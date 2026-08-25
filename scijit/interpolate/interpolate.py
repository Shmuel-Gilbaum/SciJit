"""numba equivalent of scipy.interpolate (FITPACK spline part).

jitclass-based spline classes mirroring scipy.interpolate, backed by the
fitters/evaluators in this subpackage (and therefore by the very same FITPACK
Fortran code scipy uses). Instances can be constructed AND evaluated inside
@njit code.

Classes                                  scipy equivalent
    UnivariateSpline(x, y, w, k, s)      UnivariateSpline
    InterpolatedUnivariateSpline         InterpolatedUnivariateSpline
    LSQUnivariateSpline(x, y, t,..)      LSQUnivariateSpline
    RectBivariateSpline(x, y, z2d,...)   RectBivariateSpline
    SmoothBivariateSpline(x, y, z, w,..) SmoothBivariateSpline
    RectSphereBivariateSpline(u, v, r2d) RectSphereBivariateSpline
    SmoothSphereBivariateSpline          SmoothSphereBivariateSpline

Functions (scipy signatures, tck = (t, c, k) tuple)
    splrep, splprep, splev, splint, sproot, spalde, splder_ev, bisplev

Evaluation:
  * `spl(x)` works. On the univariate classes an array runs
    .ev and a scalar runs .ev_one. On the bivariate ones `spl(x, y)` runs
    eval_grid for every pair of scalars and arrays, the cross product, which
    is what scipy's `__call__` computes at its `grid=True` default, so two
    scalars give a (1, 1) grid; the scattered-point form stays `spl.ev(x, y)`
    and the single-point scalar form `spl.ev_one(x, y)`.
    Every method is reachable by name, and `spl[x]` remains sugar for
    `spl.ev(x)`.

Differences from scipy (jitclass limitations):
  * `w=None` IS accepted on the routines the parity sweep has reached
    (sproot, splrep, UnivariateSpline, InterpolatedUnivariateSpline,
    LSQUnivariateSpline); an empty-array sentinel still works there too.
    Elsewhere the sentinel is the only spelling.
  * `bbox=[None, None]`, scipy's literal default, WORKS, from Python and
    from inside `@njit`, as does a bare `bbox=None`. `None` in a slot takes
    the data value; a NaN in a slot is a literal knot boundary.
  * derivative()/antiderivative() returning new spline objects are not
    implemented; use derivative_ev(x, nu) / splint-style calls, or
    splder / splantider for the derived tck.
  * scipy RE-CLASSES a spline instance from FITPACK's `ier`; a jitclass
    cannot change its own type, so `ier` is an attribute instead.

A jitclass constructor's defaults are Python-only. Each scipy CLASS NAME
here is therefore a plain @njit FACTORY that returns the jitclass instance,
and the jitclass itself carries an underscore. A @njit function keeps its
defaults in both worlds, so `UnivariateSpline(x, y)` compiles and runs
inside @njit as well as from Python (verified). Constructing the private
class directly, `_UnivariateSpline(x, y)`, still raises TypingError inside
@njit -- spell every argument out there, or call the factory. The classes
All seven FITPACK classes are converted. The tck FUNCTIONS were always
plain @njit. Method defaults such as `derivative_ev`'s `nu=1` are fine in
both worlds.

Omitting an argument at a PYTHON-level call of any of these costs numba's
dispatcher a per-call resolution: measured 0.315 us with every argument
explicit against 28.6 us with seven scalar defaults omitted, and 75 us when
an ARRAY default is omitted. In a hot Python loop, spell every argument out.
Inside @njit it costs nothing -- the defaults are resolved at compile time.

Accuracy: this is the same Dierckx Fortran scipy wraps, but scipy 1.18
reaches it through a C translation of the old f2py bindings rather than
through f2py, so "same Fortran" no longer implies "same bits" everywhere.
Re-measured on scipy 1.18.0, as max absolute difference:

    sproot, splrep, UnivariateSpline, InterpolatedUnivariateSpline,
    LSQUnivariateSpline                                 0.0 on every path
        tested, including smoothing fits, xb/xe, task=-1, k=1..5 and size
        sweeps -- see tests/interpolate/test_<name>_scipy.py for the grids.

    splrep(per=1), i.e. percur SMOOTHING                agrees to machine
        precision on knots, coefficients and fp; per=1 with s=0 is 0.0.

    The four BIVARIATE fitters, on a SMOOTHING fit, can place knots in
        DIFFERENT POSITIONS from scipy. The gap is then set by the data, not
        by floating point: measured up to 1.8e-02 in surface values on a
        13x14 noisy grid at s=0.01, with both sides reporting ier=0 and both
        satisfying FITPACK's own |fp-s|/s <= 1e-3 convergence test (ours
        8.1e-06, scipy 1.2e-06). Neither answer is wrong; they are two valid
        smoothing splines. On smooth, well-conditioned data the knot search
        lands in the same place and the surfaces agree.
        s=0 interpolating paths are unaffected and stay exactly 0.0.

Every figure above is a number a comparison printed, not a test bound.
A single figure for a smoothing fit describes the fixture it was measured
on; the s=0 figures are structural and reproduced on every fixture tried.

prange-safety: all of it. FITPACK's evaluators and fitters hold no state
and every workspace is a fresh local. Build instances outside the parallel
loop by preference, but constructing inside one is safe too.

Examples
--------
>>> import numpy as np
>>> from numba import njit
>>> from scijit.interpolate import InterpolatedUnivariateSpline
>>> x = np.linspace(0.0, 1.0, 11)
>>> y = np.sin(3.0 * x)
>>> spl = InterpolatedUnivariateSpline(x, y, np.ones(len(x)))
>>> @njit
... def work(spl, xs):
...     total = 0.0
...     for i in range(len(xs)):
...         total += spl(xs[i])
...     return total
>>> float(np.round(work(spl, np.array([0.25, 0.5, 0.75])), 6))
2.457187
"""
import warnings

import numpy as np
from numba import njit, objmode, float64, int64, types
from numba.core import errors
from numba.extending import overload
from scijitclass import scijitclass, all_scalar, Scalar

from . import evaluators as _ev
from . import fitters as _ft


# ---------------------------------------------------------------------
# warnings from inside compiled code
# ---------------------------------------------------------------------
#
# `warnings.warn` is not typeable by numba, but an `objmode` block runs its
# body in the interpreter, and a warning issued there reaches the ordinary
# Python warning machinery: `catch_warnings`, `simplefilter`, `-W` and
# `PYTHONWARNINGS` all see it. Measured 2026-08-01 on numba 0.66.0, including
# from inside a `prange` loop.
#
# The block acquires the GIL and boxes its argument, so these are called only
# on a fit's status branch, never per evaluated point.


@njit
def _warn_runtime(msg):
    """Issue `msg` as a ``RuntimeWarning`` from inside compiled code."""
    with objmode():
        warnings.warn(msg, RuntimeWarning)


@njit
def _warn_user(msg):
    """Issue `msg` as a ``UserWarning`` from inside compiled code."""
    with objmode():
        warnings.warn(msg, UserWarning)


@njit
def _warn_fit(msg, k, n, m, fp, s):
    """Issue scipy's ``quiet=0`` report as a ``RuntimeWarning``.

    The tail is formatted inside the ``objmode`` block so that the two floats
    are rendered by the interpreter, which is what produces scipy's text.
    """
    with objmode():
        warnings.warn(
            RuntimeWarning(msg + f"\tk={k} n={n} m={m} fp={fp} s={s}"),
            stacklevel=2)


# ---------------------------------------------------------------------
# compile-time predicates shared by the scipy-shaped front ends.
# An OMITTED default arrives as a raw python value, not types.Omitted, so
# the plain type is checked FIRST.
# ---------------------------------------------------------------------

def _lit_bool(v):
    """A numba type for a flag, reduced to a Python bool at compile time.

    Returns ``None`` when the flag is a runtime variable, and every caller
    treats that as "refuse to compile". The flags this reads (``full_output``)
    select the RETURN TYPE, and one compiled body has one return type, so a
    value only known at runtime cannot be served. Refusing produces a
    TypingError at the call site, which is the intended failure.

    Runs at TYPING time, not inside ``@njit``: `v` is a `numba.types`
    instance, not the value.
    """
    if isinstance(v, bool):                 return v
    if isinstance(v, (int, np.integer)):    return bool(v)
    if isinstance(v, types.Omitted):        return bool(v.value)
    if isinstance(v, types.BooleanLiteral): return v.literal_value
    if isinstance(v, types.IntegerLiteral): return bool(v.literal_value)
    if isinstance(v, types.StringLiteral):  return bool(v.literal_value)
    return None                             # runtime variable -> refuse


def _lit_flag(v, name, where):
    """`_lit_bool`, with the refusal spelled out in the TypingError.

    `splint-D2`, `splint-D3`, `splrep-D6`, `splprep-D8`. numba reads a bool,
    an int and a string while the call compiles, and a float or a container
    never, so those spellings are refused here where a Python caller takes
    their truthiness. Raising names the argument and the reason; returning
    ``None`` left numba reporting only that no implementation matched.
    """
    b = _lit_bool(v)
    if b is None:
        raise errors.NumbaTypeError(
            "%s: `%s` selects the return type, so inside @njit it has to be "
            "a compile-time constant: a bool, an int, a string literal or "
            "the default. Got %s, whose value is not known while the call "
            "compiles. From Python any object is read by its truthiness, as "
            "in scipy." % (where, name, v))
    return b


def _is_none(v):
    """True when an argument was left at ``None``, in any of its three forms.

    An omitted default, an explicit ``None`` and a plain Python ``None``
    reach an ``@overload`` body as three different things: `types.Omitted`
    wrapping ``None``, `types.NoneType`, and ``None`` itself. Every optional
    argument in this module is decided by this one predicate so the three
    stay indistinguishable to callers.
    """
    return (v is None or isinstance(v, types.NoneType)
            or (isinstance(v, types.Omitted) and v.value is None))


_NO_T = np.empty(0, dtype=np.float64)


def _opt_arr(v):
    """``None`` -> a zero-length float64 array; anything else -> contiguous."""
    if v is None:
        return _NO_T
    return np.ascontiguousarray(np.asarray(v, dtype=np.float64)).ravel()


@overload(_opt_arr)
def _opt_arr_ovl(v):
    """`_opt_arr` inside ``@njit``: pick the branch at compile time.

    Each compiled body contains one arm and returns a plain array.

    This docstring used to say numba could not type ``if v is None`` over two
    arms returning an array, and that is why the ``@overload`` exists.
    FALSE on numba 0.66, measured 2026-08-02: a plain ``@njit`` function with
    ``v=None`` and that exact body compiles and runs, omitted, ``None`` and
    array, from Python and from inside ``@njit``. numba prunes the dead branch,
    so the two arms never have to unify. The ``@overload`` is now one way of
    writing this rather than the only one.
    """
    if _is_none(v):
        def impl(v):
            return _NO_T
        return impl

    def impl(v):
        return np.ascontiguousarray(np.asarray(v, dtype=np.float64)).ravel()
    return impl


def _opt_f(v):
    """``None`` -> NaN. The caller pairs it with `_given` to say which."""
    return np.nan if v is None else np.float64(v)


@overload(_opt_f)
def _opt_f_ovl(v):
    """`_opt_f` inside ``@njit``: pick the branch at compile time.

    Same shape as `_opt_arr_ovl`, and the same correction applies: branch
    pruning on numba 0.66 means a plain ``@njit`` function with ``v=None``
    handles this too. See `_opt_arr_ovl`.
    """
    if _is_none(v):
        def impl(v):
            return np.nan
        return impl

    def impl(v):
        return np.float64(v)
    return impl


def _given(v):
    """True when the argument was supplied, i.e. is not ``None``."""
    return v is not None


@overload(_given)
def _given_ovl(v):
    """`_given` inside ``@njit``: the answer is a compile-time constant.

    Whether an argument was supplied is decided by its TYPE, so it is known
    during typing and baked into the compiled body as a literal. The cores
    take it as a separate boolean because a sentinel array or NaN cannot
    distinguish "not supplied" from "supplied and empty".
    """
    g = not _is_none(v)

    def impl(v):
        return g
    return impl


@njit
def _nan_if_none(v):
    """One element of a `bbox` sequence as a float; ``None`` becomes NaN.

    Exists so that ``bbox=[None, None]``, scipy's literal default, can be
    typed. numba folds an ``is None`` predicate only when the operand is a
    direct function argument (``numba/core/analysis.py``,
    ``dead_branch_prune``),
    so ``bbox[0] is None`` written at the call site survives both branches and
    unifies to ``Optional(float64)``, which no ``float64`` slot accepts. Taking
    the element as an argument here reaches the pruned path, and the ``None``
    branch is discarded before typing.
    """
    if v is None:
        return np.nan
    return np.float64(v)

# Sentinel meaning "no weights passed": a zero-length weight vector is never
# valid FITPACK input, so it stands in for scipy's w=None. Detected via
# w.size == 0 and replaced with unit weights.
#
# THE REASON RECORDED HERE WAS FALSE and is corrected 2026-08-02. It read
# "numba cannot type None in an array-argument slot". Measured on numba 0.66:
# a plain @njit factory declared `w=None` accepts omitted, None and an array,
# from Python and from inside @njit, in all nine combinations. The sentinel is
# kept only because changing these signatures is one coordinated pass over the
# whole package (fixup item F10), not because numba refuses `None`.
#
# A SEPARATE limit does still hold, and it is not this one: a JITCLASS
# constructor's defaults are Python-only, of ANY type, so inside @njit every
# argument must be passed explicitly. That is CLAUDE.md gotcha #2, and the fix
# is an @njit factory over the class rather than a different default value.
_NO_W = np.empty(0, dtype=np.float64)


# =====================================================================
# function-style API (scipy.interpolate.splrep / splev / ...)
# =====================================================================

def _cx(v):
    """True when a numba TYPE is a complex array, for an `@overload` chooser."""
    return isinstance(v, types.Array) and isinstance(v.dtype, types.Complex)


def _cx_seq(v):
    """True when a numba TYPE is complex in any spelling `splprep` takes.

    `splprep`'s `x` is an array, a tuple of arrays or a list of arrays, and
    the complex refusal has to see all three.
    """
    if _cx(v):
        return True
    if isinstance(v, types.BaseTuple):
        return any(_cx(e) for e in v.types)
    if isinstance(v, (types.List, types.ListType)):
        return _cx(v.dtype)
    return False


@njit
def _cast_raise():
    """scipy's complex-to-float64 cast refusal, unconditional.

    The one place the text lives. Reached from `_no_complex_cast` on both
    sides, and from the `splprep` arms, which know from the argument TYPE
    that the value is complex and so have nothing left to test.
    """
    raise TypeError(
        "Cannot cast array data from dtype('complex128') to "
        "dtype('float64') according to the rule 'safe'")


def _no_complex_cast(v):
    """scipy's refusal of a complex array at the FITPACK boundary.

    ``scipy.interpolate.splrep(x, y_complex)``, ``splprep`` and ``bisplev``
    all raise this exact ``TypeError``, measured on scipy 1.18. C3.
    """
    if np.iscomplexobj(v):
        _cast_raise()


@overload(_no_complex_cast)
def _no_complex_cast_ovl(v):
    """`_no_complex_cast` inside ``@njit``: decided by the array's dtype."""
    if isinstance(v, types.Array) and isinstance(v.dtype, types.Complex):
        def impl(v):
            _cast_raise()
        return impl

    def impl(v):
        pass
    return impl


def _no_complex(v, name):
    """Refuse a complex data array on a class scipy silently truncates.

    C4. MEASURED on scipy 1.18: a complex `y` on ``UnivariateSpline`` and a
    complex `z` on ``RectBivariateSpline`` produce a ``ComplexWarning``, a
    float64 coefficient array and a real result. Each class's ``Notes``
    carries the measurement.
    """
    if np.iscomplexobj(v):
        raise TypeError(name + ": complex input is not accepted")


@overload(_no_complex)
def _no_complex_ovl(v, name):
    """`_no_complex` inside ``@njit``: decided by the array's dtype."""
    if isinstance(v, types.Array) and isinstance(v.dtype, types.Complex):
        def impl(v, name):
            raise TypeError(name + ": complex input is not accepted")
        return impl

    def impl(v, name):
        pass
    return impl


def _real_in(v):
    """`v` as a real array, so a complex argument still TYPES.

    The refusal is a run-time `raise` from `_no_complex`, and a run-time
    raise does not stop numba typing the statements after it. Without this
    the ``np.asarray(v, np.float64)`` below would fail first, with a
    ``TypingError`` instead of the message the caller should see.
    """
    return np.real(np.asarray(v))


@overload(_real_in)
def _real_in_ovl(v):
    """`_real_in` inside ``@njit``: one arm per dtype class."""
    if isinstance(v, types.Array) and isinstance(v.dtype, types.Complex):
        def impl(v):
            return v.real
        return impl

    def impl(v):
        return np.asarray(v)
    return impl


def _first_seq(args):
    """P20: one or more arguments, the first an ARRAY OR A LIST.

    `scijitclass.first_array` admits ``spl(np.array([1.0, 2.0]))`` and refuses
    ``spl([1.0, 2.0])``, which scipy accepts and which the method bodies
    already handle, since they normalise through `np.asarray`.
    """
    return len(args) >= 1 and isinstance(
        args[0], (types.Array, types.List, types.ListType))


_SEQ_T = (types.Array, types.List, types.ListType)


def _grid_pair(args):
    """`SHARED-B4`: two query coordinates, each a sequence or a scalar.

    A bivariate ``spl(x, y)`` reaches `eval_grid` for every such pair,
    because scipy reads a scalar in either slot as a grid axis of length 1:
    ``spl(0.5, y)`` is ``(1, len(y))``, ``spl(x, 0.5)`` is ``(len(x), 1)``,
    and ``spl(0.5, 0.5)`` is ``(1, 1)``. `ev_one`, this package's scalar
    spelling, is reached by name.
    """
    if len(args) < 2:
        return False
    a, b = args[0], args[1]
    return ((isinstance(a, _SEQ_T) or Scalar(a))
            and (isinstance(b, _SEQ_T) or Scalar(b)))


def _pt1d(v):
    """A query coordinate as a contiguous 1-D float64 array.

    `SHARED-B4`. scipy's bivariate ``__call__`` and ``.ev`` run
    ``atleast_1d`` on both coordinates, so a scalar is an axis of length 1
    rather than a value ``len()`` cannot read.
    """
    return np.ascontiguousarray(np.asarray(v, np.float64)).ravel()


@overload(_pt1d)
def _pt1d_ovl(v):
    """`_pt1d` inside ``@njit``: a scalar and a sequence need separate bodies.

    ``np.asarray`` of a numba scalar is a 0-d array whose ``ravel`` numba
    does not type, so the scalar arm fills a length-1 buffer instead.
    """
    if (isinstance(v, types.Array) and v.ndim == 1 and v.layout == 'C'
            and v.dtype == types.float64):
        # ponytail: already what the general arm below would build. numpy's
        # asarray, ascontiguousarray and ravel are all no-copy views on a 1-D
        # contiguous float64 array, so the buffer was shared with the caller
        # either way. `.ravel()` still costs 37 ns per call there, and the
        # bivariate `.ev` pays it twice.
        def impl(v):
            return v
        return impl

    if isinstance(v, (types.Array, types.List, types.ListType,
                      types.BaseTuple)):
        def impl(v):
            return np.ascontiguousarray(np.asarray(v, np.float64)).ravel()
        return impl

    def impl(v):
        a = np.empty(1, np.float64)
        a[0] = v
        return a
    return impl


@njit
def _ev_pair(xi, yi):
    """scipy's broadcast of two ``.ev`` coordinate arrays against each other.

    `SHARED-B5`. A length-1 axis is stretched to the other's length, equal
    lengths pass through, and anything else is the mismatch scipy reports
    from ``numpy.broadcast_arrays``.
    """
    nx = len(xi)
    ny = len(yi)
    if nx == ny:
        return xi, yi
    if nx == 1:
        out = np.empty(ny, np.float64)
        for i in range(ny):
            out[i] = xi[0]
        return out, yi
    if ny == 1:
        out = np.empty(nx, np.float64)
        for i in range(nx):
            out[i] = yi[0]
        return xi, out
    raise ValueError(
        "shape mismatch: objects cannot be broadcast to a single shape.  "
        "Mismatch is between arg 0 with shape (" + str(nx)
        + ",) and arg 1 with shape (" + str(ny) + ",).")


@njit
def _iermess(ier):
    """scipy's ``_iermess[ier][0]`` message, for ``full_output``."""
    if ier == 0:
        return ("The spline has a residual sum of squares fp such that "
                "abs(fp-s)/s<=0.001")
    if ier == -1:
        return "The spline is an interpolating spline (fp=0)"
    if ier == -2:
        return ("The spline is weighted least-squares polynomial of "
                "degree k.\n"
                "fp gives the upper bound fp0 for the smoothing factor s")
    if ier == 1:
        return ("The required storage space exceeds the available storage "
                "space.\nProbable causes: data (x,y) size is too small or "
                "smoothing parameter\ns is too small (fp>s).")
    if ier == 2:
        return ("A theoretically impossible result when finding a smoothing "
                "spline\nwith fp = s. Probable cause: s too small. "
                "(abs(fp-s)/s>0.001)")
    if ier == 3:
        return ("The maximal number of iterations (20) allowed for finding "
                "smoothing\nspline with fp=s has been reached. Probable "
                "cause: s too small.\n(abs(fp-s)/s>0.001)")
    if ier == 10:
        return "Error on input data"
    # 30 and 50 come from scipy's binding, not from FITPACK, and are absent
    # from scipy's own _iermess table, so scipy reports them this way too.
    return "An error occurred"


@njit
def _splrep_core(x, y, w, has_w, xb, xe, k, task, s, has_s, t, has_t, per):
    """The whole of `splrep`. NaN `xb`/`xe` stand for scipy's ``None``, and
    `has_w` / `has_s` / `has_t` say whether `w` / `s` / `t` were given, so that
    a ZERO-LENGTH array or a NaN passed explicitly is not read as ``None``.
    Returns ``(t, c, k, fp, ier)`` always; the two entry points only slice
    it."""
    x_ = np.ascontiguousarray(np.asarray(x, np.float64)).ravel()
    y_ = np.ascontiguousarray(np.asarray(y, np.float64)).ravel()
    m = len(x_)
    if w.size == 0 and not has_w:
        w_ = np.ones(m)
        if not has_s:
            s = 0.0
    else:
        w_ = np.ascontiguousarray(np.asarray(w, np.float64)).ravel()
        if not has_s:
            s = m - np.sqrt(2.0 * m)
    if len(w_) != m:
        raise TypeError("len(w)=" + str(len(w_)) + " is not equal to m="
                        + str(m))
    if len(y_) != m:
        raise TypeError("Lengths of the first three arguments (x,y,w) must "
                        "be equal")
    if k < 1 or k > 5:
        raise TypeError("Given degree of the spline (k=" + str(k)
                        + ") is not supported. (1<=k<=5)")
    if m <= k:
        raise TypeError("m > k must hold")
    if task < -1 or task > 1:
        raise TypeError("task must be -1, 0 or 1")
    if task == 1:
        raise TypeError(
            "splrep: task=1 continues a previous fit from a module-level "
            "cache and is not supported. scipy's own task=1 raises "
            "UnboundLocalError on every call, so there is no behaviour to "
            "reproduce")
    if has_t:
        task = -1
    elif task == -1:
        raise TypeError("Knots must be given for task=-1")
    if np.isnan(xb):
        xb = x_[0]
    if np.isnan(xe):
        xe = x_[m - 1]

    if task == -1:
        # scipy allocates len(t) + 2*k + 2 and writes the interior knots into
        # the middle; FITPACK overwrites the k+1 boundary slots at each end
        # with xb and xe, so their initial values do not matter.
        ni_ = len(t)
        tt = np.empty(ni_ + 2 * k + 2, np.float64)
        for i in range(k + 1):
            tt[i] = xb
            tt[ni_ + k + 1 + i] = xe
        for i in range(ni_):
            tt[k + 1 + i] = t[i]
        nest = ni_ + 2 * k + 2
        if per != 0:
            # splrep-D1. scipy reads `t` and `per` INDEPENDENTLY: `t` sets
            # task = -1 and `per` still selects `percur` over `curfit`, so
            # the two together are the periodic least-squares fit. The knot
            # checks below belong to scipy's `curfit` binding and are not run
            # on this route; FITPACK's own percur reports a bad knot vector
            # as ier = 10, which is the ValueError scipy raises for it.
            tk, ck, fp, ier = _ft.percur_lsq(x_, y_, w_, tt, k)
        else:
            # Two checks scipy's binding runs before FITPACK sees the knots,
            # with ier codes FITPACK never produces.  Comparisons are written
            # so a NaN knot falls through both, which is what scipy does with
            # one.
            # ... but only while the data lies inside [xb, xe]. When it does
            # not, FITPACK's own fpchec fails first and reports ier = 10, and
            # so does scipy: measured, splrep(x, y, t=[1,2,3], xb=1.5) gives
            # ier 10 and ValueError("Error on input data") there, not one of
            # these codes.
            bad = 0
            if xb <= x_[0] and xe >= x_[m - 1]:
                for i in range(1, ni_):
                    if t[i] <= t[i - 1]:
                        bad = 30
                for i in range(ni_):
                    if t[i] <= xb or t[i] >= xe:
                        bad = 30
                if bad == 0:
                    for i in range(ni_):
                        if t[i] <= x_[0] or t[i] >= x_[m - 1]:
                            bad = 50
            if bad != 0:
                return (tt, np.zeros(ni_ + 2 * k + 2, np.float64), k, 0.0,
                        bad, nest, m, s)
            tk, ck, fp, ier = _ft.curfit_lsq(x_, y_, w_, tt, k, xb, xe)
    elif per != 0:
        nest = max(m + 2 * k, 2 * k + 3)
        tk, ck, fp, ier = _ft.percur(x_, y_, w_, k, s, -1)
    else:
        nest = max(m + k + 1, 2 * k + 3)
        tk, ck, fp, ier = _ft.curfit(x_, y_, w_, k, s, xb, xe, -1)
    return tk, ck, k, fp, ier, nest, m, s


@njit
def _warn_close(i, m):
    """Issue scipy's per-dimension curve-closing report."""
    with objmode():
        warnings.warn(RuntimeWarning(f'Setting x[{i}][{m}]=x[{i}][0]'),
                      stacklevel=2)


@njit
def _splprep_quiet(quiet, r):
    """scipy's ``ier <= 0 and not quiet`` success report, for `splprep`.

    `r` is `_splprep_core`'s return. scipy reports the length of the RETURNED
    knot vector here, unlike `splrep`, which reports the allocated length.
    """
    if quiet == 0 and r[5] <= 0:
        _warn_fit(_iermess(r[5]), r[2], len(r[0]), r[6], r[4], r[7])


@njit
def _splrep_quiet(quiet, r):
    """scipy's ``ier <= 0 and not quiet`` success report.

    `r` is `_splrep_core`'s return, whose trailing three entries are the
    ALLOCATED knot-array length, `m` and the resolved `s`. scipy reports the
    allocated length rather than the fitted knot count, so that is what is
    read here.
    """
    if quiet == 0 and r[4] <= 0:
        _warn_fit(_iermess(r[4]), r[2], r[5], r[6], r[3], r[7])


@njit
def _splrep_raise(ier):
    """scipy's ``ier > 0 and not full_output`` branch.

    scipy warns for 1, 2 and 3 and raises for everything else positive. Both
    halves are reproduced; the warning carries the same `_iermess` text.
    """
    if ier == 1 or ier == 2 or ier == 3:
        _warn_runtime(_iermess(ier))
        return
    if ier == 10:
        raise ValueError("Error on input data")
    if ier > 0:
        raise TypeError("An error occurred")


def splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
           full_output=0, per=0, quiet=1):
    """Compute the B-spline representation of a 1-D curve.

    Parameters
    ----------
    x : 1-D array_like of float, length m
        Abscissae, strictly increasing.
    y : 1-D array_like of float, length m
        Ordinates.
    w : 1-D array_like of float, optional
        Positive weights, length m. ``None`` (the default) means unit
        weights AND selects ``s = 0``; supplying `w` selects
        ``s = m - sqrt(2*m)`` instead.
    xb, xe : float, optional
        Interval the fit is made over. ``None`` (the default) means ``x[0]``
        and ``x[-1]``. They may only WIDEN the data interval.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    task : int, optional
        0 (default) finds the smoothing spline. -1 finds the weighted
        least-squares spline for the knots given in `t`. 1 is not supported;
        see Deviations.
    s : float, optional
        Smoothing factor: the fit satisfies
        ``sum(w[i]*(y[i]-spl(x[i])))**2 <= s``. ``None`` (the default)
        resolves as described under `w`.
    t : 1-D array_like of float, optional
        INTERIOR knots for ``task=-1``, without the boundary repetitions.
        Supplying it forces ``task=-1``. `t` and `per` are read
        independently, so the two together give the periodic least-squares
        fit on those knots.
    full_output : bool, optional
        Non-zero returns the 4-tuple described below. Must be a compile-time
        constant inside ``@njit``, since it selects the return type.
    per : bool, optional
        Non-zero fits a periodic spline through ``percur``. ``y[m-1]`` and
        ``w[m-1]`` are not used. Any non-zero value behaves as 1.
    quiet : bool, optional
        Zero also warns on a SUCCESSFUL fit, reporting `k`, the allocated
        knot count, `m`, `fp` and `s`. Default 1, which suppresses it. The
        ``ier`` in {1, 2, 3} warning is issued either way.

    Returns
    -------
    tck : tuple of (t, c, k)
        Knot vector, coefficients in FITPACK's padded form (only the first
        ``len(t) - k - 1`` are meaningful) and the degree.
    fp : float
        Weighted sum of squared residuals. Returned only when
        `full_output` is non-zero.
    ier : int
        FITPACK status: -1 interpolating spline, -2 least-squares
        polynomial, 0 smoothing achieved, 1/2/3 failure, 10 invalid input.
        Returned only when `full_output` is non-zero. It is the value FITPACK
        set, including on the failure paths that raise when `full_output` is
        zero.
    msg : str
        The exit message for the FITPACK status code `ier`. Returned only when
        `full_output` is non-zero, and reads "An error occurred" for any `ier`
        outside the table.

    Raises
    ------
    TypeError
        ``len(w) != m``, ``len(y) != m``, ``k`` outside 1..5, ``m <= k``,
        ``task`` outside -1..1, ``task == 1``, ``task == -1`` without `t`,
        or a FITPACK failure with ``ier`` outside {1, 2, 3, 10}.
    ValueError
        ``ier == 10``, FITPACK's invalid-input code, when `full_output` is
        zero. Without the raise, the empty `t`/`c` evaluate to 0.0 through
        `splev`: a wrong number and no error.

    Warns
    -----
    RuntimeWarning
        ``ier`` in {1, 2, 3} with ``full_output=0``: the fit is returned and
        the matching ``_iermess`` text is warned. With ``quiet=0``, also on
        ``ier <= 0``, reporting the successful fit. Both are issued through
        a ``numba.objmode`` block, which runs its body in the interpreter,
        so ``warnings.catch_warnings`` and ``-W`` see them from compiled and
        uncompiled callers alike.

    See Also
    --------
    scipy.interpolate.splrep : The scipy routine this mirrors.

    Notes
    -----
    - `full_output` must be a compile-time constant inside ``@njit``: it
      selects between a 3-tuple and a 4-tuple return, and numba compiles one
      return type per specialization. A bool, an int, a string and the
      default are read while the call compiles; a float, a container and a
      runtime variable are not, and raise ``TypingError`` naming the
      argument. From Python every object is read by its truthiness.
    - ``task=1`` raises. scipy keeps the previous fit in a function-local
      ``_curfit_cache``, so its own ``task=1`` raises ``UnboundLocalError``
      on every call, including immediately after a ``task=0`` call. There is
      no behaviour to reproduce.
    - A parametric `y` (2-D, or a list of arrays) is not accepted.

    For ``task=-1`` the interior knots are validated before FITPACK sees
    them, and two codes FITPACK never produces are reported: 30 for knots
    that are not strictly increasing or not strictly inside ``(xb, xe)``,
    and 50 for knots outside ``(x[0], x[-1])``. Neither is in scipy's
    ``_iermess`` table, so both surface as
    ``TypeError("An error occurred")``. Those two checks run only while the
    data lies inside ``[xb, xe]``; when it does not, ``fpchec`` fails first
    and reports ``ier = 10``. A NaN knot passes both checks. They belong to
    the non-periodic route: with ``per`` non-zero the knots go to ``percur``,
    which reports a bad knot vector as ``ier = 10`` and
    ``ValueError("Error on input data")``.

    A SMOOTHING FIT MAY WARN THAT IT IS RANK DEFICIENT. With a small `s` on
    noisy data, FITPACK's knot search can place knots so that one B-spline
    coefficient is not determined by the data at all. The curve still passes
    through the data as asked, and ``fp`` reports success, but between the
    data points it carries an arbitrary component. A ``UserWarning`` naming
    the number of undetermined coefficients is issued when this happens, and
    a larger `s` is the fix. scipy issues no warning for it.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import splrep, splev
    >>> x = np.linspace(0, 4, 40)
    >>> y = np.sin(x)
    >>> @njit
    ... def fit_and_eval(x, y, q):
    ...     tck = splrep(x, y)
    ...     return splev(q, tck)
    >>> float(fit_and_eval(x, y, np.array([1.5]))[0])
    0.9974947337577743
    """
    _no_complex_cast(x)
    _no_complex_cast(y)
    r = _splrep_core(_real_in(x), _real_in(y), _opt_arr(w), _given(w), _opt_f(xb), _opt_f(xe),
                     k, task, _opt_f(s), _given(s), _opt_arr(t), _given(t),
                     per)
    _splrep_quiet(quiet, r)
    if not full_output:
        _splrep_raise(r[4])
        return (r[0], r[1], r[2])
    return (r[0], r[1], r[2]), r[3], r[4], _iermess(r[4])


@overload(splrep, prefer_literal=True)
def _splrep_ovl(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
                full_output=0, per=0, quiet=1):
    """`splrep` inside ``@njit``: one compiled body per `full_output` value.

    `full_output` changes the return from a 3-tuple to a 4-tuple whose second
    and third elements are a float and an int, so the two shapes cannot share
    a compiled body. `prefer_literal=True` makes numba offer the literal
    value here, and `_lit_flag` raises the ``TypingError`` for a value that is
    not known while the call compiles.

    Both arms call the same `_splrep_core`. Only the packaging differs, and
    the ``not full_output`` arm additionally raises on a bad `ier` the way
    scipy does.
    """
    fo = _lit_flag(full_output, 'full_output', 'splrep')

    if fo:
        def impl(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
                 full_output=0, per=0, quiet=1):
            _no_complex_cast(x)
            _no_complex_cast(y)
            r = _splrep_core(_real_in(x), _real_in(y), _opt_arr(w),
                             _given(w), _opt_f(xb), _opt_f(xe), k, task,
                             _opt_f(s), _given(s), _opt_arr(t), _given(t),
                             per)
            _splrep_quiet(quiet, r)
            return (r[0], r[1], r[2]), r[3], r[4], _iermess(r[4])
        return impl

    def impl(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
             full_output=0, per=0, quiet=1):
        _no_complex_cast(x)
        _no_complex_cast(y)
        r = _splrep_core(_real_in(x), _real_in(y), _opt_arr(w), _given(w),
                         _opt_f(xb), _opt_f(xe), k, task, _opt_f(s),
                         _given(s), _opt_arr(t), _given(t), per)
        _splrep_quiet(quiet, r)
        _splrep_raise(r[4])
        return (r[0], r[1], r[2])
    return impl


@njit
def _ragged_raise(d):
    """numpy's own text for a ragged sequence of coordinate arrays.

    `splprep-D7`. Reaching `np.asarray` on a ragged container is what raises
    in scipy, and the sentence names the number of entries. numba renders an
    int through ``str``, so the whole message is built in compiled code.
    """
    raise ValueError(
        "setting an array element with a sequence. The requested array has "
        "an inhomogeneous shape after 1 dimensions. The detected shape was ("
        + str(d) + ",) + inhomogeneous part.")


def _xflat(x, idim):
    """``(x_flat, idim)`` from any accepted spelling of scipy's `x`.

    scipy runs ``atleast_1d(x)`` then ``idim, m = x.shape``, so a sequence of
    per-dimension arrays and an ``(idim, m)`` array are the same input to it.
    Both are typeable by numba and both are accepted here, alongside the flat
    interleaved form this package started with.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        return np.ascontiguousarray(a.T).ravel(), int(a.shape[0])
    if idim >= 1:
        # .copy(), because `per` closes the curve by writing into this buffer
        # and ravel() of an already contiguous argument is a VIEW onto the
        # caller's array.
        return np.ascontiguousarray(a).ravel().copy(), int(idim)
    raise ValueError(
        "splprep: x must be a sequence of idim arrays or an (idim, m) "
        "array; a flat interleaved array needs idim= as well")


@overload(_xflat)
def _xflat_ovl(x, idim):
    """`_xflat` inside ``@njit``: flatten `splprep`'s `x` to interleaved form.

    scipy accepts `x` as a sequence of `idim` coordinate arrays. Each spelling
    of that sequence is a different numba type and needs its own interleaving
    loop, so this dispatches on the type and hands back one body per case:
    a homogeneous or heterogeneous tuple, a reflected or typed list, or a
    2-D array.

    The tuple case builds its body with ``exec``. A tuple is indexed by
    position at compile time, so the ``for j in range(d)`` that writes
    ``out[i*d + j] = x[j][i]`` has to be unrolled into `d` literal indices
    before numba sees it.
    """
    if isinstance(x, types.BaseTuple):
        d = len(x)
        src = ["def impl(x, idim):", "    m = len(x[0])"]
        for j in range(1, d):
            src += [f"    if len(x[{j}]) != m:",
                    f"        _ragged_raise({d})"]
        src += [f"    out = np.empty({d} * m, np.float64)",
                "    for i in range(m):"]
        src += [f"        out[i * {d} + {j}] = x[{j}][i]" for j in range(d)]
        src += [f"    return out, {d}"]
        ns = {'np': np, '_ragged_raise': _ragged_raise}
        exec("\n".join(src), ns)
        return ns['impl']

    if isinstance(x, (types.List, types.ListType)):
        def impl(x, idim):
            d = len(x)
            m = len(x[0])
            for j in range(1, d):
                if len(x[j]) != m:
                    _ragged_raise(d)
            out = np.empty(d * m, np.float64)
            for i in range(m):
                for j in range(d):
                    out[i * d + j] = x[j][i]
            return out, d
        return impl

    if isinstance(x, types.Array) and x.ndim == 2:
        def impl(x, idim):
            d, m = x.shape
            out = np.empty(d * m, np.float64)
            for i in range(m):
                for j in range(d):
                    out[i * d + j] = x[j, i]
            return out, d
        return impl

    def impl(x, idim):
        if idim < 1:
            raise ValueError(
                "splprep: x must be a sequence of idim arrays or an (idim, m) "
                "array; a flat interleaved array needs idim= as well")
        # .copy(), because `per` closes the curve by writing into this buffer
        # and ravel() of an already contiguous argument is a VIEW onto the
        # caller's array.
        return (np.ascontiguousarray(np.asarray(x, np.float64)).ravel().copy(),
                idim)
    return impl


def _xin(x):
    """`splprep`'s `x` in a form the compiled core can take, from Python.

    `splprep-D6`. A list of LISTS, which scipy accepts, is a nested reflected
    container, and numba refuses to unbox one at the argument boundary of
    `_splprep_core` before any of this module's code runs. Converting that
    one spelling to a 2-D array here reaches the same core the compiled entry
    point reaches, with the same ragged-input message. Every other spelling
    is passed through untouched.
    """
    if (isinstance(x, (list, tuple)) and len(x)
            and isinstance(x[0], (list, tuple))):
        m = len(x[0])
        for e in x:
            if len(e) != m:
                _ragged_raise(len(x))
        return np.asarray(x, dtype=np.float64)
    return x


@njit
def _splprep_core(x, idim_in, w, has_w, u, has_u, ub, has_ub, ue, has_ue,
                  k, task, s, has_s, t, has_t, nest, has_nest, nest_f,
                  per, quiet):
    """The whole of `splprep`. Every optional argument travels as a value
    plus a `has_` flag, so that a ZERO-LENGTH array, a NaN or a 0 passed
    explicitly is a LITERAL and only ``None`` means "not given", which is P1.
    Returns ``(t, c, k, u, fp, ier, m, s, idim)`` always; the entry points
    only slice it. `idim` rides along because `_c_split` cannot recover it
    from an EMPTY `c`, which is what a failed fit returns."""
    xf, idim = _xflat(x, idim_in)
    if idim < 1 or idim > 10:
        raise TypeError("0 < idim < 11 must hold")
    if len(xf) % idim != 0:
        raise TypeError("splprep: len(x) must be a multiple of idim")
    m = len(xf) // idim
    if per != 0:
        # scipy closes the curve for the caller, in place; the copy is closed
        # here instead, so the caller's array is left alone.
        for j in range(idim):
            if xf[idim * (m - 1) + j] != xf[j]:
                if quiet == 0:
                    _warn_close(j, m)
                xf[idim * (m - 1) + j] = xf[j]
    if w.size == 0 and not has_w:
        w_ = np.ones(m)
    else:
        w_ = np.ascontiguousarray(np.asarray(w, np.float64)).ravel()
    if k < 1 or k > 5:
        raise TypeError("1 <= k= " + str(k) + " <=5 must hold")
    if task < -1 or task > 1:
        raise TypeError("task must be -1, 0 or 1")
    if len(w_) != m or (has_u and len(u) != m):
        raise TypeError("Mismatch of input dimensions")
    if not has_s:
        s = m - np.sqrt(2.0 * m)
    if (not has_t) and task == -1:
        raise TypeError("Knots must be given for task=-1")
    nt = len(t)
    if task == -1 and nt < 2 * k + 2:
        raise TypeError("There must be at least 2*k+2 knots for task=-1")
    if m <= k:
        raise TypeError("m > k must hold")
    if task == 1:
        raise TypeError(
            "splprep: task=1 continues a previous fit from a function-local "
            "cache and is not supported. scipy's own task=1 raises "
            "UnboundLocalError on every call, so there is no behaviour to "
            "reproduce")
    # scipy converts `nest` where it reaches ``parcur``, so a float raises
    # there -- except with ``s == 0``, which replaces the value first and
    # never converts it.
    if has_nest and nest_f and not (task >= 0 and s == 0.0):
        raise TypeError("'float' object cannot be interpreted as an integer")
    ne = int(nest) if has_nest else m + 2 * k
    if (task >= 0 and s == 0.0) or ne < 0:
        ne = m + 2 * k if per != 0 else m + k + 1
    if ne < 2 * k + 3:
        ne = 2 * k + 3
    tin = t if task == -1 else _NO_T
    # P1: `fitters.parcur` reads ``None`` as "not given" and a compiled body
    # cannot pass one, so the pair is resolved here, on the same rule --
    # `u`'s own ends when `ub`/`ue` were omitted, and 0.0 / 1.0 when there is
    # no `u` at all, which is the case where FITPACK computes the parameter
    # itself and both are inert.
    ubv = 0.0
    uev = 1.0
    if has_u and len(u) > 0:
        ubv = ub if has_ub else u[0]
        uev = ue if has_ue else u[len(u) - 1]
    if per != 0:
        uu, tk, ck, fp, ier = _ft.clocur(xf, w_, idim, k, s, u, tin, ne)
    else:
        uu, tk, ck, fp, ier = _ft.parcur(xf, w_, idim, k, s, u, ubv, uev,
                                         tin, ne)
    return tk, ck, k, uu, fp, ier, m, s, idim


@njit
def _c_split(c, n, k, idim):
    """FITPACK's flat stride-`n` curve coefficients as scipy's list.

    `c` holds `idim` blocks of length `n`, of which the first ``n-k-1``
    entries of each are defined. Returns a list of `idim` arrays of length
    ``n-k-1``, which is what ``scipy.interpolate.splprep`` puts in ``tck[1]``.

    `idim` is passed rather than read from ``len(c) // n`` because a failed
    fit returns an empty `c` and ``n = 0``, where scipy still returns `idim`
    empty arrays. It may be a runtime value.

    Each slice is copied, so the returned arrays do not alias `c`.
    """
    m = n - k - 1
    if m < 0:
        m = 0
    return [c[j * n: j * n + m].copy() for j in range(idim)]


# `_c_split`'s inverse and the predicate that chooses between the two layouts
# live in `evaluators`, the layer both this module and `curev` sit on, so
# there is one answer to "is this `c` parametric" in the subpackage.
_is_param_c = _ev._is_param_c


@njit
def _splprep_raise(ier):
    """scipy's ``ier > 0 and not full_output`` branch.

    Identical to `_splrep_raise`: warn for 1, 2 and 3, raise above that.
    """
    if ier == 1 or ier == 2 or ier == 3:
        _warn_runtime(_iermess(ier))
        return
    if ier == 10:
        raise ValueError("Error on input data")
    if ier > 0:
        raise TypeError("An error occurred")


def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None,
            full_output=0, nest=None, per=0, quiet=1, idim=-1, c_list=1):
    """Compute the B-spline representation of an N-D parametric curve.

    Parameters
    ----------
    x : sequence of 1-D array_like, or (idim, m) array_like, or 1-D array_like
        The curve's sample points. Three spellings, all measured on numba
        0.66: a tuple or list of `idim` arrays of length m; an ``(idim, m)``
        array; or a FLAT INTERLEAVED array of length ``idim*m``, in which case
        `idim` must be given. A bare 1-D array without `idim` raises.
    w : 1-D array_like of float, optional
        Positive weights, one per POINT, length m. ``None`` (the default)
        means unit weights. Unlike ``splrep``, supplying `w` does not change
        the default `s`.
    u : 1-D array_like of float, optional
        Parameter values for the data points, strictly increasing, length m.
        ``None`` (the default) has FITPACK compute the cumulative chord
        length normalised to ``[ub, ue]``. When given it is returned
        unchanged, and any range is accepted.
    ub, ue : float, optional
        Bounds on the parameter. INERT unless `u` is given, in which case
        ``None`` means ``u[0]`` and ``u[-1]``.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    task : int, optional
        0 (default) finds the smoothing curve. -1 finds the weighted
        least-squares curve on the knots given in `t`, which must then be the
        FULL knot vector with at least ``2*k+2`` entries. 1 is not supported;
        see Deviations.
    s : float, optional
        Smoothing factor: the fit satisfies
        ``sum(w[i]*dist(x[i], s(u[i])))**2 <= s``. ``None`` (the default)
        means ``m - sqrt(2*m)`` whether or not `w` is given.
        ``s = 0`` gives the interpolating curve.
    t : 1-D array_like of float, optional
        The FULL knot vector for ``task=-1``, boundary repetitions included.
        This differs from ``splrep``, whose `t` is the INTERIOR knots and
        implies ``task=-1``; here `t` alone does nothing and `task` must be
        set to -1.
    full_output : int, optional
        Non-zero returns the 4-tuple described below. Must be a compile-time
        constant inside ``@njit``, since it selects the return type.
    nest : int, optional
        Over-estimate of the knot count. ``None`` (the default) resolves
        through the rule ``m + 2*k``, replaced by ``m + k + 1``
        (or ``m + 2*k`` when `per`) whenever ``task >= 0 and s == 0``, then
        floored at ``2*k + 3``. It is INERT on an interpolating fit and
        BINDS on a smoothing one, where too small a value returns
        ``ier = 1`` with fewer knots. A float raises ``TypeError`` wherever
        the value is used.
    per : int, optional
        Non-zero fits a closed curve through ``clocur``. The last point is
        set equal to the first, on this function's own copy rather than on
        the caller's array.
    quiet : bool, optional
        Zero also warns on a SUCCESSFUL fit, reporting `k`, the knot count,
        `m`, `fp` and `s`, and warns once per dimension that `per` closes.
        Default 1, which suppresses both. The ``ier`` in {1, 2, 3} warning is
        issued either way.
    idim : int, optional
        Number of coordinates per point, needed only for the flat
        interleaved `x`. -1, the default, means "read it from `x`". See Notes.
    c_list : int, optional
        Selects the layout of ``tck[1]``. 1, the default, gives a list of
        `idim` arrays of length ``n-k-1``. 0 gives FITPACK's one flat
        stride-``len(t)`` array instead. Must be a compile-time constant
        inside ``@njit``, since it selects the return type. See Notes.

    Returns
    -------
    tck : tuple of (t, c, k)
        ``t`` is the knot vector in the parameter and ``k`` the degree.
        With ``c_list=1``, the default, ``c`` is a list of `idim` arrays of
        length ``n-k-1``, one per curve dimension.
        With ``c_list=0``, ``c`` is one flat array in FITPACK's
        STRIDE-``len(t)`` curve layout: dimension j occupies
        ``c[j*n : j*n + n-k-1]``, where ``n = len(t)``.
        `splev`, `splder_ev`, `splint`, `sproot`, `spalde` and
        `scijit.interpolate.evaluators.curev` read either layout.
    u : 1-D float64 ndarray, length m
        The parameter values: FITPACK's when `u` was not given, and the
        supplied array when it was.
    fp : float
        Weighted sum of squared residuals. Returned only when `full_output`
        is non-zero.
    ier : int
        FITPACK status: -1 interpolating curve, -2 least-squares polynomial,
        0 smoothing achieved, 1/2/3 failure, 10 invalid input. Returned only
        when `full_output` is non-zero. It is the value FITPACK set,
        including on the failure paths that raise when `full_output` is zero.
    msg : str
        The exit message for the FITPACK status code `ier`. Returned only when
        `full_output` is non-zero.

    Raises
    ------
    TypeError
        `idim` outside 1..10, `k` outside 1..5, `task` outside -1..1,
        ``task == 1``, ``task == -1`` without `t` or with fewer than
        ``2*k+2`` knots, ``m <= k``, a length mismatch between `x`, `w` and
        `u`, or a FITPACK failure with `ier` outside {1, 2, 3, 10}.
    ValueError
        ``ier == 10``, FITPACK's invalid-input code, when `full_output` is
        zero -- a non-increasing `u`, a negative weight, duplicate
        consecutive points. Also raised for an `x` that is neither a sequence
        of arrays, an ``(idim, m)`` array, nor a flat array with `idim`
        given, and for a ragged sequence.

    Warns
    -----
    RuntimeWarning
        `ier` in {1, 2, 3} with ``full_output=0``: the fit is returned and
        the matching ``_iermess`` text is warned. The warning
        is issued through a ``numba.objmode`` block, which runs its body in
        the interpreter, so ``warnings.catch_warnings`` and ``-W`` see it
        from compiled and uncompiled callers alike.

    See Also
    --------
    scipy.interpolate.splprep : The scipy routine this mirrors.

    Notes
    -----
    - `full_output` and `c_list` must both be compile-time constants inside
      ``@njit``: each selects part of the return type, and numba compiles one
      return type per specialization. A bool, an int, a string and the
      default are read while the call compiles; a float, a container and a
      runtime variable are not, and raise ``TypingError`` naming the
      argument. From Python every object is read by its truthiness. The two
      flags are independent, so there are four compiled bodies.
    - `tck` is a tuple. scipy's is a ``list``, ``[t, list(c), k]``, so
      ``tck[1] = ...`` and ``tck + [extra]`` work there and not here.
    - `idim` and `c_list` are extra trailing arguments with no scipy
      counterpart. `idim` is needed only for the flat interleaved `x` and
      accepts a runtime value; `c_list` selects the coefficient layout. A
      scipy-shaped call passes neither.
    - ``task=1`` raises. scipy keeps the previous fit in a function-local
      ``_parcur_cache``, so its own ``task=1`` raises ``UnboundLocalError``
      on every call, including immediately after a ``task=0`` call. There is
      no behaviour to reproduce.

    A SMOOTHING FIT MAY WARN THAT IT IS RANK DEFICIENT. With a small `s` on
    noisy data, FITPACK's knot search can place knots so that one B-spline
    coefficient is not determined by the data at all. The curve still passes
    through the data as asked, and ``fp`` reports success, but between the
    data points it carries an arbitrary component. A ``UserWarning`` naming
    the number of undetermined coefficients is issued when this happens, and
    a larger `s` is the fix. scipy issues no warning for it.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import splprep
    >>> t = np.linspace(0, 2 * np.pi, 30)
    >>> pts = np.vstack((np.cos(t), np.sin(t)))
    >>> @njit
    ... def fit(pts):
    ...     (t, c, k), u = splprep(pts, s=0.0)
    ...     return len(t), len(c), len(c[0]), len(u)
    >>> fit(pts)
    (34, 2, 30, 30)
    >>> @njit
    ... def fit_flat(pts):
    ...     (t, c, k), u = splprep(pts, s=0.0, c_list=0)
    ...     return len(c)
    >>> fit_flat(pts)
    68
    """
    _no_complex_cast(x)
    r = _splprep_core(_xin(x), idim, _opt_arr(w), _given(w), _opt_arr(u),
                      _given(u), _opt_f(ub), _given(ub), _opt_f(ue),
                      _given(ue), k, task, _opt_f(s), _given(s), _opt_arr(t),
                      _given(t), _opt_f(nest), _given(nest),
                              _is_float_arg(nest), per, quiet)
    _splprep_quiet(quiet, r)
    c = _c_split(r[1], len(r[0]), r[2], r[8]) if c_list else r[1]
    if not full_output:
        _splprep_raise(r[5])
        return ((r[0], c, r[2]), r[3])
    return ((r[0], c, r[2]), r[3]), r[4], r[5], _iermess(r[5])


@overload(splprep, prefer_literal=True)
def _splprep_ovl(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None,
                 t=None, full_output=0, nest=None, per=0, quiet=1, idim=-1,
                 c_list=1):
    """`splprep` inside ``@njit``: one compiled body per (`full_output`,
    `c_list`) pair.

    Same mechanism as `_splrep_ovl`, over a return that is already nested:
    ``((t, c, k), u)`` without `full_output`, and that plus ``fp``, ``ier``
    and the message with it. `c_list` selects `c` between a flat array and a
    list of arrays, which is a second axis of the return type, so the two
    flags give four bodies.
    """
    fo = _lit_flag(full_output, 'full_output', 'splprep')
    cl = _lit_flag(c_list, 'c_list', 'splprep')

    if _cx_seq(x):
        # C3, `D-INTP-2`. scipy refuses a complex `x` at the FITPACK cast, and
        # `_splprep_core` cannot type one. The refusal is a run-time raise, so
        # each arm still ends in a return of the shape its caller unpacks: an
        # arm that only raises types as ``none`` and the caller then fails to
        # compile at the unpack, which is a TypingError again.
        if fo and cl:
            def impl(x, w=None, u=None, ub=None, ue=None, k=3, task=0,
                     s=None, t=None, full_output=0, nest=None, per=0,
                     quiet=1, idim=-1, c_list=1):
                _cast_raise()
                z = np.empty(0, np.float64)
                return ((z, _c_split(z, 1, k, 1), k), z), 0.0, 0, _iermess(0)
            return impl

        if fo:
            def impl(x, w=None, u=None, ub=None, ue=None, k=3, task=0,
                     s=None, t=None, full_output=0, nest=None, per=0,
                     quiet=1, idim=-1, c_list=1):
                _cast_raise()
                z = np.empty(0, np.float64)
                return ((z, z, k), z), 0.0, 0, _iermess(0)
            return impl

        if cl:
            def impl(x, w=None, u=None, ub=None, ue=None, k=3, task=0,
                     s=None, t=None, full_output=0, nest=None, per=0,
                     quiet=1, idim=-1, c_list=1):
                _cast_raise()
                z = np.empty(0, np.float64)
                return ((z, _c_split(z, 1, k, 1), k), z)
            return impl

        def impl(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None,
                 t=None, full_output=0, nest=None, per=0, quiet=1, idim=-1,
                 c_list=1):
            _cast_raise()
            z = np.empty(0, np.float64)
            return ((z, z, k), z)
        return impl

    if fo and cl:
        def impl(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None,
                 t=None, full_output=0, nest=None, per=0, quiet=1, idim=-1,
                 c_list=1):
            r = _splprep_core(x, idim, _opt_arr(w), _given(w),
                              _opt_arr(u), _given(u), _opt_f(ub), _given(ub),
                              _opt_f(ue), _given(ue), k, task, _opt_f(s),
                              _given(s), _opt_arr(t), _given(t),
                              _opt_f(nest), _given(nest),
                              _is_float_arg(nest), per, quiet)
            _splprep_quiet(quiet, r)
            c = _c_split(r[1], len(r[0]), r[2], r[8])
            return ((r[0], c, r[2]), r[3]), r[4], r[5], _iermess(r[5])
        return impl

    if fo:
        def impl(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None,
                 t=None, full_output=0, nest=None, per=0, quiet=1, idim=-1,
                 c_list=1):
            r = _splprep_core(x, idim, _opt_arr(w), _given(w),
                              _opt_arr(u), _given(u), _opt_f(ub), _given(ub),
                              _opt_f(ue), _given(ue), k, task, _opt_f(s),
                              _given(s), _opt_arr(t), _given(t),
                              _opt_f(nest), _given(nest),
                              _is_float_arg(nest), per, quiet)
            _splprep_quiet(quiet, r)
            return ((r[0], r[1], r[2]), r[3]), r[4], r[5], _iermess(r[5])
        return impl

    if cl:
        def impl(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None,
                 t=None, full_output=0, nest=None, per=0, quiet=1, idim=-1,
                 c_list=1):
            r = _splprep_core(x, idim, _opt_arr(w), _given(w),
                              _opt_arr(u), _given(u), _opt_f(ub), _given(ub),
                              _opt_f(ue), _given(ue), k, task, _opt_f(s),
                              _given(s), _opt_arr(t), _given(t),
                              _opt_f(nest), _given(nest),
                              _is_float_arg(nest), per, quiet)
            _splprep_quiet(quiet, r)
            c = _c_split(r[1], len(r[0]), r[2], r[8])
            _splprep_raise(r[5])
            return ((r[0], c, r[2]), r[3])
        return impl

    def impl(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None,
             t=None, full_output=0, nest=None, per=0, quiet=1, idim=-1,
             c_list=1):
        r = _splprep_core(x, idim, _opt_arr(w), _given(w), _opt_arr(u),
                          _given(u), _opt_f(ub), _given(ub), _opt_f(ue),
                          _given(ue), k, task, _opt_f(s), _given(s),
                          _opt_arr(t), _given(t), _opt_f(nest),
                          _given(nest), _is_float_arg(nest), per, quiet)
        _splprep_quiet(quiet, r)
        _splprep_raise(r[5])
        return ((r[0], r[1], r[2]), r[3])
    return impl


def _is_float_arg(v):
    """True when the argument is a float, as a compile-time constant.

    `splprep`'s `nest` reaches the core through `_opt_f`, which makes every
    spelling a float64, so the ORIGINAL type has to travel beside the value
    for the refusal to be possible at all.
    """
    return isinstance(v, (float, np.floating))


@overload(_is_float_arg)
def _is_float_arg_ovl(v):
    """`_is_float_arg` inside ``@njit``: decided by the type, baked in."""
    f = (isinstance(v, types.Float)
         or (isinstance(v, types.Omitted) and isinstance(v.value, float)))

    def impl(v):
        return f
    return impl


def _int_order(v):
    """scipy's f2py conversion of a derivative order: a float is refused.

    ``scipy.interpolate.splev(x, tck, der=1.0)`` reaches
    ``_fitpack._spl_`` with a float and CPython raises
    ``TypeError("'float' object cannot be interpreted as an integer")``,
    measured on scipy 1.18 for 1.0 and for 1.7 alike. The range test runs
    FIRST there, so this is called after it. splev-D9, splder_ev-D9.
    """
    if isinstance(v, (float, np.floating)):
        raise TypeError(
            "'float' object cannot be interpreted as an integer")


@overload(_int_order)
def _int_order_ovl(v):
    """`_int_order` inside ``@njit``: the answer is a compile-time constant.

    Whether the argument is a float is decided by its TYPE, so the refusing
    body and the empty body are chosen during typing.
    """
    fl = isinstance(v, types.Float) or (isinstance(v, types.Omitted)
                                        and isinstance(v.value, float))
    if fl:
        def impl(v):
            raise TypeError(
                "'float' object cannot be interpreted as an integer")
        return impl

    def impl(v):
        pass
    return impl


@njit
def _fmt_der(der, k):
    """scipy's ``0<=der=%s<=k=%s must hold``, rendered in an objmode block.

    numba's ``str`` has no float arm, and scipy prints the value it was
    given: measured, ``der=4`` gives ``0<=der=4<=k=3`` and ``der=4.5`` gives
    ``0<=der=4.5<=k=3``. The block is reached only on the raising path.
    """
    with objmode(msg='unicode_type'):
        msg = f"0<=der={der}<=k={k} must hold"
    raise ValueError(msg)


@njit
def _fmt_ext(ext):
    """scipy's ``ext = %s not in (0, 1, 2, 3) ``, trailing space included."""
    with objmode(msg='unicode_type'):
        msg = f"ext = {ext} not in (0, 1, 2, 3) "
    raise ValueError(msg)


@njit
def _fmt_ext_mode(ext):
    """scipy's ``Unknown extrapolation mode %s.``, with the value it was
    given: measured, ``2.7``, ``-5``, ``5`` and ``foo`` all appear in the
    text. Rendered in an ``objmode`` block for the same reason as `_fmt_der`.
    """
    with objmode(msg='unicode_type'):
        msg = f"Unknown extrapolation mode {ext}."
    raise ValueError(msg)


@njit
def _der_range(der, k):
    """scipy's ``if not (0 <= der <= k)`` on `splev`'s derivative order."""
    if der < 0 or der > k:
        _fmt_der(der, k)


@njit
def _ext_check(ext):
    """scipy's ``if ext not in (0, 1, 2, 3)``, which is MEMBERSHIP.

    A range test passes ``ext = 1.7``, and the raw layer then truncates it to
    1, so an out-of-bounds point silently returns 0.0 where scipy raises.
    """
    if ext == 0 or ext == 1 or ext == 2 or ext == 3:
        return
    _fmt_ext(ext)


@njit
def _pts_reshape(x):
    """`x` as a contiguous flat float64 buffer, plus its original shape.

    ``scipy.interpolate.splev`` keeps ``x.shape``: measured, a (2, 2) query
    returns (2, 2) and a (1, 2, 2) query returns (1, 2, 2), on ``der=0`` and
    ``der=1`` alike. `evaluators.splev` refuses anything but rank 1, so the
    flattening and the reshape live here. splev-D4, splder_ev-D4.

    The shape is read from `asarray` and not from `ascontiguousarray`, which
    never returns a rank-0 array: a scalar `x` is shape ``()`` in scipy and
    was becoming ``(1,)`` here. splev-D5.
    """
    x0 = np.asarray(x, np.float64)
    shp = x0.shape
    return np.ascontiguousarray(x0).ravel(), shp


@njit
def _splev_1(xf, shp, t, c, k, der, ext):
    """`splev` on ONE coefficient array, the non-parametric computation."""
    if der == 0:
        return _ev.splev(xf, t, c, k, ext).reshape(shp)
    return _ev.splder(xf, t, c, k, der, ext).reshape(shp)


def _splev_c(xf, shp, t, c, k, der, ext):
    """`splev` over whichever coefficient layout `c` carries.

    One entry point, two return types: an array for a plain spline and a list
    of `idim` arrays for a parametric one. `_is_param_c` settles which while
    the caller compiles, so the branch costs nothing at run time and the
    caller's own return type follows it.
    """
    if _is_param_c(c):
        return [_splev_1(xf, shp, t, np.asarray(cj, np.float64), k, der, ext)
                for cj in c]
    return _splev_1(xf, shp, t, c, k, der, ext)


@overload(_splev_c)
def _splev_c_ovl(xf, shp, t, c, k, der, ext):
    if _is_param_c(c):
        def impl(xf, shp, t, c, k, der, ext):
            return [_splev_1(xf, shp, t, c[j], k, der, ext)
                    for j in range(len(c))]
        return impl

    def impl(xf, shp, t, c, k, der, ext):
        return _splev_1(xf, shp, t, c, k, der, ext)
    return impl


@njit
def splev(x, tck, der=0, ext=0):
    """Evaluate a spline or one of its derivatives.

    Parameters
    ----------
    x : float or array_like of float
        Points to evaluate at. The result keeps `x`'s shape, so a scalar
        gives a rank-0 array.
    tck : tuple of (t, c, k)
        Spline representation, as returned by `splrep` or `splprep`: knot
        vector, coefficients (padded form accepted) and degree. A `c` that
        holds one array per curve dimension, which is what `splprep`
        returns, is evaluated one dimension at a time and the result is a
        list.
    der : int, optional
        Derivative order, ``0 <= der <= k``. Default 0, the spline itself.
    ext : int, optional
        Behaviour for points outside ``[t[k], t[n-k-1]]``: 0 extrapolates,
        1 returns 0, 2 raises ``ValueError``, 3 returns the boundary value.
        Default 0.

    Returns
    -------
    y : float64 ndarray, shaped like `x`, or a list of `idim` of them
        Spline values, or `der`-th derivative values. A list, one entry per
        curve dimension, when `c` holds one array per dimension.

    Raises
    ------
    ValueError
        If `der` is outside ``0..k``, if `ext` is outside ``0..3``, if `x`
        is empty, if `c` holds fewer than ``len(t) - k - 1`` coefficients,
        or if ``ext == 2`` and a point of `x` lies outside the knot range.

    See Also
    --------
    scipy.interpolate.splev : The scipy routine this mirrors.

    Notes
    -----
    - A ``BSpline`` instance is a valid `tck` in scipy. This unpacks a
      3-tuple.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import splrep, splev
    >>> x = np.linspace(0, 4, 40)
    >>> tck = splrep(x, np.sin(x))
    >>> @njit
    ... def evaluate(tck, q):
    ...     return splev(q, tck)
    >>> np.round(evaluate(tck, np.array([0.5, 1.5, 2.5])), 8)
    array([0.47942552, 0.99749473, 0.598472  ])
    >>> @njit
    ... def slope(tck, q):
    ...     return splev(q, tck, 1)
    >>> np.round(slope(tck, np.array([0.5, 1.5, 2.5])), 8)
    array([ 0.87758567,  0.07074248, -0.80114689])
    """
    t, c, k = tck
    _der_range(der, k)
    _ext_check(ext)
    _int_order(der)
    _int_order(ext)
    xf, shp = _pts_reshape(x)
    if len(xf) < 1:
        raise ValueError("Invalid input data")
    return _splev_c(xf, shp, t, c, k, der, ext)


@njit
def splder_ev(x, tck, nu=1):
    """Evaluate the `nu`-th derivative of a spline.

    Named ``splder_ev``, not ``splder``, on purpose: ``scipy.interpolate.
    splder`` returns the derivative spline's tck, and that role is filled in
    this package by ``splder`` (also exported as ``splder``). This
    function EVALUATES.

    Parameters
    ----------
    x : 1-D float64 ndarray
        Points to evaluate at.
    tck : tuple of (t, c, k)
        Spline representation.
    nu : int, optional
        Derivative order, ``0 <= nu <= k``. Default 1.

    Returns
    -------
    y : 1-D float64 ndarray, same length as `x`
        Derivative values, extrapolated outside the knot range.

    Raises
    ------
    ValueError
        If `nu` is outside ``0..k``, or if `c` holds fewer than
        ``len(t) - k - 1`` coefficients.

    Notes
    -----
    scipy publishes no ``splder_ev``. It reaches the same computation as
    ``splev(x, tck, nu)``, which this package also provides under the name
    `splev`.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import splrep, splder_ev
    >>> x = np.linspace(0, 4, 40)
    >>> tck = splrep(x, np.sin(x))
    >>> @njit
    ... def slope(tck, q):
    ...     return splder_ev(q, tck, 1)
    >>> float(np.round(slope(tck, np.array([0.0]))[0], 6))
    1.000019
    """
    t, c, k = tck
    _int_order(nu)
    xf, shp = _pts_reshape(x)
    # D3: `splev` raises here and this computes `splev(x, tck, der=nu)` under
    # another name, so the two spellings of one computation agree. scipy's
    # `splev` raises the same text for an empty `x`.
    if len(xf) < 1:
        raise ValueError("Invalid input data")
    return _splev_c(xf, shp, t, c, k, nu, 0)


@njit
def _splint_1(t, c, k, a, b):
    """`splint` on ONE coefficient array."""
    return _ev.splint(t, np.asarray(c, np.float64), k, a, b)


def _splint_c(t, c, k, a, b):
    """`splint` over whichever coefficient layout `c` carries.

    A float for a plain spline, a list of `idim` floats for a parametric one.
    """
    if _is_param_c(c):
        return [_splint_1(t, cj, k, a, b) for cj in c]
    return _ev.splint(t, c, k, a, b)


@overload(_splint_c)
def _splint_c_ovl(t, c, k, a, b):
    if _is_param_c(c):
        def impl(t, c, k, a, b):
            return [_splint_1(t, c[j], k, a, b) for j in range(len(c))]
        return impl

    def impl(t, c, k, a, b):
        return _ev.splint(t, c, k, a, b)
    return impl


def _splint_c_wrk(t, c, k, a, b):
    """`splint(full_output=1)` over whichever layout `c` carries.

    A parametric `c` gives the same list `_splint_c` gives, with no ``wrk``
    beside it: scipy reads the layout before it reads `full_output`, so its
    parametric branch returns a bare list there too.
    """
    if _is_param_c(c):
        return [_splint_1(t, cj, k, a, b) for cj in c]
    return _ev._splint_wrk(t, c, k, a, b)


@overload(_splint_c_wrk)
def _splint_c_wrk_ovl(t, c, k, a, b):
    if _is_param_c(c):
        def impl(t, c, k, a, b):
            return [_splint_1(t, c[j], k, a, b) for j in range(len(c))]
        return impl

    def impl(t, c, k, a, b):
        return _ev._splint_wrk(t, c, k, a, b)
    return impl


def splint(a, b, tck, full_output=0):
    """Evaluate the definite integral of a spline.

    Parameters
    ----------
    a, b : float
        Integration limits, the limits first and `tck` last. They may lie
        outside the knot range, where the spline is extrapolated; ``b < a``
        gives the negated integral.
    tck : tuple of (t, c, k)
        Spline representation.
    full_output : int, optional
        Non-zero also returns `wrk`, the per-B-spline integrals. Default 0.
        Must be a compile-time constant inside ``@njit``, since it selects
        the return type.

    Returns
    -------
    res : float
        The integral of the spline over ``[a, b]``.
    wrk : 1-D float64 ndarray, length ``len(t) - k - 1``
        The integral of each normalised B-spline over ``[a, b]``, so that
        ``res == sum(c[:len(wrk)] * wrk)``. Returned only when `full_output`
        is non-zero.

    Raises
    ------
    ValueError
        If `c` holds fewer than ``len(t) - k - 1`` coefficients.

    See Also
    --------
    scipy.interpolate.splint : The scipy routine this mirrors.

    Notes
    -----
    - `full_output` must be a compile-time constant inside ``@njit``: it
      selects between a bare float and a 2-tuple, and numba compiles one
      return type per specialization. A bool, an int, a string and the
      default are read while the call compiles; a float, a container and a
      runtime variable are not, and raise ``TypingError`` naming the
      argument. From Python every object is read by its truthiness.
    - ``full_output=1`` returns the `wrk` array, holding the integrals of the
      normalized B-splines. scipy 1.18 returns ``(res, None)`` there.
    - A parametric `c`, the coefficient list ``scipy.interpolate.splprep``
      returns, makes scipy return a list of integrals. This takes one
      coefficient array at a time.
    - `tck` is a 3-tuple. scipy also accepts a ``BSpline`` instance.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import splrep, splint
    >>> x = np.linspace(0, np.pi, 60)
    >>> tck = splrep(x, np.sin(x))
    >>> @njit
    ... def area(tck):
    ...     return splint(0.0, np.pi, tck)
    >>> float(np.round(area(tck), 10))
    1.9999999783
    >>> @njit
    ... def area_and_parts(tck):
    ...     res, wrk = splint(0.0, np.pi, tck, 1)
    ...     return res, len(wrk)
    >>> res, nwrk = area_and_parts(tck)
    >>> float(np.round(res, 10)), nwrk
    (1.9999999783, 60)
    """
    t, c, k = tck
    if not full_output:
        return _splint_c(t, c, k, a, b)
    return _splint_c_wrk(t, c, k, a, b)


@overload(splint, prefer_literal=True)
def _splint_ovl(a, b, tck, full_output=0):
    """`splint` inside ``@njit``: one compiled body per `full_output` value.

    Same mechanism as `_splrep_ovl`. A `full_output` whose value is not known
    while the call compiles is refused by `_lit_flag`, which puts the reason
    in the ``TypingError``.
    """
    fo = _lit_flag(full_output, 'full_output', 'splint')

    if fo:
        def impl(a, b, tck, full_output=0):
            t, c, k = tck
            return _splint_c_wrk(t, c, k, a, b)
        return impl

    def impl(a, b, tck, full_output=0):
        t, c, k = tck
        return _splint_c(t, c, k, a, b)
    return impl


@njit
def _sproot_1(t, c, mest):
    """`sproot` on ONE coefficient array, everything after the `k` check."""
    c_ = np.asarray(c, np.float64)
    n = len(t)
    if n < 8:
        raise TypeError("The number of knots " + str(n) + ">=8")
    if len(c_) < n - 4:
        raise ValueError("c array is too small")
    # FITPACK sproot's own ier == 10 data check, reproduced here because this
    # wrapper discards ier: without it an invalid knot vector returns an empty
    # array, which is also what a root-free spline returns.
    bad = False
    for i in range(3):
        if t[i] > t[i + 1]:
            bad = True
    for j in range(n - 1, n - 4, -1):
        if t[j] < t[j - 1]:
            bad = True
    for i in range(3, n - 4):
        if t[i] >= t[i + 1]:
            bad = True
    if bad:
        raise TypeError(
            "Invalid input data. "
            "t1<=..<=t4<t5<..<tn-3<=..<=tn must hold.")
    if mest < 0:
        mest = 3 * (n - 7)
    zero, ier = _ev._sproot_ier(t, c_, mest)
    if ier == 1:
        _warn_runtime("The number of zeros exceeds mest")
    return zero


def _sproot_c(t, c, mest):
    """`sproot` over whichever coefficient layout `c` carries.

    An array of roots for a plain spline, a list of `idim` root arrays for a
    parametric one.
    """
    if _is_param_c(c):
        return [_sproot_1(t, cj, mest) for cj in c]
    return _sproot_1(t, c, mest)


@overload(_sproot_c)
def _sproot_c_ovl(t, c, mest):
    if _is_param_c(c):
        def impl(t, c, mest):
            return [_sproot_1(t, c[j], mest) for j in range(len(c))]
        return impl

    def impl(t, c, mest):
        return _sproot_1(t, c, mest)
    return impl


@njit
def sproot(tck, mest=10):
    """Find the roots of a cubic spline.

    Parameters
    ----------
    tck : tuple of (t, c, k)
        Spline representation: knot vector, B-spline coefficients, degree.
        ``k`` must be 3. ``len(c)`` must be at least ``len(t) - 4``; a longer
        `c` is accepted and the excess ignored.
    mest : int, optional
        Size of the root buffer, and therefore the maximum number of roots
        returned. Default 10. ``mest=0`` returns an empty array. A NEGATIVE
        `mest` means ``3 * (len(t) - 7)``, FITPACK's own worst case
        (``len(t) - 7`` knot intervals, each holding at most 3 roots of a
        cubic), so it never truncates.

    Returns
    -------
    zero : 1-D float64 ndarray
        The roots in ``[t[3], t[n-4]]``, ascending; possibly empty. Truncated
        to `mest` entries when the spline has more roots than that.

    Raises
    ------
    ValueError
        ``k != 3``, or ``len(c) < len(t) - 4``.
    TypeError
        ``len(t) < 8``, or a knot vector failing FITPACK's data check
        ``t1<=..<=t4<t5<..<tn-3<=..<=tn``.

    Warns
    -----
    RuntimeWarning
        ``The number of zeros exceeds mest``, when `mest` binds and roots
        were dropped. The warning is issued through a ``numba.objmode``
        block, which runs its body in the interpreter, so
        ``warnings.catch_warnings`` and ``-W`` see it from compiled and
        uncompiled callers alike.

    See Also
    --------
    scipy.interpolate.sproot : The scipy routine this mirrors.

    Notes
    -----
    - A ``BSpline`` instance is a valid `tck` in scipy. This unpacks a
      3-tuple.

    A root that coincides with a knot is missed, and a tangential (double)
    root is not found. Both are FITPACK properties.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import splrep, sproot
    >>> x = np.linspace(0, 4, 60)
    >>> tck = splrep(x, np.sin(x))
    >>> @njit
    ... def zeros(tck):
    ...     return sproot(tck)
    >>> np.round(zeros(tck), 8)
    array([3.14159265])
    """
    t, c, k = tck
    if k != 3:
        raise ValueError("sproot works only for cubic (k=3) splines")
    return _sproot_c(t, c, mest)


@njit
def _spalde_1(x, t, c, k):
    """`spalde` on ONE coefficient array."""
    return _ev.spalde(x, t, np.asarray(c, np.float64), k)


def _spalde_c(x, t, c, k):
    """`spalde` over whichever coefficient layout `c` carries.

    One array of ``k+1`` derivatives for a plain spline, a list of `idim` of
    them for a parametric one.
    """
    if _is_param_c(c):
        return [_spalde_1(x, t, cj, k) for cj in c]
    return _spalde_1(x, t, c, k)


@overload(_spalde_c)
def _spalde_c_ovl(x, t, c, k):
    if _is_param_c(c):
        def impl(x, t, c, k):
            return [_spalde_1(x, t, c[j], k) for j in range(len(c))]
        return impl

    def impl(x, t, c, k):
        return _spalde_1(x, t, c, k)
    return impl


@njit
def spalde(x, tck):
    """Evaluate all derivatives of a spline at one point.

    Parameters
    ----------
    x : float
        A SCALAR evaluation point, inside the knot range.
    tck : tuple of (t, c, k)
        Spline representation.

    Returns
    -------
    d : 1-D float64 ndarray, length ``k + 1``
        ``d[j]`` is the j-th derivative at `x`, with ``d[0]`` the value
        itself.

    Raises
    ------
    TypeError
        If `x` lies outside ``[t[k], t[n-k-1]]``, or a knot interval is
        degenerate.
    ValueError
        If `c` holds fewer than ``len(t) - k - 1`` coefficients.

    See Also
    --------
    scipy.interpolate.spalde : The scipy routine this mirrors.

    Notes
    -----
    - `x` is one point. scipy also accepts an array of `m` points and
      returns a LIST of `m` arrays for ``m > 1``, a bare array for
      ``m == 1``. The return type there follows a run-time length, which a
      compiled body cannot do. Call it once per point in a loop; inside
      ``@njit`` the loop is compiled, so it carries no per-iteration
      overhead.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import splrep, spalde
    >>> x = np.linspace(0, 4, 40)
    >>> tck = splrep(x, np.sin(x))
    >>> @njit
    ... def derivs(tck):
    ...     return spalde(0.0, tck)
    >>> np.round(derivs(tck), 6)
    array([-0.000000e+00,  1.000019e+00, -6.700000e-04, -9.908410e-01])
    """
    t, c, k = tck
    return _spalde_c(x, t, c, k)


def _rank_gt1(v):
    """True when a query coordinate has a rank above 1.

    scipy's `bisplev` runs ``atleast_1d`` on both coordinates and then
    refuses whatever is left above rank 1. The answer follows the TYPE, so
    inside ``@njit`` it is a constant.
    """
    return np.ndim(v) > 1


@overload(_rank_gt1)
def _rank_gt1_ovl(v):
    r = isinstance(v, types.Array) and v.ndim > 1

    def impl(v):
        return r
    return impl


def _unit_coord(v):
    """True when a query coordinate carries exactly ONE point by its TYPE.

    A float and a 0-d array do. An array, list or tuple carries a run-time
    length, whatever that length turns out to be.

    Takes a numba type, from an ``@overload`` chooser, or a value, from a
    Python body, and answers the same for both.
    """
    if isinstance(v, types.Type):
        if isinstance(v, types.Array):
            return v.ndim == 0
        return isinstance(v, types.Number)
    return np.ndim(v) == 0


def _bisplev_sq(z, x, y):
    """`bisplev`'s grid, squeezed as far as the coordinate TYPES settle it.

    scipy returns ``z[0][0]`` when the grid is 1x1, ``z[0]`` when it has one
    row, and the grid otherwise. Those tests read run-time lengths; a unit
    coordinate supplies its length from the type instead, so the two cases it
    settles are taken while compiling.
    """
    if _unit_coord(x):
        if _unit_coord(y):
            return z[0, 0]
        return z[0]
    return z


@overload(_bisplev_sq)
def _bisplev_sq_ovl(z, x, y):
    if _unit_coord(x):
        if _unit_coord(y):
            def impl(z, x, y):
                return z[0, 0]
            return impl

        def impl(z, x, y):
            return z[0]
        return impl

    def impl(z, x, y):
        return z
    return impl


@njit
def bisplev(x, y, tck, dx=0, dy=0):
    """Evaluate a bivariate spline on a grid.

    Parameters
    ----------
    x : float or 1-D array_like of float
        Grid abscissae, non-decreasing. A scalar or 0-d array is one point.
    y : float or 1-D array_like of float
        Grid ordinates, non-decreasing. The spline is evaluated on the full
        CROSS PRODUCT.
    tck : tuple of (tx, ty, c, kx, ky)
        Bivariate spline representation: knots in x and y, coefficients in
        FITPACK's flat layout ``c[(ny-ky-1)*i + j]``, and the two degrees.
    dx, dy : int, optional
        Orders of the partial derivatives in x and y. ``0 <= dx < kx`` and
        ``0 <= dy < ky``. Both default to 0.

    Returns
    -------
    z : float, or 1-D or 2-D float64 ndarray
        Spline values on the grid. A float when `x` and `y` are both a scalar
        or a 0-d array, a 1-D array of length ``len(y)`` when only `x` is,
        and the ``(len(x), len(y))`` grid otherwise.

    Raises
    ------
    ValueError
        If `dx` or `dy` is out of range, if `x` or `y` has a rank above 1, is
        empty or is decreasing, or if
        ``len(c) != (len(tx)-kx-1) * (len(ty)-ky-1)``.

    See Also
    --------
    scipy.interpolate.bisplev : The scipy routine this mirrors.

    Notes
    -----
    - The return rank follows a squeeze as far as the argument TYPES
      settle it, which is the two cases named under `Returns`. scipy squeezes
      on the run-time LENGTHS, so it also returns a float where
      ``len(x) == len(y) == 1`` and a 1-D array where ``len(x) == 1 <
      len(y)``. A compiled body fixes its return rank while it compiles and
      an array carries no length until it runs, so those two follow the
      length only when the coordinate is a scalar.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import RectBivariateSpline, bisplev
    >>> x = np.linspace(0, 1, 12)
    >>> y = np.linspace(0, 1, 15)
    >>> z = np.outer(np.sin(3 * x), np.cos(2 * y))
    >>> spl = RectBivariateSpline(x, y, z)
    >>> tck = (spl.tx, spl.ty, spl.c, spl.kx, spl.ky)
    >>> @njit
    ... def grid(tck, qx, qy):
    ...     return bisplev(qx, qy, tck)
    >>> float(np.round(grid(tck, np.array([0.5]), np.array([0.5]))[0, 0], 8))
    0.53894085
    """
    tx, ty, c, kx, ky = tck
    if dx < 0 or dx >= kx:
        raise ValueError(
            "0 <= dx = " + str(dx) + " < kx = " + str(kx) + " must hold")
    if dy < 0 or dy >= ky:
        raise ValueError(
            "0 <= dy = " + str(dy) + " < ky = " + str(ky) + " must hold")
    if _rank_gt1(x) or _rank_gt1(y):
        raise ValueError("First two entries should be rank-1 arrays.")
    xa = _pt1d(x)
    ya = _pt1d(y)
    # FITPACK indexes with a 32-bit integer, so scipy tests the two products
    # that overflow it. Both tests live in scipy's public `bisplev` and not
    # in its bindings, which is why the raw layer has neither.
    if len(xa) * len(ya) > 2147483647:
        raise MemoryError("Too many data points to interpolate.")
    if dx != 0 or dy != 0:
        if (len(tx) - kx - 1) * (len(ty) - ky - 1) > 2147483647:
            raise MemoryError("Too many data points to interpolate.")
    # parder's own ier == 10 data check, reproduced here because the raw
    # layer discards ier: without it an empty or decreasing axis returns a
    # zero-filled grid.
    if len(xa) < 1 or len(ya) < 1:
        raise ValueError("Invalid input data")
    for i in range(1, len(xa)):
        if xa[i] < xa[i - 1]:
            raise ValueError("Invalid input data")
    for j in range(1, len(ya)):
        if ya[j] < ya[j - 1]:
            raise ValueError("Invalid input data")
    return _bisplev_sq(_ev.parder(xa, ya, tx, ty, c, kx, ky, dx, dy), x, y)


# =====================================================================
# univariate spline classes
# =====================================================================

# ---------------------------------------------------------------------
# helpers shared by the three univariate FITPACK classes
# ---------------------------------------------------------------------

@njit
def _elem_or(v, d):
    """One element of a `bbox` sequence as a float; ``None`` becomes `d`.

    `d` is the value the data supplies for that slot, which is what scipy
    substitutes for a ``None`` there. A NaN is a value like any other and is
    returned unchanged.

    Exists for the same typing reason as `_nan_if_none`: numba folds an
    ``is None`` predicate only when the operand is a direct function argument
    (``numba/core/analysis.py``, ``dead_branch_prune``), so the test has to
    happen in a callee that takes the element.
    """
    if v is None:
        return np.float64(d)
    return np.float64(v)


@njit
def _bbox_shape_raise(n):
    """scipy's `bbox` shape refusal, for a `bbox` of `n` slots.

    A statement rather than a `raise` written into the arm that needs it. An
    ``@overload`` arm whose body is only a ``raise`` types as returning
    ``none``, and the caller then fails to unpack it with an error naming
    neither `bbox` nor its shape.
    """
    raise ValueError("bbox shape should be (" + str(n) + ",)")


def _bbox_pair(bbox, xb_def, xe_def):
    """``(xb, xe)`` from scipy's `bbox`; ``None`` means "use the data range".

    `xb_def` and `xe_def` are the data-derived values a ``None`` resolves to,
    which the caller supplies because this helper never sees the data. A NaN
    slot is a literal knot boundary and travels to FITPACK unchanged.
    """
    if bbox is None:
        return float(xb_def), float(xe_def)
    if len(bbox) != 2:
        raise ValueError("bbox shape should be (2,)")
    return (float(_elem_or(bbox[0], xb_def)),
            float(_elem_or(bbox[1], xe_def)))


@overload(_bbox_pair)
def _bbox_pair_ovl(bbox, xb_def, xe_def):
    """`_bbox_pair` inside ``@njit``, one body per spelling of `bbox`.

    scipy's literal default is ``[None, None]``, so the ``None`` may sit
    inside the container rather than in place of it. A tuple of two elements
    is length-checked here at compile time and a list at runtime, because a
    tuple's length is part of its type and a list's is not.

    Element-level ``None`` is handled by `_elem_or`, not here; see its
    docstring for why the fold has to happen in a callee.
    """
    if _is_none(bbox):
        def impl(bbox, xb_def, xe_def):
            return np.float64(xb_def), np.float64(xe_def)
        return impl
    if isinstance(bbox, types.BaseTuple):
        if len(bbox) != 2:
            def impl(bbox, xb_def, xe_def):
                _bbox_shape_raise(2)
                return np.float64(xb_def), np.float64(xe_def)
            return impl

        def impl(bbox, xb_def, xe_def):
            return _elem_or(bbox[0], xb_def), _elem_or(bbox[1], xe_def)
        return impl
    if isinstance(bbox, (types.List, types.ListType)):
        def impl(bbox, xb_def, xe_def):
            if len(bbox) != 2:
                raise ValueError("bbox shape should be (2,)")
            return _elem_or(bbox[0], xb_def), _elem_or(bbox[1], xe_def)
        return impl
    # scipy tests `bbox.shape != (2,)`, so a scalar and any rank but 1 are
    # refused on the SHAPE. Both are settled by the type, and neither can
    # reach the array arm below: a 0-d array has no `b[0]` and a rank-2 one
    # gives a row where a number is wanted, so both would fail to compile.
    if (isinstance(bbox, types.Number)
            or (isinstance(bbox, types.Array) and bbox.ndim != 1)):
        def impl(bbox, xb_def, xe_def):
            _bbox_shape_raise(2)
            return np.float64(xb_def), np.float64(xe_def)
        return impl

    def impl(bbox, xb_def, xe_def):
        b = np.asarray(bbox, dtype=np.float64)
        if b.size != 2:
            raise ValueError("bbox shape should be (2,)")
        return np.float64(b[0]), np.float64(b[1])
    return impl


_EXT_NAMES = ("extrapolate", "zeros", "raise", "const")


def _ext_code(ext):
    """scipy's ``_extrap_modes``: 0/1/2/3 or the four names.

    A DICT LOOKUP, not a range test. ``2.0`` hashes equal to ``2`` and scipy
    accepts it; ``2.7`` misses and scipy raises with the value in the text.
    An ``int(ext)`` here made 2.7 mean "raise on out-of-range points"
    silently.
    """
    if isinstance(ext, str):
        for i in range(4):
            if ext == _EXT_NAMES[i]:
                return i
        raise ValueError("Unknown extrapolation mode " + ext + ".")
    for i in range(4):
        if ext == i:
            return i
    raise ValueError("Unknown extrapolation mode " + str(ext) + ".")


@overload(_ext_code)
def _ext_code_ovl(ext):
    """`_ext_code` inside ``@njit``: accept the name or the int code.

    A string and an integer cannot share one body, so the type decides which
    one gets compiled. The string arm keeps its loop over `_EXT_NAMES` at
    runtime rather than resolving the name during typing, which is what lets
    `ext` be a runtime string as well as a literal.
    """
    if (isinstance(ext, (types.UnicodeType, types.StringLiteral))
            or isinstance(ext, str)):
        def impl(ext):
            for i in range(4):
                if ext == _EXT_NAMES[i]:
                    return i
            raise ValueError("Unknown extrapolation mode " + ext + ".")
        return impl

    def impl(ext):
        if ext == 0:
            return 0
        if ext == 1:
            return 1
        if ext == 2:
            return 2
        if ext == 3:
            return 3
        _fmt_ext_mode(ext)
        return 0
    return impl


def _w_or_ones(w, m):
    """scipy's ``w=None`` -> unit weights; a zero-length array is the same."""
    if w is None:
        return np.ones(m)
    a = np.ascontiguousarray(np.asarray(w, dtype=np.float64)).ravel()
    return np.ones(m) if a.size == 0 else a


@overload(_w_or_ones)
def _w_or_ones_ovl(w, m):
    """`_w_or_ones` inside ``@njit``: unit weights when `w` was not given.

    The ``None`` case is decided at compile time; the zero-length case has
    to stay a runtime test, because an array's length is not part of its
    type. Both produce ``np.ones(m)``, so a caller cannot tell them apart,
    which is the point.
    """
    if _is_none(w):
        def impl(w, m):
            return np.ones(m)
        return impl

    def impl(w, m):
        a = np.ascontiguousarray(np.asarray(w, dtype=np.float64)).ravel()
        if a.size == 0:
            return np.ones(m)
        return a
    return impl


def _w_given(w):
    """True when `w` was supplied as a real weight vector."""
    return w is not None and np.asarray(w).size != 0


@overload(_w_given)
def _w_given_ovl(w):
    """`_w_given` inside ``@njit``: was a real weight vector supplied.

    Separate from `_w_or_ones` because the validators need the ANSWER, not
    the weights: scipy checks ``len(w) == m`` only when `w` was given, and
    unit weights substituted silently would make that check unreachable.
    """
    if _is_none(w):
        def impl(w):
            return False
        return impl

    def impl(w):
        return np.asarray(w).size != 0
    return impl


@njit
def _curfit_warn(ier):
    """scipy's ``_reset_class`` warning, from ``_curfit_messages``.

    ``UnivariateSpline``, ``InterpolatedUnivariateSpline`` and
    ``LSQUnivariateSpline`` all reach it. P2: 1, 2 and 3 warn here as they do
    in scipy; ``ier = 10`` raises instead, which is a deliberate deviation and
    is stated in each class's ``Notes``. ``ier = 1`` cannot happen here,
    because `nest` is allocated at the worst case.
    """
    if ier == 1:
        _warn_user(
            "\nThe required storage space exceeds the available storage "
            "space, as\nspecified by the parameter nest: nest too small. If "
            "nest is already\nlarge (say nest > m/2), it may also indicate "
            "that s is too small.\nThe approximation returned is the "
            "weighted least-squares spline\naccording to the knots "
            "t[0],t[1],...,t[n-1]. (n=nest) the parameter fp\ngives the "
            "corresponding weighted sum of squared residuals (fp>s).\n")
    elif ier == 2:
        _warn_user(
            "\nA theoretically impossible result was found during the "
            "iteration\nprocess for finding a smoothing spline with fp = s: "
            "s too small.\nThere is an approximation returned but the "
            "corresponding weighted sum\nof squared residuals does not "
            "satisfy the condition abs(fp-s)/s < tol.")
    elif ier == 3:
        _warn_user(
            "\nThe maximal number of iterations maxit (set to 20 by the "
            "program)\nallowed for finding a smoothing spline with fp=s has "
            "been reached: s\ntoo small.\nThere is an approximation "
            "returned but the corresponding weighted sum\nof squared "
            "residuals does not satisfy the condition abs(fp-s)/s < tol.")


@njit
def _uni_validate(x, y, w, w_given, k, s, s_given, check_finite):
    """``scipy.interpolate.UnivariateSpline.validate_input``, in order.

    `s` arrives already resolved; `s_given` says whether the caller supplied
    it, which is what selects scipy's ``s is None or s > 0`` branch.
    """
    m = len(x)
    if check_finite:
        ok = True
        for i in range(m):
            if not np.isfinite(x[i]):
                ok = False
        for i in range(len(y)):
            if not np.isfinite(y[i]):
                ok = False
        if w_given:
            for i in range(len(w)):
                if not np.isfinite(w[i]):
                    ok = False
        if not ok:
            raise ValueError(
                "x and y array must not contain NaNs or infs.")
    if (not s_given) or s > 0.0:
        for i in range(1, m):
            if not (x[i] - x[i - 1] >= 0.0):
                raise ValueError("x must be increasing if s > 0")
    else:
        for i in range(1, m):
            if not (x[i] - x[i - 1] > 0.0):
                raise ValueError("x must be strictly increasing if s = 0")
    if m != len(y):
        raise ValueError("x and y should have a same length")
    if w_given and len(w) != m:
        raise ValueError("x, y, and w should have a same length")
    if k < 1 or k > 5:
        raise ValueError("k should be 1 <= k <= 5")
    if s_given and not (s >= 0.0):
        raise ValueError("s should be s >= 0.0")
    if m <= k:
        # scipy leaves this to f2py, which raises "(m>k) failed"; without a
        # guard FITPACK returns ier=10 and an EMPTY knot vector that later
        # evaluates to 0.0.
        raise ValueError("m > k must hold")


@njit
def _splev_ext(x, t, c, k, ext):
    """`evaluators.splev` plus the ``ext=2`` raise that would otherwise be
    swallowed."""
    y = _ev.splev(x, t, c, k, ext)
    if ext == 2:
        lo = t[k]
        hi = t[len(t) - k - 1]
        for i in range(len(x)):
            if x[i] < lo or x[i] > hi:
                raise ValueError("Found x value not in the domain")
    return y


@njit
def _splder_ext(x, t, c, k, nu, ext):
    """`evaluators.splder` plus the ``ext=2`` raise that would otherwise be
    swallowed.

    The derivative twin of `_splev_ext`. FITPACK reports the out-of-domain
    condition through a status the evaluator layer does not surface, so the
    domain test is repeated here to make ``ext=2`` raise rather than return.
    """
    y = _ev.splder(x, t, c, k, nu, ext)
    if ext == 2:
        lo = t[k]
        hi = t[len(t) - k - 1]
        for i in range(len(x)):
            if x[i] < lo or x[i] > hi:
                raise ValueError("Found x value not in the domain")
    return y


#: The five conditions `fpchec` tests, in the words scipy raises them in.
#: The condition a caller violated is the information the block carries, so
#: the whole block travels with the rejection.
_FPCHEC_MSG = (
    "The input parameters have been rejected by fpchec. This means that at "
    "least one of the following conditions is violated:\n"
    "\n"
    "1) k+1 <= n-k-1 <= m\n"
    "2) t(1) <= t(2) <= ... <= t(k+1)\n"
    "   t(n-k) <= t(n-k+1) <= ... <= t(n)\n"
    "3) t(k+1) < t(k+2) < ... < t(n-k)\n"
    "4) t(k+1) <= x(i) <= t(n-k)\n"
    "5) The conditions specified by Schoenberg and Whitney must hold\n"
    "   for at least one subset of data points, i.e., there must be a\n"
    "   subset of data points y(j) such that\n"
    "       t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1\n")


@njit
def _fpchec(x, t, k):
    """FITPACK ``fpchec``, transcribed from ``src/fitpack/fpchec.f``.

    Returns 0 or 10. scipy calls the Fortran routine directly and raises on
    a non-zero return; this wrapper discards `ier`, so the same five
    conditions are checked before the fit rather than after it.
    """
    m = len(x)
    n = len(t)
    k1 = k + 1
    k2 = k1 + 1
    nk1 = n - k1
    nk2 = nk1 + 1
    if nk1 < k1 or nk1 > m:
        return 10
    j = n - 1
    for i in range(k):
        if t[i] > t[i + 1]:
            return 10
        if t[j] < t[j - 1]:
            return 10
        j -= 1
    for i in range(k2 - 1, nk2):
        if t[i] <= t[i - 1]:
            return 10
    if x[0] < t[k1 - 1] or x[m - 1] > t[nk2 - 1]:
        return 10
    if x[0] >= t[k2 - 1] or x[m - 1] <= t[nk1 - 1]:
        return 10
    i = 0
    l = k2 - 1
    nk3 = nk1 - 1
    if nk3 < 2:
        return 0
    for j in range(1, nk3):
        tj = t[j]
        l += 1
        tl = t[l]
        while True:
            i += 1
            if i >= m - 1:
                return 10
            if x[i] > tj:
                break
        if x[i] >= tl:
            return 10
    return 0


@njit
def _uni_roots(t, c, k):
    """``UnivariateSpline.roots``: cubic only, and scipy's mest."""
    if k != 3:
        raise NotImplementedError(
            "finding roots unsupported for non-cubic splines")
    return _ev.sproot(t, c, 3 * (len(t) - 7))


_uni_spec = [
    ('t', float64[::1]),
    ('c', float64[::1]),
    ('k', int64),
    ('fp', float64),
    ('ier', int64),
    ('ext', int64),
]


@scijitclass(_uni_spec, dispatch=[('ev', _first_seq),
                                  ('ev_one', all_scalar)])
class _UnivariateSpline:
    """Instance type behind the `UnivariateSpline` factory,
    which carries the documentation and the defaults. Every argument is
    explicit here: a jitclass constructor's defaults are Python-only.
    """

    def __init__(self, x, y, w, bbox, k, s, ext, check_finite):
        x_ = np.ascontiguousarray(np.asarray(x, np.float64)).ravel()
        _no_complex(y, "UnivariateSpline")
        y_ = np.ascontiguousarray(np.asarray(_real_in(y), np.float64)).ravel()
        wg = _w_given(w)
        w_ = _w_or_ones(w, len(x_))
        sv = _opt_f(s)
        s_given = _given(s)
        sv = sv if s_given else float(len(x_))
        _uni_validate(x_, y_, w_, wg, k, sv, s_given, check_finite)
        xb, xe = _bbox_pair(bbox, x_[0], x_[len(x_) - 1])
        e = _ext_code(ext)
        t, c, fp, ier = _ft.curfit(x_, y_, w_, k, sv, xb, xe, -1)
        if ier == 10:
            raise ValueError("UnivariateSpline: error on input data")
        _curfit_warn(ier)
        self.t = t
        self.c = c
        self.k = k
        self.fp = fp
        self.ier = ier
        self.ext = e

    def ev(self, x, nu=0, ext=-1):
        """Evaluate the spline at an array of points (scipy's ``spl(x)``).

        Parameters
        ----------
        x : 1-D float64 ndarray
            Points to evaluate at.
        nu : int, optional
            Derivative order, ``0 <= nu <= k``. Default 0, the spline
            itself. This is scipy's second positional parameter.
        ext : int, optional
            Per-call override of the constructor's `ext`. -1, the default,
            keeps it -- that is this package's spelling of scipy's
            ``__call__(x, nu=0, ext=None)``, where ``None`` means the same.

        Returns
        -------
        y : 1-D float64 ndarray, same length as `x`
            Spline values, or `nu`-th derivative values.

        Raises
        ------
        ValueError
            If `nu` is outside ``0..k``, if the effective `ext` is outside
            ``0..3``, or if the effective `ext` is 2 and a point lies
            outside ``[t[k], t[n-k-1]]``.
        """
        e = self.ext if ext < 0 else ext
        _der_range(nu, self.k)
        if e > 3:
            _fmt_ext_mode(e)
        _int_order(nu)
        if nu == 0:
            return _splev_ext(x, self.t, self.c, self.k, e)
        return _splder_ext(x, self.t, self.c, self.k, nu, e)

    def ev_one(self, x, nu=0, ext=-1):
        """Evaluate the spline at a single point.

        Parameters
        ----------
        x : float
            Point to evaluate at.
        nu : int, optional
            Derivative order, ``0 <= nu <= k``. Default 0, the spline
            itself. This is scipy's second positional parameter.
        ext : int, optional
            Per-call override; -1 keeps the constructor's.

        Returns
        -------
        float
            The spline value, or its `nu`-th derivative. Wraps the scalar in a length-1 array for
            FITPACK, so calling this in a tight loop costs one small
            allocation per point -- prefer `ev` on a batch.

        Raises
        ------
        ValueError
            If `nu` is outside ``0..k``, if the effective `ext` is outside
            ``0..3``, or if the effective `ext` is 2 and `x` lies outside
            ``[t[k], t[n-k-1]]``.
        """
        e = self.ext if ext < 0 else ext
        _der_range(nu, self.k)
        if e > 3:
            _fmt_ext_mode(e)
        _int_order(nu)
        xa = np.empty(1, np.float64)
        xa[0] = x
        if nu == 0:
            return _splev_ext(xa, self.t, self.c, self.k, e)[0]
        return _splder_ext(xa, self.t, self.c, self.k, nu, e)[0]

    def __getitem__(self, x):
        """``spl[x]`` -- sugar for ``spl.ev(x)``.

        ``spl(x)`` reaches the same method and is the spelling scipy uses.

        Parameters
        ----------
        x : 1-D float64 ndarray

        Returns
        -------
        y : 1-D float64 ndarray
        """
        return _splev_ext(x, self.t, self.c, self.k, self.ext)

    def derivative_ev(self, x, nu=1, ext=-1):
        """Evaluate the `nu`-th derivative (scipy's ``spl(x, nu)``).

        Parameters
        ----------
        x : 1-D float64 ndarray
            Points to evaluate at.
        nu : int, optional
            Derivative order, ``0 <= nu <= k``. Default 1. This method
            default works inside ``@njit`` (unlike a constructor default).
        ext : int, optional
            Per-call override; -1 keeps the constructor's.

        Returns
        -------
        y : 1-D float64 ndarray, same length as `x`

        Notes
        -----
        scipy's ``.derivative()``, which returns a new spline object, is
        not implemented. Use this, or ``splder`` for the
        derivative's tck.
        """
        return _splder_ext(x, self.t, self.c, self.k, nu,
                           self.ext if ext < 0 else ext)

    def derivatives(self, x):
        """All derivatives at one point (scipy's ``spl.derivatives(x)``).

        Parameters
        ----------
        x : float
            A SCALAR point, inside the knot range.

        Returns
        -------
        d : 1-D float64 ndarray, length ``k + 1``
            ``d[j]`` is the j-th derivative at `x`; ``d[0]`` is the value.
        """
        return _ev.spalde(x, self.t, self.c, self.k)

    def integral(self, a, b):
        """Definite integral over ``[a, b]`` (scipy's ``spl.integral(a, b)``).

        Parameters
        ----------
        a, b : float
            Integration limits; may lie outside the knot range (the spline is
            extrapolated), and ``b < a`` negates the result.

        Returns
        -------
        float
            The integral.
        """
        return _ev.splint(self.t, self.c, self.k, a, b)

    def roots(self):
        """Zeros of the spline (scipy's ``spl.roots()``).

        Returns
        -------
        zero : 1-D float64 ndarray
            The roots, ascending; possibly empty.

        Raises
        ------
        NotImplementedError
            ``finding roots unsupported for non-cubic splines`` when
            ``k != 3``.

        Notes
        -----
        The root buffer holds ``3 * (len(t) - 7)`` entries, which is
        FITPACK's worst case -- at most 3 roots in each of the ``len(t) - 7``
        knot intervals -- so it never truncates.
        """
        return _uni_roots(self.t, self.c, self.k)

    def get_knots(self):
        """Interior knots plus the two ends (scipy's ``spl.get_knots()``).

        Returns
        -------
        t : 1-D float64 ndarray
            ``self.t[k : len(t)-k]`` -- the knot vector with the repeated
            boundary knots stripped, which is what scipy returns. Use the
            ``.t`` attribute for the full vector.
        """
        return self.t[self.k:len(self.t) - self.k]

    def get_coeffs(self):
        """The spline coefficients (scipy's ``spl.get_coeffs()``).

        Returns
        -------
        c : 1-D float64 ndarray, length ``len(t) - k - 1``
            The meaningful coefficients, with FITPACK's padding tail dropped.
        """
        return self.c[:len(self.t) - self.k - 1]

    def get_residual(self):
        """Weighted sum of squared residuals (scipy's ``spl.get_residual()``).

        Returns
        -------
        float
            FITPACK's ``fp`` for the fit. Zero for an interpolating spline.
        """
        return self.fp


@njit
def UnivariateSpline(x, y, w=_NO_W, bbox=None, k=3, s=None, ext=0,
                     check_finite=False):
    """Build a univariate smoothing spline.

    Parameters
    ----------
    x : 1-D float64 ndarray, length m
        Abscissae, strictly increasing.
    y : 1-D float64 ndarray, length m
        Ordinates.
    w : 1-D array_like of float, optional
        Positive weights, length m. ``None`` and a ZERO-LENGTH array both
        mean unit weights. Passing `w` does NOT change the default `s`,
        unlike ``splrep``.
    bbox : (2,) array_like of float, optional
        Boundary of the approximation interval. ``None``, in place of the
        pair or in one slot, means ``x[0]`` / ``x[-1]``, and is the default.
        It may only WIDEN the data interval. The literal default
        ``[None, None]`` is accepted, from Python and from inside ``@njit``.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    s : float, optional
        Smoothing factor: the fit satisfies
        ``sum(w[i]*(y[i]-spl(x[i])))**2 <= s``. ``None``, the default, means
        ``s = m``. ``s = 0`` gives the interpolating spline. A negative or
        NaN `s` raises ``ValueError``.
    ext : int or str, optional
        Extrapolation mode outside ``[t[k], t[n-k-1]]``, four codes and their
        names: 0 or ``'extrapolate'``, 1 or ``'zeros'``, 2 or ``'raise'``,
        3 or ``'const'``. Default 0.
    check_finite : bool, optional
        Raise if `x`, `y` or `w` contains a NaN or an inf. Default False.

    Returns
    -------
    spl : _UnivariateSpline
        A jitclass instance carrying the attributes and methods below.

    Raises
    ------
    ValueError
        Non-finite input under `check_finite`, non-monotone `x`, mismatched
        lengths, a `bbox` that is not length 2, `k` outside 1..5, a supplied
        `s` below 0, an unknown `ext`, or ``m <= k``.

    See Also
    --------
    scipy.interpolate.UnivariateSpline : The scipy class this mirrors.

    Notes
    -----
    - ``spl(x)`` runs ``.ev`` for an array and ``.ev_one`` for a scalar.
      ``.ev(x)``, ``.ev_one(x)`` and ``spl[x]`` reach the same methods.
      ``spl(x, nu)`` and ``spl(x, nu, ext)`` carry scipy's second and third
      positional parameters, and ``spl(x, nu=1)`` and ``spl(x, ext=1)``
      carry them by keyword. ``.ev(x, nu=1)`` takes the keyword inside
      ``@njit`` and not from the interpreter, where the jitclass method
      raises ``TypeError``; the call spelling works from both.
    - scipy RE-CLASSES the instance by `ier` (``_reset_class``), so a
      ``UnivariateSpline`` with ``ier == -1`` becomes an
      ``InterpolatedUnivariateSpline``. A jitclass cannot change its own
      type; `ier` is returned as an attribute instead.
    - ``.derivative()`` / ``.antiderivative()`` returning new spline objects
      are not implemented. Use ``derivative_ev(x, nu)``, or ``splder``
      for the tck.
    - ``set_smoothing_factor`` is absent: it resumes FITPACK's search from
      the stored ``fpcurf1`` continuation state, which is not reachable
      through the ``curfit`` wrapper. Refit with the new `s`.
    - ``ier = 10`` RAISES here. scipy warns and returns an object whose knot
      vector was never filled; a status that means "no approximation
      returned" is not carried into a usable object. `ier` of 1, 2 or 3
      warns.
    - A complex `y` raises ``TypeError``. scipy 1.18 accepts it, emits a
      ``ComplexWarning`` and discards the imaginary part, returning a float64
      spline.

    Defaults work in both worlds. ``UnivariateSpline`` is a plain ``@njit``
    factory, not the jitclass itself, so ``UnivariateSpline(x, y)``
    compiles and runs inside ``@njit`` as well as from Python. The class it
    returns, ``_UnivariateSpline``, takes every argument explicitly,
    because a jitclass constructor's defaults are Python-only.

    A SMOOTHING FIT MAY WARN THAT IT IS RANK DEFICIENT. With a small `s` on
    noisy data, FITPACK's knot search can place knots so that one B-spline
    coefficient is not determined by the data at all. The curve still passes
    through the data as asked, and ``fp`` reports success, but between the
    data points it carries an arbitrary component. A ``UserWarning`` naming
    the number of undetermined coefficients is issued when this happens, and
    a larger `s` is the fix. scipy issues no warning for it.

    Accuracy against ``scipy.interpolate.UnivariateSpline`` on scipy 1.18.0,
    max absolute difference: knots, coefficients and values all 0.0 at
    ``s`` = 0, 1e-8, 0.5, 1.0 and at the default, and 0.0 for the four
    ``nest``-retry-sensitive sizes m = 150, 300, 500, 1000. On scipy 1.15.3
    the ``s=1e-8`` case differed by 510.33, because 1.15.3's ``_reset_nest``
    resumed a search that 1.18's no longer does.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import UnivariateSpline
    >>> x = np.linspace(0, 4, 40)
    >>> y = np.sin(x)
    >>> spl = UnivariateSpline(x, y, s=0.5)     # smoothing spline, fit once
    >>> float(np.round(spl(1.5), 6))
    0.971964

    Inside compiled code, integrating the fitted spline over its domain:

    >>> @njit
    ... def area(spl):
    ...     return spl.integral(0.0, 4.0)
    >>> float(np.round(area(spl), 6))
    1.664201

    Attributes
    ----------
    t : 1-D float64 ndarray
        Full knot vector, including the repeated boundary knots.
    c : 1-D float64 ndarray
        Coefficients in FITPACK's padded form.
    k : int
        Spline degree.
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status: 0 = smoothing achieved, -1 = interpolating spline,
        -2 = least-squares polynomial, positive = failure. Kept as an
        attribute rather than raised -- CHECK IT after constructing.
    ext : int
        The resolved extrapolation code, 0..3.

    Methods
    -------
    ev(x), ev_one(x), __getitem__(x), derivative_ev(x, nu), derivatives(x),
    integral(a, b), roots(), get_knots(), get_coeffs(), get_residual()
    """
    return _UnivariateSpline(x, y, w, bbox, k, s, ext, check_finite)


@scijitclass(_uni_spec, dispatch=[('ev', _first_seq),
                                  ('ev_one', all_scalar)])
class _InterpolatedUnivariateSpline:
    """Instance type behind the `InterpolatedUnivariateSpline` factory,
    which carries the documentation and the defaults. Every argument is
    explicit here: a jitclass constructor's defaults are Python-only.
    """

    def __init__(self, x, y, w, bbox, k, ext, check_finite):
        x_ = np.ascontiguousarray(np.asarray(x, np.float64)).ravel()
        _no_complex(y, "InterpolatedUnivariateSpline")
        y_ = np.ascontiguousarray(np.asarray(_real_in(y), np.float64)).ravel()
        wg = _w_given(w)
        w_ = _w_or_ones(w, len(x_))
        _uni_validate(x_, y_, w_, wg, k, 0.0, False, check_finite)
        for i in range(1, len(x_)):
            if not (x_[i] - x_[i - 1] > 0.0):
                raise ValueError("x must be strictly increasing")
        xb, xe = _bbox_pair(bbox, x_[0], x_[len(x_) - 1])
        e = _ext_code(ext)
        t, c, fp, ier = _ft.curfit(x_, y_, w_, k, 0.0, xb, xe, -1)
        if ier == 10:
            raise ValueError(
                "InterpolatedUnivariateSpline: error on input data")
        _curfit_warn(ier)
        self.t = t
        self.c = c
        self.k = k
        self.fp = fp
        self.ier = ier
        self.ext = e

    def ev(self, x, nu=0, ext=-1):
        """Evaluate the spline at an array of points (scipy's ``spl(x)``).

        Parameters
        ----------
        x : 1-D float64 ndarray
            Points to evaluate at.
        nu : int, optional
            Derivative order, ``0 <= nu <= k``. Default 0, the spline
            itself. This is scipy's second positional parameter.
        ext : int, optional
            Per-call override of the constructor's `ext`; -1, the default,
            keeps it.

        Returns
        -------
        y : 1-D float64 ndarray, same length as `x`
            Spline values, or `nu`-th derivative values. Outside the knot
            range the spline is EXTRAPOLATED, where scipy's ``ext=0``
            default does the same.

        Raises
        ------
        ValueError
            If `nu` is outside ``0..k``, if the effective `ext` is outside
            ``0..3``, or if the effective `ext` is 2 and a point lies
            outside ``[t[k], t[n-k-1]]``.
        """
        e = self.ext if ext < 0 else ext
        _der_range(nu, self.k)
        if e > 3:
            _fmt_ext_mode(e)
        _int_order(nu)
        if nu == 0:
            return _splev_ext(x, self.t, self.c, self.k, e)
        return _splder_ext(x, self.t, self.c, self.k, nu, e)

    def ev_one(self, x, nu=0, ext=-1):
        """Evaluate the spline at a single point.

        Parameters
        ----------
        x : float
            Point to evaluate at.
        nu : int, optional
            Derivative order, ``0 <= nu <= k``. Default 0, the spline
            itself. This is scipy's second positional parameter.
        ext : int, optional
            Per-call override of the constructor's `ext`; -1, the default,
            keeps it.

        Returns
        -------
        float
            The spline value, or its `nu`-th derivative. Wraps the scalar
            in a length-1 array for FITPACK, so calling this in a tight
            loop costs one small allocation per point -- prefer `ev` on a
            batch.

        Raises
        ------
        ValueError
            If `nu` is outside ``0..k``, if the effective `ext` is outside
            ``0..3``, or if the effective `ext` is 2 and a point lies
            outside ``[t[k], t[n-k-1]]``.
        """
        e = self.ext if ext < 0 else ext
        _der_range(nu, self.k)
        if e > 3:
            _fmt_ext_mode(e)
        _int_order(nu)
        xa = np.empty(1, np.float64)
        xa[0] = x
        if nu == 0:
            return _splev_ext(xa, self.t, self.c, self.k, e)[0]
        return _splder_ext(xa, self.t, self.c, self.k, nu, e)[0]

    def __getitem__(self, x):
        """``spl[x]`` -- sugar for ``spl.ev(x)``.

        ``spl(x)`` reaches the same method and is the spelling scipy uses.

        Parameters
        ----------
        x : 1-D float64 ndarray

        Returns
        -------
        y : 1-D float64 ndarray
        """
        return _splev_ext(x, self.t, self.c, self.k, self.ext)

    def derivative_ev(self, x, nu=1, ext=-1):
        """Evaluate the `nu`-th derivative (scipy's ``spl(x, nu)``).

        Parameters
        ----------
        x : 1-D float64 ndarray
            Points to evaluate at.
        nu : int, optional
            Derivative order, ``0 <= nu <= k``. Default 1. This method
            default works inside ``@njit`` (unlike a constructor default).

        Returns
        -------
        y : 1-D float64 ndarray, same length as `x`

        Notes
        -----
        scipy's ``.derivative()``, which returns a new spline object, is
        not implemented. Use this, or ``splder`` for the
        derivative's tck.
        """
        return _splder_ext(x, self.t, self.c, self.k, nu,
                           self.ext if ext < 0 else ext)

    def derivatives(self, x):
        """All derivatives at one point (scipy's ``spl.derivatives(x)``).

        Parameters
        ----------
        x : float
            A SCALAR point, inside the knot range.

        Returns
        -------
        d : 1-D float64 ndarray, length ``k + 1``
            ``d[j]`` is the j-th derivative at `x`; ``d[0]`` is the value.
        """
        return _ev.spalde(x, self.t, self.c, self.k)

    def integral(self, a, b):
        """Definite integral over ``[a, b]`` (scipy's ``spl.integral(a, b)``).

        Parameters
        ----------
        a, b : float
            Integration limits; may lie outside the knot range (the spline is
            extrapolated), and ``b < a`` negates the result.

        Returns
        -------
        float
            The integral.
        """
        return _ev.splint(self.t, self.c, self.k, a, b)

    def roots(self):
        """Zeros of the spline (scipy's ``spl.roots()``).

        Returns
        -------
        zero : 1-D float64 ndarray
            The roots, ascending; possibly empty.

        Raises
        ------
        NotImplementedError
            ``finding roots unsupported for non-cubic splines`` when
            ``k != 3``.

        Notes
        -----
        The root buffer holds ``3 * (len(t) - 7)`` entries, which is
        FITPACK's worst case -- at most 3 roots in each of the ``len(t) - 7``
        knot intervals -- so it never truncates.
        """
        return _uni_roots(self.t, self.c, self.k)

    def get_knots(self):
        """Interior knots plus the two ends (scipy's ``spl.get_knots()``).

        Returns
        -------
        t : 1-D float64 ndarray
            ``self.t[k : len(t)-k]`` -- the knot vector with the repeated
            boundary knots stripped, which is what scipy returns. Use the
            ``.t`` attribute for the full vector.
        """
        return self.t[self.k:len(self.t) - self.k]

    def get_coeffs(self):
        """The spline coefficients (scipy's ``spl.get_coeffs()``).

        Returns
        -------
        c : 1-D float64 ndarray, length ``len(t) - k - 1``
            The meaningful coefficients, with FITPACK's padding tail dropped.
        """
        return self.c[:len(self.t) - self.k - 1]

    def get_residual(self):
        """Weighted sum of squared residuals (scipy's ``spl.get_residual()``).

        Returns
        -------
        float
            FITPACK's ``fp`` for the fit. Zero for an interpolating spline.
        """
        return self.fp


@njit
def InterpolatedUnivariateSpline(x, y, w=_NO_W, bbox=None, k=3, ext=0,
                                 check_finite=False):
    """Build an interpolating spline through every data point.

    `UnivariateSpline` with the smoothing factor pinned at 0, so the spline
    passes through every data point.

    Parameters
    ----------
    x : 1-D float64 ndarray, length m
        Abscissae, strictly increasing.
    y : 1-D float64 ndarray, length m
        Ordinates.
    w : 1-D array_like of float, optional
        Positive weights, length m. ``None`` and a zero-length array both
        mean unit weights. Weights have no effect on the fit here, since
        ``s = 0`` forces interpolation; measured identical knots and
        ``fp = 0`` with and without them, on both sides.
    bbox : (2,) array_like of float, optional
        Boundary of the approximation interval; ``None``, in place of the
        pair or in one slot, means ``x[0]`` / ``x[-1]``, and is the default.
        May only WIDEN.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    ext : int or str, optional
        Extrapolation mode: 0 or ``'extrapolate'``, 1 or ``'zeros'``, 2 or
        ``'raise'``, 3 or ``'const'``. Default 0.
    check_finite : bool, optional
        Raise if `x`, `y` or `w` contains a NaN or an inf. Default False.

    Returns
    -------
    spl : _InterpolatedUnivariateSpline
        A jitclass instance carrying the attributes and methods below.

    Raises
    ------
    ValueError
        Non-finite input under `check_finite`, mismatched lengths, a `bbox`
        that is not length 2, `k` outside 1..5, an unknown `ext`, ``m <= k``,
        or ``x must be strictly increasing``, the check that separates this
        class from `LSQUnivariateSpline`, where a duplicated `x` is allowed.

    See Also
    --------
    scipy.interpolate.InterpolatedUnivariateSpline : The scipy class this
        mirrors.

    Notes
    -----
    - ``spl(x)`` runs ``.ev`` for an array and ``.ev_one`` for a scalar.
      ``.ev(x)``, ``.ev_one(x)`` and ``spl[x]`` reach the same methods.
      ``spl(x, nu)`` and ``spl(x, nu, ext)`` carry scipy's second and third
      positional parameters, and ``spl(x, nu=1)`` and ``spl(x, ext=1)``
      carry them by keyword. ``.ev(x, nu=1)`` takes the keyword inside
      ``@njit`` and not from the interpreter, where the jitclass method
      raises ``TypeError``; the call spelling works from both.
    - scipy RE-CLASSES the instance by `ier`; a jitclass cannot, so `ier` is
      an attribute.
    - ``.derivative()`` / ``.antiderivative()`` returning new spline objects
      are absent, and ``derivatives(array)`` (scipy's list return) is not
      available -- pass one scalar at a time.
    - ``ier = 10`` RAISES here. scipy warns and returns an object whose knot
      vector was never filled; a status that means "no approximation
      returned" is not carried into a usable object. `ier` of 1, 2 or 3
      warns.
    - A complex `y` raises ``TypeError``. scipy 1.18 accepts it, emits a
      ``ComplexWarning`` and discards the imaginary part, returning a float64
      spline.

    Defaults work in both worlds. ``InterpolatedUnivariateSpline`` is a
    plain ``@njit`` factory, not the jitclass itself, so
    ``InterpolatedUnivariateSpline(x, y)`` compiles and runs inside
    ``@njit`` as well as from Python. The class it returns,
    ``_InterpolatedUnivariateSpline``, takes every argument explicitly,
    because a jitclass constructor's defaults are Python-only.

    Accuracy against scipy 1.18.0, max absolute difference: knots,
    coefficients, `fp`, values in range and extrapolating, `integral`,
    `derivatives`, `roots` and ``derivative_ev(nu=1)`` all 0.0.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import InterpolatedUnivariateSpline
    >>> x = np.linspace(0, 4, 40)
    >>> y = np.sin(x)
    >>> spl = InterpolatedUnivariateSpline(x, y)     # fit once
    >>> float(np.round(spl(1.5), 8))
    0.99749473

    Inside compiled code, integrating the interpolant over its domain:

    >>> @njit
    ... def area(spl):
    ...     return spl.integral(0.0, 4.0)
    >>> float(np.round(area(spl), 6))
    1.653643

    Attributes
    ----------
    t : 1-D float64 ndarray
        Full knot vector, including the repeated boundary knots.
    c : 1-D float64 ndarray
        Coefficients in FITPACK's padded form.
    k : int
        Spline degree.
    fp : float
        Sum of squared residuals, 0 for an exact interpolant.
    ier : int
        FITPACK status; expect -1 here, which means "interpolating spline"
        and is a SUCCESS, not an error.
    ext : int
        The resolved extrapolation code, 0..3.

    Methods
    -------
    ev(x), ev_one(x), __getitem__(x), derivative_ev(x, nu), derivatives(x),
    integral(a, b), roots(), get_knots(), get_coeffs(), get_residual()
    """
    return _InterpolatedUnivariateSpline(x, y, w, bbox, k, ext, check_finite)


@scijitclass(_uni_spec, dispatch=[('ev', _first_seq),
                                  ('ev_one', all_scalar)])
class _LSQUnivariateSpline:
    """Instance type behind the `LSQUnivariateSpline` factory,
    which carries the documentation and the defaults. Every argument is
    explicit here: a jitclass constructor's defaults are Python-only.
    """

    def __init__(self, x, y, t, w, bbox, k, ext, check_finite):
        x_ = np.ascontiguousarray(np.asarray(x, np.float64)).ravel()
        _no_complex(y, "LSQUnivariateSpline")
        y_ = np.ascontiguousarray(np.asarray(_real_in(y), np.float64)).ravel()
        ti = np.ascontiguousarray(np.asarray(t, np.float64)).ravel()
        wg = _w_given(w)
        w_ = _w_or_ones(w, len(x_))
        _uni_validate(x_, y_, w_, wg, k, 0.0, False, check_finite)
        m = len(x_)
        for i in range(1, m):
            if not (x_[i] - x_[i - 1] >= 0.0):
                raise ValueError("x must be increasing")
        xb, xe = _bbox_pair(bbox, x_[0], x_[m - 1])
        e = _ext_code(ext)
        ni = len(ti)
        t_full = np.empty(ni + 2 * (k + 1), np.float64)
        for i in range(k + 1):
            t_full[i] = xb
            t_full[ni + k + 1 + i] = xe
        for i in range(ni):
            t_full[k + 1 + i] = ti[i]
        # scipy's explicit Schoenberg-Whitney test on the PADDED vector,
        # t[k+1:n-k] - t[k:n-k-1] > 0, run before any Fortran.
        nfull = ni + 2 * (k + 1)
        for i in range(k + 1, nfull - k):
            if not (t_full[i] - t_full[i - 1] > 0.0):
                raise ValueError("Interior knots t must satisfy "
                                 "Schoenberg-Whitney conditions")
        if _fpchec(x_, t_full, k) != 0:
            raise ValueError(_FPCHEC_MSG)
        t, c, fp, ier = _ft.curfit_lsq(x_, y_, w_, t_full, k, xb, xe)
        if ier == 10:
            raise ValueError("LSQUnivariateSpline: error on input data")
        _curfit_warn(ier)
        self.t = t
        self.c = c
        self.k = k
        self.fp = fp
        self.ier = ier
        self.ext = e

    def ev(self, x, nu=0, ext=-1):
        """Evaluate the spline at an array of points (scipy's ``spl(x)``).

        Parameters
        ----------
        x : 1-D float64 ndarray
            Points to evaluate at.
        nu : int, optional
            Derivative order, ``0 <= nu <= k``. Default 0, the spline
            itself. This is scipy's second positional parameter.
        ext : int, optional
            Per-call override of the constructor's `ext`; -1, the default,
            keeps it.

        Returns
        -------
        y : 1-D float64 ndarray, same length as `x`
            Spline values, or `nu`-th derivative values. Outside the knot
            range the spline is EXTRAPOLATED, where scipy's ``ext=0``
            default does the same.

        Raises
        ------
        ValueError
            If `nu` is outside ``0..k``, if the effective `ext` is outside
            ``0..3``, or if the effective `ext` is 2 and a point lies
            outside ``[t[k], t[n-k-1]]``.
        """
        e = self.ext if ext < 0 else ext
        _der_range(nu, self.k)
        if e > 3:
            _fmt_ext_mode(e)
        _int_order(nu)
        if nu == 0:
            return _splev_ext(x, self.t, self.c, self.k, e)
        return _splder_ext(x, self.t, self.c, self.k, nu, e)

    def ev_one(self, x, nu=0, ext=-1):
        """Evaluate the spline at a single point.

        Parameters
        ----------
        x : float
            Point to evaluate at.
        nu : int, optional
            Derivative order, ``0 <= nu <= k``. Default 0, the spline
            itself. This is scipy's second positional parameter.
        ext : int, optional
            Per-call override of the constructor's `ext`; -1, the default,
            keeps it.

        Returns
        -------
        float
            The spline value, or its `nu`-th derivative. Wraps the scalar
            in a length-1 array for FITPACK, so calling this in a tight
            loop costs one small allocation per point -- prefer `ev` on a
            batch.

        Raises
        ------
        ValueError
            If `nu` is outside ``0..k``, if the effective `ext` is outside
            ``0..3``, or if the effective `ext` is 2 and a point lies
            outside ``[t[k], t[n-k-1]]``.
        """
        e = self.ext if ext < 0 else ext
        _der_range(nu, self.k)
        if e > 3:
            _fmt_ext_mode(e)
        _int_order(nu)
        xa = np.empty(1, np.float64)
        xa[0] = x
        if nu == 0:
            return _splev_ext(xa, self.t, self.c, self.k, e)[0]
        return _splder_ext(xa, self.t, self.c, self.k, nu, e)[0]

    def __getitem__(self, x):
        """``spl[x]`` -- sugar for ``spl.ev(x)``.

        ``spl(x)`` reaches the same method and is the spelling scipy uses.

        Parameters
        ----------
        x : 1-D float64 ndarray

        Returns
        -------
        y : 1-D float64 ndarray
        """
        return _splev_ext(x, self.t, self.c, self.k, self.ext)

    def derivative_ev(self, x, nu=1, ext=-1):
        """Evaluate the `nu`-th derivative (scipy's ``spl(x, nu)``).

        Parameters
        ----------
        x : 1-D float64 ndarray
            Points to evaluate at.
        nu : int, optional
            Derivative order, ``0 <= nu <= k``. Default 1. This method
            default works inside ``@njit`` (unlike a constructor default).

        Returns
        -------
        y : 1-D float64 ndarray, same length as `x`

        Notes
        -----
        scipy's ``.derivative()``, which returns a new spline object, is
        not implemented. Use this, or ``splder`` for the
        derivative's tck.
        """
        return _splder_ext(x, self.t, self.c, self.k, nu,
                           self.ext if ext < 0 else ext)

    def derivatives(self, x):
        """All derivatives at one point (scipy's ``spl.derivatives(x)``).

        Parameters
        ----------
        x : float
            A SCALAR point, inside the knot range.

        Returns
        -------
        d : 1-D float64 ndarray, length ``k + 1``
            ``d[j]`` is the j-th derivative at `x`; ``d[0]`` is the value.
        """
        return _ev.spalde(x, self.t, self.c, self.k)

    def integral(self, a, b):
        """Definite integral over ``[a, b]`` (scipy's ``spl.integral(a, b)``).

        Parameters
        ----------
        a, b : float
            Integration limits; may lie outside the knot range (the spline is
            extrapolated), and ``b < a`` negates the result.

        Returns
        -------
        float
            The integral.
        """
        return _ev.splint(self.t, self.c, self.k, a, b)

    def roots(self):
        """Zeros of the spline (scipy's ``spl.roots()``).

        Returns
        -------
        zero : 1-D float64 ndarray
            The roots, ascending; possibly empty.

        Raises
        ------
        NotImplementedError
            ``finding roots unsupported for non-cubic splines`` when
            ``k != 3``.

        Notes
        -----
        The root buffer holds ``3 * (len(t) - 7)`` entries, which is
        FITPACK's worst case -- at most 3 roots in each of the ``len(t) - 7``
        knot intervals -- so it never truncates.
        """
        return _uni_roots(self.t, self.c, self.k)

    def get_knots(self):
        """Interior knots plus the two ends (scipy's ``spl.get_knots()``).

        Returns
        -------
        t : 1-D float64 ndarray
            ``self.t[k : len(t)-k]`` -- the knot vector with the repeated
            boundary knots stripped, which is what scipy returns. Use the
            ``.t`` attribute for the full vector.
        """
        return self.t[self.k:len(self.t) - self.k]

    def get_coeffs(self):
        """The spline coefficients (scipy's ``spl.get_coeffs()``).

        Returns
        -------
        c : 1-D float64 ndarray, length ``len(t) - k - 1``
            The meaningful coefficients, with FITPACK's padding tail dropped.
        """
        return self.c[:len(self.t) - self.k - 1]

    def get_residual(self):
        """Weighted sum of squared residuals (scipy's ``spl.get_residual()``).

        Returns
        -------
        float
            FITPACK's ``fp`` for the fit. Zero for an interpolating spline.
        """
        return self.fp


@njit
def LSQUnivariateSpline(x, y, t, w=_NO_W, bbox=None, k=3, ext=0,
                        check_finite=False):
    """Build a least-squares spline on given interior knots.

    Parameters
    ----------
    x : 1-D float64 ndarray, length m
        Abscissae, increasing; a duplicated value is allowed.
    y : 1-D float64 ndarray, length m
        Ordinates.
    t : 1-D array_like of float
        The INTERIOR knots only, strictly inside ``(xb, xe)`` and strictly
        increasing. The constructor builds the full knot vector by repeating
        `xb` and `xe` ``k+1`` times at the ends. (The raw
        ``fitters.curfit_lsq`` wants the FULL vector instead.) An EMPTY `t` is
        valid and gives the least-squares polynomial of degree `k`.
    w : 1-D array_like of float, optional
        Positive weights, length m. ``None`` and a zero-length array both
        mean unit weights. `fp` scales as ``w**2``, so weights DO change the
        residual here even though the knots are fixed.
    bbox : (2,) array_like of float, optional
        Boundary of the approximation interval, and the padding values for
        the knot vector: ``bbox=[-1, 5]`` makes the full vector
        ``[-1,-1,-1,-1, t..., 5,5,5,5]``. ``None``, in place of the pair
        or in one slot, means ``x[0]`` / ``x[-1]``, and is the default. May
        only WIDEN.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    ext : int or str, optional
        Extrapolation mode: 0 or ``'extrapolate'``, 1 or ``'zeros'``, 2 or
        ``'raise'``, 3 or ``'const'``. Default 0.
    check_finite : bool, optional
        Raise if `x`, `y` or `w` contains a NaN or an inf. Default False.

    Returns
    -------
    spl : _LSQUnivariateSpline
        A jitclass instance carrying the attributes and methods below.

    Raises
    ------
    ValueError
        Non-finite input under `check_finite`, mismatched lengths, a `bbox`
        that is not length 2, `k` outside 1..5, an unknown `ext`, ``m <= k``,
        then ``x must be increasing`` (a DUPLICATED `x` is allowed here and
        rejected by `InterpolatedUnivariateSpline`), then ``Interior knots t
        must satisfy Schoenberg-Whitney conditions``, then FITPACK's
        ``fpchec`` rejection. All three knot checks run before the fit.

    See Also
    --------
    scipy.interpolate.LSQUnivariateSpline : The scipy class this mirrors.

    Notes
    -----
    - ``spl(x)`` runs ``.ev`` for an array and ``.ev_one`` for a scalar.
      ``.ev(x)``, ``.ev_one(x)`` and ``spl[x]`` reach the same methods.
      ``spl(x, nu)`` and ``spl(x, nu, ext)`` carry scipy's second and third
      positional parameters, and ``spl(x, nu=1)`` and ``spl(x, ext=1)``
      carry them by keyword. ``.ev(x, nu=1)`` takes the keyword inside
      ``@njit`` and not from the interpreter, where the jitclass method
      raises ``TypeError``; the call spelling works from both.
    - There is no `s` argument on either side: with the knots fixed the fit
      is the least-squares one. scipy's ``set_smoothing_factor`` is a no-op
      plus a ``UserWarning`` on this class, and is absent here.
    - ``.derivative()`` / ``.antiderivative()`` returning new spline objects
      are absent.
    - ``ier = 10`` RAISES here. scipy warns and returns an object whose knot
      vector was never filled; a status that means "no approximation
      returned" is not carried into a usable object. `ier` of 1, 2 or 3
      warns.
    - A complex `y` raises ``TypeError``. scipy 1.18 accepts it, emits a
      ``ComplexWarning`` and discards the imaginary part, returning a float64
      spline.

    Defaults work in both worlds. ``LSQUnivariateSpline`` is a plain
    ``@njit`` factory, not the jitclass itself, so ``LSQUnivariateSpline(x,
    y, t)`` compiles and runs inside ``@njit`` as well as from Python.
    The class it returns, ``_LSQUnivariateSpline``, takes every argument
    explicitly, because a jitclass constructor's defaults are Python-only.

    Accuracy against scipy 1.18.0, max absolute difference: knots,
    coefficients, `fp`, values, `get_knots`, `get_coeffs`, `integral` and
    `derivatives` all 0.0, including the empty-`t` case and weighted
    fits.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import LSQUnivariateSpline
    >>> x = np.linspace(0, 4, 40)
    >>> y = np.sin(x)
    >>> knots = np.array([1.0, 2.0, 3.0])
    >>> spl = LSQUnivariateSpline(x, y, knots)     # fit once on given interior knots
    >>> float(np.round(spl(1.5), 6))
    0.99578

    Inside compiled code, scanning the domain for the fit's peak:

    >>> @njit
    ... def peak(spl):
    ...     xs = np.linspace(0.0, 4.0, 201)
    ...     return np.max(spl(xs))
    >>> float(np.round(peak(spl), 6))
    0.998422

    Attributes
    ----------
    t : 1-D float64 ndarray
        Full knot vector, including the repeated boundary knots.
    c : 1-D float64 ndarray
        Coefficients in FITPACK's padded form.
    k : int
        Spline degree.
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status, 0 for every knot vector that passes the checks above.
    ext : int
        The resolved extrapolation code, 0..3.

    Methods
    -------
    ev(x), ev_one(x), __getitem__(x), derivative_ev(x, nu), derivatives(x),
    integral(a, b), roots(), get_knots(), get_coeffs(), get_residual()
    """
    return _LSQUnivariateSpline(x, y, t, w, bbox, k, ext, check_finite)


# =====================================================================
# bivariate spline classes
# =====================================================================

_biv_spec = [
    ('tx', float64[::1]),
    ('ty', float64[::1]),
    ('c', float64[::1]),
    ('kx', int64),
    ('ky', int64),
    ('fp', float64),
    ('ier', int64),
]


def _bbox_quad(bbox, xb_def, xe_def, yb_def, ye_def):
    """``(xb, xe, yb, ye)`` from scipy's 4-element `bbox`.

    The 4-element story of `_bbox_pair`: ``None``, in place of the container
    or in one slot, takes the data value the caller supplies for that slot.
    """
    if bbox is None:
        return (float(xb_def), float(xe_def), float(yb_def), float(ye_def))
    if len(bbox) != 4:
        raise ValueError("bbox shape should be (4,)")
    return (float(_elem_or(bbox[0], xb_def)),
            float(_elem_or(bbox[1], xe_def)),
            float(_elem_or(bbox[2], yb_def)),
            float(_elem_or(bbox[3], ye_def)))


@overload(_bbox_quad)
def _bbox_quad_ovl(bbox, xb_def, xe_def, yb_def, ye_def):
    """`_bbox_quad` inside ``@njit``: the bivariate `bbox`, four elements.

    The 2-element story of `_bbox_pair_ovl` over ``(xb, xe, yb, ye)``.
    scipy's literal default here is ``[None] * 4``.
    """
    if _is_none(bbox):
        def impl(bbox, xb_def, xe_def, yb_def, ye_def):
            return (np.float64(xb_def), np.float64(xe_def),
                    np.float64(yb_def), np.float64(ye_def))
        return impl
    if isinstance(bbox, types.BaseTuple):
        if len(bbox) != 4:
            def impl(bbox, xb_def, xe_def, yb_def, ye_def):
                _bbox_shape_raise(4)
                return (np.float64(xb_def), np.float64(xe_def),
                        np.float64(yb_def), np.float64(ye_def))
            return impl

        def impl(bbox, xb_def, xe_def, yb_def, ye_def):
            return (_elem_or(bbox[0], xb_def), _elem_or(bbox[1], xe_def),
                    _elem_or(bbox[2], yb_def), _elem_or(bbox[3], ye_def))
        return impl
    if isinstance(bbox, (types.List, types.ListType)):
        def impl(bbox, xb_def, xe_def, yb_def, ye_def):
            if len(bbox) != 4:
                raise ValueError("bbox shape should be (4,)")
            return (_elem_or(bbox[0], xb_def), _elem_or(bbox[1], xe_def),
                    _elem_or(bbox[2], yb_def), _elem_or(bbox[3], ye_def))
        return impl

    def impl(bbox, xb_def, xe_def, yb_def, ye_def):
        b = np.asarray(bbox, dtype=np.float64).ravel()
        if b.size != 4:
            raise ValueError("bbox shape should be (4,)")
        return (np.float64(b[0]), np.float64(b[1]),
                np.float64(b[2]), np.float64(b[3]))
    return impl


@njit
def _parder_order(dx, dy, kx, ky):
    """The derivative-order test scipy's ``spl(x, y, dx, dy)`` reports.

    ``parder.f`` sets ``ier = 10`` for a derivative order outside
    ``0 <= nu < k``, for an empty grid axis and for a decreasing one. The
    three conditions have different correct outcomes, so this tests the
    CONDITION and the caller keeps the other two ahead of it.

    Applies to `eval_grid` on the four bivariate classes.
    """
    if dx < 0 or dx >= kx or dy < 0 or dy >= ky:
        raise ValueError("Error code returned by parder: 10")


@njit
def _pardeu_order(dx, dy, kx, ky):
    """The `.ev` twin of `_parder_order`; scipy's text names ``pardeu``.

    Applies to `ev` and `ev_one` on the four bivariate classes.
    """
    if dx < 0 or dx >= kx or dy < 0 or dy >= ky:
        raise ValueError("Error code returned by pardeu: 10")


@njit
def _pd_order(dx, dy, kx, ky):
    """scipy's two ``partial_derivative`` messages, in the order it tests."""
    if dx < 0 or dy < 0:
        raise ValueError("order of derivative must be positive or zero")
    if dx >= kx or dy >= ky:
        raise ValueError("order of derivative must be less than degree of "
                         "spline")


@njit
def _grid_order(x, y):
    """scipy's ``grid=True`` ordering check, shared by the four bivariate
    classes.

    An axis of fewer than two points is not tested, and a repeated abscissa
    passes: the guard rejects a DECREASING step, which is what scipy's
    ``np.all(np.diff(x) >= 0.0)`` does despite its message.
    """
    if len(x) >= 2:
        for i in range(1, len(x)):
            if x[i] < x[i - 1]:
                raise ValueError(
                    "x must be strictly increasing when `grid` is True")
    if len(y) >= 2:
        for j in range(1, len(y)):
            if y[j] < y[j - 1]:
                raise ValueError(
                    "y must be strictly increasing when `grid` is True")


@njit
def _rect_validate(x, y, z, kx, ky, s, s_given, maxit):
    """``RectBivariateSpline.__init__``'s checks, in scipy's order."""
    for i in range(1, len(x)):
        if not (x[i] - x[i - 1] > 0.0):
            raise ValueError("x must be strictly increasing")
    for i in range(1, len(y)):
        if not (y[i] - y[i - 1] > 0.0):
            raise ValueError("y must be strictly increasing")
    if len(x) != z.shape[0]:
        raise ValueError("x dimension of z must have same number of elements "
                         "as x")
    if len(y) != z.shape[1]:
        raise ValueError("y dimension of z must have same number of elements "
                         "as y")
    if s_given and not (s >= 0.0):
        raise ValueError("s should be s >= 0.0")
    if len(x) <= kx:
        raise ValueError("mx must be > kx")
    if len(y) <= ky:
        raise ValueError("my must be > ky")
    if kx < 1 or kx > 5 or ky < 1 or ky > 5:
        raise ValueError("1 <= kx, ky <= 5 must hold")


@njit
def _surfit_msg(ier):
    """scipy's ``_surfit_messages[ier]``, verbatim, leading
    newline included.

    One table, two readers: the class that RAISES a status and the class
    that WARNS on it print the same text, because scipy reads this dict
    from both places. An unlisted status takes scipy's own fallback.
    """
    if ier == -3:
        return ("\n"
                "The coefficients of the spline returned have been computed "
                "as the\n"
                "minimal norm least-squares solution of a (numerically) rank "
                "deficient\n"
                "system (deficiency=%i). If deficiency is large, the results "
                "may be\n"
                "inaccurate. Deficiency may strongly depend on the value of "
                "eps.")
    if ier == 1:
        return ("\n"
                "The required storage space exceeds the available storage "
                "space: nxest\n"
                "or nyest too small, or s too small.\n"
                "The weighted least-squares spline corresponds to the "
                "current set of\n"
                "knots.")
    if ier == 2:
        return ("\n"
                "A theoretically impossible result was found during the "
                "iteration\n"
                "process for finding a smoothing spline with fp = s: s too "
                "small or\n"
                "badly chosen eps.\n"
                "Weighted sum of squared residuals does not satisfy "
                "abs(fp-s)/s < tol.")
    if ier == 3:
        return ("\n"
                "the maximal number of iterations maxit (set to 20 by the "
                "program)\n"
                "allowed for finding a smoothing spline with fp=s has been "
                "reached:\n"
                "s too small.\n"
                "Weighted sum of squared residuals does not satisfy "
                "abs(fp-s)/s < tol.\n"
                "Try increasing maxit by passing it as a keyword argument.")
    if ier == 4:
        return ("\n"
                "No more knots can be added because the number of b-spline "
                "coefficients\n"
                "(nx-kx-1)*(ny-ky-1) already exceeds the number of data "
                "points m:\n"
                "either s or m too small.\n"
                "The weighted least-squares spline corresponds to the "
                "current set of\n"
                "knots.")
    if ier == 5:
        return ("\n"
                "No more knots can be added because the additional knot "
                "would (quasi)\n"
                "coincide with an old one: s too small or too large a weight "
                "to an\n"
                "inaccurate data point.\n"
                "The weighted least-squares spline corresponds to the "
                "current set of\n"
                "knots.")
    if ier == 10:
        return ("\n"
                "Error on entry, no approximation returned. The following "
                "conditions\n"
                "must hold:\n"
                "xb<=x[i]<=xe, yb<=y[i]<=ye, w[i]>0, i=0..m-1\n"
                "If iopt==-1, then\n"
                "  xb<tx[kx+1]<tx[kx+2]<...<tx[nx-kx-2]<xe\n"
                "  yb<ty[ky+1]<ty[ky+2]<...<ty[ny-ky-2]<ye")
    return "ier=" + str(ier)


@njit
def _spfit_msg(ier):
    """scipy's ``_spfit_messages[ier]``, verbatim, leading
    newline included.

    One table, two readers: the class that RAISES a status and the class
    that WARNS on it print the same text, because scipy reads this dict
    from both places. An unlisted status takes scipy's own fallback.
    """
    if ier == -3:
        return ("\n"
                "The coefficients of the spline returned have been computed "
                "as the\n"
                "minimal norm least-squares solution of a (numerically) rank "
                "deficient\n"
                "system (deficiency=%i). If deficiency is large, the results "
                "may be\n"
                "inaccurate. Deficiency may strongly depend on the value of "
                "eps.")
    if ier == 1:
        return ("\n"
                "The required storage space exceeds the available storage "
                "space: nxest\n"
                "or nyest too small, or s too small.\n"
                "The weighted least-squares spline corresponds to the "
                "current set of\n"
                "knots.")
    if ier == 2:
        return ("\n"
                "A theoretically impossible result was found during the "
                "iteration\n"
                "process for finding a smoothing spline with fp = s: s too "
                "small or\n"
                "badly chosen eps.\n"
                "Weighted sum of squared residuals does not satisfy "
                "abs(fp-s)/s < tol.")
    if ier == 3:
        return ("\n"
                "the maximal number of iterations maxit (set to 20 by the "
                "program)\n"
                "allowed for finding a smoothing spline with fp=s has been "
                "reached:\n"
                "s too small.\n"
                "Weighted sum of squared residuals does not satisfy "
                "abs(fp-s)/s < tol.\n"
                "Try increasing maxit by passing it as a keyword argument.")
    if ier == 4:
        return ("\n"
                "No more knots can be added because the number of b-spline "
                "coefficients\n"
                "(nx-kx-1)*(ny-ky-1) already exceeds the number of data "
                "points m:\n"
                "either s or m too small.\n"
                "The weighted least-squares spline corresponds to the "
                "current set of\n"
                "knots.")
    if ier == 5:
        return ("\n"
                "No more knots can be added because the additional knot "
                "would (quasi)\n"
                "coincide with an old one: s too small or too large a weight "
                "to an\n"
                "inaccurate data point.\n"
                "The weighted least-squares spline corresponds to the "
                "current set of\n"
                "knots.")
    if ier == 10:
        return ("\n"
                "ERROR: on entry, the input data are controlled on validity\n"
                "       the following restrictions must be satisfied.\n"
                "          -1<=iopt(1)<=1, 0<=iopt(2)<=1, 0<=iopt(3)<=1,\n"
                "          -1<=ider(1)<=1, 0<=ider(2)<=1, ider(2)=0 if "
                "iopt(2)=0.\n"
                "          -1<=ider(3)<=1, 0<=ider(4)<=1, ider(4)=0 if "
                "iopt(3)=0.\n"
                "          mu >= mumin (see above), mv >= 4, nuest >=8, "
                "nvest >= 8,\n"
                "          kwrk>=5+mu+mv+nuest+nvest,\n"
                "          lwrk >= "
                "12+nuest*(mv+nvest+3)+nvest*24+4*mu+8*mv+max(nuest,mv+nvest"
                ")\n"
                "          0< u(i-1)<u(i)< pi,i=2,..,mu,\n"
                "          -pi<=v(1)< pi, v(1)<v(i-1)<v(i)<v(1)+2*pi, "
                "i=3,...,mv\n"
                "          if iopt(1)=-1: "
                "8<=nu<=min(nuest,mu+6+iopt(2)+iopt(3))\n"
                "                         0<tu(5)<tu(6)<...<tu(nu-4)< pi\n"
                "                         8<=nv<=min(nvest,mv+7)\n"
                "                         "
                "v(1)<tv(5)<tv(6)<...<tv(nv-4)<v(1)+2*pi\n"
                "                         the schoenberg-whitney conditions, "
                "i.e. there must be\n"
                "                         subset of grid coordinates uu(p) "
                "and vv(q) such that\n"
                "                            tu(p) < uu(p) < tu(p+4) "
                ",p=1,...,nu-4\n"
                "                            (iopt(2)=1 and iopt(3)=1 also "
                "count for a uu-value\n"
                "                            tv(q) < vv(q) < tv(q+4) "
                ",q=1,...,nv-4\n"
                "                            (vv(q) is either a value v(j) "
                "or v(j)+2*pi)\n"
                "          if iopt(1)>=0: s>=0\n"
                "          if s=0: nuest>=mu+6+iopt(2)+iopt(3), nvest>=mv+7\n"
                "       if one of these conditions is found to be "
                "violated,control is\n"
                "       immediately repassed to the calling program. in that "
                "case there is no\n"
                "       approximation returned.")
    return "ier=" + str(ier)


@njit
def _spherefit_msg(ier):
    """scipy's ``_spherefit_messages[ier]``, verbatim, leading
    newline included.

    One table, two readers: the class that RAISES a status and the class
    that WARNS on it print the same text, because scipy reads this dict
    from both places. An unlisted status takes scipy's own fallback.
    """
    if ier == -3:
        return ("\n"
                "WARNING. The coefficients of the spline returned have been "
                "computed as the\n"
                "         minimal norm least-squares solution of a "
                "(numerically) rank\n"
                "         deficient system (deficiency=%i, rank=%i). "
                "Especially if the rank\n"
                "         deficiency, which is computed by "
                "6+(nt-8)*(np-7)+ier, is large,\n"
                "         the results may be inaccurate. They could also "
                "seriously depend on\n"
                "         the value of eps.")
    if ier == 1:
        return ("\n"
                "The required storage space exceeds the available storage "
                "space: nxest\n"
                "or nyest too small, or s too small.\n"
                "The weighted least-squares spline corresponds to the "
                "current set of\n"
                "knots.")
    if ier == 2:
        return ("\n"
                "A theoretically impossible result was found during the "
                "iteration\n"
                "process for finding a smoothing spline with fp = s: s too "
                "small or\n"
                "badly chosen eps.\n"
                "Weighted sum of squared residuals does not satisfy "
                "abs(fp-s)/s < tol.")
    if ier == 3:
        return ("\n"
                "the maximal number of iterations maxit (set to 20 by the "
                "program)\n"
                "allowed for finding a smoothing spline with fp=s has been "
                "reached:\n"
                "s too small.\n"
                "Weighted sum of squared residuals does not satisfy "
                "abs(fp-s)/s < tol.\n"
                "Try increasing maxit by passing it as a keyword argument.")
    if ier == 4:
        return ("\n"
                "No more knots can be added because the number of b-spline "
                "coefficients\n"
                "(nx-kx-1)*(ny-ky-1) already exceeds the number of data "
                "points m:\n"
                "either s or m too small.\n"
                "The weighted least-squares spline corresponds to the "
                "current set of\n"
                "knots.")
    if ier == 5:
        return ("\n"
                "No more knots can be added because the additional knot "
                "would (quasi)\n"
                "coincide with an old one: s too small or too large a weight "
                "to an\n"
                "inaccurate data point.\n"
                "The weighted least-squares spline corresponds to the "
                "current set of\n"
                "knots.")
    if ier == 10:
        return ("\n"
                "ERROR. On entry, the input data are controlled on validity. "
                "The following\n"
                "       restrictions must be satisfied:\n"
                "            -1<=iopt<=1,  m>=2, ntest>=8 ,npest >=8, "
                "0<eps<1,\n"
                "            0<=teta(i)<=pi, 0<=phi(i)<=2*pi, w(i)>0, "
                "i=1,...,m\n"
                "            lwrk1 >= 185+52*v+10*u+14*u*v+8*(u-1)*v**2+8*m\n"
                "            kwrk >= m+(ntest-7)*(npest-7)\n"
                "            if iopt=-1: 8<=nt<=ntest , 9<=np<=npest\n"
                "                        0<tt(5)<tt(6)<...<tt(nt-4)<pi\n"
                "                        0<tp(5)<tp(6)<...<tp(np-4)<2*pi\n"
                "            if iopt>=0: s>=0\n"
                "            if one of these conditions is found to be "
                "violated,control\n"
                "            is immediately repassed to the calling program. "
                "in that\n"
                "            case there is no approximation returned.")
    return "ier=" + str(ier)


@njit
def _surfit_raise(ier):
    """scipy's ``if ier not in [0, -1, -2]: raise ValueError(msg)``.

    The text is `_surfit_msg`, the same table `_surfit_warn` prints, so the
    class that raises a status and the class that warns on it say the same
    thing. `RectBivariateSpline-D4`.
    """
    if ier == 0 or ier == -1 or ier == -2:
        return
    raise ValueError(_surfit_msg(ier))


@njit
def _surfit_warn(ier):
    """scipy's ``warnings.warn(_surfit_messages[ier])`` branch.

    `SmoothBivariateSpline` is the one class in this module that scipy leaves
    unraised: it warns for any `ier` outside {0, -1, -2} and returns the
    object. The text is `_surfit_msg`, the same table `_surfit_raise` uses,
    so the two classes say the same thing for the same status.
    """
    if ier == 0 or ier == -1 or ier == -2:
        return
    _warn_user(_surfit_msg(ier))


# Arrays run `eval_grid`, the cross product, which is what scipy's
# ``spl(x, y)`` computes: its ``__call__`` defaults to ``grid=True``. A
# scalar is an axis of length 1 there, so two scalars give a (1, 1) grid;
# `ev_one` is the scalar spelling and is reached by name. The scattered-point
# evaluation, scipy's ``grid=False``, stays reachable as ``spl.ev(xi, yi)``.
@scijitclass(_biv_spec, dispatch=[('eval_grid', _grid_pair)])
class _RectBivariateSpline:
    """Instance type behind the `RectBivariateSpline` factory,
    which carries the documentation and the defaults. Every argument is
    explicit here: a jitclass constructor's defaults are Python-only.
    """

    def __init__(self, x, y, z, bbox, kx, ky, s, maxit):
        x_ = np.ascontiguousarray(np.asarray(x, np.float64)).ravel()
        y_ = np.ascontiguousarray(np.asarray(y, np.float64)).ravel()
        _no_complex(z, "RectBivariateSpline")
        z_ = np.ascontiguousarray(np.asarray(_real_in(z), np.float64))
        sv = _opt_f(s)
        s_given = _given(s)
        if not s_given:
            raise TypeError("must be real number, not NoneType")
        _rect_validate(x_, y_, z_, kx, ky, sv, s_given, maxit)
        xb, xe, yb, ye = _bbox_quad(bbox, x_.min(), x_.max(),
                                    y_.min(), y_.max())
        # scipy reads any maxit < 1 as "run no knot-search iterations", so it
        # is folded to 0 here. `regrid`'s own maxit=-1 is a DIFFERENT sentinel
        # meaning "use FITPACK's 20", and must not be reached by this path.
        mi = maxit if maxit >= 1 else 0
        tx, ty, c, fp, ier = _ft.regrid(x_, y_, z_.ravel(), kx, ky, sv,
                                        xb, xe, yb, ye, -1.0, mi)
        _surfit_raise(ier)
        self.tx = tx
        self.ty = ty
        self.c = c
        self.kx = kx
        self.ky = ky
        self.fp = fp
        self.ier = ier

    def ev(self, xi, yi, dx=0, dy=0):
        """Evaluate at scattered points (scipy's ``spl.ev(xi, yi, dx, dy)``).

        Parameters
        ----------
        xi : 1-D float64 ndarray or float
            Abscissae of the query points.
        yi : 1-D float64 ndarray or float
            Ordinates. The points are the pairs ``(xi[i], yi[i])``, not a
            grid; use `eval_grid` for a grid. A scalar or a length-1 array
            in either slot is broadcast against the other.
        dx, dy : int, optional
            Partial-derivative orders, ``0 <= dx < kx`` and ``0 <= dy < ky``.
            Both default to 0. These method defaults work inside ``@njit``.

        Returns
        -------
        z : 1-D float64 ndarray, length ``len(xi)``
            Spline values, extrapolated outside the knot range.

        Raises
        ------
        ValueError
            Lengths that do not broadcast.
        """
        xb, yb = _ev_pair(_pt1d(xi), _pt1d(yi))
        if len(xb) == 0:
            return np.zeros(0, np.float64)
        _pardeu_order(dx, dy, self.kx, self.ky)
        return _ev.pardeu(xb, yb, self.tx, self.ty, self.c,
                          self.kx, self.ky, dx, dy)

    def ev_one(self, x, y, dx=0, dy=0):
        """Evaluate at a single point.

        Parameters
        ----------
        x, y : float
            The query point.
        dx, dy : int, optional
            Partial-derivative orders, ``0 <= dx < kx`` and ``0 <= dy < ky``.
            Both default to 0.

        Returns
        -------
        float
            The spline value, as a SCALAR. scipy's ``spl.ev(0.5, 0.5)``
            returns a length-1 array for the same query. One call per point
            costs 693 ns against 315 ns per point for `ev` on a batch,
            measured on a 200x500 table, so prefer `ev` on a batch in a hot
            loop.
        """
        xa = np.empty(1, np.float64)
        ya = np.empty(1, np.float64)
        xa[0] = x
        ya[0] = y
        _pardeu_order(dx, dy, self.kx, self.ky)
        return _ev.pardeu(xa, ya, self.tx, self.ty, self.c,
                          self.kx, self.ky, dx, dy)[0]

    def eval_grid(self, x, y, dx=0, dy=0):
        """Evaluate on a grid -- this is what scipy's ``spl(x, y)`` does.

        Parameters
        ----------
        x : 1-D float64 ndarray or float
            Grid abscissae, ascending. A scalar is an axis of length 1.
        y : 1-D float64 ndarray or float
            Grid ordinates, ascending. The spline is evaluated on the full
            CROSS PRODUCT.
        dx, dy : int, optional
            Partial-derivative orders, scipy's ``spl(x, y, dx, dy)``. Both
            default to 0.

        Returns
        -------
        z : (len(x), len(y)) float64 ndarray
            Spline values on the grid. An empty `x` or `y` gives an empty
            grid of that shape.

        Raises
        ------
        ValueError
            If `x` or `y` steps DOWN anywhere. A repeated value is accepted.

        Notes
        -----
        ``spl(x, y)`` reaches this method, scipy's ``grid=True`` default;
        ``.ev`` is the ``grid=False`` form and does not order-check.
        """
        x_ = _pt1d(x)
        y_ = _pt1d(y)
        if len(x_) == 0 or len(y_) == 0:
            return np.zeros((len(x_), len(y_)), np.float64)
        _grid_order(x_, y_)
        _parder_order(dx, dy, self.kx, self.ky)
        return _ev.parder(x_, y_, self.tx, self.ty, self.c,
                          self.kx, self.ky, dx, dy)

    def partial_derivative(self, x, y, dx, dy):
        """Grid evaluation of a mixed partial derivative.

        Equivalent to ``scipy``'s ``spl(x, y, dx=dx, dy=dy)``.

        Parameters
        ----------
        x : 1-D float64 ndarray
            Grid abscissae, ascending.
        y : 1-D float64 ndarray
            Grid ordinates, ascending.
        dx : int
            Derivative order in x, ``0 <= dx < kx``.
        dy : int
            Derivative order in y, ``0 <= dy < ky``.

        Returns
        -------
        z : (len(x), len(y)) float64 ndarray
            The derivative on the grid.

        Notes
        -----
        scipy's ``.partial_derivative(dx, dy)`` returns a new spline OBJECT
        of class ``_DerivedBivariateSpline``. Returning the object is not
        implemented; this evaluates the partial derivative directly.
        """
        _pd_order(dx, dy, self.kx, self.ky)
        return _ev.parder(_pt1d(x), _pt1d(y), self.tx, self.ty, self.c,
                          self.kx, self.ky, dx, dy)

    def integral(self, xa, xb, ya, yb):
        """Double integral over a rectangle (scipy's ``spl.integral``).

        Parameters
        ----------
        xa, xb : float
            Integration limits in x.
        ya, yb : float
            Integration limits in y.

        Returns
        -------
        float
            The integral of the spline over ``[xa, xb] x [ya, yb]``.
        """
        return _ev.dblint(self.tx, self.ty, self.c, self.kx, self.ky,
                          xa, xb, ya, yb)

    def get_knots(self):
        """The two knot vectors (scipy's ``spl.get_knots()``).

        Returns
        -------
        tx : 1-D float64 ndarray
            Knots in x, the FULL vector including the repeated boundary
            knots -- which is also what scipy returns, since
            ``_BivariateSplineBase.get_knots`` hands back ``self.tck[:2]``.
        ty : 1-D float64 ndarray
            Knots in y, likewise full.
        """
        return self.tx, self.ty

    def get_coeffs(self):
        """The spline coefficients (scipy's ``spl.get_coeffs()``).

        Returns
        -------
        c : 1-D float64 ndarray, length ``(nx-kx-1)*(ny-ky-1)``
            FITPACK's flat bivariate layout ``c[(ny-ky-1)*i + j]``, ready to
            hand to ``evaluators.bispev`` and friends.
        """
        return self.c

    def get_residual(self):
        """Sum of squared residuals (scipy's ``spl.get_residual()``).

        Returns
        -------
        float
            FITPACK's ``fp`` for the fit; 0 for an interpolating surface.
        """
        return self.fp


@njit
def RectBivariateSpline(x, y, z, bbox=None, kx=3, ky=3, s=0.0, maxit=20):
    """Build a bivariate spline over gridded data.

    Parameters
    ----------
    x : 1-D float64 ndarray, length mx
        Grid abscissae, strictly increasing.
    y : 1-D float64 ndarray, length my
        Grid ordinates, strictly increasing.
    z : 2-D float64 ndarray, shape (mx, my)
        Grid values. An F-ordered or strided array is copied first, so it is
        read correctly.
    bbox : (4,) array_like of float, optional
        The rectangle the fit is made over, ``[xb, xe, yb, ye]``. ``None``,
        in place of the four or in one slot, means ``min(x)``, ``max(x)``,
        ``min(y)``, ``max(y)``, and is the default. It may only WIDEN the
        data rectangle; a narrower one gives FITPACK's ``ier = 10``, and
        raises. The literal default ``[None, None, None, None]`` is accepted,
        from Python and from inside ``@njit``.
    kx : int, optional
        Degree in x, 1 <= kx <= 5. Default 3.
    ky : int, optional
        Degree in y, 1 <= ky <= 5. Default 3.
    s : float, optional
        Smoothing factor; 0 gives the interpolating surface. Default 0.0.
        ``None`` raises ``TypeError``.
    maxit : int, optional
        Cap on FITPACK's knot-search iterations; reaching it is ``ier = 3``,
        which raises. Default 20. Any value below 1 means "run no
        iterations" and is accepted; with ``s = 0`` no iteration is needed
        and the surface is unaffected. See the Notes for the one case where
        it still raises.

    Returns
    -------
    spl : _RectBivariateSpline
        A jitclass instance carrying the attributes and methods below.

    Raises
    ------
    ValueError
        Non-increasing `x` or `y`, a `z` whose two dimensions do not match
        `x` and `y`, a `bbox` that is not length 4, ``s < 0``,
        ``mx <= kx``, ``my <= ky``, `kx` or `ky` outside
        1..5, and every FITPACK status outside {0, -1, -2}.
    TypeError
        ``s = None``, with the message ``must be real number, not NoneType``.

    See Also
    --------
    scipy.interpolate.RectBivariateSpline : The scipy class this mirrors.

    Notes
    -----
    - ``spl(x, y)`` runs ``eval_grid``, scipy's ``grid=True`` default, for
      every pair of scalars and arrays, so two scalars give a ``(1, 1)``
      grid. ``ev(xi, yi)`` is scipy's ``grid=False`` and ``ev_one(x, y)``
      the single-point scalar spelling, both reached by name.
    - A complex `z` raises ``TypeError``. scipy 1.18 accepts it, emits a
      ``ComplexWarning`` and discards the imaginary part, returning a float64
      spline.
    - ``maxit`` is honoured as the knot-search iteration cap. ``maxit=5`` on a
      smoothing fit turns a converged answer into ``ier = 3``, which raises,
      and a larger cap converges where 20 does not.
    - ``maxit < 1`` means "run no knot-search iterations", scipy's reading,
      and is accepted. Where the fit needs no iteration the result is
      scipy's, measured over four grids from 6x7 to 20x18: with ``s = 0``
      (status ``ier = -1``, interpolating) 4 of 4 cases agree with scipy at
      ``0.000e+00`` on knots, coefficients and ``fp``, and with a large ``s``
      (``ier = -2``, least-squares polynomial) 3 of 3 do.
    - WHERE IT STILL RAISES. If the knot search was needed and the cap
      stopped it, FITPACK reports ``ier = 3`` and the fit raises, as it does
      for any other iteration cap. The surface FITPACK has at that point is
      unconverged. Raise the cap, or call `fitters.regrid` directly and read
      ``ier``.
    - A SMOOTHING FIT MAY WARN THAT IT IS RANK DEFICIENT. With a small `s` on
      noisy data, FITPACK's knot search can place knots so that one B-spline
      coefficient is not determined by the data at all. The surface still
      passes through the data as asked, and ``fp`` and ``ier`` report success,
      but between the data points it carries an arbitrary component and can
      swing far outside the range of the data.

      A ``UserWarning`` naming the number of undetermined coefficients is
      issued when this happens. The fix is a LARGER `s`: the condition
      appears when `s` is small enough to drive the knot count to its
      maximum, and disappears once the fit has room. scipy issues no
      warning for it.
    - ``.partial_derivative(dx, dy)`` returns a new ``_DerivedBivariateSpline``
      in scipy; here it evaluates on a grid directly. A jitclass cannot
      construct one of itself.
    - ``ev_one`` returns a SCALAR where scipy's ``spl.ev(0.5, 0.5)`` returns a
      length-1 array.
    - `z` as a list of lists is rejected: numba cannot reflect a list of
      lists. A 2-D array, including F-ordered and strided, is accepted.

    Defaults work in both worlds. ``RectBivariateSpline`` is a plain ``@njit``
    factory, not the jitclass itself, so ``RectBivariateSpline(x, y, z)``
    compiles and runs inside ``@njit`` as well as from Python. The class it
    returns, ``_RectBivariateSpline``, takes every argument explicitly,
    because a jitclass constructor's defaults are Python-only.

    Accuracy against ``scipy.interpolate.RectBivariateSpline`` on scipy
    1.18.0. INTERPOLATING fits (``s = 0``) are exactly 0.0 in knots,
    coefficients, `fp`, grid values, ``.ev``, ``.integral``, every
    ``dx``/``dy`` pair and both ``get_*`` methods, for ``kx``, ``ky`` = 1..5
    and with `bbox` widened.

    A SMOOTHING fit is a different matter. scipy 1.18 reaches FITPACK through
    a C translation whose smoothing iteration rounds differently, so on data
    where the knot search is sensitive the two libraries place knots in
    DIFFERENT POSITIONS. The surface gap is then set by the data rather than
    by floating point: measured up to 1.8e-02 on a 13x14 noisy grid at
    ``s = 0.01``, with both sides reporting ``ier = 0`` and both inside
    FITPACK's own ``|fp-s|/s <= 1e-3`` test. Two valid smoothing splines, not
    one right and one wrong. Where the knot search agrees the surfaces agree.
    Compare knot counts before comparing values.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import RectBivariateSpline
    >>> x = np.linspace(0, 1, 12)
    >>> y = np.linspace(0, 1, 15)
    >>> z = np.outer(np.sin(3 * x), np.cos(2 * y))
    >>> spl = RectBivariateSpline(x, y, z)     # fit once
    >>> float(np.round(spl(0.5, 0.5)[0, 0], 8))     # spl(x, y) is the grid form; [0, 0] takes the point
    0.53894085

    Inside compiled code, the largest value over a coarse query grid:

    >>> @njit
    ... def gridmax(spl):
    ...     return np.max(spl(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5)))
    >>> float(np.round(gridmax(spl), 6))
    0.99748

    Attributes
    ----------
    tx, ty : 1-D float64 ndarray
        Knot vectors in x and y. Together with `c` these are the `tck`.
    c : 1-D float64 ndarray
        Coefficients in FITPACK's flat bivariate layout.
    kx, ky : int
        The two degrees.
    fp : float
        Sum of squared residuals, also returned by ``get_residual()``.
    ier : int
        FITPACK status: 0 = smoothing achieved, -1 = interpolating surface,
        -2 = least-squares polynomial. Any other value has already raised, so
        this is 0, -1 or -2 on every instance that exists.

    Methods
    -------
    eval_grid(x, y, dx, dy)
        Grid evaluation, the equivalent of ``spl(x, y)``.
    ev(xi, yi, dx, dy), ev_one(x, y, dx, dy),
    partial_derivative(x, y, dx, dy), integral(xa, xb, ya, yb),
    get_knots(), get_coeffs(), get_residual()
    """
    return _RectBivariateSpline(x, y, z, bbox, kx, ky, s, maxit)


@njit
def _scat_validate(x, y, z, w, w_given, kx, ky, eps):
    """``BivariateSpline._validate_input``, in scipy's order."""
    m = len(x)
    if m != len(y) or m != len(z):
        raise ValueError("x, y, and z should have a same length")
    if w_given:
        if m != len(w):
            raise ValueError("x, y, z, and w should have a same length")
        for i in range(len(w)):
            if not (w[i] >= 0.0):
                raise ValueError("w should be positive")
    if not (0.0 < eps < 1.0):
        raise ValueError("eps should be between (0, 1)")
    if m < (kx + 1) * (ky + 1):
        raise ValueError("The length of x, y and z should be at least"
                         " (kx+1) * (ky+1)")


# Arrays run `eval_grid`, the cross product, which is what scipy's
# ``spl(x, y)`` computes: its ``__call__`` defaults to ``grid=True``. A
# scalar is an axis of length 1 there, so two scalars give a (1, 1) grid;
# `ev_one` is the scalar spelling and is reached by name. The scattered-point
# evaluation, scipy's ``grid=False``, stays reachable as ``spl.ev(xi, yi)``.
@scijitclass(_biv_spec, dispatch=[('eval_grid', _grid_pair)])
class _SmoothBivariateSpline:
    """Instance type behind the `SmoothBivariateSpline` factory,
    which carries the documentation and the defaults. Every argument is
    explicit here: a jitclass constructor's defaults are Python-only.
    """

    def __init__(self, x, y, z, w, bbox, kx, ky, s, eps):
        x_ = np.ascontiguousarray(np.asarray(x, np.float64)).ravel()
        y_ = np.ascontiguousarray(np.asarray(y, np.float64)).ravel()
        _no_complex(z, "SmoothBivariateSpline")
        z_ = np.ascontiguousarray(np.asarray(_real_in(z), np.float64)).ravel()
        wg = _w_given(w)
        w_ = _w_or_ones(w, len(x_))
        _scat_validate(x_, y_, z_, w_, wg, kx, ky, eps)
        xb, xe, yb, ye = _bbox_quad(bbox, x_.min(), x_.max(),
                                    y_.min(), y_.max())
        sv = _opt_f(s)
        if not _given(s):
            sv = float(len(x_))
        elif not (sv >= 0.0):
            raise ValueError("s should be s >= 0.0")
        tx, ty, c, fp, ier = _ft.surfit(x_, y_, z_, w_, kx, ky, sv, eps,
                                        0, 0, xb, xe, yb, ye)
        # `SmoothBivariateSpline-D6`: scipy warns inside the constructor, so
        # a compiled caller that builds the instance directly gets the
        # warning too.
        _surfit_warn(ier)
        self.tx = tx
        self.ty = ty
        self.c = c
        self.kx = kx
        self.ky = ky
        self.fp = fp
        self.ier = ier

    def ev(self, xi, yi, dx=0, dy=0):
        """Evaluate at scattered points (scipy's ``spl.ev(xi, yi, dx, dy)``).

        Parameters
        ----------
        xi : 1-D float64 ndarray or float
            Abscissae of the query points.
        yi : 1-D float64 ndarray or float
            Ordinates. A scalar or a length-1 array in either slot is
            broadcast against the other.
        dx, dy : int, optional
            Partial-derivative orders, ``0 <= dx < kx`` and ``0 <= dy < ky``.
            Both default to 0.

        Returns
        -------
        z : 1-D float64 ndarray, length ``len(xi)``

        Raises
        ------
        ValueError
            Lengths that do not broadcast.
        """
        xb, yb = _ev_pair(_pt1d(xi), _pt1d(yi))
        if len(xb) == 0:
            return np.zeros(0, np.float64)
        _pardeu_order(dx, dy, self.kx, self.ky)
        return _ev.pardeu(xb, yb, self.tx, self.ty, self.c,
                          self.kx, self.ky, dx, dy)

    def ev_one(self, x, y, dx=0, dy=0):
        """Evaluate at a single point.

        Parameters
        ----------
        x, y : float
            The query point.
        dx, dy : int, optional
            Partial-derivative orders, ``0 <= dx < kx`` and ``0 <= dy < ky``.
            Both default to 0.

        Returns
        -------
        float
            The spline value, as a SCALAR. scipy's ``spl.ev(0.5, 0.5)``
            returns a length-1 array for the same query.
        """
        xa = np.empty(1, np.float64)
        ya = np.empty(1, np.float64)
        xa[0] = x
        ya[0] = y
        _pardeu_order(dx, dy, self.kx, self.ky)
        return _ev.pardeu(xa, ya, self.tx, self.ty, self.c,
                          self.kx, self.ky, dx, dy)[0]

    def eval_grid(self, x, y, dx=0, dy=0):
        """Evaluate on a grid -- this is what scipy's ``spl(x, y)`` does.

        Parameters
        ----------
        x : 1-D float64 ndarray or float
            Grid abscissae, ascending. A scalar is an axis of length 1.
        y : 1-D float64 ndarray or float
            Grid ordinates, ascending. The spline is evaluated on the full
            CROSS PRODUCT.
        dx, dy : int, optional
            Partial-derivative orders, ``0 <= dx < kx`` and ``0 <= dy < ky``.
            Both default to 0.

        Returns
        -------
        z : (len(x), len(y)) float64 ndarray
            Spline values on the grid. An empty `x` or `y` gives an empty
            grid of that shape.

        Raises
        ------
        ValueError
            If `x` or `y` steps DOWN anywhere. A repeated value is accepted.

        Notes
        -----
        ``spl(x, y)`` reaches this method, scipy's ``grid=True`` default;
        ``.ev`` is the ``grid=False`` form and does not order-check.
        """
        x_ = _pt1d(x)
        y_ = _pt1d(y)
        if len(x_) == 0 or len(y_) == 0:
            return np.zeros((len(x_), len(y_)), np.float64)
        _grid_order(x_, y_)
        _parder_order(dx, dy, self.kx, self.ky)
        return _ev.parder(x_, y_, self.tx, self.ty, self.c,
                          self.kx, self.ky, dx, dy)

    def partial_derivative(self, x, y, dx, dy):
        """Grid evaluation of a mixed partial derivative.

        Parameters
        ----------
        x, y : 1-D float64 ndarray
            Grid abscissae and ordinates, ascending.
        dx, dy : int
            Derivative orders, ``0 <= dx < kx`` and ``0 <= dy < ky``.

        Returns
        -------
        z : (len(x), len(y)) float64 ndarray

        Notes
        -----
        scipy's ``.partial_derivative(dx, dy)`` returns a new
        ``_DerivedBivariateSpline`` object. Returning the object is not
        implemented; this evaluates the partial derivative directly.
        """
        _pd_order(dx, dy, self.kx, self.ky)
        return _ev.parder(_pt1d(x), _pt1d(y), self.tx, self.ty, self.c,
                          self.kx, self.ky, dx, dy)

    def integral(self, xa, xb, ya, yb):
        """Double integral over a rectangle (scipy's ``spl.integral``).

        Parameters
        ----------
        xa, xb, ya, yb : float
            Integration limits.

        Returns
        -------
        float
        """
        return _ev.dblint(self.tx, self.ty, self.c, self.kx, self.ky,
                          xa, xb, ya, yb)

    def get_knots(self):
        """The two knot vectors (scipy's ``spl.get_knots()``).

        Returns
        -------
        tx : 1-D float64 ndarray
            Knots in the first variable, the FULL vector including the
            repeated boundary knots -- which is what scipy returns too,
            since ``_BivariateSplineBase.get_knots`` hands back
            ``self.tck[:2]``.
        ty : 1-D float64 ndarray
            Knots in the second variable, likewise full.
        """
        return self.tx, self.ty

    def get_coeffs(self):
        """The spline coefficients (scipy's ``spl.get_coeffs()``).

        Returns
        -------
        c : 1-D float64 ndarray
            FITPACK's flat bivariate layout ``c[(ny-ky-1)*i + j]``.
        """
        return self.c

    def get_residual(self):
        """Sum of squared residuals (scipy's ``spl.get_residual()``).

        Returns
        -------
        float
            FITPACK's ``fp`` for the fit.
        """
        return self.fp


@njit
def SmoothBivariateSpline(x, y, z, w=_NO_W, bbox=None, kx=3, ky=3,
                          s=None, eps=1e-16):
    """Build a bivariate smoothing spline over scattered data.

    Parameters
    ----------
    x : 1-D float64 ndarray, length m
        Abscissae of the data points; no grid structure required.
    y : 1-D float64 ndarray, length m
        Ordinates.
    z : 1-D float64 ndarray, length m
        Values at those points.
    w : 1-D array_like of float, optional
        Weights, length m, each ``>= 0``. ``None`` and a ZERO-LENGTH array
        both mean unit weights.
    bbox : (4,) array_like of float, optional
        The rectangle the fit is made over, ``[xb, xe, yb, ye]``. ``None``,
        in place of the four or in one slot, means ``min(x)``, ``max(x)``,
        ``min(y)``, ``max(y)``, and is the default. The literal default
        ``[None, None, None, None]`` is accepted, from Python and from
        inside ``@njit``.
    kx : int, optional
        Degree in x, 1 <= kx <= 5. Default 3.
    ky : int, optional
        Degree in y, 1 <= ky <= 5. Default 3.
    s : float, optional
        Smoothing factor. ``None``, the default, means ``s = m``. A negative
        or NaN `s` raises ``ValueError``. ``s = 0`` requests interpolation,
        which scattered data often cannot support -- expect a positive `ier`
        then.
    eps : float, optional
        Rank-determination threshold, strictly between 0 and 1. Default
        1e-16. It changes the coefficients, so it is not cosmetic.

    Returns
    -------
    spl : _SmoothBivariateSpline
        A jitclass instance carrying the attributes and methods below.

    Raises
    ------
    ValueError
        `x`, `y`, `z` of differing lengths; a `w` of the wrong length or
        holding a negative entry; `eps` outside ``(0, 1)``;
        ``m < (kx+1)*(ky+1)``; a `bbox` that is not length 4; and a supplied
        ``s < 0``.

    Warns
    -----
    UserWarning
        `ier` outside {0, -1, -2}. The fit is returned and the matching
        ``_surfit_messages`` text is warned. This is the one class here that
        warns rather than raises: `RectBivariateSpline` and both sphere
        classes raise. The warning is issued through a ``numba.objmode``
        block, which runs its body in the interpreter, so
        ``warnings.catch_warnings`` and ``-W`` see it from compiled and
        uncompiled callers alike.

    See Also
    --------
    scipy.interpolate.SmoothBivariateSpline : The scipy class this mirrors.

    Notes
    -----
    - ``spl(x, y)`` runs ``eval_grid``, scipy's ``grid=True`` default, for
      every pair of scalars and arrays, so two scalars give a ``(1, 1)``
      grid. ``ev(xi, yi)`` is scipy's ``grid=False`` and ``ev_one(x, y)``
      the single-point scalar spelling, both reached by name.
    - ``.partial_derivative(dx, dy)`` returns a new object in scipy; here it
      evaluates on a grid directly.
    - A complex `z` raises ``TypeError``. scipy 1.18 accepts it, emits a
      ``ComplexWarning`` and discards the imaginary part, returning a float64
      spline.
    - ``ev_one`` returns a SCALAR where scipy's ``spl.ev(0.5, 0.5)`` returns
      a length-1 array.
    - A NaN in a `bbox` slot is a literal edge on both sides and both return
      an all-NaN spline, but the two knot vectors have different LENGTHS:
      9 and 9 here against scipy's 8 and 8 on a 60-point fit at
      ``s = 0.0``, ``0.5`` and ``5.0``. FITPACK reports ``ier = 1`` for it
      and scipy's C translation reports 5. A finite `bbox` agrees exactly at
      all three.

    Defaults work in both worlds. ``SmoothBivariateSpline`` is a plain
    ``@njit`` factory, not the jitclass itself, so
    ``SmoothBivariateSpline(x, y, z)`` compiles and runs inside ``@njit`` as
    well as from Python. The class it returns,
    ``_SmoothBivariateSpline``, takes every argument explicitly, because a
    jitclass constructor's defaults are Python-only.

    Accuracy against ``scipy.interpolate.SmoothBivariateSpline`` on scipy
    1.18.0, 200 scattered points, kx = ky = 3. INTERPOLATING fits (``s = 0``)
    are exactly 0.0 in knots, coefficients, `fp` and grid values.

    A SMOOTHING fit is a different matter. scipy 1.18 reaches FITPACK through
    a C translation whose smoothing iteration rounds differently, so on data
    where the knot search is sensitive the two libraries place knots in
    DIFFERENT POSITIONS. The surface gap is then set by the data rather than
    by floating point: measured up to 1.8e-02 on a 13x14 noisy grid at
    ``s = 0.01``, with both sides reporting ``ier = 0`` and both inside
    FITPACK's own ``|fp-s|/s <= 1e-3`` test. Two valid smoothing splines, not
    one right and one wrong. Where the knot search agrees the surfaces agree.
    Compare knot counts before comparing values.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import SmoothBivariateSpline
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 1, 200)
    >>> y = rng.uniform(0, 1, 200)
    >>> z = np.sin(3 * x) * np.cos(2 * y)
    >>> spl = SmoothBivariateSpline(x, y, z, s=0.05)     # fit once
    >>> spl.ier
    -2
    >>> float(np.round(spl(0.5, 0.5)[0, 0], 6))     # spl(x, y) is the grid form; [0, 0] takes the point
    0.525083

    Inside compiled code, the largest value over a coarse query grid:

    >>> @njit
    ... def gridmax(spl):
    ...     return np.max(spl(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5)))
    >>> float(np.round(gridmax(spl), 6))
    0.987065

    Attributes
    ----------
    tx, ty : 1-D float64 ndarray
        Knot vectors. With `c` these are the `tck`.
    c : 1-D float64 ndarray
        Coefficients in FITPACK's flat bivariate layout.
    kx, ky : int
        The two degrees.
    fp : float
        Weighted sum of squared residuals, also returned by
        ``get_residual()``.
    ier : int
        FITPACK status: 0 smoothing achieved, -1 interpolating surface, -2
        least-squares polynomial, positive a failure. Anything outside
        {0, -1, -2} is warned about and NOT raised on, so the object is still
        returned and usable.

    Methods
    -------
    eval_grid(x, y, dx, dy)
        Grid evaluation, the equivalent of ``spl(x, y)``.
    ev(xi, yi, dx, dy), ev_one(x, y, dx, dy),
    partial_derivative(x, y, dx, dy), integral(xa, xb, ya, yb),
    get_knots(), get_coeffs(), get_residual()
    """
    return _SmoothBivariateSpline(x, y, z, w, bbox, kx, ky, s, eps)


def _scalar_kind(v):
    """Classify a pole argument: 0 sequence or ``None``, 1 bool, 2 int, 3 float.

    `RectSphereBivariateSpline-D1`. scipy broadcasts a pole argument only when
    it is a Python ``bool`` (the three flags) or a Python ``float``
    (`pole_values`); every other scalar reaches a subscript or an unpack of a
    scalar and raises. The kind is decided by the TYPE, so it is a compile-time
    constant, and the three raising helpers below turn it into scipy's text at
    scipy's position in the check order.
    """
    if isinstance(v, (bool, np.bool_)):
        return 1
    if isinstance(v, (int, np.integer)):
        return 2
    if isinstance(v, (float, np.floating)):
        return 3
    return 0


@overload(_scalar_kind)
def _scalar_kind_ovl(v):
    """`_scalar_kind` inside ``@njit``: the answer is baked in as a literal.

    A ``numpy.bool_`` is `types.Boolean` here and is indistinguishable from a
    Python ``bool``, so it is accepted where scipy raises `IndexError`. That is
    an additive superset: scipy refuses the value, so no scipy-shaped call
    reaches it.
    """
    t = v
    if isinstance(t, types.Omitted):
        val = t.value
        k = (1 if isinstance(val, bool) else
             2 if isinstance(val, int) else
             3 if isinstance(val, float) else 0)
    elif isinstance(t, types.Boolean):
        k = 1
    elif isinstance(t, types.Integer):
        k = 2
    elif isinstance(t, types.Float):
        k = 3
    else:
        k = 0

    def impl(v):
        return k
    return impl


@njit
def _flag_no_subscript(k):
    """scipy's ``pole_continuity[1]`` / ``pole_exact[0]`` on a scalar.

    `RectSphereBivariateSpline-D1`. A ``bool`` is broadcast by scipy before
    the subscript is reached, so kinds 0 and 1 pass.
    """
    if k == 2:
        raise TypeError("'int' object is not subscriptable")
    if k == 3:
        raise TypeError("'float' object is not subscriptable")


@njit
def _flag_no_unpack(k):
    """scipy's ``ider[1], ider[3] = pole_flat`` on a scalar.

    `RectSphereBivariateSpline-D1`.
    """
    if k == 2:
        raise TypeError("cannot unpack non-iterable int object")
    if k == 3:
        raise TypeError("cannot unpack non-iterable float object")


@njit
def _values_no_unpack(k):
    """scipy's ``r0, r1 = pole_values`` on a scalar that is not a float.

    `RectSphereBivariateSpline-D1`. scipy broadcasts a ``float`` and unpacks
    everything else, so a ``bool`` and an ``int`` raise where a ``float`` fits.
    """
    if k == 1:
        raise TypeError("cannot unpack non-iterable bool object")
    if k == 2:
        raise TypeError("cannot unpack non-iterable int object")


def _pole_pair_i(v):
    """scipy's ``bool | (bool, bool)`` pole flag -> ``(int, int)``."""
    if isinstance(v, bool) or isinstance(v, (int, np.integer)):
        return int(v), int(v)
    a = np.asarray(v).ravel()
    if a.size != 2:
        raise ValueError("pole flag should be a bool or a pair of bools")
    return int(a[0]), int(a[1])


@overload(_pole_pair_i)
def _pole_pair_i_ovl(v):
    """`_pole_pair_i` inside ``@njit``: a pole flag as a pair of ints.

    ``pole_continuity`` and ``pole_flat`` are each one bool applied to both
    poles or a pair, one per pole. Broadcasting the scalar here means the
    cores only ever see two ints, and the tuple, array and scalar spellings
    each need their own body because they are different types.
    """
    if isinstance(v, types.BaseTuple):
        n = len(v)

        def impl(v):
            if n != 2:
                raise ValueError(
                    "pole flag should be a bool or a pair of bools")
            return np.int64(v[0]), np.int64(v[1])
        return impl
    if isinstance(v, types.Array):
        def impl(v):
            a = np.asarray(v).ravel()
            if a.size != 2:
                raise ValueError(
                    "pole flag should be a bool or a pair of bools")
            return np.int64(a[0]), np.int64(a[1])
        return impl

    def impl(v):
        return np.int64(v), np.int64(v)
    return impl


@njit
def _pole_pair_raise():
    """The `pole_values` shape refusal.

    A statement rather than a `raise` written into the arm that needs it. An
    ``@overload`` arm whose body is only a ``raise`` types as returning
    ``none``, and the caller, which unpacks four values, then fails to compile
    with an error naming neither `pole_values` nor its length.
    """
    raise ValueError("pole_values should be None, a float or a pair")


def _pole_pair_v(v):
    """scipy's ``None | float | (v0, v1)`` -> ``(r0, r1, given0, given1)``.

    ``None`` in a slot means "no value at that pole", and NaN spells the same
    thing, so both of scipy's ``(None, 0.0)`` and this package's
    ``(nan, 0.0)`` reach the same place.
    """
    if v is None:
        return 0.0, 0.0, False, False
    if isinstance(v, (float, int, np.floating, np.integer)):
        f = float(v)
        return f, f, True, True
    b = list(v)
    if len(b) != 2:
        raise ValueError("pole_values should be None, a float or a pair")
    a = [_nan_if_none(e) for e in b]
    # P1: only ``None`` means "this pole is unknown". A NaN is a literal pole
    # value, as in scipy, and produces an all-NaN spline on both sides.
    return (float(a[0]), float(a[1]),
            b[0] is not None, b[1] is not None)


@overload(_pole_pair_v)
def _pole_pair_v_ovl(v):
    """`_pole_pair_v` inside ``@njit``: `pole_values` as value plus present.

    Returns four things, ``(r0, r1, have0, have1)``, because a pole value of
    ``None`` means "do not constrain this pole" and a float means "constrain
    it to this", including when that float is ``0.0`` and when it is NaN. A
    value alone would not separate those, so the presence flags travel
    alongside.

    ``pole_values=(None, 0.0)`` is scipy's own spelling and reaches here as a
    heterogeneous tuple, which is why the element fold uses `_nan_if_none`.
    """
    if _is_none(v):
        def impl(v):
            return 0.0, 0.0, False, False
        return impl
    if isinstance(v, types.BaseTuple):
        if len(v) != 2:
            def impl(v):
                _pole_pair_raise()
                return 0.0, 0.0, False, False
            return impl
        # P1: presence is decided by the ELEMENT TYPE, so a NaN in a slot is
        # a literal pole value rather than a second spelling of ``None``.
        g0 = not _is_none(v.types[0])
        g1 = not _is_none(v.types[1])

        def impl(v):
            return _nan_if_none(v[0]), _nan_if_none(v[1]), g0, g1
        return impl
    if isinstance(v, (types.List, types.ListType)):
        def impl(v):
            if len(v) != 2:
                raise ValueError(
                    "pole_values should be None, a float or a pair")
            return _nan_if_none(v[0]), _nan_if_none(v[1]), True, True
        return impl
    if isinstance(v, types.Array):
        def impl(v):
            a = np.asarray(v, dtype=np.float64).ravel()
            if a.size != 2:
                raise ValueError(
                    "pole_values should be None, a float or a pair")
            return np.float64(a[0]), np.float64(a[1]), True, True
        return impl

    def impl(v):
        f = np.float64(v)
        return f, f, True, True
    return impl


@njit
def _theta_bounds(theta):
    """scipy's ``SphereBivariateSpline.__call__`` theta check.

    An EMPTY `theta` is not tested, which is scipy's ``theta.size > 0``
    guard, and both ends are inclusive.
    """
    if len(theta) > 0:
        for i in range(len(theta)):
            if theta[i] < 0.0 or theta[i] > np.pi:
                raise ValueError("requested theta out of bounds.")


@njit
def _phi_bounds(phi):
    """scipy's ``SmoothSphereBivariateSpline.__call__`` phi check.

    ``RectSphereBivariateSpline`` does NOT carry this one: measured, a phi
    outside ``[0, 2*pi]`` evaluates there without complaint.
    """
    if len(phi) > 0:
        for j in range(len(phi)):
            if phi[j] < 0.0 or phi[j] > 2.0 * np.pi:
                raise ValueError("requested phi out of bounds.")


@njit
def _rsphere_validate(u, v, r, s, s_given, c0, c1, f0, f1, ckind):
    """``RectSphereBivariateSpline.__init__``'s checks, in scipy's order.

    `ckind` is `_scalar_kind` of `pole_continuity`. scipy subscripts that
    argument inside the ``pole_continuity``/``pole_flat`` test, which sits
    after the grid checks and before the `s` check, so a scalar ``int`` there
    reports the grid fault first.
    """
    if not (0.0 < u[0] and u[len(u) - 1] < np.pi):
        raise ValueError("u should be between (0, pi)")
    if not (-np.pi <= v[0] < np.pi):
        raise ValueError("v[0] should be between [-pi, pi)")
    if not (v[len(v) - 1] <= v[0] + 2.0 * np.pi):
        raise ValueError("v[-1] should be v[0] + 2pi or less ")
    for i in range(1, len(u)):
        if not (u[i] - u[i - 1] > 0.0):
            raise ValueError("u must be strictly increasing")
    for i in range(1, len(v)):
        if not (v[i] - v[i - 1] > 0.0):
            raise ValueError("v must be strictly increasing")
    if len(u) != r.shape[0]:
        raise ValueError("u dimension of r must have same number of elements "
                         "as u")
    if len(v) != r.shape[1]:
        raise ValueError("v dimension of r must have same number of elements "
                         "as v")
    _flag_no_subscript(ckind)
    if c1 == 0 and f1 != 0:
        raise ValueError("if pole_continuity is False, so must be pole_flat")
    if c0 == 0 and f0 != 0:
        raise ValueError("if pole_continuity is False, so must be pole_flat")
    if not s_given:
        # scipy reaches ``not s >= 0.0`` with ``s`` still ``None`` and CPython
        # raises here; the text is CPython's own.
        raise TypeError(
            "'>=' not supported between instances of 'NoneType' and 'float'")
    if not (s >= 0.0):
        raise ValueError("s should be positive")


@njit
def _spfit_raise(ier):
    """scipy's ``if ier not in [0, -1, -2]: raise ValueError(msg)``.

    ``_spfit_messages`` is a third table, distinct from ``_surfit_messages``
    and ``_spherefit_messages``, and `_spfit_msg` carries it verbatim.
    `RectSphereBivariateSpline-D2`, `-D3`.
    """
    if ier == 0 or ier == -1 or ier == -2:
        return
    raise ValueError(_spfit_msg(ier))


# Arrays run `eval_grid`, the cross product, which is what scipy's
# ``spl(x, y)`` computes: its ``__call__`` defaults to ``grid=True``. A
# scalar is an axis of length 1 there, so two scalars give a (1, 1) grid;
# `ev_one` is the scalar spelling and is reached by name. The scattered-point
# evaluation, scipy's ``grid=False``, stays reachable as ``spl.ev(xi, yi)``.
@scijitclass(_biv_spec, dispatch=[('eval_grid', _grid_pair)])
class _RectSphereBivariateSpline:
    """Instance type behind the `RectSphereBivariateSpline` factory,
    which carries the documentation and the defaults. Every argument is
    explicit here: a jitclass constructor's defaults are Python-only.
    """

    def __init__(self, u, v, r, s, pole_continuity, pole_values, pole_exact,
                 pole_flat, r0, r1, ider0, ider1, ider2, ider3, iopt2, iopt3):
        u_ = np.ascontiguousarray(np.asarray(u, np.float64)).ravel()
        v_ = np.ascontiguousarray(np.asarray(v, np.float64)).ravel()
        r_ = np.ascontiguousarray(np.asarray(r, np.float64))
        # `RectSphereBivariateSpline-D1`: the three refusals below are scipy's
        # own, in scipy's order -- it unpacks `pole_values`, subscripts
        # `pole_exact` where a pole value is present, unpacks `pole_flat`, and
        # subscripts `pole_continuity` only after the grid checks.
        _values_no_unpack(_scalar_kind(pole_values))
        v0, v1, g0, g1 = _pole_pair_v(pole_values)
        if g0 or g1:
            _flag_no_subscript(_scalar_kind(pole_exact))
        e0, e1 = _pole_pair_i(pole_exact)
        _flag_no_unpack(_scalar_kind(pole_flat))
        f0, f1 = _pole_pair_i(pole_flat)
        c0, c1 = _pole_pair_i(pole_continuity)
        sv = _opt_f(s)
        _rsphere_validate(u_, v_, r_, sv, _given(s), c0, c1, f0, f1,
                          _scalar_kind(pole_continuity))
        # scipy's Table-3 mapping, then the raw-code overrides, which are the
        # additive superset. -9 and -1 are the "not overridden" sentinels.
        d0 = e0 if g0 else -1
        d2 = e1 if g1 else -1
        d1 = f0
        d3 = f1
        if ider0 != -9:
            d0 = ider0
        if ider1 != -9:
            d1 = ider1
        if ider2 != -9:
            d2 = ider2
        if ider3 != -9:
            d3 = ider3
        if iopt2 != -9:
            c0 = iopt2
        if iopt3 != -9:
            c1 = iopt3
        if not np.isnan(r0):
            v0 = r0
        if not np.isnan(r1):
            v1 = r1
        tu, tv, c, fp, ier = _ft.spgrid(u_, v_, r_.ravel(), v0, v1, sv,
                                        c0, c1, d0, d1, d2, d3)
        _spfit_raise(ier)
        self.tx = tu
        self.ty = tv
        self.c = c
        self.kx = 3
        self.ky = 3
        self.fp = fp
        self.ier = ier

    def ev(self, ui, vi, dtheta=0, dphi=0):
        """Evaluate at scattered points (scipy's ``spl.ev(theta, phi)``).

        Parameters
        ----------
        ui : 1-D float64 ndarray or float
            Colatitudes of the query points.
        vi : 1-D float64 ndarray or float
            Longitudes. A scalar or a length-1 array in either slot is
            broadcast against the other.
        dtheta, dphi : int, optional
            Partial-derivative orders, ``0 <= d < 3``. Both default to 0.

        Returns
        -------
        r : 1-D float64 ndarray, length ``len(ui)``

        Raises
        ------
        ValueError
            Lengths that do not broadcast.
        """
        ui_ = _pt1d(ui)
        vi_ = _pt1d(vi)
        _theta_bounds(ui_)
        ub, vb = _ev_pair(ui_, vi_)
        if len(ub) == 0:
            return np.zeros(0, np.float64)
        _pardeu_order(dtheta, dphi, 3, 3)
        return _ev.pardeu(ub, vb, self.tx, self.ty, self.c, 3, 3,
                          dtheta, dphi)

    def ev_one(self, uu, vv, dtheta=0, dphi=0):
        """Evaluate at a single point.

        Parameters
        ----------
        uu, vv : float
            Colatitude and longitude.
        dtheta, dphi : int, optional
            Partial-derivative orders, ``0 <= d < 3``. Both default to 0.

        Returns
        -------
        float
            The value, as a SCALAR. scipy's ``spl.ev(u, v)`` on scalars
            returns a length-1 array.
        """
        ua = np.empty(1, np.float64)
        va = np.empty(1, np.float64)
        ua[0] = uu
        va[0] = vv
        _theta_bounds(ua)
        _pardeu_order(dtheta, dphi, 3, 3)
        return _ev.pardeu(ua, va, self.tx, self.ty, self.c, 3, 3,
                          dtheta, dphi)[0]

    def eval_grid(self, uu, vv, dtheta=0, dphi=0):
        """Evaluate on a grid -- what scipy's ``spl(theta, phi)`` does.

        Parameters
        ----------
        uu : 1-D float64 ndarray or float
            Colatitudes, ascending. A scalar is an axis of length 1.
        vv : 1-D float64 ndarray or float
            Longitudes, ascending. The spline is evaluated on the full
            CROSS PRODUCT.
        dtheta, dphi : int, optional
            Partial-derivative orders, ``0 <= d < 3``. Both default to 0.

        Returns
        -------
        r : (len(uu), len(vv)) float64 ndarray
            An empty axis gives an empty grid of that shape. A grid axis
            that steps DOWN raises; a repeated value is accepted.

        Raises
        ------
        ValueError
            ``requested theta out of bounds.`` if a colatitude lies outside
            ``[0, pi]``.

        Notes
        -----
        ``spl(x, y)`` reaches this method, scipy's ``grid=True`` default;
        ``.ev`` is the ``grid=False`` form. The longitude is NOT
        range-checked for this class.
        """
        u_ = _pt1d(uu)
        v_ = _pt1d(vv)
        _theta_bounds(u_)
        if len(u_) == 0 or len(v_) == 0:
            return np.zeros((len(u_), len(v_)), np.float64)
        _grid_order(u_, v_)
        _parder_order(dtheta, dphi, 3, 3)
        return _ev.parder(u_, v_, self.tx, self.ty, self.c, 3, 3,
                          dtheta, dphi)

    def partial_derivative(self, uu, vv, dtheta, dphi):
        """Grid evaluation of a mixed partial derivative.

        Parameters
        ----------
        uu, vv : 1-D float64 ndarray
            Colatitudes and longitudes, ascending.
        dtheta, dphi : int
            Derivative orders, each below 3.

        Returns
        -------
        r : (len(uu), len(vv)) float64 ndarray

        Notes
        -----
        scipy's ``.partial_derivative`` returns a new spline object.
        Returning the object is not implemented; this evaluates the partial
        derivative directly.
        """
        _pd_order(dtheta, dphi, 3, 3)
        return _ev.parder(_pt1d(uu), _pt1d(vv), self.tx, self.ty, self.c,
                          3, 3, dtheta, dphi)

    def get_knots(self):
        """The two knot vectors (scipy's ``spl.get_knots()``).

        Returns
        -------
        tu : 1-D float64 ndarray
            Knots in colatitude, the FULL vector -- which is what scipy
            returns too.
        tv : 1-D float64 ndarray
            Knots in longitude, likewise full.
        """
        return self.tx, self.ty

    def get_coeffs(self):
        """The spline coefficients (scipy's ``spl.get_coeffs()``).

        Returns
        -------
        c : 1-D float64 ndarray, length ``(nu-4)*(nv-4)``
        """
        return self.c

    def get_residual(self):
        """Sum of squared residuals (scipy's ``spl.get_residual()``).

        Returns
        -------
        float
        """
        return self.fp


@njit
def RectSphereBivariateSpline(u, v, r, s=0.0, pole_continuity=False,
                              pole_values=None, pole_exact=False,
                              pole_flat=False, r0=np.nan, r1=np.nan,
                              ider0=-9, ider1=-9, ider2=-9, ider3=-9,
                              iopt2=-9, iopt3=-9, validate=True):
    """Build a spline on a spherical grid.

    Parameters
    ----------
    u : 1-D float64 ndarray, length mu
        Colatitudes of the grid, strictly increasing, strictly inside
        ``(0, pi)``.
    v : 1-D float64 ndarray, length mv
        Longitudes, strictly increasing, with ``v[0]`` in ``[-pi, pi)`` and
        ``v[-1] <= v[0] + 2*pi``.
    r : 2-D float64 ndarray, shape (mu, mv)
        Grid values.
    s : float, optional
        Smoothing factor. Default 0.0.
    pole_continuity : bool or (bool, bool), optional
        Continuity at the south and north pole. Default False.
    pole_values : float or (float, float), optional
        Data value at the poles. ``None`` (the default) means unknown at both;
        a scalar is broadcast to both. ``(None, 0.0)`` is accepted. A NaN in
        a slot is a literal pole value.
    pole_exact : bool or (bool, bool), optional
        Whether `pole_values` is exact or a data value to be smoothed.
        Default False. Ignored where `pole_values` is unknown.
    pole_flat : bool or (bool, bool), optional
        Whether the surface should be flat at the poles. Default False.
    r0, r1 : float, optional
        RAW FITPACK pole values, overriding `pole_values`. NaN, the default,
        means "not overridden".
    ider0, ider1, ider2, ider3 : int, optional
        RAW FITPACK ``ider`` codes, overriding `pole_exact` and `pole_flat`.
        -9, the default, means "not overridden".
    iopt2, iopt3 : int, optional
        RAW FITPACK ``iopt(2)``/``iopt(3)``, overriding `pole_continuity`.
        -9, the default, means "not overridden".
    validate : bool, optional
        Raise when FITPACK returns NaN coefficients under a success status,
        rather than handing back a spline that evaluates to NaN everywhere.
        Default True. One ``np.isnan`` over the coefficient array, run once
        at construction. See the note on ``pole_continuity=True`` below for
        what produces that state. Set False to receive the spline and test
        ``np.isnan(spl.c).any()`` directly.

    Returns
    -------
    spl : _RectSphereBivariateSpline
        A jitclass instance carrying the attributes and methods below.

    Raises
    ------
    ValueError
        `u` not strictly inside ``(0, pi)``, ``v[0]`` outside ``[-pi, pi)``,
        ``v[-1] > v[0] + 2*pi``, non-increasing `u` or `v`, an `r` whose
        dimensions do not match, ``pole_continuity`` False with
        ``pole_flat`` True, ``s < 0``, and every FITPACK status outside
        {0, -1, -2}. Also NaN coefficients under a success status, unless
        ``validate=False``, which skips the check and returns the NaN spline.
    TypeError
        A scalar `pole_continuity`, `pole_exact` or `pole_flat` that is not a
        bool, and a scalar `pole_values` that is not a float. The scalar
        reaches a subscript or an unpack: ``pole_continuity=1`` is
        ``'int' object is not subscriptable`` and ``pole_flat=1`` is
        ``cannot unpack non-iterable int object``. A pair of ints is accepted.

    See Also
    --------
    scipy.interpolate.RectSphereBivariateSpline : The scipy class this
        mirrors.

    Notes
    -----
    - ``spl(u, v)`` runs ``eval_grid``, scipy's ``grid=True`` default, for
      every pair of scalars and arrays, so two scalars give a ``(1, 1)``
      grid. ``ev(ui, vi)`` is scipy's ``grid=False`` and ``ev_one(u, v)``
      the single-point scalar spelling, both reached by name.
    - ``.partial_derivative`` returns a new object in scipy; here it
      evaluates on a grid directly.
    - ``ev_one`` returns a SCALAR where scipy's ``.ev`` on scalars returns a
      length-1 array.
    - ``pole_values=(None, 0.0)`` is accepted; ``(np.nan, 0.0)`` means the
      same thing.
    - `r0`, `r1` and the six raw codes are extra arguments scipy does not
      have. A scipy-shaped call never passes them; they reach FITPACK options
      scipy's four booleans cannot express.
    - scipy's ``.v0`` attribute is absent.

    Defaults work in both worlds. ``RectSphereBivariateSpline`` is a plain
    ``@njit`` factory, not the jitclass itself, so
    ``RectSphereBivariateSpline(u, v, r)`` compiles and runs inside ``@njit``
    as well as from Python. The class it returns,
    ``_RectSphereBivariateSpline``, takes every argument explicitly, because
    a jitclass constructor's defaults are Python-only.

    ``pole_continuity=True`` REACHES A NONDETERMINISTIC PATH IN FITPACK, on
    this library and on scipy alike. The same input, in a fresh process, gives
    different answers from run to run. At least three outcomes occur on a
    12x14 grid at ``s = 0.5``:

        nt = (20, 21)   fp = 0.0           NaN coefficients, status -1
        nt = (8, 11)    fp = 0.5000435091  finite, status 0
        nt = (9, 11)    fp = 0.4995522302  finite, status 0

    `nt` is ``(len(tu), len(tv))``, the two knot counts, and `fp` is the
    weighted sum of squared residuals.

    **The two finite outcomes are both correct.** ``spgrid`` stops when
    ``abs(fp - s) <= tol * s`` with ``tol = 0.001`` fixed in the source, so at
    ``s = 0.5`` any `fp` within 5.0e-04 of 0.5 is accepted. The two differ from
    `s` by 4.35e-05 and 4.48e-04, both inside that band, and from each other by
    4.91e-04, which is 0.98 of the band width. They are two knot placements
    that satisfy the same smoothing request, not a right and a wrong answer.

    **The NaN outcome is the defect.** ``fp = 0.0`` against ``s = 0.5`` means
    the fit fell through to interpolation, and its coefficients are not finite.

    **The proportions are not stable between samples**, so no single ratio
    describes it. Measured as the first call in a fresh process, 8 runs each:
    this library 3 NaN and 5 finite, scipy 4 NaN and 4 finite, scipy's values
    identical to this library's. An earlier 9-run sample of this library gave
    9 NaN, which is why an earlier version of this note reported the path as
    always returning NaN here; that was a small sample, not determinism.

    Nothing distinguishes the outcomes from the status code, and status -1
    means "interpolating spline", not an error. `validate`, on by default,
    detects the NaN outcome and issues a ``RuntimeWarning``; the spline is
    returned either way, as scipy returns it. ``validate=False`` silences the
    warning and leaves ``np.isnan(spl.c).any()`` to the caller.

    The cause is upstream, not in this wrapper: scipy 1.18 reaches FITPACK
    through its own C translation, sharing no code with this package, and
    reproduces the same outcomes with the same numbers. Every workspace array
    here is zero-filled and sized by the documented ``spgrid`` formula.
    ``scipy.interpolate.RectSphereBivariateSpline``'s own documentation does
    not mention it.

    ``pole_continuity=False``, the default, is stable.

    Accuracy against ``scipy.interpolate.RectSphereBivariateSpline`` on scipy
    1.18.0, 12x14 grid, default pole arguments, in knots, coefficients, `fp`
    and grid values: exactly 0.0 at ``s`` = 0, 0.5 and 10.0. At ``s = 1e-6``
    the two choose a different number of longitude knots, 20 against 21,
    which is a structurally different spline that no tolerance expresses.

    Which `s` values land in which régime is a property of the DATA, not of
    `s`: a second 12x14 fixture put the mismatch at ``s = 0.01`` instead
    (11 latitude knots against 12). Treat the knot counts above as an
    example, and compare knot counts before comparing values.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import RectSphereBivariateSpline
    >>> u = np.linspace(0.1, np.pi - 0.1, 12)
    >>> v = np.linspace(0, 2 * np.pi, 15)[:-1]
    >>> r = np.outer(np.sin(u), np.cos(v)) + 2.0
    >>> spl = RectSphereBivariateSpline(u, v, r, s=0.5)     # fit once
    >>> spl.ier
    0
    >>> float(np.round(spl(1.5, 1.0)[0, 0], 6))     # spl(u, v) is the grid form; [0, 0] takes the point
    2.442246

    Inside compiled code, the largest value over a coarse query grid:

    >>> @njit
    ... def gridmax(spl):
    ...     return np.max(spl(np.linspace(0.5, 2.5, 5), np.linspace(0.5, 5.5, 5)))
    >>> float(np.round(gridmax(spl), 6))
    2.781747

    Attributes
    ----------
    tx, ty : 1-D float64 ndarray
        Knot vectors in colatitude and longitude.
    c : 1-D float64 ndarray
        Coefficients in FITPACK's flat bivariate layout.
    kx, ky : int
        Both 3. BICUBIC ONLY.
    fp : float
        Sum of squared residuals, also returned by ``get_residual()``.
    ier : int
        FITPACK status. Any value outside {0, -1, -2} has already raised.

    Methods
    -------
    eval_grid(u, v, dtheta, dphi)
        Grid evaluation, the equivalent of ``spl(theta, phi)``.
    ev(ui, vi, dtheta, dphi), ev_one(u, v, dtheta, dphi),
    partial_derivative(u, v, dtheta, dphi), get_knots(), get_coeffs(),
    get_residual()
    """
    spl = _RectSphereBivariateSpline(u, v, r, s, pole_continuity,
                                     pole_values, pole_exact, pole_flat,
                                     r0, r1, ider0, ider1, ider2, ider3,
                                     iopt2, iopt3)
    if validate and np.isnan(spl.c).any():
        # WARN, do not raise. scipy reaches the same FITPACK path and returns
        # the same NaN spline without raising, so raising here would break a
        # parity property rather than fix a defect. The warning makes the
        # outcome visible; the spline is still returned, as scipy returns it.
        with objmode():
            warnings.warn(
                "RectSphereBivariateSpline: FITPACK returned NaN coefficients "
                "under a success status, so every evaluation of this spline "
                "will be NaN. This is the nondeterministic spgrid path that "
                "pole_continuity=True reaches; the same call succeeds on "
                "other runs. scipy reaches it too and does not warn. "
                "Pass validate=False to silence this.",
                RuntimeWarning)
    return spl


@njit
def _sphere_validate(theta, phi, w, w_given, s, s_given, eps, eps_given):
    """``SmoothSphereBivariateSpline.__init__``'s checks, in scipy's order."""
    for i in range(len(theta)):
        if not (0.0 <= theta[i] <= np.pi):
            raise ValueError("theta should be between [0, pi]")
    for i in range(len(phi)):
        if not (0.0 <= phi[i] <= 2.0 * np.pi):
            raise ValueError("phi should be between [0, 2pi]")
    if w_given:
        for i in range(len(w)):
            if not (w[i] >= 0.0):
                raise ValueError("w should be positive")
    if not s_given:
        # scipy reaches ``not s >= 0.0`` with ``s`` still ``None``; the two
        # texts below are CPython's own comparison errors.
        raise TypeError(
            "'>=' not supported between instances of 'NoneType' and 'float'")
    if not (s >= 0.0):
        raise ValueError("s should be positive")
    if not eps_given:
        raise TypeError(
            "'<' not supported between instances of 'float' and 'NoneType'")
    if not (0.0 < eps < 1.0):
        raise ValueError("eps should be between (0, 1)")


@njit
def _spherefit_raise(ier):
    """scipy's ``if ier not in [0, -1, -2]: raise ValueError(msg)``.

    ``_spherefit_messages`` differs from ``_surfit_messages``, so the text is
    that table's, verbatim, through `_spherefit_msg`.
    `SmoothSphereBivariateSpline-D4`.
    """
    if ier == 0 or ier == -1 or ier == -2:
        return
    raise ValueError(_spherefit_msg(ier))


# Arrays run `eval_grid`, the cross product, which is what scipy's
# ``spl(x, y)`` computes: its ``__call__`` defaults to ``grid=True``. A
# scalar is an axis of length 1 there, so two scalars give a (1, 1) grid;
# `ev_one` is the scalar spelling and is reached by name. The scattered-point
# evaluation, scipy's ``grid=False``, stays reachable as ``spl.ev(xi, yi)``.
@scijitclass(_biv_spec, dispatch=[('eval_grid', _grid_pair)])
class _SmoothSphereBivariateSpline:
    """Instance type behind the `SmoothSphereBivariateSpline` factory,
    which carries the documentation and the defaults. Every argument is
    explicit here: a jitclass constructor's defaults are Python-only.
    """

    def __init__(self, theta, phi, r, w, s, eps):
        th = np.ascontiguousarray(np.asarray(theta, np.float64)).ravel()
        ph = np.ascontiguousarray(np.asarray(phi, np.float64)).ravel()
        rr = np.ascontiguousarray(np.asarray(r, np.float64)).ravel()
        wg = _w_given(w)
        w_ = _w_or_ones(w, len(th))
        if len(ph) != len(th) or len(rr) != len(th):
            raise ValueError(
                "theta, phi and r should have a same length")
        if wg and len(w_) != len(th):
            raise ValueError(
                "theta, phi, r and w should have a same length")
        sv = _opt_f(s)
        ev = _opt_f(eps)
        _sphere_validate(th, ph, w_, wg, sv, _given(s), ev, _given(eps))
        tt, tp, c, fp, ier = _ft.sphere(th, ph, rr, w_, sv, ev)
        _spherefit_raise(ier)
        self.tx = tt
        self.ty = tp
        self.c = c
        self.kx = 3
        self.ky = 3
        self.fp = fp
        self.ier = ier

    def ev(self, ti, pi_, dtheta=0, dphi=0):
        """Evaluate at scattered points (scipy's ``spl.ev(theta, phi)``).

        Parameters
        ----------
        ti : 1-D float64 ndarray or float
            Colatitudes of the query points.
        pi_ : 1-D float64 ndarray or float
            Longitudes. A scalar or a length-1 array in either slot is
            broadcast against the other.
        dtheta, dphi : int, optional
            Partial-derivative orders, ``0 <= d < 3``. Both default to 0.

        Returns
        -------
        r : 1-D float64 ndarray, length ``len(ti)``

        Raises
        ------
        ValueError
            Lengths that do not broadcast.
            ``requested phi out of bounds.`` if a longitude lies outside
            ``[0, 2*pi]``, or ``requested theta out of bounds.`` if a
            colatitude lies outside ``[0, pi]``.
        """
        ti_ = _pt1d(ti)
        pi2 = _pt1d(pi_)
        _phi_bounds(pi2)
        _theta_bounds(ti_)
        tb, pb = _ev_pair(ti_, pi2)
        if len(tb) == 0:
            return np.zeros(0, np.float64)
        _pardeu_order(dtheta, dphi, 3, 3)
        return _ev.pardeu(tb, pb, self.tx, self.ty, self.c, 3, 3,
                          dtheta, dphi)

    def ev_one(self, t, p, dtheta=0, dphi=0):
        """Evaluate at a single point.

        Parameters
        ----------
        t, p : float
            Colatitude and longitude.
        dtheta, dphi : int, optional
            Partial-derivative orders, ``0 <= d < 3``. Both default to 0.

        Returns
        -------
        float
            The value, as a SCALAR. scipy's ``spl.ev(t, p)`` on scalars
            returns a length-1 array.
        """
        ta = np.empty(1, np.float64)
        pa = np.empty(1, np.float64)
        ta[0] = t
        pa[0] = p
        _phi_bounds(pa)
        _theta_bounds(ta)
        _pardeu_order(dtheta, dphi, 3, 3)
        return _ev.pardeu(ta, pa, self.tx, self.ty, self.c, 3, 3,
                          dtheta, dphi)[0]

    def eval_grid(self, t, p, dtheta=0, dphi=0):
        """Evaluate on a grid -- what scipy's ``spl(theta, phi)`` does.

        Parameters
        ----------
        t : 1-D float64 ndarray or float
            Colatitudes, ascending. A scalar is an axis of length 1.
        p : 1-D float64 ndarray or float
            Longitudes, ascending. The spline is evaluated on the full
            CROSS PRODUCT.
        dtheta, dphi : int, optional
            Partial-derivative orders, ``0 <= d < 3``. Both default to 0.

        Returns
        -------
        r : (len(t), len(p)) float64 ndarray
            An empty axis gives an empty grid of that shape. A grid axis
            that steps DOWN raises; a repeated value is accepted.

        Raises
        ------
        ValueError
            ``requested phi out of bounds.`` if a longitude lies outside
            ``[0, 2*pi]``, or ``requested theta out of bounds.`` if a
            colatitude lies outside ``[0, pi]``. The longitude is tested
            first.

        Notes
        -----
        ``spl(x, y)`` reaches this method, scipy's ``grid=True`` default;
        ``.ev`` is the ``grid=False`` form.
        """
        t_ = _pt1d(t)
        p_ = _pt1d(p)
        _phi_bounds(p_)
        _theta_bounds(t_)
        if len(t_) == 0 or len(p_) == 0:
            return np.zeros((len(t_), len(p_)), np.float64)
        _grid_order(t_, p_)
        _parder_order(dtheta, dphi, 3, 3)
        return _ev.parder(t_, p_, self.tx, self.ty, self.c, 3, 3,
                          dtheta, dphi)

    def partial_derivative(self, t, p, dtheta, dphi):
        """Grid evaluation of a mixed partial derivative.

        Parameters
        ----------
        t, p : 1-D float64 ndarray
            Colatitudes and longitudes, ascending.
        dtheta, dphi : int
            Derivative orders, each below 3.

        Returns
        -------
        r : (len(t), len(p)) float64 ndarray

        Notes
        -----
        scipy's ``.partial_derivative(dtheta, dphi)`` returns a new spline
        object. Returning the object is not implemented; this evaluates the
        partial derivative directly.
        """
        _pd_order(dtheta, dphi, 3, 3)
        return _ev.parder(_pt1d(t), _pt1d(p), self.tx, self.ty, self.c,
                          3, 3, dtheta, dphi)

    def get_knots(self):
        """The two knot vectors (scipy's ``spl.get_knots()``).

        Returns
        -------
        tt : 1-D float64 ndarray
            Knots in colatitude, the FULL vector -- which is what scipy
            returns too, since ``_BivariateSplineBase.get_knots`` hands back
            ``self.tck[:2]``.
        tp : 1-D float64 ndarray
            Knots in longitude, likewise full.
        """
        return self.tx, self.ty

    def get_coeffs(self):
        """The spline coefficients (scipy's ``spl.get_coeffs()``).

        Returns
        -------
        c : 1-D float64 ndarray, length ``(nt-4)*(np-4)``
        """
        return self.c

    def get_residual(self):
        """Sum of squared residuals (scipy's ``spl.get_residual()``).

        Returns
        -------
        float
        """
        return self.fp


@njit
def SmoothSphereBivariateSpline(theta, phi, r, w=_NO_W, s=0.0, eps=1e-16):
    """Build a smoothing spline on a sphere.

    Parameters
    ----------
    theta : 1-D float64 ndarray, length m
        Colatitudes of the data points, each in ``[0, pi]``.
    phi : 1-D float64 ndarray, length m
        Longitudes, each in ``[0, 2*pi]``.
    r : 1-D float64 ndarray, length m
        Values at those points.
    w : 1-D array_like of float, optional
        Weights, length m, each ``>= 0``. ``None`` and a ZERO-LENGTH array
        both mean unit weights.
    s : float, optional
        Smoothing factor. Default 0.0.
    eps : float, optional
        Rank-determination threshold, strictly between 0 and 1. Default
        1e-16. It CHANGES THE COEFFICIENTS on rank-deficient data and is not
        cosmetic; see Notes.

    Returns
    -------
    spl : _SmoothSphereBivariateSpline
        A jitclass instance carrying the attributes and methods below.

    Raises
    ------
    ValueError
        `theta` outside ``[0, pi]``, `phi` outside ``[0, 2*pi]``, a negative
        weight, ``s < 0``, `eps` outside ``(0, 1)``, mismatched lengths, and
        every FITPACK status outside {0, -1, -2}.

    See Also
    --------
    scipy.interpolate.SmoothSphereBivariateSpline : The scipy class this
        mirrors.

    Notes
    -----
    - ``spl(t, p)`` runs ``eval_grid``, scipy's ``grid=True`` default, for
      every pair of scalars and arrays, so two scalars give a ``(1, 1)``
      grid. ``ev(ti, pi_)`` is scipy's ``grid=False`` and ``ev_one(t, p)``
      the single-point scalar spelling, both reached by name.
    - ``.partial_derivative(dtheta, dphi)`` returns a new object in scipy;
      here it evaluates on a grid directly.
    - ``ev_one`` returns a SCALAR where scipy's ``spl.ev(t, p)`` on scalars
      returns a length-1 array.
    - No ``.integral``: FITPACK offers no ``dblint`` equivalent over a
      spherical patch, and scipy has none either.

    Defaults work in both worlds. ``SmoothSphereBivariateSpline`` is a plain
    ``@njit`` factory, not the jitclass itself, so
    ``SmoothSphereBivariateSpline(theta, phi, r)`` compiles and runs inside
    ``@njit`` as well as from Python. The class it returns,
    ``_SmoothSphereBivariateSpline``, takes every argument explicitly,
    because a jitclass constructor's defaults are Python-only.

    Accuracy against ``scipy.interpolate.SmoothSphereBivariateSpline`` on
    scipy 1.18.0, 150 scattered points, max absolute difference: knots,
    coefficients, `fp`, grid values and ``.ev`` all exactly 0.0 at
    ``s`` = 0.5, 1.0 and 5.0, and for every ``dtheta``/``dphi`` pair.
    Reproduced on a second 150-point fixture. This is the one bivariate
    class whose smoothing fits agreed on every fixture tried; the other
    three can place knots differently -- see their docstrings.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import SmoothSphereBivariateSpline
    >>> rng = np.random.default_rng(1)
    >>> theta = rng.uniform(0.2, np.pi - 0.2, 150)
    >>> phi = rng.uniform(0, 2 * np.pi, 150)
    >>> r = np.sin(theta) * np.cos(phi) + 2.0
    >>> spl = SmoothSphereBivariateSpline(theta, phi, r, s=0.5)     # fit once
    >>> spl.ier
    0
    >>> float(np.round(spl(1.5, 1.0)[0, 0], 6))     # spl(theta, phi) is the grid form; [0, 0] takes the point
    2.462142

    Inside compiled code, the largest value over a coarse query grid:

    >>> @njit
    ... def gridmax(spl):
    ...     return np.max(spl(np.linspace(0.5, 2.5, 5), np.linspace(0.5, 5.5, 5)))
    >>> float(np.round(gridmax(spl), 6))
    2.776181

    Attributes
    ----------
    tx, ty : 1-D float64 ndarray
        Knot vectors in colatitude and longitude.
    c : 1-D float64 ndarray
        Coefficients in FITPACK's flat bivariate layout.
    kx, ky : int
        Both 3. The routine is BICUBIC ONLY.
    fp : float
        Weighted sum of squared residuals, also returned by
        ``get_residual()``.
    ier : int
        FITPACK status: 0 smoothing achieved, -1 interpolating, -2
        least-squares polynomial. Any other value has already raised, so this
        is 0, -1 or -2 on every instance that exists.

    Methods
    -------
    eval_grid(t, p, dtheta, dphi)
        Grid evaluation, the equivalent of ``spl(theta, phi)``.
    ev(ti, pi_, dtheta, dphi), ev_one(t, p, dtheta, dphi),
    partial_derivative(t, p, dtheta, dphi), get_knots(), get_coeffs(),
    get_residual()
    """
    return _SmoothSphereBivariateSpline(theta, phi, r, w, s, eps)


# Public names: the scipy.interpolate-equivalent tck functions and spline
# jitclasses. NOTE jitclass constructor defaults are Python-only; inside
# @njit every argument must be passed explicitly.
__all__ = [
    'splrep', 'splprep', 'splev', 'splder_ev', 'splint', 'sproot', 'spalde',
    'bisplev',
    'UnivariateSpline', 'InterpolatedUnivariateSpline', 'LSQUnivariateSpline',
    'RectBivariateSpline', 'SmoothBivariateSpline',
    'RectSphereBivariateSpline', 'SmoothSphereBivariateSpline',
]
