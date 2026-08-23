"""BSpline, make_interp_spline and Akima1DInterpolator, in numba.

Everything here is plain numba (jitclasses + @njit functions): no ``.so``, no
ctypes callback, no module state -> **fully prange-safe** and constructible
*and* evaluable inside ``@njit`` code. This complements the FITPACK-backed
spline classes in ``interpolate.py`` (which wrap Dierckx' Fortran and take
whatever knots FITPACK picks): ``BSpline`` here accepts an **arbitrary**
``(t, c, k)`` triple, so it also evaluates hand-built / textbook / non-FITPACK
knot vectors.

Classes / functions                       scipy equivalent
    BSpline(t, c, k, extrapolate, axis)   scipy.interpolate.BSpline
    make_interp_spline(x, y, k, bc_type, axis)
                                          scipy.interpolate.make_interp_spline
    Akima1DInterpolator(x, y, axis, method, extrapolate)
                                          scipy.interpolate.Akima1DInterpolator
    splder(t, c, k, nu) or (tck, nu)  scipy.interpolate.splder
    splantider(t, c, k, nu) or (tck, nu)
                                          scipy.interpolate.splantider

Algorithm: the de Boor recursion, following scipy's ``_deBoor_D`` /
``evaluate_spline`` / ``find_interval`` (``scipy/interpolate/_bspl.pyx``,
now ``_dierckx``). FITPACK's ``splev`` is *not* used -- it only accepts
FITPACK's own knot layout, whereas ``BSpline`` must take arbitrary knots, and
writing it in numba keeps the class prange-safe with no shared-library call.

Methods (matching the jitclass naming used throughout this package). The
shapes below are for a 1-D ``c`` or ``y``; for an N-D one an array query
returns that shape with the interpolation axis replaced by ``len(xs)``, and a
scalar query returns it with that axis removed.
    .ev(xs)                   batch evaluation, xs 1-D -> 1-D
    .ev_one(x)                single point -> scalar
    .derivative_ev(xs, nu)    nu-th derivative, batch   (nu > k -> zeros)
    .antiderivative_ev(xs, nu)  nu-th antiderivative, batch   (BSpline only)
    .integral(a, b)           definite integral, == scipy ``.integrate(a, b)``
    .get_knots() / .get_coeffs()
    ``spl(xs)`` runs ``.ev``, ``spl(x)`` runs ``.ev_one``, and ``spl[xs]``
    is sugar for ``spl.ev(xs)``.

A jitclass constructor's defaults are Python-only, of any type, so inside
``@njit`` every argument has to be passed explicitly. ``BSpline`` and
``Akima1DInterpolator`` here are ``@njit`` FACTORIES over the private classes
``_BSpline`` and ``_Akima1DInterpolator``, and a plain ``@njit`` function's
defaults do work in both worlds, so ``BSpline(t, c, k)`` and
``Akima1DInterpolator(x, y)`` construct from Python and from inside ``@njit``
alike. ``make_interp_spline`` is a plain ``@njit`` function for the same
reason.

Boundary conditions for ``make_interp_spline``, as scipy spells them:
    'not-a-knot'  scipy's default; any k >= 1
    'natural'     second derivative = 0 at both ends (k = 3 only)
    'clamped'     first  derivative = 0 at both ends (k = 3 only)

Deviations from scipy (documented, deliberate)
    * Real coefficients only; scipy's complex ``c`` is not supported. An
      N-D ``c`` IS supported, at rank 1, 2 or 3, with ``axis`` naming the
      interpolation axis; rank 4 and above raise.
    * No ``bc_type=((order, value), ...)`` pairs on
      ``make_interp_spline`` -- only the three conditions above.
    * ``make_interp_spline`` takes no explicit knot vector ``t`` (build the
      ``BSpline`` directly to supply a knot vector) and no weights.
    * The collocation system is solved densely (``np.linalg.solve`` ->
      LAPACK ``dgesv``) instead of scipy's banded ``dgbsv``. Same partial
      pivoting; measured agreement on 14 random nodes is exactly 0.0 for
      k = 2, 3, 4, 5 and 1.1e-16 for k = 1, both in the coefficients and in
      the evaluated values. Cost is O(n^3) time and O(n^2) memory, so keep n
      to a few thousand. (Same trade-off as ``_cubic.py``'s tridiagonal
      solve.)
    * ``nu > k`` returns zeros.
    * Extra trailing coefficients are ignored, exactly like scipy: only the
      first ``n = len(t) - k - 1`` entries of ``c`` are used, so a FITPACK
      ``tck`` (where ``len(c) == len(t)``) can be passed straight in. In
      ``splder`` / ``splantider`` those are the only two lengths accepted and
      any other raises, where scipy validates nothing.
    * ``Akima1DInterpolator`` takes both of scipy's slope rules, ``method=0``
      or ``'akima'`` and ``method=1`` or ``'makima'``, real ``y`` only, and
      keeps scipy's default of NOT extrapolating -> NaN outside
      ``[x[0], x[-1]]``. That default is the OPPOSITE of ``BSpline``'s.

Naming, to avoid a real collision: ``splder`` / ``splantider`` here
are the counterparts of scipy's ``splder`` / ``splantider``, and the package
also exports them under those scipy names as aliases -- the same objects.
They are NOT the ``splder`` found in ``scijit.interpolate.evaluators``, which
is the raw FITPACK routine with Dierckx's own calling convention. Do not mix
the two up.

Both take scipy's ``tck`` tuple, ``splder(tck, nu)``, the three components
spelled out, ``splder(t, c, k, nu)``, and a ``BSpline``, ``splder(spl, nu)``.
The first argument's type selects; a ``tck`` or the components return a
``tck`` triple and a ``BSpline`` returns a ``BSpline``. The order argument is
named ``nu`` where scipy names it ``n``, so scipy's ``splder(tck, n=2)`` is
written ``splder(tck, 2)``.

Examples
--------
>>> import numpy as np
>>> from numba import njit
>>> from scijit.interpolate import make_interp_spline
>>> x = np.linspace(0.0, 1.0, 11)
>>> spl = make_interp_spline(x, np.sin(3.0 * x), 3, None, 'not-a-knot')
>>> @njit
... def work(spl, xs):
...     return spl(xs), spl.derivative_ev(xs, 1), spl.integral(0.0, 1.0)
>>> v, d, i = work(spl, np.array([0.25, 0.75]))
>>> np.round(v, 6)
array([0.681629, 0.778063])
>>> float(np.round(i, 6))
0.663332
"""
import numpy as np
from numba import njit, float64, int64, boolean, types
from numba.core.errors import TypingError
from numba.extending import overload
from scijitclass import scijitclass

from ._cubic import (_build_c, _find_interval as _find_seg, _eval_seg,
                     _bc_code, _check_finite, _eval_seg_nd,
                     _extrap_code, _EXTRAP_OFF, _EXTRAP_ON, _EXTRAP_PERIODIC,
                     _akima_real_y, _wrap_batch, _wrap_one)
from . import _ndaxis as _nda
from ._ndaxis import (_ND_RANKS, _check_axis, _coeff_body, _define,
                      _dispatch_branches, _rank_phrase,
                      _shape_words)


# ---------------------------------------------------------------------------
# de Boor core (follows scipy _bspl.pyx / _dierckx)
# ---------------------------------------------------------------------------

@njit
def _find_interval(t, k, xval, prev_l, extrapolate):
    """Locate the knot interval containing `xval`. Internal.

    Follows scipy's ``find_interval`` (``_bspl.pyx``).

    Parameters
    ----------
    t : 1-D float64 ndarray
        Non-decreasing knot vector.
    k : int
        Spline degree.
    xval : float
        Point to locate.
    prev_l : int
        Interval index found for the previous point; used as a search hint,
        which makes a sorted batch O(1) per point. Pass `k` if there is none.
    extrapolate : bool
        If False, a point outside the base interval ``[t[k], t[n]]`` is
        reported as invalid instead of being clamped.

    Returns
    -------
    ell : int
        Index with ``t[ell] <= xval < t[ell+1]`` and ``k <= ell <= n-1``, or
        -1 for a NaN `xval`, or for an out-of-base-interval `xval` when
        `extrapolate` is False. Callers turn -1 into NaN output.
    """
    n = t.shape[0] - k - 1
    tb = t[k]
    te = t[n]

    if xval != xval:                 # NaN
        return -1
    if ((xval < tb) or (xval > te)) and not extrapolate:
        return -1

    if k < prev_l < n:
        ell = prev_l
    else:
        ell = k

    while xval < t[ell] and ell != k:
        ell -= 1
    ell += 1
    while xval >= t[ell] and ell != n:
        ell += 1
    return ell - 1


@njit
def _deboor_d(t, x, k, ell, m, result, hh):
    """The ``k+1`` non-zero B-spline basis values at `x`. Internal.

    Follows scipy's ``_deBoor_D``: ``k - m`` plain de Boor iterations
    followed by ``m`` derivative iterations.

    Parameters
    ----------
    t : 1-D float64 ndarray
        Knot vector.
    x : float
        Evaluation point.
    k : int
        Spline degree.
    ell : int
        Interval index from `_find_interval`.
    m : int
        Derivative order, 0 <= m <= k.
    result : 1-D float64 ndarray, length k+1
        OUTPUT scratch buffer, overwritten with the ``m``-th derivatives of
        the basis functions ``B_{i,k}`` for ``i = ell-k .. ell`` (the only
        ones non-zero on this interval).
    hh : 1-D float64 ndarray, length k+1
        Working scratch buffer, contents meaningless on return.

    Returns
    -------
    None
        The answer is written into `result`.
    """
    result[0] = 1.0
    for j in range(1, k - m + 1):
        for i in range(j):
            hh[i] = result[i]
        result[0] = 0.0
        for nn in range(1, j + 1):
            ind = ell + nn
            xb = t[ind]
            xa = t[ind - j]
            if xb == xa:
                result[nn] = 0.0
                continue
            w = hh[nn - 1] / (xb - xa)
            result[nn - 1] += w * (xb - x)
            result[nn] = w * (x - xa)

    for j in range(k - m + 1, k + 1):
        for i in range(j):
            hh[i] = result[i]
        result[0] = 0.0
        for nn in range(1, j + 1):
            ind = ell + nn
            xb = t[ind]
            xa = t[ind - j]
            if xb == xa:
                result[nn] = 0.0
                continue
            w = j * hh[nn - 1] / (xb - xa)
            result[nn - 1] -= w
            result[nn] = w


@njit
def _eval_spline(t, c, k, xs, nu, extrapolate):
    """Evaluate the `nu`-th derivative of the spline ``(t, c, k)``. Internal.

    Parameters
    ----------
    t : 1-D float64 ndarray
        Knot vector.
    c : 1-D float64 ndarray
        B-spline coefficients; only the first ``len(t) - k - 1`` are read.
    k : int
        Spline degree.
    xs : 1-D float64 ndarray
        Points to evaluate at, in any order (a sorted batch is faster: the
        interval search reuses the previous index).
    nu : int
        Derivative order. Negative raises ``ValueError``; ``nu > k`` returns
        all zeros, which is what the derivative of a degree-k spline is.
    extrapolate : bool
        If False, points outside ``[t[k], t[n]]`` give NaN.

    Returns
    -------
    out : 1-D float64 ndarray, same length as `xs`
    """
    m = xs.shape[0]
    out = np.empty(m, np.float64)
    if nu < 0:
        raise NotImplementedError(
            "Cannot do derivative order nu= " + str(nu))
    if nu > k:
        for j in range(m):
            out[j] = 0.0
        return out

    work = np.empty(k + 1, np.float64)
    hh = np.empty(k + 1, np.float64)
    ell = k
    for j in range(m):
        xp = xs[j]
        ell = _find_interval(t, k, xp, ell, extrapolate)
        if ell < 0:
            out[j] = np.nan
            ell = k
            continue
        _deboor_d(t, xp, k, ell, nu, work, hh)
        acc = 0.0
        for a in range(k + 1):
            acc += c[ell + a - k] * work[a]
        out[j] = acc
    return out


@njit
def _eval_spline_one(t, c, k, x, nu, extrapolate):
    """Scalar version of `_eval_spline`. Internal.

    Parameters
    ----------
    t, c, k : as in `_eval_spline`
    x : float
        Point to evaluate at.
    nu : int
        Derivative order; negative raises ``ValueError``, ``nu > k`` returns
        0.0.
    extrapolate : bool
        If False, a point outside ``[t[k], t[n]]`` returns NaN.

    Returns
    -------
    float
    """
    if nu < 0:
        raise ValueError("derivative order nu must be non-negative")
    if nu > k:
        return 0.0
    ell = _find_interval(t, k, x, k, extrapolate)
    if ell < 0:
        return np.nan
    work = np.empty(k + 1, np.float64)
    hh = np.empty(k + 1, np.float64)
    _deboor_d(t, x, k, ell, nu, work, hh)
    acc = 0.0
    for a in range(k + 1):
        acc += c[ell + a - k] * work[a]
    return acc


# ---------------------------------------------------------------------------
# knot / coefficient transforms (scipy _fitpack_impl.splder / splantider)
# ---------------------------------------------------------------------------

@njit
def _pad_c(t, c):
    """Pad or truncate `c` to ``len(t)``, the FITPACK convention. Internal.

    Parameters
    ----------
    t : 1-D float64 ndarray
        Knot vector; its length sets the output length.
    c : 1-D float64 ndarray
        Coefficients; shorter is zero-filled, longer is truncated.

    Returns
    -------
    out : 1-D float64 ndarray, length ``len(t)``
    """
    nt = t.shape[0]
    out = np.zeros(nt, np.float64)
    nc = c.shape[0]
    if nc > nt:
        nc = nt
    for i in range(nc):
        out[i] = c[i]
    return out


_C_LEN_MSG = "Knots, coefficients and degree are inconsistent."


@njit
def _check_c_len(nt, nc, k):
    """Refuse a `c` that is neither spelling of the coefficient array.

    Two lengths are meaningful: ``len(t) - k - 1``, the number of basis
    functions, and ``len(t)``, the FITPACK convention with ``k + 1`` trailing
    zeros that `splrep` returns. Anything else is a mistake.
    ``scipy.interpolate.splder`` validates nothing and reaches a numpy
    broadcast whose outcome depends on how wrong the length was, so this
    guard is a deliberate deviation. The text is scipy's own, raised by
    ``BSpline.__init__`` for the same fault.
    """
    if nc != nt - k - 1 and nc != nt:
        raise ValueError(_C_LEN_MSG)


@njit
def _splder_core(t, c, k, nu):
    """`splder`'s algorithm. Both entry points call this and nothing else.

    Kept separate so the two accepted spellings of the arguments have one
    implementation to disagree about rather than two.
    """
    if nu < 0:
        raise ValueError("order of derivative must be non-negative")
    _check_c_len(t.shape[0], c.shape[0], k)
    if nu > k:
        raise ValueError("Order of derivative (n = " + str(nu)
                         + ") must be <= order of spline (k = "
                         + str(k) + ")")
    tt = t.astype(np.float64)
    cc = _pad_c(tt, c)
    kk = k
    for _ in range(nu):
        nt = tt.shape[0]
        m = nt - kk - 2
        cn = np.zeros(nt - 2, np.float64)
        for i in range(m):
            dt = tt[kk + 1 + i] - tt[1 + i]
            if dt == 0.0:
                raise ValueError(
                    "The spline has internal repeated knots and is not "
                    "differentiable " + str(nu) + " times")
            cn[i] = (cc[i + 1] - cc[i]) * kk / dt
        tn = np.empty(nt - 2, np.float64)
        for i in range(nt - 2):
            tn[i] = tt[i + 1]
        tt = tn
        cc = cn
        kk -= 1
    return tt, cc, kk


@njit
def _splder_any(t, c, k, nu):
    """`_splder_signed` over a 1-D `c` or an ``(n, T)`` one.

    `c.ndim` is part of `c`'s numba type, so the branch is resolved while
    compiling and the return type follows the rank that survives.
    """
    if c.ndim == 1:
        return _splder_signed(t, c, k, nu)
    elif c.ndim == 2:
        return _splder_signed_nd(t, c, k, nu)
    raise ValueError("c must have rank 1 or 2")


@njit
def _splantider_any(t, c, k, nu):
    """`_splantider_signed` over a 1-D `c` or an ``(n, T)`` one."""
    if c.ndim == 1:
        return _splantider_signed(t, c, k, nu)
    elif c.ndim == 2:
        return _splantider_signed_nd(t, c, k, nu)
    raise ValueError("c must have rank 1 or 2")


def _is_bspl(v):
    """Is `v` an instance of one of the `BSpline` jitclasses?

    Python side. `_BSPL_CLASSES` is built after the generated N-D classes
    exist, further down this module.
    """
    return isinstance(v, _BSPL_CLASSES)


def _is_bspl_ty(v):
    """`_is_bspl` for an ``@overload`` chooser, which sees a numba type."""
    return (isinstance(v, types.ClassInstanceType)
            and v.classname in _BSPL_CLASSNAMES)


def _bspl_rebuild(spl, t2, c2, k2):
    """Rebuild `spl`'s own class around a new ``(t, c, k)``. Python side."""
    if c2.ndim == 1:
        return _BSpline(t2, c2, k2, spl.extrapolate, spl.periodic)
    cls = globals()["_BSplineND%d" % spl.rest.shape[0]]
    return cls(t2, c2, k2, spl.extrapolate, spl.rest, spl.axis, spl.periodic)


def _bspl_arm(ty, core):
    """Build the ``@overload`` body that takes a `BSpline` and returns one.

    `ty` is the argument's `ClassInstanceType`, so the class to construct is
    known while compiling and is baked in as a constant. The result is
    ``@njit`` because an ``@overload`` body can only call compiled code.
    """
    cls = globals()[ty.classname]
    if ty.classname == "_BSpline":
        def impl(spl, nu):
            t2, c2, k2 = core(spl.t, spl.c, spl.k, nu)
            return cls(t2, c2, k2, spl.extrapolate, spl.periodic)
        return njit(impl)

    def impl(spl, nu):
        t2, c2, k2 = core(spl.t, spl.c, spl.k, nu)
        return cls(t2, c2, k2, spl.extrapolate, spl.rest, spl.axis,
                   spl.periodic)
    return njit(impl)


def _tck_absent(v):
    """Was this optional argument omitted, or passed as ``None``?

    `v` is a plain Python value on the interpreter path and a numba type
    inside an ``@overload`` body, so all three spellings are tested. Same
    shape as ``interpolate._lit_bool``, which reads a literal flag the same
    way. Internal.
    """
    return (v is None or isinstance(v, types.NoneType)
            or (isinstance(v, types.Omitted) and v.value is None))


#: What a `c` holding one coefficient array per dimension is refused with.
#: `splprep` produces exactly that layout, so this is the message a caller who
#: feeds a parametric fit straight into `splder` reads.
_PARAM_C_TAIL = (" does not take a parametric c, one coefficient array per "
                 "dimension, which is what splprep returns. Pass one "
                 "dimension at a time as (t, c[j], k, n), or a rank-2 c with "
                 "one column per dimension, which is done in one call.")


def _param_c(v):
    """True when a coefficient container holds one array per dimension.

    A rank-2 `c` is one column per curve and IS differentiated here, in one
    call, so it is not this case. `v` is a numba type inside an ``@overload``
    body and a plain value on the interpreter path. Internal.
    """
    if isinstance(v, types.Type):
        if isinstance(v, (types.List, types.ListType)):
            return isinstance(v.dtype, types.Array)
        if isinstance(v, types.BaseTuple):
            return (len(v.types) > 0
                    and all(isinstance(e, types.Array) for e in v.types))
        return False
    if isinstance(v, np.ndarray):
        return False
    if isinstance(v, (list, tuple)):
        return len(v) > 0 and hasattr(v[0], '__len__')
    return False


_FLAT_C_TAIL = (" takes c as an ndarray. A list or tuple of numbers is not a "
                "coefficient array here; pass np.asarray(c).")


def _flat_c(v):
    """True when a coefficient container holds numbers rather than arrays.

    A 1-D coefficient vector spelled as a Python container. `v` is a numba
    type inside an ``@overload`` body and a plain value on the interpreter
    path. Internal.
    """
    if isinstance(v, types.Type):
        if isinstance(v, (types.List, types.ListType)):
            return isinstance(v.dtype, types.Number)
        if isinstance(v, types.BaseTuple):
            return (len(v.types) > 0
                    and all(isinstance(e, types.Number) for e in v.types))
        return False
    if isinstance(v, np.ndarray):
        return False
    if isinstance(v, (list, tuple)):
        return len(v) > 0 and not hasattr(v[0], '__len__')
    return False


def _c_arm(v):
    """Which refusal a coefficient container earns: 0 none, 1 parametric,
    2 a flat container. Internal."""
    if _param_c(v):
        return 1
    if _flat_c(v):
        return 2
    return 0


#: Message tail per `_c_arm` code, so the two refusals are worded once.
_C_TAIL = (None, _PARAM_C_TAIL, _FLAT_C_TAIL)


@njit
def _param_c_raise(anti):
    """Refuse a parametric `c`, as a statement.

    A statement rather than a `raise` written into the arm that needs it. An
    ``@overload`` arm whose body is only a ``raise`` types as returning
    ``none``, and the caller, which unpacks ``(t, c, k)``, then fails to
    compile with an error naming neither `c` nor its layout.
    """
    if anti:
        raise ValueError("splantider" + _PARAM_C_TAIL)
    raise ValueError("splder" + _PARAM_C_TAIL)


@njit
def _flat_c_raise(anti):
    """Refuse a flat container `c`, as a statement. Same shape as above."""
    if anti:
        raise ValueError("splantider" + _FLAT_C_TAIL)
    raise ValueError("splder" + _FLAT_C_TAIL)


@njit
def _bad_c_raise(anti, code):
    """Refuse a coefficient container, as a statement. `code` from `_c_arm`,
    baked in while the arm compiles."""
    if code == 1:
        _param_c_raise(anti)
    _flat_c_raise(anti)


def _param_c_of_tck(t):
    """The `c` slot of a ``tck`` argument, whether type or value. Internal."""
    if isinstance(t, types.BaseTuple):
        return t.types[1] if len(t.types) == 3 else None
    return t[1] if len(t) == 3 else None


def splder(t, c=None, k=None, n=1):
    """Spline representation of the `n`-th derivative of a spline.

    Two argument spellings: ``splder(t, c, k, n)`` passes the three
    components of the spline, ``splder(tck, n)`` passes the ``(t, c, k)``
    tuple.

    Parameters
    ----------
    t : 1-D float64 ndarray, tuple ``(t, c, k)``, or a `BSpline`
        Knot vector of the input spline, the whole spline representation, or
        a spline object. A tuple selects the ``tck`` spelling and a `BSpline`
        the object spelling; under both, `k` is not passed and the second
        positional argument is the derivative order. A `BSpline` in gives a
        `BSpline` out, carrying `extrapolate` and `periodic` through.
    c : float64 ndarray, optional
        B-spline coefficients. A FITPACK-style `c` (padded to ``len(t)``) or a
        bare length-``len(t)-k-1`` array both work; it is padded internally.
        Any OTHER length raises ``ValueError``. A rank-2 `c` holds one column
        per curve and every column is differentiated at once. Under the
        ``tck`` and object spellings this slot carries `n`.
    k : int, optional
        Degree of the input spline. Not passed under the ``tck`` spelling.
    n : int, optional
        Derivative order. A NEGATIVE order gives the antiderivative of the
        opposite order, so ``splder(tck, -1)`` is ``splantider(tck, 1)``.
        ``n > k`` raises ``ValueError``. Default 1.

    Returns
    -------
    t2 : 1-D float64 ndarray, length ``len(t) - 2*n``
        Knot vector of the derivative spline (the input's, with `n` knots
        stripped from each end).
    c2 : float64 ndarray, first axis ``len(t2)``
        Coefficients of the derivative spline, zero-padded to ``len(t2)``,
        and carrying a rank-2 `c`'s columns.
    k2 : int
        Degree of the derivative spline, ``k - n``.

    See Also
    --------
    scipy.interpolate.splder : The scipy routine this mirrors.

    Notes
    -----
    - Also exported under the name ``splder``.
      ``scijit.interpolate.evaluators.splder`` is a DIFFERENT function, the
      raw FITPACK routine with Dierckx's own calling convention.
    - ``scipy.interpolate.splder`` is marked legacy in scipy's own
      documentation, which points at ``BSpline.derivative`` for new code.
    - A `BSpline` argument returns a `BSpline` rather than the triple, as it
      does in scipy.
    - The two spellings share one implementation, `_splder_core`, and reach it
      from a Python body and from an ``@overload``. The first argument's TYPE
      selects, so they cannot be mixed: a tuple followed by `c` and `k` raises
      ``ValueError`` from Python and ``TypingError`` inside ``@njit``, and so
      does a two-argument call whose first argument is an array. The return
      ``(t2, c2, k2)`` is a tck triple under either spelling, and feeds
      ``splev`` unchanged.
    - Inside ``@njit`` the ``tck`` spelling takes a TUPLE. Measured on numba
      0.66: a heterogeneous list ``[t, c, k]`` has no numba type, as an
      argument or as a construction, so scipy's list spelling reaches only the
      Python entry, which accepts it.
    - Raises ``ValueError`` if an interior knot is repeated enough to make the
      spline non-differentiable that many times.
    - A `c` whose length is neither ``len(t) - k - 1`` nor ``len(t)`` raises
      ``ValueError``. ``scipy.interpolate.splder`` validates nothing and
      reaches a numpy broadcast whose outcome depends on how wrong the length
      was, so this is a deliberate deviation; the text is scipy's own, raised
      by ``BSpline.__init__`` for the same fault.
    - A `c` of rank 3 or more raises ``ValueError``, where scipy carries any
      number of trailing dimensions.
    - A PARAMETRIC `c`, one coefficient array per dimension, which is what
      `splprep` returns, raises ``ValueError`` naming the two spellings that
      work: one dimension at a time, ``splder(t, c[j], k, n)``, or a rank-2
      `c` with one column per dimension, differentiated in one call.
      ``scipy.interpolate.splder`` raises ``AttributeError: 'list' object has
      no attribute 'shape'`` on its own `splprep` output, and refuses the
      rank-2 spelling with a broadcast error.
    - A `c` given as a flat list or tuple of numbers raises ``ValueError``.
      ``scipy.interpolate.splder`` raises ``AttributeError: 'list' object has
      no attribute 'shape'`` for it.

    Accuracy vs ``scipy.interpolate.splder`` on a random k=3 spline with 12
    knots: knots, coefficients and degree all match exactly (0.0) for
    ``n = 1, 2, 3``, including the returned array lengths, and the two
    spellings return the same bytes.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import splrep, splev, splder
    >>> x = np.linspace(0.0, 3.0, 40)
    >>> tck = splrep(x, np.sin(x))
    >>> @njit
    ... def slope_at(tck, q):
    ...     return splev(q, splder(tck, 1))
    >>> np.round(slope_at(tck, np.array([0.0, 1.5])), 6)
    array([1.000006, 0.070737])
    >>> t2, c2, k2 = splder(tck[0], tck[1], tck[2], 1)
    >>> k2, bool(np.array_equal(t2, splder(tck, 1)[0]))
    (2, True)
    """
    if _is_bspl(t):
        if k is not None:
            raise ValueError(
                "splder takes either (t, c, k, n) or (tck, n), "
                "not a spline object followed by c and k")
        order = n if c is None else c
        return _bspl_rebuild(t, *_splder_any(t.t, t.c, t.k, order))
    if isinstance(t, (tuple, list)):
        if k is not None:
            raise ValueError(
                "splder takes either (t, c, k, n) or (tck, n), "
                "not a tck tuple followed by c and k")
        tt, cc, kk = t
        arm = _c_arm(cc)
        if arm:
            raise ValueError("splder" + _C_TAIL[arm])
        return _splder_any(tt, cc, kk, n if c is None else c)
    if c is None or k is None:
        raise ValueError(
            "splder needs t, c and k, or a single tck tuple")
    arm = _c_arm(c)
    if arm:
        raise ValueError("splder" + _C_TAIL[arm])
    return _splder_any(t, c, k, n)


@overload(splder)
def _splder_ovl(t, c=None, k=None, n=1):
    """`splder` inside ``@njit``: one compiled body per argument spelling.

    The first argument's type decides. A `BSpline` instance gives another
    `BSpline`; a tuple is a ``tck`` and its second positional argument is
    `n`; an array is `t` and needs `c` and `k` beside it. A mixed call
    returns ``None`` from here, which numba reports as a ``TypingError`` at
    the call site.
    """
    if _is_bspl_ty(t):
        if not _tck_absent(k):
            return None             # spline object plus c and k
        arm = _bspl_arm(t, _splder_any)
        if _tck_absent(c):
            def impl(t, c=None, k=None, n=1):
                return arm(t, n)
            return impl

        def impl(t, c=None, k=None, n=1):
            return arm(t, c)
        return impl

    if isinstance(t, types.BaseTuple):
        if not _tck_absent(k):
            return None             # tck plus c and k -> TypingError
        arm = _c_arm(_param_c_of_tck(t))
        if arm:
            def impl(t, c=None, k=None, n=1):
                _bad_c_raise(False, arm)
                return (np.empty(0, np.float64), np.empty(0, np.float64), 0)
            return impl
        if _tck_absent(c):
            def impl(t, c=None, k=None, n=1):
                tt, cc, kk = t
                return _splder_any(tt, cc, kk, n)
            return impl

        def impl(t, c=None, k=None, n=1):
            tt, cc, kk = t
            return _splder_any(tt, cc, kk, c)
        return impl

    if _tck_absent(c) or _tck_absent(k):
        return None                 # t without c or k -> TypingError

    arm = _c_arm(c)
    if arm:
        def impl(t, c=None, k=None, n=1):
            _bad_c_raise(False, arm)
            return (np.empty(0, np.float64), np.empty(0, np.float64), 0)
        return impl

    def impl(t, c=None, k=None, n=1):
        return _splder_any(t, c, k, n)
    return impl


@njit
def _splantider_core(t, c, k, nu):
    """`splantider`'s algorithm. Both entry points call this and nothing
    else.

    Kept separate so the two accepted spellings of the arguments have one
    implementation to disagree about rather than two.
    """
    if nu < 0:
        raise ValueError("order of antiderivative must be non-negative")
    _check_c_len(t.shape[0], c.shape[0], k)
    tt = t.astype(np.float64)
    cc = _pad_c(tt, c)
    kk = k
    for _ in range(nu):
        nt = tt.shape[0]
        m = nt - kk - 1
        cn = np.zeros(nt + 2, np.float64)
        run = 0.0
        for i in range(m):
            run += cc[i] * (tt[kk + 1 + i] - tt[i])
            cn[i + 1] = run / (kk + 1)
        last = cn[m]
        for i in range(m + 1, nt + 2):
            cn[i] = last
        tn = np.empty(nt + 2, np.float64)
        tn[0] = tt[0]
        for i in range(nt):
            tn[i + 1] = tt[i]
        tn[nt + 1] = tt[nt - 1]
        tt = tn
        cc = cn
        kk += 1
    return tt, cc, kk


@njit
def _splder_signed(t, c, k, nu):
    """`splder`'s algorithm for an order of either sign.

    A negative order gives the antiderivative of the opposite order.
    """
    if nu < 0:
        return _splantider_core(t, c, k, -nu)
    return _splder_core(t, c, k, nu)


@njit
def _splantider_signed(t, c, k, nu):
    """`splantider`'s algorithm for an order of either sign.

    A negative order gives the derivative of the opposite order.
    """
    if nu < 0:
        return _splder_core(t, c, k, -nu)
    return _splantider_core(t, c, k, nu)


def splantider(t, c=None, k=None, n=1):
    """Spline representation of the `n`-th antiderivative of a spline.

    Two argument spellings: ``splantider(t, c, k, n)`` passes the three
    components of the spline, ``splantider(tck, n)`` passes the ``(t, c, k)``
    tuple.

    Parameters
    ----------
    t : 1-D float64 ndarray, tuple ``(t, c, k)``, or a `BSpline`
        Knot vector of the input spline, the whole spline representation, or
        a spline object. A `BSpline` in gives a `BSpline` out. Under the tuple
        and object spellings `k` is not passed and the second positional
        argument is the antiderivative order.
    c : float64 ndarray, optional
        B-spline coefficients; padded to ``len(t)`` internally, so both the
        FITPACK-padded and the bare form work, and any other length raises
        ``ValueError``. A rank-2 `c` holds one column per curve. Under the
        ``tck`` and object spellings this slot carries `n`.
    k : int, optional
        Degree of the input spline. Not passed under the ``tck`` spelling.
    n : int, optional
        Antiderivative order. A NEGATIVE order gives the derivative of the
        opposite order, so ``splantider(tck, -1)`` is ``splder(tck, 1)``.
        There is no upper limit. Default 1.

    Returns
    -------
    t2 : 1-D float64 ndarray, length ``len(t) + 2*n``
        Knot vector with the first and last knot repeated `n` more times.
    c2 : float64 ndarray, first axis ``len(t2)``
        Coefficients of the antiderivative, carrying a rank-2 `c`'s columns.
    k2 : int
        Degree, ``k + n``.

    See Also
    --------
    scipy.interpolate.splantider : The scipy routine this mirrors.

    Notes
    -----
    - Also exported under the name ``splantider``.
    - ``scipy.interpolate.splantider`` is marked legacy in scipy's own
      documentation, which points at ``BSpline.antiderivative`` for new code.
    - A `BSpline` argument returns a `BSpline` rather than the triple, as it
      does in scipy.
    - The integration constant is scipy's: the antiderivative vanishes at the
      left edge of the knot vector.
    - The two spellings share one implementation, `_splantider_core`. See
      `splder` for how the first argument's type selects between them,
      what a mixed call raises, and why a list reaches only the Python entry.
    - A `c` whose length is neither ``len(t) - k - 1`` nor ``len(t)`` raises
      ``ValueError``, where scipy validates nothing and reaches a numpy
      broadcast. Deliberate; see `splder`.
    - A `c` of rank 3 or more raises ``ValueError``, where scipy carries any
      number of trailing dimensions.
    - A PARAMETRIC `c`, one coefficient array per dimension, which is what
      `splprep` returns, raises ``ValueError`` naming the two spellings that
      work: one dimension at a time, ``splantider(t, c[j], k, n)``, or a rank-2
      `c` with one column per dimension, integrated in one call.
      ``scipy.interpolate.splantider`` raises ``AttributeError: 'list'
      object has no attribute 'shape'`` on its own `splprep` output, and
      refuses the rank-2 spelling with a broadcast error.
    - A `c` given as a flat list or tuple of numbers raises ``ValueError``.
      ``scipy.interpolate.splantider`` raises ``AttributeError: 'list' object
      has no attribute 'shape'`` for it.

    Accuracy vs ``scipy.interpolate.splantider`` on a random k=3 spline with
    12 knots: knots, coefficients, degree and array lengths all match exactly
    (0.0) for ``n = 1`` and ``n = 2``, and the two spellings return the same
    bytes.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import splrep, splev, splantider
    >>> x = np.linspace(0.0, np.pi, 60)
    >>> tck = splrep(x, np.sin(x))
    >>> @njit
    ... def running_area(tck, q):
    ...     return splev(q, splantider(tck, 1))
    >>> np.round(running_area(tck, np.array([np.pi / 2, np.pi])), 6)
    array([1., 2.])
    """
    if _is_bspl(t):
        if k is not None:
            raise ValueError(
                "splantider takes either (t, c, k, n) or (tck, n), "
                "not a spline object followed by c and k")
        order = n if c is None else c
        return _bspl_rebuild(t, *_splantider_any(t.t, t.c, t.k, order))
    if isinstance(t, (tuple, list)):
        if k is not None:
            raise ValueError(
                "splantider takes either (t, c, k, n) or (tck, n), "
                "not a tck tuple followed by c and k")
        tt, cc, kk = t
        arm = _c_arm(cc)
        if arm:
            raise ValueError("splantider" + _C_TAIL[arm])
        return _splantider_any(tt, cc, kk, n if c is None else c)
    if c is None or k is None:
        raise ValueError(
            "splantider needs t, c and k, or a single tck tuple")
    arm = _c_arm(c)
    if arm:
        raise ValueError("splantider" + _C_TAIL[arm])
    return _splantider_any(t, c, k, n)


@overload(splantider)
def _splantider_ovl(t, c=None, k=None, n=1):
    """`splantider` inside ``@njit``: one compiled body per spelling.

    `_splder_ovl`'s mechanism, over `_splantider_core`.
    """
    if _is_bspl_ty(t):
        if not _tck_absent(k):
            return None             # spline object plus c and k
        arm = _bspl_arm(t, _splantider_any)
        if _tck_absent(c):
            def impl(t, c=None, k=None, n=1):
                return arm(t, n)
            return impl

        def impl(t, c=None, k=None, n=1):
            return arm(t, c)
        return impl

    if isinstance(t, types.BaseTuple):
        if not _tck_absent(k):
            return None             # tck plus c and k -> TypingError
        arm = _c_arm(_param_c_of_tck(t))
        if arm:
            def impl(t, c=None, k=None, n=1):
                _bad_c_raise(True, arm)
                return (np.empty(0, np.float64), np.empty(0, np.float64), 0)
            return impl
        if _tck_absent(c):
            def impl(t, c=None, k=None, n=1):
                tt, cc, kk = t
                return _splantider_any(tt, cc, kk, n)
            return impl

        def impl(t, c=None, k=None, n=1):
            tt, cc, kk = t
            return _splantider_any(tt, cc, kk, c)
        return impl

    if _tck_absent(c) or _tck_absent(k):
        return None                 # t without c or k -> TypingError

    arm = _c_arm(c)
    if arm:
        def impl(t, c=None, k=None, n=1):
            _bad_c_raise(True, arm)
            return (np.empty(0, np.float64), np.empty(0, np.float64), 0)
        return impl

    def impl(t, c=None, k=None, n=1):
        return _splantider_any(t, c, k, n)
    return impl


# ---------------------------------------------------------------------------
# BSpline jitclass
# ---------------------------------------------------------------------------

@njit
def _eval_spline_nd(t, c2, k, xs, nu, extrapolate):
    """Evaluate T independent splines sharing ``(t, k)``, shape ``(m, T)``.

    The knot vector and the interval search are shared; only the coefficient
    lookup carries a trailing index. That is why the trailing axes are stored
    together rather than as separate splines.
    """
    m = xs.shape[0]
    tt = c2.shape[1]
    out = np.empty((m, tt), np.float64)
    if nu < 0:
        raise ValueError("derivative order nu must be non-negative")
    if nu > k:
        for j in range(m):
            for a in range(tt):
                out[j, a] = 0.0
        return out

    work = np.empty(k + 1, np.float64)
    hh = np.empty(k + 1, np.float64)
    ell = k
    for j in range(m):
        xp = xs[j]
        ell = _find_interval(t, k, xp, ell, extrapolate)
        if ell < 0:
            for a in range(tt):
                out[j, a] = np.nan
            ell = k
            continue
        _deboor_d(t, xp, k, ell, nu, work, hh)
        for a in range(tt):
            acc = 0.0
            for b in range(k + 1):
                acc += c2[ell + b - k, a] * work[b]
            out[j, a] = acc
    return out


@njit
def _eval_spline_one_nd(t, c2, k, x, nu, extrapolate):
    """Evaluate T independent splines at one point, shape ``(T,)``."""
    tt = c2.shape[1]
    out = np.empty(tt, np.float64)
    if nu < 0:
        raise ValueError("derivative order nu must be non-negative")
    if nu > k:
        for a in range(tt):
            out[a] = 0.0
        return out
    ell = _find_interval(t, k, x, k, extrapolate)
    if ell < 0:
        for a in range(tt):
            out[a] = np.nan
        return out
    work = np.empty(k + 1, np.float64)
    hh = np.empty(k + 1, np.float64)
    _deboor_d(t, x, k, ell, nu, work, hh)
    for a in range(tt):
        acc = 0.0
        for b in range(k + 1):
            acc += c2[ell + b - k, a] * work[b]
        out[a] = acc
    return out


@njit
def _splantider_nd(t, c2, k, nu):
    """Antiderivative representation for T series, coefficients ``(na, T)``.

    The knot vector and degree depend on `t`, `k` and `nu` alone, so they are
    the same for every series and are taken from the first column.
    """
    tt = c2.shape[1]
    ta, ca0, ka = _splantider_core(t, np.ascontiguousarray(c2[:, 0]), k, nu)
    na = ca0.shape[0]
    ca = np.empty((na, tt), np.float64)
    for i in range(na):
        ca[i, 0] = ca0[i]
    for j in range(1, tt):
        _, cj, _ = _splantider_core(t, np.ascontiguousarray(c2[:, j]), k, nu)
        for i in range(na):
            ca[i, j] = cj[i]
    return ta, ca, ka


@njit
def _splder_nd(t, c2, k, nu):
    """`_splder_core` for T series, coefficients ``(nd, T)``.

    The knot vector and degree depend on `t`, `k` and `nu` alone, so they are
    the same for every series and are taken from the first column.
    """
    tt = c2.shape[1]
    td, cd0, kd = _splder_core(t, np.ascontiguousarray(c2[:, 0]), k, nu)
    nd = cd0.shape[0]
    cd = np.empty((nd, tt), np.float64)
    for i in range(nd):
        cd[i, 0] = cd0[i]
    for j in range(1, tt):
        _, cj, _ = _splder_core(t, np.ascontiguousarray(c2[:, j]), k, nu)
        for i in range(nd):
            cd[i, j] = cj[i]
    return td, cd, kd


@njit
def _splder_signed_nd(t, c2, k, nu):
    """`_splder_signed` for the ``(n, T)`` coefficient layout."""
    if nu < 0:
        return _splantider_nd(t, c2, k, -nu)
    return _splder_nd(t, c2, k, nu)


@njit
def _splantider_signed_nd(t, c2, k, nu):
    """`_splantider_signed` for the ``(n, T)`` coefficient layout."""
    if nu < 0:
        return _splder_nd(t, c2, k, -nu)
    return _splantider_nd(t, c2, k, nu)


@njit
def _check_bspline(t, ncoef, k):
    """Constructor checks shared by the 1-D and N-D BSpline classes.

    Returns the number of coefficients the knot vector implies.
    """
    if k < 0:
        raise ValueError("Spline order cannot be negative.")
    nt = t.shape[0]
    n = nt - k - 1
    if n < k + 1:
        raise ValueError("Need at least " + str(2 * k + 2)
                         + " knots for degree " + str(k))
    for i in range(nt - 1):
        if t[i + 1] < t[i]:
            raise ValueError("Knots must be in a non-decreasing order.")
    if not np.isfinite(t).all():
        raise ValueError("Knots should not have nans or infs.")
    two = False
    for i in range(k, n):
        if t[i + 1] > t[i]:
            two = True
            break
    if not two:
        raise ValueError("Need at least two internal knots.")
    if ncoef < n:
        raise ValueError("Knots, coefficients and degree are "
                         "inconsistent.")
    return n


@njit
def _periodic_area(ta, ca, ka, tb, te, a, b):
    """Definite integral of a periodic spline over ``[a, b]``.

    `ta, ca, ka` are the ANTIDERIVATIVE's tck and ``[tb, te]`` the base
    interval. The complete periods spanned by ``[a, b]`` are counted and the
    remainder added, so a range wider than the base interval is not
    extrapolated with the end polynomial. Same construction as
    ``_cubic._periodic_integral``, which is measured against scipy.
    """
    period = te - tb
    interval = b - a
    n_periods = np.floor(interval / period)
    left = interval - n_periods * period
    total = 0.0
    if n_periods != 0.0:
        total = n_periods * (_eval_spline_one(ta, ca, ka, te, 0, True)
                             - _eval_spline_one(ta, ca, ka, tb, 0, True))
    aa = tb + (a - tb) % period
    bb = aa + left
    if bb <= te:
        total += (_eval_spline_one(ta, ca, ka, bb, 0, True)
                  - _eval_spline_one(ta, ca, ka, aa, 0, True))
    else:
        total += (_eval_spline_one(ta, ca, ka, te, 0, True)
                  - _eval_spline_one(ta, ca, ka, aa, 0, True))
        total += (_eval_spline_one(ta, ca, ka, tb + left + aa - te, 0, True)
                  - _eval_spline_one(ta, ca, ka, tb, 0, True))
    return total


@njit
def _periodic_area_nd(ta, ca, ka, tb, te, a, b):
    """`_periodic_area` for the stored ``(n, T)`` coefficient layout."""
    period = te - tb
    interval = b - a
    n_periods = np.floor(interval / period)
    left = interval - n_periods * period
    total = np.zeros(ca.shape[1], np.float64)
    if n_periods != 0.0:
        total = n_periods * (_eval_spline_one_nd(ta, ca, ka, te, 0, True)
                             - _eval_spline_one_nd(ta, ca, ka, tb, 0, True))
    aa = tb + (a - tb) % period
    bb = aa + left
    if bb <= te:
        total = total + (_eval_spline_one_nd(ta, ca, ka, bb, 0, True)
                         - _eval_spline_one_nd(ta, ca, ka, aa, 0, True))
    else:
        total = total + (_eval_spline_one_nd(ta, ca, ka, te, 0, True)
                         - _eval_spline_one_nd(ta, ca, ka, aa, 0, True))
        total = total + (
            _eval_spline_one_nd(ta, ca, ka, tb + left + aa - te, 0, True)
            - _eval_spline_one_nd(ta, ca, ka, tb, 0, True))
    return total


_bspline_spec = [
    ('t', float64[:]),
    ('c', float64[:]),
    ('k', int64),
    ('n', int64),
    ('extrapolate', boolean),
    ('periodic', boolean),
]


@scijitclass(_bspline_spec)
class _BSpline:
    """Instance type behind the `BSpline` factory, which carries the
    documentation and the defaults. Every argument is explicit here: a
    jitclass constructor's defaults are Python-only.
    """

    def __init__(self, t, c, k, extrapolate=True, periodic=False):
        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        nt = t.shape[0]
        n = nt - k - 1
        if n < k + 1:
            raise ValueError("Need at least " + str(2 * k + 2)
                             + " knots for degree " + str(k))
        for i in range(nt - 1):
            if t[i + 1] < t[i]:
                raise ValueError("Knots must be in a non-decreasing order.")
        if not np.isfinite(t).all():
            raise ValueError("Knots should not have nans or infs.")
        # need at least two distinct knots in t[k:n+1]
        two = False
        for i in range(k, n):
            if t[i + 1] > t[i]:
                two = True
                break
        if not two:
            raise ValueError("Need at least two internal knots.")
        if c.shape[0] < n:
            raise ValueError("Knots, coefficients and degree are "
                             "inconsistent.")

        tf = np.empty(nt, np.float64)
        for i in range(nt):
            tf[i] = t[i]
        cf = np.empty(n, np.float64)
        for i in range(n):
            cf[i] = c[i]

        self.t = tf
        self.c = cf
        self.k = k
        self.n = n
        self.extrapolate = extrapolate
        self.periodic = periodic

    def ev(self, xs):
        """Evaluate the spline at an array of points (scipy's ``spl(xs)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at, in any order; a sorted batch is faster
            because the interval search reuses the previous index.

        Returns
        -------
        out : 1-D float64 ndarray, same length as `xs`
            Spline values. Entries outside ``[t[k], t[n]]`` are extrapolated
            if ``extrapolate`` is True and NaN otherwise; a NaN input gives a
            NaN output either way.
        """
        return _eval_spline(
            self.t, self.c, self.k,
            _wrap_bspl(self.t, self.k, self.n, xs, self.periodic), 0,
            self.extrapolate)

    def ev_one(self, x):
        """Evaluate the spline at a single point.

        Parameters
        ----------
        x : float
            Point to evaluate at.

        Returns
        -------
        float
            The spline value, or NaN if `x` is outside ``[t[k], t[n]]`` with
            ``extrapolate=False``.
        """
        return _eval_spline_one(
            self.t, self.c, self.k,
            _wrap_bspl_one(self.t, self.k, self.n, x, self.periodic), 0,
            self.extrapolate)

    def __getitem__(self, xs):
        """``spl[xs]`` -- sugar for ``spl.ev(xs)``.

        ``spl(xs)`` reaches the same method and is the spelling scipy uses.

        Parameters
        ----------
        xs : 1-D float64 ndarray

        Returns
        -------
        out : 1-D float64 ndarray
        """
        return _eval_spline(
            self.t, self.c, self.k,
            _wrap_bspl(self.t, self.k, self.n, xs, self.periodic), 0,
            self.extrapolate)

    def derivative_ev(self, xs, nu=1):
        """Evaluate the `nu`-th derivative (scipy's ``spl(xs, nu)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at.
        nu : int, optional
            Derivative order. A negative order raises
            ``NotImplementedError``. ``nu > k`` returns zeros, which is what
            the derivative of a degree-k spline is. Default 1. This method
            default works inside ``@njit`` (unlike a constructor default).

        Returns
        -------
        out : 1-D float64 ndarray, same length as `xs`
        """
        return _eval_spline(
            self.t, self.c, self.k,
            _wrap_bspl(self.t, self.k, self.n, xs, self.periodic), nu,
            self.extrapolate)

    def antiderivative_ev(self, xs, nu=1):
        """Evaluate the `nu`-th antiderivative.

        Equivalent to ``scipy.interpolate.BSpline.antiderivative(nu)(xs)``.

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at.
        nu : int, optional
            Antiderivative order, >= 0 (negative raises ``ValueError``).
            Default 1. The integration constant is scipy's -- the
            antiderivative vanishes at the left edge of the knot vector.

        Returns
        -------
        out : 1-D float64 ndarray, same length as `xs`

        Notes
        -----
        Builds the antiderivative's ``(t, c, k)`` with `splantider` on
        every call, so hoist it out of a hot loop if `xs` is small. Measured
        exactly 0.0 against scipy for ``nu=1`` at every degree k = 0..5.
        """
        ta, ca, ka = splantider(self.t, self.c, self.k, nu)
        return _eval_spline(ta, ca, ka, xs, 0, self.extrapolate)

    def integral(self, a, b):
        """Definite integral over ``[a, b]`` (scipy's ``BSpline.integrate``).

        Parameters
        ----------
        a, b : float
            Integration limits. ``b < a`` returns the negated integral, as
            scipy does. If ``extrapolate`` is False the limits are first
            clamped into the base interval ``[t[k], t[n]]``, again matching
            scipy; if it is True the end polynomials are integrated on into
            the extrapolated region; under ``'periodic'`` the whole periods
            spanned are counted.

        Returns
        -------
        float
            The definite integral.

        Notes
        -----
        Measured exactly 0.0 against ``scipy.interpolate.BSpline.integrate``
        over the base interval for every degree k = 0..5 on random knots.
        Builds the antiderivative tck on each call.
        """
        ta, ca, ka = splantider(self.t, self.c, self.k, 1)
        tb = self.t[self.k]
        te = self.t[self.n]
        if self.periodic:
            return _periodic_area(ta, ca, ka, tb, te, a, b)
        aa = a
        bb = b
        sign = 1.0
        if bb < aa:
            aa = b
            bb = a
            sign = -1.0
        if not self.extrapolate:
            if aa < tb:
                aa = tb
            if bb > te:
                bb = te
            if bb < aa:
                bb = aa
        hi = _eval_spline_one(ta, ca, ka, bb, 0, True)
        lo = _eval_spline_one(ta, ca, ka, aa, 0, True)
        return sign * (hi - lo)

    def get_knots(self):
        """Return the knot vector (scipy's ``BSpline.t``).

        Returns
        -------
        t : 1-D float64 ndarray
            The full knot vector, including the repeated boundary knots. A
            reference to the stored array, not a copy.
        """
        return self.t

    def get_coeffs(self):
        """Return the B-spline coefficients (scipy's ``BSpline.c``).

        Returns
        -------
        c : 1-D float64 ndarray, length ``n = len(t) - k - 1``
            Only the coefficients actually used -- if a FITPACK-style padded
            `c` was passed to the constructor, the trailing entries are not
            here. A reference, not a copy.
        """
        return self.c


_bspline_nd_spec = [
    ('t', float64[:]),
    ('c', float64[:, :]),
    ('k', int64),
    ('n', int64),
    ('extrapolate', boolean),
    ('rest', int64[:]),
    ('axis', int64),
    ('periodic', boolean),
]

_BSPLINE_ND_SRC = '''
class _BSplineND{trail}:
    """Instance type behind `BSpline` for a rank-{rank} coefficient array.

    T independent splines share one knot vector and one degree. `rest` carries
    the shape of the coefficient axes other than the interpolation axis,
    `axis` where that axis sat in the caller's `c`.

    Generated by ``_make_bspline_nd({rank})`` in
    ``scijit/interpolate/_bspline.py`` and bound there under this name. One
    class exists per supported rank of `c`; they share this source, the spec
    and every kernel, and differ only in the rank they hand back.
    `scijit.interpolate._ndaxis` says why.
    """

    def __init__(self, t, c2, k, extrapolate, rest, axis, periodic):
        n = _check_bspline(t, c2.shape[0], k)
        nt = t.shape[0]
        tf = np.empty(nt, np.float64)
        for i in range(nt):
            tf[i] = t[i]
        cf = np.empty((n, c2.shape[1]), np.float64)
        for i in range(n):
            for j in range(c2.shape[1]):
                cf[i, j] = c2[i, j]
        self.t = tf
        self.c = cf
        self.k = k
        self.n = n
        self.extrapolate = extrapolate
        self.rest = rest
        self.axis = axis
        self.periodic = periodic

    def ev(self, xs):
        """Evaluate every spline at an array of points (scipy's ``spl(xs)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at. With ``extrapolate=False`` a point outside
            ``[t[k], t[n]]`` gives NaN across every series.

        Returns
        -------
        out : {rank}-D float64 ndarray
            The shape of `c`, with the interpolation axis replaced by
            ``len(xs)``.
        """
        return _nda._restore{rank}(
            _eval_spline_nd(
                self.t, self.c, self.k,
                _wrap_bspl(self.t, self.k, self.n, xs, self.periodic), 0,
                self.extrapolate),
            self.rest, self.axis)

    def ev_one(self, x):
        """Evaluate every spline at a single point.

        Returns
        -------
        out : {trail}-D float64 ndarray
            The shape of `c` with the interpolation axis removed.
        """
        return _nda._point{rank}(
            _eval_spline_one_nd(
                self.t, self.c, self.k,
                _wrap_bspl_one(self.t, self.k, self.n, x, self.periodic), 0,
                self.extrapolate), self.rest)

    def __getitem__(self, xs):
        """``spl[xs]`` -- sugar for ``spl.ev(xs)``."""
        return _nda._restore{rank}(
            _eval_spline_nd(
                self.t, self.c, self.k,
                _wrap_bspl(self.t, self.k, self.n, xs, self.periodic), 0,
                self.extrapolate),
            self.rest, self.axis)

    def derivative_ev(self, xs, nu=1):
        """Evaluate the `nu`-th derivative of every spline.

        Returns
        -------
        out : {rank}-D float64 ndarray
            Shaped as `ev`. ``nu > k`` gives zeros.
        """
        return _nda._restore{rank}(
            _eval_spline_nd(
                self.t, self.c, self.k,
                _wrap_bspl(self.t, self.k, self.n, xs, self.periodic), nu,
                self.extrapolate),
            self.rest, self.axis)

    def antiderivative_ev(self, xs, nu=1):
        """Evaluate the `nu`-th antiderivative of every spline.

        Returns
        -------
        out : {rank}-D float64 ndarray
            Shaped as `ev`.
        """
        ta, ca, ka = _splantider_nd(self.t, self.c, self.k, nu)
        return _nda._restore{rank}(
            _eval_spline_nd(ta, ca, ka, xs, 0, self.extrapolate),
            self.rest, self.axis)

    def integral(self, a, b):
        """Definite integral of every spline over ``[a, b]``.

        Returns
        -------
        out : {trail}-D float64 ndarray
            The shape of `c` with the interpolation axis removed.
        """
        ta, ca, ka = _splantider_nd(self.t, self.c, self.k, 1)
        tb = self.t[self.k]
        te = self.t[self.n]
        if self.periodic:
            return _nda._point{rank}(
                _periodic_area_nd(ta, ca, ka, tb, te, a, b), self.rest)
        aa = a
        bb = b
        sign = 1.0
        if bb < aa:
            aa = b
            bb = a
            sign = -1.0
        if not self.extrapolate:
            if aa < tb:
                aa = tb
            if bb > te:
                bb = te
            if bb < aa:
                bb = aa
        hi = _eval_spline_one_nd(ta, ca, ka, bb, 0, True)
        lo = _eval_spline_one_nd(ta, ca, ka, aa, 0, True)
        return _nda._point{rank}(sign * (hi - lo), self.rest)

    def get_knots(self):
        """Return the knot vector (scipy's ``BSpline.t``).

        A reference to the stored array, not a copy.
        """
        return self.t

    def get_coeffs(self):
        """Return the coefficients (scipy's ``BSpline.c``).

        Returns
        -------
        c : (n, {shape}) float64 ndarray
            The interpolation axis stays at the front, which is where scipy
            keeps it too: ``BSpline(t, c, k, axis=2).c`` is stored moved, with
            ``.axis`` recording where it came from.
        """
{coeffs}
'''


def _make_bspline_nd(rank):
    """Build the `BSpline` N-D class for a `c` of this rank.

    Parameters
    ----------
    rank : int
        Rank of the caller's `c`, from 2 to ``1 + _ND_MAXTRAIL``.

    Returns
    -------
    cls : jitclass
        The class, decorated with `scijitclass` against `_bspline_nd_spec` and
        bound in this module as ``_BSplineND<rank-1>``.
    """
    trail = rank - 1
    src = _BSPLINE_ND_SRC.format(
        rank=rank, trail=trail, shape=_shape_words(trail),
        coeffs=_coeff_body(trail, "self.n"))
    return scijitclass(_bspline_nd_spec)(
        _define(src, globals(), "_BSplineND%d" % trail))


for _rank in _ND_RANKS:
    globals()["_BSplineND%d" % (_rank - 1)] = _make_bspline_nd(_rank)
del _rank


_BSPLINE_DISPATCH_SRC = '''
def _bspline_nd(t, c, k, extrapolate, axis, periodic):
    """Construct the `BSpline` class that matches the rank of `c`.

    `c.ndim` is part of `c`'s numba type, so every branch but one is removed
    while compiling and the survivor fixes the class this returns. A rank past
    the cap leaves only the ``raise``.

    Generated by ``_make_bspline_dispatch()``; see
    `scijit.interpolate._ndaxis`.
    """
{branches}
    raise ValueError("c must have rank {phrase}")
'''


def _make_bspline_dispatch():
    """Build `_bspline_nd`, one branch per supported rank of `c`."""
    def body(rank):
        return ("        flat, rest, ax = _nda._flatten%d(c, axis)\n"
                "        return _BSplineND%d(t, flat, k, extrapolate,\n"
                "                            rest, ax, periodic)"
                % (rank, rank - 1))
    src = _BSPLINE_DISPATCH_SRC.format(
        branches=_dispatch_branches("c", body), phrase=_rank_phrase())
    return njit(_define(src, globals(), "_bspline_nd"))


_bspline_nd = _make_bspline_dispatch()

# The jitclasses `splder` and `splantider` accept in place of a tck, and their
# names, which is all an `@overload` chooser sees. Built here because the N-D
# classes do not exist until the loop above has run.
_BSPL_CLASSES = tuple([_BSpline]
                      + [globals()["_BSplineND%d" % (r - 1)]
                         for r in _ND_RANKS])
_BSPL_CLASSNAMES = frozenset(c.class_type.class_name for c in _BSPL_CLASSES)


@njit
def BSpline(t, c, k, extrapolate=True, axis=0):
    """Univariate spline in the B-spline basis.

    ``S(x) = sum_j c[j] * B_{j,k;t}(x)``, evaluated with the de Boor
    recursion. This accepts an ARBITRARY knot vector, so hand-built or
    textbook knots work as well as a FITPACK ``tck``.

    Parameters
    ----------
    t : 1-D float64 ndarray
        Knot vector, non-decreasing and finite. Must hold at least ``2*k+2``
        knots and at least two distinct values in ``t[k:n+1]``; otherwise
        ``ValueError``. A decreasing pair, a NaN or an inf also raises.
    c : 1-D float64 ndarray
        B-spline coefficients. Only the first ``n = len(t) - k - 1`` entries
        are used and stored, so a FITPACK ``tck`` (whose `c` is padded to
        ``len(t)``) can be passed straight in. ``len(c) < n`` raises
        ``ValueError``. 1-D real only.
    k : int
        Spline degree, >= 0. Negative raises ``ValueError``.
    extrapolate : bool, str or None, optional
        What a query outside the base interval ``[t[k], t[n]]`` gives.

            True           the first/last polynomial piece, extended
            False          NaN
            'periodic'     the query folded into ``[t[k], t[n]]`` first
            None           NaN

        Default True. A string other than ``'periodic'`` raises
        ``ValueError``. 0 and 1 are accepted for False and True. May be a
        runtime value, not only a literal.
    axis : int, optional
        Which axis of `c` the interpolation runs along, default 0. Negative
        values count from the end. Ignored for a 1-D `c`. ``ev(xs)`` returns
        the shape of `c` with this axis replaced by ``len(xs)``; a scalar
        query returns it with this axis removed.

    Returns
    -------
    _BSpline
        A callable jitclass instance: ``spl(xs)`` runs `ev` and ``spl(x)``
        runs `ev_one`. A rank-2 or rank-3 `c` gives `_BSplineND1` or
        `_BSplineND2`, which carry the same methods.

    See Also
    --------
    scipy.interpolate.BSpline : The scipy routine this mirrors.

    Notes
    -----
    Deviations from scipy: no ``.derivative()`` / ``.antiderivative()``
    returning new objects (use the ``_ev`` methods, or ``splder`` /
    ``splantider`` for the tck); no ``.roots()``, ``.design_matrix``,
    ``.basis_element`` or ``.from_power_basis``. The resolved mode is split
    across the `extrapolate` and `periodic` attributes, where scipy keeps the
    caller's value in one; an unrecognised string raises where scipy
    extrapolates. An `axis` naming an axis of `c` that does not exist raises
    ``numpy.exceptions.AxisError``. `c` is float64, where scipy derives the
    coefficient dtype from `c` and supports ``complex128``. Evaluation is
    ``ev(xs)`` and ``derivative_ev(xs, nu)``, where scipy spells both as
    ``spl(xs, nu)``; a two-argument call raises on arity here.

    ``BSpline`` is an ``@njit`` factory over the jitclass ``_BSpline``, so
    ``extrapolate`` may be omitted from Python AND from inside ``@njit``.
    Method defaults such as ``derivative_ev``'s ``nu=1`` work in both too.

    Measured against scipy 1.18 on a k=3 spline over ``[0, 1]`` queried at 1.3
    and -0.2: ``extrapolate=True`` gives -10.21 and -1.672,
    ``extrapolate=False`` and ``extrapolate=None`` NaN,
    ``extrapolate='periodic'`` 1.018 and 1.74. ``integrate(-1, 2)`` under
    ``'periodic'`` gives 3.5625, three times the 1.1875 of one period.

    Accuracy vs ``scipy.interpolate.BSpline`` on random knots and
    coefficients, 137 points across the base interval, for every degree
    k = 0..5: values, first derivative, first antiderivative and
    ``integrate(t[k], t[n])`` all EXACTLY 0.0. With ``extrapolate=False`` the
    NaN pattern matches scipy's element for element and the finite values
    agree exactly.

    prange-safe: yes -- pure de Boor, no shared state, no library call.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import BSpline
    >>> t = np.array([0., 0., 0., 0., 1., 2., 3., 3., 3., 3.])
    >>> c = np.array([0., 1., -1., 2., 0., 1.])
    >>> spl = BSpline(t, c, 3)     # build once from knots and coefficients
    >>> np.round(spl(np.array([0.5, 1.5, 2.5])), 6)
    array([0.375, 0.5  , 0.625])

    Inside compiled code, scanning the domain for the curve's peak:

    >>> @njit
    ... def peak(spl):
    ...     xs = np.linspace(0.0, 3.0, 301)
    ...     return np.max(spl(xs))
    >>> float(np.round(peak(spl), 6))
    1.0

    Attributes
    ----------
    t : 1-D float64 ndarray
        The knot vector (a copy).
    c : 1-D float64 ndarray, length n
        The coefficients actually used (a copy, trailing entries dropped).
    k, n : int
        Degree and number of coefficients, ``n = len(t) - k - 1``.
    extrapolate : bool
        Whether a query outside the base interval is evaluated at all.
    periodic : bool
        Whether the query is folded into the base interval first.

    Methods
    -------
    ev(xs), ev_one(x), __getitem__(xs), derivative_ev(xs, nu),
    antiderivative_ev(xs, nu), integral(a, b), get_knots(), get_coeffs()
    """
    _check_axis(axis)
    _nda._check_axis_range(axis, c.ndim)
    # scipy stores an explicit `extrapolate=None` and reads it for truth, so
    # None gives NaN outside the base interval rather than the declared True.
    code = _extrap_code(extrapolate, _EXTRAP_OFF)
    on = code != _EXTRAP_OFF
    per = code == _EXTRAP_PERIODIC
    if c.ndim == 1:
        return _BSpline(t, c, k, on, per)
    else:
        return _bspline_nd(t, c, k, on, axis, per)


# ---------------------------------------------------------------------------
# make_interp_spline
# ---------------------------------------------------------------------------

@njit
def _not_a_knot_knots(x, k):
    """Not-a-knot knot vector for interpolation. Internal.

    Follows scipy's ``_not_a_knot``, handling odd and even `k` differently
    (even degrees place the interior knots at data midpoints).

    Parameters
    ----------
    x : 1-D float64 ndarray
        Interpolation abscissae, strictly increasing.
    k : int
        Spline degree, >= 1.

    Returns
    -------
    t : 1-D float64 ndarray, length ``len(x) + k + 1``
        Knot vector with ``x[0]`` and ``x[-1]`` each repeated ``k+1`` times.
        Measured identical (0.0) to scipy's for k = 1..5.
    """
    n = x.shape[0]
    if k % 2 == 1:
        k2 = (k + 1) // 2
        m = n - 2 * k2
        t = np.empty(m + 2 * (k + 1), np.float64)
        for i in range(m):
            t[k + 1 + i] = x[k2 + i]
    else:
        k2 = k // 2
        m = n - 1 - 2 * k2
        t = np.empty(m + 2 * (k + 1), np.float64)
        for i in range(m):
            t[k + 1 + i] = 0.5 * (x[k2 + i] + x[k2 + i + 1])
    for i in range(k + 1):
        t[i] = x[0]
        t[k + 1 + m + i] = x[n - 1]
    return t


@njit
def _augknt_knots(x, k):
    """Clamped knot vector ``[x0]*k + x + [xe]*k``. Internal.

    Follows scipy's ``_augknt``, used for the 'natural' and 'clamped'
    boundary conditions.

    Parameters
    ----------
    x : 1-D float64 ndarray
        Interpolation abscissae, strictly increasing.
    k : int
        Spline degree.

    Returns
    -------
    t : 1-D float64 ndarray, length ``len(x) + 2*k``
    """
    n = x.shape[0]
    t = np.empty(n + 2 * k, np.float64)
    for i in range(k):
        t[i] = x[0]
        t[n + k + i] = x[n - 1]
    for i in range(n):
        t[k + i] = x[i]
    return t


@njit
def _interp_coeffs(x, y, t, k, ol, vl, orr, vr):
    """Solve the collocation system for the B-spline coefficients. Internal.

    Mirrors scipy's ``_coloc`` + ``_handle_lhs_derivatives`` + banded solve,
    but assembles a DENSE matrix and calls ``np.linalg.solve`` (LAPACK
    ``dgesv``) -- same partial pivoting, O(n^3) time and O(n^2) memory.

    Parameters
    ----------
    x : 1-D float64 ndarray, length n
        Interpolation abscissae, strictly increasing.
    y : 1-D float64 ndarray, length n
        Values to interpolate.
    t : 1-D float64 ndarray
        Knot vector.
    k : int
        Spline degree.
    ol, vl : 1-D int64 ndarray, 1-D float64 ndarray
        Derivative orders imposed at ``x[0]`` and the values they take, so
        row i of the system reads ``S^(ol[i])(x[0]) = vl[i]``. Both empty when
        the left end carries no condition.
    orr, vr : 1-D int64 ndarray, 1-D float64 ndarray
        The same at ``x[-1]``, occupying the last rows.

    Returns
    -------
    c : 1-D float64 ndarray, length ``len(t) - k - 1``
        B-spline coefficients of the interpolating spline.

    Raises
    ------
    ValueError
        If the conditions do not use up exactly the coefficients the knot
        vector leaves free.
    """
    n = x.shape[0]
    nt = t.shape[0] - k - 1
    nleft = ol.shape[0]
    nright = orr.shape[0]
    if nt - n != nleft + nright:
        raise ValueError("The number of derivatives at the boundaries does "
                         "not match the number of free coefficients.")
    A = np.zeros((nt, nt), np.float64)
    rhs = np.zeros(nt, np.float64)
    work = np.empty(k + 1, np.float64)
    hh = np.empty(k + 1, np.float64)

    for i in range(nleft):
        ell = _find_interval(t, k, x[0], k, True)
        _deboor_d(t, x[0], k, ell, ol[i], work, hh)
        for a in range(k + 1):
            A[i, ell - k + a] = work[a]
        rhs[i] = vl[i]

    ell = k
    for j in range(n):
        ell = _find_interval(t, k, x[j], ell, True)
        _deboor_d(t, x[j], k, ell, 0, work, hh)
        for a in range(k + 1):
            A[nleft + j, ell - k + a] = work[a]
        rhs[nleft + j] = y[j]

    for i in range(nright):
        ell = _find_interval(t, k, x[n - 1], k, True)
        _deboor_d(t, x[n - 1], k, ell, orr[i], work, hh)
        for a in range(k + 1):
            A[nt - nright + i, ell - k + a] = work[a]
        rhs[nt - nright + i] = vr[i]

    return np.linalg.solve(A, rhs)


@njit
def _wrap_bspl(t, k, n, xs, periodic):
    """Fold query points into the base interval ``[t[k], t[n]]``.

    A periodic interpolating spline is built on knots taken on a circle and
    scipy evaluates it with ``extrapolate='periodic'``, which maps a query
    with ``x0 + (x - x0) % (xn - x0)`` first. Returns `xs` untouched when
    `periodic` is False.
    """
    if not periodic:
        return xs
    x0 = t[k]
    period = t[n] - x0
    out = np.empty(xs.shape[0], np.float64)
    for i in range(xs.shape[0]):
        out[i] = x0 + (xs[i] - x0) % period
    return out


@njit
def _wrap_bspl_one(t, k, n, xq, periodic):
    """`_wrap_bspl` for a single query point."""
    if not periodic:
        return xq
    x0 = t[k]
    return x0 + (xq - x0) % (t[n] - x0)


@njit
def _check_finite_xy(x, y, check_finite):
    """Refuse a non-finite `x` or `y`, which is what scipy's flag controls.

    scipy raises ValueError from ``np.asarray_chkfinite``. A NaN would
    otherwise pass the strictly-increasing loop, since every comparison
    against NaN is False, and reach the solve.
    """
    if not check_finite:
        return
    if not np.isfinite(x).all():
        raise ValueError("Array must not contain infs or NaNs")
    if not np.isfinite(y).all():
        raise ValueError("Array must not contain infs or NaNs")


@njit
def _periodic_knots(x, k):
    """Knots taken on a circle, scipy's ``_periodic_knots``.

    The interval widths wrap around, so the `k` knots beyond each end repeat
    the widths from the other end. An even degree also shifts the interior
    knots to the interval midpoints, as scipy does.

    Returns
    -------
    t : 1-D float64 ndarray, length ``len(x) + 2*k``
    """
    n = x.shape[0]
    xc = np.empty(n, np.float64)
    for i in range(n):
        xc[i] = x[i]
    if k % 2 == 0:
        dx0 = np.empty(n - 1, np.float64)
        for i in range(n - 1):
            dx0[i] = xc[i + 1] - xc[i]
        for i in range(1, n - 1):
            xc[i] -= dx0[i - 1] / 2.0
    dx = np.empty(n - 1, np.float64)
    for i in range(n - 1):
        dx[i] = xc[i + 1] - xc[i]
    t = np.zeros(n + 2 * k, np.float64)
    for i in range(n):
        t[k + i] = xc[i]
    for i in range(k):
        t[k - i - 1] = t[k - i] - dx[n - 2 - (i % (n - 1))]
        t[n + k + i] = t[n + k + i - 1] + dx[i % (n - 1)]
    return t


@njit
def _mis_periodic_core(x, y, t, k):
    """Coefficients of the periodic interpolating spline, dense solve.

    scipy's ``_make_interp_per_full_matr``. The first ``k-1`` rows make the
    first ``k-1`` derivatives agree at the two ends; the remaining ``n`` rows
    interpolate. scipy has a Woodbury-optimised banded route for the common
    case and falls back to this same full matrix when ``n <= k``; the two
    solve one system, and this package assembles dense systems throughout.

    Returns
    -------
    c : 1-D float64 ndarray, length ``n + k - 1``
    """
    n = x.shape[0]
    m = n + k - 1
    A = np.zeros((m, m), np.float64)
    b = np.zeros(m, np.float64)
    work = np.empty(k + 1, np.float64)
    hh = np.empty(k + 1, np.float64)

    for i in range(k - 1):
        _deboor_d(t, x[0], k, k, i + 1, work, hh)
        for a in range(k + 1):
            A[i, a] += work[a]
        _deboor_d(t, x[n - 1], k, n + k - 1, i + 1, work, hh)
        for a in range(k):
            A[i, m - k + a] -= work[a]

    for i in range(n):
        xv = x[i]
        if xv == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xv) - 1
        _deboor_d(t, xv, k, left, 0, work, hh)
        for a in range(k + 1):
            A[i + k - 1, left - k + a] = work[a]
        b[i + k - 1] = y[i]

    return np.linalg.solve(A, b)


@njit
def _mis_periodic_core_nd(x, y2, t, k):
    """`_mis_periodic_core` for T series, coefficients ``(nc, T)``."""
    tt = y2.shape[1]
    c0 = _mis_periodic_core(x, np.ascontiguousarray(y2[:, 0]), t, k)
    nc = c0.shape[0]
    c = np.empty((nc, tt), np.float64)
    for i in range(nc):
        c[i, 0] = c0[i]
    for j in range(1, tt):
        cj = _mis_periodic_core(x, np.ascontiguousarray(y2[:, j]), t, k)
        for i in range(nc):
            c[i, j] = cj[i]
    return t, c


_MIS_PERIODIC_MSG = ("First and last points does not match while periodic "
                     "case expected")


@njit
def _check_periodic_y(y2):
    """Refuse a periodic fit whose ends disagree.

    The bound is ``np.allclose(y[0], y[-1], atol=1e-15)``, which leaves
    numpy's default ``rtol=1e-5`` in force, so it is
    ``1e-15 + 1e-5 * abs(y[-1])`` and not the absolute ``1e-15`` alone.
    `_cubic._check_periodic_ends` uses a different bound, because
    `CubicSpline` passes ``rtol=1e-15`` where this passes nothing.
    """
    n = y2.shape[0]
    for j in range(y2.shape[1]):
        b = y2[n - 1, j]
        if abs(y2[0, j] - b) > 1e-15 + 1e-5 * abs(b):
            raise ValueError(_MIS_PERIODIC_MSG)


@njit
def _check_periodic_y_any(y, axis):
    """`_check_periodic_y` for an N-D `y`, before it has been flattened.

    The end-match check has to run whatever the degree, because scipy runs it
    before it resolves k, so it cannot wait for the flatten inside the rank
    dispatch.
    """
    ax = axis % y.ndim
    a = np.ascontiguousarray(np.moveaxis(y, ax, 0))
    n = a.shape[0]
    flat = np.ascontiguousarray(a.reshape(n, a.size // n))
    _check_periodic_y(flat)


@njit
def _check_given_knots(xf, tg, k, nfree):
    """Validate a caller-supplied knot vector, as scipy's checks do.

    Parameters
    ----------
    xf : 1-D float64 ndarray, length n
        Abscissae, already checked strictly increasing.
    tg : 1-D float64 ndarray
        The knot vector the caller passed.
    k : int
        Spline degree.
    nfree : int
        How many derivative conditions accompany it, so the collocation
        system is square.

    Raises
    ------
    ValueError
        If `tg` is not sorted, is shorter than ``n + k + 1``, leaves the
        system over- or under-determined, or does not cover ``[x[0], x[-1]]``.
    """
    n = xf.shape[0]
    nt = tg.shape[0]
    for i in range(nt - 1):
        if tg[i + 1] < tg[i]:
            raise ValueError("Expect t to be a 1-D sorted array_like.")
    if nt < n + k + 1:
        raise ValueError("Got fewer knots than needed: len(t) must be at "
                         "least len(x) + k + 1.")
    if nt - k - 1 != n + nfree:
        raise ValueError("The number of knots does not match the number of "
                         "data points and boundary conditions: "
                         "len(t) - k - 1 must equal len(x) plus the number "
                         "of derivatives given.")
    if xf[0] < tg[k] or xf[n - 1] > tg[nt - 1 - k]:
        raise ValueError("Out of bounds: t must cover [x[0], x[-1]].")


@njit
def _mis_core(xf, yf, k, tg, use_t, ol, vl, orr, vr, per):
    """Knot vector and coefficients for one interpolating spline.

    The cases `make_interp_spline` distinguishes, in one place so the 1-D and
    N-D entry points cannot drift apart: a caller-supplied knot vector, degree
    zero, the not-a-knot knot set, and derivative end conditions.

    Parameters
    ----------
    xf, yf : 1-D float64 ndarray
        Abscissae and values.
    k : int
        Spline degree.
    tg : 1-D float64 ndarray
        The caller's knot vector when `use_t` is True, and an empty array
        otherwise. It is an argument rather than an optional so that one
        compiled body serves both, which is what keeps the N-D dispatch from
        needing a second variant per rank.
    use_t : bool
        Whether `tg` carries a knot vector.
    ol, vl, orr, vr : ndarray
        Derivative orders and values at each end, as `_interp_coeffs` takes
        them. All four empty for the not-a-knot condition.
    """
    n = xf.shape[0]
    nfree = ol.shape[0] + orr.shape[0]
    if per:
        if use_t:
            raise NotImplementedError(
                "For periodic case t is constructed automatically and can "
                "not be passed manually")
        t = _periodic_knots(xf, k)
        return t, _mis_periodic_core(xf, yf, t, k)
    if use_t:
        if k == 0:
            raise ValueError("Too much info for k=0: t and bc_type can only "
                             "be None.")
        _check_given_knots(xf, tg, k, nfree)
        return tg, _interp_coeffs(xf, yf, tg, k, ol, vl, orr, vr)
    if k == 1 and nfree != 0:
        raise ValueError("Too much info for k=1: bc_type can only be None.")
    if k == 0:
        if nfree != 0:
            raise ValueError("Too much info for k=0: t and bc_type can only "
                             "be None.")
        t = np.empty(n + 1, np.float64)
        for i in range(n):
            t[i] = xf[i]
        t[n] = xf[n - 1]
        return t, yf
    if nfree == 0:
        t = _not_a_knot_knots(xf, k)
        return t, _interp_coeffs(xf, yf, t, k, ol, vl, orr, vr)
    # The augmented knot set leaves k-1 coefficients free, so that is exactly
    # how many derivative conditions it can absorb. scipy states the same
    # rule as a count rather than as a restriction on k.
    if nfree != k - 1:
        raise ValueError("The number of derivatives at boundaries does not "
                         "match the number of free coefficients: the "
                         "augmented knot vector leaves k-1 of them.")
    t = _augknt_knots(xf, k)
    return t, _interp_coeffs(xf, yf, t, k, ol, vl, orr, vr)


@njit
def _mis_core_nd(xf, y2, k, tg, use_t, ol, vl, orr, vr, per):
    """`_mis_core` for T independent series, coefficients ``(nc, T)``.

    The knot vector depends on `xf`, `k`, `tg` and the condition orders alone,
    so it is built once and every column solves against it.
    """
    if per:
        return _mis_periodic_core_nd(xf, y2, _periodic_knots(xf, k), k)
    tt = y2.shape[1]
    t, c0 = _mis_core(xf, np.ascontiguousarray(y2[:, 0]), k, tg, use_t,
                      ol, vl, orr, vr, per)
    nc = c0.shape[0]
    c = np.empty((nc, tt), np.float64)
    for i in range(nc):
        c[i, 0] = c0[i]
    for j in range(1, tt):
        _, cj = _mis_core(xf, np.ascontiguousarray(y2[:, j]), k, tg, use_t,
                          ol, vl, orr, vr, per)
        for i in range(nc):
            c[i, j] = cj[i]
    return t, c


# ---------------------------------------------------------------------------
# make_interp_spline's `bc_type`: scipy's own spelling, which differs from
# `CubicSpline`'s. Here each end takes a LIST of (order, value) pairs, where
# CubicSpline takes a single pair. scipy refuses each shape where the other
# belongs, and so does this.
# ---------------------------------------------------------------------------

_MIS_BC_MSG = ("bc_type must be 'not-a-knot', 'natural', 'clamped', "
               "'periodic', or a pair of LISTS of (order, value) pairs; "
               "CubicSpline's pair-of-pairs spelling belongs to that class, "
               "as it does in scipy")
_MIS_EMPTY_O = np.zeros(0, np.int64)
_MIS_EMPTY_V = np.zeros(0, np.float64)


@njit
def _mis_bc_from_code(code):
    """Expand a string's int code into per-end order and value arrays."""
    if code == 0 or code == 3:
        return (_MIS_EMPTY_O, _MIS_EMPTY_V, _MIS_EMPTY_O, _MIS_EMPTY_V)
    nu = 2 if code == 1 else 1
    ol = np.empty(1, np.int64)
    orr = np.empty(1, np.int64)
    ol[0] = nu
    orr[0] = nu
    return (ol, np.zeros(1, np.float64), orr, np.zeros(1, np.float64))


@njit
def _mis_bc_check(ol, orr, k):
    """Refuse a derivative order outside ``0 <= nu <= k``, as scipy does."""
    for i in range(ol.shape[0]):
        if ol[i] < 0 or ol[i] > k:
            raise ValueError("Bad boundary conditions at x[0]: every "
                             "derivative order must satisfy 0 <= nu <= k.")
    for i in range(orr.shape[0]):
        if orr[i] < 0 or orr[i] > k:
            raise ValueError("Bad boundary conditions at x[-1]: every "
                             "derivative order must satisfy 0 <= nu <= k.")


def _mis_is_periodic(bc_type):
    """True when `bc_type` names the periodic condition.

    Only a string can, so the other arms are resolved while compiling and the
    pair-of-lists spelling never runs a comparison.
    """
    return isinstance(bc_type, str) and bc_type == "periodic"


@overload(_mis_is_periodic)
def _mis_is_periodic_ovl(bc_type):
    """`_mis_is_periodic` inside ``@njit``."""
    if isinstance(bc_type, types.NoneType) or bc_type is None:
        def impl(bc_type):
            return False
        return impl
    if isinstance(bc_type, (types.UnicodeType, types.StringLiteral, str)):
        def impl(bc_type):
            return bc_type == "periodic"
        return impl

    def impl(bc_type):
        return False
    return impl


def _mis_bc(bc_type):
    """Resolve `make_interp_spline`'s `bc_type` to per-end arrays.

    Returns
    -------
    ol, vl, orr, vr : ndarray
        Derivative orders and values at each end, as `_interp_coeffs` takes
        them.
    """
    if bc_type is None:
        return _mis_bc_from_code(0)
    if isinstance(bc_type, str):
        return _mis_bc_from_code(_bc_code(bc_type))
    try:
        left, right = bc_type
        ol = np.array([int(p[0]) for p in left], np.int64)
        vl = np.array([float(p[1]) for p in left], np.float64)
        orr = np.array([int(p[0]) for p in right], np.int64)
        vr = np.array([float(p[1]) for p in right], np.float64)
    except Exception:
        raise ValueError("Derivatives, `bc_type`, should be specified as a "
                         "pair of iterables of pairs (order, value).")
    return ol, vl, orr, vr


@overload(_mis_bc)
def _mis_bc_ovl(bc_type):
    """`_mis_bc` inside ``@njit``, choosing the arm while compiling.

    A pair of LISTS is what scipy takes here. A pair of PAIRS, which is what
    `CubicSpline` takes, is refused during typing rather than read as one
    condition per end with the order in the value slot.
    """
    if isinstance(bc_type, types.NoneType) or bc_type is None:
        # scipy's default here is None, and it means 'not-a-knot': measured
        # 0.000e+00 between bc_type=None, the omitted argument and the
        # explicit string. `CubicSpline` does NOT accept None, and scipy
        # refuses it there too, so this arm is local to this function.
        def impl(bc_type):
            return _mis_bc_from_code(0)
        return impl
    if isinstance(bc_type, (types.UnicodeType, types.StringLiteral, str)):
        def impl(bc_type):
            return _mis_bc_from_code(_bc_code(bc_type))
        return impl
    if (isinstance(bc_type, types.BaseTuple) and len(bc_type) == 2
            and all(isinstance(e, (types.List, types.ListType,
                                   types.UniTuple, types.Tuple))
                    and not (isinstance(e, types.BaseTuple)
                             and len(e) == 2
                             and all(not isinstance(m, types.BaseTuple)
                                     for m in e))
                    for e in bc_type)):
        def impl(bc_type):
            left = bc_type[0]
            right = bc_type[1]
            ol = np.empty(len(left), np.int64)
            vl = np.empty(len(left), np.float64)
            i = 0
            for p in left:
                ol[i] = p[0]
                vl[i] = np.float64(p[1])
                i += 1
            orr = np.empty(len(right), np.int64)
            vr = np.empty(len(right), np.float64)
            i = 0
            for p in right:
                orr[i] = p[0]
                vr[i] = np.float64(p[1])
                i += 1
            return ol, vl, orr, vr
        return impl
    raise TypingError(_MIS_BC_MSG)


_MIS_DISPATCH_SRC = '''
def _mis_nd(xf, y, k, axis, tg, use_t, ol, vl, orr, vr, per):
    """Fit and construct the `make_interp_spline` class matching `y`'s rank.

    `y.ndim` is part of `y`'s numba type, so every branch but one is removed
    while compiling and the survivor fixes the class this returns. A rank past
    the cap leaves only the ``raise``.

    Generated by ``_make_mis_dispatch()``; see `scijit.interpolate._ndaxis`.
    """
{branches}
    raise ValueError("y must have rank {phrase}")
'''


def _make_mis_dispatch():
    """Build `_mis_nd`, one branch per supported rank of `y`."""
    def body(rank):
        return ("        flat, rest, ax = _nda._flatten%d(y, axis)\n"
                "        yf = np.ascontiguousarray(flat.astype(np.float64))\n"
                "        t, c2 = _mis_core_nd(xf, yf, k, tg, use_t,\n"
                "                             ol, vl, orr, vr, per)\n"
                "        return _BSplineND%d(t, c2, k, True, rest, ax, per)"
                % (rank, rank - 1))
    src = _MIS_DISPATCH_SRC.format(
        branches=_dispatch_branches("y", body), phrase=_rank_phrase())
    return njit(_define(src, globals(), "_mis_nd"))


_mis_nd = _make_mis_dispatch()


@njit
def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0,
                       check_finite=True):
    """Interpolating B-spline through ``(x, y)``.

    Parameters
    ----------
    x : 1-D float64 ndarray, length n
        Abscissae, strictly increasing. A non-increasing pair raises
        ``ValueError``. Must satisfy ``n >= k + 1``.
    y : float64 ndarray of rank 1, 2 or 3
        Values to interpolate. Its length along `axis` must be n.
    k : int, optional
        Spline degree, >= 0. Default 3. ``k = 0`` gives the
        piecewise-constant spline and takes no boundary condition.
    t : 1-D float64 ndarray, optional
        Knot vector, ``len(t) == n + k + 1`` when no derivative conditions
        accompany it, and longer by one per condition when they do. Default
        None, which builds the knot set `bc_type` implies. It must be
        non-decreasing and cover ``[x[0], x[-1]]``; otherwise ``ValueError``.
        ``bc_type='periodic'`` builds its own knots and refuses this with
        ``NotImplementedError``. ``k = 0`` refuses it with ``ValueError``.
    bc_type : str or pair of lists, optional
        Boundary condition. Default None, which selects ``'not-a-knot'``.
        The strings are ``'not-a-knot'``, which the default selects and which
        works for any ``k >= 1``; ``'natural'``,
        ``S'' = 0`` at both ends; ``'clamped'``, ``S' = 0`` at both ends; and
        ``'periodic'``, which makes the spline repeat with period
        ``x[-1] - x[0]`` and requires ``y[0] == y[-1]``.

        Otherwise a pair of LISTS of ``(order, value)`` pairs, one list per
        end, fixing ``S^(order)`` to `value` there:
        ``([(1, 0.0)], [(1, 0.0)])`` is ``'clamped'`` and
        ``([(2, 0.0)], [(2, 0.0)])`` is ``'natural'``. This shape is NOT
        `CubicSpline`'s, which takes a pair of PAIRS; each refuses the
        other's shape.

        The conditions must use up exactly the ``k - 1`` coefficients the
        augmented knot vector leaves free, so a cubic takes one per end and a
        quintic two. An unrecognised string raises ``ValueError``, and a shape
        that is neither is refused while compiling.
    axis : int, optional
        Which axis of `y` the interpolation runs along, default 0. Negative
        values count from the end. Ignored for a 1-D `y`. ``ev(xs)`` returns
        the shape of `y` with this axis replaced by ``len(xs)``; a scalar
        query returns it with this axis removed.
    check_finite : bool, optional
        Whether to refuse a non-finite `x` or `y`, default True.
        The check is worth having on: every comparison against NaN is False,
        so a NaN `x` passes the strictly-increasing loop and reaches the
        solve. With it off a NaN reaches ``np.linalg.solve``, which raises
        ``LinAlgError`` of its own.

    Returns
    -------
    spl : BSpline
        The interpolating spline. ``extrapolate=True``, except for
        ``bc_type='periodic'``, where a query is folded into
        ``[x[0], x[-1]]`` first.

    See Also
    --------
    scipy.interpolate.make_interp_spline : The scipy routine this mirrors.

    Notes
    -----
    Deviations from scipy. A derivative `value` is one scalar per end and
    applies to every series of an N-D `y`, where scipy takes an array of the
    trailing shape. `y` is capped at rank 3 and is float64. A `y` of size
    zero raises here, where scipy returns a spline with zero-size
    coefficients. The returned spline goes through the `BSpline` constructor
    and is validated a second time, where scipy builds it with
    ``construct_fast`` and skips that; a knot vector this function produced
    that the constructor rejects therefore raises here and does not there.
    scipy has no weights argument either, so there is none to match.

    ``k = 0`` and ``k = 1`` resolve before the periodic branch, in scipy and
    here, so ``bc_type='periodic'`` at those degrees gives the
    piecewise-constant and piecewise-linear splines with ordinary
    extrapolation. The ends are still required to match.

    Unlike a jitclass constructor, this is a plain ``@njit`` FUNCTION, so its
    defaults work in both worlds -- ``make_interp_spline(x, y)`` is verified
    to compile and run inside ``@njit`` -- and it is where the `bc_type`
    string resolves.

    Accuracy vs ``scipy.interpolate.make_interp_spline`` on 14 random nodes,
    evaluated at 101 points: not-a-knot gives exactly 0.0 in values,
    coefficients and knots for k = 2, 3, 4, 5, and 1.1e-16 in values and
    coefficients for k = 1 (knots exact). ``'natural'`` and ``'clamped'`` with
    k=3 are exactly 0.0 in both values and coefficients.

    On 9 nodes at [0.07, 0.3, 0.55, 0.7, 0.94], values, coefficients and
    knots together: an explicit `t` 0.000e+00; the per-end ``(order, value)``
    pairs 0.000e+00 at k=3 and 3.331e-16 with two conditions per end at k=5;
    ``'periodic'`` 0.000e+00 at k=0 and k=1, and 2.220e-16, 3.331e-16,
    3.331e-16 and 8.882e-16 at k = 2, 3, 4 and 5.

    prange-safe: yes -- pure numpy plus ``np.linalg.solve``, no state.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import make_interp_spline
    >>> x = np.linspace(0.0, 1.0, 11)
    >>> y = np.sin(3.0 * x)
    >>> spl = make_interp_spline(x, y, 3)     # build once
    >>> np.round(spl(np.array([0.25, 0.75])), 6)
    array([0.681629, 0.778063])

    Inside compiled code, scanning the domain for the curve's peak:

    >>> @njit
    ... def peak(spl):
    ...     xs = np.linspace(0.0, 1.0, 201)
    ...     return np.max(spl(xs))
    >>> float(np.round(peak(spl), 6))
    0.999979
    """
    _check_axis(axis)
    _nda._check_axis_range(axis, y.ndim)
    n = x.shape[0]
    # The length has to be measured along the interpolation axis, not along
    # axis 0, or a legitimate N-D call is rejected before it reaches the
    # branch that would resolve it.
    if y.shape[axis % y.ndim] != n:
        raise ValueError("x and y must have the same length")
    if k < 0:
        raise ValueError("Expect non-negative k.")
    for i in range(n - 1):
        if x[i + 1] <= x[i]:
            raise ValueError("Expect x to be a 1-D strictly increasing "
                             "sequence.")
    _check_finite_xy(x, y, check_finite)
    ol, vl, orr, vr = _mis_bc(bc_type)
    _mis_bc_check(ol, orr, k)
    per = _mis_is_periodic(bc_type)
    # scipy resolves k = 0 and k = 1 before it reaches its periodic branch,
    # so those two degrees return the piecewise-constant and piecewise-linear
    # splines with ordinary extrapolation even when 'periodic' was asked for.
    # The end-match check still runs, because scipy runs it first.
    per_fit = per and k >= 2
    if n < k + 1:
        raise ValueError("Not enough points for this degree: need at least "
                         "k+1 of them.")

    xf = x.astype(np.float64)
    if t is None:
        tg = np.zeros(0, np.float64)
        use_t = False
    else:
        tg = np.ascontiguousarray(t.astype(np.float64))
        use_t = True

    if y.ndim == 1:
        yf = np.ascontiguousarray(y.astype(np.float64))
        if per:
            _check_periodic_y(yf.reshape(n, 1))
        tt, c = _mis_core(xf, yf, k, tg, use_t, ol, vl, orr, vr, per_fit)
        return _BSpline(tt, c, k, True, per_fit)
    else:
        if per:
            _check_periodic_y_any(y, axis)
        return _mis_nd(xf, y, k, axis, tg, use_t, ol, vl, orr, vr, per_fit)


# ---------------------------------------------------------------------------
# Akima1DInterpolator
# ---------------------------------------------------------------------------

_AKIMA_METHODS = ("akima", "makima")
_AKIMA_MSG = "`method` must be 'akima' or 'makima'"


def _akima_method_code(method):
    """scipy's ``method`` name resolved to the internal code."""
    if not isinstance(method, str):
        raise NotImplementedError("`method`=%s is unsupported." % (method,))
    for i in range(2):
        if method == _AKIMA_METHODS[i]:
            return i
    raise NotImplementedError("`method`=" + method + " is unsupported.")


@overload(_akima_method_code)
def _akima_method_code_ovl(method):
    """`_akima_method_code` inside ``@njit``.

    The arm keeps its loop at run time, so `method` may be a runtime string as
    well as a literal. Anything that is not a string is refused while
    compiling, which is where the reason can still be reported.
    """
    if not isinstance(method, (types.UnicodeType, types.StringLiteral, str)):
        raise TypingError(_AKIMA_MSG)

    def impl(method):
        for i in range(2):
            if method == _AKIMA_METHODS[i]:
                return i
        raise NotImplementedError("`method`=" + method + " is unsupported.")
    return impl


@njit
def _akima_derivs(x, y, method):
    """Akima node slopes. Internal.

    Follows scipy's ``Akima1DInterpolator`` slope rule, including the two
    phantom slopes extrapolated at each end and scipy's ``1e-9 * max``
    degeneracy threshold, below which the slope falls back to the plain
    average of the neighbouring divided differences.

    With ``delta`` the divided differences and ``d_i`` the slope at node i,
    both methods compute

        d_i = m[i+1] + (f2 / (f1 + f2)) * (m[i+2] - m[i+1])

    and differ only in the weights:

        method 0, 'akima'    f1 = |m[i+3] - m[i+2]|
                             f2 = |m[i+1] - m[i]|
        method 1, 'makima'   the same, each plus half the corresponding
                             |sum| term, which is what removes the overshoot
                             and the 0/0 edge case

    Parameters
    ----------
    x : 1-D float64 ndarray, length n >= 3
        Abscissae, strictly increasing. The n == 2 case is handled by the
        caller.
    y : 1-D float64 ndarray, length n
        Values.
    method : int
        0 for Akima's rule, 1 for the modified rule. Resolved from scipy's
        string by `_akima_method_code` before it reaches here.

    Returns
    -------
    dydx : 1-D float64 ndarray, length n
        Node slopes for the Hermite cubic.
    """
    n = x.shape[0]
    m = np.empty(n + 3, np.float64)
    for i in range(n - 1):
        m[2 + i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    m[1] = 2.0 * m[2] - m[3]
    m[0] = 2.0 * m[1] - m[2]
    m[n + 1] = 2.0 * m[n] - m[n - 1]
    m[n + 2] = 2.0 * m[n + 1] - m[n]

    dm = np.empty(n + 2, np.float64)
    for i in range(n + 2):
        dm[i] = abs(m[i + 1] - m[i])

    f1 = np.empty(n, np.float64)
    f2 = np.empty(n, np.float64)
    if method == 1:
        for i in range(n):
            f1[i] = dm[2 + i] + 0.5 * abs(m[i + 3] + m[i + 2])
            f2[i] = dm[i] + 0.5 * abs(m[i + 1] + m[i])
    else:
        for i in range(n):
            f1[i] = dm[2 + i]
            f2[i] = dm[i]

    f12 = np.empty(n, np.float64)
    fmax = -np.inf
    for i in range(n):
        f12[i] = f1[i] + f2[i]
        if f12[i] > fmax:
            fmax = f12[i]

    thr = 1e-9 * fmax
    dydx = np.empty(n, np.float64)
    for i in range(n):
        if f12[i] > thr:
            # scipy's own grouping, not the algebraically equal
            # (f1*m[i+1] + f2*m[i+2]) / f12, which rounds differently.
            dydx[i] = m[i + 1] + (f2[i] / f12[i]) * (m[i + 2] - m[i + 1])
        else:
            dydx[i] = 0.5 * (m[3 + i] + m[i])
    return dydx


@njit
def _akima_eval(x, c, xs, nu, extrapolate, periodic):
    """Evaluate the Akima PPoly at a batch of points. Internal.

    Parameters
    ----------
    x : 1-D float64 ndarray
        Breakpoints.
    c : (4, n-1) float64 ndarray
        PPoly coefficients in scipy's layout.
    xs : 1-D float64 ndarray
        Points to evaluate at.
    nu : int
        Derivative order; 0..3 are the cubic's derivatives, anything else
        (including negative) gives 0.0.
    extrapolate : bool
        If False, points outside ``[x[0], x[-1]]`` and NaN inputs give NaN.
    periodic : bool
        If True, each point is folded into ``[x[0], x[-1]]`` first.

    Returns
    -------
    out : 1-D float64 ndarray, same length as `xs`
    """
    if nu < 0:
        raise ValueError("Order of derivative cannot be negative")
    xw = _wrap_batch(x, xs, periodic)
    m = xs.shape[0]
    out = np.empty(m, np.float64)
    lo = x[0]
    hi = x[x.shape[0] - 1]
    for j in range(m):
        xq = xw[j]
        if (not extrapolate) and ((xq < lo) or (xq > hi) or (xq != xq)):
            out[j] = np.nan
            continue
        i = _find_seg(x, xq)
        out[j] = _eval_seg(c, i, xq - x[i], nu)
    return out


@njit
def _check_akima(x, y, method):
    """Constructor checks shared by the 1-D and N-D Akima classes.

    `y` is the values in whatever rank the calling class stores, read for its
    length along the interpolation axis and for finiteness.
    """
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y must have the same length")
    if n < 2:
        raise ValueError("at least 2 points are required")
    _check_finite(x, y)
    for i in range(n - 1):
        if x[i + 1] <= x[i]:
            raise ValueError("x must be a strictly increasing sequence")
    if method < 0 or method > 1:
        raise ValueError("`method` must be 'akima' or 'makima'")
    return n


@njit
def _build_c_akima_nd(x, y2, method):
    """Akima coefficients for T independent series, shape ``(4, n-1, T)``.

    The slope rule looks at neighbouring differences within a single series,
    so each column is built on its own. The ``n == 2`` case has no neighbours
    to look at and falls back to the single straight-line slope, as the 1-D
    class does.
    """
    n = x.shape[0]
    t = y2.shape[1]
    c = np.empty((4, n - 1, t), np.float64)
    for j in range(t):
        col = np.ascontiguousarray(y2[:, j])
        if n == 2:
            dydx = np.empty(2, np.float64)
            s = (col[1] - col[0]) / (x[1] - x[0])
            dydx[0] = s
            dydx[1] = s
        else:
            dydx = _akima_derivs(x, col, method)
        cj = _build_c(x, col, dydx)
        for a in range(4):
            for i in range(n - 1):
                c[a, i, j] = cj[a, i]
    return c


@njit
def _akima_eval_nd(x, c3, xs, nu, extrapolate, periodic):
    """Evaluate every series at a batch of points, shape ``(m, T)``.

    Out-of-range handling matches the 1-D routine: with `extrapolate` off, a
    point outside the data range or a NaN gives NaN across every series, and
    with `periodic` on the point is folded into the data range first.
    """
    if nu < 0:
        raise ValueError("Order of derivative cannot be negative")
    xw = _wrap_batch(x, xs, periodic)
    m = xs.shape[0]
    t = c3.shape[2]
    out = np.empty((m, t), np.float64)
    lo = x[0]
    hi = x[x.shape[0] - 1]
    for j in range(m):
        xq = xw[j]
        if (not extrapolate) and ((xq < lo) or (xq > hi) or (xq != xq)):
            for a in range(t):
                out[j, a] = np.nan
            continue
        i = _find_seg(x, xq)
        _eval_seg_nd(c3, i, xq - x[i], nu, out[j])
    return out


@njit
def _akima_point_nd(x, c3, xq, nu, extrapolate, periodic):
    """Evaluate every series at one point, shape ``(T,)``."""
    t = c3.shape[2]
    out = np.empty(t, np.float64)
    xq = _wrap_one(x, xq, periodic)
    if (not extrapolate) and ((xq < x[0]) or (xq > x[x.shape[0] - 1])
                              or (xq != xq)):
        for a in range(t):
            out[a] = np.nan
        return out
    i = _find_seg(x, xq)
    _eval_seg_nd(c3, i, xq - x[i], nu, out)
    return out


_akima_spec = [
    ('x', float64[:]),
    ('c', float64[:, :]),
    ('n', int64),
    ('extrapolate', boolean),
    ('periodic', boolean),
]


@scijitclass(_akima_spec)
class _Akima1DInterpolator:
    """Instance type behind the `Akima1DInterpolator` factory, which
    carries the documentation and the defaults. Every argument is
    explicit here: a jitclass constructor's defaults are Python-only.
    """

    def __init__(self, x, y, extrapolate=False, method=0,
                 periodic=False):
        n = _check_akima(x, y, method)
        xf = x.astype(np.float64)
        yf = y.astype(np.float64)
        if n == 2:
            dydx = np.empty(2, np.float64)
            s = (yf[1] - yf[0]) / (xf[1] - xf[0])
            dydx[0] = s
            dydx[1] = s
        else:
            dydx = _akima_derivs(xf, yf, method)
        self.x = xf
        self.c = _build_c(xf, yf, dydx)
        self.n = n
        self.extrapolate = extrapolate
        self.periodic = periodic

    def ev(self, xs):
        """Evaluate the interpolant at an array of points (scipy's ``f(xs)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at, in any order.

        Returns
        -------
        out : 1-D float64 ndarray, same length as `xs`
            Values; NaN for points outside ``[x[0], x[-1]]`` when
            ``extrapolate`` is False (the default).
        """
        return _akima_eval(self.x, self.c, xs, 0, self.extrapolate,
                           self.periodic)

    def ev_one(self, x):
        """Evaluate the interpolant at a single point.

        Parameters
        ----------
        x : float
            Point to evaluate at.

        Returns
        -------
        float
            The interpolated value, or NaN if `x` is outside
            ``[x[0], x[-1]]`` with ``extrapolate=False``.
        """
        xq = _wrap_one(self.x, x, self.periodic)
        if (not self.extrapolate) and ((xq < self.x[0])
                                       or (xq > self.x[self.n - 1])
                                       or (xq != xq)):
            return np.nan
        i = _find_seg(self.x, xq)
        return _eval_seg(self.c, i, xq - self.x[i], 0)

    def __getitem__(self, xs):
        """``f[xs]`` -- sugar for ``f.ev(xs)``, which ``f(xs)`` also reaches.

        Parameters
        ----------
        xs : 1-D float64 ndarray

        Returns
        -------
        out : 1-D float64 ndarray
        """
        return _akima_eval(self.x, self.c, xs, 0, self.extrapolate,
                           self.periodic)

    def derivative_ev(self, xs, nu=1):
        """Evaluate the `nu`-th derivative (scipy's ``f(xs, nu)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at.
        nu : int, optional
            Derivative order. 0 is the value, 1 the slope, 2 the curvature,
            3 the piecewise-constant third derivative; ``nu >= 4`` returns
            zeros, as a cubic's higher derivatives are. Negative `nu` also
            returns zeros rather than antidifferentiating (scipy rejects
            negative orders). Default 1; this method default works inside
            ``@njit``.

        Returns
        -------
        out : 1-D float64 ndarray, same length as `xs`
            Measured 1.7e-16 against scipy for ``nu=1``.
        """
        return _akima_eval(self.x, self.c, xs, nu, self.extrapolate,
                           self.periodic)

    def get_knots(self):
        """Return the breakpoints (scipy's ``PPoly.x``).

        Returns
        -------
        x : 1-D float64 ndarray, length n
            A reference to the stored array, not a copy.
        """
        return self.x

    def get_coeffs(self):
        """Return the piecewise-polynomial coefficients (scipy's ``PPoly.c``).

        Returns
        -------
        c : (4, n-1) float64 ndarray
            ``c[k, i]`` multiplies ``(x - x[i]) ** (3 - k)`` on
            ``[x[i], x[i+1]]`` -- measured exactly equal to
            ``scipy.interpolate.Akima1DInterpolator.c``. A reference, not a
            copy.
        """
        return self.c


_akima_nd_spec = [
    ('x', float64[:]),
    ('c', float64[:, :, :]),
    ('n', int64),
    ('extrapolate', boolean),
    ('rest', int64[:]),
    ('axis', int64),
    ('periodic', boolean),
]

_AKIMA_ND_SRC = '''
class _Akima1DInterpolatorND{trail}:
    """Instance type behind `Akima1DInterpolator` for a rank-{rank} `y`.

    Holds T independent Akima interpolants over one `x`. `rest` carries the
    shape of the axes other than the interpolation axis, `axis` where that
    axis sat in the caller's `y`.

    Generated by ``_make_akima_nd({rank})`` in
    ``scijit/interpolate/_bspline.py`` and bound there under this name. One
    class exists per supported rank of `y`; they share this source, the spec
    and every kernel, and differ only in the rank they hand back.
    `scijit.interpolate._ndaxis` says why.
    """

    def __init__(self, x, y2, extrapolate, method, rest, axis,
                 periodic):
        n = _check_akima(x, y2, method)
        xf = x.astype(np.float64)
        yf = np.ascontiguousarray(y2.astype(np.float64))
        self.x = xf
        self.c = _build_c_akima_nd(xf, yf, method)
        self.n = n
        self.extrapolate = extrapolate
        self.rest = rest
        self.axis = axis
        self.periodic = periodic

    def ev(self, xs):
        """Evaluate every series at an array of points (scipy's ``f(xs)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at. With ``extrapolate=False`` a point outside
            the data range gives NaN across every series; with `periodic` on
            it is folded into the data range first.

        Returns
        -------
        out : {rank}-D float64 ndarray
            The shape of `y`, with the interpolation axis replaced by
            ``len(xs)``.
        """
        return _nda._restore{rank}(
            _akima_eval_nd(self.x, self.c, xs, 0, self.extrapolate,
                           self.periodic),
            self.rest, self.axis)

    def ev_one(self, x):
        """Evaluate every series at a single point.

        Returns
        -------
        out : {trail}-D float64 ndarray
            The shape of `y` with the interpolation axis removed, or NaN
            throughout if `x` is out of range with ``extrapolate=False``.
        """
        return _nda._point{rank}(
            _akima_point_nd(self.x, self.c, x, 0, self.extrapolate,
                            self.periodic), self.rest)

    def __getitem__(self, xs):
        """``f[xs]`` -- sugar for ``f.ev(xs)``."""
        return _nda._restore{rank}(
            _akima_eval_nd(self.x, self.c, xs, 0, self.extrapolate,
                           self.periodic),
            self.rest, self.axis)

    def derivative_ev(self, xs, nu=1):
        """Evaluate the `nu`-th derivative of every series.

        Returns
        -------
        out : {rank}-D float64 ndarray
            Shaped as `ev`.
        """
        return _nda._restore{rank}(
            _akima_eval_nd(self.x, self.c, xs, nu, self.extrapolate,
                           self.periodic),
            self.rest, self.axis)

    def get_knots(self):
        """Return the breakpoints (scipy's ``PPoly.x``).

        A reference to the stored array, not a copy.
        """
        return self.x

    def get_coeffs(self):
        """Return the coefficients (scipy's ``PPoly.c``).

        Returns
        -------
        c : (4, n-1, {shape}) float64 ndarray
            The 1-D layout with the trailing axes appended, which is scipy's.
        """
{coeffs}
'''


def _make_akima_nd(rank):
    """Build the `Akima1DInterpolator` N-D class for a `y` of this rank.

    Parameters
    ----------
    rank : int
        Rank of the caller's `y`, from 2 to ``1 + _ND_MAXTRAIL``.

    Returns
    -------
    cls : jitclass
        The class, decorated with `scijitclass` against `_akima_nd_spec` and
        bound in this module as ``_Akima1DInterpolatorND<rank-1>``.
    """
    trail = rank - 1
    src = _AKIMA_ND_SRC.format(
        rank=rank, trail=trail, shape=_shape_words(trail),
        coeffs=_coeff_body(trail, "4, self.n - 1"))
    return scijitclass(_akima_nd_spec)(
        _define(src, globals(), "_Akima1DInterpolatorND%d" % trail))


for _rank in _ND_RANKS:
    globals()["_Akima1DInterpolatorND%d" % (_rank - 1)] = _make_akima_nd(_rank)
del _rank


_AKIMA_DISPATCH_SRC = '''
def _akima_nd(x, y, extrapolate, mc, axis, periodic):
    """Construct the `Akima1DInterpolator` class matching the rank of `y`.

    `y.ndim` is part of `y`'s numba type, so every branch but one is removed
    while compiling and the survivor fixes the class this returns. A rank past
    the cap leaves only the ``raise``.

    Generated by ``_make_akima_dispatch()``; see `scijit.interpolate._ndaxis`.
    """
{branches}
    raise ValueError("y must have rank {phrase}")
'''


def _make_akima_dispatch():
    """Build `_akima_nd`, one branch per supported rank of `y`."""
    def body(rank):
        return ("        flat, rest, ax = _nda._flatten%d(y, axis)\n"
                "        return _Akima1DInterpolatorND%d(\n"
                "            x, flat, extrapolate, mc, rest, ax, periodic)"
                % (rank, rank - 1))
    src = _AKIMA_DISPATCH_SRC.format(
        branches=_dispatch_branches("y", body), phrase=_rank_phrase())
    return njit(_define(src, globals(), "_akima_nd"))


_akima_nd = _make_akima_dispatch()


@njit
def Akima1DInterpolator(x, y, axis=0, method="akima",
                        extrapolate=None):
    """Akima piecewise-cubic interpolator.

    Akima's local slope rule damps the overshoot a C2 cubic spline shows on
    step-like data, at the price of only C1 continuity.

    Parameters
    ----------
    x : 1-D float64 ndarray, length n >= 2
        Breakpoints, strictly increasing. A non-increasing pair raises
        ``ValueError``.
    y : float64 ndarray of rank 1, 2 or 3
        Values, real. Its length along `axis` must be n. Rank 4 and above
        raise ``ValueError``.
    axis : int, optional
        Which axis of `y` the interpolation runs along, default 0. Negative
        values count from the end. Ignored for a 1-D `y`. ``ev(xs)`` returns
        the shape of `y` with this axis replaced by ``len(xs)``; a scalar
        query returns it with this axis removed.
    method : str, optional
        The slope rule: ``'akima'`` (the default) for Akima's weights,
        ``'makima'`` for the modified weights of Moler and Ionita, which
        remove the overshoot Akima's rule leaves on step-like data and the
        0/0 edge case. A runtime string works as well as a literal. Anything
        else raises ``NotImplementedError``.
    extrapolate : None, bool or str, optional
        What a query outside ``[x[0], x[-1]]`` gives.

            True           the end polynomial, extended
            False          NaN, and a NaN input also gives NaN
            'periodic'     the query folded into ``[x[0], x[-1]]`` first
            None           the default, which is False

        That default is the OPPOSITE of `BSpline` and `CubicSpline`, where
        extrapolation is on. A string other than ``'periodic'`` raises
        ``ValueError``. 0 and 1 are accepted for False and True. May be a
        runtime value, not only a literal.

    Returns
    -------
    _Akima1DInterpolator
        A callable jitclass instance: ``ak(xs)`` runs `ev` and ``ak(x)`` runs
        `ev_one`. A rank-2 or rank-3 `y` gives `_Akima1DInterpolatorND1` or
        `_Akima1DInterpolatorND2`, which carry the same methods.

    See Also
    --------
    scipy.interpolate.Akima1DInterpolator : The scipy routine this mirrors.

    Notes
    -----
    Deviations from scipy. There is no ``.derivative()`` or
    ``.antiderivative()`` object API, no ``.roots()`` and no ``integrate``.
    A complex `y` raises ``ValueError`` naming ``np.real``, which is scipy's
    own refusal, from Python and from inside ``@njit``. A rank-2 `x` raises
    ``TypingError`` where scipy raises
    ``ValueError('`x` must be 1-dimensional.')``; both refuse and only the
    class differs. The resolved mode is
    split across the `extrapolate` and `periodic` attributes, where scipy
    keeps the caller's value in one; an unrecognised string raises where scipy
    extrapolates. Evaluation is ``ev(xs)`` and ``derivative_ev(xs, nu)``,
    where scipy spells both as ``ak(xs, nu)``; a two-argument call raises on
    arity here. scipy takes `method` as a keyword-only string; here it is a
    positional-or-keyword argument, and it takes the same two strings.
    The n == 2 case falls back to the single linear slope.

    ``Akima1DInterpolator`` is an ``@njit`` factory over the jitclass
    ``_Akima1DInterpolator``, so ``extrapolate`` and `method` may be omitted
    from Python AND from inside ``@njit``.

    Measured against scipy 1.18 on 5 nodes spanning ``[0, 1.4]``, queried at
    2.0 and -1.0: ``extrapolate=None`` gives NaN on both sides,
    ``extrapolate=True`` 2.90115132 and -2.3627451,
    ``extrapolate='periodic'`` 0.74828791 and 0.91284842.

    Accuracy vs ``scipy.interpolate.Akima1DInterpolator``, both methods, over
    211 points on each of five fixtures (15 random nodes, the step data from
    scipy's own docstring, a 21-point sine, all-zero `y`, and n = 3): PPoly
    coefficients EXACTLY 0.0, values 2.2e-15, first derivative 1.1e-14. The
    coefficients agree bit for bit; the residual is in the Horner evaluation.
    With ``extrapolate=True`` evaluated two units past each end, 7.1e-15. With
    ``extrapolate=False`` the outside points are NaN, in the same positions as
    scipy's.

    prange-safe: yes.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import Akima1DInterpolator
    >>> x = np.linspace(0.0, 10.0, 11)
    >>> y = np.array([0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> ak = Akima1DInterpolator(x, y)
    >>> np.round(ak(np.array([2.5, 3.5])), 6)
    array([0.5, 1. ])

    Inside compiled code, the mean of the interpolant over its domain:

    >>> @njit
    ... def mean(ak):
    ...     xs = np.linspace(0.0, 10.0, 201)
    ...     return np.mean(ak(xs))
    >>> float(np.round(mean(ak), 6))
    0.748756

    Attributes
    ----------
    x : 1-D float64 ndarray
        The breakpoints.
    c : (4, n-1) float64 ndarray
        PPoly coefficients: ``c[k, i]`` multiplies ``(x - x[i]) ** (3 - k)``
        on ``[x[i], x[i+1]]``.
    n : int
        Number of breakpoints.
    extrapolate : bool
        Whether a query outside the data range is evaluated at all.
    periodic : bool
        Whether the query is folded into the data range first.

    Methods
    -------
    ev(xs), ev_one(x), __getitem__(xs), derivative_ev(xs, nu),
    get_knots(), get_coeffs()
    """
    _check_axis(axis)
    yr = _akima_real_y(y)
    code = _extrap_code(extrapolate, _EXTRAP_OFF)
    on = code != _EXTRAP_OFF
    per = code == _EXTRAP_PERIODIC
    mc = _akima_method_code(method)
    # The rank test reads the ARGUMENT, not `yr`: numba prunes a branch on an
    # argument's `.ndim` and not on a call result's.
    if y.ndim == 1:
        return _Akima1DInterpolator(x, yr, on, mc, per)
    else:
        return _akima_nd(x, yr, on, mc, axis, per)
