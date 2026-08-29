"""scipy.interpolate.CubicSpline and PchipInterpolator, in numba.

These are numba jitclasses (constructible AND evaluable inside @njit code),
mirroring scipy's piecewise-cubic interpolators. Unlike the FITPACK-backed
spline classes in ``interpolate.py`` (which wrap Fortran), these are pure
written in numba directly -- no .so, no callbacks, no module state -> fully
prange-safe.

Public names, each an ``@njit`` factory over a private jitclass
    CubicSpline(x, y, axis=0, bc_type='not-a-knot', extrapolate=None)
                                 scipy.interpolate.CubicSpline
    PchipInterpolator(x, y, axis=0, extrapolate=None)
                                 scipy.interpolate.PchipInterpolator

Both store the piecewise-polynomial coefficients exactly like scipy's PPoly:
``c`` has shape (4, n-1) and ``c[k, i]`` is the coefficient of
``(x - x[i])**(3-k)`` on the segment ``[x[i], x[i+1]]``. For an N-D ``y`` the
trailing axes are flattened into one and ``c`` becomes (4, n-1, T);
``get_coeffs`` unflattens it to scipy's shape. Evaluation locates
the interval by binary search and applies Horner's rule on the local
polynomial. What happens out of range follows ``extrapolate``: the first/last
segment continued, NaN, or the query folded into ``[x[0], x[-1]]``. Both
classes default to continuing the segment, which is scipy's default for both,
and ``bc_type='periodic'`` makes `CubicSpline` default to folding instead.

Boundary conditions (CubicSpline, ``bc_type``), as scipy's string, each
applied to both ends:
    'not-a-knot'  scipy's default, and the default here
    'natural'     second derivative = 0 at the ends
    'clamped'     first derivative  = 0 at the ends
    'periodic'    the spline repeats with period ``x[-1] - x[0]``

or a pair of ``(order, value)`` pairs, one per end, fixing ``S'`` (order 1)
or ``S''`` (order 2) to a value there. ``y`` may be rank 1, 2 or 3, with
``axis`` naming the interpolation axis.

Methods (matching the FITPACK jitclass naming in this package). The shapes
below are for a 1-D ``y``; for an N-D ``y`` an array query returns the shape
of ``y`` with the interpolation axis replaced by ``len(xs)``, and a scalar
query returns it with that axis removed.
    .ev(xs)              batch evaluation, xs 1-D -> 1-D
    .ev_one(x)           single point -> scalar
    .derivative_ev(xs, nu)   nu-th derivative, batch
    .integral(a, b)      definite integral over [a, b]  (CubicSpline only;
                         mirrors scipy CubicSpline.integrate(a, b))

A jitclass constructor's defaults are Python-only, so both public names here
are ``@njit`` FACTORIES over a private class -- ``CubicSpline`` over
``_CubicSpline``, ``PchipInterpolator`` over ``_PchipInterpolator`` -- and a
plain ``@njit`` function's defaults do apply inside ``@njit``:
``CubicSpline(x, y)`` constructs in both worlds. The factory is also where
``bc_type`` resolves, since a plain function can branch on a string and a
jitclass constructor cannot. Method defaults are fine either way --
``cs.derivative_ev(xs)`` compiles inside ``@njit``.
``cs(xs)`` runs ``.ev`` and ``cs(x)`` runs ``.ev_one``; ``.ev`` / ``.ev_one``
/ ``cs[xs]`` reach the same methods by name.

Examples
--------
>>> import numpy as np
>>> from numba import njit
>>> from scijit.interpolate import CubicSpline
>>> x = np.linspace(0.0, 1.0, 9)
>>> y = np.sin(3.0 * x)
>>> cs = CubicSpline(x, y)     # bc_type='not-a-knot' is the default
>>> np.round(cs(np.array([0.25, 0.75])), 6)
array([0.681639, 0.778073])

Inside compiled code, scanning the domain for the curve's peak:

>>> @njit
... def peak(cs):
...     xs = np.linspace(0.0, 1.0, 201)
...     return np.max(cs(xs))
>>> float(np.round(peak(cs), 6))
0.999968
"""
import numpy as np
from numba import njit, float64, int64, types
from numba.core.errors import TypingError
from numba.extending import overload
from scijitclass import scijitclass

from . import _ndaxis as _nda
from ._ndaxis import (_ND_RANKS, _check_axis, _coeff_body, _define,
                      _dispatch_branches, _rank_phrase,
                      _shape_words)


# ---------------------------------------------------------------------------
# boundary-condition flag: scipy's string
# ---------------------------------------------------------------------------

_BC_NAMES = ("not-a-knot", "natural", "clamped", "periodic")
_BC_MSG = "bc_type must be 'not-a-knot', 'natural' or 'clamped'"
_BC_ANY_MSG = ("bc_type must be 'not-a-knot', 'natural', 'clamped', "
               "'periodic', or a pair of (order, value) pairs")
_BC_PAIR_MSG = ("A specified derivative value must be given in the form "
                "(order, value).")
_BC_ORDER_MSG = "The specified derivative order must be 1 or 2."
_BC_LEN_MSG = ("`bc_type` must contain 2 elements to specify start and end "
               "conditions.")
_BC_PERIODIC_MSG = ("'periodic' `bc_type` is defined for both curve ends and "
                    "cannot be used with other boundary conditions.")

# The resolved form, one row per end, laid out flat as
#   [kind0, order0, value0, kind1, order1, value1]
# kind 0 not-a-knot, 1 a derivative condition of `order` fixed at `value`,
# 2 periodic. 'natural' is kind 1 order 2 value 0 and 'clamped' is kind 1
# order 1 value 0, which is how scipy's own `_validate_bc` reads them, so the
# three string conditions and the (order, value) pairs share one solve.
_BC_KIND_NAK = 0
_BC_KIND_DERIV = 1
_BC_KIND_PERIODIC = 2

# The resolved form for 'not-a-knot', which is the default at every
# entry point. A jitclass constructor's defaults are Python-only, so
# this only takes effect for a Python-side `_CubicSpline(x, y)`.
_BC_NAK_SPEC = np.zeros(6, np.float64)


def _bc_code(bc_type):
    """scipy's `bc_type` string, resolved to the internal int code.

    0 not-a-knot, 1 natural, 2 clamped, 3 periodic. Kept for the callers that
    only accept a string; `_bc_spec` is what `CubicSpline` uses.
    """
    if not isinstance(bc_type, str):
        raise ValueError(_BC_MSG)
    for i in range(4):
        if bc_type == _BC_NAMES[i]:
            return i
    raise ValueError(_BC_MSG)


@overload(_bc_code)
def _bc_code_ovl(bc_type):
    """`_bc_code` inside ``@njit``.

    The string arm keeps its loop over `_BC_NAMES` at run time rather than
    resolving the name during typing, which is what lets `bc_type` be a
    runtime string as well as a literal. Anything that is not a string is
    refused while compiling, so a wrong spelling cannot reach the solve.
    """
    if not isinstance(bc_type, (types.UnicodeType, types.StringLiteral, str)):
        # Refuse while compiling, not by returning something that fails later.
        # An `impl` that only raises has no return type, so the failure would
        # surface downstream as `lt(none, int)` with the reason lost.
        raise TypingError(_BC_MSG)

    def impl(bc_type):
        for i in range(4):
            if bc_type == _BC_NAMES[i]:
                return i
        raise ValueError(_BC_MSG)
    return impl


@njit
def _bc_end_str(name):
    """One end of `bc_type`, resolved from a string to ``[kind, order, value]``.

    The three-entry row is half of the six-entry spec, which is what lets the
    two ends be resolved independently: a string at one end and an
    ``(order, value)`` pair at the other is a legal `bc_type`.
    """
    e = np.zeros(3, np.float64)
    if name == "not-a-knot":
        return e
    if name == "natural":
        e[0] = _BC_KIND_DERIV
        e[1] = 2.0
        return e
    if name == "clamped":
        e[0] = _BC_KIND_DERIV
        e[1] = 1.0
        return e
    if name == "periodic":
        e[0] = _BC_KIND_PERIODIC
        return e
    raise ValueError("bc_type=" + name + " is not allowed.")


@njit
def _bc_end_pair(order, value):
    """One end of `bc_type`, resolved from an ``(order, value)`` pair."""
    if order != 1 and order != 2:
        raise ValueError(_BC_ORDER_MSG)
    e = np.zeros(3, np.float64)
    e[0] = _BC_KIND_DERIV
    e[1] = order
    e[2] = value
    return e


@njit
def _bc_join(e0, e1):
    """Two resolved ends into the six-entry spec.

    ``'periodic'`` names a condition on the pair rather than on one end, so it
    is legal only as a single string covering both. Either end carrying it here
    means it arrived in the two-element form.
    """
    if e0[0] == _BC_KIND_PERIODIC or e1[0] == _BC_KIND_PERIODIC:
        raise ValueError(_BC_PERIODIC_MSG)
    s = np.zeros(6, np.float64)
    s[0] = e0[0]
    s[1] = e0[1]
    s[2] = e0[2]
    s[3] = e1[0]
    s[4] = e1[1]
    s[5] = e1[2]
    return s


@njit
def _bc_both(e):
    """One resolved end applied to both, which is what a single string means."""
    s = np.zeros(6, np.float64)
    s[0] = e[0]
    s[1] = e[1]
    s[2] = e[2]
    s[3] = e[0]
    s[4] = e[1]
    s[5] = e[2]
    return s


def _bc_spec(bc_type):
    """Resolve `CubicSpline`'s `bc_type` to the six-entry internal form.

    A single string applies one condition to both ends. A two-element sequence
    gives the ends separately, and each element is independently either a
    string or an ``(order, value)`` pair, so ``('natural', 'clamped')``,
    ``('not-a-knot', (1, 3.0))`` and ``((2, 0.0), 'clamped')`` all resolve.

    scipy gives `make_interp_spline` a different shape for the same idea, a
    pair of LISTS of pairs; that shape does not resolve here.
    """
    if isinstance(bc_type, str):
        return _bc_both(_bc_end_str(bc_type))
    try:
        pair = list(bc_type)
    except Exception:
        raise ValueError(_BC_LEN_MSG)
    if len(pair) != 2:
        raise ValueError(_BC_LEN_MSG)
    return _bc_join(_bc_end_py(pair[0]), _bc_end_py(pair[1]))


def _bc_end_py(bc):
    """One end of a two-element `bc_type`, from Python."""
    if isinstance(bc, str):
        return _bc_end_str(bc)
    try:
        order, value = bc
    except Exception:
        raise ValueError(_BC_PAIR_MSG)
    return _bc_end_pair(int(order), float(value))


@overload(_bc_spec)
def _bc_spec_ovl(bc_type):
    """`_bc_spec` inside ``@njit``, choosing the arm while compiling.

    The shape of `bc_type` is part of its numba type, so a string end and a
    pair end are separate compiled bodies and neither has to test what it was
    given. That is why the four combinations of the two-element form are
    spelled out rather than resolved at run time.
    """
    def _is_str(t):
        return isinstance(t, (types.UnicodeType, types.StringLiteral, str))

    def _is_pair(t):
        return isinstance(t, types.BaseTuple) and len(t) == 2

    if _is_str(bc_type):
        def impl(bc_type):
            return _bc_both(_bc_end_str(bc_type))
        return impl
    if isinstance(bc_type, types.BaseTuple) and len(bc_type) == 2:
        left, right = bc_type[0], bc_type[1]
        if _is_str(left) and _is_str(right):
            def impl(bc_type):
                return _bc_join(_bc_end_str(bc_type[0]),
                                _bc_end_str(bc_type[1]))
            return impl
        if _is_str(left) and _is_pair(right):
            def impl(bc_type):
                return _bc_join(_bc_end_str(bc_type[0]),
                                _bc_end_pair(bc_type[1][0],
                                             np.float64(bc_type[1][1])))
            return impl
        if _is_pair(left) and _is_str(right):
            def impl(bc_type):
                return _bc_join(_bc_end_pair(bc_type[0][0],
                                             np.float64(bc_type[0][1])),
                                _bc_end_str(bc_type[1]))
            return impl
        if _is_pair(left) and _is_pair(right):
            def impl(bc_type):
                return _bc_join(_bc_end_pair(bc_type[0][0],
                                             np.float64(bc_type[0][1])),
                                _bc_end_pair(bc_type[1][0],
                                             np.float64(bc_type[1][1])))
            return impl
    raise TypingError(_BC_ANY_MSG)


@njit
def _bc_summary(spec):
    """The int a class stores as ``bc_type``, derived from the resolved form.

    0 not-a-knot, 1 natural, 2 clamped, 3 periodic, 4 any other derivative
    pair. Only 3 changes what evaluation does, since a periodic spline wraps
    the query point; the rest is a record of how the spline was built.
    """
    if spec[0] == _BC_KIND_PERIODIC:
        return 3
    if spec[0] == _BC_KIND_NAK and spec[3] == _BC_KIND_NAK:
        return 0
    if (spec[0] == _BC_KIND_DERIV and spec[3] == _BC_KIND_DERIV
            and spec[2] == 0.0 and spec[5] == 0.0 and spec[1] == spec[4]):
        return 1 if spec[1] == 2.0 else 2
    return 4


# ---------------------------------------------------------------------------
# `extrapolate`, resolved to an int code
# ---------------------------------------------------------------------------
#
# scipy spells three behaviours through one parameter: a bool, the string
# 'periodic', and `None` meaning "the class default". The default is NOT the
# same for every class, so the resolved code is passed in rather than assumed:
# `CubicSpline` and `PchipInterpolator` default to extrapolating,
# `Akima1DInterpolator` to NaN outside the data, and `BSpline` declares
# `extrapolate=True` outright. Measured against scipy 1.18 at x = 2.0 on data
# spanning [0, 1.4]: CubicSpline None -> 13.120, Pchip None -> 7.077,
# Akima None -> nan.

_EXTRAP_OFF = 0
_EXTRAP_ON = 1
_EXTRAP_PERIODIC = 2

_EXTRAP_MSG = ("extrapolate must be None, a bool, 0/1, or the string "
               "'periodic'")


def _extrap_absent(v):
    """Was `extrapolate` omitted, or passed as ``None``?

    `v` is a plain Python value on the interpreter path and a numba type
    inside an ``@overload`` body, so all three spellings are tested.
    Internal.
    """
    return (v is None or isinstance(v, types.NoneType)
            or (isinstance(v, types.Omitted) and v.value is None))


def _extrap_code(extrapolate, dflt):
    """scipy's `extrapolate`, resolved to 0 (off), 1 (on) or 2 (periodic).

    Parameters
    ----------
    extrapolate : None, bool, int or str
        The caller's value. ``None`` selects `dflt`; ``'periodic'`` gives 2;
        anything else is read for truth. An unrecognised string raises.
    dflt : int
        The code `None` resolves to for the calling class.

    Returns
    -------
    int
        0, 1 or 2.
    """
    if extrapolate is None:
        return dflt
    if isinstance(extrapolate, str):
        if extrapolate == "periodic":
            return _EXTRAP_PERIODIC
        raise ValueError(_EXTRAP_MSG)
    return _EXTRAP_ON if extrapolate else _EXTRAP_OFF


@overload(_extrap_code)
def _extrap_code_ovl(extrapolate, dflt):
    """`_extrap_code` inside ``@njit``.

    The string arm keeps its comparison at run time, so `extrapolate` may be
    a runtime string as well as a literal, and an unrecognised one raises
    ``ValueError`` from here exactly as it does from Python.
    """
    if _extrap_absent(extrapolate):
        def impl(extrapolate, dflt):
            return dflt
        return impl

    if isinstance(extrapolate, (types.UnicodeType, types.StringLiteral, str)):
        def impl(extrapolate, dflt):
            if extrapolate == "periodic":
                return _EXTRAP_PERIODIC
            raise ValueError(_EXTRAP_MSG)
        return impl

    def impl(extrapolate, dflt):
        return _EXTRAP_ON if extrapolate else _EXTRAP_OFF
    return impl


# ---------------------------------------------------------------------------
# a complex `y`
# ---------------------------------------------------------------------------
#
# scipy refuses a complex `y` on the two shape-preserving interpolators and
# names `np.real` in the message. Reproduced here with the same class and the
# same text. The refusal is decided from `y`'s dtype while compiling, so no
# value is inspected at run time.

_PCHIP_COMPLEX_MSG = (
    "`PchipInterpolator` only works with real values for `y`. If you are "
    "trying to use the real components of the passed array, use `np.real` "
    "on the array before passing to `PchipInterpolator`.")

_AKIMA_COMPLEX_MSG = (
    "`Akima1DInterpolator` only works with real values for `y`. If you are "
    "trying to use the real components of the passed array, use `np.real` "
    "on the array before passing to `Akima1DInterpolator`.")


def _make_real_y(name, msg):
    """Build the guard that refuses a complex `y`, carrying `msg`.

    Parameters
    ----------
    name : str
        Name to bind the guard under, for tracebacks.
    msg : str
        The ``ValueError`` text, which is scipy's own.

    Returns
    -------
    callable
        ``real_y(y)``, which returns `y` unchanged for a real dtype and
        raises ``ValueError(msg)`` for a complex one, from Python and from
        inside ``@njit``.
    """
    def real_y(y):
        if np.iscomplexobj(y):
            raise ValueError(msg)
        return y
    real_y.__name__ = name
    real_y.__qualname__ = name
    real_y.__doc__ = "Refuse a complex `y`. Internal; see `_make_real_y`."

    def real_y_ovl(y):
        """`real_y` inside ``@njit``, one body per dtype class."""
        if isinstance(y, types.Array) and isinstance(y.dtype, types.Complex):
            def impl(y):
                # The guard always fires. The unreached `return` is what
                # gives this arm a REAL return type, so the caller's own
                # `astype(np.float64)` types instead of failing first and
                # burying the message under a TypingError.
                if y.size >= 0:
                    raise ValueError(msg)
                return np.real(y).astype(np.float64)
            return impl

        def impl(y):
            return y
        return impl

    overload(real_y)(real_y_ovl)
    return real_y


_pchip_real_y = _make_real_y("_pchip_real_y", _PCHIP_COMPLEX_MSG)
_akima_real_y = _make_real_y("_akima_real_y", _AKIMA_COMPLEX_MSG)


# ---------------------------------------------------------------------------
# shared low-level njit helpers (used by both jitclasses)
# ---------------------------------------------------------------------------

_NEG_NU_MSG = "Order of derivative cannot be negative"


@njit
def _sign(v):
    """Sign of `v` as a float: 1.0, -1.0, or 0.0 when `v` is zero.

    The Fritsch-Carlson slope limiter compares signs to decide whether a
    secant crosses, so what matters is that the three cases are distinct
    and that the zero case is its own value rather than folded into one of
    the others. NaN falls through to 0.0, which keeps the comparisons in
    `_pchip_slopes` total instead of leaving a branch unhandled.
    """
    if v > 0.0:
        return 1.0
    elif v < 0.0:
        return -1.0
    return 0.0


@njit
def _tridiag_solve(a, b, c, d):
    """Solve a tridiagonal system (sub a, diag b, super c, rhs d).

    Assembled as a dense system and solved with a partially-pivoted LU
    (np.linalg.solve -> LAPACK dgesv), matching scipy's pivoted banded
    solver closely. The not-a-knot boundary rows are not diagonally
    dominant, so pivoting (which a bare Thomas sweep lacks) matters for
    bit-close agreement with scipy under far extrapolation.
    """
    n = b.shape[0]
    A = np.zeros((n, n), np.float64)
    for i in range(n):
        A[i, i] = b[i]
        if i + 1 < n:
            A[i, i + 1] = c[i]
        if i - 1 >= 0:
            A[i, i - 1] = a[i]
    return np.linalg.solve(A, d)


@njit
def _solve_periodic(dx, slope, n):
    """Node first-derivatives for a periodic C2 cubic spline.

    Periodicity makes the system cyclic rather than tridiagonal: ``s[n-1]`` is
    ``s[0]``, so there are ``n-1`` unknowns and the matrix gains a corner on
    each side. scipy condenses it to an ``(n-2, n-2)`` tridiagonal system
    solved twice, once for the data and once for the coupling column, then
    recombines. This follows that construction step for step.

    Parameters
    ----------
    dx, slope : 1-D float64 ndarray, length n-1
        Interval widths and secant slopes.
    n : int
        Number of breakpoints, at least 4. The n = 2 and n = 3 cases are
        handled by the caller, as they are in scipy.
    """
    ab0 = np.zeros(n, np.float64)          # super-diagonal
    ab1 = np.zeros(n, np.float64)          # diagonal
    ab2 = np.zeros(n, np.float64)          # sub-diagonal
    b = np.zeros(n, np.float64)
    for i in range(1, n - 1):
        ab1[i] = 2.0 * (dx[i - 1] + dx[i])
        b[i] = 3.0 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i])
    for j in range(2, n):
        ab0[j] = dx[j - 2]
    for j in range(n - 2):
        ab2[j] = dx[j + 1]

    # The first row closes the loop: it couples s[0] to s[n-2] through the
    # last interval, which is what stops the system being tridiagonal.
    ab1[0] = 2.0 * (dx[n - 2] + dx[0])
    ab0[1] = dx[n - 2]
    b[0] = 3.0 * (dx[0] * slope[n - 2] + dx[n - 2] * slope[0])
    b[n - 2] = 3.0 * (dx[n - 2] * slope[n - 3] + dx[n - 3] * slope[n - 2])

    # The four entries dropped by condensing, named for the position they held
    # in the (n-1, n-1) system.
    a_m1_0 = dx[n - 3]
    a_m1_m2 = dx[n - 2]
    a_m1_m1 = 2.0 * (dx[n - 2] + dx[n - 3])
    a_m2_m1 = dx[n - 4]
    a_0_m1 = dx[0]

    m = n - 2
    ac = np.zeros((m, m), np.float64)
    for i in range(m):
        ac[i, i] = ab1[i]
        if i >= 1:
            ac[i - 1, i] = ab0[i]
        if i <= m - 2:
            ac[i + 1, i] = ab2[i]
    b1 = np.empty(m, np.float64)
    b2 = np.zeros(m, np.float64)
    for i in range(m):
        b1[i] = b[i]
    b2[0] = -a_0_m1
    b2[m - 1] = -a_m2_m1
    s1 = np.linalg.solve(ac, b1)
    s2 = np.linalg.solve(ac, b2)

    s_m1 = ((b[m] - a_m1_0 * s1[0] - a_m1_m2 * s1[m - 1])
            / (a_m1_m1 + a_m1_0 * s2[0] + a_m1_m2 * s2[m - 1]))
    s = np.empty(n, np.float64)
    for i in range(m):
        s[i] = s1[i] + s_m1 * s2[i]
    s[n - 2] = s_m1
    s[n - 1] = s[0]
    return s


@njit
def _solve_derivs(x, y, spec):
    """Node first-derivatives for a C2 cubic spline (scipy CubicSpline solve).

    Parameters
    ----------
    x, y : 1-D float64 ndarray, length n
        Breakpoints and values.
    spec : 1-D float64 ndarray, length 6
        The resolved boundary conditions, ``[kind0, order0, value0, kind1,
        order1, value1]``. See this module's `_bc_spec`.

    Notes
    -----
    Reproduces scipy's banded fill in ``scipy.interpolate._cubic``, including
    the value carried by a derivative condition: order 1 sets ``S'`` at the
    end and order 2 sets ``S''``, and the sign of the order-2 right-hand side
    differs between the two ends, as it does there.
    """
    n = x.shape[0]
    k0 = int(spec[0])
    o0 = int(spec[1])
    v0 = spec[2]
    k1 = int(spec[3])
    o1 = int(spec[4])
    v1 = spec[5]
    dx = np.empty(n - 1, np.float64)
    slope = np.empty(n - 1, np.float64)
    for i in range(n - 1):
        dx[i] = x[i + 1] - x[i]
        slope[i] = (y[i + 1] - y[i]) / dx[i]

    # n == 2: not-a-knot and periodic both reduce to pinning S' to the single
    # secant slope at both ends, which is scipy's own reduction.
    if n == 2:
        if k0 != _BC_KIND_DERIV:
            k0 = _BC_KIND_DERIV
            o0 = 1
            v0 = slope[0]
        if k1 != _BC_KIND_DERIV:
            k1 = _BC_KIND_DERIV
            o1 = 1
            v1 = slope[0]
    elif k0 == _BC_KIND_PERIODIC:
        if n == 3:
            # Three points leave no freedom: the derivative is the same at
            # every node and scipy computes it directly.
            t = ((slope[0] / dx[0] + slope[1] / dx[1])
                 / (1.0 / dx[0] + 1.0 / dx[1]))
            s = np.empty(3, np.float64)
            for i in range(3):
                s[i] = t
            return s
        return _solve_periodic(dx, slope, n)

    # Special case: both ends not-a-knot with n == 3 -> parabola (dense 3x3),
    # exactly as scipy (the two not-a-knot conditions would otherwise
    # coincide).
    if n == 3 and k0 == _BC_KIND_NAK and k1 == _BC_KIND_NAK:
        A = np.zeros((3, 3), np.float64)
        b3 = np.empty(3, np.float64)
        A[0, 0] = 1.0
        A[0, 1] = 1.0
        A[1, 0] = dx[1]
        A[1, 1] = 2.0 * (dx[0] + dx[1])
        A[1, 2] = dx[0]
        A[2, 1] = 1.0
        A[2, 2] = 1.0
        b3[0] = 2.0 * slope[0]
        b3[1] = 3.0 * (dx[0] * slope[1] + dx[1] * slope[0])
        b3[2] = 2.0 * slope[1]
        return np.linalg.solve(A, b3)

    sub = np.zeros(n, np.float64)
    diag = np.zeros(n, np.float64)
    sup = np.zeros(n, np.float64)
    b = np.zeros(n, np.float64)

    # interior rows i = 1 .. n-2
    for i in range(1, n - 1):
        diag[i] = 2.0 * (dx[i - 1] + dx[i])
        sup[i] = dx[i - 1]
        sub[i] = dx[i]
        b[i] = 3.0 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i])

    # start condition (row 0)
    if k0 == _BC_KIND_NAK:    # not-a-knot
        d = x[2] - x[0]
        diag[0] = dx[1]
        sup[0] = d
        b[0] = ((dx[0] + 2.0 * d) * dx[1] * slope[0]
                + dx[0] ** 2 * slope[1]) / d
    elif o0 == 1:             # S'(x0) = v0
        diag[0] = 1.0
        sup[0] = 0.0
        b[0] = v0
    else:                     # S''(x0) = v0
        diag[0] = 2.0 * dx[0]
        sup[0] = dx[0]
        b[0] = -0.5 * v0 * dx[0] ** 2 + 3.0 * (y[1] - y[0])

    # end condition (row n-1)
    if k1 == _BC_KIND_NAK:    # not-a-knot
        d = x[n - 1] - x[n - 3]
        diag[n - 1] = dx[n - 3]
        sub[n - 1] = d
        b[n - 1] = (dx[n - 2] ** 2 * slope[n - 3]
                    + (2.0 * d + dx[n - 2]) * dx[n - 3] * slope[n - 2]) / d
    elif o1 == 1:             # S'(x_{n-1}) = v1
        diag[n - 1] = 1.0
        sub[n - 1] = 0.0
        b[n - 1] = v1
    else:                     # S''(x_{n-1}) = v1
        diag[n - 1] = 2.0 * dx[n - 2]
        sub[n - 1] = dx[n - 2]
        b[n - 1] = 0.5 * v1 * dx[n - 2] ** 2 + 3.0 * (y[n - 1] - y[n - 2])

    return _tridiag_solve(sub, diag, sup, b)


@njit
def _edge_case(h0, h1, m0, m1):
    """PCHIP one-sided three-point endpoint derivative (scipy _edge_case)."""
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if _sign(d) != _sign(m0):
        d = 0.0
    elif _sign(m0) != _sign(m1) and abs(d) > 3.0 * abs(m0):
        d = 3.0 * m0
    return d


@njit
def _pchip_derivs(x, y):
    """Fritsch-Carlson monotone slopes.

    scipy's ``PchipInterpolator._find_derivatives``.
    """
    n = x.shape[0]
    dk = np.empty(n, np.float64)
    if n == 2:
        m = (y[1] - y[0]) / (x[1] - x[0])
        dk[0] = m
        dk[1] = m
        return dk

    hk = np.empty(n - 1, np.float64)
    mk = np.empty(n - 1, np.float64)
    for i in range(n - 1):
        hk[i] = x[i + 1] - x[i]
        mk[i] = (y[i + 1] - y[i]) / hk[i]

    for k in range(1, n - 1):
        m0 = mk[k - 1]
        m1 = mk[k]
        if _sign(m0) != _sign(m1) or m0 == 0.0 or m1 == 0.0:
            dk[k] = 0.0
        else:
            w1 = 2.0 * hk[k] + hk[k - 1]
            w2 = hk[k] + 2.0 * hk[k - 1]
            whmean = (w1 / m0 + w2 / m1) / (w1 + w2)
            dk[k] = 1.0 / whmean

    dk[0] = _edge_case(hk[0], hk[1], mk[0], mk[1])
    dk[n - 1] = _edge_case(hk[n - 2], hk[n - 3], mk[n - 2], mk[n - 3])
    return dk


@njit
def _build_c(x, y, dydx):
    """CubicHermiteSpline PPoly coefficients from node values + derivatives."""
    n = x.shape[0]
    c = np.empty((4, n - 1), np.float64)
    for i in range(n - 1):
        dx = x[i + 1] - x[i]
        slope = (y[i + 1] - y[i]) / dx
        t = (dydx[i] + dydx[i + 1] - 2.0 * slope) / dx
        c[0, i] = t / dx
        c[1, i] = (slope - dydx[i]) / dx - t
        c[2, i] = dydx[i]
        c[3, i] = y[i]
    return c


@njit
def _cumulative(x, c):
    """Antiderivative value at each breakpoint, F(x[0]) = 0."""
    n = x.shape[0]
    cum = np.empty(n, np.float64)
    cum[0] = 0.0
    for i in range(n - 1):
        s = x[i + 1] - x[i]
        seg = (((c[0, i] / 4.0 * s + c[1, i] / 3.0) * s
                + c[2, i] / 2.0) * s + c[3, i]) * s
        cum[i + 1] = cum[i] + seg
    return cum


@njit
def _find_interval(x, xq):
    """Index i with x[i] <= xq < x[i+1], clamped to [0, n-2].

    The clamping is what makes an out-of-range query extrapolate.
    """
    n = x.shape[0]
    if xq <= x[0]:
        return 0
    if xq >= x[n - 1]:
        return n - 2
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= xq:
            lo = mid
        else:
            hi = mid
    return lo


@njit
def _eval_seg(c, i, s, nu):
    """nu-th derivative of segment i's cubic at local coordinate s."""
    if nu == 0:
        return ((c[0, i] * s + c[1, i]) * s + c[2, i]) * s + c[3, i]
    elif nu == 1:
        return (3.0 * c[0, i] * s + 2.0 * c[1, i]) * s + c[2, i]
    elif nu == 2:
        return 6.0 * c[0, i] * s + 2.0 * c[1, i]
    elif nu == 3:
        return 6.0 * c[0, i]
    return 0.0


@njit
def _evaluate(x, c, xs, nu):
    """Evaluate the piecewise polynomial `c` over `x` at every point of `xs`.

    Both jitclasses in this module carry the same ``(x, c)`` PPoly layout and
    the same array-evaluation loop. It lives here as a free function rather
    than a method because a jitclass cannot inherit, so the alternative is
    two copies that drift.
    """
    if nu < 0:
        raise ValueError(_NEG_NU_MSG)
    m = xs.shape[0]
    out = np.empty(m, np.float64)
    for j in range(m):
        i = _find_interval(x, xs[j])
        out[j] = _eval_seg(c, i, xs[j] - x[i], nu)
    return out


@njit
def _antideriv_at(x, c, cum, xq):
    """Antiderivative of the spline from ``x[0]`` to `xq`.

    `cum` holds the integral over every whole segment, prefix-summed once at
    construction, so a definite integral costs one interval search and one
    partial segment at each end instead of a sweep. `.integral` is the
    difference of two of these.
    """
    i = _find_interval(x, xq)
    s = xq - x[i]
    seg = (((c[0, i] / 4.0 * s + c[1, i] / 3.0) * s
            + c[2, i] / 2.0) * s + c[3, i]) * s
    return cum[i] + seg


# ---------------------------------------------------------------------------
# N-D `y`: the same arithmetic, carried over T independent series at once
# ---------------------------------------------------------------------------
#
# An N-D `y` is stored with the interpolation axis first and the remaining
# axes flattened into one, shape (n, T). The coefficient array grows a
# trailing T axis and the routines below walk it. Nothing about the cubic
# changes: column j of the storage is the same spline the 1-D routines above
# would build from that column alone, which is what the tests assert.


@njit
def _build_c_nd(x, y2, spec):
    """PPoly coefficients for T independent series, shape ``(4, n-1, T)``.

    One solve per column. The columns share `x`, so they share the interval
    structure, but each has its own boundary conditions to satisfy and its own
    tridiagonal solve.
    """
    n = x.shape[0]
    t = y2.shape[1]
    c = np.empty((4, n - 1, t), np.float64)
    for j in range(t):
        col = np.ascontiguousarray(y2[:, j])
        dydx = _solve_derivs(x, col, spec)
        cj = _build_c(x, col, dydx)
        for k in range(4):
            for i in range(n - 1):
                c[k, i, j] = cj[k, i]
    return c


@njit
def _check_pchip(x, y):
    """Constructor checks shared by the 1-D and N-D pchip classes.

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
    return n


@njit
def _build_c_pchip_nd(x, y2):
    """Shape-preserving coefficients for T independent series, ``(4, n-1, T)``.

    The Fritsch-Carlson slope rule, which chooses derivatives that keep each
    segment monotone where the data is, is applied to each column on its own.
    """
    n = x.shape[0]
    t = y2.shape[1]
    c = np.empty((4, n - 1, t), np.float64)
    for j in range(t):
        col = np.ascontiguousarray(y2[:, j])
        cj = _build_c(x, col, _pchip_derivs(x, col))
        for k in range(4):
            for i in range(n - 1):
                c[k, i, j] = cj[k, i]
    return c


@njit
def _cumulative_nd(x, c3):
    """Antiderivative at each breakpoint for every series, shape ``(n, T)``."""
    n = x.shape[0]
    t = c3.shape[2]
    cum = np.zeros((n, t), np.float64)
    for j in range(t):
        for i in range(n - 1):
            s = x[i + 1] - x[i]
            seg = (((c3[0, i, j] / 4.0 * s + c3[1, i, j] / 3.0) * s
                    + c3[2, i, j] / 2.0) * s + c3[3, i, j]) * s
            cum[i + 1, j] = cum[i, j] + seg
    return cum


@njit
def _eval_seg_nd(c3, i, s, nu, out):
    """Write the `nu`-th derivative of segment `i` at `s`, for every series."""
    t = c3.shape[2]
    for j in range(t):
        if nu == 0:
            out[j] = (((c3[0, i, j] * s + c3[1, i, j]) * s
                       + c3[2, i, j]) * s + c3[3, i, j])
        elif nu == 1:
            out[j] = (3.0 * c3[0, i, j] * s + 2.0 * c3[1, i, j]) * s \
                + c3[2, i, j]
        elif nu == 2:
            out[j] = 6.0 * c3[0, i, j] * s + 2.0 * c3[1, i, j]
        elif nu == 3:
            out[j] = 6.0 * c3[0, i, j]
        else:
            out[j] = 0.0


@njit
def _evaluate_nd(x, c3, xs, nu):
    """Evaluate every series at every point of `xs`, shape ``(m, T)``.

    The interval search runs once per query point and is shared across the
    series, which is the reason the trailing axes are stored together.
    """
    if nu < 0:
        raise ValueError(_NEG_NU_MSG)
    m = xs.shape[0]
    out = np.empty((m, c3.shape[2]), np.float64)
    for q in range(m):
        i = _find_interval(x, xs[q])
        _eval_seg_nd(c3, i, xs[q] - x[i], nu, out[q])
    return out


@njit
def _point_nd(x, c3, xq, nu):
    """Evaluate every series at one point, shape ``(T,)``."""
    out = np.empty(c3.shape[2], np.float64)
    i = _find_interval(x, xq)
    _eval_seg_nd(c3, i, xq - x[i], nu, out)
    return out


@njit
def _antideriv_at_nd(x, c3, cum, xq):
    """Antiderivative from ``x[0]`` to `xq`, every series, shape ``(T,)``."""
    t = c3.shape[2]
    out = np.empty(t, np.float64)
    i = _find_interval(x, xq)
    s = xq - x[i]
    for j in range(t):
        seg = (((c3[0, i, j] / 4.0 * s + c3[1, i, j] / 3.0) * s
                + c3[2, i, j] / 2.0) * s + c3[3, i, j]) * s
        out[j] = cum[i, j] + seg
    return out


_PERIODIC_MSG = ("The first and last `y` point along the interpolation axis "
                 "must be identical (within machine precision) when "
                 "bc_type='periodic'.")


@njit
def _check_periodic_ends(y2):
    """Refuse a periodic fit whose ends disagree, as scipy does.

    `y2` is the stored ``(n, T)`` layout, so every series is checked. scipy
    compares with ``rtol=1e-15, atol=1e-15``, and this uses the same bound.
    """
    n = y2.shape[0]
    for j in range(y2.shape[1]):
        a = y2[0, j]
        b = y2[n - 1, j]
        if abs(a - b) > 1e-15 + 1e-15 * abs(b):
            raise ValueError(_PERIODIC_MSG)


@njit
def _check_finite(x, y):
    """Refuse a non-finite `x` or `y`.

    A ``nan`` in `x` also defeats the strictly-increasing test, since
    ``nan <= nan`` is False, so this runs before it.
    """
    if not np.all(np.isfinite(x)):
        raise ValueError("`x` must contain only finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("`y` must contain only finite values.")


@njit
def _check_cubic(x, y, bc_type):
    """Constructor checks shared by the 1-D and N-D cubic classes.

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
    if bc_type < 0 or bc_type > 4:
        raise ValueError("bc_type must be 0 (not-a-knot), 1 (natural), "
                         "2 (clamped), 3 (periodic) or 4 (a derivative pair)")
    return n


@njit
def _wrap_batch(x, xs, periodic):
    """Fold query points into the base period, for a periodic spline.

    scipy sets ``extrapolate='periodic'`` whenever ``bc_type='periodic'``, and
    a periodic PPoly maps a query with ``x0 + (x - x0) % (xn - x0)`` before
    evaluating. Returns `xs` untouched when `periodic` is False.
    """
    if not periodic:
        return xs
    n = x.shape[0]
    x0 = x[0]
    period = x[n - 1] - x0
    out = np.empty(xs.shape[0], np.float64)
    for i in range(xs.shape[0]):
        out[i] = x0 + (xs[i] - x0) % period
    return out


@njit
def _wrap_one(x, xq, periodic):
    """`_wrap_batch` for a single query point."""
    if not periodic:
        return xq
    n = x.shape[0]
    x0 = x[0]
    return x0 + (xq - x0) % (x[n - 1] - x0)


@njit
def _periodic_integral(x, c, cum, a, b):
    """Definite integral of a periodic spline, whole periods included.

    scipy's ``PPoly.integrate`` with ``extrapolate='periodic'`` counts the
    complete periods spanned by ``[a, b]`` and adds the remainder, so a range
    wider than the data is not extrapolated with the end segment. This follows
    that construction.
    """
    n = x.shape[0]
    xs = x[0]
    xe = x[n - 1]
    period = xe - xs
    interval = b - a
    n_periods = np.floor(interval / period)
    left = interval - n_periods * period
    total = 0.0
    if n_periods != 0.0:
        total = n_periods * (_antideriv_at(x, c, cum, xe)
                             - _antideriv_at(x, c, cum, xs))
    aa = xs + (a - xs) % period
    bb = aa + left
    if bb <= xe:
        total += _antideriv_at(x, c, cum, bb) - _antideriv_at(x, c, cum, aa)
    else:
        total += _antideriv_at(x, c, cum, xe) - _antideriv_at(x, c, cum, aa)
        total += (_antideriv_at(x, c, cum, xs + left + aa - xe)
                  - _antideriv_at(x, c, cum, xs))
    return total


@njit
def _periodic_integral_nd(x, c, cum, a, b):
    """`_periodic_integral` for the stored ``(4, n-1, T)`` layout."""
    n = x.shape[0]
    xs = x[0]
    xe = x[n - 1]
    period = xe - xs
    interval = b - a
    n_periods = np.floor(interval / period)
    left = interval - n_periods * period
    total = np.zeros(c.shape[2], np.float64)
    if n_periods != 0.0:
        total = n_periods * (_antideriv_at_nd(x, c, cum, xe)
                             - _antideriv_at_nd(x, c, cum, xs))
    aa = xs + (a - xs) % period
    bb = aa + left
    if bb <= xe:
        total = total + (_antideriv_at_nd(x, c, cum, bb)
                         - _antideriv_at_nd(x, c, cum, aa))
    else:
        total = total + (_antideriv_at_nd(x, c, cum, xe)
                         - _antideriv_at_nd(x, c, cum, aa))
        total = total + (_antideriv_at_nd(x, c, cum, xs + left + aa - xe)
                         - _antideriv_at_nd(x, c, cum, xs))
    return total


# ---------------------------------------------------------------------------
# evaluation under the three extrapolation modes
# ---------------------------------------------------------------------------
#
# One routine per shape, each taking the resolved code. Mode 1 is the plain
# clamped-interval evaluation the kernels already do; mode 2 folds the query
# into the base period first; mode 0 evaluates and then masks the points that
# were outside, which is where scipy puts its NaN.


@njit
def _outside(x, xq):
    """True when `xq` lies outside ``[x[0], x[-1]]``, or is NaN."""
    return (xq < x[0]) or (xq > x[x.shape[0] - 1]) or (xq != xq)


@njit
def _eval_mode(x, c, xs, nu, extrap):
    """`_evaluate` under the extrapolation mode `extrap`."""
    out = _evaluate(x, c, _wrap_batch(x, xs, extrap == _EXTRAP_PERIODIC), nu)
    if extrap == _EXTRAP_OFF:
        for j in range(xs.shape[0]):
            if _outside(x, xs[j]):
                out[j] = np.nan
    return out


@njit
def _eval_one_mode(x, c, xq, nu, extrap):
    """`_eval_mode` for a single query point."""
    if extrap == _EXTRAP_OFF and _outside(x, xq):
        return np.nan
    xw = _wrap_one(x, xq, extrap == _EXTRAP_PERIODIC)
    i = _find_interval(x, xw)
    return _eval_seg(c, i, xw - x[i], nu)


@njit
def _eval_mode_nd(x, c3, xs, nu, extrap):
    """`_evaluate_nd` under the extrapolation mode `extrap`."""
    out = _evaluate_nd(x, c3, _wrap_batch(x, xs, extrap == _EXTRAP_PERIODIC),
                       nu)
    if extrap == _EXTRAP_OFF:
        for j in range(xs.shape[0]):
            if _outside(x, xs[j]):
                for k in range(out.shape[1]):
                    out[j, k] = np.nan
    return out


@njit
def _point_mode_nd(x, c3, xq, nu, extrap):
    """`_point_nd` under the extrapolation mode `extrap`."""
    if extrap == _EXTRAP_OFF and _outside(x, xq):
        out = np.empty(c3.shape[2], np.float64)
        for k in range(c3.shape[2]):
            out[k] = np.nan
        return out
    return _point_nd(x, c3, _wrap_one(x, xq, extrap == _EXTRAP_PERIODIC), nu)


@njit
def _integral_mode(x, c, cum, a, b, extrap):
    """Definite integral under the extrapolation mode `extrap`.

    Mode 0 gives NaN when either limit is outside the data, which is what
    ``scipy.interpolate.PPoly.integrate`` returns there.
    """
    if extrap == _EXTRAP_PERIODIC:
        return _periodic_integral(x, c, cum, a, b)
    if extrap == _EXTRAP_OFF and (_outside(x, a) or _outside(x, b)):
        return np.nan
    return _antideriv_at(x, c, cum, b) - _antideriv_at(x, c, cum, a)


@njit
def _integral_mode_nd(x, c, cum, a, b, extrap):
    """`_integral_mode` for the stored ``(4, n-1, T)`` layout."""
    if extrap == _EXTRAP_PERIODIC:
        return _periodic_integral_nd(x, c, cum, a, b)
    if extrap == _EXTRAP_OFF and (_outside(x, a) or _outside(x, b)):
        out = np.empty(c.shape[2], np.float64)
        for k in range(c.shape[2]):
            out[k] = np.nan
        return out
    return (_antideriv_at_nd(x, c, cum, b)
            - _antideriv_at_nd(x, c, cum, a))


# ---------------------------------------------------------------------------
# jitclasses
# ---------------------------------------------------------------------------

_cubic_spec = [
    ('x', float64[:]),
    ('c', float64[:, :]),
    ('cum', float64[:]),
    ('n', int64),
    ('bc_type', int64),
    ('extrapolate', int64),
]


@scijitclass(_cubic_spec)
class _CubicSpline:
    """Instance type behind the `CubicSpline` factory, which carries the
    documentation and the defaults. Every argument is explicit here: a
    jitclass constructor's defaults are Python-only.
    """

    def __init__(self, x, y, bc_type=0, spec=_BC_NAK_SPEC, extrapolate=1):
        n = _check_cubic(x, y, bc_type)
        xf = x.astype(np.float64)
        yf = np.ascontiguousarray(y.astype(np.float64))
        if bc_type == 3:
            _check_periodic_ends(yf.reshape(n, 1))
        dydx = _solve_derivs(xf, yf, spec)
        c = _build_c(xf, yf, dydx)
        self.x = xf
        self.c = c
        self.cum = _cumulative(xf, c)
        self.n = n
        self.bc_type = bc_type
        self.extrapolate = extrapolate

    def ev(self, xs):
        """Evaluate the spline at an array of points (scipy's ``f(xs)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at, in any order. What happens outside
            ``[x[0], x[-1]]`` follows the `extrapolate` code the object
            carries: 1 continues the end segment polynomial, 0 gives NaN,
            2 folds the query into the base period.

        Returns
        -------
        out : 1-D float64 ndarray, same length as `xs`
            Spline values.
        """
        return _eval_mode(self.x, self.c, xs, 0, self.extrapolate)

    def ev_one(self, x):
        """Evaluate the spline at a single point.

        Parameters
        ----------
        x : float
            Point to evaluate at; outside the data range the object's
            `extrapolate` code decides, as in `ev`.

        Returns
        -------
        float
            The spline value.
        """
        return _eval_one_mode(self.x, self.c, x, 0, self.extrapolate)

    def __getitem__(self, xs):
        """``f[xs]`` -- sugar for ``f.ev(xs)``.

        ``f(xs)`` reaches the same method and is the spelling scipy uses.

        Parameters
        ----------
        xs : 1-D float64 ndarray

        Returns
        -------
        out : 1-D float64 ndarray
        """
        return _eval_mode(self.x, self.c, xs, 0, self.extrapolate)

    def derivative_ev(self, xs, nu=1):
        """Evaluate the `nu`-th derivative (scipy's ``f(xs, nu)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at; extrapolated outside the data range.
        nu : int, optional
            Derivative order. 0 gives the value itself, 1 the slope, 2 the
            curvature, 3 the (piecewise-constant) third derivative; any
            ``nu >= 4`` returns zeros, which is what a cubic's higher
            derivatives are. A negative `nu` raises ``ValueError``; the
            antiderivative is `integral`. Default 1. Unlike a constructor
            default, this method default DOES work inside ``@njit``.

        Returns
        -------
        out : 1-D float64 ndarray, same length as `xs`
        """
        return _eval_mode(self.x, self.c, xs, nu, self.extrapolate)

    def integral(self, a, b):
        """Definite integral over ``[a, b]``.

        scipy's ``CubicSpline.integrate``.

        Parameters
        ----------
        a, b : float
            Integration limits. Either may lie outside ``[x[0], x[-1]]``,
            where the object's `extrapolate` code decides: 1 integrates the
            end segment polynomial on into the extrapolated region, 0 gives
            NaN, 2 counts whole periods. ``b < a`` gives the negated
            integral, as scipy does.

        Returns
        -------
        float
            The definite integral.

        Notes
        -----
        O(1) per call: the antiderivative at every breakpoint is precomputed
        in the constructor. Measured against
        ``scipy.interpolate.CubicSpline.integrate`` over the full data range,
        8.9e-16 to 2.2e-15 depending on the bc type.
        """
        return _integral_mode(self.x, self.c, self.cum, a, b,
                              self.extrapolate)

    def get_knots(self):
        """Return the breakpoints (scipy's ``PPoly.x``).

        Returns
        -------
        x : 1-D float64 ndarray, length n
            The breakpoint array held by the object -- a reference, not a
            copy, so do not modify it in place.
        """
        return self.x

    def get_coeffs(self):
        """Return the piecewise-polynomial coefficients (scipy's ``PPoly.c``).

        Returns
        -------
        c : (4, n-1) float64 ndarray
            ``c[k, i]`` is the coefficient of ``(x - x[i]) ** (3 - k)`` on the
            segment ``[x[i], x[i+1]]`` -- the same layout and orientation as
            ``scipy.interpolate.CubicSpline.c``. A reference, not a copy.
        """
        return self.c


_cubic_nd_spec = [
    ('x', float64[:]),
    ('c', float64[:, :, :]),
    ('cum', float64[:, :]),
    ('n', int64),
    ('bc_type', int64),
    ('rest', int64[:]),
    ('axis', int64),
    ('extrapolate', int64),
]

_CUBIC_ND_SRC = '''
class _CubicSplineND{trail}:
    """Instance type behind `CubicSpline` for a rank-{rank} `y`.

    Holds T independent splines over one `x`. `rest` carries the shape of the
    axes other than the interpolation axis, and `axis` where that axis sat in
    the caller's `y`, so a result can be handed back in the caller's layout.

    Generated by ``_make_cubic_nd({rank})`` in ``scijit/interpolate/_cubic.py``
    and bound there under this name. One class exists per supported rank of
    `y`; they share this source, the spec and every kernel, and differ only in
    the rank they hand back. `scijit.interpolate._ndaxis` says why.
    """

    def __init__(self, x, y2, bc_type, rest, axis, spec, extrapolate):
        n = _check_cubic(x, y2, bc_type)
        xf = x.astype(np.float64)
        yf = np.ascontiguousarray(y2.astype(np.float64))
        if bc_type == 3:
            _check_periodic_ends(yf)
        c = _build_c_nd(xf, yf, spec)
        self.x = xf
        self.c = c
        self.cum = _cumulative_nd(xf, c)
        self.n = n
        self.bc_type = bc_type
        self.rest = rest
        self.axis = axis
        self.extrapolate = extrapolate

    def ev(self, xs):
        """Evaluate every series at an array of points (scipy's ``f(xs)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at, in any order. Outside ``[x[0], x[-1]]``
            the object's `extrapolate` code decides, as in the 1-D class.

        Returns
        -------
        out : {rank}-D float64 ndarray
            The shape of `y`, with the interpolation axis replaced by
            ``len(xs)``.
        """
        return _nda._restore{rank}(
            _eval_mode_nd(self.x, self.c, xs, 0, self.extrapolate),
            self.rest, self.axis)

    def ev_one(self, x):
        """Evaluate every series at a single point.

        Parameters
        ----------
        x : float
            Point to evaluate at; outside the data range the object's
            `extrapolate` code decides.

        Returns
        -------
        out : {trail}-D float64 ndarray
            The shape of `y` with the interpolation axis removed. A scalar
            query drops that axis instead of replacing it, so nothing moves.
        """
        return _nda._point{rank}(
            _point_mode_nd(self.x, self.c, x, 0, self.extrapolate),
            self.rest)

    def __getitem__(self, xs):
        """``f[xs]`` -- sugar for ``f.ev(xs)``."""
        return _nda._restore{rank}(
            _eval_mode_nd(self.x, self.c, xs, 0, self.extrapolate),
            self.rest, self.axis)

    def derivative_ev(self, xs, nu=1):
        """Evaluate the `nu`-th derivative of every series.

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at; the object's `extrapolate` code decides
            outside the data range.
        nu : int, optional
            Derivative order, default 1. ``nu >= 4`` gives zeros and a
            negative `nu` raises ``ValueError``, as in the 1-D class.

        Returns
        -------
        out : {rank}-D float64 ndarray
            Shaped as `ev`.
        """
        return _nda._restore{rank}(
            _eval_mode_nd(self.x, self.c, xs, nu, self.extrapolate),
            self.rest, self.axis)

    def integral(self, a, b):
        """Definite integral of every series over ``[a, b]``.

        Parameters
        ----------
        a, b : float
            Limits; either may lie outside the data range, where the
            object's `extrapolate` code decides, and ``b < a`` negates, as
            in the 1-D class.

        Returns
        -------
        out : {trail}-D float64 ndarray
            The shape of `y` with the interpolation axis removed.
        """
        return _nda._point{rank}(
            _integral_mode_nd(self.x, self.c, self.cum, a, b,
                              self.extrapolate), self.rest)

    def get_knots(self):
        """Return the breakpoints (scipy's ``PPoly.x``).

        A reference to the stored array, not a copy.
        """
        return self.x

    def get_coeffs(self):
        """Return the piecewise-polynomial coefficients (scipy's ``PPoly.c``).

        Returns
        -------
        c : (4, n-1, {shape}) float64 ndarray
            The 1-D layout with the trailing axes appended, which is scipy's.
            The interpolation axis is NOT restored here: scipy leaves the
            trailing axes of ``.c`` in their original order too.
        """
{coeffs}
'''


def _make_cubic_nd(rank):
    """Build the `CubicSpline` N-D class for a `y` of this rank.

    Parameters
    ----------
    rank : int
        Rank of the caller's `y`, from 2 to ``1 + _ND_MAXTRAIL``.

    Returns
    -------
    cls : jitclass
        The class, decorated with `scijitclass` against `_cubic_nd_spec` and
        bound in this module as ``_CubicSplineND<rank-1>``.
    """
    trail = rank - 1
    src = _CUBIC_ND_SRC.format(
        rank=rank, trail=trail, shape=_shape_words(trail),
        coeffs=_coeff_body(trail, "4, self.n - 1"))
    return scijitclass(_cubic_nd_spec)(
        _define(src, globals(), "_CubicSplineND%d" % trail))


for _rank in _ND_RANKS:
    globals()["_CubicSplineND%d" % (_rank - 1)] = _make_cubic_nd(_rank)
del _rank


_CUBIC_DISPATCH_SRC = '''
def _cubic_nd(x, y, bc, axis, spec, extrap):
    """Construct the `CubicSpline` class that matches the rank of `y`.

    `y.ndim` is part of `y`'s numba type, so every branch but one is removed
    while compiling and the survivor fixes the class this returns. A rank past
    the cap leaves only the ``raise``.

    Generated by ``_make_cubic_dispatch()``; see `scijit.interpolate._ndaxis`.
    """
{branches}
    raise ValueError("y must have rank {phrase}")
'''


def _make_cubic_dispatch():
    """Build `_cubic_nd`, one branch per supported rank of `y`."""
    def body(rank):
        return ("        flat, rest, ax = _nda._flatten%d(y, axis)\n"
                "        return _CubicSplineND%d(x, flat, bc, rest, ax,\n"
                "                                spec, extrap)"
                % (rank, rank - 1))
    src = _CUBIC_DISPATCH_SRC.format(
        branches=_dispatch_branches("y", body), phrase=_rank_phrase())
    return njit(_define(src, globals(), "_cubic_nd"))


_cubic_nd = _make_cubic_dispatch()


@njit
def CubicSpline(x, y, axis=0, bc_type="not-a-knot", extrapolate=None):
    """C2 cubic spline interpolator over piecewise-cubic segments.

    The interpolant is twice continuously differentiable, and the boundary
    condition fixes the two remaining degrees of freedom.

    Parameters
    ----------
    x : 1-D float64 ndarray, length n >= 2
        Breakpoints, strictly increasing. A non-increasing pair raises
        ``ValueError``.
    y : float64 ndarray of rank 1, 2 or 3
        Values at the breakpoints. Its length along `axis` must be n. Every
        position on the other axes is an independent series over the same `x`.
        Rank 4 and above raise ``ValueError``.
    axis : int, optional
        Which axis of `y` the interpolation runs along, default 0. Negative
        values count from the end. Ignored for a 1-D `y`, which has only one
        axis. ``ev(xs)`` returns the shape of `y` with this axis replaced by
        ``len(xs)``; ``ev_one`` and `integral` return the shape of `y` with
        this axis removed.
    bc_type : str or 2-element sequence, optional
        Boundary condition. A single string applies the same condition at
        BOTH ends::

            'not-a-knot'   the default
            'natural'      second derivative zero at the ends
            'clamped'      first derivative zero at the ends
            'periodic'     the spline repeats with period ``x[-1] - x[0]``;
                           requires ``y[0] == y[-1]``, and evaluation folds
                           the query into that period rather than
                           extrapolating

        Otherwise a 2-element sequence giving the ends separately. Each
        element is independently one of the first three strings above or an
        ``(order, value)`` pair fixing ``S^(order)`` to `value` at that end,
        with `order` 1 or 2. So ``((1, 0.0), (1, 0.0))`` is ``'clamped'``,
        ``((2, 1.5), (1, -2.0))`` is a curvature at the left and a slope at
        the right, and ``('not-a-knot', (1, 3.0))`` mixes the two forms.
        ``'periodic'`` names a condition on the pair rather than on one end
        and is legal only as the single string; in the 2-element form it
        raises ``ValueError``.

        An unrecognised string raises ``ValueError``; a shape that is neither
        is refused while compiling. A string may be a runtime value, not only
        a literal. The default applies inside ``@njit`` as well as from
        Python.
    extrapolate : None, bool or str, optional
        What a query outside ``[x[0], x[-1]]`` gives::

            True           the end segment polynomial, continued
            False          NaN
            'periodic'     the query folded into ``[x[0], x[-1]]`` first
            None           the default: ``'periodic'`` when
                           ``bc_type='periodic'``, otherwise True

        A string other than ``'periodic'`` raises ``ValueError``. 0 and 1 are
        accepted for False and True. May be a runtime value, not only a
        literal.

    Returns
    -------
    _CubicSpline
        A callable jitclass instance: ``cs(xs)`` runs `ev` and ``cs(x)`` runs
        `ev_one`. A rank-2 or rank-3 `y` gives `_CubicSplineND1` or
        `_CubicSplineND2`, which carry the same methods.

    See Also
    --------
    scipy.interpolate.CubicSpline : The scipy routine this mirrors.

    Notes
    -----
    Deviations from scipy. `extrapolate` reads back as the resolved int code
    rather than the caller's value, and an unrecognised string raises where
    scipy extrapolates. A derivative `value` is one scalar per end and applies
    to every series of an N-D `y`, where scipy takes an array of the trailing
    shape. `y` is capped at rank 3 and is float64, where scipy takes any rank
    and promotes a complex `y` to ``complex128``; a jitclass field has a fixed
    rank and a fixed dtype, so each supported rank is a separate class. A rank-2
    `x` raises ``TypingError`` where scipy raises
    ``ValueError('`x` must be 1-dimensional.')``; both refuse and only the
    class differs. Evaluation is ``ev(xs)`` and ``derivative_ev(xs, nu)``,
    where scipy spells both as ``cs(xs, nu)``; a two-argument call raises on
    arity here. There is no ``.derivative()`` or ``.antiderivative()``
    returning a new object, and no ``solve``, ``roots``, ``extend``,
    ``construct_fast``, ``from_spline`` or ``from_bernstein_basis``.

    ``CubicSpline`` is an ``@njit`` factory over the jitclass
    ``_CubicSpline``, so ``bc_type`` and `extrapolate` may be omitted from
    Python AND from inside ``@njit``, and the strings resolve there, before
    the jitclass is constructed. Method defaults such as ``derivative_ev``'s
    ``nu=1`` work in both too.

    Measured against scipy 1.18 on 5 nodes spanning ``[0, 1.4]``, queried at
    2.0 and -1.0: ``extrapolate=None`` gives 13.1204501 and -13.35208741 on
    both sides, ``extrapolate=False`` NaN on both, ``extrapolate='periodic'``
    0.80812133 and 0.96731898, and ``bc_type='periodic'`` with
    ``extrapolate=None`` 0.83153846 and 1.00846154.

    Accuracy vs ``scipy.interpolate.CubicSpline`` on 15 random nodes over 211
    in-range points, for each of the three bc types: values 4.6e-15 to
    5.4e-15, first derivative 7.1e-15 to 1.4e-14, second derivative 2.8e-13 to
    4.0e-13, ``integrate(x[0], x[-1])`` 8.9e-16 to 2.2e-15, and PPoly
    coefficients 2.0e-13 (natural) to 4.7e-11 (not-a-knot) in absolute terms
    on coefficients whose magnitude reaches 5.0e3, i.e. about 1e-14
    relative. On 9 nodes at five points, values, coefficients, first
    derivative and ``integral`` together: the per-end ``(order, value)`` pairs
    2.220e-16, and ``'periodic'`` 0.000e+00 at n=9 with 2.132e-13 on the
    coefficients at n=12.

    Extrapolating 50 units past each end: 2.3e-10 absolute,
    8.9e-16 relative. The n=2 not-a-knot case is exactly 0.0 and the n=3
    parabola special case 4.4e-16.

    prange-safe: yes -- pure numpy, no module state, no callback. Both
    construction and evaluation may run inside a ``prange`` body.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import CubicSpline
    >>> x = np.linspace(0.0, 1.0, 9)
    >>> y = np.sin(3.0 * x)
    >>> cs = CubicSpline(x, y)          # 'not-a-knot', extrapolating
    >>> np.round(cs(np.array([0.25, 0.75])), 6)
    array([0.681639, 0.778073])

    Inside compiled code, the mean of the curve over its domain:

    >>> @njit
    ... def mean(cs):
    ...     xs = np.linspace(0.0, 1.0, 201)
    ...     return np.mean(cs(xs))
    >>> float(np.round(mean(cs), 6))
    0.660382

    Attributes
    ----------
    x : 1-D float64 ndarray
        The breakpoints, as float64.
    c : (4, n-1) float64 ndarray, or (4, n-1, T) for an N-D `y`
        PPoly coefficients: ``c[k, i]`` multiplies ``(x - x[i]) ** (3 - k)``
        on segment ``[x[i], x[i+1]]``. For an N-D `y` the trailing axes are
        flattened into one; `get_coeffs` unflattens them.
    n : int
        Number of breakpoints.
    bc_type : int
        Which condition built the spline: 0 not-a-knot, 1 natural, 2 clamped,
        3 periodic, 4 any other derivative pair.
    extrapolate : int
        The resolved extrapolation mode: 0 NaN outside the data, 1 the end
        segment continued, 2 the query folded into the base period.

    Methods
    -------
    ev(xs), ev_one(x), __getitem__(xs), derivative_ev(xs, nu),
    integral(a, b), get_knots(), get_coeffs()
    """
    _check_axis(axis)
    spec = _bc_spec(bc_type)
    bc = _bc_summary(spec)
    # scipy's `extrapolate=None` is 'periodic' for a periodic fit and True
    # otherwise, which is the only place the default depends on another
    # argument.
    if bc == 3:
        extrap = _extrap_code(extrapolate, _EXTRAP_PERIODIC)
    else:
        extrap = _extrap_code(extrapolate, _EXTRAP_ON)
    if y.ndim == 1:
        return _CubicSpline(x, y, bc, spec, extrap)
    else:
        return _cubic_nd(x, y, bc, axis, spec, extrap)


_pchip_spec = [
    ('x', float64[:]),
    ('c', float64[:, :]),
    ('n', int64),
    ('extrapolate', int64),
]


@scijitclass(_pchip_spec)
class _PchipInterpolator:
    """Instance type behind the `PchipInterpolator` factory, which carries the
    documentation. Every argument is explicit here: a jitclass constructor's
    defaults are Python-only.
    """

    def __init__(self, x, y, extrapolate=1):
        n = _check_pchip(x, y)
        xf = x.astype(np.float64)
        yf = y.astype(np.float64)
        dydx = _pchip_derivs(xf, yf)
        self.x = xf
        self.c = _build_c(xf, yf, dydx)
        self.n = n
        self.extrapolate = extrapolate

    def ev(self, xs):
        """Evaluate the spline at an array of points (scipy's ``f(xs)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at, in any order. What happens outside
            ``[x[0], x[-1]]`` follows the `extrapolate` code the object
            carries: 1 continues the end segment polynomial, 0 gives NaN,
            2 folds the query into the base period.

        Returns
        -------
        out : 1-D float64 ndarray, same length as `xs`
            Spline values.
        """
        return _eval_mode(self.x, self.c, xs, 0, self.extrapolate)

    def ev_one(self, x):
        """Evaluate the spline at a single point.

        Parameters
        ----------
        x : float
            Point to evaluate at; outside the data range the object's
            `extrapolate` code decides, as in `ev`.

        Returns
        -------
        float
            The spline value.
        """
        return _eval_one_mode(self.x, self.c, x, 0, self.extrapolate)

    def __getitem__(self, xs):
        """``f[xs]`` -- sugar for ``f.ev(xs)``.

        ``f(xs)`` reaches the same method and is the spelling scipy uses.

        Parameters
        ----------
        xs : 1-D float64 ndarray

        Returns
        -------
        out : 1-D float64 ndarray
        """
        return _eval_mode(self.x, self.c, xs, 0, self.extrapolate)

    def derivative_ev(self, xs, nu=1):
        """Evaluate the `nu`-th derivative (scipy's ``f(xs, nu)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at; extrapolated outside the data range.
        nu : int, optional
            Derivative order. 0 gives the value itself, 1 the slope, 2 the
            curvature, 3 the (piecewise-constant) third derivative; any
            ``nu >= 4`` returns zeros, which is what a cubic's higher
            derivatives are. A negative `nu` raises ``ValueError``; the
            antiderivative is `integral`. Default 1. Unlike a constructor
            default, this method default DOES work inside ``@njit``.

        Returns
        -------
        out : 1-D float64 ndarray, same length as `xs`
        """
        return _eval_mode(self.x, self.c, xs, nu, self.extrapolate)

    def get_knots(self):
        """Return the breakpoints (scipy's ``PPoly.x``).

        Returns
        -------
        x : 1-D float64 ndarray, length n
            The breakpoint array held by the object -- a reference, not a
            copy, so do not modify it in place.
        """
        return self.x

    def get_coeffs(self):
        """Return the piecewise-polynomial coefficients (scipy's ``PPoly.c``).

        Returns
        -------
        c : (4, n-1) float64 ndarray
            ``c[k, i]`` is the coefficient of ``(x - x[i]) ** (3 - k)`` on the
            segment ``[x[i], x[i+1]]`` -- the same layout and orientation as
            ``scipy.interpolate.CubicSpline.c``. A reference, not a copy.
        """
        return self.c


_pchip_nd_spec = [
    ('x', float64[:]),
    ('c', float64[:, :, :]),
    ('n', int64),
    ('rest', int64[:]),
    ('axis', int64),
    ('extrapolate', int64),
]

_PCHIP_ND_SRC = '''
class _PchipInterpolatorND{trail}:
    """Instance type behind `PchipInterpolator` for a rank-{rank} `y`.

    Holds T independent shape-preserving interpolators over one `x`. `rest`
    carries the shape of the axes other than the interpolation axis, `axis`
    where that axis sat in the caller's `y`.

    Generated by ``_make_pchip_nd({rank})`` in ``scijit/interpolate/_cubic.py``
    and bound there under this name. One class exists per supported rank of
    `y`; they share this source, the spec and every kernel, and differ only in
    the rank they hand back. `scijit.interpolate._ndaxis` says why.
    """

    def __init__(self, x, y2, rest, axis, extrapolate):
        n = _check_pchip(x, y2)
        xf = x.astype(np.float64)
        yf = np.ascontiguousarray(y2.astype(np.float64))
        self.x = xf
        self.c = _build_c_pchip_nd(xf, yf)
        self.n = n
        self.rest = rest
        self.axis = axis
        self.extrapolate = extrapolate

    def ev(self, xs):
        """Evaluate every series at an array of points (scipy's ``f(xs)``).

        Parameters
        ----------
        xs : 1-D float64 ndarray
            Points to evaluate at. Outside ``[x[0], x[-1]]`` the object's
            `extrapolate` code decides, as in the 1-D class.

        Returns
        -------
        out : {rank}-D float64 ndarray
            The shape of `y`, with the interpolation axis replaced by
            ``len(xs)``.
        """
        return _nda._restore{rank}(
            _eval_mode_nd(self.x, self.c, xs, 0, self.extrapolate),
            self.rest, self.axis)

    def ev_one(self, x):
        """Evaluate every series at a single point.

        Returns
        -------
        out : {trail}-D float64 ndarray
            The shape of `y` with the interpolation axis removed.
        """
        return _nda._point{rank}(
            _point_mode_nd(self.x, self.c, x, 0, self.extrapolate),
            self.rest)

    def __getitem__(self, xs):
        """``f[xs]`` -- sugar for ``f.ev(xs)``."""
        return _nda._restore{rank}(
            _eval_mode_nd(self.x, self.c, xs, 0, self.extrapolate),
            self.rest, self.axis)

    def derivative_ev(self, xs, nu=1):
        """Evaluate the `nu`-th derivative of every series.

        Returns
        -------
        out : {rank}-D float64 ndarray
            Shaped as `ev`.
        """
        return _nda._restore{rank}(
            _eval_mode_nd(self.x, self.c, xs, nu, self.extrapolate),
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


def _make_pchip_nd(rank):
    """Build the `PchipInterpolator` N-D class for a `y` of this rank.

    Parameters
    ----------
    rank : int
        Rank of the caller's `y`, from 2 to ``1 + _ND_MAXTRAIL``.

    Returns
    -------
    cls : jitclass
        The class, decorated with `scijitclass` against `_pchip_nd_spec` and
        bound in this module as ``_PchipInterpolatorND<rank-1>``.
    """
    trail = rank - 1
    src = _PCHIP_ND_SRC.format(
        rank=rank, trail=trail, shape=_shape_words(trail),
        coeffs=_coeff_body(trail, "4, self.n - 1"))
    return scijitclass(_pchip_nd_spec)(
        _define(src, globals(), "_PchipInterpolatorND%d" % trail))


for _rank in _ND_RANKS:
    globals()["_PchipInterpolatorND%d" % (_rank - 1)] = _make_pchip_nd(_rank)
del _rank


_PCHIP_DISPATCH_SRC = '''
def _pchip_nd(x, y, axis, extrap):
    """Construct the `PchipInterpolator` class matching the rank of `y`.

    `y.ndim` is part of `y`'s numba type, so every branch but one is removed
    while compiling and the survivor fixes the class this returns. A rank past
    the cap leaves only the ``raise``.

    Generated by ``_make_pchip_dispatch()``; see `scijit.interpolate._ndaxis`.
    """
{branches}
    raise ValueError("y must have rank {phrase}")
'''


def _make_pchip_dispatch():
    """Build `_pchip_nd`, one branch per supported rank of `y`."""
    def body(rank):
        return ("        flat, rest, ax = _nda._flatten%d(y, axis)\n"
                "        return _PchipInterpolatorND%d(x, flat, rest, ax,\n"
                "                                      extrap)"
                % (rank, rank - 1))
    src = _PCHIP_DISPATCH_SRC.format(
        branches=_dispatch_branches("y", body), phrase=_rank_phrase())
    return njit(_define(src, globals(), "_pchip_nd"))


_pchip_nd = _make_pchip_dispatch()


@njit
def PchipInterpolator(x, y, axis=0, extrapolate=None):
    """Monotone piecewise-cubic Hermite interpolator (Fritsch-Carlson slopes).

    Preserves monotonicity of the data and does not overshoot, at the price of
    being only C1 (the second derivative jumps at the nodes) where
    `CubicSpline` is C2.

    Parameters
    ----------
    x : 1-D float64 ndarray, length n >= 2
        Breakpoints, strictly increasing. A non-increasing pair raises
        ``ValueError``.
    y : float64 ndarray of rank 1, 2 or 3
        Values at the breakpoints. Its length along `axis` must be n. Every
        position on the other axes is an independent series over the same `x`.
        Rank 4 and above raise ``ValueError``.
    axis : int, optional
        Which axis of `y` the interpolation runs along, default 0. Negative
        values count from the end. Ignored for a 1-D `y`. ``ev(xs)`` returns
        the shape of `y` with this axis replaced by ``len(xs)``; ``ev_one``
        returns it with this axis removed.
    extrapolate : None, bool or str, optional
        What a query outside ``[x[0], x[-1]]`` gives::

            True           the end segment polynomial, continued
            False          NaN
            'periodic'     the query folded into ``[x[0], x[-1]]`` first
            None           the default, which is True

        A string other than ``'periodic'`` raises ``ValueError``. 0 and 1 are
        accepted for False and True. May be a runtime value, not only a
        literal.

    Returns
    -------
    _PchipInterpolator
        A callable jitclass instance: ``f(xs)`` runs `ev` and ``f(x)`` runs
        `ev_one`. An N-D `y` gives `_PchipInterpolatorND1` or
        `_PchipInterpolatorND2`, which carry the same methods.

    See Also
    --------
    scipy.interpolate.PchipInterpolator : The scipy routine this mirrors.

    Notes
    -----
    Deviations from scipy. `extrapolate` reads back as the resolved int code
    rather than the caller's value, and an unrecognised string raises where
    scipy extrapolates. `y` is capped at rank 3 and is float64, where scipy
    takes any rank; a jitclass field has a fixed rank and a fixed dtype, so
    each supported rank is a separate class. A rank-2 `x` raises
    ``TypingError`` where scipy raises
    ``ValueError('`x` must be 1-dimensional.')``; both refuse and only the
    class differs. A complex `y` raises
    ``ValueError`` naming ``np.real``, which is scipy's own refusal, from
    Python and from inside ``@njit``. Evaluation is ``ev(xs)`` and
    ``derivative_ev(xs, nu)``, where scipy spells both as ``f(xs, nu)``; a
    two-argument call raises on arity here. There is no ``.derivative()`` or
    ``.antiderivative()`` object-returning API, no ``.roots()``, and no
    ``integral`` method, which `CubicSpline` here does carry.

    ``PchipInterpolator`` is an ``@njit`` factory over the jitclass
    ``_PchipInterpolator``, which is the shape every other evaluable name in
    this subpackage uses, so `extrapolate` may be omitted from Python AND from
    inside ``@njit``: a jitclass constructor's defaults are Python-only and a
    plain ``@njit`` function's are not.

    Measured against scipy 1.18 on 5 nodes spanning ``[0, 1.4]``, queried at
    2.0 and -1.0: ``extrapolate=None`` gives 7.07678571 and 10.1547619 on both
    sides, ``extrapolate=False`` NaN on both, ``extrapolate='periodic'``
    0.7145361 and 0.8756787.

    Accuracy vs ``scipy.interpolate.PchipInterpolator`` on 15 random nodes
    over 211 points: values 4.4e-16, first derivative 7.1e-15, and the PPoly
    coefficient array exactly 0.0.

    prange-safe: yes -- pure numpy, no state.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import PchipInterpolator
    >>> x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> y = np.array([0.0, 1.0, 1.0, 3.0, 3.0])
    >>> f = PchipInterpolator(x, y)
    >>> np.round(f(np.array([0.5, 2.5])), 6)
    array([0.6875, 2.    ])

    Inside compiled code, scanning the domain for the interpolant's peak:

    >>> @njit
    ... def peak(f):
    ...     xs = np.linspace(0.0, 4.0, 201)
    ...     return np.max(f(xs))
    >>> float(np.round(peak(f), 6))
    3.0

    Attributes
    ----------
    x : 1-D float64 ndarray
        The breakpoints, as float64.
    c : (4, n-1) float64 ndarray, or (4, n-1, T) for an N-D `y`
        PPoly coefficients: ``c[k, i]`` multiplies ``(x - x[i]) ** (3 - k)``
        on ``[x[i], x[i+1]]``. For an N-D `y` the trailing axes are flattened
        into one; `get_coeffs` unflattens them.
    n : int
        Number of breakpoints.
    extrapolate : int
        The resolved extrapolation mode: 0 NaN outside the data, 1 the end
        segment continued, 2 the query folded into the base period.

    Methods
    -------
    ev(xs), ev_one(x), __getitem__(xs), derivative_ev(xs, nu),
    get_knots(), get_coeffs()
    """
    _check_axis(axis)
    yr = _pchip_real_y(y)
    extrap = _extrap_code(extrapolate, _EXTRAP_ON)
    # The rank test reads the ARGUMENT, not `yr`: numba prunes a branch on an
    # argument's `.ndim` while compiling and does not prune one on a call
    # result, so reading `yr.ndim` here would type both branches and hand the
    # rank-2 array to the 1-D class.
    if y.ndim == 1:
        return _PchipInterpolator(x, yr, extrap)
    else:
        return _pchip_nd(x, yr, axis, extrap)
