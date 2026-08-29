"""Numba-callable wrappers for the FITPACK evaluator routines.

Exposes the "evaluator group" via the bind(c) wrappers in
fitpack/wrappers.f90:

    splev, splder, splint, spalde, sproot, fourco, insert   (univariate)
    curev, cualde                                           (parametric curves)
    parder, pardeu, dblint, profil, surev                   (bivariate)

Convention: evaluation points first, then knots/coeffs, then degrees. All
arrays are passed as raw pointers, so every function is usable inside @njit
code. Workspace formulas come from the FITPACK docstrings.

Coefficient layouts (FITPACK conventions). A layout mistake here produces a
plausible curve or surface and raises nothing, so it is worth reading before
the first call:
    univariate   : c[j], j=0..n-k-2, with the tail up to len(t) unused
    curve (idim) : the coefficients of dimension j start at c[j*n], where
                   n = len(t). STRIDE n, not n-k-1. scipy's ``splprep``
                   returns a list of shorter arrays (length n-k-1 padded to
                   n-k-1), so concatenating them gives the WRONG stride and
                   a plausible-looking but wrong curve; copy each dimension
                   into an ``np.zeros(idim*n)`` at offset ``j*n`` instead.
    bivariate    : c[(ny-ky-1)*i + j]  ==  np.outer(cx, cy).ravel()

Relation to scipy: of the 16 routines here, scipy.interpolate publishes
6 under the same names, `splev`, `splder`, `splint`, `spalde`, `sproot` and
`insert`. One of those six names means a different thing on each side:
``scipy.interpolate.splder(tck, n)`` returns the tck of the derivative
spline, while `splder` here evaluates the derivative at points. The
scipy-shaped counterpart of the scipy name is
``scijit.interpolate.splder``. scipy reaches the other 10 routines only from
inside its own spline classes, or not at all, and keeps them in a private
module. So `bispev`, `bispeu`, `curev`, `cualde`, `parder`, `pardeu`,
`dblint`, `profil`, `surev` and `fourco` have no scipy name to be compared
against; their docstrings state what each one computes rather than a
difference.

``fourco`` is the one routine here whose values carry more than rounding
error. FITPACK's ``fpcsin`` uses a truncated series, so it is measured
against ``scipy.integrate.quad`` of the same integrand rather than against a
FITPACK result: on a 30-node cubic sine spline over ``[t[k], t[n-k-1]]``, at
``alfa`` 0.25, 0.5, 1, 2, 4 and 8, the worst absolute difference of THESE
routines from that reference is 1.087e-08 for the sine transform and
1.636e-08 for the cosine. Read the absolute figure and not a relative one:
both transforms pass through zero as ``alfa`` varies.

prange-safety: all of these are pure in-out Fortran routines with no
callbacks and no saved state, and every workspace is a fresh local, so
they are prange-safe.

Argument validation: shapes, sizes and ranges are checked before the Fortran
runs and a failure raises ``ValueError``. The evaluation-point arrays must be
rank 1, tested first, on `splev`, `splder`, `curev`, `parder`, `pardeu`,
`bispev`, `bispeu` and `surev`, and so must `fourco`'s ``alfa``. Then
``len(c)`` on `splev`,
`splder`, `splint`, `spalde` and `sproot`; ``len(c) == (nx-kx-1)*(ny-ky-1)``,
an equality that rejects a padded `c` as well as a short one, on `bispev`,
`bispeu`, `parder` and `pardeu`; ``len(x) == len(y)`` on `bispeu` and
`pardeu`; the `e` range on `splev` and `splder`; the `nu` range on `splder`;
``0 <= nux < kx`` and ``0 <= nuy < ky`` on `parder` and `pardeu`;
``len(t) >= 8`` on `sproot`. Each routine's ``Raises`` section gives its own
tests in the order they run.

Error handling: FITPACK's ``ier`` status is raised on in three places, all
where it reports work that was not done. `spalde` raises ``TypeError`` for
an evaluation point outside ``[t[k], t[n-k-1]]``, and `splev` and `splder`
raise ``ValueError`` for one under ``e == 2``; FITPACK returns without
writing the output buffer in all three cases. Every other ``ier`` value is
received and discarded, including the ``ier = 10`` `splev` and `splder`
report for an empty `x`, which returns an empty array. So `fourco`, `insert`,
`curev`, `cualde`, `parder`, `pardeu`, `profil`, `surev`, `bispev`, `bispeu`
and `sproot` can return zeros, or a truncated result, for input FITPACK
rejects: a knot vector inconsistent with the degree, a grid axis that is not
non-decreasing, an empty grid axis, too small an ``mest``. Validate before
calling if the input is not under the caller's control.
"""
from numba import njit, types
from numba.extending import overload
import numpy as np
from .._lib._load import load

_lib, _sig = load(__file__, "libfitpack")


_splev  = _sig(_lib.splev_wrapper, 9)
_splder = _sig(_lib.splder_wrapper, 11)
_splint = _sig(_lib.splint_wrapper, 8)
_spalde = _sig(_lib.spalde_wrapper, 7)
_sproot = _sig(_lib.sproot_wrapper, 7)
_fourco = _sig(_lib.fourco_wrapper, 10)
_insert = _sig(_lib.insert_wrapper, 11)
_curev  = _sig(_lib.curev_wrapper, 11)
_cualde = _sig(_lib.cualde_wrapper, 10)
_parder = _sig(_lib.parder_wrapper, 19)
_pardeu = _sig(_lib.pardeu_wrapper, 18)
_dblint = _sig(_lib.dblint_wrapper, 13)
_profil = _sig(_lib.profil_wrapper, 12)
_surev  = _sig(_lib.surev_wrapper, 17)


@njit
def splev(x, t, c, k, e=0):
    """Evaluate a univariate spline, FITPACK ``splev``.

    Takes the knot vector, coefficients and degree as separate arguments,
    Dierckx's own argument order.

    Parameters
    ----------
    x : 1-D array_like of float
        Points to evaluate at. Wrapped in ``ascontiguousarray``, so strided
        views are safe.
    t : 1-D array_like of float
        Knot vector, length n.
    c : 1-D array_like of float
        B-spline coefficients. Only the first ``n - k - 1`` are used; a
        FITPACK ``tck`` whose `c` is padded to ``len(t)`` passes straight in.
    k : int
        Spline degree, 1 <= k <= 5.
    e : int, optional
        What to do with points outside ``[t[k], t[n-k-1]]``: 0 = extrapolate,
        1 = return 0, 2 = raise ``ValueError``, 3 = return the boundary value.
        Default 0.

    Returns
    -------
    y : 1-D float64 ndarray, same length as `x`
        Spline values.

    Raises
    ------
    ValueError
        If `t`, `c` or `x` has a rank other than 1, if `e` is outside
        0..3, if `c` holds fewer than ``n - k - 1`` coefficients, or if
        ``e == 2`` and a point
        of `x` lies outside ``[t[k], t[n-k-1]]``. The rank test runs first.

    See Also
    --------
    scipy.interpolate.splev : The scipy routine this mirrors.

    Notes
    -----
    An EMPTY `x` returns an empty array, where
    ``scipy.interpolate.splev`` raises ``ValueError("Invalid input data")``.

    prange-safe: yes.
    """
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    if t_.ndim != 1 or c_.ndim != 1 or x_.ndim != 1:
        raise ValueError("object too deep for desired array")
    if e < 0 or e > 3:
        raise ValueError("e must be between 0 and 3")
    if len(c_) < len(t_) - k - 1:
        raise ValueError("c array is too small")
    n = np.array(len(t_), np.int32)
    m = np.array(len(x_), np.int32)
    k_ = np.array(k, np.int32)
    e_ = np.array(e, np.int32)
    y = np.zeros(len(x_), np.float64)
    ier = np.zeros(1, np.int32)
    _splev(t_.ctypes.data, n.ctypes.data, c_.ctypes.data, k_.ctypes.data,
           x_.ctypes.data, y.ctypes.data, m.ctypes.data, e_.ctypes.data,
           ier.ctypes.data)
    if ier[0] == 1:
        raise ValueError("Found x value not in the domain")
    return y


@njit
def splder(x, t, c, k, nu=1, e=0):
    """Evaluate the `nu`-th derivative of a spline, FITPACK ``splder``.

    Evaluates the derivative at points, with Dierckx's own argument order.

    Parameters
    ----------
    x : 1-D array_like of float
        Points to evaluate at.
    t : 1-D array_like of float
        Knot vector, length n.
    c : 1-D array_like of float
        B-spline coefficients (padded tck form accepted).
    k : int
        Spline degree, 1 <= k <= 5.
    nu : int, optional
        Derivative order, ``0 <= nu <= k``. Default 1.
    e : int, optional
        Out-of-range behaviour: 0 = extrapolate, 1 = return 0, 2 = raise
        ``ValueError``, 3 = boundary value. Default 0.

    Returns
    -------
    y : 1-D float64 ndarray, same length as `x`
        The `nu`-th derivative values.

    Raises
    ------
    ValueError
        If `t`, `c` or `x` has a rank other than 1, if `nu` is outside
        0..k, if `e` is
        outside 0..3, if `c` holds fewer than ``n - k - 1`` coefficients, or
        if ``e == 2`` and a point of `x` lies outside ``[t[k], t[n-k-1]]``.
        The rank test runs first.

    See Also
    --------
    scipy.interpolate.splev : Evaluates a spline derivative through ``der=nu``.

    Notes
    -----
    Name clash: ``scipy.interpolate.splder`` is a different routine, returning
    the tck of the derivative spline. The derivative-at-points computation this
    function performs is ``scipy.interpolate.splev(x, tck, der=nu)``. The tck
    counterpart of the scipy ``splder`` name is ``scijit.interpolate.splder``.

    An EMPTY `x` returns an empty array, as in `splev`.

    prange-safe: yes.
    """
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    if t_.ndim != 1 or c_.ndim != 1 or x_.ndim != 1:
        raise ValueError("object too deep for desired array")
    if nu < 0 or nu > k:
        raise ValueError("nu must be between 0 and k")
    if e < 0 or e > 3:
        raise ValueError("e must be between 0 and 3")
    if len(c_) < len(t_) - k - 1:
        raise ValueError("c array is too small")
    n = np.array(len(t_), np.int32)
    m = np.array(len(x_), np.int32)
    k_ = np.array(k, np.int32)
    nu_ = np.array(nu, np.int32)
    e_ = np.array(e, np.int32)
    y = np.zeros(len(x_), np.float64)
    wrk = np.zeros(len(t_), np.float64)
    ier = np.zeros(1, np.int32)
    _splder(t_.ctypes.data, n.ctypes.data, c_.ctypes.data, k_.ctypes.data,
            nu_.ctypes.data, x_.ctypes.data, y.ctypes.data, m.ctypes.data,
            e_.ctypes.data, wrk.ctypes.data, ier.ctypes.data)
    if ier[0] == 1:
        raise ValueError("Found x value not in the domain")
    return y


@njit
def splint(t, c, k, a, b):
    """Definite integral of a spline, FITPACK ``splint``.

    Integrates the spline given by ``(t, c, k)`` over ``[a, b]``.

    Parameters
    ----------
    t : 1-D array_like of float
        Knot vector, length n.
    c : 1-D array_like of float
        B-spline coefficients (padded tck form accepted).
    k : int
        Spline degree.
    a, b : float
        Integration limits. They may lie outside ``[t[k], t[n-k-1]]``, in
        which case the spline is extrapolated. ``b < a`` gives the negated
        integral.

    Returns
    -------
    res : float
        The integral of s(x) over ``[a, b]``.

    Raises
    ------
    ValueError
        If `t` or `c` has a rank other than 1, or `c` holds fewer than
        ``n - k - 1`` coefficients. The rank test runs first.

    See Also
    --------
    scipy.interpolate.splint : The scipy routine this mirrors.

    Notes
    -----
    Dierckx wrote this as a Fortran FUNCTION; the bind(c) wrapper turns it
    into a subroutine with a ``res`` out-argument, which is why the
    wrapper reads
    ``res[0]``.

    prange-safe: yes.
    """
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    if t_.ndim != 1 or c_.ndim != 1:
        raise ValueError("object too deep for desired array")
    if len(c_) < len(t_) - k - 1:
        raise ValueError("The length of c must be >=n-k-1")
    n = np.array(len(t_), np.int32)
    k_ = np.array(k, np.int32)
    a_ = np.array(a, np.float64)
    b_ = np.array(b, np.float64)
    wrk = np.zeros(len(t_), np.float64)
    res = np.zeros(1, np.float64)
    _splint(t_.ctypes.data, n.ctypes.data, c_.ctypes.data, k_.ctypes.data,
            a_.ctypes.data, b_.ctypes.data, wrk.ctypes.data, res.ctypes.data)
    return res[0]


@njit
def _splint_wrk(t, c, k, a, b):
    """`splint`, returning FITPACK's ``wrk`` alongside the integral.

    ``splint.f`` fills ``wrk(j)`` with the integral of the j-th normalised
    B-spline over ``[a, b]``, and `splint` above discards it. This is the same
    call with that array kept, and it is what backs ``full_output`` on the
    tck-level `scijit.interpolate.splint`.

    Carries `splint`'s coefficient-length guard, so the two spellings of one
    integral raise on the same input.
    """
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    if t_.ndim != 1 or c_.ndim != 1:
        raise ValueError("object too deep for desired array")
    if len(c_) < len(t_) - k - 1:
        raise ValueError("The length of c must be >=n-k-1")
    n = np.array(len(t_), np.int32)
    k_ = np.array(k, np.int32)
    a_ = np.array(a, np.float64)
    b_ = np.array(b, np.float64)
    wrk = np.zeros(len(t_), np.float64)
    res = np.zeros(1, np.float64)
    _splint(t_.ctypes.data, n.ctypes.data, c_.ctypes.data, k_.ctypes.data,
            a_.ctypes.data, b_.ctypes.data, wrk.ctypes.data, res.ctypes.data)
    return res[0], wrk[:len(t_) - k - 1]


@njit
def spalde(x, t, c, k):
    """All derivatives at one point, FITPACK ``spalde``.

    Returns every derivative from the value up to order `k` at one scalar
    point.

    Parameters
    ----------
    x : float
        A SCALAR evaluation point (unlike `splev`, which takes an array).
        Must lie inside ``[t[k], t[n-k-1]]``.
    t : 1-D array_like of float
        Knot vector, length n.
    c : 1-D array_like of float
        B-spline coefficients (padded tck form accepted).
    k : int
        Spline degree.

    Returns
    -------
    d : 1-D float64 ndarray, length ``k + 1``
        ``d[j]`` is the j-th derivative at `x`: ``d[0]`` is s(x), ``d[1]`` is
        s'(x), up to ``d[k]``.

    Raises
    ------
    ValueError
        If `t` or `c` has a rank other than 1, or `c` holds fewer than
        ``n - k - 1`` coefficients. The rank test runs first.
    TypeError
        If `x` lies outside ``[t[k], t[n-k-1]]``, or the knot interval
        containing it is degenerate.

    See Also
    --------
    scipy.interpolate.spalde : The scipy routine this mirrors.

    Notes
    -----
    prange-safe: yes.
    """
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    if t_.ndim != 1 or c_.ndim != 1:
        raise ValueError("object too deep for desired array")
    if len(c_) < len(t_) - k - 1:
        raise ValueError("c array is too small")
    n = np.array(len(t_), np.int32)
    k1 = np.array(k + 1, np.int32)
    x_ = np.array(x, np.float64)
    d = np.zeros(k + 1, np.float64)
    ier = np.zeros(1, np.int32)
    _spalde(t_.ctypes.data, n.ctypes.data, c_.ctypes.data, k1.ctypes.data,
            x_.ctypes.data, d.ctypes.data, ier.ctypes.data)
    if ier[0] == 10:
        raise TypeError(
            "Invalid input data. t(k)<=x<=t(n-k+1) must hold.")
    return d


@njit
def sproot(t, c, mest=-1):
    """Zeros of a cubic spline, FITPACK ``sproot``.

    Locates the zeros of a cubic spline in ``[t[3], t[n-4]]``.

    Parameters
    ----------
    t : 1-D array_like of float
        Knot vector, length n >= 8. CUBIC ONLY -- FITPACK's ``sproot`` is
        defined for k=3 and nothing else, so there is no `k` argument.
    c : 1-D array_like of float
        B-spline coefficients (padded tck form accepted).
    mest : int, optional
        Size of the root buffer, and therefore the maximum number of roots
        returned. A NEGATIVE `mest` (the default is ``-1``) means
        ``3 * (len(t) - 7)``, FITPACK's own worst case, so the default never
        truncates. ``mest=0`` returns an empty array. If the spline has more
        zeros than `mest` the extra ones are dropped and FITPACK flags it in
        ``ier``, which this wrapper does not check -- pass a larger `mest`
        if that is a risk.

    Returns
    -------
    zero : 1-D float64 ndarray, length m
        The roots found in ``[t[3], t[n-4]]``, ascending. Length m is however
        many FITPACK located, so it can be 0.

    Raises
    ------
    ValueError
        If `t` or `c` has a rank other than 1, if `t` holds fewer than 8
        knots, or `c` holds fewer than ``n - 4`` coefficients. The rank test
        runs first.

    See Also
    --------
    scipy.interpolate.sproot : The scipy routine this mirrors.

    Notes
    -----
    prange-safe: yes.
    """
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    if t_.ndim != 1 or c_.ndim != 1:
        raise ValueError("object too deep for desired array")
    if len(t_) < 8:
        raise ValueError("t array must have at least 8 elements")
    if len(c_) < len(t_) - 4:
        raise ValueError("c array is too small")
    n = np.array(len(t_), np.int32)
    if mest < 0:
        mest = 3 * (len(t_) - 7)
    mest_ = np.array(mest, np.int32)
    zero = np.zeros(mest, np.float64)
    m = np.zeros(1, np.int32)
    ier = np.zeros(1, np.int32)
    _sproot(t_.ctypes.data, n.ctypes.data, c_.ctypes.data, zero.ctypes.data,
            mest_.ctypes.data, m.ctypes.data, ier.ctypes.data)
    return zero[:m[0]]


@njit
def _sproot_ier(t, c, mest):
    """`sproot`, returning FITPACK's ``ier`` alongside the roots.

    ``sproot.f`` sets ``ier = 1`` when the spline has more zeros than `mest`
    and the extra ones were dropped, and `sproot` above discards that. The
    tck-level `scijit.interpolate.sproot` reads it to reproduce scipy's
    truncation warning. ``m == mest`` is not a substitute: a spline with
    exactly `mest` roots fills the buffer with ``ier = 0``.

    Carries `sproot`'s two size guards, so the two spellings of one root
    search raise on the same input.
    """
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    if t_.ndim != 1 or c_.ndim != 1:
        raise ValueError("object too deep for desired array")
    if len(t_) < 8:
        raise ValueError("t array must have at least 8 elements")
    if len(c_) < len(t_) - 4:
        raise ValueError("c array is too small")
    n = np.array(len(t_), np.int32)
    if mest < 0:
        mest = 3 * (len(t_) - 7)
    mest_ = np.array(mest, np.int32)
    zero = np.zeros(mest, np.float64)
    m = np.zeros(1, np.int32)
    ier = np.zeros(1, np.int32)
    _sproot(t_.ctypes.data, n.ctypes.data, c_.ctypes.data, zero.ctypes.data,
            mest_.ctypes.data, m.ctypes.data, ier.ctypes.data)
    return zero[:m[0]], ier[0]


@njit
def fourco(t, c, alfa):
    """Fourier integrals of a cubic spline, FITPACK ``fourco``.

    Computes, for each frequency ``a`` in `alfa`, the pair of integrals of
    ``s(x) sin(a x)`` and ``s(x) cos(a x)`` over ``[t[3], t[n-4]]``.

    Parameters
    ----------
    t : 1-D array_like of float
        Knot vector, length n. CUBIC ONLY (k=3), hence no `k` argument.
    c : 1-D array_like of float
        B-spline coefficients (padded tck form accepted).
    alfa : 1-D array_like of float
        Frequencies at which to compute the integrals.

    Returns
    -------
    ress : 1-D float64 ndarray, same length as `alfa`
        ``ress[i]`` is the integral of ``s(x) * sin(alfa[i] * x)``.
    resc : 1-D float64 ndarray, same length as `alfa`
        ``resc[i]`` is the integral of ``s(x) * cos(alfa[i] * x)``.

    Raises
    ------
    ValueError
        If `t`, `c` or `alfa` has a rank other than 1.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    ACCURACY IS NOT MACHINE EPSILON. FITPACK's ``fpcsin`` uses a truncated
    series, so this is the one routine in the module that does not agree with
    a reference to ~1e-16. Measured against ``scipy.integrate.quad`` of the
    same integrand on a 30-node cubic sine spline over ``[t[k], t[n-k-1]]``,
    at ``alfa`` = 0.25, 0.5, 1, 2, 4 and 8, the worst ABSOLUTE difference of
    THIS routine from that reference is 1.087e-08 for the sine transform and
    1.636e-08 for the cosine.

    READ THE ABSOLUTE FIGURE, NOT A RELATIVE ONE. Both transforms pass
    through zero as `alfa` varies: at alfa=0.5 the sine reference is
    2.057e-16 and at alfa=1 the cosine reference is 5.929e-17. Dividing by
    those gives relative errors of 3.997e+07 and 2.759e+08, which measure the
    cancellation in the reference rather than anything about `fourco`. Where
    the reference is order 1, the relative difference is 1.7e-09 to
    3.5e-09.

    prange-safe: yes.
    """
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    alfa_ = np.ascontiguousarray(np.asarray(alfa, np.float64))
    if t_.ndim != 1 or c_.ndim != 1 or alfa_.ndim != 1:
        raise ValueError("object too deep for desired array")
    n = np.array(len(t_), np.int32)
    m = np.array(len(alfa_), np.int32)
    ress = np.zeros(len(alfa_), np.float64)
    resc = np.zeros(len(alfa_), np.float64)
    wrk1 = np.zeros(len(t_), np.float64)
    wrk2 = np.zeros(len(t_), np.float64)
    ier = np.zeros(1, np.int32)
    _fourco(t_.ctypes.data, n.ctypes.data, c_.ctypes.data, alfa_.ctypes.data,
            m.ctypes.data, ress.ctypes.data, resc.ctypes.data,
            wrk1.ctypes.data, wrk2.ctypes.data, ier.ctypes.data)
    return ress, resc


@njit
def insert(t, c, k, x, iopt=0):
    """Insert one knot into a spline, FITPACK ``insert``.

    The spline itself is unchanged; only its representation gains a knot.

    Parameters
    ----------
    t : 1-D array_like of float
        Knot vector, length n.
    c : 1-D array_like of float
        B-spline coefficients (padded tck form accepted).
    k : int
        Spline degree.
    x : float
        Knot to insert; must lie in ``[t[k], t[n-k-1]]``. FITPACK flags
        anything else in ``ier``, which this wrapper does not check.
    iopt : int, optional
        0 = treat the spline as non-periodic, anything non-zero = periodic.
        Default 0.

    Returns
    -------
    tt : 1-D float64 ndarray, length n+1
        The new knot vector.
    cc : 1-D float64 ndarray, length n+1
        The new coefficients. Only the first ``len(tt) - k - 1`` are
        meaningful; the tail is padding, zero here because the buffer is
        allocated with ``np.zeros``.

    Raises
    ------
    ValueError
        If `t` or `c` has a rank other than 1.

    See Also
    --------
    scipy.interpolate.insert : The scipy routine this mirrors.

    Notes
    -----
    The padding tail of `cc` is zero here, where ``scipy.interpolate.insert``
    leaves uninitialised values there. Slice to ``len(tt) - k - 1`` before
    comparing coefficient arrays, or the tail will look like a difference.

    prange-safe: yes.
    """
    if np.asarray(t).ndim != 1 or np.asarray(c).ndim != 1:
        raise ValueError("object too deep for desired array")
    n = len(t)
    nest = n + 1
    t_ = np.zeros(nest, np.float64)
    c_ = np.zeros(nest, np.float64)
    t_[:n] = np.ascontiguousarray(np.asarray(t, np.float64))
    c_[:n] = np.ascontiguousarray(np.asarray(c, np.float64))
    n_ = np.array(n, np.int32)
    nest_ = np.array(nest, np.int32)
    k_ = np.array(k, np.int32)
    iopt_ = np.array(iopt, np.int32)
    x_ = np.array(x, np.float64)
    tt = np.zeros(nest, np.float64)
    cc = np.zeros(nest, np.float64)
    nn = np.zeros(1, np.int32)
    ier = np.zeros(1, np.int32)
    _insert(iopt_.ctypes.data, t_.ctypes.data, n_.ctypes.data, c_.ctypes.data,
            k_.ctypes.data, x_.ctypes.data, tt.ctypes.data, nn.ctypes.data,
            cc.ctypes.data, nest_.ctypes.data, ier.ctypes.data)
    return tt[:nn[0]], cc[:nn[0]]


def _is_param_c(v):
    """True when a coefficient container holds one curve per entry.

    ``scipy.interpolate``'s FITPACK layer probes ``c[0][0]`` and, when that
    succeeds, treats `c` as one spline per entry. Measured on scipy 1.18: a
    list of arrays, a tuple of arrays, an ``(idim, m)`` array and a list of
    lists all take that branch; a flat 1-D array does not.

    Takes a numba type, from an ``@overload`` chooser, or a value, from a
    Python body. In the first case the answer is a compile-time constant, so
    a caller's return type can follow it.
    """
    if isinstance(v, types.Type):
        if isinstance(v, types.Array):
            return v.ndim == 2
        if isinstance(v, (types.List, types.ListType)):
            return isinstance(v.dtype, types.Array)
        if isinstance(v, types.BaseTuple):
            return (len(v.types) > 0
                    and all(isinstance(e, types.Array) for e in v.types))
        return False
    if isinstance(v, np.ndarray):
        return v.ndim == 2
    if isinstance(v, (list, tuple)):
        return len(v) > 0 and hasattr(v[0], '__len__')
    return False


@njit
def _c_join(c, n):
    """A per-dimension coefficient container as FITPACK's flat buffer.

    `c` holds ``idim`` entries of length ``n-k-1``; the result is one array of
    length ``idim*n`` with dimension j at ``[j*n : j*n + n-k-1]`` and the
    padding zero, which is the layout `curev` and `cualde` read.
    """
    idim = len(c)
    out = np.zeros(idim * n, np.float64)
    for j in range(idim):
        cj = np.ascontiguousarray(np.asarray(c[j], np.float64))
        out[j * n: j * n + len(cj)] = cj
    return out


def _c_curve(c, n, idim):
    """A curve's coefficients as the flat stride-`n` buffer, either layout."""
    if _is_param_c(c):
        if len(c) != idim:
            raise ValueError("idim does not match the number of coefficient "
                             "arrays in c")
        return _c_join(c, n)
    return np.ascontiguousarray(np.asarray(c, np.float64))


@overload(_c_curve)
def _c_curve_ovl(c, n, idim):
    if _is_param_c(c):
        def impl(c, n, idim):
            if len(c) != idim:
                raise ValueError("idim does not match the number of "
                                 "coefficient arrays in c")
            return _c_join(c, n)
        return impl

    def impl(c, n, idim):
        return np.ascontiguousarray(np.asarray(c, np.float64))
    return impl


@njit
def curev(u, t, c, k, idim):
    """Evaluate a parametric spline curve, FITPACK ``curev``.

    Evaluates the parametric curve given by one knot vector and `idim` sets of
    coefficients at the parameter values `u`.

    Parameters
    ----------
    u : 1-D array_like of float
        Parameter values to evaluate at.
    t : 1-D array_like of float
        Knot vector in the parameter, length n.
    c : array_like of float
        Two layouts. A FLAT array of length ``idim * n`` with STRIDE n, where
        the coefficients of dimension j occupy ``c[j*n : j*n + n]`` and n is
        ``len(t)``, NOT ``n - k - 1``. Or `idim` per-dimension arrays of
        length ``n - k - 1``, as a list, a tuple or an ``(idim, m)`` array,
        which is what ``splprep`` puts in ``tck[1]``.
    k : int
        Spline degree, 1 <= k <= 5.
    idim : int
        Number of curve dimensions, 1 <= idim <= 10 (FITPACK's limit).

    Returns
    -------
    x : (m, idim) float64 ndarray
        Row i is the curve point at ``u[i]``.

    Raises
    ------
    ValueError
        If `t` or `u` has a rank other than 1, if a flat `c` has a rank other
        than 1, or if `idim` disagrees with the number of per-dimension
        arrays in `c`.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine. The nearest call
    is ``scipy.interpolate.splev(u, tck)`` on a tck from ``splprep``, which
    returns a list of per-dimension arrays, shape ``(idim, m)``: the return
    here is the transpose, one ``(m, idim)`` array.

    prange-safe: yes.
    """
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    u_ = np.ascontiguousarray(np.asarray(u, np.float64))
    if t_.ndim != 1 or u_.ndim != 1:
        raise ValueError("object too deep for desired array")
    c_ = _c_curve(c, len(t_), idim)
    if c_.ndim != 1:
        raise ValueError("object too deep for desired array")
    n = np.array(len(t_), np.int32)
    nc = np.array(len(c_), np.int32)
    m = np.array(len(u_), np.int32)
    k_ = np.array(k, np.int32)
    idim_ = np.array(idim, np.int32)
    mx = np.array(len(u_) * idim, np.int32)
    x = np.zeros(len(u_) * idim, np.float64)
    ier = np.zeros(1, np.int32)
    _curev(idim_.ctypes.data, t_.ctypes.data, n.ctypes.data, c_.ctypes.data,
           nc.ctypes.data, k_.ctypes.data, u_.ctypes.data, m.ctypes.data,
           x.ctypes.data, mx.ctypes.data, ier.ctypes.data)
    return x.reshape((len(u_), idim))


@njit
def cualde(u, t, c, k, idim):
    """All derivatives of a parametric curve at one point, FITPACK ``cualde``.

    The parametric analogue of `spalde`.

    Parameters
    ----------
    u : float
        A SCALAR parameter value, inside ``[t[k], t[n-k-1]]``.
    t : 1-D array_like of float
        Knot vector, length n.
    c : array_like of float
        The two layouts `curev` takes. A FLAT array of length ``idim * n``
        with STRIDE n = ``len(t)`` per dimension. Or `idim` per-dimension
        arrays of length ``n - k - 1``, as a list, a tuple or an
        ``(idim, m)`` array, which is what ``splprep`` puts in ``tck[1]``.
    k : int
        Spline degree.
    idim : int
        Number of curve dimensions.

    Returns
    -------
    d : (k+1, idim) float64 ndarray
        ``d[l, j]`` is the l-th derivative of coordinate j at `u`; row 0 is
        the point itself.

    Raises
    ------
    ValueError
        If `t` has a rank other than 1, if a flat `c` has a rank other than 1,
        or if `idim` disagrees with the number of per-dimension arrays in `c`.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    prange-safe: yes.
    """
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    if t_.ndim != 1:
        raise ValueError("object too deep for desired array")
    c_ = _c_curve(c, len(t_), idim)
    if c_.ndim != 1:
        raise ValueError("object too deep for desired array")
    n = np.array(len(t_), np.int32)
    nc = np.array(len(c_), np.int32)
    k1 = np.array(k + 1, np.int32)
    idim_ = np.array(idim, np.int32)
    nd = np.array((k + 1) * idim, np.int32)
    u_ = np.array(u, np.float64)
    d = np.zeros((k + 1) * idim, np.float64)
    ier = np.zeros(1, np.int32)
    _cualde(idim_.ctypes.data, t_.ctypes.data, n.ctypes.data, c_.ctypes.data,
            nc.ctypes.data, k1.ctypes.data, u_.ctypes.data, d.ctypes.data,
            nd.ctypes.data, ier.ctypes.data)
    return d.reshape((k + 1, idim))


@njit
def parder(x, y, tx, ty, c, kx, ky, nux, nuy):
    """Partial derivative of a bivariate spline on a grid, FITPACK ``parder``.

    Evaluates the mixed partial derivative on the full cross product of two
    axes.

    Parameters
    ----------
    x : 1-D array_like of float
        Grid abscissae, ascending.
    y : 1-D array_like of float
        Grid ordinates, ascending. The derivative is evaluated on the FULL
        CROSS PRODUCT ``x`` x ``y`` -- use `pardeu` for scattered points.
    tx : 1-D array_like of float
        Knots in x, length nx.
    ty : 1-D array_like of float
        Knots in y, length ny.
    c : 1-D array_like of float
        Bivariate coefficients in FITPACK's flat layout
        ``c[(ny-ky-1)*i + j]``, which equals ``np.outer(cx, cy).ravel()``.
    kx, ky : int
        Degrees in x and y, 1 <= k <= 5.
    nux : int
        Derivative order in x, ``0 <= nux < kx``.
    nuy : int
        Derivative order in y, ``0 <= nuy < ky``.

    Returns
    -------
    z : (len(x), len(y)) float64 ndarray
        The mixed partial derivative on the grid.

    Raises
    ------
    ValueError
        If `tx`, `ty`, `c`, `x` or `y` has a rank other than 1; if
        ``len(c) != (nx-kx-1)*(ny-ky-1)``, a test that is an equality, so a
        `c` padded to ``nx*ny`` is rejected as well as a short one; if `nux`
        is outside ``0 <= nux < kx``; or if `nuy` is outside
        ``0 <= nuy < ky``, the range FITPACK requires. Tested in that order.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine. It reaches the
    same computation through ``bisplev(x, y, tck, dx, dy)`` and
    ``RectBivariateSpline.__call__(x, y, dx=, dy=)``.

    Workspace sizes come from the FITPACK docstring:
    ``lwrk = mx*(kx+1-nux) + my*(ky+1-nuy) + (nx-kx-1)*(ny-ky-1)`` and
    ``kwrk = mx + my``. The integer workspace MUST be int32; float64 there
    gives garbage or a segfault.

    prange-safe: yes.
    """
    tx_ = np.ascontiguousarray(np.asarray(tx, np.float64))
    ty_ = np.ascontiguousarray(np.asarray(ty, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    if (tx_.ndim != 1 or ty_.ndim != 1 or c_.ndim != 1 or x_.ndim != 1
            or y_.ndim != 1):
        raise ValueError("object too deep for desired array")
    if len(c_) != (len(tx_) - kx - 1) * (len(ty_) - ky - 1):
        raise ValueError("c must have length (nx-kx-1)*(ny-ky-1)")
    if nux < 0 or nux >= kx:
        raise ValueError("0 <= nux < kx must hold")
    if nuy < 0 or nuy >= ky:
        raise ValueError("0 <= nuy < ky must hold")
    # ponytail: parder rebuilds the derivative coefficients over a len(c)
    # workspace on every call, exactly as pardeu does below. bispev is the
    # no-derivative grid evaluator and agrees exactly. See pardeu for the
    # crossover measurement behind the 1/256.
    #
    # The guard sits above the int32 boxes because bispev builds its own and
    # uses none of these, so on the fast path all eight are dead allocations.
    if nux == 0 and nuy == 0 and (len(x_) * len(y_) * 256
                                  <= (len(tx_) - kx - 1) * (len(ty_) - ky - 1)):
        return bispev(x_, y_, tx_, ty_, c_, kx, ky)
    nx = np.array(len(tx_), np.int32)
    ny = np.array(len(ty_), np.int32)
    mx = np.array(len(x_), np.int32)
    my = np.array(len(y_), np.int32)
    kx_ = np.array(kx, np.int32)
    ky_ = np.array(ky, np.int32)
    nux_ = np.array(nux, np.int32)
    nuy_ = np.array(nuy, np.int32)
    lw = len(x_) * (kx + 1 - nux) + len(y_) * (ky + 1 - nuy) \
        + (len(tx_) - kx - 1) * (len(ty_) - ky - 1)
    kw = len(x_) + len(y_)
    lwrk = np.array(lw, np.int32)
    kwrk = np.array(kw, np.int32)
    wrk = np.zeros(lw, np.float64)
    iwrk = np.zeros(kw, np.int32)
    z = np.zeros(len(x_) * len(y_), np.float64)
    ier = np.zeros(1, np.int32)
    _parder(tx_.ctypes.data, nx.ctypes.data, ty_.ctypes.data, ny.ctypes.data,
            c_.ctypes.data, kx_.ctypes.data, ky_.ctypes.data,
            nux_.ctypes.data, nuy_.ctypes.data, x_.ctypes.data,
            mx.ctypes.data, y_.ctypes.data, my.ctypes.data, z.ctypes.data,
            wrk.ctypes.data, lwrk.ctypes.data, iwrk.ctypes.data,
            kwrk.ctypes.data, ier.ctypes.data)
    return z.reshape((len(x_), len(y_)))


@njit
def pardeu(x, y, tx, ty, c, kx, ky, nux, nuy):
    """Partial derivative at scattered points, FITPACK ``pardeu``.

    The scattered-point sibling of `parder`, behind
    ``RectBivariateSpline.ev(x, y, dx=, dy=)``.

    Parameters
    ----------
    x : 1-D array_like of float
        Abscissae of the query points.
    y : 1-D array_like of float
        Ordinates, SAME LENGTH as `x` -- the points are the pairs
        ``(x[i], y[i])``, not a grid.
    tx : 1-D array_like of float
        Knots in x, length nx.
    ty : 1-D array_like of float
        Knots in y, length ny.
    c : 1-D array_like of float
        Coefficients in the flat layout ``c[(ny-ky-1)*i + j]``.
    kx, ky : int
        Degrees in x and y.
    nux : int
        Derivative order in x, ``0 <= nux < kx``.
    nuy : int
        Derivative order in y, ``0 <= nuy < ky``.

    Returns
    -------
    z : 1-D float64 ndarray, length ``len(x)``
        The derivative at each point.

    Raises
    ------
    ValueError
        If `tx`, `ty`, `c`, `x` or `y` has a rank other than 1, or
        ``len(c) != (nx-kx-1)*(ny-ky-1)``, or `x` and `y` have different
        lengths, or `nux` is outside ``0 <= nux < kx``, or `nuy` is outside
        ``0 <= nuy < ky``. Tested in that order.

    Notes
    -----
    prange-safe: yes.
    """
    tx_ = np.ascontiguousarray(np.asarray(tx, np.float64))
    ty_ = np.ascontiguousarray(np.asarray(ty, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    if (tx_.ndim != 1 or ty_.ndim != 1 or c_.ndim != 1 or x_.ndim != 1
            or y_.ndim != 1):
        raise ValueError("object too deep for desired array")
    if len(c_) != (len(tx_) - kx - 1) * (len(ty_) - ky - 1):
        raise ValueError("Invalid c array dimensions")
    if len(y_) != len(x_):
        raise ValueError("x and y arrays must have same length")
    if nux < 0 or nux >= kx:
        raise ValueError("0 <= nux < kx must hold")
    if nuy < 0 or nuy >= ky:
        raise ValueError("0 <= nuy < ky must hold")
    nc = (len(tx_) - kx - 1) * (len(ty_) - ky - 1)
    # ponytail: FITPACK's pardeu rebuilds the derivative coefficients over a
    # workspace of len(c) on EVERY call, so one point against a 200x500 table
    # costs 28 us where bispeu costs 935 ns. With no derivative asked for the
    # two agree exactly (measured 0.000e+00). bispeu only wins for few points:
    # pardeu is ~0.28 ns/coefficient once plus ~230 ns/point, bispeu ~289
    # ns/point flat, so the crossover is m ~ nc/211. 1/256 is the conservative
    # round number; raise it if a batch profile asks for it.
    if nux == 0 and nuy == 0 and len(x_) * 256 <= nc:
        return bispeu(x_, y_, tx_, ty_, c_, kx, ky)
    nx = np.array(len(tx_), np.int32)
    ny = np.array(len(ty_), np.int32)
    m = np.array(len(x_), np.int32)
    kx_ = np.array(kx, np.int32)
    ky_ = np.array(ky, np.int32)
    nux_ = np.array(nux, np.int32)
    nuy_ = np.array(nuy, np.int32)
    lw = nc + (kx + 1 - nux) * len(x_) + (ky + 1 - nuy) * len(x_)
    kw = 2 * len(x_)
    lwrk = np.array(lw, np.int32)
    kwrk = np.array(kw, np.int32)
    wrk = np.zeros(lw, np.float64)
    iwrk = np.zeros(kw, np.int32)
    z = np.zeros(len(x_), np.float64)
    ier = np.zeros(1, np.int32)
    _pardeu(tx_.ctypes.data, nx.ctypes.data, ty_.ctypes.data, ny.ctypes.data,
            c_.ctypes.data, kx_.ctypes.data, ky_.ctypes.data,
            nux_.ctypes.data, nuy_.ctypes.data, x_.ctypes.data,
            y_.ctypes.data, z.ctypes.data, m.ctypes.data, wrk.ctypes.data,
            lwrk.ctypes.data, iwrk.ctypes.data, kwrk.ctypes.data,
            ier.ctypes.data)
    return z


@njit
def dblint(tx, ty, c, kx, ky, xb, xe, yb, ye):
    """Double integral of a bivariate spline, FITPACK ``dblint``.

    The routine behind ``RectBivariateSpline.integral(xb, xe, yb, ye)``.

    Parameters
    ----------
    tx : 1-D array_like of float
        Knots in x, length nx.
    ty : 1-D array_like of float
        Knots in y, length ny.
    c : 1-D array_like of float
        Coefficients in the flat layout ``c[(ny-ky-1)*i + j]``.
    kx, ky : int
        Degrees in x and y.
    xb, xe : float
        Integration limits in x.
    yb, ye : float
        Integration limits in y.

    Returns
    -------
    res : float
        The integral of s(x, y) over the rectangle
        ``[xb, xe] x [yb, ye]``.

    Raises
    ------
    ValueError
        If `tx`, `ty` or `c` has a rank other than 1.

    Notes
    -----
    Another Dierckx FUNCTION wrapped as a subroutine with a ``res``
    out-argument.

    prange-safe: yes.
    """
    tx_ = np.ascontiguousarray(np.asarray(tx, np.float64))
    ty_ = np.ascontiguousarray(np.asarray(ty, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    if tx_.ndim != 1 or ty_.ndim != 1 or c_.ndim != 1:
        raise ValueError("object too deep for desired array")
    nx = np.array(len(tx_), np.int32)
    ny = np.array(len(ty_), np.int32)
    kx_ = np.array(kx, np.int32)
    ky_ = np.array(ky, np.int32)
    xb_ = np.array(xb, np.float64)
    xe_ = np.array(xe, np.float64)
    yb_ = np.array(yb, np.float64)
    ye_ = np.array(ye, np.float64)
    wrk = np.zeros(len(tx_) + len(ty_) - kx - ky - 2, np.float64)
    res = np.zeros(1, np.float64)
    _dblint(tx_.ctypes.data, nx.ctypes.data, ty_.ctypes.data, ny.ctypes.data,
            c_.ctypes.data, kx_.ctypes.data, ky_.ctypes.data,
            xb_.ctypes.data, xe_.ctypes.data, yb_.ctypes.data,
            ye_.ctypes.data, wrk.ctypes.data, res.ctypes.data)
    return res[0]


@njit
def profil(tx, ty, c, kx, ky, u, iopt=0):
    """Univariate profile of a bivariate spline, FITPACK ``profil``.

    Returns the 1-D spline obtained by fixing one variable.

    Parameters
    ----------
    tx : 1-D array_like of float
        Knots in x, length nx.
    ty : 1-D array_like of float
        Knots in y, length ny.
    c : 1-D array_like of float
        Coefficients in the flat layout ``c[(ny-ky-1)*i + j]``.
    kx, ky : int
        Degrees in x and y.
    u : float
        The value at which the other variable is held fixed.
    iopt : int, optional
        Which variable to fix: 0 = fix x at `u`, giving ``f(y) = s(u, y)`` on
        the knots `ty` with degree `ky`; 1 = fix y at `u`, giving
        ``g(x) = s(x, u)`` on the knots `tx` with degree `kx`. Default 0.

    Returns
    -------
    cu : 1-D float64 ndarray
        B-spline coefficients of the profile, of length ``len(ty)`` when
        ``iopt == 0`` and ``len(tx)`` when ``iopt == 1``. Feed it back to
        `splev` together with the matching knot vector and degree -- the
        return value is the coefficients ONLY, not a tck.

    Raises
    ------
    ValueError
        If `tx`, `ty` or `c` has a rank other than 1.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    prange-safe: yes.
    """
    tx_ = np.ascontiguousarray(np.asarray(tx, np.float64))
    ty_ = np.ascontiguousarray(np.asarray(ty, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    if tx_.ndim != 1 or ty_.ndim != 1 or c_.ndim != 1:
        raise ValueError("object too deep for desired array")
    nx = np.array(len(tx_), np.int32)
    ny = np.array(len(ty_), np.int32)
    kx_ = np.array(kx, np.int32)
    ky_ = np.array(ky, np.int32)
    iopt_ = np.array(iopt, np.int32)
    u_ = np.array(u, np.float64)
    if iopt == 0:
        nu = len(ty_)
    else:
        nu = len(tx_)
    nu_ = np.array(nu, np.int32)
    cu = np.zeros(nu, np.float64)
    ier = np.zeros(1, np.int32)
    _profil(iopt_.ctypes.data, tx_.ctypes.data, nx.ctypes.data,
            ty_.ctypes.data, ny.ctypes.data, c_.ctypes.data,
            kx_.ctypes.data, ky_.ctypes.data, u_.ctypes.data,
            nu_.ctypes.data, cu.ctypes.data, ier.ctypes.data)
    return cu


@njit
def surev(u, v, tu, tv, c, idim=1):
    """Evaluate a parametric spline surface on a grid, FITPACK ``surev``.

    The evaluator for the surfaces built by ``parsur``. BICUBIC ONLY:
    FITPACK's ``surev`` assumes degree 3 in both parameters, which is why
    there are no `ku` / `kv` arguments.

    Parameters
    ----------
    u : 1-D array_like of float
        First parameter values, ascending.
    v : 1-D array_like of float
        Second parameter values, ascending. The surface is evaluated on the
        full cross product ``u`` x ``v``.
    tu : 1-D array_like of float
        Knots in u, length nu.
    tv : 1-D array_like of float
        Knots in v, length nv.
    c : 1-D array_like of float
        Coefficients for all `idim` dimensions, in FITPACK's ``parsur``
        layout.
    idim : int, optional
        Number of surface dimensions, 1 <= idim <= 3 (FITPACK's limit for
        this routine). Default 1.

    Returns
    -------
    f : (idim, len(u), len(v)) float64 ndarray
        ``f[j]`` is the grid of coordinate j. Note the dimension index comes
        FIRST here, unlike `curev`, where it comes last.

    Raises
    ------
    ValueError
        If `tu`, `tv`, `c`, `u` or `v` has a rank other than 1.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    Workspace: ``lwrk = 4*(mu + mv)``, ``kwrk = mu + mv``; the integer
    workspace must be int32.

    For ``idim == 1`` on a bicubic tck the result is `bispev`'s grid.

    prange-safe: yes.
    """
    tu_ = np.ascontiguousarray(np.asarray(tu, np.float64))
    tv_ = np.ascontiguousarray(np.asarray(tv, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    u_ = np.ascontiguousarray(np.asarray(u, np.float64))
    v_ = np.ascontiguousarray(np.asarray(v, np.float64))
    if (tu_.ndim != 1 or tv_.ndim != 1 or c_.ndim != 1 or u_.ndim != 1
            or v_.ndim != 1):
        raise ValueError("object too deep for desired array")
    nu = np.array(len(tu_), np.int32)
    nv = np.array(len(tv_), np.int32)
    mu = np.array(len(u_), np.int32)
    mv = np.array(len(v_), np.int32)
    idim_ = np.array(idim, np.int32)
    mf_len = len(u_) * len(v_) * idim
    mf = np.array(mf_len, np.int32)
    lw = 4 * (len(u_) + len(v_))
    kw = len(u_) + len(v_)
    lwrk = np.array(lw, np.int32)
    kwrk = np.array(kw, np.int32)
    f = np.zeros(mf_len, np.float64)
    wrk = np.zeros(lw, np.float64)
    iwrk = np.zeros(kw, np.int32)
    ier = np.zeros(1, np.int32)
    _surev(idim_.ctypes.data, tu_.ctypes.data, nu.ctypes.data,
           tv_.ctypes.data, nv.ctypes.data, c_.ctypes.data, u_.ctypes.data,
           mu.ctypes.data, v_.ctypes.data, mv.ctypes.data, f.ctypes.data,
           mf.ctypes.data, wrk.ctypes.data, lwrk.ctypes.data,
           iwrk.ctypes.data, kwrk.ctypes.data, ier.ctypes.data)
    return f.reshape((idim, len(u_), len(v_)))


# ---------------- bivariate spline evaluation ----------------

_bispev = _sig(_lib.bispev_wrapper, 17)
_bispeu = _sig(_lib.bispeu_wrapper, 14)


@njit
def bispev(x, y, tx, ty, c, kx, ky):
    """Evaluate a bivariate spline on a grid, FITPACK ``bispev``.

    The evaluator that backs bivariate spline evaluation on the full cross
    product of two axes.

    Parameters
    ----------
    x : 1-D array_like of float
        Grid abscissae, ascending.
    y : 1-D array_like of float
        Grid ordinates, ascending. The spline is evaluated on the full CROSS
        PRODUCT ``x`` x ``y`` -- use `bispeu` for scattered points.
    tx : 1-D array_like of float
        Knots in x, length nx.
    ty : 1-D array_like of float
        Knots in y, length ny.
    c : 1-D array_like of float, length ``(nx-kx-1)*(ny-ky-1)``
        Coefficients in FITPACK's flat layout ``c[(ny-ky-1)*i + j]``, equal to
        ``np.outer(cx, cy).ravel()``.
    kx : int
        Degree in x, 1 <= kx <= 5.
    ky : int
        Degree in y, 1 <= ky <= 5.

    Returns
    -------
    z : (len(x), len(y)) float64 ndarray
        Spline values on the grid.

    Raises
    ------
    ValueError
        If `tx`, `ty`, `c`, `x` or `y` has a rank other than 1, or
        ``len(c) != (nx-kx-1)*(ny-ky-1)``. The coefficient test is an
        equality, so a `c` padded to ``nx*ny`` is rejected as well as a short
        one. Tested in that order.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine. It reaches the
    same computation only through ``bisplev(x, y, tck)`` and
    ``RectBivariateSpline.__call__``.

    Workspace: ``lwrk = mx*(kx+1) + my*(ky+1)``, ``kwrk = mx + my``; the
    integer workspace must be int32.

    Points outside the knot range are extrapolated. There is no ``e``/``ext``
    flag on this routine, unlike `splev`.

    prange-safe: yes.

    Examples
    --------
    Evaluate a bilinear spline for ``f(x, y) = x + 2*y`` at the grid centre,
    from inside ``@njit``:

    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import bispev
    >>> @njit
    ... def go():
    ...     tx = np.array([0., 0., 1., 1.])
    ...     ty = np.array([0., 0., 1., 1.])
    ...     c = np.array([0., 2., 1., 3.])  # corner values of x + 2*y
    ...     return bispev(np.array([0.5]), np.array([0.5]), tx, ty, c, 1, 1)
    >>> go()
    array([[1.5]])
    """
    tx_ = np.ascontiguousarray(np.asarray(tx, np.float64))
    ty_ = np.ascontiguousarray(np.asarray(ty, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    if (tx_.ndim != 1 or ty_.ndim != 1 or c_.ndim != 1 or x_.ndim != 1
            or y_.ndim != 1):
        raise ValueError("object too deep for desired array")
    if len(c_) != (len(tx_) - kx - 1) * (len(ty_) - ky - 1):
        raise ValueError("c must have length (nx-kx-1)*(ny-ky-1)")
    nx = np.array(len(tx_), np.int32)
    ny = np.array(len(ty_), np.int32)
    mx = np.array(len(x_), np.int32)
    my = np.array(len(y_), np.int32)
    kx_ = np.array(kx, np.int32)
    ky_ = np.array(ky, np.int32)
    lw = len(x_) * (kx + 1) + len(y_) * (ky + 1)
    kw = len(x_) + len(y_)
    lwrk = np.array(lw, np.int32)
    kwrk = np.array(kw, np.int32)
    z = np.zeros(len(x_) * len(y_), np.float64)
    wrk = np.zeros(lw, np.float64)
    iwrk = np.zeros(kw, np.int32)
    ier = np.zeros(1, np.int32)
    _bispev(tx_.ctypes.data, nx.ctypes.data, ty_.ctypes.data, ny.ctypes.data,
            c_.ctypes.data, kx_.ctypes.data, ky_.ctypes.data, x_.ctypes.data,
            mx.ctypes.data, y_.ctypes.data, my.ctypes.data, z.ctypes.data,
            wrk.ctypes.data, lwrk.ctypes.data, iwrk.ctypes.data,
            kwrk.ctypes.data, ier.ctypes.data)
    return z.reshape((len(x_), len(y_)))


@njit
def bispeu(x, y, tx, ty, c, kx, ky):
    """Evaluate a bivariate spline at scattered points, FITPACK ``bispeu``.

    The scattered-point sibling of `bispev`: it takes the query points as the
    pairs ``(x[i], y[i])`` rather than a grid.

    Parameters
    ----------
    x : 1-D array_like of float
        Abscissae of the query points.
    y : 1-D array_like of float
        Ordinates, SAME LENGTH as `x` -- the points are the pairs
        ``(x[i], y[i])``, not a grid. This is the difference from `bispev`.
    tx : 1-D array_like of float
        Knots in x, length nx.
    ty : 1-D array_like of float
        Knots in y, length ny.
    c : 1-D array_like of float
        Coefficients in the flat layout ``c[(ny-ky-1)*i + j]``.
    kx : int
        Degree in x, 1 <= kx <= 5.
    ky : int
        Degree in y, 1 <= ky <= 5.

    Returns
    -------
    z : 1-D float64 ndarray, length ``len(x)``
        Spline value at each point.

    Raises
    ------
    ValueError
        If `tx`, `ty`, `c`, `x` or `y` has a rank other than 1, if `x` and
        `y` have different
        lengths, or if ``len(c) != (nx-kx-1)*(ny-ky-1)``. The coefficient
        test is an equality, so a padded `c` is rejected as well as a short
        one. Tested in that order.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine. It reaches the
    same computation only through ``RectBivariateSpline.ev(x, y)``.

    Workspace is only ``kx + ky + 2`` doubles: this routine walks the points
    one at a time, where `bispev` allocates ``mx*(kx+1) + my*(ky+1)``.

    prange-safe: yes.

    Examples
    --------
    Evaluate a bilinear spline for ``f(x, y) = x + 2*y`` at two scattered
    points, from inside ``@njit``:

    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import bispeu
    >>> @njit
    ... def go():
    ...     tx = np.array([0., 0., 1., 1.])
    ...     ty = np.array([0., 0., 1., 1.])
    ...     c = np.array([0., 2., 1., 3.])  # corner values of x + 2*y
    ...     return bispeu(np.array([0.25, 0.75]), np.array([0.5, 0.5]),
    ...                   tx, ty, c, 1, 1)
    >>> go()
    array([1.25, 1.75])
    """
    tx_ = np.ascontiguousarray(np.asarray(tx, np.float64))
    ty_ = np.ascontiguousarray(np.asarray(ty, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    if (tx_.ndim != 1 or ty_.ndim != 1 or c_.ndim != 1 or x_.ndim != 1
            or y_.ndim != 1):
        raise ValueError("object too deep for desired array")
    if len(y_) != len(x_):
        raise ValueError("x and y must have same length")
    if len(c_) != (len(tx_) - kx - 1) * (len(ty_) - ky - 1):
        raise ValueError("c must have length (nx-kx-1)*(ny-ky-1)")
    nx = np.array(len(tx_), np.int32)
    ny = np.array(len(ty_), np.int32)
    m = np.array(len(x_), np.int32)
    kx_ = np.array(kx, np.int32)
    ky_ = np.array(ky, np.int32)
    lw = kx + ky + 2
    lwrk = np.array(lw, np.int32)
    z = np.zeros(len(x_), np.float64)
    wrk = np.zeros(lw, np.float64)
    ier = np.zeros(1, np.int32)
    _bispeu(tx_.ctypes.data, nx.ctypes.data, ty_.ctypes.data, ny.ctypes.data,
            c_.ctypes.data, kx_.ctypes.data, ky_.ctypes.data, x_.ctypes.data,
            y_.ctypes.data, z.ctypes.data, m.ctypes.data, wrk.ctypes.data,
            lwrk.ctypes.data, ier.ctypes.data)
    return z


# Public names: the raw FITPACK evaluator routines, argument-for-argument
# as Dierckx defines them. The scipy-shaped wrappers live in
# scijit.interpolate itself.
__all__ = [
    'splev', 'splder', 'splint', 'spalde', 'sproot', 'fourco', 'insert',
    'curev', 'cualde', 'parder', 'pardeu', 'dblint', 'profil', 'surev',
    'bispev', 'bispeu',
]
