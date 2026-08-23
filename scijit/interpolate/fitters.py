"""Numba-callable wrappers for the FITPACK fitting routines.

Companion to evaluators.py. Exposes the "fitting group" via the
bind(c) wrappers in fitpack/wrappers.f90:

    curfit, curfit_lsq, percur          (univariate smoothing / periodic)
    parcur, clocur, concur              (parametric / closed / constrained)
    cocosp, concon                      (convexity-constrained)
    regrid, parsur, pogrid, spgrid      (gridded surfaces)
    sphere, surfit                      (scattered-data surfaces)
    polar, evapol                       (polar domains, rad() callback)

All workspace sizes (nest, lwrk, kwrk, ...) are computed automatically from
the "always large enough" formulas in the FITPACK docstrings, so callers
only supply data.  Every function is
@njit-compiled and usable inside other @njit code.

Outputs are trimmed to their actual size and use the same layouts the
evaluators expect, so results feed straight into splev/curev/bispev/...:
    univariate   : (t[:n], c[:n], fp, ier)
    curve (idim) : c[:idim*n] with dimension j starting at c[j*n]
    bivariate    : c[:(nx-kx-1)*(ny-ky-1)]   (row = x index)

The polar/evapol callback: pass a plain ``@njit`` function ``rad(v) -> float``
giving the boundary radius r(v):

    from numba import njit
    @njit
    def rad(v):
        return 1.0                      # unit-disc boundary r(v)
    tu, tv, c, u, v, fp, ier = polar(x, y, z, w, rad, s)

THE ``ier`` RETURN IS NOT RAISED ON. Every fitter hands FITPACK's status
back as its last return value instead of throwing, so a caller that ignores
it can silently use a failed fit. Dierckx's convention, shared by all of
these:
    ier =  0  normal return, the requested smoothing was achieved
    ier = -1  normal, but the result is the INTERPOLATING spline (fp = 0);
              this is what s=0 gives, so it is expected, not an error
    ier = -2  normal, but the result is the weighted least-squares
              polynomial (fp = the upper bound on s); the request was for more
              smoothing than the data supports
    ier =  1  the workspace estimate (nest / nxest / nuest ...) was too
              small
    ier =  2,3  the smoothing parameter could not be reached in the
              allowed number of iterations
    ier = 10  invalid input (the checks in Dierckx's block A failed)
Treat ier in (0, -1, -2) as success and anything positive as failure.

``tol`` AND ``maxit`` ARE ARGUMENTS HERE. The twelve smoothing fitters
(`curfit` `percur` `parcur` `clocur` `concur` `regrid` `parsur` `pogrid`
`spgrid` `sphere` `surfit` `polar`) take both as trailing keywords. ``tol``
is the width of the band a fit is accepted in, ``abs(fp - s) <= tol*s``;
``maxit`` caps the knot-search iterations and returns ``ier = 3`` when it is
reached. The defaults are sentinels, ``tol=-1.0`` and ``maxit=-1``, which
resolve to the values the Fortran sources set internally, so a call that
leaves them alone runs the computation it ran before. This forks the
vendored Dierckx sources: the two constants are dummy arguments in
``src/fitpack/*.f`` rather than locals.

Relation to scipy: scipy.interpolate publishes NONE of these 17 names. It
reaches the same Fortran through `splrep`, `splprep` and the spline classes,
and keeps the fitters themselves private. So there is no scipy call to compare
a signature against, and each docstring states what the routine does rather
than how it differs.

A RANK-DEFICIENCY ``UserWarning`` IS ISSUED HERE AND NOWHERE IN SCIPY.
`curfit` and `parcur` check the fitted knot set against the
Schoenberg-Whitney condition and warn when a B-spline coefficient is not
determined by the data; `curfit_lsq`, `percur`, `clocur` and the surface
fitters do not run the check, so the same condition passes unreported there.
Under ``-W error`` a fit that scipy completes silently can therefore stop
here.

``polar`` and ``evapol`` take the ``rad()`` boundary as a plain ``@njit``
function ``rad(v) -> float``. scipy publishes no name for either routine.

A SMOOTHING fit can land somewhere ``RectBivariateSpline`` does not, and
`regrid` is where it shows. scipy 1.18 replaced the f2py FITPACK bindings
with a C translation whose smoothing iteration rounds differently, so on
sensitive data the knot search lands in different POSITIONS while returning
the same knot COUNT. Measured over four random 13x14 grids at s = 1e-6,
1e-4, 1e-2 and 1.0: `regrid` against ``RectBivariateSpline`` ranges from
exactly 0.0 up to 2.854e+02 in grid values and 1.215e+03 in coefficients,
with both sides reporting ``ier = 0`` and both inside FITPACK's own
``|fp-s|/s <= 1e-3`` convergence test. Neither answer is wrong; they are two
valid smoothing splines.

Those figures describe their fixtures. Do not read the largest as a bound:
the gap is set by how sensitive the data makes the knot search, and a
fixture worse than any tried here is possible.

One further consequence: at ``s`` where the searches diverge, one library
can converge and the other give up. On one 13x14 fixture at ``s = 0.01``
`regrid` returned ``ier = 0`` with ``fp`` equal to ``s`` while scipy's
``RectBivariateSpline`` raised ``ValueError`` reporting that ``maxit = 20``
had been reached.

Array arguments are read as flat buffers of a length the other arguments
fix, so a rank other than 1 raises ``ValueError`` with the text FITPACK's
own bindings use, ``object too deep for desired array``. The value array of
a gridded fit is the exception: it is sized from the product of the grid
lengths, so ``regrid``, ``pogrid``, ``spgrid`` and ``parsur`` test that
product instead and take the grid shape as well as the flat one.

prange-safety: the twelve routines with no callback are pure in-out
Fortran with fresh local workspaces, so they are prange-safe. ``polar``
and ``evapol`` are prange-safe too, provided the ``rad`` function supplied is
reentrant.
"""
import ctypes as ct
import warnings
from numba import njit, objmode, cfunc, types
from numba.core.errors import TypingError
from numba.extending import overload
import numpy as np
import os
import platform

rootdir = os.path.dirname(os.path.abspath(__file__))

if platform.uname()[0] == "Windows":
    _name = "\\libfitpack.dll"
elif platform.uname()[0] == "Linux":
    _name = "/libfitpack.so"
else:
    _name = "/libfitpack.dylib"

_lib = ct.CDLL(rootdir + _name)


def _sig(fn, nargs):
    """Give a ctypes handle the all-pointer signature the wrappers use.

    Every ``*_wrapper`` in ``src/fitpack/wrappers.f90`` takes its arguments by
    Fortran reference and returns nothing.

    Parameters
    ----------
    fn : ctypes function handle
        Symbol from the loaded shared library.
    nargs : int
        Argument count of the Fortran wrapper. A wrong count surfaces as a
        cryptic numba ``ExternalFunctionPointer`` error -- recount against
        ``wrappers.f90``.

    Returns
    -------
    fn : ctypes function handle
        The same object, mutated in place.
    """
    fn.argtypes = [ct.c_void_p] * nargs
    fn.restype = None
    return fn


_curfit = _sig(_lib.curfit_wrapper, 20)
_percur = _sig(_lib.percur_wrapper, 18)
_parcur = _sig(_lib.parcur_wrapper, 24)
_clocur = _sig(_lib.clocur_wrapper, 22)
_concur = _sig(_lib.concur_wrapper, 30)
_cocosp = _sig(_lib.cocosp_wrapper, 18)
_concon = _sig(_lib.concon_wrapper, 21)
_regrid = _sig(_lib.regrid_wrapper, 28)
_parsur = _sig(_lib.parsur_wrapper, 24)
_pogrid = _sig(_lib.pogrid_wrapper, 25)
_spgrid = _sig(_lib.spgrid_wrapper, 25)
_sphere = _sig(_lib.sphere_wrapper, 25)
_surfit = _sig(_lib.surfit_wrapper, 32)
_polar  = _sig(_lib.polar_wrapper, 28)
_evapol = _sig(_lib.evapol_wrapper, 9)

# "argument not supplied": a zero-length float64 array. No FITPACK array
# argument is ever legitimately empty, so length 0 is unambiguous.
#
# THE REASON RECORDED HERE WAS FALSE and is corrected 2026-08-02. It read
# "numba cannot type None in an array-argument slot". Measured on numba 0.66:
# a plain @njit function declared `w=None` accepts omitted, None and an array,
# from Python and from inside @njit; numba prunes the dead branch. The
# sentinel is kept because moving these signatures to `None` is one
# coordinated pass over the whole package (fixup item F10), not because numba
# refuses `None`.
_NO_ARR = np.empty(0, dtype=np.float64)

# FITPACK's own tuning constants, which the twelve smoothing fitters used to
# set as locals and now take as trailing arguments. TWO DIFFERENT NUMBERS:
# `curfit.f` wrote `tol = 0.1d-02`, the other eleven wrote `tol = 0.1e-02`,
# and in fixed-form Fortran `0.1e-02` is a default REAL literal, so it reaches
# `real*8 tol` as the widened float32. gfortran prints the pair as
# 1.00000000000000002E-03 and 1.00000004749745131E-03. Each routine keeps the
# number its own source held.
_TOL_D = 1e-3                            # curfit.f's 0.1d-02
_TOL_E = float(np.float32(1e-3))         # the other eleven, 0.1e-02
_MAXIT = 20


@njit
def _resolve_tol(tol, default):
    """Resolve the ``tol`` sentinel and reject the values FITPACK cannot use.

    Parameters
    ----------
    tol : float
        Caller's value. ``-1.0`` selects `default`.
    default : float
        The literal the routine's own Fortran source held.

    Returns
    -------
    tol : float
        `default` for the sentinel, otherwise `tol` unchanged.

    Raises
    ------
    ValueError
        ``tol <= 0`` and not the sentinel. ``acc = tol*s`` is a width, and a
        non-positive width accepts nothing.
    """
    if tol == -1.0:
        return default
    if tol <= 0.0:
        raise ValueError("tol must be > 0, or -1.0 for FITPACK's own value")
    return tol


@njit
def _resolve_maxit(maxit):
    """Resolve the ``maxit`` sentinel and reject values below zero.

    Parameters
    ----------
    maxit : int
        Caller's value. ``-1`` selects FITPACK's 20.

    Returns
    -------
    maxit : int
        20 for the sentinel, otherwise `maxit` unchanged.

    Raises
    ------
    ValueError
        ``maxit < 0``, or ``maxit == -1`` is not meant as the sentinel. The
        two negative spellings cannot be told apart, so ``-1`` is the
        sentinel and anything below it is refused.

    Notes
    -----
    ``maxit = 0`` is accepted and runs no knot-placement iteration at all.
    FITPACK's worker loops ``do 350 iter = 1, maxit``, so a zero cap skips the
    body and control falls through to ``ier = 3``. The caller decides whether
    that status is meaningful: with ``s = 0`` no iteration was needed and the
    flag is spurious, while for a smoothing fit it says the knot search never
    ran. `RectBivariateSpline` makes that distinction; see its `maxit`.
    """
    if maxit == -1:
        return _MAXIT
    if maxit < 0:
        raise ValueError("maxit must be >= 0, or -1 for FITPACK's own value")
    return maxit


# ---------------------------------------------------------------------
# rank check on the fitted knot set
# ---------------------------------------------------------------------
#
# The Schoenberg-Whitney condition: every B-spline basis function must have at
# least one data point strictly inside its support, or the least-squares
# matrix is rank deficient and the coefficient of the unsupported basis
# function is not determined by the data. The spline still reproduces the data
# points, so `fp` and `ier` look healthy, while the curve between them carries
# an arbitrary component.
#
# FITPACK's knot search can produce such a set: `fpknot` skips an interval
# holding NO data point but will split one holding exactly ONE, and that
# leaves an empty half. Measured on this build, 8 seeds each, noisy data:
# `curfit` at s=0.05, `parcur` at s=0.05, and `regrid` at s=0.01 and s=0.05
# all violate it, always at the interval next to a boundary.
#
# The scipy half of that was re-measured 2026-08-09 rather than inferred:
# 40 noisy points, s=0.05, seeds 0..7, the same fit through
# `scipy.interpolate.splrep`. On seeds 0 and 7 BOTH libraries return a knot
# set with one unsupported basis function; scipy issued no warning and raised
# nothing on either. That is why the check lives here.

@njit
def _sw_count(t, x, k):
    """Basis functions with no data point strictly inside their support."""
    n = len(t) - k - 1
    bad = 0
    for i in range(n):
        lo = t[i]
        hi = t[i + k + 1]
        found = False
        for j in range(len(x)):
            if x[j] > lo and x[j] < hi:
                found = True
                break
        if not found:
            bad += 1
    return bad


@njit
def _warn_rank(nbad, which):
    """Warn that the fit is rank deficient. `which`: 0 none, 1 x, 2 y."""
    with objmode():
        where = "" if which == 0 else (" in x" if which == 1 else " in y")
        warnings.warn(
            "the fitted spline is rank deficient: %d B-spline coefficient(s)%s"
            " are not determined by the data, because the knot search left a"
            " basis function with no data point in its support"
            " (Schoenberg-Whitney). The spline still reproduces the data"
            " points; between them it carries an arbitrary component."
            " A larger s, or fewer knots, avoids it." % (nbad, where),
            UserWarning)


@njit
def _check_rank(t, x, k, which):
    """Run the Schoenberg-Whitney check and warn if it fails."""
    nbad = _sw_count(t, x, k)
    if nbad > 0:
        _warn_rank(nbad, which)
    return nbad


# ---------------- univariate curves ----------------

@njit
def curfit(x, y, w=None, k=3, s=0.0, xb=None, xe=None, nest=-1,
           tol=-1.0, maxit=-1):
    """Univariate smoothing spline, FITPACK ``curfit`` with ``iopt=0``.

    Fits a smoothing spline to one-dimensional data.

    Parameters
    ----------
    x : 1-D array_like of float, length m
        Abscissae, strictly increasing.
    y : 1-D array_like of float, length m
        Ordinates.
    w : 1-D array_like of float, optional
        Positive weights, length m. ``None``, the default, is unit weights; an
        explicit ``np.ones(m)`` gives the same fit. The `None` and the array
        reach the same code: the branch is resolved at this entry point and
        numba prunes the arm it did not take.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    s : float, optional
        Smoothing factor: the fit satisfies
        ``sum(w[i]*(y[i]-s(x[i])))**2 <= s``. ``s = 0`` gives the
        interpolating spline. Default 0.0.
    xb, xe : float, optional
        Interval the fit is made over. ``None``, the default, means ``x[0]``
        and ``x[-1]``. A NaN is a literal knot boundary and reaches FITPACK
        unchanged. They may only WIDEN the data interval; narrowing gives
        ``ier = 10``.
    nest : int, optional
        Over-estimate of the knot count, and the size of the `t` and `c`
        buffers. A negative value, the default, means
        ``max(m + k + 1, 2*k + 3)``. Raise it to let FITPACK add more knots
        before it returns ``ier = 1``.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1d-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    t : 1-D float64 ndarray, length n
        Knot vector, trimmed to the n FITPACK chose.
    c : 1-D float64 ndarray, length n
        B-spline coefficients, in FITPACK's padded form: only the first
        ``n - k - 1`` are meaningful.
    fp : float
        Weighted sum of squared residuals of the returned fit.
    ier : int
        FITPACK status. 0 = smoothing achieved, -1 = interpolating spline
        (the expected value for ``s=0``), -2 = least-squares polynomial,
        positive = failure. NOT raised on -- check it.

    See Also
    --------
    scipy.interpolate.splrep : The scipy routine this mirrors.

    Notes
    -----
    The `s` default is 0.0, the interpolating spline. ``scipy.interpolate``'s
    ``splrep`` resolves its ``s=None`` to 0.0 for an unweighted call and to
    ``m - sqrt(2*m)`` when `w` is supplied; ``scijit.interpolate.splrep``
    reproduces that rule, this raw routine does not.

    Workspaces are sized from the FITPACK docstring formulas:
    ``nest = max(m + k + 1, 2*k + 3)``, ``lwrk = m*(k+1) + nest*(7+3*k)``.
    The integer workspace is int32, which is mandatory. The floor
    ``2*k + 3`` binds only at ``m == k + 1`` and the extra slot is never
    used there.

    The default `nest` is the worst case, allocated once, so the knot search
    cannot exhaust it and ``ier = 1`` does not arise.
    ``scipy.interpolate.UnivariateSpline`` sizes it at
    ``max(m // 2, 2*(k + 1))`` for a smoothing fit and re-enters ``curfit``
    with ``iopt = 1`` when that returns ``ier = 1``, continuing the knot
    search from the stored state. Measured on 40 noisy points at
    ``s`` = 0.05, 0.5 and 2.0: passing ``nest = 20`` here returns ``ier = 1``
    where the default returns ``ier = 0``, and the knots this routine
    returns match the ones that class arrives at after its retry, 0.000e+00
    in all three.

    prange-safe: yes.
    """
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    if w is None:
        w_ = np.ones(len(x_), np.float64)
    else:
        w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    if x_.ndim != 1 or y_.ndim != 1 or w_.ndim != 1:
        raise ValueError("object too deep for desired array")
    m = len(x_)
    if nest < 0:
        nest = max(m + k + 1, 2 * k + 3)
    lwrk = m * (k + 1) + nest * (7 + 3 * k)
    iopt = np.array(0, np.int32)
    m_ = np.array(m, np.int32)
    if xb is None:
        xb_v = x_[0]
    else:
        xb_v = float(xb)
    if xe is None:
        xe_v = x_[m - 1]
    else:
        xe_v = float(xe)
    xb = np.array(xb_v, np.float64)
    xe = np.array(xe_v, np.float64)
    k_ = np.array(k, np.int32)
    s_ = np.array(s, np.float64)
    nest_ = np.array(nest, np.int32)
    n = np.zeros(1, np.int32)
    t = np.zeros(nest, np.float64)
    c = np.zeros(nest, np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(nest, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_D), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _curfit(iopt.ctypes.data, m_.ctypes.data, x_.ctypes.data, y_.ctypes.data,
            w_.ctypes.data, xb.ctypes.data, xe.ctypes.data, k_.ctypes.data,
            s_.ctypes.data, nest_.ctypes.data, n.ctypes.data, t.ctypes.data,
            c.ctypes.data, fp.ctypes.data, wrk.ctypes.data, lwrk_.ctypes.data,
            iwrk.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nn = n[0]
    _check_rank(t[:nn], x_, k, 0)
    return t[:nn], c[:nn], fp[0], ier[0]


@njit
def curfit_lsq(x, y, w, t, k=3, xb=None, xe=None):
    """Least-squares spline on given knots, FITPACK ``curfit``.

    Runs ``curfit`` at ``iopt=-1``: fits the least-squares spline for a fixed
    knot vector.

    Parameters
    ----------
    x : 1-D array_like of float, length m
        Abscissae, strictly increasing.
    y : 1-D array_like of float, length m
        Ordinates.
    w : 1-D array_like of float, length m
        Positive weights (pass ones for unweighted).
    t : 1-D array_like of float
        The FULL knot vector: the boundary values repeated ``k+1`` times at
        each end, with the interior knots between them. It must satisfy the
        Schoenberg-Whitney conditions or FITPACK returns ``ier = 10``.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    xb, xe : float, optional
        Interval the fit is made over. ``None``, the default, means ``x[0]``
        and ``x[-1]``. A NaN is a literal knot boundary and reaches FITPACK
        unchanged. FITPACK overwrites the k+1 boundary knots at each end of
        `t` with these two values.

    Returns
    -------
    t : 1-D float64 ndarray
        The knot vector passed in (a copy; FITPACK may write to it).
    c : 1-D float64 ndarray, same length as `t`
        B-spline coefficients, padded -- only the first ``len(t) - k - 1``
        are meaningful.
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    See Also
    --------
    scipy.interpolate.splrep : Reaches the same fit through ``t=`` and
        ``task=-1``.

    Notes
    -----
    `t` is the full knot vector, where ``scipy.interpolate.splrep(t=...)``
    takes only the interior knots.

    There is no `s` argument: with a fixed knot vector the fit is the
    least-squares one, so smoothing does not apply.

    prange-safe: yes.
    """
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    if (x_.ndim != 1 or y_.ndim != 1 or w_.ndim != 1
            or np.asarray(t).ndim != 1):
        raise ValueError("object too deep for desired array")
    m = len(x_)
    nest = len(t)
    lwrk = m * (k + 1) + nest * (7 + 3 * k)
    iopt = np.array(-1, np.int32)
    m_ = np.array(m, np.int32)
    if xb is None:
        xb_v = x_[0]
    else:
        xb_v = float(xb)
    if xe is None:
        xe_v = x_[m - 1]
    else:
        xe_v = float(xe)
    xb = np.array(xb_v, np.float64)
    xe = np.array(xe_v, np.float64)
    k_ = np.array(k, np.int32)
    s_ = np.array(0.0, np.float64)
    nest_ = np.array(nest, np.int32)
    n = np.array(nest, np.int32)
    t_ = np.ascontiguousarray(np.asarray(t, np.float64).copy())
    c = np.zeros(nest, np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(nest, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_TOL_D, np.float64)
    maxit_ = np.array(_MAXIT, np.int32)
    _curfit(iopt.ctypes.data, m_.ctypes.data, x_.ctypes.data, y_.ctypes.data,
            w_.ctypes.data, xb.ctypes.data, xe.ctypes.data, k_.ctypes.data,
            s_.ctypes.data, nest_.ctypes.data, n.ctypes.data, t_.ctypes.data,
            c.ctypes.data, fp.ctypes.data, wrk.ctypes.data, lwrk_.ctypes.data,
            iwrk.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    return t_, c, fp[0], ier[0]


@njit
def percur(x, y, w, k=3, s=0.0, nest=-1, tol=-1.0, maxit=-1):
    """Periodic smoothing spline, FITPACK ``percur``.

    Fits a periodic smoothing spline to one-dimensional data.

    Parameters
    ----------
    x : 1-D array_like of float, length m
        Abscissae, strictly increasing, spanning exactly one period:
        ``x[m-1] - x[0]`` IS the period.
    y : 1-D array_like of float, length m
        Ordinates, with ``y[m-1] == y[0]`` -- FITPACK requires the last value
        to repeat the first and returns ``ier = 10`` otherwise.
    w : 1-D array_like of float, length m
        Positive weights (pass ones for unweighted).
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    s : float, optional
        Smoothing factor; 0 gives the interpolating periodic spline. Default
        0.0.
    nest : int, optional
        Over-estimate of the knot count, and the size of the `t` and `c`
        buffers. A negative value, the default, means
        ``max(m + 2*k, 2*k + 3)``.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    t : 1-D float64 ndarray, length n
        Knot vector, trimmed.
    c : 1-D float64 ndarray, length n
        B-spline coefficients (padded form).
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    See Also
    --------
    scipy.interpolate.splrep : Reaches the same fit through ``per=1``.

    Notes
    -----
    The `s` default is 0.0, the interpolating periodic spline, which differs
    from ``scipy.interpolate.splrep``'s ``s=None`` default resolution.

    Workspaces: ``nest = max(m + 2*k, 2*k + 3)`` (the floor binds only at
    ``m == k + 1``), ``lwrk = m*(k+1) + nest*(8+5*k)`` -- both larger than
    `curfit` needs, because the periodic system wraps.

    prange-safe: yes.
    """
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    if x_.ndim != 1 or y_.ndim != 1 or w_.ndim != 1:
        raise ValueError("object too deep for desired array")
    m = len(x_)
    if nest < 0:
        nest = max(m + 2 * k, 2 * k + 3)
    lwrk = m * (k + 1) + nest * (8 + 5 * k)
    iopt = np.array(0, np.int32)
    m_ = np.array(m, np.int32)
    k_ = np.array(k, np.int32)
    s_ = np.array(s, np.float64)
    nest_ = np.array(nest, np.int32)
    n = np.zeros(1, np.int32)
    t = np.zeros(nest, np.float64)
    c = np.zeros(nest, np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(nest, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _percur(iopt.ctypes.data, m_.ctypes.data, x_.ctypes.data, y_.ctypes.data,
            w_.ctypes.data, k_.ctypes.data, s_.ctypes.data, nest_.ctypes.data,
            n.ctypes.data, t.ctypes.data, c.ctypes.data, fp.ctypes.data,
            wrk.ctypes.data, lwrk_.ctypes.data, iwrk.ctypes.data,
            tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nn = n[0]
    return t[:nn], c[:nn], fp[0], ier[0]


@njit
def percur_lsq(x, y, w, t, k=3):
    """Periodic least-squares spline on given knots, FITPACK ``percur``.

    Runs ``percur`` at ``iopt=-1``: fits the least-squares periodic spline for
    a fixed knot vector.

    Parameters
    ----------
    x : 1-D array_like of float, length m
        Abscissae, strictly increasing, spanning exactly one period:
        ``x[m-1] - x[0]`` IS the period.
    y : 1-D array_like of float, length m
        Ordinates, with ``y[m-1] == y[0]``.
    w : 1-D array_like of float, length m
        Positive weights (pass ones for unweighted).
    t : 1-D array_like of float, length n
        The FULL knot vector, as `curfit_lsq` takes it: ``2*k+2`` boundary
        slots and the interior knots between them, so
        ``2*k+2 <= n <= min(len(t), m+2*k)``. FITPACK reads
        ``t[k+1] .. t[n-k-2]`` and writes the boundary slots itself, so their
        values on entry do not matter.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.

    Returns
    -------
    t : 1-D float64 ndarray, length n
        The knot vector, boundary slots filled in by FITPACK.
    c : 1-D float64 ndarray, length n
        B-spline coefficients (padded form).
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    See Also
    --------
    scipy.interpolate.splrep : Reaches the same fit through ``t=`` and
        ``per=1``.

    Notes
    -----
    `t` is the full knot vector, where ``scipy.interpolate.splrep(t=...)``
    takes only the interior knots.

    There is no `s` argument: with a fixed knot vector the fit is the
    least-squares one, so smoothing does not apply. There is no `xb`/`xe`
    pair either, because ``percur`` takes neither.

    Workspaces: ``nest = len(t)``, ``lwrk = m*(k+1) + nest*(8+5*k)``.

    prange-safe: yes.
    """
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    if (x_.ndim != 1 or y_.ndim != 1 or w_.ndim != 1
            or np.asarray(t).ndim != 1):
        raise ValueError("object too deep for desired array")
    m = len(x_)
    nest = len(t)
    lwrk = m * (k + 1) + nest * (8 + 5 * k)
    iopt = np.array(-1, np.int32)
    m_ = np.array(m, np.int32)
    k_ = np.array(k, np.int32)
    s_ = np.array(0.0, np.float64)
    nest_ = np.array(nest, np.int32)
    n = np.array(nest, np.int32)
    t_ = np.ascontiguousarray(np.asarray(t, np.float64).copy())
    c = np.zeros(nest, np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(nest, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_TOL_E, np.float64)
    maxit_ = np.array(_MAXIT, np.int32)
    _percur(iopt.ctypes.data, m_.ctypes.data, x_.ctypes.data, y_.ctypes.data,
            w_.ctypes.data, k_.ctypes.data, s_.ctypes.data, nest_.ctypes.data,
            n.ctypes.data, t_.ctypes.data, c.ctypes.data, fp.ctypes.data,
            wrk.ctypes.data, lwrk_.ctypes.data, iwrk.ctypes.data,
            tol_.ctypes.data, maxit_.ctypes.data, ier.ctypes.data)
    nn = n[()]
    return t_[:nn], c[:nn], fp[0], ier[0]


# ---------------- parametric curves ----------------

@njit
def parcur(x, w, idim, k=3, s=0.0, u_in=_NO_ARR, ub=None, ue=None,
           t_in=_NO_ARR, nest=-1, tol=-1.0, maxit=-1):
    """Parametric smoothing spline curve, FITPACK ``parcur``.

    Fits a smoothing spline to a parametric curve given as interleaved data
    points.

    Parameters
    ----------
    x : 1-D array_like of float, length ``idim * m``
        The data points, FLAT AND INTERLEAVED: ``x[idim*i + j]`` is
        coordinate j of point i. This is the input layout; note it is NOT the
        stride-n layout the OUTPUT coefficients use.
    w : 1-D array_like of float, length m
        Positive weights, one per POINT (not per coordinate). Pass ones for
        unweighted.
    idim : int
        Number of coordinates per point, 1 <= idim <= 10.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    s : float, optional
        Smoothing factor; 0 gives the interpolating curve. Default 0.0.
    u_in : 1-D array_like of float, optional
        Parameter values, length m, strictly increasing. A ZERO-LENGTH array,
        the default, selects FITPACK's ``ipar=0``: the chord lengths
        normalised to [0, 1]. Supplying it selects ``ipar=1``.
    ub, ue : float, optional
        Bounds on the parameter, used only with `u_in`, and required to
        satisfy ``ub <= u_in[0]`` and ``ue >= u_in[-1]``. ``None``, the
        default, means ``u_in[0]`` and ``u_in[-1]``; a NaN is a literal bound
        and reaches FITPACK unchanged. Ignored when `u_in` is empty, where
        FITPACK sets them to 0 and 1 itself.
    t_in : 1-D array_like of float, optional
        A FULL knot vector, selecting FITPACK's ``iopt=-1``, the weighted
        least-squares curve on those knots. A ZERO-LENGTH array, the default,
        selects ``iopt=0``.
    nest : int, optional
        Over-estimate of the knot count, and the size of the `t` and `c`
        buffers. A negative value, the default, means ``m + k + 1``, the
        interpolation bound. Lower it to reach FITPACK's ``ier = 1``.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    u : 1-D float64 ndarray, length m
        The parameter values, computed by FITPACK when `u_in` is empty and
        returned unchanged when it is not.
    t : 1-D float64 ndarray, length n
        Knot vector in the parameter, trimmed.
    c : 1-D float64 ndarray, length ``idim * n``
        Coefficients in FITPACK's STRIDE-n curve layout -- dimension j starts
        at ``c[j*n]``. This feeds `curev` and `cualde` directly. It is a
        different layout from the interleaved input `x`.
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    See Also
    --------
    scipy.interpolate.splprep : The scipy routine this mirrors.

    Notes
    -----
    The `s` default is 0.0, the interpolating curve, which differs from
    ``scipy.interpolate.splprep``'s ``s=None`` default of ``m - sqrt(2*m)``.
    The default `nest` is ``m + k + 1``; ``splprep`` sizes it ``m + 2*k`` for
    a smoothing fit and ``m + k + 1`` for ``s = 0``.

    prange-safe: yes.
    """
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    if x_.ndim != 1 or w_.ndim != 1:
        raise ValueError("object too deep for desired array")
    mx = len(x_)
    m = mx // idim
    nn0 = len(t_in)
    if nest < 0:
        nest = m + k + 1
    if nest < nn0:
        nest = nn0
    nc = nest * idim
    lwrk = m * (k + 1) + nest * (6 + idim + 3 * k)
    iopt = np.array(-1 if nn0 > 0 else 0, np.int32)
    ipar = np.array(1 if len(u_in) > 0 else 0, np.int32)
    idim_ = np.array(idim, np.int32)
    m_ = np.array(m, np.int32)
    u = np.zeros(m, np.float64)
    if len(u_in) > 0:
        for i in range(min(m, len(u_in))):
            u[i] = u_in[i]
    mx_ = np.array(mx, np.int32)
    if ub is None:
        ub_u = 0.0
        ub_giv = False
    else:
        ub_u = float(ub)
        ub_giv = True
    if ue is None:
        ue_u = 1.0
        ue_giv = False
    else:
        ue_u = float(ue)
        ue_giv = True
    ub_v = 0.0
    ue_v = 1.0
    if len(u_in) > 0:
        ub_v = ub_u if ub_giv else u_in[0]
        ue_v = ue_u if ue_giv else u_in[len(u_in) - 1]
    ub = np.array(ub_v, np.float64)
    ue = np.array(ue_v, np.float64)
    k_ = np.array(k, np.int32)
    s_ = np.array(s, np.float64)
    nest_ = np.array(nest, np.int32)
    n = np.zeros(1, np.int32)
    n[0] = nn0
    t = np.zeros(nest, np.float64)
    for i in range(nn0):
        t[i] = t_in[i]
    nc_ = np.array(nc, np.int32)
    c = np.zeros(nc, np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(nest, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _parcur(iopt.ctypes.data, ipar.ctypes.data, idim_.ctypes.data,
            m_.ctypes.data, u.ctypes.data, mx_.ctypes.data, x_.ctypes.data,
            w_.ctypes.data, ub.ctypes.data, ue.ctypes.data, k_.ctypes.data,
            s_.ctypes.data, nest_.ctypes.data, n.ctypes.data, t.ctypes.data,
            nc_.ctypes.data, c.ctypes.data, fp.ctypes.data, wrk.ctypes.data,
            lwrk_.ctypes.data, iwrk.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nn = n[0]
    # the knots live in the PARAMETER u, so the check runs against u, not x
    _check_rank(t[:nn], u, k, 0)
    return u, t[:nn], c[:idim * nn], fp[0], ier[0]


@njit
def clocur(x, w, idim, k=3, s=0.0, u_in=_NO_ARR, t_in=_NO_ARR, nest=-1,
           tol=-1.0, maxit=-1):
    """Closed parametric smoothing curve, FITPACK ``clocur``.

    The periodic sibling of `parcur`, fitting a closed parametric curve.

    Parameters
    ----------
    x : 1-D array_like of float, length ``idim * m``
        Data points, flat and interleaved as in `parcur`. The LAST point must
        repeat the first (``x[idim*(m-1) + j] == x[j]`` for every j) or
        FITPACK returns ``ier = 10``.
    w : 1-D array_like of float, length m
        Positive weights, one per point.
    idim : int
        Number of coordinates per point, 1 <= idim <= 10.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    s : float, optional
        Smoothing factor; 0 gives the interpolating closed curve. Default 0.0.
    u_in : 1-D array_like of float, optional
        Parameter values, length m, strictly increasing. A ZERO-LENGTH array,
        the default, selects FITPACK's ``ipar=0``. ``clocur`` has no ``ub``
        or ``ue``: the period is taken from `u_in` itself.
    t_in : 1-D array_like of float, optional
        A FULL knot vector, selecting ``iopt=-1``. A ZERO-LENGTH array, the
        default, selects ``iopt=0``.
    nest : int, optional
        Over-estimate of the knot count. A negative value, the default, means
        ``m + 2*k``.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    u : 1-D float64 ndarray, length m
        Parameter values, computed by FITPACK when `u_in` is empty and
        returned unchanged when it is not.
    t : 1-D float64 ndarray, length n
        Knot vector, trimmed.
    c : 1-D float64 ndarray, length ``idim * n``
        Coefficients in the STRIDE-n curve layout (dimension j at ``c[j*n]``),
        ready for `curev`.
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    See Also
    --------
    scipy.interpolate.splprep : Reaches the same fit through ``per=1``.

    Notes
    -----
    Workspaces are larger than `parcur`'s because of the periodic wrap:
    the default ``nest = m + 2*k``, ``lwrk = m*(k+1) + nest*(7+idim+5*k)``.

    prange-safe: yes.
    """
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    if x_.ndim != 1 or w_.ndim != 1:
        raise ValueError("object too deep for desired array")
    mx = len(x_)
    m = mx // idim
    nn0 = len(t_in)
    if nest < 0:
        nest = m + 2 * k
    if nest < nn0:
        nest = nn0
    nc = nest * idim
    lwrk = m * (k + 1) + nest * (7 + idim + 5 * k)
    iopt = np.array(-1 if nn0 > 0 else 0, np.int32)
    ipar = np.array(1 if len(u_in) > 0 else 0, np.int32)
    idim_ = np.array(idim, np.int32)
    m_ = np.array(m, np.int32)
    u = np.zeros(m, np.float64)
    if len(u_in) > 0:
        for i in range(min(m, len(u_in))):
            u[i] = u_in[i]
    mx_ = np.array(mx, np.int32)
    k_ = np.array(k, np.int32)
    s_ = np.array(s, np.float64)
    nest_ = np.array(nest, np.int32)
    n = np.zeros(1, np.int32)
    n[0] = nn0
    t = np.zeros(nest, np.float64)
    for i in range(nn0):
        t[i] = t_in[i]
    nc_ = np.array(nc, np.int32)
    c = np.zeros(nc, np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(nest, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _clocur(iopt.ctypes.data, ipar.ctypes.data, idim_.ctypes.data,
            m_.ctypes.data, u.ctypes.data, mx_.ctypes.data, x_.ctypes.data,
            w_.ctypes.data, k_.ctypes.data, s_.ctypes.data, nest_.ctypes.data,
            n.ctypes.data, t.ctypes.data, nc_.ctypes.data, c.ctypes.data,
            fp.ctypes.data, wrk.ctypes.data, lwrk_.ctypes.data,
            iwrk.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nn = n[0]
    return u, t[:nn], c[:idim * nn], fp[0], ier[0]


@njit
def concur(u, x, w, idim, db, de, k=3, s=0.0, tol=-1.0, maxit=-1):
    """Constrained parametric curve, FITPACK ``concur``.

    A smoothing spline curve whose value and derivatives are PINNED at both
    ends.

    Parameters
    ----------
    u : 1-D array_like of float, length m
        Parameter values, GIVEN here (unlike `parcur`, which computes them),
        strictly increasing.
    x : 1-D array_like of float, length ``idim * m``
        Data points, flat and interleaved as in `parcur`.
    w : 1-D array_like of float, length m
        Positive weights, one per point.
    idim : int
        Number of coordinates per point.
    db : 1-D array_like of float, length ``idim * ib``
        Constraints at the START of the curve, flat: the values of derivative
        orders 0 .. ib-1, `idim` numbers per order. ``ib`` is inferred as
        ``len(db) // idim``. Pass ``np.zeros(0)`` for no start constraint.
    de : 1-D array_like of float, length ``idim * ie``
        The same at the END of the curve; ``ie = len(de) // idim``. Pass
        ``np.zeros(0)`` for none.
    k : int, optional
        Spline degree, 1 <= k <= 5. Default 3.
    s : float, optional
        Smoothing factor; 0 gives the interpolating constrained curve.
        Default 0.0.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    t : 1-D float64 ndarray, length n
        Knot vector, trimmed.
    c : 1-D float64 ndarray, length ``idim * n``
        Coefficients in the STRIDE-n curve layout, ready for `curev`.
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on. Note ``u``
        is NOT returned here -- it was supplied.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    The knot estimate grows with the constraints:
    ``nest = m + k + 1 + max(0, ib-1) + max(0, ie-1)``. Empty ``db``/``de``
    are padded to length 1 internally because Fortran cannot take a
    zero-length array argument.

    prange-safe: yes.
    """
    u_ = np.ascontiguousarray(np.asarray(u, np.float64))
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    db_in = np.ascontiguousarray(np.asarray(db, np.float64))
    de_in = np.ascontiguousarray(np.asarray(de, np.float64))
    if (u_.ndim != 1 or x_.ndim != 1 or w_.ndim != 1 or db_in.ndim != 1
            or de_in.ndim != 1):
        raise ValueError("object too deep for desired array")
    m = len(u_)
    mx = m * idim
    ib = len(db_in) // idim
    ie = len(de_in) // idim
    nb = max(1, idim * ib)
    ne = max(1, idim * ie)
    db_ = np.zeros(nb, np.float64)
    db_[:len(db_in)] = db_in
    de_ = np.zeros(ne, np.float64)
    de_[:len(de_in)] = de_in
    nest = m + k + 1 + max(0, ib - 1) + max(0, ie - 1)
    nc = nest * idim
    npc = 2 * (k + 1) * idim
    lwrk = m * (k + 1) + nest * (6 + idim + 3 * k)
    iopt = np.array(0, np.int32)
    idim_ = np.array(idim, np.int32)
    m_ = np.array(m, np.int32)
    mx_ = np.array(mx, np.int32)
    xx = np.zeros(mx, np.float64)
    ib_ = np.array(ib, np.int32)
    nb_ = np.array(nb, np.int32)
    ie_ = np.array(ie, np.int32)
    ne_ = np.array(ne, np.int32)
    k_ = np.array(k, np.int32)
    s_ = np.array(s, np.float64)
    nest_ = np.array(nest, np.int32)
    n = np.zeros(1, np.int32)
    t = np.zeros(nest, np.float64)
    nc_ = np.array(nc, np.int32)
    c = np.zeros(nc, np.float64)
    np_ = np.array(npc, np.int32)
    cp = np.zeros(npc, np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(nest, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _concur(iopt.ctypes.data, idim_.ctypes.data, m_.ctypes.data,
            u_.ctypes.data, mx_.ctypes.data, x_.ctypes.data, xx.ctypes.data,
            w_.ctypes.data, ib_.ctypes.data, db_.ctypes.data, nb_.ctypes.data,
            ie_.ctypes.data, de_.ctypes.data, ne_.ctypes.data, k_.ctypes.data,
            s_.ctypes.data, nest_.ctypes.data, n.ctypes.data, t.ctypes.data,
            nc_.ctypes.data, c.ctypes.data, np_.ctypes.data, cp.ctypes.data,
            fp.ctypes.data, wrk.ctypes.data, lwrk_.ctypes.data,
            iwrk.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nn = n[0]
    return t[:nn], c[:idim * nn], fp[0], ier[0]


# ---------------- convexity-constrained cubics ----------------

@njit
def cocosp(x, y, w, t, e):
    """Convexity-constrained least-squares cubic, FITPACK ``cocosp``.

    Least-squares cubic spline on a GIVEN knot vector, subject to sign
    constraints on the second derivative.

    Parameters
    ----------
    x : 1-D array_like of float, length m
        Abscissae, strictly increasing.
    y : 1-D array_like of float, length m
        Ordinates.
    w : 1-D array_like of float, length m
        Positive weights.
    t : 1-D array_like of float, length n
        The FULL knot vector, boundary knots repeated 4 times at each end.
        CUBIC ONLY -- there is no `k` argument.
    e : 1-D array_like of float, length n
        Convexity flag per knot, by SIGN: ``e[j] > 0`` forces ``s'' <= 0``
        (concave) at ``t[j]``, ``e[j] < 0`` forces ``s'' >= 0`` (convex),
        ``e[j] == 0`` leaves that knot unconstrained. Only the sign matters,
        not the magnitude. Copied internally because FITPACK writes to it.

    Returns
    -------
    c : 1-D float64 ndarray, length n
        B-spline coefficients of the constrained fit (padded form).
    sq : float
        Weighted sum of squared residuals.
    sx : 1-D float64 ndarray, length m
        The fitted values ``s(x[i])``.
    bind : 1-D int32 ndarray, length n
        Which constraints are ACTIVE at the solution: non-zero where the
        constraint at that knot is binding.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    Internally ``maxtr = 100`` (search-tree records) and
    ``maxbin = max(1, n-6)`` (simultaneously binding constraints). Those are
    fixed here; a problem that needs more comes back as a positive ``ier``
    rather than silently truncating.

    prange-safe: yes.
    """
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    t_ = np.ascontiguousarray(np.asarray(t, np.float64))
    e_ = np.ascontiguousarray(np.asarray(e, np.float64).copy())
    if (x_.ndim != 1 or y_.ndim != 1 or w_.ndim != 1 or t_.ndim != 1
            or e_.ndim != 1):
        raise ValueError("object too deep for desired array")
    m = len(x_)
    n = len(t_)
    maxtr = 100
    maxbin = max(1, n - 6)
    lwrk = m * 4 + n * 7 + maxbin * (maxbin + n + 1)
    kwrk = maxtr * 4 + 2 * (maxbin + 1)
    m_ = np.array(m, np.int32)
    n_ = np.array(n, np.int32)
    maxtr_ = np.array(maxtr, np.int32)
    maxbin_ = np.array(maxbin, np.int32)
    c = np.zeros(n, np.float64)
    sq = np.zeros(1, np.float64)
    sx = np.zeros(m, np.float64)
    bnd = np.zeros(n, np.int32)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(kwrk, np.int32)
    kwrk_ = np.array(kwrk, np.int32)
    ier = np.zeros(1, np.int32)
    _cocosp(m_.ctypes.data, x_.ctypes.data, y_.ctypes.data, w_.ctypes.data,
            n_.ctypes.data, t_.ctypes.data, e_.ctypes.data,
            maxtr_.ctypes.data, maxbin_.ctypes.data, c.ctypes.data,
            sq.ctypes.data, sx.ctypes.data, bnd.ctypes.data, wrk.ctypes.data,
            lwrk_.ctypes.data, iwrk.ctypes.data, kwrk_.ctypes.data,
            ier.ctypes.data)
    return c, sq[0], sx, bnd, ier[0]


@njit
def concon(x, y, w, v, s):
    """Convexity-constrained smoothing cubic, FITPACK ``concon``.

    Like `cocosp`, but FITPACK chooses the knots to meet a smoothing target
    instead of taking them as arguments.

    Parameters
    ----------
    x : 1-D array_like of float, length m
        Abscissae, strictly increasing.
    y : 1-D array_like of float, length m
        Ordinates.
    w : 1-D array_like of float, length m
        Positive weights.
    v : 1-D array_like of float, length m
        Convexity flag per DATA POINT (not per knot, unlike `cocosp`'s `e`),
        by sign: ``v[i] > 0`` forces ``s'' <= 0`` at ``x[i]``, ``v[i] < 0``
        forces ``s'' >= 0``, ``v[i] == 0`` leaves it unconstrained. Copied
        internally because FITPACK writes to it.
    s : float
        Smoothing factor: the fit satisfies
        ``sum(w[i]*(y[i]-s(x[i])))**2 <= s``. Required, with no default.

    Returns
    -------
    t : 1-D float64 ndarray, length n
        Knot vector FITPACK chose, trimmed.
    c : 1-D float64 ndarray, length n
        B-spline coefficients (padded form).
    sq : float
        Weighted sum of squared residuals.
    sx : 1-D float64 ndarray, length m
        The fitted values ``s(x[i])``.
    bind : 1-D int32 ndarray, length n
        Non-zero where the convexity constraint at that knot is active.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    CUBIC ONLY (no `k` argument). ``nest = m + 4``, ``maxtr = 100``,
    ``maxbin = max(1, nest-6)`` are fixed internally.

    prange-safe: yes.
    """
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    v_ = np.ascontiguousarray(np.asarray(v, np.float64).copy())
    if (x_.ndim != 1 or y_.ndim != 1 or w_.ndim != 1 or v_.ndim != 1):
        raise ValueError("object too deep for desired array")
    m = len(x_)
    nest = m + 4
    maxtr = 100
    maxbin = max(1, nest - 6)
    lwrk = m * 4 + nest * 8 + maxbin * (maxbin + nest + 1)
    kwrk = maxtr * 4 + 2 * (maxbin + 1)
    iopt = np.array(0, np.int32)
    m_ = np.array(m, np.int32)
    s_ = np.array(s, np.float64)
    nest_ = np.array(nest, np.int32)
    maxtr_ = np.array(maxtr, np.int32)
    maxbin_ = np.array(maxbin, np.int32)
    n = np.zeros(1, np.int32)
    t = np.zeros(nest, np.float64)
    c = np.zeros(nest, np.float64)
    sq = np.zeros(1, np.float64)
    sx = np.zeros(m, np.float64)
    bnd = np.zeros(nest, np.int32)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(kwrk, np.int32)
    kwrk_ = np.array(kwrk, np.int32)
    ier = np.zeros(1, np.int32)
    _concon(iopt.ctypes.data, m_.ctypes.data, x_.ctypes.data, y_.ctypes.data,
            w_.ctypes.data, v_.ctypes.data, s_.ctypes.data, nest_.ctypes.data,
            maxtr_.ctypes.data, maxbin_.ctypes.data, n.ctypes.data,
            t.ctypes.data, c.ctypes.data, sq.ctypes.data, sx.ctypes.data,
            bnd.ctypes.data, wrk.ctypes.data, lwrk_.ctypes.data,
            iwrk.ctypes.data, kwrk_.ctypes.data, ier.ctypes.data)
    nn = n[0]
    return t[:nn], c[:nn], sq[0], sx, bnd[:nn], ier[0]


# ---------------- gridded surfaces ----------------

@njit
def regrid(x, y, z, kx=3, ky=3, s=0.0, xb=None, xe=None,
           yb=None, ye=None, tol=-1.0, maxit=-1):
    """Gridded bivariate smoothing spline, FITPACK ``regrid``.

    Fits a smoothing spline to values on a rectangular grid.

    Parameters
    ----------
    x : 1-D array_like of float, length mx
        Grid abscissae, strictly increasing.
    y : 1-D array_like of float, length my
        Grid ordinates, strictly increasing.
    z : array_like of float, ``mx * my`` values
        Grid values, x-index major: ``z[my*i + j] == f(x[i], y[j])``. A
        C-order ``(mx, my)`` array holds them in that order, so both it and
        its ``.ravel()`` are accepted. Any other number of values raises
        ``ValueError``.
    kx : int, optional
        Degree in x, 1 <= kx <= 5. Default 3.
    ky : int, optional
        Degree in y, 1 <= ky <= 5. Default 3.
    s : float, optional
        Smoothing factor; 0 gives the interpolating surface. Default 0.0.
    xb, xe, yb, ye : float, optional
        The rectangle the fit is made over. ``None``, the default, means the
        data range ``x[0]``, ``x[-1]``, ``y[0]``, ``y[-1]``. A NaN is a
        literal edge and reaches FITPACK unchanged. They may only WIDEN it; a
        narrower rectangle gives ``ier = 10``.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    tx : 1-D float64 ndarray, length nx
        Knots in x, trimmed.
    ty : 1-D float64 ndarray, length ny
        Knots in y, trimmed.
    c : 1-D float64 ndarray, length ``(nx-kx-1)*(ny-ky-1)``
        Coefficients in the flat bivariate layout ``c[(ny-ky-1)*i + j]``,
        which is what `bispev`, `bispeu`, `parder`, `pardeu` and `dblint`
        expect -- no reshaping needed.
    fp : float
        Sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    See Also
    --------
    scipy.interpolate.RectBivariateSpline : The scipy routine this mirrors.

    Notes
    -----
    On a smoothing fit (``s > 0``) the knot search can land in different
    positions from ``scipy.interpolate.RectBivariateSpline`` on sensitive data,
    while returning the same knot count. Both answers are valid smoothing
    splines. The module docstring gives measured figures.

    Unweighted: FITPACK's ``regrid`` has no weight array, so there is no `w`
    argument.

    prange-safe: yes.
    """
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    z_ = np.ascontiguousarray(np.asarray(z, np.float64))
    if x_.ndim != 1 or y_.ndim != 1:
        raise ValueError("object too deep for desired array")
    mx = len(x_)
    my = len(y_)
    if z_.size != mx * my:
        raise ValueError("z array length must equal mx*my")
    nxest = max(mx + kx + 1, 2 * (kx + 1))
    nyest = max(my + ky + 1, 2 * (ky + 1))
    lwrk = (4 + nxest * (my + 2 * kx + 5) + nyest * (2 * ky + 5) +
            mx * (kx + 1) + my * (ky + 1) + max(my, nxest))
    kwrk = 3 + mx + my + nxest + nyest
    iopt = np.array(0, np.int32)
    mx_ = np.array(mx, np.int32)
    my_ = np.array(my, np.int32)
    if xb is None:
        xb_v = x_[0]
    else:
        xb_v = float(xb)
    if xe is None:
        xe_v = x_[mx - 1]
    else:
        xe_v = float(xe)
    if yb is None:
        yb_v = y_[0]
    else:
        yb_v = float(yb)
    if ye is None:
        ye_v = y_[my - 1]
    else:
        ye_v = float(ye)
    xb = np.array(xb_v, np.float64)
    xe = np.array(xe_v, np.float64)
    yb = np.array(yb_v, np.float64)
    ye = np.array(ye_v, np.float64)
    kx_ = np.array(kx, np.int32)
    ky_ = np.array(ky, np.int32)
    s_ = np.array(s, np.float64)
    nxest_ = np.array(nxest, np.int32)
    nyest_ = np.array(nyest, np.int32)
    nx = np.zeros(1, np.int32)
    ny = np.zeros(1, np.int32)
    tx = np.zeros(nxest, np.float64)
    ty = np.zeros(nyest, np.float64)
    c = np.zeros((nxest - kx - 1) * (nyest - ky - 1), np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(kwrk, np.int32)
    kwrk_ = np.array(kwrk, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _regrid(iopt.ctypes.data, mx_.ctypes.data, x_.ctypes.data,
            my_.ctypes.data, y_.ctypes.data, z_.ctypes.data, xb.ctypes.data,
            xe.ctypes.data, yb.ctypes.data, ye.ctypes.data, kx_.ctypes.data,
            ky_.ctypes.data, s_.ctypes.data, nxest_.ctypes.data,
            nyest_.ctypes.data, nx.ctypes.data, tx.ctypes.data,
            ny.ctypes.data, ty.ctypes.data, c.ctypes.data, fp.ctypes.data,
            wrk.ctypes.data, lwrk_.ctypes.data, iwrk.ctypes.data,
            kwrk_.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nnx = nx[0]
    nny = ny[0]
    _check_rank(tx[:nnx], x_, kx, 1)
    _check_rank(ty[:nny], y_, ky, 2)
    return (tx[:nnx], ty[:nny], c[:(nnx - kx - 1) * (nny - ky - 1)],
            fp[0], ier[0])


@njit
def parsur(u, v, f, idim, ipar0=0, ipar1=0, s=0.0, tol=-1.0, maxit=-1):
    """Parametric gridded surface, FITPACK ``parsur``.

    A bicubic smoothing spline through gridded `idim`-dimensional points, the
    fitter whose output `surev` evaluates.

    Parameters
    ----------
    u : 1-D array_like of float, length mu
        First parameter grid, strictly increasing.
    v : 1-D array_like of float, length mv
        Second parameter grid, strictly increasing.
    f : array_like of float, ``idim * mu * mv`` values
        Grid values: ``f[mu*mv*l + mv*i + j]`` is coordinate l of the point
        at ``(u[i], v[j])``. Dimension index is OUTERMOST here. An
        ``(idim, mu, mv)`` C-order array holds them in that order and is
        accepted. Any other number of values raises ``ValueError``.
    idim : int
        Number of surface dimensions, 1 <= idim <= 3.
    ipar0 : int, optional
        1 makes the surface periodic in u, 0 does not. Default 0.
    ipar1 : int, optional
        1 makes the surface periodic in v, 0 does not. Default 0.
    s : float, optional
        Smoothing factor; 0 gives the interpolating surface. Default 0.0.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    tu : 1-D float64 ndarray, length nu
        Knots in u, trimmed.
    tv : 1-D float64 ndarray, length nv
        Knots in v, trimmed.
    c : 1-D float64 ndarray, length ``(nu-4)*(nv-4)*idim``
        Coefficients in the layout `surev` expects.
    fp : float
        Sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    BICUBIC ONLY -- ``parsur`` fixes the degree at 3 in both parameters, so
    there are no `ku` / `kv` arguments. Unweighted.

    prange-safe: yes.
    """
    u_ = np.ascontiguousarray(np.asarray(u, np.float64))
    v_ = np.ascontiguousarray(np.asarray(v, np.float64))
    f_ = np.ascontiguousarray(np.asarray(f, np.float64))
    if u_.ndim != 1 or v_.ndim != 1:
        raise ValueError("object too deep for desired array")
    mu = len(u_)
    mv = len(v_)
    if f_.size != idim * mu * mv:
        raise ValueError("f array length must equal idim*mu*mv")
    nuest = max(mu + 4 + 2 * ipar0, 8)
    nvest = max(mv + 4 + 2 * ipar1, 8)
    lwrk = (4 + nuest * (mv * idim + 11 + 4 * ipar0) +
            nvest * (11 + 4 * ipar1) + 4 * (mu + mv) +
            max(mv, nuest) * idim)
    kwrk = 3 + mu + mv + nuest + nvest
    iopt = np.array(0, np.int32)
    ipar = np.zeros(2, np.int32)
    ipar[0] = ipar0
    ipar[1] = ipar1
    idim_ = np.array(idim, np.int32)
    mu_ = np.array(mu, np.int32)
    mv_ = np.array(mv, np.int32)
    s_ = np.array(s, np.float64)
    nuest_ = np.array(nuest, np.int32)
    nvest_ = np.array(nvest, np.int32)
    nu = np.zeros(1, np.int32)
    nv = np.zeros(1, np.int32)
    tu = np.zeros(nuest, np.float64)
    tv = np.zeros(nvest, np.float64)
    c = np.zeros((nuest - 4) * (nvest - 4) * idim, np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(kwrk, np.int32)
    kwrk_ = np.array(kwrk, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _parsur(iopt.ctypes.data, ipar.ctypes.data, idim_.ctypes.data,
            mu_.ctypes.data, u_.ctypes.data, mv_.ctypes.data, v_.ctypes.data,
            f_.ctypes.data, s_.ctypes.data, nuest_.ctypes.data,
            nvest_.ctypes.data, nu.ctypes.data, tu.ctypes.data,
            nv.ctypes.data, tv.ctypes.data, c.ctypes.data, fp.ctypes.data,
            wrk.ctypes.data, lwrk_.ctypes.data, iwrk.ctypes.data,
            kwrk_.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nnu = nu[0]
    nnv = nv[0]
    return (tu[:nnu], tv[:nnv], c[:(nnu - 4) * (nnv - 4) * idim],
            fp[0], ier[0])


@njit
def pogrid(u, v, z, z0, r, s=0.0, iopt2=0, iopt3=0, ider0=-1, ider1=0,
           validate=True, tol=-1.0, maxit=-1):
    """Gridded fit on a disc, FITPACK ``pogrid``.

    A bicubic spline through data given on a polar grid inside a disc of
    radius `r`.

    Parameters
    ----------
    u : 1-D array_like of float, length mu
        Radial grid, strictly increasing in ``(0, r]``.
    v : 1-D array_like of float, length mv
        Angular grid, strictly increasing, spanning one turn.
    z : array_like of float, ``mu * mv`` values
        Values, radius-major: ``z[mv*i + j]`` is the value at polar
        coordinates ``(u[i], v[j])``. A C-order ``(mu, mv)`` array holds them
        in that order and is accepted. Any other number of values raises
        ``ValueError``.
    z0 : float
        Value at the ORIGIN, used only when ``ider0 >= 0``.
    r : float
        Disc radius.
    s : float, optional
        Smoothing factor; 0 gives the interpolating surface. Default 0.0.
    iopt2 : int, optional
        Requested ORDER OF CONTINUITY at the origin: 0 or 1. Default 0.
        Only 0 and 1 exist here. The sibling `polar` also accepts 2; passing
        2 to `pogrid` returns ``ier = 10``, measured.
    iopt3 : int, optional
        0 = none, 1 = the SURFACE ITSELF vanishes on the boundary circle.
        Default 0. This constrains the value, not a derivative: measured on
        an 8x14 disc fixture, ``max|f|`` at ``u = r`` is 1.297163e-03 at
        ``iopt3 = 0`` and exactly 0.000000e+00 at ``iopt3 = 1``, with the
        interior unchanged (2.248866 against 2.248867 at ``u = 0.5``).
        ``iopt3 = 1`` also requires ``u[-1] < r``, strictly.
    ider0 : int, optional
        Whether there is a data value `z0` at the origin: -1 = there is
        none, 0 = `z0` is an ordinary data value like the rest, 1 = `z0` is
        the right function value and is fitted exactly. Default -1.
    ider1 : int, optional
        0 = free, 1 = the surface has VANISHING PARTIAL DERIVATIVES AT THE
        ORIGIN. Default 0. This is an origin condition, not a boundary one.
        ``ider1 = 1`` requires ``iopt2 = 1``; the pair ``iopt2 = 0,
        ider1 = 1`` returns ``ier = 10``, measured.
    validate : bool, optional
        Raise on a `nuest` or `nvest` this routine cannot legally pass to
        FITPACK, rather than passing it and letting FITPACK report
        ``ier = 10``. Default True. Set False for a parameter sweep, where
        an exception would strand the run, and filter on `ier` instead.

        `nuest` and `nvest` are not arguments: this routine computes them
        as ``mu + 5 + iopt2 + iopt3`` and ``mv + 7``, and FITPACK documents
        ``nuest >= 8, nvest >= 8``. A caller therefore has no way to correct
        an out-of-range value except by changing the grid, which is what the
        message says.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    tu : 1-D float64 ndarray, length nu
        Knots in the radial parameter, trimmed.
    tv : 1-D float64 ndarray, length nv
        Knots in the angular parameter, trimmed.
    c : 1-D float64 ndarray, length ``(nu-4)*(nv-4)``
        Coefficients in the flat bivariate layout, usable with `bispev` in
        the (u, v) parametrisation.
    fp : float
        Sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    BICUBIC ONLY. Unweighted. The `iopt` and `ider` arrays FITPACK wants are
    assembled from the scalar arguments here.

    **When the `validate` guard can fire.** ``nuest < 8`` means
    ``mu + iopt2 + iopt3 < 3``. FITPACK separately requires ``mu >= mumin``
    with ``mumin = 4 - iopt3 - ider1``, one less again when ``ider0 >= 0``,
    so the loosest `mumin` is ``3 - iopt2 - iopt3``. The two conditions are
    the same inequality, and enumerating the whole legal flag space for
    ``mu`` in 1..7 gives 99 legal combinations, **none** with ``nuest < 8``;
    the smallest `nuest` any legal combination reaches is exactly 8.
    ``nvest = mv + 7`` with ``mv >= 4`` required, so the smallest legal
    `nvest` is 11.

    So the guard fires only on a grid that is too small on more than one
    count: ``mu <= 2`` with the default flags. A ``mu`` that clears the
    `nuest` bound can still be below `mumin` -- ``mu = 3`` gives
    ``nuest = 8`` and still returns ``ier = 10`` -- and that case is left to
    FITPACK rather than raised on, because `mu` is the caller's to choose
    while `nuest` is this routine's to compute.

    prange-safe: yes.
    """
    u_ = np.ascontiguousarray(np.asarray(u, np.float64))
    v_ = np.ascontiguousarray(np.asarray(v, np.float64))
    z_ = np.ascontiguousarray(np.asarray(z, np.float64))
    if u_.ndim != 1 or v_.ndim != 1:
        raise ValueError("object too deep for desired array")
    mu = len(u_)
    mv = len(v_)
    if z_.size != mu * mv:
        raise ValueError("z array length must equal mu*mv")
    nuest = mu + 5 + iopt2 + iopt3
    nvest = mv + 7
    if validate:
        # FITPACK documents nuest >= 8, nvest >= 8 (pogrid.f:112) and
        # enforces it at pogrid.f:356 by returning ier=10, which conflates
        # this with a dozen other input faults. nuest and nvest are computed
        # here, not passed in, so the check belongs on this side.
        if nuest < 8:
            raise ValueError(
                "pogrid computes nuest as mu + 5 + iopt2 + iopt3, and that "
                "is below FITPACK's documented minimum of 8. The grid is "
                "too small in the radial direction: use at least 3 u "
                "points, more for the default flags. Pass validate=False "
                "to call FITPACK anyway and read ier.")
        if nvest < 8:
            raise ValueError(
                "pogrid computes nvest as mv + 7, and that is below "
                "FITPACK's documented minimum of 8. FITPACK also requires "
                "mv >= 4. Pass validate=False to call FITPACK anyway and "
                "read ier.")
    lwrk = (8 + nuest * (mv + nvest + 3) + nvest * 21 + 4 * mu + 6 * mv +
            max(mv + nvest, nuest))
    kwrk = 4 + mu + mv + nuest + nvest
    iopt = np.zeros(3, np.int32)
    iopt[1] = iopt2
    iopt[2] = iopt3
    ider = np.zeros(2, np.int32)
    ider[0] = ider0
    ider[1] = ider1
    mu_ = np.array(mu, np.int32)
    mv_ = np.array(mv, np.int32)
    z0_ = np.array(z0, np.float64)
    r_ = np.array(r, np.float64)
    s_ = np.array(s, np.float64)
    nuest_ = np.array(nuest, np.int32)
    nvest_ = np.array(nvest, np.int32)
    nu = np.zeros(1, np.int32)
    nv = np.zeros(1, np.int32)
    tu = np.zeros(nuest, np.float64)
    tv = np.zeros(nvest, np.float64)
    c = np.zeros((nuest - 4) * (nvest - 4), np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(kwrk, np.int32)
    kwrk_ = np.array(kwrk, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _pogrid(iopt.ctypes.data, ider.ctypes.data, mu_.ctypes.data,
            u_.ctypes.data, mv_.ctypes.data, v_.ctypes.data, z_.ctypes.data,
            z0_.ctypes.data, r_.ctypes.data, s_.ctypes.data,
            nuest_.ctypes.data, nvest_.ctypes.data, nu.ctypes.data,
            tu.ctypes.data, nv.ctypes.data, tv.ctypes.data, c.ctypes.data,
            fp.ctypes.data, wrk.ctypes.data, lwrk_.ctypes.data,
            iwrk.ctypes.data, kwrk_.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nnu = nu[0]
    nnv = nv[0]
    return tu[:nnu], tv[:nnv], c[:(nnu - 4) * (nnv - 4)], fp[0], ier[0]


@njit
def spgrid(u, v, r, r0, r1, s=0.0, iopt2=0, iopt3=0,
           ider0=-1, ider1=0, ider2=-1, ider3=0, tol=-1.0, maxit=-1):
    """Gridded fit on a sphere, FITPACK ``spgrid``.

    A bicubic smoothing spline through data on a spherical grid.

    Parameters
    ----------
    u : 1-D array_like of float, length mu
        Colatitude grid, strictly increasing in ``(0, pi)``.
    v : 1-D array_like of float, length mv
        Longitude grid, strictly increasing, spanning one turn.
    r : array_like of float, ``mu * mv`` values
        Values, colatitude-major: ``r[mv*i + j]`` is the value at
        ``(u[i], v[j])``. A C-order ``(mu, mv)`` array holds them in that
        order and is accepted. Any other number of values raises
        ``ValueError``.
    r0 : float
        Value at the NORTH pole (u = 0), used only when ``ider0 >= 0``.
    r1 : float
        Value at the SOUTH pole (u = pi), used only when ``ider2 >= 0``.
    s : float, optional
        Smoothing factor; 0 gives the interpolating surface. Default 0.0.
    iopt2 : int, optional
        Requested ORDER OF CONTINUITY at the north pole ``u = 0``: 0 or 1.
        Default 0. Only 0 and 1 exist; passing 2 returns ``ier = 10``,
        measured.
    iopt3 : int, optional
        Same, at the south pole ``u = pi``. Default 0.
    ider0 : int, optional
        Whether there is a data value `r0` at the north pole: -1 = there is
        none, 0 = `r0` is an ordinary data value like the rest, 1 = `r0` is
        the right function value and is fitted exactly (``s(0, v) = r0``).
        Default -1. A vanishing gradient is `ider1`, a separate flag.
    ider1 : int, optional
        0 = free, 1 = the approximation has vanishing derivatives at the
        north pole. Default 0. Requires ``iopt2 = 1``.
    ider2 : int, optional
        South-pole counterpart of `ider0`, using `r1`. Default -1.
    ider3 : int, optional
        South-pole counterpart of `ider1`. Requires ``iopt3 = 1``.
        Default 0.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    tu : 1-D float64 ndarray, length nu
        Knots in colatitude, trimmed.
    tv : 1-D float64 ndarray, length nv
        Knots in longitude, trimmed.
    c : 1-D float64 ndarray, length ``(nu-4)*(nv-4)``
        Coefficients in the flat bivariate layout.
    fp : float
        Sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    See Also
    --------
    scipy.interpolate.RectSphereBivariateSpline : The scipy routine this
        mirrors.

    Notes
    -----
    BICUBIC ONLY. Unweighted.

    The longitude knots `tv` carry no offset. Sphere fits are the one place
    where a convention mismatch would be plausible, so check `tv` against an
    independent reference if the longitude range changes.

    **`nuest` and `nvest` are not arguments**, and there is no `validate`
    flag here, unlike the sibling `pogrid`. The knot bounds are computed from
    the grid shape, so a grid too small for FITPACK is rejected before the fit
    runs. Measured at ``mv = 6``: ``mu`` of 1, 2 and 3 give ``ier = 10``,
    while ``mu`` of 4 and 5 fit with ``nu`` of 10 and 11. Enumerating the
    whole legal flag space for ``mu`` in 1..7 gives 482 legal combinations,
    none of which reaches ``nuest < 8``.

    prange-safe: yes.
    """
    u_ = np.ascontiguousarray(np.asarray(u, np.float64))
    v_ = np.ascontiguousarray(np.asarray(v, np.float64))
    r_ = np.ascontiguousarray(np.asarray(r, np.float64))
    if u_.ndim != 1 or v_.ndim != 1:
        raise ValueError("object too deep for desired array")
    mu = len(u_)
    mv = len(v_)
    if r_.size != mu * mv:
        raise ValueError("r array length must equal mu*mv")
    # scipy's own rule: the interpolation bound at s=0, and a heuristic floor
    # under it for a smoothing fit.
    if s == 0.0:
        nuest = mu + 6 + iopt2 + iopt3
        nvest = mv + 7
    else:
        nuest = max(mu + 6 + iopt2 + iopt3, 8 + int(np.sqrt(mu / 2.0)))
        nvest = max(mv + 7, 8 + int(np.sqrt(mv / 2.0)))
    lwrk = (12 + nuest * (mv + nvest + 3) + nvest * 24 + 4 * mu + 8 * mv +
            max(mv + nvest, nuest))
    kwrk = 5 + mu + mv + nuest + nvest
    iopt = np.zeros(3, np.int32)
    iopt[1] = iopt2
    iopt[2] = iopt3
    ider = np.zeros(4, np.int32)
    ider[0] = ider0
    ider[1] = ider1
    ider[2] = ider2
    ider[3] = ider3
    mu_ = np.array(mu, np.int32)
    mv_ = np.array(mv, np.int32)
    r0_ = np.array(r0, np.float64)
    r1_ = np.array(r1, np.float64)
    s_ = np.array(s, np.float64)
    nuest_ = np.array(nuest, np.int32)
    nvest_ = np.array(nvest, np.int32)
    nu = np.zeros(1, np.int32)
    nv = np.zeros(1, np.int32)
    tu = np.zeros(nuest, np.float64)
    tv = np.zeros(nvest, np.float64)
    c = np.zeros((nuest - 4) * (nvest - 4), np.float64)
    fp = np.zeros(1, np.float64)
    wrk = np.zeros(lwrk, np.float64)
    lwrk_ = np.array(lwrk, np.int32)
    iwrk = np.zeros(kwrk, np.int32)
    kwrk_ = np.array(kwrk, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _spgrid(iopt.ctypes.data, ider.ctypes.data, mu_.ctypes.data,
            u_.ctypes.data, mv_.ctypes.data, v_.ctypes.data, r_.ctypes.data,
            r0_.ctypes.data, r1_.ctypes.data, s_.ctypes.data,
            nuest_.ctypes.data, nvest_.ctypes.data, nu.ctypes.data,
            tu.ctypes.data, nv.ctypes.data, tv.ctypes.data, c.ctypes.data,
            fp.ctypes.data, wrk.ctypes.data, lwrk_.ctypes.data,
            iwrk.ctypes.data, kwrk_.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nnu = nu[0]
    nnv = nv[0]
    return tu[:nnu], tv[:nnv], c[:(nnu - 4) * (nnv - 4)], fp[0], ier[0]


# ---------------- scattered-data surfaces ----------------

@njit
def sphere(teta, phi, r, w, s, eps=1e-16, ntest=0, npest=0,
           tol=-1.0, maxit=-1):
    """Scattered-data fit on a sphere, FITPACK ``sphere``.

    Fits a smoothing spline to scattered data on the surface of a sphere.

    Parameters
    ----------
    teta : 1-D array_like of float, length m
        Colatitudes of the data points, in ``[0, pi]``.
    phi : 1-D array_like of float, length m
        Longitudes, in ``[0, 2*pi]``.
    r : 1-D array_like of float, length m
        Values at those points.
    w : 1-D array_like of float, length m
        Positive weights (pass ones for unweighted).
    s : float
        Smoothing factor; required, with no default.
    eps : float, optional
        Rank-determination threshold for the least-squares system, in
        ``(0, 1)``. Default 1e-16. It CHANGES THE COEFFICIENTS on
        rank-deficient data, so it is not cosmetic: on a polar cap at
        ``s = 1e-4`` the surface moves by 3.379e-05 between 1e-16 and 1e-10.
    ntest : int, optional
        Upper bound on the number of colatitude knots. ``<= 0`` (the default)
        picks ``8 + int(sqrt(m/2))``. Too small gives ``ier = 1``.
    npest : int, optional
        Upper bound on the number of longitude knots; ``<= 0`` auto-picks the
        same formula. Default 0.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    tt : 1-D float64 ndarray, length nt
        Knots in colatitude, trimmed.
    tp : 1-D float64 ndarray, length np
        Knots in longitude, trimmed.
    c : 1-D float64 ndarray, length ``(nt-4)*(np-4)``
        Coefficients in the flat bivariate layout.
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    See Also
    --------
    scipy.interpolate.SmoothSphereBivariateSpline : The scipy routine this
        mirrors.

    Notes
    -----
    BICUBIC ONLY.

    prange-safe: yes.
    """
    teta_ = np.ascontiguousarray(np.asarray(teta, np.float64))
    phi_ = np.ascontiguousarray(np.asarray(phi, np.float64))
    r_ = np.ascontiguousarray(np.asarray(r, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    if (teta_.ndim != 1 or phi_.ndim != 1 or r_.ndim != 1
            or w_.ndim != 1):
        raise ValueError("object too deep for desired array")
    m = len(teta_)
    if ntest <= 0:
        ntest = 8 + int(np.sqrt(m / 2.0))
    if npest <= 0:
        npest = 8 + int(np.sqrt(m / 2.0))
    uu = ntest - 7
    vv = npest - 7
    lwrk1 = 185 + 52 * vv + 10 * uu + 14 * uu * vv + \
        8 * (uu - 1) * vv * vv + 8 * m
    lwrk2 = 48 + 21 * vv + 7 * uu * vv + 4 * (uu - 1) * vv * vv
    kwrk = m + (ntest - 7) * (npest - 7)
    iopt = np.array(0, np.int32)
    m_ = np.array(m, np.int32)
    s_ = np.array(s, np.float64)
    ntest_ = np.array(ntest, np.int32)
    npest_ = np.array(npest, np.int32)
    eps_ = np.array(eps, np.float64)
    nt = np.zeros(1, np.int32)
    npp = np.zeros(1, np.int32)
    tt = np.zeros(ntest, np.float64)
    tp = np.zeros(npest, np.float64)
    c = np.zeros((ntest - 4) * (npest - 4), np.float64)
    fp = np.zeros(1, np.float64)
    wrk1 = np.zeros(lwrk1, np.float64)
    lwrk1_ = np.array(lwrk1, np.int32)
    wrk2 = np.zeros(lwrk2, np.float64)
    lwrk2_ = np.array(lwrk2, np.int32)
    iwrk = np.zeros(kwrk, np.int32)
    kwrk_ = np.array(kwrk, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _sphere(iopt.ctypes.data, m_.ctypes.data, teta_.ctypes.data,
            phi_.ctypes.data, r_.ctypes.data, w_.ctypes.data, s_.ctypes.data,
            ntest_.ctypes.data, npest_.ctypes.data, eps_.ctypes.data,
            nt.ctypes.data, tt.ctypes.data, npp.ctypes.data, tp.ctypes.data,
            c.ctypes.data, fp.ctypes.data, wrk1.ctypes.data,
            lwrk1_.ctypes.data, wrk2.ctypes.data, lwrk2_.ctypes.data,
            iwrk.ctypes.data, kwrk_.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nnt = nt[0]
    nnp = npp[0]
    return tt[:nnt], tp[:nnp], c[:(nnt - 4) * (nnp - 4)], fp[0], ier[0]


@njit
def surfit(x, y, z, w, kx=3, ky=3, s=None, eps=1e-16, nxest=0, nyest=0,
           xb=None, xe=None, yb=None, ye=None, tol=-1.0, maxit=-1):
    """Scattered-data bivariate smoothing spline, FITPACK ``surfit``.

    Fits a smoothing spline to scattered data points.

    Parameters
    ----------
    x : 1-D array_like of float, length m
        Abscissae of the scattered data points (no grid structure needed).
    y : 1-D array_like of float, length m
        Ordinates.
    z : 1-D array_like of float, length m
        Values at those points.
    w : 1-D array_like of float, length m
        Positive weights (pass ones for unweighted).
    kx : int, optional
        Degree in x, 1 <= kx <= 5. Default 3.
    ky : int, optional
        Degree in y, 1 <= ky <= 5. Default 3.
    s : float or None, optional
        Smoothing factor. ``None``, the default, means ``s = m``. ``s = 0``
        requests interpolation, which scattered data often cannot support --
        expect a positive `ier` then. A negative `s` reaches FITPACK, which
        rejects it with ``ier = 10``.
    eps : float, optional
        Rank-determination threshold in ``(0, 1)``. Default 1e-16.
    nxest : int, optional
        Upper bound on the number of x knots. ``<= 0`` (the default)
        auto-picks ``max(kx + 1 + int(sqrt(m/2)), 2*(kx+1))``. Too small gives
        ``ier = 1``.
    nyest : int, optional
        Same for y knots; ``<= 0`` auto-picks. Default 0.
    xb, xe, yb, ye : float, optional
        The rectangle the fit is made over. ``None``, the default, means
        ``min(x)``, ``max(x)``, ``min(y)``, ``max(y)``. A NaN is a literal
        edge and reaches FITPACK unchanged.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    tx : 1-D float64 ndarray, length nx
        Knots in x, trimmed.
    ty : 1-D float64 ndarray, length ny
        Knots in y, trimmed.
    c : 1-D float64 ndarray, length ``(nx-kx-1)*(ny-ky-1)``
        Coefficients in the flat bivariate layout, feeding `bispev`,
        `bispeu`, `parder`, `pardeu` and `dblint` directly.
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on. ``-2``
        (least-squares polynomial) is common here when `s` is large.

    See Also
    --------
    scipy.interpolate.SmoothBivariateSpline : The scipy routine this mirrors.

    Notes
    -----
    The ``nxest``/``nyest`` bound only shows in the result when it BINDS,
    which on 200 scattered points is at ``s = 0``. A bound one knot too high
    there moves the knot vector and the surface with it.

    prange-safe: yes.
    """
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    z_ = np.ascontiguousarray(np.asarray(z, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    if (x_.ndim != 1 or y_.ndim != 1 or z_.ndim != 1 or w_.ndim != 1):
        raise ValueError("object too deep for desired array")
    m = len(x_)
    if s is None:
        s_v = float(m)
    else:
        s_v = float(s)
    if nxest <= 0:
        nxest = max(kx + 1 + int(np.sqrt(m / 2.0)), 2 * (kx + 1))
    if nyest <= 0:
        nyest = max(ky + 1 + int(np.sqrt(m / 2.0)), 2 * (ky + 1))
    nmax = max(nxest, nyest)
    uu = nxest - kx - 1
    vv = nyest - ky - 1
    km = max(kx, ky) + 1
    ne = max(nxest, nyest)
    bx = kx * vv + ky + 1
    by = ky * uu + kx + 1
    if bx <= by:
        b1 = bx
        b2 = b1 + vv - ky
    else:
        b1 = by
        b2 = b1 + uu - kx
    lwrk1 = (uu * vv * (2 + b1 + b2) +
             2 * (uu + vv + km * (m + ne) + ne - kx - ky) + b2 + 1)
    lwrk2 = uu * vv * (b2 + 1) + b2
    kwrk = m + (nxest - 2 * kx - 1) * (nyest - 2 * ky - 1)
    iopt = np.array(0, np.int32)
    m_ = np.array(m, np.int32)
    if xb is None:
        xb_v = x_.min()
    else:
        xb_v = float(xb)
    if xe is None:
        xe_v = x_.max()
    else:
        xe_v = float(xe)
    if yb is None:
        yb_v = y_.min()
    else:
        yb_v = float(yb)
    if ye is None:
        ye_v = y_.max()
    else:
        ye_v = float(ye)
    xb = np.array(xb_v, np.float64)
    xe = np.array(xe_v, np.float64)
    yb = np.array(yb_v, np.float64)
    ye = np.array(ye_v, np.float64)
    kx_ = np.array(kx, np.int32)
    ky_ = np.array(ky, np.int32)
    s_ = np.array(s_v, np.float64)
    nxest_ = np.array(nxest, np.int32)
    nyest_ = np.array(nyest, np.int32)
    nmax_ = np.array(nmax, np.int32)
    eps_ = np.array(eps, np.float64)
    nx = np.zeros(1, np.int32)
    ny = np.zeros(1, np.int32)
    tx = np.zeros(nmax, np.float64)
    ty = np.zeros(nmax, np.float64)
    c = np.zeros((nxest - kx - 1) * (nyest - ky - 1), np.float64)
    fp = np.zeros(1, np.float64)
    wrk1 = np.zeros(lwrk1, np.float64)
    lwrk1_ = np.array(lwrk1, np.int32)
    wrk2 = np.zeros(lwrk2, np.float64)
    lwrk2_ = np.array(lwrk2, np.int32)
    iwrk = np.zeros(kwrk, np.int32)
    kwrk_ = np.array(kwrk, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _surfit(iopt.ctypes.data, m_.ctypes.data, x_.ctypes.data, y_.ctypes.data,
            z_.ctypes.data, w_.ctypes.data, xb.ctypes.data, xe.ctypes.data,
            yb.ctypes.data, ye.ctypes.data, kx_.ctypes.data, ky_.ctypes.data,
            s_.ctypes.data, nxest_.ctypes.data, nyest_.ctypes.data,
            nmax_.ctypes.data, eps_.ctypes.data, nx.ctypes.data,
            tx.ctypes.data, ny.ctypes.data, ty.ctypes.data, c.ctypes.data,
            fp.ctypes.data, wrk1.ctypes.data, lwrk1_.ctypes.data,
            wrk2.ctypes.data, lwrk2_.ctypes.data, iwrk.ctypes.data,
            kwrk_.ctypes.data, tol_.ctypes.data,
            maxit_.ctypes.data, ier.ctypes.data)
    nnx = nx[0]
    nny = ny[0]
    return (tx[:nnx], ty[:nny], c[:(nnx - kx - 1) * (nny - ky - 1)],
            fp[0], ier[0])


# ---------------- polar domains (rad() callback) ----------------

# The ``rad`` boundary is supplied as a plain ``@njit`` function ``rad(v) ->
# float``. FITPACK passes the angle ``v`` BY REFERENCE, so the ``@cfunc`` it
# calls back through has signature ``float64(CPointer(float64))``. That adapter
# is built HERE, at typing time, from the user's scalar function and its
# ``.address`` frozen into the compiled body, so nothing user-facing takes an
# address. Established pattern; the twin is ``scijit.integrate._solve_ivp``'s
# ``_adapter_rhs``.
_RAD_SIG = types.float64(types.CPointer(types.float64))
_RAD_ADAPTERS = {}


def _adapter_rad(py):
    """``@cfunc(float64(CPointer(float64)))`` around a plain ``rad(v)->float``.

    Dereferences the by-reference angle FITPACK passes, so the user writes an
    ordinary scalar function. Cached per underlying Python function so the same
    ``rad`` reused across a fit and its `evapol` evaluations shares one adapter.
    """
    hit = _RAD_ADAPTERS.get(py)
    if hit is not None:
        return hit
    inner = njit(py)

    @cfunc(_RAD_SIG)
    def adapter(v_ptr):
        return inner(v_ptr[0])

    _RAD_ADAPTERS[py] = adapter
    return adapter


#: Refusal for a raw ``@cfunc`` address / integer pointer in the ``rad`` slot.
#: The address route is no longer accepted; pass the plain ``@njit`` function.
_RAD_ADDR_MSG = (
    "rad must be a plain @njit function rad(v) -> float giving the boundary "
    "radius r(v). A @cfunc address or raw integer pointer is not accepted: "
    "pass the @njit function itself.")


def _rad_address(rad):
    """Python-side: reject a raw address, build the ``@cfunc``, return its
    ``.address``. Accepts a plain ``@njit`` function or a bare Python
    function."""
    if isinstance(rad, (bool, int, np.integer)):
        raise ValueError(_RAD_ADDR_MSG)
    py = getattr(rad, 'py_func', rad)
    return _adapter_rad(py).address


@njit
def _polar_impl(x, y, z, w, rad_addr, s, eps, nuest, nvest,
                iopt2, iopt3, tol, maxit):
    """Internal engine for `polar`. Takes the ``rad`` boundary as a raw
    ``@cfunc`` address (``rad_addr``) and all arguments explicitly; the public
    `polar` builds the ``@cfunc`` and supplies the defaults. Not exported."""
    x_ = np.ascontiguousarray(np.asarray(x, np.float64))
    y_ = np.ascontiguousarray(np.asarray(y, np.float64))
    z_ = np.ascontiguousarray(np.asarray(z, np.float64))
    w_ = np.ascontiguousarray(np.asarray(w, np.float64))
    if (x_.ndim != 1 or y_.ndim != 1 or z_.ndim != 1 or w_.ndim != 1):
        raise ValueError("object too deep for desired array")
    m = len(x_)
    if nuest <= 0:
        nuest = 8 + int(np.sqrt(m / 2.0)) + 1
    if nvest <= 0:
        nvest = 8 + int(np.sqrt(m / 2.0)) + 1
    kk = nuest - 7
    ll = nvest - 7
    pp = 1 + iopt2 * (iopt2 + 3) // 2
    qq = kk + 2 - iopt2 - iopt3
    lwrk1 = (129 + 10 * kk + 21 * ll + kk * ll +
             (pp + ll * qq) * (1 + 8 * ll + pp) + 8 * m)
    lwrk2 = (pp + ll * qq + 1) * (4 * ll + pp) + pp + ll * qq
    kwrk = m + (nuest - 7) * (nvest - 7)
    iopt = np.zeros(3, np.int32)
    iopt[1] = iopt2
    iopt[2] = iopt3
    m_ = np.array(m, np.int32)
    s_ = np.array(s, np.float64)
    nuest_ = np.array(nuest, np.int32)
    nvest_ = np.array(nvest, np.int32)
    eps_ = np.array(eps, np.float64)
    nu = np.zeros(1, np.int32)
    nv = np.zeros(1, np.int32)
    tu = np.zeros(nuest, np.float64)
    tv = np.zeros(nvest, np.float64)
    u = np.zeros(m, np.float64)
    v = np.zeros(m, np.float64)
    c = np.zeros((nuest - 4) * (nvest - 4), np.float64)
    fp = np.zeros(1, np.float64)
    wrk1 = np.zeros(lwrk1, np.float64)
    lwrk1_ = np.array(lwrk1, np.int32)
    wrk2 = np.zeros(lwrk2, np.float64)
    lwrk2_ = np.array(lwrk2, np.int32)
    iwrk = np.zeros(kwrk, np.int32)
    kwrk_ = np.array(kwrk, np.int32)
    ier = np.zeros(1, np.int32)
    tol_ = np.array(_resolve_tol(tol, _TOL_E), np.float64)
    maxit_ = np.array(_resolve_maxit(maxit), np.int32)
    _polar(iopt.ctypes.data, m_.ctypes.data, x_.ctypes.data, y_.ctypes.data,
           z_.ctypes.data, w_.ctypes.data, rad_addr, s_.ctypes.data,
           nuest_.ctypes.data, nvest_.ctypes.data, eps_.ctypes.data,
           nu.ctypes.data, tu.ctypes.data, nv.ctypes.data, tv.ctypes.data,
           u.ctypes.data, v.ctypes.data, c.ctypes.data, fp.ctypes.data,
           wrk1.ctypes.data, lwrk1_.ctypes.data, wrk2.ctypes.data,
           lwrk2_.ctypes.data, iwrk.ctypes.data, kwrk_.ctypes.data,
           tol_.ctypes.data,
           maxit_.ctypes.data, ier.ctypes.data)
    nnu = nu[0]
    nnv = nv[0]
    return (tu[:nnu], tv[:nnv], c[:(nnu - 4) * (nnv - 4)], u, v,
            fp[0], ier[0])


def polar(x, y, z, w, rad, s, eps=1e-10, nuest=0, nvest=0,
          iopt2=0, iopt3=0, tol=-1.0, maxit=-1):
    """Scattered-data fit on a polar domain, FITPACK ``polar``.

    Fits ``z = s(u, v)`` where the domain boundary is an arbitrary
    star-shaped curve given by a CALLBACK ``r(v)``: the domain is
    ``x = u*r(v)*cos(v)``, ``y = u*r(v)*sin(v)`` for ``0 <= u <= 1``.

    Parameters
    ----------
    x : 1-D array_like of float, length m
        Cartesian abscissae of the scattered data points.
    y : 1-D array_like of float, length m
        Cartesian ordinates.
    z : 1-D array_like of float, length m
        Values at those points.
    w : 1-D array_like of float, length m
        Positive weights.
    rad : @njit function ``rad(v) -> float``
        The boundary radius r(v) as a plain ``@njit`` function of the angle
        ``v``. Works from Python and from inside ``@njit``.
    s : float
        Smoothing factor; required, with no default.
    eps : float, optional
        Rank-determination threshold in ``(0, 1)``. Default 1e-10.
    nuest : int, optional
        Upper bound on the number of u knots; ``<= 0`` (the default)
        auto-picks ``8 + int(sqrt(m/2)) + 1``.
    nvest : int, optional
        Same for v knots; ``<= 0`` auto-picks. Default 0.
    iopt2 : int, optional
        Constraint at the ORIGIN: 0 = none, 1 = the spline value at the
        origin is a free parameter shared by all directions (C0), 2 = C1 at
        the origin as well. Default 0.
    iopt3 : int, optional
        0 = none, 1 = the SURFACE ITSELF vanishes on the boundary curve.
        Default 0. This constrains the value, not a derivative.
    tol : float, optional
        Width of the band FITPACK accepts a fit in: the knot search stops
        when ``abs(fp - s) <= tol*s``. ``-1.0``, the default, resolves to
        ``0.1e-02``, the value the Fortran source set internally, so a call
        that leaves it alone runs the computation it ran before. A smaller
        value narrows the set of knot placements the routine will accept and
        costs iterations. Any other value ``<= 0`` raises ``ValueError``.
    maxit : int, optional
        Iteration cap on that same search; reaching it returns ``ier = 3``.
        ``-1``, the default, resolves to 20, the value the Fortran source
        set internally. Values ``< 1`` raise ``ValueError``.

    Returns
    -------
    tu : 1-D float64 ndarray, length nu
        Knots in the radial parameter u, trimmed.
    tv : 1-D float64 ndarray, length nv
        Knots in the angular parameter v, trimmed.
    c : 1-D float64 ndarray, length ``(nu-4)*(nv-4)``
        Coefficients in the flat bivariate layout. Evaluate with `evapol`,
        which applies the same ``rad`` mapping -- NOT with `bispev`, unless
        the query points must be converted to (u, v) first.
    u : 1-D float64 ndarray, length m
        The radial parameter FITPACK computed for each input point.
    v : 1-D float64 ndarray, length m
        The angular parameter for each input point.
    fp : float
        Weighted sum of squared residuals.
    ier : int
        FITPACK status; see the module docstring. Not raised on.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    BICUBIC ONLY. The seven-element return is the longest in this module --
    note that ``u`` and ``v`` come AFTER ``c``, unlike `parcur`, where ``u``
    comes first.

    **The default `nuest` and `nvest` carry a deliberate ``+1``.** They are
    ``8 + int(sqrt(m/2)) + 1``. FITPACK's own recommendation, and `sphere`
    in this module, use that expression without the trailing ``+1``. The
    extra knot of room is kept, and it changes answers. Pass `nuest` and
    `nvest` explicitly to pin either spelling.

    Characterised over 210 fixtures (three closed-form surfaces on the unit
    disc, ``m`` from 40 to 800, ``s`` from 1e-4 to 5), comparing the two
    spellings against the function each fixture was generated from at 400
    points off the fitting set. Every proportion below is a property of that
    fixture set, not of the ``+1``; an earlier sweep on different fixtures
    gave visibly different shares.

    What it buys. The fit converges more often: ``ier`` went from 1 to 0 on
    39 of the 210. An ``ier = 1`` fit has run out of knots and stopped
    tracking `s` at all -- on one fixture its ``fp`` stayed at 0.2609 for
    every ``s`` from 1e-4 to 0.2, while the ``+1`` fit reached ``fp = s`` on
    all five.

    What it costs. It is NOT a strict superset: on one fixture of the 210
    (``m = 80``, ``s = 1e-4``) it returned ``ier = 4``, a hard failure,
    where the bare form returned a usable ``ier = 1``. Where both spellings
    already converged the surfaces are usually identical (88 of 102 rows),
    but where the answer changes at all the ``+1`` is the worse fit more
    often than the better one: 83 rows worse against 39 better, geometric
    mean of the error ratio 1.791 over the 122 rows that differ. Splitting
    by `s` shows why -- the ratio is 2.657 at ``s = 1e-4`` and 0.6434 at
    ``s = 5``, so the extra room helps when the smoothing request is
    reasonable and hurts when `s` is small enough that honouring it makes
    the surface oscillate. Of the 39 rescued fits, 15 are better against the
    underlying function than the ``ier = 1`` result they replaced.

    Knot counts differ on 122 of 210. More room does not mean more knots:
    ``nu`` moved by -10 to +1, and was LOWER with the ``+1`` on 31 rows.
    ``nv`` moved by 0 or +2, never down.

    Workspace grows 13.4% to 47.2% (median 26.5%), largest at small ``m``.
    Wall clock over the 210 fits was 2.9600 s against 2.6064 s, a ratio of
    1.1357, with per-call medians of 2.400 ms and 2.004 ms.

    prange-safe: concurrent fits do not clash, provided the ``rad`` function
    is itself reentrant.
    """
    return _polar_impl(x, y, z, w, _rad_address(rad), s, eps, nuest, nvest,
                       iopt2, iopt3, tol, maxit)


@overload(polar)
def _polar_ovl(x, y, z, w, rad, s, eps=1e-10, nuest=0, nvest=0,
               iopt2=0, iopt3=0, tol=-1.0, maxit=-1):
    """Compiled body for `polar`. Builds the ``rad`` ``@cfunc`` at typing time
    from the plain ``@njit`` function and freezes its address into the body."""
    if isinstance(rad, types.Integer):
        raise TypingError(_RAD_ADDR_MSG)
    if not isinstance(rad, types.Dispatcher):
        return None
    addr = _adapter_rad(rad.dispatcher.py_func).address

    def impl(x, y, z, w, rad, s, eps=1e-10, nuest=0, nvest=0,
             iopt2=0, iopt3=0, tol=-1.0, maxit=-1):
        return _polar_impl(x, y, z, w, addr, s, eps, nuest, nvest,
                           iopt2, iopt3, tol, maxit)
    return impl


@njit
def _evapol_impl(tu, tv, c, rad_addr, x, y):
    """Internal engine for `evapol`. Takes the ``rad`` boundary as a raw
    ``@cfunc`` address (``rad_addr``); the public `evapol` builds it. Not
    exported."""
    tu_ = np.ascontiguousarray(np.asarray(tu, np.float64))
    tv_ = np.ascontiguousarray(np.asarray(tv, np.float64))
    c_ = np.ascontiguousarray(np.asarray(c, np.float64))
    if tu_.ndim != 1 or tv_.ndim != 1 or c_.ndim != 1:
        raise ValueError("object too deep for desired array")
    nu = np.array(len(tu_), np.int32)
    nv = np.array(len(tv_), np.int32)
    x_ = np.array(x, np.float64)
    y_ = np.array(y, np.float64)
    res = np.zeros(1, np.float64)
    _evapol(tu_.ctypes.data, nu.ctypes.data, tv_.ctypes.data, nv.ctypes.data,
            c_.ctypes.data, rad_addr, x_.ctypes.data, y_.ctypes.data,
            res.ctypes.data)
    return res[0]


def evapol(tu, tv, c, rad, x, y):
    """Evaluate a polar-domain spline, FITPACK ``evapol``.

    The evaluator for a `polar` fit. It is here rather than in
    ``evaluators.py`` because it needs the same ``rad`` callback.

    Parameters
    ----------
    tu : 1-D array_like of float
        Knots in the radial parameter, as returned by `polar`.
    tv : 1-D array_like of float
        Knots in the angular parameter.
    c : 1-D array_like of float
        Coefficients from `polar`, flat bivariate layout.
    rad : @njit function ``rad(v) -> float``
        The SAME boundary function used for the fit, as a plain ``@njit``
        function of the angle ``v``. Passing a different boundary silently
        evaluates a different surface.
    x, y : float
        A single CARTESIAN query point. This routine takes scalars, not
        arrays -- loop over the points (the loop is compiled, so that is
        cheap inside ``@njit``).

    Returns
    -------
    res : float
        The spline value at ``(x, y)``. Points outside the boundary curve are
        extrapolated; there is no range flag.

    Notes
    -----
    ``scipy.interpolate`` publishes no name for this routine.

    Dierckx wrote this as a Fortran FUNCTION; the bind(c) wrapper turns it
    into a subroutine with a ``res`` out-argument.

    prange-safe: yes, provided the ``rad`` function is reentrant.
    """
    return _evapol_impl(tu, tv, c, _rad_address(rad), x, y)


@overload(evapol)
def _evapol_ovl(tu, tv, c, rad, x, y):
    """Compiled body for `evapol`. Builds the ``rad`` ``@cfunc`` at typing time
    from the plain ``@njit`` function and freezes its address into the body."""
    if isinstance(rad, types.Integer):
        raise TypingError(_RAD_ADDR_MSG)
    if not isinstance(rad, types.Dispatcher):
        return None
    addr = _adapter_rad(rad.dispatcher.py_func).address

    def impl(tu, tv, c, rad, x, y):
        return _evapol_impl(tu, tv, c, addr, x, y)
    return impl


# Public names: the raw FITPACK fitting routines, argument-for-argument
# as Dierckx defines them. The scipy-shaped wrappers live in
# scijit.interpolate itself.
__all__ = [
    'curfit', 'curfit_lsq', 'percur', 'percur_lsq', 'parcur', 'clocur',
    'concur',
    'cocosp', 'concon', 'regrid', 'parsur', 'pogrid', 'spgrid', 'sphere',
    'surfit', 'polar', 'evapol',
]
