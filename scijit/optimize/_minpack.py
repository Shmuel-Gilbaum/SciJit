"""Numba-callable wrappers for MINPACK (nonlinear systems & least squares).

Backed by libminpack (fortran-lang/minpack modern module + bind(c)
wrappers.f90 with module-variable callback adapters, no trampolines, no
executable stack, loads on hardened distros).

Signature-compatible with Nicholas Wogan's NumbaMinpack API (`minpack_sig`,
`hybrd`, `lmdif`, same return convention), extended with the analytic-
Jacobian drivers `hybrj` and `lmder`.

User callbacks are numba @cfunc functions; pass `.address` as `funcptr`.
Fortran passes by reference, so every cfunc argument is a pointer
(indexing auto-derefs inside numba):

    minpack_sig      void(x*, fvec*, args*)                  hybrd / lmdif
    minpack_jac_sig  void(x*, fvec*, fjac*, iflag*, args*)   hybrj / lmder

Two-phase protocol of the Jacobian drivers (read iflag[0]):
    iflag == 1 -> fill fvec only
    iflag == 2 -> fill fjac only, COLUMN-major flattened:
                  fjac[i + j*ld] = d f_i / d x_j
                  ld = n for hybrj, ld = m for lmder

The Fortran wrapper parks the callback in a module variable carrying
`!$omp threadprivate`, so each thread gets its own slot: concurrent
solves are safe and these are callable from a `numba.prange` loop.
Nested and sequential solves inside one @njit function are fine too.
"""
import operator
import warnings
from collections import namedtuple

from numba import carray, cfunc, njit, objmode, prange, types, typeof
from numba.core.errors import TypingError
from numba.extending import overload
import numpy as np
from .._lib._load import load
from .._lib._typing import _K_BOOL, _K_FLOAT, _K_INT, _is_none, _lit_bool, _lit_str
from .._lib._typing import _arg_kinds as _arg_kinds_base
from .._lib._typing import _arg_kinds_ty as _arg_kinds_ty_base

minpack_sig = types.void(types.CPointer(types.double),     # x         (in)
                         types.CPointer(types.double),     # fvec      (out)
                         types.CPointer(types.double))     # args      (in)

"""numba signature of a MINPACK residual callback --
``void(float64* x, float64* fvec, float64* args)``.

Every argument is a pointer because Fortran passes by reference; indexing
auto-dereferences inside numba. ``x`` has length n, ``fvec`` length n for root
finding or m for least squares, and ``args`` whatever was passed. The callback
returns nothing: it **writes** the residuals into ``fvec``.

Used by :func:`hybrd`, :func:`lmdif`, :func:`fsolve`, :func:`leastsq`
and :func:`root` on their derivative-free paths. Identical
to NumbaMinpack's ``minpack_sig``, so an existing NumbaMinpack callback works
unchanged.

Callback **style B**: build a ``@cfunc`` and pass its ``.address``. Take
``.address`` at PYTHON level, never inside ``@njit``.

Solving ``x0**2 + x1**2 = args[0]`` and ``x0 - x1 = args[1]``::

    import numpy as np
    from numba import cfunc, njit
    from scijit.optimize import fsolve
    from scijit.optimize._minpack import minpack_sig

    @cfunc(minpack_sig)
    def resid(x, fvec, args):
        fvec[0] = x[0] * x[0] + x[1] * x[1] - args[0]
        fvec[1] = x[0] - x[1] - args[1]

    ptr = resid.address              # at PYTHON level, once

    @njit
    def solve():
        return fsolve(ptr, np.array([1.0, 1.0]), np.array([8.0, 0.0]))

    solve()                          # -> array([2., 2.])

Read parameters out of ``args`` rather than from a captured global: numba
freezes a ``@cfunc``'s globals at compile time.

A wrong-arity address is undefined behaviour rather than an error, so the
signature the ``@cfunc`` was built with must match the entry point being
called.
"""

minpack_jac_sig = types.void(types.CPointer(types.double),   # x       (in)
                             types.CPointer(types.double),   # fvec    (out)
                             types.CPointer(types.double),   # fjac    (out)
                             types.CPointer(types.int32),    # iflag   (in)
                             types.CPointer(types.double))   # args    (in)

"""numba signature of a MINPACK residual-and-Jacobian callback --
``void(float64* x, float64* fvec, float64* fjac, int32* iflag,
float64* args)``.

One callback serves both jobs, selected by ``iflag[0]``: ``1`` means write the
residuals into ``fvec``, ``2`` means write the Jacobian into ``fjac``. ``fjac``
is COLUMN-MAJOR and flat, so the derivative of residual ``i`` with respect to
variable ``j`` lives at ``fjac[i + j * m]``.

Used by :func:`hybrj`, :func:`lmder`, :func:`fsolve` with `fprime`,
:func:`leastsq` with `Dfun` and :func:`root` with `jac`.

Callback **style B**::

    import numpy as np
    from numba import cfunc, njit
    from scijit.optimize import fsolve
    from scijit.optimize._minpack import minpack_jac_sig

    @cfunc(minpack_jac_sig)
    def resid_jac(x, fvec, fjac, iflag, args):
        if iflag[0] == 1:
            fvec[0] = x[0] * x[0] + x[1] * x[1] - args[0]
            fvec[1] = x[0] - x[1] - args[1]
        else:                        # column-major, leading dimension 2
            fjac[0] = 2.0 * x[0]     # d f0 / d x0   [0 + 0*2]
            fjac[1] = 1.0            # d f1 / d x0   [1 + 0*2]
            fjac[2] = 2.0 * x[1]     # d f0 / d x1   [0 + 1*2]
            fjac[3] = -1.0           # d f1 / d x1   [1 + 1*2]

    ptr = resid_jac.address

    @njit
    def solve():
        return fsolve(ptr, np.array([1.0, 1.0]), np.array([8.0, 0.0]), ptr)

    solve()                          # -> array([2., 2.])

The same address is passed as both `func` and `fprime`, because this one
callback carries both jobs.
"""

from .._probe import wrote_residual as _wrote_residual
from .._probe import wrote_jac_residual as _wrote_jac_residual
from .._probe import residual_status as _resid_status
from .._probe import jac_residual_status as _jac_status
from .._probe import _call_p3, SENTINEL_A

_lib, _sig = load(__file__, "libminpack")


# argument counts recounted against wrappers.f90, a miscount surfaces as
# a cryptic numba ExternalFunctionPointer typing error -- but only when it
# DISAGREES with the call site.  A count that is wrong in both places raises
# nothing at all and runs into undefined behaviour, so a green result is not
# evidence that the arity is right.
_hybrd = _sig(_lib.hybrd_wrapper, 11)
_lmdif = _sig(_lib.lmdif_wrapper, 13)
_hybrj1 = _sig(_lib.hybrj1_wrapper, 12)
_lmder1 = _sig(_lib.lmder1_wrapper, 14)
# The three below have no caller in this module. They bind the entry points
# `lmstr`, `chkder` and `enorm` reach, and those three moved to the
# development-only `legacy.optimize` on 2026-08-11, which imports these back
# out. The bindings stay here, beside libminpack.
_lmstr1 = _sig(_lib.lmstr1_wrapper, 14)
_chkder = _sig(_lib.chkder_wrapper, 10)
_enorm = _sig(_lib.enorm_wrapper, 3)

# argument counts recounted against wrappers_scipy.f90
_lmdif_sp = _sig(_lib.lmdif_sp_wrapper, 21)
_lmder_sp = _sig(_lib.lmder_sp_wrapper, 21)


@njit
def hybrd(funcptr, x_init, args=np.array([0.0]), tol=1.49012e-8, maxfev=0,
          validate=True):
    """Roots of n nonlinear equations in n variables.

    Modified Powell hybrid method with a forward-difference Jacobian.

    Fortran wrapper for MINPACK's ``hybrd``. No scipy counterpart: scipy
    reaches this driver through ``fsolve``, which is the documented entry
    point. Use this when a raw ``.address`` and the untranslated ``info``
    are wanted.

    **Callback: a ``@cfunc``, passed by ``.address``.** ``funcptr`` is the ``.address`` of a
    ``@cfunc(minpack_sig)``, i.e. ``void(double* x, double* fvec,
    double* args)``. See :data:`minpack_sig` for a worked example.

    Parameters
    ----------
    funcptr : int
        Address of a ``@cfunc(minpack_sig)`` callback. Take ``.address``
        at Python level, outside the ``@njit`` function.
    x_init : np.ndarray, float64, shape (n,)
        Initial guess. Copied, so the caller's array is not modified.
    args : np.ndarray, float64, shape (k,), optional
        Passed straight through to the callback's third pointer. May be
        empty in effect; the default ``np.array([0.0])`` is a one-element
        dummy. Wrapped in ``np.ascontiguousarray`` internally, so a
        strided view is safe.
    tol : float, optional
        Relative error between two consecutive iterates at termination,
        MINPACK's ``xtol``. Default 1.49012e-8. Must be >= 0.
    maxfev : int, optional
        Maximum callback evaluations. Default 0, which selects the hybrd1
        default ``200*(n+1)``.

    validate : bool, optional
        Raise on a bad solution rather than only reporting it, default
        True.  The CHECK runs either way and always sets ``info``; this
        flag decides whether a nonzero status also raises ``ValueError``.
        Set False for a parameter sweep, where an exception would strand
        the run, and filter on ``success`` instead.

        Three stages, cheapest first.  Any nonzero entry in the returned
        ``fvec`` settles it with no extra callback evaluation, which is the
        ordinary case.  If ``fvec`` is all zero the residual is evaluated
        again at the solution: a fresh value that is NOT zero disagrees
        with what the solver reported, so it did not converge
        (``info = -2``).  If the fresh value is zero too, one perturbation
        per component follows; all zero again means the residual carries no
        information and every point is a root (``info = -1``).

        The empty-output-pointer check is separate, unconditional, and
        always raises: a callback that never writes its output is a
        programming error, identical for every parameter, so it cannot
        strand a sweep.

    Returns
    -------
    x : np.ndarray, float64, shape (n,)
        Final iterate.
    fvec : np.ndarray, float64, shape (n,)
        Residuals at ``x``.
    success : bool
        ``info == 1``.
    info : int32
        MINPACK termination code, hybrd1 numbering (the Fortran wrapper
        replicates the hybrd1 body, which folds hybrd's raw ``info = 5``
        into 4, so 5 is never returned):

        * 0, improper input parameters.
        * 1, relative error between two consecutive iterates <= ``tol``.
        * 2, ``maxfev`` reached or exceeded.
        * 3, ``tol`` too small, no further improvement possible.
        * 4, the iteration is not making good progress.

    Notes
    -----
    Returns a 4-tuple instead of scipy's bare ``x``/``OptimizeResult``;
    numba has no result objects.

    The callback is a compiled ``@cfunc``, so the floating-point expression
    it evaluates is fixed at compile time. Where a Python residual reorders
    the same arithmetic, the root moves by the last few ulp: measured
    1.3e-15 max |dx| on a reordered variant of the same residual.

    Safe to call from a ``numba.prange`` loop: the Fortran wrapper parks
    ``funcptr`` in a module variable carrying ``!$omp threadprivate``, so
    each thread gets its own slot.
    """
    if not _wrote_residual(funcptr, x_init,
                                       x_init.size, args):
        raise ValueError(
            "the residual callback never wrote fvec. Check the "
            "@cfunc signature and argument order")
    n = np.int32(x_init.size)
    n_ = np.array(n, np.int32)
    x = x_init.astype(np.float64).copy()
    fvec = np.zeros(n, np.float64)
    tol_ = np.array(tol, np.float64)
    if maxfev <= 0:
        maxfev_ = np.array(200 * (n + 1), np.int32)
    else:
        maxfev_ = np.array(maxfev, np.int32)
    info = np.zeros(1, np.int32)
    lwa = np.int32((n * (3 * n + 13)) // 2 + 2)
    wa = np.zeros(lwa, np.float64)
    lwa_ = np.array(lwa, np.int32)
    # ascontiguousarray: a strided view's .ctypes.data points into memory
    # the Fortran side would read CONTIGUOUSLY, silently wrong values.
    args_ = np.ascontiguousarray(np.asarray(args, np.float64))
    nargs = np.array(args_.size, np.int32)

    _hybrd(funcptr, n_.ctypes.data, x.ctypes.data, fvec.ctypes.data,
           tol_.ctypes.data, maxfev_.ctypes.data, info.ctypes.data,
           wa.ctypes.data, lwa_.ctypes.data,
           args_.ctypes.data, nargs.ctypes.data)

    _status = _resid_status(funcptr, x, fvec, x_init.size, args)
    if _status != 0:
        # 1 = residual identically zero, 2 = the residual the solver
        # reported disagrees with a fresh evaluation at x.  The status
        # is set either way so a sweep can filter on it; validate
        # decides whether to also raise.
        if validate:
            if _status == 1:
                raise ValueError(
                    "the residual is identically zero, so every point is a "
                    "root and this result is meaningless. Check the callback; "
                    "pass validate=False to get info=-1 instead")
            raise ValueError(
                "the residual the solver reported at the solution does not "
                "match a fresh evaluation there, so it did not converge. "
                "Check the callback for state or side effects; pass "
                "validate=False to get info=-2 instead")
        return x, fvec, False, np.int32(-_status)
    return x, fvec, info[0] == 1, info[0]


@njit
def lmdif(funcptr, x_init, neqs, args=np.array([0.0]), tol=1.49012e-8,
          maxfev=0, validate=True):
    """Minimize the sum of squares of m functions in n variables.

    Levenberg-Marquardt with a forward-difference Jacobian.

    Fortran wrapper for MINPACK's ``lmdif``. No scipy counterpart: scipy
    reaches this driver through ``leastsq``, which is the documented entry
    point. Use this when ``maxfev`` passthrough or the raw ``info`` is
    wanted.

    **Callback: a ``@cfunc``, passed by ``.address``.** ``funcptr`` is the ``.address`` of a
    ``@cfunc(minpack_sig)``; ``fvec`` has length ``neqs``, not n. See
    :data:`minpack_sig`.

    Parameters
    ----------
    funcptr : int
        Address of a ``@cfunc(minpack_sig)`` callback writing ``neqs``
        residuals.
    x_init : np.ndarray, float64, shape (n,)
        Initial guess. Copied.
    neqs : int
        Number of residuals m. Must satisfy ``m >= n``; MINPACK returns
        ``info = 0`` otherwise. Required: a ``@cfunc`` address carries no
        shape information, so there is nothing to infer it from.
    args : np.ndarray, float64, shape (k,), optional
        Passed through to the callback. Default a one-element dummy.
    tol : float, optional
        Used for BOTH ``ftol`` and ``xtol``. Default 1.49012e-8.
        ``gtol`` is fixed at 0.0 by the lmdif1 driver.
    maxfev : int, optional
        Maximum callback evaluations. Default 0, which selects the lmdif1
        default ``200*(n+1)``.

    validate : bool, optional
        Raise on a bad solution rather than only reporting it, default
        True.  The CHECK runs either way and always sets ``info``; this
        flag decides whether a nonzero status also raises ``ValueError``.
        Set False for a parameter sweep, where an exception would strand
        the run, and filter on ``success`` instead.

        Three stages, cheapest first.  Any nonzero entry in the returned
        ``fvec`` settles it with no extra callback evaluation, which is the
        ordinary case.  If ``fvec`` is all zero the residual is evaluated
        again at the solution: a fresh value that is NOT zero disagrees
        with what the solver reported, so it did not converge
        (``info = -2``).  If the fresh value is zero too, one perturbation
        per component follows; all zero again means the residual carries no
        information and every point is a root (``info = -1``).

        The empty-output-pointer check is separate, unconditional, and
        always raises: a callback that never writes its output is a
        programming error, identical for every parameter, so it cannot
        strand a sweep.

    Returns
    -------
    x : np.ndarray, float64, shape (n,)
        Final parameter estimate.
    fvec : np.ndarray, float64, shape (m,)
        Residuals at ``x``.
    success : bool
        ``1 <= info <= 4``, the same acceptance rule scipy.leastsq
        applies to its ``ier``.
    info : int32
        MINPACK termination code, lmdif1 numbering (the Fortran wrapper
        replicates the lmdif1 body, which folds raw ``info = 8`` into 4,
        so 8 is never returned):

        * 0, improper input parameters (including ``m < n``).
        * 1, actual and predicted relative reductions in the sum of
          squares are both <= ``tol``.
        * 2, relative error between two consecutive iterates <= ``tol``.
        * 3, conditions 1 and 2 both hold.
        * 4, ``fvec`` is orthogonal to the Jacobian columns to machine
          precision.
        * 5, ``maxfev`` reached or exceeded.
        * 6, ``tol`` too small, no further reduction in the sum of
          squares possible.
        * 7, ``tol`` too small, no further improvement in ``x``
          possible.

    Notes
    -----
    Returns 4 values where scipy returns ``(x, ier)`` or a longer tuple
    with ``full_output``; the covariance matrix is not produced here (see
    :func:`curve_fit` for a finite-difference estimate).

    Agreement with scipy.leastsq: exactly 0.0 max |dx| on the suite's fit
    problem (``tests/optimize/test_optimize.py``, "leastsq == scipy.leastsq"), and
    0.0 again on a 2-parameter exponential fit measured while writing
    these docs. Same Fortran, same defaults.

    Safe to call from a ``numba.prange`` loop: the Fortran wrapper parks
    ``funcptr`` in a module variable carrying ``!$omp threadprivate``, so
    each thread gets its own slot.
    """
    if not _wrote_residual(funcptr, x_init, neqs, args):
        raise ValueError(
            "the residual callback never wrote fvec. Check the "
            "@cfunc signature and argument order")
    m = np.int32(neqs)
    n = np.int32(x_init.size)
    m_ = np.array(m, np.int32)
    n_ = np.array(n, np.int32)
    x = x_init.astype(np.float64).copy()
    fvec = np.zeros(m, np.float64)
    tol_ = np.array(tol, np.float64)
    if maxfev <= 0:
        maxfev_ = np.array(200 * (n + 1), np.int32)
    else:
        maxfev_ = np.array(maxfev, np.int32)
    info = np.zeros(1, np.int32)
    iwa = np.zeros(n, np.int32)
    lwa = np.int32(m * n + 5 * n + m + 2)
    wa = np.zeros(lwa, np.float64)
    lwa_ = np.array(lwa, np.int32)
    # ascontiguousarray: a strided view's .ctypes.data points into memory
    # the Fortran side would read CONTIGUOUSLY, silently wrong values.
    args_ = np.ascontiguousarray(np.asarray(args, np.float64))
    nargs = np.array(args_.size, np.int32)

    _lmdif(funcptr, m_.ctypes.data, n_.ctypes.data, x.ctypes.data,
           fvec.ctypes.data, tol_.ctypes.data, maxfev_.ctypes.data,
           info.ctypes.data, iwa.ctypes.data, wa.ctypes.data,
           lwa_.ctypes.data, args_.ctypes.data, nargs.ctypes.data)

    _status = _resid_status(funcptr, x, fvec, neqs, args)
    if _status != 0:
        # 1 = residual identically zero, 2 = the residual the solver
        # reported disagrees with a fresh evaluation at x.  The status
        # is set either way so a sweep can filter on it; validate
        # decides whether to also raise.
        if validate:
            if _status == 1:
                raise ValueError(
                    "the residual is identically zero, so every point is a "
                    "root and this result is meaningless. Check the callback; "
                    "pass validate=False to get info=-1 instead")
            raise ValueError(
                "the residual the solver reported at the solution does not "
                "match a fresh evaluation there, so it did not converge. "
                "Check the callback for state or side effects; pass "
                "validate=False to get info=-2 instead")
        return x, fvec, False, np.int32(-_status)
    return x, fvec, 1 <= info[0] <= 4, info[0]


@njit
def hybrj(funcptr, x_init, args=np.array([0.0]), tol=1.49012e-8,
          validate=True):
    """Roots of n nonlinear equations with an analytic Jacobian.

    MINPACK ``hybrj1``, reached from `fsolve` with ``fprime`` and from `root`
    with ``method='hybr'`` and ``jac``.

    **Callback: a ``@cfunc``, passed by ``.address``.** ``funcptr`` is the ``.address`` of a
    ``@cfunc(minpack_jac_sig)`` serving both phases; see
    :data:`minpack_jac_sig`.

    Parameters
    ----------
    funcptr : int
        Address of a ``@cfunc(minpack_jac_sig)``. On ``iflag[0] == 1``
        fill ``fvec`` (length n); on ``iflag[0] == 2`` fill ``fjac``,
        column-major n x n flattened, ``fjac[i + j*n] = d f_i / d x_j``.
    x_init : np.ndarray, float64, shape (n,)
        Initial guess. Copied.
    args : np.ndarray, float64, shape (k,), optional
        Passed through to the callback. Default a one-element dummy.
    tol : float, optional
        Relative error between consecutive iterates at termination.
        Default 1.49012e-8, matching scipy's ``xtol``.

    validate : bool, optional
        Raise on a bad solution rather than only reporting it, default
        True.  The CHECK runs either way and always sets ``info``; this
        flag decides whether a nonzero status also raises ``ValueError``.
        Set False for a parameter sweep, where an exception would strand
        the run, and filter on ``success`` instead.

        Three stages, cheapest first.  Any nonzero entry in the returned
        ``fvec`` settles it with no extra callback evaluation, which is the
        ordinary case.  If ``fvec`` is all zero the residual is evaluated
        again at the solution: a fresh value that is NOT zero disagrees
        with what the solver reported, so it did not converge
        (``info = -2``).  If the fresh value is zero too, one perturbation
        per component follows; all zero again means the residual carries no
        information and every point is a root (``info = -1``).

        The empty-output-pointer check is separate, unconditional, and
        always raises: a callback that never writes its output is a
        programming error, identical for every parameter, so it cannot
        strand a sweep.

    Returns
    -------
    x : np.ndarray, float64, shape (n,)
        Final iterate.
    fvec : np.ndarray, float64, shape (n,)
        Residuals at ``x``.
    success : bool
        ``info == 1``.
    info : int32
        hybrj1 code:

        * 0, improper input parameters.
        * 1, estimated relative error between ``x`` and the solution is
          at most ``tol``.
        * 2, callback calls with ``iflag == 1`` reached ``100*(n+1)``.
        * 3, ``tol`` too small, no further improvement possible.
        * 4, the iteration is not making good progress.

    Notes
    -----
    There is NO ``maxfev`` parameter: the hybrj1 driver hardcodes the
    evaluation budget at ``100*(n+1)``. This differs from :func:`hybrd`,
    whose wrapper replicates the hybrd1 body specifically to keep a
    ``maxfev`` passthrough.

    Verified while writing these docs to land on the same root as
    :func:`hybrd` to exactly 0.0 on a 2x2 system with an exact Jacobian.

    Safe to call from a ``numba.prange`` loop: the module variable holding
    the callback carries ``!$omp threadprivate``.
    """
    if not _wrote_jac_residual(
            funcptr, x_init, x_init.size,
            x_init.size * x_init.size, args):
        raise ValueError(
            "the callback never wrote fvec at iflag=1. The argument "
            "order is (x, fvec, fjac, iflag, args)")
    n = np.int32(x_init.size)
    n_ = np.array(n, np.int32)
    x = x_init.astype(np.float64).copy()
    fvec = np.zeros(n, np.float64)
    fjac = np.zeros(n * n, np.float64)
    ldfjac = np.array(n, np.int32)
    tol_ = np.array(tol, np.float64)
    info = np.zeros(1, np.int32)
    lwa = np.int32((n * (n + 13)) // 2 + 2)
    wa = np.zeros(lwa, np.float64)
    lwa_ = np.array(lwa, np.int32)
    # ascontiguousarray: a strided view's .ctypes.data points into memory
    # the Fortran side would read CONTIGUOUSLY, silently wrong values.
    args_ = np.ascontiguousarray(np.asarray(args, np.float64))
    nargs = np.array(args_.size, np.int32)

    _hybrj1(funcptr, n_.ctypes.data, x.ctypes.data, fvec.ctypes.data,
            fjac.ctypes.data, ldfjac.ctypes.data, tol_.ctypes.data,
            info.ctypes.data, wa.ctypes.data, lwa_.ctypes.data,
            args_.ctypes.data, nargs.ctypes.data)

    _status = _jac_status(funcptr, x, fvec, x_init.size, x_init.size * x_init.size, args)
    if _status != 0:
        # 1 = residual identically zero, 2 = the residual the solver
        # reported disagrees with a fresh evaluation at x.  The status
        # is set either way so a sweep can filter on it; validate
        # decides whether to also raise.
        if validate:
            if _status == 1:
                raise ValueError(
                    "the residual is identically zero, so every point is a "
                    "root and this result is meaningless. Check the callback; "
                    "pass validate=False to get info=-1 instead")
            raise ValueError(
                "the residual the solver reported at the solution does not "
                "match a fresh evaluation there, so it did not converge. "
                "Check the callback for state or side effects; pass "
                "validate=False to get info=-2 instead")
        return x, fvec, False, np.int32(-_status)
    return x, fvec, info[0] == 1, info[0]


@njit
def lmder(funcptr, x_init, neqs, args=np.array([0.0]), tol=1.49012e-8,
          validate=True):
    """Least squares of m functions in n variables with an analytic Jacobian.

    MINPACK ``lmder1``, reached from `leastsq` with ``Dfun`` and from `root`
    with ``method='lm'`` and ``jac``.

    **Callback: a ``@cfunc``, passed by ``.address``.** ``funcptr`` is the ``.address`` of a
    ``@cfunc(minpack_jac_sig)``; see :data:`minpack_jac_sig`.

    Parameters
    ----------
    funcptr : int
        Address of a ``@cfunc(minpack_jac_sig)``. On ``iflag[0] == 1``
        fill ``fvec`` (length m); on ``iflag[0] == 2`` fill ``fjac``,
        column-major m x n flattened, ``fjac[i + j*m] = d f_i / d x_j``.
        Note the leading dimension is m here, n in :func:`hybrj`.
    x_init : np.ndarray, float64, shape (n,)
        Initial guess. Copied.
    neqs : int
        Number of residuals m; must satisfy ``m >= n``.
    args : np.ndarray, float64, shape (k,), optional
        Passed through to the callback. Default a one-element dummy.
    tol : float, optional
        Sets both ``ftol`` and ``xtol``. Default 1.49012e-8,
        matching scipy.

    validate : bool, optional
        Raise on a bad solution rather than only reporting it, default
        True.  The CHECK runs either way and always sets ``info``; this
        flag decides whether a nonzero status also raises ``ValueError``.
        Set False for a parameter sweep, where an exception would strand
        the run, and filter on ``success`` instead.

        Three stages, cheapest first.  Any nonzero entry in the returned
        ``fvec`` settles it with no extra callback evaluation, which is the
        ordinary case.  If ``fvec`` is all zero the residual is evaluated
        again at the solution: a fresh value that is NOT zero disagrees
        with what the solver reported, so it did not converge
        (``info = -2``).  If the fresh value is zero too, one perturbation
        per component follows; all zero again means the residual carries no
        information and every point is a root (``info = -1``).

        The empty-output-pointer check is separate, unconditional, and
        always raises: a callback that never writes its output is a
        programming error, identical for every parameter, so it cannot
        strand a sweep.

    Returns
    -------
    x : np.ndarray, float64, shape (n,)
        Final parameter estimate.
    fvec : np.ndarray, float64, shape (m,)
        Residuals at ``x``.
    success : bool
        ``1 <= info <= 4``.
    info : int32
        lmder1 code:

        * 0, improper input parameters.
        * 1, estimated relative error in the sum of squares <= ``tol``.
        * 2, estimated relative error between ``x`` and the solution
          <= ``tol``.
        * 3, conditions 1 and 2 both hold.
        * 4, ``fvec`` orthogonal to the Jacobian columns to machine
          precision.
        * 5, callback calls with ``iflag == 1`` reached ``100*(n+1)``.
        * 6, ``tol`` too small, no further reduction in the sum of
          squares possible.
        * 7, ``tol`` too small, no further improvement in ``x``
          possible.

    Notes
    -----
    No ``maxfev`` parameter: lmder1 hardcodes ``100*(n+1)``.

    ``ipvt`` (the column-pivot permutation) and the QR factors are
    computed internally but not returned; scipy exposes them only through
    ``full_output``.

    Safe to call from a ``numba.prange`` loop: the module variable holding
    the callback carries ``!$omp threadprivate``.
    """
    if not _wrote_jac_residual(
            funcptr, x_init, neqs, neqs * x_init.size, args):
        raise ValueError(
            "the callback never wrote fvec at iflag=1. The argument "
            "order is (x, fvec, fjac, iflag, args)")
    m = np.int32(neqs)
    n = np.int32(x_init.size)
    m_ = np.array(m, np.int32)
    n_ = np.array(n, np.int32)
    x = x_init.astype(np.float64).copy()
    fvec = np.zeros(m, np.float64)
    fjac = np.zeros(m * n, np.float64)
    ldfjac = np.array(m, np.int32)
    tol_ = np.array(tol, np.float64)
    info = np.zeros(1, np.int32)
    ipvt = np.zeros(n, np.int32)
    lwa = np.int32(5 * n + m + 2)
    wa = np.zeros(lwa, np.float64)
    lwa_ = np.array(lwa, np.int32)
    # ascontiguousarray: a strided view's .ctypes.data points into memory
    # the Fortran side would read CONTIGUOUSLY, silently wrong values.
    args_ = np.ascontiguousarray(np.asarray(args, np.float64))
    nargs = np.array(args_.size, np.int32)

    _lmder1(funcptr, m_.ctypes.data, n_.ctypes.data, x.ctypes.data,
            fvec.ctypes.data, fjac.ctypes.data, ldfjac.ctypes.data,
            tol_.ctypes.data, info.ctypes.data, ipvt.ctypes.data,
            wa.ctypes.data, lwa_.ctypes.data,
            args_.ctypes.data, nargs.ctypes.data)

    _status = _jac_status(funcptr, x, fvec, neqs, neqs * x_init.size, args)
    if _status != 0:
        # 1 = residual identically zero, 2 = the residual the solver
        # reported disagrees with a fresh evaluation at x.  The status
        # is set either way so a sweep can filter on it; validate
        # decides whether to also raise.
        if validate:
            if _status == 1:
                raise ValueError(
                    "the residual is identically zero, so every point is a "
                    "root and this result is meaningless. Check the callback; "
                    "pass validate=False to get info=-1 instead")
            raise ValueError(
                "the residual the solver reported at the solution does not "
                "match a fresh evaluation there, so it did not converge. "
                "Check the callback for state or side effects; pass "
                "validate=False to get info=-2 instead")
        return x, fvec, False, np.int32(-_status)
    return x, fvec, 1 <= info[0] <= 4, info[0]





# ==========================================================================
# scipy.optimize.leastsq
#
# One core, two entry points.  `_core_lmdif` / `_core_lmder` hold the whole
# algorithm; `leastsq` (python) and its `@overload` (njit) only resolve the
# callback and slice the return, so the two cannot disagree.
# ==========================================================================

class _Mapping:
    """String-key access on a namedtuple result.

    scipy hands back a ``dict`` from `leastsq` and `fsolve` and an
    ``OptimizeResult`` from `root`, so ``res['x']``, ``res.get('nfev')`` and
    ``res.keys()`` are part of the contract.  A namedtuple is what crosses
    the ``@njit`` boundary, and these three methods sit on top of it.
    Integer indexing and unpacking are untouched.
    """

    __slots__ = ()

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return getattr(self, key)
            except AttributeError:
                raise KeyError(key)
        return tuple.__getitem__(self, key)

    def keys(self):
        #  A real ``dict_keys`` view, as scipy's is: it supports the set
        #  operations a caller writes against scipy, ``res.keys() - {'x'}``.
        #  Built from the field NAMES alone, so no value is copied.
        return dict.fromkeys(self._fields).keys()

    def values(self):
        return tuple(self)

    def items(self):
        return tuple(zip(self._fields, self))

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._fields
        return tuple.__contains__(self, key)


def _result(name, fields):
    """A namedtuple carrying `_Mapping`, so both accesses reach one object."""
    return type(name, (_Mapping, namedtuple(name, fields)), {'__slots__': ()})


class OptimizeResult(_Mapping):
    """The result `minimize`, `root` and the other solvers return.

    Notes
    -----
    **The field set is the solver's, not a fixed one.** Each routine, and on
    `minimize` and `root` each method, returns the fields scipy's own result
    carries for that call and no others. A field the solver did not compute
    is ABSENT: reading it raises `AttributeError` from Python and is a
    `numba.TypingError` when the call compiles, and ``hasattr`` is ``False``.

    `keys` lists what a given result holds. Attribute access, ``res['x']``,
    ``res.get('nfev')``, ``in``, `values` and `items` all read the same
    object, from Python and from inside ``@njit``.

    The CONTAINER differs from scipy's, which is a ``dict`` subclass:
    ``isinstance(res, scipy.optimize.OptimizeResult)`` is ``False``, integer
    indexing reads a value here and raises ``KeyError`` there, and unpacking
    yields this result's VALUES where scipy's yields its field NAMES.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import minimize
    >>> @njit
    ... def rosen(x):
    ...     return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    >>> res = minimize(rosen, np.array([0.5, 0.5]), method='Nelder-Mead')
    >>> sorted(res.keys())
    ['final_simplex', 'fun', 'message', 'nfev', 'nit', 'status', 'success', 'x']
    >>> hasattr(res, 'hess_inv')
    False

    Attributes
    ----------
    x : ndarray
        The solution.
    fun : float or ndarray
        The objective, or the residual vector, at `x`.
    success : bool
        Whether the solver met its termination condition.
    message : str
        What the solver reported.
    """

    __slots__ = ()


#: Every distinct field set gets ONE type, built once at import.  Two calls
#: asking for the same fields get the SAME class, so numba compiles one
#: result type per field set rather than one per call site.
_OPT_RESULT_CACHE = {}


def _opt_result(fields):
    """An `OptimizeResult` carrying exactly `fields`.

    The class is named ``OptimizeResult`` whatever its fields are, because
    that is the name scipy publishes and the name `isinstance` is written
    against. What distinguishes two of them is the field set.
    """
    key = tuple(fields)
    cls = _OPT_RESULT_CACHE.get(key)
    if cls is None:
        cls = type('OptimizeResult',
                   (OptimizeResult, namedtuple('OptimizeResult', key)),
                   {'__slots__': ()})
        _OPT_RESULT_CACHE[key] = cls
    return cls


@overload(operator.getitem)
def _result_getitem(obj, key):
    """``res['x']`` inside ``@njit``, for the results `_result` builds.

    The field name is a compile-time string, so the lookup becomes the
    ordinary tuple index and costs nothing at run time.  Restricted to this
    module's result types, and declining on anything else leaves every other
    ``getitem`` alone.
    """
    if not isinstance(obj, types.BaseNamedTuple):
        return None
    cls = getattr(obj, 'instance_class', None)
    if cls is None or not issubclass(cls, _Mapping):
        return None
    if not isinstance(key, types.StringLiteral):
        return None
    if key.literal_value not in cls._fields:
        raise TypingError("%s has no field %r" % (cls.__name__,
                                                  key.literal_value))
    i = cls._fields.index(key.literal_value)

    def impl(obj, key):
        return obj[i]
    return impl


#: scipy's ``infodict`` for the forward-difference path, as a namedtuple.
LsqInfo = _result('LsqInfo', ['fvec', 'nfev', 'fjac', 'ipvt', 'qtf'])
#: scipy's ``infodict`` for the analytic-jacobian path (adds ``njev``).
LsqInfoJ = _result('LsqInfoJ',
                   ['fvec', 'nfev', 'njev', 'fjac', 'ipvt', 'qtf'])

_EPS = float(np.finfo(np.float64).eps)     # scipy's epsfcn=None resolution


def _as_x0(x0):
    """1-D contiguous float64 copy of ``x0``; a scalar becomes length 1.

    Any rank is FLATTENED, matching scipy's ``x0 = asarray(x0).flatten()``.
    """
    return np.atleast_1d(np.asarray(x0, dtype=np.float64)).flatten()


@overload(_as_x0)
def _as_x0_ovl(x0):
    """@njit implementation of `_as_x0`, one body per argument shape.

    The function above is a plain Python one and cannot be reached from
    inside ``@njit``, so the njit path is written here. It gets three
    bodies rather than one because ``x0``'s shape is already known at
    this point: a scalar and a tuple are written straight into the output
    buffer, and only the array-like case needs the
    contiguous-then-ravel-then-cast chain.
    """
    if isinstance(x0, (types.Float, types.Integer)):
        def impl(x0):
            out = np.empty(1, np.float64)
            out[0] = np.float64(x0)
            return out
        return impl
    if isinstance(x0, types.BaseTuple):
        k = len(x0)

        def impl(x0):
            out = np.empty(k, np.float64)
            for i in range(k):
                out[i] = np.float64(x0[i])
            return out
        return impl

    def impl(x0):
        # ascontiguousarray: a strided view's .ctypes.data points into memory
        # the Fortran side would read CONTIGUOUSLY, silently wrong values.
        return np.ascontiguousarray(np.asarray(x0)).ravel().astype(np.float64)
    return impl


# --------------------------------------------------------------------------
# scipy's `args`: a TUPLE of objects, splatted into the callback as
# ``f(x, *args)``.  The scipy-shaped front ends take a plain @njit callback
# only, so the @cfunc is built when the call compiles and the tuple's arity
# and every element's type are compile-time constants.  MINPACK still reaches
# the callback through one ``double*``, so the tuple is flattened into that
# buffer by a packer and rebuilt by the adapter; both halves are generated
# from the same kinds and cannot drift.  A scalar takes one slot and an array
# takes its shape followed by its data.
#
# The raw ``.address`` drivers below (hybrd, hybrj, lmdif, lmder) are
# untouched by any of this: they take the flat float64 buffer directly.
# --------------------------------------------------------------------------

#: kind codes: a float, an integer and a boolean scalar; ``k >= 1`` is an
#: array of that rank.  A scalar keeps its own type through the buffer,
#: which is a cast in the adapter; an array's data crosses as float64.

_ARGS_ELEM_MSG = (
    "an args entry must be a real number or an array of real numbers. The "
    "extra parameters cross MINPACK's double* argument slot, which carries "
    "no other type.")

_ADDRESS_DRIVERS = "hybrd, hybrj, lmdif or lmder"


def _address_msg(name, drivers=_ADDRESS_DRIVERS, spelling="f(x, *args)"):
    """The refusal a front end gives a caller who passes a callback address.

    Positive: it names the plain-``@njit`` spelling this routine wants AND
    the drivers an address still reaches, because a caller holding one has
    somewhere to go.
    """
    return ("%s: the callback must be a plain @njit function, called as "
            "%s with one argument per entry of args. The .address of a "
            "@cfunc is not accepted: pass the @njit function itself and %s "
            "builds the callback. An address still reaches %s, which take it "
            "deliberately." % (name, spelling, name, drivers))


_ADDRESS_MSG_FSOLVE = _address_msg('fsolve')
_ADDRESS_MSG_LEASTSQ = _address_msg('leastsq')
_ADDRESS_MSG_ROOT = _address_msg('root')


def _arg_kinds(args):
    """Element kinds of a Python-level ``args`` tuple."""
    return _arg_kinds_base(args, _ARGS_ELEM_MSG)


def _arg_kinds_ty(args):
    """Element kinds of the numba TYPES of ``args``, at typing time."""
    return _arg_kinds_ty_base(args, _ARGS_ELEM_MSG)


def _as_args_tuple(args):
    """scipy's coercion: a non-tuple ``args`` is a one-item tuple.

    ``None`` is a non-tuple, so it becomes ``(None,)`` and adds one extra
    parameter, which is scipy's rule and not "no parameters".  The entry
    that follows carries no real number, so `_arg_kinds` refuses it.
    """
    if isinstance(args, tuple):
        return args
    return (args,)


def _args_types(args):
    """The element types of ``args`` as the chooser sees them.

    An omitted default is the empty tuple; any other non-tuple is a one-item
    tuple, which is scipy's own coercion, and that includes ``None``.  An
    OMITTED default reaches a chooser as the RAW PYTHON VALUE rather than as
    a numba type.
    """
    if isinstance(args, types.Omitted):
        args = args.value
    if isinstance(args, types.Type):
        if isinstance(args, types.BaseTuple):
            return tuple(args)
        return (args,)
    return tuple(typeof(v) for v in _as_args_tuple(args))


def _args_is_tuple(args):
    """True when ``args`` is a tuple rather than a lone value."""
    if isinstance(args, types.Omitted):
        args = args.value
    if isinstance(args, types.Type):
        return isinstance(args, types.BaseTuple)
    return isinstance(args, tuple)


def _unpack_lines(kinds, off, ind, p='e', buf='a', ctr='o'):
    """Adapter lines rebuilding ``args`` from the flat buffer ``buf``.

    ``off`` is where the payload starts, behind whatever counts the caller's
    header holds; ``p`` prefixes the generated names, so two payloads can be
    unpacked into one body.  Returns the lines and the names to splat into
    the call.
    """
    lines = ["%s%s = %s" % (ind, ctr, off)]
    names = []
    for i, k in enumerate(kinds):
        if k < 0:
            # A scalar INTEGER slot carries the int64 BITS the packer wrote,
            # read back through the same view, so a value above 2**53 is not
            # rounded on the way in. The kind is known per argument when the
            # adapter is generated, so nothing is decided from the bits.
            cast = {_K_INT: "%s.view(np.int64)[%s]" % (buf, ctr),
                    _K_BOOL: "%s[%s] != 0.0" % (buf, ctr),
                    _K_FLOAT: "%s[%s]" % (buf, ctr)}[k]
            lines.append("%s%s%d = %s" % (ind, p, i, cast))
            lines.append("%s%s += 1" % (ind, ctr))
        else:
            dims = ", ".join("int(%s[%s + %d])" % (buf, ctr, j)
                             for j in range(k))
            lines.append("%s%sd%d = (%s,)" % (ind, p, i, dims))
            lines.append("%s%s += %d" % (ind, ctr, k))
            lines.append("%s%sm%d = %s" % (
                ind, p, i, " * ".join("%sd%d[%d]" % (p, i, j)
                                      for j in range(k))))
            sl = "%s[%s:%s + %sm%d]" % (buf, ctr, ctr, p, i)
            if k == 1:
                lines.append("%s%s%d = %s" % (ind, p, i, sl))
            else:
                lines.append("%s%s%d = %s.reshape(%sd%d)"
                             % (ind, p, i, sl, p, i))
            lines.append("%s%s += %sm%d" % (ind, ctr, p, i))
        names.append("%s%d" % (p, i))
    return "\n".join(lines), "".join(", " + s for s in names)


def _pack_args(args):
    """scipy's ``args``, flattened into the buffer the adapter unflattens.

    A scalar INTEGER entry crosses its slot as the int64 BITS, read back
    through the same view by the adapter, so a value above ``2**53`` is not
    rounded on the way. The slot's kind is known per argument when the
    adapter is generated, so nothing has to be decided from the bits.

    `_arg_kinds` runs first so a non-numeric entry is refused here rather
    than packed as a NaN.  Without it ``_pack_args(None)`` returned
    ``[nan]`` while the compiled body raised.
    """
    at = _as_args_tuple(args)
    _arg_kinds(at)
    parts = []
    for v in at:
        a = v if isinstance(v, np.ndarray) else np.asarray(v)
        if a.ndim == 0 and a.dtype.kind in 'iu':
            parts.append((_K_INT, np.int64(a)))
            continue
        a = np.asarray(v, np.float64)
        if a.ndim == 0:
            parts.append((_K_FLOAT, (a.reshape(1),)))
        else:
            parts.append((a.ndim, (np.asarray(a.shape, np.float64),
                                   np.ascontiguousarray(a).ravel())))
    tot = 0
    for k, p in parts:
        tot += 1 if k < 0 else k + p[1].size
    out = np.empty(tot, np.float64)
    o = 0
    for k, p in parts:
        if k == _K_INT:
            out.view(np.int64)[o] = p
            o += 1
        elif k < 0:
            out[o] = p[0][0]
            o += 1
        else:
            for q in p:
                out[o:o + q.size] = q
                o += q.size
    return out


@overload(_pack_args)
def _pack_args_ovl(args):
    """Compiled packer, one body per distinct ``args`` type."""
    kinds = _arg_kinds_ty(_args_types(args))
    one = not _args_is_tuple(args)
    lines = ["def impl(args):"]
    sizes = []
    for i, k in enumerate(kinds):
        src = "args" if one else "args[%d]" % i
        if k < 0:
            sizes.append("1")
        else:
            lines.append("    v%d = np.asarray(%s).astype(np.float64).ravel()"
                         % (i, src))
            lines.append("    s%d = %s.shape" % (i, src))
            sizes.append("%d + v%d.size" % (k, i))
    lines.append("    tot = %s" % (" + ".join(sizes) if sizes else "0"))
    lines.append("    out = np.empty(tot, np.float64)")
    lines.append("    o = 0")
    for i, k in enumerate(kinds):
        src = "args" if one else "args[%d]" % i
        if k == _K_INT:
            # the int64 BITS, so a value above 2**53 is not rounded
            lines.append("    out.view(np.int64)[o] = np.int64(%s)" % src)
            lines.append("    o += 1")
        elif k < 0:
            lines.append("    out[o] = np.float64(%s)" % src)
            lines.append("    o += 1")
        else:
            for j in range(k):
                lines.append("    out[o + %d] = np.float64(s%d[%d])"
                             % (j, i, j))
            lines.append("    o += %d" % k)
            lines.append("    out[o:o + v%d.size] = v%d" % (i, i))
            lines.append("    o += v%d.size" % i)
    lines.append("    return out")
    ns = {'np': np}
    exec("\n".join(lines), ns)                           # noqa: S102
    return ns['impl']


def _call_cb(f, x, args):
    """``f(x, *args)``, scipy's calling convention, from either entry point.

    Used for the probe call that reads the residual count off the callback.
    """
    return f(x, *_as_args_tuple(args))


@overload(_call_cb)
def _call_cb_ovl(f, x, args):
    """Compiled `_call_cb`: the splat is unrolled, since numba cannot splat
    a runtime value into a call."""
    if not _args_is_tuple(args):
        def impl(f, x, args):
            return f(x, args)               # scipy's one-item coercion
        return impl
    k = len(_args_types(args))
    ns = {}
    exec("def impl(f, x, args):\n    return f(x%s)\n"     # noqa: S102
         % "".join(", args[%d]" % i for i in range(k)), ns)
    return ns['impl']


def _check_arity(py, nargs, name, first='x'):
    """scipy's calling convention, checked where the message can name it.

    scipy reaches ``func(x, *args)`` and lets Python report the mismatch.
    A ``@cfunc`` swallows the same report, so it is made here.
    """
    code = getattr(py, '__code__', None)
    if code is None or code.co_flags & 0x04:             # *args: no fixed arity
        return
    want = 1 + nargs
    ndef = len(getattr(py, '__defaults__', None) or ())
    if code.co_argcount - ndef <= want <= code.co_argcount:
        return
    raise TypeError(
        "%s: the callback takes %d argument(s), and an args of length %d "
        "calls it as f(%s, *args), which needs %d."
        % (name, code.co_argcount, nargs, first, want))


def _front_pyfunc(func, nargs, name, msg, first='x'):
    """The plain Python function behind a front end's callback.

    Refuses an address, a ``@cfunc`` object and anything not callable with
    `msg`, then checks the arity ``args`` implies.
    """
    if isinstance(func, (bool, int, np.integer)) or hasattr(func, 'address'):
        raise ValueError(msg)
    py = getattr(func, 'py_func', func)
    if not callable(py):
        raise ValueError(msg)
    _check_arity(py, nargs, name, first)
    return py


def _front_pyfunc_ty(func, nargs, name, msg, first='x'):
    """`_front_pyfunc` at typing time; every refusal is a ``TypingError``."""
    if not isinstance(func, types.Dispatcher):
        raise TypingError(msg)
    py = func.dispatcher.py_func
    try:
        _check_arity(py, nargs, name, first)
    except TypeError as exc:
        raise TypingError(str(exc))
    return py


# --------------------------------------------------------------------------
# Callback dimensions.  minpack_sig is void(double*, double*, double*), so the
# adapter writes exactly the slots its header says and can neither notice a
# longer residual nor stop short of a shorter one: the first is truncated, the
# second reads past the end of the callback's own array.  The lengths ARE
# knowable one level up, where the plain @njit callback is still a function,
# so the check lives there.  It cannot live in the adapter: a `raise` inside a
# ``@cfunc`` is printed and swallowed.
#
# The message is scipy's, built in two halves.  The half naming the routine,
# the argument and the callback is fixed once the callback is known, so it is
# assembled in Python -- in an ``@overload`` chooser for the compiled body --
# and reaches the guard as one constant string.  The half naming the two
# shapes is run-time and is concatenated inside ``@njit``.
# --------------------------------------------------------------------------
def _jac_not_an_address(jac):
    """Refuse a callback address where `root` reads ``jac`` for truth.

    scipy's rule is ``not callable(jac)``, so any non-callable is read for
    its truth value.  A ``@cfunc`` object and an integer outside ``{0, 1}``
    are refused instead, because reading a function pointer as a value is
    the failure `_address_msg` exists to prevent, and ``bool`` is the only
    non-callable ``jac`` scipy documents.
    """
    if hasattr(jac, 'address'):
        raise ValueError(_ADDRESS_MSG_ROOT)
    if (isinstance(jac, (int, np.integer)) and not isinstance(jac, bool)
            and int(jac) not in (0, 1)):
        raise ValueError(_ADDRESS_MSG_ROOT)


def _jac_not_an_address_ty(jac):
    """`_jac_not_an_address` at typing time."""
    v = jac.value if isinstance(jac, types.Omitted) else jac
    if isinstance(v, types.IntegerLiteral):
        v = v.literal_value
    if isinstance(v, types.Type):
        return
    if hasattr(v, 'address'):
        raise TypingError(_ADDRESS_MSG_ROOT)
    if (isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            and int(v) not in (0, 1)):
        raise TypingError(_ADDRESS_MSG_ROOT)


def _cb_args(at, kinds):
    """``args`` as the ADAPTER will present them to the callback.

    The flat buffer carries float64 and the adapter rebuilds a C-contiguous
    array of the recorded rank, so a list, an int array or an F-order array
    reaches the callback as a C-order float64 array.  The dimension probes
    call the callback directly and must hand it the same objects, or a
    callback written for what the solve will pass fails to compile.
    """
    out = []
    for v, k in zip(at, kinds):
        if k == _K_INT:
            out.append(int(v))
        elif k == _K_BOOL:
            out.append(bool(v))
        elif k == _K_FLOAT:
            out.append(float(v))
        else:
            out.append(np.ascontiguousarray(np.asarray(v, np.float64)))
    return tuple(out)


def _shape_prefix(name, argname, py):
    """scipy's ``_check_func`` message, up to the two shapes."""
    fn = getattr(py, '__name__', None)
    return ("%s: there is a mismatch between the input and output shape of "
            "the '%s' argument%s Shape should be "
            % (name, argname, " '%s'." % fn if fn else "."))


@njit
def _check_resid_len(prefix, got, want):
    """The residual count scipy checks against ``(n,)``."""
    if got != want:
        raise TypeError(prefix + "(" + str(want) + ",) but it is ("
                        + str(got) + ",).")


@njit
def _check_jac_shape(prefix, g0, g1, w0, w1):
    """The Jacobian shape scipy checks against ``(w0, w1)``."""
    if g0 != w0 or g1 != w1:
        raise TypeError(prefix + "(" + str(w0) + ", " + str(w1)
                        + ") but it is (" + str(g0) + ", " + str(g1) + ").")


@njit
def _prepend_nm(args, n, m):
    """args buffer the ADAPTER path expects: ``[n, m, nargs, *args]``.

    A ``@cfunc`` receives one ``double*`` and no lengths, so the counts the
    adapter needs travel in front of the user's own parameters.  ``nargs`` is
    there so an ``f(x, args)`` residual can be handed a correctly sized view
    of what follows.  Applies to the adapter path only; a raw ``.address``
    gets ``args`` untouched.
    """
    out = np.empty(args.size + 3, np.float64)
    out[0] = np.float64(n)
    out[1] = np.float64(m)
    out[2] = np.float64(args.size)
    for i in range(args.size):
        out[i + 3] = args[i]
    return out


# --------------------------------------------------------------------------
# Hiding the @cfunc.  ``func`` is a plain @njit ``f(x) -> residuals``, and the
# pointer-shaped adapter is built here, once, and cached.  The cache also OWNS
# the cfunc: drop the reference and the baked-in address dangles.
# --------------------------------------------------------------------------
_LSQ_ADAPTERS = {}


_LSQ_RESID_SRC = """
def _make(inner, cfunc, sig, carray):
    @cfunc(sig)
    def adapter(x_ptr, f_ptr, a_ptr):
        n = int(a_ptr[0])
        m = int(a_ptr[1])
        na = int(a_ptr[2])
        x = carray(x_ptr, n)
        a = carray(a_ptr, 3 + na)
%(unpack)s
        out = inner(x%(argl)s)
        for i in range(m):
            f_ptr[i] = out[i]
    return adapter
"""

_LSQ_JAC_SRC = """
def _make(f_in, j_in, cfunc, sig, carray):
    @cfunc(sig)
    def adapter(x_ptr, f_ptr, jac_ptr, iflag_ptr, a_ptr):
        n = int(a_ptr[0])
        m = int(a_ptr[1])
        na = int(a_ptr[2])
        x = carray(x_ptr, n)
        a = carray(a_ptr, 3 + na)
%(unpack)s
        if iflag_ptr[0] == 1:
            out = f_in(x%(argl)s)
            for i in range(m):
                f_ptr[i] = out[i]
        else:
            jj = j_in(x%(argl)s)
            for j in range(n):
                for i in range(m):
                    jac_ptr[i + j * m] = %(jj)s
    return adapter
"""

_LSQ_PAIR_SRC = """
def _make(inner, cfunc, sig, carray):
    @cfunc(sig)
    def adapter(x_ptr, f_ptr, jac_ptr, iflag_ptr, a_ptr):
        n = int(a_ptr[0])
        m = int(a_ptr[1])
        na = int(a_ptr[2])
        c = 3 + na
        x = carray(x_ptr, n)
        a = carray(a_ptr, c + 1 + n + m * n)
%(unpack)s
        if iflag_ptr[0] == 1:
            out, jj = inner(x%(argl)s)
            for i in range(m):
                f_ptr[i] = out[i]
            for i in range(n):
                a[c + 1 + i] = x[i]
            for j in range(n):
                for i in range(m):
                    a[c + 1 + n + i + j * m] = %(jj)s
            a[c] = 1.0
        else:
            hit = a[c] != 0.0
            if hit:
                for i in range(n):
                    if a[c + 1 + i] != x[i]:
                        hit = False
            if hit:
                for j in range(n):
                    for i in range(m):
                        jac_ptr[i + j * m] = a[c + 1 + n + i + j * m]
            else:
                out, jj = inner(x%(argl)s)
                for j in range(n):
                    for i in range(m):
                        jac_ptr[i + j * m] = %(jj)s
    return adapter
"""

_RESID_SRC = """
def _make(inner, cfunc, sig, carray):
    @cfunc(sig)
    def adapter(x_ptr, f_ptr, a_ptr):
        n = int(a_ptr[0])
        na = int(a_ptr[1])
        x = carray(x_ptr, n)
        a = carray(a_ptr, 2 + na)
%(unpack)s
        out = inner(x%(argl)s)
        for i in range(n):
            f_ptr[i] = out[i]
    return adapter
"""

_JAC_SRC = """
def _make(f_in, j_in, cfunc, sig, carray):
    @cfunc(sig)
    def adapter(x_ptr, f_ptr, jac_ptr, iflag_ptr, a_ptr):
        n = int(a_ptr[0])
        na = int(a_ptr[1])
        x = carray(x_ptr, n)
        a = carray(a_ptr, 2 + na)
%(unpack)s
        if iflag_ptr[0] == 1:
            out = f_in(x%(argl)s)
            for i in range(n):
                f_ptr[i] = out[i]
        else:
            jj = j_in(x%(argl)s)
            for j in range(n):
                for i in range(n):
                    jac_ptr[i + j * n] = %(jj)s
    return adapter
"""

_PAIR_SRC = """
def _make(inner, cfunc, sig, carray):
    @cfunc(sig)
    def adapter(x_ptr, f_ptr, jac_ptr, iflag_ptr, a_ptr):
        n = int(a_ptr[0])
        na = int(a_ptr[1])
        c = 2 + na
        x = carray(x_ptr, n)
        a = carray(a_ptr, c + 1 + n + n * n)
%(unpack)s
        if iflag_ptr[0] == 1:
            out, jj = inner(x%(argl)s)
            for i in range(n):
                f_ptr[i] = out[i]
                a[c + 1 + i] = x[i]
            for j in range(n):
                for i in range(n):
                    a[c + 1 + n + i + j * n] = %(jj)s
            a[c] = 1.0
        else:
            hit = a[c] != 0.0
            if hit:
                for i in range(n):
                    if a[c + 1 + i] != x[i]:
                        hit = False
            if hit:
                for j in range(n):
                    for i in range(n):
                        jac_ptr[i + j * n] = a[c + 1 + n + i + j * n]
            else:
                out, jj = inner(x%(argl)s)
                for j in range(n):
                    for i in range(n):
                        jac_ptr[i + j * n] = %(jj)s
    return adapter
"""


def _mk_cfunc(src, subs, inners, sig):
    """Compile one generated adapter body and wrap it in a ``@cfunc``.

    ``inners`` is the compiled callback, or the residual/jacobian pair.
    """
    if not isinstance(inners, tuple):
        inners = (inners,)
    ns = {'cfunc': cfunc, 'carray': carray, 'np': np}
    exec(src % subs, ns)                                 # noqa: S102
    return ns['_make'](*inners, cfunc=cfunc, sig=sig, carray=carray)


def _adapter_lsq_resid(py, kinds=()):
    """cfunc(minpack_sig) around a plain @njit residual ``f(x, *args)``.

    The user's parameters sit behind the ``n``/``m``/``nargs`` counts
    ``_prepend_nm`` writes, flattened by `_pack_args`, and are rebuilt here
    from `kinds` before the call.
    """
    key = (py, 'lsq_resid', kinds)
    hit = _LSQ_ADAPTERS.get(key)
    if hit is not None:
        return hit
    unpack, argl = _unpack_lines(kinds, 3, ' ' * 8)
    adapter = _mk_cfunc(_LSQ_RESID_SRC, {'unpack': unpack, 'argl': argl},
                        njit(py), minpack_sig)
    _LSQ_ADAPTERS[key] = adapter
    return adapter


def _adapter_lsq_jac(pyf, pyj, col_deriv=False, kinds=()):
    """cfunc(minpack_jac_sig) around @njit residual and jacobian.

    Both are called ``f(x, *args)``, so the two share one arity.
    Branches on ``iflag`` and writes ``fjac``
    COLUMN-major with leading dimension ``m``, which is what MINPACK's lmder
    expects.

    ``col_deriv`` carries scipy's meaning, which on this path is a shape as
    well as an orientation: ``False`` reads ``jac(x)`` as ``(m, n)`` and
    ``True`` reads it as ``(n, m)``, the two shapes scipy's own ``_check_func``
    demands.  The two build separate cfuncs and are cached separately.
    """
    key = (pyf, pyj, 'lsq_jac', bool(col_deriv), kinds)
    hit = _LSQ_ADAPTERS.get(key)
    if hit is not None:
        return hit
    unpack, argl = _unpack_lines(kinds, 3, ' ' * 8)
    adapter = _mk_cfunc(
        _LSQ_JAC_SRC, {'unpack': unpack, 'argl': argl,
                       'jj': 'jj[j, i]' if col_deriv else 'jj[i, j]'},
        (njit(pyf), njit(pyj)), minpack_jac_sig)
    _LSQ_ADAPTERS[key] = adapter
    return adapter


# --------------------------------------------------------------------------
# raw drivers
# --------------------------------------------------------------------------
@njit
def _run_lmdif(fp, x, args, m, ftol, xtol, gtol, maxfev, epsfcn, mode, diag,
               factor):
    """Full lmdif. Returns (x, fvec, fjac, ipvt, qtf, nfev, info).

    ``fjac`` is allocated (n, m) C-order, which is byte-for-byte the
    ``(ldfjac=m, n)`` column-major array MINPACK writes, and is the shape
    scipy reports in ``infodict['fjac']``.
    """
    n = x.size
    n_ = np.array(n, np.int32)
    m_ = np.array(m, np.int32)
    fvec = np.zeros(m, np.float64)
    fjac = np.zeros((n, m), np.float64)
    ipvt = np.zeros(n, np.int32)
    qtf = np.zeros(n, np.float64)
    ftol_ = np.array(ftol, np.float64)
    xtol_ = np.array(xtol, np.float64)
    gtol_ = np.array(gtol, np.float64)
    maxfev_ = np.array(maxfev, np.int32)
    eps_ = np.array(epsfcn, np.float64)
    mode_ = np.array(mode, np.int32)
    factor_ = np.array(factor, np.float64)
    ldfjac_ = np.array(m, np.int32)
    info = np.zeros(1, np.int32)
    nfev = np.zeros(1, np.int32)
    nargs = np.array(args.size, np.int32)

    _lmdif_sp(fp, m_.ctypes.data, n_.ctypes.data, x.ctypes.data,
              fvec.ctypes.data, ftol_.ctypes.data, xtol_.ctypes.data,
              gtol_.ctypes.data, maxfev_.ctypes.data, eps_.ctypes.data,
              mode_.ctypes.data, diag.ctypes.data, factor_.ctypes.data,
              info.ctypes.data, nfev.ctypes.data, fjac.ctypes.data,
              ldfjac_.ctypes.data, ipvt.ctypes.data, qtf.ctypes.data,
              args.ctypes.data, nargs.ctypes.data)
    for i in range(n):                 # MINPACK's ipvt is 1-based; scipy
        ipvt[i] = ipvt[i] - 1          # reports it 0-based
    return x, fvec, fjac, ipvt, qtf, nfev[0], info[0]


@njit
def _run_lmder(fp, x, args, m, ftol, xtol, gtol, maxfev, mode, diag, factor):
    """Full lmder. Returns (x, fvec, fjac, ipvt, qtf, nfev, njev, info)."""
    n = x.size
    n_ = np.array(n, np.int32)
    m_ = np.array(m, np.int32)
    fvec = np.zeros(m, np.float64)
    fjac = np.zeros((n, m), np.float64)
    ipvt = np.zeros(n, np.int32)
    qtf = np.zeros(n, np.float64)
    ftol_ = np.array(ftol, np.float64)
    xtol_ = np.array(xtol, np.float64)
    gtol_ = np.array(gtol, np.float64)
    maxfev_ = np.array(maxfev, np.int32)
    mode_ = np.array(mode, np.int32)
    factor_ = np.array(factor, np.float64)
    ldfjac_ = np.array(m, np.int32)
    info = np.zeros(1, np.int32)
    nfev = np.zeros(1, np.int32)
    njev = np.zeros(1, np.int32)
    nargs = np.array(args.size, np.int32)

    _lmder_sp(fp, m_.ctypes.data, n_.ctypes.data, x.ctypes.data,
              fvec.ctypes.data, fjac.ctypes.data, ldfjac_.ctypes.data,
              ftol_.ctypes.data, xtol_.ctypes.data, gtol_.ctypes.data,
              maxfev_.ctypes.data, mode_.ctypes.data, diag.ctypes.data,
              factor_.ctypes.data, info.ctypes.data, nfev.ctypes.data,
              njev.ctypes.data, ipvt.ctypes.data, qtf.ctypes.data,
              args.ctypes.data, nargs.ctypes.data)
    for i in range(n):
        ipvt[i] = ipvt[i] - 1
    return x, fvec, fjac, ipvt, qtf, nfev[0], njev[0], info[0]


# --------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------
@njit
def _lsq_maxfev(n, maxfev, jac_path):
    """scipy's budget: 200*(n+1), or 100*(n+1) when Dfun is given."""
    if maxfev == 0:
        return (100 if jac_path else 200) * (n + 1)
    return maxfev


@njit
def _lsq_diag_auto(n):
    """No user diag -> mode 1.  MINPACK WRITES the scaling it derives into
    this buffer, so one must be passed even in mode 1."""
    return np.ones(n, np.float64), 1


@njit
def _lsq_diag_given(diagarr, n):
    """User diag -> mode 2.  Copied: MINPACK may write to it.

    ``np.asarray`` accepts a list or a tuple as well as an array from both
    entry points, which is the sequence scipy documents.  Entries past the
    ``n`` MINPACK reads are ignored.  A diag SHORTER than ``n`` is refused,
    because the alternative is a read past the end of the caller's array.
    """
    d = np.asarray(diagarr)
    if d.size < n:
        raise TypeError("diag must have at least one entry per variable")
    return np.ascontiguousarray(d[:n]).astype(np.float64), 2


@njit
def _trtri_upper(r):
    """Inverse of an upper-triangular matrix; ``(invR, info)``.

    Transcribes LAPACK ``dtrti2('U','N')`` operation for operation.  Measured
    against ``scipy.linalg.lapack.dtrtri`` on five random well-conditioned
    upper-triangular matrices at each of fourteen sizes, 2026-08-10: bit
    identical for ``n <= 16``, and worst RELATIVE difference 9.766e-13 at
    ``n = 32``, 8.552e-12 at ``n = 64`` and 6.194e-11 at ``n = 200``, because
    the installed ``dtrtri`` blocks from ``n = 32`` and blocking reassociates
    the sums.  ``info`` is LAPACK's: ``0`` ok, otherwise the 1-based index of
    the first zero diagonal entry.
    """
    n = r.shape[0]
    a = r.copy()
    for i in range(n):
        if a[i, i] == 0.0:
            return a, i + 1
    for j in range(n):
        a[j, j] = 1.0 / a[j, j]
        ajj = -a[j, j]
        for k in range(j):                      # DTRMV('U','N','N')
            if a[k, j] != 0.0:
                temp = a[k, j]
                for i in range(k):
                    a[i, j] += temp * a[i, k]
                a[k, j] = temp * a[k, k]
        for i in range(j):                      # DSCAL
            a[i, j] = a[i, j] * ajj
    return a, 0


@njit
def _lsq_cov(fjac, ipvt, n, ier):
    """scipy's ``cov_x``: ``(J^T J)^-1`` via the QR factor lmdif returns.

    Follows ``_minpack_py.leastsq`` step for step -- upper-triangularise the
    first ``n`` rows of ``fjac.T``, invert with ``trtri``, undo the column
    pivoting, then form ``invR @ invR.T``.

    Returns an ``(n, n)`` array, or ``None`` exactly where scipy returns
    ``None``: when ``ier`` is outside ``{1, 2, 3, 4}`` or the triangular
    factor is singular.

    The return type is ``Optional(float64[:, :])``. numba unwraps it with a
    runtime check, so a caller that indexes the result keeps compiling and
    keeps working wherever the covariance exists, and gets
    ``TypeError: expected array(float64, 2d, C), got None`` where it does
    not. This function is the ONE construction site: an ``Optional`` value
    reaches a namedtuple field only when the field is filled from a single
    expression whose type is already ``Optional``, never from two branches
    building the namedtuple, which fails to unify.
    """
    if ier < 1 or ier > 4:
        return None
    r = np.zeros((n, n), np.float64)
    for i in range(n):
        for j in range(i, n):
            r[i, j] = fjac[j, i]                # triu(fjac.T[:n, :])
    invr, tinfo = _trtri_upper(r)
    if tinfo != 0:
        return None
    perm = np.empty((n, n), np.float64)
    for i in range(n):
        p = ipvt[i]
        if p < 0 or p >= n:
            return None
        for j in range(n):
            perm[p, j] = invr[i, j]             # invR[perm] = invR.copy()
    return np.dot(perm, perm.T)


@njit
def _fmt6(v):
    """``%f`` -- fixed 6 decimals, matching scipy's f-string ``{ftol:f}``.

    Only the FRACTIONAL part is scaled, so the ``int64`` range bounds the
    value rather than a millionth of it. The earlier spelling multiplied the
    WHOLE value by 1e6 before the cast and so returned garbage above about
    9.2e12, silently::

        _fmt6(1e13)   ->  '-9223372036855.224193'
        python %f     ->  '10000000000000.000000'

    No call site was ever affected: each passes `ftol`, `xtol` or `gtol`,
    which are tiny. The ceiling is now about 9.2e18.

    A NaN and an infinity render as python's ``%f`` does. A finite value
    beyond the ``int64`` range renders as an EMPTY field, which is worse to
    read than the digits and better than a wrong number.
    """
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "-inf" if v < 0.0 else "inf"
    neg = v < 0.0
    a = -v if neg else v
    if a >= 9.0e18:
        return ""
    w = np.floor(a)
    fi = np.int64((a - w) * 1e6 + 0.5)
    if fi >= 1000000:                 # the fraction rounded up to a whole
        fi -= 1000000
        w += 1.0
    fs = str(fi)
    while len(fs) < 6:
        fs = "0" + fs
    out = str(np.int64(w)) + "." + fs
    return "-" + out if neg else out


@njit
def _lsq_mesg(ier, ftol, xtol, gtol, maxfev):
    """scipy's ``leastsq`` message strings, verbatim, plus ``-1``/``-2``."""
    if ier == -1:
        return ("The residual is identically zero, so every point is a "
                "solution and this result is meaningless.")
    elif ier == -2:
        return ("The residual the solver reported at the solution does not "
                "match a fresh evaluation there, so it did not converge.")
    elif ier == 0:
        return "Improper input parameters."
    elif ier == 1:
        return ("Both actual and predicted relative reductions in the sum "
                "of squares\n  are at most " + _fmt6(ftol))
    elif ier == 2:
        return ("The relative error between two consecutive iterates is at "
                "most " + _fmt6(xtol))
    elif ier == 3:
        return ("Both actual and predicted relative reductions in the sum "
                "of squares\n  are at most " + _fmt6(ftol) + " and the "
                "relative error between two consecutive iterates is at \n  "
                "most " + _fmt6(xtol))
    elif ier == 4:
        return ("The cosine of the angle between func(x) and any column of "
                "the\n  Jacobian is at most " + _fmt6(gtol) + " in absolute "
                "value")
    elif ier == 5:
        return ("Number of calls to function has reached maxfev = "
                + str(maxfev) + ".")
    elif ier == 6:
        return ("ftol=" + _fmt6(ftol) + " is too small, no further reduction "
                "in the sum of squares\n  is possible.")
    elif ier == 7:
        return ("xtol=" + _fmt6(xtol) + " is too small, no further "
                "improvement in the approximate\n  solution is possible.")
    elif ier == 8:
        return ("gtol=" + _fmt6(gtol) + " is too small, func(x) is "
                "orthogonal to the columns of\n  the Jacobian to machine "
                "precision.")
    return "An error occurred."


# --------------------------------------------------------------------------
# THE cores.  Both entry points below only slice these returns.
# --------------------------------------------------------------------------
@njit
def _core_lmdif(fp, x, a, m, ftol, xtol, gtol, mf, epsfcn, mode, dd, factor,
                validate):
    """`leastsq` with a forward-difference Jacobian: the whole algorithm.

    Everything `leastsq` does beyond resolving the callback and slicing
    the return lives here, so its Python body and its ``@overload`` body
    cannot drift apart. `_core_hybrd` records what happened the one time
    a public entry point in this module kept its own parallel Python body.

    Returns ``(x, fvec, fjac, ipvt, qtf, nfev, ier)``, from which both
    entry points take what their ``full_output`` asks for. `_core_lmder`
    is the analytic-Jacobian twin.
    """
    n = x.size
    if n > m:
        raise TypeError(
            "Improper input: func input vector length N=" + str(n)
            + " must not exceed func output vector length M=" + str(m))
    # `_wrote_residual` evaluates the residual once; counted, so `nfev`
    # reports the work the callback actually did.
    if not _wrote_residual(fp, x, m, a):
        raise ValueError(
            "the residual callback never wrote fvec. Check the "
            "@cfunc signature and argument order")
    x, fvec, fjac, ipvt, qtf, nfev, ier = _run_lmdif(
        fp, x, a, m, ftol, xtol, gtol, mf, epsfcn, mode, dd, factor)
    nfev = np.int32(nfev + 1)
    # ier = 0 means MINPACK rejected the arguments and never ran, so there is
    # no solution to classify.  scipy reports it the same way.
    st = 0 if ier == 0 else _resid_status(fp, x, fvec, m, a)
    if st != 0:
        if validate:
            raise ValueError(_lsq_mesg(-st, ftol, xtol, gtol, mf))
        ier = np.int32(-st)
    return x, fvec, fjac, ipvt, qtf, nfev, ier


@njit
def _core_lmder(fp, x, a, m, ftol, xtol, gtol, mf, mode, dd, factor, validate):
    """`leastsq` with an analytic Jacobian: the whole algorithm.

    The `_core_lmdif` twin; see there for why the algorithm lives in a
    core rather than in the entry points. The callback ABI is the
    difference: one function serves both phases and ``iflag`` selects
    them, so the probe checks that ``fvec`` was written at ``iflag = 1``
    rather than that the residual was written at all.

    Returns ``(x, fvec, fjac, ipvt, qtf, nfev, njev, ier)``, one longer
    than `_core_lmdif` because MINPACK counts Jacobian evaluations too.
    """
    n = x.size
    if n > m:
        raise TypeError(
            "Improper input: func input vector length N=" + str(n)
            + " must not exceed func output vector length M=" + str(m))
    # `_wrote_jac_residual` evaluates the residual branch once; counted.
    if not _wrote_jac_residual(fp, x, m, m * n, a):
        raise ValueError(
            "the callback never wrote fvec at iflag=1. The argument "
            "order is (x, fvec, fjac, iflag, args)")
    x, fvec, fjac, ipvt, qtf, nfev, njev, ier = _run_lmder(
        fp, x, a, m, ftol, xtol, gtol, mf, mode, dd, factor)
    nfev = np.int32(nfev + 1)
    st = 0 if ier == 0 else _jac_status(fp, x, fvec, m, m * n, a)
    if st != 0:
        if validate:
            raise ValueError(_lsq_mesg(-st, ftol, xtol, gtol, mf))
        ier = np.int32(-st)
    return x, fvec, fjac, ipvt, qtf, nfev, njev, ier


@njit
def _lsq_raise_bad(ier):
    """``ier = 0`` raises, exactly where scipy's ``leastsq`` raises.

    The class is scipy's too. ``TypeError`` was previously ruled out on the
    grounds that numba constructs only a fixed set of exception types;
    measured 2026-08-03 on numba 0.66, a ``TypeError`` raised inside ``@njit``
    reaches the caller as a ``TypeError`` carrying its message.
    """
    if ier == 0:
        raise TypeError("Improper input parameters.")


def _emit_lsq_warning(ier, ftol, xtol, gtol, maxfev):
    warnings.warn(_lsq_mesg.py_func(ier, ftol, xtol, gtol, maxfev),
                  RuntimeWarning, stacklevel=2)


def _emit_root_cb_warning(method):
    """scipy's own `RuntimeWarning` for a `callback` its method ignores."""
    warnings.warn(f'Method {method} does not accept callback.',
                  RuntimeWarning, stacklevel=2)


@njit
def _root_warn_cb(method):
    """`_emit_root_cb_warning` from compiled code.

    The ``objmode`` block sits in its own function so that lowering it
    pickles this dispatcher rather than a caller holding a ctypes entry
    point.
    """
    with objmode():
        _emit_root_cb_warning(method)


@njit
def _lsq_warn_bad(ier, ftol, xtol, gtol, maxfev):
    """``RuntimeWarning`` on a bare call, exactly where scipy warns.

    scipy's gate on ``full_output=False`` is ``ier`` in ``{5, 6, 7, 8}``,
    its ``LEASTSQ_FAILURE`` list: `maxfev` exhausted and the three
    tolerance-too-small codes. The message is the one `_lsq_mesg` builds,
    so there is one table rather than two.

    ``warnings.warn`` is not typeable by numba; an ``objmode`` block runs
    its body in the interpreter, so ``catch_warnings``, ``simplefilter``,
    ``-W`` and ``PYTHONWARNINGS`` all select it normally. The block takes
    the GIL and is reached only on a failing bare call. `ier = 0` still
    raises through `_lsq_raise_bad`, and the scijit-only ``-1`` and ``-2``
    are left alone.
    """
    if ier == 5 or ier == 6 or ier == 7 or ier == 8:
        with objmode():
            _emit_lsq_warning(ier, ftol, xtol, gtol, maxfev)


# --------------------------------------------------------------------------
# the public entry point
# --------------------------------------------------------------------------
def leastsq(func, x0, args=(), Dfun=None, full_output=False, col_deriv=False,
            ftol=1.49012e-8, xtol=1.49012e-8, gtol=0.0, maxfev=0,
            epsfcn=None, factor=100.0, diag=None, m=-1, validate=True):
    """Minimize the sum of squares of a set of residuals.

    Wraps MINPACK's ``lmdif``, Levenberg-Marquardt with a forward-difference
    Jacobian, or ``lmder`` when an analytic Jacobian is supplied.

    Callable from Python and from inside ``@njit``.

    Parameters
    ----------
    func : callable
        A plain ``@njit`` ``f(x) -> residuals`` or ``f(x, args)``. The
        residual count is read from what it returns.
    x0 : float or array_like
        Initial parameter guess. A scalar, list or tuple is accepted. Any
        rank is FLATTENED.
    args : tuple, optional
        Extra parameters, passed to the callback as ``f(x, *args)``, one
        per entry. Entries may be scalars or arrays and need not share a
        type. A non-tuple `args` is coerced to the ONE argument ``(args,)``.
    Dfun : callable or None, optional
        ``None`` (default) uses a forward-difference Jacobian (MINPACK
        lmdif). A plain ``@njit`` ``jac(x) -> (m, n)`` selects the analytic
        path (lmder); the ``iflag`` branch and MINPACK's column-major layout
        are handled internally. Its shape must be ``(m, n)``, or ``(n, m)``
        with ``col_deriv=True``.
    full_output : bool, optional
        ``False`` (default) returns ``(x, ier)``. ``True`` returns
        ``(x, cov_x, infodict, mesg, ier)``. Must be a compile-time constant
        inside ``@njit``: a literal, or left at its default.
    col_deriv : bool, optional
        A statement about `Dfun`, selecting a shape. ``False`` (default) reads
        ``jac(x)`` as ``(m, n)``; ``True`` reads it as ``(n, m)``. Ignored
        without `Dfun`. A runtime variable is fine.
    ftol : float, optional
        Relative error desired in the sum of squares. Default 1.49012e-8.
    xtol : float, optional
        Relative error desired in the solution. Default 1.49012e-8.
    gtol : float, optional
        Orthogonality desired between the residual vector and the Jacobian
        columns. Default 0.0.
    maxfev : int, optional
        Maximum callback evaluations. ``0`` (default) means ``200*(n+1)``, or
        ``100*(n+1)`` when ``Dfun`` is given.
    epsfcn : float or None, optional
        Forward-difference step. ``None`` (default) resolves to machine
        epsilon. Ignored on the `Dfun` path.
    factor : float, optional
        Initial step bound, in ``(0.1, 100)``. Default 100.
    diag : sequence or None, optional
        Per-variable scale factors, all positive. An array, list or tuple.
        The first ``x0.size`` entries are used and any others ignored.
        ``None`` (default) lets MINPACK derive them (``mode = 1``); a value
        selects ``mode = 2``.
    m : int, optional
        Number of residuals. ``-1`` (default) reads it from the callback by
        calling it once. A value skips that call.
    validate : bool, optional
        ``True`` (default) raises when the returned solution is degenerate.
        ``False`` reports the same conditions through `ier` instead. The
        separate check that the callback wrote ``fvec`` at all is always
        on.

    Returns
    -------
    x : ndarray, shape (n,)
        The solution, 1-D.
    cov_x : ndarray of shape (n, n), or None
        Only with ``full_output=True``. ``(J^T J)^-1``, NOT scaled by the
        residual variance. ``None`` when `ier` is outside ``{1, 2, 3, 4}``
        or the triangular factor is singular.
    infodict : LsqInfo or LsqInfoJ
        Only with ``full_output=True``. Namedtuple with fields ``fvec``,
        ``nfev``, ``fjac``, ``ipvt`` and ``qtf``, plus ``njev`` after ``nfev``
        on the `Dfun` path, reached as attributes. ``fjac`` has shape
        ``(n, m)`` and ``ipvt`` is 0-based.
    mesg : str
        Only with ``full_output=True``. The message matching `ier`.
    ier : int
        MINPACK's status. ``1`` to ``4`` are the success set; ``5`` `maxfev`
        reached, ``6`` `ftol` too small, ``7`` `xtol` too small, ``8`` `gtol`
        too small. ``-1`` and ``-2`` are this package's degenerate-solution
        codes, reachable only under ``validate=False``: ``-1`` the residual is
        identically zero, ``-2`` the reported residual disagrees with a fresh
        evaluation at `x`.

    Raises
    ------
    TypeError
        If MINPACK reports ``ier = 0``, or if ``n`` exceeds ``m``. Also if
        `Dfun` returns an array of the wrong shape, or if `diag` has fewer
        entries than `x0`.
    ValueError
        If `func` or `Dfun` is not a plain ``@njit`` function; if an `args`
        entry is neither a real number nor an array of real numbers; or,
        under ``validate=True``, if the returned solution is degenerate.

    See Also
    --------
    scipy.optimize.leastsq : The scipy routine this mirrors.
    scijit.optimize.curve_fit : Fit a model to data, built on this.
    scijit.optimize.root : ``method='lm'`` reaches the same driver.
    scijit.optimize.lsq_linear : Bounded LINEAR least squares.

    Notes
    -----
    `infodict` is a namedtuple rather than a dict. ``info['nfev']``,
    ``info.get('nfev')``, ``info.keys()``, ``info.values()``,
    ``info.items()`` and ``'nfev' in info`` all work, from Python and from
    inside ``@njit``, and its field order is scipy's key insertion order, so
    positional unpacking agrees too. Iterating it yields VALUES, where
    iterating a dict yields keys.

    ``nfev`` counts the evaluation this package makes before the solver runs
    to check that the callback writes ``fvec``, and MINPACK's own count. The
    call that reads the residual count off the callback, the calls the
    degenerate-solution check makes when the final residual is all zero, and
    the call `Dfun` costs for its shape check are not counted, so ``nfev`` is
    lower than the number of times the callback ran by between one and
    ``n + 2``.

    `cov_x` is ``None`` in the two cases scipy returns ``None``. Its type is
    ``Optional(float64[:, :])``, which numba unwraps with a runtime check:
    code that indexes it keeps working wherever the covariance exists, and
    gets a ``TypeError`` naming the array type where it does not.

    Each `args` entry must be a real number or an array of real numbers. A
    mixed tuple is served: ``(matrix, scale)`` reaches the callback as two
    separate parameters and returns ``[3., 3.]`` on the residual
    ``x - matrix.diagonal() * scale``, which is what scipy returns on the
    same call. An entry of any other type raises.

    ``ier = 0`` raises ``TypeError('Improper input parameters.')``, scipy's
    class and scipy's text.

    ``args=None`` raises. scipy reads it as the one-item tuple ``(None,)``
    and calls ``f(x, None)``, and ``None`` is not a real number.

    A `diag` with fewer entries than `x0` raises before the solver runs.
    scipy reads past the end of the array instead, and what MINPACK then does
    depends on the values that happen to follow it.

    `cov_x` is built by inverting the QR factor with a transcription of
    LAPACK ``dtrti2``. Measured against ``scipy.linalg.lapack.dtrtri``
    2026-08-10, five random well-conditioned upper-triangular matrices at
    each of fourteen sizes: bit identical for ``n <= 16``, worst RELATIVE
    difference 9.766e-13 at ``n = 32`` and 6.194e-11 at ``n = 200``, because
    the installed ``dtrtri`` blocks from ``n = 32``.

    `validate` has no scipy counterpart; its default reports a degenerate
    solution that scipy would return silently.

    Safe to call from a ``numba.prange`` loop. MINPACK reaches the callback
    through a Fortran module variable, which carries ``!$omp threadprivate``
    and so resolves to one slot per thread.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html

    Examples
    --------
    Fit a straight line to noisy measurements by minimizing its residuals. The
    data arrays are built once and reach the residual through `args`, so no
    array is rebuilt on an evaluation:

    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import leastsq
    >>> T = np.linspace(0.0, 4.0, 20)
    >>> rng = np.random.default_rng(1)
    >>> Y = 2.0 * T + 1.0 + rng.normal(0.0, 0.1, T.size)   # noisy line
    >>> @njit
    ... def resid(p, t, y):
    ...     return y - (p[0] * t + p[1])
    >>> @njit
    ... def run():
    ...     return leastsq(resid, np.array([0.0, 0.0]), args=(T, Y))
    >>> p, ier = run()
    >>> np.round(p, 3)
    array([1.984, 1.037])
    >>> ier in (1, 2, 3, 4)
    True
    """
    at = _as_args_tuple(args)
    kinds = _arg_kinds(at)
    ca = _cb_args(at, kinds)
    pyf = _front_pyfunc(func, len(kinds), 'leastsq', _ADDRESS_MSG_LEASTSQ)
    x = _as_x0(x0)
    a = _pack_args(at)
    n = x.size
    eps = _EPS if epsfcn is None else float(epsfcn)
    if diag is None:
        dd, mode = _lsq_diag_auto(n)
    else:
        dd, mode = _lsq_diag_given(np.asarray(diag, dtype=np.float64), n)

    if Dfun is None:
        mm = int(m) if m >= 0 else int(np.asarray(func(x, *ca)).size)
        fp = _adapter_lsq_resid(pyf, kinds).address
        ab = _prepend_nm(a, n, mm)
        mf = _lsq_maxfev(n, maxfev, False)
        x, fvec, fjac, ipvt, qtf, nfev, ier = _core_lmdif(
            fp, x, ab, mm, ftol, xtol, gtol, mf, eps, mode, dd, factor,
            validate)
        if not full_output:
            _lsq_raise_bad(ier)
            _lsq_warn_bad(ier, ftol, xtol, gtol, mf)
            return x, int(ier)
        return (x, _lsq_cov(fjac, ipvt, n, ier),
                LsqInfo(fvec, nfev, fjac, ipvt, qtf),
                _lsq_mesg(ier, ftol, xtol, gtol, mf), int(ier))

    pyj = _front_pyfunc(Dfun, len(kinds), 'leastsq', _ADDRESS_MSG_LEASTSQ)
    mm = int(m) if m >= 0 else int(np.asarray(func(x, *ca)).size)
    jv = np.atleast_1d(Dfun(x, *ca))
    w0, w1 = (n, mm) if col_deriv else (mm, n)
    jpfx = _shape_prefix('leastsq', 'Dfun', pyj)
    if jv.ndim != 2:
        raise TypeError("%s(%d, %d) but it is %s." % (jpfx, w0, w1, jv.shape))
    _check_jac_shape(jpfx, np.int64(jv.shape[0]), np.int64(jv.shape[1]),
                     np.int64(w0), np.int64(w1))
    fp = _adapter_lsq_jac(pyf, pyj, bool(col_deriv), kinds).address
    ab = _prepend_nm(a, n, mm)
    mf = _lsq_maxfev(n, maxfev, True)
    x, fvec, fjac, ipvt, qtf, nfev, njev, ier = _core_lmder(
        fp, x, ab, mm, ftol, xtol, gtol, mf, mode, dd, factor, validate)
    if not full_output:
        _lsq_raise_bad(ier)
        _lsq_warn_bad(ier, ftol, xtol, gtol, mf)
        return x, int(ier)
    return (x, _lsq_cov(fjac, ipvt, n, ier),
            LsqInfoJ(fvec, nfev, njev, fjac, ipvt, qtf),
            _lsq_mesg(ier, ftol, xtol, gtol, mf), int(ier))


@njit
def _leastsq_ptr(fp, x0, args, m, ftol, xtol, gtol, maxfev, epsfcn, factor,
                 diag, validate):
    """`leastsq(full_output=True)` driven by a raw ``@cfunc(minpack_sig)``.

    `curve_fit` builds its residual ``@cfunc`` from the user's MODEL and hands
    the address to the solver, so it needs the address route that the public
    `leastsq` no longer offers.  Same core, same five-tuple return, and the
    defaults `leastsq` would have resolved are resolved here.
    """
    n = x0.size
    if diag is None:
        dd, mode = _lsq_diag_auto(n)
    else:
        dd, mode = _lsq_diag_given(diag, n)
    if epsfcn is None:
        eps = _EPS
    else:
        eps = np.float64(epsfcn)
    mf = _lsq_maxfev(n, maxfev, False)
    x, fvec, fjac, ipvt, qtf, nfev, ier = _core_lmdif(
        fp, x0, args, m, ftol, xtol, gtol, mf, eps, mode, dd, factor,
        validate)
    return (x, _lsq_cov(fjac, ipvt, n, ier),
            LsqInfo(fvec, nfev, fjac, ipvt, qtf),
            _lsq_mesg(ier, ftol, xtol, gtol, mf), ier)


_M_REQUIRED = (
    "m is required when func is a raw @cfunc .address: a pointer carries no "
    "arity, so the residual count cannot be read from it. Pass a plain @njit "
    "f(x) -> residuals instead and m is inferred, as scipy does")


@overload(leastsq, prefer_literal=True)
def _leastsq_ovl(func, x0, args=(), Dfun=None, full_output=False,
                 col_deriv=False, ftol=1.49012e-8, xtol=1.49012e-8, gtol=0.0,
                 maxfev=0, epsfcn=None, factor=100.0, diag=None, m=-1,
                 validate=True):
    """@njit implementation of `leastsq`, resolved at compile time.

    The callback is the reason this exists. ``func`` is a plain ``@njit``
    residual, from which a pointer-shaped adapter is built here and its
    address baked into the body -- an address is a compile-time constant, so
    it cannot be chosen while the code runs. ``full_output`` picks the return
    type and ``Dfun`` picks between `_core_lmdif` and `_core_lmder`. Returning
    ``None`` declines the call, which numba reports as a TypingError
    naming the argument that could not be served.
    """
    kinds = _arg_kinds_ty(_args_types(args))
    nargs = len(kinds)
    fo = _lit_bool(full_output)
    if fo is None:
        return None                        # -> TypingError, as designed
    cd_runtime, addr_t, addr_f = False, 0, 0
    no_jac = _is_none(Dfun)
    no_diag = _is_none(diag)
    no_eps = _is_none(epsfcn)

    # Resolve the callback AT COMPILE TIME: the adapter is built here and its
    # address baked in, so the callback never becomes user-facing.  An
    # address is refused rather than reinterpreted.
    pyf = _front_pyfunc_ty(func, nargs, 'leastsq', _ADDRESS_MSG_LEASTSQ)
    jpfx = ''
    if no_jac:
        addr = _adapter_lsq_resid(pyf, kinds).address
    else:
        pyj = _front_pyfunc_ty(Dfun, nargs, 'leastsq', _ADDRESS_MSG_LEASTSQ)
        jpfx = _shape_prefix('leastsq', 'Dfun', pyj)
        # `col_deriv` selects a different cfunc rather than a runtime
        # branch: an address is a compile-time constant.  A literal or
        # omitted flag builds one adapter; a runtime variable builds both
        # and picks between two baked addresses, so no call is refused.
        cd = _lit_bool(col_deriv)
        addr = _adapter_lsq_jac(pyf, pyj, bool(cd), kinds).address
        addr_t = addr if cd is not None else _adapter_lsq_jac(
            pyf, pyj, True, kinds).address
        addr_f = addr if cd is not None else _adapter_lsq_jac(
            pyf, pyj, False, kinds).address
        cd_runtime = cd is None

    if no_jac:
        def impl(func, x0, args=(), Dfun=None, full_output=False,
                 col_deriv=False, ftol=1.49012e-8, xtol=1.49012e-8, gtol=0.0,
                 maxfev=0, epsfcn=None, factor=100.0, diag=None, m=-1,
                 validate=True):
            x = _as_x0(x0)
            a = _pack_args(args)
            n = x.size
            if no_eps:
                eps = _EPS
            else:
                eps = np.float64(epsfcn)
            if no_diag:
                dd, mode = _lsq_diag_auto(n)
            else:
                dd, mode = _lsq_diag_given(diag, n)
            if m >= 0:
                mm = np.int64(m)
            else:
                mm = np.int64(_call_cb(func, x, args).size)
            fp, ab = addr, _prepend_nm(a, n, mm)
            mf = _lsq_maxfev(n, maxfev, False)
            x, fvec, fjac, ipvt, qtf, nfev, ier = _core_lmdif(
                fp, x, ab, mm, ftol, xtol, gtol, mf, eps, mode, dd, factor,
                validate)
            ier = np.int64(ier)   # python's int, not MINPACK's int32
            if fo:
                return (x, _lsq_cov(fjac, ipvt, n, ier),
                        LsqInfo(fvec, nfev, fjac, ipvt, qtf),
                        _lsq_mesg(ier, ftol, xtol, gtol, mf), ier)
            _lsq_raise_bad(ier)
            _lsq_warn_bad(ier, ftol, xtol, gtol, mf)
            return x, ier
        return impl

    def impl(func, x0, args=(), Dfun=None, full_output=False,
             col_deriv=False, ftol=1.49012e-8, xtol=1.49012e-8, gtol=0.0,
             maxfev=0, epsfcn=None, factor=100.0, diag=None, m=-1,
             validate=True):
        x = _as_x0(x0)
        a = _pack_args(args)
        n = x.size
        if no_diag:
            dd, mode = _lsq_diag_auto(n)
        else:
            dd, mode = _lsq_diag_given(diag, n)
        if m >= 0:
            mm = np.int64(m)
        else:
            mm = np.int64(_call_cb(func, x, args).size)
        jv = _call_cb(Dfun, x, args)
        if col_deriv:
            w0, w1 = np.int64(n), mm
        else:
            w0, w1 = mm, np.int64(n)
        _check_jac_shape(jpfx, np.int64(jv.shape[0]), np.int64(jv.shape[1]),
                         w0, w1)
        if cd_runtime:
            fp = addr_t if col_deriv else addr_f
        else:
            fp = addr
        ab = _prepend_nm(a, n, mm)
        mf = _lsq_maxfev(n, maxfev, True)
        x, fvec, fjac, ipvt, qtf, nfev, njev, ier = _core_lmder(
            fp, x, ab, mm, ftol, xtol, gtol, mf, mode, dd, factor, validate)
        ier = np.int64(ier)   # python's int, not MINPACK's int32
        if fo:
            return (x, _lsq_cov(fjac, ipvt, n, ier),
                    LsqInfoJ(fvec, nfev, njev, fjac, ipvt, qtf),
                    _lsq_mesg(ier, ftol, xtol, gtol, mf), ier)
        _lsq_raise_bad(ier)
        _lsq_warn_bad(ier, ftol, xtol, gtol, mf)
        return x, ier
    return impl


# ==========================================================================
# scipy.optimize.root, its two MINPACK-backed methods
#
# scipy's `root` is a dispatcher: 'hybr' runs `_root_hybr`, which is fsolve's
# engine, and 'lm' runs `leastsq`.  So does this one -- the 'hybr' arm calls
# the same `_core_hybrd`/`_core_hybrj` that `fsolve` uses, and the 'lm'
# arm calls the `_core_lmdif`/`_core_lmder` above.  No solver logic lives here.
# ==========================================================================

#: ``root(method='hybr')`` result, scipy's ``OptimizeResult`` key set.
RootHybr = _opt_result(['x', 'success', 'status', 'method',
                        'nfev', 'fjac', 'r', 'qtf', 'fun',
                        'message'])
#: ``root(method='hybr', jac=...)`` -- adds ``njev``.
RootHybrJ = _opt_result(
    ['x', 'success', 'status', 'method', 'nfev', 'njev',
     'fjac', 'r', 'qtf', 'fun', 'message'])
#: ``root(method='lm')`` result, scipy's ``OptimizeResult`` key set.
RootLm = _opt_result(['x', 'message', 'status', 'success', 'cov_x',
                      'fun', 'method', 'nfev', 'fjac', 'ipvt',
                      'qtf'])
#: ``root(method='lm', jac=...)`` -- adds ``njev``.
RootLmJ = _opt_result(
    ['x', 'message', 'status', 'success', 'cov_x', 'fun',
     'method', 'nfev', 'njev', 'fjac', 'ipvt', 'qtf'])

_ROOT_METHODS = ('hybr', 'lm')

_ROOT_OPTION_KEYS = {
    'hybr': ('col_deriv', 'xtol', 'maxfev', 'band', 'eps', 'factor', 'diag'),
    'lm': ('col_deriv', 'ftol', 'xtol', 'gtol', 'maxiter', 'eps', 'factor',
           'diag'),
}

#: What kind of value each `root` option key carries. A dict literal whose
#: values are all of one type reaches the overload as a `DictType`, and a
#: `DictType` of numbers cannot hold `band`'s tuple or `diag`'s array.
_ROOT_OPT_KIND = {
    'col_deriv': 'num', 'xtol': 'num', 'ftol': 'num', 'gtol': 'num',
    'maxfev': 'num', 'maxiter': 'num', 'eps': 'num', 'factor': 'num',
    'band': 'tuple', 'diag': 'array',
}

_ROOT_OPT_KIND_MSG = {
    'band': "root: options['band'] is a (ml, mu) tuple, and a dict whose "
            "values are all numbers cannot carry one. Write the pair into "
            "the dict alongside a value of another type, or pass band=.",
    'diag': "root: options['diag'] is an array, and a dict whose values are "
            "all numbers cannot carry one. Write the array into the dict "
            "alongside a value of another type, or pass diag=.",
}


def _opt_carries(vt, kind):
    """Whether a dict whose value type is `vt` can hold a `kind` option."""
    if kind == 'num':
        return isinstance(vt, types.Number)
    if kind == 'tuple':
        return isinstance(vt, types.BaseTuple)
    return isinstance(vt, types.Array)


def _emit_root_opt_warning(msg):
    """scipy's own `OptimizeWarning` for option keys a method does not read."""
    # `OptimizeWarning` is imported here rather than at module scope: `_lsq`
    # imports `leastsq` from this module, so a top-level import would close
    # the cycle.
    from ._lsq import OptimizeWarning
    warnings.warn("Unknown solver options: " + msg, OptimizeWarning,
                  stacklevel=4)


@njit
def _root_warn_opt(msg):
    """`_emit_root_opt_warning` from compiled code.

    The ``objmode`` block sits in its own function so that lowering it
    pickles this dispatcher rather than a caller holding a ctypes entry
    point.
    """
    with objmode():
        _emit_root_opt_warning(msg)


def _mk_root_opt_scan(allowed):
    """The unknown-key scan for a dict whose keys are not known until run time.

    A dict literal whose values share one type carries no key information
    into the overload, so the comparison against `allowed` happens in the
    compiled body. Insertion order is preserved, so the message lists the
    keys in the order they were written.
    """
    @njit
    def scan(options):
        msg = ''
        for k in options:
            known = False
            for a in allowed:
                if k == a:
                    known = True
                    break
            if not known:
                msg = k if msg == '' else msg + ', ' + k
        if msg != '':
            _root_warn_opt(msg)
    return scan


def _root_options_src(options, meth, cb_msg, jac_src, cd_base,
                      no_tol, no_eps, no_band):
    """Source of `root`'s compiled body when `options` is given.

    ``None`` declines the call, which numba reports as a TypingError.

    The keys are read here and `root` is called again with `options` set to
    ``None``, so the four solver bodies are reached unchanged and each arm
    has one implementation rather than two. `method` and `jac` are written
    into the generated call as literals, which is what keeps the second call
    on the same compile-time branch as the first.

    Two dict shapes arrive. A dict literal with values of two different
    types is a `LiteralStrKeyDict`, whose `.fields` are readable here, so
    every key is resolved at compile time. A dict whose values share one
    type is a `DictType`, whose keys are not, so each read becomes
    ``options['k'] if 'k' in options else <the argument>`` and the
    unknown-key scan moves into the body.
    """
    allowed = _ROOT_OPTION_KEYS[meth]
    fields = getattr(options, 'fields', None)
    warn_msg = None
    scan_keys = None
    guards = []
    diag_runtime = False
    if fields is not None:
        served = [k for k in fields if k in allowed]
        unknown = [k for k in fields if k not in allowed]
        if unknown:
            warn_msg = ', '.join(unknown)

        def read(k, base):
            return "options['%s']" % k if k in served else base
    elif (isinstance(options, types.DictType)
            and options.key_type == types.unicode_type):
        vt = options.value_type
        served = [k for k in allowed if _opt_carries(vt, _ROOT_OPT_KIND[k])]
        if not served:
            return None
        scan_keys = tuple(served)
        for k in ('band', 'diag'):
            if k in allowed and k not in served:
                guards.append("    if '%s' in options:\n"
                              "        raise ValueError(%r)"
                              % (k, _ROOT_OPT_KIND_MSG[k]))
        diag_runtime = 'diag' in served

        def read(k, base):
            if k not in served:
                return base
            return "(options['%s'] if '%s' in options else %s)" % (k, k, base)
    else:
        return None

    # scipy runs `options.setdefault('xtol', tol)`, so an explicit
    # `options['xtol']` WINS over `tol`. Resolving `tol` into the base and
    # passing `tol=None` on keeps that order.
    xt_base = 'xtol' if no_tol else 'np.float64(tol)'
    # `eps` resolves to a float here so that the second call never has to
    # decide between `None` and a value at run time. MINPACK derives the
    # difference step as sqrt(max(eps, epsmch)), and the two spellings
    # produce the identical step, which is measured in the `hybr` arm below.
    eps_base = (('_EPS' if meth == 'hybr' else '0.0') if no_eps
                else 'np.float64(eps)')
    # `(-10, -10)` is the pair the solver arm builds for an absent `band`,
    # so it is the value that means "no band" rather than a sentinel this
    # function invents.
    band_base = ('(-10, -10)' if (scan_keys and 'band' in scan_keys
                                  and no_band) else 'band')

    def call(diag_expr, pad=''):
        return (pad
                + "    return _root(fun, x0, args, %r, %s, None, None, None,\n"
                "                 %s, np.float64(%s), np.float64(%s),\n"
                "                 np.float64(%s), np.int64(%s), np.int64(%s),\n"
                "                 %s, np.float64(%s), np.float64(%s), %s,\n"
                "                 m, validate)"
                % (meth, jac_src,
                   read('col_deriv', cd_base),
                   read('xtol', xt_base),
                   read('ftol', 'ftol'),
                   read('gtol', 'gtol'),
                   read('maxfev', 'maxfev'),
                   read('maxiter', 'maxiter'),
                   read('band', band_base),
                   read('eps', eps_base),
                   read('factor', 'factor'),
                   diag_expr))

    src = ["def impl(fun, x0, args=(), method='hybr', jac=None, tol=None,",
           "         callback=None, options=None, col_deriv=0,",
           "         xtol=1.49012e-8, ftol=1.49012e-8, gtol=0.0, maxfev=0,",
           "         maxiter=0, band=None, eps=None, factor=100.0,",
           "         diag=None, m=-1, validate=True):"]
    # scipy warns about `callback` before it warns about an unknown option
    # key, so both warnings are emitted here and the second call is made
    # with `callback=None`.
    if cb_msg is not None:
        src.append("    _root_warn_cb(%r)" % cb_msg)
    if warn_msg is not None:
        src.append("    _root_warn_opt(%r)" % warn_msg)
    if scan_keys is not None:
        src.append("    _opt_scan(options)")
    src.extend(guards)
    if diag_runtime:
        # `diag` is an array where its absence is `None`, and one expression
        # cannot be both, so the branch is on the call rather than on the
        # value. Both arms return the same namedtuple.
        src.append("    if 'diag' in options:")
        src.append(call("options['diag']", '    '))
        src.append(call('diag'))
    else:
        src.append(call(read('diag', 'diag')))
    ns = {'_root': root, 'np': np, '_EPS': _EPS,
          '_root_warn_cb': _root_warn_cb, '_root_warn_opt': _root_warn_opt,
          '_opt_scan': (_mk_root_opt_scan(scan_keys)
                        if scan_keys is not None else None)}
    exec("\n".join(src), ns)                              # noqa: S102
    return ns['impl']


def _root_method(method):
    """scipy's resolution: ``.lower()``, no strip, unknown -> ValueError."""
    meth = method.lower()
    if meth not in _ROOT_METHODS:
        raise ValueError(
            f"Unknown solver {method}. scijit wraps the two MINPACK-backed "
            f"methods, 'hybr' and 'lm'; scipy's other eight are pure Python "
            f"and have no compiled backend to reach")
    return meth


def root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None,
         options=None, col_deriv=0, xtol=1.49012e-8, ftol=1.49012e-8,
         gtol=0.0, maxfev=0, maxiter=0, band=None, eps=None, factor=100.0,
         diag=None, m=-1, validate=True):
    """Find a root of a vector function.

    Dispatches over MINPACK's two drivers: a modified Powell hybrid method,
    and Levenberg-Marquardt for a non-square system.

    Callable from Python and from inside ``@njit``.

    Parameters
    ----------
    fun : callable
        A plain ``@njit`` ``f(x, *args) -> residuals``.
    x0 : float or array_like
        Initial guess. A scalar, list or tuple is accepted. Any rank is
        FLATTENED.
    args : tuple, optional
        Extra parameters, passed to the callback as ``f(x, *args)``, one
        per entry. Entries may be scalars or arrays and need not share a
        type. A non-tuple `args` is coerced to the ONE argument ``(args,)``.
    method : {'hybr', 'lm'}, optional
        ``'hybr'`` (default) modified Powell hybrid, ``'lm'``
        Levenberg-Marquardt. Matched case-insensitively and NOT stripped, so
        ``'hybr '`` raises. Must be a compile-time constant inside ``@njit``:
        it selects the return type.
    jac : callable, bool or None, optional
        ``None`` (default) uses a forward-difference Jacobian. A plain
        ``@njit`` ``jac(x, *args)`` selects the analytic path. Any other
        value is read for its truth value: a truthy one means `fun` returns
        the pair ``(residuals, jacobian)`` and a falsy one means the same as
        ``None``, so ``1`` and ``0`` behave as ``True`` and ``False``. An
        integer other than ``0`` or ``1`` is refused. A non-callable `jac`
        must be written out inside ``@njit``, not held in a variable.
    tol : float or None, optional
        Convenience tolerance. ``None`` (default) leaves the per-method
        defaults. A value sets ``xtol`` on BOTH methods, and loses to an
        explicit ``options['xtol']``.
    callback : callable, optional
        Accepted and ignored, with a ``RuntimeWarning``. Neither method reads
        it.
    options : dict or None, optional
        Solver options. Recognised keys are ``col_deriv``, ``xtol``,
        ``maxfev``, ``band``, ``eps``, ``factor``, ``diag`` for ``'hybr'``;
        ``col_deriv``, ``ftol``, ``xtol``, ``gtol``, ``maxiter``, ``eps``,
        ``factor``, ``diag`` for ``'lm'``. A key the method does not read
        draws an ``OptimizeWarning`` and is ignored, and every key is also
        reachable as the argument of the same name.

        Inside ``@njit`` a dict whose values are all of ONE type carries
        only the keys of that kind, so a dict of numbers cannot hold
        ``band``'s tuple or ``diag``'s array and raises when it names
        either. A dict holding values of two different types carries every
        key: ``{'xtol': 1e-10, 'band': (0, 0)}`` and
        ``{'xtol': 1e-10, 'diag': d}`` both work. An empty dict literal is
        not typeable by numba; ``None`` is the empty case.
    col_deriv : int, optional
        A statement about `jac`. ``0`` (default) reads the Jacobian as
        ``J[i, j] = dF_i/dx_j``; a nonzero value reads it as already
        transposed, which on ``'lm'`` means the shape ``(n, m)`` rather than
        ``(m, n)``. Ignored without `jac`. A runtime variable is fine.
    xtol, ftol, gtol : float, optional
        Tolerances. ``ftol``/``gtol`` are read by ``'lm'`` only.
    maxfev : int, optional
        ``'hybr'`` evaluation budget. ``0`` (default) means ``200*(n+1)``, or
        ``100*(n+1)`` with ``jac``.
    maxiter : int, optional
        ``'lm'`` evaluation budget.
    band : tuple of int or None, optional
        ``(ml, mu)`` for a banded Jacobian. ``'hybr'`` only.
    eps : float or None, optional
        Forward-difference step. ``None`` (default) resolves to machine
        epsilon on ``'hybr'`` and to ``0.0`` on ``'lm'``. MINPACK derives the
        step as ``sqrt(max(eps_given, eps_machine))``, so the two spellings
        produce an identical step and identical evaluation counts.
    factor : float, optional
        Initial step bound, in ``(0.1, 100)``. Default 100.
    diag : sequence or None, optional
        Per-variable scale factors, all positive. An array, list or tuple.
        The first ``x0.size`` entries are used and any others ignored.
    m : int, optional
        Residual count, ``'lm'`` only. ``-1`` (default) reads it from the
        callback. A value skips that call.
    validate : bool, optional
        ``True`` (default) raises when the returned solution is degenerate;
        ``False`` reports it as ``status = -1`` or ``-2`` instead. The check
        that the callback wrote anything at all is always on.

    Returns
    -------
    result : RootHybr, RootHybrJ, RootLm or RootLmJ
        A namedtuple whose fields depend on `method` and on whether `jac` was
        given, reached as attributes: ``x``, ``success``, ``status``,
        ``method``,
        ``fun``, ``fjac``, ``qtf``, ``nfev``, ``message``, plus ``r`` on
        ``'hybr'``, ``ipvt`` and ``cov_x`` on ``'lm'``, and ``njev`` whenever
        ``jac`` is given. ``fun``, ``fjac``, ``qtf`` and ``r`` are only
        meaningful when ``success``.

    Raises
    ------
    TypeError
        If the residual count disagrees with ``x0.size`` on ``'hybr'``, if
        `jac` returns an array of the wrong shape, or if `diag` has fewer
        entries than `x0`.
    ValueError
        If `method` is neither ``'hybr'`` nor ``'lm'``; if `fun` is not a
        plain ``@njit`` function; if an `args` entry is neither a real number
        nor an array of real numbers; if an `options` dict inside ``@njit``
        names ``band`` or ``diag`` and its value type cannot carry one; or,
        under ``validate=True``, if the returned solution is degenerate.

    See Also
    --------
    scipy.optimize.root : The scipy routine this mirrors.
    scijit.optimize.fsolve : ``'hybr'`` alone, returning the ``fsolve`` shape.
    scijit.optimize.leastsq : ``'lm'`` alone, returning the ``leastsq`` shape.
    scijit.optimize.root_scalar : One equation in one variable.

    Notes
    -----
    scipy dispatches ``root`` over ten methods. This one implements ``hybr``
    and ``lm``, calling the MINPACK drivers directly. The remaining eight
    (``broyden1``, ``broyden2``, ``anderson``, ``linearmixing``,
    ``diagbroyden``, ``excitingmixing``, ``krylov``, ``df-sane``) are pure
    Python in scipy and are planned, not unavailable in principle.

    The result is a namedtuple rather than an ``OptimizeResult``.
    ``res['x']``, ``res.get('x')``, ``res.keys()``, ``res.values()``,
    ``res.items()`` and ``'x' in res`` all work, from Python and from inside
    ``@njit``, and its field order is scipy's key insertion order, which
    differs between the two methods as it does in scipy, so positional
    unpacking agrees too. Iterating it yields VALUES, where iterating a dict
    yields keys, and it is not an instance of
    ``scipy.optimize.OptimizeResult``.

    `method`, and whether `jac` is ``None``, must be compile-time constants
    inside ``@njit``: both change the return type.

    `callback` draws a ``RuntimeWarning`` and is ignored, and an unrecognised
    `options` key draws an ``OptimizeWarning`` and is ignored. Both are
    scipy's class and scipy's text on these two methods.

    ``cov_x`` on ``'lm'`` is ``None`` in the two cases scipy returns
    ``None``: an `status` outside ``{1, 2, 3, 4}``, or a singular triangular
    factor.

    ``nfev`` counts the evaluations this package makes before the solver
    runs as well as MINPACK's own: one on ``'hybr'`` to read the residual
    count, two more on ``'hybr'`` without `jac` under ``validate=True`` for
    the read probe, and one on ``'lm'`` to check that the callback writes
    the residual. scipy counts two of its own, so on ``'hybr'`` it is one
    higher than scipy's without `jac` under ``validate=True`` and one lower
    on the other two, and on ``'lm'`` it is one lower. On ``'lm'`` the
    evaluation that reads the residual length is not counted, as scipy does
    not count its own, and nor is the one `jac` costs for its shape check.

    ``args=None`` raises. scipy reads it as the one-item tuple ``(None,)``
    and calls ``f(x, None)``, and ``None`` is not a real number.

    A `diag` with fewer entries than `x0` raises before the solver runs.
    scipy reads past the end of the array instead, and what MINPACK then does
    depends on the values that happen to follow it. A non-positive entry
    reaches MINPACK, which reports ``status = 0``.

    `validate` has no scipy counterpart; its default reports a degenerate
    solution that scipy would return silently.

    Safe to call from a ``numba.prange`` loop. MINPACK reaches the callback
    through a Fortran module variable, which carries ``!$omp threadprivate``
    and so resolves to one slot per thread.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import root
    >>> @njit
    ... def f(x):
    ...     return np.array([x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0,
    ...                      0.5 * (x[1] - x[0]) ** 3 + x[1]])
    >>> @njit
    ... def run():
    ...     return root(f, np.array([0.0, 0.0]))
    >>> res = run()
    >>> res.x
    array([0.8411639, 0.1588361])
    >>> res.success
    True
    """
    meth = _root_method(method)
    # scipy's rule, verbatim: any non-callable `jac` is read for its truth
    # value, a truthy one meaning `fun` returns `(f, J)` and a falsy one
    # meaning no Jacobian.  So `1` and `0` behave as `True` and `False`, and
    # `None` falls through the same branch to no Jacobian.
    jac_pair = False
    if not callable(jac):
        _jac_not_an_address(jac)
        jac_pair = bool(jac)
        jac = fun if jac_pair else None
    if callback is not None:
        warnings.warn(f'Method {method} does not accept callback.',
                      RuntimeWarning, stacklevel=2)
    if options:
        allowed = _ROOT_OPTION_KEYS[meth]
        unknown = [k for k in options if k not in allowed]
        if unknown:
            # `OptimizeWarning` is imported here rather than at module scope:
            # `_lsq` imports `leastsq` from this module, so a top-level import
            # would close the cycle.
            from ._lsq import OptimizeWarning
            warnings.warn(
                "Unknown solver options: " + ", ".join(str(k) for k in unknown),
                OptimizeWarning, stacklevel=4)
        col_deriv = options.get('col_deriv', col_deriv)
        eps = options.get('eps', eps)
        factor = options.get('factor', factor)
        diag = options.get('diag', diag)
        if meth == 'hybr':
            maxfev = options.get('maxfev', maxfev)
            band = options.get('band', band)
        else:
            ftol = options.get('ftol', ftol)
            gtol = options.get('gtol', gtol)
            maxiter = options.get('maxiter', maxiter)
        # scipy runs `options.setdefault('xtol', tol)`, so an explicit
        # `options['xtol']` WINS over `tol`.  Resolving `xtol` after the rest
        # of the dict keeps that order.
        if 'xtol' in options:
            xtol = options['xtol']
        elif tol is not None:
            xtol = tol
    elif tol is not None:
        xtol = tol                      # scipy sets xtol on BOTH methods

    at = _as_args_tuple(args)
    kinds = _arg_kinds(at)
    ca = _cb_args(at, kinds)
    pyf = _front_pyfunc(fun, len(kinds), 'root', _ADDRESS_MSG_ROOT)
    x = _as_x0(x0)
    a = _pack_args(at)
    n = x.size
    if diag is None:
        dd, mode = _lsq_diag_auto(n)
    else:
        dd, mode = _lsq_diag_given(np.asarray(diag, dtype=np.float64), n)

    if meth == 'hybr':
        # scipy resolves an absent `eps` to finfo.eps here and to 0.0 on the
        # `lm` branch below. DO NOT TIDY THE ASYMMETRY AWAY -- it is
        # reproduced deliberately, and it is cosmetic. MINPACK derives the
        # difference step as `eps = sqrt(max(Epsfcn, epsmch))`
        # (minpack.f90:474 in fdjac1, :553 in fdjac2), which maps 0.0 and
        # machine epsilon to the same number. Measured on a 2-D system:
        # eps None / 0.0 / finfo.eps give |dx| 0.000e+00 and the identical
        # nfev on both methods, and scipy behaves the same way.
        epsfcn = _EPS if eps is None else float(eps)
        ml, mu = (-10, -10) if band is None else (int(band[0]), int(band[1]))
        mf = _maxfev(n, maxfev, jac is not None)
        probe = fun(x, *ca)
        rv = probe[0] if jac_pair else probe
        _check_resid_len(_shape_prefix('fsolve', 'func', pyf),
                         np.int64(np.atleast_1d(rv).size), np.int64(n))
        if jac is None:
            fp = _adapter_resid(pyf, kinds).address
            ab = _prepend_n(a, n)
            xo, fvec, fjac, r, qtf, nfev, st = _core_hybrd(
                fp, x, ab, xtol, mf, ml, mu, epsfcn, mode, dd, factor,
                validate, 1)
            return RootHybr(xo, st == 1, int(st), 'hybr', nfev, fjac, r,
                            qtf, fvec, _mesg(st, mf, xtol))
        if jac_pair:
            jv, pyj = probe[1], pyf
            fp = _adapter_pair(pyf, bool(col_deriv), kinds).address
        else:
            pyj = _front_pyfunc(jac, len(kinds), 'root', _ADDRESS_MSG_ROOT)
            jv = jac(x, *ca)
            fp = _adapter_jac(pyf, pyj, bool(col_deriv), kinds).address
        jv = np.atleast_1d(jv)
        jpfx = _shape_prefix('fsolve', 'fprime', pyj)
        if jv.ndim != 2:
            raise TypeError("%s(%d, %d) but it is %s." % (jpfx, n, n, jv.shape))
        _check_jac_shape(jpfx, np.int64(jv.shape[0]), np.int64(jv.shape[1]),
                         np.int64(n), np.int64(n))
        ab = _prepend_n_pair(a, n) if jac_pair else _prepend_n(a, n)
        xo, fvec, fjac, r, qtf, nfev, njev, st = _core_hybrj(
            fp, x, ab, xtol, mf, mode, dd, factor, validate, 1)
        return RootHybrJ(xo, st == 1, int(st), 'hybr', nfev, njev, fjac, r,
                         qtf, fvec, _mesg(st, mf, xtol))

    # 0.0, not _EPS: scipy's `_root_leastsq` differs from `_root_hybr` here.
    # See the note on the hybr branch -- the two resolve to the same step.
    epsfcn = 0.0 if eps is None else float(eps)
    if jac is None:
        mm = int(m) if m >= 0 else int(np.asarray(fun(x, *ca)).size)
        fp = _adapter_lsq_resid(pyf, kinds).address
        ab = _prepend_nm(a, n, mm)
        mf = _lsq_maxfev(n, maxiter, False)
        xo, fvec, fjac, ipvt, qtf, nfev, st = _core_lmdif(
            fp, x, ab, mm, ftol, xtol, gtol, mf, epsfcn, mode, dd, factor,
            validate)
        return RootLm(xo, _lsq_mesg(st, ftol, xtol, gtol, mf), int(st),
                      1 <= st <= 4, _lsq_cov(fjac, ipvt, n, st), fvec, 'lm',
                      nfev, fjac, ipvt, qtf)
    # scipy checks `Dfun`'s shape against `m`, so the residual is evaluated
    # here only when `m` has to be inferred from it.
    if jac_pair:
        lprobe = fun(x, *ca)
        jv, pyj = lprobe[1], pyf
        mm = int(m) if m >= 0 else int(np.asarray(lprobe[0]).size)
        fp = _adapter_lsq_pair(pyf, bool(col_deriv), kinds).address
    else:
        pyj = _front_pyfunc(jac, len(kinds), 'root', _ADDRESS_MSG_ROOT)
        mm = int(m) if m >= 0 else int(np.asarray(fun(x, *ca)).size)
        jv = jac(x, *ca)
        fp = _adapter_lsq_jac(pyf, pyj, bool(col_deriv), kinds).address
    jv = np.atleast_1d(jv)
    w0, w1 = (n, mm) if col_deriv else (mm, n)
    jpfx = _shape_prefix('leastsq', 'Dfun', pyj)
    if jv.ndim != 2:
        raise TypeError("%s(%d, %d) but it is %s." % (jpfx, w0, w1, jv.shape))
    _check_jac_shape(jpfx, np.int64(jv.shape[0]), np.int64(jv.shape[1]),
                     np.int64(w0), np.int64(w1))
    ab = _prepend_nm_pair(a, n, mm) if jac_pair else _prepend_nm(a, n, mm)
    mf = _lsq_maxfev(n, maxiter, True)
    xo, fvec, fjac, ipvt, qtf, nfev, njev, st = _core_lmder(
        fp, x, ab, mm, ftol, xtol, gtol, mf, mode, dd, factor, validate)
    return RootLmJ(xo, _lsq_mesg(st, ftol, xtol, gtol, mf), int(st),
                   1 <= st <= 4, _lsq_cov(fjac, ipvt, n, st), fvec, 'lm',
                   nfev, njev, fjac, ipvt, qtf)


@overload(root, prefer_literal=True)
def _root_ovl(fun, x0, args=(), method='hybr', jac=None, tol=None,
              callback=None, options=None, col_deriv=0, xtol=1.49012e-8,
              ftol=1.49012e-8, gtol=0.0, maxfev=0, maxiter=0, band=None,
              eps=None, factor=100.0, diag=None, m=-1, validate=True):
    """@njit implementation of `root`, resolved at compile time.

    ``method`` has to be a string LITERAL, because it selects both which
    core runs and which of the four result namedtuples comes back, and a
    numba function has one return type. Alongside it the callback is
    resolved the way `_leastsq_ovl` describes: a raw address stays as it
    is, a plain ``@njit`` residual gets an adapter built and its address
    baked in. Returning ``None`` declines the call, which numba reports
    as a TypingError naming the argument that could not be served.
    """
    ms = _lit_str(method)
    if ms is None:
        return None                     # runtime method -> TypingError
    meth = _root_method(ms)
    # scipy PUBLISHES `callback` and neither method reads it: it warns and
    # carries on.  Measured on scipy 1.18, hybr and lm both make 0 callback
    # calls and both emit "Method <m> does not accept callback."  The python
    # body warns at `_minpack.py:2593`; this is the same contract compiled,
    # with the message built here and baked in as a constant.
    CB_GIVEN = not _is_none(callback)
    CB_MSG = ms
    # scipy's `if not callable(jac)`: a truthy non-callable means `fun`
    # returns `(f, J)`, a falsy one means no Jacobian, so `1` and `0` behave
    # as `True` and `False`.  The truth value has to be known while the body
    # is typed, since it picks the adapter, so a literal or an omitted
    # default is served and a runtime value declines the call.  A
    # ``types.Dispatcher`` is the callable case and passes through.
    jac_pair = False
    jac_src = 'None'
    if _is_none(jac):
        jac = None
    elif isinstance(jac, types.Dispatcher):
        jac_src = 'jac'
    elif isinstance(jac, types.Omitted):
        _jac_not_an_address_ty(jac)
        jac_pair = bool(jac.value)
        jac = fun if jac_pair else None
    elif not isinstance(jac, types.Type):
        _jac_not_an_address_ty(jac)     # numba hands the raw python default
        jac_pair = bool(jac)
        jac = fun if jac_pair else None
    elif isinstance(jac, types.Literal):
        _jac_not_an_address_ty(jac)
        jac_pair = bool(jac.literal_value)
        jac = fun if jac_pair else None
    else:
        return None                     # runtime non-callable -> TypingError
    if jac_src != 'jac':
        jac_src = 'True' if jac_pair else 'None'
    no_jac = _is_none(jac)
    no_diag = _is_none(diag)
    no_eps = _is_none(eps)
    no_band = _is_none(band)
    no_tol = _is_none(tol)
    if not _is_none(options):
        # The keys are read in `_root_options_src` and `root` is called
        # again with `options=None`, so the four bodies below serve both
        # spellings and there is one implementation of each arm.
        cd = _lit_bool(col_deriv)
        return _root_options_src(options, meth, ms if CB_GIVEN else None,
                                 jac_src,
                                 'col_deriv' if cd is None else repr(int(cd)),
                                 no_tol, no_eps, no_band)
    kinds = _arg_kinds_ty(_args_types(args))
    nargs = len(kinds)

    cd_runtime, addr_t, addr_f = False, 0, 0
    # Resolve the callback AT COMPILE TIME: the adapter is built here and its
    # address baked in, so the callback never becomes user-facing.  An
    # address is refused rather than reinterpreted.
    pyf = _front_pyfunc_ty(fun, nargs, 'root', _ADDRESS_MSG_ROOT)
    # scipy's `_check_func` is called from inside the arm, so it names the arm
    # rather than `root`: 'fsolve'/'func'/'fprime' on hybr, 'leastsq'/'Dfun'
    # on lm.
    fpfx = _shape_prefix('fsolve', 'func', pyf)
    jpfx = ''
    if no_jac:
        addr = (_adapter_resid(pyf, kinds).address if meth == 'hybr'
                else _adapter_lsq_resid(pyf, kinds).address)
    else:
        # `col_deriv` selects a different cfunc rather than a runtime
        # branch: an address is a compile-time constant.  A literal or
        # omitted flag builds one adapter; a runtime variable builds both
        # and picks between two baked addresses, so no call is refused.
        cd = _lit_bool(col_deriv)
        if jac_pair:
            _mk1 = (_adapter_pair if meth == 'hybr' else _adapter_lsq_pair)

            def _mk(pf, pj, c):
                return _mk1(pf, c, kinds)
            pyj = pyf
        else:
            _mk2 = _adapter_jac if meth == 'hybr' else _adapter_lsq_jac

            def _mk(pf, pj, c):
                return _mk2(pf, pj, c, kinds)
            pyj = _front_pyfunc_ty(jac, nargs, 'root', _ADDRESS_MSG_ROOT)
        addr = _mk(pyf, pyj, bool(cd)).address
        addr_t = addr if cd is not None else _mk(pyf, pyj, True).address
        addr_f = addr if cd is not None else _mk(pyf, pyj, False).address
        cd_runtime = cd is None
        jpfx = (_shape_prefix('fsolve', 'fprime', pyj) if meth == 'hybr'
                else _shape_prefix('leastsq', 'Dfun', pyj))

    # One probe call per callback layout, selected here rather than branched
    # on in the body: with ``jac=True`` the runtime ``jac`` is the boolean,
    # not a function, so the two spellings cannot share a call site.
    if no_jac:
        _dims = _jonly = None
    elif jac_pair:
        @njit
        def _dims(fun, jac, x, args):
            pr = _call_cb(fun, x, args)
            return pr[0], pr[1]

        @njit
        def _jonly(fun, jac, x, args):
            return _call_cb(fun, x, args)[1]
    else:
        @njit
        def _dims(fun, jac, x, args):
            return _call_cb(fun, x, args), _call_cb(jac, x, args)

        @njit
        def _jonly(fun, jac, x, args):
            return _call_cb(jac, x, args)
    # The `jac=True` adapter memoises its own Jacobian, in scratch this
    # buffer carries; see `_prepend_n_pair`.
    _pre_h = _prepend_n_pair if jac_pair else _prepend_n
    _pre_l = _prepend_nm_pair if jac_pair else _prepend_nm

    if meth == 'hybr' and no_jac:
        def impl(fun, x0, args=(), method='hybr', jac=None, tol=None,
                 callback=None, options=None, col_deriv=0, xtol=1.49012e-8,
                 ftol=1.49012e-8, gtol=0.0, maxfev=0, maxiter=0, band=None,
                 eps=None, factor=100.0, diag=None, m=-1, validate=True):
            x = _as_x0(x0)
            if CB_GIVEN:
                _root_warn_cb(CB_MSG)
            a = _pack_args(args)
            n = x.size
            xt = xtol if no_tol else np.float64(tol)
            epsfcn = _EPS if no_eps else np.float64(eps)
            if no_band:
                ml, mu = -10, -10
            else:
                ml, mu = np.int64(band[0]), np.int64(band[1])
            if no_diag:
                dd, mode = _lsq_diag_auto(n)
            else:
                dd, mode = _lsq_diag_given(diag, n)
            fp, ab = addr, _prepend_n(a, n)
            mf = _maxfev(n, maxfev, False)
            _check_resid_len(fpfx, np.int64(_call_cb(fun, x, args).size),
                             np.int64(n))
            xo, fvec, fjac, r, qtf, nfev, st = _core_hybrd(
                fp, x, ab, xt, mf, ml, mu, epsfcn, mode, dd, factor, validate,
                1)
            st = np.int64(st)   # python's int, not MINPACK's int32
            return RootHybr(xo, st == 1, st, 'hybr', nfev, fjac, r, qtf,
                            fvec, _mesg(st, mf, xt))
        return impl

    if meth == 'hybr':
        def impl(fun, x0, args=(), method='hybr', jac=None, tol=None,
                 callback=None, options=None, col_deriv=0, xtol=1.49012e-8,
                 ftol=1.49012e-8, gtol=0.0, maxfev=0, maxiter=0, band=None,
                 eps=None, factor=100.0, diag=None, m=-1, validate=True):
            x = _as_x0(x0)
            if CB_GIVEN:
                _root_warn_cb(CB_MSG)
            a = _pack_args(args)
            n = x.size
            xt = xtol if no_tol else np.float64(tol)
            if no_diag:
                dd, mode = _lsq_diag_auto(n)
            else:
                dd, mode = _lsq_diag_given(diag, n)
            if cd_runtime:
                fp = addr_t if col_deriv else addr_f
            else:
                fp = addr
            ab = _pre_h(a, n)
            mf = _maxfev(n, maxfev, True)
            rv, jv = _dims(fun, jac, x, args)
            _check_resid_len(fpfx, np.int64(rv.size), np.int64(n))
            _check_jac_shape(jpfx, np.int64(jv.shape[0]),
                             np.int64(jv.shape[1]), np.int64(n), np.int64(n))
            xo, fvec, fjac, r, qtf, nfev, njev, st = _core_hybrj(
                fp, x, ab, xt, mf, mode, dd, factor, validate, 1)
            st = np.int64(st)   # python's int, not MINPACK's int32
            return RootHybrJ(xo, st == 1, st, 'hybr', nfev, njev, fjac, r,
                             qtf, fvec, _mesg(st, mf, xt))
        return impl

    if no_jac:
        def impl(fun, x0, args=(), method='hybr', jac=None, tol=None,
                 callback=None, options=None, col_deriv=0, xtol=1.49012e-8,
                 ftol=1.49012e-8, gtol=0.0, maxfev=0, maxiter=0, band=None,
                 eps=None, factor=100.0, diag=None, m=-1, validate=True):
            x = _as_x0(x0)
            if CB_GIVEN:
                _root_warn_cb(CB_MSG)
            a = _pack_args(args)
            n = x.size
            xt = xtol if no_tol else np.float64(tol)
            epsfcn = 0.0 if no_eps else np.float64(eps)
            if no_diag:
                dd, mode = _lsq_diag_auto(n)
            else:
                dd, mode = _lsq_diag_given(diag, n)
            if m >= 0:
                mm = np.int64(m)
            else:
                mm = np.int64(_call_cb(fun, x, args).size)
            fp, ab = addr, _prepend_nm(a, n, mm)
            mf = _lsq_maxfev(n, maxiter, False)
            xo, fvec, fjac, ipvt, qtf, nfev, st = _core_lmdif(
                fp, x, ab, mm, ftol, xt, gtol, mf, epsfcn, mode, dd, factor,
                validate)
            st = np.int64(st)   # python's int, not MINPACK's int32
            return RootLm(xo, _lsq_mesg(st, ftol, xt, gtol, mf), st,
                          1 <= st <= 4, _lsq_cov(fjac, ipvt, n, st), fvec,
                          'lm', nfev, fjac, ipvt, qtf)
        return impl

    def impl(fun, x0, args=(), method='hybr', jac=None, tol=None,
             callback=None, options=None, col_deriv=0, xtol=1.49012e-8,
             ftol=1.49012e-8, gtol=0.0, maxfev=0, maxiter=0, band=None,
             eps=None, factor=100.0, diag=None, m=-1, validate=True):
        x = _as_x0(x0)
        if CB_GIVEN:
            _root_warn_cb(CB_MSG)
        a = _pack_args(args)
        n = x.size
        xt = xtol if no_tol else np.float64(tol)
        if no_diag:
            dd, mode = _lsq_diag_auto(n)
        else:
            dd, mode = _lsq_diag_given(diag, n)
        # scipy checks `Dfun`'s shape against `m`, so the residual is
        # evaluated here only when `m` has to be inferred from it.
        if m >= 0:
            mm = np.int64(m)
            jv = _jonly(fun, jac, x, args)
        else:
            rv, jv = _dims(fun, jac, x, args)
            mm = np.int64(rv.size)
        if col_deriv:
            w0, w1 = np.int64(n), mm
        else:
            w0, w1 = mm, np.int64(n)
        _check_jac_shape(jpfx, np.int64(jv.shape[0]), np.int64(jv.shape[1]),
                         w0, w1)
        if cd_runtime:
            fp = addr_t if col_deriv else addr_f
        else:
            fp = addr
        ab = _pre_l(a, n, mm)
        mf = _lsq_maxfev(n, maxiter, True)
        xo, fvec, fjac, ipvt, qtf, nfev, njev, st = _core_lmder(
            fp, x, ab, mm, ftol, xt, gtol, mf, mode, dd, factor, validate)
        st = np.int64(st)   # python's int, not MINPACK's int32
        return RootLmJ(xo, _lsq_mesg(st, ftol, xt, gtol, mf), st,
                       1 <= st <= 4, _lsq_cov(fjac, ipvt, n, st), fvec, 'lm',
                       nfev, njev, fjac, ipvt, qtf)
    return impl


# ==========================================================================
# scipy.optimize.fsolve
#
# One core, two entry points.  `_core_hybrd` / `_core_hybrj` hold the whole
# algorithm; `fsolve` (python) and its `@overload` (njit) only resolve the
# callback and slice the return.
# ==========================================================================
# argument counts recounted against wrappers_scipy.f90
_hybrd_sp = _sig(_lib.hybrd_sp_wrapper, 20)
_hybrj_sp = _sig(_lib.hybrj_sp_wrapper, 18)

#: scipy's ``infodict`` for the forward-difference path, as a namedtuple.
InfoDict = _result('InfoDict', ['nfev', 'fjac', 'r', 'qtf', 'fvec'])
#: scipy's ``infodict`` for the analytic-jacobian path (adds ``njev``).
InfoDictJ = _result('InfoDictJ',
                    ['nfev', 'njev', 'fjac', 'r', 'qtf', 'fvec'])



# --------------------------------------------------------------------------
# Hiding the @cfunc.  A user passes a plain @njit function; we build the
# pointer-shaped adapter ourselves, once, and cache it.
#
# The cache is not an optimisation: it also OWNS the cfunc.  Drop the reference
# and the baked-in address dangles.
#
# The adapter needs `n`, which is unknown when it is built -- x0's numba TYPE
# carries no length.  So `_prepend_n_args` PREPENDS n to the args buffer
# and the adapter reads it at call time.
# --------------------------------------------------------------------------
_MIXED_CALLBACK_MSG = (
    "fsolve: func is a raw @cfunc .address and fprime is a plain @njit "
    "function. Mixing the two spellings is not served: a raw jacobian "
    "address carries BOTH jobs and func is unused, and a plain @njit "
    "fprime needs a plain @njit func to build the adapter from. Give both "
    "in the same form.")

_ADAPTERS = {}


def _adapter_resid(py, kinds=()):
    """cfunc(minpack_sig) around a plain @njit ``f(x, *args) -> residuals``."""
    key = (py, 'resid', kinds)
    hit = _ADAPTERS.get(key)
    if hit is not None:
        return hit
    unpack, argl = _unpack_lines(kinds, 2, ' ' * 8)
    adapter = _mk_cfunc(_RESID_SRC, {'unpack': unpack, 'argl': argl},
                        njit(py), minpack_sig)
    _ADAPTERS[key] = adapter
    return adapter


def _adapter_jac(pyf, pyj, col_deriv=False, kinds=()):
    """cfunc(minpack_jac_sig) around plain @njit f(x)->resid and jac(x)->(n,n).

    Branches on ``iflag`` and writes ``fjac`` COLUMN-major, so the user never
    sees either detail.

    ``col_deriv`` carries scipy's meaning: it is a statement about the USER's
    jacobian, not about the internal layout. ``False`` reads ``jac(x)`` as the
    ordinary ``J[i, j] = dF_i/dx_j``; ``True`` reads it as already transposed,
    ``J[j, i]``, which is what a scipy call with ``col_deriv=1`` supplies. The
    two build separate cfuncs and are cached separately.
    """
    key = (pyf, pyj, 'jac', bool(col_deriv), kinds)
    hit = _ADAPTERS.get(key)
    if hit is not None:
        return hit
    unpack, argl = _unpack_lines(kinds, 2, ' ' * 8)
    adapter = _mk_cfunc(
        _JAC_SRC, {'unpack': unpack, 'argl': argl,
                   'jj': 'jj[j, i]' if col_deriv else 'jj[i, j]'},
        (njit(pyf), njit(pyj)), minpack_jac_sig)
    _ADAPTERS[key] = adapter
    return adapter


def _adapter_pair(py, col_deriv=False, kinds=()):
    """cfunc(minpack_jac_sig) around one @njit ``f(x) -> (resid, jac)``.

    scipy's ``jac=True`` spelling: one callable returns both, and scipy
    splits them with ``MemoizeJac``. MINPACK asks for the two separately
    through ``iflag``, so the callable runs once per phase rather than once
    per point. Both phases are evaluations of `fun`, and ``nfev`` counts the
    residual ones.

    ``col_deriv`` has the same meaning as in `_adapter_jac`.
    """
    key = (py, 'pair', bool(col_deriv), kinds)
    hit = _ADAPTERS.get(key)
    if hit is not None:
        return hit
    unpack, argl = _unpack_lines(kinds, 2, ' ' * 8)
    adapter = _mk_cfunc(
        _PAIR_SRC, {'unpack': unpack, 'argl': argl,
                    'jj': 'jj[j, i]' if col_deriv else 'jj[i, j]'},
        njit(py), minpack_jac_sig)
    _ADAPTERS[key] = adapter
    return adapter


def _adapter_lsq_pair(py, col_deriv=False, kinds=()):
    """`_adapter_pair` for the least-squares layout: ``m`` residuals, an
    ``(m, n)`` jacobian, leading dimension ``m``."""
    key = (py, 'lsq_pair', bool(col_deriv), kinds)
    hit = _LSQ_ADAPTERS.get(key)
    if hit is not None:
        return hit
    unpack, argl = _unpack_lines(kinds, 3, ' ' * 8)
    adapter = _mk_cfunc(
        _LSQ_PAIR_SRC, {'unpack': unpack, 'argl': argl,
                        'jj': 'jj[j, i]' if col_deriv else 'jj[i, j]'},
        njit(py), minpack_jac_sig)
    _LSQ_ADAPTERS[key] = adapter
    return adapter


@njit
def _prepend_n(args, n):
    """args buffer the ADAPTER path expects: ``[n, nargs, *args]``.

    ``nargs`` is there so an ``f(x, args)`` residual can be handed a
    correctly sized view of what follows.
    """
    out = np.empty(args.size + 2, np.float64)
    out[0] = np.float64(n)
    out[1] = np.float64(args.size)
    for i in range(args.size):
        out[i + 2] = args[i]
    return out


@njit
def _prepend_n_pair(args, n):
    """`_prepend_n` with the memo the ``jac=True`` adapter writes into.

    `_adapter_pair` stores the point and the Jacobian it computed at
    ``iflag = 1`` in the ``1 + n + n*n`` slots behind the caller's
    parameters, and reuses them at ``iflag = 2`` when the point is
    unchanged, which is what scipy's ``MemoizeJac`` does for the same
    spelling. The memo lives in this buffer rather than in the adapter
    because one buffer is allocated per solve, so two solves in a
    ``numba.prange`` loop cannot share it.
    """
    out = np.zeros(args.size + 3 + n + n * n, np.float64)
    out[0] = np.float64(n)
    out[1] = np.float64(args.size)
    for i in range(args.size):
        out[i + 2] = args[i]
    return out


@njit
def _prepend_nm_pair(args, n, m):
    """`_prepend_nm` with the memo `_adapter_lsq_pair` writes into.

    See `_prepend_n_pair`; the Jacobian is ``(m, n)`` on this layout.
    """
    out = np.zeros(args.size + 4 + n + m * n, np.float64)
    out[0] = np.float64(n)
    out[1] = np.float64(m)
    out[2] = np.float64(args.size)
    for i in range(args.size):
        out[i + 3] = args[i]
    return out


# --------------------------------------------------------------------------
# args-length probe.  minpack_sig is void(double*, double*, double*), so the
# length of `args` appears in no argument and in no type: it is unknowable
# from the CALLBACK, not merely from its address (passing the @cfunc object
# instead does not recover it).  scipy learns it from Python, which unpacks
# the tuple into named parameters.  This probe recovers it, turning a silent
# out-of-bounds read into a clean ValueError.  Run under `validate`; see the
# module docstring.
#
# The residual COUNT is checked in the glue instead, by `_check_resid_len`,
# where the callback is still a function and its return length can be read.
# --------------------------------------------------------------------------
_PAD = 8            # slots probed past the end; catches the usual off-by-k

# The read probe needs two values that SURVIVE arithmetic.  _probe's SENTINEL_A
# and SENTINEL_B are denormals, deliberately unlike any real residual -- but
# `1.0 - 1e-320 == 1.0`, so a callback reading them produces identical output
# either way and the probe sees nothing.  Two ordinary, well-separated
# magnitudes instead.  They only ever fill padding the callback must not read,
# so they cannot collide with a legitimate value.
_READ_A = 1.0e6
_READ_B = -7.0e5


@njit
def _reads_beyond(fn_addr, x, n, args):
    """True if the callback reads PAST the end of ``args``.

    Differential padding: the same call is made twice with two different
    fill values past ``args.size``.  Identical inputs inside the buffer, so
    any difference in the residual proves the tail was read.  No false
    positive unless the callback is non-deterministic, which the existing
    ``residual_status`` check already assumes it is not.  False negative if
    the value read does not reach the output.
    """
    k = args.size
    a1 = np.empty(k + _PAD, np.float64)
    a2 = np.empty(k + _PAD, np.float64)
    for i in range(k):
        a1[i] = args[i]
        a2[i] = args[i]
    for i in range(k, k + _PAD):
        a1[i] = _READ_A
        a2[i] = _READ_B
    f1 = np.zeros(n, np.float64)
    f2 = np.zeros(n, np.float64)
    _call_p3(fn_addr, x.ctypes.data, f1.ctypes.data, a1.ctypes.data)
    _call_p3(fn_addr, x.ctypes.data, f2.ctypes.data, a2.ctypes.data)
    for i in range(n):
        if f1[i] != f2[i] and not (np.isnan(f1[i]) and np.isnan(f2[i])):
            return True
    return False


@njit
def _check_bounds(fn_addr, x, n, args):
    """The args-length probe, with the message scipy's own error corresponds
    to."""
    if _reads_beyond(fn_addr, x, n, args):
        raise ValueError(
            "the callback reads past the end of args. scipy raises TypeError "
            "here, because it unpacks the tuple into named parameters; "
            "minpack_sig carries no length for args, so this is detected by "
            "probing. Pass validate=False to skip this check")


# --------------------------------------------------------------------------
# concrete workers: always compute everything, the caller slices
# --------------------------------------------------------------------------
@njit
def _run_hybrd(funcptr, x, args, xtol, maxfev, ml, mu, epsfcn, mode, diag,
               factor):
    """Full hybrd. Returns (x, fvec, fjac, r, qtf, nfev, info)."""
    n = x.size
    lr = (n * (n + 1)) // 2
    n_ = np.array(n, np.int32)
    fvec = np.zeros(n, np.float64)
    fjac = np.zeros((n, n), np.float64)
    r = np.zeros(lr, np.float64)
    qtf = np.zeros(n, np.float64)
    xtol_ = np.array(xtol, np.float64)
    maxfev_ = np.array(maxfev, np.int32)
    ml_ = np.array(ml, np.int32)
    mu_ = np.array(mu, np.int32)
    eps_ = np.array(epsfcn, np.float64)
    mode_ = np.array(mode, np.int32)
    factor_ = np.array(factor, np.float64)
    lr_ = np.array(lr, np.int32)
    info = np.zeros(1, np.int32)
    nfev = np.zeros(1, np.int32)
    nargs = np.array(args.size, np.int32)

    _hybrd_sp(funcptr, n_.ctypes.data, x.ctypes.data, fvec.ctypes.data,
              xtol_.ctypes.data, maxfev_.ctypes.data, ml_.ctypes.data,
              mu_.ctypes.data, eps_.ctypes.data, mode_.ctypes.data,
              diag.ctypes.data, factor_.ctypes.data, info.ctypes.data,
              nfev.ctypes.data, fjac.ctypes.data, r.ctypes.data,
              lr_.ctypes.data, qtf.ctypes.data, args.ctypes.data,
              nargs.ctypes.data)
    return x, fvec, fjac, r, qtf, nfev[0], info[0]


@njit
def _run_hybrj(funcptr, x, args, xtol, maxfev, mode, diag, factor):
    """Full hybrj. Returns (x, fvec, fjac, r, qtf, nfev, njev, info)."""
    n = x.size
    lr = (n * (n + 1)) // 2
    n_ = np.array(n, np.int32)
    fvec = np.zeros(n, np.float64)
    fjac = np.zeros((n, n), np.float64)
    r = np.zeros(lr, np.float64)
    qtf = np.zeros(n, np.float64)
    xtol_ = np.array(xtol, np.float64)
    maxfev_ = np.array(maxfev, np.int32)
    mode_ = np.array(mode, np.int32)
    factor_ = np.array(factor, np.float64)
    lr_ = np.array(lr, np.int32)
    info = np.zeros(1, np.int32)
    nfev = np.zeros(1, np.int32)
    njev = np.zeros(1, np.int32)
    nargs = np.array(args.size, np.int32)

    _hybrj_sp(funcptr, n_.ctypes.data, x.ctypes.data, fvec.ctypes.data,
              fjac.ctypes.data, xtol_.ctypes.data, maxfev_.ctypes.data,
              mode_.ctypes.data, diag.ctypes.data, factor_.ctypes.data,
              info.ctypes.data, nfev.ctypes.data, njev.ctypes.data,
              r.ctypes.data, lr_.ctypes.data, qtf.ctypes.data,
              args.ctypes.data, nargs.ctypes.data)
    return x, fvec, fjac, r, qtf, nfev[0], njev[0], info[0]


# --------------------------------------------------------------------------
# shared setup: resolve scipy's defaults.  Returns (x, args, diag, mode,
# maxfev, ml, mu) so both paths agree on every derived value.
# --------------------------------------------------------------------------
@njit
def _maxfev(n, maxfev, jac_path):
    """scipy's default budget: 200*(n+1), or 100*(n+1) with a jacobian."""
    if maxfev == 0:
        return (100 if jac_path else 200) * (n + 1)
    return maxfev


@njit
def _diag_auto(n, mode):
    """No user diag -> scipy's mode 1.  MINPACK WRITES the scaling it
    derives into this buffer, so one must be passed even in mode 1."""
    return np.ones(n, np.float64), (1 if mode < 0 else mode)


@njit
def _diag_given(diagarr, n, mode):
    """User diag -> scipy's mode 2.  Copied, because MINPACK may write
    to it and the caller's array must not be clobbered.

    ``np.asarray`` accepts a list or a tuple as well as an array from both
    entry points, which is the sequence scipy documents.  Entries past the
    ``n`` MINPACK reads are ignored.  A non-positive entry is not tested
    here: MINPACK's own ``mode == 2`` test reports it as ``ier = 0``, which
    is the status scipy returns.  Only a diag SHORTER than ``n`` is refused,
    because the alternative is a read past the end of the caller's array.
    """
    d = np.asarray(diagarr)
    if d.size < n:
        raise TypeError("diag must have at least one entry per variable")
    return (np.ascontiguousarray(d[:n]).astype(np.float64),
            (2 if mode < 0 else mode))


@njit
def _mesg(ier, maxfev, xtol):
    """scipy's message strings, verbatim, plus the two scijit-only codes.

    ``-1``/``-2`` diverge from scipy deliberately: scipy reports ``ier=1``,
    "The solution converged", for a residual that is zero everywhere.
    """
    if ier == -1:
        return ("The residual is identically zero, so every point is a root "
                "and this result is meaningless.")
    elif ier == -2:
        return ("The residual the solver reported at the solution does not "
                "match a fresh evaluation there, so it did not converge.")
    elif ier == 0:
        return "Improper input parameters were entered."
    elif ier == 1:
        return "The solution converged."
    elif ier == 2:
        return ("The number of calls to function has reached maxfev = "
                + str(maxfev) + ".")
    elif ier == 3:
        return ("xtol=" + _fmt6(xtol) + " is too small, no further "
                "improvement in the approximate\n solution is possible.")
    elif ier == 4:
        return ("The iteration is not making good progress, as measured by "
                "the \n improvement from the last five "
                "Jacobian evaluations.")
    elif ier == 5:
        return ("The iteration is not making good progress, as measured by "
                "the \n improvement from the last ten iterations.")
    return "An error occurred."


# --------------------------------------------------------------------------
# THE cores.  Both entry points below only slice these returns.
# --------------------------------------------------------------------------
@njit
def _core_hybrd(fp, x, a, xtol, mf, ml, mu, epsfcn, m, dd, factor, validate,
                nextra):
    """`fsolve` with a forward-difference Jacobian: the whole algorithm.

    Everything `fsolve` does beyond resolving the callback and slicing
    the return lives here, so its Python body and its ``@overload`` body
    cannot drift apart. That is not hypothetical -- a parallel Python
    body once passed ``func`` where the overload passed ``fprime``, and
    nothing caught it.

    ``nextra`` is the number of evaluations the entry point already made
    before calling in, so one counter covers the whole solve.

    Returns ``(x, fvec, fjac, r, qtf, nfev, ier)``, from which both entry
    points take what their ``full_output`` asks for. `_core_hybrj` is the
    analytic-Jacobian twin.
    """
    n = x.size
    # Every evaluation of the user's residual is counted, this package's own
    # probes included, so `nfev` reports the work the callback actually did.
    # `_check_bounds` evaluates two in `_reads_beyond`; `nextra` carries the
    # entry point's own count.
    nprobe = nextra
    if validate:
        _check_bounds(fp, x, n, a)
        nprobe += 2
    x, fvec, fjac, r, qtf, nfev, ier = _run_hybrd(
        fp, x, a, xtol, mf, ml, mu, epsfcn, m, dd, factor)
    nfev = np.int32(nfev + nprobe)
    # ier = 0 means MINPACK rejected the arguments and never ran, so there is
    # no solution to classify.  Without this gate a bad `factor` or `diag`
    # surfaced as "the residual disagrees with a fresh evaluation", where
    # scipy reports ier = 0, "Improper input parameters were entered."
    st = 0 if ier == 0 else _resid_status(fp, x, fvec, n, a)
    if st != 0:
        if validate:
            raise ValueError(_mesg(-st, mf, xtol))
        ier = np.int32(-st)
    return x, fvec, fjac, r, qtf, nfev, ier


@njit
def _core_hybrj(fp, x, a, xtol, mf, m, dd, factor, validate, nextra):
    """`fsolve` with an analytic Jacobian: the whole algorithm.

    The `_core_hybrd` twin; see there for why the algorithm lives in a
    core rather than in the entry points. The callback ABI is the
    difference: one function serves both phases and ``iflag`` selects
    them. There is no ``band`` or ``epsfcn`` argument either -- both
    belong to the difference approximation this path replaces.

    ``nextra`` is the number of residual evaluations the entry point already
    made before calling in.

    Returns ``(x, fvec, fjac, r, qtf, nfev, njev, ier)``.
    """
    n = x.size
    x, fvec, fjac, r, qtf, nfev, njev, ier = _run_hybrj(
        fp, x, a, xtol, mf, m, dd, factor)
    nfev = np.int32(nfev + nextra)
    st = 0 if ier == 0 else _jac_status(fp, x, fvec, n, n * n, a)
    if st != 0:
        if validate:
            raise ValueError(_mesg(-st, mf, xtol))
        ier = np.int32(-st)
    return x, fvec, fjac, r, qtf, nfev, njev, ier


@njit
def _hybrd_raise_bad(ier):
    """``ier = 0`` raises, exactly where scipy's ``fsolve`` raises.

    MINPACK reports ``ier = 0`` when it rejected the arguments and never
    ran, so ``x`` comes back as the STARTING POINT. Returning that quietly
    is the failure mode this package exists to hunt: a plausible answer with
    nothing to distinguish it from a converged one.

    scipy's gate is reproduced exactly, measured on scipy 1.18: a bare call
    raises ``TypeError('Improper input parameters were entered.')`` and a
    ``full_output=True`` call returns with ``ier = 0`` instead, so a caller
    who asked for the status gets to read it. `_lsq_raise_bad` is the
    `leastsq` twin, whose message text differs by two words because scipy's
    does.
    """
    if ier == 0:
        raise TypeError("Improper input parameters were entered.")


def _emit_hybrd_warning(ier, maxfev, xtol):
    warnings.warn(_mesg.py_func(ier, maxfev, xtol), RuntimeWarning,
                  stacklevel=2)


@njit
def _hybrd_warn_bad(ier, maxfev, xtol):
    """``RuntimeWarning`` on a bare call, exactly where scipy warns.

    scipy's gate on ``full_output=False`` is ``ier`` in ``{2, 3, 4, 5}``:
    `maxfev` exhausted, `xtol` too small, and both not-making-progress
    codes. The message is the one `_mesg` builds, so there is one table
    rather than two.

    ``warnings.warn`` is not typeable by numba; an ``objmode`` block runs
    its body in the interpreter, so ``catch_warnings``, ``simplefilter``,
    ``-W`` and ``PYTHONWARNINGS`` all select it normally. The block takes
    the GIL and is reached only on a non-converged bare call. `ier = 0`
    still raises through `_hybrd_raise_bad`, and the scijit-only ``-1``
    and ``-2`` are left alone.
    """
    if ier == 2 or ier == 3 or ier == 4 or ier == 5:
        with objmode():
            _emit_hybrd_warning(ier, maxfev, xtol)




# --------------------------------------------------------------------------
# the public entry point
# --------------------------------------------------------------------------
def fsolve(func, x0, args=(), fprime=None, full_output=False, col_deriv=0,
           xtol=1.49012e-8, maxfev=0, band=None, epsfcn=None, factor=100.0,
           diag=None, mode=-1, validate=True, keep_shape=False):
    """Find the roots of a system of nonlinear equations.

    Wraps MINPACK's ``hybrd``, a modified Powell hybrid method, or ``hybrj``
    when an analytic Jacobian is supplied.

    Callable from Python and from inside ``@njit``.

    Parameters
    ----------
    func : callable
        A plain ``@njit`` ``f(x) -> residuals`` or ``f(x, args) -> residuals``.
        The arity is detected, so the `args` form is used only when `args` is
        passed. The residual count must equal ``x0.size``.
    x0 : float or array_like
        Initial guess. A scalar, list or tuple is accepted. Any rank is
        FLATTENED, so an ``(n, m)`` guess is one system of ``n*m`` variables
        rather than a batch.
    args : tuple, optional
        Extra parameters, passed to the callback as ``f(x, *args)``, one
        per entry. Entries may be scalars or arrays and need not share a
        type. A non-tuple `args` is coerced to the ONE argument ``(args,)``.
    fprime : callable or None, optional
        ``None`` (default) uses a forward-difference Jacobian. A plain
        ``@njit`` ``jac(x) -> (n, n)`` selects the analytic path, with the
        ``iflag`` branch and MINPACK's column-major layout handled
        internally.
    full_output : bool, optional
        ``False`` (default) returns `x` alone. ``True`` returns the 4-tuple.
        Compile-time constant inside ``@njit``: a literal, or left at its
        default.
    col_deriv : int, optional
        A statement about `fprime`. ``0`` (default) reads ``jac(x)`` as
        ``J[i, j] = dF_i/dx_j``; a nonzero value reads it as already
        transposed. Ignored without `fprime`. A runtime variable is fine.
    xtol : float, optional
        Relative error desired in the solution. Default 1.49012e-8.
    maxfev : int, optional
        Maximum callback evaluations. ``0`` (default) means ``200*(n+1)``, or
        ``100*(n+1)`` with `fprime`.
    band : tuple of int or None, optional
        ``(ml, mu)``, the sub- and super-diagonal counts of a banded Jacobian.
        ``None`` (default) treats it as dense.
    epsfcn : float or None, optional
        Forward-difference step. ``None`` (default) resolves to machine
        epsilon.
    factor : float, optional
        Initial step bound, in ``(0.1, 100)``. Default 100.
    diag : sequence or None, optional
        Per-variable scale factors, all positive. An array, list or tuple.
        The first ``x0.size`` entries are used and any others ignored.
        ``None`` (default) lets MINPACK derive them from the Jacobian column
        norms.
    mode : int, optional
        Scaling mode. ``1`` scales internally, ``2`` uses `diag` unchanged.
        ``-1`` (default) resolves to ``1`` when `diag` is ``None`` and ``2``
        otherwise. A runtime variable is fine: it does not change the return
        type.
    validate : bool, optional
        ``True`` (default) raises when the returned solution is degenerate,
        and probes whether the callback reads past the end of `args`.
        ``False`` reports the degenerate conditions through `ier` instead and
        skips the read probe. The check that the callback wrote ``fvec``, and
        the residual-count and Jacobian-shape checks, are always on. With
        ``full_output=False`` a ``validate=False`` failure is invisible, so
        pair the two.
    keep_shape : bool, optional
        ``False`` (default) returns `x` 1-D. ``True`` reshapes it back to
        `x0`'s rank, which reads better for a field on a grid. Affects `x`
        only. Compile-time constant.

    Returns
    -------
    x : ndarray
        The solution, 1-D unless ``keep_shape=True``.
    infodict : InfoDict or InfoDictJ
        Only with ``full_output=True``. Namedtuple with fields ``nfev``,
        ``fjac``, ``r``, ``qtf`` and ``fvec``, plus ``njev`` after ``nfev`` on
        the `fprime` path, reached as attributes or by name. All but ``nfev``
        are meaningful only when ``ier == 1``.
    ier : int
        MINPACK's status: ``0`` the arguments were rejected and the solver
        never ran, reachable only here since a bare call raises on it;
        ``1`` converged, ``2`` `maxfev` reached, ``3`` `xtol`
        too small, ``4`` and ``5`` not making progress. ``-1`` and ``-2`` are
        this package's degenerate-solution codes, reachable only under
        ``validate=False``: ``-1`` the residual is identically zero, ``-2`` the
        reported residual disagrees with a fresh evaluation at `x`.
    mesg : str
        The message matching `ier`.

    Raises
    ------
    TypeError
        If MINPACK reports ``ier = 0``, meaning it rejected the arguments and
        never ran, so `x` would come back as `x0`. Raised only on a bare
        call: with ``full_output=True`` the status is returned as ``ier = 0``
        instead. Also if the residual count disagrees with ``x0.size``, if
        `fprime` returns something other than an ``(n, n)`` array, or if
        `diag` has fewer entries than `x0`.
    ValueError
        If `func` or `fprime` is not a plain ``@njit`` function; if an `args`
        entry is neither a real number nor an array of real numbers; if the
        callback never wrote ``fvec``; or, under ``validate=True``, if the
        returned solution is degenerate or the callback read past the end of
        `args`.

    See Also
    --------
    scipy.optimize.fsolve : The scipy routine this mirrors.
    scijit.optimize.root : The same solvers behind a ``method`` argument.
    scijit.optimize.leastsq : Least squares rather than a square system.

    Notes
    -----
    `infodict` is a namedtuple rather than a dict. ``info['nfev']``,
    ``info.get('nfev')``, ``info.keys()``, ``info.values()``,
    ``info.items()`` and ``'nfev' in info`` all work, from Python and from
    inside ``@njit``, and its field order is scipy's key insertion order, so
    positional unpacking agrees too. Iterating it yields VALUES, where
    iterating a dict yields keys.

    ``nfev`` counts every evaluation of `func`, including the ones this
    package makes before the solver runs: one to read the residual count off
    the callback, and two more under ``validate=True`` for the read probe.
    scipy counts two of its own, so ``nfev`` is one higher than scipy's on
    the forward-difference path at ``validate=True`` and one lower on the
    other three. The
    evaluation `fprime` costs for its shape check is not counted, and scipy
    does not count its own either.

    Each `args` entry must be a real number or an array of real numbers. A
    mixed tuple is served: ``(matrix, scale)`` reaches the callback as two
    separate parameters and returns ``[3., 3.]`` on the residual
    ``x - matrix.diagonal() * scale``, which is what scipy returns on the
    same call. An entry of any other type raises, and ``args=None`` is one:
    scipy reads it as the one-item tuple ``(None,)`` and calls
    ``f(x, None)``.

    Whether the callback reads past the end of `args` is recovered by probing
    it under ``validate=True``. An overrun larger than 8 slots still escapes
    that probe.

    A `diag` with fewer entries than `x0` raises before the solver runs.
    scipy reads past the end of the array instead, and what MINPACK then does
    depends on the values that happen to follow it.

    `mode`, `validate` and `keep_shape` have no scipy counterpart. Their
    defaults reproduce scipy: `mode` derives from `diag` exactly as scipy
    derives it, and `keep_shape=False` returns the flat `x` scipy returns.

    Safe to call from a ``numba.prange`` loop. MINPACK reaches the callback
    through a Fortran module variable, which carries ``!$omp threadprivate``
    and so resolves to one slot per thread.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import fsolve
    >>> @njit
    ... def f(x):
    ...     return np.array([x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0,
    ...                      0.5 * (x[1] - x[0]) ** 3 + x[1]])
    >>> @njit
    ... def run():
    ...     return fsolve(f, np.array([0.0, 0.0]))
    >>> run()
    array([0.8411639, 0.1588361])

    Parameters reach the callback through `args`, splatted as
    ``func(x, *args)``:

    >>> @njit
    ... def fa(x, c):
    ...     return np.array([x[0] ** 2 - c])
    >>> @njit
    ... def run_args():
    ...     return fsolve(fa, np.array([1.0]), (2.0,))
    >>> run_args()
    array([1.41421356])

    With `full_output`, which must be a literal inside ``@njit``:

    >>> @njit
    ... def run_full():
    ...     return fsolve(f, np.array([0.0, 0.0]), full_output=True)
    >>> x, infodict, ier, mesg = run_full()
    >>> ier, infodict.nfev
    (1, 15)
    >>> mesg
    'The solution converged.'
    """
    at = _as_args_tuple(args)
    kinds = _arg_kinds(at)
    ca = _cb_args(at, kinds)
    pyf = _front_pyfunc(func, len(kinds), 'fsolve', _ADDRESS_MSG_FSOLVE)
    x = _as_x0(x0)
    a = _pack_args(at)
    n = x.size
    if diag is None:
        dd, m = _diag_auto(n, mode)
    else:
        dd, m = _diag_given(np.asarray(diag, dtype=np.float64), n, mode)
    mf = _maxfev(n, maxfev, fprime is not None)
    if band is None:
        ml, mu = -1, -1
    else:
        # scipy's own `ml, mu = band[:2]`, so a band of fewer than two
        # entries raises ValueError rather than IndexError.
        ml, mu = band[:2]
        ml, mu = int(ml), int(mu)
    eps = _EPS if epsfcn is None else float(epsfcn)
    _check_resid_len(_shape_prefix('fsolve', 'func', pyf),
                     np.int64(np.atleast_1d(func(x, *ca)).size), np.int64(n))

    if fprime is None:
        fp, ab = (_adapter_resid(pyf, kinds).address, _prepend_n(a, n))
        x, fvec, fjac, r, qtf, nfev, ier = _core_hybrd(
            fp, x, ab, xtol, mf, ml, mu, eps, m, dd, factor, validate, 1)
        xo = _finish(x, x0, keep_shape)
        if not full_output:
            _hybrd_raise_bad(ier)
            _hybrd_warn_bad(ier, mf, xtol)
            return xo
        return (xo, InfoDict(nfev, fjac, r, qtf, fvec), int(ier),
                _mesg(ier, mf, xtol))

    pyj = _front_pyfunc(fprime, len(kinds), 'fsolve', _ADDRESS_MSG_FSOLVE)
    _jj = np.atleast_1d(fprime(x, *ca))
    _pfx = _shape_prefix('fsolve', 'fprime', pyj)
    if _jj.ndim != 2:
        raise TypeError("%s(%d, %d) but it is %s." % (_pfx, n, n, _jj.shape))
    _check_jac_shape(_pfx, np.int64(_jj.shape[0]), np.int64(_jj.shape[1]),
                     np.int64(n), np.int64(n))
    fp = _adapter_jac(pyf, pyj, bool(col_deriv), kinds).address
    ab = _prepend_n(a, n)
    x, fvec, fjac, r, qtf, nfev, njev, ier = _core_hybrj(
        fp, x, ab, xtol, mf, m, dd, factor, validate, 1)
    xo = _finish(x, x0, keep_shape)
    if not full_output:
        _hybrd_raise_bad(ier)
        _hybrd_warn_bad(ier, mf, xtol)
        return xo
    return (xo, InfoDictJ(nfev, njev, fjac, r, qtf, fvec), int(ier),
            _mesg(ier, mf, xtol))




def _finish(xflat, x0, keep_shape):
    """scipy's flat solution, or reshaped back to ``x0``'s rank."""
    if keep_shape and np.ndim(x0) > 1:
        return xflat.reshape(np.shape(x0))
    return xflat


@overload(_finish, prefer_literal=True)
def _finish_ovl(xflat, x0, keep_shape):
    """@njit implementation of `_finish`.

    The reshape has to be decided while the body is typed: an array's
    rank is part of its numba type, so one function cannot return a
    ``(2, 3)`` array on one call and a flat one on the next. Both
    ``keep_shape`` and ``x0.ndim`` are known here, so the branch
    disappears and the common case compiles to a passthrough.
    """
    ks = _lit_bool(keep_shape)
    if ks and isinstance(x0, types.Array) and x0.ndim > 1:
        def impl(xflat, x0, keep_shape):
            return xflat.reshape(x0.shape)
        return impl

    def impl(xflat, x0, keep_shape):
        return xflat
    return impl


@overload(fsolve, prefer_literal=True)
def _fsolve_ovl(func, x0, args=(), fprime=None, full_output=False,
                col_deriv=0, xtol=1.49012e-8, maxfev=0, band=None,
                epsfcn=None, factor=100.0, diag=None, mode=-1,
                validate=True, keep_shape=False):
    """@njit implementation of `fsolve`, resolved at compile time.

    The callback is the reason this exists. ``func`` is a plain ``@njit``
    residual, from which a pointer-shaped adapter is built here and its
    address baked into the body -- an address is a compile-time constant, so
    it cannot be chosen while the code runs. ``full_output`` picks the return
    type, ``fprime`` picks between `_core_hybrd` and `_core_hybrj`, and
    ``keep_shape`` reaches `_finish`, whose return rank is also a type.
    Returning ``None`` declines the call, which numba reports as a
    TypingError naming the argument that could not be served.
    """
    fo = _lit_bool(full_output)
    if fo is None:
        return None                   # -> TypingError, as designed
    kinds = _arg_kinds_ty(_args_types(args))
    no_jac = _is_none(fprime)
    no_diag = _is_none(diag)
    no_band = _is_none(band)
    no_eps = _is_none(epsfcn)
    cd_runtime, addr_t, addr_f = False, 0, 0

    # Resolve the callback AT COMPILE TIME: the adapter is built here and its
    # address baked in, so the callback never becomes user-facing.  An
    # address is refused rather than reinterpreted.
    pyf = _front_pyfunc_ty(func, len(kinds), 'fsolve', _ADDRESS_MSG_FSOLVE)
    fpfx = _shape_prefix('fsolve', 'func', pyf)
    jpfx = ''
    if no_jac:
        addr = _adapter_resid(pyf, kinds).address
    else:
        pyj = _front_pyfunc_ty(fprime, len(kinds), 'fsolve',
                               _ADDRESS_MSG_FSOLVE)
        jpfx = _shape_prefix('fsolve', 'fprime', pyj)
        # scipy reads `col_deriv` only on the analytic-jacobian path, and
        # it selects a different cfunc rather than a runtime branch: an
        # address is a compile-time constant.  A literal or omitted flag
        # builds one adapter; a runtime variable builds both and picks
        # between two baked addresses, so no call is refused.
        cd = _lit_bool(col_deriv)
        addr = _adapter_jac(pyf, pyj, bool(cd), kinds).address
        addr_t = addr if cd is not None else _adapter_jac(
            pyf, pyj, True, kinds).address
        addr_f = addr if cd is not None else _adapter_jac(
            pyf, pyj, False, kinds).address
        cd_runtime = cd is None

    if no_jac:
        def impl(func, x0, args=(), fprime=None, full_output=False,
                 col_deriv=0, xtol=1.49012e-8, maxfev=0, band=None,
                 epsfcn=None, factor=100.0, diag=None, mode=-1,
                 validate=True, keep_shape=False):
            x = _as_x0(x0)
            a = _pack_args(args)
            n = x.size
            fp, ab = addr, _prepend_n(a, n)
            if no_diag:
                dd, m = _diag_auto(n, mode)
            else:
                dd, m = _diag_given(diag, n, mode)
            if no_band:
                ml, mu = -1, -1
            else:
                ml, mu = np.int64(band[0]), np.int64(band[1])
            mf = _maxfev(n, maxfev, False)
            if no_eps:
                eps = _EPS
            else:
                eps = np.float64(epsfcn)
            _check_resid_len(fpfx, np.int64(_call_cb(func, x, args).size),
                             np.int64(n))
            x, fvec, fjac, r, qtf, nfev, ier = _core_hybrd(
                fp, x, ab, xtol, mf, ml, mu, eps, m, dd, factor, validate, 1)
            ier = np.int64(ier)   # python's int, not MINPACK's int32
            xo = _finish(x, x0, keep_shape)
            if fo:
                return (xo, InfoDict(nfev, fjac, r, qtf, fvec), ier,
                        _mesg(ier, mf, xtol))
            _hybrd_raise_bad(ier)
            _hybrd_warn_bad(ier, mf, xtol)
            return xo
        return impl

    def impl(func, x0, args=(), fprime=None, full_output=False,
             col_deriv=0, xtol=1.49012e-8, maxfev=0, band=None,
             epsfcn=None, factor=100.0, diag=None, mode=-1, validate=True,
             keep_shape=False):
        x = _as_x0(x0)
        a = _pack_args(args)
        n = x.size
        if cd_runtime:
            fp = addr_t if col_deriv else addr_f
        else:
            fp = addr
        ab = _prepend_n(a, n)
        if no_diag:
            dd, m = _diag_auto(n, mode)
        else:
            dd, m = _diag_given(diag, n, mode)
        mf = _maxfev(n, maxfev, True)
        _check_resid_len(fpfx, np.int64(_call_cb(func, x, args).size),
                         np.int64(n))
        jj0 = _call_cb(fprime, x, args)
        _check_jac_shape(jpfx, np.int64(jj0.shape[0]), np.int64(jj0.shape[1]),
                         np.int64(n), np.int64(n))
        x, fvec, fjac, r, qtf, nfev, njev, ier = _core_hybrj(
            fp, x, ab, xtol, mf, m, dd, factor, validate, 1)
        ier = np.int64(ier)   # python's int, not MINPACK's int32
        xo = _finish(x, x0, keep_shape)
        if fo:
            return (xo, InfoDictJ(nfev, njev, fjac, r, qtf, fvec), ier,
                    _mesg(ier, mf, xtol))
        _hybrd_raise_bad(ier)
        _hybrd_warn_bad(ier, mf, xtol)
        return xo
    return impl


