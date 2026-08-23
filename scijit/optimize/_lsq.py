"""Linear and nonlinear least-squares solvers callable from ``@njit``.

Three public routines: ``nnls`` (non-negative least squares by
Lawson-Hanson), ``lsq_linear`` (bound-constrained linear least squares by
trust-region-reflective or BVLS), and ``curve_fit`` (nonlinear model fit
with a covariance estimate). All three run from Python and from inside
``@njit``.

Notes
-----
``nnls`` and ``lsq_linear`` take no callback and hold no state, so both are
safe to call from a ``numba.prange`` loop. ``lsq_linear``'s
``method='trf'`` reaches LAPACK through ``scijit.linalg.qr_pivot``, which is
prange-safe as well.

``curve_fit`` runs on the MINPACK ``leastsq`` solver, whose callback slot is
a module variable carrying ``!$omp threadprivate``, one slot per thread, so
it is prange-safe too.
"""
import warnings
from collections import namedtuple

import numpy as np
from numba import carray, cfunc, njit, objmode, types
from numba.core.errors import TypingError
from numba.extending import intrinsic, overload
import llvmlite.ir as ir

from ._lapack import qr_pivot as _qr_pivot
from ._lapack import solve_triangular as _solve_triangular
from ._lbfgsb import _is_none                # noqa: E402
from ._minpack import (leastsq, minpack_sig,  # noqa: F401
                       LsqInfo, _lit_bool, _leastsq_ptr, _address_msg,
                       _check_arity, _opt_result)


@intrinsic
def _call_resid(typingctx, fn_addr, xp, fp, ap):
    """Call a ``@cfunc(minpack_sig)`` through its raw address.

    ``curve_fit`` rebuilds the Jacobian by finite differences at the
    solution, which means evaluating the user residual from inside
    ``@njit`` -- and a ``@cfunc`` address is an integer there, not
    something numba will call. The generated code casts the address to
    ``void (*)(double*, double*, double*)`` and passes the three buffers
    as ``intp`` data pointers, in MINPACK's order: ``x`` in, ``fvec``
    out, ``args`` in.
    """
    signature = types.void(types.intp, types.intp, types.intp, types.intp)

    def codegen(context, builder, sg, args):
        fnaddr, p0, p1, p2 = args
        dp = ir.DoubleType().as_pointer()
        fnty = ir.FunctionType(ir.VoidType(), [dp, dp, dp])
        fptr = builder.inttoptr(fnaddr, fnty.as_pointer())
        builder.call(fptr, [builder.inttoptr(p0, dp),
                            builder.inttoptr(p1, dp),
                            builder.inttoptr(p2, dp)])
        return context.get_dummy_value()

    return signature, codegen


@njit
def _resid_eval(resid_ptr, p, args, m):
    """Evaluate the user residual fvec (length m) at parameters p."""
    fvec = np.zeros(m, np.float64)
    p_ = np.ascontiguousarray(np.asarray(p, np.float64))
    a_ = np.ascontiguousarray(np.asarray(args, np.float64))
    _call_resid(resid_ptr, p_.ctypes.data, fvec.ctypes.data, a_.ctypes.data)
    return fvec


# --- shared: least squares on a column subset of A -----------------------
@njit
def _lstsq_cols(A, cols, rhs):
    """min || A[:, cols] z - rhs ||_2 ; returns z (length cols.size)."""
    m = A.shape[0]
    k = cols.size
    Asub = np.empty((m, k), np.float64)
    for jj in range(k):
        c = cols[jj]
        for i in range(m):
            Asub[i, jj] = A[i, c]
    return np.linalg.lstsq(Asub, rhs)[0]


#: `maxiter=None`. Out of reach of any budget a caller could mean, so a
#: negative `maxiter` stays a real budget, as it is in scipy.
_NNLS_DEFAULT = -(1 << 62)

#: scipy's texts, `_nnls.py:67-77`, up to the shape each renders.
_NNLS_MSG_A2D = "Expected a 2D array, but the shape of A is "
_NNLS_MSG_B1D = ("Expected a 1D array,(or 2D with one column), but the, shape "
                 "of b is ")


@njit
def _shape_str(a):
    """``repr`` of an array's shape, python's own rendering of a tuple."""
    if a.ndim == 1:
        return "(" + str(a.shape[0]) + ",)"
    s = "("
    for i in range(a.ndim):
        s = s + str(a.shape[i])
        if i + 1 < a.ndim:
            s = s + ", "
    return s + ")"


@njit
def _chk_finite_any(a):
    """``numpy.asarray_chkfinite``'s test, at any rank."""
    f = a.ravel()
    for i in range(f.size):
        if not np.isfinite(f[i]):
            raise ValueError(_CF_MSG_FINITE)


@njit
def _nnls_bad_a(A):
    """scipy's non-2-D `A` message, carrying the real shape."""
    raise ValueError(_NNLS_MSG_A2D + _shape_str(A))


@njit
def _nnls_bad_b(b):
    """scipy's `b` message, carrying the real shape."""
    raise ValueError(_NNLS_MSG_B1D + _shape_str(b))


@njit
def _nnls_core(A, b, maxiter, validate):
    """Lawson-Hanson active set, the compiled core both `nnls` entries reach.

    `nnls` is the Python entry: it applies scipy's ravel rule for a 2-D
    single-column `b`, turns ``maxiter=None`` into the sentinel ``-1``, and
    calls this. Inside ``@njit`` an overload calls this directly. Keeping the
    loop in one function is what makes the two entries return the same values
    rather than two implementations that agree today.

    Returns ``(x, rnorm)`` with `rnorm` the residual 2-norm, neither squared
    nor halved. `_bvls` in this module returns a HALVED SQUARED cost instead,
    because that is what scipy's `lsq_linear` result carries; the two are not
    interchangeable.

    `maxiter` below zero is the sentinel for scipy's ``None``, which means
    ``3 * n``. `validate` has no scipy counterpart: scipy raises on
    non-convergence unconditionally, so ``True`` reproduces scipy and
    ``False`` returns the current iterate. The raise names the flag.

    The active-set tolerance is ``10 * eps * ||A||_1 * max(m, n)``, scipy's
    suggested relaxation. Measured against ``scipy.optimize.nnls`` over 15
    random 30x8 problems: worst 3.6e-16 in `x` and 8.9e-16 in `rnorm`.

    An ``m == 0`` problem leaves the loop at ``x = 0`` with ``rnorm = 0``.
    That is the correct answer rather than a special case: with no equations
    the residual vector has length zero, so every non-negative `x` minimises
    it, and the zero vector is the minimum-norm choice among them. scipy 1.18
    returns uninitialised memory there.
    """
    A = np.ascontiguousarray(np.asarray(A, np.float64))
    b = np.ascontiguousarray(np.asarray(b, np.float64))
    # the order is scipy's, `_nnls.py:64-81`: both arrays are read for
    # finiteness before either shape is looked at
    _chk_finite_any(A)
    _chk_finite_any(b)
    if A.ndim != 2:
        _nnls_bad_a(A)
    if b.ndim != 1:
        _nnls_bad_b(b)
    m, n = A.shape
    if m != b.size:
        raise ValueError("Incompatible dimensions. The first dimension of A "
                         "is " + str(m) + ", while the shape of b is ("
                         + str(b.size) + ",)")

    # scipy's rule, `_nnls.py:85`: `if not maxiter`, so 0 and None both mean
    # the default and a negative value is passed through as a real budget.
    if maxiter == _NNLS_DEFAULT or maxiter == 0:
        maxiter = 3 * n
    # tolerance on the projected residual A^T(b - Ax), scipy's suggested
    # relaxation: max(m,n) * ||A||_1 * eps
    eps = np.finfo(np.float64).eps
    a1 = 0.0
    for j in range(n):
        s = 0.0
        for i in range(m):
            s += abs(A[i, j])
        if s > a1:
            a1 = s
    tol = 10.0 * eps * a1 * (m if m > n else n)

    x = np.zeros(n, np.float64)
    P = np.zeros(n, np.bool_)                 # passive (in-solution) set
    w = A.T @ (b - A @ x)
    it = 0
    while True:
        # most-positive gradient among the currently-zero variables
        wmax = -1.0e300
        t = -1
        for j in range(n):
            if (not P[j]) and w[j] > wmax:
                wmax = w[j]
                t = j
        if t == -1 or wmax <= tol:
            break
        P[t] = True
        while True:                            # inner (feasibility) loop
            it += 1
            if it > maxiter:
                break
            idx = np.where(P)[0]
            k = idx.size
            zsub = _lstsq_cols(A, idx, b)
            z = np.zeros(n, np.float64)
            zmin = 1.0e300
            for jj in range(k):
                z[idx[jj]] = zsub[jj]
                if zsub[jj] < zmin:
                    zmin = zsub[jj]
            if zmin > 0.0:
                x = z
                break
            # ratio test, largest feasible step toward z
            alpha = 1.0e300
            for jj in range(k):
                j = idx[jj]
                if z[j] <= 0.0:
                    val = x[j] / (x[j] - z[j])
                    if val < alpha:
                        alpha = val
            for j in range(n):
                x[j] = x[j] + alpha * (z[j] - x[j])
            for j in range(n):
                if P[j] and x[j] <= 0.0:
                    P[j] = False
                    x[j] = 0.0
        if it > maxiter:
            # scipy raises here unconditionally and has no flag for it.
            # `validate=False` is this package's escape, named in the
            # message, for a sweep that cannot afford an exception.
            if validate:
                raise RuntimeError("Maximum number of iterations reached.")
            break
        w = A.T @ (b - A @ x)

    r = A @ x - b
    rnorm = np.sqrt(r @ r)
    return x, rnorm


@njit
def _kkt_optimality(g, on_bound):
    """Max KKT violation: |g| on free vars, g*on_bound on bounded vars.

    A TRUE maximum, so it can be negative.  scipy's
    ``compute_kkt_optimality`` returns ``np.max(g_kkt)`` with no floor, and
    when every variable sits on a bound at the solution every entry of
    ``g * on_bound`` is negative -- measured -1.20982 on a 30x6 problem
    where a 0.0 floor reported 0.  Flooring at zero left the termination
    test unchanged (both values pass ``< tol``) and only misreported the
    field.
    """
    n = g.size
    mx = -np.inf
    for j in range(n):
        if on_bound[j] == 0.0:
            v = abs(g[j])
        else:
            v = g[j] * on_bound[j]
        if v > mx:
            mx = v
    return mx


def nnls(A, b, *, maxiter=None, validate=True):
    """Non-negative least squares.

    ``min ||A x - b||_2`` subject to ``x >= 0``, by the classical
    Lawson-Hanson active-set method.  Callable from Python and from inside
    ``@njit``; both entries run the same compiled core.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Coefficient matrix. Cast to float64 and copied to a contiguous
        buffer, so strided views are safe. Must be 2-D.
    b : array_like, shape (m,)
        Right-hand side, with ``b.size == A.shape[0]``. A 2-D `b` of one
        column is ravelled.
    maxiter : int or None, optional
        Cap on the inner feasibility iterations. ``None`` and ``0`` both
        mean ``3 * n``; a negative value is a real budget and is exhausted at
        once. Keyword-only. See Notes for what the cap counts.
    validate : bool, optional
        ``True`` (default) raises ``RuntimeError`` when `maxiter` is reached.
        ``False`` returns the current iterate instead.

    Returns
    -------
    x : ndarray, shape (n,)
        Non-negative solution vector.
    rnorm : float
        Residual 2-norm ``||A x - b||_2``, not squared and not halved.

    Raises
    ------
    ValueError
        If `A` or `b` holds a non-finite entry; if `A` is not 2-D; if `b` has
        rank above 2, or rank 2 with more than one column; or if
        ``b.size != A.shape[0]``. The finiteness of both arrays is read
        before either shape is.
    RuntimeError
        If `maxiter` is reached, unless ``validate=False``.

    See Also
    --------
    scipy.optimize.nnls : The scipy routine this mirrors.
    scijit.optimize.lsq_linear : General lower and upper bounds, not just
        ``x >= 0``.
    scijit.optimize.leastsq : Nonlinear least squares.

    Notes
    -----
    `maxiter` caps the INNER feasibility iterations of the Lawson-Hanson
    loop, where scipy's caps the outer ones, so a small explicit `maxiter`
    can stop the two at different points, converging here while scipy raises
    ``RuntimeError``. The default and any cap large enough to converge agree.

    Inside ``@njit``, `maxiter` and `validate` may also be passed
    positionally: numba's dispatcher does not enforce keyword-only. The Python
    entry enforces it.

    `validate` has no counterpart in scipy's signature. scipy raises on
    non-convergence unconditionally, so the default reproduces scipy.
    ``validate=False`` returns the current iterate instead, for a run over
    many points where an exception would abort the whole loop.

    An `A` with zero rows returns ``(zeros(n), 0.0)``, the same values on every
    call. With no equations the residual vector has length zero, so
    ``||A x - b||`` is 0 for any `x`; every non-negative `x` is a minimiser and
    the zero vector is the minimum-norm one among them. scipy 1.18 reads
    uninitialised memory on that input: consecutive calls to
    ``scipy.optimize.nnls(np.zeros((0, 3)), np.zeros(0))`` return a different
    `x` each time.

    Pure ``@njit``, no callback, no module state and no library handle, so it
    is safe to call from a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import nnls
    >>> A = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> b = np.array([2.0, 1.0, -1.0])
    >>> @njit
    ... def run():
    ...     return nnls(A, b)
    >>> x, rnorm = run()
    >>> x
    array([1.5, 0. ])
    >>> round(rnorm, 10)
    1.2247448714
    """
    # scipy's own two lines, `_nnls.py:64-65`: both arrays are read for
    # finiteness before either shape is looked at.
    A = np.asarray_chkfinite(A, dtype=np.float64, order='C')
    b = np.asarray_chkfinite(b, dtype=np.float64)
    # checked HERE, not only in the core: a rank other than 2 fails to type
    # inside @njit at the `m, n = A.shape` unpack, and would surface as a
    # TypingError instead of scipy's ValueError.
    if A.ndim != 2:
        raise ValueError(_NNLS_MSG_A2D + str(A.shape))
    # scipy's own rule, `_nnls.py:70-74`: rank > 2, or rank 2 with more than
    # one column, is an error; a single column is ravelled.
    if b.ndim > 2 or (b.ndim == 2 and b.shape[1] != 1):
        raise ValueError(_NNLS_MSG_B1D + str(b.shape))
    if b.ndim == 2:
        b = b.ravel()
    x, rnorm = _nnls_core(
        A, b, _NNLS_DEFAULT if maxiter is None else int(maxiter), validate)
    return x, rnorm


@overload(nnls)
def _nnls_ovl(A, b, maxiter=None, validate=True):
    """@njit implementation of `nnls`.

    ``maxiter=None`` means scipy's ``3 * n``, and ``None`` cannot be
    compared against an integer inside ``@njit``, so its presence is
    settled here and the body sees the sentinel ``-1`` instead.

    A rank is part of a numba array type, so a shape scipy refuses is known
    while the call compiles. The refusal is still made by a compiled body,
    which reads both arrays for finiteness first and then raises scipy's own
    ``ValueError``, so the two entry points refuse in the same order with
    the same class. A rank-2 `b` reaches the same ravel the Python entry
    applies, its column count being the one part that is not in the type.
    """
    mi_none = _is_none(maxiter)
    bad_a = isinstance(A, types.Array) and A.ndim != 2
    b2d = isinstance(b, types.Array) and b.ndim == 2
    bad_b = isinstance(b, types.Array) and b.ndim > 2

    if bad_a:
        def impl(A, b, maxiter=None, validate=True):
            _chk_finite_any(A)
            _chk_finite_any(b)
            _nnls_bad_a(A)
        return impl
    if bad_b:
        def impl(A, b, maxiter=None, validate=True):
            _chk_finite_any(A)
            _chk_finite_any(b)
            _nnls_bad_b(b)
        return impl
    if b2d:
        def impl(A, b, maxiter=None, validate=True):
            if mi_none:
                mi = _NNLS_DEFAULT
            else:
                mi = np.int64(maxiter)
            _chk_finite_any(A)
            _chk_finite_any(b)
            if b.shape[1] != 1:
                _nnls_bad_b(b)
            return _nnls_core(A, np.ascontiguousarray(b).ravel(), mi,
                              validate)
        return impl

    def impl(A, b, maxiter=None, validate=True):
        if mi_none:
            mi = _NNLS_DEFAULT
        else:
            mi = np.int64(maxiter)
        return _nnls_core(A, b, mi, validate)
    return impl


# --------------------------------------------------------------------------
# ``lsq_linear(verbose=...)``: the progress table and the summary lines.
#
# The text is built by the four Python functions below, which are
# `scipy/optimize/_lsq/common.py`'s ``print_header_linear`` and
# ``print_iteration_linear`` and the two ``print`` pairs in
# `_lsq/lsq_linear.py`.  Compiled code reaches them through the ``@njit``
# wrappers, each holding a ``numba.objmode`` block in its own module-level
# function because lowering such a block pickles the function that encloses
# it.  ``scijit.integrate.romb`` prints its Richardson table the same way.
#
# A row's cost-reduction and step-norm columns are 15 spaces on the first
# iteration, where scipy has ``None``; a compiled loop carries a flag beside
# each value instead.
# --------------------------------------------------------------------------


def _lsql_py_header():
    """The five column titles."""
    print("{:^15}{:^15}{:^15}{:^15}{:^15}"
          .format("Iteration", "Cost", "Cost reduction", "Step norm",
                  "Optimality"))


def _lsql_py_row(iteration, cost, cost_reduction, step_norm, optimality,
                 has_cr, has_sn):
    """One table row.  `has_cr` and `has_sn` blank their column when False."""
    cr = f"{cost_reduction:^15.2e}" if has_cr else " " * 15
    sn = f"{step_norm:^15.2e}" if has_sn else " " * 15
    print(f"{iteration:^15}{cost:^15.4e}{cr}{sn}{optimality:^15.2e}")


def _lsql_py_summary(message, nit, initial_cost, cost, optimality):
    """The two lines after a solver loop."""
    print(message)
    print(f"Number of iterations {nit}, initial cost {initial_cost:.4e}, "
          f"final cost {cost:.4e}, first-order optimality {optimality:.2e}.")


def _lsql_py_short(message, cost, optimality):
    """The two lines the unconstrained short circuit prints instead."""
    print(message)
    print(f"Final cost {cost:.4e}, first-order optimality {optimality:.2e}")


@njit
def _lsql_show_header():
    with objmode():
        _lsql_py_header()


@njit
def _lsql_show_row(iteration, cost, cost_reduction, step_norm, optimality,
                   has_cr, has_sn):
    with objmode():
        _lsql_py_row(iteration, cost, cost_reduction, step_norm, optimality,
                     has_cr, has_sn)


@njit
def _lsql_show_summary(message, nit, initial_cost, cost, optimality):
    with objmode():
        _lsql_py_summary(message, nit, initial_cost, cost, optimality)


@njit
def _lsql_show_short(message, cost, optimality):
    with objmode():
        _lsql_py_short(message, cost, optimality)


@njit
def _bvls(A, b, lb, ub, tol, max_iter, verbose):
    """Stark and Parker BVLS active set, the compiled core `lsq_linear` wraps.

    Takes ``(A, b, lb, ub, tol, max_iter, verbose)`` and returns the
    solution, the cost and the optimality measure. `lsq_linear` adds the
    residual vector, the active mask and the unconstrained solve that scipy's
    result carries, and validates its arguments before it gets here; the
    checks below are for a direct call.

    `verbose` at 2 prints the header and one row per iteration, at the point
    in the loop where scipy prints them. The two summary lines belong to
    `lsq_linear`, which is where scipy prints those.

    Returns ``(x, cost, optimality, nit, term, initial_cost)``. `cost` is
    ``0.5 * ||A x - b||^2``, the HALVED SQUARED residual scipy's
    `lsq_linear` reports, not the plain 2-norm `_nnls_core` returns.
    `initial_cost` is the cost at the projected starting point, which
    `lsq_linear` prints and scipy deletes from its result.

    `term` is NOT scipy's `status`. It is 3 for the short-circuit below, 1
    for optimality under `tol`, 2 for a cost change under ``tol * cost``, and
    ``-1`` when the loop runs out of iterations, where scipy's code for that
    is 0. `_lsql_pack` does the ``-1 -> 0`` translation, so anything calling
    `_bvls` directly gets the raw code.

    Every argument is required, and an unconstrained problem passes
    explicit ``-np.inf`` and ``np.inf`` ARRAYS. This is not a numba limit:
    a tuple default and a ``None`` default both work in an ``@njit``
    function (measured 2026-08-02), and ``curve_fit`` in this same module
    already carries ``bounds=(-np.inf, np.inf)``. `_bvls` is internal and
    takes per-element bounds, which a 2-tuple cannot express. `max_iter` below zero is the sentinel for scipy's
    ``None``, which means ``n``; as in scipy the initialization iterations
    are added on top of that budget rather than counted against it. `tol`
    serves both tests scipy uses it for, first-order optimality ``< tol`` and
    relative cost decrease ``< tol * cost``.

    A feasible unconstrained least-squares solution short-circuits the
    active-set loop, exactly as in scipy. Measured against
    ``scipy.optimize.lsq_linear(method='bvls')`` over 15 random bounded 30x6
    problems: exactly 0.0 in `x`. `_trf_linear` is the other method
    `lsq_linear` dispatches to, and it stops on a different criterion.

    An empty problem raises rather than short-circuiting. scipy reaches an
    illegal LAPACK call on the same input.
    """
    A = np.ascontiguousarray(np.asarray(A, np.float64))
    b = np.ascontiguousarray(np.asarray(b, np.float64))
    lb = np.ascontiguousarray(np.asarray(lb, np.float64))
    ub = np.ascontiguousarray(np.asarray(ub, np.float64))
    if A.ndim != 2:
        raise ValueError("lsq_linear: A must be 2-D")
    m, n = A.shape
    if b.size != m:
        raise ValueError("lsq_linear: A.shape[0] must equal b.size")
    if lb.size != n or ub.size != n:
        raise ValueError("lsq_linear: lb/ub length must equal A.shape[1]")
    if m == 0 or n == 0:
        raise ValueError("lsq_linear: empty problem")
    for j in range(n):
        if lb[j] >= ub[j]:
            raise ValueError(_LSQL_MSG_BOUNDS)

    # unconstrained least-squares solution; return it if already feasible
    x_lsq = np.linalg.lstsq(A, b)[0]
    inb = True
    for j in range(n):
        if x_lsq[j] < lb[j] or x_lsq[j] > ub[j]:
            inb = False
            break
    if inb:
        r = A @ x_lsq - b
        cost = 0.5 * (r @ r)
        g = A.T @ r
        opt = 0.0
        for j in range(n):
            if abs(g[j]) > opt:
                opt = abs(g[j])
        return x_lsq, cost, opt, 0, 3, cost

    # project the infeasible start onto the box, marking active bounds
    x = x_lsq.copy()
    on_bound = np.zeros(n, np.float64)          # -1 lower, +1 upper, 0 free
    for j in range(n):
        if x[j] <= lb[j]:
            x[j] = lb[j]
            on_bound[j] = -1.0
        elif x[j] >= ub[j]:
            x[j] = ub[j]
            on_bound[j] = 1.0

    r = A @ x - b
    cost = 0.5 * (r @ r)
    initial_cost = cost
    g = A.T @ r
    iteration = 0
    cost_change = 0.0
    step_norm = 0.0
    has_prev = False

    if verbose == 2:
        _lsql_show_header()

    # initialization loop: reach a free-set least-squares-feasible state
    while True:
        free_idx = np.where(on_bound == 0.0)[0]
        if free_idx.size == 0:
            break
        if verbose == 2:
            _lsql_show_row(iteration, cost, cost_change, step_norm,
                           _kkt_optimality(g, on_bound), has_prev, has_prev)
        iteration += 1
        active = (on_bound != 0.0).astype(np.float64)
        b_free = b - A @ (x * active)
        x_free_old = x[free_idx].copy()
        z = _lstsq_cols(A, free_idx, b_free)
        anyv = False
        vflag = np.zeros(free_idx.size, np.bool_)
        for jj in range(free_idx.size):
            j = free_idx[jj]
            if z[jj] < lb[j]:
                x[j] = lb[j]
                on_bound[j] = -1.0
                vflag[jj] = True
                anyv = True
            elif z[jj] > ub[j]:
                x[j] = ub[j]
                on_bound[j] = 1.0
                vflag[jj] = True
                anyv = True
        for jj in range(free_idx.size):
            if not vflag[jj]:
                x[free_idx[jj]] = z[jj]
        r = A @ x - b
        cost_new = 0.5 * (r @ r)
        cost_change = cost - cost_new
        cost = cost_new
        g = A.T @ r
        step_norm = np.linalg.norm(x[free_idx] - x_free_old)
        has_prev = True
        if not anyv:
            break

    if max_iter < 0:
        max_iter = n
    max_iter += iteration

    term = -1
    opt = _kkt_optimality(g, on_bound)
    it = iteration
    while it < max_iter:                        # main BVLS loop A
        if verbose == 2:
            _lsql_show_row(it, cost, cost_change, step_norm, opt,
                           has_prev, has_prev)
        if opt < tol:
            term = 1
        if term != -1:
            break
        # free the bounded variable with the largest KKT violation
        mv = -1
        mval = -1.0e300
        for j in range(n):
            val = g[j] * on_bound[j]
            if val > mval:
                mval = val
                mv = j
        on_bound[mv] = 0.0

        while True:                             # loop B
            free_idx = np.where(on_bound == 0.0)[0]
            k = free_idx.size
            x_free_old = x[free_idx].copy()
            if k == 0:
                # scipy runs this pass as well, on an empty free set: the
                # inner solve is over no columns, nothing moves, and the
                # step norm is that of a zero-length vector.
                step_norm = np.linalg.norm(x[free_idx] - x_free_old)
                has_prev = True
                break
            active = (on_bound != 0.0).astype(np.float64)
            b_free = b - A @ (x * active)
            z = _lstsq_cols(A, free_idx, b_free)
            best = 1.0e300
            bi = -1
            bkind = 0
            hasv = False
            for jj in range(k):
                j = free_idx[jj]
                if z[jj] < lb[j]:
                    hasv = True
                    al = (lb[j] - x[j]) / (z[jj] - x[j])
                    if al < best:
                        best = al
                        bi = jj
                        bkind = -1
                elif z[jj] > ub[j]:
                    hasv = True
                    al = (ub[j] - x[j]) / (z[jj] - x[j])
                    if al < best:
                        best = al
                        bi = jj
                        bkind = 1
            if hasv:
                alpha = best
                for jj in range(k):
                    j = free_idx[jj]
                    x[j] = x[j] * (1.0 - alpha) + alpha * z[jj]
                on_bound[free_idx[bi]] = np.float64(bkind)
            else:
                for jj in range(k):
                    x[free_idx[jj]] = z[jj]
                step_norm = np.linalg.norm(x[free_idx] - x_free_old)
                has_prev = True
                break

        r = A @ x - b
        cost_new = 0.5 * (r @ r)
        cost_change = cost - cost_new
        if cost_change < tol * cost:
            term = 2
        cost = cost_new
        g = A.T @ r
        opt = _kkt_optimality(g, on_bound)
        it += 1

    return x, cost, opt, it, term, initial_cost


# --------------------------------------------------------------------------
# Trust Region Reflective for a LINEAR least-squares problem, scipy's
# ``method='trf'`` default.
#
# scipy keeps this in its own file, `_lsq/trf_linear.py`, separate from the
# nonlinear ``trf`` that ``least_squares`` runs: it is a self-contained solver
# over `numpy`, `scipy.linalg` ``qr`` and ``solve_triangular``, the compiled
# ``givens_elimination`` helper, and eleven small functions from
# `_lsq/common.py`.  All of that is below, one function per scipy function,
# named for its scipy counterpart.
#
# ``scijit.linalg.qr_pivot`` IS ``scipy.linalg.qr(pivoting=True,
# mode='economic')``, including the 0-based permutation and the economy shape
# when m < n, so QR needed no port.
# --------------------------------------------------------------------------

_TRFL_EPS = np.finfo(np.float64).eps


@njit
def _trfl_in_bounds(x, lb, ub):
    """``lb <= x <= ub`` on every component, ``_lsq/common.py`` ``in_bounds``."""
    for i in range(x.size):
        if not (x[i] >= lb[i] and x[i] <= ub[i]):
            return False
    return True


@njit
def _trfl_step_to_bound(x, s, lb, ub):
    """``step_size_to_bound``: the stride along `s` that first hits a bound.

    Returns ``(min_step, hits)``, `hits` being ``-1`` where the lower bound
    is reached, ``+1`` the upper and ``0`` elsewhere.  A zero component of
    `s` never hits, which is scipy's ``np.sign(s)`` factor.
    """
    n = x.size
    steps = np.empty(n, np.float64)
    for i in range(n):
        if s[i] != 0.0:
            steps[i] = max((lb[i] - x[i]) / s[i], (ub[i] - x[i]) / s[i])
        else:
            steps[i] = np.inf
    min_step = np.inf
    for i in range(n):
        if steps[i] < min_step:
            min_step = steps[i]
    hits = np.zeros(n, np.int64)
    for i in range(n):
        if steps[i] == min_step:
            if s[i] > 0.0:
                hits[i] = 1
            elif s[i] < 0.0:
                hits[i] = -1
    return min_step, hits


@njit
def _trfl_active(x, lb, ub, rtol):
    """``find_active_constraints``, the ``-1``/``0``/``+1`` mask.

    The threshold is `rtol` times the magnitude of the closer bound, and an
    infinite bound is never active.  ``rtol = 0`` compares against the bound
    itself.  Where both tests pass the upper bound wins, which is the order
    scipy assigns them in.
    """
    n = x.size
    active = np.zeros(n, np.float64)
    if rtol == 0.0:
        for i in range(n):
            if x[i] <= lb[i]:
                active[i] = -1.0
            if x[i] >= ub[i]:
                active[i] = 1.0
        return active
    for i in range(n):
        ld = x[i] - lb[i]
        ud = ub[i] - x[i]
        if np.isfinite(lb[i]):
            if ld <= min(ud, rtol * max(1.0, abs(lb[i]))):
                active[i] = -1.0
        if np.isfinite(ub[i]):
            if ud <= min(ld, rtol * max(1.0, abs(ub[i]))):
                active[i] = 1.0
    return active


@njit
def _trfl_feasible(x, lb, ub, rstep):
    """``make_strictly_feasible``: push an iterate off its bounds.

    ``rstep = 0`` steps one ULP inward with ``np.nextafter``; otherwise the
    shift is `rstep` times the magnitude of the bound.  A box too narrow for
    either collapses to its midpoint.
    """
    n = x.size
    xn = x.copy()
    act = _trfl_active(x, lb, ub, rstep)
    for i in range(n):
        if act[i] == -1.0:
            if rstep == 0.0:
                xn[i] = np.nextafter(lb[i], ub[i])
            else:
                xn[i] = lb[i] + rstep * max(1.0, abs(lb[i]))
        elif act[i] == 1.0:
            if rstep == 0.0:
                xn[i] = np.nextafter(ub[i], lb[i])
            else:
                xn[i] = ub[i] - rstep * max(1.0, abs(ub[i]))
    for i in range(n):
        if xn[i] < lb[i] or xn[i] > ub[i]:
            xn[i] = 0.5 * (lb[i] + ub[i])
    return xn


@njit
def _trfl_cl_scaling(x, g, lb, ub):
    """``CL_scaling_vector``: the Coleman-Li vector `v` and its derivative."""
    n = x.size
    v = np.ones(n, np.float64)
    dv = np.zeros(n, np.float64)
    for i in range(n):
        if g[i] < 0.0 and np.isfinite(ub[i]):
            v[i] = ub[i] - x[i]
            dv[i] = -1.0
        elif g[i] > 0.0 and np.isfinite(lb[i]):
            v[i] = x[i] - lb[i]
            dv[i] = 1.0
    return v, dv


@njit
def _trfl_reflective(y, lb, ub):
    """``reflective_transformation``, the point only.

    The sign vector scipy returns beside it is discarded at both call sites
    in ``trf_linear``, so it is not built.
    """
    n = y.size
    x = y.copy()
    if _trfl_in_bounds(y, lb, ub):
        return x
    for i in range(n):
        lf = np.isfinite(lb[i])
        uf = np.isfinite(ub[i])
        if lf and not uf:
            x[i] = max(y[i], 2.0 * lb[i] - y[i])
        elif uf and not lf:
            x[i] = min(y[i], 2.0 * ub[i] - y[i])
        elif lf and uf:
            d = ub[i] - lb[i]
            t = (y[i] - lb[i]) % (2.0 * d)
            x[i] = lb[i] + min(t, 2.0 * d - t)
    return x


@njit
def _trfl_givens(S, v, diag):
    """``givens_elimination``: rotate a diagonal block into the zeros.

    ``[S; diag(d)]`` with `S` upper triangular becomes upper triangular
    again, `S` and `v` in place.  scipy ships this as a Cython extension
    over BLAS ``drotg`` and ``drot``.

    The rotation is LAPACK 3.10's ``drotg``: ``d = sqrt(f*f + g*g)``,
    ``c = |f| / d``, ``r = copysign(d, f)``, ``s = g / r``.  Measured against
    the shipped extension over 2000 random cases of size 1 to 7, including a
    zero in `diag` and a zero on `S`'s diagonal: worst absolute difference
    1.776e-15 and 866 of 2000 bit-identical.  The remainder is that
    OpenBLAS's ``drotg`` accumulates in long double on x86-64.  Three other
    spellings of the same rotation were measured and are worse: MINPACK's
    ``qrsolv`` form 8.016e+00, the reference ``drotg`` sign rule 8.745e+00,
    and ``hypot`` in place of the square root 1.887e-15 at 541 of 2000.
    """
    n = diag.size
    drow = np.zeros(n, np.float64)
    for i in range(n):
        if diag[i] == 0.0:
            continue
        for k in range(i, n):
            drow[k] = 0.0
        drow[i] = diag[i]
        u = 0.0
        for j in range(i, n):
            if drow[j] == 0.0:
                continue
            f = S[j, j]
            g = drow[j]
            if g == 0.0:
                c = 1.0
                s = 0.0
                r = f
            elif f == 0.0:
                c = 0.0
                s = 1.0 if g > 0.0 else -1.0
                r = abs(g)
            else:
                d = np.sqrt(f * f + g * g)
                c = abs(f) / d
                r = d if f > 0.0 else -d
                s = g / r
            S[j, j] = r
            drow[j] = 0.0
            for k in range(j + 1, n):
                a1 = S[j, k]
                a2 = drow[k]
                S[j, k] = c * a1 + s * a2
                drow[k] = c * a2 - s * a1
            b1 = v[j]
            v[j] = c * b1 + s * u
            u = c * u - s * b1


@njit
def _trfl_reg_lsq_qr(m, n, R, QTb, perm, diag):
    """``regularized_lsq_with_qr``: solve ``[A; diag(d)] x = [b; 0]``.

    `R` and `QTb` come from the pivoted QR of `A`; `R` is overwritten.  The
    rows whose rotated diagonal falls under ``eps * max(m, n) * max|diag R|``
    are dropped, which is scipy's rank test, and the triangular solve runs on
    what is left.
    """
    v = QTb.copy()
    dp = np.empty(n, np.float64)
    for i in range(n):
        dp[i] = diag[perm[i]]
    _trfl_givens(R, v, dp)
    mx = 0.0
    for i in range(n):
        if abs(R[i, i]) > mx:
            mx = abs(R[i, i])
    threshold = _TRFL_EPS * max(m, n) * mx
    nk = 0
    for i in range(n):
        if abs(R[i, i]) > threshold:
            nk += 1
    nns = np.empty(nk, np.int64)
    k = 0
    for i in range(n):
        if abs(R[i, i]) > threshold:
            nns[k] = i
            k += 1
    x = np.zeros(n, np.float64)
    if nk == 0:
        return x
    Rs = np.empty((nk, nk), np.float64)
    vs = np.empty(nk, np.float64)
    for a in range(nk):
        vs[a] = v[nns[a]]
        for c in range(nk):
            Rs[a, c] = R[nns[a], nns[c]]
    sol = _solve_triangular(Rs, vs)
    for a in range(nk):
        x[perm[nns[a]]] = sol[a]
    return x


@njit
def _trfl_eval_quad(A, g, s):
    """``evaluate_quadratic`` with no diagonal term, `s` one-dimensional."""
    Js = A @ s
    return 0.5 * (Js @ Js) + (s @ g)


@njit
def _trfl_eval_quad_h(A, d, g_h, s, diag):
    """``evaluate_quadratic`` against the column-scaled ``A diag(d)``.

    scipy builds that matrix as a ``LinearOperator`` whose ``matvec`` is
    ``A.dot(x * d)``, so the scaling multiplies the VECTOR.  ``(A * d) @ s``
    rounds differently and is not used.
    """
    Js = A @ (s * d)
    return 0.5 * (Js @ Js + (s * diag) @ s) + (s @ g_h)


@njit
def _trfl_build_quad(A, d, g_h, s, diag):
    """``build_quadratic_1d`` without `s0`: the ``t**2`` and ``t`` terms."""
    v = A @ (s * d)
    a = v @ v + (s * diag) @ s
    a *= 0.5
    return a, g_h @ s


@njit
def _trfl_build_quad_s0(A, d, g_h, s, s0, diag):
    """``build_quadratic_1d`` with `s0`: the same, plus the free term."""
    v = A @ (s * d)
    a = v @ v + (s * diag) @ s
    a *= 0.5
    b = g_h @ s
    u = A @ (s0 * d)
    b += u @ v
    c = 0.5 * (u @ u) + (g_h @ s0)
    b += (s0 * diag) @ s
    c += 0.5 * ((s0 * diag) @ s0)
    return a, b, c


@njit
def _trfl_min_quad(a, b, lo, hi, c):
    """``minimize_quadratic_1d``: the smaller of the ends and the extremum."""
    t = np.empty(3, np.float64)
    t[0] = lo
    t[1] = hi
    nt = 2
    if a != 0.0:
        ex = -0.5 * b / a
        if lo < ex < hi:
            t[2] = ex
            nt = 3
    bi = 0
    by = t[0] * (a * t[0] + b) + c
    for i in range(1, nt):
        y = t[i] * (a * t[i] + b) + c
        if y < by:
            by = y
            bi = i
    return t[bi], by


@njit
def _trfl_select_step(x, A, d, g_h, c_h, p, p_h, lb, ub, theta):
    """``select_step``: the best of the Gauss-Newton, reflected and gradient steps.

    `p` and `p_h` are SCALED IN PLACE, which is what scipy does and what the
    backtracking branch of the caller then reads.
    """
    if _trfl_in_bounds(x + p, lb, ub):
        return p
    p_stride, hits = _trfl_step_to_bound(x, p, lb, ub)
    r_h = p_h.copy()
    for i in range(r_h.size):
        if hits[i] != 0:
            r_h[i] = -r_h[i]
    r = d * r_h
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p
    r_stride_u, _rh = _trfl_step_to_bound(x_on_bound, r, lb, ub)
    r_stride_l = (1.0 - theta) * r_stride_u
    r_stride_u = r_stride_u * theta
    if r_stride_u > 0.0:
        qa, qb, qc = _trfl_build_quad_s0(A, d, g_h, r_h, p_h, c_h)
        r_stride, r_value = _trfl_min_quad(qa, qb, r_stride_l, r_stride_u, qc)
        r_h = p_h + r_h * r_stride
        r = d * r_h
    else:
        r_value = np.inf
    p_h *= theta
    p *= theta
    p_value = _trfl_eval_quad_h(A, d, g_h, p_h, c_h)
    ag_h = -g_h
    ag = d * ag_h
    ag_stride_u, _ah = _trfl_step_to_bound(x, ag, lb, ub)
    ag_stride_u = ag_stride_u * theta
    ga, gb = _trfl_build_quad(A, d, g_h, ag_h, c_h)
    ag_stride, ag_value = _trfl_min_quad(ga, gb, 0.0, ag_stride_u, 0.0)
    ag = ag * ag_stride
    if p_value < r_value and p_value < ag_value:
        return p
    elif r_value < p_value and r_value < ag_value:
        return r
    return ag


@njit
def _trfl_backtracking(A, g, x, p, theta, p_dot_g, lb, ub):
    """``backtracking``: halve the stride until the quadratic model drops.

    Returns the step and the cost change.  `x` is NOT advanced here, which
    is scipy's own control flow: its ``backtracking`` returns the `x` it was
    given, so the iterate stands still on this branch.
    """
    alpha = 1.0
    while True:
        x_new = _trfl_reflective(x + alpha * p, lb, ub)
        step = x_new - x
        cost_change = -_trfl_eval_quad(A, g, step)
        if cost_change > -0.1 * alpha * p_dot_g:
            break
        alpha *= 0.5
    act = _trfl_active(x_new, lb, ub, 1e-10)
    for i in range(act.size):
        if act[i] != 0.0:
            x_new = _trfl_reflective(x + theta * alpha * p, lb, ub)
            x_new = _trfl_feasible(x_new, lb, ub, 0.0)
            step = x_new - x
            cost_change = -_trfl_eval_quad(A, g, step)
            break
    return step, cost_change


@njit
def _trf_linear(A, b, lb, ub, tol, max_iter, verbose):
    """Trust Region Reflective for bounded LINEAR least squares.

    The compiled core behind ``lsq_linear(method='trf')``, scipy's default.
    Takes ``(A, b, lb, ub, tol, max_iter, verbose)`` with `max_iter` below
    zero as the sentinel for scipy's ``None``, which means 100 here.

    `verbose` at 2 prints the header and one row per iteration, at the point
    in the loop where scipy prints them, and nothing on the short circuit.
    The two summary lines belong to `lsq_linear`.

    Returns ``(x, cost, optimality, nit, status, active_mask,
    initial_cost)``.  `status` is scipy's own, ``-1`` no progress, ``0``
    iterations exhausted, ``1`` optimality under `tol`, ``2`` relative cost
    change under `tol`, ``3`` the unconstrained solution was feasible.
    `optimality` is the uniform norm of the SCALED gradient, which is a
    different quantity from the KKT measure `_bvls` reports.
    `initial_cost` is the cost at the strictly feasible starting point,
    which `lsq_linear` prints and scipy deletes from its result.

    A feasible unconstrained least-squares solution short-circuits, exactly
    as in scipy, and is reported with ``nit = 0`` and an all-zero
    `active_mask`.  Every iterate after that is strictly interior to the box,
    so `active_mask` is decided within `tol` of a bound rather than exactly.

    Measured against ``scipy.optimize.lsq_linear(method='trf')`` over 40
    random bounded problems, m in 3..19 and n in 1..8: worst ``|dx|``
    3.608e-16, worst ``|dcost|`` 1.776e-15, and `nit`, `status` and
    `active_mask` identical on every one.  On a 200x40 the worst ``|dx|`` is
    9.021e-17.  The residual difference is the two LAPACK builds: this
    package's ``dgeqp3`` and scipy's bundled one differ at 5.204e-16 on the
    same matrix.
    """
    m, n = A.shape
    x_lsq = np.linalg.lstsq(A, b, -1.0)[0]
    if _trfl_in_bounds(x_lsq, lb, ub):
        r0 = A @ x_lsq - b
        g0 = A.T @ r0
        o0 = 0.0
        for j in range(n):
            if abs(g0[j]) > o0:
                o0 = abs(g0[j])
        c0 = 0.5 * (r0 @ r0)
        return x_lsq, c0, o0, 0, 3, np.zeros(n, np.float64), c0

    x = _trfl_feasible(_trfl_reflective(x_lsq, lb, ub), lb, ub, 0.1)
    Q, R0, perm = _qr_pivot(A)
    QT = np.ascontiguousarray(Q.T)
    R = np.zeros((n, n), np.float64)
    k = min(m, n)
    for i in range(k):
        for j in range(n):
            R[i, j] = R0[i, j]
    QTr = np.zeros(n, np.float64)

    r = A @ x - b
    g = A.T @ r
    cost = 0.5 * (r @ r)
    initial_cost = cost
    if max_iter < 0:
        max_iter = 100

    term = -2                      # scipy's `termination_status is None`
    g_norm = 0.0
    iteration = 0
    cost_change = 0.0
    step_norm = 0.0
    has_prev = False
    dperm = np.empty(n, np.float64)
    Rd = np.empty((n, n), np.float64)
    if verbose == 2:
        _lsql_show_header()
    for iteration in range(max_iter):
        v, dv = _trfl_cl_scaling(x, g, lb, ub)
        g_scaled = g * v
        g_norm = 0.0
        for i in range(n):
            if abs(g_scaled[i]) > g_norm:
                g_norm = abs(g_scaled[i])
        if g_norm < tol:
            term = 1
        if verbose == 2:
            _lsql_show_row(iteration, cost, cost_change, step_norm, g_norm,
                           has_prev, has_prev)
        if term != -2:
            break
        diag_h = g * dv
        diag_root_h = np.sqrt(diag_h)
        d = np.sqrt(v)
        g_h = d * g
        for i in range(n):
            dperm[i] = d[perm[i]]
        for i in range(n):
            for j in range(n):
                Rd[i, j] = R[i, j] * dperm[j]
        QTrp = QT @ r
        for i in range(k):
            QTr[i] = QTrp[i]
        p_h = -_trfl_reg_lsq_qr(m, n, Rd, QTr, perm, diag_root_h)
        p = d * p_h
        p_dot_g = p @ g
        if p_dot_g > 0.0:
            term = -1
        theta = 1.0 - min(0.005, g_norm)
        step = _trfl_select_step(x, A, d, g_h, diag_h, p, p_h, lb, ub, theta)
        cost_change = -_trfl_eval_quad(A, g, step)
        if cost_change < 0.0:
            step, cost_change = _trfl_backtracking(
                A, g, x, p, theta, p_dot_g, lb, ub)
        else:
            x = _trfl_feasible(x + step, lb, ub, 0.0)
        step_norm = np.linalg.norm(step)
        has_prev = True
        r = A @ x - b
        g = A.T @ r
        if cost_change < tol * cost:
            term = 2
        cost = 0.5 * (r @ r)
    if term == -2:
        term = 0
    return (x, cost, g_norm, iteration + 1, term,
            _trfl_active(x, lb, ub, tol), initial_cost)


#: ``lsq_linear``'s result, scipy's key set.  scipy carries the same ten on
#: every method, measured over 'trf', 'bvls' and a bounded fit.
LsqLinearResult = _opt_result(
    ['x', 'fun', 'cost', 'optimality', 'active_mask', 'unbounded_sol',
     'nit', 'status', 'message', 'success'])

#: scipy's ``TERMINATION_MESSAGES``, indexed by ``status + 1`` so that the
#: ``-1`` ``trf`` reports has a text of its own.
_LSQL_MESSAGES = (
    'The algorithm was not able to make progress on the last iteration.',
    'The maximum number of iterations is exceeded.',
    'The first-order optimality measure is less than `tol`.',
    'The relative change of the cost function is less than `tol`.',
    'The unconstrained solution is optimal.',
)

#: scipy's own texts, `_lsq/lsq_linear.py:346-405` and `_lsq/common.py`.
_LSQL_MSG_BOUNDS = ("Each lower bound must be strictly less than each upper "
                    "bound.")
_LSQL_MSG_B2D = "`b` must have at most 1 dimension."
_LSQL_MSG_MAXITER = "`max_iter` must be None or positive integer."
_LSQL_MSG_LSMRIT = "`lsmr_maxiter` must be None or positive integer."
_LSQL_MSG_SOLVER = "`solver` must be None, 'exact' or 'lsmr'."
_LSQL_MSG_LSMR_BVLS = "method='bvls' can't be used with lsq_solver='lsmr'"
_LSQL_MSG_VERBOSE = "`verbose` must be in [0, 1, 2]."
_LSQL_MSG_A2D = "`A` must have at most 2 dimensions."
_LSQL_MSG_AB = "Inconsistent shapes between `A` and `b`."
_LSQL_MSG_BOUNDS2 = "`bounds` must contain 2 elements."
_LSQL_MSG_BSHAPE = "Bounds have wrong shape."
_LSQL_MSG_LSMRTOL = "`lsmr_tol` must be None, 'auto', or positive float."

_LSQL_MSG_METHOD = "`method` must be 'trf' or 'bvls'"

_LSQL_MSG_METHOD_LIT = (
    "lsq_linear: `method` must be a literal string inside @njit. The two "
    "methods report `nit` and `active_mask` differently, so the choice is "
    "made while the call compiles.")

_LSQL_LSMR_MSG = (
    "lsq_linear: lsq_solver='lsmr' is not implemented. The dense default, "
    "lsq_solver='exact', runs a pivoted QR and is what a call that names "
    "neither uses. Pass lsq_solver=None or 'exact'.")


@njit
def _active_mask(x, lb, ub):
    """scipy's ``active_mask``: -1 on the lower bound, +1 on the upper, 0 free."""
    n = x.size
    out = np.zeros(n, np.float64)
    for j in range(n):
        if x[j] <= lb[j]:
            out[j] = -1.0
        elif x[j] >= ub[j]:
            out[j] = 1.0
    return out


def _lsql_a2d(A):
    """`A` as a C-contiguous 2-D float64 array, scipy's ``np.atleast_2d``.

    A one-dimensional `A` becomes a single ROW, which is what scipy solves;
    three dimensions or more is the shape it refuses.
    """
    Ac = np.ascontiguousarray(np.asarray(A, np.float64))
    if Ac.ndim == 1:
        return Ac.reshape(1, Ac.size)
    if Ac.ndim != 2:
        raise ValueError(_LSQL_MSG_A2D)
    return Ac


@overload(_lsql_a2d)
def _lsql_a2d_ovl(A):
    nd = A.ndim if isinstance(A, types.Array) else 0
    if nd == 1:
        def impl(A):
            Ac = np.ascontiguousarray(np.asarray(A).astype(np.float64))
            return Ac.reshape(1, Ac.size)
        return impl
    if nd != 2:
        raise TypingError(_LSQL_MSG_A2D)

    def impl(A):
        return np.ascontiguousarray(np.asarray(A).astype(np.float64))
    return impl


def _lsql_bounds_len(bounds):
    """scipy's ``prepare_bounds`` gate, ``_lsq/lsq_linear.py:14-16``.

    ``len`` of something with no length is CPython's own ``TypeError``,
    which is what scipy reaches for a scalar `bounds`.
    """
    if len(bounds) != 2:
        raise ValueError(_LSQL_MSG_BOUNDS2)


@njit
def _lsql_check_lu(lb, ub, n):
    """The bound shapes and the strict ordering, scipy `:302-307`."""
    if lb.size != n or ub.size != n:
        raise ValueError(_LSQL_MSG_BSHAPE)
    for j in range(n):
        if lb[j] >= ub[j]:
            raise ValueError(_LSQL_MSG_BOUNDS)


def _lsql_check_lsmr_tol(lsmr_tol):
    """scipy's own condition, ``_lsq/lsq_linear.py:312-314``.

    A positive ``float`` instance, ``'auto'`` or ``None``. An integer is
    refused whatever its value, and so is ``True``, because neither is a
    ``float``; ``numpy.float64`` is one and passes.
    """
    if not ((isinstance(lsmr_tol, float) and lsmr_tol > 0)
            or lsmr_tol in ('auto', None)):
        raise ValueError(_LSQL_MSG_LSMRTOL)


@njit
def _split_bounds_lu(bounds, n):
    """scipy's ``bounds`` 2-tuple -> (lb, ub) arrays of length n.

    Each element is a scalar broadcast across all variables or an array of
    length n, which is what scipy accepts.
    """
    lo = np.asarray(bounds[0], np.float64)
    hi = np.asarray(bounds[1], np.float64)
    lb = np.full(n, lo.item()) if lo.ndim == 0 else np.ascontiguousarray(lo)
    ub = np.full(n, hi.item()) if hi.ndim == 0 else np.ascontiguousarray(hi)
    return lb, ub


def lsq_linear(A, b, bounds=(-np.inf, np.inf), method='trf', tol=1e-10,
               lsq_solver=None, lsmr_tol=None, max_iter=None, verbose=0, *,
               lsmr_maxiter=None):
    """Bounded-variable linear least squares.

    Solves ``min 0.5 * ||A x - b||^2`` subject to ``lb <= x <= ub``, by the
    trust-region-reflective method by default and by Stark and Parker's BVLS
    active-set method on request.  Callable from Python and from inside
    ``@njit``; both entries run the same compiled cores.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Design matrix. Dense only. A one-dimensional `A` is read as a single
        row.
    b : array_like, shape (m,)
        Right-hand side.
    bounds : 2-tuple, optional
        ``(lb, ub)``, exactly two elements. Each is a scalar or an array of
        length ``n``. Default ``(-inf, inf)``, unbounded.
    method : {'trf', 'bvls'}, optional
        ``'trf'``, the default, is the trust-region-reflective solver, whose
        iterates stay strictly inside the box. ``'bvls'`` is the active-set
        solver, which reaches the bounds exactly. Inside ``@njit`` this must
        be a literal string, since the two report `nit` and `active_mask`
        differently.
    tol : float, optional
        Termination tolerance on the first-order optimality measure and on
        the relative change in cost. Default 1e-10.
    lsq_solver : {None, 'exact'}, optional
        The inner solve. ``None`` and ``'exact'`` both run the dense pivoted
        QR. ``'lsmr'`` raises: on ``method='bvls'`` it is not a valid pairing,
        and on ``method='trf'`` it selects an iterative solver this package
        does not have. Any other string raises.
    lsmr_tol : None, 'auto' or float, optional
        Tunes the LSMR inner solver, which the ``'lsmr'`` value of
        `lsq_solver` selects, so it is not read here. Validated to ``None``,
        ``'auto'`` or a positive float; an integer is refused whatever its
        value. Default ``None``.
    max_iter : int or None, optional
        Iteration cap. ``None`` (default) resolves to 100 on ``'trf'`` and to
        ``n`` on ``'bvls'``. Anything at or below zero raises.
    verbose : {0, 1, 2}, optional
        Progress reporting on stdout. ``0``, the default, is silent. ``1``
        prints the termination message and a summary line after the solve.
        ``2`` adds a five-column table with one row per iteration. See Notes.
    lsmr_maxiter : int or None, keyword-only, optional
        Tunes the LSMR inner solver, so it is not read here. Validated to
        ``None`` or an integer at least 1. Default ``None``.

    Returns
    -------
    res : LsqLinearResult
        ``x``, ``fun``, ``cost``, ``optimality``, ``active_mask``, ``nit``,
        ``status``, ``unbounded_sol``, ``message``, ``success``, reached as
        attributes. ``status`` is ``-1`` no progress on the last iteration,
        reachable on ``'trf'`` only, ``0`` iteration limit, ``1`` optimality
        below ``tol``, ``2`` cost change below ``tol``, ``3`` the
        unconstrained solution was already optimal.
        ``optimality`` is the uniform norm of the scaled gradient on
        ``'trf'`` and the KKT measure on ``'bvls'``. ``active_mask`` is
        ``-1`` on a lower bound, ``+1`` on an upper one and ``0`` free, and is
        all zero whenever ``status`` is ``3``. On ``'trf'`` every iterate is
        strictly interior, so the mask is decided within ``tol`` of a bound
        rather than on equality. ``unbounded_sol`` is the whole
        ``numpy.linalg.lstsq`` tuple, ``(solution, residuals, rank,
        singular_values)``, taken at ``rcond=-1``.

    Raises
    ------
    ValueError
        If `method` is neither ``'trf'`` nor ``'bvls'``; if `lsq_solver` is
        not ``None``, ``'exact'`` or ``'lsmr'``, or is ``'lsmr'``; if
        `verbose` is outside ``0``, ``1``, ``2``; if `A` has
        more than two dimensions; if `max_iter` is at or below zero; if `b`
        has more than one dimension or a length other than ``A.shape[0]``;
        if `bounds` does not hold exactly two elements, a bound has the
        wrong length, or a lower bound is not strictly below its upper one;
        if `lsmr_maxiter` is below 1 or `lsmr_tol` is neither ``None``,
        ``'auto'`` nor a positive float; or if either dimension of `A` is
        zero.
    TypeError
        If `bounds` has no length at all, which is CPython's own message.
    numpy.linalg.LinAlgError
        On a non-finite entry in `A` or `b`, from the unconstrained solve.

    See Also
    --------
    scipy.optimize.lsq_linear : The scipy routine this mirrors.
    scijit.optimize.nnls : The ``x >= 0`` special case.
    scijit.optimize.leastsq : Nonlinear least squares.
    scijit.optimize.curve_fit : Fit a model to data.

    Notes
    -----
    The two methods solve the same problem and stop on different criteria, so
    they land in different places and report different accounting. On the 3x2
    problem below, ``'trf'`` takes 16 iterations to ``x[0] =
    1.4999954223632812`` and ``'bvls'`` takes 2 to an exact ``1.5``, the two
    4.578e-06 apart. scipy behaves the same way, and the default is ``'trf'``
    on both sides.

    `res` carries the ten fields in ONE order on every exit path:
    ``unbounded_sol`` sits between `active_mask` and `nit`. scipy has two
    orders, that one on the ``status = 3`` short circuit and a second
    elsewhere with ``unbounded_sol`` after `status`, because its two paths
    build the object differently. Reaching a field by name or by attribute
    is unaffected on both sides; what differs is ``list(res.keys())`` on the
    paths where ``status`` is not ``3``.

    ``unbounded_sol``'s third member, the rank, is an ``int64`` here and an
    ``int32`` in scipy.

    A sparse or ``LinearOperator`` `A` is not accepted, and neither is a
    `bounds` given as a ``scipy.optimize.Bounds`` object; pass the pair, nor
    ``lsq_solver='lsmr'``, which selects an iterative inner solve.

    `verbose` above ``0`` prints, and printing takes the GIL, so a
    ``prange`` loop over solves is serialized for the duration of each line.
    ``verbose=0`` reaches no printing code.

    An `A` with a zero dimension raises. scipy 1.18 answers instead: shape
    ``(12, 0)`` returns ``[]`` while LAPACK prints ``On entry to DLASCL
    parameter number 4 had an illegal value``. That answer is produced
    through an invalid LAPACK call.

    No callback of either style and no module state, so it is safe to call
    from a ``numba.prange`` loop; ``'trf'`` reaches LAPACK ``dgeqp3`` through
    ``scijit.linalg.qr_pivot``, which is prange-safe too (verified: 24
    concurrent solves reproduce the serial answer to 2.78e-16).

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import lsq_linear
    >>> A = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> b = np.array([2.0, 1.0, -1.0])
    >>> @njit
    ... def run():
    ...     return lsq_linear(A, b, (np.array([0.0, 0.0]),
    ...                              np.array([1.5, 1.0])))
    >>> res = run()
    >>> res.x
    array([1.49999542e+000, 4.94065646e-324])
    >>> res.status, res.success
    (1, True)

    The active-set method reaches the bound exactly, in two iterations:

    >>> bvls = lsq_linear(A, b, (np.array([0.0, 0.0]),
    ...                          np.array([1.5, 1.0])), 'bvls')
    >>> bvls.x, bvls.nit
    (array([1.5, 0. ]), 2)
    """
    # the order is scipy's, `_lsq/lsq_linear.py:249-314`
    if method not in ('trf', 'bvls'):
        raise ValueError(_LSQL_MSG_METHOD)
    if lsq_solver is not None and lsq_solver not in ('exact', 'lsmr'):
        raise ValueError(_LSQL_MSG_SOLVER)
    if verbose != 0 and verbose != 1 and verbose != 2:
        raise ValueError(_LSQL_MSG_VERBOSE)
    if method == 'bvls' and lsq_solver == 'lsmr':
        raise ValueError(_LSQL_MSG_LSMR_BVLS)
    if lsq_solver == 'lsmr':
        raise ValueError(_LSQL_LSMR_MSG)
    A = _lsql_a2d(A)
    if max_iter is not None and max_iter <= 0:
        raise ValueError(_LSQL_MSG_MAXITER)
    b = np.ascontiguousarray(np.asarray(b, np.float64))
    if b.ndim > 1:
        raise ValueError(_LSQL_MSG_B2D)
    b = b.ravel()
    n = A.shape[1]
    if b.size != A.shape[0]:
        raise ValueError(_LSQL_MSG_AB)
    _lsql_bounds_len(bounds)
    lb, ub = _split_bounds_lu(bounds, n)
    _lsql_check_lu(lb, ub, n)
    if lsmr_maxiter is not None and lsmr_maxiter < 1:
        raise ValueError(_LSQL_MSG_LSMRIT)
    _lsql_check_lsmr_tol(lsmr_tol)
    mi = -1 if max_iter is None else int(max_iter)
    if method == 'trf':
        x, cost, opt, nit, st, am, ic = _trf_linear(A, b, lb, ub, tol, mi,
                                                    verbose)
        res = _lsql_pack_trf(A, b, x, cost, opt, nit, st, am)
    else:
        x, cost, opt, nit, term, ic = _bvls(A, b, lb, ub, tol, mi, verbose)
        res = _lsql_pack(A, b, x, cost, opt, nit, term, lb, ub)
    if verbose > 0:
        if res.status == 3:
            _lsql_py_short(res.message, res.cost, res.optimality)
        else:
            _lsql_py_summary(res.message, res.nit, ic, res.cost,
                             res.optimality)
    return res


@njit
def _lsql_pack(A, b, x, cost, opt, nit, term, lb, ub):
    """Assemble scipy's result fields from the core's five outputs.

    ``nit`` follows scipy's own accounting, which is not a plain count.
    ``scipy/optimize/_lsq/bvls.py`` runs ``for iteration in range(iteration,
    max_iter)`` and returns ``nit=iteration + 1``, so its ``iteration`` is
    the index of the LAST pass.  Our counter increments at the END of a
    pass, so it equals that index when the loop BREAKS on convergence
    (report ``it + 1``) and exceeds it by one when the loop exhausts the cap
    (report ``it``).  The unconstrained short circuit never enters the loop
    and reports 0, as scipy does.  Both branches were measured against
    scipy over 45 bounded problems and at ``max_iter=1``.

    ``active_mask`` is ``zeros(n)`` on the short circuit whatever the
    solution is, which is scipy's ``:340``: nothing was made active there,
    so a solution that happens to sit on a finite bound is still free.
    """
    st = 0 if term == -1 else term
    nout = nit if (st == 0 or st == 3) else nit + 1
    if st == 3:
        am = np.zeros(x.size, np.float64)
    else:
        am = _active_mask(x, lb, ub)
    return LsqLinearResult(x, A @ x - b, cost, opt, am,
                           np.linalg.lstsq(A, b, -1.0), nout, st,
                           _LSQL_MESSAGES[st + 1], st > 0)


@njit
def _lsql_pack_trf(A, b, x, cost, opt, nit, st, am):
    """The same result fields from ``_trf_linear``'s six outputs.

    `nit` and `active_mask` arrive finished here, unlike the BVLS route:
    ``trf`` counts its own iterations the way scipy's result reports them,
    and it decides the mask within `tol` of a bound rather than on equality.
    `status` reaches ``-1`` on this route, which is why the message table is
    indexed from ``-1``.
    """
    return LsqLinearResult(x, A @ x - b, cost, opt, am,
                           np.linalg.lstsq(A, b, -1.0), nit, st,
                           _LSQL_MESSAGES[st + 1], st > 0)


@overload(lsq_linear, prefer_literal=True)
def _lsq_linear_ovl(A, b, bounds=(-np.inf, np.inf), method='trf', tol=1e-10,
                    lsq_solver=None, lsmr_tol=None, max_iter=None,
                    verbose=0, *, lsmr_maxiter=None):
    """@njit implementation of `lsq_linear`.

    ``method`` picks the solver while the call compiles, because the two
    cores return their accounting fields differently; a non-literal
    ``method`` is refused rather than resolved at run time.
    ``lsq_solver='lsmr'`` selects machinery this package does not have and
    is refused here too, and so is an `lsmr_tol` whose TYPE scipy refuses;
    ``max_iter=None`` becomes the sentinel ``-1``. Returning ``None``
    declines the call, which numba reports as a TypingError naming the
    argument that could not be served.
    """
    meth = 'trf' if _is_none(method) else _cf_lit_str(method)
    if meth is None:
        raise TypingError(_LSQL_MSG_METHOD_LIT)
    if meth not in ('trf', 'bvls'):
        raise TypingError(_LSQL_MSG_METHOD)
    if not _is_none(lsq_solver):
        sv = _cf_lit_str(lsq_solver)
        if sv == 'lsmr':
            raise TypingError(_LSQL_MSG_LSMR_BVLS if meth == 'bvls'
                              else _LSQL_LSMR_MSG)
        if sv != 'exact':
            raise TypingError(_LSQL_MSG_SOLVER)
    mi_none = _is_none(max_iter)
    it_none = _is_none(lsmr_maxiter)
    if isinstance(b, types.Array) and b.ndim > 1:
        raise TypingError(_LSQL_MSG_B2D)
    # scipy takes a positive float, 'auto' or None, and nothing else; only
    # the sign is left to run time.
    tol_none = _is_none(lsmr_tol)
    if not tol_none:
        if isinstance(lsmr_tol, (types.UnicodeType, types.StringLiteral)):
            if _cf_lit_str(lsmr_tol) != 'auto':
                raise TypingError(_LSQL_MSG_LSMRTOL)
            tol_none = True
        elif not isinstance(lsmr_tol, types.Float):
            raise TypingError(_LSQL_MSG_LSMRTOL)
    # a tuple carries its length in its type; an array does not
    if isinstance(bounds, types.BaseTuple) and len(bounds) != 2:
        raise TypingError(_LSQL_MSG_BOUNDS2)
    bnd_arr = isinstance(bounds, types.Array)

    use_trf = meth == 'trf'

    def impl(A, b, bounds=(-np.inf, np.inf), method='trf', tol=1e-10,
             lsq_solver=None, lsmr_tol=None, max_iter=None, verbose=0,
             lsmr_maxiter=None):
        if verbose != 0 and verbose != 1 and verbose != 2:
            raise ValueError(_LSQL_MSG_VERBOSE)
        Ac = _lsql_a2d(A)
        if mi_none:
            mi = -1
        else:
            if max_iter <= 0:
                raise ValueError(_LSQL_MSG_MAXITER)
            mi = np.int64(max_iter)
        bc = np.ascontiguousarray(np.asarray(b).astype(np.float64)).ravel()
        if bc.size != Ac.shape[0]:
            raise ValueError(_LSQL_MSG_AB)
        if bnd_arr:
            if bounds.size != 2:
                raise ValueError(_LSQL_MSG_BOUNDS2)
        lb, ub = _split_bounds_lu(bounds, Ac.shape[1])
        _lsql_check_lu(lb, ub, Ac.shape[1])
        if not it_none:
            if lsmr_maxiter < 1:
                raise ValueError(_LSQL_MSG_LSMRIT)
        if not tol_none:
            if lsmr_tol <= 0.0:
                raise ValueError(_LSQL_MSG_LSMRTOL)
        if use_trf:
            xt, ct, ot, nt, stt, amt, ict = _trf_linear(Ac, bc, lb, ub, tol,
                                                        mi, verbose)
            res = _lsql_pack_trf(Ac, bc, xt, ct, ot, nt, stt, amt)
            ic = ict
        else:
            x, cost, opt, nit, term, icb = _bvls(Ac, bc, lb, ub, tol, mi,
                                                 verbose)
            res = _lsql_pack(Ac, bc, x, cost, opt, nit, term, lb, ub)
            ic = icb
        if verbose > 0:
            if res.status == 3:
                _lsql_show_short(res.message, res.cost, res.optimality)
            else:
                _lsql_show_summary(res.message, res.nit, ic, res.cost,
                                   res.optimality)
        return res
    return impl


#: scipy's two `sigma` texts, `_minpack_py.py:996` and `:1001`.
_CF_MSG_SIGMA_SHAPE = "`sigma` has incorrect shape."
_CF_MSG_SIGMA_PD = "`sigma` must be positive definite."

_CF_MSG_SCOPE = (
    "curve_fit: bounds, method and jac must be left at their defaults. bounds "
    "and method select least_squares (trf/dogbox), which is not implemented; "
    "jac is a callable or a str code, which the same two methods consume.")

_CF_MSG_IER = "Optimal parameters not found: "

#: scipy's text for an empty `ydata`, `_minpack_py.py:952`.
_CF_MSG_EMPTY_Y = "`ydata` must not be empty!"

_CF_MSG_EMPTY_P0 = (
    "curve_fit: p0 must hold at least one parameter. scipy 1.18 reaches "
    "`TypeError: object of type 'numpy.float64' has no len()` on the same "
    "input, from inside its own covariance branch.")

#: scipy's text when a NaN survives to the finiteness check,
#: `numpy.asarray_chkfinite`.
_CF_MSG_FINITE = "array must not contain infs or NaNs"

#: scipy's ``curve_fit`` infodict for the 'lm' method, as a namedtuple.
#: Same five keys leastsq reports, which IS scipy's 'lm' key set.
CurveFitInfo = LsqInfo


class OptimizeWarning(UserWarning):
    """Warning category for recoverable problems during optimization.

    See Also
    --------
    scipy.optimize.OptimizeWarning : The scipy category this mirrors.

    Notes
    -----
    This is a DIFFERENT CLASS from ``scipy.optimize.OptimizeWarning``, with
    the same name and the same base. A filter naming this class selects it,
    and so does one naming ``UserWarning`` or a bare
    ``warnings.catch_warnings(record=True)``. A filter naming scipy's class
    does NOT: ``simplefilter('error', scipy.optimize.OptimizeWarning)``
    leaves this warning as a warning, and ``except
    scijit.optimize.OptimizeWarning`` does not catch a scipy one. Ported
    code that turns scipy's class into an exception has to name this one as
    well. Importing scipy's would put scipy in this package's runtime
    dependencies, which it is not.

    ``curve_fit`` is the one site that raises it here, on the branch where
    the covariance could not be estimated. scipy also raises it for an
    unknown solver option and from several ``linprog`` and ``fminbound``
    conditions.
    """


#: scipy's own text at this site, read off scipy 1.18 on 2026-08-01.
_CF_MSG_PCOV = "Covariance of the parameters could not be estimated"


def _emit_pcov_warning():
    warnings.warn(_CF_MSG_PCOV, OptimizeWarning, stacklevel=2)


@njit
def _warn_pcov():
    """Warn exactly where scipy warns: ``pcov`` left all-inf.

    ``warnings.warn`` is not typeable by numba; an ``objmode`` block runs
    its body in the interpreter and the warning reaches the ordinary Python
    machinery, so ``catch_warnings``, ``simplefilter``, ``-W`` and
    ``PYTHONWARNINGS`` all see it.  The block takes the GIL, and is reached
    only on the branch that could not form the covariance.
    """
    with objmode():
        _emit_pcov_warning()



# --------------------------------------------------------------------------
# scipy's model, `f(x, *p)`.
#
# scipy's `curve_fit` takes a MODEL, not a residual: `f(xdata, a, b, ...)`
# returning the predicted y. The residual MINPACK needs is built here, once
# per model, and its address frozen into the compiled body. The parameter
# count comes from the model's own signature, which is what lets `p0` default
# to `ones(n)` exactly as scipy's does.
#
# The body has to be generated rather than written, because the number of
# parameters is part of the CALL and numba has no way to splat an array into
# a compiled call (measured: `f(x, *p)` with `p` an array is a TypingError).
# --------------------------------------------------------------------------

_CF_MODELS = {}
_CF_INNER = {}

_CF_MSG_MODEL_LEN = (
    "curve_fit: the model returned %d values for %d data points. scipy "
    "subtracts the model output from `ydata` and numpy raises on the same "
    "shapes.")


def _cf_model_out(v, m):
    """The model's return as a length-`m` float64 array.

    A scalar is broadcast, which is what scipy's ``f(xdata, *p) - ydata``
    does through numpy. Any other length raises.
    """
    a = np.asarray(v, np.float64).ravel()
    if a.ndim == 0 or a.size == 1:
        return np.full(m, float(a.reshape(1)[0]))
    if a.size != m:
        raise ValueError(_CF_MSG_MODEL_LEN % (a.size, m))
    return a


@overload(_cf_model_out)
def _cf_model_out_ovl(v, m):
    """Compiled twin. The scalar case is decided by TYPE, the length at run
    time; the raise is reached inside a ``@cfunc``, where numba prints it and
    the callback returns without writing `fvec`, which the residual probe
    then reports."""
    if not isinstance(v, types.Array):
        def impl(v, m):
            return np.full(m, np.float64(v))
        return impl

    def impl(v, m):
        a = np.ascontiguousarray(v).astype(np.float64).ravel()
        if a.size == 1:
            return np.full(m, a[0])
        if a.size != m:
            raise ValueError("curve_fit: the model returned a number of "
                             "values that is not len(ydata)")
        return a
    return impl


_CF_MODEL_SRC = """
def _make(inner, np_):
    @cfunc(minpack_sig)
    def adapter(p_ptr, f_ptr, a_ptr):
        m = int(a_ptr[0])
        kind = int(a_ptr[1])
        if kind == 1:
            ln = 2 + 3 * m
        elif kind == 2:
            ln = 2 + 2 * m + m * m
        else:
            ln = 2 + 2 * m
        buf = carray(a_ptr, ln)
        p = carray(p_ptr, np_)
        out = _cf_model_out(inner(buf[2:2 + m], %(plist)s), m)
        for i in range(m):
            f_ptr[i] = out[i] - buf[2 + m + i]
        if kind == 1:
            for i in range(m):
                f_ptr[i] = f_ptr[i] * buf[2 + 2 * m + i]
        elif kind == 2:
            o = 2 + 2 * m
            for i in range(m):
                s = f_ptr[i]
                for k in range(i):
                    s -= buf[o + i * m + k] * f_ptr[k]
                f_ptr[i] = s / buf[o + i * m + i]
    return adapter
"""


#: Largest parameter count served for a ``f(x, *params)`` model.  The arity
#: is not in the signature, so one adapter is generated per count up to this
#: and the call picks by ``len(p0)``.
_CF_STAR_NMAX = 15

_CF_MSG_STAR_P0 = "Unable to determine number of fit parameters."

_CF_MSG_STAR_MAX = (
    "curve_fit: a model taking *params is served for up to %d parameters, "
    "and `p0` names more. A model with a written-out parameter list has no "
    "such limit." % _CF_STAR_NMAX)

_CF_MSG_STAR_BAD = (
    "curve_fit: the model does not compile at the number of parameters "
    "`p0` names.")

_CF_MSG_STAR_NONE = (
    "curve_fit: the model taking *params does not compile at any parameter "
    "count from 1 to %d." % _CF_STAR_NMAX)


def _model_is_star(py):
    """``f(x, *params)``: the parameter count is not in the signature."""
    return bool(py.__code__.co_flags & 0x04)      # CO_VARARGS


def _model_nparams(py):
    """The model's parameter count: everything after `x`.

    ``-1`` for a ``f(x, *params)`` model, whose count is settled by `p0`.
    """
    if _model_is_star(py):
        return -1
    na = py.__code__.co_argcount
    if na < 2:
        raise ValueError(
            "curve_fit: the model must be f(x, p0, p1, ...) with at least "
            "one parameter; got %d argument(s)" % na)
    return na - 1


def _cf_arity_msgs(py, npar):
    """CPython's two texts for a `p0` that disagrees with the model.

    scipy calls ``f(xdata, *params)``, so a `p0` of the wrong length reaches
    CPython's argument binding and the message is CPython's rather than
    scipy's.  The model's name and its parameter names are known once the
    model is, so the SHORT direction is one finished string per count and
    only the LONG direction carries a number the call site supplies.

    Returns ``(miss, over)``: `miss` is indexed by the number of parameters
    GIVEN, and `over` wants ``str(n + 1) + " were given"`` appended.
    """
    names = py.__code__.co_varnames[1:npar + 1]
    fn = getattr(py, '__name__', '<lambda>')
    miss = []
    for g in range(npar):
        q = ["'%s'" % w for w in names[g:]]
        if len(q) == 1:
            lst = q[0]
        elif len(q) == 2:
            lst = "%s and %s" % (q[0], q[1])
        else:
            lst = "%s, and %s" % (", ".join(q[:-1]), q[-1])
        miss.append("%s() missing %d required positional argument%s: %s"
                    % (fn, len(q), "s" if len(q) > 1 else "", lst))
    return tuple(miss), "%s() takes %d positional arguments but " % (
        fn, npar + 1)


@njit
def _cf_check_arity(n, m, npar, miss, over):
    """`p0`'s length against the model's, where CPython would raise.

    The adapter is generated with the model's own parameter count and writes
    exactly that many slots, so a shorter `p0` reads past the end of the
    parameter buffer and a longer one fits a variable the model never sees.
    Neither is visible inside a ``@cfunc``, where a raise is printed and
    swallowed, so the count is compared here, one level up, while both
    lengths are still known.

    Three conditions raise before the model is ever called, on both sides:
    an empty `ydata`, an empty `p0`, and more parameters than data points
    outside the ``m == 1`` exemption.  This test yields to all three.
    """
    if n < 1 or m < 1 or (m != 1 and n > m):
        return
    if n < npar:
        raise TypeError(miss[n])
    if n > npar:
        raise TypeError(over + str(n + 1) + " were given")


def _build_adapter(inner, npar):
    """One ``cfunc(minpack_sig)`` residual at a fixed parameter count."""
    plist = ", ".join("p[%d]" % k for k in range(npar))
    ns = {"cfunc": cfunc, "carray": carray, "minpack_sig": minpack_sig,
          "_cf_model_out": _cf_model_out}
    exec(_CF_MODEL_SRC % {"plist": plist}, ns)
    return ns["_make"](inner, npar)


def _adapter_model(py, npar=None):
    """cfunc(minpack_sig) residual around a plain @njit model ``f(x, *p)``.

    Reads ``[m, kind, *xdata, *ydata, *transform]`` out of the args buffer,
    which is what `_pack_xy` puts there, and calls the model once with the
    whole `xdata`.

    With `npar` given, returns the single adapter at that parameter count.
    With `npar` left out, it comes from the model's signature; a
    ``f(x, *params)`` model has none, and the return is then a TUPLE of
    ``_CF_STAR_NMAX`` addresses, one per count from 1 up, which a compiled
    caller indexes by ``len(p0)``.  A homogeneous tuple can be indexed by a
    run-time integer inside ``@njit``, which is what makes the choice
    reachable from compiled code.  The Python entry knows ``len(p0)`` and
    asks for that one count, so it never pays for the other fourteen.
    """
    if npar is None:
        npar = _model_nparams(py)
    key = (py, npar)
    hit = _CF_MODELS.get(key)
    if hit is not None:
        return hit
    inner = _CF_INNER.get(py)
    if inner is None:
        inner = njit(py)
        _CF_INNER[py] = inner
    if npar < 0:
        # A ``*params`` body still only types at the counts it indexes, so
        # the ones it refuses are kept as the address 0 and reported at the
        # call rather than here.
        addrs = []
        for k in range(1, _CF_STAR_NMAX + 1):
            try:
                addrs.append(_adapter_model(py, k).address)
            except Exception:                             # noqa: BLE001
                addrs.append(0)
        if not any(addrs):
            raise ValueError(_CF_MSG_STAR_NONE)
        made = tuple(addrs)
    else:
        made = _build_adapter(inner, npar)
    _CF_MODELS[key] = made
    return made


@njit
def _pack_xy(xd, yd, tr, kind):
    """The args buffer the model adapter reads.

    ``[m, kind, *xdata, *ydata, *transform]``. `kind` is scipy's three
    weighting cases: ``0`` no `sigma`, ``1`` a per-point multiplier
    ``1 / sigma``, ``2`` the lower Cholesky factor of a covariance `sigma`,
    stored row-major and applied by forward substitution.
    """
    m = yd.size
    out = np.empty(2 + 2 * m + tr.size, np.float64)
    out[0] = np.float64(m)
    out[1] = np.float64(kind)
    for i in range(m):
        out[2 + i] = xd[i]
        out[2 + m + i] = yd[i]
    for i in range(tr.size):
        out[2 + 2 * m + i] = tr[i]
    return out


@njit
def _cf_tr_scalar(s, m):
    """One standard deviation for every point: the multiplier ``1 / sigma``."""
    return np.full(m, 1.0 / s), 1


@njit
def _cf_tr_1d(s, m):
    """A length-`m` `sigma` of standard deviations, or a length-1 broadcast."""
    if s.size == 1:
        return np.full(m, 1.0 / s[0]), 1
    if s.size != m:
        raise ValueError(_CF_MSG_SIGMA_SHAPE)
    tr = np.empty(m, np.float64)
    for i in range(m):
        tr[i] = 1.0 / s[i]
    return tr, 1


@njit
def _cf_tr_2d(s, m):
    """An ``(m, m)`` covariance `sigma` -> its lower Cholesky factor `L`.

    The residual then solves ``L z = r``, which is scipy's
    ``solve_triangular(transform, r, lower=True)``. The factorisation is
    written out rather than taken from ``np.linalg.cholesky`` so that a
    matrix which is not positive definite reaches scipy's own message
    instead of a ``LinAlgError``.
    """
    if s.size == 1:
        return np.full(m, 1.0 / s[0, 0]), 1
    if s.shape[0] != m or s.shape[1] != m:
        raise ValueError(_CF_MSG_SIGMA_SHAPE)
    L = np.zeros((m, m), np.float64)
    for i in range(m):
        for j in range(i + 1):
            acc = s[i, j]
            for k in range(j):
                acc -= L[i, k] * L[j, k]
            if i == j:
                if acc <= 0.0:
                    raise ValueError(_CF_MSG_SIGMA_PD)
                L[i, j] = np.sqrt(acc)
            else:
                L[i, j] = acc / L[j, j]
    return L.ravel(), 2


def _cf_transform(sigma, m):
    """scipy's `sigma` -> ``(transform, kind)``, ``_minpack_py.py:986-1003``.

    Rank decides which of the three helpers runs, and rank is a compile-time
    property of a numba array type, so the branch is taken while the call is
    typing rather than in the body. A body branching on ``s.ndim`` would have
    to type an ``s[i]`` beside an ``s[i, j]``.
    """
    s = np.asarray(sigma, np.float64)
    if s.ndim == 0:
        return _cf_tr_scalar(s.item(), m)
    if s.ndim == 1:
        return _cf_tr_1d(np.ascontiguousarray(s), m)
    if s.ndim == 2:
        return _cf_tr_2d(np.ascontiguousarray(s), m)
    raise ValueError(_CF_MSG_SIGMA_SHAPE)


@overload(_cf_transform)
def _cf_transform_ovl(sigma, m):
    if isinstance(sigma, types.Array):
        if sigma.ndim == 1:
            def impl(sigma, m):
                return _cf_tr_1d(np.ascontiguousarray(
                    np.asarray(sigma).astype(np.float64)), m)
            return impl
        if sigma.ndim == 2:
            def impl(sigma, m):
                return _cf_tr_2d(np.ascontiguousarray(
                    np.asarray(sigma).astype(np.float64)), m)
            return impl
        raise TypingError(_CF_MSG_SIGMA_SHAPE)

    def impl(sigma, m):
        return _cf_tr_scalar(np.float64(sigma), m)
    return impl


_CF_MSG_F = (
    "curve_fit: f must be a plain @njit model f(x, p0, p1, ...) returning "
    "the predicted y. A python callable cannot be reached from compiled "
    "code, and a raw @cfunc .address is not accepted.")

_CF_MSG_XY = "curve_fit: xdata and ydata must have the same length"

#: scipy's three `nan_policy` texts, `_minpack_py.py:955-963`.
_CF_MSG_NP_PROP = "`nan_policy='propagate'` is not supported by this function."
_CF_MSG_NP_BAD = "nan_policy must be one of {None, 'raise', 'omit'}"
_CF_MSG_NP_RAISE = "The input contains nan values"

@njit
def _cf_chkfinite(xd, yd):
    """scipy's ``np.asarray_chkfinite`` on both arrays, ydata first."""
    for i in range(yd.size):
        if not np.isfinite(yd[i]):
            raise ValueError(_CF_MSG_FINITE)
    for i in range(xd.size):
        if not np.isfinite(xd[i]):
            raise ValueError(_CF_MSG_FINITE)


@njit
def _cf_nan_raise(xd, yd):
    """scipy's ``nan_policy='raise'``: NaN only, infinities pass."""
    for i in range(xd.size):
        if np.isnan(xd[i]):
            raise ValueError(_CF_MSG_NP_RAISE)
    for i in range(yd.size):
        if np.isnan(yd[i]):
            raise ValueError(_CF_MSG_NP_RAISE)


@njit
def _cf_nan_omit(xd, yd):
    """scipy's ``nan_policy='omit'``: drop every index NaN in either array.

    Infinities are kept, which is scipy's rule: its mask is built from
    ``np.isnan`` on both arrays and never consults ``np.isfinite``.
    """
    m = yd.size
    keep = np.zeros(m, np.bool_)
    k = 0
    for i in range(m):
        if not (np.isnan(xd[i]) or np.isnan(yd[i])):
            keep[i] = True
            k += 1
    xo = np.empty(k, np.float64)
    yo = np.empty(k, np.float64)
    j = 0
    for i in range(m):
        if keep[i]:
            xo[j] = xd[i]
            yo[j] = yd[i]
            j += 1
    return xo, yo


@njit
def _cf_broadcast_xy(xd, yd, tr, kind):
    """numpy's broadcast of the model output against `ydata`.

    scipy forms the residual as ``f(xdata, *p) - ydata``, so a length-1
    `ydata` is repeated across the model's output and a length-1 `xdata` the
    other way round, and any other length disagreement is numpy's own
    broadcast failure. `sigma` is validated against the length `ydata` had,
    so a transform holding one entry is repeated with it.
    """
    if yd.size == 1 and xd.size > 1:
        yd = np.full(xd.size, yd[0])
        if kind == 1 and tr.size == 1:
            tr = np.full(xd.size, tr[0])
    elif xd.size == 1 and yd.size > 1:
        xd = np.full(yd.size, xd[0])
    elif xd.size != yd.size:
        raise ValueError("operands could not be broadcast together with "
                         "shapes (" + str(xd.size) + ",) ("
                         + str(yd.size) + ",) ")
    return xd, yd, tr


@njit
def _cf_omit_keep(xd, yd):
    """The mask ``nan_policy='omit'`` keeps: neither array NaN at that index."""
    keep = np.ones(yd.size, np.bool_)
    for i in range(yd.size):
        if np.isnan(xd[i]) or np.isnan(yd[i]):
            keep[i] = False
    return keep


@njit
def _cf_boolidx(axis, size, nkeep):
    """numpy's text for a boolean index whose length is not the axis length."""
    return ("boolean index did not match indexed array along axis "
            + str(axis) + "; size of axis is " + str(size)
            + " but size of corresponding boolean axis is " + str(nkeep))


@njit
def _cf_omit_len_msg(nx, ny):
    """numpy's text for ``has_nan |= np.isnan(ydata)`` with two lengths.

    scipy builds the mask from `xdata` and ORs `ydata`'s into it in place,
    so the failure names three shapes: the two operands and the output,
    which is `xdata`'s.
    """
    return ("operands could not be broadcast together with shapes ("
            + str(nx) + ",) (" + str(ny) + ",) (" + str(nx) + ",) ")


@njit
def _cf_omit_1d(s, keep):
    """``sigma[~has_nan]`` for a vector of standard deviations."""
    if keep.all():
        return s
    if s.size != keep.size:
        raise IndexError(_cf_boolidx(0, s.size, keep.size))
    k = 0
    for i in range(keep.size):
        if keep[i]:
            k += 1
    out = np.empty(k, np.float64)
    j = 0
    for i in range(keep.size):
        if keep[i]:
            out[j] = s[i]
            j += 1
    return out


@njit
def _cf_omit_2d(s, keep):
    """``sigma[~has_nan, :][:, ~has_nan]`` for a covariance matrix."""
    if keep.all():
        return s
    if s.shape[0] != keep.size:
        raise IndexError(_cf_boolidx(0, s.shape[0], keep.size))
    if s.shape[1] != keep.size:
        raise IndexError(_cf_boolidx(1, s.shape[1], keep.size))
    k = 0
    for i in range(keep.size):
        if keep[i]:
            k += 1
    out = np.empty((k, k), np.float64)
    r = 0
    for i in range(keep.size):
        if keep[i]:
            c = 0
            for j in range(keep.size):
                if keep[j]:
                    out[r, c] = s[i, j]
                    c += 1
            r += 1
    return out


def _cf_sigma_omit(sigma, keep):
    """`sigma` with the omitted points dropped, ``_minpack_py.py:977-984``.

    A vector loses the same entries as the data; a covariance loses the same
    rows and the same columns; a scalar is left alone. An all-True `keep` is
    left alone as well, which is scipy's rule: the mask is applied only on
    the branch where a NaN was found.
    """
    s = np.asarray(sigma)
    if s.ndim == 1:
        return _cf_omit_1d(np.ascontiguousarray(s.astype(np.float64)), keep)
    if s.ndim == 2:
        return _cf_omit_2d(np.ascontiguousarray(s.astype(np.float64)), keep)
    return sigma


@overload(_cf_sigma_omit)
def _cf_sigma_omit_ovl(sigma, keep):
    if isinstance(sigma, types.Array):
        if sigma.ndim == 1:
            def impl(sigma, keep):
                return _cf_omit_1d(np.ascontiguousarray(
                    np.asarray(sigma).astype(np.float64)), keep)
            return impl
        if sigma.ndim == 2:
            def impl(sigma, keep):
                return _cf_omit_2d(np.ascontiguousarray(
                    np.asarray(sigma).astype(np.float64)), keep)
            return impl

    def impl(sigma, keep):
        return sigma
    return impl


@njit
def _pairwise_sum(a, off, n):
    """``numpy.sum`` over ``a[off:off + n]``, in numpy's own summation order.

    scipy's reduced chi-square is ``np.sum(fvec ** 2)``, and numpy does not
    accumulate that sequentially: below 8 elements it runs a plain loop,
    up to 128 it carries eight partial accumulators and combines them as a
    balanced tree, and above that it splits in half at a multiple of 8 and
    recurses. A sequential accumulator differs from it by a last bit, which
    is enough to move `pcov`.

    Verified against ``np.sum(v ** 2)`` over every length from 0 to 299 and
    at 512, 1000 and 4097: bit-identical at all 303 sizes.
    """
    if n < 8:
        s = 0.0
        for i in range(n):
            s += a[off + i]
        return s
    if n <= 128:
        r0 = a[off]
        r1 = a[off + 1]
        r2 = a[off + 2]
        r3 = a[off + 3]
        r4 = a[off + 4]
        r5 = a[off + 5]
        r6 = a[off + 6]
        r7 = a[off + 7]
        i = 8
        while i < n - (n % 8):
            r0 += a[off + i]
            r1 += a[off + i + 1]
            r2 += a[off + i + 2]
            r3 += a[off + i + 3]
            r4 += a[off + i + 4]
            r5 += a[off + i + 5]
            r6 += a[off + i + 6]
            r7 += a[off + i + 7]
            i += 8
        s = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            s += a[off + i]
            i += 1
        return s
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum(a, off, n2) + _pairwise_sum(a, off + n2, n - n2)


def _cf_lit_str(v):
    """Compile-time str out of whatever numba hands the overload.

    The shapes are `_lit_bool`'s: an OMITTED argument arrives as the raw
    Python default, an explicit one as a ``types.StringLiteral``. ``None``
    means a runtime variable, which cannot be served because the policy
    selects which validation runs.
    """
    if isinstance(v, str):
        return v
    if isinstance(v, types.Omitted) and isinstance(v.value, str):
        return v.value
    if isinstance(v, types.StringLiteral):
        return v.literal_value
    return None


def _cf_nan_flags(check_finite, nan_policy):
    """Resolve scipy's `check_finite` / `nan_policy` pair at typing time.

    Returns ``(cf_dynamic, cf_const, np_prop, np_bad, np_raise, np_omit)``.
    scipy's rule, ``_minpack_py.py:934-965``: `check_finite` at ``None``
    resolves to ``True`` when `nan_policy` is also ``None`` and to ``False``
    otherwise, and `nan_policy` is validated only on the branch where the
    finiteness check did not run.
    """
    npol = None if _is_none(nan_policy) else _cf_lit_str(nan_policy)
    if not _is_none(nan_policy) and npol is None:
        raise TypingError(
            "curve_fit: nan_policy must be a compile-time string literal "
            "inside @njit; it selects which validation runs.")
    if _is_none(check_finite):
        cf_dynamic, cf_const = False, npol is None
    else:
        lit = _lit_bool(check_finite)
        if lit is None:
            cf_dynamic, cf_const = True, False
        else:
            cf_dynamic, cf_const = False, lit
    return (cf_dynamic, cf_const, npol == 'propagate',
            npol is not None and npol not in ('raise', 'omit', 'propagate'),
            npol == 'raise', npol == 'omit')


@njit
def _cf_core(resid_ptr, p0, args, m, ftol, xtol, gtol, maxfev, epsfcn,
             factor, diag, absolute_sigma, mchk):
    """THE algorithm. Both entry points only slice this return.

    `pcov` is MINPACK's own covariance, the ``cov_x`` the inner `leastsq`
    returns, built by inverting the QR factor ``lmdif`` already produced and
    undoing the column pivoting. scipy takes the same matrix from the same
    place, so the two run one estimator rather than two estimates of one
    quantity. The scaling and the three unestimable branches below follow
    ``_minpack_py.curve_fit`` statement for statement.
    """
    n = p0.size
    if m < 1:
        raise ValueError(_CF_MSG_EMPTY_Y)
    if n < 1:
        raise ValueError(_CF_MSG_EMPTY_P0)
    # scipy's guard, `_minpack_py.py:1021-1023`. It counts the entries
    # `ydata` HAD, and exempts a single one because that is the broadcast
    # case; MINPACK's own message covers the exempted cell on both sides.
    if mchk != 1 and n > mchk:
        raise TypeError("The number of func parameters=" + str(n)
                        + " must not exceed the number of data points="
                        + str(mchk))

    # `_leastsq_ptr` rather than `leastsq`: the residual here is a @cfunc
    # this module built from the user's model, and the public `leastsq`
    # takes a plain @njit callback only.  Same core, same return.
    popt, cov, info, mesg, ier = _leastsq_ptr(
        resid_ptr, p0, args, m, ftol, xtol, gtol, maxfev, epsfcn, factor,
        diag, True)
    fvec = info.fvec

    # scipy returns an all-inf pcov whenever the covariance is not
    # estimable: no `cov_x`, because MINPACK never reached a solution or its
    # triangular factor was singular; a NaN in the covariance; or too few
    # residuals for the parameters. Filling with ZEROS instead would read as
    # a perfectly determined fit, the opposite of what is true.
    pcov = np.full((n, n), np.inf, np.float64)
    warn_cov = True
    if cov is not None:
        cv = cov                        # narrows the Optional to an array
        bad = False
        for i in range(n):
            for j in range(n):
                if np.isnan(cv[i, j]):
                    bad = True
        if not bad:
            if absolute_sigma:
                pcov = cv
                warn_cov = False
            elif m > n:
                ssr = _pairwise_sum(fvec * fvec, 0, m)
                pcov = cv * (ssr / (m - n))
                warn_cov = False
    # scipy warns here and returns; `popt` in the same return is usable and
    # is often the only thing the caller wanted.  This sits in the shared
    # core, so it fires once whether the call came from Python or from
    # inside @njit.
    if warn_cov:
        _warn_pcov()
    return popt, pcov, info, mesg, ier


def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
              check_finite=None, bounds=(-np.inf, np.inf), method=None,
              jac=None, *, full_output=False, nan_policy=None,
              args=np.array([0.0]), tol=1.49012e-8, maxfev=0, ftol=None,
              xtol=None, gtol=0.0, epsfcn=None, factor=100.0, diag=None,
              col_deriv=False):
    """Nonlinear least-squares curve fit.

    Fits a model ``f(x, p0, p1, ...)`` to data by nonlinear least squares and
    returns the best-fit parameters with a covariance estimate. Callable from
    Python and from inside ``@njit``; both entries run the same compiled core.

    Parameters
    ----------
    f : callable
        A MODEL ``f(x, p0, p1, ...)`` returning the predicted y, and the
        parameter count comes from its signature. A plain ``@njit`` model is
        reached from both entry points; a plain Python function is compiled
        here and so is reached from the Python entry.
    xdata, ydata : array_like
        Independent and dependent data, one dimension each. The residual
        count ``m`` is the longer of the two; an array holding a single
        value is repeated across the other, and any other disagreement
        raises.
    p0 : array_like or None, optional
        Initial parameters. ``None`` (default) is ``ones(n)`` with `n` read
        off the model's signature. A `p0` of any other length than the model
        takes raises ``TypeError``.
    sigma : array_like or float or None, optional
        Uncertainties on `ydata`. A scalar or a length-`m` array holds
        standard deviations and the residual becomes
        ``(model - ydata) / sigma``; an ``(m, m)`` array is a covariance and
        the residual is transformed by its lower Cholesky factor. Under
        ``nan_policy='omit'`` the entries dropped from the data are dropped
        from `sigma` as well. See Notes.
    absolute_sigma : bool, optional
        ``False`` (default) scales ``pcov`` by the reduced chi-square
        ``ssr / (m - n)``. ``True`` returns MINPACK's covariance unscaled.
    check_finite : bool or None, optional
        ``None`` (default) is ``True`` when `nan_policy` is ``None`` and
        ``False`` otherwise. ``True`` raises ``ValueError`` on a non-finite
        entry in either array.
    bounds, method, jac : optional
        Accepted only at their defaults. See Notes.
    full_output : bool, keyword-only, optional
        ``False`` (default) returns ``(popt, pcov)``. ``True`` returns
        ``(popt, pcov, infodict, mesg, ier)``. Compile-time constant inside
        ``@njit``: it selects the return type.
    nan_policy : {None, 'raise', 'omit'}, keyword-only, optional
        Read only when the finiteness check did not run. ``'raise'`` refuses
        a NaN, ``'omit'`` drops the points where either array is NaN.
    args : ndarray, optional
        BEYOND SCIPY. A flat buffer accepted for signature compatibility and
        not read: the model reads its data from `xdata` and `ydata`.
    tol : float, optional
        BEYOND SCIPY. Sets both `ftol` and `xtol` when neither is given.
    maxfev, ftol, xtol, gtol, epsfcn, factor, diag, col_deriv : optional
        Passed to the inner ``leastsq``. `ftol` and `xtol` default to `tol`.
        `maxfev` at ``0`` means MINPACK's ``200 * (n + 1)``. `col_deriv`
        selects the layout of an analytic Jacobian, which this route does not
        pass, so it reaches nothing.

    Returns
    -------
    popt : ndarray, shape (n,)
        Best-fit parameters.
    pcov : ndarray, shape (n, n)
        Covariance estimate, MINPACK's own, formed by inverting the QR
        factor ``lmdif`` produced and undoing the column pivoting.
        ALL-INF where the covariance is not estimable: MINPACK reached no
        solution, its triangular factor was singular, a NaN reached the
        covariance, or ``m <= n`` with ``absolute_sigma=False``. That branch
        issues :class:`OptimizeWarning` with the text "Covariance of the
        parameters could not be estimated", through a ``numba.objmode``
        block, so ``warnings.catch_warnings`` and ``-W`` see it from compiled
        and interpreted calls alike. All-zero on an exact fit.
    infodict : CurveFitInfo
        With ``full_output``. ``fvec``, ``nfev``, ``fjac``, ``ipvt``,
        ``qtf``, in the order the fields are reached positionally.
    mesg : str
        With ``full_output``. MINPACK's termination message.
    ier : int
        With ``full_output``. ``1..4`` is success; anything else raises
        ``RuntimeError`` before returning.

    Raises
    ------
    ValueError
        If a non-finite entry survives the finiteness check, or a NaN
        survives ``nan_policy='raise'``; if the model takes ``*params`` and
        `p0` is ``None``, names more than 15 parameters, or names a count
        the model body does not accept; if `sigma` has the wrong shape or
        is not positive definite; if `method` or `jac` is anything but
        ``None``, or `bounds` anything but ``(-inf, inf)``; if `f` is not a
        plain ``@njit`` model; if `xdata` and `ydata` have lengths that do
        not broadcast; if `p0` holds no parameters; or if `ydata` is empty.
    TypeError
        If the parameter count exceeds ``len(ydata)``, or `p0` is not as
        long as the model's parameter list.
    IndexError
        If ``nan_policy='omit'`` is given with a `sigma` that is not as
        long as the data.
    RuntimeError
        If the inner ``leastsq`` returns an `ier` outside ``1..4``.

    Warns
    -----
    OptimizeWarning
        `pcov` could not be estimated and is returned all-inf.

    See Also
    --------
    scipy.optimize.curve_fit : The scipy routine this mirrors.
    scijit.optimize.leastsq : The solver underneath.
    scijit.optimize.lsq_linear : Bounded LINEAR least squares.

    Notes
    -----
    `bounds`, `method` and `jac` are accepted only at their defaults.
    `bounds`, and ``method='trf'`` or ``'dogbox'``, select the nonlinear
    ``least_squares`` this package does not have. ``method='lm'`` and `jac`
    reach the ``leastsq`` path and are not wired through this front end.

    `sigma` and ``nan_policy='omit'`` act on the residual: `sigma` folds the
    weights into it, and ``'omit'`` shortens the data it reads. A `sigma` of
    shape ``(1, 1)`` is read as one standard deviation broadcast across the
    data, scipy's own ``sigma.size == 1`` rule.

    The model is called once per iteration with the whole `xdata` array and
    must return one value per point. A model that returns a single value is
    broadcast across the data, as ``f(xdata, *p) - ydata`` in scipy. A
    multi-dimensional `xdata` is not accepted, and neither is a callable
    object rather than a function, because the parameter count is read from
    the model's signature.

    A model written as ``f(x, *params)`` is accepted, with `p0` required.
    `p0` omitted raises ``ValueError('Unable to determine number of fit
    parameters.')``, and more than 15 parameters raises.

    A `p0` holding no parameters raises. scipy 1.18 reaches ``TypeError:
    object of type 'numpy.float64' has no len()`` on the same input.

    A model written with numpy operations needs no change. Two spellings do
    not, and each has a one-token fix: ``math.sin(x)`` becomes ``np.sin(x)``,
    and a branch on `x` such as ``if x > 0.5`` becomes
    ``np.where(x > 0.5, ...)``.

    `nfev` is one higher than scipy's on the same fit: one residual runs
    before MINPACK starts, and the count reports the callback's own work
    rather than MINPACK's counter.

    `infodict` is a namedtuple rather than a dict, and the 5-tuple return is
    a tuple rather than an ``OptimizeResult``.

    `full_output` must be a compile-time literal inside ``@njit``, and
    `nan_policy` must be a literal string.

    `tol` sets both ``ftol`` and ``xtol`` when neither is given. `args` is
    accepted for signature compatibility and not read; scipy rejects the
    keyword with ``ValueError("'args' is not a supported keyword
    argument.")``.

    Safe to call from a ``numba.prange`` loop: MINPACK holds the callback in
    a module variable carrying ``!$omp threadprivate``, one slot per thread.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    Examples
    --------
    Fit ``a * exp(-b * x)`` to noisy measurements. The model is scipy's spelling,
    ``f(x, *params)``, and `p0` defaults to ``ones(n)`` off its signature:

    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import curve_fit
    >>> @njit
    ... def model(x, a, b):
    ...     return a * np.exp(-b * x)
    >>> x = np.linspace(0.0, 4.0, 20)
    >>> rng = np.random.default_rng(0)
    >>> y = model(x, 2.5, 0.4) + rng.normal(0.0, 0.05, x.size)   # noisy measurements
    >>> @njit
    ... def run():
    ...     return curve_fit(model, x, y)
    >>> popt, pcov = run()
    >>> np.round(popt, 3)
    array([2.521, 0.411])
    """
    if (method is not None or jac is not None
            or bounds[0] != -np.inf or bounds[1] != np.inf):
        raise ValueError(_CF_MSG_SCOPE)
    # Only a plain @njit model is accepted; a raw @cfunc .address is refused.
    if isinstance(f, (int, np.integer)):
        raise ValueError(_CF_MSG_F)
    # scipy's resolution, `_minpack_py.py:934-935`
    if check_finite is None:
        check_finite = True if nan_policy is None else False
    xd = np.ascontiguousarray(np.asarray(xdata, np.float64)).ravel()
    yd = np.ascontiguousarray(np.asarray(ydata, np.float64)).ravel()
    if check_finite:
        _cf_chkfinite(xd, yd)
    # scipy's order, `_minpack_py.py:951` before `:956`
    if yd.size == 0:
        raise ValueError(_CF_MSG_EMPTY_Y)
    if not check_finite and nan_policy is not None:
        if nan_policy == 'propagate':
            raise ValueError(_CF_MSG_NP_PROP)
        if nan_policy not in ('raise', 'omit'):
            raise ValueError(_CF_MSG_NP_BAD)
        if nan_policy == 'raise':
            _cf_nan_raise(xd, yd)
        else:
            if xd.size != yd.size:
                # scipy ORs the two NaN masks together before it indexes
                raise ValueError(_cf_omit_len_msg(xd.size, yd.size))
            # scipy drops the same entries from `sigma`, `_minpack_py.py:977`
            if sigma is not None:
                sigma = _cf_sigma_omit(sigma, _cf_omit_keep(xd, yd))
            xd, yd = _cf_nan_omit(xd, yd)
    # the count scipy tests `p0` against is the one `ydata` arrived with
    m0 = yd.size
    # a Dispatcher carries the model as `py_func`; a plain Python
    # function IS the model and is compiled here, which is how scipy's
    # own spelling `curve_fit(lambda x, a, b: ..., ...)` is reached.
    py = f.py_func if hasattr(f, 'py_func') else f
    if not hasattr(py, '__code__'):
        raise ValueError(_CF_MSG_F)
    npar = _model_nparams(py)
    # scipy reads the parameter count off the model's signature and
    # defaults p0 to ones(n). The signature is readable here too, unless
    # the model takes *params, where scipy refuses for the same reason.
    if p0 is None:
        if npar < 0:
            raise ValueError(_CF_MSG_STAR_P0)
        p0 = np.ones(npar, np.float64)
    if sigma is None:
        tr, kind = np.empty(0, np.float64), 0
    else:
        tr, kind = _cf_transform(sigma, m0)
    if npar < 0:
        _n0 = int(np.asarray(p0).size)
        if _n0 > _CF_STAR_NMAX:
            raise ValueError(_CF_MSG_STAR_MAX)
        try:
            _made = _adapter_model(py, _n0).address
        except Exception:                             # noqa: BLE001
            raise ValueError(_CF_MSG_STAR_BAD)
    else:
        _miss, _over = _cf_arity_msgs(py, npar)
        _cf_check_arity(np.int64(np.asarray(p0).size), np.int64(m0),
                        npar, _miss, _over)
        _made = _adapter_model(py).address
    xd, yd, tr = _cf_broadcast_xy(xd, yd, tr, kind)
    fp, ab = _made, _pack_xy(xd, yd, tr, kind)
    # `.copy()`, because the solver writes the solution into this array and
    # `ravel()` of an already-contiguous float64 input is a VIEW of the
    # caller's `p0`. The compiled body reaches `.astype`, which always copies.
    p0 = np.ascontiguousarray(np.asarray(p0, np.float64)).ravel().copy()
    popt, pcov, info, mesg, ier = _cf_core(
        fp, p0, ab, yd.size, tol if ftol is None else ftol,
        tol if xtol is None else xtol, gtol, maxfev, epsfcn, factor, diag,
        bool(absolute_sigma), m0)
    if ier not in (1, 2, 3, 4):
        raise RuntimeError(_CF_MSG_IER + mesg)
    if full_output:
        return popt, pcov, info, mesg, ier
    return popt, pcov


@overload(curve_fit, prefer_literal=True)
def _curve_fit_ovl(f, xdata, ydata, p0=None, sigma=None,
                   absolute_sigma=False, check_finite=None,
                   bounds=(-np.inf, np.inf), method=None, jac=None, *,
                   full_output=False, nan_policy=None,
                   args=np.array([0.0]), tol=1.49012e-8, maxfev=0, ftol=None,
                   xtol=None, gtol=0.0, epsfcn=None, factor=100.0, diag=None,
                   col_deriv=False):
    """@njit implementation of `curve_fit`.

    ``full_output`` picks between a 2-tuple and a 5-tuple, so it must be a
    compile-time constant; ``absolute_sigma`` is read at run time.  ``p0``
    is required on the ``.address`` path, because the parameter count comes
    from the model signature there and a ``@cfunc`` address does not carry
    one. The remaining arguments select scipy machinery this package does
    not have and are refused rather than ignored. Returning ``None``
    declines the call, which numba reports as a TypingError naming the
    argument that could not be served.
    """
    fo = _lit_bool(full_output)
    if fo is None:
        return None                     # runtime flag -> TypingError
    if not (_is_none(method) and _is_none(jac)):
        raise TypingError(_CF_MSG_SCOPE)
    no_sigma = _is_none(sigma)
    ftol_none = _is_none(ftol)
    xtol_none = _is_none(xtol)
    (cf_dynamic, cf_const, np_prop, np_bad,
     np_raise, np_omit) = _cf_nan_flags(check_finite, nan_policy)
    no_p0 = _is_none(p0)

    # Resolve the callback AT COMPILE TIME. A Dispatcher `f` is scipy's
    # MODEL, for which the residual adapter is built here and the data
    # packed into the buffer it reads. A raw @cfunc .address is refused.
    if isinstance(f, types.Dispatcher):
        py = f.dispatcher.py_func
        npar = _model_nparams(py)
        made = _adapter_model(py)
        star = npar < 0
        if star:
            if no_p0:
                raise TypingError(_CF_MSG_STAR_P0)
            addr, miss, over = made, ('',), ''
            npar = 1        # types only; the star branch never reads it
        else:
            addr = made.address
            miss, over = _cf_arity_msgs(py, npar)
        model = True
    else:
        raise TypingError(_CF_MSG_F)

    def impl(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
             check_finite=None, bounds=(-np.inf, np.inf), method=None,
             jac=None, full_output=False, nan_policy=None,
             args=np.array([0.0]), tol=1.49012e-8, maxfev=0, ftol=None,
             xtol=None, gtol=0.0, epsfcn=None, factor=100.0, diag=None,
             col_deriv=False):
        if bounds[0] != -np.inf or bounds[1] != np.inf:
            raise ValueError(_CF_MSG_SCOPE)
        xd = np.ascontiguousarray(np.asarray(xdata).astype(np.float64)).ravel()
        yd = np.ascontiguousarray(np.asarray(ydata).astype(np.float64)).ravel()
        # all-True means no point was dropped, so `sigma` is left alone
        keep = np.ones(0, np.bool_)
        if cf_dynamic:
            do_check = bool(check_finite)
        else:
            do_check = cf_const
        if do_check:
            _cf_chkfinite(xd, yd)
        # scipy's order, `_minpack_py.py:951` before `:956`
        if yd.size == 0:
            raise ValueError(_CF_MSG_EMPTY_Y)
        if not do_check:
            if np_prop:
                raise ValueError(_CF_MSG_NP_PROP)
            if np_bad:
                raise ValueError(_CF_MSG_NP_BAD)
            if np_raise:
                _cf_nan_raise(xd, yd)
            if np_omit:
                if xd.size != yd.size:
                    raise ValueError(_cf_omit_len_msg(xd.size, yd.size))
                keep = _cf_omit_keep(xd, yd)
                xd, yd = _cf_nan_omit(xd, yd)
        m0 = yd.size
        if model:
            if no_p0:
                pf = np.ones(npar, np.float64)
            else:
                pf = np.ascontiguousarray(
                    np.asarray(p0).astype(np.float64)).ravel()
            if star:
                # one adapter per parameter count; `addr` is the tuple of
                # their addresses and `p0` settles which one runs.
                if pf.size < 1 or pf.size > _CF_STAR_NMAX:
                    raise ValueError(_CF_MSG_STAR_MAX)
                fp = addr[pf.size - 1]
                if fp == 0:
                    raise ValueError(_CF_MSG_STAR_BAD)
            else:
                fp = addr
            if no_sigma:
                tr = np.empty(0, np.float64)
                kind = 0
            else:
                tr, kind = _cf_transform(_cf_sigma_omit(sigma, keep), m0)
            if not star:
                _cf_check_arity(np.int64(pf.size), np.int64(m0), npar,
                                miss, over)
            xd, yd, tr = _cf_broadcast_xy(xd, yd, tr, kind)
            ab_buf = _pack_xy(xd, yd, tr, kind)
        else:
            fp = f
            ab_buf = np.ascontiguousarray(
                np.asarray(args).astype(np.float64)).ravel()
            pf = np.ascontiguousarray(
                np.asarray(p0).astype(np.float64)).ravel()
        if ftol_none:
            ft = tol
        else:
            ft = ftol
        if xtol_none:
            xt = tol
        else:
            xt = xtol
        popt, pcov, info, mesg, ier = _cf_core(
            fp, pf, ab_buf, yd.size, ft, xt, gtol, maxfev, epsfcn, factor,
            diag, absolute_sigma, m0)
        if ier < 1 or ier > 4:
            raise RuntimeError(_CF_MSG_IER + mesg)
        if fo:
            return popt, pcov, info, mesg, ier
        return popt, pcov
    return impl
