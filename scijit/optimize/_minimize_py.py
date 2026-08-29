"""Multivariate unconstrained minimizers, in numba ``@njit``.

These are the ``scipy.optimize`` minimizers that are **pure Python** in
scipy (Nelder-Mead, Powell, nonlinear CG, BFGS), so they transcribe
directly to ``@njit`` with no compiled backend and no module state.

They are **prange-safe**. So are the Fortran L-BFGS-B and SLSQP behind
:func:`scijit.optimize.minimize`, which are reverse communication with
their state in caller-owned arrays.

Callback convention (as in ``_scalar.py``): the objective and gradient are
PLAIN ``@njit`` functions passed as first-class arguments --

    ``func(x, *args) -> float64``      (x is 1-D float64)
    ``grad(x, *args) -> 1-D float64``  (cg / bfgs only, optional)

``args`` is a tuple, unpacked into the argument list after ``x``; its
elements may be of any type a compiled call accepts.  The default ``()``
calls ``func(x)``.  An ndarray or a list `args` arrives as ONE argument,
``func(x, args)``.  No ``@cfunc``, no ``.address``, no captured globals ->
reentrant -> safe to call from inside a ``numba.prange`` loop (many
independent minimizations at once, verified in the test suite).

--------------------------------------------------------------------------
PUBLIC API -- scipy's signatures, names and defaults:

  fmin(func, x0, args, xtol, ftol, maxiter, maxfun, full_output, disp,
       retall, callback, initial_simplex)          Nelder-Mead simplex
  fmin_cg(f, x0, fprime, args, gtol, norm, epsilon, maxiter, full_output,
          disp, retall, callback, c1, c2)          Polak-Ribiere+ CG
  fmin_bfgs(f, x0, fprime, args, gtol, norm, epsilon, maxiter, full_output,
            disp, retall, callback, xrtol, c1, c2, hess_inv0)      BFGS
  fmin_powell(func, x0, args, xtol, ftol, maxiter) Powell direction-set

``full_output`` and ``retall`` select the return shape, so inside ``@njit``
they must be compile-time literals.

--------------------------------------------------------------------------
DEVIATIONS from scipy:

  * A bare call returns an ndarray and ``full_output`` returns a plain
    tuple, as in scipy: 5, 6, 5 and 7 elements for ``fmin``,
    ``fmin_powell``, ``fmin_cg`` and ``fmin_bfgs``.  scipy's ``fmin*``
    functions return no ``OptimizeResult`` either; that belongs to
    ``minimize``.
  * ``allvecs`` under ``retall`` is a python list of 1-D arrays, as
    scipy's is.  From inside ``@njit`` it is a ``numba.typed.List`` of
    the same arrays.
  * ``args`` is a tuple, unpacked into the objective's argument list as
    scipy's is, and its elements may be of any type a compiled call
    accepts.  An ndarray or a list ``args`` arrives as ONE argument
    instead.  ``fmin`` and ``fmin_powell`` agree with scipy there, since
    scipy raises for both spellings; ``fmin_cg`` and ``fmin_bfgs``
    diverge, because scipy unpacks those two element by element and the
    arity of a compiled call is fixed when it compiles.  The routines
    behind ``minimize`` take the flat float64 buffer the ``double*`` of
    the Fortran packs imposes.
  * ``callback`` takes a plain Python function or an ``@njit`` function,
    and both halt on ``StopIteration``, as scipy's does.  The Python
    spelling is reached through a ``numba.objmode`` block over a
    module-level slot, so it takes the GIL per iteration and is not
    reentrant.  The compiled entry point takes the ``@njit`` spelling
    only, and any exception raised inside one halts the run.
  * ``disp`` IS implemented, at scipy's default, so a bare call prints
    scipy's summary and warns with scipy's class on a limit path.  Both
    take the GIL, so ``disp`` SERIALIZES a ``prange`` loop; ``disp=0`` is
    the parallel path.  Measured numbers are in the per-routine Notes.
  * NOT IMPLEMENTED, in Powell: the bounded path, an independent
    ``maxfun`` (it is pinned to ``N*1000``), and the final direction
    set.
  * DELIBERATELY DIFFERENT: cg / bfgs check that the gradient returns an
    array of ``len(x0)`` and raise ``ValueError`` otherwise.  numba has
    no bounds-checking.  scipy 1.18 covers part of the same ground and
    the coverage is uneven: cg raises on a long gradient and broadcasts
    a short one, bfgs raises on either unless the gradient is small
    enough for the norm test to halt first.  The per-routine Notes
    carry the measurements.

TOLERANCE: Nelder-Mead / Powell reach the true minimizer to ~xtol/ftol.
cg / bfgs run scipy's own line search (MINPACK ``DCSRCH`` first,
``scalar_search_wolfe2`` on failure) over scipy's ``ScalarFunction``
memoization, so ``xopt``, ``fopt``, ``nfev`` and ``njev`` reproduce scipy.
"""
import warnings

import numpy as np
from numba import njit, objmode, types
from numba.core.errors import TypingError
from numba.extending import overload
from numba.typed import List

__all__ = ['fmin', 'fmin_powell', 'fmin_cg', 'fmin_bfgs']


# ------------------------------------------------------------------ helpers

@njit
def _argsort_simplex(sim, fsim):
    """Sort simplex rows + fsim ascending by fsim (in place)."""
    n = fsim.shape[0]
    order = np.argsort(fsim)
    fsim_new = fsim[order].copy()
    sim_new = np.empty_like(sim)
    for k in range(n):
        sim_new[k, :] = sim[order[k], :]
    for k in range(n):
        fsim[k] = fsim_new[k]
        sim[k, :] = sim_new[k, :]


def _allvecs_list(av):
    """The rows of `av` as a sequence of 1-D arrays, scipy's `allvecs`.

    The four cores accumulate the iterates into one ``(nit + 1, N)`` buffer.
    scipy's `retall` returns a list of 1-D arrays instead, so the rows are
    split at the return.
    """
    return [av[i].copy() for i in range(av.shape[0])]


@overload(_allvecs_list)
def _allvecs_list_ovl(av):
    """`_allvecs_list` inside compiled code.

    A python `list` does not exist in compiled code; the compiled entry
    returns a ``numba.typed.List`` of the same 1-D arrays, indexable and
    iterable from both sides.
    """
    def impl(av):
        out = List.empty_list(types.float64[::1])
        for i in range(av.shape[0]):
            out.append(av[i].copy())
        return out
    return impl


# =================================================================== fmin
# Nelder-Mead simplex.  Port of scipy _minimize_neldermead.  `initial_simplex`
# IS supported; `adaptive` and `bounds` are not.  (This comment used to say
# "no initial_simplex", which was false: the argument reaches `_nm_core` as
# `sim0`/`use_sim` and reproduces scipy's iterate exactly.  Corrected
# 2026-08-02 after measuring max|diff| 0.000e+00 against scipy on the same
# simplex, with matching nit and nfev.)

from ._lbfgsb import _lit_bool, _is_none          # noqa: E402
from ._lsq import OptimizeWarning                 # noqa: E402
from ._callback import (_cb_noop, _cb_install, _cb_release,   # noqa: E402
                        _cb_resolve, _cb_resolve_ty,
                        _cb_halt_get, _cb_halt_clear)

# scipy resolves an omitted maxiter/maxfun to np.inf in one branch and to
# N*200 in another; an int64 sentinel stands in for inf inside @njit.
_NM_INF = np.int64(1) << 62


def _budget(v):
    """A `maxiter` or `maxfun` as the int64 the cores compare against.

    scipy tests `maxfun == np.inf` explicitly and otherwise compares against
    a python int, so an infinite budget is a legal input. `int(np.inf)`
    raises `OverflowError`, so the infinities take the sentinel instead of
    the cast.
    """
    if v == np.inf:
        return _NM_INF
    if v == -np.inf:
        return -_NM_INF
    return int(v)


@njit
def _budget_nb(v):
    """`_budget` inside compiled code.

    `np.int64(np.inf)` is an out-of-range `fptosi`, which is poison rather
    than an error, so the infinities are taken before the cast.
    """
    if v == np.inf:
        return _NM_INF
    if v == -np.inf:
        return -_NM_INF
    return np.int64(v)

# ======================================================= scipy's `disp` output
# scipy prints a summary when `disp` is set, and its default IS set.  The four
# routines here reproduce that text.  Two facts decide how.
#
# WHERE SCIPY PRINTS AND WHERE IT WARNS.  `fmin` prints only when it
# converged: on a limit it calls `warnings.warn(msg, RuntimeWarning)` and
# prints nothing at all.  `fmin_powell`, `fmin_cg` and `fmin_bfgs` route the
# STATUS line through `_print_success_message_or_warn`, which prints it on
# success and warns it on failure, and then print the counter lines either
# way.  Measured 2026-08-03 by capturing stdout from scipy 1.18.
#
# THE WARNING IS REPRODUCED, and it costs a GIL acquisition.  `warnings.warn`
# reaches compiled code only through a `numba.objmode` block, which takes the
# GIL, so a `disp` that fires inside a `prange` loop serializes it.  MEASURED
# at 32 threads over 1000 concurrent solves of the 2-D Rosenbrock function
# from (0.5, 0.5), one emission per solve, numba 0.66.0:
#
#                    silent    printing    print + warn
#     fmin          3.83 ms   198.15 ms        61.87 ms
#     fmin_powell   1.79 ms   175.77 ms       192.36 ms
#     fmin_cg       2.27 ms   217.99 ms       246.39 ms
#     fmin_bfgs    37.14 ms   225.84 ms       237.70 ms
#
# `fmin`'s third column is the LOWER one because its limit path warns and
# prints nothing, where the other three print the counters as well.  The
# printing ALREADY serializes, so for those three the warning adds to a cost
# that was there either way rather than introducing one.  `disp=0` is the
# parallel path and is what a prange sweep wants.
#
# FORMATTING.  numba has no `%` operator on strings and no f-string format
# spec, so `{fval:f}` is built by hand.
_DISP_SUCCESS = "Optimization terminated successfully."

#: scipy's `_status_message`, indexed by the warnflag each family reports.
#: Nelder-Mead and Powell: 1 maxfev, 2 maxiter, 3 nan.
_NM_STATUS = (_DISP_SUCCESS,
              "Maximum number of function evaluations has been exceeded.",
              "Maximum number of iterations has been exceeded.",
              "NaN result encountered.",
              "The result is outside of the provided bounds.")
#: CG and BFGS: 1 maxiter, 2 precision loss, 3 nan.
_GR_STATUS = (_DISP_SUCCESS,
              "Maximum number of iterations has been exceeded.",
              "Desired error not necessarily achieved due to precision loss.",
              "NaN result encountered.")


@njit
def _status_of(table, k):
    """`table[k]`, clamped, so an unmapped warnflag still gives a string."""
    if k < 0 or k >= len(table):
        return table[0]
    return table[k]


@njit
def _warn_status(msg, runtime_class):
    """scipy's warning on a limit path, with scipy's class.

    `runtime_class` selects ``RuntimeWarning``, which Nelder-Mead and Powell
    use, from ``OptimizeWarning``, which CG and BFGS use through
    ``_print_success_message_or_warn``'s default.
    """
    with objmode():
        if runtime_class:
            warnings.warn(msg, RuntimeWarning)
        else:
            warnings.warn(msg, OptimizeWarning)


@njit
def _disp_status(warnflag, msg, runtime_class):
    """scipy's `_print_success_message_or_warn`: print it or warn it."""
    if warnflag == 0:
        print(msg)
    else:
        _warn_status(msg, runtime_class)


@njit
def _fmt_f(v):
    """``%f``: six decimals, matching an f-string ``{v:f}``.

    Returns an EMPTY string for a value this spelling cannot carry -- a NaN,
    an infinity, or an integer part beyond ``int64``. `_print_fval` prints
    those through numba's own float printing, which is Python's ``repr``.
    Only the integer part is scaled, so the ``int64`` range is the limit
    rather than a millionth of it.
    """
    if np.isnan(v) or np.isinf(v):
        return ""
    neg = v < 0.0
    a = -v if neg else v
    if a >= 9.0e18:
        return ""
    w = np.floor(a)
    fi = np.int64((a - w) * 1e6 + 0.5)
    if fi >= 1000000:
        fi -= 1000000
        w += 1.0
    fs = str(fi)
    while len(fs) < 6:
        fs = "0" + fs
    out = str(np.int64(w)) + "." + fs
    return "-" + out if neg else out


@njit
def _print_fval(v):
    """scipy's ``print(f"         Current function value: {fval:f}")``."""
    s = _fmt_f(v)
    if len(s) == 0:
        print("         Current function value:", v)
    else:
        print("         Current function value: " + s)


@njit
def _disp_nm(warnflag, fval, nit, nfev):
    """`fmin`'s ``disp`` output.

    scipy prints the four-line summary only on success. On a limit it warns
    and prints NOTHING, not even the counters, which is where it differs from
    the other three.
    """
    if warnflag != 0:
        _warn_status(_status_of(_NM_STATUS, warnflag), True)
        return
    print(_DISP_SUCCESS)
    _print_fval(fval)
    print("         Iterations: " + str(nit))
    print("         Function evaluations: " + str(nfev))


@njit
def _disp_counters(warnflag, msg, fval, nit, nfev, runtime_class):
    """`fmin_powell`'s ``disp`` output: status line or warning, then the
    counters either way."""
    _disp_status(warnflag, msg, runtime_class)
    _print_fval(fval)
    print("         Iterations: " + str(nit))
    print("         Function evaluations: " + str(nfev))


@njit
def _disp_grad(warnflag, msg, fval, nit, nfev, njev):
    """`fmin_cg` and `fmin_bfgs`: `_disp_counters` plus the gradient count.

    Their warning class is ``OptimizeWarning``, scipy's default in
    ``_print_success_message_or_warn``, where Powell passes
    ``RuntimeWarning`` explicitly.
    """
    _disp_counters(warnflag, msg, fval, nit, nfev, False)
    print("         Gradient evaluations: " + str(njev))


@njit
def _nm_limits(n, maxiter, maxfun, mi_none, mf_none):
    """scipy's asymmetric default: `_minimize_neldermead` lines 353-364.

    Neither given -> both N*200.  One given -> the OTHER becomes inf, unless
    the given one IS inf, in which case the other takes the N*200 default so
    the loop cannot run unbounded.
    """
    if mi_none and mf_none:
        return n * 200, n * 200
    if mi_none:
        if maxfun >= _NM_INF:
            return n * 200, maxfun
        return _NM_INF, maxfun
    if mf_none:
        if maxiter >= _NM_INF:
            return maxiter, n * 200
        return maxiter, _NM_INF
    return maxiter, maxfun


@njit
def _nm_call(func, x, args, cnt, maxfun):
    """One objective evaluation, refused once `maxfun` is reached.

    scipy wraps the objective in `_wrap_scalar_function_maxfun_validation`,
    which RAISES `_MaxFuncCallError` BEFORE evaluating when the budget is
    spent; the caller catches it and abandons the iteration mid-way.  The
    boolean stands in for the exception.
    """
    if cnt[0] >= maxfun:
        return 0.0, False
    cnt[0] += 1
    if isinstance(args, tuple):
        return func(x, *args), True
    return func(x, args), True


@njit
def _nm_min(fsim):
    """`np.min`, which PROPAGATES NaN, where fsim[0] after an argsort does not."""
    m = fsim[0]
    for i in range(1, fsim.shape[0]):
        if np.isnan(fsim[i]) or fsim[i] < m:
            m = fsim[i]
    return m


@njit
def _nm_grow(buf, k):
    """Double `buf`'s row count when row k would not fit."""
    if k < buf.shape[0]:
        return buf
    out = np.empty((2 * buf.shape[0] + 1, buf.shape[1]), np.float64)
    for i in range(buf.shape[0]):
        for j in range(buf.shape[1]):
            out[i, j] = buf[i, j]
    return out


@njit
def _nm_core(func, x0, args, xtol, ftol, maxiter, maxfun, sim0, use_sim,
             retall, cb=_cb_noop, use_cb=False):
    """Nelder-Mead.

    Returns ``(x, fval, iterations, funcalls, warnflag, allvecs, sim,
    fsim)``.  The last two are the final simplex and its objective
    values, which scipy reports as ``final_simplex``.

    Transcribes `scipy.optimize._minimize_neldermead` along its default path
    (non-adaptive, no bounds), INCLUDING its control flow: an iteration
    abandoned because `maxfun` ran out does NOT increment the iteration
    counter, but the simplex is still re-sorted and `allvecs` still grows,
    because that path goes through `except _MaxFuncCallError: pass` and falls
    into the loop body below it.  The CONVERGENCE break leaves the loop before
    that point, so it records nothing.

    `allvecs` is (iterations, n) when `retall`, and (0, n) otherwise.
    """
    x0f = np.ascontiguousarray(np.asarray(x0)).astype(np.float64).ravel()
    if use_sim:
        if sim0.ndim != 2 or sim0.shape[0] != sim0.shape[1] + 1:
            raise ValueError(
                "`initial_simplex` should be an array of shape (N+1,N)")
        if x0f.size != sim0.shape[1]:
            raise ValueError(
                "Size of `initial_simplex` is not consistent with `x0`")
        n = sim0.shape[1]
        sim = np.ascontiguousarray(sim0).astype(np.float64).copy()
    else:
        n = x0f.size
        sim = np.empty((n + 1, n), np.float64)
        sim[0, :] = x0f
        for k in range(n):
            y = x0f.copy()
            if y[k] != 0.0:
                y[k] = 1.05 * y[k]          # (1 + nonzdelt) * y[k]
            else:
                y[k] = 0.00025             # zdelt
            sim[k + 1, :] = y

    nrows = 64 if retall else 0
    allvecs = np.empty((nrows, max(n, 1)), np.float64)
    nvec = 0
    if retall:
        allvecs = _nm_grow(allvecs, nvec)
        for j in range(n):
            allvecs[nvec, j] = sim[0, j]
        nvec += 1

    rho = 1.0
    chi = 2.0
    psi = 0.5
    sigma = 0.5

    fsim = np.full(n + 1, np.inf, np.float64)
    cnt = np.zeros(1, np.int64)

    for k in range(n + 1):                  # the initial simplex
        v, ok = _nm_call(func, sim[k, :], args, cnt, maxfun)
        if not ok:
            break
        fsim[k] = v
    _argsort_simplex(sim, fsim)

    iterations = 1
    xbar = np.empty(n, np.float64)

    # Drop whatever an earlier solve on this thread left in the halt
    # slot, so the first read below cannot see it.  A standalone entry
    # point does not clear it on the way out: only `minimize` takes it.
    _cb_halt_clear()

    while cnt[0] < maxfun and iterations < maxiter:
        do_break = False
        aborted = False

        # scipy's test is `np.max(np.abs(...)) <= tol` on both spreads, and
        # numpy's max PROPAGATES NaN: one NaN makes the whole max NaN, and
        # `nan <= tol` is False, so a NaN-valued objective never satisfies
        # the convergence test and the run ends at maxiter with warnflag 1.
        # A plain `if d > dmax` loop leaves dmax at 0.0 instead, because a
        # NaN is never greater than anything -- so it declared convergence
        # on the first pass and reported warnflag 0. Measured on
        # f(x) = nan for x[0] > 1 from (2, 2): scipy iter=100 nfev=400
        # warnflag=1, ours had iter=11 nfev=43 warnflag=0.
        dmax = 0.0
        for j in range(1, n + 1):
            for i in range(n):
                d = abs(sim[j, i] - sim[0, i])
                if np.isnan(d):
                    dmax = np.nan
                    break
                if d > dmax:
                    dmax = d
            if np.isnan(dmax):
                break
        fmax = 0.0
        for j in range(1, n + 1):
            d = abs(fsim[0] - fsim[j])
            if np.isnan(d):
                fmax = np.nan
                break
            if d > fmax:
                fmax = d
        if dmax <= xtol and fmax <= ftol:
            do_break = True

        if not do_break:
            for i in range(n):
                s = 0.0
                for j in range(n):
                    s += sim[j, i]
                xbar[i] = s / n

            xr = (1.0 + rho) * xbar - rho * sim[n, :]
            fxr, ok = _nm_call(func, xr, args, cnt, maxfun)
            if not ok:
                aborted = True
            doshrink = 0

            if not aborted and fxr < fsim[0]:
                xe = (1.0 + rho * chi) * xbar - rho * chi * sim[n, :]
                fxe, ok = _nm_call(func, xe, args, cnt, maxfun)
                if not ok:
                    aborted = True
                elif fxe < fxr:
                    sim[n, :] = xe
                    fsim[n] = fxe
                else:
                    sim[n, :] = xr
                    fsim[n] = fxr
            elif not aborted:
                if fxr < fsim[n - 1]:
                    sim[n, :] = xr
                    fsim[n] = fxr
                else:
                    if fxr < fsim[n]:
                        xc = (1.0 + psi * rho) * xbar - psi * rho * sim[n, :]
                        fxc, ok = _nm_call(func, xc, args, cnt, maxfun)
                        if not ok:
                            aborted = True
                        elif fxc <= fxr:
                            sim[n, :] = xc
                            fsim[n] = fxc
                        else:
                            doshrink = 1
                    else:
                        xcc = (1.0 - psi) * xbar + psi * sim[n, :]
                        fxcc, ok = _nm_call(func, xcc, args, cnt, maxfun)
                        if not ok:
                            aborted = True
                        elif fxcc < fsim[n]:
                            sim[n, :] = xcc
                            fsim[n] = fxcc
                        else:
                            doshrink = 1

                    if (not aborted) and doshrink == 1:
                        for j in range(1, n + 1):
                            sim[j, :] = sim[0, :] + sigma * (sim[j, :]
                                                             - sim[0, :])
                            v, ok = _nm_call(func, sim[j, :], args, cnt,
                                             maxfun)
                            if not ok:
                                aborted = True
                                break
                            fsim[j] = v
            if not aborted:
                iterations += 1

        # Runs on the convergence break and on an abort.
        _argsort_simplex(sim, fsim)
        # The convergence break does NOT record a point.  scipy's loop is
        # try/except (not try/finally), so `break` on the convergence test
        # leaves the loop before `allvecs.append(sim[0])`; the maxfun abort
        # goes through `except _MaxFuncCallError: pass` and DOES append.
        # Appending on the break too produced a trailing DUPLICATE of the
        # previous row -- measured 86 rows against scipy's 85 on rosenbrock
        # from (-1.2, 1.0), with rows 84 and 85 identical.  The simplex is
        # already sorted on that pass, so the sort above stays: it is a no-op
        # there and x is unaffected.
        if retall and not do_break:
            allvecs = _nm_grow(allvecs, nvec)
            for j in range(n):
                allvecs[nvec, j] = sim[0, j]
            nvec += 1
        if do_break:
            break
        # scipy calls the callback here, after the `retall` append and before
        # the loop's next test, and leaves the loop when it raises
        # `StopIteration`.  The convergence break above never reaches it,
        # because scipy's break sits inside the `try` and leaves the while
        # loop before the append.
        if use_cb:
            cb(sim[0, :].copy(), fsim[0])
            if _cb_halt_get():
                break

    x = sim[0, :].copy()
    fval = _nm_min(fsim)
    warnflag = 0
    if cnt[0] >= maxfun:
        warnflag = 1
    elif iterations >= maxiter:
        warnflag = 2
    #  `sim` and `fsim` are returned so a caller can report scipy's
    #  `final_simplex`; `fmin` ignores them, `minimize` reports them.
    return (x, fval, iterations, np.int64(cnt[0]), warnflag,
            allvecs[:nvec], sim, fsim)


def fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None,
         initial_simplex=None):
    """Minimize a function using the Nelder-Mead simplex algorithm.

    Derivative-free: only function values are used. Callable from Python and
    from inside ``@njit``; both entries run the same compiled core.

    Parameters
    ----------
    func : callable
        A plain ``@njit`` ``f(x, *args) -> value``. No ``@cfunc``, no
        ``.address``.
    x0 : array_like
        Initial guess. Any rank is flattened.
    args : tuple or ndarray, optional
        Extra parameters. A tuple is unpacked into separate arguments after
        `x`, so ``args=(a, b)`` calls ``func(x, a, b)``. Its elements may be
        of any type a compiled call accepts, arrays and strings included.
        Default ``()``, which calls ``func(x)``. An ndarray or a list
        arrives as ONE argument instead, ``func(x, args)``. See Notes.
    xtol, ftol : float, optional
        Absolute convergence tolerances on ``x`` and on ``f``. Both must be
        met. Defaults 1e-4.
    maxiter, maxfun : int or None, optional
        Iteration and evaluation budgets. ``None`` (default) means
        ``N*200`` for BOTH, but supplying only ONE leaves the OTHER
        unbounded rather than at the default.
    full_output : bool, optional
        ``False`` (default) returns ``xopt``. ``True`` returns
        ``(xopt, fopt, iter, funcalls, warnflag)``. Compile-time constant
        inside ``@njit``: it selects the return type.
    disp : int, optional
        ``1`` (default) prints a summary on success and raises a
        ``RuntimeWarning`` on a limit. ``0`` is silent, and is what a
        ``prange`` sweep wants: see Notes.
    retall : bool, optional
        Append ``allvecs`` to the return. Compile-time constant.
    callback : callable or None, optional
        Called once per iteration as ``callback(xk)``, with a copy of the
        current iterate. A plain Python function may instead be written
        ``callback(intermediate_result)`` and is handed an object carrying
        ``x`` and ``fun``. Raising ``StopIteration`` stops the run and
        returns the current best. From inside ``@njit`` the callback is an
        ``@njit`` ``callback(xk)``.
    initial_simplex : array_like or None, optional
        ``(N+1, N)``. Overrides the simplex built around ``x0``; ``x0`` is
        then used only for the length check.

    Returns
    -------
    xopt : ndarray
        The minimizer.
    fopt : float
        ``func(xopt)``. With ``full_output``.
    iter : int
        Iterations performed. With ``full_output``.
    funcalls : int
        Objective evaluations. With ``full_output``.
    warnflag : int
        ``0`` converged, ``1`` maxfun reached, ``2`` maxiter reached. NOTE
        that ``fmin_cg`` and ``fmin_bfgs`` use a DIFFERENT table, where ``1``
        means maxiter. With ``full_output``.
    allvecs : list of ndarray
        The best point after each iteration, ``iter`` entries of shape
        ``(N,)``, the last being `xopt`. With ``retall``. The first is `x0`
        unless an `initial_simplex` was given, in which case it is that
        simplex's first row. From inside
        ``@njit`` this is a ``numba.typed.List``.

    Raises
    ------
    ValueError
        If `callback` is neither ``None`` nor a callable nor an ``@njit``
        function, or `initial_simplex` has a shape other than ``(N + 1, N)``.

    See Also
    --------
    scipy.optimize.fmin : The scipy routine this mirrors.
    scijit.optimize.fmin_powell : Derivative-free, direction-set.
    scijit.optimize.fmin_bfgs : Uses a gradient, usually far fewer
        evaluations.
    scijit.optimize.minimize_scalar : One variable.

    Notes
    -----
    Under `retall`, ``allvecs`` is a python list of 1-D arrays, as scipy's
    is. From inside ``@njit`` it is a ``numba.typed.List`` of the same
    arrays.

    `args` is unpacked into the objective's argument list, ``func(x, *args)``,
    which is scipy's contract. An ndarray or a list `args` reaches `func` as
    ONE argument instead. scipy accepts neither spelling at this name, so the
    one-argument form is additive and no scipy-shaped call reaches it.

    `full_output` and `retall` must be compile-time literals inside ``@njit``:
    they select the return type.

    `callback` takes a plain Python function or an ``@njit`` function, and
    both halt on ``StopIteration``. The Python spelling reaches the compiled
    core through a ``numba.objmode`` block over a module-level slot: it takes
    the GIL once per iteration, and two concurrent solves that both install
    one overwrite each other, so a ``prange`` loop over it serializes and is
    not reentrant.

    Two differences from scipy follow from the compiled path. A plain Python
    callable cannot be an argument of a compiled function, so a call written
    inside ``@njit`` passes the ``@njit`` spelling and a Python one is refused
    by numba. And numba matches no exception class, ``except StopIteration``
    being a ``TypingError`` on 0.66, so any exception raised inside an
    ``@njit`` callback stops the run the way ``StopIteration`` does, where
    scipy propagates it. A Python callback keeps scipy's behaviour: only
    ``StopIteration`` halts, and anything else reaches the caller.

    ``disp`` reproduces scipy's summary and scipy's warning, at scipy's
    default, so a bare call writes to stdout and warns on a limit path. The
    warning takes the GIL through a ``numba.objmode`` block and the printing
    through Python's stdout, so with ``disp`` set a ``prange`` loop over this
    routine still gives the right answer but serializes. ``disp=0`` is the
    parallel path.

    Pure ``@njit``, so it is safe to call from a ``numba.prange`` loop with
    ``disp=0`` and with `callback` left at ``None`` or given as an ``@njit``
    function.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import fmin
    >>> @njit
    ... def rosen(x):
    ...     s = 0.0
    ...     for i in range(x.size - 1):
    ...         s += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    ...     return s
    >>> @njit
    ... def run():
    ...     return fmin(rosen, np.array([0.0, 0.0]), disp=0)
    >>> np.round(run(), 4)
    array([1., 1.])
    """
    cbf, use_cb, pycb = _cb_resolve('fmin', callback)
    a = args
    xf = np.ascontiguousarray(np.atleast_1d(
        np.asarray(x0, dtype=np.float64))).ravel()
    if initial_simplex is None:
        sim0, use_sim = np.zeros((0, 0)), False
    else:
        sim0 = np.ascontiguousarray(
            np.atleast_2d(np.asarray(initial_simplex, dtype=np.float64)))
        use_sim = True
    mi, mf = _nm_limits(xf.size if not use_sim else sim0.shape[1],
                        0 if maxiter is None else _budget(maxiter),
                        0 if maxfun is None else _budget(maxfun),
                        maxiter is None, maxfun is None)
    prev = _cb_install(pycb)
    try:
        x, fval, it, nfe, wf, av, _sim, _fsim = _nm_core(
            func, xf, a, xtol, ftol, mi, mf, sim0, use_sim, bool(retall),
            cbf, use_cb)
    finally:
        _cb_release(prev)
    if disp:
        _disp_nm(wf, fval, it, nfe)
    if full_output:
        if retall:
            return x, fval, int(it), int(nfe), int(wf), _allvecs_list(av)
        return x, fval, int(it), int(nfe), int(wf)
    if retall:
        return x, _allvecs_list(av)
    return x


@overload(fmin, prefer_literal=True)
def _fmin_ovl(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
              maxfun=None, full_output=0, disp=1, retall=0, callback=None,
              initial_simplex=None):
    """@njit implementation of `fmin`, resolved at compile time.

    ``full_output`` and ``retall`` between them select one of four return
    shapes, and whether ``initial_simplex`` was given decides the shape
    of the array handed to `_nm_core`, so all three have to be known
    while the body is typed. A ``maxiter`` or ``maxfun`` of ``None``
    becomes the sentinel 0 and its ABSENCE travels as a separate flag,
    because `_nm_limits` resolves the two asymmetrically and needs to
    know which was omitted. Returning ``None`` declines the call, which
    numba reports as a
    TypingError naming the argument that could not be served.
    """
    fo = _lit_bool(full_output)
    ra = _lit_bool(retall)
    if fo is None or ra is None:
        return None                     # runtime flag -> TypingError
    CBF, USE_CB = _cb_resolve_ty('fmin', callback)
    mi_none = _is_none(maxiter)
    mf_none = _is_none(maxfun)
    no_sim = _is_none(initial_simplex)

    def impl(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
             maxfun=None, full_output=0, disp=1, retall=0, callback=None,
             initial_simplex=None):
        a = args
        xf = np.ascontiguousarray(
            np.atleast_1d(np.asarray(x0))).ravel().astype(np.float64)
        if no_sim:
            sim0 = np.zeros((0, 0), np.float64)
            nn = xf.size
        else:
            sim0 = np.ascontiguousarray(
                np.atleast_2d(np.asarray(initial_simplex))
            ).astype(np.float64)
            nn = sim0.shape[1]
        if mi_none:
            mi_v = np.int64(0)
        else:
            mi_v = _budget_nb(maxiter)
        if mf_none:
            mf_v = np.int64(0)
        else:
            mf_v = _budget_nb(maxfun)
        mi, mf = _nm_limits(nn, mi_v, mf_v, mi_none, mf_none)
        x, fval, it, nfe, wf, av, _sim, _fsim = _nm_core(
            func, xf, a, xtol, ftol, mi, mf, sim0, not no_sim, ra,
            CBF, USE_CB)
        if disp:
            _disp_nm(wf, fval, it, nfe)
        if fo:
            if ra:
                return x, fval, it, nfe, wf, _allvecs_list(av)
            return x, fval, it, nfe, wf
        if ra:
            return x, _allvecs_list(av)
        return x
    return impl


# ================================================================= fmin_powell
# Powell direction-set method.  Port of scipy _minimize_powell +
# _linesearch_powell (unbounded path) + _minimize_scalar_brent.

@njit
def _phi_dir(func, p, xi, args, alpha, cnt, maxfun):
    """``phi(alpha) = func(p + alpha * xi)``, counted and budgeted.

    Powell's method minimizes along one direction at a time, and the
    bracketing and Brent routines below want a scalar function of
    ``alpha``. A numba closure cannot capture a runtime function value,
    so ``func``, ``p`` and ``xi`` travel as arguments instead, and the
    evaluation counter travels as a length-1 array because an ``int``
    would be passed by value.

    Returns ``(value, live)``. ``live`` is False once `maxfun` evaluations
    have been spent, and then the value is not a value. scipy refuses the
    call from the same place, the wrapper it puts around the objective, by
    raising its private ``_MaxFuncCallError`` and letting it unwind through
    the bracket, the Brent search and the line search into the sweep loop,
    which catches it. Nothing unwinds through four compiled frames, so the
    refusal travels as a flag and every frame between here and the sweep
    loop returns on it without using what it holds.
    """
    if cnt[0] >= maxfun:
        return 0.0, False
    cnt[0] += 1
    if isinstance(args, tuple):
        return func(p + alpha * xi, *args), True
    return func(p + alpha * xi, args), True


@njit
def _bracket_dir(func, p, xi, args, cnt, maxfun):
    """Downhill bracket of phi(a)=func(p+a*xi).  Returns
    (ok, xa, xb, xc, fa, fb, fc, live).

    ``live`` False means the `maxfun` budget ran out inside the bracket, and
    then every other element is discarded by the caller.
    """
    _gold = 1.618034
    _vs = 1e-21
    grow_limit = 110.0
    maxiter = 1000
    xa = 0.0
    xb = 1.0
    fa, live = _phi_dir(func, p, xi, args, xa, cnt, maxfun)
    if not live:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
    fb, live = _phi_dir(func, p, xi, args, xb, cnt, maxfun)
    if not live:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
    if fa < fb:
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = xb + _gold * (xb - xa)
    fc, live = _phi_dir(func, p, xi, args, xc, cnt, maxfun)
    if not live:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
    it = 0
    while fc < fb:
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if abs(val) < _vs:
            denom = 2.0 * _vs
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        if it > maxiter:
            raise RuntimeError(
                "No valid bracket was found before the iteration limit was "
                "reached. Consider trying different initial points or "
                "increasing `maxiter`.")
        it += 1
        if (w - xc) * (xb - w) > 0.0:
            fw, live = _phi_dir(func, p, xi, args, w, cnt, maxfun)
            if not live:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
            if fw < fc:
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif fw > fb:
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw, live = _phi_dir(func, p, xi, args, w, cnt, maxfun)
            if not live:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw, live = _phi_dir(func, p, xi, args, w, cnt, maxfun)
            if not live:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        elif (w - wlim) * (xc - w) > 0.0:
            fw, live = _phi_dir(func, p, xi, args, w, cnt, maxfun)
            if not live:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
            if fw < fc:
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw, live = _phi_dir(func, p, xi, args, w, cnt, maxfun)
                if not live:
                    return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        else:
            w = xc + _gold * (xc - xb)
            fw, live = _phi_dir(func, p, xi, args, w, cnt, maxfun)
            if not live:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw

    # scipy's three validity conditions.  cond3 is the only one that tests
    # finiteness, so it is what rejects a bracket that ran off to +-inf in x.
    cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    cond2 = (xa < xb < xc) or (xc < xb < xa)
    cond3 = np.isfinite(xa) and np.isfinite(xb) and np.isfinite(xc)
    ok = cond1 and cond2 and cond3
    return ok, xa, xb, xc, fa, fb, fc, True


@njit
def _brent_dir(func, p, xi, args, tol, cnt, maxfun):
    """Brent minimizer of phi(a)=func(p+a*xi).  Returns (amin, fmin, live).

    An invalid bracket is not an error here, and it is not "stay put"
    either. scipy intercepts its own ``BracketError`` in
    ``_recover_from_bracket_error`` and salvages the three bracket points:
    NaN anywhere among ``xa, xb, xc, fa, fb, fc`` gives ``(nan, nan)``,
    otherwise the best of the three by ``argmin(fs)``. Returning the
    incoming ``fval`` instead HID the NaN behind the last finite value --
    the same shape of failure as brute's argmin and fmin's convergence
    test: a wrong answer with nothing to signal it.

    ``live`` False means the `maxfun` budget ran out, and then ``amin`` and
    ``fmin`` are discarded by the caller.
    """
    ok, xa, xb, xc, fa, fb, fc, live = _bracket_dir(func, p, xi, args, cnt,
                                                    maxfun)
    if not live:
        return 0.0, 0.0, False
    if not ok:
        if (np.isnan(xa) or np.isnan(xb) or np.isnan(xc)
                or np.isnan(fa) or np.isnan(fb) or np.isnan(fc)):
            return np.nan, np.nan, True
        if fa <= fb and fa <= fc:
            return xa, fa, True
        if fb <= fc:
            return xb, fb, True
        return xc, fc, True
    _mintol = 1.0e-11
    _cg = 0.3819660
    maxiter = 500
    x = xb
    w = xb
    v = xb
    fw = fb
    fv = fb
    fx = fb
    if xa < xc:
        a = xa
        b = xc
    else:
        a = xc
        b = xa
    deltax = 0.0
    rat = 0.0
    it = 0
    while it < maxiter:
        tol1 = tol * abs(x) + _mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        if abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break
        if abs(deltax) <= tol1:
            if x >= xmid:
                deltax = a - x
            else:
                deltax = b - x
            rat = _cg * deltax
        else:
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            pp = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                pp = -pp
            tmp2 = abs(tmp2)
            dx_temp = deltax
            deltax = rat
            if ((pp > tmp2 * (a - x)) and (pp < tmp2 * (b - x))
                    and (abs(pp) < abs(0.5 * tmp2 * dx_temp))):
                rat = pp / tmp2
                u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    if xmid - x >= 0.0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if x >= xmid:
                    deltax = a - x
                else:
                    deltax = b - x
                rat = _cg * deltax
        if abs(rat) < tol1:
            if rat >= 0.0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu, live = _phi_dir(func, p, xi, args, u, cnt, maxfun)
        if not live:
            return 0.0, 0.0, False
        if fu > fx:
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        it += 1
    return x, fx, True


@njit
def _linesearch_powell(func, p, xi, args, tol, fval, cnt, maxfun):
    """Minimize func(p + a*xi) over a.  Returns
    (fret, p_new, xi_scaled, live).

    There is no failure branch: `_brent_dir` always returns a point, taking
    scipy's bracket-error recovery when the bracket is invalid. The move is
    then unconditional, as scipy's `xi = alpha_min * xi; return fret, p + xi,
    xi` is.

    ``live`` False means the `maxfun` budget ran out inside the search. The
    other three are then the values that came in, and the caller keeps the
    point it already had, which is what scipy keeps: its exception unwinds
    past the assignment ``fval, x, direc1 = _linesearch_powell(...)``, so
    the tuple never lands.
    """
    if not np.any(xi):
        return fval, p, xi, True
    alpha_min, fret, live = _brent_dir(func, p, xi, args, tol, cnt, maxfun)
    if not live:
        return fval, p, xi, False
    xi2 = alpha_min * xi
    return fret, p + xi2, xi2, True


@njit
def _powell_core(func, x0, args, xtol, ftol, maxiter, maxfun, direc0,
                 use_direc, retall, cb=_cb_noop, use_cb=False):
    """Powell's direction-set method: the whole algorithm.

    Returns ``(x, fval, direc, itn, nfev, warnflag, allvecs)``. Both
    `fmin_powell` entries take what their flags ask for and nothing else
    lives outside here, so the two cannot drift apart.

    `direc0` is the initial direction set, ``(N, N)``, used only when
    `use_direc` is set; otherwise the identity. `allvecs` is
    ``(itn + 1, N)`` under `retall` and ``(0, N)`` otherwise, the same
    convention `_nm_core` follows.
    """
    x = np.ascontiguousarray(np.asarray(x0)).ravel().astype(np.float64)
    N = x.shape[0]
    if N == 0:
        raise ValueError("fmin_powell: x0 must be non-empty")
    if use_direc:
        if direc0.shape[0] != N or direc0.shape[1] != N:
            raise ValueError("fmin_powell: direc must be (len(x0), len(x0))")
        direc = np.ascontiguousarray(direc0).astype(np.float64).copy()
        if np.linalg.matrix_rank(direc) != N:
            _warn_status("direc input is not full rank, some parameters may "
                         "not be optimized", False)
    else:
        direc = np.eye(N, dtype=np.float64)

    # The budget covers the first evaluation too, so a budget of zero or
    # less has nothing to spend on it.  scipy refuses that call rather than
    # returning: its wrapper raises `_MaxFuncCallError` from OUTSIDE the
    # try that its sweep loop catches on, so the whole run raises.  The
    # check sits here, after every other refusal, so that it preempts none
    # of them.
    if maxfun <= 0:
        raise ValueError(
            "fmin_powell: maxfun must be at least 1. The first evaluation "
            "is spent from the same budget, so there is nothing to compute "
            "a starting value with.")

    cnt = np.zeros(1, dtype=np.int64)

    nrows = 16 if retall else 0
    allvecs = np.empty((nrows, max(N, 1)), np.float64)
    nvec = 0
    if retall:
        allvecs = _nm_grow(allvecs, nvec)
        for j in range(N):
            allvecs[nvec, j] = x[j]
        nvec += 1

    if isinstance(args, tuple):
        fval = func(x, *args)
    else:
        fval = func(x, args)
    cnt[0] += 1
    x1 = x.copy()
    itn = 0
    warnflag = 0

    # Drop whatever an earlier solve on this thread left in the halt
    # slot, so the first read below cannot see it.  A standalone entry
    # point does not clear it on the way out: only `minimize` takes it.
    _cb_halt_clear()

    spent = False
    while True:
        fx = fval
        bigind = 0
        delta = 0.0
        for i in range(N):
            direc1 = direc[i, :].copy()
            fx2 = fval
            fr, xn, dn, live = _linesearch_powell(
                func, x, direc1, args, xtol * 100.0, fval, cnt, maxfun)
            if not live:
                spent = True
                break
            fval, x, direc1 = fr, xn, dn
            if (fx2 - fval) > delta:
                delta = fx2 - fval
                bigind = i
        # A sweep the budget cut short is not a sweep: scipy's exception
        # leaves the loop from inside the `for`, before `iter += 1`, before
        # the `retall` append and before the callback.
        if spent:
            break
        itn += 1
        if retall:
            allvecs = _nm_grow(allvecs, nvec)
            for j in range(N):
                allvecs[nvec, j] = x[j]
            nvec += 1

        # scipy's call site: after the `retall` append and before the
        # convergence test below it.
        if use_cb:
            cb(x.copy(), fval)
            if _cb_halt_get():
                break

        bnd = ftol * (abs(fx) + abs(fval)) + 1e-20
        if 2.0 * (fx - fval) <= bnd:
            break
        if cnt[0] >= maxfun:
            break
        if itn >= maxiter:
            break
        if np.isnan(fx) and np.isnan(fval):
            break

        # extrapolated direction.  The `cnt[0] >= maxfun` test above has
        # already left the loop when the budget is gone, so this evaluation
        # is always one the budget allows and needs no test of its own.
        # scipy's own order is the same.
        direc1 = x - x1
        x1 = x.copy()
        x2 = x + direc1
        if isinstance(args, tuple):
            fx2 = func(x2, *args)
        else:
            fx2 = func(x2, args)
        cnt[0] += 1

        if fx > fx2:
            t = 2.0 * (fx + fx2 - 2.0 * fval)
            temp = fx - fval - delta
            t *= temp * temp
            temp = fx - fx2
            t -= delta * temp * temp
            if t < 0.0:
                fr, xn, dn, live = _linesearch_powell(
                    func, x, direc1, args, xtol * 100.0, fval, cnt, maxfun)
                if not live:
                    break
                fval, x, direc1 = fr, xn, dn
                if np.any(direc1):
                    direc[bigind, :] = direc[N - 1, :]
                    direc[N - 1, :] = direc1

    # scipy derives warnflag AFTER the loop, from the FINAL state, in this
    # order. Setting it at the break instead reported 0 whenever the loop
    # left by the convergence test with a NaN in hand: the bail-out test is
    # `isnan(fx) and isnan(fval)`, which needs BOTH, while the flag test is
    # `isnan(fval) or isnan(x).any()`, which needs either. (scipy's warnflag
    # 4, out-of-bounds, belongs to the bounded path, which is not wrapped.)
    if cnt[0] >= maxfun:
        warnflag = 1
    elif itn >= maxiter:
        warnflag = 2
    elif np.isnan(fval) or np.isnan(x).any():
        warnflag = 3

    return x, fval, direc, itn, cnt[0], warnflag, allvecs[:nvec]


@njit
def _powell_limits(n, maxiter, maxfun, mi_none, mf_none):
    """scipy's `_minimize_powell` default rule, which is `_nm_limits` at
    ``N * 1000`` rather than ``N * 200``."""
    if mi_none and mf_none:
        return n * 1000, n * 1000
    if mi_none:
        if maxfun >= _NM_INF:
            return n * 1000, maxfun
        return _NM_INF, maxfun
    if mf_none:
        if maxiter >= _NM_INF:
            return maxiter, n * 1000
        return maxiter, _NM_INF
    return maxiter, maxfun


def fmin_powell(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
                maxfun=None, full_output=0, disp=1, retall=0, callback=None,
                direc=None):
    """Minimize a function using Powell's direction-set method.

    Derivative-free. Minimizes along a set of directions that is updated each
    sweep, using a scalar Brent line search.

    Callable from Python and from inside ``@njit``. Both entries run the same
    compiled core, so they cannot disagree.

    Parameters
    ----------
    func : callable
        A plain ``@njit`` ``func(x, *args) -> float``.
    x0 : array_like
        Initial guess. Any rank is flattened.
    args : tuple or ndarray, optional
        Extra parameters. A tuple is unpacked into separate arguments after
        `x`, so ``args=(a, b)`` calls ``func(x, a, b)``. Its elements may be
        of any type a compiled call accepts, arrays and strings included.
        Default ``()``, which calls ``func(x)``. An ndarray or a list
        arrives as ONE argument instead, ``func(x, args)``. See Notes.
    xtol : float, optional
        Line-search tolerance. Default 1e-4. The inner Brent search runs at
        ``xtol * 100``.
    ftol : float, optional
        Relative convergence tolerance on `f` between sweeps. Default 1e-4.
    maxiter, maxfun : int, float or None, optional
        Sweep and evaluation budgets. ``None`` (default) means ``N*1000``
        for BOTH, but supplying only ONE leaves the OTHER unbounded rather
        than at the default. ``np.inf`` is accepted and means unbounded.
    full_output : bool, optional
        ``False`` (default) returns `xopt` alone. ``True`` returns the
        6-tuple. Compile-time constant inside ``@njit``: it selects the
        return type.
    disp : int, optional
        ``1`` (default) prints a summary. The status line appears only on
        success and becomes a ``RuntimeWarning`` otherwise; the counters
        print either way. ``0`` is silent, and is what a ``prange`` sweep
        wants: see Notes.
    retall : bool, optional
        Append ``allvecs`` to the return. Compile-time constant.
    callback : callable or None, optional
        Called once per iteration as ``callback(xk)``, with a copy of the
        current iterate. A plain Python function may instead be written
        ``callback(intermediate_result)`` and is handed an object carrying
        ``x`` and ``fun``. Raising ``StopIteration`` stops the run and
        returns the current best. From inside ``@njit`` the callback is an
        ``@njit`` ``callback(xk)``.
    direc : array_like or None, optional
        ``(N, N)`` initial direction set. ``None`` (default) is the identity.

    Returns
    -------
    xopt : ndarray
        The minimizer.
    fopt : float
        ``func(xopt, args)``. With ``full_output``.
    direc : ndarray, shape (N, N)
        The final direction set. With ``full_output``.
    iter : int
        Sweeps performed. With ``full_output``.
    funcalls : int
        Objective evaluations used. With ``full_output``.
    warnflag : int
        ``0`` converged, ``1`` `maxfun` reached, ``2`` `maxiter` reached,
        ``3`` the search entered a NaN region. With ``full_output``.
    allvecs : list of ndarray
        The best point after each sweep, ``iter + 1`` entries of shape
        ``(N,)``, the first being `x0`. With ``retall``. From inside
        ``@njit`` this is a ``numba.typed.List`` of the same arrays.

    Raises
    ------
    ValueError
        Empty `x0`, a `direc` whose shape is not ``(N, N)``, a `maxfun` at or
        below zero, or a `callback` that is neither a callable nor an
        ``@njit`` function. A `maxiter` of ``0`` or below does NOT raise:
        one sweep runs and ``warnflag`` is 2.

    Warns
    -----
    OptimizeWarning
        A `direc` that is not full rank.

    See Also
    --------
    scipy.optimize.fmin_powell : The scipy routine this mirrors.
    scijit.optimize.fmin : Derivative-free, simplex.
    scijit.optimize.fmin_bfgs : Uses a gradient.
    scijit.optimize.minimize : ``method='Powell'`` reaches the same core.

    Notes
    -----
    Covers scipy's ``_minimize_powell`` along its unbounded path only, where
    ``bounds=None``. ``warnflag`` 4, the out-of-bounds code, therefore never
    appears.

    `full_output` and `retall` must be compile-time literals inside ``@njit``:
    they select the return type.

    `callback` takes a plain Python function or an ``@njit`` function, and
    both halt on ``StopIteration``. The Python spelling reaches the compiled
    core through a ``numba.objmode`` block over a module-level slot: it takes
    the GIL once per iteration, and two concurrent solves that both install
    one overwrite each other, so a ``prange`` loop over it serializes and is
    not reentrant.

    Two differences from scipy follow from the compiled path. A plain Python
    callable cannot be an argument of a compiled function, so a call written
    inside ``@njit`` passes the ``@njit`` spelling and a Python one is refused
    by numba. And numba matches no exception class, ``except StopIteration``
    being a ``TypingError`` on 0.66, so any exception raised inside an
    ``@njit`` callback stops the run the way ``StopIteration`` does, where
    scipy propagates it. A Python callback keeps scipy's behaviour: only
    ``StopIteration`` halts, and anything else reaches the caller.

    A `direc` whose shape is not ``(N, N)`` raises `ValueError` here. scipy
    does not check the shape: a short one raises `IndexError` from inside the
    sweep and a wide one runs with the extra columns ignored.

    An empty `x0` raises `ValueError` here, and so does a `maxfun` at or
    below zero. scipy refuses both too, through its private
    ``_MaxFuncCallError``, which subclasses ``RuntimeError`` and is not
    importable; the empty `x0` reaches it because the default `maxfun` of
    ``N*1000`` is zero when ``N`` is zero.

    ``disp`` reproduces scipy's summary and scipy's warning, at scipy's
    default, so a bare call writes to stdout and warns on a limit path. The
    warning takes the GIL through a ``numba.objmode`` block and the printing
    through Python's stdout, so with ``disp`` set a ``prange`` loop over this
    routine still gives the right answer but serializes. ``disp=0`` is the
    parallel path.

    Under `retall`, ``allvecs`` is a python list of 1-D arrays, as scipy's
    is. From inside ``@njit`` it is a ``numba.typed.List`` of the same
    arrays.

    `args` is unpacked into the objective's argument list, ``func(x, *args)``,
    which is scipy's contract. An ndarray or a list `args` reaches `func` as
    ONE argument instead. scipy accepts neither spelling at this name, so the
    one-argument form is additive and no scipy-shaped call reaches it.

    Pure ``@njit``, so it is safe to call from a ``numba.prange`` loop with
    ``disp=0`` and with `callback` left at ``None`` or given as an ``@njit``
    function.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_powell.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import fmin_powell
    >>> @njit
    ... def rosen(x):
    ...     s = 0.0
    ...     for i in range(x.size - 1):
    ...         s += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    ...     return s
    >>> @njit
    ... def run():
    ...     return fmin_powell(rosen, np.array([0.0, 0.0]), (), disp=0)
    >>> np.round(run(), 8)
    array([1., 1.])
    """
    cbf, use_cb, pycb = _cb_resolve('fmin_powell', callback)
    a = args
    xf = np.ascontiguousarray(np.atleast_1d(
        np.asarray(x0, dtype=np.float64))).ravel()
    mi_none = maxiter is None
    mf_none = maxfun is None
    mi, mf = _powell_limits(xf.size,
                            0 if mi_none else _budget(maxiter),
                            0 if mf_none else _budget(maxfun),
                            mi_none, mf_none)
    if direc is None:
        d0, ud = np.zeros((0, 0)), False
    else:
        d0 = np.ascontiguousarray(np.asarray(direc, dtype=np.float64))
        ud = True
    prev = _cb_install(pycb)
    try:
        r = _powell_core(func, xf, a, xtol, ftol, mi, mf, d0, ud,
                         bool(retall), cbf, use_cb)
    finally:
        _cb_release(prev)
    if disp:
        _disp_counters(r[5], _status_of(_NM_STATUS, r[5]), r[1], r[3],
                       r[4], True)
    if full_output:
        out = (r[0], r[1], r[2], int(r[3]), int(r[4]), int(r[5]))
        if retall:
            return out + (_allvecs_list(r[6]),)
        return out
    if retall:
        return r[0], _allvecs_list(r[6])
    return r[0]


@overload(fmin_powell, prefer_literal=True)
def _fmin_powell_ovl(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
                     maxfun=None, full_output=0, disp=1, retall=0,
                     callback=None, direc=None):
    """@njit implementation of `fmin_powell`, resolved at compile time.

    ``full_output`` and ``retall`` between them select one of four return
    shapes, and whether ``direc`` was given decides the shape of the array
    handed to `_powell_core`, so all three have to be known while the body
    is typed. A ``maxiter`` or ``maxfun`` of ``None`` becomes the sentinel
    0 and its ABSENCE travels as a separate flag, because `_powell_limits`
    resolves the two asymmetrically and needs to know which was omitted.
    Returning ``None`` declines the call, which numba reports as a
    TypingError naming the argument that could not be served.
    """
    fo = _lit_bool(full_output)
    ra = _lit_bool(retall)
    if fo is None or ra is None:
        return None                     # runtime flag -> TypingError
    CBF, USE_CB = _cb_resolve_ty('fmin_powell', callback)
    mi_none = _is_none(maxiter)
    mf_none = _is_none(maxfun)
    no_direc = _is_none(direc)

    def impl(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None,
             maxfun=None, full_output=0, disp=1, retall=0, callback=None,
             direc=None):
        a = args
        xf = np.ascontiguousarray(
            np.atleast_1d(np.asarray(x0))).ravel().astype(np.float64)
        if no_direc:
            d0 = np.zeros((0, 0), np.float64)
        else:
            d0 = np.ascontiguousarray(np.asarray(direc)).astype(np.float64)
        if mi_none:
            mi_v = np.int64(0)
        else:
            mi_v = _budget_nb(maxiter)
        if mf_none:
            mf_v = np.int64(0)
        else:
            mf_v = _budget_nb(maxfun)
        mi, mf = _powell_limits(xf.size, mi_v, mf_v, mi_none, mf_none)
        r = _powell_core(func, xf, a, xtol, ftol, mi, mf, d0, not no_direc,
                         ra, CBF, USE_CB)
        if disp:
            _disp_counters(r[5], _status_of(_NM_STATUS, r[5]), r[1], r[3],
                       r[4], True)
        if fo:
            if ra:
                return (r[0], r[1], r[2], r[3], r[4], r[5],
                        _allvecs_list(r[6]))
            return r[0], r[1], r[2], r[3], r[4], r[5]
        if ra:
            return r[0], _allvecs_list(r[6])
        return r[0]
    return impl
# ================================================== scipy's ScalarFunction
# The memoize-and-count layer `fmin_cg` and `fmin_bfgs` share.  See
# `_sf_fun` and `_sf_grad`.

_EPS_FD = 1.4901161193847656e-08        # scipy `_epsilon`, sqrt(finfo.eps)


def _eps_vec(epsilon, n):
    """`epsilon` as a length-`n` forward-difference step vector.

    scipy hands `epsilon` to `approx_derivative` as `abs_step`, which is
    documented as "float or ndarray" and broadcast against `x`. Measured on
    scipy 1.18: a length-`n` array gives a per-element step, a length-1
    array broadcasts, and any other length raises `ValueError` out of the
    broadcast.
    """
    e = np.asarray(epsilon, dtype=np.float64).ravel()
    if e.size == 1:
        return np.full(n, float(e[0]))
    if e.size != n:
        raise ValueError("epsilon must be a scalar or have length len(x0)")
    return np.ascontiguousarray(e)


@overload(_eps_vec)
def _eps_vec_ovl(epsilon, n):
    """`_eps_vec` inside compiled code.

    The scalar and the array take different bodies because the two cannot
    be typed by one expression; which one applies is known while the call
    is typed.
    """
    if isinstance(epsilon, types.Array):
        def impl(epsilon, n):
            e = np.ascontiguousarray(
                np.asarray(epsilon)).ravel().astype(np.float64)
            if e.size == 1:
                return np.full(n, e[0])
            if e.size != n:
                raise ValueError(
                    "epsilon must be a scalar or have length len(x0)")
            return e
        return impl

    def impl(epsilon, n):
        return np.full(n, np.float64(epsilon))
    return impl


@njit
def _no_grad(x, *args):
    """Type-correct stand-in for ``fprime=None``.  Never called."""
    return np.zeros(x.shape[0], np.float64)


@njit
def _vecnorm(x, ordv):
    """scipy `_optimize.vecnorm`.  ``inf`` max|x|, ``-inf`` min|x|, else p-norm.

    The three reductions are `np.amax`, `np.amin` and `np.sum`, and all three
    PROPAGATE a NaN: one NaN anywhere makes the whole norm NaN.  A plain
    ``if a > m`` loop drops it instead, because a NaN compares False against
    everything, and then `fmin_cg`'s and `fmin_bfgs`' ``while gnorm > gtol``
    keeps running on a corrupted iterate until the iteration cap.  Same shape
    as the Nelder-Mead convergence test above.

    A zero-length `x` reduces to ``0.0`` for a finite `ordv` and raises for
    ``+-inf``, where the message names maximum or minimum after the reduction
    that has no identity.
    """
    n = x.shape[0]
    if ordv == np.inf:
        if n == 0:
            raise ValueError(
                "zero-size array to reduction operation maximum which has no "
                "identity")
        m = abs(x[0])
        for i in range(1, n):
            a = abs(x[i])
            if np.isnan(a) or a > m:
                m = a
        return m
    if ordv == -np.inf:
        if n == 0:
            raise ValueError(
                "zero-size array to reduction operation minimum which has no "
                "identity")
        m = abs(x[0])
        for i in range(1, n):
            a = abs(x[i])
            if np.isnan(a) or a < m:
                m = a
        return m
    s = 0.0
    for i in range(n):
        s += abs(x[i]) ** ordv
    return s ** (1.0 / ordv)


@njit(error_model='numpy')
def _fd_grad(func, x, args, f0, eps):
    """Forward differences, `approx_derivative(method='2-point', abs_step=eps)`.

    The denominator is the REALIZED step ``(x+h) - x``, not ``h``.  A step
    that rounds away entirely falls back to ``sqrt(eps)*sign(x)*max(1,|x|)``.

    `eps` is a length-`n` step vector, which is `abs_step` broadcast against
    `x` as scipy broadcasts it.
    """
    n = x.shape[0]
    g = np.empty(n, np.float64)
    x1 = x.copy()
    for i in range(n):
        h = eps[i]
        if (x[i] + h) - x[i] == 0.0:
            if x[i] >= 0.0:
                sgn = 1.0
            else:
                sgn = -1.0
            a = abs(x[i])
            if a < 1.0:
                a = 1.0
            h = _EPS_FD * sgn * a
        x1[i] = x[i] + h
        dx = x1[i] - x[i]
        if isinstance(args, tuple):
            g[i] = (func(x1, *args) - f0) / dx
        else:
            g[i] = (func(x1, args) - f0) / dx
        x1[i] = x[i]
    return g


@njit
def _sf_setx(SX, SFL, x):
    """`ScalarFunction._update_x`, guarded by `np.array_equal`."""
    n = x.shape[0]
    same = True
    for i in range(n):
        if not (SX[i] == x[i]):
            same = False
            break
    if not same:
        for i in range(n):
            SX[i] = x[i]
        SFL[0] = 0
        SFL[1] = 0


@njit
def _sf_fun(func, args, SX, SF, SG, SFL, SC, x):
    """`ScalarFunction.fun`: ``f`` at ``x``, memoized and counted.

    `fmin_cg` and `fmin_bfgs` reach the objective through scipy's
    ``_prepare_scalar_function``, which caches ``f`` and ``g`` at one
    point and counts every evaluation; reproducing that cache is what
    makes ``nfev`` readable at all. The state that scipy keeps in
    instance attributes lives in the caller's ``SX`` (the cached point),
    ``SF``/``SG`` (the values), ``SFL`` (validity flags) and ``SC``
    (counters) arrays, since numba has nowhere else to put it.
    """
    _sf_setx(SX, SFL, x)
    if SFL[0] == 0:
        if isinstance(args, tuple):
            SF[0] = func(SX, *args)
        else:
            SF[0] = func(SX, args)
        SC[0] += 1
        SFL[0] = 1
    return SF[0]


@njit
def _sf_grad(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC, x):
    """`ScalarFunction.grad`: gradient at ``x``, memoized and counted.

    The counting is the reason this is not just a call to ``grad``: under
    ``use_fd`` a gradient costs ``n`` further objective calls where ``f``
    is already cached at ``x`` and ``n + 1`` where it is not, and that
    split is what makes ``nfev`` line up. The length check on an analytic
    gradient is not redundant either -- numba does not bounds-check, so a
    short return would be read past its end into ``SG`` and the search
    would converge to a wrong point with no error raised.

    Returns a COPY, because the caller holds the vector across further
    evaluations that would otherwise overwrite ``SG``.
    """
    _sf_setx(SX, SFL, x)
    if SFL[1] == 0:
        if use_fd:
            if SFL[0] == 0:
                if isinstance(args, tuple):
                    SF[0] = func(SX, *args)
                else:
                    SF[0] = func(SX, args)
                SC[0] += 1
                SFL[0] = 1
            gg = _fd_grad(func, SX, args, SF[0], eps)
            SC[0] += SX.shape[0]
        else:
            if isinstance(args, tuple):
                gg = grad(SX, *args)
            else:
                gg = grad(SX, args)
            if gg.shape[0] != SX.shape[0]:
                raise ValueError(
                    "the gradient must return an array of len(x0)")
        # element-wise store, so a non-float64 gradient converts without the
        # extra whole-array copy `.astype` would make
        for i in range(SX.shape[0]):
            SG[i] = gg[i]
        SC[1] += 1
        SFL[1] = 1
    return SG.copy()


# ============================================ MINPACK DCSRCH (wolfe1 search)
# Port of `scipy.optimize._dcsrch`.  `task` is an int code:
# 0 START, 1 FG, 2 CONVERGENCE, 3 WARNING, 4 ERROR.
# State travels in two arrays instead of instance attributes:
#   D[0] ginit  D[1] gtest  D[2] gx  D[3] gy  D[4] finit  D[5] fx  D[6] fy
#   D[7] stx    D[8] sty    D[9] stmin  D[10] stmax  D[11] width  D[12] width1
#   Di[0] stage  Di[1] brackt

_T_START = 0
_T_FG = 1
_T_CONV = 2
_T_WARN = 3
_T_ERR = 4


@njit
def _check_hess_inv0(H, n):
    """`_optimize._check_positive_definite`, plus the (N, N) shape check."""
    if H.ndim != 2 or H.shape[0] != n or H.shape[1] != n:
        raise ValueError("'hess_inv0' must have shape (N, N)")
    ok = True
    for i in range(n):
        for j in range(n):
            if H[i, j] != H[j, i]:
                ok = False
    if ok:
        L = np.zeros((n, n), np.float64)
        for i in range(n):
            for j in range(i + 1):
                s = H[i, j]
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                if i == j:
                    if s <= 0.0:
                        ok = False
                        break
                    L[i, i] = np.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]
            if not ok:
                break
    if not ok:
        raise ValueError("'hess_inv0' matrix isn't positive definite.")


@njit
def _sgn(v):
    """Sign of ``v``: ``+1.0``, ``-1.0``, ``0.0``, or NaN for a NaN input.

    `_dcstep` needs the quantity scipy writes as ``dx / abs(dx)``, which
    is NaN whenever ``dx`` is, so NaN is kept distinct here rather than
    folded into the zero case. Not the same function as `_scalar._sgn`,
    which returns ``0.0`` for NaN because its callers multiply two signs
    to test for a bracket; the two must not be merged.
    """
    if v > 0.0:
        return 1.0
    if v < 0.0:
        return -1.0
    if v == 0.0:
        return 0.0
    return np.nan


@njit
def _max3(a, b, c):
    """Largest of three values, ``a`` first.

    `_dcstep` scales its cubic interpolant by
    ``max(|theta|, |dx|, |dp|)`` on every branch. NaN is not handled: the
    ``>`` tests skip a NaN in ``b`` or ``c``, while a NaN in ``a`` is
    returned, which is the ordering scipy's ``max`` of the same three
    also produces.
    """
    m = a
    if b > m:
        m = b
    if c > m:
        m = c
    return m


@njit(error_model='numpy')
def _dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """`_dcsrch.dcstep`.  Returns (stx, fx, dx, sty, fy, dy, stp, brackt)."""
    sgnd = _sgn(dp) * _sgn(dx)
    stpf = stp

    if fp > fx:
        # A higher function value.  The minimum is bracketed.
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = _max3(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        if abs(stpc - stx) <= abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        # A lower function value and derivatives of opposite sign.
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = _max3(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        # A lower function value, same-sign derivatives, |derivative| falling.
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = _max3(abs(theta), abs(dx), abs(dp))
        rad = (theta / s) ** 2 - (dx / s) * (dp / s)
        if not (rad > 0.0):
            rad = 0.0
        gamma = s * np.sqrt(rad)
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if r < 0.0 and gamma != 0.0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                b = stp + 0.66 * (sty - stp)
                if b < stpf:
                    stpf = b
            else:
                b = stp + 0.66 * (sty - stp)
                if b > stpf:
                    stpf = b
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stpf < stpmin:
                stpf = stpmin
            elif stpf > stpmax:
                stpf = stpmax
    else:
        # A lower function value, same-sign derivatives, |derivative| flat.
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = _max3(abs(theta), abs(dy), abs(dp))
            gamma = s * np.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpf = stp + r * (sty - stp)
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0.0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    return stx, fx, dx, sty, fy, dy, stpf, brackt


@njit(error_model='numpy')
def _dcsrch_iterate(stp, f, g, task, ftol, gtol, xtol, stpmin, stpmax, D, Di):
    """`DCSRCH._iterate`.  Returns (stp, f, g, task)."""
    p5 = 0.5
    p66 = 0.66
    xtrapl = 1.1
    xtrapu = 4.0

    if task == _T_START:
        if stp < stpmin or stp > stpmax or g >= 0.0 or ftol < 0.0 \
                or gtol < 0.0 or xtol < 0.0 or stpmin < 0.0 \
                or stpmax < stpmin:
            return stp, f, g, _T_ERR

        Di[1] = 0                       # brackt
        Di[0] = 1                       # stage
        D[4] = f                        # finit
        D[0] = g                        # ginit
        D[1] = ftol * D[0]              # gtest
        D[11] = stpmax - stpmin         # width
        D[12] = D[11] / p5              # width1
        D[7] = 0.0                      # stx
        D[5] = D[4]                     # fx
        D[2] = D[0]                     # gx
        D[8] = 0.0                      # sty
        D[6] = D[4]                     # fy
        D[3] = D[0]                     # gy
        D[9] = 0.0                      # stmin
        D[10] = stp + xtrapu * stp      # stmax
        return stp, f, g, _T_FG

    brackt = Di[1] != 0
    ftest = D[4] + stp * D[1]

    if Di[0] == 1 and f <= ftest and g >= 0.0:
        Di[0] = 2

    if brackt and (stp <= D[9] or stp >= D[10]):
        task = _T_WARN
    if brackt and D[10] - D[9] <= xtol * D[10]:
        task = _T_WARN
    if stp == stpmax and f <= ftest and g <= D[1]:
        task = _T_WARN
    if stp == stpmin and (f > ftest or g >= D[1]):
        task = _T_WARN

    if f <= ftest and abs(g) <= gtol * -D[0]:
        task = _T_CONV

    if task == _T_WARN or task == _T_CONV:
        return stp, f, g, task

    if Di[0] == 1 and f <= D[5] and f > ftest:
        fm = f - stp * D[1]
        fxm = D[5] - D[7] * D[1]
        fym = D[6] - D[8] * D[1]
        gm = g - D[1]
        gxm = D[2] - D[1]
        gym = D[3] - D[1]
        (D[7], fxm, gxm, D[8], fym, gym, stp, brackt) = _dcstep(
            D[7], fxm, gxm, D[8], fym, gym, stp, fm, gm, brackt, D[9], D[10])
        D[5] = fxm + D[7] * D[1]
        D[6] = fym + D[8] * D[1]
        D[2] = gxm + D[1]
        D[3] = gym + D[1]
    else:
        (D[7], D[5], D[2], D[8], D[6], D[3], stp, brackt) = _dcstep(
            D[7], D[5], D[2], D[8], D[6], D[3], stp, f, g, brackt, D[9], D[10])

    if brackt:
        if abs(D[8] - D[7]) >= p66 * D[12]:
            stp = D[7] + p5 * (D[8] - D[7])
        D[12] = D[11]
        D[11] = abs(D[8] - D[7])

    if brackt:
        D[9] = min(D[7], D[8])
        D[10] = max(D[7], D[8])
    else:
        D[9] = stp + xtrapl * (stp - D[7])
        D[10] = stp + xtrapu * (stp - D[7])

    if stp < stpmin:
        stp = stpmin
    elif stp > stpmax:
        stp = stpmax

    if (brackt and (stp <= D[9] or stp >= D[10])) or \
            (brackt and D[10] - D[9] <= xtol * D[10]):
        stp = D[7]

    Di[1] = 1 if brackt else 0
    return stp, f, g, _T_FG


# =================================================== Wolfe line searches
# `_line_search_wolfe12`: DCSRCH first, `scalar_search_wolfe2` on failure.

@njit(error_model='numpy')
def _prp_step(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
              xk, pk, gfk, deltak, ordv, alpha, gfkp1, have_g,
              cx, cp, cg, cs, cv):
    """`_minimize_cg.polak_ribiere_powell_step`, written into the cache."""
    n = xk.shape[0]
    xkp1 = xk + alpha * pk
    if have_g:
        g1 = gfkp1
    else:
        g1 = _sf_grad(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC, xkp1)
    yk = g1 - gfk
    beta = np.dot(yk, g1) / deltak
    if not (beta > 0.0):            # `max(0, ...)`; NaN goes to 0, as in python
        beta = 0.0
    pkp1 = -g1 + beta * pk
    gn = _vecnorm(g1, ordv)
    for i in range(n):
        cx[i] = xkp1[i]
        cp[i] = pkp1[i]
        cg[i] = g1[i]
    cs[0] = alpha
    cs[1] = gn
    cv[0] = 1
    return gn


@njit(error_model='numpy')
def _cg_descent_ok(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
                   xk, pk, gfk, deltak, ordv, gtol, sigma_3,
                   alpha, gfkp1, have_g, cx, cp, cg, cs, cv):
    """`_minimize_cg.descent_condition`.  Fills the cached step either way."""
    n = xk.shape[0]
    gn = _prp_step(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
                   xk, pk, gfk, deltak, ordv, alpha, gfkp1, have_g,
                   cx, cp, cg, cs, cv)
    if gn <= gtol:
        return True
    d1 = 0.0
    d2 = 0.0
    for i in range(n):
        d1 += cp[i] * cg[i]
        d2 += cg[i] * cg[i]
    return d1 <= -sigma_3 * d2


@njit(error_model='numpy')
def _wolfe1(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
            xk, pk, gfk, phi0, old_phi0, derphi0, c1, c2, amin, amax, xtol):
    """`line_search_wolfe1`.  Returns (found, stp, phi1, gval)."""
    n = xk.shape[0]
    if derphi0 != 0.0:
        alpha1 = 1.01 * 2.0 * (phi0 - old_phi0) / derphi0
        # scipy writes `min(1.0, ...)`, which KEEPS 1.0 unless the operand is
        # strictly smaller, so a NaN operand gives 1.0.  `if alpha1 > 1.0`
        # leaves the NaN in place and the whole line search then steps to a
        # NaN iterate.
        if not (alpha1 < 1.0):
            alpha1 = 1.0
        if alpha1 < 0.0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    D = np.zeros(13, np.float64)
    Di = np.zeros(2, np.int64)
    gval = gfk.copy()
    phi1 = phi0
    derphi1 = derphi0
    task = _T_START
    stp = alpha1
    found = False

    for _ in range(100):
        stp, phi1, derphi1, task = _dcsrch_iterate(
            stp, phi1, derphi1, task, c1, c2, xtol, amin, amax, D, Di)
        if not np.isfinite(stp):
            task = _T_WARN
            break
        if task == _T_FG:
            alpha1 = stp
            xa = xk + stp * pk
            phi1 = _sf_fun(func, args, SX, SF, SG, SFL, SC, xa)
            gval = _sf_grad(func, grad, use_fd, eps, args,
                            SX, SF, SG, SFL, SC, xa)
            derphi1 = np.dot(gval, pk)
        else:
            break
    else:
        task = _T_WARN

    if task == _T_CONV:
        found = True
    return found, stp, phi1, gval


@njit(error_model='numpy')
def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """`_linesearch._cubicmin`.  Returns (ok, xmin)."""
    C = fpa
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    d00 = dc * dc
    d01 = -db * db
    # `x ** np.float64(3)` calls libm `pow`; `x ** 3` compiles to a repeated
    # multiply, which rounds twice and drifts from CPython by an ulp.
    d10 = -dc ** np.float64(3)
    d11 = db ** np.float64(3)
    r0 = fb - fa - C * db
    r1 = fc - fa - C * dc
    A = (d00 * r0 + d01 * r1) / denom
    B = (d10 * r0 + d11 * r1) / denom
    radical = B * B - 3.0 * A * C
    xmin = a + (-B + np.sqrt(radical)) / (3.0 * A)
    if not np.isfinite(xmin):
        return False, 0.0
    return True, xmin


@njit(error_model='numpy')
def _quadmin(a, fa, fpa, b, fb):
    """`_linesearch._quadmin`.  Returns (ok, xmin)."""
    D = fa
    C = fpa
    db = b - a * 1.0
    B = (fb - D - C * db) / (db * db)
    xmin = a - C / (2.0 * B)
    if not np.isfinite(xmin):
        return False, 0.0
    return True, xmin


@njit(error_model='numpy')
def _zoom2(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
           xk, pk, a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi0, derphi0,
           c1, c2, use_extra, gfk, deltak, ordv, gtol, sigma_3,
           cx, cp, cg, cs, cv):
    """`_linesearch._zoom`.  Returns (ok, a_star, phi_star, gval, have_g)."""
    maxiter = 10
    i = 0
    delta1 = 0.2
    delta2 = 0.1
    phi_rec = phi0
    a_rec = 0.0
    gval = np.zeros(xk.shape[0], np.float64)
    a_j = 0.0
    phi_aj = phi0
    while True:
        dalpha = a_hi - a_lo
        if dalpha < 0.0:
            a = a_hi
            b = a_lo
        else:
            a = a_lo
            b = a_hi

        ok_c = False
        cchk = delta1 * dalpha
        if i > 0:
            ok_c, a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                                  a_rec, phi_rec)
        if (i == 0) or (not ok_c) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            ok_q, a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (not ok_q) or (a_j > b - qchk) or (a_j < a + qchk):
                a_j = a_lo + 0.5 * dalpha

        xa = xk + a_j * pk
        phi_aj = _sf_fun(func, args, SX, SF, SG, SFL, SC, xa)
        if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            gval = _sf_grad(func, grad, use_fd, eps, args,
                            SX, SF, SG, SFL, SC, xa)
            derphi_aj = np.dot(gval, pk)
            if abs(derphi_aj) <= -c2 * derphi0:
                accept = True
                if use_extra:
                    accept = _cg_descent_ok(
                        func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
                        xk, pk, gfk, deltak, ordv, gtol, sigma_3,
                        a_j, gval, True, cx, cp, cg, cs, cv)
                if accept:
                    return True, a_j, phi_aj, gval, True
            if derphi_aj * (a_hi - a_lo) >= 0.0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if i > maxiter:
            return False, 0.0, 0.0, gval, False


@njit(error_model='numpy')
def _wolfe2(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
            xk, pk, phi0, old_phi0, derphi0, c1, c2, amax,
            use_extra, gfk, deltak, ordv, gtol, sigma_3,
            cx, cp, cg, cs, cv):
    """`scalar_search_wolfe2`.  Returns (ok, a_star, phi_star, gval, have_g)."""
    maxiter = 10
    n = xk.shape[0]
    gval = np.zeros(n, np.float64)
    alpha0 = 0.0
    if derphi0 != 0.0:
        alpha1 = 1.01 * 2.0 * (phi0 - old_phi0) / derphi0
        # `min(1.0, ...)`, as in `_wolfe1`: a NaN operand gives 1.0.
        if not (alpha1 < 1.0):
            alpha1 = 1.0
    else:
        alpha1 = 1.0
    if alpha1 < 0.0:
        alpha1 = 1.0
    if alpha1 > amax:
        alpha1 = amax

    phi_a1 = _sf_fun(func, args, SX, SF, SG, SFL, SC, xk + alpha1 * pk)
    phi_a0 = phi0
    derphi_a0 = derphi0

    for i in range(maxiter):
        if alpha1 == 0.0 or alpha0 > amax:
            return False, 0.0, phi0, gval, False

        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
                (phi_a1 >= phi_a0 and i > 0):
            return _zoom2(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
                          xk, pk, alpha0, alpha1, phi_a0, phi_a1, derphi_a0,
                          phi0, derphi0, c1, c2, use_extra, gfk, deltak,
                          ordv, gtol, sigma_3, cx, cp, cg, cs, cv)

        xa = xk + alpha1 * pk
        gval = _sf_grad(func, grad, use_fd, eps, args,
                        SX, SF, SG, SFL, SC, xa)
        derphi_a1 = np.dot(gval, pk)
        if abs(derphi_a1) <= -c2 * derphi0:
            accept = True
            if use_extra:
                accept = _cg_descent_ok(
                    func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
                    xk, pk, gfk, deltak, ordv, gtol, sigma_3,
                    alpha1, gval, True, cx, cp, cg, cs, cv)
            if accept:
                return True, alpha1, phi_a1, gval, True

        if derphi_a1 >= 0.0:
            return _zoom2(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
                          xk, pk, alpha1, alpha0, phi_a1, phi_a0, derphi_a1,
                          phi0, derphi0, c1, c2, use_extra, gfk, deltak,
                          ordv, gtol, sigma_3, cx, cp, cg, cs, cv)

        alpha2 = 2.0 * alpha1
        if alpha2 > amax:
            alpha2 = amax
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = _sf_fun(func, args, SX, SF, SG, SFL, SC, xk + alpha1 * pk)
        derphi_a0 = derphi_a1

    # maxiter reached: the step is accepted, the gradient at it is not known
    return True, alpha1, phi_a1, gval, False


@njit(error_model='numpy')
def _ls_wolfe12(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
                xk, pk, gfk, phi0, old_phi0, c1, c2, amin, amax,
                use_extra, deltak, ordv, gtol, sigma_3,
                cx, cp, cg, cs, cv):
    """`_line_search_wolfe12`.  Returns (ok, alpha, new_fval, gval, have_g)."""
    if not (0.0 < c1 and c1 < c2 and c2 < 1.0):
        raise ValueError("'c1' and 'c2' do not satisfy '0 < c1 < c2 < 1'.")
    cv[0] = 0
    derphi0 = np.dot(gfk, pk)
    ok, stp, phi1, gval = _wolfe1(
        func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
        xk, pk, gfk, phi0, old_phi0, derphi0, c1, c2, amin, amax, 1e-14)
    if ok and use_extra:
        if not _cg_descent_ok(func, grad, use_fd, eps, args,
                              SX, SF, SG, SFL, SC, xk, pk, gfk, deltak,
                              ordv, gtol, sigma_3, stp, gval, True,
                              cx, cp, cg, cs, cv):
            ok = False
            cv[0] = 0
    if ok:
        return True, stp, phi1, gval, True

    cv[0] = 0
    ok2, a2, phi2, gval2, have_g = _wolfe2(
        func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
        xk, pk, phi0, old_phi0, derphi0, c1, c2, amax,
        use_extra, gfk, deltak, ordv, gtol, sigma_3, cx, cp, cg, cs, cv)
    return ok2, a2, phi2, gval2, have_g


# =================================================================== fmin_cg

@njit(error_model='numpy')
def _cg_core(func, grad, use_fd, x0, args, gtol, ordv, eps, maxiter,
             c1, c2, retall, cb=_cb_noop, use_cb=False):
    """`_minimize_cg`.  Returns (x, fval, gfk, nfev, njev, warnflag, allvecs)."""
    n = x0.shape[0]
    SX = np.full(n, np.nan, np.float64)
    SF = np.zeros(1, np.float64)
    SG = np.zeros(n, np.float64)
    SFL = np.zeros(2, np.int64)
    SC = np.zeros(2, np.int64)
    cx = np.zeros(n, np.float64)
    cp = np.zeros(n, np.float64)
    cg = np.zeros(n, np.float64)
    cs = np.zeros(2, np.float64)
    cv = np.zeros(1, np.int64)

    xk = x0.copy()
    old_fval = _sf_fun(func, args, SX, SF, SG, SFL, SC, xk)
    gfk = _sf_grad(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC, xk)

    k = 0
    old_old_fval = old_fval + np.sqrt(np.dot(gfk, gfk)) / 2.0

    nrows = 64 if retall else 0
    allvecs = np.empty((nrows, max(n, 1)), np.float64)
    nvec = 0
    if retall:
        allvecs = _nm_grow(allvecs, nvec)
        for j in range(n):
            allvecs[nvec, j] = xk[j]
        nvec += 1

    warnflag = 0
    pk = -gfk
    gnorm = _vecnorm(gfk, ordv)
    sigma_3 = 0.01

    # Drop whatever an earlier solve on this thread left in the halt
    # slot, so the first read below cannot see it.  A standalone entry
    # point does not clear it on the way out: only `minimize` takes it.
    _cb_halt_clear()

    while gnorm > gtol and k < maxiter:
        deltak = np.dot(gfk, gfk)
        ok, alpha_k, new_fval, gfkp1, have_g = _ls_wolfe12(
            func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
            xk, pk, gfk, old_fval, old_old_fval, c1, c2, 1e-100, 1e100,
            True, deltak, ordv, gtol, sigma_3, cx, cp, cg, cs, cv)
        if not ok:
            warnflag = 2
            break
        old_old_fval = old_fval
        old_fval = new_fval

        if cv[0] == 1 and alpha_k == cs[0]:
            for i in range(n):
                xk[i] = cx[i]
            pk = cp.copy()
            gfk = cg.copy()
            gnorm = cs[1]
        else:
            gnorm = _prp_step(func, grad, use_fd, eps, args,
                              SX, SF, SG, SFL, SC, xk, pk, gfk, deltak, ordv,
                              alpha_k, gfkp1, have_g, cx, cp, cg, cs, cv)
            for i in range(n):
                xk[i] = cx[i]
            pk = cp.copy()
            gfk = cg.copy()

        if retall:
            allvecs = _nm_grow(allvecs, nvec)
            for j in range(n):
                allvecs[nvec, j] = xk[j]
            nvec += 1
        k += 1
        # scipy's call site: after the `retall` append and the counter
        # increment, on `old_fval`.
        if use_cb:
            cb(xk.copy(), old_fval)
            if _cb_halt_get():
                break

    fval = old_fval
    if warnflag == 2:
        pass
    elif k >= maxiter:
        warnflag = 1
    elif np.isnan(gnorm) or np.isnan(fval) or np.any(np.isnan(xk)):
        warnflag = 3
    # `k` is APPENDED, so every existing index into this tuple is unchanged.
    # `fmin_cg` does not report it; `minimize` does, as scipy's `nit`.
    return (xk, fval, gfk, np.int64(SC[0]), np.int64(SC[1]), warnflag,
            allvecs[:nvec], np.int64(k))


def fmin_cg(f, x0, fprime=None, args=(), gtol=1e-5, norm=np.inf,
            epsilon=_EPS_FD, maxiter=None, full_output=0, disp=1, retall=0,
            callback=None, c1=1e-4, c2=0.4):
    """Minimize a function with the Polak-Ribiere+ conjugate gradient method.

    Reaches the same core as ``minimize(method='CG')``.  Callable from
    Python and from inside ``@njit``; both entries run the same compiled core.

    Parameters
    ----------
    f : callable
        A plain ``@njit`` ``f(x, *args) -> float``.  No ``@cfunc``, no
        ``.address``.
    x0 : array_like
        Initial guess.  Any rank is flattened.
    fprime : callable or None, optional
        A plain ``@njit`` ``fprime(x, *args) -> 1-D float64 array``.  ``None``
        (default) selects forward differences with step ``epsilon``.  A
        result whose length is not ``len(x0)`` raises ``ValueError``.
    args : tuple or ndarray, optional
        Extra parameters. A tuple is unpacked into separate arguments after
        `x`, so ``args=(a, b)`` calls ``f(x, a, b)`` and
        ``fprime(x, a, b)``. Its elements may be of any type a compiled call
        accepts, arrays and strings included. Default ``()``, which calls
        ``f(x)``. An ndarray or a list arrives as ONE argument instead,
        ``f(x, args)``. See Notes.
    gtol : float, optional
        Stop when ``vecnorm(g, norm) <= gtol``.  Default 1e-5.
    norm : float, optional
        Order of the gradient norm.  ``inf`` (default) is ``max|g_i|``,
        ``-inf`` is ``min|g_i|``, any other value is the p-norm.
    epsilon : float or ndarray, optional
        Absolute step for the forward-difference gradient.  An array gives
        a per-element step and must have length ``len(x0)`` or length 1.
    maxiter : int, float or None, optional
        Iteration cap.  ``None`` (default) means ``N*200``.  ``np.inf`` is
        accepted and means unbounded.
    full_output : bool, optional
        ``False`` (default) returns ``xopt``.  ``True`` returns
        ``(xopt, fopt, func_calls, grad_calls, warnflag)``.  Compile-time
        constant inside ``@njit``: it selects the return type.
    disp : int, optional
        ``1`` (default) prints a summary. The status line appears only on
        success and becomes an ``OptimizeWarning`` otherwise; the counters
        print either way. ``0`` is silent, and is what a ``prange`` sweep
        wants: see Notes.
    retall : bool, optional
        Append ``allvecs`` to the return.  Compile-time constant.
    callback : callable or None, optional
        Called once per iteration as ``callback(xk)``, with a copy of the
        current iterate. A plain Python function may instead be written
        ``callback(intermediate_result)`` and is handed an object carrying
        ``x`` and ``fun``. Raising ``StopIteration`` stops the run and
        returns the current best. From inside ``@njit`` the callback is an
        ``@njit`` ``callback(xk)``.
    c1, c2 : float, optional
        Armijo and curvature parameters of the line search.  Defaults 1e-4
        and 0.4.  ``0 < c1 < c2 < 1`` is required, checked at the first line
        search.

    Returns
    -------
    xopt : ndarray
        The minimizer.
    fopt : float
        ``f(xopt)``.  With ``full_output``.
    func_calls : int
        Objective evaluations, including those consumed by a
        finite-difference gradient.  With ``full_output``.
    grad_calls : int
        Gradient evaluations.  With ``full_output``.
    warnflag : int
        ``0`` converged, ``1`` ``maxiter`` reached, ``2`` precision loss
        (the line search failed), ``3`` NaN encountered.  NOTE that ``fmin``
        uses a DIFFERENT table, where ``1`` means ``maxfun``.  With
        ``full_output``.
    allvecs : list of ndarray
        ``xk`` after each iteration, ``nit + 1`` entries of shape ``(N,)``,
        the first being `x0`.  With ``retall``.  From inside ``@njit`` this
        is a ``numba.typed.List``.

    Raises
    ------
    ValueError
        If `callback` is neither ``None`` nor a callable nor an ``@njit``
        function, or `fprime` returns a gradient whose
        length is not ``len(x0)``.

    See Also
    --------
    scipy.optimize.fmin_cg : The scipy routine this mirrors.
    scijit.optimize.fmin_bfgs : Stores an inverse-Hessian estimate; usually
        fewer iterations at O(N^2) memory.
    scijit.optimize.fmin_l_bfgs_b : Bounds, and limited memory at large N.
    scijit.optimize.minimize : Dispatches over the Fortran-backed methods.

    Notes
    -----
    Under `retall`, ``allvecs`` is a python list of 1-D arrays, as scipy's
    is. From inside ``@njit`` it is a ``numba.typed.List`` of the same
    arrays.

    `args` is unpacked into the argument lists of `f` and `fprime`,
    ``f(x, *args)``, which is scipy's contract.

    DELIBERATELY DIFFERENT: an ndarray or a list `args` reaches `f` and
    `fprime` as ONE argument, ``f(x, args)``. scipy unpacks those two element
    by element. The arity of a compiled call is fixed when it compiles, so a
    sequence whose length is known only at run time cannot be unpacked:
    ``f(x, *args)`` on an ndarray inside ``@njit`` is a ``TypingError``. An
    objective written to scipy's shape refuses the ndarray rather than
    reading it wrongly.

    `full_output` and `retall` must be compile-time literals inside ``@njit``:
    they select the return type.

    `callback` takes a plain Python function or an ``@njit`` function, and
    both halt on ``StopIteration``. The Python spelling reaches the compiled
    core through a ``numba.objmode`` block over a module-level slot: it takes
    the GIL once per iteration, and two concurrent solves that both install
    one overwrite each other, so a ``prange`` loop over it serializes and is
    not reentrant.

    Two differences from scipy follow from the compiled path. A plain Python
    callable cannot be an argument of a compiled function, so a call written
    inside ``@njit`` passes the ``@njit`` spelling and a Python one is refused
    by numba. And numba matches no exception class, ``except StopIteration``
    being a ``TypingError`` on 0.66, so any exception raised inside an
    ``@njit`` callback stops the run the way ``StopIteration`` does, where
    scipy propagates it. A Python callback keeps scipy's behaviour: only
    ``StopIteration`` halts, and anything else reaches the caller.

    ``disp`` reproduces scipy's summary and scipy's warning, at scipy's
    default, so a bare call writes to stdout and warns on a limit path. The
    warning takes the GIL through a ``numba.objmode`` block and the printing
    through Python's stdout, so with ``disp`` set a ``prange`` loop over this
    routine still gives the right answer but serializes. ``disp=0`` is the
    parallel path.

    A gradient whose length is not ``len(x0)`` raises. scipy 1.18 separates
    the two cases. A LONGER gradient raises ``ValueError: operands could not
    be broadcast together``. A SHORTER one is broadcast and the run finishes:
    ``warnflag`` is 0 with `x` at the starting point when the gradient is
    small enough that the norm test halts first, and 2 otherwise. The point
    can also move, measured ``[1.524, 1.524]`` from ``x0 = [0.5, 0.5]`` with a
    length-1 gradient of ``-0.001``.

    Pure ``@njit``, so it is safe to call from a ``numba.prange`` loop with
    ``disp=0`` and with `callback` left at ``None`` or given as an ``@njit``
    function.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import fmin_cg
    >>> @njit
    ... def rosen(x):
    ...     s = 0.0
    ...     for i in range(x.size - 1):
    ...         s += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    ...     return s
    >>> @njit
    ... def rosen_g(x):
    ...     g = np.zeros(x.size)
    ...     for i in range(x.size - 1):
    ...         g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
    ...         g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    ...     return g
    >>> @njit
    ... def run():
    ...     return fmin_cg(rosen, np.array([0.0, 0.0]), rosen_g, disp=0)
    >>> np.round(run(), 6)
    array([0.999992, 0.999984])
    """
    cbf, use_cb, pycb = _cb_resolve('fmin_cg', callback)
    a = args
    xf = np.ascontiguousarray(np.atleast_1d(
        np.asarray(x0, dtype=np.float64))).ravel()
    mi = xf.size * 200 if maxiter is None else _budget(maxiter)
    prev = _cb_install(pycb)
    try:
        if fprime is None:
            r = _cg_core(f, _no_grad, True, xf, a, gtol, float(norm),
                         _eps_vec(epsilon, xf.size), mi, c1, c2,
                         bool(retall), cbf, use_cb)
        else:
            r = _cg_core(f, fprime, False, xf, a, gtol, float(norm),
                         _eps_vec(epsilon, xf.size), mi, c1, c2,
                         bool(retall), cbf, use_cb)
    finally:
        _cb_release(prev)
    if disp:
        _disp_grad(r[5], _status_of(_GR_STATUS, r[5]), r[1], r[7],
                       r[3], r[4])
    if full_output:
        out = (r[0], r[1], int(r[3]), int(r[4]), int(r[5]))
        if retall:
            return out + (_allvecs_list(r[6]),)
        return out
    if retall:
        return r[0], _allvecs_list(r[6])
    return r[0]


@overload(fmin_cg, prefer_literal=True)
def _fmin_cg_ovl(f, x0, fprime=None, args=(), gtol=1e-5, norm=np.inf,
                 epsilon=_EPS_FD, maxiter=None, full_output=0, disp=1,
                 retall=0, callback=None, c1=1e-4, c2=0.4):
    """@njit implementation of `fmin_cg`, resolved at compile time.

    ``full_output`` and ``retall`` between them select one of four return
    shapes, and an absent ``fprime`` swaps the gradient slot for the
    `_no_grad` placeholder and flips `_cg_core`'s finite-difference
    switch -- a type change either way. Returning ``None`` declines the
    call, which numba reports as a TypingError naming the argument that
    could not be served.
    """
    fo = _lit_bool(full_output)
    ra = _lit_bool(retall)
    if fo is None or ra is None:
        return None                     # runtime flag -> TypingError
    CBF, USE_CB = _cb_resolve_ty('fmin_cg', callback)
    fd = _is_none(fprime)
    mi_none = _is_none(maxiter)

    def impl(f, x0, fprime=None, args=(), gtol=1e-5, norm=np.inf,
             epsilon=_EPS_FD, maxiter=None, full_output=0, disp=1,
             retall=0, callback=None, c1=1e-4, c2=0.4):
        a = args
        xf = np.ascontiguousarray(
            np.atleast_1d(np.asarray(x0))).ravel().astype(np.float64)
        if mi_none:
            mi = np.int64(xf.size * 200)
        else:
            mi = _budget_nb(maxiter)
        if fd:
            r = _cg_core(f, _no_grad, True, xf, a, gtol, np.float64(norm),
                         _eps_vec(epsilon, xf.size), mi, c1, c2, ra,
                         CBF, USE_CB)
        else:
            r = _cg_core(f, fprime, False, xf, a, gtol, np.float64(norm),
                         _eps_vec(epsilon, xf.size), mi, c1, c2, ra,
                         CBF, USE_CB)
        if disp:
            _disp_grad(r[5], _status_of(_GR_STATUS, r[5]), r[1], r[7],
                       r[3], r[4])
        if fo:
            if ra:
                return r[0], r[1], r[3], r[4], r[5], _allvecs_list(r[6])
            return r[0], r[1], r[3], r[4], r[5]
        if ra:
            return r[0], _allvecs_list(r[6])
        return r[0]
    return impl


# ================================================================= fmin_bfgs

@njit(error_model='numpy')
def _bfgs_core(func, grad, use_fd, x0, args, gtol, ordv, eps, maxiter,
               c1, c2, xrtol, H0, use_h0, retall, cb=_cb_noop, use_cb=False):
    """`_minimize_bfgs`.

    Returns (x, fval, gfk, Hk, nfev, njev, warnflag, allvecs).
    """
    n = x0.shape[0]
    SX = np.full(n, np.nan, np.float64)
    SF = np.zeros(1, np.float64)
    SG = np.zeros(n, np.float64)
    SFL = np.zeros(2, np.int64)
    SC = np.zeros(2, np.int64)
    cx = np.zeros(1, np.float64)
    cp = np.zeros(1, np.float64)
    cg = np.zeros(1, np.float64)
    cs = np.zeros(2, np.float64)
    cv = np.zeros(1, np.int64)

    xk = x0.copy()
    old_fval = _sf_fun(func, args, SX, SF, SG, SFL, SC, xk)
    gfk = _sf_grad(func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC, xk)

    k = 0
    I = np.eye(n, dtype=np.float64)
    if use_h0:
        Hk = H0.copy()
    else:
        Hk = np.eye(n, dtype=np.float64)
    old_old_fval = old_fval + np.sqrt(np.dot(gfk, gfk)) / 2.0

    nrows = 64 if retall else 0
    allvecs = np.empty((nrows, max(n, 1)), np.float64)
    nvec = 0
    if retall:
        allvecs = _nm_grow(allvecs, nvec)
        for j in range(n):
            allvecs[nvec, j] = xk[j]
        nvec += 1

    warnflag = 0
    gnorm = _vecnorm(gfk, ordv)

    # Drop whatever an earlier solve on this thread left in the halt
    # slot, so the first read below cannot see it.  A standalone entry
    # point does not clear it on the way out: only `minimize` takes it.
    _cb_halt_clear()

    while gnorm > gtol and k < maxiter:
        pk = -np.dot(Hk, gfk)
        ok, alpha_k, new_fval, gfkp1, have_g = _ls_wolfe12(
            func, grad, use_fd, eps, args, SX, SF, SG, SFL, SC,
            xk, pk, gfk, old_fval, old_old_fval, c1, c2, 1e-100, 1e100,
            False, 0.0, ordv, gtol, 0.0, cx, cp, cg, cs, cv)
        if not ok:
            warnflag = 2
            break
        old_old_fval = old_fval
        old_fval = new_fval

        sk = alpha_k * pk
        xkp1 = xk + sk
        if retall:
            allvecs = _nm_grow(allvecs, nvec)
            for j in range(n):
                allvecs[nvec, j] = xkp1[j]
            nvec += 1
        xk = xkp1
        if not have_g:
            gfkp1 = _sf_grad(func, grad, use_fd, eps, args,
                             SX, SF, SG, SFL, SC, xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        k += 1
        # scipy's call site: after the counter increment and BEFORE the
        # gradient-norm test below it, on `old_fval`.
        if use_cb:
            cb(xk.copy(), old_fval)
            if _cb_halt_get():
                break
        gnorm = _vecnorm(gfk, ordv)
        if gnorm <= gtol:
            break

        if alpha_k * _vecnorm(pk, 2.0) <= xrtol * (xrtol + _vecnorm(xk, 2.0)):
            break
        if not np.isfinite(old_fval):
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        if rhok_inv == 0.0:
            rhok = 1000.0
        else:
            rhok = 1.0 / rhok_inv

        A1 = I - np.outer(sk, yk) * rhok
        A2 = I - np.outer(yk, sk) * rhok
        # scipy writes `rhok * sk[:, None] * sk[None, :]`, which scales the
        # COLUMN first: (rhok*sk_i)*sk_j, not rhok*(sk_i*sk_j).
        Hk = np.dot(A1, np.dot(Hk, A2)) + np.outer(rhok * sk, sk)

    fval = old_fval
    if warnflag == 2:
        pass
    elif k >= maxiter:
        warnflag = 1
    elif np.isnan(gnorm) or np.isnan(fval) or np.any(np.isnan(xk)):
        warnflag = 3
    # `k` is APPENDED, so every existing index into this tuple is unchanged.
    # `fmin_bfgs` does not report it; `minimize` does, as scipy's `nit`.
    return (xk, fval, gfk, Hk, np.int64(SC[0]), np.int64(SC[1]), warnflag,
            allvecs[:nvec], np.int64(k))


def fmin_bfgs(f, x0, fprime=None, args=(), gtol=1e-5, norm=np.inf,
              epsilon=_EPS_FD, maxiter=None, full_output=0, disp=1,
              retall=0, callback=None, xrtol=0, c1=1e-4, c2=0.9,
              hess_inv0=None):
    """Minimize a function with the quasi-Newton BFGS method.

    Reaches the same core as ``minimize(method='BFGS')``.  Callable from
    Python and from inside ``@njit``; both entries run the same compiled core.

    Parameters
    ----------
    f : callable
        A plain ``@njit`` ``f(x, *args) -> float``.
    x0 : array_like
        Initial guess.  Any rank is flattened.
    fprime : callable or None, optional
        A plain ``@njit`` ``fprime(x, *args) -> 1-D float64 array``.  ``None``
        (default) selects forward differences with step ``epsilon``.  A
        result whose length is not ``len(x0)`` raises ``ValueError``.
    args : tuple or ndarray, optional
        Extra parameters. A tuple is unpacked into separate arguments after
        `x`, so ``args=(a, b)`` calls ``f(x, a, b)`` and
        ``fprime(x, a, b)``. Its elements may be of any type a compiled call
        accepts, arrays and strings included. Default ``()``, which calls
        ``f(x)``. An ndarray or a list arrives as ONE argument instead,
        ``f(x, args)``. See Notes.
    gtol : float, optional
        Stop when ``vecnorm(g, norm) <= gtol``.  Default 1e-5.
    norm : float, optional
        Order of the gradient norm.  ``inf`` (default) is ``max|g_i|``.
    epsilon : float or ndarray, optional
        Absolute step for the forward-difference gradient.  An array gives
        a per-element step and must have length ``len(x0)`` or length 1.
    maxiter : int, float or None, optional
        Iteration cap.  ``None`` (default) means ``N*200``.  ``np.inf`` is
        accepted and means unbounded.
    full_output : bool, optional
        ``False`` (default) returns ``xopt``.  ``True`` returns
        ``(xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag)``.
        Compile-time constant inside ``@njit``.
    disp : int, optional
        ``1`` (default) prints a summary. The status line appears only on
        success and becomes an ``OptimizeWarning`` otherwise; the counters
        print either way. ``0`` is silent, and is what a ``prange`` sweep
        wants: see Notes.
    retall : bool, optional
        Append ``allvecs`` to the return.  Compile-time constant.
    callback : callable or None, optional
        Called once per iteration as ``callback(xk)``, with a copy of the
        current iterate. A plain Python function may instead be written
        ``callback(intermediate_result)`` and is handed an object carrying
        ``x`` and ``fun``. Raising ``StopIteration`` stops the run and
        returns the current best. From inside ``@njit`` the callback is an
        ``@njit`` ``callback(xk)``.
    xrtol : float, optional
        Relative tolerance on ``x``.  Stop when the step is shorter than
        ``xrtol * (xrtol + |xk|_2)``.  Default 0.
    c1, c2 : float, optional
        Armijo and curvature parameters.  Defaults 1e-4 and 0.9.
        ``0 < c1 < c2 < 1`` is required.
    hess_inv0 : ndarray or None, optional
        Initial inverse-Hessian estimate, shape ``(N, N)``.  ``None``
        (default) is the identity.  A non-symmetric or non-positive-definite
        matrix raises ``ValueError``.

    Returns
    -------
    xopt : ndarray
        The minimizer.
    fopt : float
        ``f(xopt)``.  With ``full_output``.
    gopt : ndarray
        Gradient at ``xopt``.  With ``full_output``.
    Bopt : ndarray, shape (N, N)
        The inverse-Hessian estimate at ``xopt``.  With ``full_output``.
    func_calls : int
        Objective evaluations, including those consumed by a
        finite-difference gradient.  With ``full_output``.
    grad_calls : int
        Gradient evaluations.  With ``full_output``.
    warnflag : int
        ``0`` converged, ``1`` ``maxiter`` reached, ``2`` precision loss,
        ``3`` NaN encountered.  With ``full_output``.
    allvecs : list of ndarray
        ``xk`` after each iteration, ``nit + 1`` entries of shape ``(N,)``,
        the first being `x0`.  With ``retall``.  From inside ``@njit`` this
        is a ``numba.typed.List``.

    Raises
    ------
    ValueError
        If `callback` is neither ``None`` nor a callable nor an ``@njit``
        function; if `fprime` returns a gradient whose
        length is not ``len(x0)``; or if `hess_inv0` is not ``(N, N)``.

    See Also
    --------
    scipy.optimize.fmin_bfgs : The scipy routine this mirrors.
    scijit.optimize.fmin_cg : No stored matrix; O(N) memory.
    scijit.optimize.fmin_l_bfgs_b : Bounds, and limited memory at large N.
    scijit.optimize.minimize : Dispatches over the Fortran-backed methods.

    Notes
    -----
    Under `retall`, ``allvecs`` is a python list of 1-D arrays, as scipy's
    is. From inside ``@njit`` it is a ``numba.typed.List`` of the same
    arrays.

    `args` is unpacked into the argument lists of `f` and `fprime`,
    ``f(x, *args)``, which is scipy's contract.

    DELIBERATELY DIFFERENT: an ndarray or a list `args` reaches `f` and
    `fprime` as ONE argument, ``f(x, args)``. scipy unpacks those two element
    by element. The arity of a compiled call is fixed when it compiles, so a
    sequence whose length is known only at run time cannot be unpacked:
    ``f(x, *args)`` on an ndarray inside ``@njit`` is a ``TypingError``. An
    objective written to scipy's shape refuses the ndarray rather than
    reading it wrongly.

    `full_output` and `retall` must be compile-time literals inside ``@njit``:
    they select the return type.

    `callback` takes a plain Python function or an ``@njit`` function, and
    both halt on ``StopIteration``. The Python spelling reaches the compiled
    core through a ``numba.objmode`` block over a module-level slot: it takes
    the GIL once per iteration, and two concurrent solves that both install
    one overwrite each other, so a ``prange`` loop over it serializes and is
    not reentrant.

    Two differences from scipy follow from the compiled path. A plain Python
    callable cannot be an argument of a compiled function, so a call written
    inside ``@njit`` passes the ``@njit`` spelling and a Python one is refused
    by numba. And numba matches no exception class, ``except StopIteration``
    being a ``TypingError`` on 0.66, so any exception raised inside an
    ``@njit`` callback stops the run the way ``StopIteration`` does, where
    scipy propagates it. A Python callback keeps scipy's behaviour: only
    ``StopIteration`` halts, and anything else reaches the caller.

    ``disp`` reproduces scipy's summary and scipy's warning, at scipy's
    default, so a bare call writes to stdout and warns on a limit path. The
    warning takes the GIL through a ``numba.objmode`` block and the printing
    through Python's stdout, so with ``disp`` set a ``prange`` loop over this
    routine still gives the right answer but serializes. ``disp=0`` is the
    parallel path.

    A gradient whose length is not ``len(x0)`` raises. scipy 1.18 raises on
    the same input, ``ValueError: shapes (2,2) and (1,) not aligned``, from
    the inverse-Hessian product. The one case it misses is a wrong-length
    gradient small enough for the gradient-norm test to halt at iteration 0,
    where it returns `x` at the starting point with ``warnflag`` 0.

    Pure ``@njit``, so it is safe to call from a ``numba.prange`` loop with
    ``disp=0`` and with `callback` left at ``None`` or given as an ``@njit``
    function.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import fmin_bfgs
    >>> @njit
    ... def rosen(x):
    ...     s = 0.0
    ...     for i in range(x.size - 1):
    ...         s += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    ...     return s
    >>> @njit
    ... def rosen_g(x):
    ...     g = np.zeros(x.size)
    ...     for i in range(x.size - 1):
    ...         g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
    ...         g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    ...     return g
    >>> @njit
    ... def run():
    ...     return fmin_bfgs(rosen, np.array([0.0, 0.0]), rosen_g,
    ...                      disp=0)
    >>> np.round(run(), 6)
    array([0.999999, 0.999998])
    """
    cbf, use_cb, pycb = _cb_resolve('fmin_bfgs', callback)
    a = args
    xf = np.ascontiguousarray(np.atleast_1d(
        np.asarray(x0, dtype=np.float64))).ravel()
    mi = xf.size * 200 if maxiter is None else _budget(maxiter)
    if hess_inv0 is None:
        H0, uh = np.zeros((0, 0)), False
    else:
        H0 = np.ascontiguousarray(np.asarray(hess_inv0, dtype=np.float64))
        _check_hess_inv0(H0, xf.size)
        uh = True
    prev = _cb_install(pycb)
    try:
        if fprime is None:
            r = _bfgs_core(f, _no_grad, True, xf, a, gtol, float(norm),
                           _eps_vec(epsilon, xf.size), mi, c1, c2,
                           float(xrtol), H0, uh,
                           bool(retall), cbf, use_cb)
        else:
            r = _bfgs_core(f, fprime, False, xf, a, gtol, float(norm),
                           _eps_vec(epsilon, xf.size), mi, c1, c2,
                           float(xrtol), H0, uh,
                           bool(retall), cbf, use_cb)
    finally:
        _cb_release(prev)
    if disp:
        _disp_grad(r[6], _status_of(_GR_STATUS, r[6]), r[1], r[8],
                       r[4], r[5])
    if full_output:
        out = (r[0], r[1], r[2], r[3], int(r[4]), int(r[5]), int(r[6]))
        if retall:
            return out + (_allvecs_list(r[7]),)
        return out
    if retall:
        return r[0], _allvecs_list(r[7])
    return r[0]


@overload(fmin_bfgs, prefer_literal=True)
def _fmin_bfgs_ovl(f, x0, fprime=None, args=(), gtol=1e-5, norm=np.inf,
                   epsilon=_EPS_FD, maxiter=None, full_output=0, disp=1,
                   retall=0, callback=None, xrtol=0, c1=1e-4, c2=0.9,
                   hess_inv0=None):
    """@njit implementation of `fmin_bfgs`, resolved at compile time.

    Three arguments change types rather than values: ``full_output`` and
    ``retall`` pick one of four return shapes, an absent ``fprime`` swaps
    the gradient slot for the `_no_grad` placeholder, and an absent
    ``hess_inv0`` makes the initial inverse Hessian an empty ``(0, 0)``
    array with a companion flag instead of a real one. Returning ``None``
    declines the call, which numba reports as a TypingError naming the
    argument that could not be served.
    """
    fo = _lit_bool(full_output)
    ra = _lit_bool(retall)
    if fo is None or ra is None:
        return None
    CBF, USE_CB = _cb_resolve_ty('fmin_bfgs', callback)
    fd = _is_none(fprime)
    mi_none = _is_none(maxiter)
    h_none = _is_none(hess_inv0)

    def impl(f, x0, fprime=None, args=(), gtol=1e-5, norm=np.inf,
             epsilon=_EPS_FD, maxiter=None, full_output=0, disp=1,
             retall=0, callback=None, xrtol=0, c1=1e-4, c2=0.9,
             hess_inv0=None):
        a = args
        xf = np.ascontiguousarray(
            np.atleast_1d(np.asarray(x0))).ravel().astype(np.float64)
        if mi_none:
            mi = np.int64(xf.size * 200)
        else:
            mi = _budget_nb(maxiter)
        if h_none:
            H0 = np.zeros((0, 0), np.float64)
            uh = False
        else:
            H0 = np.ascontiguousarray(
                np.asarray(hess_inv0)).astype(np.float64)
            _check_hess_inv0(H0, xf.size)
            uh = True
        if fd:
            r = _bfgs_core(f, _no_grad, True, xf, a, gtol, np.float64(norm),
                           _eps_vec(epsilon, xf.size), mi, c1, c2,
                           np.float64(xrtol), H0, uh, ra, CBF, USE_CB)
        else:
            r = _bfgs_core(f, fprime, False, xf, a, gtol, np.float64(norm),
                           _eps_vec(epsilon, xf.size), mi, c1, c2,
                           np.float64(xrtol), H0, uh, ra, CBF, USE_CB)
        if disp:
            _disp_grad(r[6], _status_of(_GR_STATUS, r[6]), r[1], r[8],
                       r[4], r[5])
        if fo:
            if ra:
                return (r[0], r[1], r[2], r[3], r[4], r[5], r[6],
                        _allvecs_list(r[7]))
            return r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        if ra:
            return r[0], _allvecs_list(r[7])
        return r[0]
    return impl
