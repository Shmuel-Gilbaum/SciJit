"""Numba-callable SLSQP (constrained nonlinear minimization).

Backed by libslsqp (jacobwilliams/slsqp modern refactor of Dieter
Kraft's SLSQP, BSD-3). Equivalent of
scipy.optimize.minimize(method='SLSQP').

slsqp is REVERSE COMMUNICATION via mode: the @njit driver loop below
calls the user's PLAIN @njit functions when the solver asks, no
@cfunc, no address, and no module-level solver state (all of it lives
in caller arrays; the refactor's F77 SAVE variables are carried in
flat sdat/ldat arrays), so concurrent solves are safe. A PYTHON
`callback` is the one exception: it travels through a module-level
slot.

The user supplies two @njit functions (mirroring scipy's fun/jac split):

    @njit
    def fc(x, *args):               # mode == 1 request
        ...
        return f, c        # objective value, constraints array (len
                           # max(m,1); equality constraints FIRST,
                           # c[i] == 0 required for i < meq, then
                           # inequalities c[i] >= 0; if m == 0 return
                           # a dummy np.zeros(1))

    @njit
    def gj(x, *args):               # mode == -1 request
        ...
        return g, jac      # gradient (n,), constraint Jacobian
                           # (m, n) C-ordered 2-D array (jac[i, j] =
                           # d c_i / d x_j; if m == 0 pass
                           # np.zeros((1, n)))

Problem shape (m, meq) is passed explicitly, function results cannot
be introspected before compilation.
"""

from numba import njit, objmode, types
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
import numpy as np
from .._lib._load import load

_lib, _sig = load(__file__, "libslsqp")


_slsqp = _sig(_lib.slsqp_wrapper, 26)


_EPSILON = float(np.sqrt(np.finfo(np.float64).eps))     # scipy's _epsilon

# `bounds` coercion and the compile-time helpers are shared with the L-BFGS-B
# public entry point rather than copied.
from ._lbfgsb import (                       # noqa: E402
    _split_bounds, _as_bounds, _lit_bool, _is_none, _BOUNDS_LEN_MSG,
    _eps_at, _check_eps, _call_args,
)
from ._callback import (_cb_noop, _cb_install, _cb_release,   # noqa: E402
                        _cb_resolve, _cb_resolve_ty,
                        _cb_halt_get, _cb_halt_clear)


# --------------------------------------------------------------------------
# placeholders for the absent callbacks
# --------------------------------------------------------------------------
@njit
def _no_c(x, *args):
    """Constraint values for an absent constraint family: length 0.

    Every callback slot the core takes is a concrete ``@njit`` function,
    so an absent family is served by one that returns nothing rather than
    by a compile-time branch through the core. ``meq`` and ``mineq`` are
    read from the returned lengths, so they come out 0 with no special
    casing. `_no_j` and `_no_g` are the same idea for the Jacobian and
    the objective gradient.
    """
    return np.zeros(0, np.float64)


@njit
def _no_nlead(x, *args):
    """Leading-row count for an absent scalar-entry family: zero.

    Fills the same slot `_combine_nlead` builds, so the call site is one
    expression rather than a branch on a compile-time flag. See `_no_c`.
    """
    return 0


@njit
def _no_j(x, *args):
    """Constraint-Jacobian placeholder, shape ``(0, n)``.

    Never called: the core reaches the analytic Jacobian only under
    ``has_jeq``/``has_jieq`` and takes `_fd_jac` otherwise. It exists so
    the slot always holds a concrete ``@njit`` function of the right
    return shape, which is what lets the core be typed at all. See
    `_no_c`.
    """
    return np.zeros((0, x.size), np.float64)


@njit
def _no_g(x, *args):
    """Objective-gradient placeholder, ``n`` zeros.

    Never called: the core reaches ``fprime`` only under ``has_gf`` and
    takes `_fd_grad` otherwise. Same reason as `_no_j` -- the slot has to
    hold something concrete for the core to type. See `_no_c`.
    """
    return np.zeros(x.size, np.float64)


# --------------------------------------------------------------------------
# scipy's LIST of scalar constraints -> the one vector-valued function the
# solver takes
# --------------------------------------------------------------------------
#: Built functions, keyed on the identities of the constraints they combine.
#: The tuple is stored beside the function so the identities stay valid: a
#: collected dispatcher would let a later object reuse its id.
_COMBINED_CONS = {}
_COMBINED_NLEAD = {}


def _as_1d(v):
    """One constraint entry's value as a 1-D array.

    An entry may return a single number or several at once, which is
    ``np.atleast_1d`` in scipy and a type question here.
    """
    return np.atleast_1d(np.asarray(v))


@overload(_as_1d)
def _as_1d_ovl(v):
    if isinstance(v, types.Array):
        if v.ndim == 1:
            def impl(v):
                return v
            return impl

        def impl(v):
            return np.ravel(v)
        return impl

    def impl(v):
        out = np.empty(1, np.float64)
        out[0] = v
        return out
    return impl


def _combine_nlead(fns):
    """``(x, *args) -> int``, the number of rows a tuple of entries adds.

    Read only where an analytic Jacobian describes the vector-valued function
    alone, so the rows these entries add have to be differenced. An entry may
    contribute more than one row, so the count is a run-time fact.
    """
    key = tuple(id(f) for f in fns)
    hit = _COMBINED_NLEAD.get(key)
    if hit is not None and hit[0] == fns:
        return hit[1]
    ns = {'np': np, '_as_1d': _as_1d, '_call_args': _call_args}
    src = ['def _cons_nlead(x, *args):', '    tot = 0']
    for i in range(len(fns)):
        ns['_c%d' % i] = fns[i]
        src.append('    tot += _as_1d(_c%d(x, *args)).size' % i)
    src.append('    return tot')
    exec('\n'.join(src), ns)
    fn = njit(ns['_cons_nlead'])
    _COMBINED_NLEAD[key] = (fns, fn)
    return fn


def _combine_cons(fns, vec=None):
    """One ``(x, *args) -> ndarray`` from a tuple of scalar constraints.

    scipy's `eqcons` and `ieqcons` are a sequence of callables, one scalar
    constraint each; `f_eqcons` and `f_ieqcons` are one callable returning
    all of them at once. The solver takes the second shape, so the first is
    rewritten into it here, when the call compiles.

    Each entry is called as ``g(x, *args) -> float``, which is scipy's own
    convention.

    `vec` is the vector-valued function of the same kind, when one was given
    as well. scipy CONCATENATES the two spellings rather than preferring
    either, scalars first, so the result does too.

    The body is generated because the number of calls is the length of
    `fns`, which is known here and nowhere later. The result is cached on
    the identities of `fns` and `vec`, so the same constraint list does not
    rebuild it.
    """
    key = tuple(id(f) for f in fns) + (id(vec),)
    hit = _COMBINED_CONS.get(key)
    if hit is not None and hit[0] == (fns, vec):
        return hit[1]
    k = len(fns)
    ns = {'np': np, '_as_1d': _as_1d, '_call_args': _call_args}
    src = ['def _cons_vec(x, *args):']
    # each entry may return one value or several, as scipy's
    # `sum(map(len, [atleast_1d(c['fun'](x, *args)) ...]))` allows, so the
    # block is sized from what the entries return rather than from `k`
    for i in range(k):
        ns['_c%d' % i] = fns[i]
        src.append('    r%d = _as_1d(_c%d(x, *args))' % (i, i))
    tot = ' + '.join('r%d.size' % i for i in range(k))
    if vec is None:
        src.append('    out = np.empty(%s, np.float64)' % tot)
    else:
        ns['_v'] = vec
        src.append('    v = _v(x, *args)')
        src.append('    out = np.empty(%s + v.size, np.float64)' % tot)
    src.append('    p = 0')
    for i in range(k):
        src.append('    out[p:p + r%d.size] = r%d' % (i, i))
        src.append('    p += r%d.size' % i)
    if vec is not None:
        src.append('    out[p:] = v')
    src.append('    return out')
    exec('\n'.join(src), ns)
    fn = njit(ns['_cons_vec'])
    _COMBINED_CONS[key] = ((fns, vec), fn)
    return fn


_CONS_MSG = (
    "fmin_slsqp: every entry of eqcons/ieqcons must be an @njit function "
    "g(x, *args) -> float. A plain Python callable cannot be reached from "
    "compiled code")

#: scipy's own text, `_slsqp_py.py:406-408`, and its own exception class.
_SLSQP_BOUNDS_LEN_MSG = ('SLSQP Error: the length of bounds is not '
                         'compatible with that of x0.')


def _cons_dispatchers(seq):
    """The `@njit` dispatchers in `seq`, as a tuple. Raises otherwise."""
    out = []
    for f in seq:
        if not isinstance(f, Dispatcher):
            if not callable(f):
                raise TypeError("'%s' object is not callable"
                                % type(f).__name__)
            raise ValueError(_CONS_MSG)
        out.append(f)
    return tuple(out)


def _cons_arg_type(t):
    """The dispatchers behind a tuple-of-functions ARGUMENT TYPE.

    Returns a tuple, empty when the argument is scipy's default. Returns
    ``None`` when the argument is something else, which declines the call.

    An argument left at its default arrives here in one of two shapes: the
    raw Python value ``()``, or ``types.Omitted(())``. Both are tested, as
    the `bounds` handling below already does.
    """
    if isinstance(t, types.Omitted):
        t = t.value
    if isinstance(t, types.BaseTuple):
        out = []
        for e in t.types:
            if not isinstance(e, types.Dispatcher):
                return None
            out.append(e.dispatcher)
        return tuple(out)
    return () if t == () else None


# --------------------------------------------------------------------------
# forward differences, scipy's approx_derivative(method='2-point',
# abs_step=epsilon, bounds=...) including _adjust_scheme_to_bounds
# --------------------------------------------------------------------------
@njit
def _fd_h(xi, lb, ub, eps):
    """The step `approx_derivative` takes at one coordinate."""
    lower_dist = xi - lb
    upper_dist = ub - xi
    h = eps
    violated = (xi + h < lb) or (xi + h > ub)
    fitting = abs(h) <= max(lower_dist, upper_dist)
    if violated and fitting:
        return -h
    if not fitting:
        return upper_dist if upper_dist >= lower_dist else -lower_dist
    return h


@njit
def _fd_grad(f, x, args, eps, lb, ub):
    """Gradient of a scalar `f(x, *args)` by forward differences.

    ``eps`` is a scalar or one step per coordinate; ``_eps_at`` reads it.
    """
    n = x.size
    f0 = _call_args(f, x, args)
    g = np.empty(n, np.float64)
    xt = x.copy()
    for i in range(n):
        h = _fd_h(x[i], lb[i], ub[i], _eps_at(eps, i))
        xt[i] = x[i] + h
        g[i] = (_call_args(f, xt, args) - f0) / (xt[i] - x[i])
        xt[i] = x[i]
    return g


@njit
def _fd_jac(fc, x, args, eps, lb, ub):
    """Jacobian of a vector `fc(x, *args)` by forward differences, (m, n).

    ``eps`` is a scalar or one step per coordinate; ``_eps_at`` reads it.
    """
    n = x.size
    f0 = _call_args(fc, x, args)
    m = f0.size
    j = np.empty((m, n), np.float64)
    xt = x.copy()
    for i in range(n):
        h = _fd_h(x[i], lb[i], ub[i], _eps_at(eps, i))
        xt[i] = x[i] + h
        d = xt[i] - x[i]
        f1 = _call_args(fc, xt, args)
        for k in range(m):
            j[k, i] = (f1[k] - f0[k]) / d
        xt[i] = x[i]
    return j


@njit
def _mixed_jac(fc, jac, k, x, args, eps, lb, ub):
    """Jacobian of a combined constraint family, ``(m, n)``.

    scipy appends the two spellings of one constraint kind, the scalar
    entries first, and attaches the analytic Jacobian to the vector one
    alone. Rows ``0`` to ``k - 1`` therefore come from forward
    differences and the rest from `jac`.

    The differenced Jacobian is computed for the whole family and its
    trailing rows are overwritten. The values discarded that way are the
    ones `jac` supplies, so the result is what scipy computes.
    """
    j = _fd_jac(fc, x, args, eps, lb, ub)
    a = _call_args(jac, x, args)
    if k + a.shape[0] != j.shape[0] or a.shape[1] != x.size:
        raise ValueError("fmin_slsqp: the constraint jacobian must be "
                         "(len(f_eqcons(x)), len(x0)) for the VECTOR "
                         "constraint alone, not for the scalar entries")
    for r in range(a.shape[0]):
        for c in range(x.size):
            j[k + r, c] = a[r, c]
    return j


@njit
def _mult_offset(n, la):
    """Index of the QP multiplier block in the packed working array.

    ``slsqp`` carves its one-dimensional ``w`` into blocks and hands the
    slices to ``slsqpb`` as separate arguments
    (``src/slsqp/04_slsqp_core.f90:245-254``)::

        im = 1 ; il = im + la ; ix = il + n1*n/2 + 1 ; ir = ix + n

    ``w(ir)`` is ``slsqpb``'s ``r``, declared ``dimension(m+n+n+2)`` at
    ``04_slsqp_core.f90:298``, whose leading ``m`` entries are the
    multipliers of the general constraints and whose trailing ``2n + 2``
    belong to the bound rows. The value returned is the 0-based offset.

    NOT ``w(1) ... w(m)``, which ``04_slsqp_core.f90:188-190`` calls "the
    multipliers associated with the general constraints". That block is
    ``mu``, the L1 merit penalty vector, and it is a different quantity:
    on the three-constraint fixture it reads ``[1.011, 0, 0]`` where the
    multipliers are ``[0.8, 0, 0]``.
    """
    return la + (n * (n + 1)) // 2 + n + 1


@njit
def _clip_box(x, lb, ub):
    """scipy clips x into the box before the first evaluation, and again
    before every callback (gh11403: SLSQP can exceed a bound by 1-2 ulp)."""
    out = x.copy()
    for i in range(x.size):
        if out[i] < lb[i]:
            out[i] = lb[i]
        if out[i] > ub[i]:
            out[i] = ub[i]
    return out


@njit
def _slsqp_msg(mode):
    """scipy's ``exit_modes`` table, verbatim."""
    if mode == -1:
        return "Gradient evaluation required (g & a)"
    elif mode == 0:
        return "Optimization terminated successfully"
    elif mode == 1:
        return "Function evaluation required (f & c)"
    elif mode == 2:
        return "More equality constraints than independent variables"
    elif mode == 3:
        return "More than 3*n iterations in LSQ subproblem"
    elif mode == 4:
        return "Inequality constraints incompatible"
    elif mode == 5:
        return "Singular matrix E in LSQ subproblem"
    elif mode == 6:
        return "Singular matrix C in LSQ subproblem"
    elif mode == 7:
        return "Rank-deficient equality constraint subproblem HFTI"
    elif mode == 8:
        return "Positive directional derivative for linesearch"
    elif mode == 9:
        return "Iteration limit reached"
    # scipy indexes the table, so a mode outside -1..9 is a KeyError naming
    # the mode. Reachable only from a backend neither implementation expects.
    raise KeyError(mode)


@njit
def _disp_slsqp(mode, fx, nit, nfev, njev):
    """scipy's ``iprint >= 1`` summary, verbatim.

    scipy prints ``fx`` and the three counters with a bare ``print``, so no
    format spec is involved and numba's own repr reproduces the text
    exactly. The 12-space indent is scipy's, and differs from the 9 spaces
    the `fmin` family uses.
    """
    print(_slsqp_msg(mode) + "    (Exit mode " + str(mode) + ")")
    print("            Current function value:", fx)
    print("            Iterations:", nit)
    print("            Function evaluations:", nfev)
    print("            Gradient evaluations:", njev)


#: scipy's per-major-iteration header, `_slsqp_py.py`
#: ``f"{'NIT':>5} {'FC':>5} {'OBJFUN':>16} {'GNORM':>16}"``.
_SLSQP_HEADER = "  NIT    FC           OBJFUN            GNORM"


def _slsqp_row(nit, nfev, fx, gn):
    """One row of scipy's ``iprint >= 2`` table.

    The columns carry a width and a precision, and numba types neither
    ``%`` nor ``format`` on a float, so the row is built in Python and
    reached through ``objmode``. The block sits on the ``iprint >= 2``
    branch, so a default call never enters it and never takes the GIL.
    """
    print(f"{nit:5d} {nfev:5d} {fx:16.6E} {gn:16.6E}")


@njit
def _slsqp_print_row(nit, nfev, fx, gn):
    """`_slsqp_row` reached from compiled code.

    The ``objmode`` block lives here rather than in the driver: lowering it
    pickles the enclosing function, and the driver holds the ctypes handle
    to the Fortran entry point, which raises
    ``ValueError: ctypes objects containing pointers cannot be pickled``.
    """
    with objmode():
        _slsqp_row(nit, nfev, fx, gn)


# --------------------------------------------------------------------------
# THE core.  Both entry points below only slice its return.
# --------------------------------------------------------------------------
@njit
def _slsqp_run(func, fprime, feq, jeq, fieq, jieq, x0, lb, ub, args, acc,
               maxiter, eps, has_gf, has_jeq, has_jieq, k_eq=0, k_ieq=0,
               iprint=0, cb=_cb_noop, use_cb=False):
    """Reverse-communication SLSQP driver.

    Returns ``(x, fx, majiter, mode, nfev, njev, g, multipliers)``.  Every
    callback slot is a concrete function; ``has_*`` select the analytic path
    over forward differences and are ordinary runtime booleans, since neither
    changes a type.

    ``multipliers`` is the ``m`` constraint multipliers of the last QP
    subproblem, read out of the packed working array at `_mult_offset`.
    Length ``m``, so ``(0,)`` when there is no constraint.

    ``k_eq`` and ``k_ieq`` are how many leading rows of the matching
    constraint family came from scipy's list-of-scalars spelling. A
    positive value under ``has_j*`` selects `_mixed_jac`, which differences
    those rows and takes the rest from the analytic Jacobian. Both default
    to ``0``, so a caller with one constraint spelling passes neither.

    ``iprint >= 2`` prints scipy's per-major-iteration table. It defaults
    to ``0``, silent.
    """
    n = x0.size
    x = _clip_box(np.ascontiguousarray(np.asarray(x0)).astype(np.float64), lb,
                  ub)

    # meq / mineq are READ FROM THE CALLBACKS, exactly as scipy reads them
    ceq0 = _call_args(feq, x, args)
    cineq0 = _call_args(fieq, x, args)
    meq = ceq0.size
    mineq_c = cineq0.size
    m = meq + mineq_c
    la = max(m, 1)
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = ((3 * n1 + m) * (n1 + 1) + (n1 - meq + 1) * (mineq + 2)
             + 2 * mineq + (n1 + mineq) * (n1 - meq) + 2 * meq + n1
             + ((n + 1) * n) // 2 + 2 * m + 3 * n + 3 * n1 + 1)

    # NaN is the Fortran's ignore-this-bound sentinel, and it is what scipy
    # writes ("Mark infinite bounds with nans; the C code expects this").
    # `enforce_bounds` and `lsq`'s constraint assembly both test
    # `ieee_is_nan` first, so a NaN bound is dropped rather than enforced.
    # A large finite sentinel is NOT equivalent: the guard beside the NaN
    # test is `xl(i) <= -infbnd` with `infbnd = huge(1.0d0)`, so any value
    # short of huge survives it and enters the QP as a real constraint row.
    xl = np.empty(n, np.float64)
    xu = np.empty(n, np.float64)
    for i in range(n):
        xl[i] = lb[i] if np.isfinite(lb[i]) else np.nan
        xu[i] = ub[i] if np.isfinite(ub[i]) else np.nan

    m_ = np.array(m, np.int32)
    meq_ = np.array(meq, np.int32)
    la_ = np.array(la, np.int32)
    n_ = np.array(n, np.int32)
    f_ = np.zeros(1, np.float64)
    c_buf = np.zeros(la, np.float64)
    g_buf = np.zeros(n + 1, np.float64)
    a_buf = np.zeros(la * (n + 1), np.float64)      # column-major (la, n+1)
    acc_ = np.array(acc, np.float64)
    iter_ = np.zeros(1, np.int32)
    # majiter is BOTH the limit going in and the iteration count coming out.
    # scipy passes the limit through unchanged: `_slsqp_py.py` builds
    # `{"itermax": int(maxiter)}`.  scipy <= 1.15.3 passed `maxiter - 1` to the
    # Fortran solver; 1.18 replaced that solver with the `._slsqplib` backend
    # and dropped the decrement.  scijit targets 1.18.
    iter_[0] = maxiter
    mode_ = np.zeros(1, np.int32)
    w = np.zeros(len_w, np.float64)
    len_w_ = np.array(len_w, np.int32)
    sdat_r = np.zeros(10, np.float64)
    sdat_i = np.zeros(8, np.int32)
    ldat_r = np.zeros(18, np.float64)
    alphamin = np.array(0.1, np.float64)
    alphamax = np.array(1.0, np.float64)
    tolf = np.array(-1.0, np.float64)               # disabled extras
    toldf = np.array(-1.0, np.float64)
    toldx = np.array(-1.0, np.float64)
    max_iter_ls = np.array(0, np.int32)
    nnls_mode = np.array(1, np.int32)

    nfev = 0
    njev = 0

    # mode 0 on entry: evaluate f, c, g and a once
    xc = _clip_box(x, lb, ub)
    f_[0] = _call_args(func, xc, args)
    nfev += 1
    for i in range(meq):
        c_buf[i] = ceq0[i]
    for i in range(mineq_c):
        c_buf[meq + i] = cineq0[i]
    if has_gf:
        gv = _call_args(fprime, xc, args)
        njev += 1
    else:
        gv = _fd_grad(func, xc, args, eps, lb, ub)
        njev += 1
        nfev += n
    if gv.size != n:
        raise ValueError("fmin_slsqp: the gradient length does not match "
                         "len(x0)")
    for j in range(n):
        g_buf[j] = gv[j]
    g_buf[n] = 0.0
    if meq > 0:
        if has_jeq and k_eq > 0:
            aeq = _mixed_jac(feq, jeq, k_eq, xc, args, eps, lb, ub)
        elif has_jeq:
            aeq = _call_args(jeq, xc, args)
        else:
            aeq = _fd_jac(feq, xc, args, eps, lb, ub)
        if aeq.shape[0] != meq or aeq.shape[1] != n:
            raise ValueError("fmin_slsqp: the equality-constraint jacobian "
                             "must be (len(f_eqcons(x)), len(x0))")
        for i in range(meq):
            for j in range(n):
                a_buf[i + j * la] = aeq[i, j]
    if mineq_c > 0:
        if has_jieq and k_ieq > 0:
            ain = _mixed_jac(fieq, jieq, k_ieq, xc, args, eps, lb, ub)
        elif has_jieq:
            ain = _call_args(jieq, xc, args)
        else:
            ain = _fd_jac(fieq, xc, args, eps, lb, ub)
        if ain.shape[0] != mineq_c or ain.shape[1] != n:
            raise ValueError("fmin_slsqp: the inequality-constraint jacobian "
                             "must be (len(f_ieqcons(x)), len(x0))")
        for i in range(mineq_c):
            for j in range(n):
                a_buf[meq + i + j * la] = ain[i, j]

    if iprint >= 2:
        print(_SLSQP_HEADER)
    iter_prev = 0
    # Drop whatever an earlier solve on this thread left in the halt
    # slot, so the first read below cannot see it.  A standalone entry
    # point does not clear it on the way out: only `minimize` takes it.
    _cb_halt_clear()
    while True:
        _slsqp(m_.ctypes.data, meq_.ctypes.data, la_.ctypes.data,
               n_.ctypes.data, x.ctypes.data, xl.ctypes.data,
               xu.ctypes.data, f_.ctypes.data, c_buf.ctypes.data,
               g_buf.ctypes.data, a_buf.ctypes.data, acc_.ctypes.data,
               iter_.ctypes.data, mode_.ctypes.data, w.ctypes.data,
               len_w_.ctypes.data, sdat_r.ctypes.data,
               sdat_i.ctypes.data, ldat_r.ctypes.data,
               alphamin.ctypes.data, alphamax.ctypes.data,
               tolf.ctypes.data, toldf.ctypes.data, toldx.ctypes.data,
               max_iter_ls.ctypes.data, nnls_mode.ctypes.data)

        # scipy evaluates, then calls the solver, then prints, so its `fx`
        # and `nfev` are one evaluation AHEAD of this loop's at the same
        # point while its `g` is not. Printing after the branches below puts
        # all three where scipy has them, and it is measured: the four rows
        # of its table are reproduced character for character.
        bumped = iter_[0] > iter_prev
        iter_prev = iter_[0]
        pending = iprint >= 2 and bumped
        done = False

        if mode_[0] == 1:                   # need f and c
            xc = _clip_box(x, lb, ub)
            f_[0] = _call_args(func, xc, args)
            nfev += 1
            ce = _call_args(feq, xc, args)
            ci = _call_args(fieq, xc, args)
            for i in range(meq):
                c_buf[i] = ce[i]
            for i in range(mineq_c):
                c_buf[meq + i] = ci[i]
        elif mode_[0] == -1:                # need g and a
            xc = _clip_box(x, lb, ub)
            if has_gf:
                gv = _call_args(fprime, xc, args)
            else:
                gv = _fd_grad(func, xc, args, eps, lb, ub)
                nfev += n
            njev += 1
            for j in range(n):
                g_buf[j] = gv[j]
            g_buf[n] = 0.0
            if meq > 0:
                if has_jeq and k_eq > 0:
                    aeq = _mixed_jac(feq, jeq, k_eq, xc, args, eps, lb, ub)
                elif has_jeq:
                    aeq = _call_args(jeq, xc, args)
                else:
                    aeq = _fd_jac(feq, xc, args, eps, lb, ub)
                for i in range(meq):
                    for j in range(n):
                        a_buf[i + j * la] = aeq[i, j]
            if mineq_c > 0:
                if has_jieq and k_ieq > 0:
                    ain = _mixed_jac(fieq, jieq, k_ieq, xc, args, eps, lb, ub)
                elif has_jieq:
                    ain = _call_args(jieq, xc, args)
                else:
                    ain = _fd_jac(fieq, xc, args, eps, lb, ub)
                for i in range(mineq_c):
                    for j in range(n):
                        a_buf[meq + i + j * la] = ain[i, j]
        else:
            done = True

        # scipy calls the callback on a MAJOR-iteration increment, after the
        # mode branch, ahead of the table row and ahead of its own exit test,
        # and leaves the loop on a halt with `mode` untouched.
        if bumped and use_cb:
            cb(x.copy(), f_[0])
            if _cb_halt_get():
                break
        if pending:
            gn = 0.0
            for j in range(n + 1):
                gn += g_buf[j] * g_buf[j]
            _slsqp_print_row(np.int64(iter_[0]), np.int64(nfev),
                             np.float64(f_[0]), np.sqrt(gn))
        if done:
            break

    g_out = np.empty(n, np.float64)
    for j in range(n):
        g_out[j] = g_buf[j]
    # The solver increments before it tests the limit
    # (src/slsqp/04_slsqp_core.f90:467-469 is `iter = iter + 1` then
    # `if ( iter>itermx ) return`), so an exit on the iteration limit reports
    # itermx + 1 while only itermx iterations ran.  Clamping to the limit
    # reports iterations PERFORMED, which is what the counter is documented to
    # hold.  A converged exit leaves the loop elsewhere and is already correct,
    # so this is a no-op on every other path.
    nit = iter_[0]
    if nit > maxiter:
        nit = maxiter
    # Copied out rather than returned as a view, so the whole `len_w`
    # buffer is not kept alive by an `m`-element result.
    ir0 = _mult_offset(np.int64(n), np.int64(la))
    mult = w[ir0:ir0 + m].copy()
    return x, f_[0], nit, mode_[0], nfev, njev, g_out, mult


@njit
def _slsqp_bounds(n, bounds_lo, bounds_hi):
    """lower/upper vectors -> the +-inf arrays the driver wants."""
    lb = np.empty(n, np.float64)
    ub = np.empty(n, np.float64)
    has_l = bounds_lo.size == n
    has_u = bounds_hi.size == n
    if bounds_lo.size != 0 and not has_l:
        raise ValueError(_BOUNDS_LEN_MSG)
    if bounds_hi.size != 0 and not has_u:
        raise ValueError(_BOUNDS_LEN_MSG)
    bad = False
    for i in range(n):
        lb[i] = bounds_lo[i] if has_l else -np.inf
        ub[i] = bounds_hi[i] if has_u else np.inf
        if lb[i] > ub[i]:
            bad = True
    if bad:
        # scipy renders the whole comparison mask into the message, one
        # entry per bound pair: "... in bounds True, False."
        s = ""
        for i in range(n):
            if i > 0:
                s = s + ", "
            s = s + ("True" if lb[i] > ub[i] else "False")
        raise ValueError("SLSQP Error: lb > ub in bounds " + s + ".")
    return lb, ub


def fmin_slsqp(func, x0, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None,
               bounds=(), fprime=None, fprime_eqcons=None,
               fprime_ieqcons=None, args=(), iter=100, acc=1.0e-6, iprint=1,
               disp=None, full_output=0, epsilon=_EPSILON, callback=None):
    """Minimize a function subject to equality, inequality and bound constraints.

    SLSQP -- sequential least-squares programming -- replaces the problem at
    each iteration with a quadratic model under linearized constraints, and
    solves that subproblem as a least-squares fit.

    Callable from Python and from inside ``@njit``. Both entries run the same
    compiled driver. Where a refused argument is reported differs: from
    Python as a ``ValueError`` while the call runs, from ``@njit`` as a
    ``TypingError`` while it compiles, both naming the argument.

    Parameters
    ----------
    func : callable
        A plain ``@njit`` ``f(x, *args) -> value``.
    x0 : array_like
        Initial guess. Any rank is flattened, and it is clipped into `bounds`
        before the first evaluation.
    eqcons : tuple of callables, optional
        Equality constraints ``g(x, *args) -> float`` or
        ``g(x, *args) -> array``, feasible where the result is ``0``. An entry
        returning an array contributes one constraint per element. Every entry
        is a plain ``@njit`` function. Empty (default) is no equality
        constraint. Inside ``@njit`` the tuple is a compile-time constant: its
        length and the functions in it decide what is compiled. Given together
        with `f_eqcons` the two are APPENDED, these entries first, and these
        entries are differenced even when `f_eqcons` carries an analytic
        Jacobian.
    f_eqcons : callable or None, optional
        ``ceq(x, *args) -> array``, all equality constraints at once, feasible
        where the result is ``0``. Compile-time constant inside ``@njit``.
    ieqcons : tuple of callables, optional
        Inequality constraints, feasible where the result is ``>= 0``.
        Otherwise as `eqcons`, and appended ahead of `f_ieqcons` the same way.
    f_ieqcons : callable or None, optional
        ``cineq(x, *args) -> array``, all inequality constraints at once,
        feasible where the result is ``>= 0``.
    bounds : sequence of (min, max) pairs, (n, 2) ndarray, or (), optional
        Every spelling :func:`~scijit.optimize.fmin_l_bfgs_b` accepts. Empty
        (default) and ``None`` are both unbounded, where `fmin_l_bfgs_b`
        takes ``None`` alone and refuses an empty sequence. A length other
        than ``len(x0)`` is an `IndexError` here and a `ValueError` there.
        The pairs are read by transposition, so each row must hold exactly
        two entries.
    fprime : callable or None, optional
        ``grad(x, *args) -> array``. ``None`` (default) computes the gradient
        by forward differences with absolute step `epsilon`.
    fprime_eqcons, fprime_ieqcons : callable or None, optional
        ``(m, n)`` Jacobian of `f_eqcons` / `f_ieqcons` alone, not of the
        `eqcons` / `ieqcons` entries appended before it. ``None`` (default)
        computes it by forward differences. Ignored when the matching vector
        function is absent.
    args : tuple or ndarray, optional
        Extra values for every callback. A tuple is unpacked into their
        argument lists, ``f(x, *args)``; the entries may be of any types the
        callbacks accept. ``()`` (default) calls ``f(x)``.
    iter : int, optional
        Maximum major iterations. Default 100.
    acc : float, optional
        Requested accuracy. Default 1e-6.
    iprint : int, optional
        ``1`` (default) prints the exit summary. ``0`` and below are silent.
        ``2`` and above add a per-iteration table, one row per major
        iteration, ahead of the summary.
    disp : int or None, optional
        Overrides `iprint` when it is not ``None``.
    full_output : bool, optional
        ``False`` (default) returns `x` alone. ``True`` returns the 5-tuple.
        Compile-time constant inside ``@njit``: it selects the return type.
    epsilon : float or ndarray, optional
        Absolute step for every finite-difference path. Default ``sqrt(eps)``.
        An array gives one step per coordinate; a size-1 array applies to
        every coordinate.
    callback : callable, optional
        Called once per major iteration, as ``callback(xk)`` or
        ``callback(intermediate_result)``. Two spellings are served: a plain
        Python callable, which halts the solve when it raises
        ``StopIteration``, and a numba ``@njit`` ``callback(xk)``, which halts
        when it raises any exception. See Notes.

    Returns
    -------
    out : ndarray, or tuple
        `x` alone, or with `full_output` the 5-tuple ``(x, fx, its, imode,
        smode)``: the minimizer, its objective value, the major-iteration
        count, the exit mode and the message for it. ``imode == 0`` is
        success, ``8`` a bad line search and ``9`` the iteration limit.

    Raises
    ------
    IndexError
        If the length of `bounds` is neither 0 nor ``len(x0)``.
    ValueError
        If `callback` is a ``@cfunc``, a raw function address or a
        non-callable; if an entry of `eqcons` or `ieqcons` is a plain Python
        callable; if a row of `bounds` does not hold two entries; if any
        lower bound exceeds its upper bound; if `epsilon` does not broadcast
        against `x0` while a forward difference is taken; or if a gradient or
        constraint Jacobian comes back with the wrong shape.
    TypeError
        If an entry of `eqcons` or `ieqcons` is not callable, or if a row of
        `bounds` is a number rather than a pair.

    See Also
    --------
    scipy.optimize.fmin_slsqp : The scipy routine this mirrors.
    scijit.optimize.minimize : Dispatches here for ``method='SLSQP'``.
    scijit.optimize.fmin_cobyla : Derivative-free, nonlinear constraints.
    scijit.optimize.fmin_l_bfgs_b : Bounds only, no general constraints.

    Notes
    -----
    `eqcons` and `ieqcons` are a TUPLE of ``@njit`` functions, where scipy
    takes a list of Python callables. The tuple is combined into one
    vector-valued function when the call compiles and handed to the same slot
    `f_eqcons` and `f_ieqcons` fill, so the two spellings run the same code
    and their Jacobians come from the same forward differences.

    An ndarray or a list `args` reaches every callback as ONE argument,
    ``f(x, args)``, where scipy unpacks those two element by element. The
    arity of a compiled call is fixed when it compiles, so a sequence whose
    length is known only at run time cannot be unpacked.

    Whether `f_eqcons`, `f_ieqcons`, `fprime`, `fprime_eqcons` and
    `fprime_ieqcons` are ``None`` is a compile-time constant inside ``@njit``,
    since numba has no ``None``-able function argument. `full_output` is
    likewise a compile-time literal, because it selects the return type.

    A Python `callback` is reached from compiled code through a module-level
    slot and a ``numba.objmode`` block, so it takes the GIL and pays an
    interpreter round trip once per major iteration. It is also not
    ``prange``-safe, because the slot is module state that two concurrent
    solves share. Inside
    ``@njit`` only the ``@njit`` spelling is accepted, since a Python callable
    cannot cross into compiled code as an argument.

    The two spellings differ in what halts the solve. The Python one halts on
    ``StopIteration``, which is scipy's contract; anything else it raises
    reaches the caller. The ``@njit`` one halts on ANY exception, because
    numba matches no exception class: ``except StopIteration`` does not
    compile.

    `iprint` prints scipy's exit summary at its scipy default of ``1``, so a
    bare call writes to stdout.

    A `bounds` LONGER than `x0` raises the same ``IndexError`` as a shorter
    one. scipy reaches an ``np.clip`` before its own length test in that
    direction and surfaces a numpy broadcast ``ValueError`` instead.

    A gradient or constraint Jacobian of the wrong shape raises. numba applies
    no bounds checking, so an unchecked wrong shape would read past the end of
    the buffer rather than fail.

    Safe to call from a ``numba.prange`` loop with `callback` at ``None`` or
    an ``@njit`` function: the solver is reverse communication, so its state
    lives in caller-owned arrays. A Python `callback` is not, and serializes
    the loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html

    Examples
    --------
    Minimize ``x0**2 + x1**2`` subject to ``x0 + x1 >= 1``:

    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import fmin_slsqp
    >>> @njit
    ... def obj(x):
    ...     return x[0] ** 2 + x[1] ** 2
    >>> @njit
    ... def cineq(x):
    ...     return np.array([x[0] + x[1] - 1.0])
    >>> @njit
    ... def run():
    ...     return fmin_slsqp(obj, np.array([2.0, 1.0]), f_ieqcons=cineq)
    >>> run()
    Optimization terminated successfully    (Exit mode 0)
                Current function value: 0.5
                Iterations: 4
                Function evaluations: 12
                Gradient evaluations: 4
    array([0.5, 0.5])

    `iprint` defaults to scipy's ``1``, so the exit summary above is what a
    bare call writes to stdout. With `full_output`, which must be a literal
    inside ``@njit``:

    >>> @njit
    ... def run_full():
    ...     return fmin_slsqp(obj, np.array([2.0, 1.0]), f_ieqcons=cineq,
    ...                       full_output=True)
    >>> x, fx, its, imode, smode = run_full()
    Optimization terminated successfully    (Exit mode 0)
                Current function value: 0.5
                Iterations: 4
                Function evaluations: 12
                Gradient evaluations: 4
    >>> fx, its, imode
    (0.5, 4, 0)
    >>> smode
    'Optimization terminated successfully'
    """
    cbf, use_cb, pycb = _cb_resolve('fmin_slsqp', callback)
    # scipy attaches fprime_eqcons to the f_eqcons constraint dict and to
    # nothing else, so without f_eqcons it is IGNORED. Measured on scipy
    # 1.18: a deliberately wrong fprime_eqcons with no f_eqcons changes no
    # answer.
    use_jeq = fprime_eqcons is not None and f_eqcons is not None
    use_jieq = fprime_ieqcons is not None and f_ieqcons is not None
    # scipy APPENDS the two spellings of each constraint kind, scalars
    # first, rather than preferring either. Same order here.
    k_eq = len(eqcons)
    k_ieq = len(ieqcons)
    eq_fns = _cons_dispatchers(eqcons) if k_eq else ()
    ieq_fns = _cons_dispatchers(ieqcons) if k_ieq else ()
    if k_eq:
        f_eqcons = _combine_cons(eq_fns, f_eqcons)
    if k_ieq:
        f_ieqcons = _combine_cons(ieq_fns, f_ieqcons)
    x = np.ascontiguousarray(np.atleast_1d(
        np.asarray(x0, dtype=np.float64))).ravel().copy()
    n = x.size

    feq = _no_c if f_eqcons is None else f_eqcons
    fieq = _no_c if f_ieqcons is None else f_ieqcons
    jeq = fprime_eqcons if use_jeq else _no_j
    jieq = fprime_ieqcons if use_jieq else _no_j
    gf = _no_g if fprime is None else fprime

    if bounds is None or len(bounds) == 0:
        lo = hi = np.zeros(0)
    else:
        # scipy transposes the pairs into lower and upper BEFORE it counts
        # them, so a malformed pair is reported ahead of a wrong count
        lo, hi = _as_bounds(bounds)
        if lo.size != n:
            # and it sizes the constraint block between the two, by calling
            # every constraint once at x0, so a constraint that raises there
            # is reported ahead of the count
            _call_args(feq, x, args)
            _call_args(fieq, x, args)
            raise IndexError(_SLSQP_BOUNDS_LEN_MSG)
    lb, ub = _slsqp_bounds(n, lo, hi)

    # `epsilon` is read only where a forward difference is taken: for the
    # objective gradient without `fprime`, and for a constraint block no
    # analytic Jacobian covers
    if (fprime is None
            or (f_eqcons is not None and (not use_jeq or k_eq))
            or (f_ieqcons is not None and (not use_jieq or k_ieq))):
        _check_eps(np.asarray(epsilon).size, np.asarray(epsilon).ndim, n)
    ea = np.asarray(epsilon, dtype=np.float64)
    epsilon = float(ea) if ea.ndim == 0 else np.ascontiguousarray(ea)
    # scipy resolves `disp` over `iprint` BEFORE the run, because the table
    # header is printed there
    ip = iprint if disp is None else int(disp)
    # how many LEADING rows the analytic Jacobian does not describe. An entry
    # may contribute more than one row, so the count comes from the entries.
    n_eq = (_call_args(_combine_nlead(eq_fns), x, args)
            if (use_jeq and k_eq) else 0)
    n_ieq = (_call_args(_combine_nlead(ieq_fns), x, args)
             if (use_jieq and k_ieq) else 0)
    prev = _cb_install(pycb)
    try:
        r = _slsqp_run(func, gf, feq, jeq, fieq, jieq, x, lb, ub, args, acc,
                       iter, epsilon, fprime is not None, use_jeq, use_jieq,
                       n_eq, n_ieq, ip, cbf, use_cb)
    finally:
        _cb_release(prev)
    if ip >= 1:
        _disp_slsqp(r[3], r[1], r[2], r[4], r[5])
    if not full_output:
        return r[0]
    return r[0], r[1], int(r[2]), int(r[3]), _slsqp_msg(r[3])


@overload(fmin_slsqp, prefer_literal=True)
def _fmin_slsqp_ovl(func, x0, eqcons=(), f_eqcons=None, ieqcons=(),
                    f_ieqcons=None, bounds=(), fprime=None,
                    fprime_eqcons=None, fprime_ieqcons=None, args=(),
                    iter=100, acc=1.0e-6, iprint=1, disp=None, full_output=0,
                    epsilon=_EPSILON, callback=None):
    """@njit implementation of `fmin_slsqp`, resolved at compile time.

    ``full_output`` picks the return type, and which of the five
    callbacks are present decides which slots take a placeholder and
    which take the caller's function -- both are types, so neither can be
    settled at runtime. Returning ``None`` declines the call, which numba
    reports as a TypingError naming the argument that could not be
    served.
    """
    fo = _lit_bool(full_output)
    if fo is None:
        return None                     # runtime flag -> TypingError
    CBF, USE_CB = _cb_resolve_ty('fmin_slsqp', callback)
    has_eq = not _is_none(f_eqcons)
    has_ieq = not _is_none(f_ieqcons)
    has_gf = not _is_none(fprime)
    has_jeq = not _is_none(fprime_eqcons)
    has_jieq = not _is_none(fprime_ieqcons)
    # scipy's LIST-of-scalars spelling.  The tuple's length and the
    # dispatchers in it are both known here, so the combined vector-valued
    # function is built now and reaches `impl` as a constant.  A tuple
    # given together with `f_eqcons` is APPENDED to it, scalars first,
    # which is what scipy does with the two spellings.
    eq_fns = _cons_arg_type(eqcons)
    ieq_fns = _cons_arg_type(ieqcons)
    if eq_fns is None or ieq_fns is None:
        return None
    # scipy attaches fprime_eqcons to the f_eqcons constraint dict alone, so
    # without f_eqcons it is IGNORED. A scalar list given as well contributes
    # LEADING rows the analytic Jacobian does not describe; `k_eq` tells the
    # core how many of them to difference.
    use_jeq = has_jeq and has_eq
    use_jieq = has_jieq and has_ieq
    k_eq = len(eq_fns) if use_jeq else 0
    k_ieq = len(ieq_fns) if use_jieq else 0
    # an entry may contribute more than one row, so the leading-row count is
    # a run-time fact and the counter is only reached where it is read
    nlead_eq = _combine_nlead(eq_fns) if k_eq else _no_nlead
    nlead_ieq = _combine_nlead(ieq_fns) if k_ieq else _no_nlead
    eq_slot = _no_c
    if eq_fns:
        eq_slot = _combine_cons(eq_fns,
                                f_eqcons.dispatcher if has_eq else None)
    ieq_slot = _no_c
    if ieq_fns:
        ieq_slot = _combine_cons(ieq_fns,
                                 f_ieqcons.dispatcher if has_ieq else None)
    # the combined function REPLACES the argument in its slot
    use_eq_arg = has_eq and not eq_fns
    use_ieq_arg = has_ieq and not ieq_fns
    # bounds=() is scipy's "unconstrained".  An OMITTED default arrives as
    # types.Omitted(()), not as a BaseTuple, so both spellings are tested.
    _bt = bounds.value if isinstance(bounds, types.Omitted) else bounds
    no_bounds = (_is_none(bounds) or _bt == ()
                 or (isinstance(_bt, types.BaseTuple) and len(_bt) == 0))
    # scipy: `if disp is not None: iprint = disp`, then it prints at
    # iprint >= 1.  Whether `disp` was given is a type question, so it is
    # settled here and the comparison itself stays a runtime one.
    disp_none = _is_none(disp)
    eps_arr = isinstance(epsilon, types.Array)
    # `epsilon` is read only where a forward difference is taken, so nothing
    # about it is tested when `fprime` is given and every constraint block
    # carries an analytic Jacobian
    eps_check = eps_arr and (
        not has_gf
        or ((has_eq or eq_fns) and (not use_jeq or k_eq))
        or ((has_ieq or ieq_fns) and (not use_jieq or k_ieq)))

    def impl(func, x0, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None,
             bounds=(), fprime=None, fprime_eqcons=None, fprime_ieqcons=None,
             args=(), iter=100, acc=1.0e-6, iprint=1, disp=None,
             full_output=0, epsilon=_EPSILON, callback=None):
        x = np.ascontiguousarray(np.asarray(x0)).ravel().astype(np.float64)
        n = x.size
        if use_eq_arg:
            feq = f_eqcons
        else:
            feq = eq_slot
        if use_ieq_arg:
            fieq = f_ieqcons
        else:
            fieq = ieq_slot
        if no_bounds:
            lo = np.zeros(0, np.float64)
            hi = np.zeros(0, np.float64)
        else:
            # scipy transposes the pairs into lower and upper BEFORE it
            # counts them; neither the count nor the WIDTH of a pair is in
            # an array TYPE, so both are run-time tests
            lo, hi = _split_bounds(bounds, n)
            if lo.size != 0 and lo.size != n:
                # and it sizes the constraint block between the two, by
                # calling every constraint once at x0, so a constraint that
                # raises there is reported ahead of the count
                _call_args(feq, x, args)
                _call_args(fieq, x, args)
                raise IndexError(_SLSQP_BOUNDS_LEN_MSG)
        if eps_check:
            _check_eps(epsilon.size, epsilon.ndim, n)
        lb, ub = _slsqp_bounds(n, lo, hi)
        if use_jeq:
            jeq = fprime_eqcons
        else:
            jeq = _no_j
        if use_jieq:
            jieq = fprime_ieqcons
        else:
            jieq = _no_j
        if has_gf:
            gf = fprime
        else:
            gf = _no_g
        if disp_none:
            ip = np.int64(iprint)
        else:
            ip = np.int64(disp)
        r = _slsqp_run(func, gf, feq, jeq, fieq, jieq, x, lb, ub, args, acc,
                       iter, epsilon, has_gf, use_jeq, use_jieq,
                       _call_args(nlead_eq, x, args),
                       _call_args(nlead_ieq, x, args), ip, CBF, USE_CB)
        if ip >= 1:
            _disp_slsqp(r[3], r[1], r[2], r[4], r[5])
        if fo:
            return r[0], r[1], r[2], r[3], _slsqp_msg(r[3])
        return r[0]
    return impl


@njit
def _minimize_slsqp_fg(fg, x0, lower, upper, args, acc, maxiter,
                       cb=_cb_noop, use_cb=False):
    """Bounds-only SLSQP driven by an fg-style function (m = 0).

    Backend for minimize(method='SLSQP'): the unified front-end takes
    one (x, args) -> (f, g) function like L-BFGS-B; this loop feeds it
    to the constraint-free SLSQP protocol. For actual constraints use
    `fmin_slsqp`, which takes the constraint functions and their
    Jacobians as separate arguments.

    Returns ``(x, fx, success, mode, majiter, g, nfev, njev,
    multipliers)``. The last is read out of the packed working array at
    `_mult_offset`, the same slice `_slsqp_run` takes; ``m`` is 0 on this
    route, so its length is 0 by construction rather than by assertion.
    """
    n = np.int32(x0.size)
    if lower.size != 0 and lower.size != n:
        raise ValueError("minimize: lower must be empty or len(x0)")
    if upper.size != 0 and upper.size != n:
        raise ValueError("minimize: upper must be empty or len(x0)")
    la = np.int32(1)
    n1 = np.int32(n + 1)
    mineq = np.int32(n1 + n1)                       # m=0, meq=0
    len_w = np.int32((3 * n1) * (n1 + 1)
                     + (n1 + 1) * (mineq + 2) + 2 * mineq
                     + (n1 + mineq) * n1 + n1
                     + ((n + 1) * n) // 2 + 3 * n + 3 * n1 + 1)

    m_ = np.zeros(1, np.int32)
    meq_ = np.zeros(1, np.int32)
    la_ = np.array(1, np.int32)
    n_ = np.array(n, np.int32)
    x = np.ascontiguousarray(x0.astype(np.float64)).copy()

    xl = np.zeros(n, np.float64)
    xu = np.zeros(n, np.float64)
    has_l = lower.size == n
    has_u = upper.size == n
    for i in range(n):
        lo = lower[i] if has_l else -np.inf
        hi = upper[i] if has_u else np.inf
        xl[i] = lo if np.isfinite(lo) else np.nan
        xu[i] = hi if np.isfinite(hi) else np.nan

    # Clip the starting point into the box before the first evaluation.
    # SLSQP maintains feasibility as an invariant: the QP subproblem bounds
    # the step by xl - x <= d <= xu - x, so an x inside the box admits d = 0,
    # the QP optimum is a descent direction, and x + s stays inside. Only the
    # INITIAL point can break that. Started outside, all three exit paths
    # report mode = 0 -- two of them returning an infeasible x, the third a
    # feasible point that is not the minimum. Clipping against xl/xu rather
    # than lower/upper is deliberate: xl/xu are always length n and carry the
    # NaN sentinel for an absent bound, and both comparisons against NaN are
    # False, so an absent bound is a no-op clip.
    for i in range(n):
        if x[i] < xl[i]:
            x[i] = xl[i]
        elif x[i] > xu[i]:
            x[i] = xu[i]

    f_ = np.zeros(1, np.float64)
    c_buf = np.zeros(1, np.float64)
    g_buf = np.zeros(n + 1, np.float64)
    a_buf = np.zeros(la * (n + 1), np.float64)
    acc_ = np.array(acc, np.float64)
    iter_ = np.zeros(1, np.int32)
    iter_[0] = maxiter
    mode_ = np.zeros(1, np.int32)
    w = np.zeros(len_w, np.float64)
    len_w_ = np.array(len_w, np.int32)
    sdat_r = np.zeros(10, np.float64)
    sdat_i = np.zeros(8, np.int32)
    ldat_r = np.zeros(18, np.float64)
    alphamin = np.array(0.1, np.float64)
    alphamax = np.array(1.0, np.float64)
    tolf = np.array(-1.0, np.float64)
    toldf = np.array(-1.0, np.float64)
    toldx = np.array(-1.0, np.float64)
    max_iter_ls = np.array(0, np.int32)
    nnls_mode = np.array(1, np.int32)

    # An fg-style objective returns f and g from one call, and the RC
    # protocol asks for them separately: mode = 1 wants f, mode = -1 wants g,
    # and mode = -1 almost always follows a mode = 1 at the SAME x. Caching
    # the last (x, f, g) turns that pair into one call. Without the cache the
    # driver evaluated the objective twice per accepted step -- 80 calls where
    # 47 sufficed on a 2-D Rosenbrock.
    #
    # nfev counts calls that actually reach `fg`; njev counts gradient
    # requests, served from the cache or not.
    nfev = 0
    njev = 0
    x_memo = np.empty(n, np.float64)
    g_memo = np.zeros(n, np.float64)
    f_memo = 0.0

    fv, gv = _call_args(fg, x, args)
    nfev += 1
    njev += 1
    if len(gv) != n:
        raise ValueError("minimize: fg returned a gradient whose "
                         "length != len(x)")
    f_[0] = fv
    f_memo = fv
    for j in range(n):
        g_buf[j] = gv[j]
        g_memo[j] = gv[j]
        x_memo[j] = x[j]
    g_buf[n] = 0.0

    iter_prev = 0
    # Drop whatever an earlier solve on this thread left in the halt
    # slot, so the first read below cannot see it.  A standalone entry
    # point does not clear it on the way out: only `minimize` takes it.
    _cb_halt_clear()
    while True:
        _slsqp(m_.ctypes.data, meq_.ctypes.data, la_.ctypes.data,
               n_.ctypes.data, x.ctypes.data, xl.ctypes.data,
               xu.ctypes.data, f_.ctypes.data, c_buf.ctypes.data,
               g_buf.ctypes.data, a_buf.ctypes.data, acc_.ctypes.data,
               iter_.ctypes.data, mode_.ctypes.data, w.ctypes.data,
               len_w_.ctypes.data, sdat_r.ctypes.data,
               sdat_i.ctypes.data, ldat_r.ctypes.data,
               alphamin.ctypes.data, alphamax.ctypes.data,
               tolf.ctypes.data, toldf.ctypes.data, toldx.ctypes.data,
               max_iter_ls.ctypes.data, nnls_mode.ctypes.data)

        bumped = iter_[0] > iter_prev
        iter_prev = iter_[0]
        done = False
        if mode_[0] == 1 or mode_[0] == -1:
            fresh = False
            for j in range(n):
                if x[j] != x_memo[j]:
                    fresh = True
                    break
            if fresh:
                fv, gv = _call_args(fg, x, args)
                nfev += 1
                f_memo = fv
                for j in range(n):
                    g_memo[j] = gv[j]
                    x_memo[j] = x[j]
            if mode_[0] == 1:              # need f
                f_[0] = f_memo
            else:                          # need g
                njev += 1
                for j in range(n):
                    g_buf[j] = g_memo[j]
        else:
            done = True
        # scipy calls the callback on a major-iteration increment, ahead of
        # its own exit test, and leaves the loop on a halt with `mode`
        # untouched.
        if bumped and use_cb:
            cb(x.copy(), f_[0])
            if _cb_halt_get():
                break
        if done:
            break

    ir0 = _mult_offset(np.int64(n), np.int64(la))
    mult = w[ir0:ir0 + m_[0]].copy()
    return (x, f_[0], mode_[0] == 0, mode_[0], iter_[0],
            np.ascontiguousarray(g_buf[:n]).copy(), nfev, njev, mult)
