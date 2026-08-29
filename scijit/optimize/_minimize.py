"""Unified minimization entry points.

`minimize` carries scipy's signature and reaches eleven algorithms. Seven
answer to scipy's own `method` string. The other four are the remaining
PRIMA derivative-free solvers, which scipy does not expose.

The objective is a plain ``@njit`` function called ``fun(x, *args)`` for
every method. Where the backend needs a C function pointer, the ``@cfunc``
adapter is built when the call compiles, so no ``.address`` is required.

`tol` maps to each tolerance the chosen method has. ``tol=None`` and
``maxiter=0`` keep the method's own defaults.

Routine listings
----------------
minimize
    Eleven algorithms behind scipy's signature, the four PRIMA
    derivative-free solvers among them.
"""
import warnings
from collections import namedtuple

import numpy as np
from numba import njit, objmode, typeof, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.extending import overload
from scijitclass import scijitclass

from ._lbfgsb import (fmin_l_bfgs_b, _is_none, _lbfgsb_run, _lbfgsb_bounds,
                      _lbfgsb_msg, _lbfgsb_flag, _clip_to_box, _ev_fg,
                      _ev_split, _ev_approx)
from ._minpack import (_arg_kinds_ty, _args_types, _mk_cfunc,
                       _opt_result, _pack_args, _unpack_lines)
from ._slsqp import (_minimize_slsqp_fg, _slsqp_run, _slsqp_bounds,
                     _no_c, _no_j, _EPSILON)
from ._minimize_py import (_nm_core, _cg_core, _bfgs_core, _powell_core,
                           _powell_limits, _nm_limits, _no_grad, _EPS_FD,
                           _eps_vec)
from ._prima import (minimize_uobyqa, minimize_newuoa, minimize_bobyqa,
                     minimize_lincoa, fmin_cobyla, prima_sig,
                     _prepend_n_args)
from ._callback import (_cb_noop, _cb_install, _cb_release,
                        _cb_resolve, _cb_resolve_ty,
                        _cb_halt_clear, _cb_halt_take,
                        CB_STOP_STATUS, CB_STOP_MESSAGE)

_MEPS = float(np.finfo(np.float64).eps)


@njit
def _pack_bounds(n, lower, upper):
    """lower/upper vectors -> scipy's (n, 2) bounds array, +-inf for open."""
    if lower.size == 0 and upper.size == 0:
        return np.zeros((0, 2), np.float64)
    if lower.size != 0 and lower.size != n:
        raise ValueError("minimize: lower must be empty or len(x0)")
    if upper.size != 0 and upper.size != n:
        raise ValueError("minimize: upper must be empty or len(x0)")
    b = np.empty((n, 2), np.float64)
    for i in range(n):
        b[i, 0] = lower[i] if lower.size == n else -np.inf
        b[i, 1] = upper[i] if upper.size == n else np.inf
    return b


@njit(error_model='numpy')
def _pair_rho(sk, yk):
    """``1 / (yk[i] @ sk[i])`` for each stored correction pair.

    ``error_model='numpy'`` so a pair of zero curvature gives ``inf``, which
    is what the same division gives in scipy.

    Parameters
    ----------
    sk, yk : ndarray, shape (n_corrs, n)
        The correction pairs.

    Returns
    -------
    rho : ndarray, shape (n_corrs,)
    """
    r = np.empty(sk.shape[0], np.float64)
    for i in range(sk.shape[0]):
        r[i] = 1.0 / np.dot(yk[i], sk[i])
    return r


@njit
def _hess_inv_matvec(sk, yk, rho, v):
    """The inverse-Hessian product from stored correction pairs.

    Nocedal's two-loop recursion. `sk` and `yk` are ``(n_corrs, n)``, oldest
    correction first, and `v` is ``(n,)``.

    Private, and it matches scipy: the counterpart is
    ``LbfgsInvHessProduct._matvec``, which scipy also keeps private. The
    public spelling is `HessInv.__call__`, as scipy's is ``op @ v``.

    Parameters
    ----------
    sk : ndarray, shape (n_corrs, n)
        Steps between successive iterates.
    yk : ndarray, shape (n_corrs, n)
        Differences between successive gradients.
    rho : ndarray, shape (n_corrs,)
        ``1 / (yk[i] @ sk[i])`` per pair, as `_pair_rho` computes it.
    v : ndarray, shape (n,)
        The vector to apply the operator to.

    Returns
    -------
    r : ndarray, shape (n,)
        ``H @ v``, with ``H`` the limited-memory inverse Hessian. With
        ``n_corrs == 0`` this is `v` unchanged, the identity.
    """
    k = sk.shape[0]
    q = v.astype(np.float64).copy()
    alpha = np.zeros(k, np.float64)
    for i in range(k - 1, -1, -1):
        alpha[i] = rho[i] * np.dot(sk[i], q)
        q = q - alpha[i] * yk[i]
    r = q
    for i in range(k):
        beta = rho[i] * np.dot(yk[i], r)
        r = r + sk[i] * (alpha[i] - beta)
    return r


_HESS_SPEC = [('sk', types.float64[:, ::1]),
              ('yk', types.float64[:, ::1]),
              ('mat', types.float64[:, ::1]),
              ('n', types.int64),
              ('n_corrs', types.int64),
              ('rho', types.float64[::1]),
              ('shape', types.UniTuple(types.int64, 2))]


@scijitclass(_HESS_SPEC)
class HessInv:
    """The inverse-Hessian estimate a `minimize` result carries.

    Two backends produce one, and they produce different things. BFGS carries
    a dense ``(n, n)`` matrix. L-BFGS-B never forms a matrix: it keeps the
    ``(s, y)`` correction pairs and applies the operator through them, which
    is what makes it usable at large `n`. Both are reached the same way, and
    ``.todense()`` gives the matrix from either.

    Apply the operator with ``op(v)``, ``op.matvec(v)`` or ``op.dot(v)``. The
    ``op @ v`` and ``op * v`` spellings and a ``dtype`` attribute are absent.

    Raises
    ------
    ValueError
        If `sk` and `yk` do not have matching shape.

    See Also
    --------
    scipy.optimize.LbfgsInvHessProduct : The operator scipy's L-BFGS-B result
        carries.

    Notes
    -----
    scipy uses two types where this uses one: a plain ndarray on the BFGS
    result and ``scipy.optimize.LbfgsInvHessProduct`` on the L-BFGS-B result.
    A compiled function has one return type, so one class covers both here.

    Only a ``'BFGS'`` or ``'L-BFGS-B'`` result carries a ``hess_inv``. On
    every other method the field is absent, as it is in scipy, so an operator
    is reached only after one of those two.

    A correction pair of zero curvature gives a `rho` entry of ``inf`` and a
    matrix of ``nan``. numpy raises a ``RuntimeWarning`` on that division; this
    does not.

    CONSTRUCTOR DEFAULTS ARE PYTHON-ONLY is the package-wide jitclass trap,
    but this constructor declares none, so all four arguments are always
    required, in Python and inside ``@njit`` alike.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import minimize
    >>> @njit
    ... def fg(x):
    ...     f = (x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2
    ...     return f, np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.5)])
    >>> @njit
    ... def run():
    ...     op = minimize(fg, np.array([0.0, 0.0]), method='BFGS').hess_inv
    ...     return op.has_hess(), op.shape, op.todense().shape
    >>> run()
    (True, (2, 2), (2, 2))

    Attributes
    ----------
    sk, yk : ndarray, shape (n_corrs, n)
        The stored correction pairs. Both are ``(0, n)`` unless the method
        was ``'L-BFGS-B'``.
    rho : ndarray, shape (n_corrs,)
        ``1 / (yk[i] @ sk[i])`` per pair.
    shape : tuple of int
        ``(n, n)``.
    n_corrs : int
        The number of stored correction pairs. ``0`` for the dense form.
    mat : ndarray, shape (n, n) or (0, 0)
        The dense estimate. ``(0, 0)`` unless the method was ``'BFGS'``.
    n : int
        The problem dimension.
    """

    def __init__(self, sk, yk, mat, n):
        if sk.shape[0] != yk.shape[0] or sk.shape[1] != yk.shape[1]:
            raise ValueError(
                'sk and yk must have matching shape, (n_corrs, n)')
        self.sk = sk
        self.yk = yk
        self.mat = mat
        self.n = n
        self.n_corrs = sk.shape[0]
        self.shape = (np.int64(n), np.int64(n))
        self.rho = _pair_rho(sk, yk)

    def ev(self, v):
        """Apply the operator. Reached as ``res.hess_inv(v)``.

        Parameters
        ----------
        v : ndarray, shape (n,)
            The vector to apply the operator to.

        Returns
        -------
        r : ndarray, shape (n,)
            ``H @ v``.
        """
        if self.mat.shape[0] != 0:
            return self.mat @ v
        return _hess_inv_matvec(self.sk, self.yk, self.rho, v)

    def matvec(self, v):
        """Apply the operator, the ``LinearOperator.matvec`` spelling.

        Parameters
        ----------
        v : ndarray, shape (n,)
            The vector to apply the operator to.

        Returns
        -------
        r : ndarray, shape (n,)
            ``H @ v``.
        """
        return self.ev(v)

    def dot(self, v):
        """Apply the operator, the ``LinearOperator.dot`` spelling.

        Parameters
        ----------
        v : ndarray, shape (n,)
            The vector to apply the operator to.

        Returns
        -------
        r : ndarray, shape (n,)
            ``H @ v``.
        """
        return self.ev(v)

    def todense(self):
        """The operator as an ``(n, n)`` array.

        Returns
        -------
        mat : ndarray, shape (n, n)
            The inverse-Hessian estimate materialised as a dense matrix.
        """
        if self.mat.shape[0] != 0:
            return self.mat.copy()
        out = np.zeros((self.n, self.n), np.float64)
        e = np.zeros(self.n, np.float64)
        for j in range(self.n):
            e[:] = 0.0
            e[j] = 1.0
            col = _hess_inv_matvec(self.sk, self.yk, self.rho, e)
            for i in range(self.n):
                out[i, j] = col[i]
        return out

    def has_hess(self):
        """Whether the method produced an estimate at all.

        Returns
        -------
        flag : bool
            ``True`` for ``'BFGS'`` and ``'L-BFGS-B'``, ``False`` otherwise.
        """
        return self.mat.shape[0] != 0 or self.sk.shape[0] != 0


@njit
def _dense_hess(mat):
    """The BFGS operator: a dense ``(n, n)`` inverse-Hessian estimate."""
    n = mat.shape[0]
    return HessInv(np.zeros((0, n), np.float64), np.zeros((0, n), np.float64),
                   np.ascontiguousarray(mat), np.int64(n))


@njit
def _pairs_hess(sk, yk, n):
    """The L-BFGS-B operator: the stored ``(s, y)`` correction pairs."""
    return HessInv(np.ascontiguousarray(sk), np.ascontiguousarray(yk),
                   np.zeros((0, 0), np.float64), np.int64(n))


# --------------------------------------------------------------------------
# The result types, one per FIELD SET.
#
# Each method returns the fields scipy's result carries for that method and
# no others, so a field the method did not compute is ABSENT rather than
# holding a sentinel.  `_opt_result` caches on the field set, so 'BFGS' and
# 'L-BFGS-B' share one class.
#
# THE ORDER IS THIS PACKAGE'S, and it is the same convention for every
# method: `x`, `fun`, the method's own outputs, the counters, then `status`,
# `message`, `success`.  scipy's per-method insertion orders disagree with
# each other -- measured on scipy 1.18, 'BFGS' and 'L-BFGS-B' carry the SAME
# ten fields in DIFFERENT orders, and 'COBYLA' opens with `x, status,
# success` where 'CG' ends with `x, nit`.  That is how each solver builds its
# object rather than a contract, and reproducing it is not parity.  Reaching
# a field by name or by attribute is unaffected.
# --------------------------------------------------------------------------

#: ``'Nelder-Mead'``.  `final_simplex` is scipy's ``(sim, fsim)`` pair.
_MR_NM = _opt_result(
    ['x', 'fun', 'nit', 'nfev', 'final_simplex', 'status', 'message',
     'success'])
#: ``'Powell'``, whose `direc` is the direction set the run ended with.
_MR_POWELL = _opt_result(
    ['x', 'fun', 'direc', 'nit', 'nfev', 'status', 'message', 'success'])
#: ``'CG'``.
_MR_GRAD = _opt_result(
    ['x', 'fun', 'jac', 'nit', 'nfev', 'njev', 'status', 'message',
     'success'])
#: ``'SLSQP'``, whose `multipliers` are the constraint multipliers of the
#: QP subproblem the solver finished on.  `_MR_GRAD`'s fields plus that one,
#: so the two are separate classes.
_MR_SLSQP = _opt_result(
    ['x', 'fun', 'jac', 'multipliers', 'nit', 'nfev', 'njev', 'status',
     'message', 'success'])
#: ``'BFGS'`` and ``'L-BFGS-B'``, which report an inverse-Hessian estimate.
_MR_HESS = _opt_result(
    ['x', 'fun', 'jac', 'nit', 'nfev', 'njev', 'hess_inv', 'status',
     'message', 'success'])
#: ``'COBYLA'``, whose `maxcv` is the largest constraint violation.
_MR_COBYLA = _opt_result(
    ['x', 'fun', 'maxcv', 'nfev', 'status', 'message', 'success'])
#: ``'UOBYQA'``, ``'NEWUOA'``, ``'BOBYQA'`` and ``'LINCOA'``.  scipy has no
#: counterpart to any of the four, so this shape has none to match.
_MR_PRIMA = _opt_result(
    ['x', 'fun', 'nfev', 'status', 'message', 'success'])


@njit
def _cb_halt_over(st, msg, ok, use_cb):
    """scipy's post-step for a `callback` that raised ``StopIteration``.

    scipy applies it once in `minimize`, after whichever solver ran
    (`scipy/optimize/_minimize.py:823-826`), so the six methods that serve a
    callback report the same three values.

    It overrides the three FIELDS rather than rebuilding the result, because
    the six methods no longer share one result type and a namedtuple cannot
    be rebuilt generically inside ``@njit``.  `_cb_halt_take` clears the
    flag, so every runner calls this exactly once.
    """
    if use_cb and _cb_halt_take():
        return np.int64(CB_STOP_STATUS), CB_STOP_MESSAGE, False
    return st, msg, ok

_MIN_CB_METHOD_MSG = (
    "minimize: callback is served by 'Nelder-Mead', 'Powell', 'CG', 'BFGS', "
    "'L-BFGS-B' and 'SLSQP'. 'COBYLA', 'UOBYQA', 'NEWUOA', 'BOBYQA' and "
    "'LINCOA' reach PRIMA, whose wrapper passes no callback_fcn, so there is "
    "no slot for one to arrive through.")

_MIN_METHOD_RUNTIME_MSG = (
    "minimize: inside @njit `method` must be a literal string written at the "
    "call site, not a variable. Each method returns the field set scipy's "
    "result carries for THAT method, and a compiled function has one return "
    "type per signature, so the method has to be known while the call "
    "compiles. Spell it literally, minimize(f, x0, method='BFGS'), or call "
    "from Python, where any string works. `root` refuses a runtime `method` "
    "for the same reason")

_MIN_CONS_MSG = (
    "minimize: constraints must be empty. SLSQP and COBYLA take them "
    "through fmin_slsqp and fmin_cobyla, whose constraint callables are "
    "@njit functions rather than a list of dicts.")

# The five ignored-argument warnings, in scipy's own emission order
# (`scipy/optimize/_minimize.py:617-646`).  A call emits one per class, each
# naming the method as the CALLER spelled it.
_W_JAC, _W_HESS, _W_HESSP, _W_CONS, _W_BOUNDS = 1, 2, 4, 8, 16

#: Methods that read no gradient, so `jac` reaching them is ignored.
_NO_JAC = ('NELDER-MEAD', 'POWELL', 'COBYLA', 'UOBYQA', 'NEWUOA', 'BOBYQA',
           'LINCOA')
#: Methods that take a box.  Nelder-Mead and Powell are absent here and
#: present in scipy's list; see the `bounds` paragraph under Notes.
_TAKES_BOUNDS = ('L-BFGS-B', 'SLSQP', 'COBYLA', 'BOBYQA', 'LINCOA')
#: Methods scipy forwards `constraints` to.
_TAKES_CONS = ('SLSQP', 'COBYLA')
#: Methods with a callback call site.  The other five reach PRIMA.
_TAKES_CALLBACK = ('NELDER-MEAD', 'POWELL', 'CG', 'BFGS', 'L-BFGS-B',
                   'SLSQP')


def _emit_min_warnings(method, mask):
    """The `RuntimeWarning`s scipy raises for an argument it will ignore."""
    if mask & _W_JAC:
        warnings.warn(
            f'Method {method} does not use gradient information (jac).',
            RuntimeWarning, stacklevel=2)
    if mask & _W_HESS:
        warnings.warn(
            f'Method {method} does not use Hessian information (hess).',
            RuntimeWarning, stacklevel=2)
    if mask & _W_HESSP:
        warnings.warn(
            f'Method {method} does not use Hessian-vector product'
            ' information (hessp).', RuntimeWarning, stacklevel=2)
    if mask & _W_CONS:
        warnings.warn(f'Method {method} cannot handle constraints.',
                      RuntimeWarning, stacklevel=2)
    if mask & _W_BOUNDS:
        warnings.warn(f'Method {method} cannot handle bounds.',
                      RuntimeWarning, stacklevel=2)


@njit
def _min_warn(method, mask):
    """`_emit_min_warnings` from compiled code.

    The `objmode` block sits in its own function so that lowering it pickles
    this dispatcher rather than a caller holding a ctypes entry point, and
    behind a test on `mask` so a call with nothing to report never takes the
    GIL.
    """
    if mask != 0:
        with objmode():
            _emit_min_warnings(method, mask)


def _min_warn_mask(m, jac_given, hess_given, hessp_given, cons_given,
                   bounds_given):
    """The warning mask for one call, from the method and five flags."""
    mask = 0
    if jac_given and m in _NO_JAC:
        mask |= _W_JAC
    if hess_given:
        mask |= _W_HESS
    if hessp_given:
        mask |= _W_HESSP
    if cons_given and m not in _TAKES_CONS:
        mask |= _W_CONS
    if bounds_given and m not in _TAKES_BOUNDS:
        mask |= _W_BOUNDS
    return mask

_LB_MESSAGES = ('CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL',
                'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT',
                'ABNORMAL: TERMINATION')
_SQ_MESSAGES = ('Optimization terminated successfully',
                'Iteration limit reached',
                'Singular matrix C in LSQ subproblem')

# =========================================================================
# the method table
# =========================================================================
#: Every accepted `method`, upper-cased.  scipy matches case-insensitively
#: and so does this.
_METHOD_NAMES = ('NELDER-MEAD', 'POWELL', 'CG', 'BFGS', 'L-BFGS-B', 'SLSQP',
                 'COBYLA', 'UOBYQA', 'NEWUOA', 'BOBYQA', 'LINCOA')

#: The four PRIMA names, in the order `_run_prima` branches on.
_PRIMA_CODE = {'NEWUOA': 0, 'UOBYQA': 1, 'BOBYQA': 2, 'LINCOA': 3}

#: Methods reached through a PRIMA `prima_sig` address.
_DF_PRIMA = ('UOBYQA', 'NEWUOA', 'BOBYQA', 'LINCOA')

_MIN_METHOD_LIST = (
    "'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'SLSQP', 'COBYLA', "
    "and additionally 'UOBYQA', 'NEWUOA', 'BOBYQA', 'LINCOA'")

#: scipy's own text, `scipy/optimize/_minimize.py:814`, which names the
#: `method` as the caller spelled it.
_MIN_UNKNOWN = 'Unknown solver '

_MIN_METHOD_TYPE_MSG = (
    "minimize: method must be a string. Accepted: " + _MIN_METHOD_LIST +
    " (case-insensitive).")

_MIN_JAC_TYPE_MSG = (
    "minimize: jac must be None, False, 0, or a plain @njit "
    "jac(x, *args) -> array. "
    "scipy's '2-point', '3-point' and 'cs' strings are not implemented; "
    "jac=None already forward-differences where the method needs a "
    "gradient. jac=True is spelled by having `fun` return (f, g).")

_MIN_JAC_FG_MSG = (
    "minimize: jac was given together with an objective that already returns "
    "(f, gradient). Pass either fg(x, *args) -> (f, g) with jac=None, or "
    "fun(x, *args) -> f with jac=grad.")

_MIN_FUN_MSG = (
    "minimize: fun must be a plain @njit function of (x, *args), returning "
    "either f or (f, gradient). A python callable cannot be reached from "
    "compiled code, and a raw @cfunc .address is not accepted.")

_MIN_ARITY_TMPL = (
    "minimize: %s is called %s(x, *args), so with the `args` given it takes "
    "%d argument%s: x, then %d element%s of `args`. The function passed binds "
    "no such call.")


def _arity_msg(name, atypes):
    """`_MIN_ARITY_TMPL` filled in for one callable and one `args` shape."""
    k = len(atypes)
    return _MIN_ARITY_TMPL % (name, name, k + 1, '' if k == 0 else 's',
                              k, '' if k == 1 else 's')


#: The type `x` is called with, everywhere below.
_F64_ARR = types.Array(types.float64, 1, 'C')


def _fun_shape(ftype, atypes=(), name='fun'):
    """``(returns_fg, nargs)`` for an objective or gradient type.

    scipy calls the objective ``fun(x, *args)``, so the arity is one plus the
    length of `args` and the parameter types after `x` are the ELEMENTS of
    `args`. `atypes` carries those element types, read from the call site.

    Two spellings of the objective ship. ``fg(x, *args) -> (f, g)`` returns
    the value and the gradient together; ``fun(x, *args) -> f`` returns the
    value. Which one it is follows from the return type, which numba resolves
    when the call compiles, so it needs no flag and costs no evaluation.

    Raises `TypeError` naming the callable and the expected argument count
    when that call does not bind, rather than letting numba's bind failure
    surface from inside the probe.
    """
    tc = cpu_target.typing_context
    tc.refresh()
    try:
        sig = ftype.get_call_type(tc, (_F64_ARR,) + tuple(atypes), {})
    except Exception:
        raise TypeError(_arity_msg(name, atypes))
    return isinstance(sig.return_type, types.BaseTuple), 1 + len(atypes)


#: Split objectives, keyed on the python function behind the dispatcher.
_FG_SPLITS = {}


def _split_fg(pyf):
    """``f(x, *args)`` and ``g(x, *args)`` from one ``fg(x, *args)``.

    Nelder-Mead, Powell, CG, BFGS, COBYLA and the PRIMA solvers take a
    scalar objective, and CG and BFGS take the gradient separately. An
    ``fg`` objective is adapted here rather than refused.

    Each returned function calls `pyf` once and drops half of what it
    computed, so an ``fg`` objective on CG or BFGS does the gradient work
    on an objective evaluation and the objective work on a gradient
    evaluation. The evaluation COUNTS are the same as scipy's; the work per
    evaluation is not.

    The two sit in a callback slot AND pass `args` on, so each declares
    ``*args`` and forwards it unchanged. One spelling then serves the splat
    and the single-argument arm alike: with a tuple `args` the caller
    unpacks it here, and with an ndarray `args` the tuple holds that one
    array.
    """
    hit = _FG_SPLITS.get(pyf)
    if hit is not None:
        return hit
    inner = njit(pyf)

    @njit
    def _f_of_fg(x, *args):
        return inner(x, *args)[0]

    @njit
    def _g_of_fg(x, *args):
        return inner(x, *args)[1]

    _FG_SPLITS[pyf] = (_f_of_fg, _g_of_fg)
    return _f_of_fg, _g_of_fg


#: Joined objectives, keyed on the two python functions behind them.
_FG_JOINS = {}


def _join_fg(pyf, pyg):
    """One ``fg(x, *args) -> (f, g)`` from a scalar objective and a gradient.

    L-BFGS-B takes both spellings itself; SLSQP's fg-driven backend does
    not, so scipy's separate `fun` and `jac` are joined here.
    """
    key = (pyf, pyg)
    hit = _FG_JOINS.get(key)
    if hit is not None:
        return hit
    f_in, g_in = njit(pyf), njit(pyg)

    @njit
    def _fg(x, *args):
        return f_in(x, *args), g_in(x, *args)

    _FG_JOINS[key] = _fg
    return _fg


@njit
def _no_cons(x, *args):
    """Zero nonlinear constraints, for COBYLA on an unconstrained problem."""
    return np.zeros(0, np.float64)


# =========================================================================
# one runner per backend.  Both entry points call these, so the python and
# the @njit paths cannot disagree.
# =========================================================================
_NM_MESSAGES = ('Optimization terminated successfully.',
                'Maximum number of function evaluations has been exceeded.',
                'Maximum number of iterations has been exceeded.',
                'NaN result encountered.')
_GR_MESSAGES = ('Optimization terminated successfully.',
                'Maximum number of iterations has been exceeded.',
                'Desired error not necessarily achieved due to precision '
                'loss.',
                'NaN result encountered.')
_PR_MESSAGES = ('Optimization terminated successfully.',
                'The budget of objective evaluations was exhausted.')


@njit
def _msg_of(table, k):
    """`table[k]`, clamped, so an unmapped code still returns a string."""
    if k < 0 or k >= len(table):
        return table[len(table) - 1]
    return table[k]


@njit
def _run_nm(f, xf, args, tol, maxiter, cb=_cb_noop, use_cb=False,
            fatol=np.nan, maxfev=0, has_mf=False):
    """method='Nelder-Mead' -> the `fmin` core.

    `tol` is the simplex tolerance and `fatol` the objective one. scipy
    keeps them apart, so ``options={'xatol': v}`` moves one and leaves the
    other at its default. `maxfev` is the evaluation budget and `has_mf`
    whether it was given at all, which the default rule keys off.
    """
    xtol = 1e-4 if np.isnan(tol) else tol
    fat = 1e-4 if np.isnan(fatol) else fatol
    # scipy's rule is asymmetric: give it `maxiter` and `maxfev` becomes
    # unbounded. `mf` was pinned to N*200 whatever `maxiter` said, which
    # stopped a long run early with status 1 where scipy carries on. Powell
    # took the same correction; the two now share the shape.
    absent = maxiter <= 0
    mi, mf = _nm_limits(xf.size, np.int64(maxiter), np.int64(maxfev),
                        absent, not has_mf)
    if use_cb:
        _cb_halt_clear()
    x, fv, it, nfe, wf, _av, sim, fsim = _nm_core(
        f, xf, args, xtol, fat, mi, mf,
        np.zeros((0, 0)), False, False, cb, use_cb)
    st, msg, ok = _cb_halt_over(np.int64(wf), _msg_of(_NM_MESSAGES, wf),
                                wf == 0, use_cb)
    return _MR_NM(
        np.ascontiguousarray(x), np.float64(fv),
        np.int64(it), np.int64(nfe),
        (np.ascontiguousarray(sim), np.ascontiguousarray(fsim)),
        st, msg, ok)


@njit
def _run_powell(f, xf, args, tol, maxiter, cb=_cb_noop, use_cb=False,
                ftol=np.nan, maxfev=0, has_mf=False):
    """method='Powell' -> `fmin_powell`.

    `tol` is the simplex tolerance and `ftol` the objective one. scipy
    keeps them apart, so ``options={'ftol': v}`` moves one and leaves the
    other at its default. `maxfev` is the evaluation budget and `has_mf`
    whether it was given at all, which the default rule keys off.
    """
    xtol = 1e-4 if np.isnan(tol) else tol
    ft = 1e-4 if np.isnan(ftol) else ftol
    # `minimize` never prints, so it reaches the core rather than
    # `fmin_powell`, whose scipy default `disp=1` writes a summary.
    absent = maxiter <= 0
    mi, mf = _powell_limits(xf.size, np.int64(maxiter), np.int64(maxfev),
                            absent, not has_mf)
    if use_cb:
        _cb_halt_clear()
    r = _powell_core(f, xf, args, xtol, ft, mi, mf,
                     np.zeros((0, 0), np.float64), False, False,
                     cb, use_cb)
    x, fv, it, nfe, wf = r[0], r[1], r[3], r[4], r[5]
    st, msg, ok = _cb_halt_over(np.int64(wf), _msg_of(_NM_MESSAGES, wf),
                                wf == 0, use_cb)
    return _MR_POWELL(
        np.ascontiguousarray(x), np.float64(fv),
        np.ascontiguousarray(r[2]),
        np.int64(it), np.int64(nfe), st, msg, ok)


@njit
def _run_cg(f, g, use_fd, xf, args, tol, maxiter, cb=_cb_noop,
            use_cb=False):
    """method='CG' -> the `fmin_cg` core, which also reports the gradient."""
    gtol = 1e-5 if np.isnan(tol) else tol
    mi = maxiter if maxiter > 0 else xf.size * 200
    # `_cg_core` takes the finite-difference step as a per-element VECTOR.
    if use_cb:
        _cb_halt_clear()
    r = _cg_core(f, g, use_fd, xf, args, gtol, np.inf,
                 _eps_vec(_EPS_FD, xf.size), mi, 1e-4, 0.4, False,
                 cb, use_cb)
    st, msg, ok = _cb_halt_over(np.int64(r[5]), _msg_of(_GR_MESSAGES, r[5]),
                                r[5] == 0, use_cb)
    return _MR_GRAD(
        np.ascontiguousarray(r[0]), np.float64(r[1]),
        np.ascontiguousarray(r[2]), np.int64(r[7]), np.int64(r[3]),
        np.int64(r[4]), st, msg, ok)


@njit
def _run_bfgs(f, g, use_fd, xf, args, tol, maxiter, cb=_cb_noop,
              use_cb=False):
    """method='BFGS' -> the `fmin_bfgs` core."""
    gtol = 1e-5 if np.isnan(tol) else tol
    mi = maxiter if maxiter > 0 else xf.size * 200
    # `_bfgs_core` takes the finite-difference step as a per-element VECTOR.
    if use_cb:
        _cb_halt_clear()
    r = _bfgs_core(f, g, use_fd, xf, args, gtol, np.inf,
                   _eps_vec(_EPS_FD, xf.size), mi,
                   1e-4, 0.9, 0.0, np.zeros((0, 0)), False, False,
                   cb, use_cb)
    st, msg, ok = _cb_halt_over(np.int64(r[6]), _msg_of(_GR_MESSAGES, r[6]),
                                r[6] == 0, use_cb)
    return _MR_HESS(
        np.ascontiguousarray(r[0]), np.float64(r[1]),
        np.ascontiguousarray(r[2]), np.int64(r[8]), np.int64(r[4]),
        np.int64(r[5]), _dense_hess(r[3]), st, msg, ok)


@njit
def _lbfgsb_prepare(xf, lo, up):
    """`_pack_bounds`'s validation, then L-BFGS-B's own ``nbd`` encoding.

    `fmin_l_bfgs_b` does this internally and returns only scipy's three
    values, so a caller that also wants the correction pairs reaches the
    driver directly. The preparation is repeated here rather than the
    algorithm: `_lbfgsb_run` stays the one implementation.
    """
    n = xf.size
    bnd = _pack_bounds(n, lo, up)
    if bnd.shape[0] == 0:
        lov = np.zeros(0, np.float64)
        hiv = np.zeros(0, np.float64)
    else:
        lov = np.ascontiguousarray(bnd[:, 0])
        hiv = np.ascontiguousarray(bnd[:, 1])
    l, u, nbd = _lbfgsb_bounds(n, lov, hiv)
    return _clip_to_box(xf, l, u, nbd), l, u, nbd


@njit
def _lbfgsb_pack(x, f, g, conv, sr, nfev, njev, nit, which, sk, yk, n,
                 use_cb=False):
    """The shared packing for both L-BFGS-B entries.

    The halt post-step is applied HERE rather than by the caller, so the two
    entries cannot disagree about it and `_cb_halt_take` still runs once.
    """
    st = _lbfgsb_flag(conv, sr)
    stv, msg, ok = _cb_halt_over(
        np.int64(st), _LB_MESSAGES[st] if st < 3 else _LB_MESSAGES[2],
        st == 0, use_cb)
    # Both return sites must produce the SAME numba type, so every integer
    # field is cast explicitly -- the backends report their counters in
    # different widths.
    return _MR_HESS(
        np.ascontiguousarray(x), np.float64(f), np.ascontiguousarray(g),
        np.int64(nit), np.int64(nfev), np.int64(njev),
        _pairs_hess(sk, yk, n), stv, msg, ok)


@njit
def _run_lbfgsb(fg, xf, args, lo, up, tol, maxiter, cb=_cb_noop,
                use_cb=False, gtol=np.nan, maxfun=0, has_mfun=False):
    """method=None.  `fg` returns (f, g); the driver takes that.

    `tol` is scipy's `ftol` and `gtol` its own. scipy seeds both from
    `tol` and keeps them apart afterwards, so ``options={'gtol': v}``
    moves one and leaves the other at its default.

    `maxfun` is scipy's evaluation budget and is INDEPENDENT of `maxiter`:
    its default is 15000 whatever the iteration cap is. `has_mfun` carries
    its absence, because zero and negative budgets are meaningful.
    """
    # The driver takes `ftol` as `factr = ftol / eps`; the 1e7 default is
    # scipy's own `ftol` 2.220446049250313e-09 divided by the same eps.
    pgtol = 1e-5 if np.isnan(gtol) else gtol
    factr = 1e7 if np.isnan(tol) else tol / _MEPS
    mi = maxiter if maxiter > 0 else 15000
    mfn = maxfun if has_mfun else _LB_MAXFUN_DEFAULT
    x0, l, u, nbd = _lbfgsb_prepare(xf, lo, up)
    if use_cb:
        _cb_halt_clear()
    r = _lbfgsb_run(_ev_fg, fg, fg, x0, l, u, nbd, args, 10, factr, pgtol,
                    1e-8, mfn, mi, -1, 20, cb, use_cb)
    # One evaluator call yields both f and g, so the two counters coincide.
    return _lbfgsb_pack(r[0], r[1], r[2], r[3], r[4], r[5], r[5], r[6],
                        r[7], r[8], r[9], xf.size, use_cb)


@njit
def _run_lbfgsb_sep(f, g, has_g, xf, args, lo, up, tol, maxiter,
                    cb=_cb_noop, use_cb=False, gtol=np.nan, maxfun=0,
                    has_mfun=False):
    """method=None with a scalar objective.

    The driver takes all three protocols itself: `fg`, a scalar objective
    with `fprime`, and a scalar objective with `approx_grad`.

    `tol` is scipy's `ftol` and `gtol` its own; `maxfun` and `has_mfun` are
    the evaluation budget and its absence. See `_run_lbfgsb`.
    """
    # The driver takes `ftol` as `factr = ftol / eps`; the 1e7 default is
    # scipy's own `ftol` 2.220446049250313e-09 divided by the same eps.
    pgtol = 1e-5 if np.isnan(gtol) else gtol
    factr = 1e7 if np.isnan(tol) else tol / _MEPS
    mi = maxiter if maxiter > 0 else 15000
    mfn = maxfun if has_mfun else _LB_MAXFUN_DEFAULT
    x0, l, u, nbd = _lbfgsb_prepare(xf, lo, up)
    if use_cb:
        _cb_halt_clear()
    if has_g:
        r = _lbfgsb_run(_ev_split, f, g, x0, l, u, nbd, args, 10, factr,
                        pgtol, 1e-8, mfn, mi, -1, 20, cb, use_cb)
        njev = r[5]
    else:
        r = _lbfgsb_run(_ev_approx, f, f, x0, l, u, nbd, args, 10, factr,
                        pgtol, 1e-8, mfn, mi, -1, 20, cb, use_cb)
        # No gradient is evaluated on this path: the driver asks for one
        # `n + 1` times per request, which is exactly what `_ev_approx`
        # reports, so the request count is the reported total over `n + 1`.
        njev = r[5] // (xf.size + 1)
    return _lbfgsb_pack(r[0], r[1], r[2], r[3], r[4], r[5], njev, r[6],
                        r[7], r[8], r[9], xf.size, use_cb)


@njit
def _run_slsqp_fg(fg, xf, args, lo, up, tol, maxiter, cb=_cb_noop,
                  use_cb=False):
    """method='SLSQP' with an fg objective."""
    acc = 1e-6 if np.isnan(tol) else tol
    mi = maxiter if maxiter > 0 else 100
    if use_cb:
        _cb_halt_clear()
    x, f, ok, mode, nit, gv, nfev, njev, mult = _minimize_slsqp_fg(
        fg, xf, lo, up, args, acc, mi, cb, use_cb)
    return _slsqp_result(x, f, ok, mode, nit, gv, mult, nfev, njev, use_cb)


@njit
def _run_slsqp_sep(f, g, has_g, xf, args, lo, up, tol, maxiter,
                   cb=_cb_noop, use_cb=False):
    """method='SLSQP' with a scalar objective and an optional gradient."""
    acc = 1e-6 if np.isnan(tol) else tol
    mi = maxiter if maxiter > 0 else 100
    lb, ub = _slsqp_bounds(xf.size, lo, up)
    if use_cb:
        _cb_halt_clear()
    x, fx, nit, mode, nfev, njev, gv, mult = _slsqp_run(
        f, g, _no_c, _no_j, _no_c, _no_j, xf, lb, ub, args, acc, mi,
        _EPSILON, has_g, False, False, 0, 0, 0, cb, use_cb)
    return _slsqp_result(x, fx, mode == 0, mode, nit, gv, mult, nfev, njev,
                         use_cb)


@njit
def _slsqp_result(x, f, ok, mode, nit, gv, mult, nfev, njev, use_cb=False):
    """The SLSQP branches' shared packing, halt post-step included."""
    # scipy's SLSQP reports `majiter`, which on the iteration-limit path
    # equals `maxiter` exactly. Our driver's counter increments once more
    # before the limit test fires, so the cap path was reporting maxiter + 1.
    # MEASURED against scipy: maxiter 1/2/5 -> scipy nit 1/2/5, ours 2/3/6.
    # The CONVERGENCE path already agreed (46 == 46, 37 == 37), so only the
    # cap path is corrected -- the same convergence-vs-exhaustion split
    # lsq_linear's `nit` needed.
    n = nit
    if mode == 9 and n > 0:
        n -= 1
    if mode == 0:
        msg = _SQ_MESSAGES[0]
    elif mode == 9:
        msg = _SQ_MESSAGES[1]
    else:
        msg = _SQ_MESSAGES[2]
    st, msg, ok = _cb_halt_over(np.int64(mode), msg, ok, use_cb)
    return _MR_SLSQP(
        np.ascontiguousarray(x), np.float64(f), np.ascontiguousarray(gv),
        np.ascontiguousarray(mult),
        np.int64(n), np.int64(nfev), np.int64(njev), st, msg, ok)


@njit
def _run_cobyla(f, cons, xf, args, lo, up, tol, maxiter, rhobeg):
    """method='COBYLA' -> PRIMA, through `fmin_cobyla`."""
    rhoend = 1e-4 if np.isnan(tol) else tol
    mf = maxiter if maxiter > 0 else 1000
    # `catol` is scipy's own `None` here, not the `nan` this call site used
    # to spell: `fmin_cobyla`'s default moved to scipy's 2e-4 and kept the
    # `nan` path only for this caller.
    r = fmin_cobyla(f, xf, cons, args, None, rhobeg, rhoend, mf, None, None,
                    m_nlcon=-1, lower=lo, upper=up, full_output=True)
    return _MR_COBYLA(
        np.ascontiguousarray(r.x), np.float64(r.fun), np.float64(r.maxcv),
        np.int64(r.nfev), np.int64(r.status), r.message, r.success)


#: `prima_sig` adapters, keyed on the python objective and the `args` kinds.
#: The cache also OWNS the cfunc: nothing else holds a reference, and a
#: collected cfunc leaves a dangling address.
_PRIMA_ADAPTERS = {}

_PRIMA_OBJ_SRC = """
def _make(f_in, cfunc, sig, carray):
    @cfunc(sig)
    def adapter(x_ptr, f_ptr, a_ptr):
        n = int(a_ptr[0])
        na = int(a_ptr[1])
        x = carray(x_ptr, n)
        a = carray(a_ptr, 2 + na)[2:]
%(unpack)s
        f_ptr[0] = f_in(x%(argl)s)
    return adapter
"""


def _adapter_star(pyf, kinds):
    """``cfunc(prima_sig)`` around a plain @njit ``f(x, *args) -> float``.

    UOBYQA, NEWUOA, BOBYQA and LINCOA reach the objective through a C
    function pointer, which carries one ``double*`` and no argument list, so
    `args` crosses flattened by `_pack_args` and is rebuilt here from
    `kinds` before the call. The header ``_prepend_n_args`` writes holds
    ``n`` and the buffer length; the payload starts at offset 0 of the slice
    behind it.
    """
    key = (pyf, kinds)
    hit = _PRIMA_ADAPTERS.get(key)
    if hit is not None:
        return hit
    unpack, argl = _unpack_lines(kinds, 0, ' ' * 8)
    adapter = _mk_cfunc(_PRIMA_OBJ_SRC, {'unpack': unpack, 'argl': argl},
                        njit(pyf), prima_sig)
    _PRIMA_ADAPTERS[key] = adapter
    return adapter


def _prima_args(args):
    """`args` flattened for the ``prima_sig`` adapter, tuple or not.

    The non-tuple arm is scipy's own coercion of `args` to a one-item tuple,
    so the buffer holds the same elements the seven scipy methods splat.
    """
    if isinstance(args, tuple):
        return _pack_args(args)
    return _pack_args((args,))


@overload(_prima_args)
def _prima_args_ovl(args):
    if isinstance(args, types.BaseTuple) or (isinstance(args, tuple)
                                             and len(args) == 0):
        def impl(args):
            return _pack_args(args)
        return impl

    def impl(args):
        return _pack_args((args,))
    return impl


@njit
def _run_prima(addr, which, xf, args, lo, up, tol, maxiter, rhobeg, npt):
    """methods 'NEWUOA', 'UOBYQA', 'BOBYQA' and 'LINCOA'."""
    rhoend = 1e-6 if np.isnan(tol) else tol
    # The substitution the other nine runners make, spelled here too. The
    # PRIMA entries apply the same `500 * n` when `maxfun <= 0`, so this
    # changes no value; it stops one runner out of ten relying on the callee
    # for a default the docstring attributes to this function.
    mf = maxiter if maxiter > 0 else 500 * xf.size
    z2 = np.zeros((0, 0))
    z1 = np.zeros(0)
    if which == 0:
        x, f, ok, nf, info = minimize_newuoa(addr, xf, args, rhobeg, rhoend,
                                             mf, npt)
    elif which == 1:
        x, f, ok, nf, info = minimize_uobyqa(addr, xf, args, rhobeg, rhoend,
                                             mf)
    elif which == 2:
        x, f, ok, nf, info = minimize_bobyqa(addr, xf, lo, up, args, rhobeg,
                                             rhoend, mf, npt)
    else:
        x, f, cstrv, ok, nf, info = minimize_lincoa(
            addr, xf, z2, z1, z2, z1, lo, up, args, rhobeg, rhoend, mf, npt)
    return _MR_PRIMA(
        np.ascontiguousarray(x), np.float64(f), np.int64(nf),
        np.int64(info), _msg_of(_PR_MESSAGES, 0 if ok else 1), ok)


@njit
def _bounds_to_lu(bounds, n):
    """scipy's ``bounds`` -- an (n, 2) array_like -- to (lower, upper)."""
    b = np.ascontiguousarray(np.asarray(bounds, np.float64))
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("minimize: bounds must be one (min, max) pair per "
                         "variable, an (n, 2) array_like")
    if b.shape[0] != n:
        raise ValueError("minimize: bounds must have one pair per variable")
    for i in range(b.shape[0]):
        if b[i, 1] < b[i, 0]:
            raise ValueError(
                "An upper bound is less than the corresponding lower bound.")
    return (np.ascontiguousarray(b[:, 0]).copy(),
            np.ascontiguousarray(b[:, 1]).copy())


_MIN_COMPLEX_MSG = "minimize: complex `x0` is not accepted"


def _complex_ty(t):
    """True when a numba type carries complex values."""
    if isinstance(t, types.Array):
        return isinstance(t.dtype, types.Complex)
    if isinstance(t, (types.List, types.ListType)):
        return isinstance(t.dtype, types.Complex)
    if isinstance(t, types.BaseTuple):
        return any(isinstance(e, types.Complex) for e in t)
    return isinstance(t, types.Complex)


def _no_complex_x0(x0):
    """Refuse a complex `x0`. C4: scipy accepts one and does not minimise it.

    MEASURED on scipy 1.18, ``f(z) = |z - (2+3j)|**2``, whose minimiser is
    ``2+3j`` and whose minimum is 0: the imaginary part never moves from its
    initial value and ``fun`` is 9.0 from every start. `minimize`'s ``Notes``
    carries the numbers.
    """
    if np.iscomplexobj(x0):
        raise TypeError(_MIN_COMPLEX_MSG)


@overload(_no_complex_x0)
def _no_complex_x0_ovl(x0):
    """`_no_complex_x0` inside ``@njit``: decided by the argument's type."""
    if _complex_ty(x0):
        def impl(x0):
            raise TypeError(_MIN_COMPLEX_MSG)
        return impl

    def impl(x0):
        pass
    return impl


def _real_x0(x0):
    """`x0` as a real array, so a complex argument still TYPES.

    A run-time `raise` does not stop numba typing the statements after it,
    so ``np.asarray(x0, np.float64)`` would fail first and the caller would
    see a ``TypingError`` in place of `_MIN_COMPLEX_MSG`.
    """
    return np.real(np.asarray(x0))


@overload(_real_x0)
def _real_x0_ovl(x0):
    if _complex_ty(x0):
        def impl(x0):
            return np.asarray(x0).real
        return impl

    def impl(x0):
        return np.asarray(x0)
    return impl


@njit
def _min_prep_core(x0, bounds, lower, upper):
    """`x0`, `lower` and `upper` in the shape every runner takes."""
    _no_complex_x0(x0)
    x0a = np.asarray(_real_x0(x0), np.float64)
    if x0a.ndim > 1:
        raise ValueError("'x0' must only have one dimension.")
    xf = np.ascontiguousarray(x0a).ravel()
    b = np.ascontiguousarray(np.asarray(bounds, np.float64))
    if b.size > 0:
        lo, up = _bounds_to_lu(b, xf.size)
    else:
        lo = np.ascontiguousarray(np.asarray(lower, np.float64)).ravel()
        up = np.ascontiguousarray(np.asarray(upper, np.float64)).ravel()
    return xf, lo, up


def _min_prep(x0, bounds, lower, upper):
    """`_min_prep_core` with ``None`` accepted for the three bound spellings.

    scipy's default for `bounds` is ``None``, and an EMPTY ARRAY cannot be a
    default on a function carrying an ``@overload``: numba validates the two
    signatures against each other and compares every default with ``==``,
    which an empty array answers with an empty array.
    """
    z2 = np.zeros((0, 2), np.float64)
    z1 = np.zeros(0, np.float64)
    return _min_prep_core(x0,
                          z2 if bounds is None else bounds,
                          z1 if lower is None else lower,
                          z1 if upper is None else upper)


@overload(_min_prep)
def _min_prep_ovl(x0, bounds, lower, upper):
    nb, nl, nu = _is_none(bounds), _is_none(lower), _is_none(upper)

    def impl(x0, bounds, lower, upper):
        z2 = np.zeros((0, 2), np.float64)
        z1 = np.zeros(0, np.float64)
        if nb:
            b = z2
        else:
            b = np.ascontiguousarray(np.asarray(bounds, np.float64))
        if nl:
            lw = z1
        else:
            lw = np.ascontiguousarray(np.asarray(lower, np.float64)).ravel()
        if nu:
            uw = z1
        else:
            uw = np.ascontiguousarray(np.asarray(upper, np.float64)).ravel()
        return _min_prep_core(x0, b, lw, uw)
    return impl


def _cons_given_ty(constraints):
    """``constraints`` non-empty, read from its numba TYPE.

    A tuple carries its length in its type. Anything else, a list or a
    reflected container, does not, so it is taken as given: this function
    only ever gates a warning or a refusal, and both are right there.
    """
    ct = (constraints.value if isinstance(constraints, types.Omitted)
          else constraints)
    if isinstance(ct, tuple):
        return len(ct) != 0
    if isinstance(ct, types.BaseTuple):
        return len(ct.types) != 0
    return not _is_none(ct)


def _jac_absent(jac):
    """``True`` when `jac` means no gradient at all.

    scipy's rule is ``jac is None or bool(jac) is False``
    (`scipy/optimize/_minimize.py:661-663`), so ``False`` and ``0`` reach the
    solver as no gradient rather than as an error. Only the two spellings an
    ``@overload`` chooser can also read as a literal count here, so both
    entry points answer the same.
    """
    return jac is None or (isinstance(jac, (bool, int)) and not jac)


def _jac_absent_ty(jac):
    """`_jac_absent` read from an ``@overload`` argument's TYPE."""
    if _is_none(jac):
        return True
    jt = jac.value if isinstance(jac, types.Omitted) else jac
    return (isinstance(jt, (types.BooleanLiteral, types.IntegerLiteral))
            and not jt.literal_value)


def _auto_method(bounds_given, cons_given):
    """scipy's rule for ``method=None``: SLSQP, L-BFGS-B or BFGS."""
    if cons_given:
        return 'SLSQP'
    if bounds_given:
        return 'L-BFGS-B'
    return 'BFGS'


def _tol_or_nan(tol):
    """`tol` as a float, with ``None`` carried through as ``nan``.

    Every runner reads ``nan`` as "use this method's own tolerance", which
    leaves ``0.0`` and every negative value meaning themselves, as they do
    in scipy.
    """
    return np.nan if tol is None else np.float64(tol)


@overload(_tol_or_nan)
def _tol_or_nan_ovl(tol):
    if _is_none(tol):
        def impl(tol):
            return np.nan
        return impl

    def impl(tol):
        return np.float64(tol)
    return impl


# =========================================================================
# `options`
# =========================================================================
#: The per-method tolerance names scipy's `options` carries, in the order
#: they are read.  A method with ONE tolerance reads its name from here; the
#: three methods with a PAIR read theirs through `_min_opt_nm`,
#: `_min_opt_pw` and `_min_opt_lb`, which give each name its own slot.
_MIN_OPT_TOL = ('xtol', 'ftol', 'gtol', 'pgtol', 'acc', 'rhoend', 'tol')

#: The three methods whose tolerances come in PAIRS, and the pair each reads.
#: MEASURED, with both members of a pair folded onto the single `tol` slot:
#: Nelder-Mead ``options={'xatol': 1e-2}`` returned ``[0., 0.00025]`` where
#: scipy returned ``[1.00143, 2.49749]``; Powell ``options={'ftol': 1e-2}``
#: returned ``[1., 1.]`` where scipy returned ``[0.709792, 0.503804]``; and
#: L-BFGS-B ``options={'gtol': 1e-2}`` returned ``[0.622821, 0.386286]``
#: where scipy returned ``[0.999683, 0.999379]``.  Naming EITHER member
#: moved the other, and a call at EQUAL tolerances cannot show it.
_MIN_OPT_NM = ('xatol', 'fatol')
_MIN_OPT_PW = ('xtol', 'ftol')
_MIN_OPT_LB = ('ftol', 'gtol')

#: The two methods that read ``options['maxfev']``.  scipy's key set is per
#: method, measured over 14 keys x 7 methods: `maxfev` reaches
#: ``_minimize_neldermead`` and ``_minimize_powell`` and draws an
#: ``OptimizeWarning`` on the other five.
_TAKES_MAXFEV = ('NELDER-MEAD', 'POWELL')

#: The one method that reads ``options['maxfun']``.  scipy names the key on
#: ``_minimize_lbfgsb`` alone; the other ten draw an ``OptimizeWarning``.
#: Its default there is 15000, independent of `maxiter`.
_TAKES_MAXFUN = ('L-BFGS-B',)

#: scipy's ``_minimize_lbfgsb`` default for both caps.
_LB_MAXFUN_DEFAULT = 15000

#: Every `options` key that reaches an argument.  `rhobeg` is scipy's own
#: COBYLA key and PRIMA's; `npt` is PRIMA's alone.
_MIN_OPT_KEYS = ('maxiter', 'rhobeg', 'npt') + _MIN_OPT_NM + _MIN_OPT_TOL

#: scipy's text, `scipy/optimize/_minimize.py` ``_check_unknown_options``.
_MIN_OPT_UNKNOWN = 'Unknown solver options: '

_MIN_OPT_TYPE_MSG = (
    "minimize: options must be a mapping or None. Its keys are read when "
    "the call compiles, so a dict LITERAL at the call site is what reaches "
    "compiled code; a dict built at run time and passed in is not.")


def _min_opt_warn(text):
    """scipy's `OptimizeWarning` for an `options` key nothing here reads.

    Imported inside the function because `_lsq` imports from `_minpack`,
    which this module imports from, so a top-level import closes the cycle.
    """
    from ._lsq import OptimizeWarning
    warnings.warn(text, OptimizeWarning, stacklevel=3)


@njit
def _min_opt_warn_nb(text):
    """`_min_opt_warn` from compiled code.

    The ``objmode`` block sits in its own function for the reason
    `_min_warn`'s does: lowering one pickles the enclosing function, and the
    bodies that call this reach a ctypes entry point.
    """
    if len(text) > 0:
        with objmode():
            _min_opt_warn(_MIN_OPT_UNKNOWN + text)


def _min_options(options, tol, maxiter, mf=False, mfun=False):
    """`options` folded onto the arguments that carry it.

    Returns ``(tol, maxiter, rhobeg, npt)``, with `tol` already carried
    through `_tol_or_nan`. A key nothing here reads draws an
    `OptimizeWarning` naming it and is ignored, which is what scipy does
    for a key IT does not read.

    `mf` says the calling method reads ``maxfev``, which Nelder-Mead and
    Powell do and the other nine do not. `mfun` says it reads ``maxfun``,
    which L-BFGS-B does and the other ten do not. The key set scipy accepts
    is per method, so the warning is too.
    """
    rhobeg, npt = 1.0, 0
    if options is None:
        return _tol_or_nan(tol), np.int64(maxiter), np.float64(rhobeg), \
            np.int64(npt)
    if not hasattr(options, 'keys'):
        raise TypeError(_MIN_OPT_TYPE_MSG)
    unknown = [str(k) for k in options
               if k not in _MIN_OPT_KEYS and not (mf and k == 'maxfev')
               and not (mfun and k == 'maxfun')]
    if unknown:
        _min_opt_warn(_MIN_OPT_UNKNOWN + ', '.join(unknown))
    if 'maxiter' in options:
        maxiter = options['maxiter']
    if 'rhobeg' in options:
        rhobeg = options['rhobeg']
    if 'npt' in options:
        npt = options['npt']
    for k in _MIN_OPT_TOL:
        if k in options:
            tol = options[k]
            break
    return (_tol_or_nan(tol), np.int64(maxiter), np.float64(rhobeg),
            np.int64(npt))


def _min_opt_mf(options):
    """``options['maxfev']``, and whether it was there at all.

    Read by Nelder-Mead and Powell, the two methods scipy's own
    ``_minimize_neldermead`` and ``_minimize_powell`` take it on. Its
    ABSENCE travels separately from its value, because zero and negative
    budgets are meaningful and the default rule keys off absence.
    """
    if options is not None and 'maxfev' in options:
        return np.int64(options['maxfev']), True
    return np.int64(0), False


@overload(_min_opt_mf)
def _min_opt_mf_ovl(options):
    if _is_none(options):
        def impl(options):
            return np.int64(0), False
        return impl

    if isinstance(options, types.DictType):
        def impl(options):
            if 'maxfev' in options:
                return np.int64(options['maxfev']), True
            return np.int64(0), False
        return impl

    if isinstance(options, types.LiteralStrKeyDict):
        HAS_MAXFEV = 'maxfev' in options.fields

        def impl(options):
            if HAS_MAXFEV:
                return np.int64(options['maxfev']), True
            return np.int64(0), False
        return impl

    raise TypingError(_MIN_OPT_TYPE_MSG)


def _min_opt_mfun(options):
    """``options['maxfun']``, and whether it was there at all.

    Read by L-BFGS-B, the one method scipy's own ``_minimize_lbfgsb`` takes
    it on. Its ABSENCE travels separately from its value, because zero and
    negative budgets are meaningful and the default keys off absence:
    scipy's `maxfun` default is 15000 whatever `maxiter` is.
    """
    if options is not None and 'maxfun' in options:
        return np.int64(options['maxfun']), True
    return np.int64(0), False


@overload(_min_opt_mfun)
def _min_opt_mfun_ovl(options):
    if _is_none(options):
        def impl(options):
            return np.int64(0), False
        return impl

    if isinstance(options, types.DictType):
        def impl(options):
            if 'maxfun' in options:
                return np.int64(options['maxfun']), True
            return np.int64(0), False
        return impl

    if isinstance(options, types.LiteralStrKeyDict):
        HAS_MAXFUN = 'maxfun' in options.fields

        def impl(options):
            if HAS_MAXFUN:
                return np.int64(options['maxfun']), True
            return np.int64(0), False
        return impl

    raise TypingError(_MIN_OPT_TYPE_MSG)


def _min_opt_nm(options, tol):
    """Nelder-Mead's two tolerance slots, ``(xatol, fatol)``.

    scipy seeds both from `tol` and lets an explicit ``options`` entry
    override one of them, `scipy/optimize/_minimize.py` ``setdefault``.
    """
    t = _tol_or_nan(tol)
    xa, fa = t, t
    if options is not None:
        if 'xatol' in options:
            xa = options['xatol']
        if 'fatol' in options:
            fa = options['fatol']
    return np.float64(xa), np.float64(fa)


@overload(_min_opt_nm)
def _min_opt_nm_ovl(options, tol):
    if _is_none(options):
        def impl(options, tol):
            t = _tol_or_nan(tol)
            return t, t
        return impl

    if isinstance(options, types.DictType):
        def impl(options, tol):
            t = _tol_or_nan(tol)
            xa, fa = t, t
            if 'xatol' in options:
                xa = np.float64(options['xatol'])
            if 'fatol' in options:
                fa = np.float64(options['fatol'])
            return xa, fa
        return impl

    if isinstance(options, types.LiteralStrKeyDict):
        HAS_XATOL = 'xatol' in options.fields
        HAS_FATOL = 'fatol' in options.fields

        def impl(options, tol):
            t = _tol_or_nan(tol)
            xa, fa = t, t
            if HAS_XATOL:
                xa = np.float64(options['xatol'])
            if HAS_FATOL:
                fa = np.float64(options['fatol'])
            return xa, fa
        return impl

    raise TypingError(_MIN_OPT_TYPE_MSG)


def _min_opt_pw(options, tol):
    """Powell's two tolerance slots, ``(xtol, ftol)``.

    scipy seeds both from `tol` and lets an explicit ``options`` entry
    override one of them, `scipy/optimize/_minimize.py` ``setdefault``.
    """
    t = _tol_or_nan(tol)
    a, b = t, t
    if options is not None:
        if 'xtol' in options:
            a = options['xtol']
        if 'ftol' in options:
            b = options['ftol']
    return np.float64(a), np.float64(b)


@overload(_min_opt_pw)
def _min_opt_pw_ovl(options, tol):
    if _is_none(options):
        def impl(options, tol):
            t = _tol_or_nan(tol)
            return t, t
        return impl

    if isinstance(options, types.DictType):
        def impl(options, tol):
            t = _tol_or_nan(tol)
            a, b = t, t
            if 'xtol' in options:
                a = np.float64(options['xtol'])
            if 'ftol' in options:
                b = np.float64(options['ftol'])
            return a, b
        return impl

    if isinstance(options, types.LiteralStrKeyDict):
        HAS_A = 'xtol' in options.fields
        HAS_B = 'ftol' in options.fields

        def impl(options, tol):
            t = _tol_or_nan(tol)
            a, b = t, t
            if HAS_A:
                a = np.float64(options['xtol'])
            if HAS_B:
                b = np.float64(options['ftol'])
            return a, b
        return impl

    raise TypingError(_MIN_OPT_TYPE_MSG)


def _min_opt_lb(options, tol):
    """L-BFGS-B's two tolerance slots, ``(ftol, gtol)``.

    scipy seeds both from `tol` and lets an explicit ``options`` entry
    override one of them, `scipy/optimize/_minimize.py` ``setdefault``.
    """
    t = _tol_or_nan(tol)
    a, b = t, t
    if options is not None:
        if 'ftol' in options:
            a = options['ftol']
        if 'gtol' in options:
            b = options['gtol']
    return np.float64(a), np.float64(b)


@overload(_min_opt_lb)
def _min_opt_lb_ovl(options, tol):
    if _is_none(options):
        def impl(options, tol):
            t = _tol_or_nan(tol)
            return t, t
        return impl

    if isinstance(options, types.DictType):
        def impl(options, tol):
            t = _tol_or_nan(tol)
            a, b = t, t
            if 'ftol' in options:
                a = np.float64(options['ftol'])
            if 'gtol' in options:
                b = np.float64(options['gtol'])
            return a, b
        return impl

    if isinstance(options, types.LiteralStrKeyDict):
        HAS_A = 'ftol' in options.fields
        HAS_B = 'gtol' in options.fields

        def impl(options, tol):
            t = _tol_or_nan(tol)
            a, b = t, t
            if HAS_A:
                a = np.float64(options['ftol'])
            if HAS_B:
                b = np.float64(options['gtol'])
            return a, b
        return impl

    raise TypingError(_MIN_OPT_TYPE_MSG)


@overload(_min_options)
def _min_options_ovl(options, tol, maxiter, mf=False, mfun=False):
    """`_min_options` inside ``@njit``, one body per dict flavour.

    A dict LITERAL whose values all take one type is a ``DictType``, whose
    keys are readable at RUN time; a literal carrying two value types is a
    ``LiteralStrKeyDict``, whose keys are readable while the call compiles
    and whose entries can only be indexed by a key written into the body.
    Both are served. The second arm's ``if HAS_<key>`` tests are freevar
    constants, so numba prunes the branch and no absent key is indexed.

    `mf` and `mfun` arrive as run-time booleans, so the second arm builds
    every unknown-key string while the call compiles and picks between them.
    No method reads both keys, so three strings cover the four combinations.
    """
    if _is_none(options):
        def impl(options, tol, maxiter, mf=False, mfun=False):
            return (_tol_or_nan(tol), np.int64(maxiter), np.float64(1.0),
                    np.int64(0))
        return impl

    if isinstance(options, types.DictType):
        if not isinstance(options.key_type, types.UnicodeType):
            raise TypingError(_MIN_OPT_TYPE_MSG)
        KEYS = _MIN_OPT_KEYS
        TOLK = _MIN_OPT_TOL

        def impl(options, tol, maxiter, mf=False, mfun=False):
            bad = ''
            for k in options:
                known = (mf and k == 'maxfev') or (mfun and k == 'maxfun')
                for j in range(len(KEYS)):
                    if k == KEYS[j]:
                        known = True
                        break
                if not known:
                    bad = k if len(bad) == 0 else bad + ', ' + k
            _min_opt_warn_nb(bad)
            t = _tol_or_nan(tol)
            m = np.int64(maxiter)
            rb = np.float64(1.0)
            nt = np.int64(0)
            if 'maxiter' in options:
                m = np.int64(options['maxiter'])
            if 'rhobeg' in options:
                rb = np.float64(options['rhobeg'])
            if 'npt' in options:
                nt = np.int64(options['npt'])
            for j in range(len(TOLK)):
                if TOLK[j] in options:
                    t = np.float64(options[TOLK[j]])
                    break
            return t, m, rb, nt
        return impl

    if isinstance(options, types.LiteralStrKeyDict):
        fields = options.fields
        BAD = ', '.join(str(k) for k in fields if k not in _MIN_OPT_KEYS)
        BAD_MF = ', '.join(str(k) for k in fields
                           if k not in _MIN_OPT_KEYS and k != 'maxfev')
        BAD_MFUN = ', '.join(str(k) for k in fields
                             if k not in _MIN_OPT_KEYS and k != 'maxfun')
        HAS_MAXITER = 'maxiter' in fields
        HAS_RHOBEG = 'rhobeg' in fields
        HAS_NPT = 'npt' in fields
        # One flag per tolerance name, because the key has to be written
        # into the body; the first present one wins, so the later flags are
        # cleared here rather than in the body.
        tolk = None
        for k in _MIN_OPT_TOL:
            if k in fields:
                tolk = k
                break
        (T_XTOL, T_FTOL, T_GTOL, T_PGTOL, T_ACC,
         T_RHOEND, T_TOL) = (k == tolk for k in _MIN_OPT_TOL)

        def impl(options, tol, maxiter, mf=False, mfun=False):
            _min_opt_warn_nb(BAD_MF if mf else (BAD_MFUN if mfun else BAD))
            t = _tol_or_nan(tol)
            m = np.int64(maxiter)
            rb = np.float64(1.0)
            nt = np.int64(0)
            if HAS_MAXITER:
                m = np.int64(options['maxiter'])
            if HAS_RHOBEG:
                rb = np.float64(options['rhobeg'])
            if HAS_NPT:
                nt = np.int64(options['npt'])
            if T_XTOL:
                t = np.float64(options['xtol'])
            if T_FTOL:
                t = np.float64(options['ftol'])
            if T_GTOL:
                t = np.float64(options['gtol'])
            if T_PGTOL:
                t = np.float64(options['pgtol'])
            if T_ACC:
                t = np.float64(options['acc'])
            if T_RHOEND:
                t = np.float64(options['rhoend'])
            if T_TOL:
                t = np.float64(options['tol'])
            return t, m, rb, nt
        return impl

    raise TypingError(_MIN_OPT_TYPE_MSG)


def _plan(fun_d, jac_d, is_fg):
    """Everything the runners need, resolved from the two callables.

    Returns ``(f_scalar, gradient, has_gradient, fg)``. `fun_d` and `jac_d`
    are dispatchers. `fg` is ``None`` where there is nothing to build one
    from.
    """
    if is_fg:
        f_s, g_s = _split_fg(fun_d.py_func)
        return f_s, g_s, True, fun_d
    if jac_d is None:
        return fun_d, _no_grad, False, None
    return fun_d, jac_d, True, _join_fg(fun_d.py_func, jac_d.py_func)


def minimize(fun, x0, args=(), method=None, jac=None,
             hess=None, hessp=None, bounds=None,
             constraints=(), tol=None, callback=None, options=None,
             lower=None, upper=None, maxiter=0):
    """Minimize a scalar function of one or more variables.

    Eleven algorithms behind one `method` string. Seven answer to the
    standard method names; four more are the remaining PRIMA derivative-free
    solvers.

    Callable from Python and from inside ``@njit``. Both entries run the
    same compiled backends and reach them through the same runners. They
    differ in the CLASS an argument error carries, which the Raises section
    names, because the compiled entry decides those while the call compiles.

    `fun` is a plain ``@njit`` function in every case, including for the four
    PRIMA methods.

    Parameters
    ----------
    fun : callable
        A plain ``@njit`` function called ``fun(x, *args)``, in either of two
        shapes: ``-> f`` returns the objective, ``-> (f, g)`` returns it
        together with its gradient. The return type is read when the call
        compiles, so neither shape needs a flag and neither costs an
        evaluation.
    x0 : array_like
        Initial guess, one dimension.
    args : tuple, optional
        Extra parameters, unpacked into the argument list of `fun` and `jac`
        after `x`, so ``args=(a, b)`` calls ``fun(x, a, b)``. Its elements
        may be of different types and shapes. Default ``()``, which calls
        ``fun(x)``. A non-tuple `args` is a one-item tuple, so an ndarray
        reaches `fun` as ONE argument.
    method : str or None, optional
        ``'Nelder-Mead'``, ``'Powell'``, ``'CG'``, ``'BFGS'``,
        ``'L-BFGS-B'``, ``'SLSQP'``, ``'COBYLA'``, or one of the additional
        ``'UOBYQA'``, ``'NEWUOA'``, ``'BOBYQA'``, ``'LINCOA'``. Matched
        case-insensitively. ``None`` (default) selects ``'SLSQP'`` with
        constraints, ``'L-BFGS-B'`` with bounds and ``'BFGS'`` otherwise.
        From inside ``@njit`` it must be a literal written at the call
        site: the result's field set is the method's, and a compiled
        function has one return type per signature. A variable there raises
        `numba.TypingError`. The Python entry takes any string.
    jac : callable, bool, int or None, optional
        ``jac(x, *args) -> array``, the gradient, called with the same
        `args` as `fun`.
        ``None`` (default), ``False`` and ``0`` all mean no gradient: the
        gradient then comes from an ``fg``-style `fun`, or, where the method
        needs one and `fun` returns only ``f``, from forward differences.
    hess, hessp : callable or None, optional
        Accepted and ignored, with a ``RuntimeWarning`` naming the method.
        No method reached from here uses second-order information.
    bounds : (n, 2) array_like or None, optional
        One ``(min, max)`` pair per variable. ``None`` (default) is
        unbounded. Use ``-np.inf`` / ``np.inf`` for a one-sided bound. Used
        by ``'L-BFGS-B'``, ``'SLSQP'``, ``'COBYLA'``, ``'BOBYQA'`` and
        ``'LINCOA'``. The other six warn and ignore them. A ``(0, 2)`` array
        is also unbounded; it counts as bounds for the `method` default,
        because that is settled from the type rather than the length. See
        Notes.
    constraints : tuple, optional
        Accepted only as empty. ``'SLSQP'`` and ``'COBYLA'`` raise on a
        non-empty one; the other nine warn and ignore it.
        :func:`~scijit.optimize.fmin_slsqp` and
        :func:`~scijit.optimize.fmin_cobyla` take constraints.
    tol : float or None, optional
        Headline tolerance for the chosen method. ``None`` (default) uses
        that method's own: ``xatol``/``fatol`` 1e-4 for Nelder-Mead,
        ``xtol``/``ftol`` 1e-4 for Powell, ``gtol`` 1e-5 for CG and BFGS,
        ``gtol`` 1e-5 and ``ftol`` 2.220446049250313e-09 for L-BFGS-B,
        ``acc`` 1e-6 for SLSQP,
        ``rhoend`` 1e-4 for COBYLA and 1e-6 for the PRIMA four. A value
        given sets every tolerance the method has, so ``0.0`` runs it to its
        iteration cap.
    callback : callable or None, optional
        Called once per iteration as ``callback(xk)``, with `xk` the current
        iterate. Either a plain Python callable or an ``@njit`` function of
        that shape. A Python callable whose single parameter is named
        ``intermediate_result`` is called with an
        :class:`~scijit.optimize.OptimizeResult` instead, carrying ``x``
        and ``fun``. Raising ``StopIteration`` halts the solve, and the result
        then carries ``status`` 99 and ``success`` ``False``. Served by
        ``'Nelder-Mead'``, ``'Powell'``, ``'CG'``, ``'BFGS'``, ``'L-BFGS-B'``
        and ``'SLSQP'``.
    options : dict or None, optional
        Solver options. ``maxiter`` reaches `maxiter`. Three methods have
        a tolerance PAIR and each name reaches a slot of its own: ``xatol``
        and ``fatol`` on Nelder-Mead, ``xtol`` and ``ftol`` on Powell,
        ``ftol`` and ``gtol`` on L-BFGS-B. For the rest, ``xtol``,
        ``ftol``, ``gtol``, ``pgtol``, ``acc``, ``rhoend`` and ``tol``
        reach `tol`;
        ``rhobeg`` reaches ``'COBYLA'`` and the PRIMA four, ``npt``
        reaches ``'NEWUOA'``, ``'BOBYQA'`` and ``'LINCOA'``, ``maxfev``
        reaches ``'Nelder-Mead'`` and ``'Powell'``, and ``maxfun``
        reaches ``'L-BFGS-B'``. Any other key
        draws an ``OptimizeWarning`` naming it and is ignored, and the key
        set is per method. From inside
        ``@njit`` this is a dict LITERAL at the call site.
    lower, upper : ndarray, optional
        The bounds as two separate arrays, which is the form the Fortran
        drivers take. Each is empty (default) or length n. Ignored when
        `bounds` is non-empty.
    maxiter : int, optional
        Iteration budget. ``0`` (default) uses the method default: ``200*n``
        for Nelder-Mead, CG and BFGS, ``1000*n`` for Powell, 15000 for
        L-BFGS-B, 100 for SLSQP, 1000 objective evaluations for COBYLA and
        ``500*n`` for the PRIMA four.

    Returns
    -------
    res : OptimizeResult
        `x` is the minimizer and `fun` its objective value. Every method
        also carries `nfev`, `status`, `message` and `success`. The rest of
        the field set is the METHOD's, and a field the method did not
        compute is absent:

        ==============  ===============================================
        method          also carries
        ==============  ===============================================
        'Nelder-Mead'   ``nit``, ``final_simplex``
        'Powell'        ``nit``, ``direc``
        'CG'            ``nit``, ``jac``, ``njev``
        'BFGS'          ``nit``, ``jac``, ``njev``, ``hess_inv``
        'L-BFGS-B'      ``nit``, ``jac``, ``njev``, ``hess_inv``
        'SLSQP'         ``nit``, ``jac``, ``njev``, ``multipliers``
        'COBYLA'        ``maxcv``
        the PRIMA four  nothing further
        ==============  ===============================================

        Reading an absent field raises `AttributeError`, and
        `numba.TypingError` from inside ``@njit``. ``res.keys()`` lists what
        a given result holds.

    Raises
    ------
    ValueError
        If `method` is not one of the eleven; if `constraints` is
        non-empty on ``'SLSQP'`` or
        ``'COBYLA'``; if `jac` is neither ``None`` nor an ``@njit`` function;
        if `jac` is given together with an ``fg``-style `fun`; if `bounds`
        is not ``(n, 2)`` or
        has an upper bound below its lower one; if `x0` has more than one
        dimension; if `lower` or `upper` has a length that is neither 0 nor
        n; or if `fun` or `jac` returns a gradient whose length is not
        ``len(x0)``.

        From inside ``@njit`` the first four of those are
        `numba.TypingError` instead, because the condition is decided while
        the call compiles rather than while it runs.

        A `callback` given to ``'COBYLA'``, ``'UOBYQA'``, ``'NEWUOA'``,
        ``'BOBYQA'`` or ``'LINCOA'`` raises `ValueError`, and
        `numba.TypingError` from inside ``@njit`` where `method` is a literal.
        Those five reach PRIMA, whose wrapper passes no ``callback_fcn``.
    TypeError
        If `fun` or `jac` does not bind ``(x, *args)`` for the `args` given,
        which from inside ``@njit`` is a `numba.TypingError`; if `x0` is
        complex, in any of the array, list and tuple spellings, from both
        entry points; if `options` is neither a mapping nor ``None``, which
        from inside ``@njit`` is a `numba.TypingError`.

    See Also
    --------
    scipy.optimize.minimize : The scipy routine this mirrors.
    scijit.optimize.fmin_l_bfgs_b : L-BFGS-B with its full control set.
    scijit.optimize.fmin_slsqp : SLSQP with constraints.
    scijit.optimize.fmin_cobyla : COBYLA with nonlinear constraints.
    scijit.optimize.minimize_scalar : One variable, no starting point.

    Notes
    -----
    THE METHOD TABLE. ``gradient`` is what the method does with `jac`;
    ``bounds`` is whether it uses them; ``prange`` is whether concurrent
    calls are safe.

    ==============  ==========  ======  ======  =========
    method          gradient    bounds  prange  backend
    ==============  ==========  ======  ======  =========
    'Nelder-Mead'   ignored     no      yes     pure port
    'Powell'        ignored     no      yes     pure port
    'CG'            uses        no      yes     pure port
    'BFGS'          uses        no      yes     pure port
    'L-BFGS-B'      uses        yes     yes     Fortran
    'SLSQP'         uses        yes     yes     Fortran
    'COBYLA'        ignored     yes     yes     Fortran
    'UOBYQA'        ignored     no      yes     Fortran
    'NEWUOA'        ignored     no      yes     Fortran
    'BOBYQA'        ignored     yes     yes     Fortran
    'LINCOA'        ignored     yes     yes     Fortran
    ==============  ==========  ======  ======  =========

    Every method is safe to call from a ``numba.prange`` loop. The
    reverse-communication solvers keep their state in caller-owned arrays;
    PRIMA reaches its callback through a Fortran module variable carrying
    ``!$omp threadprivate``, so that resolves to one slot per thread. The
    32-thread measurement behind the PRIMA half is in
    `scijit.optimize.minimize_newuoa`; the pure ports hold no shared state
    to corrupt.

    `constraints` is empty for every method. Nonlinear constraints reach
    COBYLA through :func:`~scijit.optimize.fmin_cobyla`, equality and
    inequality constraints reach SLSQP through
    :func:`~scijit.optimize.fmin_slsqp`, and linear constraints reach LINCOA
    through :func:`~scijit.optimize.minimize_lincoa`.

    ``nit`` and ``jac``. Nelder-Mead, Powell, CG, BFGS, L-BFGS-B and SLSQP
    report an iteration count. COBYLA and the PRIMA four count objective
    evaluations instead and carry no ``nit``. CG, BFGS, L-BFGS-B and SLSQP
    report the gradient at the solution; the other seven carry no ``jac``.

    THE RESULT'S FIELD ORDER is the same for every method: ``x``, ``fun``,
    the method's own outputs, the counters, then ``status``, ``message``,
    ``success``. scipy's orders differ from each other, ``'BFGS'`` and
    ``'L-BFGS-B'`` carrying one field set in two orders, so there is no one
    order to match. Reaching a field by name or by attribute is unaffected.

    ``multipliers`` on ``'SLSQP'`` holds the constraint multipliers of the
    quadratic subproblem the solver finished on, one per constraint, the
    equalities first. Its length is the number of constraints, so ``(0,)``
    where there are none. Bound rows contribute no entry.

    WHICH SCIPY METHODS ARE ABSENT. ``'Newton-CG'``, ``'TNC'``, ``'COBYQA'``,
    ``'dogleg'``, ``'trust-ncg'``, ``'trust-krylov'``, ``'trust-exact'`` and
    ``'trust-constr'`` raise ``ValueError('Unknown solver <method>')``, which
    is scipy's own text for a name it does not have. A callable `method`,
    which scipy forwards ``**options`` to, raises as well.

    scipy 1.18's ``'COBYLA'`` is PRIMA, which is the library behind this one
    too, so the two run the same implementation rather than two versions of
    the same idea.

    THE OBJECTIVE. Both shapes are accepted and told apart by the return
    type. An ``fg`` objective handed to a method that takes a scalar one is
    split into two compiled functions, each of which calls it and drops half
    of what it computed: the evaluation counts match scipy's, the work per
    evaluation does not. A scalar objective with `jac` handed to SLSQP is
    joined into one ``fg`` for the same reason.

    `args` is unpacked into the objective's argument list, ``fun(x, *args)``.
    A non-tuple `args` is read as a one-item tuple and so reaches `fun` as one
    argument. The four PRIMA methods reach the objective through a C function
    pointer, which carries one ``double*``: the elements cross it flattened
    and are rebuilt before the call, so the same spellings work.

    `jac` accepts ``None``, ``False``, ``0`` or a compiled gradient. scipy's
    ``'2-point'``, ``'3-point'`` and ``'cs'`` are not implemented;
    ``jac=None`` already forward-differences on the methods that need a
    gradient, which is what scipy's ``'2-point'`` does. ``jac=True`` is
    spelled by having `fun` return ``(f, g)``. From inside ``@njit``,
    ``False`` and ``0`` have to be written at the call site: a variable
    holding one is refused, because its value is not known when the call
    compiles, and scipy accepts it.

    `constraints` is unavailable as a list of dicts holding callables.

    THE CALLBACK. From inside ``@njit`` only an ``@njit`` `callback` is
    reachable, because a Python callable cannot cross into compiled code as
    an argument. A Python `callback` reaches the solver through a
    module-level slot and takes the GIL once per iteration, so two solves
    running at once in a ``numba.prange`` loop overwrite each other's
    callback and the loop stops running in parallel. An ``@njit`` `callback`
    travels as an argument and does neither.

    An exception other than ``StopIteration`` from a Python `callback`
    reaches the caller after the solve has run its exit path, where scipy
    raises it from inside the solver loop, so the traceback carries no solver
    frames. An ``@njit`` `callback` halts the solve on ANY exception, and the
    result then carries the ``status`` 99 a ``StopIteration`` gives.

    `res` is a `scijit.optimize.OptimizeResult`, not scipy's. Its field set
    is the solver's, as scipy's is: COBYLA carries ``maxcv``, Nelder-Mead
    ``final_simplex``, Powell ``direc``, SLSQP ``multipliers``, and BFGS and
    L-BFGS-B ``hess_inv``. A field the method did not compute is absent, and
    reading it raises ``AttributeError`` from Python and a ``TypingError``
    when the call compiles.

    ``'Nelder-Mead'`` and ``'Powell'`` DO NOT USE `bounds`, and warn that they
    cannot. scipy's do use them: on the box ``[(0, 0.5), (0, 0.5)]`` with
    ``(x[0] - 1)**2 + (x[1] - 2)**2`` from ``(0, 0)``, scipy returns
    ``[0.5, 0.5]`` and ``[0.49997985, 0.5]`` where this returns ``[1., 2.]``
    on both. The warning is what makes that visible, and it is a warning
    scipy does not raise.

    A ``(0, 2)`` `bounds` array is unbounded and still counts as bounds for
    the ``method=None`` default, because that is settled from the type rather
    than the length. scipy has no counterpart for this spelling.

    ``x0`` with every variable pinned by equal bounds goes to the solver
    here. scipy short-circuits it and returns a result built without calling
    the solver, whose field set is smaller again.

    A gradient whose length is not ``len(x0)`` raises, and so does a `lower`
    or `upper` of the wrong length. scipy 1.18 raises on neither of the two
    Fortran methods offered here.

    `options` reaches the arguments listed under Parameters, and every
    other key draws an ``OptimizeWarning`` naming it, which is scipy's
    class and scipy's text for a key IT does not read. So a key scipy reads
    and this does not, ``disp``, ``return_all``, ``maxcor``,
    ``eps``, ``maxls``, ``initial_simplex``, ``direc``, ``norm``,
    ``adaptive``, ``catol`` and ``iprint`` among them, is announced rather
    than honoured.

    ``tol`` and `tol` name one quantity twice, and so do ``maxiter`` and
    `maxiter`. Where both are given, ``options`` wins, which is scipy's
    ``setdefault`` order.

    A complex `x0` raises ``TypeError``. scipy 1.18 accepts one and returns
    ``complex128``, and the imaginary part never moves from its starting
    value: measured on ``f(z) = |z - (2+3j)|**2``, whose minimiser is
    ``2+3j`` and whose minimum is 0, ``x0 = 1+1j`` gives
    ``x = 1.9999999504278225+1j`` and ``fun = 9.0``, ``x0 = 0j`` gives
    ``x = 2.000000136913649+0j`` and ``fun = 9.0``, and ``x0 = 5-2j`` gives
    ``x = 2.0000004414283588-2j`` and ``fun = 9.0``, each with
    ``success=True``.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Examples
    --------
    The objective returns the value and the gradient together:

    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import minimize
    >>> @njit
    ... def fg(x):
    ...     f = (x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2
    ...     return f, np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.5)])
    >>> @njit
    ... def run():
    ...     return minimize(fg, np.array([0.0, 0.0]))
    >>> res = run()
    >>> res.x
    array([1. , 2.5])
    >>> res.success
    True

    scipy's shape, an objective returning only the value:

    >>> @njit
    ... def f(x):
    ...     return (x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2
    >>> @njit
    ... def run_nm():
    ...     return minimize(f, np.array([0.0, 0.0]), method='Nelder-Mead')
    >>> np.round(run_nm().x, 6)
    array([1.000008, 2.499982])

    Nelder-Mead stops on a simplex smaller than ``xatol``, 1e-4 by default,
    so the last digits above are the tolerance rather than the arithmetic.

    Extra parameters, one array and one scalar, unpacked after `x`:

    >>> @njit
    ... def fa(x, target, w):
    ...     return w * ((x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2)
    >>> @njit
    ... def run_args():
    ...     return minimize(fa, np.array([0.0, 0.0]),
    ...                     args=(np.array([1.0, 2.5]), 3.0))
    >>> np.round(run_args().x, 6)
    array([1. , 2.5])

    A derivative-free PRIMA method, from the same plain function:

    >>> @njit
    ... def run_newuoa():
    ...     return minimize(f, np.array([0.0, 0.0]), method='NEWUOA')
    >>> np.round(run_newuoa().x, 6)
    array([1. , 2.5])

    Bounded, with SLSQP:

    >>> @njit
    ... def run_bounded():
    ...     return minimize(fg, np.array([0.0, 0.0]), method='SLSQP',
    ...                     bounds=np.array([[0.0, 0.5], [0.0, 0.5]]))
    >>> run_bounded().x
    array([0.5, 0.5])
    """
    cbf, use_cb, pycb = _cb_resolve('minimize', callback)
    cons_given = len(constraints) != 0
    if method is None:
        method = _auto_method(bounds is not None, cons_given)
    tol_in = tol
    tol, maxiter, rhobeg, npt = _min_options(
        options, tol, maxiter,
        isinstance(method, str) and method.upper() in _TAKES_MAXFEV,
        isinstance(method, str) and method.upper() in _TAKES_MAXFUN)
    if not isinstance(method, str):
        raise ValueError(_MIN_METHOD_TYPE_MSG)
    m = method.upper()
    if m not in _METHOD_NAMES:
        raise ValueError(_MIN_UNKNOWN + method)
    if cons_given and m in _TAKES_CONS:
        raise ValueError(_MIN_CONS_MSG)
    if use_cb and m not in _TAKES_CALLBACK:
        raise ValueError(_MIN_CB_METHOD_MSG)
    jac_absent = _jac_absent(jac)
    _emit_min_warnings(
        method, _min_warn_mask(m, not jac_absent, hess is not None,
                               hessp is not None, cons_given,
                               bounds is not None))
    xf, lo, up = _min_prep(x0, bounds, lower, upper)

    # Only a plain @njit objective is accepted; a raw @cfunc .address is
    # refused, including for the derivative-free PRIMA methods.
    if not isinstance(fun, Dispatcher):
        raise ValueError(_MIN_FUN_MSG)

    if jac_absent:
        jac_d = None
    elif isinstance(jac, Dispatcher):
        jac_d = jac
    else:
        raise ValueError(_MIN_JAC_TYPE_MSG)
    atypes = _args_types(args)
    is_fg = _fun_shape(typeof(fun), atypes, 'fun')[0]
    if jac_d is not None:
        _fun_shape(typeof(jac_d), atypes, 'jac')
    if is_fg and jac_d is not None:
        raise ValueError(_MIN_JAC_FG_MSG)
    f_s, g_s, has_g, fg = _plan(fun, jac_d, is_fg)

    prev = _cb_install(pycb)
    try:
        if m == 'NELDER-MEAD':
            nm_xa, nm_fa = _min_opt_nm(options, tol_in)
            nm_mf, nm_hm = _min_opt_mf(options)
            return _run_nm(f_s, xf, args, nm_xa, maxiter, cbf, use_cb,
                           nm_fa, nm_mf, nm_hm)
        if m == 'POWELL':
            pw_x, pw_f = _min_opt_pw(options, tol_in)
            pw_mf, pw_hm = _min_opt_mf(options)
            return _run_powell(f_s, xf, args, pw_x, maxiter, cbf, use_cb,
                               pw_f, pw_mf, pw_hm)
        if m == 'CG':
            return _run_cg(f_s, g_s, not has_g, xf, args, tol, maxiter,
                           cbf, use_cb)
        if m == 'BFGS':
            return _run_bfgs(f_s, g_s, not has_g, xf, args, tol, maxiter,
                             cbf, use_cb)
        if m == 'L-BFGS-B':
            lb_f, lb_g = _min_opt_lb(options, tol_in)
            lb_mf, lb_hm = _min_opt_mfun(options)
            if is_fg:
                return _run_lbfgsb(fg, xf, args, lo, up, lb_f, maxiter,
                                   cbf, use_cb, lb_g, lb_mf, lb_hm)
            return _run_lbfgsb_sep(f_s, g_s, has_g, xf, args, lo, up, lb_f,
                                   maxiter, cbf, use_cb, lb_g, lb_mf, lb_hm)
        if m == 'SLSQP':
            if fg is not None:
                return _run_slsqp_fg(fg, xf, args, lo, up, tol, maxiter,
                                     cbf, use_cb)
            return _run_slsqp_sep(f_s, g_s, has_g, xf, args, lo, up, tol,
                                  maxiter, cbf, use_cb)
        if m == 'COBYLA':
            return _run_cobyla(f_s, _no_cons, xf, args, lo, up, tol, maxiter,
                               rhobeg)
        addr = _adapter_star(f_s.py_func, _arg_kinds_ty(atypes)).address
        return _run_prima(addr, _PRIMA_CODE[m], xf,
                          _prepend_n_args(_prima_args(args), xf.size),
                          lo, up, tol, maxiter, rhobeg, npt)
    finally:
        _cb_release(prev)


@overload(minimize, prefer_literal=True)
def _minimize_ovl(fun, x0, args=(), method=None, jac=None,
                  hess=None, hessp=None, bounds=None,
                  constraints=(), tol=None, callback=None, options=None,
                  lower=None, upper=None, maxiter=0):
    """@njit implementation of `minimize`.

    A literal `method` selects one backend and compiles that alone, and
    fixes the result type, which is what the ``prefer_literal=True`` split
    is for. A `method` that is a value is refused here, in the chooser, so
    the refusal is a compile-time `numba.TypingError`.

    The objective's shape, ``f`` or ``(f, g)``, is read from its return type
    here, and where the backend needs a ``@cfunc`` the adapter is built here
    and its address frozen into the body as a constant.
    """
    CBF, USE_CB = _cb_resolve_ty('minimize', callback)

    # The five ignored-argument flags.  Every one of them is decidable here,
    # so the warning mask a call emits is a constant in the compiled body
    # except where `method` itself is a value.
    JAC_ABSENT = _jac_absent_ty(jac)
    JAC_GIVEN = not JAC_ABSENT
    HESS_GIVEN = not _is_none(hess)
    HESSP_GIVEN = not _is_none(hessp)
    BOUNDS_GIVEN = not _is_none(bounds)
    CONS_GIVEN = _cons_given_ty(constraints)

    # `m` is the method, which must be known HERE: it fixes both the
    # backend that compiles and the result TYPE.  A value rather than a
    # literal is refused below, so every branch of the dispatch that
    # returns leaves `m` a string: the three that assign take `.upper()` of
    # a literal or of `_auto_method`, which answers 'BFGS', 'L-BFGS-B' or
    # 'SLSQP' over its whole input space, and the other two raise.
    mt = method.value if isinstance(method, types.Omitted) else method
    if isinstance(mt, types.StringLiteral):
        m = mt.literal_value.upper()
        MNAME = mt.literal_value
    elif isinstance(mt, str):
        m = mt.upper()
        MNAME = mt
    elif mt is None or isinstance(mt, types.NoneType):
        MNAME = _auto_method(BOUNDS_GIVEN, CONS_GIVEN)
        m = MNAME.upper()
    elif isinstance(mt, types.UnicodeType):
        #  A RUNTIME method string.  Refused HERE, in the chooser, which runs
        #  in Python while the call compiles, so the refusal is a compile-time
        #  TypingError.  It must NOT be an impl that raises: such an arm types
        #  as returning `none` and the caller fails on the return type first,
        #  so the message is never seen.
        raise TypingError(_MIN_METHOD_RUNTIME_MSG)
    else:
        raise TypingError(_MIN_METHOD_TYPE_MSG)

    MASK = _min_warn_mask(m, JAC_GIVEN, HESS_GIVEN, HESSP_GIVEN,
                          CONS_GIVEN, BOUNDS_GIVEN)
    if CONS_GIVEN and m in _TAKES_CONS:
        # Refused HERE rather than by an impl that raises, for the reason
        # the runtime-`method` refusal gives above: an impl whose body is
        # only a `raise` TYPES AS RETURNING `none`, so a caller that reads
        # `res.x` fails on the return type first and the message is never
        # seen. Measured: "Unknown attribute 'x' of type none". The three
        # refusals in this chooser all take the same route.
        raise TypingError(_MIN_CONS_MSG)

    if m not in _METHOD_NAMES:
        raise TypingError(_MIN_UNKNOWN + MNAME)

    if USE_CB and m not in _TAKES_CALLBACK:
        raise TypingError(_MIN_CB_METHOD_MSG)

    # Only a plain @njit objective is accepted; a raw @cfunc .address is
    # refused, including for the derivative-free PRIMA methods.
    if not isinstance(fun, types.Dispatcher):
        raise TypingError(_MIN_FUN_MSG)
    if JAC_ABSENT:
        jac_d = None
    elif isinstance(jac, types.Dispatcher):
        jac_d = jac.dispatcher
    else:
        raise TypingError(_MIN_JAC_TYPE_MSG)
    atypes = _args_types(args)
    try:
        is_fg = _fun_shape(fun, atypes, 'fun')[0]
        if jac_d is not None:
            _fun_shape(typeof(jac_d), atypes, 'jac')
    except TypeError as exc:
        raise TypingError(str(exc))
    if is_fg and jac_d is not None:
        raise TypingError(_MIN_JAC_FG_MSG)

    F_S, G_S, HAS_G, FG = _plan(fun.dispatcher, jac_d, is_fg)
    USE_FD = not HAS_G
    ADDR = (_adapter_star(F_S.py_func, _arg_kinds_ty(atypes)).address
            if m in _DF_PRIMA else 0)
    WHICH = _PRIMA_CODE.get(m, 0)

    if m == 'NELDER-MEAD':
        def impl(fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None,
                 constraints=(), tol=None, callback=None, options=None,
                 lower=None, upper=None, maxiter=0):
            _min_warn(MNAME, MASK)
            t, maxiter, rhobeg, npt = _min_options(
                options, tol, maxiter, True)
            xf, lo, up = _min_prep(x0, bounds, lower, upper)
            nm_xa, nm_fa = _min_opt_nm(options, tol)
            nm_mf, nm_hm = _min_opt_mf(options)
            return _run_nm(F_S, xf, args, nm_xa, maxiter,
                           CBF, USE_CB, nm_fa, nm_mf, nm_hm)
    elif m == 'POWELL':
        def impl(fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None,
                 constraints=(), tol=None, callback=None, options=None,
                 lower=None, upper=None, maxiter=0):
            _min_warn(MNAME, MASK)
            t, maxiter, rhobeg, npt = _min_options(
                options, tol, maxiter, True)
            xf, lo, up = _min_prep(x0, bounds, lower, upper)
            pw_x, pw_f = _min_opt_pw(options, tol)
            pw_mf, pw_hm = _min_opt_mf(options)
            return _run_powell(F_S, xf, args, pw_x, maxiter,
                               CBF, USE_CB, pw_f, pw_mf, pw_hm)
    elif m == 'CG':
        def impl(fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None,
                 constraints=(), tol=None, callback=None, options=None,
                 lower=None, upper=None, maxiter=0):
            _min_warn(MNAME, MASK)
            t, maxiter, rhobeg, npt = _min_options(
                options, tol, maxiter)
            xf, lo, up = _min_prep(x0, bounds, lower, upper)
            return _run_cg(F_S, G_S, USE_FD, xf, args, t, maxiter,
                           CBF, USE_CB)
    elif m == 'BFGS':
        def impl(fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None,
                 constraints=(), tol=None, callback=None, options=None,
                 lower=None, upper=None, maxiter=0):
            _min_warn(MNAME, MASK)
            t, maxiter, rhobeg, npt = _min_options(
                options, tol, maxiter)
            xf, lo, up = _min_prep(x0, bounds, lower, upper)
            return _run_bfgs(F_S, G_S, USE_FD, xf, args, t,
                             maxiter, CBF, USE_CB)
    elif m == 'L-BFGS-B' and is_fg:
        def impl(fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None,
                 constraints=(), tol=None, callback=None, options=None,
                 lower=None, upper=None, maxiter=0):
            _min_warn(MNAME, MASK)
            t, maxiter, rhobeg, npt = _min_options(
                options, tol, maxiter, False, True)
            xf, lo, up = _min_prep(x0, bounds, lower, upper)
            lb_f, lb_g = _min_opt_lb(options, tol)
            lb_mf, lb_hm = _min_opt_mfun(options)
            return _run_lbfgsb(FG, xf, args, lo, up, lb_f, maxiter,
                               CBF, USE_CB, lb_g, lb_mf, lb_hm)
    elif m == 'L-BFGS-B':
        def impl(fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None,
                 constraints=(), tol=None, callback=None, options=None,
                 lower=None, upper=None, maxiter=0):
            _min_warn(MNAME, MASK)
            t, maxiter, rhobeg, npt = _min_options(
                options, tol, maxiter, False, True)
            xf, lo, up = _min_prep(x0, bounds, lower, upper)
            lb_f, lb_g = _min_opt_lb(options, tol)
            lb_mf, lb_hm = _min_opt_mfun(options)
            return _run_lbfgsb_sep(F_S, G_S, HAS_G, xf, args, lo,
                                   up, lb_f, maxiter, CBF, USE_CB, lb_g,
                                   lb_mf, lb_hm)
    elif m == 'SLSQP' and FG is not None:
        def impl(fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None,
                 constraints=(), tol=None, callback=None, options=None,
                 lower=None, upper=None, maxiter=0):
            _min_warn(MNAME, MASK)
            t, maxiter, rhobeg, npt = _min_options(
                options, tol, maxiter)
            xf, lo, up = _min_prep(x0, bounds, lower, upper)
            return _run_slsqp_fg(FG, xf, args, lo, up, t, maxiter,
                                 CBF, USE_CB)
    elif m == 'SLSQP':
        def impl(fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None,
                 constraints=(), tol=None, callback=None, options=None,
                 lower=None, upper=None, maxiter=0):
            _min_warn(MNAME, MASK)
            t, maxiter, rhobeg, npt = _min_options(
                options, tol, maxiter)
            xf, lo, up = _min_prep(x0, bounds, lower, upper)
            return _run_slsqp_sep(F_S, G_S, HAS_G, xf, args, lo,
                                  up, t, maxiter, CBF, USE_CB)
    elif m == 'COBYLA':
        def impl(fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None,
                 constraints=(), tol=None, callback=None, options=None,
                 lower=None, upper=None, maxiter=0):
            _min_warn(MNAME, MASK)
            t, maxiter, rhobeg, npt = _min_options(
                options, tol, maxiter)
            xf, lo, up = _min_prep(x0, bounds, lower, upper)
            return _run_cobyla(F_S, _no_cons, xf, args, lo, up,
                               t, maxiter, rhobeg)
    else:
        def impl(fun, x0, args=(), method=None, jac=None,
                 hess=None, hessp=None, bounds=None,
                 constraints=(), tol=None, callback=None, options=None,
                 lower=None, upper=None, maxiter=0):
            _min_warn(MNAME, MASK)
            t, maxiter, rhobeg, npt = _min_options(
                options, tol, maxiter)
            xf, lo, up = _min_prep(x0, bounds, lower, upper)
            return _run_prima(ADDR, WHICH, xf,
                              _prepend_n_args(_prima_args(args), xf.size),
                              lo, up, t, maxiter, rhobeg, npt)
    return impl


