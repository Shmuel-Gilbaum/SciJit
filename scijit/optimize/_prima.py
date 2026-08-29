"""Numba-callable PRIMA (Powell's derivative-free optimizers).

Backed by libprima (libprima/prima, Zaikun Zhang's modern-Fortran
reference implementation, BSD-3). These are the successors of the
solvers behind scipy's COBYLA (scipy 1.16+ ships a pure-Python PRIMA
port as minimize(method='COBYLA')).

No gradients required anywhere, the user supplies objective values
only, as a numba @cfunc (pass .address):

    prima_sig      void(x*, f*, args*)            uobyqa/newuoa/
                                                  bobyqa/lincoa
    prima_con_sig  void(x*, f*, constr*, args*)   cobyla

cobyla constraints follow the scijit/scipy convention: fill
constr with c(x) where c >= 0 means feasible (the wrapper converts to
PRIMA's internal <= 0 form).

Solvers (unconstrained -> most constrained):
    minimize_uobyqa   unconstrained, quadratic models (small n)
    minimize_newuoa   unconstrained (npt interpolation points)
    minimize_bobyqa   bound constraints
    minimize_lincoa   linear constraints  Aineq x <= bineq, Aeq x = beq
    fmin_cobyla       nonlinear (+ linear + bound) constraints

The callback state lives in Fortran module variables carrying
`!$omp threadprivate`, so each thread gets its own slot and concurrent
solves are safe: these are callable from a `numba.prange` loop.
"""
from collections import namedtuple

from numba import carray, cfunc, njit, objmode, types
from numba.core.errors import TypingError
from numba.extending import overload
import numpy as np
from .._lib._load import load
import warnings

prima_sig = types.void(types.CPointer(types.double),     # x      (in)
                       types.CPointer(types.double),     # f      (out)
                       types.CPointer(types.double))     # args   (in)
"""numba signature of a PRIMA objective --
``void(float64*, float64*, float64*)``.

scipy has no counterpart: a Python callable crosses no ABI, so nothing in
``scipy.optimize`` describes the shape of a callback.

Every argument is a pointer because Fortran passes by reference; indexing
auto-dereferences inside numba.  The callback returns nothing: it **writes**
the objective value into ``f[0]``.  ``x`` has length n, ``args`` whatever was
passed.  No gradient is ever requested -- these are derivative-free solvers.

Callback **style B**: build a ``@cfunc`` and pass its ``.address``::

    import numpy as np
    from numba import cfunc, njit
    from scijit.optimize._prima import prima_sig, minimize_newuoa

    @cfunc(prima_sig)
    def obj(x, f, args):
        f[0] = (x[0] - args[0]) ** 2 + 10.0 * (x[1] - x[0] ** 2) ** 2

    PTR = obj.address              # take .address at PYTHON level, once

    @njit
    def run():
        x, f, ok, nf, info = minimize_newuoa(
            PTR, np.array([0.0, 0.0]), np.array([1.0]), 1.0, 1e-6, 0, 0)
        return x

Read parameters out of ``args``, not from a captured global -- numba freezes
a ``@cfunc``'s globals at compile time.

Used by :func:`minimize_uobyqa`, :func:`minimize_newuoa`,
:func:`minimize_bobyqa`, :func:`minimize_lincoa` and the
:func:`~scijit.optimize.minimize` front-end.  COBYLA needs
:data:`prima_con_sig` instead -- a different arity, so the two are not
interchangeable and a wrong-signature address is undefined behaviour rather
than an error.
"""

from ._lbfgsb import _lit_bool                # noqa: E402
from ._minpack import (                     # noqa: E402
                       _arg_kinds, _arg_kinds_ty, _args_types, _as_args_tuple,
                       _pack_args, _unpack_lines, _call_cb, _check_arity,
                       _front_pyfunc, _front_pyfunc_ty, _address_msg)

_ADDRESS_MSG_COBYLA = _address_msg(
    'fmin_cobyla',
    'minimize_uobyqa, minimize_newuoa, minimize_bobyqa or minimize_lincoa')
from .._probe import wrote_residual as _wrote_obj
from .._probe import wrote_objcon as _wrote_objcon

prima_con_sig = types.void(types.CPointer(types.double),  # x     (in)
                           types.CPointer(types.double),  # f     (out)
                           types.CPointer(types.double),  # constr(out)
                           types.CPointer(types.double))  # args  (in)
"""numba signature of a COBYLA objective-with-constraints --
``void(float64*, float64*, float64*, float64*)``.

Like :data:`prima_sig` plus a third pointer: write the ``m_nlcon``
nonlinear constraint values into ``constr``.

**Sign convention: ``c(x) >= 0`` means feasible**, the same convention
:func:`~scijit.optimize.fmin_slsqp` uses.  PRIMA internally wants ``<= 0``;
the Fortran adapter negates, so constraints are written in the ``>= 0`` form.

Callback **style B**::

    import numpy as np
    from numba import cfunc
    from scijit.optimize._prima import prima_con_sig

    @cfunc(prima_con_sig)
    def objcon(x, f, c, args):
        f[0] = x[0] ** 2 + x[1] ** 2
        c[0] = x[0] + x[1] - args[0]      # feasible where x0 + x1 >= args[0]

The number of constraints is fixed when the adapter is built, and writing past
it corrupts memory silently.

:func:`fmin_cobyla` builds one of these internally from a plain ``@njit``
objective and its `cons`, and takes those two rather than an address.
"""

_lib, _sig = load(__file__, "libprima")


_uobyqa = _sig(_lib.uobyqa_wrapper, 11)
_newuoa = _sig(_lib.newuoa_wrapper, 12)
_bobyqa = _sig(_lib.bobyqa_wrapper, 14)
_lincoa = _sig(_lib.lincoa_wrapper, 21)
_cobyla = _sig(_lib.cobyla_wrapper, 23)


@njit
def _bounds(n, lower, upper):
    """Expand the optional bound arrays into the dense pair PRIMA reads.

    The three bounded solvers -- BOBYQA, LINCOA and COBYLA -- always read
    a length-``n`` ``xl``/``xu``, while the public entry points accept an empty
    array meaning unbounded, so the absent side is filled with
    ``-inf``/``+inf`` here rather than branched on at every call. The
    length check is explicit because numba does not bounds-check: a short
    ``lower`` would otherwise be read past its end.
    """
    if lower.size != 0 and lower.size != n:
        raise ValueError("prima: lower must be empty or len(x0)")
    if upper.size != 0 and upper.size != n:
        raise ValueError("prima: upper must be empty or len(x0)")
    xl = np.full(n, -np.inf)
    xu = np.full(n, np.inf)
    if lower.size == n:
        for i in range(n):
            xl[i] = lower[i]
    if upper.size == n:
        for i in range(n):
            xu[i] = upper[i]
    return xl, xu


@njit
def _uobyqa_core(funcptr, x0, args=np.zeros(1), rhobeg=1.0,
                    rhoend=1e-6, maxfun=0):
    """Compiled core of :func:`minimize_uobyqa`.

    Takes the callback as a raw ``@cfunc(prima_sig)`` address only. The
    public name resolves a plain ``@njit`` objective to an address and
    documents every argument.

    Parameters
    ----------
    funcptr : int
        ``.address`` of a ``@cfunc(prima_sig)`` writing the objective
        value into ``f[0]``.
    x0 : float64 array, shape (n,)
        Initial guess.  Cast to float64 and copied.
    args : float64 array, optional
        Extra data reachable in the callback as ``args[i]``.  Default
        ``np.zeros(1)`` -- a dummy, since numba needs a concrete type.
    rhobeg : float, optional
        Initial trust-region radius, i.e. the scale of the first
        exploratory steps.  Default 1.0.  Set it to roughly the
        distance over which the objective changes appreciably.
    rhoend : float, optional
        Final trust-region radius -- the resolution the answer is
        wanted to.  Default 1e-6, PRIMA's default.
    maxfun : int, optional
        Objective-evaluation budget.  ``0`` (default) means PRIMA's
        ``500 * n``.

    Returns
    -------
    x : float64 array, shape (n,)
        Best point found.
    f : float
        Objective value at ``x``.
    success : bool
        ``info in (0, 1)``, i.e. SMALL_TR_RADIUS (the radius shrank to
        ``rhoend``) or FTARGET_ACHIEVED.
    nf : int
        Objective evaluations used.
    info : int
        Raw PRIMA exit code: ``0`` small trust-region radius (normal),
        ``1`` target objective reached, ``2`` ``maxfun`` exhausted,
        ``3`` maximum iterations reached, ``7`` NaN in ``x``,
        ``8`` NaN or +inf objective, ``13`` ``x0`` contains NaN.

    Notes
    -----
    Measured on a 2-D Rosenbrock-like objective from ``[0, 0]`` with
    ``rhoend = 1e-8``: converged to the true minimizer within 1.6e-12,
    ``info = 0``, 57 evaluations.

    Safe to call from a ``numba.prange`` loop: the callback address is stored
    in a Fortran module variable carrying ``!$omp threadprivate``, so each
    thread gets its own slot.
    """
    if not _wrote_obj(funcptr, x0, 1, args):
        raise ValueError(
            "the objective callback never wrote f. Check the @cfunc "
            "signature and argument order")
    n = np.int32(x0.size)
    n_ = np.array(n, np.int32)
    x = np.ascontiguousarray(x0.astype(np.float64)).copy()
    f = np.zeros(1, np.float64)
    rb = np.array(rhobeg, np.float64)
    re = np.array(rhoend, np.float64)
    mf = np.array(maxfun if maxfun > 0 else 500 * n, np.int32)
    nf = np.zeros(1, np.int32)
    info = np.zeros(1, np.int32)
    args_ = np.ascontiguousarray(np.asarray(args, np.float64))
    na = np.array(args_.size, np.int32)

    _uobyqa(funcptr, n_.ctypes.data, x.ctypes.data, f.ctypes.data,
            rb.ctypes.data, re.ctypes.data, mf.ctypes.data,
            nf.ctypes.data, info.ctypes.data,
            args_.ctypes.data, na.ctypes.data)
    return x, f[0], info[0] == 0 or info[0] == 1, nf[0], info[0]


@njit
def _newuoa_core(funcptr, x0, args=np.zeros(1), rhobeg=1.0,
                    rhoend=1e-6, maxfun=0, npt=0):
    """Compiled core of :func:`minimize_newuoa`.

    Takes the callback as a raw ``@cfunc(prima_sig)`` address only. The
    public name resolves a plain ``@njit`` objective to an address and
    documents every argument.

    Parameters
    ----------
    funcptr : int
        ``.address`` of a ``@cfunc(prima_sig)``.
    x0 : float64 array, shape (n,)
        Initial guess.  Cast to float64 and copied.
    args : float64 array, optional
        Extra data for the callback.  Default ``np.zeros(1)``.
    rhobeg : float, optional
        Initial trust-region radius.  Default 1.0.
    rhoend : float, optional
        Final trust-region radius.  Default 1e-6, PRIMA's default.
    maxfun : int, optional
        Evaluation budget.  ``0`` (default) means ``500 * n``.
    npt : int, optional
        Number of interpolation points.  ``0`` (default) means the
        recommended ``2n + 1``.  Valid range is ``n + 2`` to
        ``(n+1)(n+2)/2``; more points give a better model at a higher
        per-iteration cost.

    Returns
    -------
    x : float64 array, shape (n,)
        Best point found.
    f : float
        Objective value at ``x``.
    success : bool
        ``info in (0, 1)`` -- SMALL_TR_RADIUS or FTARGET_ACHIEVED.
    nf : int
        Objective evaluations used.
    info : int
        Raw PRIMA exit code; see :func:`minimize_uobyqa` for the list.

    Notes
    -----
    Measured on the same 2-D objective from ``[0, 0]`` with
    ``rhoend = 1e-8``: within 1.6e-10 of the true minimizer,
    ``info = 0``, 80 evaluations -- more evaluations than UOBYQA but
    each far cheaper.

    Safe to call from a ``numba.prange`` loop: the Fortran module variable
    holding the callback carries ``!$omp threadprivate``.
    """
    if not _wrote_obj(funcptr, x0, 1, args):
        raise ValueError(
            "the objective callback never wrote f. Check the @cfunc "
            "signature and argument order")
    n = np.int32(x0.size)
    n_ = np.array(n, np.int32)
    x = np.ascontiguousarray(x0.astype(np.float64)).copy()
    f = np.zeros(1, np.float64)
    rb = np.array(rhobeg, np.float64)
    re = np.array(rhoend, np.float64)
    mf = np.array(maxfun if maxfun > 0 else 500 * n, np.int32)
    np_ = np.array(npt if npt > 0 else 2 * n + 1, np.int32)
    nf = np.zeros(1, np.int32)
    info = np.zeros(1, np.int32)
    args_ = np.ascontiguousarray(np.asarray(args, np.float64))
    na = np.array(args_.size, np.int32)

    _newuoa(funcptr, n_.ctypes.data, x.ctypes.data, f.ctypes.data,
            rb.ctypes.data, re.ctypes.data, mf.ctypes.data,
            np_.ctypes.data, nf.ctypes.data, info.ctypes.data,
            args_.ctypes.data, na.ctypes.data)
    return x, f[0], info[0] == 0 or info[0] == 1, nf[0], info[0]


@njit
def _bobyqa_core(funcptr, x0, lower=np.zeros(0), upper=np.zeros(0),
                    args=np.zeros(1), rhobeg=1.0, rhoend=1e-6,
                    maxfun=0, npt=0):
    """Compiled core of :func:`minimize_bobyqa`.

    Takes the callback as a raw ``@cfunc(prima_sig)`` address only. The
    public name resolves a plain ``@njit`` objective to an address and
    documents every argument.

    Parameters
    ----------
    funcptr : int
        ``.address`` of a ``@cfunc(prima_sig)``.
    x0 : float64 array, shape (n,)
        Initial guess.  Cast to float64 and copied.
    lower, upper : float64 array, shape (n,) or (0,), optional
        Element-wise bounds; ``+-np.inf`` for an unbounded side, an
        empty array (the default) for no bound at all on that side.  A
        length that is neither 0 nor ``n`` raises ``ValueError``.
    args : float64 array, optional
        Extra data for the callback.  Default ``np.zeros(1)``.
    rhobeg : float, optional
        Initial trust-region radius.  Default 1.0.  BOBYQA requires
        room to move: if ``x0`` sits closer than ``rhobeg`` to a bound,
        PRIMA prints a warning and *shifts* ``x0`` away from it.  Lower
        ``rhobeg`` for a narrow box.
    rhoend : float, optional
        Final trust-region radius.  Default 1e-6.
    maxfun : int, optional
        Evaluation budget.  ``0`` (default) means ``500 * n``.
    npt : int, optional
        Interpolation points.  ``0`` (default) means ``2n + 1``.

    Returns
    -------
    x : float64 array, shape (n,)
        Best point found, inside the box.
    f : float
        Objective value at ``x``.
    success : bool
        ``info in (0, 1)``.
    nf : int
        Objective evaluations used.
    info : int
        Raw PRIMA exit code; see :func:`minimize_uobyqa`.

    Notes
    -----
    Measured with an active bound (``upper[0] = 0.9`` on an objective
    whose free minimum is at 1.0): converged to the constrained
    optimum ``[0.9, 0.81]``, ``info = 0``.

    Safe to call from a ``numba.prange`` loop: the Fortran module variable
    holding the callback carries ``!$omp threadprivate``.
    """
    if not _wrote_obj(funcptr, x0, 1, args):
        raise ValueError(
            "the objective callback never wrote f. Check the @cfunc "
            "signature and argument order")
    n = np.int32(x0.size)
    xl, xu = _bounds(n, lower, upper)
    n_ = np.array(n, np.int32)
    x = np.ascontiguousarray(x0.astype(np.float64)).copy()
    f = np.zeros(1, np.float64)
    rb = np.array(rhobeg, np.float64)
    re = np.array(rhoend, np.float64)
    mf = np.array(maxfun if maxfun > 0 else 500 * n, np.int32)
    np_ = np.array(npt if npt > 0 else 2 * n + 1, np.int32)
    nf = np.zeros(1, np.int32)
    info = np.zeros(1, np.int32)
    args_ = np.ascontiguousarray(np.asarray(args, np.float64))
    na = np.array(args_.size, np.int32)

    _bobyqa(funcptr, n_.ctypes.data, x.ctypes.data, xl.ctypes.data,
            xu.ctypes.data, f.ctypes.data, rb.ctypes.data,
            re.ctypes.data, mf.ctypes.data, np_.ctypes.data,
            nf.ctypes.data, info.ctypes.data,
            args_.ctypes.data, na.ctypes.data)
    return x, f[0], info[0] == 0 or info[0] == 1, nf[0], info[0]


@njit
def _lincoa_core(funcptr, x0, Aineq=np.zeros((0, 0)),
                    bineq=np.zeros(0), Aeq=np.zeros((0, 0)),
                    beq=np.zeros(0), lower=np.zeros(0),
                    upper=np.zeros(0), args=np.zeros(1), rhobeg=1.0,
                    rhoend=1e-6, maxfun=0, npt=0):
    """Compiled core of :func:`minimize_lincoa`.

    Takes the callback as a raw ``@cfunc(prima_sig)`` address only. The
    public name resolves a plain ``@njit`` objective to an address and
    documents every argument.

    Parameters
    ----------
    funcptr : int
        ``.address`` of a ``@cfunc(prima_sig)``.
    x0 : float64 array, shape (n,)
        Initial guess.  Cast to float64 and copied.
    Aineq : float64 array, shape (mi, n), optional
        Inequality matrix, ``Aineq @ x <= bineq``.  C-ordered;
        transposed to Fortran layout internally.  Default ``(0, 0)``.
        A shape inconsistent with ``bineq`` raises ``ValueError``.
    bineq : float64 array, shape (mi,), optional
        Inequality right-hand side.  Its length defines ``mi``.
    Aeq : float64 array, shape (me, n), optional
        Equality matrix, ``Aeq @ x = beq``.  Default ``(0, 0)``.
    beq : float64 array, shape (me,), optional
        Equality right-hand side.  Its length defines ``me``.
    lower, upper : float64 array, shape (n,) or (0,), optional
        Element-wise bounds; ``+-np.inf`` or empty for none.
    args : float64 array, optional
        Extra data for the callback.  Default ``np.zeros(1)``.
    rhobeg : float, optional
        Initial trust-region radius.  Default 1.0.
    rhoend : float, optional
        Final trust-region radius.  Default 1e-6.
    maxfun : int, optional
        Evaluation budget.  ``0`` (default) means ``500 * n``.
    npt : int, optional
        Interpolation points.  ``0`` (default) means ``2n + 1``.

    Returns
    -------
    x : float64 array, shape (n,)
        Best point found.
    f : float
        Objective value at ``x``.
    cstrv : float
        Constraint violation at ``x`` -- the amount by which the linear
        constraints are breached, ideally at rounding level.  This
        element is absent from every other solver's return tuple, and
        it is dropped by the :func:`~scijit.optimize.minimize`
        front-end.
    success : bool
        ``info in (0, 1)``.
    nf : int
        Objective evaluations used.
    info : int
        Raw PRIMA exit code; see :func:`minimize_uobyqa`.

    Notes
    -----
    Measured with ``x0 + x1 <= 1.5`` active: converged on the
    constraint face at ``sum(x) = 1.5`` with ``cstrv = 1.4e-08``,
    ``info = 0``.  A residual ``cstrv`` of order ``rhoend`` is normal.

    **Start from a feasible point.**  By Powell's design LINCOA
    *modifies* ``bineq`` to make an infeasible ``x0`` feasible, so an
    infeasible start silently changes the problem being solved rather
    than raising.

    Safe to call from a ``numba.prange`` loop: the Fortran module variable
    holding the callback carries ``!$omp threadprivate``.
    """
    if not _wrote_obj(funcptr, x0, 1, args):
        raise ValueError(
            "the objective callback never wrote f. Check the @cfunc "
            "signature and argument order")
    n = np.int32(x0.size)
    mi = np.int32(bineq.size)
    me = np.int32(beq.size)
    if mi > 0 and (Aineq.ndim != 2 or Aineq.shape[0] != mi
                   or Aineq.shape[1] != n):
        raise ValueError("lincoa: Aineq must be (len(bineq), len(x0))")
    if me > 0 and (Aeq.ndim != 2 or Aeq.shape[0] != me
                   or Aeq.shape[1] != n):
        raise ValueError("lincoa: Aeq must be (len(beq), len(x0))")
    xl, xu = _bounds(n, lower, upper)
    n_ = np.array(n, np.int32)
    mi_ = np.array(mi, np.int32)
    me_ = np.array(me, np.int32)
    x = np.ascontiguousarray(x0.astype(np.float64)).copy()
    # column-major (m, n) = C-contiguous transpose
    ait = np.ascontiguousarray(Aineq.astype(np.float64).T).copy()
    aet = np.ascontiguousarray(Aeq.astype(np.float64).T).copy()
    bi = np.ascontiguousarray(np.asarray(bineq, np.float64))
    be = np.ascontiguousarray(np.asarray(beq, np.float64))
    f = np.zeros(1, np.float64)
    cstrv = np.zeros(1, np.float64)
    rb = np.array(rhobeg, np.float64)
    re = np.array(rhoend, np.float64)
    mf = np.array(maxfun if maxfun > 0 else 500 * n, np.int32)
    np_ = np.array(npt if npt > 0 else 2 * n + 1, np.int32)
    nf = np.zeros(1, np.int32)
    info = np.zeros(1, np.int32)
    args_ = np.ascontiguousarray(np.asarray(args, np.float64))
    na = np.array(args_.size, np.int32)

    _lincoa(funcptr, n_.ctypes.data, x.ctypes.data, mi_.ctypes.data,
            ait.ctypes.data, bi.ctypes.data, me_.ctypes.data,
            aet.ctypes.data, be.ctypes.data, xl.ctypes.data,
            xu.ctypes.data, f.ctypes.data, cstrv.ctypes.data,
            rb.ctypes.data, re.ctypes.data, mf.ctypes.data,
            np_.ctypes.data, nf.ctypes.data, info.ctypes.data,
            args_.ctypes.data, na.ctypes.data)
    return (x, f[0], cstrv[0], info[0] == 0 or info[0] == 1,
            nf[0], info[0])


#: ``minimize(method='COBYLA')``'s result keys, as a namedtuple.
CobylaResult = namedtuple('CobylaResult',
                          ['x', 'fun', 'maxcv', 'message', 'nfev', 'status',
                           'success'])

_M_NLCON_REQUIRED = (
    "m_nlcon is required when func is a raw @cfunc .address: prima_con_sig "
    "carries no constraint count in any argument or type, so it cannot be "
    "read from the callback. Pass plain @njit f(x, args) and cons(x, args) "
    "instead and it is inferred, as scipy infers it")

# One message per rejection, shared by the python body and the chooser, so
# both entry points name the same argument and give the same reason.
_CALLBACK_MSG = (
    "fmin_cobyla: callback must be None. PRIMA's cobyla takes a callback_fcn "
    "and src/prima/wrappers.f90 passes none, so there is no slot for one to "
    "arrive through.")

_CONS_REQUIRED = (
    "fmin_cobyla: cons is required. Pass a plain @njit c(x, *consargs) -> "
    "array of all constraints, feasible where the result is >= 0, or a "
    "sequence of such functions")

#: scipy's own text for a `cons` that is neither callable nor a sequence of
#: callables, `_cobyla_py.py:146-147`.
_CONS_TYPE_ERR = ("cons must be a sequence of callable functions or a single"
                  " callable function.")


def _cons_seq(cons):
    """`cons` as a sequence, by scipy's algorithm at `_cobyla_py.py:148-158`.

    The decisions are CPython's rather than a shape test: ``len`` says whether
    it is a sequence, and ``callable`` decides each element.  Reproducing the
    algorithm rather than testing the shape is what makes a string, a numpy
    array and a list holding one non-callable raise the same way they do in
    scipy.
    """
    try:
        len(cons)
    except TypeError as exc:
        if callable(cons):
            return [cons]
        raise TypeError(_CONS_TYPE_ERR) from exc
    for thisfunc in cons:
        if not callable(thisfunc):
            raise TypeError(_CONS_TYPE_ERR)
    return cons


def _cons_callable_ty(t):
    """Whether a numba type is one a compiled call could invoke."""
    return isinstance(t, (types.Dispatcher, types.Function,
                          types.FunctionType))


def _cons_msg_ty(cons):
    """The refusal text for a `cons` type the chooser cannot resolve.

    scipy's text where scipy would refuse it too, and the ``@njit`` spelling
    where the object is callable and scipy would have accepted it.
    """
    if isinstance(cons, types.BaseTuple):
        ok = len(cons.types) > 0 and all(_cons_callable_ty(c)
                                         for c in cons.types)
    else:
        ok = _cons_callable_ty(cons)
    return _CONS_REQUIRED if ok else _CONS_TYPE_ERR

_COBYLA_FUNC_MSG = (
    "fmin_cobyla: func must be a plain @njit f(x, *args). A python callable "
    "cannot be reached from compiled code.")


# --------------------------------------------------------------------------
# Hiding the @cfunc.  A user may pass a raw prima_con_sig address (the
# historical spelling) OR a plain @njit objective plus a plain @njit
# constraint function.  For the latter the pointer-shaped adapter is built
# here, once, and cached -- the cache also OWNS the cfunc.
# --------------------------------------------------------------------------
_COBYLA_ADAPTERS = {}


@njit
def _c1d(v):
    """A constraint value as a 1-D float64 array, scalar or array in.

    scipy's `cons` entries return a scalar each and its `NonlinearConstraint`
    also accepts a vector-valued one. Both spellings reach PRIMA through the
    same buffer, so they are normalised here rather than at every call site.
    """
    return np.atleast_1d(np.asarray(v)).astype(np.float64)


#: Fused constraint dispatchers, keyed on the tuple of python functions, so
#: repeated calls with the same `cons` reuse one compiled chain and one cfunc.
_FUSED_CONS = {}


def _cons_fn(src, ns):
    """Compile one generated constraint wrapper."""
    ns = dict(ns, np=np, _c1d=_c1d, njit=njit)
    exec(src, ns)                                        # noqa: S102
    return njit(ns['fn'])


def _pair_cons(left, right, argl):
    """``c(x, *consargs)`` concatenating two constraint functions' values."""
    return _cons_fn(
        "def fn(x%s):\n"
        "    return np.concatenate((_c1d(left(x%s)), _c1d(right(x%s))))\n"
        % (argl, argl, argl), {'left': left, 'right': right})


def _one_cons(only, argl):
    """``c(x, *consargs)`` normalising one constraint function's value."""
    return _cons_fn("def fn(x%s):\n    return _c1d(only(x%s))\n"
                    % (argl, argl), {'only': only})


def _fuse_cons(pycs, ncargs=1):
    """One ``@njit c(x, *consargs) -> array`` from a sequence of them.

    Folded pairwise, so each closure captures exactly two compiled functions
    as freevars and numba never has to iterate a tuple of functions.
    """
    key = (tuple(pycs), ncargs)
    hit = _FUSED_CONS.get(key)
    if hit is not None:
        return hit
    argl = "".join(", c%d" % i for i in range(ncargs))
    fused = _one_cons(njit(pycs[0]), argl)
    for py in pycs[1:]:
        fused = _pair_cons(fused, njit(py), argl)
    _FUSED_CONS[key] = fused
    return fused


def _cons_pyfuncs(cons):
    """The python functions behind `cons`, or ``None`` if it is not one.

    Accepts a single ``@njit`` function or a sequence of them, which is
    scipy's own spelling.
    """
    if hasattr(cons, 'py_func'):
        return [cons.py_func]
    if isinstance(cons, (list, tuple)):
        if not all(hasattr(c, 'py_func') for c in cons):
            return None
        return [c.py_func for c in cons]
    return None


def _cons_pyfuncs_ty(cons):
    """`_cons_pyfuncs` for a chooser, reading numba types."""
    if isinstance(cons, types.Dispatcher):
        return [cons.dispatcher.py_func]
    if isinstance(cons, types.BaseTuple):
        if not all(isinstance(c, types.Dispatcher) for c in cons.types):
            return None
        return [c.dispatcher.py_func for c in cons.types]
    return None


def _adapter_objcon(pyf, pyc, kinds=(), ckinds=()):
    """cfunc(prima_con_sig) around @njit ``f(x, *args) -> float`` and
    ``c(x, *consargs) -> array``.

    `pyc` is either a python function or the already-compiled fusion of a
    sequence of them that `_fuse_cons` returns.
    """
    key = (pyf, pyc, 'objcon', kinds, ckinds)
    hit = _COBYLA_ADAPTERS.get(key)
    if hit is not None:
        return hit
    f_in = njit(pyf)
    c_in = pyc if hasattr(pyc, 'py_func') else njit(pyc)
    # Buffer layout, from `_prepend_nm_args`:
    #   [n, m, len(args), len(consargs), *args, *consargs]
    aun, aargl = _unpack_lines(kinds, 4, ' ' * 8)
    cun, cargl = _unpack_lines(ckinds, 0, ' ' * 8, p='g', buf='cbuf', ctr='q')
    ns = {'cfunc': cfunc, 'carray': carray, 'np': np,
          'prima_con_sig': prima_con_sig, 'f_in': f_in, 'c_in': c_in}
    exec(_OBJCON_SRC % {'aunpack': aun, 'aargl': aargl,      # noqa: S102
                        'cunpack': cun, 'cargl': cargl}, ns)
    adapter = ns['_make']()
    _COBYLA_ADAPTERS[key] = adapter
    return adapter


_OBJCON_SRC = """
def _make():
    @cfunc(prima_con_sig)
    def adapter(x_ptr, f_ptr, c_ptr, a_ptr):
        n = int(a_ptr[0])
        m = int(a_ptr[1])
        na = int(a_ptr[2])
        nc = int(a_ptr[3])
        x = carray(x_ptr, n)
        a = carray(a_ptr, 4 + na + nc)
        cbuf = a[4 + na:]
%(aunpack)s
%(cunpack)s
        f_ptr[0] = f_in(x%(aargl)s)
        cv = c_in(x%(cargl)s)
        # NO negation here.  PRIMA wants constr(x) <= 0, but the Fortran
        # `calcfc_adapter` in src/prima/wrappers.f90 already negates, so the
        # cfunc ABI is c(x) >= 0 feasible -- scipy's and SLSQP's convention.
        # Negating again silently drops the constraint.
        for i in range(m):
            c_ptr[i] = cv[i]
    return adapter
"""


#: Objective-only adapters, keyed on the python function behind the
#: dispatcher.  The cache also OWNS the cfunc: nothing else holds a
#: reference, and a collected cfunc leaves a dangling address.
_OBJ_ADAPTERS = {}


def _adapter_obj(pyf):
    """cfunc(prima_sig) around a plain @njit ``f(x, args) -> float``.

    UOBYQA, NEWUOA, BOBYQA and LINCOA take a ``prima_sig`` address. This
    builds one from an ordinary compiled function, so `minimize` can offer
    them the same objective every other method takes. The address is a
    plain int and freezes into an ``@overload`` impl as a constant.

    The adapter reads ``n`` and ``len(args)`` out of the head of the buffer,
    which is what `_prepend_n_args` puts there.
    """
    hit = _OBJ_ADAPTERS.get(pyf)
    if hit is not None:
        return hit
    f_in = njit(pyf)

    @cfunc(prima_sig)
    def adapter(x_ptr, f_ptr, a_ptr):
        n = int(a_ptr[0])
        na = int(a_ptr[1])
        x = carray(x_ptr, n)
        a = carray(a_ptr, 2 + na)[2:]
        f_ptr[0] = f_in(x, a)

    _OBJ_ADAPTERS[pyf] = adapter
    return adapter


@njit
def _prepend_n_args(args, n):
    """args buffer the objective ADAPTER expects: ``[n, len(args), *args]``."""
    out = np.empty(args.size + 2, np.float64)
    out[0] = np.float64(n)
    out[1] = np.float64(args.size)
    for i in range(args.size):
        out[i + 2] = args[i]
    return out


@njit
def _prepend_nm_args(args, cargs, n, m):
    """args buffer the ADAPTER path expects.

    ``[n, m, len(args), len(cargs), *args, *cargs]``. The objective reads the
    first block and the constraints read the second, which is what lets
    `consargs` differ from `args` as it does in scipy.
    """
    out = np.empty(args.size + cargs.size + 4, np.float64)
    out[0] = np.float64(n)
    out[1] = np.float64(m)
    out[2] = np.float64(args.size)
    out[3] = np.float64(cargs.size)
    for i in range(args.size):
        out[i + 4] = args[i]
    for i in range(cargs.size):
        out[i + 4 + args.size] = cargs[i]
    return out


# --------------------------------------------------------------------------
# Hiding the @cfunc on the four derivative-free solvers.
#
# Each takes EITHER a raw `@cfunc(prima_sig)` `.address`, the historical
# spelling, OR a plain `@njit` objective. For the second the adapter is built
# when the call compiles and its address frozen into the body, the same shape
# `minimize` already uses for these four. `_prepend_n_args` supplies the two
# slots the adapter reads.
#
# The objective may be `f(x, args)` or `f(x)`; the arity is read off the
# dispatcher.
# --------------------------------------------------------------------------


from .._lib._typing import _is_none    # noqa: E402


def _as_arr1(v, fill):
    """A 1-D float64 buffer. ``None`` -> length 1 of `fill`, or length 0."""
    if v is None:
        return np.zeros(0) if fill is None else np.full(1, fill)
    return np.ascontiguousarray(np.asarray(v, np.float64)).ravel()


def _as_arr2(v):
    """A 2-D float64 buffer. ``None`` -> ``(0, 0)``."""
    if v is None:
        return np.zeros((0, 0))
    return np.ascontiguousarray(np.asarray(v, np.float64))


_OBJ1_ADAPTERS = {}


def _adapter_obj1(pyf):
    """`_adapter_obj` for an objective written ``f(x)``, with no `args`."""
    hit = _OBJ1_ADAPTERS.get(pyf)
    if hit is not None:
        return hit
    f_in = njit(pyf)

    @cfunc(prima_sig)
    def adapter(x_ptr, f_ptr, a_ptr):
        n = int(a_ptr[0])
        f_ptr[0] = f_in(carray(x_ptr, n))

    _OBJ1_ADAPTERS[pyf] = adapter
    return adapter


def _obj_arity(py):
    """1 for ``f(x)``, 2 for ``f(x, args)``. Anything else raises."""
    na = py.__code__.co_argcount
    if na not in (1, 2):
        raise ValueError(
            "the objective must be f(x) or f(x, args); got %d arguments" % na)
    return na


def _obj_address(py):
    """The adapter address for a plain python objective of either arity."""
    if _obj_arity(py) == 1:
        return _adapter_obj1(py).address
    return _adapter_obj(py).address


def _prima_addr_py(func):
    """(address, pad) for the PYTHON entries. `pad` selects the args prefix."""
    if isinstance(func, (int, np.integer)):
        return int(func), False
    return _obj_address(func.py_func), True


def _prima_addr_ty(func):
    """(address, pad) for the CHOOSERS, or None to decline the call."""
    if isinstance(func, types.Integer):
        return None, False
    if isinstance(func, types.Dispatcher):
        return _obj_address(func.dispatcher.py_func), True
    return None, None


_PRIMA_FUNC_MSG = (
    "the first argument must be the .address of a @cfunc(prima_sig), or a "
    "plain @njit f(x, args) or f(x). A python callable cannot be reached "
    "from compiled code.")


def minimize_uobyqa(funcptr, x0, args=None, rhobeg=1.0, rhoend=1e-6,
                    maxfun=0):
    """Unconstrained derivative-free minimization by UOBYQA.

    Powell's Unconstrained Optimization BY Quadratic Approximation.
    No direct scipy counterpart; ``scipy.optimize.minimize`` has no
    UOBYQA method.  Builds a full quadratic model from
    ``(n+1)(n+2)/2`` interpolation points, so it converges in few
    evaluations but costs O(n^4) per iteration -- use it only for small
    ``n`` (roughly n <= 10) and prefer :func:`minimize_newuoa` above
    that.

    Parameters
    ----------
    funcptr : njit function or int
        The objective.  Either a plain ``@njit`` ``f(x, args)`` or
        ``f(x)`` returning the value, for which the
        ``@cfunc(prima_sig)`` is built and cached internally, or the
        ``.address`` of a ``@cfunc(prima_sig)`` writing the value into
        ``f[0]``.
    x0 : float64 array, shape (n,)
        Initial guess.  Cast to float64 and copied.
    args : float64 array, optional
        Extra data reachable in the objective as ``args[i]``.  Default
        ``None``, which passes a length-1 dummy.
    rhobeg : float, optional
        Initial trust-region radius, i.e. the scale of the first
        exploratory steps.  Default 1.0.  Set it to roughly the
        distance over which the objective changes appreciably.
    rhoend : float, optional
        Final trust-region radius -- the resolution the answer is
        wanted to.  Default 1e-6, which is PRIMA's own, where
        `fmin_cobyla` carries scipy's 1e-4 instead.
    maxfun : int, optional
        Objective-evaluation budget.  ``0`` (default) means PRIMA's
        ``500 * n``, where `fmin_cobyla` carries scipy's 1000.

    Returns
    -------
    x : float64 array, shape (n,)
        Best point found.
    f : float
        Objective value at ``x``.
    success : bool
        ``info in (0, 1)``, i.e. SMALL_TR_RADIUS (the radius shrank to
        ``rhoend``) or FTARGET_ACHIEVED.
    nf : int
        Objective evaluations used.
    info : int
        Raw PRIMA exit code: ``0`` small trust-region radius (normal),
        ``1`` target objective reached, ``2`` ``maxfun`` exhausted,
        ``3`` maximum iterations reached, ``7`` NaN in ``x``,
        ``8`` NaN or +inf objective, ``13`` ``x0`` contains NaN.

    Notes
    -----
    Measured on a 2-D Rosenbrock-like objective from ``[0, 0]`` with
    ``rhoend = 1e-8``: converged to the true minimizer within 1.6e-12,
    ``info = 0``, 57 evaluations.

    Safe to call from a ``numba.prange`` loop: the callback address is
    stored in a Fortran module variable carrying ``!$omp threadprivate``,
    so each thread gets its own slot.
    """
    fp, pad = _prima_addr_py(funcptr)
    a = _as_arr1(args, 1.0)
    x = np.ascontiguousarray(np.asarray(x0, np.float64))
    ab = _prepend_n_args(a, x.size) if pad else a
    return _uobyqa_core(fp, x, ab, rhobeg, rhoend, maxfun)


@overload(minimize_uobyqa)
def _minimize_uobyqa_ovl(funcptr, x0, args=None, rhobeg=1.0,
                         rhoend=1e-6, maxfun=0):
    addr, pad = _prima_addr_ty(funcptr)
    if pad is None:
        raise TypingError(_PRIMA_FUNC_MSG)
    no_args = _is_none(args)

    def impl(funcptr, x0, args=None, rhobeg=1.0, rhoend=1e-6,
             maxfun=0):
        if no_args:
            a = np.full(1, 1.0, np.float64)
        else:
            a = np.ascontiguousarray(
                np.asarray(args).astype(np.float64)).ravel()
        x = np.ascontiguousarray(np.asarray(x0).astype(np.float64))
        if pad:
            return _uobyqa_core(addr, x, _prepend_n_args(a, x.size), rhobeg,
                                rhoend, maxfun)
        return _uobyqa_core(funcptr, x, a, rhobeg, rhoend, maxfun)
    return impl


def minimize_newuoa(funcptr, x0, args=None, rhobeg=1.0, rhoend=1e-6,
                    maxfun=0, npt=0):
    """Unconstrained derivative-free minimization by NEWUOA.

    Powell's NEW Unconstrained Optimization Algorithm.
    No direct scipy counterpart.  Builds an *underdetermined* quadratic
    model from ``npt`` interpolation points, which makes it far cheaper
    per iteration than :func:`minimize_uobyqa` and the usual first
    choice for unconstrained derivative-free work.

    Parameters
    ----------
    funcptr : njit function or int
        The objective.  Either a plain ``@njit`` ``f(x, args)`` or
        ``f(x)`` returning the value, or the ``.address`` of a
        ``@cfunc(prima_sig)``.
    x0 : float64 array, shape (n,)
        Initial guess.  Cast to float64 and copied.
    args : float64 array, optional
        Extra data for the objective.  Default ``None``, which passes a
        length-1 dummy.
    rhobeg : float, optional
        Initial trust-region radius.  Default 1.0.
    rhoend : float, optional
        Final trust-region radius.  Default 1e-6, which is PRIMA's own,
        where `fmin_cobyla` carries scipy's 1e-4 instead.
    maxfun : int, optional
        Evaluation budget.  ``0`` (default) means PRIMA's ``500 * n``,
        where `fmin_cobyla` carries scipy's 1000.
    npt : int, optional
        Number of interpolation points.  ``0`` (default) means the
        recommended ``2n + 1``.  Valid range is ``n + 2`` to
        ``(n+1)(n+2)/2``; more points give a better model at a higher
        per-iteration cost.

    Returns
    -------
    x : float64 array, shape (n,)
        Best point found.
    f : float
        Objective value at ``x``.
    success : bool
        ``info in (0, 1)`` -- SMALL_TR_RADIUS or FTARGET_ACHIEVED.
    nf : int
        Objective evaluations used.
    info : int
        Raw PRIMA exit code; see :func:`minimize_uobyqa` for the list.

    Notes
    -----
    Measured on the same 2-D objective from ``[0, 0]`` with
    ``rhoend = 1e-8``: within 1.6e-10 of the true minimizer,
    ``info = 0``, 80 evaluations -- more evaluations than UOBYQA but
    each far cheaper.

    Safe to call from a ``numba.prange`` loop: the Fortran module
    variable holding the callback carries ``!$omp threadprivate``.
    """
    fp, pad = _prima_addr_py(funcptr)
    a = _as_arr1(args, 1.0)
    x = np.ascontiguousarray(np.asarray(x0, np.float64))
    ab = _prepend_n_args(a, x.size) if pad else a
    return _newuoa_core(fp, x, ab, rhobeg, rhoend, maxfun, npt)


@overload(minimize_newuoa)
def _minimize_newuoa_ovl(funcptr, x0, args=None, rhobeg=1.0,
                         rhoend=1e-6, maxfun=0, npt=0):
    addr, pad = _prima_addr_ty(funcptr)
    if pad is None:
        raise TypingError(_PRIMA_FUNC_MSG)
    no_args = _is_none(args)

    def impl(funcptr, x0, args=None, rhobeg=1.0, rhoend=1e-6,
             maxfun=0, npt=0):
        if no_args:
            a = np.full(1, 1.0, np.float64)
        else:
            a = np.ascontiguousarray(
                np.asarray(args).astype(np.float64)).ravel()
        x = np.ascontiguousarray(np.asarray(x0).astype(np.float64))
        if pad:
            return _newuoa_core(addr, x, _prepend_n_args(a, x.size), rhobeg,
                                rhoend, maxfun, npt)
        return _newuoa_core(funcptr, x, a, rhobeg, rhoend, maxfun, npt)
    return impl


def minimize_bobyqa(funcptr, x0, lower=None, upper=None,
                    args=None, rhobeg=1.0, rhoend=1e-6, maxfun=0,
                    npt=0):
    """Bound-constrained derivative-free minimization by BOBYQA.

    Bound Optimization BY Quadratic Approximation.
    No direct scipy counterpart.  NEWUOA's model machinery with a box
    constraint; every evaluation point stays inside the bounds, which
    matters when the objective is undefined outside them.

    Parameters
    ----------
    funcptr : njit function or int
        The objective.  Either a plain ``@njit`` ``f(x, args)`` or
        ``f(x)`` returning the value, or the ``.address`` of a
        ``@cfunc(prima_sig)``.
    x0 : float64 array, shape (n,)
        Initial guess.  Cast to float64 and copied.
    lower, upper : float64 array, shape (n,), optional
        Element-wise bounds; ``+-np.inf`` for an unbounded side,
        ``None`` (the default) for no bound at all on that side.  A
        length that is neither 0 nor ``n`` raises ``ValueError``.
    args : float64 array, optional
        Extra data for the objective.  Default ``None``, which passes a
        length-1 dummy.
    rhobeg : float, optional
        Initial trust-region radius.  Default 1.0.  BOBYQA requires
        room to move: if ``x0`` sits closer than ``rhobeg`` to a bound,
        PRIMA prints a warning and *shifts* ``x0`` away from it.  Lower
        ``rhobeg`` for a narrow box.
    rhoend : float, optional
        Final trust-region radius.  Default 1e-6, which is PRIMA's own,
        where `fmin_cobyla` carries scipy's 1e-4 instead.
    maxfun : int, optional
        Evaluation budget.  ``0`` (default) means PRIMA's ``500 * n``,
        where `fmin_cobyla` carries scipy's 1000.
    npt : int, optional
        Interpolation points.  ``0`` (default) means ``2n + 1``.

    Returns
    -------
    x : float64 array, shape (n,)
        Best point found, inside the box.
    f : float
        Objective value at ``x``.
    success : bool
        ``info in (0, 1)``.
    nf : int
        Objective evaluations used.
    info : int
        Raw PRIMA exit code; see :func:`minimize_uobyqa`.

    Notes
    -----
    Measured with an active bound (``upper[0] = 0.9`` on an objective
    whose free minimum is at 1.0): converged to the constrained
    optimum ``[0.9, 0.81]``, ``info = 0``.

    Safe to call from a ``numba.prange`` loop: the Fortran module
    variable holding the callback carries ``!$omp threadprivate``.
    """
    fp, pad = _prima_addr_py(funcptr)
    a = _as_arr1(args, 1.0)
    lo, up = _as_arr1(lower, None), _as_arr1(upper, None)
    x = np.ascontiguousarray(np.asarray(x0, np.float64))
    ab = _prepend_n_args(a, x.size) if pad else a
    return _bobyqa_core(fp, x, lo, up, ab, rhobeg, rhoend, maxfun, npt)


@overload(minimize_bobyqa)
def _minimize_bobyqa_ovl(funcptr, x0, lower=None, upper=None,
                         args=None, rhobeg=1.0, rhoend=1e-6,
                         maxfun=0, npt=0):
    addr, pad = _prima_addr_ty(funcptr)
    if pad is None:
        raise TypingError(_PRIMA_FUNC_MSG)
    no_args, no_lo, no_up = _is_none(args), _is_none(lower), _is_none(upper)

    def impl(funcptr, x0, lower=None, upper=None,
             args=None, rhobeg=1.0, rhoend=1e-6, maxfun=0, npt=0):
        if no_args:
            a = np.full(1, 1.0, np.float64)
        else:
            a = np.ascontiguousarray(
                np.asarray(args).astype(np.float64)).ravel()
        if no_lo:
            lo = np.zeros(0, np.float64)
        else:
            lo = np.ascontiguousarray(
                np.asarray(lower).astype(np.float64)).ravel()
        if no_up:
            up = np.zeros(0, np.float64)
        else:
            up = np.ascontiguousarray(
                np.asarray(upper).astype(np.float64)).ravel()
        x = np.ascontiguousarray(np.asarray(x0).astype(np.float64))
        if pad:
            return _bobyqa_core(addr, x, lo, up,
                                _prepend_n_args(a, x.size), rhobeg, rhoend,
                                maxfun, npt)
        return _bobyqa_core(funcptr, x, lo, up, a, rhobeg, rhoend,
                            maxfun, npt)
    return impl


def minimize_lincoa(funcptr, x0, Aineq=None, bineq=None,
                    Aeq=None, beq=None,
                    lower=None, upper=None, args=None,
                    rhobeg=1.0, rhoend=1e-6, maxfun=0, npt=0):
    """Linearly-constrained derivative-free minimization by LINCOA.

    LINearly Constrained Optimization Algorithm.
    Minimizes ``f(x)`` subject to ``Aineq @ x <= bineq``,
    ``Aeq @ x = beq`` and bounds.  No direct scipy counterpart; scipy's
    ``LinearConstraint`` path requires a gradient-based method.
    Constraints are data, not code, so nonlinear ones need
    :func:`fmin_cobyla` instead.

    Parameters
    ----------
    funcptr : njit function or int
        The objective.  Either a plain ``@njit`` ``f(x, args)`` or
        ``f(x)`` returning the value, or the ``.address`` of a
        ``@cfunc(prima_sig)``.
    x0 : float64 array, shape (n,)
        Initial guess.  Cast to float64 and copied.
    Aineq : float64 array, shape (mi, n), optional
        Inequality matrix, ``Aineq @ x <= bineq``.  C-ordered;
        transposed to Fortran layout internally.  Default ``None``.
        A shape inconsistent with ``bineq`` raises ``ValueError``.
    bineq : float64 array, shape (mi,), optional
        Inequality right-hand side.  Its length defines ``mi``.
    Aeq : float64 array, shape (me, n), optional
        Equality matrix, ``Aeq @ x = beq``.  Default ``None``.
    beq : float64 array, shape (me,), optional
        Equality right-hand side.  Its length defines ``me``.
    lower, upper : float64 array, shape (n,), optional
        Element-wise bounds; ``+-np.inf`` or ``None`` for none.
    args : float64 array, optional
        Extra data for the objective.  Default ``None``, which passes a
        length-1 dummy.
    rhobeg : float, optional
        Initial trust-region radius.  Default 1.0.
    rhoend : float, optional
        Final trust-region radius.  Default 1e-6, which is PRIMA's own,
        where `fmin_cobyla` carries scipy's 1e-4 instead.
    maxfun : int, optional
        Evaluation budget.  ``0`` (default) means PRIMA's ``500 * n``,
        where `fmin_cobyla` carries scipy's 1000.
    npt : int, optional
        Interpolation points.  ``0`` (default) means ``2n + 1``.

    Returns
    -------
    x : float64 array, shape (n,)
        Best point found.
    f : float
        Objective value at ``x``.
    cstrv : float
        Constraint violation at ``x`` -- the amount by which the linear
        constraints are breached, ideally at rounding level.  This
        element is absent from every other solver's return tuple, and
        it is dropped by the :func:`~scijit.optimize.minimize`
        front-end.
    success : bool
        ``info in (0, 1)``.
    nf : int
        Objective evaluations used.
    info : int
        Raw PRIMA exit code; see :func:`minimize_uobyqa`.

    Notes
    -----
    Measured with ``x0 + x1 <= 1.5`` active: converged on the
    constraint face at ``sum(x) = 1.5`` with ``cstrv = 1.4e-08``,
    ``info = 0``.  A residual ``cstrv`` of order ``rhoend`` is normal.

    **Start from a feasible point.**  By Powell's design LINCOA
    *modifies* ``bineq`` to make an infeasible ``x0`` feasible, so an
    infeasible start silently changes the problem being solved rather
    than raising.

    Safe to call from a ``numba.prange`` loop: the Fortran module
    variable holding the callback carries ``!$omp threadprivate``.
    """
    fp, pad = _prima_addr_py(funcptr)
    a = _as_arr1(args, 1.0)
    x = np.ascontiguousarray(np.asarray(x0, np.float64))
    ab = _prepend_n_args(a, x.size) if pad else a
    return _lincoa_core(fp, x, _as_arr2(Aineq), _as_arr1(bineq, None),
                        _as_arr2(Aeq), _as_arr1(beq, None),
                        _as_arr1(lower, None), _as_arr1(upper, None), ab,
                        rhobeg, rhoend, maxfun, npt)


@overload(minimize_lincoa)
def _minimize_lincoa_ovl(funcptr, x0, Aineq=None,
                         bineq=None, Aeq=None,
                         beq=None, lower=None,
                         upper=None, args=None, rhobeg=1.0,
                         rhoend=1e-6, maxfun=0, npt=0):
    addr, pad = _prima_addr_ty(funcptr)
    if pad is None:
        raise TypingError(_PRIMA_FUNC_MSG)
    n_a, n_ai, n_bi, n_ae, n_be, n_lo, n_up = (
        _is_none(args), _is_none(Aineq), _is_none(bineq), _is_none(Aeq),
        _is_none(beq), _is_none(lower), _is_none(upper))

    def impl(funcptr, x0, Aineq=None, bineq=None,
             Aeq=None, beq=None, lower=None,
             upper=None, args=None, rhobeg=1.0, rhoend=1e-6,
             maxfun=0, npt=0):
        # Normalised exactly as the python body's `_as_arr1`/`_as_arr2` do.
        # The 1-D arguments are RAVELLED on both sides and the matrices on
        # neither, and an absent `args` is a length-1 buffer of 1.0 on both:
        # on the raw-address path that slot reaches the caller's own
        # ``@cfunc``, where 0.0 and 1.0 are two different objectives.
        if n_a:
            a = np.full(1, 1.0, np.float64)
        else:
            a = np.ascontiguousarray(
                np.asarray(args).astype(np.float64)).ravel()
        if n_ai:
            Ai = np.zeros((0, 0), np.float64)
        else:
            Ai = np.ascontiguousarray(np.asarray(Aineq).astype(np.float64))
        if n_bi:
            bi = np.zeros(0, np.float64)
        else:
            bi = np.ascontiguousarray(
                np.asarray(bineq).astype(np.float64)).ravel()
        if n_ae:
            Ae = np.zeros((0, 0), np.float64)
        else:
            Ae = np.ascontiguousarray(np.asarray(Aeq).astype(np.float64))
        if n_be:
            be = np.zeros(0, np.float64)
        else:
            be = np.ascontiguousarray(
                np.asarray(beq).astype(np.float64)).ravel()
        if n_lo:
            lo = np.zeros(0, np.float64)
        else:
            lo = np.ascontiguousarray(
                np.asarray(lower).astype(np.float64)).ravel()
        if n_up:
            up = np.zeros(0, np.float64)
        else:
            up = np.ascontiguousarray(
                np.asarray(upper).astype(np.float64)).ravel()
        x = np.ascontiguousarray(np.asarray(x0).astype(np.float64))
        if pad:
            return _lincoa_core(addr, x, Ai, bi, Ae, be, lo,
                                up, _prepend_n_args(a, x.size), rhobeg,
                                rhoend, maxfun, npt)
        return _lincoa_core(funcptr, x, Ai, bi, Ae, be, lo, up,
                            a, rhobeg, rhoend, maxfun, npt)
    return impl


#: PRIMA's CTOL_DFT, and scipy's minimize(method='COBYLA') default.
_CTOL_DFT = float(np.sqrt(np.finfo(np.float64).eps))

#: pyprima's text for a `catol` it refuses, `common/preproc.py:242`.  The
#: number is the value it substitutes, so it is rendered from `_CTOL_DFT`.
_CTOL_WARN = ("COBYLA: Invalid CTOL; it should be a nonnegative number; "
              "it is set to " + repr(_CTOL_DFT))


def _emit_ctol_warning():
    warnings.warn(_CTOL_WARN, UserWarning, stacklevel=3)


@njit
def _warn_ctol():
    """The `UserWarning` pyprima emits for a negative or `nan` `catol`.

    ``warnings.warn`` is not typeable by numba; an ``objmode`` block runs its
    body in the interpreter.  The block sits in its own function because
    lowering one pickles the enclosing function.
    """
    with objmode():
        _emit_ctol_warning()


@njit
def _cobyla_status(info, cstrv, catol):
    """scipy's COBYLA ``status``, which is PRIMA's ``info`` unchanged.

    scipy 1.16 replaced Powell's COBYLA with a Python translation of PRIMA and
    reports the solver's own code: ``status=result.info`` in
    ``_cobyla_py._minimize_cobyla``.  So the code passes straight through, with
    no mapping onto the pre-1.16 1/2/3/4/5 table.  ``cstrv`` and ``catol`` are
    taken and ignored: the constraint post-check belongs to ``success``, which
    is where scipy 1.18 applies it.
    """
    return info


@njit
def _cobyla_success(info, cstrv, catol):
    """scipy 1.18's COBYLA ``success``.

    ``_cobyla_py`` computes it as: the violation post-check first, then normal
    termination.  ``cstrv > catol`` fails regardless of why the solver stopped;
    otherwise only SMALL_TR_RADIUS (0) and FTARGET_ACHIEVED (1) are successes.
    """
    if cstrv > catol:
        return False
    return info == 0 or info == 1


@njit
def _cobyla_message(status, cstrv, catol):
    """scipy 1.18's COBYLA message, verbatim.

    The violation post-check wins over the solver's own text, matching the
    order in ``_cobyla_py``.  The remaining strings are pyprima's
    ``get_info_string('COBYLA', info)``.
    """
    if cstrv > catol:
        return ("Did not converge to a solution satisfying the constraints. "
                "See `maxcv` for the magnitude of the violation.")
    if status == 0:
        return ("Return from COBYLA because the trust region radius reaches "
                "its lower bound.")
    if status == 1:
        return ("Return from COBYLA because the target function value is "
                "achieved.")
    if status == 2:
        return ("Return from COBYLA because a trust region step has failed to "
                "reduce the quadratic model.")
    if status == 3:
        return ("Return from COBYLA because the objective function has been "
                "evaluated MAXFUN times.")
    if status == 20:
        return ("Return from COBYLA because the maximal number of trust "
                "region iterations has been reached.")
    if status == -1:
        return "Return from COBYLA because NaN or Inf occurs in x."
    if status == -2:
        return ("Return from COBYLA because the objective function returns "
                "NaN/+Inf.")
    if status == -3:
        return "Return from COBYLA because NaN or Inf occurs in the models."
    if status == 6:
        return ("Return from COBYLA because there is no space between the "
                "lower and upper bounds of variable.")
    if status == 7:
        return ("Return from COBYLA because rounding errors are becoming "
                "damaging.")
    if status == 8:
        return ("Return from COBYLA because one of the linear constraints has "
                "a zero gradient")
    if status == 30:
        return ("Return from COBYLA because the callback function requested "
                "termination")
    return "Unknown exit status."


_DISP_MSG = ("disp argument to minimize must be 0, 1, 2, or 3,"
             "                          received ")


@njit
def _cobyla_disp(show, status, cstrv, catol):
    """scipy's failure line, ``_cobyla_py.py:178-180``.

    ``if disp and not sol['success']: print(f"COBYLA failed to find a
    solution: {sol.message}")``. The solver's own ``iprint`` output is a
    separate stream and is not reproduced.
    """
    if show and not _cobyla_success(status, cstrv, catol):
        print("COBYLA failed to find a solution: "
              + _cobyla_message(status, cstrv, catol))


@njit
def _cobyla_run(funcptr, m_nlcon, x0, Aineq, bineq, Aeq, beq, lower, upper,
                args, rhobeg, rhoend, maxfun, ctol):
    """The raw PRIMA call.  Returns (x, f, cstrv, nf, info)."""
    n = x0.size
    m = m_nlcon
    mi = bineq.size
    me = beq.size
    if mi > 0 and (Aineq.ndim != 2 or Aineq.shape[0] != mi
                   or Aineq.shape[1] != n):
        raise ValueError("cobyla: Aineq must be (len(bineq), len(x0))")
    if me > 0 and (Aeq.ndim != 2 or Aeq.shape[0] != me
                   or Aeq.shape[1] != n):
        raise ValueError("cobyla: Aeq must be (len(beq), len(x0))")
    xl, xu = _bounds(n, lower, upper)
    m_ = np.array(m, np.int32)
    n_ = np.array(n, np.int32)
    mi_ = np.array(mi, np.int32)
    me_ = np.array(me, np.int32)
    x = np.ascontiguousarray(np.asarray(x0)).astype(np.float64).copy()
    ait = np.ascontiguousarray(Aineq.astype(np.float64).T).copy()
    aet = np.ascontiguousarray(Aeq.astype(np.float64).T).copy()
    bi = np.ascontiguousarray(np.asarray(bineq, np.float64))
    be = np.ascontiguousarray(np.asarray(beq, np.float64))
    f = np.zeros(1, np.float64)
    cstrv = np.zeros(1, np.float64)
    nlconstr = np.zeros(max(m, 1), np.float64)
    rb = np.array(rhobeg, np.float64)
    re = np.array(rhoend, np.float64)
    ct = np.array(ctol, np.float64)
    mf = np.array(maxfun if maxfun > 0 else 500 * n, np.int32)
    nf = np.zeros(1, np.int32)
    info = np.zeros(1, np.int32)
    na = np.array(args.size, np.int32)

    _cobyla(funcptr, m_.ctypes.data, n_.ctypes.data, x.ctypes.data,
            mi_.ctypes.data, ait.ctypes.data, bi.ctypes.data,
            me_.ctypes.data, aet.ctypes.data, be.ctypes.data,
            xl.ctypes.data, xu.ctypes.data, f.ctypes.data,
            cstrv.ctypes.data, nlconstr.ctypes.data, rb.ctypes.data,
            re.ctypes.data, ct.ctypes.data, mf.ctypes.data, nf.ctypes.data,
            info.ctypes.data, args.ctypes.data, na.ctypes.data)
    return x, f[0], cstrv[0], nf[0], info[0]


@njit
def _cobyla_core(funcptr, m_nlcon, x0, Aineq, bineq, Aeq, beq, lower, upper,
                 args, rhobeg, rhoend, maxfun, catol):
    """Probe, solve, classify.  Returns (x, f, cstrv, nf, status).

    `nf` carries the write probe below as well as PRIMA's own count, so it is
    every objective evaluation this routine makes. scipy's `nfev` is the same
    quantity: measured on ``min x0**2 + x1**2`` subject to ``x0 >= 1`` from
    ``(3, 3)``, ``scipy`` reported 39 against 39 calls actually made.
    """
    n = x0.size
    if not _wrote_objcon(funcptr, np.ascontiguousarray(
            np.asarray(x0)).astype(np.float64), max(m_nlcon, 1), args):
        raise ValueError(
            "the objective callback never wrote f or the constraint vector. "
            "The argument order is (x, f, constr, args)")
    x, f, cstrv, nf, info = _cobyla_run(
        funcptr, m_nlcon, x0, Aineq, bineq, Aeq, beq, lower, upper, args,
        rhobeg, rhoend, maxfun, catol)
    return x, f, cstrv, nf + 1, _cobyla_status(info, cstrv, catol)


def fmin_cobyla(func, x0, cons, args=(), consargs=None, rhobeg=1.0,
                rhoend=1e-4, maxfun=1000, disp=None, catol=2e-4,
                *, callback=None, m_nlcon=-1, Aineq=None, bineq=None,
                Aeq=None, beq=None, lower=None, upper=None,
                full_output=False):
    """Minimize a function subject to nonlinear inequality constraints.

    COBYLA builds a linear model of the objective and of every constraint from
    values alone, and minimizes it inside a trust region. No derivatives are
    required.

    Callable from Python and from inside ``@njit``.

    Parameters
    ----------
    func : callable
        A plain ``@njit`` ``f(x, *args) -> value``.
    x0 : array_like
        Initial guess. Any rank is flattened.
    cons : callable or sequence of callables
        Constraint functions, feasible where the result is ``>= 0``. Either
        one plain ``@njit`` ``c(x, *consargs)`` returning all of them at once,
        or a list or tuple of such functions returning a scalar or an array
        each. A tuple is the spelling that types inside ``@njit``.
    args : tuple or ndarray, optional
        Extra parameters for `func`, packed into one flat float64 buffer.
    consargs : tuple, ndarray or None, optional
        Extra parameters for `cons`. ``None`` (default) reuses `args`.
    rhobeg : float, optional
        Initial trust-region radius. Default 1.0.
    rhoend : float, optional
        Final trust-region radius. Default 1e-4.
    maxfun : int, optional
        Maximum objective evaluations. Default 1000.
    disp : int or None, optional
        ``0``, ``1``, ``2``, ``3`` or ``None``; anything else is a
        ``ValueError``. Truthy prints ``COBYLA failed to find a solution:``
        and the message when the solve fails. The solver's own per-iteration
        stream is not reproduced.
    catol : float or None, optional
        Constraint-violation tolerance, reaching the solver as PRIMA's
        ``ctol``. It governs which point the solver returns, and it decides
        ``success``: a solve leaving ``maxcv > catol`` reports
        ``success=False``. Raising it trades feasibility for objective value.
        Default 2e-4. ``None`` resolves to ``sqrt(eps)``, 1.49e-08, which is
        what ``minimize(method='COBYLA')`` uses. A negative or ``nan`` value
        raises a ``UserWarning`` and the solver runs on ``sqrt(eps)``, while
        ``success`` and `message` still read the value as given, so a ``nan``
        makes the violation test pass whatever ``maxcv`` is.
    callback : None, optional
        Accepted only as ``None``. See Notes. Keyword-only, as is every
        parameter after it, so ten is the largest number of positional
        arguments.
    m_nlcon : int, optional
        Number of nonlinear constraints. ``-1`` (default) reads it from
        `cons` by evaluating the constraints once at `x0`.
    Aineq, bineq, Aeq, beq, lower, upper : ndarray or None, optional
        Linear constraints ``Aineq @ x <= bineq`` and ``Aeq @ x == beq``, and
        simple bounds, all taken by PRIMA natively. ``None`` (default) for
        each leaves it unset.
    full_output : bool, optional
        ``False`` (default) returns `x` alone. ``True`` returns a
        ``CobylaResult``. Compile-time constant inside ``@njit``: it selects
        the return type.

    Returns
    -------
    x : ndarray
        The minimizer.
    result : CobylaResult
        Only with ``full_output=True``. Namedtuple with fields ``x``,
        ``fun``, ``maxcv``, ``message``, ``nfev``, ``status`` and ``success``.
        ``status`` is PRIMA's own exit code: ``0`` the trust-region radius
        reached its lower bound, ``1`` the target value was achieved, ``2`` a
        trust-region step failed to reduce the model, ``3`` `maxfun` was
        exhausted, ``7`` damaging rounding errors, and negative codes for
        NaN or infinity in `x`, in the objective, or in the models.
        ``success`` is ``True`` when ``maxcv <= catol`` and ``status`` is
        ``0`` or ``1``. ``nfev`` counts the one probe evaluation this routine
        makes before the solve as well as the solver's own calls.

    Raises
    ------
    TypeError
        If `cons` is omitted, or is neither callable nor a sequence of
        callables.
    ValueError
        If `func` is not a plain ``@njit`` function; if `disp` is outside
        ``{0, 1, 2, 3, None}``; or if `callback` is not ``None``. The compiled
        entry point raises the same text as a ``TypingError`` wherever the
        argument's type settles the refusal.

    See Also
    --------
    scipy.optimize.fmin_cobyla : The scipy routine this mirrors.
    scijit.optimize.fmin_slsqp : Constrained, gradient-based.
    scijit.optimize.minimize : Reaches PRIMA's unconstrained and bounded
        solvers by name.

    Notes
    -----
    The returned point is not scipy's. scipy 1.16 replaced Powell's COBYLA
    with pyprima, a Python translation of the same algorithm, while this wraps
    PRIMA itself. Measured on ``min x0**2 + x1**2`` subject to ``x0 >= 1``
    from ``(3, 3)``: ``|dx|`` 1.18e-04 and ``|df|`` 1.38e-08, with PRIMA the
    nearer of the two to the true optimum, ``f - 1`` being 6.8e-14 against
    scipy's 1.4e-08.

    `catol` moves the answer, so it is worth setting deliberately. On the
    problem above, the 2e-4 default returns ``fun = 0.999900005`` at
    ``maxcv = 5.0e-05``; ``catol=None`` returns ``1.000000002`` at
    ``maxcv = 0``; ``catol=1e-2`` returns ``0.999003146`` at
    ``maxcv = 5.0e-04``; ``catol=1`` returns ``0.910296579`` at
    ``maxcv = 5.0e-02``.

    `args` and `consargs` each pack into one flat float64 buffer, so they
    carry numbers rather than arbitrary objects.

    scipy calls `callback` once per iteration, with either ``callback(xk)`` or
    ``callback(intermediate_result)``. Neither is served here: PRIMA's
    ``cobyla`` takes a ``callback_fcn`` and ``src/prima/wrappers.f90`` passes
    none, so the Fortran has no slot for one to arrive through.

    `disp` prints the failure line and not the solver's own per-iteration
    stream, which PRIMA emits from Fortran.

    `Aineq`, `bineq`, `Aeq`, `beq`, `lower` and `upper` have no scipy
    counterpart; scipy's COBYLA can express linear constraints and bounds only
    by folding them into `cons`. A scipy-shaped call leaves all six unset and
    behaves as scipy does. `full_output` has no scipy counterpart either:
    scipy returns `x` alone and discards `fun`, `maxcv`, `nfev`, `status`,
    `message` and `success`, which is what the ``False`` default reproduces.

    Safe to call from a ``numba.prange`` loop. PRIMA reaches the callback
    through a Fortran module variable, which carries ``!$omp threadprivate``
    and so resolves to one slot per thread.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cobyla.html

    Examples
    --------
    Minimize ``x0**2 + x1**2`` subject to ``x0 >= 1``:

    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import fmin_cobyla
    >>> @njit
    ... def obj(x):
    ...     return x[0] ** 2 + x[1] ** 2
    >>> @njit
    ... def cons(x):
    ...     return np.array([x[0] - 1.0])
    >>> @njit
    ... def run():
    ...     return fmin_cobyla(obj, np.array([3.0, 3.0]), cons)
    >>> np.round(run(), 4)
    array([1., 0.])

    With `full_output`, which must be a literal inside ``@njit``:

    >>> @njit
    ... def run_full():
    ...     return fmin_cobyla(obj, np.array([3.0, 3.0]), cons,
    ...                        full_output=True)
    >>> res = run_full()
    >>> round(res.fun, 9), res.status, res.success
    (0.999900005, 0, True)
    >>> round(res.maxcv, 9), res.nfev
    (5e-05, 37)

    A tuple of one-constraint functions, which is scipy's own spelling:

    >>> @njit
    ... def c_lo(x):
    ...     return x[0] - 1.0
    >>> @njit
    ... def c_hi(x):
    ...     return 5.0 - x[1]
    >>> @njit
    ... def run_pair():
    ...     return fmin_cobyla(obj, np.array([3.0, 3.0]), (c_lo, c_hi))
    >>> np.round(run_pair(), 4)
    array([1., 0.])
    """
    # scipy's validation order: `cons` in `fmin_cobyla` itself, then `disp`
    # and the objective inside `_minimize_cobyla`, then `callback`.
    cons = _cons_seq(cons)
    if disp is not None and disp != 0 and disp != 1 and disp != 2 \
            and disp != 3:
        raise ValueError(_DISP_MSG + str(disp))
    x = np.ascontiguousarray(np.atleast_1d(
        np.asarray(x0, dtype=np.float64))).ravel().copy()
    n = x.size
    at = _as_args_tuple(args)
    cat = at if consargs is None else _as_args_tuple(consargs)
    kinds, ckinds = _arg_kinds(at), _arg_kinds(cat)
    a, ca = _pack_args(at), _pack_args(cat)
    pyf = _front_pyfunc(func, len(kinds), 'fmin_cobyla',
                        _ADDRESS_MSG_COBYLA)
    if callback is not None:
        raise ValueError(_CALLBACK_MSG)
    pycs = _cons_pyfuncs(cons)
    if pycs is None or len(pycs) == 0:
        raise ValueError(_CONS_REQUIRED)
    for pc in pycs:
        _check_arity(pc, len(ckinds), 'fmin_cobyla')
    cf = _fuse_cons(pycs, len(ckinds))
    mm = int(m_nlcon) if m_nlcon >= 0 else int(cf(x, *cat).size)
    fp = _adapter_objcon(pyf, cf, kinds, ckinds).address
    ab = _prepend_nm_args(a, ca, n, mm)
    z2 = np.zeros((0, 0))
    z1 = np.zeros(0)
    # Two values, as in scipy.  `ct` is what the solver runs on and carries
    # the "nonnegative number" coercion pyprima applies in `preproc`; `cchk`
    # is the raw `catol` the post-check compares `maxcv` against.  A `nan`
    # makes every comparison False, which is what scipy reports.
    if catol is None:
        ct = cchk = _CTOL_DFT
    else:
        cchk = catol
        if np.isnan(catol) or catol < 0.0:
            _emit_ctol_warning()
            ct = _CTOL_DFT
        else:
            ct = catol
    xo, f, cstrv, nf, st = _cobyla_core(
        fp, mm, x,
        z2 if Aineq is None else np.asarray(Aineq, np.float64),
        z1 if bineq is None else np.asarray(bineq, np.float64),
        z2 if Aeq is None else np.asarray(Aeq, np.float64),
        z1 if beq is None else np.asarray(beq, np.float64),
        z1 if lower is None else np.asarray(lower, np.float64),
        z1 if upper is None else np.asarray(upper, np.float64),
        ab, rhobeg, rhoend, maxfun, ct)
    _cobyla_disp(disp is not None and disp != 0, st, cstrv, cchk)
    if not full_output:
        return xo
    # bool(): st is an int32 here and numpy's bool_ fails an `is True`
    # identity test, where scipy returns a python bool
    return CobylaResult(xo, f, cstrv, _cobyla_message(st, cstrv, cchk),
                        int(nf), int(st),
                        bool(_cobyla_success(st, cstrv, cchk)))


@overload(fmin_cobyla, prefer_literal=True)
def _fmin_cobyla_ovl(func, x0, cons, args=(), consargs=None, rhobeg=1.0,
                     rhoend=1e-4, maxfun=1000, disp=None, catol=2e-4,
                     *, callback=None, m_nlcon=-1, Aineq=None, bineq=None,
                     Aeq=None, beq=None, lower=None, upper=None,
                     full_output=False):
    """@njit implementation of `fmin_cobyla`, resolved at compile time.

    Three things have to be settled before the body is typed and none of
    them can be decided at runtime: ``full_output`` picks the return
    type, an objective/constraint pair of ``@njit`` functions has to be
    fused into one ``prima_con_sig`` adapter address, and each optional
    matrix argument has to be known present or absent so the body can
    skip it. Returning ``None`` declines the call, which numba reports as
    a TypingError naming the argument that could not be served.
    """
    fo = _lit_bool(full_output)
    if fo is None:
        return None                     # runtime flag -> TypingError
    # scipy's validation order: `cons` first, then the objective, then
    # `callback`.  `disp` sits between the first two there and is a run-time
    # value here, so it is checked in the body; see the module's Notes.
    pycs = _cons_pyfuncs_ty(cons)
    if pycs is None or len(pycs) == 0:
        raise TypingError(_cons_msg_ty(cons))
    nca = _is_none(consargs)
    kinds = _arg_kinds_ty(_args_types(args))
    ckinds = kinds if nca else _arg_kinds_ty(_args_types(consargs))
    pyf = _front_pyfunc_ty(func, len(kinds), 'fmin_cobyla',
                           _ADDRESS_MSG_COBYLA)
    if not _is_none(callback):
        raise TypingError(_CALLBACK_MSG)
    for pc in pycs:
        try:
            _check_arity(pc, len(ckinds), 'fmin_cobyla')
        except TypeError as exc:
            raise TypingError(str(exc))
    cfused = _fuse_cons(pycs, len(ckinds))
    addr = _adapter_objcon(pyf, cfused, kinds, ckinds).address
    nAi, nbi = _is_none(Aineq), _is_none(bineq)
    nAe, nbe = _is_none(Aeq), _is_none(beq)
    nlo, nup = _is_none(lower), _is_none(upper)
    nct = _is_none(catol)
    ndisp = _is_none(disp)
    # numba's `str` carries an integer but renders a float as
    # '<object type:float64>', so the value is appended only where it reads.
    disp_int = isinstance(disp, (types.Integer, types.Omitted))

    def impl(func, x0, cons, args=(), consargs=None, rhobeg=1.0,
             rhoend=1e-4, maxfun=1000, disp=None, catol=2e-4, callback=None,
             m_nlcon=-1, Aineq=None, bineq=None, Aeq=None, beq=None,
             lower=None, upper=None, full_output=False):
        if ndisp:
            show = False
        else:
            if disp != 0 and disp != 1 and disp != 2 and disp != 3:
                if disp_int:
                    raise ValueError(_DISP_MSG + str(disp))
                raise ValueError(_DISP_MSG)
            show = disp != 0
        x = np.ascontiguousarray(np.atleast_1d(
            np.asarray(x0))).ravel().astype(np.float64)
        n = x.size
        a = _pack_args(args)
        if nca:
            ca = a
        else:
            ca = _pack_args(consargs)
        if m_nlcon >= 0:
            mm = np.int64(m_nlcon)
        elif nca:
            mm = np.int64(_call_cb(cfused, x, args).size)
        else:
            mm = np.int64(_call_cb(cfused, x, consargs).size)
        fp, ab = addr, _prepend_nm_args(a, ca, n, mm)
        z2 = np.zeros((0, 0))
        z1 = np.zeros(0)
        # Cast exactly as the python body does: an integer-dtype matrix must
        # reach `_cobyla_core` as float64 from both entry points.
        Ai = z2 if nAi else np.asarray(Aineq).astype(np.float64)
        bi = z1 if nbi else np.asarray(bineq).astype(np.float64)
        Ae = z2 if nAe else np.asarray(Aeq).astype(np.float64)
        be = z1 if nbe else np.asarray(beq).astype(np.float64)
        lo = z1 if nlo else np.asarray(lower).astype(np.float64)
        up = z1 if nup else np.asarray(upper).astype(np.float64)
        # Two values, as in scipy.  `ct` runs the solver and carries
        # pyprima's "nonnegative number" coercion; `cchk` is the raw `catol`
        # the post-check compares `maxcv` against.
        if nct:
            ct = _CTOL_DFT
            cchk = _CTOL_DFT
        else:
            cchk = catol
            if np.isnan(catol) or catol < 0.0:
                _warn_ctol()
                ct = _CTOL_DFT
            else:
                ct = catol
        xo, f, cstrv, nf, st = _cobyla_core(
            fp, mm, x, Ai, bi, Ae, be, lo, up, ab, rhobeg,
            rhoend, maxfun, ct)
        _cobyla_disp(show, st, cstrv, cchk)
        if fo:
            return CobylaResult(xo, f, cstrv,
                                _cobyla_message(st, cstrv, cchk), nf, st,
                                _cobyla_success(st, cstrv, cchk))
        return xo
    return impl
