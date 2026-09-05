"""The LSODA layer: the ODEPACK binding, and the machinery over it.

Backed by liblsoda (Nicholaswogan's thread-safe modern-Fortran ODEPACK),
built from ``src/odepack``. The vendored pack provides two solvers, DLSODA
and DLSODAR: Adams/BDF with automatic stiff/nonstiff switching, and the same
with root finding.

Two front ends sit on this module. :func:`~scijit.integrate.odeint` is
scipy's ``odeint`` and lives in ``_odeint.py``; ``solve_ivp(method='LSODA')``
lives in ``_solve_ivp.py``. Everything they share is here: the ctypes
binding, the three callback ABIs, the ``@cfunc`` adapters built from a plain
``@njit`` callback, and the solver drivers.

The signature objects describe the C ABI LSODA calls back through. All three
are private and none is part of any call: both front ends take a plain
``@njit`` function, build the ``@cfunc`` themselves, and refuse an address in
that slot.

    _lsoda_sig        = void(float64 t, float64* y, float64* ydot, float64* args)
    _lsoda_yfirst_sig = void(float64* y, float64 t, float64* ydot, float64* args)
    _lsoda_jac_sig    = void(t, y*, pd*, neq, ml, mu, nrowpd, args*)

LSODA reaches the callback through a Fortran module variable holding its
address. That slot is ``!$omp threadprivate``, so each OS thread gets its own
copy and independent integrations run correctly under prange. The solver's
own state is allocated per call, so a nested integration is fine as well.

Not a public API.
"""
import inspect

import numpy as np
from numba import carray, cfunc, njit, types
from numba.extending import overload

from .._lib._load import load


_lsoda_sig = types.void(types.double,                       # t     (value)
                        types.CPointer(types.double),       # y     (in)
                        types.CPointer(types.double),       # ydot  (out)
                        types.CPointer(types.double))       # args  (in)
"""numba signature of an LSODA right-hand side --
``void(float64, float64*, float64*, float64*)``.  Private.

``t`` arrives by value, ``y``/``ydot``/``args`` as pointers.  The callback
returns nothing: it **writes** the ``neq`` derivatives into ``ydot``.

This is the ``tfirst=True`` argument order.  ``odeint``'s default is
``tfirst=False``, whose order is ``(y, t, ydot, args)`` and whose signature
object is :data:`_lsoda_yfirst_sig`.

No public routine in ``scijit.integrate`` accepts a ``@cfunc`` built against
this signature, and a caller cannot pass one.  ``odeint`` and
``solve_ivp(method='LSODA')`` take a plain ``@njit`` right-hand side and build
the ``@cfunc`` around it at typing time, which is what the adapter builders in
``_odeint`` and ``_solve_ivp`` use this signature for.
"""


_lib, _sig = load(__file__, "liblsoda")


# Argument count recounted against wrappers_scipy.f90, one continuation line
# at a time rather than by adjusting the previous number:
#   cfcn tfirst neq y0 t ntime yout                   7   (7)
#   itol rtol atol usetcrit tcrit ntcrit              6  (13)
#   h0 hmax hmin ixpr mxstep mxhnil                   6  (19)
#   mxordn mxords istate_out                          3  (22)
#   hu tcur tolsf tsw nst nfe nje                     7  (29)
#   nqu mused imxer lenrw leniw                       5  (34)
#   args nargs jt ml mu cjac                          6  (40)
_odeint_sp = _sig(_lib.odeint_scipy_wrapper, 40)


# Argument count recounted against wrappers.f90, one continuation line at a
# time rather than by adjusting the previous number:
#   cfcn neq u0 t0 tf                                 5   (5)
#   itol rtol atol h0 hmax hmin                       6  (11)
#   mxstep maxsteps maxord nsteps_out                 4  (15)
#   t_hist h_hist hu_hist nq_hist nqn_hist yh_hist    6  (21)
#   istate_out nfev_out njev_out                      3  (24)
#   args nargs jt ml mu cjac                          6  (30)
#   istate_in rwork lrw iwork liw                     5  (35)
#   cstate_r cstate_i y_out t_out                     4  (39)
_lsoda_dense = _sig(_lib.lsoda_dense_wrapper, 39)


#: Length of the flattened LSODA common blocks, which travel between the
#: segments of one history run beside RWORK and IWORK.  From
#: ``src/odepack/01_odepack_common.f90``: DLS001 ``reals(218)``/``ints(37)``,
#: DLSA01 ``reals(22)``/``ints(9)``, DLSR01 ``reals(5)``/``ints(9)``.
#: `lsoda_dense_wrapper` declares the same two lengths from its own
#: parameters, and ``tests/integrate/test_solve_ivp_scipy.py`` reads both files and
#: asserts the four numbers agree.
_NCS_R = 218 + 22 + 5


_NCS_I = 37 + 9 + 9


#: LSODA's largest Adams order, so the Nordsieck array is (neq, 13) at most.
#: MXORDN is set to 12 in the wrapper and MXORDS to 5, so 12 bounds both.
_MAXORD = 12


#: numba signature of a y-first right-hand side --
#: ``void(float64*, float64, float64*, float64*)``, i.e. ``f(y, t, ydot,
#: args)``.  This is the shape scipy's default ``func(y, t, *args)`` has, and
#: the one :func:`odeint` expects unless ``tfirst=True``.  The callback writes
#: the ``neq`` derivatives into ``ydot``::
#:
#:     @cfunc(_lsoda_yfirst_sig)
#:     def rhs(y, t, dy, args):
#:         dy[0] = y[1]
#:         dy[1] = -args[0] * y[0]
#:
#: ``tfirst=True`` uses ``_lsoda_sig`` instead, whose parameter order is
#: ``(t, y, ydot, args)``.  Both are private; ``odeint`` builds the ``@cfunc``
#: internally from a plain ``@njit`` right-hand side and accepts no address.
_lsoda_yfirst_sig = types.void(types.CPointer(types.double),   # y    (in)
                              types.double,                   # t    (value)
                              types.CPointer(types.double),   # ydot (out)
                              types.CPointer(types.double))   # args (in)


# --------------------------------------------------------------------------
# Hiding the @cfunc.  ``func`` may be a raw cfunc address (int) OR a plain
# @njit rhs returning the derivatives: f(y, t) -> ydot (scipy's order,
# tfirst=False) or f(t, y) -> ydot (tfirst=True).  The adapter is built here,
# once, and cached; the cache OWNS the cfunc, so dropping the reference would
# dangle the baked-in address.
#
# ``neq`` is unknown when the adapter is built, so the glue PREPENDS it to the
# args buffer -- adapter path only; a raw address gets the user's args
# untouched.  This also removes the Windows ABI exposure for adapter users: a
# raw t-first address under a y-first default relies on the two ABIs
# coinciding (true on x86-64 System V, false on Windows x64), whereas an
# adapter states the argument order explicitly in its own body.
# --------------------------------------------------------------------------
_ADAPTERS = {}


def _py_func_of(fn, what="the right-hand side"):
    """The Python function behind a plain ``@njit`` dispatcher.

    A ``@cfunc`` OBJECT is the common mistake in this slot: it is neither an
    address nor a dispatcher, and reaching for ``.py_func`` on one raises an
    ``AttributeError`` naming this file rather than the caller's error.
    """
    py = getattr(fn, 'py_func', None)
    if py is None:
        raise ValueError(
            "%s must be a plain @njit function. A @cfunc object or its "
            ".address is not accepted: pass the @njit function itself and "
            "the callback is built internally." % what)
    return py


def _rhs_arity(py):
    """Positional parameter count of a plain ``@njit`` right-hand side.

    Two means ``f(y, t)``, three means ``f(y, t, args)``.  Anything else is
    refused here rather than surfacing as a numba binding error from inside
    the adapter, where the traceback points at this file instead of the
    caller's function.
    """
    n = 0
    for p in inspect.signature(py).parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
        elif p.kind == p.VAR_POSITIONAL:
            raise ValueError(
                "a plain @njit right-hand side cannot take *args; write "
                "f(y, t) or f(y, t, args), where args is one flat float64 "
                "array")
    if n not in (2, 3):
        raise ValueError(
            "a plain @njit right-hand side takes 2 or 3 arguments, "
            "f(y, t) or f(y, t, args); this one takes %d" % n)
    return n


def _adapter_rhs(py, tfirst, argbase=1):
    """cfunc around a plain @njit ``f(y, t)`` (or ``f(t, y)``) -> ydot.

    A three-argument right-hand side also receives ``args``, matching
    scipy's ``func(y, t, *args)``.  ``argbase`` is where the args buffer
    records its own length: the buffer is ``[neq, nargs, *args]`` for
    ``odeint`` and ``solve_ivp`` and ``[neq, ng, nargs, *args]`` for
    the event path, so slot 0 is ``neq`` either way and the count sits at
    ``argbase``.
    """
    nparam = _rhs_arity(py)
    key = (py, bool(tfirst), int(argbase), nparam)
    hit = _ADAPTERS.get(key)
    if hit is not None:
        return hit
    inner = njit(py)
    base = int(argbase)
    # A wrong-length return writes NOTHING rather than reading past the end
    # of `out`. numba has no bounds checking, so the loop would otherwise
    # copy whatever follows the array into ydot and the solver would
    # integrate it. Writing nothing leaves the caller's sentinel intact,
    # which is what the pre-flight probe looks for, so a short rhs reaches
    # a ValueError instead of a plausible wrong answer.
    if nparam == 3:
        if tfirst:
            @cfunc(_lsoda_sig)
            def adapter(t, y_ptr, dy_ptr, a_ptr):
                neq = int(a_ptr[0])
                na = int(a_ptr[base])
                out = inner(t, carray(y_ptr, neq),
                            carray(a_ptr, base + 1 + na)[base + 1:])
                if out.size == neq:
                    for i in range(neq):
                        dy_ptr[i] = out[i]
        else:
            @cfunc(_lsoda_yfirst_sig)
            def adapter(y_ptr, t, dy_ptr, a_ptr):
                neq = int(a_ptr[0])
                na = int(a_ptr[base])
                out = inner(carray(y_ptr, neq), t,
                            carray(a_ptr, base + 1 + na)[base + 1:])
                if out.size == neq:
                    for i in range(neq):
                        dy_ptr[i] = out[i]
    elif tfirst:
        @cfunc(_lsoda_sig)
        def adapter(t, y_ptr, dy_ptr, a_ptr):
            neq = int(a_ptr[0])
            out = inner(t, carray(y_ptr, neq))
            if out.size == neq:
                for i in range(neq):
                    dy_ptr[i] = out[i]
    else:
        @cfunc(_lsoda_yfirst_sig)
        def adapter(y_ptr, t, dy_ptr, a_ptr):
            neq = int(a_ptr[0])
            out = inner(carray(y_ptr, neq), t)
            if out.size == neq:
                for i in range(neq):
                    dy_ptr[i] = out[i]
    _ADAPTERS[key] = adapter
    return adapter


#: numba signature of the Fortran ``numba_jac`` interface in
#: ``src/odepack/wrappers_scipy.f90``.  Internal: a Jacobian is supplied as a
#: plain ``@njit`` function and the ``@cfunc`` around it is built here, so
#: nothing published takes an address.
#:
#: ``void(double t, double* y, double* pd, int32 neq, int32 ml, int32 mu,
#: int32 nrowpd, double* args)``.  ``neq``, ``ml``, ``mu`` and ``nrowpd`` are
#: LSODA's and travel by value, because none of them is known when the
#: adapter is built.
_lsoda_jac_sig = types.void(types.double,                     # t     (value)
                           types.CPointer(types.double),      # y     (in)
                           types.CPointer(types.double),      # pd    (out)
                           types.int32,                       # neq   (value)
                           types.int32,                       # ml    (value)
                           types.int32,                       # mu    (value)
                           types.int32,                       # nrowpd(value)
                           types.CPointer(types.double))      # args  (in)


_JAC_ADAPTERS = {}


_JAC_ARITY_MSG = (
    "a plain @njit jacobian takes 2 or 3 arguments, j(t, y) or "
    "j(t, y, args), where args is one flat float64 array; this one takes %d")


_JAC_ARGS_MSG = (
    "a three-argument jacobian j(t, y, args) needs the right-hand side to be "
    "a plain @njit function too. Alongside a raw @cfunc address for `fun` the "
    "parameter buffer carries no length, so the jacobian cannot find them: "
    "write j(t, y) and close over the parameters, or pass `fun` as a plain "
    "@njit function")


_JAC_SHAPE_FULL_MSG = (
    "the jacobian must return an (n, n) array with jac[i, j] = d f_i / d y_j")


_JAC_SHAPE_BAND_MSG = (
    "with lband/uband the jacobian must return an (lband + uband + 1, n) "
    "array with jac[uband + i - j, j] = d f_i / d y_j, the layout "
    "scipy.linalg.solve_banded takes")


def _jac_arity(py, what="jacobian"):
    """Positional parameter count of a plain ``@njit`` Jacobian.

    Two means ``j(t, y)``, three means ``j(t, y, args)``.
    """
    n = 0
    for prm in inspect.signature(py).parameters.values():
        if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD):
            n += 1
        elif prm.kind == prm.VAR_POSITIONAL:
            raise ValueError(
                "a plain @njit %s cannot take *args; write j(t, y) or "
                "j(t, y, args), where args is one flat float64 array" % what)
    if n not in (2, 3):
        raise ValueError(_JAC_ARITY_MSG % n)
    return n


def _adapter_jac(py, banded, tfirst=True, padded=True, coldiv=False,
                 argbase=1):
    """``@cfunc`` around a plain ``@njit`` Jacobian, built at TYPING time.

    The address is frozen into the compiled body, so a Jacobian is a plain
    ``@njit`` function everywhere a caller can see.  The cache OWNS the
    ``@cfunc``; dropping the reference would dangle the baked-in address.

    Two shapes of user function, selected by ``banded``, and both write into
    the same LSODA buffer.  ``pd`` is Fortran-ordered ``(nrowpd, neq)``, so
    element ``(r, c)`` sits at ``pd[r + c*nrowpd]``.

    **Full**, ``jt = 1``.  ``nrowpd == neq`` and ODEPACK wants
    ``PD(i, j) = d f(i)/d y(j)`` (``odepack.f:596-598``), so element
    ``(i, j)`` of the user's row-major ``(n, n)`` array lands at
    ``pd[i + j*nrowpd]``.

    **Banded**, ``jt = 4``.  ``nrowpd == 2*ml + mu + 1`` and ODEPACK wants
    ``PD(i-j+MU+1, j)`` 1-based (``odepack.f:599-603``), while scipy asks the
    caller for ``jac_packed[uband + i - j, j]`` in an ``(ml + mu + 1, n)``
    array (``_ivp/lsoda.py:60-67``).  The row index is the same on both sides,
    so the copy is row for row, and the two triangular corners, which DLSODA
    overwrites (``odepack.f:604-606``), come along harmlessly.

    ``coldiv`` is ``odeint``'s ``col_deriv``: the user's array arrives
    transposed, so the full case reads ``j[j, i]`` and the banded case takes
    an ``(n, ml + mu + 1)`` array.  Measured in scipy: feeding the transpose
    under ``col_deriv=1`` reproduces the ``col_deriv=0`` answer digit for
    digit, both full and banded.

    A wrong-returned SHAPE writes NOTHING.  It cannot raise from here: an
    exception inside a ``@cfunc`` is printed as "Exception ignored in" and the
    function returns, measured, so the caller never sees it and LSODA would
    carry on regardless.  The shape is therefore checked once in the glue,
    before the run, by :func:`_check_jac_shape_full` and
    :func:`_check_jac_shape_band`, which raise a real ``ValueError``.  This branch is what remains if a
    Jacobian returns different shapes on different calls.
    """
    nparam = _jac_arity(py)
    if nparam == 3 and not padded:
        raise ValueError(_JAC_ARGS_MSG)
    key = (py, bool(banded), bool(tfirst), bool(padded), bool(coldiv),
           int(argbase), nparam)
    hit = _JAC_ADAPTERS.get(key)
    if hit is not None:
        return hit
    inner = njit(py)
    base = int(argbase)
    wants3 = nparam == 3
    tf = bool(tfirst)
    cd = bool(coldiv)

    if banded and cd:
        @cfunc(_lsoda_jac_sig)
        def adapter(t, y_ptr, pd_ptr, neq, ml, mu, nrowpd, a_ptr):
            n = int(neq)
            yv = carray(y_ptr, n)
            if wants3:
                na = int(a_ptr[base])
                if tf:
                    j = inner(t, yv, carray(a_ptr, base + 1 + na)[base + 1:])
                else:
                    j = inner(yv, t, carray(a_ptr, base + 1 + na)[base + 1:])
            elif tf:
                j = inner(t, yv)
            else:
                j = inner(yv, t)
            mband = int(ml) + int(mu) + 1
            if j.shape[0] == n and j.shape[1] == mband:
                nr = int(nrowpd)
                for c in range(n):
                    for r in range(mband):
                        pd_ptr[r + c * nr] = j[c, r]
    elif cd:
        @cfunc(_lsoda_jac_sig)
        def adapter(t, y_ptr, pd_ptr, neq, ml, mu, nrowpd, a_ptr):
            n = int(neq)
            yv = carray(y_ptr, n)
            if wants3:
                na = int(a_ptr[base])
                if tf:
                    j = inner(t, yv, carray(a_ptr, base + 1 + na)[base + 1:])
                else:
                    j = inner(yv, t, carray(a_ptr, base + 1 + na)[base + 1:])
            elif tf:
                j = inner(t, yv)
            else:
                j = inner(yv, t)
            if j.shape[0] == n and j.shape[1] == n:
                nr = int(nrowpd)
                for c in range(n):
                    for r in range(n):
                        pd_ptr[r + c * nr] = j[c, r]
    elif banded:
        @cfunc(_lsoda_jac_sig)
        def adapter(t, y_ptr, pd_ptr, neq, ml, mu, nrowpd, a_ptr):
            n = int(neq)
            yv = carray(y_ptr, n)
            if wants3:
                na = int(a_ptr[base])
                if tf:
                    j = inner(t, yv, carray(a_ptr, base + 1 + na)[base + 1:])
                else:
                    j = inner(yv, t, carray(a_ptr, base + 1 + na)[base + 1:])
            elif tf:
                j = inner(t, yv)
            else:
                j = inner(yv, t)
            mband = int(ml) + int(mu) + 1
            if j.shape[0] == mband and j.shape[1] == n:
                nr = int(nrowpd)
                for c in range(n):
                    for r in range(mband):
                        pd_ptr[r + c * nr] = j[r, c]
    else:
        @cfunc(_lsoda_jac_sig)
        def adapter(t, y_ptr, pd_ptr, neq, ml, mu, nrowpd, a_ptr):
            n = int(neq)
            yv = carray(y_ptr, n)
            if wants3:
                na = int(a_ptr[base])
                if tf:
                    j = inner(t, yv, carray(a_ptr, base + 1 + na)[base + 1:])
                else:
                    j = inner(yv, t, carray(a_ptr, base + 1 + na)[base + 1:])
            elif tf:
                j = inner(t, yv)
            else:
                j = inner(yv, t)
            if j.shape[0] == n and j.shape[1] == n:
                nr = int(nrowpd)
                for c in range(n):
                    for r in range(n):
                        pd_ptr[r + c * nr] = j[r, c]
    _JAC_ADAPTERS[key] = adapter
    return adapter


@njit
def _check_jac_shape_full(j, n):
    """Raise unless ``j`` is the ``(n, n)`` array a full Jacobian must be.

    One call to the Jacobian, before the run, which is what scipy pays to
    learn the same thing: it reports
    "Expected a Jacobian array with shape (80, 80), but got (3, 80)".
    The check cannot live inside the ``@cfunc`` adapter, because an exception
    raised there is printed and swallowed.
    """
    if j.shape[0] != n or j.shape[1] != n:
        raise ValueError(_JAC_SHAPE_FULL_MSG)


@njit
def _check_jac_shape_band(j, n, ml, mu):
    """Raise unless ``j`` is the ``(ml + mu + 1, n)`` packed array a banded
    Jacobian must be.  scipy reports "Expected a banded Jacobian array with
    shape (3, 80), but got (80, 80)" for the same mistake."""
    if j.shape[0] != ml + mu + 1 or j.shape[1] != n:
        raise ValueError(_JAC_SHAPE_BAND_MSG)


@njit
def _prepend_neq(args, neq):
    """args buffer the ADAPTER path expects: ``[neq, nargs, *args]``."""
    out = np.empty(args.size + 2, np.float64)
    out[0] = np.float64(neq)
    out[1] = np.float64(args.size)
    for i in range(args.size):
        out[i + 2] = args[i]
    return out


# --------------------------------------------------------------------------
# input coercion
# --------------------------------------------------------------------------
def _as_1d(a):
    """1-D contiguous float64 copy; a scalar becomes length 1.

    Rank >= 2 is NOT flattened -- scipy rejects it, and the caller checks
    the rank before getting here.
    """
    return np.atleast_1d(np.asarray(a, dtype=np.float64)).copy()


@overload(_as_1d)
def _as_1d_ovl(a):
    """Compiled body for :func:`_as_1d`, one per argument shape.

    A scalar, a tuple and an array each need different code to become a
    contiguous 1-D float64 buffer, and which one applies is known when the
    call compiles.
    """
    if isinstance(a, (types.Float, types.Integer)):
        def impl(a):
            out = np.empty(1, np.float64)
            out[0] = np.float64(a)
            return out
        return impl
    if isinstance(a, types.BaseTuple):
        n = len(a)

        def impl(a):
            out = np.empty(n, np.float64)
            for i in range(n):
                out[i] = np.float64(a[i])
            return out
        return impl

    def impl(a):
        # ascontiguousarray: a strided view's .ctypes.data points into the
        # base buffer and Fortran reads contiguously (package gotcha 0).
        return np.ascontiguousarray(np.asarray(a)).astype(np.float64)
    return impl


@njit
def _msg(istate):
    """scipy's ``_msgs`` dict, verbatim (``_odepack_py.py`` lines 21-31)."""
    if istate == 2:
        return "Integration successful."
    elif istate == 1:
        return "Nothing was done; the integration time was 0."
    elif istate == -1:
        return "Excess work done on this call (perhaps wrong Dfun type)."
    elif istate == -2:
        return "Excess accuracy requested (tolerances too small)."
    elif istate == -3:
        return "Illegal input detected (internal error)."
    elif istate == -4:
        return "Repeated error test failures (internal error)."
    elif istate == -5:
        return ("Repeated convergence failures (perhaps bad Jacobian or "
                "tolerances).")
    elif istate == -6:
        return "Error weight became zero during problem."
    elif istate == -7:
        return "Internal workspace insufficient to finish (internal error)."
    elif istate == -8:
        return "Run terminated (internal error)."
    return "An error occurred."


@njit
def _lsoda_dense_work(neq, jt, ml, mu):
    """The state one LSODA history run continues through.

    Returns ``(rwork, iwork, cstate_r, cstate_i)``, the four buffers
    :func:`_run_lsoda_dense` takes.  ODEPACK continues an integration from
    RWORK and IWORK left exactly as the previous call returned them, at the
    same addresses (``odepack.f:568``), and this edition keeps the rest of
    the solver state in a derived type, which travels in the two ``cstate``
    buffers.  One set of four serves every segment of one integration.

    The RWORK requirement is ``odepack.f:499-517``, the fixed-length case:
    ``MAX(LRN, LRS)`` with ``LRN = 20 + 16*NEQ`` and ``LRS = 22 + 9*NEQ +
    NEQ**2`` for ``jt`` 1 and 2, ``22 + 10*NEQ + (2*ML + MU)*NEQ`` for 4 and
    5.  IWORK is ``20 + NEQ`` (``odepack.f:542``).  LSODA switches between
    the nonstiff and stiff methods on its own, so both regimes are allocated
    for, not the current one.

    A buffer shorter than LSODA needs is memory corruption rather than an
    exception, so `lsoda_dense_wrapper` recomputes both lengths from its own
    copy of the same expression and refuses a short one with ``istate = -3``.
    ``tests/integrate/test_solve_ivp_scipy.py`` pins them equal by handing
    the wrapper one element less than this returns and reading that refusal.
    """
    if jt >= 4:
        lrw = 22 + neq * max(16, 10 + 2 * ml + mu)
    else:
        lrw = 22 + neq * max(16, neq + 9)
    return (np.zeros(lrw, np.float64),
            np.zeros(20 + neq, np.int32),
            np.zeros(_NCS_R, np.float64),
            np.zeros(_NCS_I, np.int32))


@njit
def _run_lsoda_dense(funcptr, y0, t0, tf, rtol, atol, mxstep, maxsteps,
                     args, jt, ml, mu, jacptr, itol, h0, hmax, hmin,
                     rwork, iwork, cstate_r, cstate_i, istate_in):
    """Drive LSODA one step at a time, keeping the Nordsieck history.

    The history is what makes a callable dense solution possible: ODEPACK's
    ``DINTDY`` interpolates only inside the step it is standing on, so a
    solution over the whole span has to be assembled from a snapshot per
    step.

    Returns ``(nsteps, t, h, yh, istate, nfev, njev, y_end, t_end)``.
    ``t[k]`` is the time step ``k`` reached, ``h[k]`` the step size ``yh[k]``
    is scaled for, and ``yh[k]`` is ``(n, maxord + 1)`` with columns
    ``h^j/j! * y^(j)``, so evaluation is a polynomial in ``(t - t[k]) /
    h[k]``.  Columns past the order actually used are zero and contribute
    nothing.  ``y_end`` and ``t_end`` are where this call stopped.

    ``rwork``, ``iwork``, ``cstate_r`` and ``cstate_i`` are LSODA's own
    state, allocated by :func:`_lsoda_dense_work` and owned by the caller.
    ``istate_in`` is 1 to start an integration and 2 to CONTINUE the one
    those four describe, passing back the previous call's ``y_end`` and
    ``t_end`` as ``y0`` and ``t0``.  A continuation writes only the steps it
    takes itself, so a history longer than one buffer costs one integration
    rather than one per attempt.  The four are passed through untouched:
    normalising them would copy them, and a copy is not the array ODEPACK
    left its state in.

    ``jt``, ``ml`` and ``mu`` carry the Jacobian treatment, as in
    :func:`_run_odeint`.  This run and the reporting run are two separate
    integrations of one problem, so both take the same setting.  The
    Nordsieck array starts at ``RWORK(21)`` whatever ``jt`` is
    (``odepack.f:1247``), so the history read below is unaffected.

    ``itol``, ``h0``, ``hmax`` and ``hmin`` are LSODA's, and the same
    argument applies to them: ``max_step``, ``first_step``, ``min_step`` and
    a vector ``rtol``/``atol`` each change the answer, so a caller that sets
    one and also asks for dense output needs it on both runs.  ``rtol`` and
    ``atol`` reach ODEPACK as buffers, so ``itol`` 2, 3 and 4 are reachable
    here as they are in :func:`_run_odeint`.  The defaults are ODEPACK's own
    (``odepack.f:1200``): ``itol`` 1 is scalar tolerances, and ``h0``,
    ``hmax`` and ``hmin`` of 0 mean solver-determined, no upper limit and no
    lower limit.
    """
    neq = y0.size
    neq_ = np.array(neq, np.int32)
    ms = np.array(maxsteps, np.int32)
    mo = np.array(_MAXORD, np.int32)
    y0_ = np.ascontiguousarray(y0.astype(np.float64))
    t0_ = np.array(t0, np.float64)
    tf_ = np.array(tf, np.float64)
    # `_as_1d` rather than `np.array`: it is the route the reporting run's
    # tolerances take, and it turns a scalar, a tuple or an array into ONE
    # contiguous 1-D float64 buffer, which is what `itol` 2, 3 and 4 read on
    # from (gotcha 0, a strided view's `.ctypes.data` points into the base).
    rtol_ = _as_1d(rtol)
    atol_ = _as_1d(atol)
    itol_ = np.array(itol, np.int32)
    h0_ = np.array(h0, np.float64)
    hmax_ = np.array(hmax, np.float64)
    hmin_ = np.array(hmin, np.float64)
    mx = np.array(mxstep, np.int32)
    nsteps = np.zeros(1, np.int32)
    istate = np.zeros(1, np.int32)
    nfev_ = np.zeros(1, np.int32)
    njev_ = np.zeros(1, np.int32)
    t_h = np.zeros(maxsteps, np.float64)
    h_h = np.zeros(maxsteps, np.float64)
    hu_h = np.zeros(maxsteps, np.float64)
    nq_h = np.zeros(maxsteps, np.int32)
    nqn_h = np.zeros(maxsteps, np.int32)
    # Fortran fills yh_hist(neq, maxord+1, maxsteps) column-major, which is
    # C-order (maxsteps, maxord+1, neq).
    yh = np.zeros((maxsteps, _MAXORD + 1, neq), np.float64)
    args_ = np.ascontiguousarray(np.asarray(args, np.float64))
    na = np.array(args_.size, np.int32)
    jt_ = np.array(jt, np.int32)
    ml_ = np.array(ml, np.int32)
    mu_ = np.array(mu, np.int32)
    is_in = np.array(istate_in, np.int32)
    lrw_ = np.array(rwork.size, np.int32)
    liw_ = np.array(iwork.size, np.int32)
    y_end = np.zeros(neq, np.float64)
    t_end = np.zeros(1, np.float64)

    # 39 pointers, in the order the count above lists them.
    _lsoda_dense(funcptr, neq_.ctypes.data, y0_.ctypes.data,
                 t0_.ctypes.data, tf_.ctypes.data,
                 itol_.ctypes.data, rtol_.ctypes.data, atol_.ctypes.data,
                 h0_.ctypes.data, hmax_.ctypes.data, hmin_.ctypes.data,
                 mx.ctypes.data, ms.ctypes.data,
                 mo.ctypes.data, nsteps.ctypes.data, t_h.ctypes.data,
                 h_h.ctypes.data, hu_h.ctypes.data, nq_h.ctypes.data,
                 nqn_h.ctypes.data, yh.ctypes.data, istate.ctypes.data,
                 nfev_.ctypes.data, njev_.ctypes.data,
                 args_.ctypes.data, na.ctypes.data, jt_.ctypes.data,
                 ml_.ctypes.data, mu_.ctypes.data, jacptr,
                 is_in.ctypes.data, rwork.ctypes.data, lrw_.ctypes.data,
                 iwork.ctypes.data, liw_.ctypes.data, cstate_r.ctypes.data,
                 cstate_i.ctypes.data, y_end.ctypes.data, t_end.ctypes.data)

    ns = int(nsteps[0])
    t_out = t_h[:ns].copy()
    h_out = h_h[:ns].copy()
    # Transpose to (ns, n, maxord+1), the orientation the evaluator reads,
    # and apply the one correction scipy makes: when the order to attempt
    # NEXT is lower than the order just used, ODEPACK never updated the
    # last column, because the next step will not read it. Rescaling it
    # puts it back on the same step size as the rest.
    yh_out = np.zeros((ns, neq, _MAXORD + 1), np.float64)
    for k in range(ns):
        nq = int(nq_h[k])
        for j in range(nq + 1):
            for i in range(neq):
                yh_out[k, i, j] = yh[k, j, i]
        if int(nqn_h[k]) < nq and hu_h[k] != 0.0:
            s = (h_h[k] / hu_h[k]) ** nq
            for i in range(neq):
                yh_out[k, i, nq] *= s
    # The counters and the resume point are APPENDED, so r[0:5] keep their
    # meanings for every existing consumer.
    return (ns, t_out, h_out, yh_out, istate[0], nfev_[0], njev_[0],
            y_end, t_end[0])


@njit
def lsoda_dense_eval(tt, d_t, d_h, d_yh):
    """Dense output from an LSODA step history, as plain arrays.

    The array-only counterpart of
    ``solve_ivp(..., method='LSODA', dense_output=True).sol``, and the LSODA
    twin of :func:`~scijit.integrate.rk_dense_eval`.  Every argument and the
    return value is an array.

    Parameters
    ----------
    tt : float64 array, shape (k,)
        Times to evaluate at.  Outside the integrated span the nearest
        step's polynomial is extrapolated, as in scipy.
    d_t : float64 array, shape (s,)
        Time each accepted step reached.
    d_h : float64 array, shape (s,)
        Step size each stored Nordsieck block is scaled for.
    d_yh : float64 array, shape (s, n, 13)
        Nordsieck history per step; column ``j`` is ``h^j/j! * y^(j)``.

    Returns
    -------
    y : float64 array, shape (k, n)
        Solution at each ``tt``, time-major.  ``res.sol(tt)`` returns the
        transpose of this, ``(n, k)``, which is scipy's orientation.

    Raises
    ------
    ValueError
        ``d_t`` is empty, which is what an integration run without
        ``dense_output=True`` leaves behind.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> from scijit.integrate._odepack import lsoda_dense_eval
    >>> @njit
    ... def rhs(t, y):
    ...     out = np.empty(2)
    ...     out[0] = y[1]
    ...     out[1] = -4.0 * y[0]
    ...     return out
    >>> res = si.solve_ivp(rhs, (0.0, 1.0), np.array([1.0, 0.0]),
    ...                    'LSODA', None, True)
    >>> @njit
    ... def at(tt, d_t, d_h, d_yh):
    ...     return lsoda_dense_eval(tt, d_t, d_h, d_yh)
    >>> at(np.array([0.25, 0.5]), res.sol.t, res.sol.h, res.sol.yh)
    array([[ 0.87716606, -0.95824058],
           [ 0.54044215, -1.68279969]])
    """
    s = d_t.size
    if s == 0:
        raise ValueError(
            "no dense-output history; call solve_ivp with dense_output=True")
    n = d_yh.shape[1]
    ncol = d_yh.shape[2]
    out = np.empty((tt.size, n))
    for m in range(tt.size):
        # locate the step that ENDS at or after tt[m]; steps are ordered in
        # the direction of travel, so one comparison serves both directions
        k = 0
        if s > 1:
            forward = d_t[s - 1] >= d_t[0]
            k = s - 1
            for q in range(s):
                if forward:
                    if d_t[q] >= tt[m]:
                        k = q
                        break
                else:
                    if d_t[q] <= tt[m]:
                        k = q
                        break
        x = (tt[m] - d_t[k]) / d_h[k]
        p = 1.0
        for i in range(n):
            out[m, i] = 0.0
        for j in range(ncol):
            for i in range(n):
                out[m, i] += d_yh[k, i, j] * p
            p *= x
    return out


# --------------------------------------------------------------------------
# concrete worker: always computes everything, the front end slices
# --------------------------------------------------------------------------
@njit
def _run_odeint(funcptr, tfirst, y0, t, rt, at, itol, usetcrit, tcrit,
                h0, hmax, hmin, ixpr, mxstep, mxhnil, mxordn, mxords, args,
                jt=2, ml=0, mu=0, jacptr=0):
    """One LSODA run over ``t``.

    Returns ``(y, hu, tcur, tolsf, tsw, nst, nfe, nje, nqu, mused, imxer,
    lenrw, leniw, istate)``.

    ``jt`` is LSODA's Jacobian-type indicator, 1 or 2 for a full Jacobian and
    4 or 5 for a banded one, with the odd values meaning the caller supplies
    it.  ``ml`` and ``mu`` are the half-bandwidths and reach ``IWORK(1)`` and
    ``IWORK(2)``; LSODA reads them only for ``jt`` 4 and 5.  The defaults
    reproduce the finite-difference full Jacobian this routine had before
    they existed.

    ``jacptr`` is the raw address of a ``@cfunc`` matching the Fortran
    ``numba_jac`` interface, read only when ``jt`` is 1 or 4.  ``0`` is the
    null pointer the other two routes pass.
    """
    neq = y0.size
    ntime = t.size
    nleg = ntime - 1
    if nleg < 0:
        nleg = 0

    tf_ = np.array(tfirst, np.int32)
    neq_ = np.array(neq, np.int32)
    nt_ = np.array(ntime, np.int32)
    # Fortran fills yout(neq, ntime) column-major == C (ntime, neq)
    y = np.zeros((ntime, neq), np.float64)
    itol_ = np.array(itol, np.int32)
    uc_ = np.array(usetcrit, np.int32)
    # `tcrit` is the whole array, not its first element: LSODA is restarted
    # per leg inside the wrapper and the wrapper walks the array as scipy
    # does.  `_as_1d` is the same route rtol and atol take, and it is what
    # guarantees a contiguous 1-D float64 buffer from a scalar, a tuple or an
    # array in both entry points; `.ctypes.data` of a strided view points
    # into the base buffer and Fortran reads it contiguously (gotcha 0).
    tc_ = _as_1d(tcrit)
    ntc_ = np.array(tc_.size, np.int32)
    h0_ = np.array(h0, np.float64)
    hmax_ = np.array(hmax, np.float64)
    hmin_ = np.array(hmin, np.float64)
    ixpr_ = np.array(ixpr, np.int32)
    mxstep_ = np.array(mxstep, np.int32)
    mxhnil_ = np.array(mxhnil, np.int32)
    mxordn_ = np.array(mxordn, np.int32)
    mxords_ = np.array(mxords, np.int32)
    istate = np.zeros(1, np.int32)

    hu = np.zeros(nleg, np.float64)
    tcur = np.zeros(nleg, np.float64)
    tolsf = np.zeros(nleg, np.float64)
    tsw = np.zeros(nleg, np.float64)
    nst = np.zeros(nleg, np.int32)
    nfe = np.zeros(nleg, np.int32)
    nje = np.zeros(nleg, np.int32)
    nqu = np.zeros(nleg, np.int32)
    mused = np.zeros(nleg, np.int32)
    imxer = np.zeros(1, np.int32)
    lenrw = np.zeros(1, np.int32)
    leniw = np.zeros(1, np.int32)
    na = np.array(args.size, np.int32)
    jt_ = np.array(jt, np.int32)
    ml_ = np.array(ml, np.int32)
    mu_ = np.array(mu, np.int32)

    _odeint_sp(funcptr, tf_.ctypes.data, neq_.ctypes.data, y0.ctypes.data,
               t.ctypes.data, nt_.ctypes.data, y.ctypes.data,
               itol_.ctypes.data, rt.ctypes.data, at.ctypes.data,
               uc_.ctypes.data, tc_.ctypes.data, ntc_.ctypes.data,
               h0_.ctypes.data,
               hmax_.ctypes.data, hmin_.ctypes.data, ixpr_.ctypes.data,
               mxstep_.ctypes.data, mxhnil_.ctypes.data, mxordn_.ctypes.data,
               mxords_.ctypes.data, istate.ctypes.data, hu.ctypes.data,
               tcur.ctypes.data, tolsf.ctypes.data, tsw.ctypes.data,
               nst.ctypes.data, nfe.ctypes.data, nje.ctypes.data,
               nqu.ctypes.data, mused.ctypes.data, imxer.ctypes.data,
               lenrw.ctypes.data, leniw.ctypes.data, args.ctypes.data,
               na.ctypes.data, jt_.ctypes.data, ml_.ctypes.data,
               mu_.ctypes.data, jacptr)
    return (y, hu, tcur, tolsf, tsw, nst, nfe, nje, nqu, mused,
            imxer[0], lenrw[0], leniw[0], istate[0])


#: LSODA's Jacobian-type indicator, ``odepack.f:622-634``.
JT_FULL_USER = 1


JT_FULL_FD = 2


JT_BANDED_USER = 4


JT_BANDED_FD = 5


def _jac_route(has_jac, banded):
    """LSODA's ``jt`` from scipy's routing table, ``_ode.py:1300-1319``.

    ==========  ========  ====
    jacobian    banded    jt
    ==========  ========  ====
    no          no        2
    no          yes       5
    yes         no        1
    yes         yes       4
    ==========  ========  ====
    """
    if banded:
        return JT_BANDED_USER if has_jac else JT_BANDED_FD
    return JT_FULL_USER if has_jac else JT_FULL_FD
