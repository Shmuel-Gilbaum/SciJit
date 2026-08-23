"""LSODA / LSODAR callback signatures and the LSODAR glue.

Backed by liblsoda (Nicholaswogan's thread-safe modern-Fortran ODEPACK).
LSODA is Adams/BDF with automatic stiff/nonstiff switching, the engine under
:func:`scijit.integrate.odeint` and under
``solve_ivp(method='LSODA')``. LSODAR adds root finding.

The two signature objects below describe the C ABI LSODA calls back through.
They are private (``_lsoda_sig``, ``_lsoda_event_sig``) and not part of any
call: :func:`~scijit.integrate.odeint` and ``solve_ivp(method='LSODA')`` take
a plain ``@njit`` function and build the ``@cfunc`` themselves, and refuse an
address in that slot.

    _lsoda_sig = void(float64 t, float64* y, float64* ydot, float64* args)

LSODA reaches the callback through a Fortran module variable holding its
address. That slot is ``!$omp threadprivate``, so each OS thread gets its own
copy and independent integrations run correctly under prange. The solver's
own state is allocated per call, so a nested integration is fine as well.
"""
import ctypes as ct
from numba import carray, cfunc, njit, types
from numba.core.errors import TypingError
from numba.extending import overload
import numpy as np
import os
import platform

# the internal RHS signature, matching numbalsoda's lsoda_sig. PRIVATE: no
# public routine accepts a @cfunc built against it.
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
``_odeint_scipy`` and ``_solve_ivp`` use this signature for.
"""
_lsoda_event_sig = _lsoda_sig
"""numba signature of an LSODAR event function -- the same object as
:data:`_lsoda_sig`, rebound.  Private.  ``_lsoda_event_sig is _lsoda_sig``
holds.

The shape matches, and the third pointer means something different: instead
of ``neq`` derivatives the callback writes ``ng`` event values ``g_i(t, y)``
into it.  A root is reported when any ``g_i`` changes sign.

scipy publishes no counterpart.  Its ``solve_ivp`` takes ordinary Python
callables in ``events``, its ``odeint`` has no event support, and LSODAR is
reached in scipy only through ``scipy.integrate.ode``.

Nothing outside this module builds a ``@cfunc`` against it.
``scijit.integrate.solve_ivp``'s event handling runs on a finished step
history through ``_events.py``, which takes a plain ``@njit``
``g(t, y)`` returning an ``(ng,)`` array.
"""

from .._probe import wrote_ydot as _wrote_ydot

rootdir = os.path.dirname(os.path.abspath(__file__))

if platform.uname()[0] == "Windows":
    _name = "\\liblsoda.dll"
elif platform.uname()[0] == "Linux":
    _name = "/liblsoda.so"
else:
    _name = "/liblsoda.dylib"

_lib = ct.CDLL(rootdir + _name)


def _sig(fn, nargs):
    """Declare a bind(c) wrapper as ``nargs`` opaque pointers returning void.

    Every Fortran argument crosses by reference, so the ctypes view is
    uniform and every call site passes ``.ctypes.data``.  ``nargs`` must
    match the wrapper's argument list in ``src/odepack/wrappers.f90``.  A
    miscount that disagrees with the call site raises; a miscount
    CONSISTENT with the call site raises nothing and runs into undefined
    behaviour, so recount against the Fortran rather than against a
    previous count.
    """
    fn.argtypes = [ct.c_void_p] * nargs
    fn.restype = None
    return fn


_lsoda = _sig(_lib.lsoda_wrapper, 12)
_lsodar = _sig(_lib.lsodar_wrapper, 16)


# ---------------------------------------------------------------------
# Hiding the @cfunc for LSODAR.  Both callbacks may be a raw ``.address``
# or a plain @njit function; the adapter is built at TYPING time and its
# address frozen into the compiled body.  The cache OWNS the cfunc, since
# the address is baked into compiled code.
#
# Neither adapter can know `neq`, `ng` or the number of parameters when it
# is built, so all three travel in the args buffer, which the adapter path
# prepends: ``[neq, ng, nargs, *args]``.  A two-argument callback reads only
# what it needs, slot 0 for the right-hand side and slots 0 and 1 for the
# event function; a three-argument one also reads `nargs` to slice out the
# parameters.
# ---------------------------------------------------------------------
_EVENT_ADAPTERS = {}


def _adapter_event(py):
    """cfunc around a plain @njit ``g(t, y) -> gout`` of length ``ng``.

    A three-argument ``g(t, y, args)`` also receives the parameters, which
    is the shape the right-hand side beside it takes.
    """
    from ._odeint_scipy import _rhs_arity
    nparam = _rhs_arity(py)
    key = (py, nparam)
    hit = _EVENT_ADAPTERS.get(key)
    if hit is not None:
        return hit
    inner = njit(py)

    # A wrong length writes NOTHING rather than reading past the end of
    # `out`. numba has no bounds checking, so the loop would otherwise copy
    # whatever follows the array. Writing nothing leaves the caller's
    # sentinel intact, which is what the pre-flight probe looks for.
    if nparam == 3:
        @cfunc(_lsoda_event_sig)
        def adapter(t, y_ptr, g_ptr, a_ptr):
            neq = int(a_ptr[0])
            ng = int(a_ptr[1])
            na = int(a_ptr[2])
            out = inner(t, carray(y_ptr, neq), carray(a_ptr, 3 + na)[3:])
            if out.size == ng:
                for i in range(ng):
                    g_ptr[i] = out[i]
    else:
        @cfunc(_lsoda_event_sig)
        def adapter(t, y_ptr, g_ptr, a_ptr):
            neq = int(a_ptr[0])
            ng = int(a_ptr[1])
            out = inner(t, carray(y_ptr, neq))
            if out.size == ng:
                for i in range(ng):
                    g_ptr[i] = out[i]

    _EVENT_ADAPTERS[key] = adapter
    return adapter


@njit
def _prepend_neq_ng(args, neq, ng):
    """args buffer the LSODAR adapter path expects: ``[neq, ng, nargs, *args]``.

    ``nargs`` is what lets a three-argument right-hand side or event
    function slice its own parameters back out of the buffer.
    """
    out = np.empty(args.size + 3, np.float64)
    out[0] = np.float64(neq)
    out[1] = np.float64(ng)
    out[2] = np.float64(args.size)
    for i in range(args.size):
        out[i + 3] = args[i]
    return out


def _fp_event(func, gfunc, args, neq, ng):
    """Resolve both LSODAR callbacks to ``(rhs addr, event addr, args)``.

    Mixing the two spellings is refused rather than guessed: the address
    path needs the caller's args untouched and the adapter path needs the
    two lengths prepended, and one buffer cannot be both.
    """
    raise NotImplementedError("_fp_event is @njit-only")


@overload(_fp_event)
def _fp_event_ovl(func, gfunc, args, neq, ng):
    f_is_addr = isinstance(func, types.Integer)
    g_is_addr = isinstance(gfunc, types.Integer)
    f_is_fn = isinstance(func, types.Dispatcher)
    g_is_fn = isinstance(gfunc, types.Dispatcher)
    if not (f_is_addr or f_is_fn) or not (g_is_addr or g_is_fn):
        return None
    if f_is_addr != g_is_addr:
        raise TypingError(
            "solve_ivp: funcptr and gfuncptr must use the SAME callback "
            "spelling. Pass a plain @njit function for both, or a @cfunc "
            ".address for both; the two need different args buffers, so a "
            "mixed call cannot be served")
    if f_is_addr:
        def impl(func, gfunc, args, neq, ng):
            return func, gfunc, args
        return impl

    # Deferred, because `_odeint_scipy` imports `_lsoda_sig` from this
    # module: at module level the two would form a cycle, and by the time
    # an overload BODY runs both are fully loaded.
    from ._odeint_scipy import _adapter_rhs
    f_addr = _adapter_rhs(func.dispatcher.py_func, True, 2).address
    g_addr = _adapter_event(gfunc.dispatcher.py_func).address

    def impl(func, gfunc, args, neq, ng):
        return f_addr, g_addr, _prepend_neq_ng(args, neq, ng)
    return impl
