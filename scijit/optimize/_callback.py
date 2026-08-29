"""The ``callback`` protocol the scipy-shaped minimizers share.

Two spellings reach one call site in each solver core.

    plain Python callable   ``callback(xk)`` or
                            ``callback(intermediate_result)``, halting on
                            ``StopIteration``.  scipy's own contract.
    ``@njit`` function      ``callback(xk)``, halting on ``StopIteration``.

The Python spelling is reached from compiled code through a module-level slot
and a ``numba.objmode`` block.  MEASURED on numba 0.66.0, 200000 iterations
with an empty callback: 0.732 ns/iteration with no callback, 0.985 ns with the
compiled spelling and 951.304 ns with the Python one, which is the interpreter
round trip and the GIL acquisition the block pays for.

The slot is module state, so two concurrent solves that both install a Python
callback overwrite each other, and the block holds the GIL, so a Python
callback serializes a ``numba.prange`` loop.  The compiled spelling travels as
a first-class ``@njit`` argument, holds no module state and takes no GIL.

HALTING.  The halt does not travel as an exception.  It is recorded in one
slot per numba thread, reached through a raw address, and a solver core reads
that slot with `_cb_halt_get` and leaves its loop.  No core wraps the call in
``try``/``except``, so an exception the callback raised for its own reasons
propagates out of the compiled frames where it was raised, with its class and
message intact, which is scipy's timing.  `_cb_halt_take` is what a caller
reads afterwards to apply scipy's post-step, ``status = 99``.  Only `minimize`
reads it, and scipy likewise applies that post-step only there.

The two spellings differ in where the halt is recognised.  On the Python path
``StopIteration`` is caught in the ``objmode`` block, where the interpreter is
available, and nothing else is caught at all.  On the compiled path numba
matches no exception CLASS: ``except StopIteration`` does not compile (``No
implementation of function Function(<intrinsic exception_match>)``), a bare
``raise`` inside a handler is ``UnsupportedBytecodeError: The re-raising of an
exception is not yet supported.``, and ``except Exception as e`` is
``UnsupportedBytecodeError: Exception object cannot be stored into variable
(e).``  So the shim `_cb_wrap_njit` builds around an ``@njit`` callback catches
``Exception``, and any exception from that callback halts the solve.

``basinhopping`` is the exception in both spellings, because scipy's contract
there already is ``callback(x, f, accept)`` returning ``True`` to stop.
"""
import inspect

import numpy as np
import numba
from numba import carray, get_thread_id, njit, objmode, types
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.extending import intrinsic

from ._minpack import _opt_result

__all__ = []


# ------------------------------------------------------------ the messages

#: What both entry points say when `callback` is a shape neither serves.
_CB_SPELLING = ("`callback` takes None, a plain Python callable "
                "`callback(xk)`, or a numba `@njit` function of the same "
                "shape. A `@cfunc` or a raw function address is not accepted")

#: What the compiled entry adds: a Python callable cannot be an ARGUMENT.
_CB_COMPILED = ("a plain Python callable cannot cross into compiled code as "
                "an argument; from inside `@njit` pass an `@njit` "
                "`callback(xk)`")

_BH_SPELLING = ("`callback` takes None, a plain Python callable "
                "`callback(x, f, accept)`, or a numba `@njit` function of the "
                "same shape returning a bool. A `@cfunc` or a raw function "
                "address is not accepted")

#: `differential_evolution` only: scipy's legacy two-argument spelling.
_DE_CONVERGENCE_MSG = (
    "differential_evolution: `callback(xk, convergence=val)` is not served. "
    "Use `callback(intermediate_result)`, whose `.x` and `.fun` carry the "
    "best member and its energy, or `callback(xk)`")


def _cb_msg(name, spelling=_CB_SPELLING):
    return "%s: %s" % (name, spelling)


# ---------------------------------------------------------- the no-op slots
# `None` cannot sit on a runtime-dead branch, measured: an @njit function is
# the default rather than a `None` plus a flag.

@njit
def _cb_noop(x, fval):
    """The callback a call without one runs."""
    pass


@njit
def _bh_noop(x, f, accept):
    """`basinhopping`'s no-op: never halts."""
    return False


# ------------------------------------------------------------ the halt flag
# A solver core swallows the exception the callback raised, so the halt has to
# reach the caller some other way.  Compiled code cannot write to a
# module-level array, a `numba.typed` container or a jitclass instance, and a
# closure that escapes cannot capture a run-time value, all measured on numba
# 0.66.0.  What it can write to is a raw address, which is the channel the
# Fortran packs already use.

#: scipy's post-step for a callback halt: ``status = 99``.
CB_STOP_STATUS = 99
CB_STOP_MESSAGE = '`callback` raised `StopIteration`.'

#: One slot per numba thread, so two solves in one ``numba.prange`` loop
#: cannot read each other's halt.
_N_HALT = int(numba.config.NUMBA_NUM_THREADS)
_HALT = np.zeros(_N_HALT, np.int64)
_HALT_ADDR = _HALT.ctypes.data


@intrinsic
def _as_voidptr(typingctx, addr):
    """An integer address as a pointer, so `numba.carray` can read it."""
    sig = types.voidptr(addr)

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen


@njit
def _halt_slot():
    """This thread's index into `_HALT`."""
    t = get_thread_id()
    if t < 0 or t >= _N_HALT:
        return 0
    return t


@njit
def _cb_halt_set():
    """Record that the callback halted the solve running on this thread."""
    carray(_as_voidptr(_HALT_ADDR), (_N_HALT,), np.int64)[_halt_slot()] = 1


@njit
def _cb_halt_clear():
    """Drop whatever an earlier solve left in this thread's slot."""
    carray(_as_voidptr(_HALT_ADDR), (_N_HALT,), np.int64)[_halt_slot()] = 0


@njit
def _cb_halt_get():
    """Whether the callback just called halted, leaving the slot set.

    What a solver core reads to leave its loop. It does not clear, because
    the runner reads the same slot afterwards through `_cb_halt_take` to
    apply ``status = 99``.
    """
    return carray(_as_voidptr(_HALT_ADDR), (_N_HALT,),
                  np.int64)[_halt_slot()] != 0


@njit
def _cb_halt_take():
    """Whether the solve just run halted, clearing the slot as it reads."""
    b = carray(_as_voidptr(_HALT_ADDR), (_N_HALT,), np.int64)
    i = _halt_slot()
    v = b[i]
    b[i] = 0
    return v != 0


# ------------------------------------------------- the Python callback slot

#: What a ``callback(intermediate_result)`` is handed: an `OptimizeResult`
#: carrying ``x`` and ``fun``, in that order.  MEASURED on scipy 1.18, which
#: hands its callback an ``OptimizeResult`` whose ``list(keys())`` is
#: ``['x', 'fun']``.
IntermediateResult = _opt_result(['x', 'fun'])

#: `fn` is the wrapped Python callback for the call in progress.
_SLOT = {'fn': None}


def _cb_wrap_py(callback):
    """Resolve which of scipy's two callback spellings this is.

    ``set(sig.parameters) == {'intermediate_result'}`` selects the
    ``callback(intermediate_result)`` spelling; everything else is called as
    ``callback(xk)``.
    """
    try:
        params = set(inspect.signature(callback).parameters)
    except (TypeError, ValueError):
        params = None
    if params == {'intermediate_result'}:
        def wrapped(x, fval):
            return callback(intermediate_result=IntermediateResult(x, fval))
        return wrapped

    def wrapped(x, fval):
        return callback(x)
    return wrapped


def _cb_invoke(x, fval):
    """The Python half of `_cb_py`'s ``objmode`` block. True halts the solve.

    ``StopIteration`` is scipy's halt and is the only class caught. Anything
    else leaves the ``objmode`` block and every compiled frame above it, so a
    callback that is simply broken raises where it was called rather than
    looking like a clean early stop.
    """
    try:
        _SLOT['fn'](x, fval)
        return False
    except StopIteration:
        return True


@njit
def _cb_py(x, fval):
    """Reach the installed Python callback once, from compiled code.

    The ``objmode`` block lives here rather than in a solver core: lowering
    one pickles the enclosing function, and the L-BFGS-B and SLSQP cores hold
    a ctypes entry point, which is not picklable.
    """
    with objmode(stop='boolean'):
        stop = _cb_invoke(x, fval)
    if stop:
        _cb_halt_set()


def _de_wrap_py(callback):
    """`differential_evolution`'s Python callback, in scipy's three spellings.

    ``callback(intermediate_result)`` and ``callback(xk)`` are served. scipy's
    legacy ``callback(xk, convergence=val)`` is refused rather than served a
    substitute: its value is ``tol / (std(energies) / |mean(energies)|)``, a
    different quantity from the ``fun`` this protocol carries, and supplying
    the wrong one under the right keyword is worse than declining.
    """
    try:
        params = set(inspect.signature(callback).parameters)
    except (TypeError, ValueError):
        params = None
    if params == {'intermediate_result'}:
        def wrapped(x, fval):
            return callback(intermediate_result=IntermediateResult(x, fval))
        return wrapped
    if params is not None and 'convergence' in params:
        raise ValueError(_DE_CONVERGENCE_MSG)

    def wrapped(x, fval):
        return callback(x)
    return wrapped


def _de_invoke(x, fval):
    """`differential_evolution`'s Python half. True halts the solve.

    scipy halts on ``StopIteration`` AND on a truthy return, and reports the
    same message for both. Anything else raised leaves the ``objmode`` block,
    which is also what scipy does.
    """
    try:
        r = _SLOT['fn'](x, fval)
        return r is not None and bool(r)
    except StopIteration:
        return True


@njit
def _de_py(x, fval):
    """Reach the installed Python callback once, from compiled code.

    Records the halt in the thread's slot, so one read in the core serves
    this spelling and the ``@njit`` one alike.
    """
    with objmode(stop='boolean'):
        stop = _de_invoke(x, fval)
    if stop:
        _cb_halt_set()


def _bh_invoke(x, f, accept):
    """`basinhopping`'s Python callback. scipy halts on a truthy return."""
    r = _SLOT['fn'](x, f, accept)
    return False if r is None else bool(r)


@njit
def _bh_py(x, f, accept):
    """`basinhopping`'s Python callback, from compiled code."""
    with objmode(stop='boolean'):
        stop = _bh_invoke(x, f, accept)
    return stop


def _cb_install(fn):
    """Put `fn` in the slot. Returns what `_cb_release` restores."""
    prev = _SLOT['fn']
    _SLOT['fn'] = fn
    return prev


def _cb_release(prev):
    """Restore the slot.

    Called from a ``finally``, so a solve that raised for its own reasons,
    including one the callback raised, still restores the slot.
    """
    _SLOT['fn'] = prev


# --------------------------------------- the user's @njit callback, adapted
# The cores call one fixed shape. A user's `callback(xk)` becomes it here,
# at typing time, so the arity is scipy's on the outside.

_NJIT_CB = {}
_NJIT_BH = {}


def _cb_wrap_njit(disp):
    """A user's ``@njit`` ``callback(xk)`` as the two-argument shim.

    The shim is the one place the compiled protocol's halt is read, because
    numba matches no exception CLASS, so ``StopIteration`` and a bug in the
    callback arrive here as the same ``except Exception``.
    """
    w = _NJIT_CB.get(disp)
    if w is None:
        @njit
        def w(x, fval):
            try:
                disp(x)
            except Exception:                      # noqa: BLE001
                _cb_halt_set()
        _NJIT_CB[disp] = w
    return w


def _bh_wrap_njit(disp):
    """A user's ``@njit`` ``callback(x, f, accept)``, unchanged in arity."""
    w = _NJIT_BH.get(disp)
    if w is None:
        @njit
        def w(x, f, accept):
            return disp(x, f, accept)
        _NJIT_BH[disp] = w
    return w


# ------------------------------------------------------------- the choosers

from .._lib._typing import _is_none as _is_none_v    # noqa: E402


def _cb_resolve(name, callback, bh=False, de=False):
    """Python entry point: what the core is called with.

    Returns ``(fn, use_cb, pyfn)``. `fn` is the ``@njit`` function the core
    calls, `use_cb` gates the call site, and `pyfn` is the Python callable to
    install in the slot, or ``None`` when the compiled spelling was given.
    """
    noop = _bh_noop if bh else _cb_noop
    spelling = _BH_SPELLING if bh else _CB_SPELLING
    if callback is None:
        return noop, False, None
    if hasattr(callback, 'py_func'):                  # an @njit Dispatcher
        return (_bh_wrap_njit(callback) if bh
                else _cb_wrap_njit(callback)), True, None
    if hasattr(callback, 'address') or isinstance(callback, (int, np.integer)):
        raise ValueError(_cb_msg(name, spelling))     # a @cfunc or an address
    if callable(callback):
        if bh:
            return _bh_py, True, callback
        if de:
            return _de_py, True, _de_wrap_py(callback)
        return _cb_py, True, _cb_wrap_py(callback)
    raise ValueError(_cb_msg(name, spelling))


def _cb_resolve_ty(name, callback, bh=False, de=False):
    """``@overload`` chooser: what the generated body is called with.

    Returns ``(fn, use_cb)``. A Python callable cannot reach here at all,
    so anything that is neither absent nor a Dispatcher is refused by name.
    `de` takes the same ``@njit`` shape as the default protocol, so it only
    selects the message; the Python-only halting rules live in `_de_invoke`.
    """
    if _is_none_v(callback):
        return (_bh_noop if bh else _cb_noop), False
    if isinstance(callback, types.Dispatcher):
        return (_bh_wrap_njit(callback.dispatcher) if bh
                else _cb_wrap_njit(callback.dispatcher)), True
    raise TypingError(_cb_msg(name, _BH_SPELLING if bh else _CB_SPELLING)
                      + '; ' + _CB_COMPILED)
