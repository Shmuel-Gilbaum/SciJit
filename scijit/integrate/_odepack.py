"""The LSODA callback signature.

Backed by liblsoda (Nicholaswogan's thread-safe modern-Fortran ODEPACK).
LSODA is Adams/BDF with automatic stiff/nonstiff switching, the engine under
:func:`scijit.integrate.odeint` and under ``solve_ivp(method='LSODA')``.

The signature object below describes the C ABI LSODA calls back through.  It
is private (``_lsoda_sig``) and not part of any call:
:func:`~scijit.integrate.odeint` and ``solve_ivp(method='LSODA')`` take a
plain ``@njit`` function and build the ``@cfunc`` themselves, and refuse an
address in that slot.

    _lsoda_sig = void(float64 t, float64* y, float64* ydot, float64* args)

LSODA reaches the callback through a Fortran module variable holding its
address. That slot is ``!$omp threadprivate``, so each OS thread gets its own
copy and independent integrations run correctly under prange. The solver's
own state is allocated per call, so a nested integration is fine as well.
"""
from numba import types

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
