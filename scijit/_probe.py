"""Pre-flight validation of pointer-writing ``@cfunc`` callbacks.

Every Fortran pack in this package receives its user function as a
``@cfunc`` whose OUTPUT travels through a pointer.  That has one failure
mode with no natural diagnostic: a callback that never writes leaves the
buffer holding whatever it held, and the Fortran reads that as a perfectly
converged problem.  Measured before this module existed:

    odeint,          RHS writes no ydot   -> returns the initial condition
    minimize_uobyqa, objective writes no f -> returns x0 after 22 evaluations
    lmdif,           residual writes no fvec -> returns x0

all three with ``success = True``.  A transposed argument order does exactly
this: writing ``@cfunc(minpack_jac_sig) def j(x, fvec, fjac, args, iflag)``
when the ABI is ``(x, fvec, fjac, iflag, args)`` makes ``iflag[0]`` read
``args[0]``, so the residual branch never runs.

scipy cannot hit this, because its callbacks RETURN a value whose shape it
validates (a ``None`` return raises ``RuntimeError``/``TypeError``).  The
exposure is a consequence of the pointer ABI, so the check belongs here.

How it works
------------
Call the user's callback ONCE before handing it to the Fortran, with the
output buffer filled with a sentinel.  If the buffer comes back untouched,
nothing was written.  Two DIFFERENT sentinels are used in sequence, so a
callback that happens to write the first sentinel verbatim is still seen to
have written.

The distinction this preserves is the important one: a genuine all-zero
residual is NOT flagged, because writing ``0.0`` still overwrites the
sentinel.  "The callback never ran" and "the problem is degenerate" become
separable, which is exactly what the Fortran cannot do.

Cost is one extra callback evaluation per solve (two in the rare case where
the first sentinel is written verbatim), against the hundreds or thousands a
solve performs.

Every entry point that takes one of these callbacks accepts ``validate``,
default ``True``.  Pass ``False`` to skip the probe.
"""

import llvmlite.ir as ir
import numpy as np
from numba import njit, types
from numba.extending import intrinsic

# Two unrelated denormal-range values.  Chosen to be things no real residual
# produces, and different from each other so that writing one verbatim still
# counts as a write.
SENTINEL_A = -1.234567890123e-301
SENTINEL_B = 9.876543210987e-302


# ``args`` and ``x`` may arrive as a tuple or list, which the public API
# accepts; numba's ``np.asarray`` handles those but ``.astype`` does not
# (gotcha #0), so every probe converts with ``asarray`` first.


def _dptr():
    return ir.DoubleType().as_pointer()


@intrinsic
def _call_p3(typingctx, fn_addr, p0, p1, p2):
    """Call ``void(double*, double*, double*)`` by raw address.

    Covers both ``minpack_sig`` and ``prima_sig``, which have the same C
    signature.
    """
    signature = types.void(types.intp, types.intp, types.intp, types.intp)

    def codegen(context, builder, sg, args):
        fnaddr, a0, a1, a2 = args
        fnty = ir.FunctionType(ir.VoidType(), [_dptr()] * 3)
        fptr = builder.inttoptr(fnaddr, fnty.as_pointer())
        builder.call(fptr, [builder.inttoptr(a0, _dptr()),
                            builder.inttoptr(a1, _dptr()),
                            builder.inttoptr(a2, _dptr())])
        return context.get_dummy_value()

    return signature, codegen


@intrinsic
def _call_p4(typingctx, fn_addr, p0, p1, p2, p3):
    """Call ``void(double*, double*, double*, double*)`` -- ``prima_con_sig``."""
    signature = types.void(types.intp, types.intp, types.intp, types.intp,
                           types.intp)

    def codegen(context, builder, sg, args):
        fnaddr, a0, a1, a2, a3 = args
        fnty = ir.FunctionType(ir.VoidType(), [_dptr()] * 4)
        fptr = builder.inttoptr(fnaddr, fnty.as_pointer())
        builder.call(fptr, [builder.inttoptr(a0, _dptr()),
                            builder.inttoptr(a1, _dptr()),
                            builder.inttoptr(a2, _dptr()),
                            builder.inttoptr(a3, _dptr())])
        return context.get_dummy_value()

    return signature, codegen


@intrinsic
def _call_tp3(typingctx, fn_addr, t, p0, p1, p2):
    """Call ``void(double, double*, double*, double*)`` -- ``lsoda_sig``.

    ``t`` arrives by VALUE, which is why this cannot reuse ``_call_p3``.
    """
    signature = types.void(types.intp, types.double, types.intp, types.intp,
                           types.intp)

    def codegen(context, builder, sg, args):
        fnaddr, tv, a0, a1, a2 = args
        fnty = ir.FunctionType(ir.VoidType(),
                               [ir.DoubleType()] + [_dptr()] * 3)
        fptr = builder.inttoptr(fnaddr, fnty.as_pointer())
        builder.call(fptr, [tv,
                            builder.inttoptr(a0, _dptr()),
                            builder.inttoptr(a1, _dptr()),
                            builder.inttoptr(a2, _dptr())])
        return context.get_dummy_value()

    return signature, codegen


@intrinsic
def _call_jac(typingctx, fn_addr, p0, p1, p2, pi, p4):
    """Call ``void(double*, double*, double*, int32*, double*)``.

    ``minpack_jac_sig``.
    """
    signature = types.void(types.intp, types.intp, types.intp, types.intp,
                           types.intp, types.intp)

    def codegen(context, builder, sg, args):
        fnaddr, a0, a1, a2, ai, a4 = args
        iptr = ir.IntType(32).as_pointer()
        fnty = ir.FunctionType(ir.VoidType(),
                               [_dptr()] * 3 + [iptr, _dptr()])
        fptr = builder.inttoptr(fnaddr, fnty.as_pointer())
        builder.call(fptr, [builder.inttoptr(a0, _dptr()),
                            builder.inttoptr(a1, _dptr()),
                            builder.inttoptr(a2, _dptr()),
                            builder.inttoptr(ai, iptr),
                            builder.inttoptr(a4, _dptr())])
        return context.get_dummy_value()

    return signature, codegen


@njit
def _untouched(buf, sentinel):
    for i in range(buf.size):
        if buf[i] != sentinel:
            return False
    return True


@njit
def wrote_residual(fn_addr, x, nout, args):
    """True if a ``minpack_sig``/``prima_sig`` callback writes its output.

    Parameters
    ----------
    fn_addr : int
        ``.address`` of the ``@cfunc``.
    x : 1-D float64 ndarray
        Point to probe at, normally the initial guess.
    nout : int
        Length of the output buffer (``m`` residuals, or 1 for a PRIMA
        objective).
    args : 1-D float64 ndarray
        The user's ``args``, passed through unchanged.
    """
    xw = np.ascontiguousarray(np.asarray(x).astype(np.float64))
    aw = np.ascontiguousarray(np.asarray(args).astype(np.float64))
    b1 = np.full(nout, SENTINEL_A)
    _call_p3(fn_addr, xw.ctypes.data, b1.ctypes.data, aw.ctypes.data)
    if not _untouched(b1, SENTINEL_A):
        return True
    b2 = np.full(nout, SENTINEL_B)
    _call_p3(fn_addr, xw.ctypes.data, b2.ctypes.data, aw.ctypes.data)
    return not _untouched(b2, SENTINEL_B)


@njit
def wrote_objcon(fn_addr, x, ncon, args):
    """True if a ``prima_con_sig`` callback writes ``f`` (COBYLA)."""
    xw = np.ascontiguousarray(np.asarray(x).astype(np.float64))
    aw = np.ascontiguousarray(np.asarray(args).astype(np.float64))
    nc = ncon if ncon > 0 else 1
    c1 = np.zeros(nc)
    b1 = np.full(1, SENTINEL_A)
    _call_p4(fn_addr, xw.ctypes.data, b1.ctypes.data, c1.ctypes.data,
             aw.ctypes.data)
    if not _untouched(b1, SENTINEL_A):
        return True
    b2 = np.full(1, SENTINEL_B)
    _call_p4(fn_addr, xw.ctypes.data, b2.ctypes.data, c1.ctypes.data,
             aw.ctypes.data)
    return not _untouched(b2, SENTINEL_B)


@njit
def wrote_ydot(fn_addr, t, y, args):
    """True if an ``lsoda_sig`` right-hand side writes ``ydot``."""
    yw = np.ascontiguousarray(np.asarray(y).astype(np.float64))
    aw = np.ascontiguousarray(np.asarray(args).astype(np.float64))
    n = yw.size
    d1 = np.full(n, SENTINEL_A)
    _call_tp3(fn_addr, t, yw.ctypes.data, d1.ctypes.data, aw.ctypes.data)
    if not _untouched(d1, SENTINEL_A):
        return True
    d2 = np.full(n, SENTINEL_B)
    _call_tp3(fn_addr, t, yw.ctypes.data, d2.ctypes.data, aw.ctypes.data)
    return not _untouched(d2, SENTINEL_B)


@njit
def wrote_jac_residual(fn_addr, x, nout, njac, args):
    """True if a ``minpack_jac_sig`` callback writes ``fvec`` at iflag=1.

    Probes the RESIDUAL branch specifically, which is the one a transposed
    ``iflag``/``args`` argument order silently skips.
    """
    xw = np.ascontiguousarray(np.asarray(x).astype(np.float64))
    aw = np.ascontiguousarray(np.asarray(args).astype(np.float64))
    jbuf = np.zeros(njac)
    iflag = np.ones(1, np.int32)
    b1 = np.full(nout, SENTINEL_A)
    _call_jac(fn_addr, xw.ctypes.data, b1.ctypes.data, jbuf.ctypes.data,
              iflag.ctypes.data, aw.ctypes.data)
    if not _untouched(b1, SENTINEL_A):
        return True
    iflag[0] = 1
    b2 = np.full(nout, SENTINEL_B)
    _call_jac(fn_addr, xw.ctypes.data, b2.ctypes.data, jbuf.ctypes.data,
              iflag.ctypes.data, aw.ctypes.data)
    return not _untouched(b2, SENTINEL_B)


@intrinsic
def _call_aprod(typingctx, fn_addr, transa, m, n, p0, p1, p2):
    """Call ``void(int32, int32, int32, double*, double*, double*)``.

    ``propack_sig``.  The first three arguments arrive BY VALUE, and the
    operation code selects ``y = A @ x`` (0) or ``y = A.T @ x`` (1).
    """
    signature = types.void(types.intp, types.int32, types.int32, types.int32,
                           types.intp, types.intp, types.intp)

    def codegen(context, builder, sg, args):
        fnaddr, ta, mv, nv, a0, a1, a2 = args
        i32 = ir.IntType(32)
        fnty = ir.FunctionType(ir.VoidType(), [i32] * 3 + [_dptr()] * 3)
        fptr = builder.inttoptr(fnaddr, fnty.as_pointer())
        builder.call(fptr, [ta, mv, nv,
                            builder.inttoptr(a0, _dptr()),
                            builder.inttoptr(a1, _dptr()),
                            builder.inttoptr(a2, _dptr())])
        return context.get_dummy_value()

    return signature, codegen


@njit
def wrote_aprod(fn_addr, m, n, args):
    """True if a ``propack_sig`` operator writes ``y`` for ``y = A @ x``.

    Probes with ``transa = 0``, so ``x`` has length ``n`` and ``y`` length
    ``m``.  ``x`` is all ones rather than zeros: a correct operator applied
    to a zero vector legitimately returns zeros, which would be
    indistinguishable from not writing at all.
    """
    aw = np.ascontiguousarray(np.asarray(args).astype(np.float64))
    xv = np.ones(n)
    y1 = np.full(m, SENTINEL_A)
    _call_aprod(fn_addr, np.int32(0), np.int32(m), np.int32(n),
                xv.ctypes.data, y1.ctypes.data, aw.ctypes.data)
    if not _untouched(y1, SENTINEL_A):
        return True
    y2 = np.full(m, SENTINEL_B)
    _call_aprod(fn_addr, np.int32(0), np.int32(m), np.int32(n),
                xv.ctypes.data, y2.ctypes.data, aw.ctypes.data)
    return not _untouched(y2, SENTINEL_B)


@njit
def _eval_resid(fn_addr, x, nout, args):
    xw = np.ascontiguousarray(np.asarray(x).astype(np.float64))
    aw = np.ascontiguousarray(np.asarray(args).astype(np.float64))
    f = np.zeros(nout)
    _call_p3(fn_addr, xw.ctypes.data, f.ctypes.data, aw.ctypes.data)
    return f


@njit
def residual_status(fn_addr, xstar, fstar, nout, args):
    """Classify the returned solution.  0 ok, 1 degenerate, 2 inconsistent.

    Three stages, cheapest first:

    1. ``fstar`` is the residual the solver already computed at ``xstar``.
       Any nonzero entry settles it, with no extra callback evaluation.
       That is every ordinary solve, so the rest is rarely reached.
    2. ``fstar`` all zero: evaluate ``f(xstar)`` directly.  If the fresh
       value is NOT zero it disagrees with what the solver reported, which
       is a convergence failure rather than a degenerate residual, and is
       reported separately (status 2).  A stale ``fvec`` from an earlier
       iterate or a callback with state both land here.
    3. Fresh value also zero: perturb one component at a time.  All zero
       again means the residual carries no information (status 1).
    """
    for i in range(nout):
        if fstar[i] != 0.0:
            return 0

    f0 = _eval_resid(fn_addr, xstar, nout, args)
    for i in range(nout):
        if f0[i] != 0.0:
            return 2

    xs = np.ascontiguousarray(np.asarray(xstar).astype(np.float64))
    for j in range(xs.size):
        xp = xs.copy()
        xp[j] = xs[j] + 1e-3 * (1.0 + np.abs(xs[j]))
        fj = _eval_resid(fn_addr, np.ascontiguousarray(xp), nout, args)
        for i in range(nout):
            if fj[i] != 0.0:
                return 0            # varies: a real residual, not degenerate
    return 1


@njit
def _eval_jac_resid(fn_addr, x, nout, njac, args):
    """Residual from a ``minpack_jac_sig`` callback, at ``iflag = 1``."""
    xw = np.ascontiguousarray(np.asarray(x).astype(np.float64))
    aw = np.ascontiguousarray(np.asarray(args).astype(np.float64))
    f = np.zeros(nout)
    jbuf = np.zeros(njac)
    iflag = np.ones(1, np.int32)
    _call_jac(fn_addr, xw.ctypes.data, f.ctypes.data, jbuf.ctypes.data,
              iflag.ctypes.data, aw.ctypes.data)
    return f


@njit
def jac_residual_status(fn_addr, xstar, fstar, nout, njac, args):
    """``residual_status`` for the analytic-Jacobian callbacks."""
    for i in range(nout):
        if fstar[i] != 0.0:
            return 0

    f0 = _eval_jac_resid(fn_addr, xstar, nout, njac, args)
    for i in range(nout):
        if f0[i] != 0.0:
            return 2

    xs = np.ascontiguousarray(np.asarray(xstar).astype(np.float64))
    for j in range(xs.size):
        xp = xs.copy()
        xp[j] = xs[j] + 1e-3 * (1.0 + np.abs(xs[j]))
        fj = _eval_jac_resid(fn_addr, np.ascontiguousarray(xp), nout, njac,
                             args)
        for i in range(nout):
            if fj[i] != 0.0:
                return 0            # varies: a real residual, not degenerate
    return 1
