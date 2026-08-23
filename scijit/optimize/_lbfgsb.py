"""Numba-callable L-BFGS-B (bound-constrained minimization, large-n).

Backed by liblbfgsb (jacobwilliams/lbfgsb modern refactor of Nocedal/
Zhu L-BFGS-B 3.0, BSD-3). Equivalent of
scipy.optimize.minimize(method='L-BFGS-B').

setulb is REVERSE COMMUNICATION: the @njit driver loop below asks the
user function for f and g when the solver requests them. Consequences:

  * the objective is a PLAIN @njit function, no @cfunc, no address:
        @njit
        def fg(x, *args):         # returns (f, gradient_array)
            ...
            return f, g
  * the solver holds no module-level state (task/workspaces live in
    caller arrays), so concurrent solves are safe (prange OK) unless a
    PYTHON `callback` is given, which travels through a module-level
    slot. The minpack pack reaches this too, by a different route: its
    callback slot is a Fortran module variable carrying
    !$omp threadprivate.

Bounds: pass `bounds` as a sequence of (min, max) pairs or an (n, 2)
array, using +-np.inf for unbounded ends, or None for an unconstrained
minimization. The nbd flags scipy computes are derived the same way
here.
"""
import ctypes as ct

from numba import njit, types
from numba.extending import overload
import numpy as np
import os
import platform

rootdir = os.path.dirname(os.path.abspath(__file__))

if platform.uname()[0] == "Windows":
    _name = "\\liblbfgsb.dll"
elif platform.uname()[0] == "Linux":
    _name = "/liblbfgsb.so"
else:
    _name = "/liblbfgsb.dylib"

_lib = ct.CDLL(rootdir + _name)


def _sig(fn, nargs):
    """Give the L-BFGS-B wrapper its ctypes signature.

    The ``bind(c)`` entry point in ``src/lbfgsb/05_wrappers.f90`` takes
    every argument by reference and returns nothing, so the signature is
    ``[c_void_p] * nargs`` with ``restype = None`` and the call sites
    pass ``array.ctypes.data``. ``nargs`` must be recounted against that
    file whenever the wrapper changes: a count that disagrees with the
    call site surfaces as a cryptic numba ``ExternalFunctionPointer``
    typing error, and a count that is wrong in both places raises nothing
    at all and runs into undefined behaviour.
    """
    fn.argtypes = [ct.c_void_p] * nargs
    fn.restype = None
    return fn


_setulb = _sig(_lib.setulb_wrapper, 19)

# `args` coercion and the result-container factory are shared with the MINPACK
# entry points rather than copied
from ._minpack import _as_args, _result      # noqa: E402
# one spelling of CPython's type names for a message, shared with _scalar.py
from ._scalar import _ty_name       # noqa: E402
from ._callback import (_cb_noop, _cb_install, _cb_release,   # noqa: E402
                        _cb_resolve, _cb_resolve_ty,
                        _cb_halt_get, _cb_halt_clear)

# Fortran CHARACTER strings are BLANK-padded, and the task buffer crosses
# the ABI as int8, so the pad byte is ASCII 32.
_BLANK = np.int8(32)


@njit
def _new_task():
    """60-byte task buffer initialised to 'START' + blanks."""
    t = np.full(60, _BLANK, np.int8)
    t[0] = 83; t[1] = 84; t[2] = 65; t[3] = 82; t[4] = 84   # START
    return t


@njit
def _set_stop(t):
    """Overwrite the task buffer with 'STOP' + blanks."""
    for i in range(60):
        t[i] = _BLANK
    t[0] = 83; t[1] = 84; t[2] = 79; t[3] = 80              # STOP


@njit
def _prefix2(t, a, b):
    """True when the task buffer opens with the two byte codes ``a``, ``b``.

    L-BFGS-B reports its reverse-communication state in a Fortran
    ``character*60``, which crosses the ABI as 60 ``int8`` because numba
    has no character type. Two bytes separate the three states the driver
    acts on, so the whole state test is this; the call sites carry the
    ASCII in a trailing comment.
    """
    return t[0] == a and t[1] == b


#: scipy's ``fmin_l_bfgs_b`` information dict. Reached by name, by position
#: and by string key, from Python and from inside ``@njit``.
LbfgsbInfo = _result('LbfgsbInfo',
                     ['grad', 'task', 'funcalls', 'nit', 'warnflag'])

_EPS = float(np.finfo(np.float64).eps)

# driver-side STOP reasons, so the message can name which limit was hit
_STOP_NONE, _STOP_MAXITER, _STOP_MAXFUN, _STOP_CALLBACK = 0, 1, 2, 3

_BOUNDS_LEN_MSG = "length of x0 != length of bounds"

#: CPython's own texts for ``lb, ub = zip(*bounds)`` and ``float(row)``.
_UNPACK_MANY_MSG = "too many values to unpack (expected 2)"
_ITEM_MSG = "can only convert an array of size 1 to a Python scalar"

#: numpy's texts for a finite-difference step that does not broadcast.
_EPS_SEQ_MSG = "setting an array element with a sequence."
_EPS_IDX_MSG = "index 1 is out of bounds for axis 0 with size 1"

_BOUNDS_SHAPE_MSG = (
    "bounds must be a sequence of (min, max) pairs, one per element of x0, "
    "or an (n, 2) array. Use None or -+inf for an absent bound. Inside @njit "
    "a MIXED list such as [(None, 5.0), (0.0, 1.0)] cannot be used: numba "
    "refuses to unbox a heterogeneous list. The same pairs as a TUPLE work, "
    "and so does the all-inf spelling.")


def _lit_bool(v):
    """Resolve an ``@overload`` argument to a compile-time bool, or None.

    An overload whose RETURN TYPE depends on a flag has to know the flag
    while it is typing the body, and the flag arrives in five different
    shapes. ``None`` means it is a runtime variable and cannot be served:
    the caller then returns ``None`` from the overload, which numba
    reports as a TypingError.

    The first two branches are not redundant with the last two. numba
    hands an OMITTED argument the RAW PYTHON DEFAULT, a builtins ``bool``
    or ``int``, never a ``types.BooleanLiteral`` -- measured on numba
    0.66, omitting the argument gives ``bool True`` where passing it
    explicitly gives ``Literal[bool](True)``. Deleting them breaks every
    call that leaves the flag out.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, types.Omitted):
        return bool(v.value)
    if isinstance(v, types.BooleanLiteral):
        return v.literal_value
    if isinstance(v, types.IntegerLiteral):
        return bool(v.literal_value)
    return None


def _is_none(v):
    """True when an ``@overload`` argument is absent.

    Three unrelated objects mean absent, and an overload deciding whether
    to serve a call has to accept all three: Python's own ``None``, which
    is what numba hands over for an OMITTED argument; a
    ``types.NoneType``, which is an explicitly passed ``None``; and a
    ``types.Omitted`` wrapping ``None``. Dropping the first test breaks
    every call that leaves the argument out.
    """
    return (v is None or isinstance(v, types.NoneType)
            or (isinstance(v, types.Omitted) and v.value is None))


def _eps_at(eps, i):
    """Coordinate ``i``'s finite-difference step.

    scipy's `eps` is a float or an ndarray. An ndarray gives one step per
    coordinate and a size-1 array applies to every coordinate, which is
    numpy broadcasting inside ``approx_derivative``. A rank other than 1 is
    flattened first, so a 0-d array reads as the scalar it holds.
    """
    if isinstance(eps, np.ndarray):
        f = np.ravel(eps)
        return float(f[0] if f.size == 1 else f[i])
    return float(eps)


@overload(_eps_at)
def _eps_at_ovl(eps, i):
    if isinstance(eps, types.Array):
        if eps.ndim == 1:
            def impl(eps, i):
                if eps.size == 1:
                    return np.float64(eps[0])
                return np.float64(eps[i])
            return impl

        def impl(eps, i):
            f = np.ravel(eps)
            if f.size == 1:
                return np.float64(f[0])
            return np.float64(f[i])
        return impl

    def impl(eps, i):
        return np.float64(eps)
    return impl


@njit
def _check_eps(size, ndim, n):
    """What numpy raises when a finite-difference step does not broadcast.

    ``approx_derivative`` adds `eps` to an ``(n,)`` iterate and indexes the
    result per coordinate, so a 1-D step of length other than 1 or `n` fails
    the broadcast, a step of rank 2 or more holding several values fails the
    assignment, and a rank-2 step holding one value leaves an array whose
    first axis has length 1.
    """
    if ndim > 1:
        if size != 1:
            raise ValueError(_EPS_SEQ_MSG)
        if n > 1:
            raise IndexError(_EPS_IDX_MSG)
    elif size != 1 and size != n:
        raise ValueError("operands could not be broadcast together with "
                         "shapes (" + str(n) + ",) (" + str(size) + ",) ")


@njit
def _check_pair_width(k):
    """What CPython raises when ``lb, ub = zip(*bounds)`` gives ``k`` groups.

    ``zip`` transposes, so `k` is the length of the shortest row of `bounds`
    and the two names bind only when it is exactly 2.
    """
    if k > 2:
        raise ValueError(_UNPACK_MANY_MSG)
    if k < 2:
        raise ValueError("not enough values to unpack (expected 2, got "
                         + str(k) + ")")


def _np_scalar_name(dt):
    """CPython's ``type(x).__name__`` for the scalar a numpy dtype yields."""
    if isinstance(dt, types.Boolean):
        return 'numpy.bool'
    if isinstance(dt, types.Integer):
        return 'numpy.%sint%d' % ('' if dt.signed else 'u', dt.bitwidth)
    if isinstance(dt, types.Float):
        return 'numpy.float%d' % dt.bitwidth
    if isinstance(dt, types.Complex):
        return 'numpy.complex%d' % dt.bitwidth
    return str(dt)


def _refusing_impl(ref):
    """A `_split_bounds` body that only raises, and still returns two arrays.

    A body whose every path raises has return type ``none``, and the caller
    then fails to unpack it with ``failed to unpack none`` rather than
    raising what the body says. The unreachable return has to stay reachable
    to numba, so the test is on `n`, whose value arrives at run time.
    """
    cls, msg = ref
    if cls is TypeError:
        def impl(bounds, n):
            if n >= 0:
                raise TypeError(msg)
            return np.zeros(0, np.float64), np.zeros(0, np.float64)
        return impl

    def impl(bounds, n):
        if n >= 0:
            raise ValueError(msg)
        return np.zeros(0, np.float64), np.zeros(0, np.float64)
    return impl


def _pair_refusal(els):
    """The refusal ``zip(*bounds)`` produces for these row types, or ``None``.

    Returns a ``(class, message)`` pair. A row that is not iterable at all
    decides the outcome first, and otherwise the count of unpacked names is
    the SHORTEST row, which is why a mixed-width `bounds` can still bind.
    """
    if not els:
        return None
    for e in els:
        if not isinstance(e, (types.BaseTuple, types.Array, types.List,
                              types.ListType)):
            return TypeError, "'" + _ty_name(e) + "' object is not iterable"
    widths = [len(e) for e in els if isinstance(e, types.BaseTuple)]
    if len(widths) != len(els):
        return None                 # an array row: its length is run-time
    k = min(widths)
    if k > 2:
        return ValueError, _UNPACK_MANY_MSG
    if k < 2:
        return ValueError, ("not enough values to unpack (expected 2, got %d)"
                            % k)
    return None


# --------------------------------------------------------------------------
# the three ways scipy's fmin_l_bfgs_b can be handed f and g.  Each is a tiny
# @njit evaluator passed to the ONE driver loop as a first-class argument --
# a numba closure cannot capture a runtime function value, so it travels as
# an argument (the pattern _multiquad._adapt uses).
# --------------------------------------------------------------------------
def _call_args(func, x, args):
    """``func(x, *args)`` for a tuple `args`, ``func(x, args)`` otherwise.

    scipy's contract is the splat. The second arm carries this package's own
    one-buffer spelling, which `minimize` passes and which serves a sequence
    whose length is a run-time fact: the arity of a compiled call is fixed
    when it compiles, so an ndarray cannot be unpacked into one.

    Shared with `_slsqp` so the two names answer `args` the same way.
    """
    if isinstance(args, tuple):
        return func(x, *args)
    return func(x, args)


@overload(_call_args)
def _call_args_ovl(func, x, args):
    if isinstance(args, types.BaseTuple) or (isinstance(args, tuple)
                                             and len(args) == 0):
        def impl(func, x, args):
            return func(x, *args)
        return impl

    def impl(func, x, args):
        return func(x, args)
    return impl


@njit
def _ev_fg(func, fprime, x, args, eps, l, u, nbd):
    """``fprime=None, approx_grad=0``: ``func(x, *args) -> (f, g)``.

    Returns ``(f, g, nfe)``; ``nfe`` is what scipy's ``sf.nfev`` counts, which
    on this protocol is one evaluation per call.
    """
    f, g = _call_args(func, x, args)
    return f, g, 1


@njit
def _ev_split(func, fprime, x, args, eps, l, u, nbd):
    """``fprime`` given: ``func(x, *args) -> f``, ``fprime(x, *args) -> g``."""
    return _call_args(func, x, args), _call_args(fprime, x, args), 1


@njit
def _ev_approx(func, fprime, x, args, eps, l, u, nbd):
    """``approx_grad=1``: forward differences with absolute step ``eps``.

    ``eps`` is a scalar or one step per coordinate; ``_eps_at`` reads it.

    Transcribes scipy's ``approx_derivative(method='2-point', abs_step=eps)``
    together with ``_adjust_scheme_to_bounds``: a step that would leave the
    box is flipped in sign if it fits on the other side, and clamped to the
    remaining distance if it fits on neither.  The divisor is the step
    taken, ``x1[i] - x0[i]``, not the requested one.
    """
    n = x.size
    f0 = _call_args(func, x, args)
    g = np.empty(n, np.float64)
    xt = x.copy()
    for i in range(n):
        lb = l[i] if (nbd[i] == 1 or nbd[i] == 2) else -np.inf
        ub = u[i] if (nbd[i] == 2 or nbd[i] == 3) else np.inf
        lower_dist = x[i] - lb
        upper_dist = ub - x[i]
        h = _eps_at(eps, i)
        violated = (x[i] + h < lb) or (x[i] + h > ub)
        fitting = abs(h) <= max(lower_dist, upper_dist)
        if violated and fitting:
            h = -h
        elif not fitting:
            h = upper_dist if upper_dist >= lower_dist else -lower_dist
        xt[i] = x[i] + h
        dx = xt[i] - x[i]
        g[i] = (_call_args(func, xt, args) - f0) / dx
        xt[i] = x[i]
    # scipy's sf.nfev counts the finite-difference probes too: 1 + n per
    # gradient, so its funcalls is (n+1) times ours would otherwise be
    return f0, g, n + 1


@njit
def _lbfgsb_run(ev, func, fprime, x0, l, u, nbd, args, m, factr, pgtol,
                epsilon, maxfun, maxiter, iprint, maxls,
                cb=_cb_noop, use_cb=False):
    """The reverse-communication driver.  Returns everything; the entries slice.

    Returns ``(x, f, g, conv, stop_reason, nfev, nit, which, sk, yk)``.

    ``sk`` and ``yk`` are the stored correction pairs, shape ``(n_corrs, n)``.
    They live in the first ``2*m*n`` slots of `wa`, which this function owns,
    so no Fortran change exposes them: ``Isave(4)`` and ``Isave(5)`` in
    ``src/lbfgsb/04_lbfgsb.f90`` set those two offsets, and ``setulb`` passes
    ``Isave(22)`` as ``mainlb``'s base, which puts ``iupdat``, the number of
    updates performed, at ``isave[30]`` counting from zero.
    """
    n = np.int32(x0.size)
    n_ = np.array(n, np.int32)
    m_ = np.array(m, np.int32)
    x = np.ascontiguousarray(np.asarray(x0)).astype(np.float64).copy()

    f_ = np.zeros(1, np.float64)
    g = np.zeros(n, np.float64)
    factr_ = np.array(factr, np.float64)
    pgtol_ = np.array(pgtol, np.float64)
    wa = np.zeros(2 * m * n + 5 * n + 11 * m * m + 8 * m, np.float64)
    iwa = np.zeros(3 * n, np.int32)
    task = _new_task()
    iprint_ = np.array(iprint, np.int32)
    maxls_ = np.array(maxls, np.int32)
    csave = np.full(60, _BLANK, np.int8)
    lsave = np.zeros(4, np.int32)
    isave = np.zeros(44, np.int32)
    dsave = np.zeros(29, np.float64)

    nfev = 0
    nit = 0
    stop_reason = _STOP_NONE
    # Drop whatever an earlier solve on this thread left in the halt
    # slot, so the first read below cannot see it.  A standalone entry
    # point does not clear it on the way out: only `minimize` takes it.
    _cb_halt_clear()
    while True:
        _setulb(n_.ctypes.data, m_.ctypes.data, x.ctypes.data,
                l.ctypes.data, u.ctypes.data, nbd.ctypes.data,
                f_.ctypes.data, g.ctypes.data, factr_.ctypes.data,
                pgtol_.ctypes.data, wa.ctypes.data, iwa.ctypes.data,
                task.ctypes.data, iprint_.ctypes.data, csave.ctypes.data,
                lsave.ctypes.data, isave.ctypes.data, dsave.ctypes.data,
                maxls_.ctypes.data)

        if _prefix2(task, 70, 71):                     # 'FG'
            fv, gv, nfe = ev(func, fprime, x, args, epsilon, l, u, nbd)
            if len(gv) != n:
                # numba does not bounds-check, and scipy's L-BFGS-B accepts a
                # short gradient and returns x0 with success=True
                raise ValueError("l_bfgs_b: the gradient length does not "
                                 "match len(x0)")
            f_[0] = fv
            for i in range(n):
                g[i] = gv[i]
            nfev += nfe
        elif _prefix2(task, 78, 69):                   # 'NE' (NEW_X)
            nit += 1
            # scipy calls the callback here, at NEW_X and before the two
            # budget tests, and lets either of them overwrite the callback's
            # own STOP.
            if use_cb:
                cb(x.copy(), f_[0])
                if _cb_halt_get():
                    stop_reason = _STOP_CALLBACK
                    _set_stop(task)
            if nit >= maxiter:
                stop_reason = _STOP_MAXITER
                _set_stop(task)
            elif nfev > maxfun:
                stop_reason = _STOP_MAXFUN
                _set_stop(task)
        else:
            break

    conv = _prefix2(task, 67, 79)                      # 'CO'NVERGENCE
    # which outcome the Fortran reported: 1 and 2 are the two convergence
    # tests, and the error and warning classes carry their own codes
    which = _task_code(task)

    nc = isave[30]
    if nc > m:
        nc = m
    if nc < 0:
        nc = 0
    sk = np.ascontiguousarray(wa[0:m * n].reshape(m, n)[:nc]).copy()
    yk = np.ascontiguousarray(wa[m * n:2 * m * n].reshape(m, n)[:nc]).copy()
    return x, f_[0], g, conv, stop_reason, nfev, nit, which, sk, yk


@njit
def _task_code(task):
    """The outcome the Fortran wrote in the task buffer, as a code.

    L-BFGS-B 3.0 reports its state as text in a ``character*60``, where
    scipy's C translation reports two integers and looks them up in two
    tables. The text is matched here and the outcome travels as an integer,
    so no caller of `_lbfgsb_run` sees a different return shape. `1` and `2`
    are the two convergence tests, which is what the code meant before the
    error and warning classes were added to it.
    """
    last = -1
    for i in range(60):
        if task[i] != _BLANK:
            last = i
    s = ''
    for i in range(last + 1):
        s = s + chr(task[i])
    if s == 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL':
        return 1
    if s == 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH':
        return 2
    if s == 'ERROR: NO FEASIBLE SOLUTION':
        return 4
    if s == 'ERROR: FACTR < 0':
        return 5
    if s == 'ERROR: FTOL < ZERO':
        return 6
    if s == 'ERROR: GTOL < ZERO':
        return 7
    if s == 'ERROR: XTOL < ZERO':
        return 8
    if s == 'ERROR: STP < STPMIN':
        return 9
    if s == 'ERROR: STP > STPMAX':
        return 10
    if s == 'ERROR: STPMIN < ZERO':
        return 11
    if s == 'ERROR: STPMAX < STPMIN':
        return 12
    if s == 'ERROR: INITIAL G >= ZERO':
        return 13
    if s == 'ERROR: M <= 0':
        return 14
    if s == 'ERROR: N <= 0':
        return 15
    if s == 'ERROR: INVALID NBD':
        return 16
    if s == 'WARNING: ROUNDING ERRORS PREVENT PROGRESS':
        return 17
    if s == 'WARNING: STP = STPMAX':
        return 18
    if s == 'WARNING: STP = STPMIN':
        return 19
    if s == 'WARNING: XTOL TEST SATISFIED':
        return 20
    return 0


@njit
def _lbfgsb_msg(conv, which, stop_reason):
    """scipy's message, which it builds from a status table and a task table.

    The vendored Fortran writes the ORIGINAL L-BFGS-B 3.0 task strings, which
    carry underscores where scipy's tables carry spaces and ``ZERO`` where
    they carry ``0``, and which name the line-search failure where scipy
    leaves the task half empty. The outcome is translated rather than the
    characters, so the two agree exactly.
    """
    if conv:
        if which == 1:
            return "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
        return "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
    if stop_reason == _STOP_MAXITER:
        return "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
    if stop_reason == _STOP_MAXFUN:
        return "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
    if stop_reason == _STOP_CALLBACK:
        return "STOP: CALLBACK REQUESTED HALT"
    if which == 4:
        return "ERROR: NO FEASIBLE SOLUTION"
    if which == 5:
        return "ERROR: FACTR < 0"
    if which == 6:
        return "ERROR: FTOL < 0"
    if which == 7:
        return "ERROR: GTOL < 0"
    if which == 8:
        return "ERROR: XTOL < 0"
    if which == 9:
        return "ERROR: STP < STPMIN"
    if which == 10:
        return "ERROR: STP > STPMAX"
    if which == 11:
        return "ERROR: STPMIN < 0"
    if which == 12:
        return "ERROR: STPMAX < STPMIN"
    if which == 13:
        return "ERROR: INITIAL G >= 0"
    if which == 14:
        return "ERROR: M <= 0"
    if which == 15:
        return "ERROR: N <= 0"
    if which == 16:
        return "ERROR: INVALID NBD"
    if which == 17:
        return "WARNING: ROUNDING ERRORS PREVENT PROGRESS"
    if which == 18:
        return "WARNING: STP = STPMAX"
    if which == 19:
        return "WARNING: STP = STPMIN"
    if which == 20:
        return "WARNING: XTOL TEST SATISFIED"
    return "ABNORMAL: "


@njit
def _lbfgsb_flag(conv, stop_reason):
    """scipy's ``warnflag``: 0 converged, 1 a limit was hit, 2 anything else.

    A callback halt is in the third class: scipy reaches ``warnflag`` 2 for
    it, because neither budget was exhausted.
    """
    if conv:
        return 0
    if stop_reason == _STOP_MAXITER or stop_reason == _STOP_MAXFUN:
        return 1
    return 2


@njit
def _lbfgsb_bounds(n, lower, upper):
    """scipy's ``nbd`` encoding, from lower/upper arrays with +-inf."""
    l = np.zeros(n, np.float64)
    u = np.zeros(n, np.float64)
    nbd = np.zeros(n, np.int32)
    has_l = lower.size == n
    has_u = upper.size == n
    if lower.size != 0 and not has_l:
        raise ValueError(_BOUNDS_LEN_MSG)
    if upper.size != 0 and not has_u:
        raise ValueError(_BOUNDS_LEN_MSG)
    for i in range(n):
        lo = lower[i] if has_l else -np.inf
        hi = upper[i] if has_u else np.inf
        if lo > hi:
            raise ValueError("LBFGSB - one of the lower bounds is greater "
                             "than an upper bound.")
        lf = np.isfinite(lo)
        uf = np.isfinite(hi)
        if lf:
            l[i] = lo
        if uf:
            u[i] = hi
        if lf and uf:
            nbd[i] = 2
        elif lf:
            nbd[i] = 1
        elif uf:
            nbd[i] = 3
    return l, u, nbd


@njit
def _clip_to_box(x, l, u, nbd):
    """scipy clips x0 into the box before the first evaluation."""
    out = x.copy()
    for i in range(x.size):
        if (nbd[i] == 1 or nbd[i] == 2) and out[i] < l[i]:
            out[i] = l[i]
        if (nbd[i] == 2 or nbd[i] == 3) and out[i] > u[i]:
            out[i] = u[i]
    return out


def _arr_to_scalar(v):
    """A one-element ndarray as the number in it, anything else unchanged."""
    return v.item() if isinstance(v, np.ndarray) else v


def _as_bounds(bounds):
    """``bounds`` -> ``(lower, upper)``, ``None`` in a slot meaning open.

    Transcribes scipy's ``old_bound_to_new``: the pairs are TRANSPOSED with
    ``zip(*bounds)``, so a row that is not a pair is what decides the
    exception, and a one-element array in a slot reads as the number in it.
    The LENGTH test is the caller's, because ``fmin_l_bfgs_b`` places it
    before the transposition and ``fmin_slsqp`` after it.
    """
    lo_seq, hi_seq = zip(*bounds)
    lo = np.array([-np.inf if v is None else float(_arr_to_scalar(v))
                   for v in lo_seq], np.float64)
    hi = np.array([np.inf if v is None else float(_arr_to_scalar(v))
                   for v in hi_seq], np.float64)
    return lo, hi


def _split_bounds(bounds, n):
    """``bounds`` -> ``(lower, upper)``, the same transposition as `_as_bounds`.

    Accepts an ``(n, 2)`` array, a sequence of ``(min, max)`` pairs, and a
    higher-rank array whose slots each hold one number. A pair containing
    ``None`` is python-only; see ``_BOUNDS_SHAPE_MSG``. The length test is the
    caller's.
    """
    return _as_bounds(bounds)


@overload(_split_bounds)
def _split_bounds_ovl(bounds, n):
    """Every ``bounds`` spelling numba can type, measured not assumed.

    ``verify/bounds_typing.txt``:  a tuple or list of float pairs, a tuple or
    list of pairs uniformly carrying ``None``, a MIXED tuple, and an ``(n, 2)``
    array all type.  Only a MIXED LIST fails, and it fails in numba's list
    unboxing before this overload is reached, so it cannot be served here.
    A ``None`` slot means "no bound in that direction": ``-inf`` for min,
    ``+inf`` for max, which is scipy's own equivalent spelling.
    """
    if isinstance(bounds, types.Array):
        if bounds.ndim == 1:
            # a flat array has no pairs to transpose, so every row is a
            # number and `zip` refuses the first one
            not_iter = ("'" + _np_scalar_name(bounds.dtype)
                        + "' object is not iterable")

            def impl(bounds, n):
                if bounds.shape[0] == 0:        # minimize()'s "no bounds"
                    return np.zeros(0, np.float64), np.zeros(0, np.float64)
                raise TypeError(not_iter)
            return impl

        if bounds.ndim == 2:
            def impl(bounds, n):
                if bounds.shape[0] == 0:        # minimize()'s "no bounds"
                    return np.zeros(0, np.float64), np.zeros(0, np.float64)
                _check_pair_width(bounds.shape[1])
                m = bounds.shape[0]
                lo = np.empty(m, np.float64)
                hi = np.empty(m, np.float64)
                for i in range(m):
                    lo[i] = bounds[i, 0]
                    hi[i] = bounds[i, 1]
                return lo, hi
            return impl

        def impl(bounds, n):
            if bounds.shape[0] == 0:            # minimize()'s "no bounds"
                return np.zeros(0, np.float64), np.zeros(0, np.float64)
            _check_pair_width(bounds.shape[1])
            m = bounds.shape[0]
            lo = np.empty(m, np.float64)
            hi = np.empty(m, np.float64)
            for i in range(m):
                p = np.ravel(bounds[i, 0])
                q = np.ravel(bounds[i, 1])
                if p.size != 1 or q.size != 1:
                    raise ValueError(_ITEM_MSG)
                lo[i] = p[0]
                hi[i] = q[0]
            return lo, hi
        return impl

    uid = [0]

    def _emit(el, j, acc, target, ind):
        """Source lines writing element `j` of pair `acc` into `target`.

        Three slot types. A ``None`` slot is the open bound. A plain number
        is cast. An ``Optional`` slot is what a MIXED LIST LITERAL produces,
        numba unifying the ``None`` and the float into
        ``Optional(float64)``; it narrows on an ``is None`` BRANCH and not
        through a cast, so that case needs statements rather than an
        expression.
        """
        open_ = "-np.inf" if j == 0 else "np.inf"
        if isinstance(el[j], types.NoneType):
            return ["%s%s = %s" % (ind, target, open_)]
        if isinstance(el[j], types.Optional):
            uid[0] += 1
            v = "_v%d" % uid[0]
            return ["%s%s = %s[%d]" % (ind, v, acc, j),
                    "%sif %s is None:" % (ind, v),
                    "%s    %s = %s" % (ind, target, open_),
                    "%selse:" % ind,
                    "%s    %s = %s" % (ind, target, v)]
        return ["%s%s = np.float64(%s[%d])" % (ind, target, acc, j)]

    if isinstance(bounds, types.BaseTuple):
        els = list(bounds)
        ref = _pair_refusal(els)
        if ref is not None:
            return _refusing_impl(ref)
        if any(not isinstance(e, types.BaseTuple) for e in els):
            return _refusing_impl((ValueError, _BOUNDS_SHAPE_MSG))
        k = len(els)
        # unrolled: a heterogeneous tuple can only be indexed by a CONSTANT
        src = ["def impl(bounds, n):",
               "    lo = np.empty(%d, np.float64)" % k,
               "    hi = np.empty(%d, np.float64)" % k]
        for i, el in enumerate(els):
            src += _emit(el, 0, "bounds[%d]" % i, "lo[%d]" % i, "    ")
            src += _emit(el, 1, "bounds[%d]" % i, "hi[%d]" % i, "    ")
        src.append("    return lo, hi")
        ns = {'np': np, '_BOUNDS_LEN_MSG': _BOUNDS_LEN_MSG}
        exec("\n".join(src), ns)
        return ns['impl']

    dt = getattr(bounds, 'dtype', None)
    if dt is not None and isinstance(dt, types.BaseTuple):
        if len(dt) != 2:
            return _refusing_impl(_pair_refusal([dt]))
        # homogeneous list: one runtime loop, the None slots fixed at compile
        # time.  A MIXED list never reaches here -- numba refuses to unbox it.
        src = ["def impl(bounds, n):",
               "    m = len(bounds)",
               "    lo = np.empty(m, np.float64)",
               "    hi = np.empty(m, np.float64)",
               "    for i in range(m):"]
        src += _emit(dt, 0, "bounds[i]", "lo[i]", "        ")
        src += _emit(dt, 1, "bounds[i]", "hi[i]", "        ")
        src.append("    return lo, hi")
        ns = {'np': np, '_BOUNDS_LEN_MSG': _BOUNDS_LEN_MSG}
        exec("\n".join(src), ns)
        return ns['impl']

    return _refusing_impl((ValueError, _BOUNDS_SHAPE_MSG))


def fmin_l_bfgs_b(func, x0, fprime=None, args=(), approx_grad=0,
                  bounds=None, m=10, factr=1e7, pgtol=1e-5, epsilon=1e-8,
                  maxfun=15000, maxiter=15000, callback=None, maxls=20,
                  iprint=-1):
    """Minimize a function of many variables subject to simple bounds.

    L-BFGS-B stores a limited number of past gradient differences instead of a
    full Hessian approximation, which is what makes it usable at large `n`.

    Callable from Python and from inside ``@njit``. Both entries run the same
    compiled driver. One argument SPELLING is python-only, named under
    `Notes`.

    Parameters
    ----------
    func : callable
        A plain ``@njit`` function. Which protocol it must follow depends on
        the two arguments below: with ``fprime=None`` and ``approx_grad=0``
        it is ``f(x, *args) -> (value, gradient)``; with `fprime` given or
        ``approx_grad=1`` it is ``f(x, *args) -> value``.
    x0 : array_like
        Initial guess. Any rank is flattened, and it is clipped into `bounds`
        before the first evaluation.
    fprime : callable or None, optional
        ``jac(x, *args) -> gradient``. ``None`` or a plain ``@njit``
        function, decided at compile time.
    args : tuple or ndarray, optional
        Extra values for `func` and `fprime`. A tuple is unpacked into their
        argument lists, ``f(x, *args)``; the entries may be of any types the
        callbacks accept. ``()`` (default) calls ``f(x)``.
    approx_grad : bool, optional
        ``True`` computes the gradient by forward differences with absolute
        step `epsilon`. Compile-time constant.
    bounds : sequence of (min, max) pairs, (n, 2) ndarray, or None, optional
        ``None`` (default) is unconstrained. A sequence of pairs and an
        ``(n, 2)`` array both work, from Python and inside ``@njit``. ``None``
        inside a pair means "no bound in that direction"; ``-np.inf`` and
        ``np.inf`` are the equivalent spelling. Pairs may mix the two, and a
        list holding mixed pairs is written at the ``@njit`` call site rather
        than passed in as an argument; see `Notes`. The pairs are read by
        transposition, so the number of rows must be ``len(x0)`` and each row
        must hold exactly two entries.
    m : int, optional
        Number of stored gradient corrections. Default 10.
    factr : float, optional
        Convergence tolerance on `f`, in units of machine epsilon. Default
        1e7. The absolute tolerance is ``factr * np.finfo(float).eps``.
    pgtol : float, optional
        Convergence tolerance on the projected gradient. Default 1e-5.
    epsilon : float or ndarray, optional
        Absolute step for `approx_grad`. Default 1e-8. An array gives one
        step per coordinate; a size-1 array applies to every coordinate.
        Read only when `approx_grad` is set.
    maxfun, maxiter : int, optional
        Evaluation and iteration budgets. Both default to 15000.
    callback : callable, optional
        Called once per iteration, as ``callback(xk)`` or
        ``callback(intermediate_result)``. Two spellings are served: a plain
        Python callable, which halts the solve when it raises
        ``StopIteration``, and a numba ``@njit`` ``callback(xk)``, which halts
        when it raises any exception. See Notes.
    maxls : int, optional
        Maximum line-search steps per iteration. Default 20. Must be positive.
    iprint : int, optional
        Verbosity passed straight to the Fortran, whose output goes to the
        process stdout rather than through Python. Default ``-1``, silent.

    Returns
    -------
    x : ndarray, shape (n,)
        The minimizer.
    f : float
        `func` at `x`.
    d : LbfgsbInfo
        Namedtuple with fields ``grad``, ``task``, ``funcalls``, ``nit`` and
        ``warnflag``, reached as attributes or by position. ``warnflag`` is
        ``0`` converged, ``1`` a limit was reached and ``2`` anything else;
        ``grad`` and ``task`` are meaningful whatever it is.

    Raises
    ------
    ValueError
        If `callback` is a ``@cfunc``, a raw function address or a
        non-callable; if `bounds` has a length other than ``len(x0)``, or
        rows that are not pairs; if any lower bound exceeds its upper bound;
        if `maxls` is below 1; if `epsilon` does not broadcast against `x0`
        while `approx_grad` is set; or if `func` or `fprime` returns a
        gradient whose length is not ``len(x0)``.
    TypeError
        If a row of `bounds` is a number rather than a pair.
    numba.core.errors.TypingError
        Inside ``@njit``, where the refusal is decided by TYPE: a Python
        `callback`, a list of pairs that mixes ``None`` with a number, and
        an `approx_grad` or `fprime` whose value is not known when the call
        compiles.

    See Also
    --------
    scipy.optimize.fmin_l_bfgs_b : The scipy routine this mirrors.
    scijit.optimize.minimize : Dispatches here for ``method='L-BFGS-B'``.
    scijit.optimize.fmin_slsqp : Constrained minimization by SLSQP.
    scijit.optimize.fmin_bfgs : Unbounded BFGS, a pure ``@njit`` port.

    Notes
    -----
    `d` is a namedtuple carrying scipy's five field names in scipy's order.
    ``d['warnflag']``, ``d.warnflag`` and ``d[4]`` all reach the flag, and
    ``d.keys()``, ``d.values()``, ``d.items()``, ``d.get()``, ``'task' in d``
    and ``dict(d)`` all work. Unpacking yields the five VALUES, where
    unpacking scipy's dict yields the five field names.

    An ndarray or a list `args` reaches `func` and `fprime` as ONE argument,
    ``f(x, args)``, where scipy unpacks those two element by element. The
    arity of a compiled call is fixed when it compiles, so a sequence whose
    length is known only at run time cannot be unpacked.

    At ``len(x0) == 0`` the reported `funcalls` is 0 where scipy reports 1:
    scipy evaluates `func` once at `x0` before the solver runs and this
    reports the calls it made, which is none. `f` comes back as a float where
    scipy returns the 0-d array it started with.

    Whether `fprime` is ``None``, and the value of `approx_grad`, are
    compile-time constants inside ``@njit``: they select which callback
    protocol gets compiled, so neither can vary at run time.

    A Python `callback` is reached from compiled code through a module-level
    slot and a ``numba.objmode`` block, so it takes the GIL and pays an
    interpreter round trip once per iteration. It is also not ``prange``-safe,
    because the slot is module state that two concurrent solves share. Inside ``@njit`` only the
    ``@njit`` spelling is accepted, since a Python callable cannot cross into
    compiled code as an argument.

    The two spellings differ in what halts the solve. The Python one halts on
    ``StopIteration``, which is scipy's contract; anything else it raises
    reaches the caller. The ``@njit`` one halts on ANY exception, because
    numba matches no exception class: ``except StopIteration`` does not
    compile.

    A MIXED list of pairs such as ``[(None, 5.0), (0.0, 1.0)]`` is written at
    the ``@njit`` call site, not passed in as an argument. numba unifies the
    ``None`` and the float into an optional type for a literal, and refuses to
    unbox a heterogeneous list handed over as an argument:
    ``TypeError: can't unbox heterogeneous list``. The same pairs as a tuple
    cross either way, and so do a list whose pairs are uniform, an
    ``(n, 2)`` array, and the ``-np.inf`` / ``np.inf`` spelling.

    A gradient whose length is not ``len(x0)`` raises. scipy 1.18 raises at no
    length and no value. What it returns depends on the gradient it was
    handed: on a 2-variable problem with a length-1 gradient it reports
    ``warnflag`` 0 at the starting point for ``0.0`` and ``-1e-6``, and
    ``warnflag`` 2 with the first coordinate moved for ``-1.0`` and ``-2.0``.
    The coordinates the short gradient does not cover are not written,
    measured coming back as ``-1.177e-310``.

    `iprint` has no scipy counterpart, and sits last in the signature so that
    every position before it is scipy's. scipy 1.18 removed both ``iprint``
    and ``disp`` from this function and from the ``minimize`` options dict.

    Safe to call from a ``numba.prange`` loop with `callback` at ``None`` or
    an ``@njit`` function: the solver is reverse communication, so its state
    lives in caller-owned arrays. A Python `callback` is not, and serializes
    the loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import fmin_l_bfgs_b
    >>> @njit
    ... def fg(x):
    ...     f = (x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2
    ...     return f, np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.5)])
    >>> @njit
    ... def run():
    ...     return fmin_l_bfgs_b(fg, np.array([0.0, 0.0]))
    >>> x, f, d = run()
    >>> x
    array([1. , 2.5])
    >>> f
    0.0
    >>> d.warnflag, d.nit
    (0, 2)
    """
    cbf, use_cb, pycb = _cb_resolve('fmin_l_bfgs_b', callback)
    x = np.ascontiguousarray(np.atleast_1d(
        np.asarray(x0, dtype=np.float64))).ravel().copy()
    n = x.size
    # scipy's order: the bounds length, then the transposition into lower and
    # upper, then lb > ub, then the clip, and `maxls` after all four
    if bounds is not None:
        if len(bounds) != n:
            raise ValueError(_BOUNDS_LEN_MSG)
        lo, hi = _as_bounds(bounds)
    else:
        lo = hi = np.zeros(0)
    l, u, nbd = _lbfgsb_bounds(n, lo, hi)
    x = _clip_to_box(x, l, u, nbd)
    if maxls < 1:
        raise ValueError('maxls must be positive.')
    ea = np.asarray(epsilon, dtype=np.float64)
    # `epsilon` is read only on the forward-difference path, so nothing about
    # it is tested on the other two
    if approx_grad:
        _check_eps(ea.size, ea.ndim, n)
    epsilon = float(ea) if ea.ndim == 0 else np.ascontiguousarray(ea)

    if approx_grad:
        ev, jac = _ev_approx, func
    elif fprime is None:
        ev, jac = _ev_fg, func
    else:
        ev, jac = _ev_split, fprime
    prev = _cb_install(pycb)
    try:
        xo, f, g, conv, sr, nfev, nit, which, _sk, _yk = _lbfgsb_run(
            ev, func, jac, x, l, u, nbd, args, m, factr, pgtol, epsilon,
            maxfun,
            maxiter, iprint, maxls, cbf, use_cb)
    finally:
        _cb_release(prev)
    return xo, f, LbfgsbInfo(g, _lbfgsb_msg(conv, which, sr), nfev, nit,
                             _lbfgsb_flag(conv, sr))


@overload(fmin_l_bfgs_b, prefer_literal=True)
def _lbfgsb_ovl(func, x0, fprime=None, args=(), approx_grad=0, bounds=None,
                m=10, factr=1e7, pgtol=1e-5, epsilon=1e-8, maxfun=15000,
                maxiter=15000, callback=None, maxls=20, iprint=-1):
    """@njit implementation of `fmin_l_bfgs_b`, resolved at compile time.

    scipy accepts the objective in three shapes -- ``func`` alone under
    ``approx_grad``, one ``func`` returning ``(f, g)``, or ``func`` and
    ``fprime`` separately -- and each needs a different evaluator inside
    the single driver loop. Which one applies is a matter of types, so it
    is settled here and the evaluator travels into the body as a
    first-class function value. Returning ``None`` declines the call,
    which numba reports as a TypingError naming the argument that could
    not be served.
    """
    CBF, USE_CB = _cb_resolve_ty('fmin_l_bfgs_b', callback)
    ag = _lit_bool(approx_grad)
    if ag is None:
        return None                     # runtime flag -> TypingError
    no_jac = _is_none(fprime)
    no_bounds = _is_none(bounds)
    # A numba array TYPE carries rank, dtype and layout, never a length, so
    # neither the number of pairs nor the WIDTH of a pair is answerable here.
    # Both are run-time tests inside `_split_bounds`, which is why an
    # `(n, 3)` array cannot be turned away by this chooser.
    # `epsilon` is a float or an ndarray, and its LENGTH is another run-time
    # question a type cannot answer.  scipy reads it only under
    # `approx_grad`, so the check is gated on the same flag.
    eps_arr = isinstance(epsilon, types.Array)
    eps_check = ag and eps_arr
    if ag:
        ev = _ev_approx
        split = False
    elif no_jac:
        ev = _ev_fg
        split = False
    else:
        ev = _ev_split
        split = True

    def impl(func, x0, fprime=None, args=(), approx_grad=0, bounds=None,
             m=10, factr=1e7, pgtol=1e-5, epsilon=1e-8, maxfun=15000,
             maxiter=15000, callback=None, maxls=20, iprint=-1):
        x = np.ascontiguousarray(np.asarray(x0)).ravel().astype(np.float64)
        n = x.size
        if no_bounds:
            lo = np.zeros(0, np.float64)
            hi = np.zeros(0, np.float64)
        else:
            if len(bounds) != n:
                raise ValueError(_BOUNDS_LEN_MSG)
            lo, hi = _split_bounds(bounds, n)
        l, u, nbd = _lbfgsb_bounds(n, lo, hi)
        x = _clip_to_box(x, l, u, nbd)
        if maxls < 1:
            raise ValueError('maxls must be positive.')
        if eps_check:
            _check_eps(epsilon.size, epsilon.ndim, n)
        if split:
            jac = fprime
        else:
            jac = func
        xo, f, g, conv, sr, nfev, nit, which, _sk, _yk = _lbfgsb_run(
            ev, func, jac, x, l, u, nbd, args, m, factr, pgtol, epsilon,
            maxfun, maxiter, iprint, maxls, CBF, USE_CB)
        return xo, f, LbfgsbInfo(g, _lbfgsb_msg(conv, which, sr), nfev, nit,
                                 _lbfgsb_flag(conv, sr))
    return impl
