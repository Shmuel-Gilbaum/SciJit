"""Adaptive numerical integration in ``@njit``, over QUADPACK."""
import ctypes as ct
import os
import platform
import warnings

import numpy as np
from numba import carray, cfunc, njit, objmode, typeof, types
from numba.core.errors import TypingError
from numba.extending import overload

# the internal integrand signature (RETURNS the value). PRIVATE: no public
# routine accepts a @cfunc built against it.
_quadpack_sig = types.float64(types.float64,               # x     (value)
                              types.CPointer(types.float64))  # args (ptr)
"""numba signature of a QUADPACK integrand, ``float64(float64, float64*)``.
Private.

The shape of the ``@cfunc`` that :func:`quad`, :func:`~scijit.integrate.nquad`,
``dblquad`` and ``tplquad`` build for an integrand.  The callback RETURNS the
integrand value.  ``x`` arrives by value; ``args`` is a pointer.  There is no
``n`` argument: the integrand is scalar to scalar.

``args`` points into a WRITABLE buffer, and this package writes through it: the
``nquad`` chain stores each level's fixed coordinate with ``buf[hc + k] = xk``
and accumulates ``neval`` and the running ``abserr`` in the same buffer, which
is why every level of the nest calls one shared core rather than :func:`quad`.

Notes
-----
No public routine takes a callback of this signature, and no caller needs to
build one.  :func:`quad` takes a plain ``@njit`` ``f(x, *args)``, builds the
``@cfunc`` when the calling function compiles, and refuses an address with a
``TypeError`` naming the plain spelling.  ``nquad``, ``dblquad`` and
``tplquad`` take the plain function only, and a multi-dimensional integrand
receives its coordinates as separate arguments, which this signature has no
slot for.

scipy publishes no name for an integrand signature.  This is a low-level
spelling with no scipy counterpart.
"""

rootdir = os.path.dirname(os.path.abspath(__file__))

if platform.uname()[0] == "Windows":
    _name = "\\libquadpack.dll"
elif platform.uname()[0] == "Linux":
    _name = "/libquadpack.so"
else:
    _name = "/libquadpack.dylib"

_lib = ct.CDLL(rootdir + _name)


def _sig(fn, nargs):
    """Declare a bind(c) wrapper as ``nargs`` opaque pointers returning void.

    Every Fortran argument crosses by reference, so the ctypes view is
    uniform and every call site passes ``.ctypes.data``.  ``nargs`` must
    match the wrapper's argument list in ``src/quadpack/wrappers.f90``.  A
    miscount that disagrees with the call site raises; a miscount CONSISTENT
    with the call site raises nothing and runs into undefined behaviour, so
    recount against the Fortran rather than against a previous count.
    """
    fn.argtypes = [ct.c_void_p] * nargs
    fn.restype = None
    return fn


# ---------------------------------------------------------------------
# Hiding the @cfunc.  QUADPACK needs a C function pointer, so a route
# that reaches Fortran cannot take a first-class @njit function.  The
# adapter closes that gap: given a plain @njit ``f(x, *args)``, it builds
# the ``@cfunc(_quadpack_sig)`` once, at TYPING time, and freezes the
# address into the compiled body.  Every QUADPACK route is then reachable
# from a plain @njit integrand, including the infinite limits, the break
# points, the weights and ``full_output``, none of which the pure port
# can do.
#
# The cache OWNS the cfunc.  The address is baked into compiled code, so
# dropping the last Python reference would leave it dangling.
#
# The ABI carries one ``double *`` for the extra parameters, so scipy's
# ``args`` tuple is flattened into that buffer and unflattened inside the
# adapter.  Both halves are generated from the same kinds, which are
# known when the caller compiles: slot 0 holds the payload length, a
# scalar takes one slot, an array takes its shape then its data.
# ---------------------------------------------------------------------
_ADAPTERS = {}

#: Replay functions, keyed the way :func:`_adapter_quad` keys adapters.
_REPLAYS = {}


_ADDRESS_MSG = (
    "quad: func must be a plain @njit function, called as f(x, *args) with "
    "one argument per entry of args. The .address of a @cfunc is not "
    "accepted: pass the @njit function itself and quad builds the callback.")

_ARGS_MSG = (
    "quad: an args entry must be a real number or an array of real numbers. "
    "The extra parameters cross QUADPACK's double* argument slot, which "
    "carries no other type.")

_ARITY_MSG = (
    "quad: func takes %d argument(s), and an args of length %d calls it as "
    "f(x, *args), which needs %d.")

_FULL_OUTPUT_MSG = (
    "quad: full_output must be a compile-time constant inside @njit. It "
    "selects the number of return values, so a runtime flag cannot be typed.")

_COMPLEX_MSG = (
    "quad: complex_func must be a compile-time constant inside @njit. It "
    "selects the return type, so a runtime flag cannot be typed.")

_COMPLEX_FO_MSG = (
    "quad: complex_func with full_output is refused inside @njit. scipy "
    "returns the two diagnostics under a dict of dicts, and a dict holding "
    "a dict cannot be lowered. Call quad from python for that pair.")

_WEIGHT_MSG = (
    "quad: weight must be a string literal or None inside @njit. It selects "
    "which QUADPACK routine runs, so it has to be known when the call "
    "compiles.")

#: scipy's own text, from CPython's conversion of a complex return to the
#: double QUADPACK reads.  A numpy complex128 return reaches a different
#: path there, which warns per evaluation and integrates the real part; that
#: one is refused here too, deliberately.
_COMPLEX_REAL_MSG = "must be real number, not complex"

#: Reached only when the integrand raised during the integration and then
#: did not raise on the repeat, which is a non-deterministic integrand.
_INTEGRAND_MSG = (
    "quad: the integrand raised during the integration and returned "
    "normally when the same point was evaluated again.")


#: kind codes: a float, an integer and a boolean scalar; ``k >= 1`` is an
#: array of that rank.  A scalar keeps its own type through the buffer,
#: which is a cast in the adapter; an array's data crosses as float64.
_K_FLOAT, _K_INT, _K_BOOL = -1, -2, -3


def _arg_kinds(args):
    """Element kinds of a Python-level ``args`` tuple.

    The packer writes what the kinds say and the adapter reads it back, so
    the two halves cannot drift.
    """
    kinds = []
    for v in args:
        a = v if isinstance(v, np.ndarray) else np.asarray(v)
        if a.dtype.kind not in 'biuf':
            raise ValueError(_ARGS_MSG)
        if a.ndim:
            kinds.append(a.ndim)
        elif a.dtype.kind == 'b':
            kinds.append(_K_BOOL)
        elif a.dtype.kind == 'f':
            kinds.append(_K_FLOAT)
        else:
            kinds.append(_K_INT)
    return tuple(kinds)


def _arg_kinds_ty(args):
    """:func:`_arg_kinds` from the numba TYPE of ``args``, at typing time."""
    kinds = []
    for t in args:
        t = types.unliteral(t)
        if isinstance(t, types.Array):
            if not isinstance(t.dtype, (types.Integer, types.Float,
                                        types.Boolean)):
                raise TypingError(_ARGS_MSG)
            kinds.append(t.ndim)
        elif isinstance(t, types.Boolean):
            kinds.append(_K_BOOL)
        elif isinstance(t, types.Float):
            kinds.append(_K_FLOAT)
        elif isinstance(t, types.Integer):
            kinds.append(_K_INT)
        else:
            raise TypingError(_ARGS_MSG)
    return tuple(kinds)


def _args_types(args):
    """The element types of ``args`` as the chooser sees them.

    ``None`` and an omitted default are the empty tuple; a non-tuple is a
    one-item tuple, which is scipy's own coercion.
    """
    if isinstance(args, types.Omitted):
        args = args.value
    if isinstance(args, types.Type):
        if _is_none(args):
            return ()
        if isinstance(args, types.BaseTuple):
            return tuple(args)
        return (args,)
    # An OMITTED default reaches a chooser as the RAW PYTHON VALUE.
    return tuple(typeof(v) for v in _as_args_tuple(args))


def _args_is_tuple(args):
    """True when ``args`` is a tuple or ``None`` rather than a lone value."""
    if isinstance(args, types.Omitted):
        args = args.value
    if isinstance(args, types.Type):
        return _is_none(args) or isinstance(args, types.BaseTuple)
    return args is None or isinstance(args, tuple)


def _as_args_tuple(args):
    """scipy's coercion: a non-tuple ``args`` is a one-item tuple.

    ``None`` is the empty tuple here, which is what ``nquad`` and the two
    fixed-depth wrappers on it mean by it. :func:`quad` reads its own
    ``args=None`` as scipy's one-item tuple instead (row quad-D14).
    """
    if args is None:
        return ()
    if isinstance(args, tuple):
        return args
    return (args,)


def _integrand_sig(kinds):
    """The signature ``f(x, *args)`` is compiled for, from the kinds."""
    sig = [types.float64]
    for k in kinds:
        if k == _K_INT:
            sig.append(types.int64)
        elif k == _K_BOOL:
            sig.append(types.boolean)
        elif k == _K_FLOAT:
            sig.append(types.float64)
        else:
            sig.append(types.Array(types.float64, k, 'C'))
    return tuple(sig)


def _link_integrand(disp, kinds):
    """Compile a plain ``@njit`` integrand for its own signature, now.

    This is what makes ``quad`` NESTABLE, and the reason is a linking one
    rather than a numerical one.  This module reaches QUADPACK through
    ctypes, and a ctypes pointer is what numba calls a dynamic global.  When
    numba first compiles a function IN a first-class-function context, which
    is what an integrand is, it does not link that symbol, and the call dies
    at RUN time with "numba jitted function aborted due to unresolved
    symbol".  Compiling the integrand standalone first links it.

    So an integrand that itself calls ``quad`` used to fail, and one that
    happened to have been called once already worked.  The order dependence
    is the whole bug.

    A signature mismatch is not an error here: the integrand may legitimately
    take something else, and the normal typing path reports that with a
    message about the caller's function rather than about this one.
    """
    try:
        disp.compile(_integrand_sig(kinds))
    except Exception:                                    # noqa: BLE001
        pass


def _unpack_src(kinds, flip_x, neg, part, replay=False):
    """Source of the ``@cfunc`` body that unflattens ``args`` and calls in.

    ``flip_x`` calls the integrand at ``-x`` and ``neg`` negates what it
    returns, which is scipy's remapping for ``weight='cos'``/``'sin'`` with
    ``a = -inf``.  ``part`` takes ``.real`` or ``.imag`` of a complex
    integrand, which is scipy's ``complex_func``.

    ``replay`` generates a plain ``@njit`` body over the same buffer,
    reached only once the adapter has recorded a failure: it calls the
    integrand at the point that raised, from a frame where the exception
    can propagate.  See :func:`_replay_quad`.
    """
    if replay:
        lines = ["def replay(x, a):",
                 "    n = int(a[0])",
                 "    o = 1"]
    else:
        lines = ["def adapter(x, p):",
                 "    n = int(p[0])",
                 "    a = carray(p, n + 3)",
                 # Once the integrand has raised, every later evaluation is
                 # a wasted one: QUADPACK still has to converge before the
                 # entry point can raise, so it converges on a zero
                 # integrand.
                 "    if a[n + 1] != 0.0:",
                 "        return 0.0",
                 "    o = 1"]
    names = []
    for i, k in enumerate(kinds):
        if k < 0:
            cast = {_K_INT: "a.view(np.int64)[o]", _K_BOOL: "a[o] != 0.0",
                    _K_FLOAT: "a[o]"}[k]
            lines.append("    e%d = %s" % (i, cast))
            lines.append("    o += 1")
        else:
            dims = ", ".join("int(a[o + %d])" % j for j in range(k))
            lines.append("    d%d = (%s,)" % (i, dims))
            lines.append("    o += %d" % k)
            lines.append("    m%d = %s" % (
                i, " * ".join("d%d[%d]" % (i, j) for j in range(k))))
            if k == 1:
                lines.append("    e%d = a[o:o + m%d]" % (i, i))
            else:
                lines.append("    e%d = a[o:o + m%d].reshape(d%d)" % (i, i, i))
            lines.append("    o += m%d" % i)
        names.append("e%d" % i)
    if replay:
        lines.append("    return inner(%s)" % ", ".join(["x"] + names))
        return "\n".join(lines)
    xarg = "-x" if flip_x else "x"
    call = "inner(%s)" % ", ".join([xarg] + names)
    if part:
        call += "." + part
    # A `raise` inside a @cfunc is printed and swallowed, so an exception
    # from the integrand is recorded here, with the point it was raised at,
    # and the entry point repeats the call.
    lines += ["    _v = 0.0",
              "    try:",
              "        _v = %s%s" % ("-" if neg else "", call),
              "    except Exception:",
              "        a[n + 1] = -1.0",
              "        a[n + 2] = %s" % xarg,
              "        _v = 0.0",
              "    return _v"]
    return "\n".join(lines)


def _returns_complex(disp, kinds):
    """True when the integrand types complex for the call it will get."""
    cres = disp.overloads.get(_integrand_sig(kinds))
    return cres is not None and isinstance(cres.signature.return_type,
                                           types.Complex)


def _adapter_quad(py, kinds, flip_x=False, neg=False, part=''):
    """cfunc around a plain @njit ``f(x, *args) -> float``."""
    key = (py, kinds, flip_x, neg, part)
    hit = _ADAPTERS.get(key)
    if hit is not None:
        return hit
    inner = njit(py)
    _link_integrand(inner, kinds)
    if not part and _returns_complex(inner, kinds):
        raise TypeError(_COMPLEX_REAL_MSG)
    ns = {'carray': carray, 'inner': inner, 'np': np}
    exec(_unpack_src(kinds, flip_x, neg, part), ns)      # noqa: S102
    adapter = cfunc(_quadpack_sig)(ns['adapter'])
    _ADAPTERS[key] = adapter
    return adapter


def _replay_quad(py, kinds):
    """The integrand, read out of the packed buffer, where a raise carries.

    scipy propagates whatever the integrand raised.  A ``@cfunc`` cannot:
    the exception is printed and the callback returns to Fortran, so the
    adapter records the point instead and this repeats that one call from
    the entry point.  What reaches the caller is then the integrand's own
    exception, of its own class and with its own message, rather than a
    restatement.
    """
    key = (py, kinds)
    hit = _REPLAYS.get(key)
    if hit is not None:
        return hit
    inner = njit(py)
    _link_integrand(inner, kinds)
    ns = {'inner': inner, 'np': np}
    exec(_unpack_src(kinds, False, False, '', True), ns)  # noqa: S102
    fn = njit(ns['replay'])
    _REPLAYS[key] = fn
    return fn


def _check_arity(py, nargs):
    """scipy's calling convention, checked where the message can name it.

    scipy reaches ``func(x, *args)`` and lets Python report the mismatch.
    A ``@cfunc`` swallows the same report, so it is made here.
    """
    code = getattr(py, '__code__', None)
    if code is None or code.co_flags & 0x04:             # *args: no fixed arity
        return
    want = 1 + nargs
    ndef = len(getattr(py, '__defaults__', None) or ())
    if code.co_argcount - ndef <= want <= code.co_argcount:
        return
    raise TypeError(_ARITY_MSG % (code.co_argcount, nargs, want))


def _callback_pyfunc(func, kinds):
    """The plain Python function behind ``func``, with its arity checked."""
    if isinstance(func, (bool, int, np.integer)) or hasattr(func, 'address'):
        raise TypeError(_ADDRESS_MSG)
    py = getattr(func, 'py_func', func)
    if not callable(py):
        raise TypeError(_ADDRESS_MSG)
    _check_arity(py, len(kinds))
    return py


def _pack_args(args):
    """scipy's ``args``, flattened into the buffer the adapter unflattens.

    Two slots follow the payload and carry no argument: the integrand
    writes a flag into the first and the point it raised at into the
    second, which is what lets the entry point repeat that call.

    A scalar INTEGER entry crosses its slot as the int64 BITS, read back
    through the same view by the adapter, so a value above ``2**53`` is not
    rounded on the way.  The slot's kind is known per argument when the
    adapter is generated, so nothing has to be decided from the bits.
    """
    parts = []
    for v in _as_args_tuple(args):
        a = v if isinstance(v, np.ndarray) else np.asarray(v)
        if a.ndim == 0 and a.dtype.kind in 'iu':
            parts.append((_K_INT, a.astype(np.int64)))
            continue
        a = np.asarray(v, np.float64)
        if a.ndim == 0:
            parts.append((_K_FLOAT, a.reshape(1)))
        else:
            parts.append((a.ndim, (np.asarray(a.shape, np.float64),
                                   a.ravel())))
    tot = 0
    for k, p in parts:
        tot += 1 if k < 0 else k + p[1].size
    out = np.zeros(tot + 3, np.float64)
    out[0] = float(tot)
    o = 1
    for k, p in parts:
        if k == _K_INT:
            out.view(np.int64)[o] = p
            o += 1
        elif k < 0:
            out[o] = p[0]
            o += 1
        else:
            for q in p:
                out[o:o + q.size] = q
                o += q.size
    return out


@overload(_pack_args)
def _pack_args_ovl(args):
    kinds = _arg_kinds_ty(_args_types(args))
    one = not _args_is_tuple(args)
    lines = ["def impl(args):"]
    sizes = []
    for i, k in enumerate(kinds):
        src = "args" if one else "args[%d]" % i
        if k < 0:
            sizes.append("1")
        else:
            lines.append("    v%d = np.asarray(%s).astype(np.float64).ravel()"
                         % (i, src))
            lines.append("    s%d = %s.shape" % (i, src))
            sizes.append("%d + v%d.size" % (k, i))
    lines.append("    tot = %s" % (" + ".join(sizes) if sizes else "0"))
    lines.append("    out = np.zeros(tot + 3, np.float64)")
    lines.append("    out[0] = np.float64(tot)")
    lines.append("    o = 1")
    for i, k in enumerate(kinds):
        src = "args" if one else "args[%d]" % i
        if k == _K_INT:
            # the int64 BITS, so a value above 2**53 is not rounded
            lines.append("    out.view(np.int64)[o] = np.int64(%s)" % src)
            lines.append("    o += 1")
        elif k < 0:
            lines.append("    out[o] = np.float64(%s)" % src)
            lines.append("    o += 1")
        else:
            for j in range(k):
                lines.append("    out[o + %d] = np.float64(s%d[%d])"
                             % (j, i, j))
            lines.append("    o += %d" % k)
            lines.append("    out[o:o + v%d.size] = v%d" % (i, i))
            lines.append("    o += v%d.size" % i)
    lines.append("    return out")
    ns = {'np': np}
    exec("\n".join(lines), ns)                           # noqa: S102
    return ns['impl']


_dqags = _sig(_lib.dqags_wrapper, 15)
_dqagi = _sig(_lib.dqagi_wrapper, 15)
_dqagp = _sig(_lib.dqagp_wrapper, 18)
# 20, not 19: `dqawo_wrapper` gained a trailing `momcom` out-argument, which
# is scipy's one infodict entry on this route that no other output carries.
_dqawo = _sig(_lib.dqawo_wrapper, 20)
_dqawf = _sig(_lib.dqawf_wrapper, 18)
_dqaws = _sig(_lib.dqaws_wrapper, 19)
_dqawc = _sig(_lib.dqawc_wrapper, 17)


# ---------------------------------------------------------------------
# routing codes, resolved from scipy's `weight` string at compile time
# ---------------------------------------------------------------------
W_NONE = 0
W_COS = 1
W_SIN = 2
W_ALG = 3
W_ALG_LOGA = 4
W_ALG_LOGB = 5
W_ALG_LOG = 6
W_CAUCHY = 7

_WEIGHT_CODES = {'cos': W_COS, 'sin': W_SIN, 'alg': W_ALG,
                 'alg-loga': W_ALG_LOGA, 'alg-logb': W_ALG_LOGB,
                 'alg-log': W_ALG_LOG, 'cauchy': W_CAUCHY}


class IntegrationWarning(UserWarning):
    """Warning category emitted on a soft failure during integration.

    Raised at two conditions: a soft QUADPACK failure (``ier`` in
    1, 2, 3, 4, 5, 7) with ``full_output`` false, and ``points`` given
    together with ``weight``.  The message carries the ``limit`` substituted
    into the ``ier = 1`` text, and separate wording for ``ier`` 1, 4 and 7
    on the cos/sin route to infinity, where a subdivision is a cycle.

    See Also
    --------
    scipy.integrate.IntegrationWarning : The scipy category this mirrors.

    Notes
    -----
    This is a DIFFERENT CLASS OBJECT from ``scipy.integrate.
    IntegrationWarning``, with the same name and the same base.  A filter
    naming this class selects it, and so does one naming ``UserWarning`` or a
    bare ``warnings.catch_warnings(record=True)``.  A filter naming scipy's
    class does NOT: ``simplefilter('error',
    scipy.integrate.IntegrationWarning)`` leaves this warning as a warning,
    and ``except scijit.integrate.IntegrationWarning`` does not catch a scipy
    one.  Ported code that turns scipy's class into an exception names this
    one as well.

    Examples
    --------
    The warning is emitted by :func:`quad`; the filter that catches it is
    Python-level, since ``warnings.catch_warnings`` does not run inside
    ``@njit``:

    >>> import numpy as np
    >>> import warnings
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def f(x):
    ...     return np.cos(x)
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter('always')
    ...     v, e = si.quad(f, 0.0, 1.0, (), False, 1.49e-8, 1.49e-8, 50,
    ...                    np.array([0.5]), 'cos', 1.0)
    >>> issubclass(w[0].category, si.IntegrationWarning)
    True
    """


def _route_name(wcode, use_points, a, b):
    """Which QUADPACK routine a call reaches, as scipy names its key sets."""
    if wcode == W_COS or wcode == W_SIN:
        if a == np.inf or a == -np.inf or b == np.inf or b == -np.inf:
            return 'qawfe'
        return 'qawoe'
    if wcode == W_NONE and use_points and not (
            a == np.inf or a == -np.inf or b == np.inf or b == -np.inf):
        return 'qagpe'
    return 'std'


# --------------------------------------------------------------------------
# infodict
#
# EVERY VALUE IS A 1-D float64 ARRAY, in both entry points.  A numba dict
# carries ONE value type, and a dict whose values do not unify is a
# `LiteralStrKey[Dict]`, which cannot vary its key set at run time and
# cannot cross back to the interpreter.  scipy's key set DOES vary at run
# time: an empty interval returns the unweighted set whatever the route,
# and the cos/sin route splits on `b` being infinite.  Measured on numba
# 0.66: a run-time branch between two `LiteralStrKey[Dict]`s does not
# error, it keeps the first arm's keys and fails later at the read.  So the
# key set and the value type cannot both be scipy's, and the key set is the
# one that varies with the call.
#
# `alist`, `blist`, `rlist` and `elist` are float64 arrays already, so the
# common type is a float64 array: an int scalar becomes a length-1 array
# and an int32 array becomes a float64 one.  Recorded under `Notes` on
# `quad`, with the measurement.
# --------------------------------------------------------------------------
def _sc_py(v):
    """A scalar as the length-1 float64 array the dict holds."""
    return np.array([float(v)], dtype=np.float64)


def _infodict(r, route):
    """scipy's ``infodict``: its key set per route, plus ``ier``.

    ``ier`` is additive.  scipy signals a soft failure by appending a
    message to the RETURN, and a compiled body has one return shape, so the
    exit code stays reachable under a key no scipy-shaped call reads.
    """
    if route == 'qawfe':
        d = {'neval': _sc_py(r[2]), 'lst': _sc_py(r[4]), 'rslst': r[7],
             'erlst': r[8], 'ierlst': r[9].astype(np.float64)}
    else:
        d = {'neval': _sc_py(r[2]), 'last': _sc_py(r[4]),
             'iord': r[9].astype(np.float64),
             'alist': r[5], 'blist': r[6], 'rlist': r[7], 'elist': r[8]}
        if route == 'qagpe':
            d['pts'] = r[10]
            d['level'] = r[11].astype(np.float64)
            d['ndin'] = r[12].astype(np.float64)
        elif route == 'qawoe':
            d['nnlog'] = r[13].astype(np.float64)
            d['chebmo'] = np.ascontiguousarray(r[14]).ravel()
            d['momcom'] = _sc_py(r[15])
    d['ier'] = _sc_py(r[3])
    return d


@njit
def _sc(v):
    """:func:`_sc_py`, compiled."""
    out = np.empty(1, np.float64)
    out[0] = np.float64(v)
    return out


@njit
def _f64(a):
    """An int32 diagnostic array as the float64 one the dict holds."""
    return a.astype(np.float64)


@njit
def _info_base(r):
    """The unweighted key set, which is also every route's at ``a == b``."""
    return {'neval': _sc(r[2]), 'last': _sc(r[4]), 'iord': _f64(r[9]),
            'alist': r[5], 'blist': r[6], 'rlist': r[7], 'elist': r[8],
            'ier': _sc(r[3])}


@njit
def _info_std(r, empty, cycle):
    return _info_base(r)


@njit
def _info_pts(r, empty, cycle):
    d = _info_base(r)
    if not empty:
        d['pts'] = r[10]
        d['level'] = _f64(r[11])
        d['ndin'] = _f64(r[12])
    return d


@njit
def _info_osc(r, empty, cycle):
    d = _info_base(r)
    if not empty:
        d['nnlog'] = _f64(r[13])
        d['chebmo'] = r[14].ravel()
        d['momcom'] = _sc(r[15])
    return d


@njit
def _info_fourier(r, empty, cycle):
    if empty:
        return _info_base(r)
    return {'neval': _sc(r[2]), 'lst': _sc(r[4]), 'rslst': r[7],
            'erlst': r[8], 'ierlst': _f64(r[9]), 'ier': _sc(r[3])}


@njit
def _info_cos(r, empty, cycle):
    """The cos/sin key set, chosen at RUN time.

    Which of ``dqawoe`` and ``dqawfe`` runs depends on an endpoint being
    infinite, and whether the interval is empty is a value question too, so
    all three key sets are reachable from one compiled body.
    """
    if empty:
        return _info_base(r)
    if cycle:
        return _info_fourier(r, False, True)
    return _info_osc(r, False, False)


_INFO_NJIT = {'std': _info_std, 'qagpe': _info_pts, 'qawoe': _info_osc,
              'qawfe': _info_fourier, 'cos': _info_cos}

_IER_MSG = {
    1: "The maximum number of subdivisions (%d) has been achieved.\n  If "
       "increasing the limit yields no improvement it is advised to analyze "
       "\n  the integrand in order to determine the difficulties.  If the "
       "position of a \n  local difficulty can be determined (singularity, "
       "discontinuity) one will \n  probably gain from splitting up the "
       "interval and calling the integrator \n  on the subranges.  Perhaps a "
       "special-purpose integrator should be used.",
    2: "The occurrence of roundoff error is detected, which prevents \n  the "
       "requested tolerance from being achieved.  The error may be \n  "
       "underestimated.",
    3: "Extremely bad integrand behavior occurs at some points of the\n  "
       "integration interval.",
    4: "The algorithm does not converge.  Roundoff error is detected\n  in "
       "the extrapolation table.  It is assumed that the requested tolerance\n"
       "  cannot be achieved, and that the returned result (if full_output = "
       "1) is \n  the best which can be obtained.",
    5: "The integral is probably divergent, or slowly convergent.",
    7: "Abnormal termination of the routine.  The estimates for result\n  and "
       "error are less reliable.  It is assumed that the requested accuracy\n  "
       "has not been achieved.",
}


_IER_MSG_CYCLE = {
    1: "The maximum number of cycles allowed has been achieved., e.e.\n  of "
       "subintervals (a+(k-1)c, a+kc) where c = (2*int(abs(omega)+1))\n  "
       "*pi/abs(omega), for k = 1, 2, ..., lst.  One can allow more cycles by "
       "increasing the value of limlst.  Look at info['ierlst'] with "
       "full_output=1.",
    4: "The extrapolation table constructed for convergence acceleration\n  of "
       "the series formed by the integral contributions over the cycles, \n  "
       "does not converge to within the requested accuracy.  Look at \n  "
       "info['ierlst'] with full_output=1.",
    7: "Bad integrand behavior occurs within one or more of the cycles.\n  "
       "Location and type of the difficulty involved can be determined from "
       "\n  the vector info['ierlist'] obtained with full_output=1.",
}


def _emit_ier_warning(ier, limit, stacklevel=2, cycle=False):
    """scipy's IntegrationWarning, same text and same category.

    ``stacklevel`` counts from here, so the frame it lands on differs
    between the two entry points: the compiled one reaches this through an
    objmode block whose caller IS the user's frame at 2, and the Python one
    has `quad` and `_quad_slice` in between.
    """
    if cycle and ier in _IER_MSG_CYCLE:
        # scipy substitutes three of the six on the cos/sin route to
        # infinity, where a "subdivision" is a cycle
        msg = _IER_MSG_CYCLE[ier]
    else:
        msg = _IER_MSG.get(ier, "Unknown error.")
        if ier == 1:
            msg = msg % limit
    warnings.warn(msg, IntegrationWarning, stacklevel=stacklevel)


def _emit_points_ignored(stacklevel=2):
    """scipy's warning for ``points`` given together with ``weight``, same
    text and same category.  Python side; :func:`_warn_points_ignored` is the
    ``@njit`` entry that reaches it."""
    warnings.warn("Break points cannot be specified when using weighted "
                  "integrand.\nContinuing, ignoring specified points.",
                  IntegrationWarning, stacklevel=stacklevel)


@njit
def _warn_ier(ier, limit, cycle=False):
    """Warn exactly where scipy warns: a soft failure with
    ``full_output=0``.  An objmode block takes the GIL, so this is only
    ever reached on an error path.

    ``cycle`` picks scipy's other wording for ``ier`` 1, 4 and 7, which it
    uses on the cos/sin route to infinity.
    """
    if ier == 1 or ier == 2 or ier == 3 or ier == 4 or ier == 5 or ier == 7:
        with objmode():
            _emit_ier_warning(ier, limit, 2, cycle)


@njit
def _warn_points_ignored():
    """Raise the ignored-break-points warning from inside compiled code.

    An objmode block takes the GIL, so this is only ever reached on the one
    path where scipy also warns, never per integrand evaluation.
    """
    with objmode():
        _emit_points_ignored()


# ---------------------------------------------------------------------
# THE core.  Every entry point below only slices what this returns.
# ---------------------------------------------------------------------
@njit
def _quad_core(funcptr, a, b, args, epsabs, epsrel, limit, points,
               use_points, wcode, w0, w1, maxp1, limlst, refptr=0):
    """Every QUADPACK route scipy's ``quad`` can reach.

    Returns ``(val, abserr, neval, ier, last, alist, blist, rlist, elist,
    iord, pts, level, ndin, nnlog, chebmo, momcom)``.  The five arrays after
    ``last`` are length ``limit``; only ``[:last]`` is meaningful, exactly as
    in scipy.  The five after those are route-specific and empty elsewhere,
    and ``momcom`` is written only by the finite cos/sin route.

    ``refptr`` is the address of the same integrand read at ``-x``, negated
    for ``'sin'``, which is the only thing scipy's ``a = -inf`` cos/sin
    route needs that the forward callback cannot supply.  A caller that
    leaves it at 0 gets a ``ValueError`` on that one route.
    """
    # Nothing is validated here.  scipy validates none of it either: an
    # unusable `limit`, `maxp1` or `limlst` reaches QUADPACK, comes back as
    # ier = 6, and the forensic tree at the bottom of this function turns
    # that into scipy's message.  So `maxp1 = 0` integrates normally on a
    # route that does not read it, and the empty-interval shortcut below
    # is reachable at any `limit`, both as in scipy.
    nlim = limit if limit > 0 else 0
    alist = np.full(nlim, np.nan)
    blist = np.full(nlim, np.nan)
    rlist = np.zeros(nlim)
    elist = np.zeros(nlim)
    iord = np.zeros(nlim, np.int32)
    # route-specific diagnostics: the break-point route's pts/level/ndin
    # and the oscillatory route's nnlog/chebmo, empty on every other route
    pts_out = np.zeros(0)
    level = np.zeros(0, np.int32)
    ndin = np.zeros(0, np.int32)
    nnlog = np.zeros(0, np.int32)
    chebmo = np.zeros((25, 0))
    # the oscillatory route's chebyshev-moment count.  QUADPACK writes it
    # only on that route, and scipy reports it only there.
    momcom = np.zeros(1, np.int32)
    # QUADPACK's iord is 1-based and scipy reports it 0-based, so the
    # first `last` entries are shifted here too.  The TAIL is where the
    # two part company, deliberately: scipy allocates every diagnostic
    # array with PyArray_SimpleNew and QUADPACK writes only `[:last]`, so
    # scipy's tail is whatever the allocator returned.  Measured against
    # scipy 1.18 on the seven routes: `[:last]` identical, tails differing
    # by up to 2.05e+09 in `iord` and 6.95e-310 in `blist`.  This one is
    # zeroed rather than left uninitialised.
    # The dqawf route puts per-cycle STATUS CODES here instead, which are
    # not indices and are not shifted.
    sub_route = True

    # scipy: shortcut for an empty interval, before the a/b normalisation
    if a == b:
        return (0.0, 0.0, 0, 0, 0, alist, blist, rlist, elist, iord,
                pts_out, level, ndin, nnlog, chebmo, momcom[0])

    flip = b < a
    lo = min(a, b)
    hi = max(a, b)

    inf_lo = np.isinf(lo) and lo < 0.0
    inf_hi = np.isinf(hi) and hi > 0.0
    if inf_hi and not inf_lo:
        infb = 1
        bound = lo
    elif inf_hi and inf_lo:
        infb = 2
        bound = 0.0
    elif inf_lo and not inf_hi:
        infb = -1
        bound = hi
    else:
        infb = 0
        bound = 0.0

    ea = np.array(epsabs, np.float64)
    er = np.array(epsrel, np.float64)
    result = np.zeros(1, np.float64)
    abserr = np.zeros(1, np.float64)
    neval = np.zeros(1, np.int32)
    ier = np.zeros(1, np.int32)
    last = np.zeros(1, np.int32)
    args_ = np.ascontiguousarray(np.asarray(args, np.float64))
    na = np.array(args_.size, np.int32)
    lim = np.array(limit, np.int32)

    if wcode == W_NONE and use_points and infb == 0:
        # ---- dqagp: user break points -------------------------------
        # scipy preprocesses: np.unique (sort + dedupe), then a STRICT
        # lo < p < hi filter, so nan/inf and the endpoints drop out.
        m = points.shape[0]
        srt = np.sort(np.ascontiguousarray(np.asarray(points, np.float64)))
        tmp = np.empty(m + 2)
        k = 0
        for i in range(m):
            v = srt[i]
            if not (v > lo and v < hi):
                continue
            if k > 0 and v == tmp[k - 1]:
                continue
            tmp[k] = v
            k += 1
        if k >= limit:
            # QUADPACK's own condition (`limit <= npts` inside dqagpe), and
            # the driver would silently widen `limit` here rather than
            # report it, so the code is set by hand and the tree below
            # writes scipy's message.
            ier[0] = 6
        else:
            npts2 = k + 2
            pts = np.zeros(npts2, np.float64)
            for i in range(k):
                pts[i] = tmp[i]

            a_ = np.array(lo, np.float64)
            b_ = np.array(hi, np.float64)
            n2 = np.array(npts2, np.int32)
            leniw = max(2 * limit + npts2, 3 * npts2 - 2)
            lenw = 2 * leniw - npts2
            liw = np.array(leniw, np.int32)
            lw = np.array(lenw, np.int32)
            iwork = np.zeros(leniw, np.int32)
            work = np.zeros(lenw, np.float64)
            _dqagp(funcptr, a_.ctypes.data, b_.ctypes.data, n2.ctypes.data,
                   pts.ctypes.data, ea.ctypes.data, er.ctypes.data,
                   result.ctypes.data, abserr.ctypes.data, neval.ctypes.data,
                   ier.ctypes.data, liw.ctypes.data, lw.ctypes.data,
                   iwork.ctypes.data, work.ctypes.data,
                   args_.ctypes.data, na.ctypes.data, last.ctypes.data)
            le = (leniw - npts2) // 2
            for i in range(min(limit, le)):
                alist[i] = work[i]
                blist[i] = work[le + i]
                rlist[i] = work[2 * le + i]
                elist[i] = work[3 * le + i]
                iord[i] = iwork[i]
            pts_out = work[4 * le:4 * le + npts2].copy()
            level = iwork[le:le + min(limit, le)].copy()
            ndin = iwork[2 * le:2 * le + npts2].copy()

    elif wcode == W_NONE and infb == 0:
        # ---- dqags: finite interval ---------------------------------
        a_ = np.array(lo, np.float64)
        b_ = np.array(hi, np.float64)
        iwork = np.zeros(nlim, np.int32)
        work = np.zeros(4 * nlim, np.float64)
        _dqags(funcptr, a_.ctypes.data, b_.ctypes.data, ea.ctypes.data,
               er.ctypes.data, result.ctypes.data, abserr.ctypes.data,
               neval.ctypes.data, ier.ctypes.data, lim.ctypes.data,
               iwork.ctypes.data, work.ctypes.data,
               args_.ctypes.data, na.ctypes.data, last.ctypes.data)
        for i in range(limit):
            alist[i] = work[i]
            blist[i] = work[limit + i]
            rlist[i] = work[2 * limit + i]
            elist[i] = work[3 * limit + i]
            iord[i] = iwork[i]

    elif wcode == W_NONE:
        # ---- dqagi: one or both limits infinite ---------------------
        if use_points:
            raise ValueError("Infinity inputs cannot be used with break "
                             "points.")
        bd = np.array(bound, np.float64)
        inf_ = np.array(infb, np.int32)
        iwork = np.zeros(nlim, np.int32)
        work = np.zeros(4 * nlim, np.float64)
        _dqagi(funcptr, bd.ctypes.data, inf_.ctypes.data, ea.ctypes.data,
               er.ctypes.data, result.ctypes.data, abserr.ctypes.data,
               neval.ctypes.data, ier.ctypes.data, lim.ctypes.data,
               iwork.ctypes.data, work.ctypes.data,
               args_.ctypes.data, na.ctypes.data, last.ctypes.data)
        for i in range(limit):
            alist[i] = work[i]
            blist[i] = work[limit + i]
            rlist[i] = work[2 * limit + i]
            elist[i] = work[3 * limit + i]
            iord[i] = iwork[i]

    elif wcode == W_COS or wcode == W_SIN:
        integr = np.array(1 if wcode == W_COS else 2, np.int32)
        om = np.array(w0, np.float64)
        mp = np.array(maxp1, np.int32)
        if infb == 0:
            # ---- dqawo: oscillatory weight, finite interval ----------
            a_ = np.array(lo, np.float64)
            b_ = np.array(hi, np.float64)
            leniw = 2 * nlim
            lenw = 2 * leniw + 25 * (maxp1 if maxp1 > 0 else 0)
            liw = np.array(leniw, np.int32)
            lw = np.array(lenw, np.int32)
            iwork = np.zeros(leniw, np.int32)
            work = np.zeros(lenw, np.float64)
            _dqawo(funcptr, a_.ctypes.data, b_.ctypes.data, om.ctypes.data,
                   integr.ctypes.data, ea.ctypes.data, er.ctypes.data,
                   result.ctypes.data, abserr.ctypes.data, neval.ctypes.data,
                   ier.ctypes.data, liw.ctypes.data, mp.ctypes.data,
                   lw.ctypes.data, iwork.ctypes.data, work.ctypes.data,
                   args_.ctypes.data, na.ctypes.data, last.ctypes.data,
                   momcom.ctypes.data)
            for i in range(limit):
                alist[i] = work[i]
                blist[i] = work[limit + i]
                rlist[i] = work[2 * limit + i]
                elist[i] = work[3 * limit + i]
                iord[i] = iwork[i]
            nnlog = iwork[limit:2 * limit].copy()
            chebmo = np.ascontiguousarray(
                work[4 * limit:4 * limit + 25 * maxp1].reshape(
                    25, maxp1 if maxp1 > 0 else 0))
        elif infb == 2:
            raise ValueError("Cannot integrate with this weight from -Inf "
                             "to +Inf.")
        else:
            # ---- dqawf: Fourier integral over (a, +inf) -------------
            # infb == -1 is scipy's remap: the same routine over
            # (-b, +inf) on f(-x), negated for 'sin'.  The reflected
            # callback is a second adapter built at typing time.
            if infb == -1:
                if refptr == 0:
                    raise ValueError(
                        "quad: weight 'cos'/'sin' with a = -inf reads the "
                        "integrand at -x, and this caller supplied no "
                        "reflected callback.")
                fptr = refptr
                alow = -hi
            else:
                fptr = funcptr
                alow = lo
            a_ = np.array(alow, np.float64)
            lst = np.array(limlst, np.int32)
            leniw = 2 * nlim + (limlst if limlst > 0 else 0)
            lenw = 2 * leniw + 25 * (maxp1 if maxp1 > 0 else 0)
            liw = np.array(leniw, np.int32)
            lw = np.array(lenw, np.int32)
            iwork = np.zeros(leniw, np.int32)
            work = np.zeros(lenw, np.float64)
            _dqawf(fptr, a_.ctypes.data, om.ctypes.data,
                   integr.ctypes.data, ea.ctypes.data, result.ctypes.data,
                   abserr.ctypes.data, neval.ctypes.data, ier.ctypes.data,
                   lst.ctypes.data, liw.ctypes.data, mp.ctypes.data,
                   lw.ctypes.data, iwork.ctypes.data, work.ctypes.data,
                   args_.ctypes.data, na.ctypes.data, last.ctypes.data)
            # dqawf has no subintervals.  It writes one entry per CYCLE and
            # there are `limlst` of those, so the three fields that carry
            # them are `limlst` long here rather than `limit` long, which
            # is the length scipy returns them at.
            sub_route = False
            nlst = limlst if limlst > 0 else 0
            rlist = np.zeros(nlst)
            elist = np.zeros(nlst)
            iord = np.zeros(nlst, np.int32)
            for i in range(nlst):
                rlist[i] = work[i]                  # rslst
                elist[i] = work[limlst + i]         # erlst
                iord[i] = iwork[i]                  # ierlst

    elif wcode == W_CAUCHY:
        # ---- dqawc: Cauchy principal value --------------------------
        if infb != 0:
            raise ValueError("Cannot integrate with this weight over an "
                             "infinite interval.")
        a_ = np.array(lo, np.float64)
        b_ = np.array(hi, np.float64)
        c_ = np.array(w0, np.float64)
        lenw = 4 * nlim
        lw = np.array(lenw, np.int32)
        iwork = np.zeros(nlim, np.int32)
        work = np.zeros(lenw, np.float64)
        _dqawc(funcptr, a_.ctypes.data, b_.ctypes.data, c_.ctypes.data,
               ea.ctypes.data, er.ctypes.data, result.ctypes.data,
               abserr.ctypes.data, neval.ctypes.data, ier.ctypes.data,
               lim.ctypes.data, lw.ctypes.data, iwork.ctypes.data,
               work.ctypes.data, args_.ctypes.data, na.ctypes.data,
               last.ctypes.data)
        for i in range(limit):
            alist[i] = work[i]
            blist[i] = work[limit + i]
            rlist[i] = work[2 * limit + i]
            elist[i] = work[3 * limit + i]
            iord[i] = iwork[i]

    else:
        # ---- dqaws: algebraico-logarithmic end-point singularities ---
        if infb != 0:
            raise ValueError("Cannot integrate with this weight over an "
                             "infinite interval.")
        integr = np.array(wcode - W_ALG + 1, np.int32)
        a_ = np.array(lo, np.float64)
        b_ = np.array(hi, np.float64)
        al = np.array(w0, np.float64)
        be = np.array(w1, np.float64)
        lenw = 4 * nlim
        lw = np.array(lenw, np.int32)
        iwork = np.zeros(nlim, np.int32)
        work = np.zeros(lenw, np.float64)
        _dqaws(funcptr, a_.ctypes.data, b_.ctypes.data, al.ctypes.data,
               be.ctypes.data, integr.ctypes.data, ea.ctypes.data,
               er.ctypes.data, result.ctypes.data, abserr.ctypes.data,
               neval.ctypes.data, ier.ctypes.data, lim.ctypes.data,
               lw.ctypes.data, iwork.ctypes.data, work.ctypes.data,
               args_.ctypes.data, na.ctypes.data, last.ctypes.data)
        for i in range(limit):
            alist[i] = work[i]
            blist[i] = work[limit + i]
            rlist[i] = work[2 * limit + i]
            elist[i] = work[3 * limit + i]
            iord[i] = iwork[i]

    if sub_route:
        nl = last[0]
        for i in range(limit):
            iord[i] = iord[i] - 1 if i < nl else 0

    val = result[0]
    if flip:
        val = -val
    if ier[0] == 6:
        # scipy's forensic tree for ier = 6, branch for branch, including
        # the branches it cannot reach: the message for an `alg` alpha or
        # beta at exactly -1 is scipy's fall-through "The input is
        # invalid.", because its own test is `min(wvar) < -1`.
        # Every branch raises where it stands: an exception message held in
        # a VARIABLE that two branches assign is lowered to the FIRST
        # assignment, silently, so a tree written that way reports one
        # message for every cell.
        if epsabs <= 0.0:
            if epsrel < max(50.0 * 2.220446049250313e-16, 5e-29):
                raise ValueError("If 'epsabs'<=0, 'epsrel' must be greater "
                                 "than both 5e-29 and 50*(machine epsilon).")
            if ((wcode == W_COS or wcode == W_SIN)
                    and (abs(a) + abs(b)) == np.inf):
                raise ValueError("Sine or cosine weighted integrals with "
                                 "infinite domain must have 'epsabs'>0.")
        elif wcode == W_NONE:
            if not use_points:
                raise ValueError("Invalid 'limit' argument. There must be "
                                 "at least one subinterval")
            if points.size > 0 and not (lo <= points.min()
                                        and points.max() <= hi):
                raise ValueError("All break points in 'points' must lie "
                                 "within the integration limits.")
            if points.size >= limit:
                # scipy prints the RAW count, not the filtered one its
                # condition is really about.
                raise ValueError("Number of break points (" +
                                 str(points.size) +
                                 ") must be less than subinterval limit (" +
                                 str(limit) + ")")
        else:
            if maxp1 < 1:
                raise ValueError("Chebyshev moment limit maxp1 must be >=1.")
            if ((wcode == W_COS or wcode == W_SIN)
                    and abs(a + b) == np.inf):
                raise ValueError("Cycle limit limlst must be >=3.")
            if wcode == W_CAUCHY:
                if w0 == lo or w0 == hi:
                    raise ValueError("Parameter 'wvar' must not equal "
                                     "integration limits 'a' or 'b'.")
            elif min(w0, w1) < -1.0:
                raise ValueError("wvar parameters (alpha, beta) must both "
                                 "be >= -1.")
        raise ValueError("The input is invalid.")
    return (val, abserr[0], neval[0], ier[0], last[0],
            alist, blist, rlist, elist, iord,
            pts_out, level, ndin, nnlog, chebmo, momcom[0])


# ---------------------------------------------------------------------
# compile-time predicates and argument resolution
# ---------------------------------------------------------------------
def _lit_bool(v):
    """The value of a boolean-ish argument at TYPING time, or ``None`` when it
    is a runtime variable.

    ``None`` means no compiled body can be chosen, because the flag selects
    the number of return values.  The overload then declines and numba raises
    a ``TypingError``, which is the intended outcome rather than a failure.
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


def _lit_str(v):
    """The value of a string argument at TYPING time, or ``None``.

    ``weight`` selects which QUADPACK routine runs, so it has to be known
    when the call compiles.
    """
    if isinstance(v, str):
        return v
    if isinstance(v, types.StringLiteral):
        return v.literal_value
    if isinstance(v, types.Omitted):
        return v.value
    return None


def _is_none(v):
    """True when an argument is ``None`` at TYPING time, in any of the three
    spellings numba hands an overload: the Python value, ``types.NoneType``,
    or an ``Omitted`` default."""
    return (v is None or isinstance(v, types.NoneType)
            or (isinstance(v, types.Omitted) and v.value is None))


def _weight_code(weight):
    """scipy's `weight` string, as an internal routing code.

    Strings only, as in scipy.  The codes themselves are private.
    """
    if weight is None:
        return W_NONE
    code = _WEIGHT_CODES.get(weight) if isinstance(weight, str) else None
    if code is None:
        raise ValueError("%s not a recognized weighting function." % (weight,))
    return code


_WVAR_SCALAR_MSG = "must be real number, not %s"
_WVAR_NOT_SEQ_MSG = "argument 4 must be 2-item sequence, not %s"
_WVAR_LEN_MSG = "argument 4 must be sequence of length 2, not %d"
_WVAR_0D_ARRAY_MSG = ("only 0-dimensional arrays can be converted to Python "
                      "scalars")


def _wvar_pair(wvar, wcode):
    """``wvar`` as the two floats every weighted route takes.

    One slot for ``omega`` or the Cauchy ``c``, two for the ``'alg'``
    family's ``(alpha, beta)``.  A scalar fills the first and leaves the
    second at 0.0, so one core signature serves every weight.

    The ``'alg'`` family takes a 2-item sequence and every other weight a
    real number, and anything else raises ``TypeError`` with the text
    scipy's argument conversion produces.  Without a weight ``wvar`` is not
    read at all.
    """
    if wcode == W_NONE:
        return 0.0, 0.0
    if W_ALG <= wcode <= W_ALG_LOG:
        if wvar is None or isinstance(wvar, (bool, int, float, np.number)):
            raise TypeError(_WVAR_NOT_SEQ_MSG
                            % ('None' if wvar is None
                               else type(wvar).__name__))
        w = np.asarray(wvar, np.float64).ravel()
        if w.size != 2:
            raise TypeError(_WVAR_LEN_MSG % w.size)
        return float(w[0]), float(w[1])
    if isinstance(wvar, np.ndarray):
        if wvar.ndim:
            raise TypeError(_WVAR_0D_ARRAY_MSG)
        return float(wvar), 0.0
    if not isinstance(wvar, (bool, int, float, np.number)):
        raise TypeError(_WVAR_SCALAR_MSG
                        % ('NoneType' if wvar is None
                           else type(wvar).__name__))
    return float(wvar), 0.0


def _check_limit(limit):
    """``limit`` as scipy's C conversion takes it: an integer and nothing
    else.  A float, including one whose value is integral, raises."""
    if isinstance(limit, (bool, int, np.integer)):
        return
    raise TypeError("'%s' object cannot be interpreted as an integer"
                    % type(limit).__name__)


def _raise_integrand(ab, py, kinds):
    """Raise what the integrand raised, if it raised.

    The adapter recorded a flag and the point, because a ``@cfunc`` cannot
    carry an exception back to Fortran.  Repeating that one call here
    reproduces the caller's own exception, which is what scipy propagates.
    """
    if ab[ab.size - 2] == 0.0:
        return
    _replay_quad(py, kinds)(ab[ab.size - 1], ab)
    raise ValueError(_INTEGRAND_MSG)


def _quad_slice(r, full_output, limit, route):
    """The ONLY thing an entry point does with the core's return."""
    if full_output:
        return r[0], r[1], _infodict(r, route)
    if r[3] in (1, 2, 3, 4, 5, 7):
        _emit_ier_warning(r[3], limit, 4, route == 'qawfe')
    return r[0], r[1]


# ---------------------------------------------------------------------
# quad
# ---------------------------------------------------------------------
def quad(func, a, b, args=(), full_output=0, epsabs=1.49e-8,
         epsrel=1.49e-8, limit=50, points=None, weight=None, wvar=None,
         wopts=None, maxp1=50, limlst=50, complex_func=False):
    """Adaptive integration of ``func`` from ``a`` to ``b``.

    One name over seven QUADPACK entry points, chosen from the values of
    ``a`` and ``b``, from ``points`` and from ``weight``.

    ==================================  =========
    call                                QUADPACK
    ==================================  =========
    ``quad(f, 0.0, 1.0)``               ``dqags``
    ``quad(f, 0.0, np.inf)``            ``dqagi``
    ``quad(f, 0.0, 1.0, points=p)``     ``dqagp``
    ``weight='cos'``/``'sin'``, finite  ``dqawo``
    ``weight='cos'``/``'sin'``, b=inf   ``dqawf``
    ``weight='alg'``...                 ``dqaws``
    ``weight='cauchy'``                 ``dqawc``
    ==================================  =========

    ``func`` is a plain ``@njit`` function, called as ``func(x, *args)``.

    **Calls nest, to any depth.**  A ``quad`` inside another ``quad``'s
    integrand is correct: on ``int_0^1 int_0^1 exp(-x y) dy dx``, whose
    value is 0.796599599297053, the nested call returns
    0.7965995992970532.  Each wrapper saves the thread's callback slot on
    entry and restores it on exit, which is what makes that work.

    :func:`~scijit.integrate.nquad` is that nesting with its depth taken
    from the length of ``ranges``, and ``dblquad`` and ``tplquad`` are
    ``nquad`` at depth two and three.

    Parameters
    ----------
    func : @njit function
        The integrand.  It is called as ``func(x, *args)``, so it takes one
        argument when ``args`` is empty and ``1 + len(args)`` otherwise, and
        returns the integrand value.
    a, b : float
        Interval endpoints.  Either may be ``np.inf`` or ``-np.inf``.
        ``b < a`` gives the negated integral and ``a == b`` gives
        ``(0.0, 0.0)``.
    args : tuple, optional
        Extra parameters, handed to ``func`` after ``x``.  A value that is
        not a tuple is read as a one-item tuple, so ``args=None`` is one
        argument and not none.  An entry is a real number or an array of
        real numbers.  A scalar keeps its type, so an integer arrives as an
        integer; an array's data arrives as float64 at its own rank and
        shape.  Default ``()``.
    full_output : bool or int, optional
        Return the diagnostics as a third value.  Default 0.  Inside
        ``@njit`` it must be a compile-time constant: it selects the
        number of return values.
    epsabs, epsrel : float, optional
        Requested absolute and relative accuracy.  Both default to 1.49e-8.
    limit : int, optional
        Maximum number of subintervals.  Default 50.
    points : float64 array or None, optional
        Interior break points, where the integrand has a known
        singularity or kink.  Sorted, deduplicated and filtered to
        ``a < p < b`` first, so unsorted input, duplicates, out-of-range
        values, the endpoints themselves and ``nan``/``inf`` all drop out.
        Only with finite limits: an infinite one raises ``ValueError``.
        Inside ``@njit`` ``None`` against an array is a compile-time
        choice.
    weight : str or None, optional
        Weight function folded into the integrand, one of ``'cos'``,
        ``'sin'``, ``'alg'``, ``'alg-loga'``, ``'alg-logb'``,
        ``'alg-log'``, ``'cauchy'``.  Case sensitive.  Inside ``@njit`` it
        must be a string literal, since it selects which QUADPACK routine
        is called.
    wvar : float or 2-element sequence, optional
        The weight's parameter: ``omega`` for ``'cos'``/``'sin'``,
        ``c`` for ``'cauchy'``, ``(alpha, beta)`` for the ``'alg'``
        family.  A value the weight does not take raises ``TypeError``.
        For a ``wvar`` ARRAY under the ``'alg'`` family the length is read
        when the call runs, so the message names the requirement and not
        the length received.
    wopts : optional
        Accepted and ignored.  Its purpose is to hand back precomputed
        Chebyshev moments.
    maxp1 : int, optional
        Upper bound on the number of Chebyshev moment sets for
        ``'cos'``/``'sin'``.  Default 50.
    limlst : int, optional
        Upper bound on the number of cycles for ``'cos'``/``'sin'``
        with ``b = inf``.  Default 50.
    complex_func : bool, optional
        Integrate a complex-valued ``func``, by integrating its real and
        its imaginary part separately and recombining.  Default False.
        Inside ``@njit`` it must be a compile-time constant, and it cannot
        be combined there with ``full_output``.

    Returns
    -------
    y : float or complex
        The integral.  Complex when ``complex_func`` is true.
    abserr : float or complex
        Estimate of the absolute error.
    infodict : dict
        Present only when ``full_output`` is true.  The key set for the
        routine the call reached: ``neval``, ``last``, ``alist``,
        ``blist``, ``rlist``, ``elist`` and ``iord`` on the five ordinary
        routes, ``pts``, ``level`` and ``ndin`` beside them on the
        break-point route, ``nnlog``, ``chebmo`` and ``momcom`` on the
        oscillatory one, and ``neval``, ``lst``, ``rslst``, ``erlst`` and
        ``ierlst`` on the cos/sin route to infinity.  An empty interval,
        ``a == b``, carries the unweighted key set whatever the route.  The
        arrays have length ``limit``, and length ``limlst`` for the cycle
        arrays on the cos/sin route to infinity; only ``[:last]``, or
        ``[:lst]`` there, is meaningful.  One key is additive: ``ier``,
        QUADPACK's exit code, 0 for success.

        EVERY VALUE IS A 1-D float64 ARRAY.  ``neval``, ``last``, ``lst``
        and ``ier`` are length 1, so ``infodict['neval']`` reads
        ``array([21.])`` and the value itself is ``infodict['neval'][0]``.
        ``chebmo`` is flat, of length ``25 * maxp1``; ``.reshape(25,
        maxp1)`` recovers a two-dimensional shape.  The value types are
        described under Notes.

        With ``complex_func`` the two runs arrive under ``'real'`` and
        ``'imag'``.

    Raises
    ------
    ValueError
        What QUADPACK reports as an invalid input: a ``limit`` below 1; a
        ``maxp1`` below 1 or a ``limlst`` below 3 on a route that reads
        them; more break points than ``limit``, or break points outside
        the interval; ``alpha`` or ``beta`` below -1; a Cauchy ``wvar``
        equal to a limit; an ``epsabs`` at or below 0 with too small an
        ``epsrel``.  Also break points with an infinite limit, a cos/sin
        weight over ``(-inf, inf)``, an ``alg`` or ``cauchy`` weight over
        an infinite interval, and an unrecognised ``weight``.
    TypeError
        ``func`` given as an address rather than as a function, with an
        arity that ``args`` does not fit, or returning a complex value
        while ``complex_func`` is false.

    An empty interval, ``a == b``, returns before any of these are
    checked.  Whatever the integrand raises is propagated.

    Warns
    -----
    IntegrationWarning
        On a soft QUADPACK failure (``ier`` in 1, 2, 3, 4, 5, 7) with
        ``full_output`` false, and on ``points`` given together with
        ``weight``, which is ignored.

    See Also
    --------
    scipy.integrate.quad : The scipy routine this mirrors.

    Notes
    -----
    Three arguments are compile-time constants inside ``@njit``:
    ``full_output``, ``weight`` and ``complex_func``.  A runtime value in one
    of those slots raises ``TypingError`` naming it.

    :class:`IntegrationWarning` is a different class object from
    ``scipy.integrate.IntegrationWarning``, with the same name and base, so
    a filter naming scipy's class does not select it.  Filter on this
    package's class or on ``UserWarning``.

    **The infodict's values.** Every value is a 1-D float64 array, where
    scipy's ``infodict`` holds an int under ``neval``, ``last``, ``lst``
    and ``momcom``, int32 arrays under ``iord``, ``level``, ``ndin``,
    ``nnlog`` and ``ierlst``, and a two-dimensional ``chebmo``.  Read a
    count as ``int(info['neval'][0])`` and the moments as
    ``info['chebmo'].reshape(25, maxp1)``.

    A ``func`` returning a complex value while ``complex_func`` is false
    raises ``TypeError``.  scipy raises the same for a Python ``complex``
    and, for a numpy ``complex128``, warns once per evaluation and
    integrates the real part.

    With ``complex_func`` true and ``full_output`` false, an empty
    interval returns ``0j``, where scipy returns ``0.0``.

    Every diagnostic array is zeroed past ``last``, where scipy leaves the
    tail uninitialised.

    On a nonzero ``ier`` the Fortran layer also prints a short line to
    standard output, which numba cannot suppress.

    Examples
    --------
    A plain ``@njit`` integrand, with its extra parameters in the ``args``
    tuple:

    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def f(x, c, d):
    ...     return c * np.exp(-d * x * x)
    >>> @njit
    ... def run():
    ...     return si.quad(f, 0.0, 1.0, (2.0, 1.0))
    >>> run()
    (1.4936482656248542, 1.658282695188145e-14)

    ``full_output`` adds the diagnostics, and must be a constant:

    >>> @njit
    ... def g(x):
    ...     return np.exp(-x * x)
    >>> @njit
    ... def run_full():
    ...     v, e, info = si.quad(g, 0.0, 1.0, (), True)
    ...     return v, info['neval'][0], info['last'][0], info['ier'][0]
    >>> run_full()
    (0.7468241328124271, 21.0, 1.0, 0.0)

    A complex integrand, with ``complex_func``:

    >>> @njit
    ... def h(x, w):
    ...     return np.exp(1j * w * x)
    >>> @njit
    ... def run_cx():
    ...     return si.quad(h, 0.0, 1.0, (2.0,), complex_func=True)
    >>> run_cx()
    ((0.45464871341284085+0.7080734182735712j), (6.045545055588785e-15+7.861194120923578e-15j))
    """
    # Row quad-D14. scipy wraps a non-tuple `args` in a one-item tuple and
    # `None` is a non-tuple, so the integrand is called with it.
    at = (None,) if args is None else _as_args_tuple(args)
    if a == b:
        # scipy's empty-interval shortcut runs ahead of every check it
        # makes, so an unrecognised `weight`, an integrand of an arity the
        # call cannot satisfy and a `func` that is not callable at all all
        # reach it and return.
        lim0 = limit if isinstance(limit, (bool, int, np.integer)) else 0
        if lim0 is not limit and full_output:
            # A non-integer `limit` is a length here only under
            # `full_output`, which is where numpy refuses it in scipy too.
            raise TypeError("expected a sequence of integers or a single "
                            "integer, got '%s'" % (limit,))
        r = _quad_core(0, a, b, np.zeros(1), 0.0, 0.0, lim0, np.zeros(0),
                       False, W_NONE, 0.0, 0.0, maxp1, limlst)
        if complex_func:
            if not full_output:
                return complex(0.0, 0.0), complex(0.0, 0.0)
            d = _infodict(r, 'std')
            return (complex(0.0, 0.0), complex(0.0, 0.0),
                    {'real': d, 'imag': d})
        return _quad_slice(r, bool(full_output), lim0, 'std')
    if points is not None and weight is not None:
        # Row quad-D11. scipy warns as soon as it sees the two together and
        # validates the weight name and `wvar` afterwards, so a call with an
        # unrecognised weight warns and then raises.
        _emit_points_ignored(3)
    wcode = _weight_code(weight)
    _check_limit(limit)
    kinds = _arg_kinds(at)
    py = _callback_pyfunc(func, kinds)
    if hasattr(func, 'compile'):
        # Link the integrand's own symbols before it is used as a
        # first-class function value, which is what lets `quad` nest.
        _link_integrand(func, kinds)
    w0, w1 = _wvar_pair(wvar, wcode)
    ab = _pack_args(at)
    if points is None or wcode != W_NONE:
        pts, usep = np.zeros(0), False
    else:
        pts = np.ascontiguousarray(np.asarray(points, np.float64).ravel())
        usep = True
    fp = fpr = 0
    if not complex_func:
        fp = _adapter_quad(py, kinds).address
        if wcode == W_COS or wcode == W_SIN:
            # scipy remaps f(x) -> +-f(-x) over (-b, +inf) when a = -inf.
            # The address is frozen at typing time, so the remapping is a
            # SECOND adapter rather than a branch inside the callback.
            fpr = _adapter_quad(py, kinds, True, wcode == W_SIN).address
    route = _route_name(wcode, usep, a, b)
    if complex_func:
        # scipy integrates the real and the imaginary part separately and
        # recombines, so each part is a QUADPACK run of its own.
        rfpr = ifpr = 0
        if wcode == W_COS or wcode == W_SIN:
            rfpr = _adapter_quad(py, kinds, True, wcode == W_SIN,
                                 'real').address
            ifpr = _adapter_quad(py, kinds, True, wcode == W_SIN,
                                 'imag').address
        rr = _quad_core(_adapter_quad(py, kinds, part='real').address,
                        a, b, ab, epsabs, epsrel, limit, pts, usep,
                        wcode, w0, w1, maxp1, limlst, rfpr)
        _raise_integrand(ab, py, kinds)
        ri = _quad_core(_adapter_quad(py, kinds, part='imag').address,
                        a, b, ab, epsabs, epsrel, limit, pts, usep,
                        wcode, w0, w1, maxp1, limlst, ifpr)
        _raise_integrand(ab, py, kinds)
        val = rr[0] + 1j * ri[0]
        err = rr[1] + 1j * ri[1]
        if not full_output:
            for _r in (rr, ri):
                if _r[3] in (1, 2, 3, 4, 5, 7):
                    _emit_ier_warning(_r[3], limit, 3, route == 'qawfe')
            return val, err
        dr = _infodict(rr, route)
        di = _infodict(ri, route)
        # scipy wraps each in the tuple its own return was sliced from.
        return val, err, {'real': (dr,), 'imag': (di,)}
    r = _quad_core(fp, a, b, ab, epsabs, epsrel, limit, pts, usep,
                   wcode, w0, w1, maxp1, limlst, fpr)
    _raise_integrand(ab, py, kinds)
    return _quad_slice(r, bool(full_output), limit, route)


def _wvar_refusal(wvar, wcode):
    """The refusal :func:`_wvar_pair` would make, from the TYPE of ``wvar``.

    ``None`` when the type is one the weight accepts, or when the length is
    a run-time fact and the reader checks it instead.
    """
    if wcode == W_NONE:
        return None
    if W_ALG <= wcode <= W_ALG_LOG:
        if _is_none(wvar):
            return _WVAR_NOT_SEQ_MSG % 'None'
        if isinstance(wvar, types.Number):
            return _WVAR_NOT_SEQ_MSG % ('bool' if isinstance(wvar,
                                                             types.Boolean)
                                        else ('int' if isinstance(
                                            wvar, types.Integer) else 'float'))
        if isinstance(wvar, types.BaseTuple) and len(wvar) != 2:
            return _WVAR_LEN_MSG % len(wvar)
        return None
    if _is_none(wvar):
        return _WVAR_SCALAR_MSG % 'NoneType'
    if isinstance(wvar, types.Array):
        return _WVAR_0D_ARRAY_MSG if wvar.ndim else None
    if not isinstance(wvar, types.Number):
        return _WVAR_SCALAR_MSG % ('tuple' if isinstance(wvar, types.BaseTuple)
                                   else 'object')
    return None


def _empty_only_impl(cls, msg, lit_fo, lit_cf, warn_pts=False):
    """A compiled body for a call that is legal only over an empty interval.

    scipy checks ``a == b`` first and returns before it validates anything,
    so a call this module would otherwise refuse when it compiles has to
    return ``(0.0, 0.0)`` at run time when the interval turns out to be
    empty, and raise ``cls(msg)`` when it does not.

    ``warn_pts`` carries row quad-D11: ``points`` given with a ``weight`` is
    warned about before the refusal, which is where scipy warns.
    """
    body = ["def impl(func, a, b, args=(), full_output=0, epsabs=1.49e-8,",
            "         epsrel=1.49e-8, limit=50, points=None, weight=None,",
            "         wvar=None, wopts=None, maxp1=50, limlst=50,",
            "         complex_func=False):",
            "    if a != b:"]
    if warn_pts:
        body.append("        _warn_points_ignored()")
    body.append("        raise %s(msg)" % cls)
    if lit_cf:
        body.append("    return complex(0.0, 0.0), complex(0.0, 0.0)")
    elif lit_fo:
        body += ["    r = _quad_core(0, a, b, np.zeros(1), 0.0, 0.0, limit,",
                 "                   np.zeros(0), False, 0, 0.0, 0.0, maxp1,",
                 "                   limlst)",
                 "    return r[0], r[1], _info_std(r, True, False)"]
    else:
        body.append("    return 0.0, 0.0")
    ns = {'msg': msg, 'np': np, '_quad_core': _quad_core,
          '_info_std': _info_std,
          '_warn_points_ignored': _warn_points_ignored}
    exec("\n".join(body), ns)                                # noqa: S102
    return ns['impl']


@overload(quad, prefer_literal=True)
def _quad_ovl(func, a, b, args=(), full_output=0, epsabs=1.49e-8,
              epsrel=1.49e-8, limit=50, points=None, weight=None, wvar=None,
              wopts=None, maxp1=50, limlst=50, complex_func=False):
    """Compiled body for :func:`quad`, one per distinct call shape.

    Everything that decides the shape is resolved here rather than at run
    time: an integer ``func`` routes to Fortran and a Dispatcher to the pure
    port; ``full_output`` picks the return arity; ``weight`` picks the
    QUADPACK routine; ``args``, ``wvar`` and ``points`` each get a tiny
    ``@njit`` reader so the body below has no branches left in it.

    Returning ``None`` declines the overload, which surfaces as a
    ``TypingError``.  That is the intended answer for a runtime
    ``full_output`` or ``weight``.
    """
    lit_fo = _lit_bool(full_output)
    if lit_fo is None:
        raise TypingError(_FULL_OUTPUT_MSG)
    lit_cf = _lit_bool(complex_func)
    if lit_cf is None:
        raise TypingError(_COMPLEX_MSG)
    if lit_cf and lit_fo:
        raise TypingError(_COMPLEX_FO_MSG)
    ws = _lit_str(weight)
    if ws is None and not _is_none(weight):
        # `weight` is a runtime value or not a string. Both are refused:
        # it routes to a different QUADPACK entry point, so it has to be
        # known when this body compiles.
        raise TypingError(_WEIGHT_MSG)
    # Every refusal below is carried to RUN time instead of being raised
    # here, because scipy's empty-interval shortcut precedes all of them and
    # `a == b` is a run-time question.  Each keeps its own class and text
    # for the interval that is not empty.  scipy reports the weight before
    # the integrand, so the order below is scipy's.
    bad = None
    kinds = ()
    # Row quad-D14. `quad`'s own `args=None` is scipy's one-item tuple
    # holding None, which QUADPACK's double* slot cannot carry, so it is
    # refused. The class and the text are the python body's, and the refusal
    # waits for `a != b` like every other one here.
    _args_none = _is_none(args) or (isinstance(args, types.Omitted)
                                    and args.value is None)
    try:
        kinds = () if _args_none else _arg_kinds_ty(_args_types(args))
    except TypingError:
        bad = ('ValueError', _ARGS_MSG)
    if _args_none:
        bad = bad or ('ValueError', _ARGS_MSG)
    wcode = W_NONE
    try:
        wcode = _weight_code(ws)
    except ValueError as exc:
        bad = bad or ('ValueError', str(exc))
    if not isinstance(func, types.Dispatcher):
        bad = bad or ('TypeError', _ADDRESS_MSG)
    elif bad is None:
        try:
            _check_arity(func.dispatcher.py_func, len(kinds))
        except TypeError as exc:
            bad = ('TypeError', str(exc))
    # Row quad-D21. `limit` reaches numpy.full as a length, so a float is a
    # typing failure here where scipy's conversion raises TypeError.
    if not isinstance(limit, (types.Integer, types.Boolean, int, bool)):
        bad = bad or ('TypeError',
                      "'%s' object cannot be interpreted as an integer"
                      % ('float' if isinstance(limit, (types.Float, float))
                         else 'object'))
    # Rows quad-D5 and quad-D6. `wvar` is validated per weight, as scipy's
    # argument conversion does, rather than read positionally: the `alg`
    # family's reader took `v[0]` and `v[1]` from a sequence of any length,
    # and numba does not bounds-check.
    _wv_bad = _wvar_refusal(wvar, wcode)
    if _wv_bad is not None:
        bad = bad or ('TypeError', _wv_bad)
    if bad is not None:
        return _empty_only_impl(bad[0], bad[1], lit_fo, lit_cf,
                                not _is_none(points) and not _is_none(weight))
    py = func.dispatcher.py_func
    _link_integrand(func.dispatcher, kinds)

    if wcode == W_NONE or _is_none(wvar):
        @njit
        def _getw(v):
            return 0.0, 0.0
    elif isinstance(wvar, types.Number):
        @njit
        def _getw(v):
            return float(v), 0.0
    elif isinstance(wvar, types.BaseTuple):
        @njit
        def _getw(v):
            return float(v[0]), float(v[1])
    else:
        # An array's LENGTH is a run-time fact, so the refusal scipy makes
        # from the length is made here. numba takes a constant message only,
        # so the count scipy prints is not in it.
        @njit
        def _getw(v):
            if v.size != 2:
                raise TypeError("argument 4 must be sequence of length 2")
            return float(v[0]), float(v[1])

    no_pts = _is_none(points)
    if no_pts:
        @njit
        def _getp(p):
            return np.zeros(0)
    elif isinstance(points, types.Number):
        @njit
        def _getp(p):
            out = np.empty(1)
            out[0] = float(p)
            return out
    else:
        @njit
        def _getp(p):
            return np.ascontiguousarray(
                np.asarray(p).astype(np.float64).ravel())

    use_pts = (not no_pts) and wcode == W_NONE
    warn_pts = (not no_pts) and wcode != W_NONE

    # The key set is chosen here, so it has to be decidable here.  It is,
    # for every route but one: break points refuse an infinite limit, so
    # `use_pts` is `dqagp`, and the `alg` and `cauchy` weights refuse one
    # too.  Only cos/sin can still reach two routines, on a run-time test.
    _info = (_info_cos if wcode == W_COS or wcode == W_SIN
             else _info_pts if use_pts else _info_std)

    is_cs = wcode == W_COS or wcode == W_SIN
    part_r = 'real' if lit_cf else ''
    _replay = _replay_quad(py, kinds)
    addr = _adapter_quad(py, kinds, part=part_r).address
    ref = 0
    if wcode == W_COS or wcode == W_SIN:
        # scipy remaps f(x) -> +-f(-x) over (-b, +inf) when a = -inf.  The
        # address is frozen at typing time, so the remapping is a SECOND
        # adapter rather than a branch inside the callback.
        ref = _adapter_quad(py, kinds, True, wcode == W_SIN, part_r).address

    if lit_cf:
        addr_i = _adapter_quad(py, kinds, part='imag').address
        ref_i = 0
        if wcode == W_COS or wcode == W_SIN:
            ref_i = _adapter_quad(py, kinds, True, wcode == W_SIN,
                                  'imag').address

        def impl(func, a, b, args=(), full_output=0, epsabs=1.49e-8,
                 epsrel=1.49e-8, limit=50, points=None, weight=None,
                 wvar=None, wopts=None, maxp1=50, limlst=50,
                 complex_func=False):
            if warn_pts and a != b:
                _warn_points_ignored()
            w0, w1 = _getw(wvar)
            ab = _pack_args(args)
            pp = _getp(points)
            rr = _quad_core(addr, a, b, ab, epsabs, epsrel, limit, pp,
                            use_pts, wcode, w0, w1, maxp1, limlst, ref)
            if ab[ab.size - 2] != 0.0:
                _replay(ab[ab.size - 1], ab)
                raise ValueError(_INTEGRAND_MSG)
            ri = _quad_core(addr_i, a, b, ab, epsabs, epsrel, limit, pp,
                            use_pts, wcode, w0, w1, maxp1, limlst, ref_i)
            if ab[ab.size - 2] != 0.0:
                _replay(ab[ab.size - 1], ab)
                raise ValueError(_INTEGRAND_MSG)
            cyc = is_cs and (b == np.inf or a == -np.inf)
            _warn_ier(rr[3], limit, cyc)
            _warn_ier(ri[3], limit, cyc)
            return (complex(rr[0], ri[0]), complex(rr[1], ri[1]))
        return impl

    def impl(func, a, b, args=(), full_output=0, epsabs=1.49e-8,
             epsrel=1.49e-8, limit=50, points=None, weight=None,
             wvar=None, wopts=None, maxp1=50, limlst=50,
             complex_func=False):
        if warn_pts and a != b:
            # scipy takes the empty-interval shortcut before it warns
            _warn_points_ignored()
        w0, w1 = _getw(wvar)
        ab = _pack_args(args)
        r = _quad_core(addr, a, b, ab, epsabs,
                       epsrel, limit, _getp(points), use_pts, wcode,
                       w0, w1, maxp1, limlst, ref)
        if ab[ab.size - 2] != 0.0:
            _replay(ab[ab.size - 1], ab)
            raise ValueError(_INTEGRAND_MSG)
        if lit_fo:
            # `a == b` and an infinite endpoint are RUN-TIME questions, and
            # scipy's key set turns on both. The same two tests select the
            # route in the python body, through `_route_name`.
            return r[0], r[1], _info(
                r, a == b,
                a == np.inf or a == -np.inf or b == np.inf or b == -np.inf)
        _warn_ier(r[3], limit, is_cs and (b == np.inf or a == -np.inf))
        return r[0], r[1]
    return impl
