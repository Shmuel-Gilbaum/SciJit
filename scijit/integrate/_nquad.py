"""Integration over ``n`` variables.

An ``n``-deep nest of :func:`~scijit.integrate.quad`.

Axis 0 is the INNERMOST integral, so ``ranges[0]`` is
integrated first and ``ranges[-1]`` last.  The depth is a compile-time
property because it comes from the LENGTH of ``ranges``, which numba knows
when the caller compiles.  The chain is built in Python at typing time, one
``@cfunc`` per level, each holding the address of the level below.

Everything that picks a QUADPACK entry point is resolved at typing time and
baked into the generated code: the depth, each axis's ``weight``, whether an
axis has break points, which shape each callback is written in, and
``full_output``.  Nothing is decided at run time that could have been
decided when the call compiled.

Buffer layout.  One flat ``float64`` block per invocation, allocated by the
entry point and threaded down the chain as ``quad``'s ``args``::

    [0]  na             length of the packed args block
    [1]  total          length of this buffer
    [2]  neval          running count, innermost level only
    [3]  abserr         running MAX over every level, which is what scipy
                        reports
    [4]  err            0, or the level that failed as ``axis + 1``,
                        negative when a callback raised rather than the
                        integration
    [5 .. 5 + 14n)      per-axis header, 14 slots each, axis 0 first:
                        epsabs epsrel limit lo hi wvar0 wvar1 poff plen
                        maxp1 limlst wcode elo ehi
    [HC .. HC + n)      coordinates, index i holding axis i's
    [HA .. HA + na)     args, packed as `_quadpack._pack_args` writes them
    [HP ..)             break points, per axis, located by poff/plen

``HC = 5 + 14n`` and ``HA = HC + n`` are compile-time; ``HP`` follows the
args block.  The slot count is ``_SLOTS`` and the offsets within a block are
the names unpacked from ``range(14)`` below.  ``elo``/``ehi`` are the limits
the axis was integrated over and they are PER AXIS: one shared pair would be
overwritten by a deeper level while the outer level was still running.
Each level writes its OWN coordinate slot and passes the same buffer down,
so no level copies and nothing is shared between levels.  The buffer is
allocated per call, so the construction is prange-safe.

An exception raised below the outermost level cannot cross the C ABI: a
``raise`` inside a ``@cfunc`` is printed and swallowed.  Each level records
itself in the ``err`` slot instead, every later evaluation returns zero so
QUADPACK converges quickly, and the entry point raises.  Where the
integration itself failed, the entry point REPEATS that level's
``_quad_core`` call on the recorded limits with an integrand that returns
zero, so the exception the caller sees is the one scipy raises, with
scipy's own message.
"""
import inspect

import numpy as np
from numba import carray, cfunc, njit, typeof, types
from numba.core.errors import TypingError
from numba.extending import overload

from ._quadpack import (W_NONE, _COMPLEX_REAL_MSG, _K_BOOL, _K_FLOAT,
                        _K_INT, _arg_kinds, _arg_kinds_ty, _integrand_sig,
                        _pack_args, _quad_core, _warn_ier, _weight_code,
                        _quadpack_sig)

# fixed head
_NU, _TOT, _NEVAL, _AERR = 0, 1, 2, 3
#: the level that failed, as ``axis + 1``; negative when a CALLBACK raised
#: rather than the integration itself.
_ERR = 4
_HDR = 5                                  # first per-axis slot
_SLOTS = 14                               # slots per axis
# offsets within an axis block.  `_ELO`/`_EHI` are the limits the axis was
# called with, which is what lets the entry point reproduce the exception,
# and they live here rather than in the fixed head because every level
# writes them: one shared pair is overwritten by a deeper level while the
# outer level's own integration is still running.
(_EA, _ER, _LIM, _LO, _HI, _W0, _W1, _POFF, _PLEN, _MAXP1,
 _LIMLST, _WC, _ELO, _EHI) = range(14)

_DEF_EA = 1.49e-8
_DEF_ER = 1.49e-8
_DEF_LIM = 50.0
_DEF_MAXP1 = 50.0
_DEF_LIMLST = 50.0

#: scipy's `opts` keys.  These are `quad`'s own keyword arguments, which is
#: what an `opts` entry is: scipy hands the dict to `quad` as `**opt`.
#: `wopts` is accepted and ignored, as in :func:`quad`.
_OPT_KEYS = ('epsabs', 'epsrel', 'limit', 'points', 'weight', 'wvar',
             'wopts', 'maxp1', 'limlst', 'complex_func')

#: The first half of `_OPT_KEY_MSG`, for the run-time key walk, which builds
#: its message by concatenation rather than by formatting.
_OPT_KEY_PRE = "quad() got an unexpected keyword argument '"

#: Built chains, keyed by everything that shapes the generated code.  The
#: cache OWNS every cfunc: the addresses are baked into compiled code, so
#: dropping a reference would leave one dangling.
_CHAINS = {}

_ARITY_MSG = (
    "%s: `func` takes %d arguments, which is neither of the two shapes "
    "this accepts at depth %d with an `args` of length %d. Write scipy's "
    "shape, one scalar per axis innermost first and then one argument per "
    "`args` entry, f(x0, ..., x%d%s); or the array shape, f(x%s) with x "
    "holding the %d coordinates.")

_LIMIT_ARITY_MSG = (
    "%s: a limit callback for axis %d takes %d arguments. It receives the "
    "%d already-fixed OUTER coordinates and then the %d `args` entries, so "
    "write g(%s) in scipy's shape, or g(coords%s) with coords holding the "
    "coordinates outermost first.")

_OPT_TUPLE_MSG = (
    "nquad: per-axis `opts` carrying a `weight` cannot be passed from inside "
    "@njit. numba lowers a dict holding a string only on its own, not inside "
    "a tuple. Three spellings work: one dict applied to every axis, "
    "opts={'weight': 'cos', 'wvar': 1.0}; per-axis dicts holding numbers "
    "only; or calling nquad from python, where every per-axis spelling "
    "works.")

_OPT_HOMO_MSG = (
    "nquad: an `opts` dict whose values are ALL THE SAME KIND does not carry "
    "its keys to compile time, and `weight` and `points` pick the QUADPACK "
    "routine for their axis, so they have to be known then. Add a numeric "
    "entry to make the dict heterogeneous and its keys readable: "
    "opts={'weight': 'cos', 'wvar': 1.0} or "
    "opts={'points': (0.5,), 'epsabs': 1.49e-8}. A dict holding numbers only "
    "needs nothing, since its settings are read at run time.")

#: Reached from `nquad`, `dblquad` and `tplquad`, since the last two
#: delegate. The caller's own name is substituted so a caller who typed
#: `dblquad` is not told about a function they have never used.
_FUNC_NJIT_MSG = (
    "%s: `func` must be a plain @njit function, called as "
    "f(x0, ..., xn, *args) with one argument per axis, innermost first, and "
    "then one per entry of `args`. The .address of a @cfunc is not accepted: "
    "pass the @njit function itself and %s builds the callback.")
_LIMIT_NJIT_MSG = (
    "%s: a limit callback must be a plain @njit function. The .address of a "
    "@cfunc is not accepted: pass the @njit function itself.")

#: An exception raised BELOW the outermost level cannot cross the C ABI, so
#: the level records itself in the buffer and the entry point raises.  Where
#: the integration itself failed, the same `_quad_core` call is repeated on
#: the same limits with an integrand that returns zero, which reproduces
#: scipy's own message rather than restating it here.
_INNER_MSG = (
    "nquad: the integration over an inner axis failed; the exception was "
    "raised below the outermost level, where it cannot be propagated across "
    "the callback boundary.")
_INNER_CB_MSG = (
    "nquad: a limit callback or the integrand raised while integrating an "
    "inner axis. The exception was raised below the outermost level, where "
    "it cannot be propagated across the callback boundary.")

#: scipy's own text.  The dict reaches `quad` as `**opt`, so an unknown key
#: is Python's unexpected-keyword-argument report on `quad` itself, and the
#: class is TypeError for the same reason.
_OPT_KEY_MSG = "quad() got an unexpected keyword argument '%s'"

#: scipy's `nquad` reaches `_NQuad(...).integrate(*args)`, so `args` is
#: UNPACKED rather than coerced: an array or a list contributes one
#: parameter per element and a bare value raises.  `quad` coerces a bare
#: value to a one-item tuple instead; the two differ in scipy and here.
_ARGS_ITER_MSG = (
    "%s: `args` is a tuple, or another iterable that unpacks into one. "
    "scipy reaches `.integrate(*args)`, where a bare value raises "
    "TypeError: argument after * must be an iterable.")

_ARGS_ARRAY_MSG = (
    "%s: an `args` ARRAY cannot be unpacked inside @njit. Its length says "
    "how many parameters the integrand takes, which has to be known when "
    "the call compiles, and an array carries no length in its type. Write "
    "the tuple, args=(a, b). An array works from python, as it does in "
    "scipy.")

_FULL_OUTPUT_MSG = (
    "nquad: full_output must be a compile-time constant inside @njit. It "
    "selects the number of return values, so a runtime flag cannot be typed.")

_OPT_CALLABLE_MSG = (
    "nquad: an `opts` entry must be a dict or a (epsabs, epsrel, limit) "
    "triple. scipy also accepts a callable returning a dict; `weight` and "
    "`points` pick the QUADPACK routine for that axis, so they are read when "
    "the call compiles and a value produced at run time cannot reach them.")

_RANGE_MSG = (
    "nquad: each entry of `ranges` is a (lo, hi) pair, or a callable "
    "returning one. A pair's members are each a float or a plain @njit "
    "function.")

#: Python's own text.  `nquad` iterates `points` before `quad` sees the dict
#: (`_quadpack_py.py:1281`), so a value that is not iterable is reported from
#: the iteration, where `quad` alone would take a scalar.
_NOT_ITER_MSG = "'%s' object is not iterable"

_CF_MSG = (
    "nquad: `complex_func` must be false. A complex value cannot cross the "
    "double-valued callback every level of the chain is called through. "
    "`quad` takes complex_func at depth one; scipy's `nquad` raises "
    "TypeError for complex_func=True on any integrand, real or complex.")


# ---------------------------------------------------------------------
# resolving which shape a callback is written in
# ---------------------------------------------------------------------
def _arity(py):
    """Positional parameter count, or ``None`` when the function is
    ``*args``."""
    try:
        sig = inspect.signature(py)
    except (TypeError, ValueError):                          # noqa: BLE001
        return None
    n = 0
    for p in sig.parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            return None
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


def _takes_coord_array(disp, kinds):
    """True when ``f(x, *args)`` types with ``x`` an ARRAY and returns a
    scalar.

    This is what separates the array shape from scipy's scalar shape at the
    one arity where they collide, a single coordinate.  A scalar-shaped
    ``f(x0, *args)`` also COMPILES against an array, because its arithmetic
    broadcasts, but then it returns an array rather than a float, so the
    return type decides where the signature cannot.
    """
    sig = (types.float64[::1],) + tuple(_integrand_sig(kinds)[1:])
    try:
        disp.compile(sig)
    except Exception:                                        # noqa: BLE001
        return False
    cres = disp.overloads.get(sig)
    if cres is None:
        return False
    return isinstance(cres.signature.return_type, types.Number)


def _returns_complex_cb(disp, form, ncoords, kinds):
    """True when the integrand types complex for the call the chain makes."""
    tail = tuple(_integrand_sig(kinds)[1:])
    if form == 'array':
        sig = (types.float64[::1],) + tail
    else:
        sig = (types.float64,) * ncoords + tail
    try:
        disp.compile(sig)
    except Exception:                                        # noqa: BLE001
        return False
    cres = disp.overloads.get(sig)
    return cres is not None and isinstance(cres.signature.return_type,
                                           types.Complex)


def _callback_form(disp, ncoords, kinds, what, axis=None, who='nquad'):
    """``'array'`` or ``'scalar'``, and the arity the callback uses.

    ``ncoords`` is how many fixed coordinates the callback receives and
    ``kinds`` describes ``args``.  scipy's shape takes one scalar per
    coordinate and then one argument per ``args`` entry.  The array shape
    takes the coordinates as one array and then the same ``args``; it is
    preferred wherever it types, so a call written against it keeps its
    meaning.
    """
    na = len(kinds)
    py = disp.py_func
    ar = _arity(py)
    want = ncoords + na
    if ar is None:                        # *args takes whatever it is given
        form, out = 'scalar', want
    elif ar == 1 + na and _takes_coord_array(disp, kinds):
        form, out = 'array', 1 + na
    elif ar == want:
        form, out = 'scalar', want
    else:
        tail = "".join(", a%d" % i for i in range(na))
        if what == 'func':
            raise TypingError(_ARITY_MSG % (who, ar, ncoords, na, ncoords - 1,
                                            tail, tail, ncoords))
        names = ", ".join(["x%d" % i
                           for i in range(axis + 1, axis + 1 + ncoords)]
                          + ["a%d" % i for i in range(na)])
        raise TypingError(_LIMIT_ARITY_MSG % (who, axis, ar, ncoords, na,
                                              names, tail))
    if what == 'func' and _returns_complex_cb(disp, form, ncoords, kinds):
        # the integrand crosses QUADPACK's double slot, as it does one
        # level down in `quad`, so it is refused where scipy refuses it
        raise TypeError(_COMPLEX_REAL_MSG)
    return form, out


# ---------------------------------------------------------------------
# generated source helpers
# ---------------------------------------------------------------------
def _args_unpack_src(kinds, base, indent):
    """Source unflattening the packed ``args`` block at ``buf[base]``.

    The layout is :func:`~scijit.integrate._quadpack._pack_args`'s: slot 0
    is the payload length, a scalar takes one slot, an array takes its shape
    and then its data.  ``kinds`` describes it once and both halves are
    generated from that description, so they cannot drift.
    """
    lines = ["%s_o = %d" % (indent, base + 1)]
    names = []
    for i, k in enumerate(kinds):
        if k < 0:
            cast = {_K_INT: "buf.view(np.int64)[_o]",
                    _K_BOOL: "buf[_o] != 0.0",
                    _K_FLOAT: "buf[_o]"}[k]
            lines.append("%s_a%d = %s" % (indent, i, cast))
            lines.append("%s_o += 1" % indent)
        else:
            dims = ", ".join("int(buf[_o + %d])" % j for j in range(k))
            lines.append("%s_d%d = (%s,)" % (indent, i, dims))
            lines.append("%s_o += %d" % (indent, k))
            lines.append("%s_m%d = %s"
                         % (indent, i,
                            " * ".join("_d%d[%d]" % (i, j) for j in range(k))))
            if k == 1:
                lines.append("%s_a%d = buf[_o:_o + _m%d]" % (indent, i, i))
            else:
                lines.append("%s_a%d = buf[_o:_o + _m%d].reshape(_d%d)"
                             % (indent, i, i, i))
            lines.append("%s_o += _m%d" % (indent, i))
        names.append("_a%d" % i)
    return lines, names


def _coord_args(n, first, form, anames):
    """The argument list a callback is called with, as source.

    ``first`` is the innermost axis whose coordinate the callback sees; it
    receives axes ``first .. n-1``.  scipy's order is innermost of those
    first, which is the buffer's own order, so the scalar shape reads
    straight out of it.  ``args`` follows, unpacked, as in scipy.
    """
    hc = _HDR + _SLOTS * n
    if form == 'array':
        return ", ".join(["cw"] + anames)
    parts = ["buf[%d]" % (hc + i) for i in range(first, n)]
    return ", ".join(parts + anames)


def _coord_window(n, first, indent):
    """Source building the outermost-first ``coords`` the array shape wants.

    The buffer holds coordinates innermost first, so the array shape gets a
    reversed copy.  The scalar shape reads the buffer directly and this is
    not emitted at all.
    """
    hc = _HDR + _SLOTS * n
    m = n - first
    src = ["%scw = np.empty(%d)" % (indent, m)]
    for j in range(m):
        src.append("%scw[%d] = buf[%d]" % (indent, j, hc + n - 1 - j))
    return src


def _limit_src(ax, j, n, indent, anames):
    """Source computing axis ``j``'s ``lo`` and ``hi``.

    Both limits are emitted together, since a range callable produces the pair
    at once and that is scipy's spelling.  ``anames`` are the unpacked
    ``args`` names the callbacks are handed after their coordinates.
    """
    base = _HDR + _SLOTS * j
    first = j + 1
    src = []
    need_window = any(
        ax[k] == 'array'
        for k in ('rng_form', 'lo_form', 'hi_form') if ax.get(k))
    if need_window and first <= n - 1:
        src += _coord_window(n, first, indent)
    elif need_window:
        src.append("%scw = np.empty(0)" % indent)
    if ax['rng'] is not None:
        call = _coord_args(n, first, ax['rng_form'], anames)
        src.append("%slo, hi = _rng%d(%s)" % (indent, j, call))
        return src
    if ax['lo'] is None:
        src.append("%slo = buf[%d]" % (indent, base + _LO))
    else:
        call = _coord_args(n, first, ax['lo_form'], anames)
        src.append("%slo = _lo%d(%s)" % (indent, j, call))
    if ax['hi'] is None:
        src.append("%shi = buf[%d]" % (indent, base + _HI))
    else:
        call = _coord_args(n, first, ax['hi_form'], anames)
        src.append("%shi = _hi%d(%s)" % (indent, j, call))
    return src


def _points_src(j, indent):
    """Source slicing axis ``j``'s break points and filtering to ``[lo, hi]``.

    scipy drops the points that fall outside the interval before calling
    ``quad``; QUADPACK rejects them otherwise.
    """
    base = _HDR + _SLOTS * j
    return [
        "%spo = int(buf[%d])" % (indent, base + _POFF),
        "%spl = int(buf[%d])" % (indent, base + _PLEN),
        "%skp = 0" % indent,
        "%sfor _i in range(pl):" % indent,
        "%s    if lo <= buf[po + _i] <= hi:" % indent,
        "%s        kp += 1" % indent,
        "%spts = np.empty(kp)" % indent,
        "%skp = 0" % indent,
        "%sfor _i in range(pl):" % indent,
        "%s    if lo <= buf[po + _i] <= hi:" % indent,
        "%s        pts[kp] = buf[po + _i]" % indent,
        "%s        kp += 1" % indent,
    ]


def _quad_call(ax, j, fo, target, indent, count_neval, top=False):
    """Source for one level's integration, plus what it does with the result.

    Every level calls ``_quad_core`` rather than :func:`quad`.  ``quad``
    normalises its ``args`` with ``.astype``, which COPIES, and the chain
    needs one shared buffer: a copy would hide each level's writes from the
    caller, which is how ``neval`` and the running ``abserr`` come back.
    Going straight to the core also drops one buffer copy per evaluation.

    Every routing argument is a compile-time constant here, so this expands
    to a call with no branches left in it.
    """
    base = _HDR + _SLOTS * j
    wcode = _weight_code(ax['weight'])
    args = [target, "lo", "hi", "buf",
            "buf[%d]" % (base + _EA), "buf[%d]" % (base + _ER),
            "int(buf[%d])" % (base + _LIM),
            "pts" if ax['points'] else "np.zeros(0)",
            "True" if ax['points'] else "False",
            str(wcode),
            "buf[%d]" % (base + _W0), "buf[%d]" % (base + _W1),
            "int(buf[%d])" % (base + _MAXP1),
            "int(buf[%d])" % (base + _LIMLST)]
    src = ["%sr = _quad_core(%s)" % (indent, ", ".join(args))]
    if top:
        # An exception below the outermost level is recorded rather than
        # raised, because a `raise` inside a @cfunc is printed and swallowed.
        # Here it can propagate, so it does, and before the warning.
        src += ["%sif buf[%d] != 0.0:" % (indent, _ERR),
                "%s    _nq_reraise(buf, buf[%d])" % (indent, _ERR)]
    src += ["%sv = r[0]" % indent,
            "%se = r[1]" % indent]
    if not fo:
        # `full_output` suppresses the warning, in scipy and in `quad`.
        src.append("%s_warn_ier(r[3], int(buf[%d]))" % (indent, base + _LIM))
    if count_neval:
        src.append("%sbuf[%d] += np.float64(r[2])" % (indent, _NEVAL))
    src += ["%sif buf[%d] < e:" % (indent, _AERR),
            "%s    buf[%d] = e" % (indent, _AERR)]
    return src


# ---------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------
def _leaf_cfunc(fpy, n, form, kinds):
    """Innermost level: fix axis 0, then call the user's integrand."""
    f = njit(fpy)
    hc = _HDR + _SLOTS * n
    ha = hc + n
    unp, anames = _args_unpack_src(kinds, ha, "    ")
    if form == 'array':
        call = ", ".join(["buf[%d:%d]" % (hc, hc + n)] + anames)
    else:
        call = ", ".join(["buf[%d]" % (hc + i) for i in range(n)] + anames)
    src = ["def leaf(x0, aptr):",
           "    buf = carray(aptr, int(aptr[%d]))" % _TOT,
           "    if buf[%d] != 0.0:" % _ERR,
           "        return 0.0",
           "    buf[%d] = x0" % hc]
    src += unp
    # A `raise` inside a @cfunc is printed and swallowed, so an exception
    # from the integrand is recorded here and raised by the entry point.
    src += ["    _v = 0.0",
            "    try:",
            "        _v = f(%s)" % call,
            "    except Exception:",
            "        buf[%d] = -1.0" % _ERR,
            "        _v = 0.0",
            "    return _v"]
    ns = {"np": np, "carray": carray, "f": f}
    exec("\n".join(src), ns)
    return cfunc(_quadpack_sig)(ns["leaf"])


def _node_cfunc(below, k, n, ax, fo, kinds):
    """Level ``k``: fix axis ``k``, then integrate axis ``k-1`` beneath it."""
    j = k - 1
    hc = _HDR + _SLOTS * n
    ha = hc + n
    unp, anames = _args_unpack_src(kinds, ha, "    ")
    src = ["def node(xk, aptr):",
           "    buf = carray(aptr, int(aptr[%d]))" % _TOT,
           # Once a level below has failed, every later evaluation is a
           # wasted one: QUADPACK still has to converge before the entry
           # point can raise, so it converges on a zero integrand.
           "    if buf[%d] != 0.0:" % _ERR,
           "        return 0.0",
           "    buf[%d] = xk" % (hc + k)]
    src += unp
    src += ["    lo = 0.0",
            "    hi = 0.0",
            "    v = 0.0",
            "    _bad = 0",
            "    try:"]
    src += _limit_src(ax, j, n, "        ", anames)
    if ax['points']:
        src.append("        pts = np.zeros(0)")
        src += _points_src(j, "        ")
    src += ["    except Exception:",
            "        _bad = 1",
            "    if _bad == 0:",
            # this axis's own slots: a deeper level writes its own pair
            # while this integration is still running
            "        buf[%d] = lo" % (_HDR + _SLOTS * j + _ELO),
            "        buf[%d] = hi" % (_HDR + _SLOTS * j + _EHI),
            "        try:"]
    # scipy counts the INNERMOST loop only, so level 1 is the one whose
    # evaluations are the integrand's own.
    src += _quad_call(ax, j, fo, "below", "            ", fo and k == 1)
    src += ["        except Exception:",
            "            _bad = 2",
            "            v = 0.0",
            "    if _bad != 0 and buf[%d] == 0.0:" % _ERR,
            "        buf[%d] = %r if _bad == 2 else %r"
            % (_ERR, float(j + 1), -float(j + 1)),
            "    return v"]
    ns = {"np": np, "carray": carray, "_quad_core": _quad_core,
          "_warn_ier": _warn_ier, "below": below}
    for tag, key in (("_rng%d" % j, 'rng'), ("_lo%d" % j, 'lo'),
                     ("_hi%d" % j, 'hi')):
        if ax[key] is not None:
            ns[tag] = njit(ax[key])
    exec("\n".join(src), ns)
    return cfunc(_quadpack_sig)(ns["node"])


def _chain_key(fpy, form, kinds, axes, n, fo):
    """Everything the generated code depends on, as a hashable key."""
    per = []
    for ax in axes:
        per.append((ax['rng'], ax['rng_form'],
                    ax['lo'], ax['lo_form'],
                    ax['hi'], ax['hi_form'],
                    ax['weight'], ax['points']))
    return (fpy, form, kinds, n, bool(fo), tuple(per))


def build_chain(fpy, form, kinds, axes, n, fo):
    """Build the nest and return the address of the OUTERMOST callable level.

    For ``n == 1`` that is the leaf itself, which the entry point hands
    straight to ``quad``.

    Parameters
    ----------
    fpy : callable
        The plain Python function behind the integrand.
    form : {'scalar', 'array'}
        Which shape the integrand is called in.
    kinds : tuple of int
        Element kinds of the packed ``args`` block.
    axes : list of dict
        One axis description per level, innermost first.
    n : int
        Nesting depth.
    fo : bool
        Whether ``full_output`` is requested.

    Returns
    -------
    int
        Address of the outermost ``@cfunc`` level.
    """
    key = _chain_key(fpy, form, kinds, axes, n, fo)
    hit = _CHAINS.get(key)
    if hit is not None:
        return hit[0]
    built = [_leaf_cfunc(fpy, n, form, kinds)]
    addr = built[0].address
    for k in range(1, n):
        node = _node_cfunc(addr, k, n, axes[k - 1], fo, kinds)
        built.append(node)
        addr = node.address
    _CHAINS[key] = (addr, built)
    return addr


@cfunc(_quadpack_sig)
def _zero_integrand(x, p):
    """Returns zero, so a failing level's integration can be REPEATED from
    the entry point where the exception can propagate."""
    return 0.0


_ZERO_ADDR = _zero_integrand.address


@njit
def _nq_reraise(buf, code):
    """Raise what the level recorded in ``code`` raised.

    A positive code is an axis whose own integration failed.  Repeating that
    ``_quad_core`` call on the same limits, with an integrand that returns
    zero, reproduces the exception scipy raises out of its nested ``quad``,
    with scipy's own message, rather than restating the message here.  A
    negative code is a callback or the integrand, whose exception carries no
    message this side can reach.
    """
    if code < 0.0:
        raise ValueError(_INNER_CB_MSG)
    base = _HDR + _SLOTS * (int(code) - 1)
    lo = buf[base + _ELO]
    hi = buf[base + _EHI]
    po = int(buf[base + _POFF])
    pl = int(buf[base + _PLEN])
    kp = 0
    for i in range(pl):
        if lo <= buf[po + i] <= hi:
            kp += 1
    pts = np.empty(kp)
    kp = 0
    for i in range(pl):
        if lo <= buf[po + i] <= hi:
            pts[kp] = buf[po + i]
            kp += 1
    _quad_core(_ZERO_ADDR, lo, hi, buf, buf[base + _EA], buf[base + _ER],
               int(buf[base + _LIM]), pts, pl > 0, int(buf[base + _WC]),
               buf[base + _W0], buf[base + _W1], int(buf[base + _MAXP1]),
               int(buf[base + _LIMLST]))
    raise ValueError(_INNER_MSG)


# ---------------------------------------------------------------------
# opts
# ---------------------------------------------------------------------
def _blank_axis():
    return {'rng': None, 'rng_form': None,
            'lo': None, 'lo_form': None,
            'hi': None, 'hi_form': None,
            'weight': None, 'points': False, 'wvar': False,
            'ea': None, 'er': None, 'lim': None}


def _check_keys(keys, from_python=False):
    """Reject an `opts` key `quad` has no argument for.

    scipy raises **TypeError** here, because the dict reaches ``quad`` as
    ``**opt`` and an unknown key is an unexpected keyword argument.
    """
    for k in keys:
        if k not in _OPT_KEYS:
            msg = _OPT_KEY_MSG % k
            raise TypeError(msg) if from_python else TypingError(msg)


def _py_kind_name(t):
    """The word Python's iteration error uses for a value of numba type `t`."""
    if isinstance(t, types.Boolean):
        return 'bool'
    if isinstance(t, types.Integer):
        return 'int'
    if isinstance(t, types.Complex):
        return 'complex'
    return 'float'


def _points_array(v):
    """``points`` as scipy materialises it, ``[x for x in opt['points']]``.

    scipy runs that comprehension at ``_quadpack_py.py:1281``, so a value
    that is not iterable raises there and raises here, with Python's own
    message rather than a restatement of it.  A generator is materialised,
    as it is there, where :func:`quad` alone would take a bare scalar.
    """
    return np.ascontiguousarray(np.asarray(list(v), np.float64).ravel())


def _check_complex_func(v, from_python):
    """Refuse a true ``complex_func``.

    scipy accepts the key and then compares a complex ``abserr`` against an
    int, so ``complex_func=True`` raises ``TypeError: '>' not supported
    between instances of 'complex' and 'int'`` there for every integrand.
    """
    if v is None or not bool(v):
        return
    raise ValueError(_CF_MSG) if from_python else TypingError(_CF_MSG)


def _opts_from_pydict(d):
    """An axis description from a real python dict."""
    # scipy materialises `points` before `quad` ever sees the dict as
    # `**opt`, so a `points` that is not iterable is reported ahead of an
    # unknown key.  Measured on 1.18: a dict carrying both reports `points`.
    pts = _points_array(d['points']) if 'points' in d else None
    _check_keys(d.keys(), True)
    _check_complex_func(d.get('complex_func'), True)
    return {'keys': set(d.keys()), 'weight': d.get('weight'),
            'points': 'points' in d, 'wvar': 'wvar' in d, 'literal': True,
            'pts': pts}


def _opts_from_type(t):
    """An axis description from whatever numba handed the overload.

    Three shapes arrive.  A HETEROGENEOUS dict literal is a
    ``LiteralStrKeyDict`` and carries its keys as plain strings, so a
    ``weight`` in it is readable here.  A HOMOGENEOUS one is an ordinary
    ``DictType`` whose keys are known only at run time, which is enough for
    the numeric settings and not enough for ``weight``.  A 3-tuple is the
    ``(epsabs, epsrel, limit)`` spelling.
    """
    if isinstance(t, types.LiteralStrKeyDict):
        keys = list(t.fields)
        # scipy iterates `points` ahead of handing the dict to `quad`, so a
        # scalar there is reported before an unknown key.  The value's TYPE
        # answers it here, which is why this one is a compile-time refusal.
        for name, vt in zip(t.fields, t.types):
            if name == 'points' and isinstance(vt, types.Number):
                raise TypingError(_NOT_ITER_MSG % _py_kind_name(vt))
        _check_keys(keys)
        w = None
        for name, vt in zip(t.fields, t.types):
            if name == 'weight':
                if not isinstance(vt, types.StringLiteral):
                    raise TypingError(_OPT_HOMO_MSG)
                w = vt.literal_value
        return {'keys': set(keys), 'weight': w, 'points': 'points' in keys,
                'wvar': 'wvar' in keys, 'literal': True,
                'cf': 'complex_func' in keys}
    if isinstance(t, types.DictType):
        # A homogeneous dict reaches run time with its keys intact and its
        # TYPE carrying none of them.  That is enough for the numeric
        # settings, read below with `in`, and not enough for anything that
        # picks a routine.  What the type cannot answer is asked at run time
        # instead, by `_runtime_opt_guards`.
        if not isinstance(t.value_type, types.Number):
            raise TypingError(_OPT_HOMO_MSG)
        return {'keys': None, 'weight': None, 'points': False,
                'wvar': False, 'literal': False, 'cf': True,
                'vkind': _py_kind_name(t.value_type)}
    if isinstance(t, types.BaseTuple) and len(t) == 3:
        return {'keys': 'triple', 'weight': None, 'points': False,
                'wvar': False, 'literal': True}
    if isinstance(t, types.Dispatcher):
        raise TypingError(_OPT_CALLABLE_MSG)
    raise TypingError(_OPT_CALLABLE_MSG)


def _opts_list(opts, n, from_python):
    """One description per axis, innermost first."""
    if from_python:
        if opts is None:
            return [{'keys': set(), 'weight': None, 'points': False,
                     'wvar': False, 'literal': True} for _ in range(n)]
        if isinstance(opts, dict):
            return [_opts_from_pydict(opts)] * n
        if callable(opts):
            raise ValueError(_OPT_CALLABLE_MSG)
        seq = list(opts)
        if len(seq) == 3 and all(_is_triple_entry(e) for e in seq):
            return [{'keys': 'triple', 'weight': None, 'points': False,
                     'wvar': False, 'literal': True} for _ in range(n)]
        if len(seq) != n:
            raise ValueError(
                "nquad: `opts` must be one dict, or one per axis; got %d for "
                "%d ranges" % (len(seq), n))
        out = []
        for e in seq:
            if callable(e):
                raise ValueError(_OPT_CALLABLE_MSG)
            if isinstance(e, dict):
                out.append(_opts_from_pydict(e))
            else:
                out.append({'keys': 'triple', 'weight': None,
                            'points': False, 'wvar': False, 'literal': True})
        return out
    # typing time
    if _is_none(opts):
        return [{'keys': set(), 'weight': None, 'points': False,
                 'wvar': False, 'literal': True} for _ in range(n)]
    if isinstance(opts, (types.LiteralStrKeyDict, types.DictType)):
        return [_opts_from_type(opts)] * n
    if isinstance(opts, types.BaseTuple):
        if len(opts) == 3 and all(isinstance(e, types.Number) for e in opts):
            return [{'keys': 'triple', 'weight': None, 'points': False,
                     'wvar': False, 'literal': True} for _ in range(n)]
        if any(isinstance(e, types.LiteralStrKeyDict) for e in opts):
            raise TypingError(_OPT_TUPLE_MSG)
        if len(opts) != n:
            raise TypingError(
                "nquad: `opts` must be one dict, or one per axis; got %d for "
                "%d ranges" % (len(opts), n))
        return [_opts_from_type(e) for e in opts]
    raise TypingError(_OPT_CALLABLE_MSG)


def _opt_expr(od, i, name, default, per_axis, cast=None):
    """Source reading one numeric setting for an axis.

    A literal dict knows at compile time whether the key is there, so the
    read is emitted only when it is.  A run-time dict is asked with ``in``,
    which costs one lookup and keeps the default when the key is absent.
    """
    sub = "opts[%d]" % i if per_axis else "opts"
    if od['keys'] == 'triple':
        slot = {'epsabs': 0, 'epsrel': 1, 'limit': 2}.get(name)
        if slot is None:                  # a triple carries no such setting
            return repr(default)
        return "np.float64(%s[%d])" % (sub, slot)
    if od['keys'] is None:
        return ("(np.float64(%s['%s']) if '%s' in %s else %r)"
                % (sub, name, name, sub, default))
    if name in od['keys']:
        return "np.float64(%s['%s'])" % (sub, name)
    return repr(default)


def _points_expr(od, i, per_axis):
    sub = "opts[%d]" % i if per_axis else "opts"
    if not od['points']:
        return "np.zeros(0)"
    return ("np.ascontiguousarray(np.asarray(%s['points'], "
            "dtype=np.float64)).ravel()" % sub)


def _runtime_opt_guards(od, i, per_axis, indent):
    """Source asking, at run time, what a homogeneous dict's TYPE cannot say.

    A dict whose values are all numbers is an ordinary ``DictType``, so its
    keys arrive with the value rather than with the type.  Three of scipy's
    answers depend on those keys, and each is asked here in scipy's own
    order: ``points``, then an unknown key, then ``complex_func``.
    """
    sub = "opts[%d]" % i if per_axis else "opts"
    src = []
    if od['keys'] is None:
        src += ["%sif 'points' in %s:" % (indent, sub),
                "%s    raise TypeError(%r)"
                % (indent, _NOT_ITER_MSG % od['vkind']),
                "%sfor _k in %s:" % (indent, sub),
                "%s    if _k not in _OPT_KEYS:" % indent,
                '%s        raise TypeError(_OPT_KEY_PRE + _k + "\'")'
                % indent,
                "%sif 'complex_func' in %s and %s['complex_func'] != 0:"
                % (indent, sub, sub),
                "%s    raise ValueError(_CF_MSG)" % indent]
    elif od.get('cf'):
        # A literal dict: the key is known now, the VALUE need not be.
        src += ["%sif %s['complex_func']:" % (indent, sub),
                "%s    raise ValueError(_CF_MSG)" % indent]
    return src


def _nq_args_tuple(args, who):
    """``args`` as scipy's ``integrate(*args)`` unpacks it."""
    if args is None:
        return ()
    if isinstance(args, tuple):
        return args
    try:
        return tuple(args)
    except TypeError:                                        # noqa: BLE001
        raise TypeError(_ARGS_ITER_MSG % who)


def _nq_args_types(args, who):
    """The element types of ``args`` as the chooser sees them."""
    if isinstance(args, types.Omitted):
        args = args.value
    if isinstance(args, types.Type):
        if _is_none(args):
            return ()
        if isinstance(args, types.BaseTuple):
            return tuple(args)
        if isinstance(args, types.Array):
            raise TypingError(_ARGS_ARRAY_MSG % who)
        raise TypingError(_ARGS_ITER_MSG % who)
    # An OMITTED default reaches a chooser as the RAW PYTHON VALUE.
    return tuple(typeof(v) for v in _nq_args_tuple(args, who))


from .._lib._typing import _is_none    # noqa: E402


from .._lib._typing import _lit_bool    # noqa: E402


# ---------------------------------------------------------------------
# ranges
# ---------------------------------------------------------------------
def _axis_from_range(r, i, n, kinds, from_python, who='nquad'):
    """One axis's limits, from ``ranges[i]``.

    Accepts scipy's two spellings and the ``(lo, hi)`` pair whose members are
    themselves callbacks.  The number of fixed coordinates an axis's callback
    sees is ``n - 1 - i``, which is zero for the outermost.
    """
    ax = _blank_axis()
    ncoords = n - 1 - i
    # scipy's own two answers, measured on 1.18: a range that is not
    # iterable at all is a TypeError ("cannot unpack non-iterable float
    # object"); one that is iterable but not a PAIR is a ValueError ("too
    # many values to unpack").  Inside @njit both are compile-time, so both
    # become TypingError.
    not_seq = TypeError if from_python else TypingError
    bad_len = ValueError if from_python else TypingError

    def as_disp(e):
        if from_python:
            return e if getattr(e, 'py_func', None) is not None else None
        return e if isinstance(e, types.Dispatcher) else None

    def pyfunc(e):
        return e.py_func if from_python else e.dispatcher.py_func

    def dispatcher(e):
        return e if from_python else e.dispatcher

    d = as_disp(r)
    if d is not None:
        form, ar = _callback_form(dispatcher(d), ncoords, kinds, 'range', i,
                                  who)
        ax['rng'] = pyfunc(d)
        ax['rng_form'] = form
        return ax
    if hasattr(r, 'address'):
        raise (TypeError if from_python else TypingError)(
            _LIMIT_NJIT_MSG % who)

    try:
        pair = tuple(r)
    except TypeError:                                        # noqa: BLE001
        raise not_seq(_RANGE_MSG)
    if len(pair) != 2:
        raise bad_len(_RANGE_MSG)

    for slot, e in zip(('lo', 'hi'), pair):
        d = as_disp(e)
        if d is not None:
            form, ar = _callback_form(dispatcher(d), ncoords, kinds, 'limit',
                                      i, who)
            ax[slot] = pyfunc(d)
            ax[slot + '_form'] = form
        elif from_python:
            if callable(e) or hasattr(e, 'address'):
                raise TypeError(_LIMIT_NJIT_MSG % who)
        elif not isinstance(e, types.Number):
            raise bad_len(_RANGE_MSG)
    return ax


def _const_limits(r, from_python):
    """The two constant limit VALUES of a range, 0.0 where a callback sits."""
    d = getattr(r, 'py_func', None) if from_python else None
    if from_python and (d is not None or callable(r)):
        return 0.0, 0.0
    if not from_python and isinstance(r, types.Dispatcher):
        return 0.0, 0.0
    out = []
    for e in tuple(r):
        if from_python:
            out.append(0.0 if callable(e) else float(e))
        else:
            out.append(0.0)
    return out[0], out[1]


# ---------------------------------------------------------------------
# public: nquad
# ---------------------------------------------------------------------
def nquad(func, ranges, args=None, opts=None, full_output=False):
    """Integration over ``n`` variables.

    ``ranges[0]`` is the INNERMOST integral and ``ranges[-1]`` the outermost.

    Parameters
    ----------
    func : @njit function
        Integrand, in either of two shapes.  The scalar shape takes one
        scalar per axis, innermost first, and then one argument per entry of
        ``args``: ``f(x0, x1, ..., a0, a1)``.  The array shape takes the
        coordinates as one array and the same ``args`` after it,
        ``f(x, a0, a1)`` with ``x[0]`` innermost.  The shape is resolved when
        the call compiles, from the function's arity and from whether it
        types against a coordinate array.
    ranges : tuple
        One entry per axis, innermost first.  Each is a ``(lo, hi)`` pair, or
        a plain ``@njit`` callable RETURNING such a pair.  A pair's members
        are each a float or a callback.  A callback for axis ``i`` receives
        the already-fixed OUTER coordinates ``x_{i+1} ... x_{n-1}``,
        innermost of those first, and then the ``args`` entries; or, in the
        array shape, one ``coords`` array holding them outermost first,
        followed by the same entries.  The tuple's LENGTH sets the nesting
        depth, so inside ``@njit`` it must be a tuple rather than a list or
        an array.
    args : tuple, optional
        Extra parameters, passed to ``func`` and to every range callback
        after their coordinates.  Each entry is a real number or an array of
        real numbers.  A value that is not a tuple becomes a one-item tuple.
    opts : dict, tuple of dicts, or None, optional
        Per-axis settings, innermost first, or one dict applied to every
        axis.  An ``opts`` entry is :func:`quad`'s own keyword arguments:
        ``epsabs``, ``epsrel``, ``limit``, ``points``, ``weight``, ``wvar``,
        ``wopts``, ``maxp1``, ``limlst``, ``complex_func``.  ``wopts`` is
        accepted and ignored.  A ``(epsabs, epsrel, limit)`` triple is
        accepted in place of a dict.

        ``weight`` and ``points`` pick the QUADPACK routine for their axis,
        so they are read when the call compiles.  Inside ``@njit`` that
        needs a dict whose values are of MIXED kinds, which is the one numba
        carries its keys on: ``{'weight': 'cos', 'wvar': 1.0}`` works,
        ``{'weight': 'cos'}`` alone does not.  A dict holding numbers only
        needs nothing, since those settings are read at run time.
    full_output : bool, optional
        Return the diagnostics as a third value.  Must be known when the
        call compiles, since it picks the number of return values.

    Returns
    -------
    value : float
        The integral.
    abserr : float
        The LARGEST error estimate over every level.
    out_dict : dict
        Only when ``full_output``.  One key, ``neval``, counting the
        innermost level's evaluations.

    Raises
    ------
    TypeError
        ``func`` or a limit callback is not a plain ``@njit`` function; an
        ``opts`` key is not one of the ten above; an ``opts`` ``points`` is
        not iterable; ``func`` returns a complex value.  ``TypingError``
        inside ``@njit`` for the first two, which are compile-time
        questions.
    ValueError
        ``ranges`` is empty; an ``opts`` sequence's length is not the number
        of ranges; a range entry is not a ``(lo, hi)`` pair; an ``args``
        entry is not a real number or an array of them; or QUADPACK refuses
        the settings of some level, which is :func:`quad`'s own set of
        conditions.

    See Also
    --------
    scipy.integrate.nquad : The scipy routine this mirrors.

    Notes
    -----
    **Prange-safe.**  Each call works in its own buffer.

    An ``opts`` entry is a dict or a triple, where scipy also accepts a
    callable returning a dict.  ``weight`` and ``points`` pick the QUADPACK
    routine for their axis and are read when the call compiles, so a value
    produced at run time cannot reach them.

    A per-axis ``opts`` carrying a ``weight`` cannot be passed from inside
    ``@njit``.  One dict applied to every axis works, and so does every
    per-axis spelling when ``nquad`` is called from python.

    An ``opts`` sequence of the wrong length raises, and an empty ``ranges``
    raises ``ValueError``.  scipy validates neither: it indexes ``opts``
    from the end, so three entries against two ranges uses the last two and
    drops the first, and an empty ``ranges`` reaches ``IndexError: list
    index out of range``.

    A ``(lo, hi)`` pair whose members are individually callbacks, a
    length-3 sequence of scalars read as one ``(epsabs, epsrel, limit)``
    triple for every axis, and the array integrand shape are additive.
    No scipy-shaped call reaches them.

    An ``opts`` dict whose values are all numbers cannot carry ``points``.
    A dict literal holding a tuple carries it,
    ``opts={'points': (0.5,), 'epsabs': 1.49e-8}``.

    ``complex_func`` is accepted only as false. scipy accepts the key and
    then compares a complex ``abserr`` against an int, so
    ``complex_func=True`` raises ``TypeError`` there for every integrand,
    real or complex.

    ``abserr`` is a float64 on every route.  scipy returns the int ``0`` on
    an integral whose range is degenerate.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def f(x0, x1):                    # scipy's shape, x0 innermost
    ...     return x0 * x1
    >>> @njit
    ... def run():
    ...     return si.nquad(f, ((0.0, 1.0), (0.0, 2.0)))
    >>> run()
    (0.9999999999999999, 1.1102230246251564e-14)

    A limit that depends on the axis outside it, and scipy's other range
    spelling side by side:

    >>> @njit
    ... def hi0(x1):                      # axis 0's upper limit
    ...     return 1.0 - x1
    >>> @njit
    ... def rng0(x1):                     # the same, as one callable
    ...     return 0.0, 1.0 - x1
    >>> @njit
    ... def tri():
    ...     a = si.nquad(f, ((0.0, hi0), (0.0, 1.0)))
    ...     b = si.nquad(f, (rng0, (0.0, 1.0)))
    ...     return a[0], b[0]
    >>> tri()
    (0.04166666666666667, 0.04166666666666667)

    Extra parameters travel as scipy's tuple and reach the integrand and the
    range callbacks alike:

    >>> @njit
    ... def g(x0, x1, c):
    ...     return c * x0 * x1
    >>> @njit
    ... def lo0(x1, c):
    ...     return 0.0 * c
    >>> @njit
    ... def scaled():
    ...     return si.nquad(g, ((lo0, 1.0), (0.0, 2.0)), (3.0,))[0]
    >>> scaled()
    3.0
    """
    return _nquad_py(func, ranges, args, opts, full_output, 'nquad')


def _nquad_py(func, ranges, args, opts, full_output, who):
    """:func:`nquad`'s python body, with the CALLER's name for the error
    messages.  ``dblquad`` and ``tplquad`` delegate here rather than to
    :func:`nquad`, so a caller who typed ``dblquad`` is told about
    ``dblquad``."""
    n = len(ranges)
    if n < 1:
        raise ValueError("%s: at least one integration range is "
                         "required" % who)
    fo = bool(full_output)
    if isinstance(func, (bool, int, np.integer)) or hasattr(func, 'address'):
        raise TypeError(_FUNC_NJIT_MSG % (who, who))
    disp = func if getattr(func, 'py_func', None) is not None else None
    if disp is None:
        raise TypeError(_FUNC_NJIT_MSG % (who, who))
    at = _nq_args_tuple(args, who)
    kinds = _arg_kinds(at)
    form, ar = _callback_form(disp, n, kinds, 'func', who=who)

    axes = [_axis_from_range(ranges[i], i, n, kinds, True, who)
            for i in range(n)]
    ods = _opts_list(opts, n, True)
    for ax, od in zip(axes, ods):
        ax['weight'] = od['weight']
        ax['points'] = od['points']
        ax['wvar'] = od['wvar']
    addr = build_chain(disp.py_func, form, kinds, axes, n, fo)

    ua = _pack_args(at)
    nu = ua.size
    hc = _HDR + _SLOTS * n
    ha = hc + n

    # `_opts_from_pydict` materialised the break points where scipy does,
    # ahead of the key check, so they are read off the description here.
    pts = [od.get('pts') if od.get('pts') is not None else np.zeros(0)
           for od in ods]
    total = ha + nu + int(sum(p.size for p in pts))
    buf = np.zeros(total)
    buf[_NU] = float(nu)
    buf[_TOT] = float(total)

    off = ha + nu
    for i, (ax, od) in enumerate(zip(axes, ods)):
        b = _HDR + _SLOTS * i
        src = (opts[i] if _seq_opts(opts, n) else opts) if opts is not None \
            else None
        buf[b + _EA] = _py_opt(src, 'epsabs', _DEF_EA, 0)
        buf[b + _ER] = _py_opt(src, 'epsrel', _DEF_ER, 1)
        buf[b + _LIM] = _py_opt(src, 'limit', _DEF_LIM, 2)
        buf[b + _MAXP1] = _py_opt(src, 'maxp1', _DEF_MAXP1, None)
        buf[b + _LIMLST] = _py_opt(src, 'limlst', _DEF_LIMLST, None)
        buf[b + _WC] = float(_weight_code(ax['weight']))
        lo, hi = _const_limits(ranges[i], True)
        buf[b + _LO], buf[b + _HI] = lo, hi
        w0, w1 = _py_wvar(src)
        buf[b + _W0], buf[b + _W1] = w0, w1
        buf[b + _POFF] = float(off)
        buf[b + _PLEN] = float(pts[i].size)
        buf[off:off + pts[i].size] = pts[i]
        off += pts[i].size
    buf[ha:ha + nu] = ua

    j = n - 1
    b = _HDR + _SLOTS * j
    ax = axes[j]
    pa = _py_arg_values(at)
    if ax['rng'] is not None:
        lo, hi = njit(ax['rng'])(*_py_call_args(ax, 'rng', pa))
    else:
        lo = (buf[b + _LO] if ax['lo'] is None
              else njit(ax['lo'])(*_py_call_args(ax, 'lo', pa)))
        hi = (buf[b + _HI] if ax['hi'] is None
              else njit(ax['hi'])(*_py_call_args(ax, 'hi', pa)))
    p = pts[j]
    if p.size:
        p = p[(p >= lo) & (p <= hi)]
    r = _quad_core(addr, lo, hi, buf, buf[b + _EA], buf[b + _ER],
                   int(buf[b + _LIM]), p, bool(ax['points']),
                   _weight_code(ax['weight']), buf[b + _W0], buf[b + _W1],
                   int(buf[b + _MAXP1]), int(buf[b + _LIMLST]))
    # An exception below the outermost level is recorded rather than raised,
    # because a `raise` inside a @cfunc is printed and swallowed.  Here it
    # can propagate, so it does.
    if buf[_ERR] != 0.0:
        _nq_reraise(buf, buf[_ERR])
    if not fo:
        _warn_ier(r[3], int(buf[b + _LIM]))
    if fo and n == 1:
        buf[_NEVAL] += float(r[2])
    ae = buf[_AERR]
    if ae < r[1]:
        ae = r[1]
    if fo:
        return r[0], ae, {'neval': int(buf[_NEVAL])}
    return r[0], ae


def _is_triple_entry(e):
    """True when ``e`` can be one number of an ``(epsabs, epsrel, limit)``.

    Row nquad-D2. ``numpy.ndim`` of a function object is 0, so a callable
    passed the test that reads a 3-element ``opts`` as one triple, and three
    callables over three ranges, which is scipy's own spelling, were read as
    a triple and reported from the float conversion instead of the refusal.
    """
    return (np.ndim(e) == 0 and not isinstance(e, dict) and not callable(e))


def _seq_opts(opts, n):
    """True when a python-level ``opts`` is one entry PER AXIS."""
    if opts is None or isinstance(opts, dict):
        return False
    seq = list(opts)
    if len(seq) == 3 and all(_is_triple_entry(e) for e in seq):
        return False
    return True


def _py_opt(src, name, default, slot):
    """One numeric setting from a python-level opts entry."""
    if src is None:
        return default
    if isinstance(src, dict):
        return float(src.get(name, default))
    if slot is None:                      # a triple carries no such setting
        return default
    return float(tuple(src)[slot])


def _py_wvar(src):
    if not isinstance(src, dict) or 'wvar' not in src:
        return 0.0, 0.0
    w = src['wvar']
    if np.ndim(w) == 0:
        return float(w), 0.0
    w = np.asarray(w, np.float64).ravel()
    return float(w[0]), float(w[1] if w.size > 1 else 0.0)


def _py_arg_values(at):
    """``args`` as the generated code will see it.

    An array's data crosses the buffer as float64, so the same conversion is
    applied here; a scalar keeps its own type, which is what the adapter
    casts it back to.
    """
    out = []
    for v in at:
        a = v if isinstance(v, np.ndarray) else np.asarray(v)
        out.append(np.ascontiguousarray(a, np.float64) if a.ndim else v)
    return tuple(out)


def _py_call_args(ax, slot, pa):
    """The arguments an OUTERMOST-axis callback is called with from python.

    The outermost axis has no fixed coordinates, so the list is ``args``
    alone; the array shape gets an empty ``coords`` in front of it.
    """
    if ax[slot + '_form'] == 'array':
        return (np.zeros(0),) + pa
    return pa


@overload(nquad, prefer_literal=True)
def _nquad_ovl(func, ranges, args=None, opts=None, full_output=False):
    """Compiled body for :func:`nquad`, one per call shape.

    The nesting depth is ``len(ranges)``, which numba knows here, so the whole
    chain of ``@cfunc`` levels is built now and only its outermost address
    survives into the compiled body.  Every routing decision is made here for
    the same reason.
    """
    who = 'nquad'
    if not isinstance(func, types.Dispatcher):
        raise TypingError(_FUNC_NJIT_MSG % (who, who))
    if not isinstance(ranges, types.BaseTuple):
        raise TypingError(
            "nquad: `ranges` must be a TUPLE. Its length is the nesting "
            "depth, which has to be known when this call compiles, so a list "
            "or an array cannot serve")
    n = len(ranges)
    if n < 1:
        raise TypingError("nquad: at least one integration range is required")
    fo = _lit_bool(full_output)
    if fo is None:
        raise TypingError(_FULL_OUTPUT_MSG)

    kinds = _arg_kinds_ty(_nq_args_types(args, who))
    form, ar = _callback_form(func.dispatcher, n, kinds, 'func', who=who)
    axes = [_axis_from_range(ranges[i], i, n, kinds, False, who)
            for i in range(n)]
    ods = _opts_list(opts, n, False)
    per_axis = (isinstance(opts, types.BaseTuple)
                and not (len(opts) == 3
                         and all(isinstance(e, types.Number) for e in opts)))
    for ax, od in zip(axes, ods):
        ax['weight'] = od['weight']
        ax['points'] = od['points']
        ax['wvar'] = od['wvar']
    addr = build_chain(func.dispatcher.py_func, form, kinds, axes, n, fo)

    hc = _HDR + _SLOTS * n
    ha = hc + n

    # `ranges` may hold callables beside floats, so it is a HETEROGENEOUS
    # tuple and a run-time loop cannot index it.  The packing is generated
    # here with literal indices, which is possible for the same reason the
    # chain is: n is known now.
    src = ["def impl(func, ranges, args=None, opts=None, full_output=False):"]
    # What a homogeneous `opts` dict's TYPE cannot answer, asked at run time
    # and before anything is integrated, which is where scipy answers it.
    for i, od in enumerate(ods):
        src += _runtime_opt_guards(od, i, per_axis, "    ")
    src += ["    ua = _pack_args(args)",
            "    nu = ua.size"]
    for i, od in enumerate(ods):
        src.append("    p%d = %s" % (i, _points_expr(od, i, per_axis)))
    tot = " + ".join(["%d + nu" % ha] + ["p%d.size" % i for i in range(n)])
    src += ["    total = %s" % tot,
            "    buf = np.zeros(total)",
            "    buf[%d] = np.float64(nu)" % _NU,
            "    buf[%d] = np.float64(total)" % _TOT,
            "    off = %d + nu" % ha]
    for i, od in enumerate(ods):
        b = _HDR + _SLOTS * i
        src += ["    buf[%d] = %s" % (b + _EA,
                                      _opt_expr(od, i, 'epsabs', _DEF_EA,
                                                per_axis)),
                "    buf[%d] = %s" % (b + _ER,
                                      _opt_expr(od, i, 'epsrel', _DEF_ER,
                                                per_axis)),
                "    buf[%d] = %s" % (b + _LIM,
                                      _opt_expr(od, i, 'limit', _DEF_LIM,
                                                per_axis)),
                "    buf[%d] = %s" % (b + _MAXP1,
                                      _opt_expr(od, i, 'maxp1', _DEF_MAXP1,
                                                per_axis)),
                "    buf[%d] = %s" % (b + _LIMLST,
                                      _opt_expr(od, i, 'limlst', _DEF_LIMLST,
                                                per_axis)),
                "    buf[%d] = %r" % (b + _WC,
                                      float(_weight_code(od['weight'])))]
        ax = axes[i]
        if ax['rng'] is None:
            if ax['lo'] is None:
                src.append("    buf[%d] = np.float64(ranges[%d][0])"
                           % (b + _LO, i))
            if ax['hi'] is None:
                src.append("    buf[%d] = np.float64(ranges[%d][1])"
                           % (b + _HI, i))
        if od['wvar']:
            sub = "opts[%d]" % i if per_axis else "opts"
            src += ["    _w = np.asarray(%s['wvar'], dtype=np.float64)"
                    ".ravel()" % sub,
                    "    buf[%d] = _w[0]" % (b + _W0),
                    "    buf[%d] = _w[1] if _w.size > 1 else 0.0"
                    % (b + _W1)]
        src += ["    buf[%d] = np.float64(off)" % (b + _POFF),
                "    buf[%d] = np.float64(p%d.size)" % (b + _PLEN, i),
                "    for _i in range(p%d.size):" % i,
                "        buf[off + _i] = p%d[_i]" % i,
                "    off += p%d.size" % i]
    src += ["    for _i in range(nu):",
            "        buf[%d + _i] = ua[_i]" % ha]

    j = n - 1
    ax = axes[j]
    unp, anames = _args_unpack_src(kinds, ha, "    ")
    src += unp
    src += _limit_src(ax, j, n, "    ", anames)
    if ax['points']:
        src += _points_src(j, "    ")
    # With one axis there is no level below, so this call IS the innermost.
    src += _quad_call(ax, j, fo, "addr", "    ", fo and n == 1, top=True)
    src += ["    ae = buf[%d]" % _AERR]
    if fo:
        src.append("    return v, ae, {'neval': int(buf[%d])}" % _NEVAL)
    else:
        src.append("    return v, ae")

    ns = {"np": np, "_quad_core": _quad_core, "_warn_ier": _warn_ier,
          "addr": addr, "_pack_args": _pack_args, "_nq_reraise": _nq_reraise,
          "_OPT_KEYS": _OPT_KEYS, "_OPT_KEY_PRE": _OPT_KEY_PRE,
          "_CF_MSG": _CF_MSG}
    for tag, key in (("_rng%d" % j, 'rng'), ("_lo%d" % j, 'lo'),
                     ("_hi%d" % j, 'hi')):
        if ax[key] is not None:
            ns[tag] = njit(ax[key])
    exec("\n".join(src), ns)
    return ns["impl"]


# ---------------------------------------------------------------------
# dblquad and tplquad, over the same chain
# ---------------------------------------------------------------------
#: Coordinate-only limit adapters, keyed by everything that shapes them.
_SWAPPED = {}

_LIMIT_SHAPE_MSG = (
    "%s: %s takes %d arguments. scipy calls it with the %d coordinate(s) "
    "alone, %s, so write it that way; a %d-argument form taking the `args` "
    "entries after them is accepted as well.")


def _limit_adapter(py, ncoords, na, swap, who, what, from_python):
    """scipy's ``dblquad``/``tplquad`` limit callback, as this chain calls it.

    Two things separate those two routines from ``nquad``.  scipy reaches
    ``gfun(args[0])`` and ``qfun(args[1], args[0])``, so the extra parameters
    never arrive there, where ``nquad``'s own range callbacks get
    ``fn_range(*args)``.  And ``qfun``/``rfun`` receive ``(x, y)``, outermost
    first, the opposite order from every other callback in the family.  Both
    are scipy's and both are reproduced by building the adapted function
    here, at typing time, rather than by indexing a ``*args`` per call.

    Returns ``None`` when the callback already has the shape the chain calls,
    which is the additive form taking the ``args`` entries after its
    coordinates.
    """
    ar = _arity(py)
    if ar is None:
        # A `*args` callback accepts any count, so the count cannot say
        # which shape it is written in.  scipy calls a limit callback with
        # its coordinates alone, so that is the shape it gets, and the
        # adapter below is what drops the extra parameters.
        pass_args = False
    else:
        #: the additive shape, which takes the `args` entries after its
        #: coordinates, as `nquad`'s own range callbacks do
        pass_args = bool(na) and ar == ncoords + na
        if not pass_args and ar != ncoords:
            coords = ", ".join(("x, y" if swap
                                else "x").split(", ")[:ncoords])
            raise (TypeError if from_python else TypingError)(
                _LIMIT_SHAPE_MSG % (who, what, ar, ncoords, coords,
                                    ncoords + na))
    if not swap and (pass_args or not na):
        return None                       # already the shape the chain calls
    key = (py, ncoords, na, swap, pass_args)
    hit = _SWAPPED.get(key)
    if hit is not None:
        return hit
    cs = ["c%d" % i for i in range(ncoords)]
    us = ["_u%d" % i for i in range(na)]
    params = ", ".join(cs + us)
    ic = list(reversed(cs)) if swap else list(cs)
    call = ", ".join(ic + us if pass_args else ic)
    ns = {"inner": njit(py)}
    exec("def swapped(%s):\n    return inner(%s)" % (params, call),
         ns)                                              # noqa: S102
    out = njit(ns["swapped"])
    _SWAPPED[key] = out
    return out


def _fix_limits(pairs, ncoords, na, swap, who, from_python):
    """Each of a limit PAIR, adapted to the shape the chain calls."""
    out = []
    for e, what in pairs:
        d = _as_disp(e, from_python)
        if d is None:
            out.append(e)
            continue
        w = _limit_adapter(_pyf(d, from_python), ncoords, na, swap, who,
                           what, from_python)
        out.append(e if w is None else w)
    return out


def _fix_limits_ty(ns, pairs, ncoords, na, swap, who):
    """The same at typing time, as the SOURCE naming each adapted limit."""
    out = []
    for e, what in pairs:
        d = _as_disp(e, False)
        if d is None:
            out.append(what)
            continue
        w = _limit_adapter(_pyf(d, False), ncoords, na, swap, who, what,
                           False)
        if w is None:
            out.append(what)
        else:
            ns["_s_" + what] = w
            out.append("_s_" + what)
    return out


def _as_disp(e, from_python):
    if from_python:
        return e if getattr(e, 'py_func', None) is not None else None
    return e if isinstance(e, types.Dispatcher) else None


def _pyf(e, from_python):
    return e.py_func if from_python else e.dispatcher.py_func


def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8,
            limit=50):
    """Double integral of ``func(y, x)`` over a curved region.

    ``int_a^b dx int_{gfun(x)}^{hfun(x)} func(y, x) dy``, an ``nquad`` of
    depth two.

    Parameters
    ----------
    func : @njit function ``func(y, x, *args) -> float``
        Integrand, with **the inner variable first**.
    a, b : float
        Outer (x) limits.  Either may be infinite.
    gfun, hfun : float, or @njit function ``g(x) -> float``
        Lower and upper inner (y) limits.  Either may be a constant or a
        callback, in any combination.  A callback receives the outer
        coordinate alone; a form taking the ``args`` entries after it is
        accepted as well.
    args : tuple, optional
        Extra parameters, passed to ``func``.  Each entry is a real number
        or an array of real numbers.
    epsabs, epsrel : float, optional
        Accuracy request, applied at both levels.
    limit : int, optional
        Subdivision cap, applied at both levels.  Default 50.

    Returns
    -------
    value : float
        Approximation to the double integral.
    abserr : float
        The LARGEST error estimate over both levels.

    Raises
    ------
    TypeError
        ``func`` or a limit callback is not a plain ``@njit`` function, a
        limit callback's arity is neither ``1`` nor ``1 + len(args)``, or
        ``func`` returns a complex value.  ``TypingError`` inside ``@njit``
        for the first two, which are compile-time questions.
    ValueError
        An ``args`` entry is not a real number or an array of them, or
        QUADPACK refuses the settings of either level, which is
        :func:`quad`'s own set of conditions.

    See Also
    --------
    scipy.integrate.dblquad : The scipy routine this mirrors.

    Notes
    -----
    ``limit`` has no scipy counterpart, and a limit callback taking the
    ``args`` entries after its coordinate is additive.  No scipy-shaped call
    reaches either.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def f(y, x):                      # inner variable FIRST
    ...     return x * y * y
    >>> @njit
    ... def hfun(x):
    ...     return 1.0 - x
    >>> @njit
    ... def run():
    ...     return si.dblquad(f, 0.0, 1.0, 0.0, hfun)[0]
    >>> run()
    0.016666666666666666
    """
    na = len(_arg_kinds(_nq_args_tuple(args, 'dblquad')))
    g, h = _fix_limits(((gfun, 'gfun'), (hfun, 'hfun')), 1, na, False,
                       'dblquad', True)
    return _nquad_py(func, ((g, h), (a, b)), args,
                     {'epsabs': epsabs, 'epsrel': epsrel, 'limit': limit},
                     False, 'dblquad')


@overload(dblquad)
def _dblquad_ovl(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8,
                 epsrel=1.49e-8, limit=50):
    """Compiled body for :func:`dblquad`: one ``nquad`` of depth two.

    The integrand is checked HERE, under the name the caller typed, before
    the delegation.  `nquad` would reject the same function a moment later
    and say `nquad`, naming a routine this caller never called.
    """
    if not isinstance(func, types.Dispatcher):
        raise TypingError(_FUNC_NJIT_MSG % ('dblquad', 'dblquad'))
    kinds = _arg_kinds_ty(_nq_args_types(args, 'dblquad'))
    _callback_form(func.dispatcher, 2, kinds, 'func', who='dblquad')
    na = len(kinds)
    ns = {"nquad": nquad}
    ge, he = _fix_limits_ty(ns, ((gfun, 'gfun'), (hfun, 'hfun')), 1, na,
                            False, 'dblquad')
    src = ["def impl(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8,",
           "         epsrel=1.49e-8, limit=50):",
           "    return nquad(func, ((%s, %s), (a, b)), args," % (ge, he),
           "                 (epsabs, epsrel, limit))"]
    exec("\n".join(src), ns)                              # noqa: S102
    return ns["impl"]


def tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs=1.49e-8,
            epsrel=1.49e-8, limit=50):
    """Triple integral of ``func(z, y, x)`` over a curved region.

    ``int_a^b dx int_{gfun(x)}^{hfun(x)} dy int_{qfun(x,y)}^{rfun(x,y)}
    func(z, y, x) dz``, an ``nquad`` of depth three.

    Parameters
    ----------
    func : @njit function ``func(z, y, x, *args) -> float``
        Integrand, **innermost variable first**.
    a, b : float
        Outer (x) limits.  Either may be infinite.
    gfun, hfun : float, or @njit function ``g(x) -> float``
        Lower and upper y limits.
    qfun, rfun : float, or @njit function ``q(x, y) -> float``
        Lower and upper z limits.  As callbacks they take ``(x, y)``,
        **outermost variable first**, the opposite order from the integrand.
        As with ``gfun``/``hfun``, a form taking the ``args`` entries after
        the coordinates is accepted as well.
    args : tuple, optional
        Extra parameters, passed to ``func``.  Each entry is a real number
        or an array of real numbers.
    epsabs, epsrel : float, optional
        Accuracy request, applied at all three levels.
    limit : int, optional
        Subdivision cap, applied at all three levels.  Default 50.

    Returns
    -------
    value : float
        Approximation to the triple integral.
    abserr : float
        The LARGEST error estimate over the three levels.

    Raises
    ------
    TypeError
        ``func`` or a limit callback is not a plain ``@njit`` function, a
        limit callback's arity is neither its coordinate count nor that plus
        ``len(args)``, or ``func`` returns a complex value.
        ``TypingError`` inside ``@njit`` for the first two, which are
        compile-time questions.
    ValueError
        An ``args`` entry is not a real number or an array of them, or
        QUADPACK refuses the settings of some level, which is :func:`quad`'s
        own set of conditions.

    See Also
    --------
    scipy.integrate.tplquad : The scipy routine this mirrors.

    Notes
    -----
    ``limit`` has no scipy counterpart, and a limit callback taking the
    ``args`` entries after its coordinates is additive.  No scipy-shaped
    call reaches either.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def f(z, y, x):                   # innermost variable FIRST
    ...     return x * y * y * z * z * z
    >>> @njit
    ... def run():                        # box 0<x<1, 0<y<2, 0<z<3
    ...     return si.tplquad(f, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0)[0]
    >>> run()
    26.999999999999996
    """
    na = len(_arg_kinds(_nq_args_tuple(args, 'tplquad')))
    q, r = _fix_limits(((qfun, 'qfun'), (rfun, 'rfun')), 2, na, True,
                       'tplquad', True)
    g, h = _fix_limits(((gfun, 'gfun'), (hfun, 'hfun')), 1, na, False,
                       'tplquad', True)
    return _nquad_py(func, ((q, r), (g, h), (a, b)), args,
                     {'epsabs': epsabs, 'epsrel': epsrel, 'limit': limit},
                     False, 'tplquad')


@overload(tplquad)
def _tplquad_ovl(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs=1.49e-8,
                 epsrel=1.49e-8, limit=50):
    """Compiled body for :func:`tplquad`: one ``nquad`` of depth three.

    ``qfun``/``rfun`` are handed to the chain already swapped, so the level
    that calls them needs to know nothing about scipy's argument order.
    """
    if not isinstance(func, types.Dispatcher):
        raise TypingError(_FUNC_NJIT_MSG % ('tplquad', 'tplquad'))
    # Checked here, under the caller's own name; see `_dblquad_ovl`.
    kinds = _arg_kinds_ty(_nq_args_types(args, 'tplquad'))
    _callback_form(func.dispatcher, 3, kinds, 'func', who='tplquad')
    na = len(kinds)
    ns = {"nquad": nquad}
    qexpr, rexpr = _fix_limits_ty(ns, ((qfun, 'qfun'), (rfun, 'rfun')), 2, na,
                                  True, 'tplquad')
    gexpr, hexpr = _fix_limits_ty(ns, ((gfun, 'gfun'), (hfun, 'hfun')), 1, na,
                                  False, 'tplquad')
    src = ["def impl(func, a, b, gfun, hfun, qfun, rfun, args=(),",
           "         epsabs=1.49e-8, epsrel=1.49e-8, limit=50):",
           "    return nquad(func, ((%s, %s), (%s, %s), (a, b)), args,"
           % (qexpr, rexpr, gexpr, hexpr),
           "                 (epsabs, epsrel, limit))"]
    exec("\n".join(src), ns)
    return ns["impl"]
