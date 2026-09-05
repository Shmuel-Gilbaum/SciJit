"""LSODA integration over a time grid: ``scijit.integrate.odeint``.

The input and output variations are resolved at compile time with
``@overload``.

**The callback is a plain ``@njit`` function.** ``func(y, t, *args)``, one
parameter per entry of ``args``, returning a new 1-D float64 array of ``neq``
derivatives.  ``tfirst=True`` switches the order to ``func(t, y, *args)``, as
it does in scipy.  The ``@cfunc`` LSODA calls back through is built here, at
typing time, and its address frozen into the compiled body; a ``@cfunc``
address passed in the ``func`` slot is refused rather than reinterpreted.

**``args`` is scipy's tuple.**  Its entries cross LSODA's single ``double *``
in a buffer this module packs and the generated adapter unpacks, so an entry
may be a float, an int, a bool or an array of any rank and arrives in the
callback with its own type and shape.  A list, an array or a bare number in
the ``args`` slot is refused with scipy's own "Extra arguments must be in a
tuple.", which is what ``PyTuple_Check`` gives on scipy's side.

**Return shape is scipy's.** ``full_output=False`` (the default) returns the
solution array alone; ``full_output=True`` returns ``(y, infodict)``.
``success_out=True`` appends a bool, which scipy has no counterpart for.

**infodict is a namedtuple**, so ``info['nfe']`` is spelled ``info.nfe``.
Field names and contents are scipy's 13 keys.
"""
import warnings
from collections import namedtuple

import numpy as np
from numba import carray, cfunc, njit, objmode, typeof, types
from numba.core.errors import TypingError
from numba.extending import overload

from ._odepack import (_as_1d, _jac_route, _lsoda_jac_sig, _lsoda_sig,
                       _lsoda_yfirst_sig, _msg, _run_odeint)
from .._lib._typing import _K_BOOL, _K_FLOAT, _K_INT, _is_none, _lit_bool
from .._lib._typing import _arg_kinds as _arg_kinds_base
from .._lib._typing import _arg_kinds_ty as _arg_kinds_ty_base

__all__ = ['odeint', 'InfoDict', 'ODEintWarning', 'ODEpackError']


class ODEintWarning(Warning):
    """Warning class :func:`odeint` uses.

    Raised on every abnormal exit, carrying the ``istate`` message and
    "Run with full_output = 1 to get quantitative information.", and on a
    normal exit under ``printmessg``.

    See Also
    --------
    scipy.integrate.ODEintWarning : The scipy warning this mirrors.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def rhs(y, t):
    ...     return -y
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     _ = si.odeint(rhs, np.array([1.0]), np.array([0.0, 1.0]),
    ...                   printmessg=1)
    >>> issubclass(w[0].category, si.ODEintWarning)
    True
    """


class ODEpackError(Exception):
    """Exception class :func:`odeint` uses for ODEPACK binding checks.

    Raised for an ``args`` that is not a tuple and for a tolerance of the
    wrong length or rank. Derives straight from ``Exception``.

    Notes
    -----
    scipy raises its private ``scipy.integrate._odepack.error`` for the same
    checks, likewise derived from ``Exception``.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def rhs(y, t):
    ...     return -y
    >>> try:                                  # args must be a tuple
    ...     si.odeint(rhs, np.array([1.0]), np.array([0.0, 1.0]), [3.0])
    ... except si.ODEpackError as e:
    ...     print(e)
    Extra arguments must be in a tuple.
    """

































#: scipy's ``full_output`` infodict, as a namedtuple.  Field names are
#: scipy's 13 dict keys, in ``sorted()`` order.
InfoDict = namedtuple(
    'InfoDict',
    ['hu', 'imxer', 'leniw', 'lenrw', 'message', 'mused', 'nfe', 'nje',
     'nqu', 'nst', 'tcur', 'tolsf', 'tsw'])

#: scipy's ``rtol``/``atol`` default, ``sqrt(finfo(float64).eps)``.
_TOL = 1.49012e-8

#: scipy's own wording, ``_odepackmodule.c:497``, embedded newline and five
#: spaces included.
_TOL_MSG = ("Tolerances must be an array of the same length as the\n"
            "     number of equations or a scalar.")

_LITERAL_MSG = (
    "odeint: %s must be a compile-time constant inside @njit. It selects "
    "%s, so a value only known while the code runs cannot be typed. Call "
    "odeint from python for a run-time flag.")

_RTOL_RANK_MSG = "Error converting relative tolerance."
_ATOL_RANK_MSG = "Error converting absolute tolerance."

#: `args` was supplied but the plain @njit right-hand side has nowhere to put
#: it. scipy fails the same call with "takes 2 positional arguments but 3 were
#: given", from Python's own argument binding.
_ARGS_MSG = ("args= was given but the right-hand side takes only (y, t), so "
             "the parameters have nowhere to go. Write f(y, t, args) and read "
             "them as args[i], or close over them and drop args=")






# --------------------------------------------------------------------------
# scipy's `args` tuple, across a single ``double *``
#
# LSODA calls back through one pointer, so the parameters have to be flat.
# What crosses the pointer does not decide what the caller writes: the
# adapter that unflattens the buffer is generated at TYPING time, where the
# tuple's arity and every element's type are compile-time constants.  Both
# halves are generated from the same kinds, so they cannot drift.  Layout:
# slot 0 is ``neq``, slot 1 the payload length, a scalar takes one slot and
# an array its shape followed by its data.
# --------------------------------------------------------------------------
#: kind codes: a float, an integer and a boolean scalar; ``k >= 1`` is an
#: array of that rank.  A scalar keeps its own type through the buffer,
#: which is a cast in the adapter; an array's data crosses as float64.

#: scipy's own wording, ``_odepackmodule.c``: ``args`` is checked with
#: ``PyTuple_Check``, so a list, an array, ``None`` and a bare number are all
#: refused.
_ARGS_TUPLE_MSG = "Extra arguments must be in a tuple."

_ARGS_KIND_MSG = (
    "odeint: an args entry must be a real number or an array of real "
    "numbers. The extra parameters cross LSODA's double* argument slot, "
    "which carries no other type.")

_ADDRESS_MSG = (
    "odeint: func must be a plain @njit function, called as f(y, t, *args) "
    "with one argument per entry of args, or f(t, y, *args) under "
    "tfirst=True. The .address of a @cfunc is not accepted: pass the @njit "
    "function itself and odeint builds the callback.")

_DFUN_ADDRESS_MSG = (
    "odeint: Dfun must be a plain @njit function, called as j(y, t, *args) "
    "with one argument per entry of args, or j(t, y, *args) under "
    "tfirst=True. The .address of a @cfunc is not accepted: pass the @njit "
    "function itself and odeint builds the callback.")

_ARITY_MSG = (
    "odeint: %s takes %%d argument(s), and an args of length %%d calls it as "
    "%s, which needs %%d.")


def _arity_msg(what, tfirst):
    """The arity refusal, naming the argument order the call will use."""
    order = "%s(t, y, *args)" if tfirst else "%s(y, t, *args)"
    return _ARITY_MSG % (what, order % ('f' if what == 'func' else 'j'))


def _arg_kinds(args):
    """Element kinds of a Python-level ``args`` tuple."""
    return _arg_kinds_base(args, _ARGS_KIND_MSG)


def _arg_kinds_ty(tys):
    """Element kinds of the numba TYPES of ``args``, at typing time."""
    return _arg_kinds_ty_base(tys, _ARGS_KIND_MSG)


def _args_tuple(args):
    """``args`` as a tuple, refusing every other spelling as scipy does.

    ``PyTuple_Check`` is the whole rule: a list, an array, ``None`` and a
    bare number all reach "Extra arguments must be in a tuple.".
    """
    if isinstance(args, tuple):
        return args
    raise ODEpackError(_ARGS_TUPLE_MSG)


def _args_types(args):
    """The element types of ``args`` as the chooser sees them."""
    if isinstance(args, types.Omitted):
        args = args.value
    if isinstance(args, types.Type):
        if isinstance(args, types.BaseTuple):
            return tuple(args)
        raise TypingError(_ARGS_TUPLE_MSG)
    # An OMITTED default reaches a chooser as the RAW PYTHON VALUE.
    if not isinstance(args, tuple):
        raise TypingError(_ARGS_TUPLE_MSG)
    return tuple(typeof(v) for v in args)


def _check_arity(py, nargs, what, tfirst):
    """scipy's calling convention, checked where the message can name it.

    scipy reaches ``func(y, t, *args)`` and lets Python report a mismatch.
    A ``@cfunc`` swallows the same report, so it is made here.
    """
    code = getattr(py, '__code__', None)
    if code is None or code.co_flags & 0x04:             # *args: no fixed arity
        return
    want = 2 + nargs
    ndef = len(getattr(py, '__defaults__', None) or ())
    if code.co_argcount - ndef <= want <= code.co_argcount:
        return
    raise TypeError(_arity_msg(what, tfirst)
                    % (code.co_argcount, nargs, want))


def _callback_target(func, msg):
    """The plain Python function behind a callback.

    An integer, a ``@cfunc`` object and anything else carrying an
    ``.address`` are refused rather than reinterpreted: an integer meant as
    a function pointer and read as a value has no signal at all.

    The arity is NOT checked here.  scipy tests that the callbacks are
    callable ahead of the ranks and the tolerances and finds a wrong arity
    only when it calls one, so the two halves sit at different points of
    :func:`odeint`'s check order.
    """
    if isinstance(func, (bool, int, np.integer)) or hasattr(func, 'address'):
        raise TypeError(msg)
    py = getattr(func, 'py_func', func)
    if not callable(py):
        raise TypeError(msg)
    return py


def _pack_args(args, neq):
    """``args``, flattened into the buffer the adapters unflatten.

    A scalar INTEGER entry crosses its slot as the int64 BITS, read back
    through the same view by the adapter, so a value above ``2**53`` is not
    rounded on the way.  The slot's kind is known per argument when the
    adapter is generated, so nothing has to be decided from the bits.
    """
    parts = []
    for v in args:
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
    out = np.empty(tot + 2, np.float64)
    out[0] = float(neq)
    out[1] = float(tot)
    o = 2
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
def _pack_args_ovl(args, neq):
    """Compiled body for :func:`_pack_args`, generated from the kinds."""
    kinds = _arg_kinds_ty(_args_types(args))
    lines = ["def impl(args, neq):"]
    sizes = []
    for i, k in enumerate(kinds):
        src = "args[%d]" % i
        if k < 0:
            sizes.append("1")
        else:
            lines.append("    v%d = np.asarray(%s).astype(np.float64).ravel()"
                         % (i, src))
            lines.append("    s%d = %s.shape" % (i, src))
            sizes.append("%d + v%d.size" % (k, i))
    lines.append("    tot = %s" % (" + ".join(sizes) if sizes else "0"))
    lines.append("    out = np.empty(tot + 2, np.float64)")
    lines.append("    out[0] = np.float64(neq)")
    lines.append("    out[1] = np.float64(tot)")
    lines.append("    o = 2")
    for i, k in enumerate(kinds):
        src = "args[%d]" % i
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


def _unpack_lines(kinds):
    """Statements that rebuild the ``args`` tuple out of the flat buffer.

    Returns the lines and the names each element landed in.  Shared by the
    right-hand-side adapter and the Jacobian adapter, so one description of
    the layout serves both.
    """
    lines = ["    n = int(p[1])",
             "    a = carray(p, n + 2)",
             "    o = 2"]
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
    return lines, names


_RHS_ADAPTERS = {}


def _rhs_src(kinds, tfirst):
    """Source of the ``@cfunc`` body that unflattens ``args`` and calls in.

    A wrong-length return writes NOTHING rather than reading past the end of
    ``out``: numba has no bounds checking, so the loop would otherwise copy
    whatever follows the array into ``ydot`` and LSODA would integrate it.
    Writing nothing leaves the caller's sentinel intact, which is what the
    pre-flight probe looks for.
    """
    head = "t, y_ptr" if tfirst else "y_ptr, t"
    lines = ["def adapter(%s, dy_ptr, p):" % head,
             "    neq = int(p[0])"]
    body, names = _unpack_lines(kinds)
    lines += body
    lines.append("    yv = carray(y_ptr, neq)")
    call = ", ".join((["t", "yv"] if tfirst else ["yv", "t"]) + names)
    lines += ["    out = inner(%s)" % call,
              "    if out.size == neq:",
              "        for i in range(neq):",
              "            dy_ptr[i] = out[i]"]
    return "\n".join(lines)


def _adapter_rhs_args(py, tfirst, kinds):
    """``@cfunc`` around a plain ``@njit`` ``f(y, t, *args) -> ydot``.

    Built at TYPING time and its address frozen into the compiled body, so
    the callback stops being user-facing.  The cache OWNS the ``@cfunc``;
    dropping the reference would dangle the baked-in address.
    """
    key = (py, bool(tfirst), kinds)
    hit = _RHS_ADAPTERS.get(key)
    if hit is not None:
        return hit
    ns = {'carray': carray, 'inner': njit(py), 'np': np}
    exec(_rhs_src(kinds, bool(tfirst)), ns)              # noqa: S102
    adapter = cfunc(_lsoda_sig if tfirst else _lsoda_yfirst_sig)(ns['adapter'])
    _RHS_ADAPTERS[key] = adapter
    return adapter


_JAC_ARGS_ADAPTERS = {}


def _jac_src(kinds, tfirst, banded, coldiv):
    """Source of the Jacobian ``@cfunc`` body.  Layouts as in
    :func:`_adapter_jac`; a wrong SHAPE writes nothing, for the reason
    :func:`_rhs_src` gives."""
    lines = ["def adapter(t, y_ptr, pd_ptr, neq, ml, mu, nrowpd, p):",
             "    nn = int(neq)"]
    body, names = _unpack_lines(kinds)
    lines += body
    lines.append("    yv = carray(y_ptr, nn)")
    call = ", ".join((["t", "yv"] if tfirst else ["yv", "t"]) + names)
    lines.append("    j = inner(%s)" % call)
    if banded:
        lines.append("    mband = int(ml) + int(mu) + 1")
        rows = "mband"
        if coldiv:
            cond = "j.shape[0] == nn and j.shape[1] == mband"
            put = "pd_ptr[r + c * nr] = j[c, r]"
        else:
            cond = "j.shape[0] == mband and j.shape[1] == nn"
            put = "pd_ptr[r + c * nr] = j[r, c]"
    else:
        rows = "nn"
        cond = "j.shape[0] == nn and j.shape[1] == nn"
        put = ("pd_ptr[r + c * nr] = j[c, r]" if coldiv
               else "pd_ptr[r + c * nr] = j[r, c]")
    lines += ["    if %s:" % cond,
              "        nr = int(nrowpd)",
              "        for c in range(nn):",
              "            for r in range(%s):" % rows,
              "                %s" % put]
    return "\n".join(lines)


def _adapter_jac_args(py, tfirst, kinds, banded, coldiv):
    """``@cfunc`` around a plain ``@njit`` ``j(y, t, *args)``."""
    key = (py, bool(tfirst), kinds, bool(banded), bool(coldiv))
    hit = _JAC_ARGS_ADAPTERS.get(key)
    if hit is not None:
        return hit
    ns = {'carray': carray, 'inner': njit(py), 'np': np}
    exec(_jac_src(kinds, bool(tfirst), bool(banded), bool(coldiv)),
         ns)                                             # noqa: S102
    adapter = cfunc(_lsoda_jac_sig)(ns['adapter'])
    _JAC_ARGS_ADAPTERS[key] = adapter
    return adapter


@njit
def _check_rhs_shape(out, neq):
    """Raise unless the right-hand side returned ``neq`` derivatives.

    scipy checks this inside its callback thunk on every call and reports
    the same two faults.  The check cannot live in the ``@cfunc`` adapter,
    where an exception is printed and swallowed, so it runs once here on a
    direct call to the plain ``@njit`` function.
    """
    if out.ndim != 1:
        raise RuntimeError(
            "The array returned by func must be one-dimensional, but got "
            "ndim=" + str(out.ndim) + ".")
    if out.size != neq:
        raise RuntimeError(
            "The size of the array returned by func (" + str(out.size) +
            ") does not match the size of y0 (" + str(neq) + ").")


@njit
def _check_jac_full_rt(j, n):
    """scipy's full-Jacobian shape report, ``_odepackmodule.c:333-371``."""
    if j.shape[0] != n or j.shape[1] != n:
        raise RuntimeError(
            "Expected a Jacobian array with shape (" + str(n) + ", " +
            str(n) + "), but got (" + str(j.shape[0]) + ", " +
            str(j.shape[1]) + ")")


@njit
def _check_jac_band_rt(j, n, ml, mu):
    """scipy's banded-Jacobian shape report."""
    if j.shape[0] != ml + mu + 1 or j.shape[1] != n:
        raise RuntimeError(
            "Expected a banded Jacobian array with shape (" +
            str(ml + mu + 1) + ", " + str(n) + "), but got (" +
            str(j.shape[0]) + ", " + str(j.shape[1]) + ")")


@njit
def _check_jac_band_t_rt(j, n, ml, mu):
    """``col_deriv`` banded: the packed array arrives transposed, which is
    the ``jac_transpose`` swap at ``_odepackmodule.c:326-331``."""
    if j.shape[0] != n or j.shape[1] != ml + mu + 1:
        raise RuntimeError(
            "Expected a banded Jacobian array with shape (" + str(n) +
            ", " + str(ml + mu + 1) + "), but got (" + str(j.shape[0]) +
            ", " + str(j.shape[1]) + ")")


# One evaluation of a callback, in each of scipy's two argument orders.
# The order is a compile-time property, so it is a NAME rather than a flag:
# a `tfirst` passed as an ordinary argument reaches the chooser as
# `types.boolean` and carries no value to branch on.
def _call_y(fn, y, t, args):
    """``fn(y, t, *args)``, scipy's default order."""
    return fn(y, t, *args)


@overload(_call_y)
def _call_y_ovl(fn, y, t, args):
    def impl(fn, y, t, args):
        return fn(y, t, *args)
    return impl


def _call_t(fn, y, t, args):
    """``fn(t, y, *args)``, the ``tfirst=True`` order."""
    return fn(t, y, *args)


@overload(_call_t)
def _call_t_ovl(fn, y, t, args):
    def impl(fn, y, t, args):
        return fn(t, y, *args)
    return impl


# --------------------------------------------------------------------------
# checks scipy performs in Python before reaching the C layer
# --------------------------------------------------------------------------
@njit
def _check_monotonic(t):
    """scipy ``_odepack_py.odeint`` lines 244-248, ported.

    scipy writes ``(dt >= 0).all() or (dt <= 0).all()``, and a NaN gap is
    False under both, so a NaN in ``t`` fails the test (row ``odeint``-D3).
    Testing ``d < 0.0`` instead reads NaN as neither, which passed the whole
    array through and returned NaN states with no signal.

    ``numpy.diff`` takes the difference along the LAST axis, so a rank >= 2
    ``t`` is tested row by row on both sides.  It reaches this function
    because scipy runs ``numpy.diff`` before the rank test.
    """
    d = np.diff(t)
    if not ((d >= 0.0).all() or (d <= 0.0).all()):
        raise ValueError(
            "The values in t must be monotonically increasing or "
            "monotonically decreasing; repeated values are allowed.")




#: appended to the message on an abnormal exit, ``_odepack_py.py:257``.
_FULL_OUTPUT_HINT = " Run with full_output = 1 to get quantitative information."


def _emit_odeint_warning(istate, printmessg, stacklevel=2):
    """scipy's ``_odepack_py.py:255-262``: warn on every abnormal exit, and
    on a normal one only when ``printmessg`` asks.

    The text is built by calling the ``@njit`` message table's ``.py_func``,
    so there is one table rather than two.  ``stacklevel`` counts frames from
    ``warnings.warn``: 2 reaches the caller through an ``objmode`` block,
    whose caller frame is the compiled function's own, and the Python entry
    passes 3 because it has one frame more.
    """
    if istate < 0:
        warnings.warn(_msg.py_func(istate) + _FULL_OUTPUT_HINT,
                      ODEintWarning, stacklevel=stacklevel)
    elif printmessg:
        warnings.warn(_msg.py_func(istate), ODEintWarning,
                      stacklevel=stacklevel)


@njit
def _odeint_warn(istate, printmessg):
    """:func:`_emit_odeint_warning` from compiled code.

    The ``objmode`` block sits on the branch that warns, so a successful run
    without ``printmessg`` never enters it and never takes the GIL.
    """
    if istate < 0 or printmessg != 0:
        with objmode():
            _emit_odeint_warning(istate, printmessg)










# --------------------------------------------------------------------------
# compile-time helpers
# --------------------------------------------------------------------------
def _lit_int(v):
    """Compile-time int out of whatever numba hands the overload, or ``None``
    when the value is only known at run time."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, types.Omitted):
        return None if v.value is None else int(v.value)
    if isinstance(v, types.IntegerLiteral):
        return int(v.literal_value)
    if isinstance(v, types.BooleanLiteral):
        return int(v.literal_value)
    return None






def _is_banded_odeint(ml, mu):
    """Whether ``odeint``'s ``ml``/``mu`` select a banded Jacobian.

    ``odeint`` uses a NEGATIVE value where ``solve_ivp`` uses ``None``:
    ``_odepack_py.py:237`` turns ``ml=None`` into ``ml=-1``, and its
    docstring reads "if either of these are not None or non-negative, then
    the Jacobian is assumed to be banded".  Measured: ``odeint(..., ml=-1)``
    is bit-identical to ``odeint`` with no ``ml`` at all, nfe 598 both.

    Returns ``None`` when the answer needs a value that is only known at run
    time, which happens for a non-literal ``ml``/``mu`` inside ``@njit``.
    """
    known = True
    hit = False
    for v in (ml, mu):
        if _is_none(v):
            continue
        n = _lit_int(v)
        if n is None:
            known = False
        elif n >= 0:
            hit = True
    if hit:
        return True
    if known:
        return False
    # A runtime value that could be either. The caller resolves it in the
    # compiled body, where the value exists.
    return None




_SEQ = (types.Array, types.BaseTuple, types.List)
if hasattr(types, 'ListType'):
    _SEQ = _SEQ + (types.ListType,)


def _is_vec(v):
    """True when the tolerance was spelled as a sequence, not a scalar.

    scipy rejects a length-1 sequence against ``neq > 1``; a bare float is
    always legal.  The distinction is only visible in the TYPE, so it is
    resolved here rather than from ``.size``.

    A 0-d array is a SCALAR, which is row ``odeint``-D5:
    ``_odepackmodule.c`` converts the tolerance with ``min_depth=0`` and
    takes ``PyArray_NDIM(...) == 0`` as the scalar case, so
    ``rtol=np.array(1e-6)`` is legal against any ``neq``.
    """
    if isinstance(v, np.ndarray):
        return v.ndim > 0
    if isinstance(v, (list, tuple)):
        return True
    if isinstance(v, types.Omitted):
        return _is_vec(v.value)
    if isinstance(v, types.Array):
        return v.ndim > 0
    return isinstance(v, _SEQ)


def _rank(v):
    """Array rank of an argument, 0 for a scalar, 1 for a sequence.

    A tuple or a list is rank 1 whichever side it arrives from, as a
    python object or as the numba TYPE of one. Reading a
    ``types.UniTuple`` as rank 0 made ``odeint(f, y0, (0.0, 0.5, 1.0))``
    look like a scalar ``t`` inside ``@njit``.
    """
    if isinstance(v, np.ndarray):
        return v.ndim
    if isinstance(v, types.Array):
        return v.ndim
    if isinstance(v, types.Omitted):
        return _rank(v.value)
    if isinstance(v, (list, tuple)) or isinstance(v, _SEQ):
        return 1
    return 0


_Y0_RANK_MSG = "Initial condition y0 must be one-dimensional."
_T_RANK_MSG = "Output times t must be one-dimensional."

#: Row ``odeint``-D4. scipy reaches ``numpy.diff(t)`` before any check of its
#: own, so a scalar or 0-d ``t`` is reported by numpy, in numpy's words.
_T_SCALAR_MSG = "diff requires input that is at least one dimensional"

#: Row ``odeint``-D18. ``_odepackmodule.c`` converts ``tcrit`` with
#: ``max_depth=1``; rank >= 2 fails the conversion.
_TCRIT_RANK_MSG = "Error constructing critical times."


# --------------------------------------------------------------------------
# public front end
# --------------------------------------------------------------------------
def odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0,
           ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0,
           hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
           mxords=5, printmessg=0, tfirst=False, success_out=False):
    """Integrate a system of ODEs with LSODA over a time grid.

    LSODA switches between Adams (non-stiff) and BDF (stiff) methods
    automatically. The result is time-major: row ``i`` is the state at
    ``t[i]``.

    Parameters
    ----------
    func : @njit function
        The right-hand side, ``f(y, t, *args)``, RETURNING a new 1-D float64
        array of ``neq`` derivatives.  One parameter per entry of ``args``.
        Under ``tfirst=True`` the order is ``f(t, y, *args)``.  A plain
        ``@njit`` function.
    y0 : float, sequence or ndarray
        Initial state at ``t[0]``. A scalar becomes length 1. Rank >= 2
        raises ``ValueError``.
    t : sequence or ndarray
        Times to report at, monotonic; repeated values are allowed.
        ``t[0]`` is the initial time. Rank >= 2 raises ``ValueError``. An
        empty ``t`` returns a ``(0, neq)`` array.
    args : tuple, optional
        Extra parameters, passed to the callback as separate arguments.  An
        entry may be a float, an int, a bool or an array of any rank, and
        arrives with its own type and shape.  Anything other than a tuple
        raises ``ODEpackError("Extra arguments must be in a tuple.")``.  A
        callback whose arity does not fit raises ``TypeError`` naming both
        counts.
    Dfun : @njit function or None, optional
        The Jacobian ``d f / d y``, ``j(y, t, *args)``, or ``j(t, y, *args)``
        under ``tfirst=True``.  ``None`` (default) leaves LSODA to build one
        by finite differences, at ``neq`` extra right-hand-side evaluations
        per rebuild.  A plain ``@njit`` function, as ``func`` is.

        Shape ``(neq, neq)`` with ``jac[i, j] = d f_i / d y_j``, or, with
        ``ml``/``mu`` set, the packed ``(ml + mu + 1, neq)`` form with
        ``jac[mu + i - j, j] = d f_i / d y_j``. ``col_deriv`` transposes
        both. A wrong shape raises ``RuntimeError``.

        Measured on Robertson, 3 states, ``rtol=1e-8 atol=1e-10``:
        ``nfe`` 502 without and 424 with, ``nje`` 26 either way, values
        agreeing to 1.369777e-11. The 78 evaluations removed are exactly
        the ``26 * 3`` the finite-difference rebuilds cost.
    col_deriv : int, optional
        Non-zero means ``Dfun`` returns the transpose: ``(neq, neq)`` read
        as ``jac[j, i] = d f_i / d y_j``, and with ``ml``/``mu`` the packed
        array transposed to ``(neq, ml + mu + 1)``. Inert with ``Dfun=None``.
        Must be a compile-time constant inside ``@njit``.
    full_output : bool, optional
        ``False`` (default) returns ``y``. ``True`` returns
        ``(y, infodict)``. Must be a compile-time constant inside
        ``@njit``.
    ml, mu : int or None, optional
        Half-bandwidths of the Jacobian, counting the sub- and
        super-diagonals and excluding the main diagonal. Setting either
        selects LSODA's banded Jacobian, ``jt=5`` with ``Dfun=None`` and
        ``jt=4`` with a ``Dfun``; the unset one becomes 0. ``None``
        (default) on both keeps the full Jacobian.  A negative value means
        the same as ``None``.

        Cost, on an 80-point heat equation by method of lines whose
        Jacobian is tridiagonal, ``rtol=1e-6 atol=1e-9``: LSODA rebuilds
        the Jacobian 6 times and a full finite-difference rebuild spends
        ``neq = 80`` right-hand-side evaluations each, against
        ``ml + mu + 1 = 3`` for the banded one. ``nfe`` 597 against 135.
    rtol, atol : float, sequence or None, optional
        Relative and absolute tolerances. ``None`` (the default) means
        1.49012e-8, ``sqrt(finfo(float64).eps)``. Either may be a scalar or a
        vector of length ``neq``; the four combinations map to LSODA's
        ``itol`` 1, 2, 3 and 4. A sequence of any other length raises
        ``ODEpackError``, and rank >= 2 raises
        ``ODEpackError("Error converting relative tolerance.")``.
    tcrit : float, sequence or None, optional
        Critical times the integrator must not step past, in the order it
        will meet them.  The first is in force from the first leg; the
        integrator advances to the next each time an output time passes the
        current one, and steps freely once the sequence is exhausted.
    h0 : float, optional
        First step size. ``0.0`` (default) lets LSODA choose.
    hmax, hmin : float, optional
        Step-size bounds. ``0.0`` (default) means no bound.
    ixpr : int, optional
        Print a message on each method switch.
    mxstep : int, optional
        Maximum internal steps between two output times. ``0`` (default)
        means LSODA's internal 500.
    mxhnil : int, optional
        Maximum "step size too small" messages. ``0`` (default) means
        LSODA's internal 10.
    mxordn, mxords : int, optional
        Maximum order for the Adams (12) and BDF (5) methods.
    printmessg : int, optional
        Non-zero asks for the success message as an :class:`ODEintWarning`.
        An abnormal exit warns whatever this is set to.
    tfirst : bool, optional
        ``False`` (default) selects the ``f(y, t, *args)`` callback order.
        ``True`` selects ``f(t, y, *args)``.  Must be a compile-time
        constant inside ``@njit``.
    success_out : bool, optional
        ``True`` appends a ``success`` bool to the return. ``False`` is the
        default. Must be a compile-time constant inside ``@njit``.

    Returns
    -------
    y : float64 ndarray, shape ``(len(t), len(y0))``
        Solution at each requested time, time-major. ``y[0]`` is ``y0``.
        On an abnormal exit the rows up to and including the failing leg
        hold real values and the rest are ``0.0``.
    infodict : InfoDict
        Only with ``full_output=True``. The 13 keys reached as
        attributes: ``hu``, ``imxer``, ``leniw``, ``lenrw``, ``message``,
        ``mused``, ``nfe``, ``nje``, ``nqu``, ``nst``, ``tcur``,
        ``tolsf``, ``tsw``. The nine vector fields have length
        ``len(t) - 1``, one entry per integration leg. Entries past a
        failure are ``0.0``. ``tolsf`` is written by LSODA only on an
        "excess accuracy" exit, so on a successful run it reads ``0.0``.
    success : bool
        Only with ``success_out=True``. ``istate == 2``.

    Raises
    ------
    TypeError
        ``func`` or ``Dfun`` is not a plain ``@njit`` function (an integer or
        other non-``@njit`` value); or its arity does not fit ``args``.
    ODEpackError
        ``args`` is not a tuple, or a tolerance has the wrong length or
        rank.
    RuntimeError
        The right-hand side returned the wrong number of derivatives, or
        ``Dfun`` returned the wrong shape.
    ValueError
        ``y0`` or ``t`` has rank >= 2, or ``t`` is not monotonic.

    Warns
    -----
    ODEintWarning
        On every abnormal exit, and on a normal one under ``printmessg``.

    See Also
    --------
    scipy.integrate.odeint : The scipy routine this mirrors.

    Notes
    -----
    ``success_out`` has no scipy counterpart.

    ``ODEpackError`` derives straight from ``Exception``, where scipy raises
    its private ``scipy.integrate._odepack.error``, also derived from
    ``Exception``.

    On an abnormal exit the tail of ``y`` past the failing leg is ``0.0``,
    and the ``infodict`` vector entries past that leg are ``0.0``; scipy
    leaves both uninitialized. ``tolsf`` reads ``0.0`` on a successful run,
    where scipy returns whatever the heap held.

    The right-hand side is called once before the run and the length of what
    it returns is checked, which is what scipy checks inside its callback on
    every call.  ``Dfun``'s shape is checked the same way.  A callback whose
    shape CHANGES mid-run is caught by scipy on the offending call and is not
    caught here.

    ``Dfun``'s arity is checked before the run as well.  scipy reaches a
    wrong arity only when LSODA calls the Jacobian, which on a non-stiff
    problem never happens.

    Two refusals arrive earlier from inside ``@njit`` than the check order
    scipy runs.  A ``func`` or ``Dfun`` that is not a plain ``@njit``
    function, and an ``args`` entry that is neither a real number nor an
    array of them, are both settled when the call compiles, so a call
    carrying one of them and a non-monotonic ``t`` reports the refusal where
    scipy reports the ``t`` error.  Called from python, both follow scipy's
    order.

    ``infodict`` is a namedtuple where scipy returns a dict, so
    ``info['nfe']`` is spelled ``info.nfe``.

    ``ixpr=1`` prints LSODA's method-switch messages on Fortran unit 6.
    scipy accepts the argument and prints nothing.  Unit 6 is not reachable
    through ``contextlib.redirect_stdout``, and concurrent solves under
    ``prange`` interleave on it, so ``ixpr=0``, the default, writes nothing.

    **Prange-safe.** The callback address, the ``args`` pointer, the
    Jacobian address and the ``tfirst`` flag travel through Fortran module
    variables, and all slots are ``!$omp threadprivate``, so each OS thread
    reads its own copy.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def rhs(y, t, k):                  # scipy's order, y first
    ...     out = np.empty(2)              # y'' = -k y
    ...     out[0] = y[1]
    ...     out[1] = -k * y[0]
    ...     return out
    >>> @njit
    ... def run():
    ...     return si.odeint(rhs, np.array([1.0, 0.0]),
    ...                      np.array([0.0, 0.5, 1.0]), (4.0,))
    >>> run()
    array([[ 1.        ,  0.        ],
           [ 0.54030231, -1.68294195],
           [-0.4161468 , -1.81859487]])

    ``full_output=True`` adds scipy's diagnostics as a namedtuple, so
    ``info['nfe']`` is spelled ``info.nfe``:

    >>> y, info = si.odeint(rhs, np.array([1.0, 0.0]),
    ...                     np.array([0.0, 0.5, 1.0]), (4.0,),
    ...                     full_output=True)
    >>> info.message
    'Integration successful.'
    """
    banded = _is_banded_odeint(ml, mu)
    jt = _jac_route(Dfun is not None, banded)
    cd = bool(col_deriv)
    ml_i = 0 if (ml is None or ml < 0) else int(ml)
    mu_i = 0 if (mu is None or mu < 0) else int(mu)
    # Rows odeint-D4 and odeint-D2. The order below is scipy's, measured on
    # a grid of two-fault calls: numpy.diff(t), which refuses a scalar or
    # 0-d `t`, then `t` monotonicity, then `args` is a tuple, then the
    # callbacks are callable, then `y0` rank, `t` rank, the tolerances and
    # `tcrit`. A callback's ARITY is not part of that sequence: scipy finds
    # it when it calls the callback, so it is reported below, past the
    # empty-`t` return.
    if np.ndim(t) == 0:
        raise ValueError(_T_SCALAR_MSG)
    tt = _as_1d(t)
    _check_monotonic(tt)
    argt = _args_tuple(args)
    _py = _callback_target(func, _ADDRESS_MSG)
    _jpy = None if Dfun is None else _callback_target(Dfun, _DFUN_ADDRESS_MSG)
    if np.ndim(y0) > 1:
        raise ValueError(_Y0_RANK_MSG)
    if np.ndim(t) > 1:
        raise ValueError(_T_RANK_MSG)
    yy = _as_1d(y0)

    neq = yy.size
    # Per ARGUMENT, rank then length, which is the order `setup_extra_inputs`
    # runs them in: an rtol of the wrong length is reported ahead of a rank-2
    # atol, measured against scipy 1.18.
    rt = np.atleast_1d(np.asarray(_TOL if rtol is None else rtol,
                                  dtype=np.float64)).copy()
    at = np.atleast_1d(np.asarray(_TOL if atol is None else atol,
                                  dtype=np.float64)).copy()
    rv = _is_vec(rtol)
    av = _is_vec(atol)
    if np.ndim(rtol) > 1:
        raise ODEpackError(_RTOL_RANK_MSG)
    if rv and rt.size != neq:
        raise ODEpackError(_TOL_MSG)
    if np.ndim(atol) > 1:
        raise ODEpackError(_ATOL_RANK_MSG)
    if av and at.size != neq:
        raise ODEpackError(_TOL_MSG)
    itol = 1 + (1 if av else 0) + (2 if rv else 0)

    if tcrit is None:
        uc, tc = 0, np.zeros(1, np.float64)
    else:
        # Row odeint-D18. tcrit is converted with max_depth=1, so rank >= 2
        # fails the conversion. Ours walked it as `tc.size` critical times.
        tc = _as_1d(tcrit)
        if tc.ndim > 1:
            raise ODEpackError(_TCRIT_RANK_MSG)
        uc, tc = 1, np.ascontiguousarray(tc).ravel()

    if tt.size == 0:
        # scipy allocates a (0, neq) result and never calls func; under
        # full_output the diagnostic vectors are sized len(t) - 1 and numpy
        # refuses the negative dimension. The tolerance and tcrit
        # conversions above still run: measured, scipy reports both on an
        # empty `t`.
        if full_output:
            raise ValueError("negative dimensions are not allowed")
        yz = np.zeros((0, yy.size), np.float64)
        if success_out:
            return yz, False
        return yz

    # Row odeint-D2. scipy reaches the callback here and Python reports a
    # wrong arity from the call itself, so a call carrying a second fault
    # reports that fault instead and an empty `t` reports nothing at all.
    # An `args` entry of a type LSODA's double* slot cannot carry is refused
    # at the same point.
    _check_arity(_py, len(argt), 'func', tfirst)
    if _jpy is not None:
        _check_arity(_jpy, len(argt), 'Dfun', tfirst)
    kinds = _arg_kinds(argt)

    tf = 1 if tfirst else 0
    _fp = _adapter_rhs_args(_py, bool(tfirst), kinds).address
    aa = _pack_args(argt, neq)
    _jp = 0
    if Dfun is not None:
        _jp = _adapter_jac_args(_jpy, bool(tfirst), kinds, banded, cd).address
        # One evaluation up front.  The shape cannot be checked inside the
        # @cfunc: an exception raised there is printed and swallowed.
        _jj = (_call_t(Dfun, yy, tt[0], argt) if tfirst
               else _call_y(Dfun, yy, tt[0], argt))
        if banded and cd:
            _check_jac_band_t_rt(_jj, neq, ml_i, mu_i)
        elif banded:
            _check_jac_band_rt(_jj, neq, ml_i, mu_i)
        else:
            _check_jac_full_rt(_jj, neq)
    _out = (_call_t(func, yy, tt[0], argt) if tfirst
            else _call_y(func, yy, tt[0], argt))
    _check_rhs_shape(_out, neq)

    (y, hu, tcur, tolsf, tsw, nst, nfe, nje, nqu, mused, imxer, lenrw,
     leniw, istate) = _run_odeint(
        _fp, tf, yy, tt, rt, at, itol, uc, tc, h0, hmax, hmin, ixpr,
        mxstep, mxhnil, mxordn, mxords, aa, jt, ml_i, mu_i, _jp)

    _emit_odeint_warning(istate, printmessg, 3)
    ok = istate == 2
    if full_output:
        info = InfoDict(hu, int(imxer), int(leniw), int(lenrw), _msg(istate),
                        mused, nfe, nje, nqu, nst, tcur, tolsf, tsw)
        if success_out:
            return y, info, ok
        return y, info
    if success_out:
        return y, ok
    return y


def _odeint_refuse(msg, odepack, fo, so, monotonic):
    """An `@overload` arm for :func:`odeint` that refuses the call.

    A body whose only statement is a ``raise`` types as returning ``none``,
    so the caller's ``y = odeint(...)`` fails to unpack before the body ever
    runs and the message never reaches anyone.  Measured on all four
    refusals: a rank-2 ``y0`` reported a ``TypingError`` from
    ``y = odeint(f, np.zeros((2, 2)), t)`` and the correct
    ``ValueError("Initial condition y0 must be one-dimensional.")`` only
    when the result was discarded.

    So the raise is a STATEMENT and a value of the shape this call asked for
    follows it.  That value is never produced: it exists so the arm carries
    the return type its caller is unpacking.

    ``monotonic`` runs scipy's ``t`` test ahead of the refusal, which is
    where scipy reports it from.  ``odepack`` picks `ODEpackError` over
    `ValueError`, and ``fo`` and ``so`` pick the number of return values.
    """
    if odepack:
        @njit
        def _raise():
            raise ODEpackError(msg)
    else:
        @njit
        def _raise():
            raise ValueError(msg)

    def impl(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0,
             ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0,
             hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
             mxords=5, printmessg=0, tfirst=False, success_out=False):
        if monotonic:
            _check_monotonic(_as_1d(t))
        _raise()
        y = np.zeros((0, 1), np.float64)
        z = np.zeros(0, np.float64)
        zi = np.zeros(0, np.int32)
        if fo:
            info = InfoDict(z, 0, 0, 0, '', zi, zi, zi, zi, zi, z, z, z)
            if so:
                return y, info, False
            return y, info
        if so:
            return y, False
        return y
    return impl


@overload(odeint, prefer_literal=True)
def _odeint_ovl(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0,
                ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0,
                hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
                mxords=5, printmessg=0, tfirst=False, success_out=False):
    """Compiled body for :func:`odeint`, one per distinct call shape.

    ``full_output``, ``success_out`` and ``tfirst`` are resolved here: the
    first two pick the number of return values and the third picks the
    callback ABI.  The unimplemented arguments raise from inside a compiled
    body rather than from the typing pass, so the message reaches the caller
    as a ``ValueError`` and not as a typing failure.
    """
    fo = _lit_bool(full_output)
    so = _lit_bool(success_out)
    tf_lit = _lit_bool(tfirst)
    if fo is None:
        raise TypingError(_LITERAL_MSG % ('full_output',
                                          'the number of return values'))
    if so is None:
        raise TypingError(_LITERAL_MSG % ('success_out',
                                          'the number of return values'))
    if tf_lit is None:
        raise TypingError(_LITERAL_MSG % ('tfirst',
                                          "the callback's argument order"))
    tf = 1 if tf_lit else 0

    # Row odeint-D4. A scalar or 0-d `t` is a TYPE fact, and `_as_1d`
    # promotes it to length 1, so no run-time test can see it. scipy reaches
    # numpy.diff(t) before every check of its own, so this refusal precedes
    # the callback resolution and the y0 rank test, as it does in the python
    # body. The impl names nothing but the message, since the reason it
    # exists may be that the rest cannot be typed at all.
    if _rank(t) == 0:
        return _odeint_refuse(_T_SCALAR_MSG, False, fo, so, False)

    # Row odeint-D2. `args` being a tuple is a TYPE fact and scipy tests it
    # AFTER the monotonicity of `t`, so the refusal is deferred into a body
    # that runs that test first. It also carries the python body's class,
    # where refusing from the typing pass gave a TypingError.
    try:
        _tys = _args_types(args)
    except TypingError:
        return _odeint_refuse(_ARGS_TUPLE_MSG, True, fo, so, True)

    # resolve the callback AT COMPILE TIME
    if not isinstance(func, types.Dispatcher):
        raise TypingError(_ADDRESS_MSG)
    _py = func.dispatcher.py_func

    # `ml`/`mu` select LSODA's banded Jacobian and `Dfun` selects whether the
    # caller supplies it. Which route the call takes is settled here, so the
    # compiled body carries a plain `jt` constant; the half-bandwidths
    # themselves stay runtime values.
    _no_dfun = _is_none(Dfun)
    # A `ml`/`mu` held in a runtime variable cannot say here whether LSODA's
    # banded Jacobian was asked for, because a negative value is scipy's
    # spelling of None. Both answers are prepared and the body picks one,
    # which is what scipy does from the value it always has.
    _banded = _is_banded_odeint(ml, mu)
    _rt_band = _banded is None
    _jt_b = _jac_route(not _no_dfun, True)
    _jt_f = _jac_route(not _no_dfun, False)
    _jt = _jt_b if _banded else _jt_f
    _no_ml = _is_none(ml)
    _no_mu = _is_none(mu)
    _cd = _lit_bool(col_deriv)
    if _cd is None:
        raise TypingError(_LITERAL_MSG % ('col_deriv',
                                          "the jacobian adapter's layout"))

    if not _no_dfun and not isinstance(Dfun, types.Dispatcher):
        raise TypingError(_DFUN_ADDRESS_MSG)

    # Row odeint-D2. Both rank refusals run scipy's monotonicity test first,
    # which is a RUN-TIME fact and precedes them on scipy's side.
    if _rank(y0) > 1:
        return _odeint_refuse(_Y0_RANK_MSG, False, fo, so, True)
    if _rank(t) > 1:
        return _odeint_refuse(_T_RANK_MSG, False, fo, so, True)

    # Row odeint-D2. A callback's arity is a typing-time fact here and a
    # run-time one for scipy, which reports it from the call itself, past the
    # tolerances, `tcrit` and the empty-`t` return. The message is captured
    # here and raised at that point in the body below. The @cfunc is then not
    # built, since it cannot be, and the calls that would type against the
    # wrong arity sit after the raise, where numba prunes them.
    _amsg = ''
    try:
        _check_arity(_py, len(_tys), 'func', bool(tf_lit))
        if not _no_dfun:
            _check_arity(Dfun.dispatcher.py_func, len(_tys), 'Dfun',
                         bool(tf_lit))
    except TypeError as _e:
        _amsg = str(_e)
    _bad_arity = bool(_amsg)

    # The right-hand side's @cfunc, and the jacobian's, are built HERE, at
    # typing time, so both are plain @njit functions for the caller and their
    # addresses are constants in the body below.
    _kinds = () if _bad_arity else _arg_kinds_ty(_tys)
    _addr = 0 if _bad_arity else _adapter_rhs_args(_py, bool(tf_lit),
                                                   _kinds).address
    _jaddr = 0
    _jaddr_b = 0
    if not _no_dfun and not _bad_arity:
        _jpy = Dfun.dispatcher.py_func
        _jaddr = _adapter_jac_args(_jpy, bool(tf_lit), _kinds,
                                   bool(_banded), _cd).address
        _jaddr_b = (_adapter_jac_args(_jpy, bool(tf_lit), _kinds, True,
                                      _cd).address if _rt_band else _jaddr)

    no_rtol = _is_none(rtol)
    no_atol = _is_none(atol)
    no_tcrit = _is_none(tcrit)
    # A rank >= 2 tolerance is refused in the BODY, with scipy's class and
    # text, rather than here: refusing at typing time gave a TypingError
    # where the python body and scipy give ODEpackError, and the refusal has
    # to sit between the rtol and atol conversions in scipy's order.
    rv = False if no_rtol else _is_vec(rtol)
    av = False if no_atol else _is_vec(atol)
    itol = 1 + (1 if av else 0) + (2 if rv else 0)

    def impl(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0,
             ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0,
             hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
             mxords=5, printmessg=0, tfirst=False, success_out=False):
        yy = _as_1d(y0)
        tt = _as_1d(t)
        _check_monotonic(tt)
        neq = yy.size

        # Rank then length, per argument, which is scipy's order. Each rank
        # test runs on the CONVERTED array and ahead of the ravel, so the
        # rank-2 specialisation still types.
        if no_rtol:
            rt = np.full(1, _TOL)
        else:
            rt0 = _as_1d(rtol)
            if rt0.ndim > 1:
                raise ODEpackError(_RTOL_RANK_MSG)
            rt = np.ascontiguousarray(rt0).ravel()
            if rv and rt.size != neq:
                raise ODEpackError(_TOL_MSG)
        if no_atol:
            at = np.full(1, _TOL)
        else:
            at0 = _as_1d(atol)
            if at0.ndim > 1:
                raise ODEpackError(_ATOL_RANK_MSG)
            at = np.ascontiguousarray(at0).ravel()
            if av and at.size != neq:
                raise ODEpackError(_TOL_MSG)

        if no_tcrit:
            uc = 0
            tc = np.zeros(1, np.float64)
        else:
            # Row odeint-D18.
            uc = 1
            tc0 = _as_1d(tcrit)
            if tc0.ndim > 1:
                raise ODEpackError(_TCRIT_RANK_MSG)
            tc = np.ascontiguousarray(tc0).ravel()

        if tt.size == 0:
            # Below the tolerance and tcrit conversions: measured, scipy
            # reports a rank-2 rtol, a bad tolerance length and a rank-2
            # tcrit on an empty `t` too.
            if fo:
                raise ValueError("negative dimensions are not allowed")
            yz = np.zeros((0, yy.size), np.float64)
            if so:
                return yz, False
            return yz

        # Row odeint-D2. scipy reaches the callback here.
        if _bad_arity:
            raise TypeError(_amsg)

        fp = _addr
        aa = _pack_args(args, neq)

        if _no_ml:
            ml_i = 0
        else:
            ml_i = max(np.int64(ml), 0)
        if _no_mu:
            mu_i = 0
        else:
            mu_i = max(np.int64(mu), 0)

        if _rt_band:
            bnd = False
            if not _no_ml:
                if ml >= 0:
                    bnd = True
            if not _no_mu:
                if mu >= 0:
                    bnd = True
            jt_r = _jt_b if bnd else _jt_f
        else:
            bnd = _banded
            jt_r = _jt

        if _no_dfun:
            jp = 0
        else:
            if _rt_band:
                jp = _jaddr_b if bnd else _jaddr
            else:
                jp = _jaddr
            if tf == 1:
                jj = _call_t(Dfun, yy, tt[0], args)
            else:
                jj = _call_y(Dfun, yy, tt[0], args)
            if bnd and _cd:
                _check_jac_band_t_rt(jj, neq, ml_i, mu_i)
            elif bnd:
                _check_jac_band_rt(jj, neq, ml_i, mu_i)
            else:
                _check_jac_full_rt(jj, neq)
        if tf == 1:
            _check_rhs_shape(_call_t(func, yy, tt[0], args), neq)
        else:
            _check_rhs_shape(_call_y(func, yy, tt[0], args), neq)

        (y, hu, tcur, tolsf, tsw, nst, nfe, nje, nqu, mused, imxer, lenrw,
         leniw, istate) = _run_odeint(
            fp, tf, yy, tt, rt, at, itol, uc, tc, h0, hmax, hmin, ixpr,
            mxstep, mxhnil, mxordn, mxords, aa, jt_r, ml_i, mu_i, jp)

        _odeint_warn(istate, printmessg)
        ok = istate == 2
        if fo:
            info = InfoDict(hu, int(imxer), int(leniw), int(lenrw),
                            _msg(istate), mused, nfe, nje, nqu, nst, tcur,
                            tolsf, tsw)
            if so:
                return y, info, ok
            return y, info
        if so:
            return y, ok
        return y
    return impl
