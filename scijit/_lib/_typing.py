"""Typing-time predicates shared by the ``@overload`` front ends.

An ``@overload`` chooser runs while the call compiles and is handed either a
numba type or the raw Python default. These read one argument in whichever
of those shapes it arrives in. Not a public API.
"""

import numpy as np
from numba import types
from numba.core.errors import TypingError

__all__ = ['_is_none', '_lit_bool', '_lit_str',
           '_K_FLOAT', '_K_INT', '_K_BOOL',
           '_arg_kinds', '_arg_kinds_ty']


def _lit_str(v):
    """The value of a string argument at TYPING time, or ``None``.

    A string argument that selects which routine is compiled in, such as
    QUADPACK's ``weight`` or ``solve_ivp``'s ``method``, has to be known
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

    A string literal is NOT read here. `scijit.interpolate` accepts one and
    resolves it in its own front end, where that contract is documented.
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


#: kind codes for one ``args`` entry: a float, an integer and a boolean
#: scalar. ``k >= 1`` is an array of that rank. A scalar keeps its own type
#: through the buffer, which is a cast in the adapter; an array's data
#: crosses as float64.
_K_FLOAT, _K_INT, _K_BOOL = -1, -2, -3


def _arg_kinds(args, msg):
    """Element kinds of a Python-level ``args`` tuple.

    The packer writes what the kinds say and the adapter reads it back, so
    the two halves cannot drift.

    Parameters
    ----------
    args : tuple
        The entries to classify.
    msg : str
        Text of the ``ValueError`` raised for a non-numeric entry. Each
        front end names its own pack's argument slot.
    """
    kinds = []
    for v in args:
        a = v if isinstance(v, np.ndarray) else np.asarray(v)
        if a.dtype.kind not in 'biuf':
            raise ValueError(msg)
        if a.ndim:
            kinds.append(a.ndim)
        elif a.dtype.kind == 'b':
            kinds.append(_K_BOOL)
        elif a.dtype.kind == 'f':
            kinds.append(_K_FLOAT)
        else:
            kinds.append(_K_INT)
    return tuple(kinds)


def _arg_kinds_ty(tys, msg):
    """:func:`_arg_kinds` from the numba TYPES of ``args``, at typing time.

    Parameters
    ----------
    tys : tuple of numba types
        The entry types to classify.
    msg : str
        Text of the ``TypingError`` raised for a non-numeric entry.
    """
    kinds = []
    for t in tys:
        t = types.unliteral(t)
        if isinstance(t, types.Array):
            if not isinstance(t.dtype, (types.Integer, types.Float,
                                        types.Boolean)):
                raise TypingError(msg)
            kinds.append(t.ndim)
        elif isinstance(t, types.Boolean):
            kinds.append(_K_BOOL)
        elif isinstance(t, types.Float):
            kinds.append(_K_FLOAT)
        elif isinstance(t, types.Integer):
            kinds.append(_K_INT)
        else:
            raise TypingError(msg)
    return tuple(kinds)
