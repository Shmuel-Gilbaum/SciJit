"""Typing-time predicates shared by the ``@overload`` front ends.

An ``@overload`` chooser runs while the call compiles and is handed either a
numba type or the raw Python default. These read one argument in whichever
of those shapes it arrives in. Not a public API.
"""

import numpy as np
from numba import types

__all__ = ['_is_none', '_lit_bool', '_lit_str']


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
