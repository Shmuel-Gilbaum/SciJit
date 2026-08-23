"""Shared helpers for an N-D ``y`` with an interpolation ``axis``.

scipy's one-dimensional interpolators accept an N-D ``y`` together with an
``axis`` naming which of its axes the interpolation runs along. Every position
on the remaining axes is an independent series sampled at the same ``x``, so
one ``x`` and one set of breakpoints serve all of them.

Notes
-----
**Storage.** The classes in this package hold the values with the
interpolation axis moved to the front and the remaining axes flattened into
one, shape ``(n, T)`` where ``T`` is the product of those axes. The trailing
axes only ever travel together, so one evaluation kernel serves every rank
instead of one per rank.

The rank reappears on the way out. A method evaluates to ``(m, T)``, reshapes
to ``(m,) + rest``, then moves the query axis from the front to wherever the
caller's ``axis`` put it. That last step is what makes the result match scipy:
the query axis lands at position ``axis``, it is not prepended.

**Why there is one class per rank, and why they are generated.** A jitclass
field has a fixed rank, and a method cannot return a rank that follows a
field's runtime contents. So each supported rank of ``y`` needs its own class,
differing from the next only in a reshape line, and ``reshape`` needs its
argument COUNT literal in the source.

`_ND_MAXTRAIL` fixes how many ranks exist. The helpers here, the N-D classes
in `_cubic`, `_bspline` and `_interp1d`, and the dispatch that picks between
them are all built in a loop from that one number, so raising the cap is a
change to the number alone. ``scijit.integrate``'s `nquad` reaches arbitrary
nesting depth the same way.

The cost is that ``inspect.getsource`` cannot show a generated class: there is
no file to point at. Every generated class and method therefore carries a
docstring naming the function that built it and the rank it was built for.

``@njit`` is numba's compiler decorator: the decorated function is compiled to
machine code, once per combination of argument types it is called with. Since
``y.ndim`` is part of an array's numba type, a factory that branches on it has
the other branches removed while compiling, and the branch that survives fixes
the returned class.
"""

import numpy as np
from numba import njit, types
from numba.core.errors import TypingError
from numba.extending import overload

__all__ = []


_AXIS_MSG = "an integer is required for the axis"


def _check_axis(axis):
    """Refuse an `axis` that is not an integer.

    `axis` sits at third or fourth position, so a call written against an
    older order lands a `bc_type` string or a `fill_value` array here. For a
    1-D `y` the axis is never used, so without this guard such a call would
    return a spline built with the DEFAULT boundary condition and say
    nothing.

    ``bool`` is a subclass of ``int`` in Python and a distinct type in numba.
    Both arms accept it and read it as the integer it is.
    """
    if not isinstance(axis, (bool, int, np.integer)):
        raise TypeError(_AXIS_MSG)


@overload(_check_axis)
def _check_axis_ovl(axis):
    """`_check_axis` inside ``@njit``, resolved while compiling.

    An OMITTED argument reaches an ``@overload`` as ``types.Omitted``, not as
    the type of its default, so the default has to be unwrapped and checked on
    its own. Without that, ``CubicSpline(x, y)`` inside ``@njit`` is refused
    while the identical call from Python is not.
    """
    if isinstance(axis, types.Omitted):
        if isinstance(axis.value, (bool, int)):
            def impl(axis):
                pass
            return impl
        raise TypingError(_AXIS_MSG)
    if isinstance(axis, (types.Integer, types.Boolean)):
        def impl(axis):
            pass
        return impl
    raise TypingError(_AXIS_MSG)


@njit
def _check_axis_range(axis, ndim):
    """Refuse an `axis` outside ``[-ndim, ndim)``.

    Raises
    ------
    numpy.exceptions.AxisError
        If `axis` is outside the range, with the message
        ``numpy.lib.array_utils.normalize_axis_index`` writes.

    Notes
    -----
    Applies to `BSpline` and `make_interp_spline`, whose `axis` names an axis
    of an array rather than a value folded by a modulo. The cubic family folds
    instead and calls `_check_axis` alone.

    The cast to ``int64`` is what lets a ``bool`` reach ``str`` inside
    ``@njit``.
    """
    ax = np.int64(axis)
    if ax < -ndim or ax >= ndim:
        raise np.exceptions.AxisError(
            "axis " + str(ax) + " is out of bounds for array of dimension "
            + str(ndim))


@njit
def _check_axis_index(axis, ndim):
    """Refuse an `axis` that is not a usable index into a shape of `ndim`.

    Raises
    ------
    IndexError
        If `axis` is outside ``[-ndim, ndim)``.

    Notes
    -----
    Applies to `interp1d`, which reads ``y.shape[axis]`` before folding the
    axis, so the range is the same as `_check_axis_range`'s and the exception
    is the one a tuple subscript raises.
    """
    ax = np.int64(axis)
    if ax < -ndim or ax >= ndim:
        raise IndexError("tuple index out of range")

# THE CAP. How many axes an N-D `y` may carry besides the interpolation axis.
# `y` may be of rank 1, served by the hand-written 1-D class, up to rank
# 1 + _ND_MAXTRAIL. Raising this number is the only edit an extra rank needs:
# the helpers below, the N-D classes in the three interpolator modules and the
# dispatch branches that choose between them are generated from it.
_ND_MAXTRAIL = 2

# The ranks of `y` that get an N-D class. Rank 1 has a hand-written class, so
# the generated ones start at rank 2.
_ND_RANKS = tuple(range(2, _ND_MAXTRAIL + 2))


def _rank_phrase():
    """Return the supported ranks as prose, ``'1, 2 or 3'`` at a cap of 2.

    Used in the ``ValueError`` a factory raises for a rank past the cap, so
    the message follows `_ND_MAXTRAIL` rather than restating it.
    """
    r = [str(i) for i in range(1, _ND_MAXTRAIL + 2)]
    return ", ".join(r[:-1]) + " or " + r[-1]


def _shape_words(trail):
    """Return the trailing shape as prose for a generated docstring.

    ``'T'`` for one trailing axis, which the class stores unflattened;
    ``'r0, r1'`` and onwards for more, which is what ``get_coeffs``
    unflattens to.
    """
    if trail == 1:
        return "T"
    return ", ".join("r%d" % i for i in range(trail))


def _coeff_body(trail, lead):
    """Return the ``get_coeffs`` body a generated class of this trail needs.

    One trailing axis is already the caller's shape, so the stored array goes
    back untouched. More than one is stored flattened and has to be unflattened
    to the rank the caller passed, which is the reshape whose argument count
    only a generated source can make literal.

    ``ascontiguousarray`` comes FIRST because a jitclass field declared
    ``float64[:, :, :]`` is typed with layout 'A', and numba's ``reshape``
    takes a contiguous array only.

    Parameters
    ----------
    trail : int
        Number of trailing axes the class serves.
    lead : str
        Source for the leading dimensions of the reshape, ``'4, self.n - 1'``
        for a PPoly class and ``'self.n'`` for a B-spline one.
    """
    if trail == 1:
        return "        return self.c"
    dims = ", ".join("self.rest[%d]" % i for i in range(trail))
    return ("        return np.ascontiguousarray(self.c).reshape(\n"
            "            %s, %s)" % (lead, dims))


def _dispatch_branches(var, body):
    """Return an ``if``/``elif`` chain over the supported ranks, as source.

    Parameters
    ----------
    var : str
        Name of the array argument whose rank selects the branch.
    body : callable
        Takes a rank and returns that branch's body, indented by 8 spaces.

    Notes
    -----
    ``ndim`` is part of an array's numba type, so a chain of literal
    comparisons against it has every branch but one removed while compiling.
    That is what lets one function return a different class per rank.
    """
    out = []
    for rank in _ND_RANKS:
        out.append("    %s %s.ndim == %d:"
                   % ("if" if rank == _ND_RANKS[0] else "elif", var, rank))
        out.append(body(rank))
    return "\n".join(out)


def _define(src, ns, name):
    """Compile one generated definition and hand back what it defines.

    Parameters
    ----------
    src : str
        Source for a single module-level ``def`` or ``class``.
    ns : dict
        Globals to compile against, normally the calling module's
        ``globals()``, so the generated code resolves the same names a
        hand-written definition would and lands in the same module.
    name : str
        The name `src` binds. It is bound in `ns` as a side effect.

    Returns
    -------
    obj : object
        The function or class just defined.

    Notes
    -----
    A syntax error in `src` raises ``SyntaxError`` from here, at import time,
    because every generator in this package runs while its module is being
    imported.
    """
    exec(src, ns)
    return ns[name]


_FLATTEN_SRC = '''
def _flatten{rank}(y, axis):
    """Lay a rank-{rank} `y` out as ``(n, T)``, interpolation axis first.

    Returns the flattened values, the trailing shape as an array, and the
    resolved axis. A negative `axis` wraps, as scipy's ``axis % y.ndim`` does.
    The {trail} trailing axes are flattened into one, so the stored array has
    the same rank whatever `y` had and one kernel evaluates every rank.

    Generated by ``_make_flatten({rank})``; see this module's Notes.
    """
    ax = axis % {rank}
    ym = np.ascontiguousarray(np.moveaxis(y, ax, 0))
    rest = np.empty({trail}, np.int64)
{assign}
    flat = np.ascontiguousarray(ym.reshape(ym.shape[0], {prod}))
    return flat, rest, ax
'''

_RESTORE_SRC = '''
def _restore{rank}(flat, rest, axis):
    """Unflatten an ``(m, T)`` result to rank {rank} and place the query axis.

    The reshape has a literal rank and runtime dimensions, which is what lets
    one stored layout serve a rank the caller chose. The query axis then moves
    from the front to position `axis`, which is where scipy leaves it.

    Generated by ``_make_restore({rank})``; see this module's Notes.
    """
    nd = flat.reshape(flat.shape[0], {dims})
    return np.ascontiguousarray(np.moveaxis(nd, 0, axis))
'''

_POINT_SRC = '''
def _point{rank}(vals, rest):
    """Shape a single-point ``(T,)`` result for a rank-{rank} `y`.

    A scalar query removes the interpolation axis instead of replacing it, so
    the result carries the trailing shape alone and nothing has to move.

    Generated by ``_make_point({rank})``; see this module's Notes.
    """
    return np.ascontiguousarray(vals.reshape({dims}))
'''


def _make_flatten(rank):
    """Build ``_flatten<rank>``, the input side of the stored layout."""
    trail = rank - 1
    src = _FLATTEN_SRC.format(
        rank=rank, trail=trail,
        assign="\n".join("    rest[%d] = ym.shape[%d]" % (i, i + 1)
                         for i in range(trail)),
        prod=" * ".join("ym.shape[%d]" % i for i in range(1, rank)))
    return njit(_define(src, globals(), "_flatten%d" % rank))


def _make_restore(rank):
    """Build ``_restore<rank>``, the output side for an array query."""
    trail = rank - 1
    src = _RESTORE_SRC.format(
        rank=rank, dims=", ".join("rest[%d]" % i for i in range(trail)))
    return njit(_define(src, globals(), "_restore%d" % rank))


def _make_point(rank):
    """Build ``_point<rank>``, the output side for a scalar query."""
    trail = rank - 1
    src = _POINT_SRC.format(
        rank=rank, dims=", ".join("rest[%d]" % i for i in range(trail)))
    return njit(_define(src, globals(), "_point%d" % rank))


for _rank in _ND_RANKS:
    globals()["_flatten%d" % _rank] = _make_flatten(_rank)
    globals()["_restore%d" % _rank] = _make_restore(_rank)
    globals()["_point%d" % _rank] = _make_point(_rank)
del _rank
