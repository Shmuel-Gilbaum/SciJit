"""scipy.interpolate.interp1d, written in numba directly.

A numba jitclass (evaluable inside @njit code) built by a small Python
factory, mirroring ``scipy.interpolate.interp1d``. The linear / nearest /
previous / next kinds select an index directly. The spline kinds are
FITTED through this package's ``make_interp_spline``, which is the route
scipy's own ``interp1d`` takes, and EVALUATED through the FITPACK ``splev``.
Fitting that way carries no degree cap, so an integer `kind` above 3 works.

Public API
    interp1d(x, y, kind='linear', bounds_error=None, fill_value=nan,
             extrapolate=False, assume_sorted=False, axis=-1)
                                                     -> _Interp1D jitclass

``kind`` takes scipy's string:
    'linear'      also 'slinear', and the same as the integer order 1
    'nearest'     ties round DOWN  (scipy side='left')
    'previous'    also 'zero', and the same as the integer order 0
    'next'        value of the next sample
    'quadratic'   the same as the integer order 2
    'cubic'       the same as the integer order 3
    'nearest-up'  ties round UP    (scipy side='right')

An INTEGER ``kind`` is the spline ORDER, which is scipy's meaning: 0 is
'zero', 1 'slinear', 2 'quadratic', 3 'cubic', and orders above 3 have no
string spelling. A negative order raises, with scipy's message.

Methods (mirroring the other classes here)
    f(xs)      xs 1-D float64 -> 1-D float64, running .ev
    f(x)       single point -> scalar, running .ev_one
    .ev(xs)    the same batch evaluation, by name
    .ev_one(x) the same single point, by name
    x[xs]      __getitem__ sugar == .ev(xs)

``fill_value`` takes scipy's spellings: a scalar, a 2-tuple read as
``(below, above)``, a sequence broadcast up to ``y``'s trailing shape and
meaning one value per series, or the string ``"extrapolate"``. The additional
bool ``extrapolate=True`` reaches the same extrapolation.

Deviations from scipy
  * ``y`` may be rank 1, 2 or 3, with ``axis`` naming the interpolation
    axis. Rank 4 and above raise: a jitclass field has a fixed rank, so
    each supported rank is a separate class.
  * ``copy=False`` is honoured for an `x` or `y` that is already contiguous
    in the dtype it is stored in, where a later mutation then shows through
    as it does in scipy. Anything else is converted, which copies, because a
    strided view read through a raw pointer gives wrong values.
  * ``x`` must be strictly increasing once sorted. A repeated abscissa raises
    ``ValueError`` where scipy is silent, and with ``assume_sorted=True`` an
    unsorted ``x`` raises where scipy returns NaN.
  * A complex ``fill_value`` on a real ``y`` raises ``ValueError``, and a
    complex ``x`` raises ``TypingError``; scipy truncates both to float64.
  * ``fill_value`` is a property returning the resolved ``(below, above)``
    pair, where scipy's returns the object last assigned; `set_fill_value`
    takes everything scipy's setter takes.

prange-safety: linear/nearest/previous/next kinds are pure array reads ->
prange-safe. The spline kinds call the stateless FITPACK ``splev`` evaluator
-> also prange-safe. Build the instances OUTSIDE the prange
loop, then evaluate many independent interpolators concurrently.

Examples
--------
>>> import numpy as np
>>> from numba import njit
>>> from scijit.interpolate import interp1d
>>> x = np.linspace(0.0, 1.0, 9)
>>> y = np.sin(3.0 * x)
>>> f = interp1d(x, y, 'cubic')      # or the integer spline order 3
>>> np.round(f(np.array([0.25, 0.75])), 6)
array([0.681639, 0.778073])

Inside compiled code, scanning the domain for the interpolant's peak:

>>> @njit
... def peak(f):
...     xs = np.linspace(0.0, 1.0, 201)
...     return np.max(f(xs))
>>> float(np.round(peak(f), 6))
0.999968
"""
import numpy as np
from numba import (njit, objmode, complex128, float64, int64, boolean,
                   types)
from numba.extending import overload
from scijitclass import scijitclass

from . import evaluators as _ev
from ._bspline import _mis_core, _mis_core_nd

# `_mis_core` takes the caller's knot vector as an argument rather
# than an optional, so one compiled body serves both cases. interp1d
# never supplies one: its spline kinds always use the not-a-knot set.
_NO_KNOTS = np.zeros(0, np.float64)
_NO_ORD = np.zeros(0, np.int64)
_NO_VAL = np.zeros(0, np.float64)
from . import _ndaxis as _nda
from ._ndaxis import (_ND_RANKS, _check_axis, _define,
                      _dispatch_branches, _rank_phrase)


# ---------------------------------------------------------------------------
# the `kind` flag: scipy's string, or an integer spline order
# ---------------------------------------------------------------------------
#
# `_KIND_NAMES[i]` is the scipy string for our code `i`. `_KIND_ALIASES` holds
# the two further names scipy accepts, each mapping onto a code we already
# have; both were measured against scipy before being admitted.

_KIND_NAMES = ("linear", "nearest", "previous", "next", "quadratic", "cubic",
               "nearest-up")
_KIND_ALIASES = (("slinear", 0), ("zero", 2))

# Spline degree carried by each code. 0 marks an index kind, which needs none.
_KIND_DEGREE = (0, 0, 0, 0, 2, 3, 0)

# Code 7 is a B-spline whose degree is held in the instance's `k`. Codes 4 and
# 5 are quadratic and cubic and predate it; 7 covers the orders that have no
# string spelling.
_SPLINE_ANY = 7

_KIND_MSG = ("kind must be one of 'linear', 'nearest', 'nearest-up', "
             "'previous', 'next', 'quadratic', 'cubic', 'slinear', 'zero', "
             "or a non-negative integer spline order")

#: The tail of scipy's ``NotImplementedError`` for an unsupported `kind`.
_KIND_UNSUPPORTED = " is unsupported: Use fitpack routines for other types."


@njit
def _raise_kind_value(k):
    """Raise scipy's `kind` refusal, with the offending value in the text.

    Row: interp1d-I5. scipy reads an integer `kind` as the spline order and
    refuses everything else, naming the value:
    ``NotImplementedError('1.5 is unsupported: Use fitpack routines for other
    types.')``. A float reaches ``str`` inside ``@njit`` as the literal text
    ``<object type:float64>``, so the message is assembled in `objmode`, which
    is the only spelling measured to render a float64. The cost sits on the
    raising path alone: 200,000 in-range calls through a body carrying this
    branch took 0.0425 s against 0.0435 s without it, and a caller's ``prange``
    over a function containing it still parallelises (0 performance warnings).
    """
    with objmode(msg='unicode_type'):
        msg = f"{k} is unsupported: Use fitpack routines for other types."
    raise NotImplementedError(msg)


@njit
def _raise_kind_none():
    """Raise scipy's `kind` refusal for ``kind=None``. Row interp1d-I5.

    A bare ``raise`` in an ``@overload`` arm makes the arm's return type
    ``none``, because the ``return`` after it is unreachable and inference
    drops it. Calling out to a raising function keeps the ``return`` live, so
    the arm still hands back the ``(code, degree)`` pair its callers unpack.
    """
    raise NotImplementedError("None" + _KIND_UNSUPPORTED)


@njit
def _order_pair(k):
    """``(code, degree)`` for scipy's integer `kind`, the spline ORDER.

    Orders 0 and 1 need no spline: scipy's ``'zero'`` is the previous sample
    and ``'slinear'`` is linear, and both are reached by index. Orders 2 and 3
    take the codes the strings ``'quadratic'`` and ``'cubic'`` already use, so
    the two spellings share one path. Anything higher gets `_SPLINE_ANY` with
    the degree beside it.
    """
    if k < 0:
        raise ValueError("Expect non-negative k.")
    if k == 0:
        return 2, 0
    if k == 1:
        return 0, 0
    if k == 2:
        return 4, 2
    if k == 3:
        return 5, 3
    return _SPLINE_ANY, k


def _kind_code(kind):
    """scipy's `kind`, resolved to ``(internal code, spline degree)``.

    A string names one of scipy's seven kinds. An INTEGER is the spline
    ORDER, which is what scipy means by an integer `kind`: 0 is ``'zero'``,
    1 ``'slinear'``, 2 ``'quadratic'``, 3 ``'cubic'``, and 4 and 5 are
    higher-order splines with no string spelling.

    The degree is returned beside the code because orders 4 and 5 share one
    code and are told apart only by it.

    Only a string and an INTEGER are accepted, which is scipy's rule: its
    chain is ``kind in ['zero', 'slinear', 'quadratic', 'cubic']``, then
    ``isinstance(kind, int)``, then the five remaining names, then
    ``NotImplementedError``. A float never satisfies ``isinstance(kind, int)``
    there, so ``kind=1.5`` and ``kind=2.0`` are both refused (row
    interp1d-I5). ``bool`` is an ``int`` in Python and reads as the order 0
    or 1.
    """
    if isinstance(kind, str):
        for i in range(7):
            if kind == _KIND_NAMES[i]:
                return i, _KIND_DEGREE[i]
        for j in range(2):
            if kind == _KIND_ALIASES[j][0]:
                return _KIND_ALIASES[j][1], 0
        raise NotImplementedError("%s%s" % (kind, _KIND_UNSUPPORTED))
    if isinstance(kind, (bool, int, np.integer)):
        return _order_pair(int(kind))
    raise NotImplementedError("%s%s" % (kind, _KIND_UNSUPPORTED))


@overload(_kind_code)
def _kind_code_ovl(kind):
    """`_kind_code` inside ``@njit``, one body per spelling of `kind`.

    A string and an integer cannot share one body, so the argument type
    decides which arm is built. The string arm keeps its loop at run time, so
    `kind` may be a runtime string rather than a literal.

    A FLOAT and ``None`` get their own arms rather than falling through to
    ``int(kind)``, which truncated 1.5 to a linear interpolator and said
    nothing (row interp1d-I5). The type is the whole decision, exactly as
    ``isinstance(kind, int)`` is in scipy, so it is taken while compiling.
    """
    if isinstance(kind, (types.UnicodeType, types.StringLiteral, str)):
        def impl(kind):
            for i in range(7):
                if kind == _KIND_NAMES[i]:
                    return i, _KIND_DEGREE[i]
            if kind == "slinear":
                return 0, 0
            if kind == "zero":
                return 2, 0
            raise NotImplementedError(kind + _KIND_UNSUPPORTED)
        return impl

    if isinstance(kind, types.NoneType) or kind is None:
        def impl(kind):
            _raise_kind_none()
            return 0, 0
        return impl

    if isinstance(kind, types.Float):
        def impl(kind):
            _raise_kind_value(kind)
            return 0, 0
        return impl

    if isinstance(kind, types.Omitted):
        kv = kind.value
        if isinstance(kv, str):
            code, deg = _kind_code(kv)

            def impl(kind):
                return code, deg
            return impl
        if not isinstance(kv, (bool, int, np.integer)):
            def impl(kind):
                raise NotImplementedError("%s%s" % (kv, _KIND_UNSUPPORTED))
            return impl

    def impl(kind):
        return _order_pair(int(kind))
    return impl


# ---------------------------------------------------------------------------
# the `fill_value` argument: scipy's (below, above) pair, or a per-series fill
# ---------------------------------------------------------------------------
#
# scipy branches on the CONTAINER. A 2-tuple, and only a 2-tuple, is
# ``(below, above)``; anything else, scalar, list or array, is broadcast up to
# the trailing shape of `y` and means one value PER SERIES. Rows interp1d-I1
# and -I2. The two ends are therefore held as flat arrays of the trailing
# length, and the tuple's two elements are broadcast separately, so
# ``fill_value=([1.0, 2.0], [3.0, 4.0])`` sets a different value per series at
# each end.

#: The names scipy's broadcast failure reports, one per position.
_FILL_ONE = "fill_value"
_FILL_BELOW = "fill_value (below)"
_FILL_ABOVE = "fill_value (above)"

#: scipy's sentinel, the only string `fill_value` accepts. Row interp1d-I3.
_FILL_EXTRAP = "extrapolate"


@njit
def _shape_text(sh):
    """A shape array as Python's tuple repr, ``(2,)`` or ``(2, 3)``.

    ``str`` of an int64 renders correctly inside ``@njit``; ``str`` of a
    float64 does not, which is why only integers are formatted here.
    """
    n = sh.shape[0]
    if n == 0:
        return "()"
    if n == 1:
        return "(" + str(sh[0]) + ",)"
    s = "(" + str(sh[0])
    for i in range(1, n):
        s += ", " + str(sh[i])
    return s + ")"


#: Refusal for a complex `fill_value` on a real `y`. scipy truncates it to
#: float64 with a ``ComplexWarning`` and returns a real result, measured; a
#: complex value answered with a real number and no signal is what this
#: raises instead. Row interp1d-I8.
_FILL_COMPLEX_MSG = "a complex fill_value needs a complex y"


@njit
def _raise_fill_complex():
    """Raise `_FILL_COMPLEX_MSG`.

    Called rather than raised inline: a bare ``raise`` in an ``@overload`` arm
    types the arm as returning ``none``, because the ``return`` after it is
    unreachable and inference drops it.
    """
    raise ValueError(_FILL_COMPLEX_MSG)


def _as_values(a):
    """`a` in the dtype the ordinates are stored in.

    complex128 for a complex `y` of any width and float64 for anything else,
    which is what scipy returns: measured, a complex64 `y` comes back
    complex128.
    """
    return np.ascontiguousarray(a).astype(
        np.complex128 if np.iscomplexobj(a) else np.float64)


@overload(_as_values)
def _as_values_ovl(a):
    """`_as_values` inside ``@njit``. The dtype is the whole decision."""
    if isinstance(a.dtype, types.Complex):
        def impl(a):
            return np.ascontiguousarray(a).astype(np.complex128)
        return impl

    def impl(a):
        return np.ascontiguousarray(a).astype(np.float64)
    return impl


def _splev_any(xs, t, c, k):
    """``splev`` over real or complex coefficients.

    FITPACK is float64, so a complex spline is evaluated as two real ones and
    recombined, which is what scipy's own complex ``BSpline`` does internally.
    """
    if np.iscomplexobj(c):
        return (_ev.splev(xs, t, np.ascontiguousarray(c.real), k, 0)
                + 1j * _ev.splev(xs, t, np.ascontiguousarray(c.imag), k, 0))
    return _ev.splev(xs, t, c, k, 0)


@overload(_splev_any)
def _splev_any_ovl(xs, t, c, k):
    """`_splev_any` inside ``@njit``. The real arm is the plain call."""
    if isinstance(c.dtype, types.Complex):
        def impl(xs, t, c, k):
            return (_ev.splev(xs, t, np.ascontiguousarray(np.real(c)), k, 0)
                    + 1j * _ev.splev(xs, t,
                                     np.ascontiguousarray(np.imag(c)), k, 0))
        return impl

    def impl(xs, t, c, k):
        return _ev.splev(xs, t, c, k, 0)
    return impl


@njit
def _bcast_fill(src, sshape, bshape, name):
    """scipy's ``_check_broadcast_up_to``, returning the ravelled result.

    Parameters
    ----------
    src : 1-D float64 ndarray
        The fill value's own C-order ravel.
    sshape : 1-D int64 ndarray
        Its shape.
    bshape : 1-D int64 ndarray
        The trailing shape of `y`, which is what the fill broadcasts up to.
    name : str
        ``'fill_value'``, ``'fill_value (below)'`` or ``'fill_value (above)'``,
        which is the position scipy names in the failure.

    Returns
    -------
    out : 1-D ndarray, length ``prod(bshape)``
        One value per series, in the dtype `src` carries and in the same C
        order the stored ``(n, T)`` ordinates use.

    Raises
    ------
    ValueError
        If the shapes do not broadcast, with scipy's text.
    """
    f = sshape.shape[0]
    r = bshape.shape[0]
    ok = f <= r
    if ok:
        for i in range(f):
            a = sshape[f - 1 - i]
            b = bshape[r - 1 - i]
            if a != 1 and a != b:
                ok = False
                break
    if not ok:
        raise ValueError(name + " argument must be able to broadcast up to "
                         "shape " + _shape_text(bshape) + " but had shape "
                         + _shape_text(sshape))
    t = 1
    for i in range(r):
        t *= bshape[i]
    out = np.empty(t, src.dtype)
    # C-order strides of the target, and of the source mapped onto it: a
    # source axis of length 1 gets stride 0 and repeats, which is what
    # broadcasting means.
    tstr = np.empty(r, np.int64)
    sstr = np.zeros(r, np.int64)
    acc = 1
    for d in range(r - 1, -1, -1):
        tstr[d] = acc
        acc *= bshape[d]
    acc = 1
    for i in range(f):
        d = r - 1 - i
        s = f - 1 - i
        sstr[d] = 0 if sshape[s] == 1 else acc
        acc *= sshape[s]
    for lin in range(t):
        j = 0
        for d in range(r):
            j += ((lin // tstr[d]) % bshape[d]) * sstr[d]
        out[lin] = src[j]
    return out


def _fill_one(value, bshape, name, proto):
    """One end of the fill, broadcast up to `bshape` and ravelled.

    `proto` is the ordinate array, and the fill is cast to its dtype. A
    complex `value` on a real `proto` raises `_FILL_COMPLEX_MSG`.
    """
    if np.iscomplexobj(value) and not np.iscomplexobj(proto):
        raise ValueError(_FILL_COMPLEX_MSG)
    dt = np.complex128 if np.iscomplexobj(proto) else np.float64
    a = np.ascontiguousarray(np.asarray(value, dtype=dt))
    return _bcast_fill(a.ravel(), np.asarray(a.shape, dtype=np.int64),
                       bshape, name)


@overload(_fill_one)
def _fill_one_ovl(value, bshape, name, proto):
    """`_fill_one` inside ``@njit``, one body per spelling of one end.

    A scalar has no shape to check, so it skips the broadcast entirely, which
    is also what scipy's ``arr_from.size == 1`` short circuit does. An array's
    rank is part of its type and its lengths are not, so the rank is read
    while compiling and the lengths at run time.
    """
    cx = isinstance(proto.dtype, types.Complex)
    # an array, a list and a homogeneous tuple all carry `dtype`, so one line
    # reaches the element type of every container spelling
    vd = getattr(value, 'dtype', value)
    if isinstance(vd, (types.Complex, complex)) and not cx:
        def impl(value, bshape, name, proto):
            _raise_fill_complex()
            return np.zeros(1, np.float64)
        return impl

    if isinstance(value, types.Array):
        nd = value.ndim

        def impl(value, bshape, name, proto):
            a = _as_values(value)
            sh = np.empty(nd, np.int64)
            for d in range(nd):
                sh[d] = a.shape[d]
            return _bcast_fill(a.ravel(), sh, bshape, name)
        return impl

    if isinstance(value, (types.List, types.ListType)):
        def impl(value, bshape, name, proto):
            a = _as_values(np.asarray(value))
            sh = np.empty(1, np.int64)
            sh[0] = a.shape[0]
            return _bcast_fill(a.ravel(), sh, bshape, name)
        return impl

    if isinstance(value, types.UniTuple):
        ln = len(value)

        def impl(value, bshape, name, proto):
            a = np.empty(ln, proto.dtype)
            for i in range(ln):
                a[i] = value[i]
            sh = np.empty(1, np.int64)
            sh[0] = ln
            return _bcast_fill(a, sh, bshape, name)
        return impl

    def impl(value, bshape, name, proto):
        t = 1
        for i in range(bshape.shape[0]):
            t *= bshape[i]
        out = np.empty(t, proto.dtype)
        for i in range(t):
            out[i] = value
        return out
    return impl


@njit
def _nan_pair(t, proto):
    """A NaN fill of length `t` in `proto`'s dtype, for the extrapolating
    sentinel, which resolves the flags and is never read."""
    out = np.empty(t, proto.dtype)
    for i in range(t):
        out[i] = np.nan
    return out


def _fill_resolve(fill_value, bshape, proto):
    """scipy's `fill_value`, resolved to ``(below, above, extrapolating)``.

    Parameters
    ----------
    fill_value : scalar, 2-tuple, sequence, or ``'extrapolate'``
    bshape : 1-D int64 ndarray
        The trailing shape of `y`, ``[1]`` for a 1-D `y`, which is the shape
        scipy broadcasts the fill up to.
    proto : ndarray
        The ordinates, whose dtype the fill is cast to.

    Returns
    -------
    below, above : 1-D ndarray, length ``prod(bshape)``
    extrapolating : bool
        True when `fill_value` was scipy's ``'extrapolate'`` sentinel, in
        which case the two arrays are NaN and never read.
    """
    if isinstance(fill_value, str):
        if fill_value != _FILL_EXTRAP:
            raise ValueError("could not convert string to float: np.str_('"
                             + fill_value + "')")
        t = int(np.prod(bshape)) if bshape.size else 1
        return _nan_pair(t, proto), _nan_pair(t, proto), True
    if isinstance(fill_value, tuple) and len(fill_value) == 2:
        return (_fill_one(fill_value[0], bshape, _FILL_BELOW, proto),
                _fill_one(fill_value[1], bshape, _FILL_ABOVE, proto), False)
    r = _fill_one(fill_value, bshape, _FILL_ONE, proto)
    return r, r.copy(), False


@overload(_fill_resolve)
def _fill_resolve_ovl(fill_value, bshape, proto):
    """`_fill_resolve` inside ``@njit``, one body per spelling.

    A tuple's LENGTH is part of its type, so scipy's
    ``isinstance(fill_value, tuple) and len(fill_value) == 2`` is answered
    while compiling. The ``'extrapolate'`` comparison is left at run time, so
    the sentinel may arrive as a variable rather than a literal.
    """
    if isinstance(fill_value, (types.UnicodeType, types.StringLiteral, str)):
        def impl(fill_value, bshape, proto):
            if fill_value != _FILL_EXTRAP:
                raise ValueError("could not convert string to float: "
                                 "np.str_('" + fill_value + "')")
            t = 1
            for i in range(bshape.shape[0]):
                t *= bshape[i]
            return _nan_pair(t, proto), _nan_pair(t, proto), True
        return impl

    if isinstance(fill_value, types.BaseTuple) and len(fill_value) == 2:
        def impl(fill_value, bshape, proto):
            return (_fill_one(fill_value[0], bshape, _FILL_BELOW, proto),
                    _fill_one(fill_value[1], bshape, _FILL_ABOVE, proto),
                    False)
        return impl

    def impl(fill_value, bshape, proto):
        r = _fill_one(fill_value, bshape, _FILL_ONE, proto)
        return r, r.copy(), False
    return impl


@njit
def _trailing_shape(y, ax):
    """`y`'s shape with the interpolation axis dropped, ``[1]`` when empty.

    This is scipy's ``broadcast_shape``: ``y.shape[:axis] + y.shape[axis+1:]``,
    replaced by ``(1,)`` for a 1-D `y`. It is what `fill_value` broadcasts up
    to, so it fixes how many series a per-series fill must supply.
    """
    nd = y.ndim
    if nd == 1:
        return np.ones(1, np.int64)
    r = np.empty(nd - 1, np.int64)
    k = 0
    for d in range(nd):
        if d != ax:
            r[k] = y.shape[d]
            k += 1
    return r


def _copy_or_view(a, copy):
    """`a` as contiguous float64, copied unless `copy` is False and safe.

    Row interp1d-I4. scipy's ``copy=False`` sets numpy's ``copy_if_needed``,
    so it keeps a reference and a later mutation of the caller's array shows
    through the interpolator. Reproducing that is only safe where no
    conversion is due: an array that is already C-contiguous float64 is
    handed straight back, and anything else is converted, which copies. A
    strided view is therefore still copied, which is what gotcha #0 requires.

    The layout and the dtype are part of the numba type, so the decision is
    taken while compiling and `copy` is read only on the branch that can
    honour it.
    """
    if copy:
        return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).copy()
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


@overload(_copy_or_view)
def _copy_or_view_ovl(a, copy):
    """`_copy_or_view` inside ``@njit``, one body per layout and dtype."""
    if (isinstance(a, types.Array) and a.dtype == types.float64
            and a.layout == 'C'):
        def impl(a, copy):
            if copy:
                return a.copy()
            return a
        return impl

    def impl(a, copy):
        return np.ascontiguousarray(np.asarray(a, np.float64))
    return impl


def _copy_or_view_y(a, copy):
    """`_copy_or_view` for the ORDINATES, which keep a complex dtype.

    `x` goes through `_copy_or_view` and is float64 there, so a complex `x` is
    refused; `y` reaches this one, which holds complex128 for a complex `y` of
    any width and float64 for anything else.
    """
    if copy:
        return _as_values(a).copy()
    return _as_values(a)


@overload(_copy_or_view_y)
def _copy_or_view_y_ovl(a, copy):
    """`_copy_or_view_y` inside ``@njit``, one body per layout and dtype."""
    if (isinstance(a, types.Array) and a.layout == 'C'
            and a.dtype in (types.float64, types.complex128)):
        def impl(a, copy):
            if copy:
                return a.copy()
            return a
        return impl

    def impl(a, copy):
        return _as_values(a)
    return impl


@njit
def _is_spline(kind):
    """True for the kinds evaluated through a B-spline rather than an index.

    Codes 4 and 5 are quadratic and cubic; code 7 is any other order, with the
    degree held beside it. Collected here so adding an order touches one place.
    """
    return kind == 4 or kind == 5 or kind == _SPLINE_ANY


def _bounds_flag(bounds_error, extrapolate):
    """scipy's ``bounds_error=None``: raise unless extrapolating."""
    if bounds_error is None:
        return not extrapolate
    return bool(bounds_error)


@overload(_bounds_flag)
def _bounds_flag_ovl(bounds_error, extrapolate):
    """`_bounds_flag` inside ``@njit``. The ``None`` arm is decided by type."""
    if isinstance(bounds_error, types.NoneType) or bounds_error is None:
        def impl(bounds_error, extrapolate):
            return not extrapolate
        return impl

    def impl(bounds_error, extrapolate):
        return bool(bounds_error)
    return impl


#: scipy's refusal when ``np.take`` receives a bool `axis`.
_AXIS_TAKE_MSG = "an integer is required for the axis"


def _check_axis_take(axis, assume_sorted):
    """Refuse a ``bool`` `axis` on the sorting path, as ``np.take`` does.

    scipy reorders with ``np.take(y, ind, axis=axis)`` when `assume_sorted` is
    False, and numpy refuses a bool there with
    ``TypeError('an integer is required for the axis')``. With
    ``assume_sorted=True`` no ``take`` runs and the bool is read as the index
    0 or 1. MEASURED over 1-D and 2-D `y` at both flags: the refusal appears
    in exactly the four ``assume_sorted=False`` cells.

    It runs after the length check, which is where scipy reaches ``take``.
    """
    if isinstance(axis, (bool, np.bool_)) and not assume_sorted:
        raise TypeError(_AXIS_TAKE_MSG)


@overload(_check_axis_take)
def _check_axis_take_ovl(axis, assume_sorted):
    """`_check_axis_take` inside ``@njit``.

    Whether `axis` is a bool is a TYPE question, answered while compiling;
    whether the sort runs is a value, so that half stays at run time.
    """
    at = axis.value if isinstance(axis, types.Omitted) else axis
    is_bool = isinstance(at, types.Boolean) or (at is True or at is False)
    if is_bool:
        def impl(axis, assume_sorted):
            if not assume_sorted:
                raise TypeError(_AXIS_TAKE_MSG)
        return impl

    def impl(axis, assume_sorted):
        pass
    return impl


@njit
def _raise_below(v, lo):
    """Raise scipy's below-range ``ValueError``, naming the value and bound.

    Row interp1d-I10. The two floats cannot be rendered by ``str`` inside
    ``@njit``, which writes the literal text ``<object type:float64>``, so the
    message is built in `objmode`. It costs nothing on the path that does not
    raise, and a caller's ``prange`` over a function reaching this one still
    parallelises.
    """
    with objmode(msg='unicode_type'):
        msg = (f"A value ({v}) in x_new is below the interpolation range's "
               f"minimum value ({lo}).")
    raise ValueError(msg)


@njit
def _raise_above(v, hi):
    """Raise scipy's above-range ``ValueError``. Row interp1d-I10."""
    with objmode(msg='unicode_type'):
        msg = (f"A value ({v}) in x_new is above the interpolation range's "
               f"maximum value ({hi}).")
    raise ValueError(msg)


@njit
def _check_bounds(xs, lo, hi):
    """scipy's ``_check_bounds``: the whole batch below, then the whole above.

    The two passes are what makes the reported value scipy's. A single pass
    reports whichever end the first offending point falls off, so
    ``x_new = [2.0, -1.0]`` would name 2.0 where scipy names -1.0.
    """
    m = xs.shape[0]
    for j in range(m):
        if xs[j] < lo:
            _raise_below(xs[j], lo)
    for j in range(m):
        if xs[j] > hi:
            _raise_above(xs[j], hi)


@njit
def _sorted_xy(x, y, assume_sorted):
    """`x`, `y` reordered by ascending `x`, unless already vouched for."""
    if assume_sorted:
        return x, y
    order = np.argsort(x, kind="mergesort")
    return np.ascontiguousarray(x[order]), np.ascontiguousarray(y[order])


# ---------------------------------------------------------------------------
# per-kind single-point njit helpers (shared by .ev and .ev_one)
# ---------------------------------------------------------------------------

@njit
def _linear_one(x, y, xv):
    """Piecewise-linear value at xv; extrapolates with the end segment."""
    n = x.shape[0]
    if xv <= x[0]:
        i = 0
    elif xv >= x[n - 1]:
        i = n - 2
    else:
        lo = 0
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x[mid] <= xv:
                lo = mid
            else:
                hi = mid
        i = lo
    return y[i] + (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (xv - x[i])


@njit
def _nearest_one(x, y, xv, up):
    """Nearest-neighbour value. up=False -> ties round down (scipy 'nearest',
    side='left'); up=True -> ties round up (scipy 'nearest-up')."""
    n = x.shape[0]
    idx = 0
    for j in range(n - 1):
        m = 0.5 * x[j] + 0.5 * x[j + 1]      # midpoint, overflow-safe
        if up:
            if m <= xv:
                idx = j + 1
            else:
                break
        else:
            if m < xv:
                idx = j + 1
            else:
                break
    return y[idx]


@njit
def _previous_one(x, y, xv):
    """Value of the previous sample (largest x[i] <= xv), clamped to node 0."""
    n = x.shape[0]
    idx = 0
    for i in range(n):
        if x[i] <= xv:
            idx = i
        else:
            break
    return y[idx]


@njit
def _next_one(x, y, xv):
    """Value of the next sample, smallest x[i] >= xv, clamped at the end."""
    n = x.shape[0]
    idx = n - 1
    for i in range(n - 1, -1, -1):
        if x[i] >= xv:
            idx = i
        else:
            break
    return y[idx]


# ---------------------------------------------------------------------------
# jitclass
# ---------------------------------------------------------------------------

@njit
def _check_increasing(xx):
    """Reject a non-increasing `x` after any sorting has been applied."""
    for i in range(xx.shape[0] - 1):
        if xx[i + 1] <= xx[i]:
            raise ValueError("x must be a strictly increasing sequence")


@njit
def _sorted_xy_nd(x, y2, assume_sorted):
    """`x` and an ``(n, T)`` `y` reordered by ascending `x`, rows together."""
    if assume_sorted:
        return x, y2
    order = np.argsort(x, kind="mergesort")
    xo = np.ascontiguousarray(x[order])
    yo = np.empty((y2.shape[0], y2.shape[1]), y2.dtype)
    for i in range(y2.shape[0]):
        src = order[i]
        for j in range(y2.shape[1]):
            yo[i, j] = y2[src, j]
    return xo, yo


@njit
def _raw_one_nd(kind, x, y2, t, c2, k, xv, out):
    """Kind-dispatched value at `xv` for every series, ignoring bounds/fill."""
    tt = y2.shape[1]
    if kind == 0:
        for j in range(tt):
            out[j] = _linear_one(x, np.ascontiguousarray(y2[:, j]), xv)
    elif kind == 1:
        for j in range(tt):
            out[j] = _nearest_one(x, np.ascontiguousarray(y2[:, j]), xv, False)
    elif kind == 6:
        for j in range(tt):
            out[j] = _nearest_one(x, np.ascontiguousarray(y2[:, j]), xv, True)
    elif kind == 2:
        for j in range(tt):
            out[j] = _previous_one(x, np.ascontiguousarray(y2[:, j]), xv)
    elif kind == 3:
        for j in range(tt):
            out[j] = _next_one(x, np.ascontiguousarray(y2[:, j]), xv)
    else:
        xa = np.empty(1, np.float64)
        xa[0] = xv
        for j in range(tt):
            yv = _splev_any(xa, t, np.ascontiguousarray(c2[:, j]), k)
            out[j] = yv[0]


@njit
def _eval_batch_nd(kind, x, y2, t, c2, k, n, xs, bounds_error, extrapolate,
                   fill_below, fill_above):
    """Evaluate every series over a batch, shape ``(m, T)``.

    Which points are out of range is decided a whole row at a time: the series
    share `x`, so a query point is out of bounds for all of them or none. WHAT
    is written there is per series, because `fill_below` and `fill_above`
    carry one value each per series (rows interp1d-I1 and -I2).
    """
    xs = np.ascontiguousarray(np.asarray(xs, np.float64))
    m = xs.shape[0]
    tt = y2.shape[1]
    x0 = x[0]
    x1 = x[n - 1]
    if bounds_error:
        _check_bounds(xs, x0, x1)
    out = np.empty((m, tt), y2.dtype)
    # the spline kinds go through one splev call per series for the whole batch
    if _is_spline(kind):
        raw = np.empty((m, tt), y2.dtype)
        for j in range(tt):
            col = _splev_any(xs, t, np.ascontiguousarray(c2[:, j]), k)
            for q in range(m):
                raw[q, j] = col[q]
    else:
        raw = np.empty((0, 0), y2.dtype)
    row = np.empty(tt, y2.dtype)
    for q in range(m):
        xv = xs[q]
        below = xv < x0
        above = xv > x1
        if below or above:
            if not extrapolate:
                if below:
                    for j in range(tt):
                        out[q, j] = fill_below[j]
                else:
                    for j in range(tt):
                        out[q, j] = fill_above[j]
                continue
        if _is_spline(kind):
            for j in range(tt):
                out[q, j] = raw[q, j]
        else:
            _raw_one_nd(kind, x, y2, t, c2, k, xv, row)
            for j in range(tt):
                out[q, j] = row[j]
        if extrapolate:
            if (kind == 2 and below) or (kind == 3 and above):
                for j in range(tt):
                    out[q, j] = np.nan
    return out


@njit
def _eval_point_nd(kind, x, y2, t, c2, k, n, xv, bounds_error, extrapolate,
                   fill_below, fill_above):
    """Evaluate every series at one point, shape ``(T,)``."""
    tt = y2.shape[1]
    out = np.empty(tt, y2.dtype)
    below = xv < x[0]
    above = xv > x[n - 1]
    if below or above:
        if bounds_error:
            if below:
                _raise_below(xv, x[0])
            else:
                _raise_above(xv, x[n - 1])
        if not extrapolate:
            if below:
                for j in range(tt):
                    out[j] = fill_below[j]
            else:
                for j in range(tt):
                    out[j] = fill_above[j]
            return out
    _raw_one_nd(kind, x, y2, t, c2, k, xv, out)
    if extrapolate:
        if (kind == 2 and below) or (kind == 3 and above):
            for j in range(tt):
                out[j] = np.nan
    return out


def _interp1d_spec_for(dt):
    """The 1-D jitclass spec for ordinates of this dtype.

    `x` and the knot vector are float64 whatever `y` is: the abscissae are
    ordered and the knots follow them.
    """
    return [
        ('kind', int64),
        ('x', float64[:]),
        ('y', dt[:]),
        ('t', float64[:]),
        ('c', dt[:]),
        ('k', int64),
        ('n', int64),
        ('bounds_error', boolean),
        ('extrapolate', boolean),
        ('fill_below', dt),
        ('fill_above', dt),
    ]


_interp1d_spec = _interp1d_spec_for(float64)

_INTERP1D_1D_SRC = '''
class {cname}:
    """1-D interpolator jitclass over {dword} ordinates, the counterpart of
    ``scipy.interpolate.interp1d``.

    Normally constructed via the `interp1d` factory below, which validates the
    input and precomputes the FITPACK ``tck`` for the spline kinds 4/5. The
    constructor can also be called directly, including inside ``@njit``, if
    a ``tck`` is already in hand.

    Parameters
    ----------
    x : 1-D float64 ndarray, length n >= 2
        Sample abscissae, strictly increasing (not re-checked here; the
        factory checks it).
    y : 1-D {dword} ndarray, length n
        Sample ordinates.
    kind : int
        Interpolation kind, as the resolved int code. The `interp1d` factory
        accepts scipy's string and resolves it here:
        0 = 'linear' (also 'slinear'),
        1 = 'nearest' (ties round DOWN, scipy side='left'),
        2 = 'previous' (also 'zero'), 3 = 'next',
        4 = 'quadratic' (k=2 interpolating spline),
        5 = 'cubic' (k=3 interpolating spline),
        6 = 'nearest-up' (ties round UP, scipy side='right').
        No default inside the class; the factory defaults to ``'linear'``,
        which is scipy's default too.
    t : 1-D float64 ndarray
        Knot vector for kinds 4/5, empty (length 0) for the other kinds.
    c : 1-D {dword} ndarray
        B-spline coefficients matching `t`, empty for the non-spline kinds.
    k : int
        Spline degree: 2 for kind 4, 3 for kind 5, 0 (unused) otherwise.
    bounds_error : bool
        If True, evaluating at any point outside ``[x[0], x[-1]]`` raises
        ``ValueError``. The factory resolves scipy's ``bounds_error=None``
        into this bool before constructing.
    extrapolate : bool
        If True, out-of-bounds points are extrapolated instead of filled.
        This is scipy's ``fill_value="extrapolate"``.
    fill_below, fill_above : {dword}
        Values returned below ``x[0]`` and above ``x[-1]`` when
        ``bounds_error`` is False and `extrapolate` is False. The factory
        resolves scipy's `fill_value` into the pair. A 1-D `y` has trailing
        shape ``(1,)``, so one number a side covers it. Both default to NaN
        through the factory.

    Attributes
    ----------
    fill_value : ({dword}, {dword})
        The resolved ``(below, above)`` pair, assignable as a pair.

    Methods
    -------
    ev(xs)
        Evaluate at a 1-D array of points -> 1-D {dword} array.
    ev_one(xv)
        Evaluate at one scalar point -> {dword}.
    set_fill_value(value)
        Re-resolve `fill_value` after construction, taking a scalar, a
        2-tuple, a sequence or ``'extrapolate'``.
    __getitem__(xs)
        ``f[xs]`` is sugar for ``f.ev(xs)``.

    Notes
    -----
    Deviations from scipy: ``y`` is capped at rank 3, where scipy takes any
    rank, and `fill_value` reads back as the resolved pair rather than the
    object last assigned.

    Generated by ``_make_interp1d_1d({dtkey!r})`` in
    ``scijit/interpolate/_interp1d.py`` and bound there under this name. One
    class exists per dtype of `y`, because a jitclass field has one dtype;
    they share this source and differ only in what they hold.

    CONSTRUCTOR DEFAULTS ARE PYTHON-ONLY. This constructor declares no
    defaults at all, so all ten arguments must always be passed. The wider
    trap still applies package-wide: a jitclass constructor's Python-level
    defaults are not visible inside ``@njit``, where an omitted argument
    raises "TypeError: invalid number of args". Pass every argument
    explicitly in compiled code, or call the `interp1d` factory, whose
    defaults do work in both worlds.

    Accuracy vs ``scipy.interpolate.interp1d`` on 12 random nodes evaluated
    at 97 in-range points: kinds 1/2/3/6 exactly 0.0, kind 0 (linear)
    1.11e-16, kinds 4 and 5 both 4.44e-16. With ``extrapolate=True`` over
    x[0]-2 .. x[-1]+2: linear and nearest exactly 0.0, quadratic 1.42e-14,
    cubic 8.53e-14.

    prange-safe: yes for every kind (pure array reads, or the stateless
    FITPACK ``splev``). Build instances outside the parallel loop.
    """

    def __init__(self, x, y, kind, t, c, k, bounds_error, extrapolate,
                 fill_below, fill_above):
        n = x.shape[0]
        # ascontiguousarray and nothing else: it hands a C-contiguous array
        # straight back, so `copy=False` reaches the field intact (row
        # interp1d-I4), and copies a strided one, which is what gotcha #0
        # exists for. The dtype is settled by the factory, which is also
        # where the copy decision is taken.
        self.x = np.ascontiguousarray(x)
        self.y = np.ascontiguousarray(y)
        self.t = np.ascontiguousarray(t)
        self.c = np.ascontiguousarray(c)
        self.kind = kind
        self.k = k
        self.n = n
        self.bounds_error = bounds_error
        self.extrapolate = extrapolate
        self.fill_below = fill_below
        self.fill_above = fill_above

    @property
    def fill_value(self):
        """The resolved ``(below, above)`` pair. Row interp1d-I9.

        scipy's getter hands back the object last assigned, a number, a pair
        or the string. One property has one type here, so this is the pair
        the object will actually use, and `set_fill_value` is the spelling
        that accepts everything scipy's setter accepts.
        """
        return self.fill_below, self.fill_above

    @fill_value.setter
    def fill_value(self, pair):
        """Assign the ``(below, above)`` pair, leaving the flags alone."""
        self.fill_below = pair[0]
        self.fill_above = pair[1]

    def set_fill_value(self, value):
        """scipy's ``fill_value`` setter, re-resolving the flags with it.

        Row interp1d-I9. A property setter on a jitclass is typed by its
        getter's return type, measured, so it cannot take a number at one
        call site and ``'extrapolate'`` at another. A METHOD can, and does:
        this accepts a scalar, a 2-tuple, a sequence, or scipy's
        ``'extrapolate'``, from Python and from inside ``@njit``.

        Parameters
        ----------
        value : float, 2-tuple, sequence, or ``'extrapolate'``
            As `interp1d`'s `fill_value`.

        Raises
        ------
        ValueError
            On ``'extrapolate'`` while `bounds_error` is set, with scipy's
            text; or on a `value` that does not broadcast.

        Notes
        -----
        ``'extrapolate'`` clears `bounds_error`, exactly as scipy's setter
        does, and leaves the numeric fill untouched because it is no longer
        read.
        """
        below, above, ex = _fill_resolve(value, np.ones(1, np.int64),
                                         self.y)
        if ex:
            if self.bounds_error:
                raise ValueError(
                    "Cannot extrapolate and raise at the same time.")
            self.bounds_error = False
            self.extrapolate = True
        else:
            self.fill_below = below[0]
            self.fill_above = above[0]
            self.extrapolate = False

    def _raw_one(self, xv):
        """Kind-dispatched value at `xv`, ignoring bounds/fill/extrapolate.

        Internal. Every branch is typed by numba even for a fixed runtime
        ``kind``, including the FITPACK ``splev`` call below.
        """
        if self.kind == 0:
            return _linear_one(self.x, self.y, xv)
        elif self.kind == 1:
            return _nearest_one(self.x, self.y, xv, False)
        elif self.kind == 6:
            return _nearest_one(self.x, self.y, xv, True)
        elif self.kind == 2:
            return _previous_one(self.x, self.y, xv)
        elif self.kind == 3:
            return _next_one(self.x, self.y, xv)
        # spline kinds 4 (k=2) / 5 (k=3)
        xa = np.empty(1, np.float64)
        xa[0] = xv
        yv = _splev_any(xa, self.t, self.c, self.k)
        return yv[0]

    def ev_one(self, xv):
        """Interpolate at a single point (scipy's ``f(scalar)``).

        Parameters
        ----------
        xv : float
            Abscissa to evaluate at. May lie outside ``[x[0], x[-1]]``; what
            happens then is set by `bounds_error`, `extrapolate` and
            `fill_value`.

        Returns
        -------
        float
            The interpolated value; `fill_value` if `xv` is out of bounds with
            ``bounds_error=False`` and ``extrapolate=False``; NaN if
            ``extrapolate=True`` with kind 2 ('previous') below the range or
            kind 3 ('next') above it, which is what scipy returns there.

        Raises
        ------
        ValueError
            If `xv` is outside ``[x[0], x[-1]]`` and ``bounds_error`` is True.
        """
        below = xv < self.x[0]
        above = xv > self.x[self.n - 1]
        if below or above:
            if self.bounds_error:
                if below:
                    _raise_below(xv, self.x[0])
                else:
                    _raise_above(xv, self.x[self.n - 1])
            if not self.extrapolate:
                return self.fill_below if below else self.fill_above
        r = self._raw_one(xv)
        if self.extrapolate:
            if self.kind == 2 and below:
                r = np.nan
            elif self.kind == 3 and above:
                r = np.nan
        return r

    def ev(self, xs):
        """Interpolate at an array of points (scipy's ``f(array)``).

        Parameters
        ----------
        xs : 1-D array of float64
            Abscissae to evaluate at, in any order. Wrapped in
            ``ascontiguousarray`` internally, so a strided view is safe.

        Returns
        -------
        out : 1-D {dword} ndarray, same length as `xs`
            Interpolated values, with out-of-bounds entries handled exactly as
            in `ev_one`.

        Raises
        ------
        ValueError
            If any entry of `xs` is out of bounds and ``bounds_error`` is
            True.

        Notes
        -----
        For the spline kinds 4 and 5 the whole batch goes through a single
        FITPACK ``splev`` call, so this is markedly faster than looping over
        `ev_one`.
        """
        xs = np.ascontiguousarray(np.asarray(xs, np.float64))
        m = xs.shape[0]
        x0 = self.x[0]
        x1 = self.x[self.n - 1]
        if self.bounds_error:
            _check_bounds(xs, x0, x1)
        out = np.empty(m, {dt})
        # evaluate the whole batch once for the spline kinds (one Fortran call)
        if _is_spline(self.kind):
            raw = _splev_any(xs, self.t, self.c, self.k)
        else:
            raw = np.empty(0, {dt})
        for j in range(m):
            xv = xs[j]
            below = xv < x0
            above = xv > x1
            if below or above:
                if not self.extrapolate:
                    out[j] = self.fill_below if below else self.fill_above
                    continue
            if _is_spline(self.kind):
                r = raw[j]
            else:
                r = self._raw_one(xv)
            if self.extrapolate:
                if self.kind == 2 and below:
                    r = np.nan
                elif self.kind == 3 and above:
                    r = np.nan
            out[j] = r
        return out

    def __getitem__(self, xs):
        """``f[xs]`` -- sugar for ``f.ev(xs)``.

        ``f(xs)`` reaches the same method and is the spelling scipy uses.

        Parameters
        ----------
        xs : 1-D array of float64

        Returns
        -------
        out : 1-D {dword} ndarray
        """
        return self.ev(xs)
'''

#: numba type, source spelling and the word the docstrings use, per dtype key.
#: 'f' holds float64 ordinates, 'c' complex128. scipy returns complex128 for a
#: complex `y` of any width, measured.
_I1D_DT = {'f': (float64, 'np.float64', 'float64'),
           'c': (complex128, 'np.complex128', 'complex128')}


def _make_interp1d_1d(dtkey):
    """Build the 1-D `interp1d` class for ordinates of this dtype.

    Parameters
    ----------
    dtkey : {'f', 'c'}

    Returns
    -------
    cls : jitclass
        Decorated with `scijitclass` against ``_interp1d_spec_for(dt)`` and
        bound in this module as ``_Interp1D`` or ``_Interp1DC``.
    """
    dt, dtsrc, dword = _I1D_DT[dtkey]
    cname = '_Interp1D' + ('C' if dtkey == 'c' else '')
    src = _INTERP1D_1D_SRC.format(cname=cname, dt=dtsrc, dword=dword,
                                  dtkey=dtkey)
    return scijitclass(_interp1d_spec_for(dt))(_define(src, globals(), cname))


#: Both 1-D classes, keyed by dtype. The factory chooses one while compiling.
_I1D_CLASSES = {k: _make_interp1d_1d(k) for k in ('f', 'c')}
_Interp1D = _I1D_CLASSES['f']


def _interp1d_nd_spec_for(dt):
    """The N-D jitclass spec for ordinates of this dtype."""
    return [
        ('kind', int64),
        ('x', float64[:]),
        ('y', dt[:, :]),
        ('t', float64[:]),
        ('c', dt[:, :]),
        ('k', int64),
        ('n', int64),
        ('bounds_error', boolean),
        ('extrapolate', boolean),
        # one fill value PER SERIES, flattened in the same C order as the
        # stored (n, T) ordinates. scipy broadcasts `fill_value` up to y's
        # trailing shape, so a length-T array is what the contract needs
        # (interp1d-I2).
        ('fill_below', dt[:]),
        ('fill_above', dt[:]),
        ('rest', int64[:]),
        ('axis', int64),
    ]


_interp1d_nd_spec = _interp1d_nd_spec_for(float64)

_INTERP1D_ND_SRC = '''
class {cname}:
    """1-D interpolator over T series of {dword} sharing one `x`, for a
    rank-{rank} `y`.

    The counterpart of ``scipy.interpolate.interp1d`` with an N-D `y`. Every
    position on the axes other than `axis` is an independent series. `rest`
    carries their shape, `axis` where the interpolation axis sat in `y`.

    A query point is out of range for every series at once, so `bounds_error`
    and `extrapolate` apply to a whole row. `fill_value` does not: `fill_below`
    and `fill_above` hold one value per series, so an out-of-range row can
    take a different number in each column.

    Generated by ``_make_interp1d_nd({rank}, {dtkey!r})`` in
    ``scijit/interpolate/_interp1d.py`` and bound there under this name. One
    class exists per supported rank of `y` and per dtype; they share this
    source and every kernel, and differ in the rank they hand back and in
    what they hold. `scijit.interpolate._ndaxis` says why.
    """

    def __init__(self, x, y2, kind, t, c2, k, bounds_error, extrapolate,
                 fill_below, fill_above, rest, axis):
        self.x = np.ascontiguousarray(x)
        self.y = np.ascontiguousarray(y2)
        self.t = np.ascontiguousarray(t)
        self.c = np.ascontiguousarray(c2)
        self.kind = kind
        self.k = k
        self.n = x.shape[0]
        self.bounds_error = bounds_error
        self.extrapolate = extrapolate
        self.fill_below = fill_below
        self.fill_above = fill_above
        self.rest = rest
        self.axis = axis

    @property
    def fill_value(self):
        """The resolved ``(below, above)`` pair, one value per series.

        Row interp1d-I9. scipy's getter returns the object last assigned;
        one property has one type here, so this is the pair the object will
        use. `set_fill_value` accepts everything scipy's setter accepts.
        """
        return self.fill_below, self.fill_above

    @fill_value.setter
    def fill_value(self, pair):
        """Assign the ``(below, above)`` pair, leaving the flags alone."""
        self.fill_below = _as_values(np.asarray(pair[0]))
        self.fill_above = _as_values(np.asarray(pair[1]))

    def set_fill_value(self, value):
        """scipy's ``fill_value`` setter, re-resolving the flags with it.

        Row interp1d-I9. Takes a scalar, a 2-tuple, a sequence broadcast up
        to the trailing shape, or scipy's ``'extrapolate'``, from Python and
        from inside ``@njit``. ``'extrapolate'`` clears `bounds_error` as
        scipy's setter does, and raises scipy's
        ``Cannot extrapolate and raise at the same time.`` when it is set.
        """
        below, above, ex = _fill_resolve(value, self.rest, self.y)
        if ex:
            if self.bounds_error:
                raise ValueError(
                    "Cannot extrapolate and raise at the same time.")
            self.bounds_error = False
            self.extrapolate = True
        else:
            self.fill_below = below
            self.fill_above = above
            self.extrapolate = False

    def ev(self, xs):
        """Interpolate every series at an array of points.

        Parameters
        ----------
        xs : 1-D array of float64
            Abscissae to evaluate at, in any order.

        Returns
        -------
        out : {rank}-D {dword} ndarray
            The shape of `y`, with the interpolation axis replaced by
            ``len(xs)``.

        Raises
        ------
        ValueError
            If any entry of `xs` is out of bounds and ``bounds_error`` is True.
        """
        return _nda._restore{rank}(
            _eval_batch_nd(self.kind, self.x, self.y, self.t, self.c, self.k,
                           self.n, xs, self.bounds_error, self.extrapolate,
                           self.fill_below, self.fill_above),
            self.rest, self.axis)

    def ev_one(self, xv):
        """Interpolate every series at a single point (scipy's ``f(scalar)``).

        Returns
        -------
        out : {trail}-D {dword} ndarray
            The shape of `y` with the interpolation axis removed.
        """
        return _nda._point{rank}(
            _eval_point_nd(self.kind, self.x, self.y, self.t, self.c, self.k,
                           self.n, xv, self.bounds_error, self.extrapolate,
                           self.fill_below, self.fill_above), self.rest)

    def __getitem__(self, xs):
        """``f[xs]`` -- sugar for ``f.ev(xs)``."""
        return self.ev(xs)
'''


def _make_interp1d_nd(rank, dtkey):
    """Build the `interp1d` N-D class for a `y` of this rank and dtype.

    Parameters
    ----------
    rank : int
        Rank of the caller's `y`, from 2 to ``1 + _ND_MAXTRAIL``.
    dtkey : {'f', 'c'}
        ``'f'`` for float64 ordinates, ``'c'`` for complex128.

    Returns
    -------
    cls : jitclass
        The class, decorated with `scijitclass` against
        ``_interp1d_nd_spec_for(dt)`` and bound in this module as
        ``_Interp1DND<rank-1>`` or ``_Interp1DCND<rank-1>``.
    """
    dt, dtsrc, dword = _I1D_DT[dtkey]
    trail = rank - 1
    cname = '_Interp1D%sND%d' % ('C' if dtkey == 'c' else '', trail)
    src = _INTERP1D_ND_SRC.format(rank=rank, trail=trail, cname=cname,
                                  dt=dtsrc, dword=dword, dtkey=dtkey)
    return scijitclass(_interp1d_nd_spec_for(dt))(
        _define(src, globals(), cname))


#: Every N-D class, keyed by (dtype key, rank of `y`).
_I1D_ND_CLASSES = {}

for _rank in _ND_RANKS:
    for _dtkey in ('f', 'c'):
        _I1D_ND_CLASSES[(_dtkey, _rank)] = _make_interp1d_nd(_rank, _dtkey)
del _rank, _dtkey

for _rank in _ND_RANKS:
    globals()["_Interp1DND%d" % (_rank - 1)] = _I1D_ND_CLASSES[('f', _rank)]
del _rank


def _i1d_cons(xx, yy, kd, t, c, k, be, extrapolate, fb, fa):
    """Construct the 1-D class matching the dtype of `yy`. Internal."""
    return _I1D_CLASSES['c' if np.iscomplexobj(yy) else 'f'](
        xx, yy, kd, t, c, k, be, extrapolate, fb, fa)


@overload(_i1d_cons)
def _i1d_cons_ovl(xx, yy, kd, t, c, k, be, extrapolate, fb, fa):
    """`_i1d_cons` inside ``@njit``. The dtype is the whole decision, so the
    class is a constant in the compiled body."""
    cls = _I1D_CLASSES['c' if isinstance(yy.dtype, types.Complex) else 'f']

    def impl(xx, yy, kd, t, c, k, be, extrapolate, fb, fa):
        return cls(xx, yy, kd, t, c, k, be, extrapolate, fb, fa)
    return impl


def _i1d_cons_nd(y, xx, yy, kd, t, c, k, be, extrapolate, fb, fa, rest, ax):
    """Construct the N-D class matching the rank and dtype of `y`. Internal.

    `y` is the caller's array, passed only so the class can be read off its
    type: the stored ordinates `yy` are ``(n, T)`` whatever the rank was.
    """
    return _I1D_ND_CLASSES['c' if np.iscomplexobj(y) else 'f', np.ndim(y)](
        xx, yy, kd, t, c, k, be, extrapolate, fb, fa, rest, ax)


@overload(_i1d_cons_nd)
def _i1d_cons_nd_ovl(y, xx, yy, kd, t, c, k, be, extrapolate, fb, fa, rest,
                     ax):
    """`_i1d_cons_nd` inside ``@njit``, one arm per class."""
    cls = _I1D_ND_CLASSES[('c' if isinstance(y.dtype, types.Complex) else 'f',
                           y.ndim)]

    def impl(y, xx, yy, kd, t, c, k, be, extrapolate, fb, fa, rest, ax):
        return cls(xx, yy, kd, t, c, k, be, extrapolate, fb, fa, rest, ax)
    return impl


def _mis_any(xx, yy, k):
    """The not-a-knot interpolating fit, over real or complex ordinates.

    The knot vector follows `xx` and `k` alone, so a complex fit is two real
    fits sharing one set of knots, which is how scipy builds a complex
    ``BSpline`` too.
    """
    if np.iscomplexobj(yy):
        t, cr = _mis_core(xx, np.ascontiguousarray(yy.real), k, _NO_KNOTS,
                          False, _NO_ORD, _NO_VAL, _NO_ORD, _NO_VAL, False)
        t2, ci = _mis_core(xx, np.ascontiguousarray(yy.imag), k, _NO_KNOTS,
                           False, _NO_ORD, _NO_VAL, _NO_ORD, _NO_VAL, False)
        return t, cr + 1j * ci
    return _mis_core(xx, yy, k, _NO_KNOTS, False, _NO_ORD, _NO_VAL, _NO_ORD,
                     _NO_VAL, False)


@overload(_mis_any)
def _mis_any_ovl(xx, yy, k):
    """`_mis_any` inside ``@njit``. The real arm is the plain fit."""
    if isinstance(yy.dtype, types.Complex):
        def impl(xx, yy, k):
            t, cr = _mis_core(xx, np.ascontiguousarray(np.real(yy)), k,
                              _NO_KNOTS, False, _NO_ORD, _NO_VAL, _NO_ORD,
                              _NO_VAL, False)
            t2, ci = _mis_core(xx, np.ascontiguousarray(np.imag(yy)), k,
                               _NO_KNOTS, False, _NO_ORD, _NO_VAL, _NO_ORD,
                               _NO_VAL, False)
            return t, cr + 1j * ci
        return impl

    def impl(xx, yy, k):
        return _mis_core(xx, yy, k, _NO_KNOTS, False, _NO_ORD, _NO_VAL,
                         _NO_ORD, _NO_VAL, False)
    return impl


def _mis_any_nd(xx, yy, k):
    """`_mis_any` for the stored ``(n, T)`` layout."""
    if np.iscomplexobj(yy):
        t, cr = _mis_core_nd(xx, np.ascontiguousarray(yy.real), k, _NO_KNOTS,
                             False, _NO_ORD, _NO_VAL, _NO_ORD, _NO_VAL, False)
        t2, ci = _mis_core_nd(xx, np.ascontiguousarray(yy.imag), k, _NO_KNOTS,
                              False, _NO_ORD, _NO_VAL, _NO_ORD, _NO_VAL,
                              False)
        return t, cr + 1j * ci
    return _mis_core_nd(xx, yy, k, _NO_KNOTS, False, _NO_ORD, _NO_VAL,
                        _NO_ORD, _NO_VAL, False)


@overload(_mis_any_nd)
def _mis_any_nd_ovl(xx, yy, k):
    """`_mis_any_nd` inside ``@njit``. The real arm is the plain fit."""
    if isinstance(yy.dtype, types.Complex):
        def impl(xx, yy, k):
            t, cr = _mis_core_nd(xx, np.ascontiguousarray(np.real(yy)), k,
                                 _NO_KNOTS, False, _NO_ORD, _NO_VAL, _NO_ORD,
                                 _NO_VAL, False)
            t2, ci = _mis_core_nd(xx, np.ascontiguousarray(np.imag(yy)), k,
                                  _NO_KNOTS, False, _NO_ORD, _NO_VAL,
                                  _NO_ORD, _NO_VAL, False)
            return t, cr + 1j * ci
        return impl

    def impl(xx, yy, k):
        return _mis_core_nd(xx, yy, k, _NO_KNOTS, False, _NO_ORD, _NO_VAL,
                            _NO_ORD, _NO_VAL, False)
    return impl


@njit
def _i1d_prep(xr, flat, kd, kdeg, assume_sorted):
    """Sort, check and (for a spline kind) fit, for an N-D `y` of any rank.

    The stored layout is ``(n, T)`` whatever the caller's rank, so this runs
    once per interpolator rather than once per rank. It is the whole body of
    `_interp1d_nd`'s branches apart from the flatten and the class name.

    Returns
    -------
    xx, yy : ndarray
        The sorted abscissae and the ``(n, T)`` ordinates.
    t, c, k : ndarray, ndarray, int
        Knots, coefficients and degree for a spline kind; empty arrays and
        ``k = 0`` for an index kind, which reads the segment directly.
    """
    xx, yy = _sorted_xy_nd(xr, _as_values(flat), assume_sorted)
    _check_increasing(xx)
    if _is_spline(kd):
        k = kdeg
        t, c = _mis_any_nd(xx, yy, k)
    else:
        k = 0
        t = np.zeros(0, np.float64)
        c = np.zeros((0, yy.shape[1]), yy.dtype)
    return xx, yy, t, c, k


_INTERP1D_DISPATCH_SRC = '''
def _interp1d_nd(xr, y, kd, kdeg, be, extrapolate, fb, fa, assume_sorted,
                 axis):
    """Construct the `interp1d` class that matches the rank of `y`.

    `y.ndim` is part of `y`'s numba type, so every branch but one is removed
    while compiling and the survivor fixes the rank; `_i1d_cons_nd` reads the
    dtype off the same type and fixes the class. A rank past the cap leaves
    only the ``raise``.

    Generated by ``_make_interp1d_dispatch()``; see
    `scijit.interpolate._ndaxis`.
    """
{branches}
    raise ValueError("y must have rank {phrase}")
'''


def _make_interp1d_dispatch():
    """Build `_interp1d_nd`, one branch per supported rank of `y`."""
    def body(rank):
        return ("        flat, rest, ax = _nda._flatten%d(y, axis)\n"
                "        xx, yy, t, c, k = _i1d_prep(xr, flat, kd, kdeg,\n"
                "                                    assume_sorted)\n"
                "        return _i1d_cons_nd(y, xx, yy, kd, t, c, k, be,\n"
                "                            extrapolate, fb, fa, rest, ax)"
                % rank)
    src = _INTERP1D_DISPATCH_SRC.format(
        branches=_dispatch_branches("y", body), phrase=_rank_phrase())
    return njit(_define(src, globals(), "_interp1d_nd"))


_interp1d_nd = _make_interp1d_dispatch()


# ---------------------------------------------------------------------------
# @njit factory
# ---------------------------------------------------------------------------

_KINDS = (0, 1, 2, 3, 4, 5, 6)


@njit
def interp1d(x, y, kind="linear", axis=-1, copy=True,
             bounds_error=None, fill_value=np.nan, assume_sorted=False,
             extrapolate=False):
    """Build an `_Interp1D` for one-dimensional interpolation.

    Parameters
    ----------
    x : 1-D array_like of float
        Sample abscissae. Sorted ascending unless `assume_sorted` says they
        already are. After sorting they must be strictly increasing; a
        repeated value raises ``ValueError``. Minimum length is 2 for kinds
        0-3 and 6, and ``k+1`` (3 for quadratic, 4 for cubic) for 4 and 5.
    y : array_like of float or complex, rank 1, 2 or 3
        Sample ordinates. Its length along `axis` must match `x`. Every
        position on the other axes is an independent series over the same
        `x`. Rank 4 and above raise ``ValueError``. A complex `y` of any
        width is held as complex128 and the interpolated result carries it;
        anything else is float64. The spline kinds fit the real and the
        imaginary parts separately on one knot vector.
    kind : str or int, optional
        Interpolation kind. Either one of these strings:

            'linear'      'slinear' is accepted for this too
            'nearest'     ties round DOWN
            'previous'    'zero' is accepted for this too
            'next'
            'quadratic'   k=2 interpolating spline
            'cubic'       k=3 interpolating spline
            'nearest-up'  ties round UP

        or an integer giving the spline ORDER, so ``kind=3`` is cubic. See
        Notes for the integer's full range. Default ``'linear'``. An
        unrecognised string raises ``NotImplementedError``. The value may be a
        runtime string, not only a literal.
    axis : int, optional
        Which axis of `y` the interpolation runs along, default -1. Negative
        values count from the end. Ignored for a 1-D `y`. ``ev(xs)`` returns
        the shape of `y` with this axis replaced by ``len(xs)``; a scalar
        query returns it with this axis removed.
    copy : bool, optional
        True, the default, copies `x` and `y`. False keeps a reference where
        the array is already contiguous float64, so a later mutation of it
        shows through; anything else is converted, which copies. With the
        default ``assume_sorted=False`` the sort copies regardless.
    bounds_error : bool or None, optional
        If True, evaluating outside ``[x[0], x[-1]]`` raises ``ValueError``.
        Default None, which means True unless the call extrapolates. Passing
        True while extrapolating raises ``ValueError``.
    fill_value : float, (below, above), array_like, or 'extrapolate', optional
        What out-of-bounds points get when `bounds_error` is False and the
        call does not extrapolate. A 2-tuple, and only a 2-tuple, is
        ``(below, above)``. Anything else, a scalar, a list or an array, is
        broadcast up to ``y.shape[:axis] + y.shape[axis+1:]`` and gives one
        value PER SERIES; a shape that does not broadcast raises
        ``ValueError``. Each element of a 2-tuple broadcasts on its own, so
        ``([1.0, 2.0], [3.0, 4.0])`` sets a different value per series at each
        end. The string ``'extrapolate'`` extrapolates instead of filling; it
        may be a runtime string, not only a literal. Default NaN.
    assume_sorted : bool, optional
        False (the default) sorts `x` ascending and reorders `y` with it. True
        takes `x` as already ascending and skips the sort. Either way the
        result must be strictly increasing. A ``bool`` `axis` raises
        ``TypeError`` on the sorting path, and is read as 0 or 1 with
        ``assume_sorted=True``.
    extrapolate : bool, optional
        If True, extrapolate out-of-bounds points instead of filling them.
        The same thing ``fill_value='extrapolate'`` does. Default False.
        Linear extrapolates along the end segment, the splines continue the
        end polynomial, nearest/nearest-up clamp to the end node, 'previous'
        returns NaN below the range and 'next' returns NaN above it.

        THE PRECEDENCE across the three parameters:
        extrapolation is on if EITHER `extrapolate` is True or `fill_value`
        is ``'extrapolate'``, so the string wins over the default
        ``extrapolate=False``; ``bounds_error=None`` then resolves to
        "raise unless extrapolating"; an explicit ``bounds_error=True``
        alongside extrapolation from either spelling raises
        ``Cannot extrapolate and raise at the same time.``; and a numeric
        `fill_value` given alongside extrapolation is never read.

    Returns
    -------
    f : _Interp1D
        A callable jitclass instance. ``f(xs)`` runs `_Interp1D.ev` and
        ``f(x)`` runs `_Interp1D.ev_one`; ``f[xs]`` reaches `ev` too.

    Raises
    ------
    NotImplementedError
        On a `kind` that is neither one of the strings nor an integer, which
        includes a float and ``None``, and the message names the value.
    ValueError
        On a `fill_value` whose shape does not broadcast up to `y`'s trailing
        shape, an `x` that is not strictly increasing once sorted, a length
        mismatch, too few points for the chosen kind, or ``bounds_error=True``
        while extrapolating.
    IndexError
        On an `axis` outside ``[-y.ndim, y.ndim)``.
    TypeError
        On a ``bool`` `axis` with ``assume_sorted=False``.

    See Also
    --------
    scipy.interpolate.interp1d : The scipy routine this mirrors.

    Notes
    -----
    This factory is an ``@njit`` function, so it runs from Python and from
    inside ``@njit``, and its keyword defaults apply in both. The ``_Interp1D``
    constructor underneath takes all ten arguments explicitly, because a
    jitclass constructor's defaults are Python-only; the factory is what
    supplies them, and it is where the `kind` string resolves.

    **An integer `kind` is the spline ORDER.** So ``0`` is ``'zero'``, ``1``
    ``'slinear'``, ``2`` ``'quadratic'`` and ``3`` ``'cubic'``; orders above 3
    have no string spelling, and a negative order raises
    ``Expect non-negative k.``

    ``scipy.interpolate.interp1d`` is marked legacy in scipy's own
    documentation, which points at `make_interp_spline` and the other
    constructors for new code.

    Deviations from scipy. `extrapolate` is an additional parameter with no
    scipy counterpart, so interp1d exposes three out-of-bounds parameters
    where scipy exposes two; a scipy-shaped call never passes it. A duplicated
    abscissa raises here, where scipy sorts, does not check, and evaluates a
    zero-width interval to inf or nan. `y` is capped at rank 3, where scipy
    takes any rank. `fill_value` is a property returning the resolved
    ``(below, above)`` pair, where scipy's returns the object last assigned,
    and `set_fill_value` is the method that takes everything scipy's setter
    takes; a property setter on a jitclass is typed by its getter, so one
    property cannot accept both a number and a string.

    A complex `fill_value` on a real `y` raises ``ValueError``, and a complex
    `x` raises ``TypingError``. scipy truncates both to float64 and returns a
    real result,
    warning only for the `fill_value`: measured, ``fill_value=1+2j`` on a real
    `y` returns float64 with a ``ComplexWarning``, and a complex `x` returns
    complex128 whose imaginary part is 0.0 everywhere.

    `set_fill_value('extrapolate')` extrapolates for every kind. scipy's own
    setter does not: it fixes the linear kernel at construction, so on a 1-D
    float64 `y` at ``kind='linear'`` the same object clamps to the end nodes
    when the string arrives afterwards and extrapolates when it arrives in
    the constructor. Measured at ``x = linspace(0, 1, 5)``,
    ``y = sin(3x)``, queried at -1 and 2: built with the string
    ``[-2.7266, -2.4067]``, set afterwards ``[0.0, 0.1411]``. The same split
    appears at ``kind='previous'`` and ``'next'``, where the setter route
    loses the NaN. This reproduces the constructor's answer in every cell.

    The spline kinds are fitted with this package's ``make_interp_spline``
    at ``bc_type='not-a-knot'``. FITPACK's ``1 <= k <= 5`` fitting cap does
    not apply, because no FITPACK fitter is involved.

    Accuracy vs scipy on 16 nodes over 50 in-range points, calling scipy with
    the same string: 'linear', 'nearest', 'nearest-up', 'previous', 'next',
    'quadratic', 'cubic' and 'zero' all exactly 0.0, and 'slinear'
    1.11e-16. An integer `kind`, which is the spline ORDER, over orders 0 to
    8: worst 4.44e-16. The two aliases: 'zero' exactly 0.0 against code 2, and
    'slinear' 2.22e-16 against code 0, that last one because scipy routes
    'slinear' through a degree-1 spline where this reads the segment
    directly. Extrapolating two units past each end: linear and nearest 0.0,
    quadratic 1.42e-14, cubic 8.53e-14.

    prange-safe: yes for evaluation of prebuilt instances.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import interp1d
    >>> x = np.linspace(0.0, 1.0, 9)
    >>> y = np.sin(3.0 * x)
    >>> f = interp1d(x, y, 'cubic')     # build once
    >>> np.round(f(np.array([0.25, 0.75])), 6)
    array([0.681639, 0.778073])

    Inside compiled code, the mean of the interpolant over its domain:

    >>> @njit
    ... def mean(f):
    ...     xs = np.linspace(0.0, 1.0, 201)
    ...     return np.mean(f(xs))
    >>> float(np.round(mean(f), 6))
    0.660382
    """
    _check_axis(axis)
    _nda._check_axis_index(axis, y.ndim)
    ax = axis % y.ndim
    xr = _copy_or_view(x, copy)
    if xr.ndim != 1:
        raise ValueError("x must be one-dimensional")
    n = xr.shape[0]
    if y.shape[ax] != n:
        raise ValueError("x and y arrays must be equal in length along "
                         "interpolation axis.")
    _check_axis_take(axis, assume_sorted)
    kd, kdeg = _kind_code(kind)
    fbv, fav, ex_fill = _fill_resolve(fill_value, _trailing_shape(y, ax),
                                      y)
    # THE PRECEDENCE, across all three of the parameters this factory carries
    # where scipy carries two. Row interp1d-I3.
    #   1. extrapolation is on if EITHER `extrapolate=True` or
    #      `fill_value='extrapolate'` asks for it, so the string wins over an
    #      `extrapolate=False` left at its default;
    #   2. `bounds_error=None`, the default, then means "raise unless
    #      extrapolating", which is scipy's rule for its own two;
    #   3. an explicit `bounds_error=True` with extrapolation from either
    #      spelling raises, as scipy does for its one spelling;
    #   4. a numeric `fill_value` passed alongside extrapolation is resolved
    #      and never read, again as in scipy.
    extrap = extrapolate or ex_fill
    be = _bounds_flag(bounds_error, extrap)
    if be and extrap:
        raise ValueError("Cannot extrapolate and raise at the same time.")
    if _is_spline(kd):
        if n < kdeg + 1:
            raise ValueError("x and y need at least k+1 entries for a "
                             "spline of order k")
    elif n < 2:
        raise ValueError("x and y must have at least 2 entries")

    if y.ndim == 1:
        yr = _copy_or_view_y(y, copy)
        xx, yy = _sorted_xy(xr, yr, assume_sorted)
        _check_increasing(xx)
        if _is_spline(kd):
            k = kdeg
            t, c = _mis_any(xx, yy, k)
            t = np.ascontiguousarray(t.astype(np.float64))
            c = np.ascontiguousarray(_as_values(c))
        else:
            k = 0
            t = np.zeros(0, np.float64)
            c = np.zeros(0, yy.dtype)
        # a 1-D `y` has trailing shape (1,), so the pair is one value a side
        return _i1d_cons(xx, yy, kd, t, c, k, be, extrap, fbv[0], fav[0])
    else:
        return _interp1d_nd(xr, y, kd, kdeg, be, extrap, fbv, fav,
                            assume_sorted, ax)


# ---------------------------------------------------------------------------
# index-only variant: the same interpolator without the FITPACK branch
# ---------------------------------------------------------------------------
#
# Kinds 0/1/2/3/6 need no FITPACK at all, so this class holds no `t`/`c`/`k`
# and never mentions `splev`.  Everything else -- bounds, fill, extrapolate,
# tie-breaking -- is identical to `_Interp1D`, and the two agree exactly.
