"""numba equivalent of scipy.interpolate.RegularGridInterpolator / interpn.

Pure-@njit N-dimensional interpolation on a regular (rectilinear) grid,
constructible from Python and evaluable INSIDE @njit code:

    method        algorithm
    ----------    ------------------------------------------------------
    "linear"      N-linear (multilinear over 2**ndim corners)
    "nearest"     nearest-neighbour (half-open rounding)
    "pchip"       shape-preserving cubic along each axis in turn, last
                  axis first, four nodes per axis
    "splinef2d"   bicubic RectBivariateSpline over a 2-D grid; `interpn`
                  only, as in scipy

scipy's "slinear", "cubic" and "quintic" raise ``ValueError``. Those three
build an N-D B-spline through a sparse iterative solve, which this package
does not carry. For a smooth 2-D grid use RectBivariateSpline
(scijit.interpolate), which is what "splinef2d" reaches.

Accuracy of this module against scipy 1.18.0, over 200 random in-range
points per case: 'nearest' is exactly 0.0 for ndim 1 to 4, since it selects
a stored value rather than computing one. 'linear' is 0.0 in 1-D, 4.441e-16
for ndim 2 and 2.220e-16 for ndim 3 and 4. The corner sum is accumulated in
a different order from scipy's, so it is machine-eps close and not
bit-identical. Extrapolation, ``fill_value=None``, measures 4.547e-13; a
descending axis 4.441e-16; a constant ``fill_value`` exactly 0.0.
'pchip' measures 2.220e-16 over 400 random 2-D points, 163 of them
bit-identical, and 8.882e-16 over 200 random 3-D points; its segment is
evaluated by Horner where scipy accumulates the powers of the offset.


**Encoding (why a jitclass needs it)**

A numba jitclass member must be a single typed array -- it cannot hold a
Python tuple/list of variable-length axis arrays. So the ragged grid is
packed into flat arrays:

Nine are passed to the constructor, in this order:

  flat_grid  float64[:]  all axis coordinates concatenated, ascending
  offsets    int64[:]    length ndim+1; axis j lives in
                         flat_grid[offsets[j] : offsets[j+1]]
  shape      int64[:]    length ndim; number of nodes per axis
  values_flat dt[:]      the grid values, C-order (row-major) ravel
  ndim       int64
  method     int64       0=linear, 1=nearest, 2=pchip. INTERNAL: the public
                         `method` is scipy's string, resolved by the factory
  bounds_error int64     1 -> raise on out-of-bounds query
  fill_value dt          value returned for OOB when bounds_error==0
  extrapolate int64      1 -> ignore fill_value, extrapolate linearly (this
                         is scipy's fill_value=None behaviour)

A tenth field is DERIVED rather than passed:

  strides    int64[:]    length ndim; C-order strides into values_flat,
                         computed in the constructor from `shape`

**dtype and vector-valued data.** `dt` above is float64 for a real `values`
of any width and complex128 for a complex one, and the interpolated result
carries it. `values` may also carry up to 2 axes beyond the grid axes, one
block of numbers per node, and the result is then
``xi.shape[:-1] + values.shape[ndim:]``. A jitclass field has one dtype and
one rank, so each combination is its own class, generated from one source
template; a vector-valued class carries two more fields, `rest` and `nvals`.
The class is chosen while the call compiles.


**Public API**

  RegularGridInterpolator(points, values, method="linear", bounds_error=True,
                          fill_value=np.nan, extrapolate=False)
      `points` = TUPLE of 1-D ascending or descending axis arrays, all of
      the same dtype and layout (descending axes are flipped like scipy).
      `method` is scipy's string, "linear", "nearest" or "pchip", and
      "pchip" needs four nodes on every axis and a real `values`.
      `fill_value=None`
      extrapolates off the edge cell, and a sequence `fill_value` is one
      value per component. Returns a jitclass usable from Python or from
      inside @njit.

  interpn(points, values, xi, method="linear", bounds_error=True,
          fill_value=np.nan, extrapolate=False)
      Functional wrapper (builds an interpolator and evaluates xi). `xi` is
      an (..., ndim) array of any rank, or a tuple of coordinate arrays that
      broadcast together; the result has shape xi.shape[:-1] followed by the
      trailing shape of `values`. `method` also takes "splinef2d", a bicubic
      spline over a real 2-D grid, which returns float64 and does not
      extrapolate; inside @njit that name is written at the call site.

  _RGI  the scalar-valued float64 jitclass. Its constructor takes the nine
        passed fields above, `strides` excepted, so it can also be
        constructed directly INSIDE @njit from pre-packed arrays; the
        vector-valued classes take `rest` and `nvals` after them, and
        `_RGI_CLASSES[(dtkey, trail)]` holds all six. Evaluation methods,
        written for the scalar case, where `rest` below is empty:
          .ev(xi)          xi shape (m, ndim) float64 -> (m,) + rest
          .ev_1d(xi)       xi rank 1, read as reshape(-1, ndim)
          .ev_point(p)     p shape (ndim,) float64   -> a scalar, or rest
          .ev_nd(xi)       xi shape (..., ndim)      -> xi.shape[:-1] + rest
        each with an `_m` twin taking a per-call `method`:
          .ev_m(xi, method)    .ev_1d_m(xi, method)
          .ev_point_m(p, method)  .ev_nd_m(xi, method)

Examples
--------
>>> import numpy as np
>>> from numba import njit
>>> from scijit.interpolate import RegularGridInterpolator
>>> x = np.linspace(0.0, 1.0, 5)
>>> y = np.linspace(0.0, 2.0, 7)
>>> vals = np.outer(x, y) ** 2
>>> rgi = RegularGridInterpolator((x, y), vals, 'linear')     # build once
>>> pts = np.array([[0.3, 0.9], [0.8, 1.1]])
>>> np.round(rgi(pts), 6)             # (m,)
array([0.083333, 0.801667])

Inside compiled code, the mean over a set of query points:

>>> @njit
... def query_mean(rgi, pts):
...     return np.mean(rgi(pts))
>>> float(np.round(query_mean(rgi, pts), 6))
0.4425

prange-safe: no module/global state, evaluation is read-only. Build the
interpolator outside the parallel loop and evaluate it inside.

CONSTRUCTOR DEFAULTS ARE PYTHON-ONLY. The interpolator constructors declare
no defaults, so every argument must always be given, but the trap it
belongs to is package-wide: a jitclass constructor's Python defaults are
invisible inside ``@njit``, where an omitted argument raises "TypeError:
invalid number of args". The ``RegularGridInterpolator`` factory below is a
``@njit``: it validates and packs the ragged grid inside compiled code, so
the interpolator can be built and evaluated without leaving ``@njit``.
scipy's ``rgi(xi)`` works as written. The rank of the array selects the
method: a stack of points, shape (m, ndim), runs ``.ev``; a rank-1 array runs
``.ev_1d``, which reads it as ``reshape(-1, ndim)`` exactly as scipy does;
rank 3 and above run ``.ev_nd``. A second argument is scipy's per-call
``method``, and reaches the ``_m`` twin. ``.ev_point(p)`` returns a scalar for
a single point and is reached by name. All eight stay reachable by name.
"""
import numpy as np
from numba import complex128, float64, int64, njit, types
from numba.core.errors import TypingError
from numba.extending import overload
from numba.np.unsafe.ndarray import to_fixed_tuple
from scijitclass import scijitclass, sig, Array, ArrayOf, String

from ._cubic import _build_c, _eval_seg, _find_interval, _pchip_derivs
from ._ndaxis import _ND_MAXTRAIL, _define
from .interpolate import RectBivariateSpline


# ---------------------------------------------------------------------------
# the `method` flag: scipy's string
# ---------------------------------------------------------------------------

_METHOD_NAMES = ("linear", "nearest", "pchip")

_METHOD_MSG = ("method must be 'linear', 'nearest' or 'pchip'; scipy's "
               "'slinear', 'cubic' and 'quintic' are not supported. `interpn` "
               "also takes 'splinef2d', written at the call site rather than "
               "held in a variable")


def _method_code(method):
    """scipy's `method` string, resolved to the internal int code."""
    if not isinstance(method, str):
        raise ValueError(_METHOD_MSG)
    for i in range(3):
        if method == _METHOD_NAMES[i]:
            return i
    raise ValueError(_METHOD_MSG)


@overload(_method_code)
def _method_code_ovl(method):
    """`_method_code` inside ``@njit``.

    The arm resolves at run time, so `method` may be a runtime string rather
    than a literal. Anything that is not a string is refused while compiling,
    which is where the reason can still be reported.
    """
    if not isinstance(method, (types.UnicodeType, types.StringLiteral, str)):
        raise TypingError(_METHOD_MSG)

    def impl(method):
        for i in range(3):
            if method == _METHOD_NAMES[i]:
                return i
        raise ValueError(_METHOD_MSG)
    return impl

_METHOD_CODE = {"linear": 0, "nearest": 1, "pchip": 2}


def _method_code_or(method, default):
    """`method` resolved to a code, with ``None`` keeping `default`.

    scipy's ``rgi(xi, method=None)`` evaluates with the method the object was
    built with.
    """
    return default if method is None else _method_code(method)


@overload(_method_code_or)
def _method_code_or_ovl(method, default):
    """`_method_code_or` inside ``@njit``. ``None`` is a type, so the branch
    is taken while compiling."""
    if method is None or isinstance(method, types.NoneType):
        def impl(method, default):
            return default
        return impl

    def impl(method, default):
        return _method_code(method)
    return impl


def _method_arg(t):
    """Dispatch predicate: the second argument of ``rgi(xi, method)``.

    A string, or ``None`` for the method the object was built with.
    """
    return String(t) or isinstance(t, types.NoneType)


def _stack_arg(t):
    """Dispatch predicate: an `xi` of rank 3 or more.

    scipy documents ``xi`` as ``(..., ndim)`` and returns ``xi.shape[:-1]``,
    so every rank above 2 reaches one method.
    """
    return Array(t) and t.ndim >= 3


# ---------------------------------------------------------------------------
# 'pchip': the tensor-product fold, one axis at a time
# ---------------------------------------------------------------------------
#
# The grid is collapsed one axis at a time, last axis first, running a 1-D
# shape-preserving cubic along each axis and folding it away. A cubic reads
# four nodes: the two its segment spans, plus one either side, which is what
# the Fritsch-Carlson slopes at those two nodes are built from. So the fold
# gathers a four-node window per axis rather than the whole axis, and the
# arithmetic it runs on that window is the arithmetic the whole axis would
# have run.

#: Nodes per axis the fold reads.
_PCHIP_WIN = 4

#: Tail of the message for an axis too short for a cubic. The double space is
#: scipy's, from the two fragments its f-string joins.
_PCHIP_DIMS_TAIL = (", but method pchip requires at least  4 points per "
                    "dimension.")

#: Raised when `values` is complex and the interpolator is BUILT with 'pchip'.
_PCHIP_CX_MSG = (
    "`PchipInterpolator` only works with real values. If you are trying to "
    "use the real components of the passed array, use `np.real` on the array "
    "before passing to `RegularGridInterpolator`.")

#: Raised when a per-call ``method='pchip'`` reaches a complex interpolator.
#: scipy words the two differently: this one comes from `PchipInterpolator`
#: itself, which it only reaches on the second route.
_PCHIP_CX_CALL_MSG = (
    "`PchipInterpolator` only works with real values for `y`. If you are "
    "trying to use the real components of the passed array, use `np.real` on "
    "the array before passing to `PchipInterpolator`.")


@njit
def _pchip_dims_raise(n, j):
    """Report an axis with fewer than four nodes. Internal."""
    raise ValueError("There are " + str(n) + " points in dimension " + str(j)
                     + _PCHIP_DIMS_TAIL)


def _pchip_no_complex(values):
    """Refuse a complex `values`, as scipy's ``__init__`` does for 'pchip'."""
    if np.iscomplexobj(values):
        raise ValueError(_PCHIP_CX_MSG)


@overload(_pchip_no_complex)
def _pchip_no_complex_ovl(values):
    """`_pchip_no_complex` inside ``@njit``. The dtype is a type, so the
    branch is taken while compiling and the real case costs nothing."""
    if isinstance(values, types.Array) and isinstance(values.dtype,
                                                      types.Complex):
        def impl(values):
            raise ValueError(_PCHIP_CX_MSG)
        return impl

    def impl(values):
        pass
    return impl


@njit
def _pchip_at(x, y, xq):
    """The 1-D shape-preserving cubic through ``(x, y)``, read at `xq`.

    `PchipInterpolator`'s arithmetic, on the four-node window the fold hands
    it. Off the ends it continues the edge segment, which is what scipy's
    ``extrapolate=True`` default does.
    """
    c = _build_c(x, y, _pchip_derivs(x, y))
    i = _find_interval(x, xq)
    return _eval_seg(c, i, xq - x[i], 0)


@njit
def _pchip_point(flat_grid, offsets, shape, strides, ndim, values_flat, nvals,
                 point, out):
    """Interpolate at one point with 'pchip', writing `nvals` components.

    Parameters
    ----------
    flat_grid, offsets, shape, strides, ndim : the interpolator's packing
    values_flat : 1-D float64 ndarray
        The grid values, C-order, `nvals` adjacent numbers per node.
    nvals : int
        Components per node, 1 for a scalar-valued grid.
    point : 1-D float64 ndarray, length ndim
    out : 1-D float64 ndarray, length nvals
        Written in place.
    """
    # the four-node window on each axis, clamped so it stays on the axis
    lo = np.empty(ndim, np.int64)
    for j in range(ndim):
        off = offsets[j]
        n = shape[j]
        left = _find_interval(flat_grid[off:off + n], point[j]) - 1
        if left < 0:
            left = 0
        if left > n - _PCHIP_WIN:
            left = n - _PCHIP_WIN
        lo[j] = left

    # gather the window block, C-order over the window indices
    nblk = 1
    for j in range(ndim):
        nblk *= _PCHIP_WIN
    buf = np.empty(nblk * nvals, np.float64)
    for b in range(nblk):
        flat = 0
        rem = b
        for j in range(ndim - 1, -1, -1):
            flat += (lo[j] + rem % _PCHIP_WIN) * strides[j]
            rem //= _PCHIP_WIN
        base = flat * nvals
        for q in range(nvals):
            buf[b * nvals + q] = values_flat[base + q]

    # fold one axis at a time, last axis first
    size = nblk * nvals
    xw = np.empty(_PCHIP_WIN, np.float64)
    col = np.empty(_PCHIP_WIN, np.float64)
    for j in range(ndim - 1, -1, -1):
        off = offsets[j] + lo[j]
        for k in range(_PCHIP_WIN):
            xw[k] = flat_grid[off + k]
        outer = size // (_PCHIP_WIN * nvals)
        new = np.empty(outer * nvals, np.float64)
        for o in range(outer):
            base = o * _PCHIP_WIN * nvals
            for q in range(nvals):
                for k in range(_PCHIP_WIN):
                    col[k] = buf[base + k * nvals + q]
                new[o * nvals + q] = _pchip_at(xw, col, point[j])
        buf = new
        size = outer * nvals

    for q in range(nvals):
        out[q] = buf[q]

def _rgi_spec_for(dt, trail):
    """The jitclass spec for a `values` of this dtype and trailing rank.

    Parameters
    ----------
    dt : numba type
        ``float64`` or ``complex128``, the dtype the grid values are held in.
    trail : int
        Number of axes `values` carries beyond the grid axes. 0 is a scalar at
        each node, which is what scipy's plain `values` gives.

    Returns
    -------
    list of (str, numba type)
    """
    spec = [
        ('flat_grid', float64[:]),
        ('offsets', int64[:]),
        ('shape', int64[:]),
        ('strides', int64[:]),
        ('values_flat', dt[:]),
        ('ndim', int64),
        ('method', int64),
        ('bounds_error', int64),
        ('fill_value', dt if trail == 0 else dt[:]),
        ('extrapolate', int64),
    ]
    if trail > 0:
        spec = spec + [('rest', int64[:]), ('nvals', int64)]
    return spec


_rgi_spec = _rgi_spec_for(float64, 0)


def _rgi_dispatch():
    """The ``scijitclass`` dispatch table every interpolator class carries.

    Every method takes an array, so the rank selects between them: `ev` reads
    a stack of points, shape (m, ndim), `ev_1d` reads scipy's rank-1 `xi`, and
    `ev_nd` reads scipy's (..., ndim). The default ev_one / ev convention
    cannot separate those. A second argument is scipy's per-call `method`.

    THE RANK-1 ARM IS `ev_1d`, NOT `ev_point` (row P6). scipy's
    `_ndim_coords_from_arrays` reshapes a rank-1 `xi` to `(-1, ndim)`, so
    `rgi(p)` for one point returns an array of shape (1,) rather than a
    scalar, and on a 1-D grid a rank-1 `xi` is a run of m points rather than
    one malformed point. `ev_point` keeps its single-point contract under its
    own name.
    """
    return [('ev', sig(ArrayOf(ndim=2))),
            ('ev_m', sig(ArrayOf(ndim=2), _method_arg)),
            ('ev_1d', sig(ArrayOf(ndim=1))),
            ('ev_1d_m', sig(ArrayOf(ndim=1), _method_arg)),
            ('ev_nd', sig(_stack_arg)),
            ('ev_nd_m', sig(_stack_arg, _method_arg))]


_RGI_HEAD_SRC = '''
class {cname}:
    """Regular-grid interpolator jitclass over {dword} {vword}, the
    counterpart of ``scipy.interpolate.RegularGridInterpolator``.

    Normally built by the `RegularGridInterpolator` factory below, which packs
    a ragged tuple of axis arrays into the flat encoding this constructor
    wants. It can also be constructed directly inside ``@njit`` from
    pre-packed arrays.

    Parameters
    ----------
    flat_grid : 1-D float64 ndarray
        All axis coordinate arrays concatenated, each ascending. Axis j
        occupies ``flat_grid[offsets[j]:offsets[j+1]]``.
    offsets : 1-D int64 ndarray, length ndim+1
        Start index of each axis in `flat_grid`, with ``offsets[0] == 0`` and
        ``offsets[ndim] == flat_grid.size``.
    shape : 1-D int64 ndarray, length ndim
        Number of nodes per axis; must agree with `offsets`.
    values_flat : 1-D {dword} ndarray, length {vlen}
        Grid values in C order (row-major ravel){vtail}
    ndim : int
        Number of grid dimensions, >= 1.
    method : int
        0 = 'linear' (multilinear over the 2**ndim cell corners),
        1 = 'nearest' (half-open rounding, ties at the cell midpoint going
        DOWN, which is what scipy's ``yi <= .5`` does), 2 = 'pchip' (a
        shape-preserving cubic along each axis in turn). No default here; the
        `RegularGridInterpolator` factory accepts scipy's string and resolves
        it to this code, defaulting to ``'linear'`` as scipy does.
    bounds_error : int
        1 = raise ``ValueError`` when a query point lies outside the grid,
        0 = fall back to `fill_value` or extrapolation. The factory defaults
        to 1, matching scipy's ``bounds_error=True``.
    fill_value : {filltype}
        {filldoc}
    extrapolate : int
        1 = ignore `fill_value` and extrapolate from the clamped edge cell,
        which is scipy's ``fill_value=None``; 0 = use `fill_value`.
{cparams}
    Attributes
    ----------
    strides : 1-D int64 ndarray, length ndim
        C-order strides into `values_flat`, in units of one node, computed in
        the constructor.
    (plus every constructor argument, stored under the same name)

    Methods
    -------
    ev(xi)
        Batch evaluation, ``xi`` of shape (m, ndim) -> {evshape}.
    ev_1d(xi)
        A rank-1 ``xi``, read as ``xi.reshape(-1, ndim)`` -> {evshape}.
    ev_point(point)
        Single point, ``point`` of shape (ndim,) -> {pointshape}.
    ev_nd(xi)
        ``xi`` of shape (..., ndim), rank 3 or more -> {ndshape}.
    ev_m(xi, method), ev_1d_m(xi, method), ev_point_m(point, method),
    ev_nd_m(xi, method)
        The same four with a per-call `method`, ``'linear'``, ``'nearest'``
        or ``'pchip'``; ``None`` keeps the method the object was built with.

    Notes
    -----
    'linear', 'nearest' and 'pchip' are implemented. scipy's 'slinear',
    'cubic' and 'quintic' raise ``ValueError``.
    ``rgi(xi)`` runs ``.ev`` for a (m, ndim) array, ``.ev_1d`` for a rank-1
    one and ``.ev_nd`` above that, and a second argument reaches the ``_m``
    twin. ``.ev_point`` takes one point and is reached by name only. A NaN
    query coordinate is out of bounds: it raises under ``bounds_error == 1``
    and returns NaN otherwise, whatever `fill_value` holds, which is what
    scipy does.

    A per-call ``'pchip'`` needs four nodes on every axis and raises
    ``ValueError`` otherwise, after the bounds check. On a complex `values`
    it raises whatever the axis lengths are.

    Generated by ``_make_rgi({dtkey!r}, {trail})`` in
    ``scijit/interpolate/_rgi.py`` and bound there under this name. One class
    exists per dtype of `values` and per number of trailing axes it carries,
    because a jitclass field has one dtype and one rank; they share this
    source and differ in the dtype they hold and the rank they hand back.

    CONSTRUCTOR DEFAULTS ARE PYTHON-ONLY (package-wide jitclass trap): this
    constructor declares none, so all {nargs} arguments are always required,
    in Python and inside ``@njit`` alike.

    Accuracy: see the module docstring.

    prange-safe: yes (read-only evaluation, no state).
    """

    def __init__(self, flat_grid, offsets, shape, values_flat, ndim,
                 method, bounds_error, fill_value, extrapolate{cargs}):
        self.flat_grid = flat_grid
        self.offsets = offsets
        self.shape = shape
        self.values_flat = values_flat
        self.ndim = ndim
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.extrapolate = extrapolate
{cinit}
        # C-order strides into values_flat, in units of one node
        strides = np.empty(ndim, np.int64)
        strides[ndim - 1] = 1
        for j in range(ndim - 2, -1, -1):
            strides[j] = strides[j + 1] * shape[j + 1]
        self.strides = strides

    def _locate(self, j, x):
        """Locate the cell of axis `j` containing `x`. Internal.

        Reproduces scipy's ``searchsorted(side='left') - 1`` followed by a
        clamp into ``[0, n-2]``, so an out-of-range coordinate lands on the
        edge cell and extrapolates along it.

        Parameters
        ----------
        j : int
            Axis index, 0 <= j < ndim.
        x : float
            Coordinate along that axis.

        Returns
        -------
        i : int
            Cell index in ``[0, n-2]``.
        y : float
            Normalised position in the cell, ``(x - x_i) / (x_i+1 - x_i)``;
            0.0 if the cell has zero width. Outside the grid it goes below 0
            or above 1, which is what makes extrapolation linear. A one-node
            axis has no cell, so it gives ``(0, 0.0)`` and contributes the
            stored node.
        """
        off = self.offsets[j]
        n = self.offsets[j + 1] - off
        if n == 1:
            return 0, 0.0
        # binary search, side='left': first idx with flat_grid[off+idx] >= x
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.flat_grid[off + mid] < x:
                lo = mid + 1
            else:
                hi = mid
        i = lo - 1
        if i < 0:
            i = 0
        if i > n - 2:
            i = n - 2
        denom = self.flat_grid[off + i + 1] - self.flat_grid[off + i]
        if denom != 0.0:
            y = (x - self.flat_grid[off + i]) / denom
        else:
            y = 0.0
        return i, y

    def _oob_dim(self, point):
        """The first axis on which `point` lies outside the grid. Internal.

        Parameters
        ----------
        point : 1-D float64 ndarray, length ndim

        Returns
        -------
        int
            The lowest axis index whose coordinate is NaN, below the first
            node or above the last, or -1 when every coordinate is in bounds.
            The bounds are inclusive, so a point exactly on the boundary is in
            bounds, as in scipy. The index is what the out-of-bounds message
            names. scipy tests ``grid[0] <= p`` and ``p <= grid[-1]``, both
            False for a NaN, so a NaN coordinate is out of bounds there too.
        """
        for j in range(self.ndim):
            off = self.offsets[j]
            lo = self.flat_grid[off]
            hi = self.flat_grid[self.offsets[j + 1] - 1]
            if np.isnan(point[j]) or point[j] < lo or point[j] > hi:
                return j
        return -1
'''

_RGI_SCALAR_SRC = '''
    def _eval_code(self, point, mcode):
        """Interpolate at one point with an explicit method code. Internal.

        The body `ev_point` and `ev_point_m` share: `mcode` is 0 for linear,
        1 for nearest and 2 for pchip, so a per-call `method` and the stored
        one reach the same arithmetic.

        Parameters
        ----------
        point : 1-D float64 ndarray, length ndim
        mcode : int
            0 = linear, 1 = nearest, 2 = pchip.

        Returns
        -------
        {dword}
        """
        if point.shape[0] != self.ndim:
            raise ValueError("point has wrong dimension for this "
                             "RegularGridInterpolator")
        d = self._oob_dim(point)
        if d >= 0 and self.bounds_error == 1:
            raise ValueError("One of the requested xi is out of bounds "
                             "in dimension " + str(d))
        # a NaN coordinate returns NaN whatever `fill_value` holds, which is
        # scipy's answer: it writes the fill first and the NaN over it
        for j in range(self.ndim):
            if np.isnan(point[j]):
                return {nan}
        if d >= 0 and self.extrapolate == 0:
            return self.fill_value
        # out of bounds while extrapolating: fall through, the clamped cell
        # gives the linear extrapolation
{pchip}
        idx = np.empty(self.ndim, np.int64)
        y = np.empty(self.ndim, np.float64)
        for j in range(self.ndim):
            ij, yj = self._locate(j, point[j])
            idx[j] = ij
            y[j] = yj

        if mcode == 1:
            # nearest neighbour
            flat = 0
            for j in range(self.ndim):
                nj = idx[j] if y[j] <= 0.5 else idx[j] + 1
                flat += nj * self.strides[j]
            return self.values_flat[flat]

        # N-linear: sum over the 2**ndim hypercube corners
        total = {zero}
        ncorner = 1 << self.ndim
        for c in range(ncorner):
            w = 1.0
            flat = 0
            for j in range(self.ndim):
                bit = (c >> j) & 1
                if bit == 1:
                    w *= y[j]
                    flat += (idx[j] + 1) * self.strides[j]
                else:
                    w *= (1.0 - y[j])
                    flat += idx[j] * self.strides[j]
            if w != 0.0:
                total += self.values_flat[flat] * w
        return total

    def _ev_code(self, xi, mcode):
        """Interpolate a stack of points with an explicit method code.

        The body `ev` and `ev_m` share.

        Parameters
        ----------
        xi : 2-D float64 ndarray, shape (m, ndim)
        mcode : int
            0 = linear, 1 = nearest, 2 = pchip.

        Returns
        -------
        out : 1-D {dword} ndarray, length m
        """
        xic = np.ascontiguousarray(np.asarray(xi))
        if xic.ndim != 2 or xic.shape[1] != self.ndim:
            raise ValueError("xi must have shape (m, ndim) matching this "
                             "RegularGridInterpolator")
        m = xic.shape[0]
        out = np.empty(m, {dt})
        for p in range(m):
            out[p] = self._eval_code(xic[p], mcode)
        return out

    def ev_point(self, point):
        """Interpolate at a single point (scipy's ``rgi(point[None, :])[0]``).

        Parameters
        ----------
        point : 1-D float64 ndarray, length ndim
            Query coordinates, one per grid axis. A wrong length raises
            ``ValueError``.

        Returns
        -------
        {dword}
            The interpolated value; NaN if any coordinate is NaN;
            `fill_value` if the point is out of bounds with
            ``bounds_error == 0`` and ``extrapolate == 0``; the extrapolated
            edge-cell value if ``extrapolate == 1``.

        Raises
        ------
        ValueError
            If ``point.shape[0] != ndim``, or if the point is out of bounds
            and ``bounds_error == 1``.
        """
        return self._eval_code(point, self.method)

    def ev_point_m(self, point, method):
        """Interpolate at a single point with a per-call `method`.

        Parameters
        ----------
        point : 1-D float64 ndarray, length ndim
        method : str or None
            ``'linear'`` or ``'nearest'``. ``None`` keeps the method the
            interpolator was built with.

        Returns
        -------
        {dword}

        Raises
        ------
        ValueError
            If `method` is neither of the two names, if
            ``point.shape[0] != ndim``, or if the point is out of bounds and
            ``bounds_error == 1``.
        """
        return self._eval_code(point, _method_code_or(method, self.method))

    def ev(self, xi):
        """Interpolate at a batch of points (scipy's ``rgi(xi)``).

        Parameters
        ----------
        xi : 2-D float64 ndarray, shape (m, ndim)
            One query point per row. Any other shape raises ``ValueError``.
            The array is passed through ``ascontiguousarray`` first, so a
            strided view (e.g. ``pts[:, ::2]``) is handled correctly rather
            than silently reading the base buffer.

        Returns
        -------
        out : 1-D {dword} ndarray, length m
            Interpolated values, one per row of `xi`, each computed exactly as
            in `ev_point`.

        Raises
        ------
        ValueError
            On a wrong `xi` shape, or on an out-of-bounds row when
            ``bounds_error == 1``.
        """
        return self._ev_code(xi, self.method)

    def ev_m(self, xi, method):
        """Interpolate a batch of points with a per-call `method`.

        Parameters
        ----------
        xi : 2-D float64 ndarray, shape (m, ndim)
        method : str or None
            ``'linear'`` or ``'nearest'``. ``None`` keeps the method the
            interpolator was built with.

        Returns
        -------
        out : 1-D {dword} ndarray, length m

        Raises
        ------
        ValueError
            If `method` is neither of the two names, on a wrong `xi` shape, or
            on an out-of-bounds row when ``bounds_error == 1``.
        """
        return self._ev_code(xi, _method_code_or(method, self.method))

    def ev_1d(self, xi):
        """Interpolate over a rank-1 ``xi``, scipy's ``reshape(-1, ndim)``.

        scipy's ``_ndim_coords_from_arrays`` gives a rank-1 array the shape
        ``(-1, ndim)``, so one coordinate per axis is a single point and comes
        back as a length-1 array, and on a 1-D grid a run of m coordinates is
        m separate points.

        Parameters
        ----------
        xi : 1-D float64 ndarray
            Query coordinates, ``m * ndim`` of them. A length that is not a
            multiple of `ndim` raises.

        Returns
        -------
        out : 1-D {dword} ndarray, length ``xi.size // ndim``

        Raises
        ------
        ValueError
            On a length that does not divide, or on an out-of-bounds point
            when ``bounds_error == 1``.
        """
        return self._ev_code(
            np.ascontiguousarray(np.asarray(xi)).reshape(-1, self.ndim),
            self.method)

    def ev_1d_m(self, xi, method):
        """`ev_1d` with a per-call `method`.

        Parameters
        ----------
        xi : 1-D float64 ndarray
        method : str or None
            ``'linear'`` or ``'nearest'``. ``None`` keeps the method the
            interpolator was built with.

        Returns
        -------
        out : 1-D {dword} ndarray
        """
        return self._ev_code(
            np.ascontiguousarray(np.asarray(xi)).reshape(-1, self.ndim),
            _method_code_or(method, self.method))

    def ev_nd(self, xi):
        """Interpolate over scipy's ``xi`` of shape ``(..., ndim)``.

        Parameters
        ----------
        xi : float64 ndarray of rank 3 or more, shape ``(..., ndim)``
            The leading axes are flattened, the points evaluated, and the
            result folded back.

        Returns
        -------
        out : {dword} ndarray of shape ``xi.shape[:-1]``

        Raises
        ------
        ValueError
            If ``xi.shape[-1] != ndim``, or on an out-of-bounds point when
            ``bounds_error == 1``.
        """
        q = np.ascontiguousarray(np.asarray(xi)).reshape(-1, xi.shape[-1])
        return self._ev_code(q, self.method).reshape(xi.shape[:-1])

    def ev_nd_m(self, xi, method):
        """`ev_nd` with a per-call `method`.

        Parameters
        ----------
        xi : float64 ndarray of rank 3 or more, shape ``(..., ndim)``
        method : str or None
            ``'linear'`` or ``'nearest'``. ``None`` keeps the method the
            interpolator was built with.

        Returns
        -------
        out : {dword} ndarray of shape ``xi.shape[:-1]``

        Raises
        ------
        ValueError
            If `method` is neither of the two names, if
            ``xi.shape[-1] != ndim``, or on an out-of-bounds point when
            ``bounds_error == 1``.
        """
        q = np.ascontiguousarray(np.asarray(xi)).reshape(-1, xi.shape[-1])
        return self._ev_code(
            q, _method_code_or(method, self.method)).reshape(xi.shape[:-1])
'''

_RGI_VEC_SRC = '''
    def _eval_code(self, point, mcode, out):
        """Interpolate at one point, writing one value per component.

        The body every evaluation method shares: `mcode` is 0 for linear, 1
        for nearest and 2 for pchip, so a per-call `method` and the stored one
        reach the same arithmetic. The result is written into `out` rather
        than returned, which is what lets a batch fill one row at a time.

        Parameters
        ----------
        point : 1-D float64 ndarray, length ndim
        mcode : int
            0 = linear, 1 = nearest, 2 = pchip.
        out : 1-D {dword} ndarray, length nvals
            Written in place.
        """
        if point.shape[0] != self.ndim:
            raise ValueError("point has wrong dimension for this "
                             "RegularGridInterpolator")
        nv = self.nvals
        d = self._oob_dim(point)
        if d >= 0 and self.bounds_error == 1:
            raise ValueError("One of the requested xi is out of bounds "
                             "in dimension " + str(d))
        # a NaN coordinate returns NaN whatever `fill_value` holds, which is
        # scipy's answer: it writes the fill first and the NaN over it
        for j in range(self.ndim):
            if np.isnan(point[j]):
                for q in range(nv):
                    out[q] = {nan}
                return
        if d >= 0 and self.extrapolate == 0:
            for q in range(nv):
                out[q] = self.fill_value[q]
            return
        # out of bounds while extrapolating: fall through, the clamped cell
        # gives the linear extrapolation
{pchip}
        idx = np.empty(self.ndim, np.int64)
        y = np.empty(self.ndim, np.float64)
        for j in range(self.ndim):
            ij, yj = self._locate(j, point[j])
            idx[j] = ij
            y[j] = yj

        if mcode == 1:
            # nearest neighbour
            flat = 0
            for j in range(self.ndim):
                nj = idx[j] if y[j] <= 0.5 else idx[j] + 1
                flat += nj * self.strides[j]
            for q in range(nv):
                out[q] = self.values_flat[flat * nv + q]
            return

        # N-linear: sum over the 2**ndim hypercube corners
        for q in range(nv):
            out[q] = {zero}
        ncorner = 1 << self.ndim
        for c in range(ncorner):
            w = 1.0
            flat = 0
            for j in range(self.ndim):
                bit = (c >> j) & 1
                if bit == 1:
                    w *= y[j]
                    flat += (idx[j] + 1) * self.strides[j]
                else:
                    w *= (1.0 - y[j])
                    flat += idx[j] * self.strides[j]
            if w != 0.0:
                base = flat * nv
                for q in range(nv):
                    out[q] += self.values_flat[base + q] * w

    def _ev_code(self, xi, mcode):
        """Interpolate a stack of points with an explicit method code.

        Parameters
        ----------
        xi : 2-D float64 ndarray, shape (m, ndim)
        mcode : int
            0 = linear, 1 = nearest, 2 = pchip.

        Returns
        -------
        out : 2-D {dword} ndarray, shape (m, nvals)
            One row per query point, flattened over the trailing axes.
        """
        xic = np.ascontiguousarray(np.asarray(xi))
        if xic.ndim != 2 or xic.shape[1] != self.ndim:
            raise ValueError("xi must have shape (m, ndim) matching this "
                             "RegularGridInterpolator")
        m = xic.shape[0]
        out = np.empty((m, self.nvals), {dt})
        for p in range(m):
            self._eval_code(xic[p], mcode, out[p])
        return out

    def ev_point(self, point):
        """Interpolate at a single point (scipy's ``rgi(point[None, :])[0]``).

        Parameters
        ----------
        point : 1-D float64 ndarray, length ndim
            Query coordinates, one per grid axis. A wrong length raises
            ``ValueError``.

        Returns
        -------
        out : {trail}-D {dword} ndarray of shape ``values.shape[ndim:]``
            NaN in every component if any coordinate is NaN; `fill_value` if
            the point is out of bounds with ``bounds_error == 0`` and
            ``extrapolate == 0``; the extrapolated edge-cell value if
            ``extrapolate == 1``.

        Raises
        ------
        ValueError
            If ``point.shape[0] != ndim``, or if the point is out of bounds
            and ``bounds_error == 1``.
        """
        out = np.empty(self.nvals, {dt})
        self._eval_code(point, self.method, out)
        return np.ascontiguousarray(out).reshape({restdims})

    def ev_point_m(self, point, method):
        """Interpolate at a single point with a per-call `method`.

        Parameters
        ----------
        point : 1-D float64 ndarray, length ndim
        method : str or None
            ``'linear'`` or ``'nearest'``. ``None`` keeps the method the
            interpolator was built with.

        Returns
        -------
        out : {trail}-D {dword} ndarray

        Raises
        ------
        ValueError
            If `method` is neither of the two names, if
            ``point.shape[0] != ndim``, or if the point is out of bounds and
            ``bounds_error == 1``.
        """
        out = np.empty(self.nvals, {dt})
        self._eval_code(point, _method_code_or(method, self.method), out)
        return np.ascontiguousarray(out).reshape({restdims})

    def ev(self, xi):
        """Interpolate at a batch of points (scipy's ``rgi(xi)``).

        Parameters
        ----------
        xi : 2-D float64 ndarray, shape (m, ndim)
            One query point per row. Any other shape raises ``ValueError``.
            The array is passed through ``ascontiguousarray`` first, so a
            strided view (e.g. ``pts[:, ::2]``) is handled correctly rather
            than silently reading the base buffer.

        Returns
        -------
        out : {evrank}-D {dword} ndarray of shape ``(m,) + values.shape[ndim:]``

        Raises
        ------
        ValueError
            On a wrong `xi` shape, or on an out-of-bounds row when
            ``bounds_error == 1``.
        """
        r = self._ev_code(xi, self.method)
        return np.ascontiguousarray(r).reshape(r.shape[0], {restdims})

    def ev_m(self, xi, method):
        """Interpolate a batch of points with a per-call `method`.

        Parameters
        ----------
        xi : 2-D float64 ndarray, shape (m, ndim)
        method : str or None
            ``'linear'`` or ``'nearest'``. ``None`` keeps the method the
            interpolator was built with.

        Returns
        -------
        out : {evrank}-D {dword} ndarray

        Raises
        ------
        ValueError
            If `method` is neither of the two names, on a wrong `xi` shape, or
            on an out-of-bounds row when ``bounds_error == 1``.
        """
        r = self._ev_code(xi, _method_code_or(method, self.method))
        return np.ascontiguousarray(r).reshape(r.shape[0], {restdims})

    def ev_1d(self, xi):
        """Interpolate over a rank-1 ``xi``, scipy's ``reshape(-1, ndim)``.

        scipy's ``_ndim_coords_from_arrays`` gives a rank-1 array the shape
        ``(-1, ndim)``, so one coordinate per axis is a single point and comes
        back with a leading length of 1, and on a 1-D grid a run of m
        coordinates is m separate points.

        Parameters
        ----------
        xi : 1-D float64 ndarray
            Query coordinates, ``m * ndim`` of them. A length that is not a
            multiple of `ndim` raises.

        Returns
        -------
        out : {evrank}-D {dword} ndarray of shape
            ``(xi.size // ndim,) + values.shape[ndim:]``

        Raises
        ------
        ValueError
            On a length that does not divide, or on an out-of-bounds point
            when ``bounds_error == 1``.
        """
        r = self._ev_code(
            np.ascontiguousarray(np.asarray(xi)).reshape(-1, self.ndim),
            self.method)
        return np.ascontiguousarray(r).reshape(r.shape[0], {restdims})

    def ev_1d_m(self, xi, method):
        """`ev_1d` with a per-call `method`.

        Parameters
        ----------
        xi : 1-D float64 ndarray
        method : str or None
            ``'linear'`` or ``'nearest'``. ``None`` keeps the method the
            interpolator was built with.

        Returns
        -------
        out : {evrank}-D {dword} ndarray
        """
        r = self._ev_code(
            np.ascontiguousarray(np.asarray(xi)).reshape(-1, self.ndim),
            _method_code_or(method, self.method))
        return np.ascontiguousarray(r).reshape(r.shape[0], {restdims})

    def ev_nd(self, xi):
        """Interpolate over scipy's ``xi`` of shape ``(..., ndim)``.

        Parameters
        ----------
        xi : float64 ndarray of rank 3 or more, shape ``(..., ndim)``
            The leading axes are flattened, the points evaluated, and the
            result folded back.

        Returns
        -------
        out : {dword} ndarray of shape
            ``xi.shape[:-1] + values.shape[ndim:]``

        Raises
        ------
        ValueError
            If ``xi.shape[-1] != ndim``, or on an out-of-bounds point when
            ``bounds_error == 1``.
        """
        q = np.ascontiguousarray(np.asarray(xi)).reshape(-1, xi.shape[-1])
        r = self._ev_code(q, self.method)
        return np.ascontiguousarray(r).reshape(xi.shape[:-1] + ({restdims},))

    def ev_nd_m(self, xi, method):
        """`ev_nd` with a per-call `method`.

        Parameters
        ----------
        xi : float64 ndarray of rank 3 or more, shape ``(..., ndim)``
        method : str or None
            ``'linear'`` or ``'nearest'``. ``None`` keeps the method the
            interpolator was built with.

        Returns
        -------
        out : {dword} ndarray of shape
            ``xi.shape[:-1] + values.shape[ndim:]``

        Raises
        ------
        ValueError
            If `method` is neither of the two names, if
            ``xi.shape[-1] != ndim``, or on an out-of-bounds point when
            ``bounds_error == 1``.
        """
        q = np.ascontiguousarray(np.asarray(xi)).reshape(-1, xi.shape[-1])
        r = self._ev_code(q, _method_code_or(method, self.method))
        return np.ascontiguousarray(r).reshape(xi.shape[:-1] + ({restdims},))
'''

#: Class name, dtype and words per dtype key. 'f' holds float64 values, 'c'
#: complex128. scipy returns float64 for real input of any width and
#: complex128 for complex input of any width, both measured.
_RGI_DT = {'f': (float64, 'np.float64', 'float64', '0.0', 'np.nan'),
           'c': (complex128, 'np.complex128', 'complex128',
                 'np.complex128(0.0)', 'np.complex128(np.nan)')}

#: The 'pchip' arm of `_eval_code`, one spelling per dtype and trailing rank.
#: A grid too short for a cubic is refused here rather than when the
#: interpolator is built, because `mcode` can also arrive from a per-call
#: `method`; scipy revalidates on the same route. A complex `values` has no
#: pchip arm at all, so the branch carries scipy's refusal instead.
_PCHIP_ARM = {
    ('f', 0): '''
        if mcode == 2:
            for j in range(self.ndim):
                if self.shape[j] < 4:
                    _pchip_dims_raise(self.shape[j], j)
            one = np.empty(1, np.float64)
            _pchip_point(self.flat_grid, self.offsets, self.shape,
                         self.strides, self.ndim, self.values_flat, 1,
                         point, one)
            return one[0]
''',
    ('f', 1): '''
        if mcode == 2:
            for j in range(self.ndim):
                if self.shape[j] < 4:
                    _pchip_dims_raise(self.shape[j], j)
            _pchip_point(self.flat_grid, self.offsets, self.shape,
                         self.strides, self.ndim, self.values_flat, nv,
                         point, out)
            return
''',
    ('c', 0): '''
        if mcode == 2:
            raise ValueError(_PCHIP_CX_CALL_MSG)
''',
    ('c', 1): '''
        if mcode == 2:
            raise ValueError(_PCHIP_CX_CALL_MSG)
''',
}


def _make_rgi(dtkey, trail):
    """Build the interpolator class for one dtype and trailing rank.

    Parameters
    ----------
    dtkey : {'f', 'c'}
        ``'f'`` for float64 values, ``'c'`` for complex128.
    trail : int
        Number of axes `values` carries beyond the grid axes, 0 to
        `_ND_MAXTRAIL`. 0 is scipy's plain scalar-valued grid.

    Returns
    -------
    cls : jitclass
        Decorated with `scijitclass` against ``_rgi_spec_for(dt, trail)`` and
        bound in this module under the name it was generated with.
    """
    dt, dtsrc, dword, zero, nan = _RGI_DT[dtkey]
    cname = '_RGI' + ('C' if dtkey == 'c' else '')
    if trail > 0:
        cname += 'V%d' % trail
    restdims = ", ".join("self.rest[%d]" % i for i in range(trail))
    head = _RGI_HEAD_SRC.format(
        cname=cname, dtkey=dtkey, trail=trail, dword=dword,
        vword='a scalar at each node' if trail == 0
        else 'a %d-D block at each node' % trail,
        vlen='prod(shape)' if trail == 0 else 'prod(shape) * nvals',
        vtail='.' if trail == 0 else
        ', so the nvals components of one node are adjacent.',
        filltype=dword if trail == 0 else '1-D %s ndarray' % dword,
        filldoc=('Value returned for out-of-bounds points when\n'
                 '        ``bounds_error == 0`` and ``extrapolate == 0``. The'
                 ' factory defaults\n        to NaN, as scipy does.')
        if trail == 0 else
        ('One value per component, returned for out-of-bounds points\n'
         '        when ``bounds_error == 0`` and ``extrapolate == 0``. The '
         'factory\n        broadcasts scipy\'s scalar or per-component '
         '`fill_value` into it and\n        defaults to NaN, as scipy does.'),
        cparams='' if trail == 0 else
        ('    rest : 1-D int64 ndarray, length %d\n'
         '        The trailing shape of `values`, ``values.shape[ndim:]``.\n'
         '    nvals : int\n'
         '        Number of components at each node, ``prod(rest)``.\n'
         % trail),
        evshape='(m,)' if trail == 0 else '(m,) + rest',
        pointshape='a scalar' if trail == 0 else 'an array of shape rest',
        ndshape='``xi.shape[:-1]``' if trail == 0
        else '``xi.shape[:-1] + rest``',
        nargs='nine' if trail == 0 else 'eleven',
        cargs='' if trail == 0 else ', rest, nvals',
        cinit='' if trail == 0 else
        '        self.rest = rest\n        self.nvals = nvals')
    body = (_RGI_SCALAR_SRC if trail == 0 else _RGI_VEC_SRC).format(
        dt=dtsrc, dword=dword, zero=zero, nan=nan, trail=trail,
        evrank=trail + 1, restdims=restdims,
        pchip=_PCHIP_ARM[(dtkey, 0 if trail == 0 else 1)].strip('\n'))
    return scijitclass(_rgi_spec_for(dt, trail),
                       dispatch=_rgi_dispatch())(
        _define(head + body, globals(), cname))


#: Every interpolator class, keyed by (dtype key, trailing rank). The build
#: chooses one while compiling: `values.dtype` and, for a tuple `points`, the
#: difference between ``values.ndim`` and ``len(points)`` are both known then.
_RGI_CLASSES = {}

for _dtkey in ('f', 'c'):
    for _trail in range(_ND_MAXTRAIL + 1):
        _RGI_CLASSES[(_dtkey, _trail)] = _make_rgi(_dtkey, _trail)
del _dtkey, _trail

#: The float64 scalar-valued class, which is what a scipy-shaped call with a
#: real `values` builds. Named for the callers and tests that reach it.
_RGI = _RGI_CLASSES[('f', 0)]


@njit
def _flip_axis_flat(vals_flat, shape, j, nvals):
    """Reverse axis ``j`` of a C-order flat buffer, out of place.

    ``np.flip(values, axis=j)`` needs a compile-time axis, so the flip is done
    as an index computation instead: walk the flat buffer, decompose each
    linear index into its per-axis subscripts, replace ``i_j`` with
    ``n_j - 1 - i_j``, and recompose. This is what lets the whole factory be
    ``@njit`` rather than Python.

    `nvals` is how many numbers each grid node carries, 1 for a scalar-valued
    grid and ``prod(values.shape[ndim:])`` for a vector-valued one. The
    components of one node are adjacent, so they move together.
    """
    nd = shape.shape[0]
    strides = np.empty(nd, np.int64)
    acc = 1
    for d in range(nd - 1, -1, -1):
        strides[d] = acc
        acc *= shape[d]
    out = np.empty_like(vals_flat)
    for lin in range(vals_flat.shape[0] // nvals):
        src = 0
        for d in range(nd):
            idx = (lin // strides[d]) % shape[d]
            if d == j:
                idx = shape[d] - 1 - idx
            src += idx * strides[d]
        for q in range(nvals):
            out[lin * nvals + q] = vals_flat[src * nvals + q]
    return out


#: scipy's `_check_fill_value` message, raised for a `fill_value` whose dtype
#: cannot be cast to the dtype the grid values are held in.
_FILL_MSG = ("fill_value must be either 'None' or of a type compatible "
             "with values")

#: Raised for a `fill_value` whose SHAPE does not broadcast to one value per
#: component. scipy reaches the same case inside a numpy assignment at
#: evaluation time; see `RegularGridInterpolator`'s Notes.
_FILL_SHAPE_MSG = ("fill_value must be a scalar, or broadcast up to the "
                   "trailing shape of values")


@njit
def _raise_fill():
    """Raise scipy's `_check_fill_value` refusal.

    Called rather than raised inline: a bare ``raise`` in an ``@overload`` arm
    types the arm as returning ``none``, because the ``return`` after it is
    unreachable and inference drops it.
    """
    raise ValueError(_FILL_MSG)


def _nan_fill(values):
    """NaN in the dtype `values` are stored in.

    A ``fill_value=None`` extrapolates, so the stored fill is never read; it
    still has to carry the dtype of the field it lands in.
    """
    return np.complex128(np.nan) if np.iscomplexobj(values) else np.nan


@overload(_nan_fill)
def _nan_fill_ovl(values):
    """`_nan_fill` inside ``@njit``. The dtype is the whole decision."""
    if isinstance(values.dtype, types.Complex):
        def impl(values):
            return np.complex128(np.nan)
        return impl

    def impl(values):
        return np.nan
    return impl


def _fill_of(fill_value, values):
    """`fill_value` cast to the dtype `values` are stored in.

    Grid values are held as complex128 when `values` is complex and as float64
    otherwise, which is what scipy returns: measured, a complex64 `values`
    comes back complex128 and a float32 or integer one comes back float64. The
    cast a `fill_value` has to survive is therefore against the STORED dtype,
    which is why an integer `values` accepts a float fill and refuses a
    complex one, as scipy does.

    Parameters
    ----------
    fill_value : scalar or array_like
        A scalar fills every component; a sequence is one value per component,
        broadcast up to ``values.shape[ndim:]`` at build time.
    values : ndarray

    Returns
    -------
    scalar or ndarray
        In the stored dtype, keeping the rank `fill_value` arrived with.

    Raises
    ------
    ValueError
        If `fill_value` cannot be cast to the stored dtype.
    """
    dt = np.complex128 if np.iscomplexobj(values) else np.float64
    if not np.can_cast(np.asarray(fill_value).dtype, dt, casting='same_kind'):
        raise ValueError(_FILL_MSG)
    if np.ndim(fill_value) == 0:
        return dt(fill_value)
    return np.ascontiguousarray(np.asarray(fill_value, dtype=dt))


@overload(_fill_of)
def _fill_of_ovl(fill_value, values):
    """`_fill_of` inside ``@njit``, one body per spelling of `fill_value`.

    The cast is a TYPE question and is answered while compiling. A complex
    `fill_value` on a real `values` still raises at run time rather than while
    compiling, so both entry points refuse it with scipy's class and text.
    """
    ft = fill_value.value if isinstance(fill_value, types.Omitted) \
        else fill_value
    cx = isinstance(values.dtype, types.Complex)
    scalar = isinstance(ft, (types.Float, types.Integer, types.Boolean,
                             types.Complex, float, int, bool, complex))
    # an array, a list and a homogeneous tuple all carry `dtype`, so one line
    # reaches the element type of every container spelling
    fd = getattr(ft, 'dtype', ft)
    fill_cx = isinstance(fd, (types.Complex, complex))
    if fill_cx and not cx:
        def impl(fill_value, values):
            _raise_fill()
            return np.float64(0.0)
        return impl
    if scalar:
        if cx:
            def impl(fill_value, values):
                return np.complex128(fill_value)
            return impl

        def impl(fill_value, values):
            return np.float64(fill_value)
        return impl
    if isinstance(ft, (types.Array, types.List, types.ListType,
                       types.BaseTuple)):
        if cx:
            def impl(fill_value, values):
                return np.ascontiguousarray(
                    np.asarray(fill_value)).astype(np.complex128)
            return impl

        def impl(fill_value, values):
            return np.ascontiguousarray(
                np.asarray(fill_value)).astype(np.float64)
        return impl
    raise TypingError(_FILL_MSG)


def _fill_pair(fill_value, extrapolate, values):
    """`fill_value` and `extrapolate` as the pair `_rgi_build` takes.

    ``None`` means extrapolate, so it resolves to the extrapolation flag and a
    `fill_value` that is never read.

    Parameters
    ----------
    fill_value : scalar, array_like or None
    extrapolate : bool
    values : ndarray
        The grid values, whose dtype the fill is cast to.

    Returns
    -------
    fv : scalar or ndarray
        The fill in the stored dtype.
    ex : int
        1 to extrapolate off the edge cell, 0 to return `fv`.
    """
    if fill_value is None:
        return _nan_fill(values), 1
    return _fill_of(fill_value, values), (1 if extrapolate else 0)


def _fill_pair_ty(fill_value):
    """`_fill_pair`'s ``None`` decision, over the numba TYPE of `fill_value`.

    An omitted argument reaches an ``@overload`` as ``types.Omitted`` carrying
    the raw Python value, so the default has to be unwrapped before the branch.

    Parameters
    ----------
    fill_value : numba type

    Returns
    -------
    bool
        True when `fill_value` is ``None``, which means extrapolate.
    """
    ft = fill_value.value if isinstance(fill_value, types.Omitted) \
        else fill_value
    return ft is None or isinstance(ft, types.NoneType)


@njit
def _fill_bcast(a, rest, nvals):
    """One fill value per component, broadcast up to the trailing shape.

    numpy's assignment rule, right-aligned: an axis of length 1 repeats, and
    anything else has to equal the target. scipy assigns into a
    ``(rows, nvals)`` selection, one row per out-of-bounds point, so the fill
    may also carry ONE leading axis of length 1 for that row axis. Measured on
    scipy: a shape ``(3,)``, a shape ``(1, 3)``, a list and a scalar all fill a
    trailing shape of ``(3,)``, while ``(2,)`` and ``(2, 1)`` are refused.

    Parameters
    ----------
    a : ndarray
        The fill, in the stored dtype.
    rest : 1-D int64 ndarray
        The trailing shape of `values`.
    nvals : int
        ``prod(rest)``.

    Returns
    -------
    out : 1-D ndarray, length `nvals`
        In the same C order the stored values use.

    Raises
    ------
    ValueError
        If the shapes do not broadcast.
    """
    r = rest.shape[0]
    # drop the row axis, if the fill carries one and it is a single row
    off = 1 if (a.ndim == r + 1 and a.shape[0] == 1) else 0
    f = a.ndim - off
    ok = f <= r
    if ok:
        for i in range(f):
            s = a.shape[a.ndim - 1 - i]
            if s != 1 and s != rest[r - 1 - i]:
                ok = False
    if not ok:
        raise ValueError(_FILL_SHAPE_MSG)
    src = np.ascontiguousarray(a).ravel()
    out = np.empty(nvals, src.dtype)
    # C-order strides of the target, and of the source mapped onto it: a
    # source axis of length 1 gets stride 0 and repeats
    tstr = np.empty(r, np.int64)
    sstr = np.zeros(r, np.int64)
    acc = 1
    for d in range(r - 1, -1, -1):
        tstr[d] = acc
        acc *= rest[d]
    acc = 1
    for i in range(f):
        d = r - 1 - i
        s = a.shape[a.ndim - 1 - i]
        sstr[d] = 0 if s == 1 else acc
        acc *= s
    for lin in range(nvals):
        j = 0
        for d in range(r):
            j += ((lin // tstr[d]) % rest[d]) * sstr[d]
        out[lin] = src[j]
    return out


def _fill_scalar(fv):
    """The one fill value a scalar-valued grid stores.

    A length-1 sequence is one value, which is what numpy's assignment makes
    of it and what scipy therefore accepts.
    """
    if np.ndim(fv) == 0:
        return fv
    a = np.ascontiguousarray(np.asarray(fv)).ravel()
    if a.size != 1:
        raise ValueError(_FILL_SHAPE_MSG)
    return a[0]


@overload(_fill_scalar)
def _fill_scalar_ovl(fv):
    """`_fill_scalar` inside ``@njit``, one body per rank."""
    if isinstance(fv, types.Array):
        def impl(fv):
            a = np.ascontiguousarray(fv).ravel()
            if a.size != 1:
                raise ValueError(_FILL_SHAPE_MSG)
            return a[0]
        return impl

    def impl(fv):
        return fv
    return impl


def _fill_vec(fv, rest, nvals):
    """One fill value per component, from a scalar or a sequence."""
    if np.ndim(fv) == 0:
        return np.full(nvals, fv)
    return _fill_bcast(np.ascontiguousarray(np.asarray(fv)), rest, nvals)


@overload(_fill_vec)
def _fill_vec_ovl(fv, rest, nvals):
    """`_fill_vec` inside ``@njit``, one body per rank."""
    if isinstance(fv, types.Array):
        def impl(fv, rest, nvals):
            return _fill_bcast(fv, rest, nvals)
        return impl

    def impl(fv, rest, nvals):
        return np.full(nvals, fv)
    return impl


def RegularGridInterpolator(points, values, method="linear",
                            bounds_error=True, fill_value=np.nan,
                            extrapolate=False):
    """Build an `_RGI` for interpolation on a regular grid.

    Parameters
    ----------
    points : tuple or list of 1-D float array_like
        Axis coordinates, one array per dimension, each strictly ascending OR
        strictly descending (a descending axis is flipped along with the
        corresponding axis of `values`). Anything non-monotone raises
        ``ValueError``. Arbitrary ndim; tested to 4-D.
    values : array_like, shape ``tuple(len(p) for p in points) + rest``
        Grid data. A real `values` of any width is held as float64 and a
        complex one as complex128, which is the dtype the interpolated result
        carries. `rest` is the shape of the block at each node, empty for one
        number per node and up to 2 axes beyond that; ``rgi(xi)`` then returns
        ``xi.shape[:-1] + rest``. A shape that disagrees with the grid raises
        ``ValueError``.
    method : str, optional
        ``'linear'`` for multilinear interpolation, ``'nearest'`` for
        nearest-neighbour, ties at the cell midpoint rounding down, or
        ``'pchip'`` for a shape-preserving cubic along each axis in turn. The
        value may be a runtime string. Anything that is not one of those three
        raises ``ValueError``. Default ``'linear'``. ``'pchip'`` needs at least
        four nodes on every axis and a real `values`, and raises
        ``ValueError`` otherwise.
    bounds_error : bool, optional
        If True, a query point outside the grid raises ``ValueError`` at
        evaluation time. Default True.
    fill_value : scalar, array_like or None, optional
        Value returned for out-of-bounds queries when ``bounds_error`` is
        False. ``None`` extrapolates linearly off the edge cell instead. A
        scalar fills every component; a sequence broadcasts up to `rest` and
        gives one value per component. A value that cannot be cast to the
        stored dtype raises ``ValueError``, so a complex `fill_value` on a
        real `values` raises. Default NaN.
    extrapolate : bool, optional
        Continue the interpolant off the edge cell for an out-of-bounds query,
        ignoring `fill_value`. The same thing ``fill_value=None`` does, and
        the two spellings reach the same code. Default False.

    Returns
    -------
    rgi : jitclass
        A jitclass instance to pass into ``@njit`` code, where it is evaluated
        with ``rgi(xi)`` or ``rgi(xi, method)``, or by name with ``.ev(xi)``,
        ``.ev_point(p)``, ``.ev_nd(xi)`` and their ``_m`` twins. Which class
        it is follows the dtype of `values` and how many trailing axes it
        carries, both fixed while compiling.

    See Also
    --------
    scipy.interpolate.RegularGridInterpolator : The scipy routine this mirrors.

    Notes
    -----
    Validation, the descending-axis flip and the ragged-to-flat packing all
    run in compiled code, so the interpolator can be built inside ``@njit``.
    The jitclass constructor underneath always takes all of its arguments.

    Only ``'linear'``, ``'nearest'`` and ``'pchip'`` are implemented. scipy's
    ``'slinear'``, ``'cubic'`` and ``'quintic'`` build tensor-product B-splines
    solved with a sparse Krylov solver this package does not ship, and raise
    here; ``'pchip'`` is the shape-preserving cubic alternative.

    `extrapolate` has no scipy counterpart. It is `fill_value=None` under a
    boolean spelling; a scipy-shaped call never passes it.

    `nu`, `solver` and `solver_args`, scipy's three keyword-only arguments for
    the spline methods, do not exist here. A callable jitclass dispatches on
    the types of its POSITIONAL arguments, so a keyword-only argument has no
    spelling on ``rgi(xi, ...)``.

    A `fill_value` whose shape does not broadcast up to `rest` raises when the
    interpolator is built. scipy raises inside a numpy assignment at
    evaluation instead, with a message naming the number of out-of-bounds
    rows, and it raises on every call, including one whose points are all in
    bounds.

    A `values` with trailing axes needs `points` as a TUPLE when the call is
    compiled: a list has no length until it runs, and the rank of the result
    is fixed while compiling. A list passed from Python is converted here, so
    the constraint applies to a list built inside ``@njit``.

    Accuracy: see the module docstring.

    prange-safe: yes. 64 interpolators built inside one ``prange`` and
    evaluated reproduce the serial build exactly, 0.000e+00.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import RegularGridInterpolator
    >>> x = np.linspace(0.0, 1.0, 5)
    >>> y = np.linspace(0.0, 1.0, 6)
    >>> vals = np.exp(x[:, None] + y[None, :])
    >>> rgi = RegularGridInterpolator((x, y), vals)     # build once
    >>> float(rgi(np.array([[0.25, 0.4]]))[0])
    1.9155408290138962

    Inside compiled code, the largest value over a set of query points:

    >>> @njit
    ... def query_max(rgi):
    ...     pts = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
    ...     return np.max(rgi(pts))
    >>> float(np.round(query_max(rgi), 6))
    5.514377
    """
    if isinstance(points, list):
        points = tuple(points)
    fv, ex = _fill_pair(fill_value, extrapolate, values)
    return _rgi_build(points, values, method, bounds_error, fv, ex)


@overload(RegularGridInterpolator, prefer_literal=True)
def _rgi_ovl(points, values, method="linear", bounds_error=True,
             fill_value=np.nan, extrapolate=False):
    """`RegularGridInterpolator` inside ``@njit``.

    ``fill_value=None`` is a TYPE, not a value, so the branch is taken while
    compiling and the compiled body carries a fill of the stored dtype.
    Everything else stays in `_rgi_build`, so this entry and the Python one
    are one implementation.
    """
    if _fill_pair_ty(fill_value):
        def impl(points, values, method="linear", bounds_error=True,
                 fill_value=np.nan, extrapolate=False):
            return _rgi_build(points, values, method, bounds_error,
                              _nan_fill(values), 1)
        return impl

    def impl(points, values, method="linear", bounds_error=True,
             fill_value=np.nan, extrapolate=False):
        return _rgi_build(points, values, method, bounds_error,
                          _fill_of(fill_value, values),
                          1 if extrapolate else 0)
    return impl


#: Raised while compiling for more trailing axes than there are classes.
_TRAIL_MSG = ("values may carry at most %d axes beyond the grid axes"
              % _ND_MAXTRAIL)

#: Raised for a vector-valued `values` whose `points` cannot be measured while
#: compiling. See `RegularGridInterpolator`'s Notes.
_TRAIL_LIST_MSG = ("values with trailing dimensions needs points as a tuple, "
                   "whose length is known while compiling")


def _values_trail(points, values):
    """How many axes `values` carries beyond the grid axes, from the TYPES.

    ``values.ndim`` is part of an array's numba type and the length of a TUPLE
    is part of a tuple's, so the difference is known while compiling, which is
    what fixes the rank of the result. A LIST has no length until it runs, so
    a list `points` is read as a scalar-valued grid and `_rgi_build`'s runtime
    check refuses anything else.

    Parameters
    ----------
    points, values : numba types

    Returns
    -------
    int
        0 to `_ND_MAXTRAIL`. A negative difference is clamped to 0, leaving
        `_rgi_build` to raise scipy's "There are N point arrays" message.

    Raises
    ------
    numba.core.errors.TypingError
        On more trailing axes than the cap.
    """
    if isinstance(points, types.BaseTuple):
        nd = len(points)
    else:
        nd = values.ndim
    trail = values.ndim - nd
    if trail < 0:
        return 0
    if trail > _ND_MAXTRAIL:
        raise TypingError(_TRAIL_MSG)
    return trail


def _rgi_cons(points, values, flat_grid, offsets, shape, values_flat, ndim,
              mcode, bounds_error, fill_value, extrapolate, rest):
    """Construct the interpolator class matching `values`. Internal.

    The last step of `_rgi_build`, split out so the dtype and the trailing
    rank, which choose the class, are read from the argument TYPES.
    """
    trail = np.ndim(values) - len(points)
    cls = _RGI_CLASSES[('c' if np.iscomplexobj(values) else 'f',
                        max(trail, 0))]
    if trail <= 0:
        return cls(flat_grid, offsets, shape, values_flat, ndim, mcode,
                   bounds_error, _fill_scalar(fill_value), extrapolate)
    nvals = int(np.prod(rest)) if rest.size else 1
    return cls(flat_grid, offsets, shape, values_flat, ndim, mcode,
               bounds_error, _fill_vec(fill_value, rest, nvals), extrapolate,
               rest, nvals)


@overload(_rgi_cons)
def _rgi_cons_ovl(points, values, flat_grid, offsets, shape, values_flat,
                  ndim, mcode, bounds_error, fill_value, extrapolate, rest):
    """`_rgi_cons` inside ``@njit``, one arm per class.

    The chosen class is a constant in the compiled body, which is what lets
    one factory return a different jitclass per dtype and per trailing rank.
    """
    trail = _values_trail(points, values)
    cls = _RGI_CLASSES[('c' if isinstance(values.dtype, types.Complex)
                        else 'f', trail)]
    if trail == 0:
        def impl(points, values, flat_grid, offsets, shape, values_flat,
                 ndim, mcode, bounds_error, fill_value, extrapolate, rest):
            return cls(flat_grid, offsets, shape, values_flat, ndim, mcode,
                       bounds_error, _fill_scalar(fill_value), extrapolate)
        return impl

    def impl(points, values, flat_grid, offsets, shape, values_flat,
             ndim, mcode, bounds_error, fill_value, extrapolate, rest):
        nvals = 1
        for i in range(rest.shape[0]):
            nvals *= rest[i]
        return cls(flat_grid, offsets, shape, values_flat, ndim, mcode,
                   bounds_error, _fill_vec(fill_value, rest, nvals),
                   extrapolate, rest, nvals)
    return impl


def _values_cast(values):
    """`values` ravelled into the dtype the grid is stored in.

    complex128 for a complex `values` of any width, float64 for anything else,
    which is what scipy returns: measured, complex64 comes back complex128 and
    float32 and integer values come back float64.
    """
    return np.ascontiguousarray(values).ravel().astype(
        np.complex128 if np.iscomplexobj(values) else np.float64)


@overload(_values_cast)
def _values_cast_ovl(values):
    """`_values_cast` inside ``@njit``. The dtype is the whole decision."""
    if isinstance(values.dtype, types.Complex):
        def impl(values):
            return np.ascontiguousarray(values).ravel().astype(np.complex128)
        return impl

    def impl(values):
        return np.ascontiguousarray(values).ravel().astype(np.float64)
    return impl


def _trail_count_ty(points, values):
    """The trailing rank the CLASS was chosen for, as a number.

    `_rgi_build` compares it against the rank `values` actually has, which is
    what refuses a vector-valued `values` whose `points` arrived as a list.
    """
    return _values_trail_py(points, values)


@overload(_trail_count_ty)
def _trail_count_ty_ovl(points, values):
    """`_trail_count_ty` inside ``@njit``, baked in while compiling."""
    trail = _values_trail(points, values)

    def impl(points, values):
        return trail
    return impl


def _values_trail_py(points, values):
    """`_values_trail` over VALUES rather than types, for the Python bodies."""
    trail = np.ndim(values) - len(points)
    return trail if trail > 0 else 0


@njit
def _rgi_build(points, values, method, bounds_error, fill_value, extrapolate):
    """The one body both `RegularGridInterpolator` entry points reach.

    `fill_value` already carries the stored dtype and `extrapolate` is already
    the 0/1 code, so ``None`` has been resolved before this runs.
    """
    mcode = _method_code(method)

    ndim = len(points)
    if ndim < 1:
        raise ValueError("points must contain at least one axis array")

    # 'pchip' runs a cubic along every axis, so every axis needs four nodes.
    # scipy checks this first, before the axes are read for anything else.
    if mcode == 2:
        for j in range(ndim):
            if points[j].shape[0] < 4:
                _pchip_dims_raise(points[j].shape[0], j)
        _pchip_no_complex(values)

    # --- pass 1: validate each axis, record its length and orientation ----
    shape_arr = np.empty(ndim, np.int64)
    desc = np.zeros(ndim, np.int64)
    total = 0
    for j in range(ndim):
        p = points[j]
        if p.ndim != 1:
            raise ValueError("The points in dimension " + str(j)
                             + " must be 1-dimensional")
        if p.shape[0] < 1:
            raise ValueError("The points in dimension " + str(j)
                             + " must have at least 1 node")
        up = True
        down = True
        for i in range(p.shape[0] - 1):
            if p[i + 1] <= p[i]:
                up = False
            if p[i + 1] >= p[i]:
                down = False
        if not up and not down:
            raise ValueError("The points in dimension " + str(j)
                             + " must be strictly ascending or descending")
        if not up:
            desc[j] = 1
        shape_arr[j] = p.shape[0]
        total += p.shape[0]

    if ndim > values.ndim:
        raise ValueError("There are " + str(ndim) + " point arrays, but "
                         "values has " + str(values.ndim) + " dimensions")
    for j in range(ndim):
        if values.shape[j] != shape_arr[j]:
            raise ValueError("There are " + str(shape_arr[j]) + " points and "
                             + str(values.shape[j]) + " values in dimension "
                             + str(j))
    # the trailing axes of `values`, one block of numbers per grid node. The
    # rank of the result follows this, so it is fixed while compiling and the
    # run-time count has to agree with the class that was chosen.
    ntrail = values.ndim - ndim
    if ntrail != _trail_count_ty(points, values):
        raise ValueError(_TRAIL_LIST_MSG)
    rest = np.empty(ntrail, np.int64)
    nvals = 1
    for i in range(ntrail):
        rest[i] = values.shape[ndim + i]
        nvals *= rest[i]

    # --- pass 2: pack the ragged axes into one flat buffer + offsets ------
    flat_grid = np.empty(total, np.float64)
    offsets = np.zeros(ndim + 1, np.int64)
    k = 0
    for j in range(ndim):
        p = points[j]
        nj = p.shape[0]
        if desc[j] == 1:
            for i in range(nj):                 # store ascending
                flat_grid[k + i] = p[nj - 1 - i]
        else:
            for i in range(nj):
                flat_grid[k + i] = p[i]
        k += nj
        offsets[j + 1] = k

    values_flat = _values_cast(values)
    for j in range(ndim):
        if desc[j] == 1:
            values_flat = _flip_axis_flat(values_flat, shape_arr, j, nvals)

    return _rgi_cons(points, values,
                     np.ascontiguousarray(flat_grid), offsets, shape_arr,
                     np.ascontiguousarray(values_flat), ndim, mcode,
                     1 if bounds_error else 0, fill_value,
                     1 if extrapolate else 0, rest)


def _xi_as_array(xi):
    """`xi` as one ``(..., ndim)`` array, scipy's `_ndim_coords_from_arrays`.

    A tuple or list of coordinate arrays is broadcast together and stacked on
    a new trailing axis, which is what lets a meshgrid be passed straight in.
    A one-element tuple is its element. Anything else is taken as the array it
    already is.

    Parameters
    ----------
    xi : ndarray, or tuple or list of ndarray

    Returns
    -------
    ndarray
        Contiguous float64, with the query dimension last.
    """
    if isinstance(xi, (tuple, list)):
        if len(xi) == 1:
            return np.ascontiguousarray(np.asarray(xi[0], np.float64))
        parts = np.broadcast_arrays(*[np.asarray(a, np.float64) for a in xi])
        return np.ascontiguousarray(np.stack(parts, axis=-1))
    return np.ascontiguousarray(np.asarray(xi, np.float64))


@overload(_xi_as_array)
def _xi_as_array_ovl(xi):
    """`_xi_as_array` inside ``@njit``.

    ``np.broadcast_arrays`` cannot be reached with a runtime-length argument
    list, so the broadcast is done as index arithmetic: a source axis of
    length 1 gets stride 0 and repeats.
    """
    if not isinstance(xi, (types.BaseTuple, types.List, types.ListType)):
        def impl(xi):
            return np.ascontiguousarray(np.asarray(xi, np.float64))
        return impl

    if isinstance(xi, types.BaseTuple) and len(xi) == 1:
        def impl(xi):
            return np.ascontiguousarray(np.asarray(xi[0], np.float64))
        return impl

    if isinstance(xi, types.UniTuple):
        el = xi.dtype
    elif isinstance(xi, (types.List, types.ListType)):
        el = xi.dtype
    else:
        raise TypingError("xi must be one array, or a tuple of coordinate "
                          "arrays of the same dtype and layout")
    ND = el.ndim

    def impl(xi):
        nt = len(xi)
        shp = np.ones(ND, np.int64)
        for j in range(nt):
            s = xi[j].shape
            for d in range(ND):
                if s[d] != 1:
                    if shp[d] != 1 and shp[d] != s[d]:
                        raise ValueError("the coordinate arrays in xi cannot "
                                         "be broadcast together")
                    shp[d] = s[d]
        m = 1
        for d in range(ND):
            m *= shp[d]
        flat = np.empty((m, nt), np.float64)
        st = np.empty(ND, np.int64)
        for j in range(nt):
            a = np.ascontiguousarray(np.asarray(xi[j], np.float64)).ravel()
            s = xi[j].shape
            acc = 1
            for d in range(ND - 1, -1, -1):
                st[d] = 0 if s[d] == 1 else acc
                acc *= s[d]
            for lin in range(m):
                rem = lin
                src = 0
                for d in range(ND - 1, -1, -1):
                    src += (rem % shp[d]) * st[d]
                    rem //= shp[d]
                flat[lin, j] = a[src]
        return flat.reshape(to_fixed_tuple(shp, ND) + (nt,))
    return impl


def _interpn_eval(interp, q):
    """Evaluate `q` on `interp` and return scipy's ``q.shape[:-1]``.

    A rank-1 `q` is scipy's ``reshape(-1, ndim)``, so one coordinate per axis
    gives a length-1 result and a 1-D grid reads a run of points.
    """
    if q.ndim == 1:
        return interp.ev(q.reshape(-1, interp.ndim))
    if q.ndim == 2:
        return interp.ev(q)
    return interp.ev_nd(q)


@overload(_interpn_eval)
def _interpn_eval_ovl(interp, q):
    """`_interpn_eval` inside ``@njit``, one arm per rank.

    numba's dead-branch pruning reads conditions built from a function's own
    ARGUMENTS, and `q` is a local in `_interpn_run`, so ``if q.ndim == 1``
    there leaves every arm to be typed and the rank-3 arm cannot type
    ``interp.ev``. Choosing the arm here resolves the rank while compiling.
    """
    if q.ndim == 1:
        def impl(interp, q):
            return interp.ev(q.reshape(-1, interp.ndim))
        return impl
    if q.ndim == 2:
        def impl(interp, q):
            return interp.ev(q)
        return impl

    def impl(interp, q):
        return interp.ev_nd(q)
    return impl


# ---------------------------------------------------------------------------
# method='splinef2d': `interpn`'s bivariate-spline arm
# ---------------------------------------------------------------------------

#: Raised when `values` is not a 2-D grid of numbers.
_SF2D_NDIM_MSG = ("The method splinef2d can only be used for 2-dimensional "
                  "input data")

#: Raised when `points` and `values` disagree on the number of axes.
_SF2D_SCALAR_MSG = ("The method splinef2d can only be used for scalar data "
                    "with one point per coordinate")

#: Raised for `fill_value=None`, or `extrapolate=True`, with `bounds_error`
#: off. A bivariate spline is not evaluated outside its rectangle.
_SF2D_EX_MSG = "The method splinef2d does not support extrapolation."

#: Raised for a complex `values`. scipy takes its real part.
_SF2D_CX_MSG = ("The method splinef2d needs a real values; pass "
                "values.real or values.imag")


@njit
def _sf2d_raise(msg):
    """Raise `msg` as a ``ValueError``. Internal.

    Called as a STATEMENT, never as the value of an expression. A bare
    ``raise`` inside an ``@overload`` arm types that arm as returning ``none``,
    so the caller stops compiling before the message is ever reached.
    """
    raise ValueError(msg)


@njit
def _sf2d_bounds(points, values):
    """Validate the two axes and return their extents. Internal.

    Returns
    -------
    lo, hi : 1-D float64 ndarray, length 2
        The smallest and largest coordinate on each axis, whichever way the
        axis runs.
    """
    lo = np.empty(2, np.float64)
    hi = np.empty(2, np.float64)
    for j in range(2):
        p = points[j]
        if p.ndim != 1:
            raise ValueError("The points in dimension " + str(j)
                             + " must be 1-dimensional")
        if p.shape[0] < 1:
            raise ValueError("The points in dimension " + str(j)
                             + " must have at least 1 node")
        up = True
        down = True
        for i in range(p.shape[0] - 1):
            if p[i + 1] <= p[i]:
                up = False
            if p[i + 1] >= p[i]:
                down = False
        if not up and not down:
            raise ValueError("The points in dimension " + str(j)
                             + " must be strictly ascending or descending")
        if up:
            lo[j] = p[0]
            hi[j] = p[p.shape[0] - 1]
        else:
            lo[j] = p[p.shape[0] - 1]
            hi[j] = p[0]
    for j in range(2):
        p = points[j]
        if values.shape[j] != p.shape[0]:
            raise ValueError("There are " + str(p.shape[0]) + " points and "
                             + str(values.shape[j]) + " values in dimension "
                             + str(j))
    return lo, hi


@njit
def _sf2d_dim_check(nlast):
    """Refuse an `xi` whose trailing length is not 2. Internal."""
    if nlast != 2:
        raise ValueError("The requested sample points xi have dimension "
                         + str(nlast) + ", but this RegularGridInterpolator "
                         "has dimension 2")


@njit
def _sf2d_arm(points, values, q2, lo, hi, bounds_error, fill_value):
    """Evaluate the bivariate spline at `q2`, shape ``(m, 2)``. Internal.

    The spline is built from the RAW `points`, so a descending axis raises
    where `method='linear'` accepts it, and the in-bounds mask is built from
    the extents, so a NaN coordinate is out of bounds rather than a special
    case.
    """
    m = q2.shape[0]
    if bounds_error:
        for j in range(2):
            for i in range(m):
                if not (lo[j] <= q2[i, j] <= hi[j]):
                    raise ValueError("One of the requested xi is out of "
                                     "bounds in dimension " + str(j))
    ok = np.empty(m, np.bool_)
    nv = 0
    for i in range(m):
        v = (lo[0] <= q2[i, 0] <= hi[0]) and (lo[1] <= q2[i, 1] <= hi[1])
        ok[i] = v
        if v:
            nv += 1
    a = np.empty(nv, np.float64)
    b = np.empty(nv, np.float64)
    k = 0
    for i in range(m):
        if ok[i]:
            a[k] = q2[i, 0]
            b[k] = q2[i, 1]
            k += 1
    spl = RectBivariateSpline(points[0], points[1], values, None, 3, 3,
                              0.0, 20)
    ev = spl.ev(a, b)
    out = np.empty(m, np.float64)
    k = 0
    for i in range(m):
        if ok[i]:
            out[i] = ev[k]
            k += 1
        else:
            out[i] = fill_value
    return out


def _sf2d_eval(points, values, q, lo, hi, bounds_error, fill_value):
    """`_sf2d_arm` over an `xi` of any rank, returning ``q.shape[:-1]``.

    A rank-1 `q` is scipy's ``reshape(-1, 2)``, so one coordinate pair is a
    single point.
    """
    if q.ndim == 1:
        return _sf2d_arm(points, values, q.reshape(-1, 2), lo, hi,
                         bounds_error, fill_value)
    _sf2d_dim_check(q.shape[q.ndim - 1])
    if q.ndim == 2:
        return _sf2d_arm(points, values, q, lo, hi, bounds_error, fill_value)
    r = _sf2d_arm(points, values, q.reshape(-1, 2), lo, hi, bounds_error,
                  fill_value)
    return r.reshape(q.shape[:-1])


@overload(_sf2d_eval)
def _sf2d_eval_ovl(points, values, q, lo, hi, bounds_error, fill_value):
    """`_sf2d_eval` inside ``@njit``, one arm per rank.

    `q` is a local in the caller, so a rank test written there leaves every
    arm to be typed and the rank-3 arm cannot type a ``(m, 2)`` index. The
    rank is resolved here instead, while the call compiles.
    """
    if q.ndim == 1:
        def impl(points, values, q, lo, hi, bounds_error, fill_value):
            return _sf2d_arm(points, values, q.reshape(-1, 2), lo, hi,
                             bounds_error, fill_value)
        return impl
    if q.ndim == 2:
        def impl(points, values, q, lo, hi, bounds_error, fill_value):
            _sf2d_dim_check(q.shape[1])
            return _sf2d_arm(points, values, q, lo, hi, bounds_error,
                             fill_value)
        return impl

    nd = q.ndim

    def impl(points, values, q, lo, hi, bounds_error, fill_value):
        _sf2d_dim_check(q.shape[nd - 1])
        r = _sf2d_arm(points, values, q.reshape(-1, 2), lo, hi, bounds_error,
                      fill_value)
        return r.reshape(q.shape[:-1])
    return impl


def _splinef2d_run(points, values, xi, bounds_error, fill_value, extrapolate):
    """The one body both `interpn` entry points reach for ``'splinef2d'``.

    The checks run in scipy's order: the rank of `values`, extrapolation, the
    number of point arrays, the axes, the trailing length of `xi`, then the
    bounds.
    """
    if np.ndim(values) != 2:
        raise ValueError(_SF2D_NDIM_MSG)
    if np.iscomplexobj(values):
        raise ValueError(_SF2D_CX_MSG)
    if not bounds_error and extrapolate:
        raise ValueError(_SF2D_EX_MSG)
    if len(points) > 2:
        raise ValueError("There are " + str(len(points))
                         + " point arrays, but values has 2 dimensions")
    if len(points) != 2:
        raise ValueError(_SF2D_SCALAR_MSG)
    lo, hi = _sf2d_bounds(tuple(points), values)
    return _sf2d_eval(tuple(points), values, _xi_as_array(xi), lo, hi,
                      bounds_error, _fill_scalar(fill_value))


@overload(_splinef2d_run)
def _splinef2d_run_ovl(points, values, xi, bounds_error, fill_value,
                       extrapolate):
    """`_splinef2d_run` inside ``@njit``.

    The rank and dtype of `values`, and the length of a tuple `points`, decide
    which body can be typed at all: a `values` that is not a real 2-D array
    never reaches a bivariate spline, and neither does a `points` that cannot
    supply two axes. Those arms report the refusal through `_sf2d_raise` and
    then return an array of the right type, so the message survives the
    caller's use of the result.
    """
    bad = ''
    if values.ndim != 2:
        bad = _SF2D_NDIM_MSG
    elif isinstance(values.dtype, types.Complex):
        bad = _SF2D_CX_MSG
    elif isinstance(points, types.BaseTuple):
        if len(points) > 2:
            bad = ("There are " + str(len(points))
                   + " point arrays, but values has 2 dimensions")
        elif len(points) != 2:
            bad = _SF2D_SCALAR_MSG
    if bad:
        msg = bad

        def impl(points, values, xi, bounds_error, fill_value, extrapolate):
            _sf2d_raise(msg)
            return np.empty(0, np.float64)
        return impl

    def impl(points, values, xi, bounds_error, fill_value, extrapolate):
        if not bounds_error and extrapolate:
            raise ValueError(_SF2D_EX_MSG)
        if len(points) > 2:
            raise ValueError("There are " + str(len(points))
                             + " point arrays, but values has 2 dimensions")
        if len(points) != 2:
            raise ValueError(_SF2D_SCALAR_MSG)
        lo, hi = _sf2d_bounds(points, values)
        return _sf2d_eval(points, values, _xi_as_array(xi), lo, hi,
                          bounds_error, _fill_scalar(fill_value))
    return impl


def _is_splinef2d(method):
    """Dispatch predicate: ``method`` is the string ``'splinef2d'``.

    An omitted default reaches an ``@overload`` as the raw Python value inside
    ``types.Omitted``, so it is unwrapped before the test. A `method` held in a
    variable inside ``@njit`` has no literal value and is not this arm.
    """
    mt = method.value if isinstance(method, types.Omitted) else method
    if isinstance(mt, str):
        return mt == 'splinef2d'
    return (isinstance(mt, types.StringLiteral)
            and mt.literal_value == 'splinef2d')


def interpn(points, values, xi, method="linear", bounds_error=True,
            fill_value=np.nan, extrapolate=False):
    """Interpolate on a regular grid.

    A one-shot wrapper: builds a `RegularGridInterpolator` and evaluates it at
    `xi`.

    Parameters
    ----------
    points : tuple or list of 1-D float array_like
        Axis coordinates, as for `RegularGridInterpolator`.
    values : array_like
        Grid data, as for `RegularGridInterpolator`: its leading shape is the
        grid shape, a complex `values` is held as complex128 and anything else
        as float64, and up to 2 trailing axes give a block of numbers at each
        node.
    xi : array_like, or tuple or list of array_like
        Query points of shape ``(..., ndim)``, any rank. A rank-1 `xi` is
        read as ``reshape(-1, ndim)``, so one coordinate per axis is a single
        point and a 1-D grid reads a run of points. A tuple or list of
        coordinate arrays is broadcast together and stacked on a new trailing
        axis, which is what lets a meshgrid be passed straight in; arrays that
        do not broadcast raise ``ValueError``.
    method : str, optional
        ``'linear'`` multilinear, ``'nearest'`` nearest-neighbour,
        ``'pchip'`` a shape-preserving cubic along each axis in turn,
        ``'splinef2d'`` a bicubic `RectBivariateSpline` over a 2-D grid.
        Default ``'linear'``. ``'pchip'`` needs at least four nodes on every
        axis and a real `values`.
    bounds_error : bool, optional
        Raise on out-of-bounds query points. Default True.
    fill_value : scalar, array_like or None, optional
        Out-of-bounds value when ``bounds_error`` is False; ``None``
        extrapolates linearly off the edge cell. A sequence is one value per
        component. A value that cannot be cast to the stored dtype, or whose
        shape does not broadcast, raises ``ValueError``. Default NaN.
        ``'splinef2d'`` takes a scalar and refuses ``None``.
    extrapolate : bool, optional
        Continue the interpolant off the edge cell for an out-of-bounds query,
        ignoring `fill_value`. The same thing ``fill_value=None`` does.
        Default False. ``'splinef2d'`` refuses it.

    Returns
    -------
    out : ndarray of shape ``xi.shape[:-1] + values.shape[ndim:]``
        Interpolated values, one per query point, in the dtype `values` are
        stored in. ``'splinef2d'`` returns float64.

    See Also
    --------
    scipy.interpolate.interpn : The scipy routine this mirrors.

    Notes
    -----
    It builds a `RegularGridInterpolator` and evaluates it, both in compiled
    code. Rebuilding the interpolator on every call is wasteful, so for
    repeated queries against one grid build it once and call it, ``rgi(xi)``.

    Only ``'linear'``, ``'nearest'``, ``'pchip'`` and ``'splinef2d'`` are
    implemented. scipy's ``'slinear'``, ``'cubic'`` and ``'quintic'`` are not,
    and raise.

    `extrapolate` has no scipy counterpart. It is `fill_value=None` under a
    boolean spelling; a scipy-shaped call never passes it.

    ``method='splinef2d'`` differs from the other methods in three ways:
    `values` must be a 2-D grid, extrapolation is refused with `bounds_error`
    off, and the spline is built from `points` as given, so a descending axis
    raises ``ValueError("x must be strictly increasing")`` where the other
    methods flip it and interpolate.

    **`method='splinef2d'` inside ``@njit``.** The arm is selected while the
    call compiles, since a 2-D grid and an N-D grid cannot be evaluated by the
    same compiled code. The name has to be written at the call site;
    ``method`` read from a variable raises ``ValueError``.

    **A complex `values` with ``method='splinef2d'``.** It raises
    ``ValueError``. scipy interpolates the real part and discards the
    imaginary one, with a ``ComplexWarning``.

    **A 1-D grid with ``method='splinef2d'``.** It raises
    ``ValueError("The method splinef2d can only be used for 2-dimensional
    input data")``. scipy raises ``IndexError: tuple index out of range``,
    from indexing the second axis of a grid that has one.

    Accuracy: see the module docstring.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.interpolate import interpn
    >>> x = np.linspace(0.0, 1.0, 5)
    >>> y = np.linspace(0.0, 1.0, 6)
    >>> vals = np.exp(x[:, None] + y[None, :])
    >>> @njit
    ... def one_shot(x, y, vals, xi):
    ...     return interpn((x, y), vals, xi)
    >>> float(one_shot(x, y, vals, np.array([[0.25, 0.4]]))[0])
    1.9155408290138962
    """
    if isinstance(points, list):
        points = tuple(points)
    fv, ex = _fill_pair(fill_value, extrapolate, values)
    if method == 'splinef2d':
        return _splinef2d_run(points, values, xi, bounds_error, fv, ex)
    return _interpn_run(points, values, xi, method, bounds_error, fv, ex)


@overload(interpn, prefer_literal=True)
def _interpn_ovl(points, values, xi, method="linear", bounds_error=True,
                 fill_value=np.nan, extrapolate=False):
    """`interpn` inside ``@njit``, the twin of the Python body above."""
    if _is_splinef2d(method):
        if _fill_pair_ty(fill_value):
            def impl(points, values, xi, method="linear", bounds_error=True,
                     fill_value=np.nan, extrapolate=False):
                return _splinef2d_run(points, values, xi, bounds_error,
                                      _nan_fill(values), 1)
            return impl

        def impl(points, values, xi, method="linear", bounds_error=True,
                 fill_value=np.nan, extrapolate=False):
            return _splinef2d_run(points, values, xi, bounds_error,
                                  _fill_of(fill_value, values),
                                  1 if extrapolate else 0)
        return impl

    if _fill_pair_ty(fill_value):
        def impl(points, values, xi, method="linear", bounds_error=True,
                 fill_value=np.nan, extrapolate=False):
            return _interpn_run(points, values, xi, method, bounds_error,
                                _nan_fill(values), 1)
        return impl

    def impl(points, values, xi, method="linear", bounds_error=True,
             fill_value=np.nan, extrapolate=False):
        return _interpn_run(points, values, xi, method, bounds_error,
                            _fill_of(fill_value, values),
                            1 if extrapolate else 0)
    return impl


@njit
def _interpn_run(points, values, xi, method, bounds_error, fill_value,
                 extrapolate):
    """The one body both `interpn` entry points reach."""
    interp = _rgi_build(points, values, method, bounds_error, fill_value,
                        extrapolate)
    return _interpn_eval(interp, _xi_as_array(xi))
