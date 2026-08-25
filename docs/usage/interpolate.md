# interpolate (FITPACK splines)

`scipy.interpolate` routines callable inside `@njit` code, for 1-D, 2-D and N-D
interpolation and smoothing. `@njit` is numba's decorator that compiles a Python
function to machine code. The spline routines are backed by the same Dierckx
FITPACK Fortran that scipy uses. See the [preamble](./index.md) for `prange`
safety and caching.

Each class name is a factory function that builds the spline object, and the
object is callable. `spl(q)` evaluates at query points. `spl(xi)` on a single
number returns one value, through `spl.ev_one(xi)`. `spl(array)` returns the
value at every point, through `spl.ev(array)`. `spl[q]` is another spelling of
`.ev`. These names hold across every spline object on this page. Derivatives,
integrals and knots have their own named methods, shown with each class below.

The factory keeps its keyword defaults from Python and inside `@njit` alike, so
`CubicSpline(x, y)` works both ways. The jitclass behind the factory, a
numba-compiled class, carries no defaults, so a partial constructor call is safe
only through the factory name: inside `@njit`, call the class by its name, or
spell every argument out.

## Reusing a spline

Building a spline is the expensive step; evaluating it is cheap. Build the object
once, outside the loop that queries it, and reuse it. A spline rebuilt inside the
loop is refit on every call.

A spline reaches an `@njit` function either as an argument, which suits one
chosen at run time, or as a module-level object the function reads directly or
closes over. The second form is what lets a spline appear inside a solver
callback, shown under "A spline inside a solver callback" below. The first call
to a compiled function pays numba's one-time compile cost; later calls run the
machine code.

Query points batch the same way, and for the same reason. A bivariate spline
object costs 693 ns per call and 315 ns per point when the points arrive
together, measured on a 200x500 table. Passing the whole array of query points
to one call is both the shorter spelling and the faster one.
[Evaluation cost per call](#evaluation-cost-per-call) has the measurement and
the case where a loop cannot batch.

## 1-D interpolation: `interp1d`

`interp1d` builds a callable interpolant from `(x, y)` samples. It is a jitclass
and is `prange`-safe, where `prange` is numba's parallel loop.

```python
import numpy as np
from numba import njit
from scijit.interpolate import interp1d

x = np.linspace(0.0, 2*np.pi, 15)
y = np.sin(x)

f_lin  = interp1d(x, y, "linear")     # the default
f_cub  = interp1d(x, y, "cubic")
f_near = interp1d(x, y, "nearest")

q = np.linspace(0.5, 5.5, 30)

@njit
def sample(f, q):
    return f(q)                       # f(scalar), f(array), or f[q]

a = sample(f_lin, np.array([1.0, 3.0]))
b = sample(f_cub, np.array([1.0, 3.0]))
```

```
f_lin(np.array([1.0, 3.0]))   [0.82589014 0.13688702]
f_cub(np.array([1.0, 3.0]))   [0.84144992 0.14109502]
```

`kind` is `"linear"`, `"nearest"`, `"previous"`, `"next"`, `"quadratic"`,
`"cubic"` or `"nearest-up"`, with `"slinear"` and `"zero"` accepted as scipy
spells them. An integer is the spline order.

`y` may be complex, for every `kind`, and the result is then `complex128`. The
spline kinds fit the real and the imaginary parts on one knot vector. A complex
`fill_value` needs a complex `y` and raises otherwise, and a complex `x` is
refused. `y` is capped at rank 3.

### Out-of-range points

A point outside `[x[0], x[-1]]` raises by default. `bounds_error=None`, the
default, means raise unless extrapolating.

```python
interp1d(x, y, "linear")(np.array([-1.0]))
```

```
ValueError: A value (-1.0) in x_new is below the interpolation range's minimum value (0.0).
```

Pass `bounds_error=False` to substitute `fill_value` instead. `fill_value` takes
one number, or scipy's `(below, above)` pair.

```python
f = interp1d(x, y, "linear", bounds_error=False, fill_value=(-1.0, 99.0))
f(np.array([-1.0, 1.0, 100.0]))
```

```
[-1.          0.82589014 99.        ]
```

`extrapolate=True` spells scipy's `fill_value="extrapolate"`. Asking to
extrapolate and to raise at the same time is rejected.

### Unsorted `x`

`x` is sorted on construction unless `assume_sorted=True` says it is already
ascending. With that flag set, an unsorted `x` is a caller error.

```python
perm = np.array([3, 0, 4, 1, 2, 5, 7, 6, 8, 9, 10, 12, 11, 13, 14])
f = interp1d(x[perm], y[perm], "linear")   # assume_sorted defaults to False
f(np.array([1.0, 3.0]))
```

```
[0.82589014 0.13688702]
```

## Cubic splines: `CubicSpline`

`CubicSpline` fits a piecewise cubic through the data with continuous first and
second derivatives.

```python
import numpy as np
from numba import njit
from scijit.interpolate import CubicSpline

xs = np.linspace(0.0, 2*np.pi, 12)
ys = np.sin(xs)
cs = CubicSpline(xs, ys, bc_type="not-a-knot")
q  = np.linspace(0.5, 5.5, 30)

@njit
def eval_cubic(cs, q):
    v  = cs(q)                   # values (extrapolates via the end segments)
    d  = cs.derivative_ev(q, 1)  # first derivative (nu=1)
    I  = cs.integral(0.0, np.pi) # definite integral
    return v, d, I

v, d, I = eval_cubic(cs, q)      # I -> 2.0001726527965884

knots  = cs.get_knots()          # the breakpoints
coeffs = cs.get_coeffs()         # PPoly c, shape (4, n-1)
```

```
cs.integral(0, pi)   2.0001726527965884
coeffs.shape         (4, 11)
```

`.get_knots()` returns the breakpoints and `.get_coeffs()` the PPoly
coefficients. Both are available on every spline class on this page.

`bc_type` takes `'not-a-knot'`, `'natural'`, `'clamped'`, `'periodic'`, or a
per-end derivative condition. `CubicSpline` and `make_interp_spline` spell the
condition DIFFERENTLY, and each refuses the other's spelling:

```
CubicSpline(x, y, bc_type=((1, 0.5), (2, 3.0)))            a pair of PAIRS
make_interp_spline(x, y, 3, bc_type=([(1, 0.5)], [(2, 3.0)]))   a pair of LISTS
```

Each pair is `(order, value)`. Order 1 fixes the first derivative at that end,
order 2 the second. `((1, 0.0), (1, 0.0))` is `'clamped'` and
`((2, 0.0), (2, 0.0))` is `'natural'`. `'periodic'` requires `y[0] == y[-1]` and
folds the query into `[x[0], x[-1]]` rather than extrapolating, so
`f(x + period) == f(x)`. A derivative value is one scalar per end and applies to
every series of an N-D `y`.

## Arbitrary knots: `make_interp_spline`, `BSpline`

`make_interp_spline` fits an interpolating B-spline and returns a `BSpline`
object. `BSpline` takes an arbitrary `(t, c, k)` triple, not only the knot
layouts FITPACK produces. A FITPACK `tck` from `splrep` passes straight in, and
the extra trailing coefficients are ignored.

```python
import numpy as np
from numba import njit
from scijit.interpolate import BSpline, make_interp_spline

x = np.linspace(0.0, 2*np.pi, 15)
spl = make_interp_spline(x, np.sin(x), 3, bc_type="not-a-knot")
q = np.linspace(0.2, 6.0, 40)

@njit
def evaluate(spl, q):
    v = spl(q)                          # values             (or spl[q])
    d = spl.derivative_ev(q, 1)         # nu-th derivative
    A = spl.antiderivative_ev(q, 1)     # nu-th antiderivative
    I = spl.integral(0.0, np.pi)        # definite integral
    return v, d, A, I

v, d, A, I = evaluate(spl, q)           # I -> 1.9999972390629823

t = np.array([0., 0., 0., 0., 1., 2., 3., 3., 3., 3.])   # any knots
c = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 0.0])

@njit
def hand_built(t, c, qs):
    s = BSpline(t, c, 3)                # a factory, defaults apply here too
    return s(qs), s(1.5)

vv, o = hand_built(t, c, np.array([0.5, 1.5, 2.5]))
```

```
hand_built  [-0.86979167  1.546875    0.19791667]   1.546875
```

`.derivative_ev(q, nu)` gives the `nu`-th derivative, `.antiderivative_ev(q, nu)`
the `nu`-th antiderivative, and `.integral(a, b)` the definite integral.

`make_interp_spline` takes an explicit knot vector as `t`, in scipy's position:

```
make_interp_spline(x, y, k=3, t=my_knots)
```

`bc_type` is `"not-a-knot"` (any `k >= 1`), `"natural"`, `"clamped"` or
`"periodic"`, the middle two requiring `k == 3`. `BSpline` takes `extrapolate`
as `True`, `False`, `'periodic'` or `None`, where `None` and `0` read as
`False`.

**Deviations.** There is no weights argument here. The collocation system is
solved densely, so the fit costs `O(n^3)` and `n` should stay in the low
thousands. `.derivative_ev(q, nu)` with `nu > k` returns zeros, where the
function `splder` raises `ValueError` for that.

## Shape-preserving interpolants: `Akima1DInterpolator`, `PchipInterpolator`

Both pick node slopes that keep the interpolant faithful to the shape of the
data. Akima suppresses overshoot near sharp changes. PCHIP keeps each segment
monotone wherever the data is, through the Fritsch-Carlson Hermite scheme.

```python
import numpy as np
from numba import njit
from scijit.interpolate import Akima1DInterpolator, PchipInterpolator

xa = np.linspace(0.0, 10.0, 11)
ya = np.array([0., 2., 1., 3., 2., 6., 5., 5., 8., 9., 9.])
ak = Akima1DInterpolator(xa, ya)

xp = np.linspace(0.0, 10.0, 11)
yp = np.array([0.,1.,1.,1.,2.,3.,3.,3.,4.,5.,5.])
pc = PchipInterpolator(xp, yp)

@njit
def evaluate(ak, pc, q):
    return ak(q), pc(q)                # pass a scalar for a single point

a_out, p_out = evaluate(ak, pc, np.array([2.5, 7.5]))
```

```
ak(q)   [1.953125 6.4375  ]
pc(q)   [1.    3.375]
```

`Akima1DInterpolator`'s `method="makima"` selects the modified slope rule.
`extrapolate` is `True`, `False`, `'periodic'` or `None`. The `None` default
gives NaN outside `[x[0], x[-1]]`.

## Smoothing and least-squares fits: `UnivariateSpline`, `InterpolatedUnivariateSpline`, `LSQUnivariateSpline`

Three univariate classes build and return a spline object. `UnivariateSpline`
fits a smoothing spline, trading closeness to the data for smoothness through the
factor `s`. `InterpolatedUnivariateSpline` passes through every point.
`LSQUnivariateSpline` fits in least squares against a knot vector the caller
supplies.

```python
import numpy as np
from numba import njit
from scijit.interpolate import UnivariateSpline

x = np.linspace(0.0, 4.0, 40)
y = np.sin(x)

@njit
def smooth(x, y, q):
    spl = UnivariateSpline(x, y, None, None, 3, 0.5, 0, False)  # k=3, s=0.5
    return spl(q)

val = smooth(x, y, 1.5)
```

```
val   0.9719636188892988
```

The example spells out every argument. When skipping arguments inside `@njit`,
pass by keyword: the third positional is `w` and the fourth is `bbox`, not `k`.

### The rank-deficiency warning

A smoothing fit with a small `s` on noisy data can place knots so that one
B-spline coefficient is not determined by any data point. FITPACK reports
success and the fit still passes through the data, but between the data points it
carries a component the data never constrained. This is detected and warned. The
warning reaches `splrep`, `splprep`, all three `UnivariateSpline` classes and
`RectBivariateSpline`.

```python
import warnings
import numpy as np
from scijit.interpolate import UnivariateSpline

rng = np.random.default_rng(1)
x = np.linspace(0.0, 1.0, 30)
y = np.sin(6*x) + 0.05*rng.standard_normal(30)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    tight = UnivariateSpline(x, y, s=1e-3)
    print(len(w), "warning(s)")
```

```
1 warning(s)
UserWarning: the fitted spline is rank deficient: 1 B-spline coefficient(s)
are not determined by the data, because the knot search left a basis function
with no data point in its support (Schoenberg-Whitney). The spline still
reproduces the data points; between them it carries an arbitrary component.
A larger s, or fewer knots, avoids it.
```

A larger `s` is the fix. The same data at `s=0.05` is silent:

```
s=1e-3   knots 24   max|c| 1.171   1 warning
s=0.05   knots  5   max|c| 1.460   0 warnings
```

The condition is the Schoenberg-Whitney criterion, which says a B-spline basis
is determined only if every basis function has a data point in its support. The
behaviour is upstream Dierckx, and an interpolating fit (`s=0`) never triggers
it.

## 2-D interpolation: `RectBivariateSpline`, `SmoothBivariateSpline`

`RectBivariateSpline` fits a spline surface over a rectangular grid.
`SmoothBivariateSpline` fits scattered `(x, y, z)` samples.

```python
import numpy as np
from numba import njit
from scijit.interpolate import RectBivariateSpline

x = np.linspace(0.0, 1.0, 12)
y = np.linspace(0.0, 1.0, 15)
z = np.outer(np.sin(3*x), np.cos(2*y))

spl = RectBivariateSpline(x, y, z, None, 3, 3, 0.0, 20)   # kx=ky=3, s=0

@njit
def evaluate(spl, xs, ys):
    grid = spl(xs, ys)             # cross product, scipy's grid=True default
    pts  = spl.ev(xs, ys)          # scattered (x[i], y[i]) pairs
    one  = spl.ev_one(0.5, 0.5)    # single point
    return grid, pts, one

xs = np.array([0.2, 0.5, 0.8])
ys = np.array([0.3, 0.6, 0.9])
grid, pts, one = evaluate(spl, xs, ys)
```

```
grid.shape   (3, 3)
grid[0]      [ 0.46601793  0.20460175 -0.1282869 ]
scattered    [ 0.46601793  0.36144435 -0.15346563]
one point    0.5389408523180532
```

On the bivariate classes the callable form `spl(x, y)` runs the full cross
product of the two axes, which is scipy's `grid=True` default. The scattered
form `spl.ev(x, y)` evaluates at the pairs `(x[i], y[i])`. `spl.ev_one(x, y)`
takes a single point.

The FITPACK status is kept as the `.ier` attribute rather than raised on the
success codes. After a smoothing fit, check it. `0` means smoothing achieved,
`-1` an interpolating surface, `-2` a least-squares polynomial. Any other value
has already raised, except on `SmoothBivariateSpline`, which warns and returns a
usable object.

`RectSphereBivariateSpline` and `SmoothSphereBivariateSpline` fit the same way
over latitude and longitude on a sphere.

## N-D interpolation on a regular grid: `RegularGridInterpolator`, `interpn`

`RegularGridInterpolator` interpolates on a regular grid, built from a tuple of
ascending (or descending) 1-D axis arrays plus the grid `values`. It is a
jitclass and is `prange`-safe. The callable `rgi(xi)` evaluates a batch, where
`xi` has shape `(m, ndim)` and the result is `(m,)`. `.ev_point(p)` returns a
scalar for a single point, where `p` has shape `(ndim,)`.

```python
import numpy as np
from numba import njit
from scijit.interpolate import RegularGridInterpolator, interpn

x = np.linspace(0.0, 1.0, 5)
y = np.linspace(0.0, 2.0, 6)
X, Y = np.meshgrid(x, y, indexing='ij')
vals = np.sin(X) + np.cos(Y)

rgi = RegularGridInterpolator((x, y), vals, method="linear")
pts = np.array([[0.3, 0.7], [0.55, 1.2]])

@njit
def sample(rgi, pts):
    batch = rgi(pts)                            # (2,), batch of points
    one   = rgi.ev_point(np.array([0.3, 0.7]))  # scalar, single point
    return batch, one

batch, one = sample(rgi, pts)
out = interpn((x, y), vals, pts, method="linear")   # functional form
```

```
batch   [1.04660356 0.88222594]
one     1.0466035556355542
out     [1.04660356 0.88222594]
```

`method` is `"linear"`, `"nearest"` or `"pchip"`. `"pchip"` runs a
shape-preserving cubic along each axis in turn, and needs at least four nodes on
every axis and a real `values`. `"slinear"`, `"cubic"` and `"quintic"` raise
`ValueError`. For a smooth 2-D grid use `RectBivariateSpline`.

`fill_value=None` continues the interpolant off the grid, linearly for
`"linear"` and along the edge segment for `"pchip"`. Otherwise out-of-bounds
points raise (`bounds_error=True`) or return `fill_value`.

The result shape follows the rank of `xi`, and each rank has its own method:

- `(m, ndim)` returns `(m,)`, a batch of points, through `.ev`.
- Rank 3 or more is scipy's `(..., ndim)` and returns `xi.shape[:-1]`, through
  `.ev_nd`.
- Rank 1 is read as `xi.reshape(-1, ndim)`, through `.ev_1d`. One coordinate per
  axis returns shape `(1,)`, and on a 1-D grid a run of `m` coordinates is `m`
  points.

The callable `rgi(xi)` reaches these the same way `.ev` and `.ev_point` do. Pass
a method name as a second argument to re-select it for that call,
`rgi(xi, "nearest")`. `None` there keeps the method the object was built with.
The `_m` twins `.ev_m`, `.ev_1d_m`, `.ev_point_m` and `.ev_nd_m` take the
per-call method by name.

`interpn` also takes `xi` as a tuple of coordinate arrays, which are broadcast
together, so a meshgrid can be passed straight in. Its result has shape
`xi.shape[:-1]` at every rank.

`interpn` carries one more method, `"splinef2d"`, which
`RegularGridInterpolator` does not. It is a bicubic surface over the grid, so it
needs two point arrays and a real 2-D `values`, and it returns `float64`. It
does not extrapolate: `fill_value=None`, and `extrapolate=True`, raise
`ValueError` when `bounds_error` is off. The surface is built from `points` as
given, so a descending axis raises `ValueError("x must be strictly
increasing")` where the other methods flip it and interpolate. Inside `@njit`
the method name must be a literal at the call site. A `method` read from a
variable raises. A complex `values` raises here, and is truncated to its real
part by scipy.

A complex `values` otherwise returns `complex128`, and a complex `fill_value`
goes with it. `values` may also carry axes beyond the grid axes, up to two of
them, so each node holds a block of numbers rather than one. The result is then
`xi.shape[:-1]` followed by that trailing shape, and a `fill_value` sequence
broadcast up to it gives one value per component. Inside `@njit`, a `values`
with trailing axes needs `points` as a tuple, not a list. A `fill_value` whose
shape does not broadcast raises when the interpolator is built.

## N-D `y` and the `axis` argument

Every 1-D interpolator above accepts a `y` of rank 1, 2 or 3, together with an
`axis` naming which of its axes the interpolation runs along. Each position on
the remaining axes is an independent series over the same `x`, fitted and
evaluated in one call.

```python
import numpy as np
from numba import njit
from scijit.interpolate import CubicSpline

t = np.linspace(0.0, 2*np.pi, 15)
sig = np.stack([np.sin(t), np.cos(t), np.sin(2*t)])   # (3, 15), 3 sensors

@njit
def resample(sig, t, q):
    cs = CubicSpline(t, sig, axis=1, bc_type="not-a-knot")  # one fit, all 3 series
    return cs(q), cs(1.0)             # array query, then a single point

q = np.array([0.5, 1.5, 2.5, 3.5])
out, one = resample(sig, t, q)
```

```
sig.shape    (3, 15)
out.shape    (3, 4)
one.shape    (3,)
```

An array query returns the shape of `y` with the interpolation axis replaced by
`len(q)`, so the result keeps the caller's layout. A scalar query removes that
axis instead. A negative `axis` counts from the end, as `axis % y.ndim`.

`BSpline` is the one exception to where the extra axes live. Its trailing axes
are on the coefficient array `c`, not on a `y`.

A `y` of rank 4 or above raises `ValueError`.

## A spline inside a solver callback

A routine in `scijit.optimize` or `scijit.integrate` compiles the function it is
given: a residual for `fsolve` or `leastsq`, a model for `curve_fit`, an
integrand for the `quad` family. A spline is evaluated inside that function. Two
routes reach it, and which one applies depends on when the spline is known.

A spline assigned at module level, or captured from an enclosing function, has a
fixed identity when the callback compiles. numba freezes it into the compiled
code, so the callback calls the spline object directly. `spl(x, y)` on a
bivariate spline is the grid form, so a single point returns a `(1, 1)` array,
read with `[0, 0]`.

```python
import numpy as np
from numba import njit
from scijit.interpolate import RectBivariateSpline, bispeu
from scijit.optimize import fsolve

# An opacity table kappa(log rho, log T), cm^2/g, on a coarse grid:
# electron scattering plus a Kramers bound-free term.
logrho = np.linspace(-8.0, -4.0, 12)          # g/cm^3
logT   = np.linspace(3.5, 6.5, 15)            # K
RHO, T = np.meshgrid(10.0**logrho, 10.0**logT, indexing='ij')
logkap = np.log10(0.34 + 3.16e23 * RHO * T**(-3.5))

spl = RectBivariateSpline(logrho, logT, logkap, None, 3, 3, 0.0, 20)  # kx=ky=3, s=0

# Locate (log rho, log T) with kappa = 1 cm^2/g and log P = log rho + log T = -1.
LOGKAP_T = 0.0
LOGP_T   = -1.0

@njit
def residual(state):
    lr, lt = state[0], state[1]
    r0 = spl(lr, lt)[0, 0] - LOGKAP_T         # spline called directly
    r1 = lr + lt - LOGP_T
    return np.array([r0, r1])

@njit
def solve_object():
    return fsolve(residual, np.array([-6.0, 5.0]))

# The same spline, carried as its field arrays through args and read with bispeu.
@njit
def residual_raw(state, tx, ty, c, kx, ky):
    lr, lt = state[0], state[1]
    r0 = bispeu(np.array([lr]), np.array([lt]), tx, ty, c, kx, ky)[0] - LOGKAP_T
    r1 = lr + lt - LOGP_T
    return np.array([r0, r1])

@njit
def solve_raw(tx, ty, c, kx, ky):
    return fsolve(residual_raw, np.array([-6.0, 5.0]), (tx, ty, c, kx, ky))

root_obj = solve_object()
root_raw = solve_raw(spl.tx, spl.ty, spl.c, spl.kx, spl.ky)
```

```
object route root [-6.04054041  5.04054041]
array route root  [-6.04054041  5.04054041]
max|diff|         0.0
```

The two routes reach the same root, to `max|diff| == 0.0`. The object route
carries no arrays through `args` and needs no field names.

The array route is the one to use when the spline is not known when the callback
compiles. A spline built from data that arrives at run time has no fixed identity
to freeze, so it cannot be called directly inside the callback. It travels
instead as its knot and coefficient arrays through the solver's `args`, and the
callback evaluates them with the raw functions of the next section: `splev` for a
univariate `(t, c, k)`, `bispeu` or `bispev` for a bivariate `tx, ty, c, kx, ky`.
A scijit spline exposes these as fields, `spl.t`, `spl.c`, `spl.k` on a
univariate object and `spl.tx`, `spl.ty`, `spl.c`, `spl.kx`, `spl.ky` on a
bivariate one, not as scipy's single `.tck`.

```python
import numpy as np
from numba import njit
from scijit.interpolate import splrep, splev
from scijit.optimize import brentq

# An opacity-vs-temperature curve measured this run, not known at compile time.
Tgrid = np.linspace(3.5, 6.5, 25)                                 # log T
kap   = np.log10(0.34 + 3.16e23 * 1e-6 * (10.0**Tgrid)**(-3.5))   # log kappa, fixed rho

TARGET = 0.0     # find log T where kappa = 1.0 cm^2/g

@njit
def resid(lt, t, c, k):
    return splev(np.array([lt]), (t, c, k))[0] - TARGET

@njit
def find_T(logT_axis, logkap):
    t, c, k = splrep(logT_axis, logkap)             # fit at run time
    return brentq(resid, 3.6, 6.4, (t, c, k))       # tck travels through args

logT_root = find_T(Tgrid, kap)
kappa_root = 10.0**splev(np.array([logT_root]), splrep(Tgrid, kap))[0]
```

```
log T root  5.0515308458456
kappa there 1.0000000000015452
```

## Working with a `tck` directly

For a caller who already holds a `tck` triple, a family of functions reads it
without building an object. A spline is fully described by a `tck` triple
`(t, c, k)`: the knot vector, the B-spline coefficients and the degree. `splrep`
fits one from data.

```python
import numpy as np
from numba import njit
from scijit.interpolate import splrep, splev, splint, splder_ev

x = np.linspace(0.0, 2*np.pi, 25)
y = np.sin(x)

@njit
def fit_and_eval(x, y, q):
    tck = splrep(x, y)                        # w, k, s optional
    ys  = splev(q, tck)                       # values on q
    d   = splder_ev(np.array([1.0]), tck, 1)  # first derivative at 1.0
    area = splint(0.0, np.pi, tck)            # integral over [0, pi]
    return ys, d, area

ys, d, area = fit_and_eval(x, y, np.array([1.0, 2.0, 3.0]))
```

```
splev      [0.84146759 0.90928749 0.14111852]
splder_ev  [0.54042097]
splint     1.9999913920492483
```

`splev` takes a derivative order through its `der` argument. `splder_ev`
evaluates the `nu`-th derivative directly. `splint` returns the definite
integral over `[a, b]`.

Two more functions read a cubic `tck`. `sproot` returns the interior roots.
`spalde` returns the value together with all `k` derivatives at one point.

```python
import numpy as np
from numba import njit
from scijit.interpolate import splrep, sproot, spalde

x = np.linspace(0.0, 2*np.pi, 25)
y = np.sin(x)

@njit
def analyse(x, y):
    tck = splrep(x, y)
    roots = sproot(tck)            # zeros of the cubic spline
    ders  = spalde(np.pi, tck)     # value and all k derivatives at pi
    return roots, ders

roots, ders = analyse(x, y)
```

```
sproot   [3.14159265]
spalde   [ 1.25896956e-16 -9.99973688e-01 -1.27222187e-15  9.94275275e-01]
```

`splder` and `splantider` take a derivative or antiderivative of the whole `tck`
and return a new `(t, c, k)` triple, or a `BSpline` when passed one.

```python
import numpy as np
from numba import njit
from scijit.interpolate import splrep, splev, splder

x = np.linspace(0.0, 3.0, 40)
tck = splrep(x, np.sin(x))

@njit
def slope_at(tck, q):
    return splev(q, splder(tck, 1))     # spline of the first derivative

vals = slope_at(tck, np.array([1.0, 2.0]))
```

```
vals   [ 0.5403022  -0.41614676]
```

Inside `@njit` the `tck` spelling takes a TUPLE `(t, c, k)`, not a list. `splder`
and `splantider` also take a `BSpline`: `splder(spl, 1)`.

### Parametric curves: `splprep`

`splprep` fits a parametric curve through points that need not be a function of
one coordinate, such as a closed loop.

```python
import numpy as np
from numba import njit
from scijit.interpolate import splprep, splev

theta = np.linspace(0.0, 2*np.pi, 40)
pts = np.stack([np.cos(theta), np.sin(theta)])   # (2, 40): points on a circle

@njit
def fit_curve(pts, u):
    tck, uu = splprep(pts, None, None, None, None, 3, 0, 0.0)  # s=0 interpolates
    return splev(u, tck)           # a list, one array per dimension

xy = fit_curve(pts, np.array([0.0, 0.25, 0.5]))
```

```
x(u)   [ 1.00000000e+00 -8.86910156e-08 -9.99998234e-01]
y(u)   [-4.30331932e-19  9.99999007e-01 -1.73819292e-15]
```

`splev` on a parametric `tck` returns a list holding one array per curve
dimension. `splprep` also returns the parameter values `uu`.

`tck[1]` is a list of one coefficient array per dimension by default
(`c_list=1`). Setting `c_list=0` gives FITPACK's single flat array instead.
`full_output` and `c_list` select the return type, so inside `@njit` both must
be compile-time constants.

### Evaluating a bivariate `tck`: `bisplev`

`bispev` and `bispeu` evaluate a bivariate spline given as the separate FITPACK
arrays `tx`, `ty`, `c`, `kx`, `ky`, rather than through a class. `bispev` takes
the full cross product of two axes. `bispeu` takes scattered `(x[i], y[i])`
pairs. `bisplev` is scipy's name for the grid form, and takes those same five
arrays bundled as one `(tx, ty, c, kx, ky)` tuple.

```python
import numpy as np
from numba import njit
from scijit.interpolate import bispev, bispeu, bisplev

@njit
def on_grid():
    tx = np.array([0., 0., 1., 1.])
    ty = np.array([0., 0., 1., 1.])
    c  = np.array([0., 2., 1., 3.])     # corners of f(x, y) = x + 2*y
    return bispev(np.array([0.5]), np.array([0.5]), tx, ty, c, 1, 1)

@njit
def on_grid_tck():
    tx = np.array([0., 0., 1., 1.])
    ty = np.array([0., 0., 1., 1.])
    c  = np.array([0., 2., 1., 3.])
    return bisplev(np.array([0.5]), np.array([0.5]), (tx, ty, c, 1, 1))

@njit
def at_points():
    tx = np.array([0., 0., 1., 1.])
    ty = np.array([0., 0., 1., 1.])
    c  = np.array([0., 2., 1., 3.])
    return bispeu(np.array([0.25, 0.75]), np.array([0.5, 0.5]),
                  tx, ty, c, 1, 1)

print("bispev", on_grid())          # [[1.5]]
print("bisplev", on_grid_tck())     # [[1.5]]
print("bispeu", at_points())        # [1.25 1.75]
```

In practice the arrays come from a fit, or from a `RectBivariateSpline`
instance's `.tx`, `.ty` and `.c` attributes, rather than being written by hand.
When written by hand, the coefficient layout is FITPACK's flat form
`c[(ny-ky-1)*i + j]`, equal to `np.outer(cx, cy).ravel()`.

### Evaluation cost per call

A spline object holds its knots and coefficients as fields of a numba
jitclass. Reading those fields costs something, and the cost falls once per
call rather than once per point. Measured on a 200x500 table with the calling
loop compiled:

```
spl.ev(a, b)                       one point per call     693 ns
bispeu(a, b, tx, ty, c, kx, ky)    one point per call     498 ns
spl.ev(qx, qy)                     100000 points at once  315 ns per point
```

A batch call is the first answer. A loop that must evaluate one point at a
time is the case where `bispeu` is worth reaching for, and the knot and
coefficient arrays have to be bound to locals ABOVE the loop. numba does not
lift a jitclass field read out of a loop, so `spl.tx` written inside the loop
is re-read on every iteration and costs more than the object call it replaced.

```python
import numpy as np
from numba import njit
from scijit.interpolate import RectBivariateSpline, bispeu

x = np.linspace(0.0, 4.0, 40)
y = np.linspace(0.0, 3.0, 30)
z = np.sin(x[:, None]) * np.cos(y[None, :])
spl = RectBivariateSpline(x, y, z, None, 3, 3, 0.0, 20)

qx = np.linspace(0.5, 3.5, 200)
qy = np.linspace(0.5, 2.5, 200)

@njit
def batched(spl, qx, qy):
    return spl.ev(qx, qy).sum()

@njit
def one_at_a_time(spl, qx, qy):
    tx, ty, c, kx, ky = spl.tx, spl.ty, spl.c, spl.kx, spl.ky   # bound once
    a = np.empty(1, np.float64)
    b = np.empty(1, np.float64)
    s = 0.0
    for i in range(qx.size):
        a[0] = qx[i]
        b[0] = qy[i]
        s += bispeu(a, b, tx, ty, c, kx, ky)[0]
    return s

print(batched(spl, qx, qy))
print(one_at_a_time(spl, qx, qy))
```

```
37.73576110810349
37.73576110810349
```

Applies to `RectBivariateSpline`, `SmoothBivariateSpline`,
`RectSphereBivariateSpline` and `SmoothSphereBivariateSpline`, through
`bispeu` for scattered points and `bispev` for a grid. The univariate classes
have no such gap: `UnivariateSpline.ev` measures 378 ns against 350 ns for
`splev` on the same spline, so the object is the right spelling there in every
case.

## Flag arguments: scipy's spelling

`kind`, `method` and `bc_type` take scipy's string. Where scipy accepts only a
string, so does this package, and an integer is refused.

`interp1d`'s `kind` is the exception, because scipy accepts an integer there and
gives it a meaning: the spline order. This package reads it the same way, so
`interp1d(x, y, 3)` is a cubic in both, and `0`, `1`, `2`, `3` are `'zero'`,
`'slinear'`, `'quadratic'` and `'cubic'`. Orders above 3 have no string
spelling. A negative order raises.

`UnivariateSpline`'s `ext` also takes either, matching scipy, which documents
both `0` to `3` and the equivalent names.

## What silence about SciPy means

Where a routine behaves differently from `scipy.interpolate`, its docstring says
so under `Notes`, and the difference is named. Silence means the two were
compared and agreed.

Five public names have no `scipy.interpolate` counterpart at all, and each says
so in its own docstring: `bispev`, `bispeu`, `splder_ev`, and the two raw
FITPACK submodules `evaluators` and `fitters`.
</content>
</invoke>
