# scijit usage guide

Every example in these pages is runnable and tested. Read
[Callbacks](#callbacks) first: it covers how a routine here receives a
function, which is the one convention that differs from calling scipy.
A plain `@njit` function passed as an ordinary argument is the callback
spelling every routine here uses.

## Pages

- [interpolate](./interpolate.md): FITPACK splines, `BSpline`/
  `make_interp_spline` from arbitrary knots, `Akima1DInterpolator`
- [optimize](./optimize.md): roots, least squares, minimization, PRIMA,
  scalar root-finders and scalar minimizers, the prange-safe `fmin`/
  `fmin_powell`/`fmin_cg`/`fmin_bfgs`
- [integrate](./integrate.md): QUADPACK `quad`, LSODA `odeint`, `solve_ivp`,
  the nestable `nquad`/`dblquad`/`tplquad`, and the sampled-data
  quadrature routines (`trapezoid`, `simpson`, `romb`, …)

For every public name and its generated page see the
[API reference](../reference/index.md).

---

## Callbacks

A routine that takes a function takes a plain `@njit` function, passed as an
ordinary argument. `@njit` is numba's decorator; it compiles the function to
machine code on first call.

```python
import numpy as np
from numba import njit, prange
from scijit.optimize import minimize, fsolve

@njit
def fg(x, args):                     # value and gradient together
    r = x[0] - args[0]
    return r * r, np.array([2.0 * r])

minimize(fg, np.array([0.0]), args=np.array([3.0]))[0]      # array([3.])

@njit
def residual(x, args):               # returns the residual vector
    return np.array([x[0] * x[0] - args[0]])

fsolve(residual, np.array([1.0]), np.array([9.0]))          # array([3.])
```

`fsolve` and `minimize` call the callback as `(x, args)`, as shown. The
signature varies by routine and is on that routine's page. `args` is a float64
array of parameters, which the routine forwards to the callback unchanged.
Anything else the callback reads is closed over from the enclosing scope, and
numba freezes those values when the callback compiles.

### `prange` safety

`numba.prange` marks a loop for numba to run across threads, under
`@njit(parallel=True)`. The same call works inside one:

```python
@njit(parallel=True)
def solve_many(targets):
    out = np.zeros(targets.shape[0])
    for i in prange(targets.shape[0]):
        out[i] = fsolve(residual, np.array([1.0]), targets[i:i + 1])[0]
    return out

solve_many(np.array([4.0, 9.0, 16.0, 25.0]))       # array([2., 3., 4., 5.])
```

Every routine in the package is `prange`-safe, per routine in the
[compatibility page](../compatibility.md).

---

## Caching

No routine in this package is declared cacheable. `@njit(cache=True)` on a
function that calls one is either refused or silently ineffective, so the
first call in a process compiles. See the
[compatibility page](../compatibility.md) for what triggers it and what to do
instead.
