# optimize

`scipy.optimize` inside `@njit` code: scalar and multivariate root finding,
linear and nonlinear least squares, local and global minimization, constrained
minimization, and the assignment problem. The backends are MINPACK, L-BFGS-B,
SLSQP, the derivative-free PRIMA family, and pure numba ports of the solvers
scipy writes in Python.

## Passing a function

`@njit` is numba's decorator: it compiles a Python function to machine code the
first time it runs on given argument types. Every routine on this page that
takes a function takes a plain `@njit` function, passed as an ordinary argument.
See the [preamble](./index.md) for the shared conventions: the `@njit` callback,
`prange` safety, and caching.

Extra parameters ride in scipy's `args` tuple. The objective is `f(x)` under the
default `args=()`, and `f(x, a, b)` under `args=(a, b)`. The tuple is unpacked
into every call, and its elements may differ in type and shape.

## Differences from scipy

Where a routine behaves differently from `scipy.optimize`, its docstring says so
under `Notes` and names the difference. Silence means the two were compared and
agreed, on the same inputs, values and counters. A routine with no scipy
counterpart says so instead. The PRIMA solvers `NEWUOA`, `UOBYQA`, `BOBYQA` and
`LINCOA`, reached through `minimize`, are that case.

## Scalar root finding: `brentq`, `root_scalar`

`brentq` finds a root of a scalar function on a bracket `[a, b]` where the sign
of `f` changes. The objective takes one float and returns one float.

```python
from numba import njit
from scijit.optimize import brentq

@njit
def f(x):
    return x * x - 2.0

@njit
def run():
    return brentq(f, 0.0, 2.0)

run()          # -> 1.4142135623731364
```

`bisect`, `brenth`, `ridder` and `toms748` take the same call shape. `brentq` is
the general-purpose choice. `bisect` uses only the sign of `f` and is slowest.
`toms748` needs the fewest evaluations on a smooth `f`.

**The result object.** `full_output=True` returns `(x, RootResults)`.
`RootResults` is a namedtuple carrying scipy's fields:

    RootResults(root, iterations, function_calls, converged, flag, method)

Read a field by attribute (`res.root`), by index (`res[1]` is `iterations`), or
by unpacking.

```python
@njit
def run_full():
    return brentq(f, 0.0, 2.0, full_output=True)

x, res = run_full()
x, res.iterations, res.function_calls, res.converged, res.method
# -> (1.4142135623731364, 8, 9, True, 'brentq')
```

`full_output` selects the return shape. Inside `@njit` the return type is fixed
when the function compiles, so an argument that changes it must be known at that
point: write a literal like `True`, keep the default, or read a module-level
constant. A value that varies at run time raises `TypingError`, numba's
compile-time type error. From Python any value works. The same rule covers a
string that selects an algorithm, and whether a derivative argument is `None`,
both seen below.

**Derivatives select the method.** `newton` picks its algorithm from the
derivatives supplied. No `fprime` runs the secant method. `fprime` alone runs
Newton-Raphson. `fprime` with `fprime2` runs Halley's. The choice is settled
when the call compiles, so each spelling goes on its own line rather than
through a variable that might hold `None`. `.method` reports the choice.

```python
from scijit.optimize import newton

@njit
def g(x):
    return x**3 - 2.0*x - 5.0             # root near 2.0945514815

@njit
def gp(x):
    return 3.0*x**2 - 2.0

@njit
def gpp(x):
    return 6.0*x

@njit
def run_newton():
    a = newton(g, 2.0)                    # secant
    b = newton(g, 2.0, gp)                # Newton-Raphson
    c = newton(g, 2.0, gp, fprime2=gpp)   # Halley
    return a, b, c

run_newton()
# -> (2.094551481542326, 2.0945514815423265, 2.0945514815423265)
```

`fprime2` follows `args`, `tol` and `maxiter` in the signature, so it is passed
by keyword, as in scipy.

**The dispatcher.** `root_scalar` reaches all eight scalar root-finders behind
one `method` argument, or selects one from what was supplied. A `bracket` gives
`'brentq'`. An `x0` with both derivatives gives `'halley'`. The names are
`'brentq'`, `'brenth'`, `'bisect'`, `'ridder'`, `'toms748'`, `'secant'`,
`'newton'` and `'halley'`, matched case-insensitively.

```python
from scijit.optimize import root_scalar

@njit
def run_dispatch():
    r1 = root_scalar(g, bracket=(2.0, 3.0), method="toms748")
    r2 = root_scalar(g, x0=2.0, fprime=gp)
    return r1.root, r1.method, r2.root, r2.method

run_dispatch()
# -> (2.094551481542327, 'toms748', 2.0945514815423265, 'newton')
```

**A sweep.** `numba.prange` is numba's parallel loop: its iterations run across
threads. Because parameters ride in `args`, one compiled objective serves every
point, and the call is safe inside `prange`.

```python
import numpy as np
from numba import prange

@njit
def fc(x, c):
    return x * x - c

@njit(parallel=True)
def solve_many(cs):
    out = np.empty(cs.size)
    for i in prange(cs.size):
        out[i] = brentq(fc, 0.0, 10.0, (cs[i],))
    return out

solve_many(np.array([2.0, 3.0, 4.0]))
# -> array([1.41421356, 1.73205081, 2.        ])
```

**Fixed points.** `fixed_point` returns the value where `f(x) == x`.

```python
from scijit.optimize import fixed_point

@njit
def cos_map(x):
    return np.cos(x)

@njit
def run_fixed_point():
    return fixed_point(cos_map, 0.5)

run_fixed_point()
# -> 0.7390851332151607
```

## Multivariate root finding: `fsolve`, `root`

`fsolve` solves a square system. The system is a plain `@njit` function
returning the residual vector, and the residual count must equal the length of
`x0`.

```python
import numpy as np
from numba import njit
from scijit.optimize import fsolve

@njit
def system(x):
    return np.array([x[0] + 0.5 * (x[0] - x[1])**3 - 1.0,
                     0.5 * (x[1] - x[0])**3 + x[1]])

@njit
def run():
    return fsolve(system, np.array([0.0, 0.0]))

run()          # -> array([0.8411639, 0.1588361])
```

`root` reaches the same two MINPACK drivers behind a `method` argument: `'hybr'`
(the default, a modified Powell hybrid) and `'lm'` (Levenberg-Marquardt).

**The info fields.** `full_output=True` on `fsolve` returns
`(x, infodict, ier, mesg)`. `infodict` is a namedtuple, not a dict. Read a field
by attribute (`infodict.nfev`), by key (`infodict['nfev']`), or by unpacking.

```python
@njit
def run_full():
    return fsolve(system, np.array([0.0, 0.0]), full_output=True)

x, infodict, ier, mesg = run_full()
ier, infodict.nfev, mesg
# -> (1, 15, 'The solution converged.')
```

**An analytic Jacobian.** Supplying `fprime` selects the derivative driver. The
Jacobian is a second `@njit` function of the same signature returning the
`(n, n)` matrix.

```python
@njit
def jac(x):
    return np.array([[1.0 + 1.5 * (x[0] - x[1])**2,
                      -1.5 * (x[0] - x[1])**2],
                     [-1.5 * (x[1] - x[0])**2,
                      1.5 * (x[1] - x[0])**2 + 1.0]])

@njit
def run_jac():
    return fsolve(system, np.array([0.0, 0.0]), fprime=jac)

run_jac()      # -> array([0.8411639, 0.1588361])
```

**The result of `root`.** `root` returns a namedtuple reached by attribute.
`method`, and whether `jac` is `None`, must be compile-time constants inside
`@njit`, because both change the return type.

```python
from scijit.optimize import root

@njit
def run_root():
    res = root(system, np.array([0.0, 0.0]))
    return res.x, res.success

run_root()     # -> (array([0.8411639, 0.1588361]), True)
```

**A sweep.** `fsolve` is safe inside a `numba.prange` loop; concurrent solves do
not collide.

```python
from numba import prange

@njit
def fa(x, c):
    return np.array([x[0]**2 - c])

@njit(parallel=True)
def solve_many_roots(cs):
    out = np.empty(cs.size)
    for i in prange(cs.size):
        out[i] = fsolve(fa, np.array([1.0]), (cs[i],))[0]
    return out

solve_many_roots(np.array([2.0, 3.0, 4.0]))
# -> array([1.41421356, 1.73205081, 2.        ])
```

## Least squares: `leastsq`, `curve_fit`, `lsq_linear`

### Nonlinear least squares

`leastsq` minimizes the sum of squares of a residual vector, which may be longer
than the parameter vector. The residual is a plain `@njit` function. The data
arrays are built once and reach it through `args`, unpacked into every call, so
no array is rebuilt on an evaluation.

```python
import numpy as np
from numba import njit
from scijit.optimize import leastsq

T = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
Y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

@njit
def resid(p, t, y):
    return y - (p[0]*t + p[1])

@njit
def run():
    return leastsq(resid, np.array([0.0, 0.0]), args=(T, Y))

p, ier = run()
np.round(p, 8)         # -> array([2., 1.])
```

The bare call returns `(x, ier)`, where `ier` in `{1, 2, 3, 4}` means success.
`full_output=True` returns `(x, cov_x, infodict, mesg, ier)`. `cov_x` is the
unscaled `(JᵀJ)⁻¹`. It is `None` when `ier` is outside the success set or the
triangular factor is singular. Supplying `Dfun` selects the analytic-Jacobian
driver.

Each entry of `args` reaches the residual as its own parameter, so data arrays
travel unpacked and one compiled residual serves a `prange` sweep.

### Nonlinear curve fitting

`curve_fit` fits a model `f(x, p0, p1, ...)` to data and returns the fitted
parameters with their covariance. The model is a plain `@njit` function.

```python
from scijit.optimize import curve_fit

XDATA = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
YDATA = 2.5 * np.exp(-0.4 * XDATA)

@njit
def model(x, a, b):
    return a * np.exp(-b * x)

@njit
def run():
    return curve_fit(model, XDATA, YDATA, np.array([1.0, 0.1]))

popt, pcov = run()
np.round(popt, 8)      # -> array([2.5, 0.4])
```

Leaving `p0` out defaults it to `ones(n)`, with `n` read off the model's
signature. A model written `f(x, *params)` is also accepted, with `p0` required;
written-out parameters compile faster. `pcov` is MINPACK's own covariance scaled
by `SSR/(m−n)`, matching scipy's `absolute_sigma=False` default.

The model must be a plain `@njit` function, not a class instance with
`__call__`; a callable object is refused. `bounds`, `method` and `jac` are
accepted only at their defaults, since all three would select a solver this
package does not implement.

### Linear least squares

`nnls` solves `min ‖Ax − b‖` with `x ≥ 0` by the Lawson-Hanson algorithm. It
and `lsq_linear` take arrays only, hold no state, and run from a `numba.prange`
loop.

```python
from scijit.optimize import nnls

A = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
b = np.array([2.0, 1.0, -1.0])

@njit
def run_nnls():
    return nnls(A, b)

x, rnorm = run_nnls()          # x = [1.5, 0.0], rnorm = 1.224744871...
```

`lsq_linear` adds general bounds `lb ≤ x ≤ ub`, `±inf` allowed. Its `method` is
`'trf'` by default (trust-region reflective, iterating strictly inside the box).
`'bvls'` selects the active-set solver that reaches the bounds exactly. Inside
`@njit` the `method` string must be a literal, since the two report `nit` and
`active_mask` differently.

```python
from scijit.optimize import lsq_linear

lo = np.array([0.0, 0.0])
hi = np.array([1.5, 1.0])

@njit
def run_bounded():
    res  = lsq_linear(A, b, (lo, hi))         # method defaults to 'trf'
    bvls = lsq_linear(A, b, (lo, hi), 'bvls')
    return np.round(res.x, 6), res.status, bvls.x, bvls.nit

run_bounded()
# -> (array([1.499995, 0.      ]), 1, array([1.5, 0. ]), 2)
```

`nnls` returns `(x, rnorm)`. `lsq_linear` returns a result carrying scipy's ten
fields as attributes: `x`, `fun`, `cost`, `optimality`, `active_mask`,
`unbounded_sol`, `nit`, `status`, `message` and `success`.

## Local minimization: `minimize`

### The unified interface: `minimize`

`minimize` reaches eleven algorithms behind one `method` string, with scipy's
signature. The objective is a plain `@njit` function of `(x, *args)`. It returns
either the value alone, or the value and its gradient together as `(f, g)`. The
return shape is read while the call compiles, so neither spelling needs a flag.

```python
import numpy as np
from numba import njit
from scijit.optimize import minimize

@njit
def rosen(x):
    return 100.0*(x[1]-x[0]**2)**2 + (1.0-x[0])**2

@njit
def run():
    return minimize(rosen, np.array([-1.2, 1.0]), method='BFGS')

res = run()
res.x          # -> array([0.9999955 , 0.99999099])
res.success    # -> True
```

`method` left unset picks `'L-BFGS-B'` with bounds and `'BFGS'` otherwise.
`minimize` takes no constraints, so scipy's `'SLSQP'`-with-constraints default
does not apply here. Inside `@njit` a given `method` must be a literal at the
call site, because each method returns its own field set.

**A separate gradient.** Returning the value and gradient together saves the
gradient methods a second pass over the same arithmetic. The gradient may
instead be a second function, passed as scipy's `jac`.

```python
@njit
def rosen_fg(x):
    f = 100.0*(x[1]-x[0]**2)**2 + (1.0-x[0])**2
    g = np.array([-400.0*x[0]*(x[1]-x[0]**2) - 2.0*(1.0-x[0]),
                  200.0*(x[1]-x[0]**2)])
    return f, g

@njit
def grad(x):
    return np.array([-400.0*x[0]*(x[1]-x[0]**2) - 2.0*(1.0-x[0]),
                     200.0*(x[1]-x[0]**2)])

@njit
def run_grad():
    a = minimize(rosen_fg, np.array([-1.2, 1.0]), method='L-BFGS-B').x
    b = minimize(rosen, np.array([-1.2, 1.0]), method='CG', jac=grad).x
    return a, b

run_grad()
# -> (array([0.99999895, 0.99999802]), array([1.00000001, 1.        ]))
```

**Result fields differ per method.** `x` is the minimizer and `fun` its value.
Every method also carries `nfev`, `status`, `message` and `success`. The rest is
the method's own, and a field the method did not compute is absent. `res.keys()`
lists what a given result holds. Reading an absent field raises `AttributeError`
from Python and `TypingError` inside `@njit`.

| method        | also carries                                |
|---------------|---------------------------------------------|
| 'Nelder-Mead' | `nit`, `final_simplex`                      |
| 'Powell'      | `nit`, `direc`                              |
| 'CG'          | `nit`, `jac`, `njev`                        |
| 'BFGS'        | `nit`, `jac`, `njev`, `hess_inv`            |
| 'L-BFGS-B'    | `nit`, `jac`, `njev`, `hess_inv`            |
| 'SLSQP'       | `nit`, `jac`, `njev`, `multipliers`         |
| 'COBYLA'      | `maxcv`                                      |
| the PRIMA four| nothing further                             |

**The inverse-Hessian operator.** `'BFGS'` and `'L-BFGS-B'` results carry
`hess_inv`, a `HessInv` operator. Apply it with `op(v)`, `op.matvec(v)` or
`op.dot(v)`, and materialise it with `op.todense()`. The other nine methods
carry an empty operator that applies as the identity, with `op.has_hess()`
False; test that rather than the values.

```python
@njit
def hess():
    op = minimize(rosen, np.array([-1.2, 1.0]), method='BFGS').hess_inv
    return op.has_hess(), op.shape, op.todense().shape

hess()         # -> (True, (2, 2), (2, 2))
```

**Bounds** are one `(min, max)` pair per variable, an `(n, 2)` array.
`'L-BFGS-B'`, `'SLSQP'`, `'COBYLA'`, `'BOBYQA'` and `'LINCOA'` use them. The
other six warn and ignore.

```python
@njit
def run_bounded():
    return minimize(rosen, np.array([0.0, 0.0]), method='L-BFGS-B',
                    bounds=np.array([[-2.0, 0.5], [-2.0, 0.5]]))

run_bounded().x    # -> array([0.5 , 0.25])
```

The eleven methods and what each uses:

| `method`        | gradient | bounds | prange | backend    |
|-----------------|----------|--------|--------|------------|
| `'Nelder-Mead'` | ignored  | no     | yes    | pure port  |
| `'Powell'`      | ignored  | no     | yes    | pure port  |
| `'CG'`          | uses     | no     | yes    | pure port  |
| `'BFGS'`        | uses     | no     | yes    | pure port  |
| `'L-BFGS-B'`    | uses     | yes    | yes    | Fortran    |
| `'SLSQP'`       | uses     | yes    | yes    | Fortran    |
| `'COBYLA'`      | ignored  | yes    | yes    | Fortran    |
| `'UOBYQA'`      | ignored  | no     | yes    | Fortran    |
| `'NEWUOA'`      | ignored  | no     | yes    | Fortran    |
| `'BOBYQA'`      | ignored  | yes    | yes    | Fortran    |
| `'LINCOA'`      | ignored  | yes    | yes    | Fortran    |

The last four are Powell solvers scipy does not expose. scipy 1.18's own
`'COBYLA'` is PRIMA, the library behind this one, so the two run the same
implementation. `'Newton-CG'`, `'dogleg'`, `'trust-ncg'`, `'trust-exact'`,
`'trust-constr'`, `'TNC'`, `'trust-krylov'` and `'COBYQA'` are not available.
Constraints do not go through `minimize`; they reach their solvers through
`fmin_cobyla` and `fmin_slsqp`.

### The standalone minimizers

`fmin` (Nelder-Mead simplex) and `fmin_powell` (direction-set) are
derivative-free. `fmin_cg` (conjugate gradient) and `fmin_bfgs` (quasi-Newton)
use a gradient, given as `fprime` or forward-differenced when `fprime=None`. All
four are pure numba ports with no compiled backend and no module state, so they
run in a `numba.prange` loop.

```python
from scijit.optimize import fmin, fmin_powell, fmin_cg, fmin_bfgs

@njit
def rosen_grad(x):
    g = np.empty(2)
    g[0] = -400.0*x[0]*(x[1]-x[0]**2) - 2.0*(1.0-x[0])
    g[1] = 200.0*(x[1]-x[0]**2)
    return g

@njit
def run_four():
    x0 = np.array([-1.2, 1.0])
    a  = ()
    xn = fmin(rosen, x0, a, disp=0)                    # Nelder-Mead
    xp = fmin_powell(rosen, x0, a, disp=0)             # Powell
    xc = fmin_cg(rosen, x0, rosen_grad, a, disp=0)     # gradient is 3rd arg
    xb = fmin_bfgs(rosen, x0, rosen_grad, a, disp=0)
    return xn, xp, xc, xb

run_four()
# -> (array([1.00002202, 1.00004222]), array([1., 1.]),
#     array([1.00000001, 1.        ]), array([0.99999997, 0.99999995]))
```

The gradient is the third argument on `fmin_cg` and `fmin_bfgs`, after `x0`.
`disp=1` (the default) prints a summary and warns on a limit path, both of which
force serial execution, so `disp=0` is the path a `prange` loop wants.
`full_output` and `retall` select the return shape, so inside `@njit` they must
be literals. Under `full_output=1` each returns scipy's tuple, whose elements
carry scipy's shapes: `fmin_bfgs`'s inverse-Hessian estimate and `fmin_powell`'s
direction set among them.

`maxiter` and `maxfun` at `None` (default) mean the method's own cap: `N*200`
for `fmin`, `N*1000` iterations and evaluations for `fmin_powell`, `N*200` for
`fmin_cg` and `fmin_bfgs`. `warnflag` is `0` for success, `1` for a function or
iteration cap, `2` for a line-search failure, `3` for a nan.

**A sweep.** Parameters ride in `args`, so one compiled objective serves the
whole loop.

```python
from numba import prange

@njit
def quad(x, tx, ty):
    return (x[0]-tx)**2 + (x[1]-ty)**2

@njit
def quad_grad(x, tx, ty):
    g = np.empty(2)
    g[0] = 2.0*(x[0]-tx)
    g[1] = 2.0*(x[1]-ty)
    return g

@njit(parallel=True)
def many(targets):                        # targets (n, 2)
    out = np.empty_like(targets)
    for i in prange(targets.shape[0]):
        out[i] = fmin_bfgs(quad, np.zeros(2), quad_grad,
                           (targets[i, 0], targets[i, 1]), disp=0)
    return out

many(np.array([[1.0, 2.0], [-3.0, 0.5]]))
# -> array([[ 1. ,  2. ], [-3. ,  0.5]])
```

### Bounds-only minimization: `fmin_l_bfgs_b`

`fmin_l_bfgs_b` minimizes subject to simple box bounds. It stores a limited
number of past gradient differences instead of a full Hessian, which is what
keeps it usable at large `n`. With no separate gradient the objective returns
the value and the gradient as a tuple. With `approx_grad=1` or an `fprime` it
returns the value alone.

```python
from scijit.optimize import fmin_l_bfgs_b

@njit
def fg(x):
    f = (x[0] - 1.0)**2 + (x[1] - 2.5)**2
    g = np.array([2.0*(x[0] - 1.0), 2.0*(x[1] - 2.5)])
    return f, g

@njit
def run_lbfgsb():
    x, f, d = fmin_l_bfgs_b(fg, np.array([0.0, 0.0]))
    return x, f, d.warnflag

run_lbfgsb()   # -> (array([1. , 2.5]), 0.0, 0)
```

`approx_grad` and whether `fprime` is present are compile-time constants: they
select which callback protocol is compiled. Bounds are one `(min, max)` pair per
variable, `None` (the default) for unconstrained. `None` inside a pair means no
bound in that direction, the same as `±np.inf`.

## Constrained minimization: `fmin_slsqp`, `fmin_cobyla`

### Equality and inequality constraints: `fmin_slsqp`

The objective and each constraint are plain `@njit` functions of `(x, *args)`.
`f_eqcons` returns all equality constraints at once (`== 0`). `f_ieqcons`
returns the inequalities (`>= 0`).

```python
from scijit.optimize import fmin_slsqp

@njit
def obj(x):                               # minimize x0^2 + x1^2
    return x[0]**2 + x[1]**2

@njit
def eqc(x):                               # subject to x0 + x1 == 1
    c = np.empty(1)
    c[0] = x[0] + x[1] - 1.0
    return c

@njit
def run():
    return fmin_slsqp(obj, np.array([2.0, 0.0]), f_eqcons=eqc, disp=0)

run()                                     # -> array([0.5, 0.5])
```

`full_output=True` adds the objective value, the major-iteration count, the exit
mode and its message:

```python
@njit
def run_full():
    return fmin_slsqp(obj, np.array([2.0, 0.0]), f_eqcons=eqc,
                      full_output=True, disp=0)

run_full()
# -> (array([0.5, 0.5]), 0.5, 4, 0, 'Optimization terminated successfully')
```

Equality and inequality constraints have two spellings. The first is the
vector-valued `f_eqcons`/`f_ieqcons` above. The second is a tuple of
one-per-entry `@njit` functions `g(x, *args)`, passed as `eqcons`/`ieqcons`. An
entry returns one value or an array, contributing one constraint per element.
Given both, the tuple entries are appended ahead of the vector function's.
Inside `@njit` the tuple is a compile-time constant.

```python
@njit
def g1(x):                                # one scalar constraint
    return x[0] + x[1] - 1.0

@njit
def run_tuple():
    return fmin_slsqp(obj, np.array([2.0, 0.0]), eqcons=(g1,), disp=0)

run_tuple()                               # -> array([0.5, 0.5])
```

`bounds` is scipy's `(n, 2)` array. `fprime` is an explicit gradient; left out,
the gradient comes from forward differences.

**The `disp` flag.** It defaults to 1, scipy's default, and writes a convergence
report per call, which forces serial execution. A `prange` loop with `disp` set
returns the right answer but stops running in parallel. `disp=0` silences it and
keeps the loop parallel. Applies to `fmin`, `fmin_powell`, `fmin_cg`,
`fmin_bfgs` and `fmin_slsqp`.

### Nonlinear constraints: `fmin_cobyla`

`fmin_cobyla` takes plain `@njit` functions, both called splatted: `f(x, *args)`
and `c(x, *consargs)`. `cons` returns the whole constraint vector at once,
`c >= 0` feasible, or a sequence of functions returning one value each. The
number of constraints is read from `cons` at `x0`.

```python
from scijit.optimize import fmin_cobyla

@njit
def cobj(x):
    return (x[0]-1.0)**2 + (x[1]-2.5)**2

@njit
def ccon(x):                              # >= 0 feasible
    c = np.empty(1)
    c[0] = x[0] - 2.0*x[1] + 2.0
    return c

@njit
def run():
    return fmin_cobyla(cobj, np.array([2.0, 0.0]), ccon)

run()                                     # -> array([1.39999392, 1.69999696])
```

`full_output=True` returns a `CobylaResult` namedtuple with the fields `x`,
`fun`, `maxcv`, `message`, `nfev`, `status` and `success`. The flag selects the
return shape and must be a literal inside `@njit`. `callback` is accepted only as
`None`. `Aineq`, `bineq`, `Aeq`, `beq`, `lower` and `upper` add linear
constraints and simple bounds, which PRIMA takes natively.

### Derivative-free minimization: the PRIMA solvers through `minimize`

`minimize` reaches Powell's four PRIMA solvers by name, taking the same plain
`@njit` objective. `'NEWUOA'` and `'UOBYQA'` are unconstrained, `'BOBYQA'`
bounded, `'LINCOA'` linearly constrained. `tol` sets PRIMA's `rhoend`, 1e-6 when
not given. These four have no `scipy.optimize` counterpart.

```python
@njit
def objective(x, a, b):
    return (x[0]-a)**2 + (x[1]-b)**2

@njit
def run_prima():
    return minimize(objective, np.array([0.0, 0.0]),
                    args=(3.0, -2.0), method='NEWUOA')

r = run_prima()
r.x, r.fun                                # -> (array([ 3., -2.]), 0.0)
```

Which minimizer to reach for:

| the problem                                | use                                          |
|--------------------------------------------|----------------------------------------------|
| a cheap gradient                           | `minimize` (L-BFGS-B), fastest               |
| a cheap gradient, no compiled backend      | `fmin_bfgs` / `fmin_cg`                       |
| bounds + gradient + equality constraints   | `fmin_slsqp`                                  |
| no gradient, unconstrained/bounds/linear   | `minimize` (NEWUOA, UOBYQA, BOBYQA, LINCOA)  |
| no gradient, no compiled backend           | `fmin` / `fmin_powell`                        |
| no gradient, nonlinear constraints         | `fmin_cobyla`                                 |

## Scalar minimization: `minimize_scalar`

`minimize_scalar` minimizes a scalar function of one float and returns scipy's
`MinimizeScalarResult`. `fminbound`, `brent` and `golden` return the minimiser
as a bare float, which is scipy's shape for those three. The objective is a plain
`@njit` function of one float.

```python
from scijit.optimize import minimize_scalar, fminbound, brent, golden

@njit
def q(x):
    return (x - 1.5)**2 + 0.5             # minimum at x = 1.5

@njit
def run():
    ra = minimize_scalar(q, bracket=(0.0, 3.0))       # method='brent'
    rb = minimize_scalar(q, bounds=(0.0, 3.0), method='bounded')
    rc = fminbound(q, -2.0, 5.0)
    rd = brent(q)
    re = golden(q)
    return ra.x, rb.x, rc, rd, re

run()
# -> (1.5000000000000004, 1.5, 1.5, 1.5, 1.5000000053299778)
```

`minimize_scalar`'s `method` takes scipy's name: `'brent'`, `'golden'` or
`'bounded'`. `method=None`, the default, is `'brent'`, or `'bounded'` when
`bounds` is given. Inside `@njit` the string must be a literal, because
`'bounded'` carries a `status` field the other two do not. `bracket` holds seed
points for the downhill search, or three points used directly. `bounds` belongs
to `'bounded'` alone.

`brent` and `golden` take their seed points as scipy's `brack`, a tuple of two
or three. `None` starts the search from `(0.0, 1.0)`. Inside `@njit` it must be a
tuple or `None`; a list raises `TypingError`.

**Bracket search: `bracket`.** `bracket` is the downhill search that `brent` and
`golden` run first, exposed under its own name. It returns the three points, the
objective at each, and the evaluation count.

```python
from scijit.optimize import bracket

@njit
def f(x):
    return 10.0 * x**2 + 3.0 * x + 5.0

@njit
def run_bracket():
    return bracket(f, 0.1, 1.0)

run_bracket()
# -> (1.0, 0.1, -1.3562306, 18.0, 5.4, 19.3249226037636, 3)
```

The three points are strictly ordered and the middle value is below both ends,
which is what makes a minimum lie inside them. No valid bracket raises
`RuntimeError`.

## Global optimization: `differential_evolution`, `basinhopping`, `brute`

For all three the objective is a plain `@njit` function `func(x, *args)`, with
`x` a 1-D float64 array. They hold no module state and carry their random-number
state per call, so each runs from a `numba.prange` loop.

**Brute-force grid search.** `brute` evaluates the objective on a full Cartesian
grid and returns the best point. It uses no randomness, so the same call gives
the same answer.

```python
import numpy as np
from numba import njit
from scijit.optimize import brute, fmin

@njit
def q(x):
    return (x[0] - 1.5)**2 + (x[1] + 0.5)**2

ranges = np.array([[-3.0, 3.0], [-3.0, 3.0]])

@njit
def run_brute():
    return brute(q, ranges, Ns=20, finish=fmin)

run_brute()                               # -> array([ 1.49996485, -0.50001094])
```

`ranges` is one `(min, max)` pair per axis. `Ns` (default 20) sets the point
count per axis. `finish` is `None` (no polish) or `fmin` (the default
Nelder-Mead polish), named at the call site because it is resolved when the call
compiles. With `full_output=True` inside `@njit`, `grid` comes back as `(M, N)`
and `Jout` as `(M,)` in C order, because a compiled function cannot return an
array whose ndim follows a runtime value. Recover scipy's layout with
`grid.T.reshape((N,) + (Ns,)*N)` and `Jout.reshape((Ns,)*N)`.

**Basin-hopping.** `basinhopping` alternates a random displacement with a local
minimization, keeping or rejecting each new basin by a Metropolis test.

```python
from scijit.optimize import basinhopping

@njit
def run_hop():
    return basinhopping(q, np.array([2.0, 2.0]), niter=60, seed=1,
                        minimizer_kwargs={'method': 'BFGS'})

res = run_hop()
res.x                                     # -> array([1.5, -0.5])
```

The local minimizer is set through `minimizer_kwargs`, carrying `'method'`
(`'Nelder-Mead'`, `'Powell'`, `'BFGS'` the default, or `'CG'`) and `'jac'` (a
plain `@njit` gradient). Inside `@njit` write that dict as a literal at the call
site, not a variable, since only a literal can carry the callable `'jac'`. The
result is `(x, fun, nit, nfev, njev, minimization_failures, success, message)`,
with `message` a list holding one string.

**Differential evolution.** `differential_evolution` runs a population-based
stochastic search over a box.

```python
from scijit.optimize import differential_evolution

@njit
def qt(x, target):
    return (x[0] - target[0])**2 + (x[1] - target[1])**2

bounds = np.array([[-3.0, 3.0], [-3.0, 3.0]])

@njit
def run_de():
    return differential_evolution(qt, bounds,
                                  (np.array([1.5, -0.5]),),   # args
                                  'best1bin', seed=3)         # strategy

res = run_de()
res.x, res.success                        # -> (array([1.5, -0.5]), True)
```

`strategy` and `init` take scipy's names, `'best1bin'` and `'latinhypercube'`
the defaults. `polish` (default `True`) runs an L-BFGS-B polish over the same
box. The polished point replaces the best member only when it improves the
objective, the polish converged, and it lies inside the bounds.
`res.population` and `res.population_energies` carry the final population.

**The random-number state.** `basinhopping` and `differential_evolution` take
scipy's `rng`: `None`, an integer, or a `numpy.random.Generator`. A bare call
draws fresh entropy, so two identical bare calls give different answers.
`rng=None` with the additive integer `seed` selects an internal generator whose
state is per-call, which is the spelling to use inside a `prange` loop: one
`Generator` shared across threads is not thread-safe. An integer `rng` builds a
`numpy.random.default_rng`, as scipy 1.18 does, so an integer seed is
reproducible.

## Assignment: `linear_sum_assignment`

`linear_sum_assignment` solves the rectangular assignment problem: pick one
entry from each row and each column of a cost matrix so the total is minimal.

```python
import numpy as np
from numba import njit
from scijit.optimize import linear_sum_assignment

cost = np.array([[4.0, 1.0, 3.0],
                 [2.0, 0.0, 5.0],
                 [3.0, 2.0, 2.0]])

@njit
def assign():
    return linear_sum_assignment(cost)

row_ind, col_ind = assign()   # -> (array([0, 1, 2]), array([1, 0, 2]))
```

`row_ind` is sorted ascending, and both outputs are `int64` of length
`min(nr, nc)`. numba does not index one array with two index arrays, so sum the
chosen entries in a loop rather than `cost[row_ind, col_ind].sum()`:

```python
@njit
def total():
    r, c = linear_sum_assignment(cost)
    s = 0.0
    for k in range(r.size):
        s += cost[r[k], c[k]]
    return s

total()   # -> 5.0
```

Pass `maximize=True` to maximize the total. The cost matrix is 2-D of any
numeric dtype; a rank other than 2 is refused while the call compiles.
Rectangular matrices are allowed. On a degenerate problem it may return a
different but equally optimal assignment.
