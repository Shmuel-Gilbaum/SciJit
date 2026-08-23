# integrate

`scipy.integrate` inside `@njit` code: adaptive and fixed-order quadrature,
multiple integrals, the sampled-data quadrature routines, and ODE solving by
LSODA and explicit Runge-Kutta. See the [preamble](./index.md) for the shared
conventions: the `@njit` callback, `prange` safety, and caching.

## Passing a function

Every routine here takes a plain `@njit` function, passed as an ordinary
argument. `@njit` is numba's decorator; it compiles a function to machine code
so it runs at full speed inside the routine. Each routine's own section gives
the shape of the function it expects.

The sampled-data routines are the exception: they integrate arrays and take no
function. Those are `trapezoid`, `simpson`, `romb`, `cumulative_trapezoid`,
`cumulative_simpson` and `newton_cotes`.

## Adaptive integration: `quad`

`quad` computes a definite integral over an interval. The integrand returns
its value and is called as `f(x, *args)`.

```python
import numpy as np
from numba import njit
from scijit.integrate import quad

@njit
def f(x, c):
    return c * np.exp(-x*x)

@njit
def run():
    val, err = quad(f, -5.0, 5.0, (1.0,))
    return val                     # -> sqrt(pi) = 1.7724538509027912
```

`args` carries fixed parameters to the integrand. It is scipy's tuple. A value
that is not a tuple is read as a one-item tuple, as scipy reads it. Each entry
is a real number or an array of real numbers.

An infinite limit is written `np.inf`, as in scipy:

```python
@njit
def run_inf():
    val, err = quad(f, -np.inf, np.inf, (1.0,))
    return val                     # -> 1.7724538509055159
```

Interior singularities or known break points go in `points`:

```python
@njit
def run_pts():
    val, err = quad(f, 0.0, 1.0, (1.0,), points=np.array([0.3]))
    return val                     # -> 0.746824132812427
```

`weight` folds a weight function into the integrand. The choices are `'cos'`,
`'sin'`, `'alg'`, `'alg-loga'`, `'alg-logb'`, `'alg-log'` and `'cauchy'`, with
the weight parameter in `wvar`:

```python
@njit
def run_wt():
    val, err = quad(f, 0.0, 1.0, (1.0,), weight='cos', wvar=1.0)
    return val                     # -> 0.6561743627315068
```

`complex_func=True` integrates a complex-valued integrand by integrating its
real and imaginary parts separately, as scipy does:

```python
@njit
def fc(x):
    return np.exp(1j * x)

@njit
def run_cf():
    val, err = quad(fc, 0.0, np.pi, complex_func=True)
    return val                     # -> 2j (up to rounding)
```

`full_output=True` adds a third return value, scipy's `infodict`. The flag has
to be a literal `True`, not a variable: it changes how many values the call
returns, and the number of return values is fixed when the function compiles.

```python
@njit
def run_full():
    val, err, info = quad(f, 0.0, 1.0, (1.0,), True)
    return val, info['neval'][0], info['ier'][0]   # ier == 0 means converged
```

The `infodict` carries `neval`, `last`, `alist`, `blist`, `rlist`, `elist` and
`iord`, plus the break-point and oscillatory keys scipy adds on those routes,
and `neval`, `lst`, `rslst`, `erlst` and `ierlst` on the cos/sin route to
infinity. One extra key, `ier`, has no scipy counterpart. Every value is a 1-D
float64 array, so `info['neval']` reads `array([21.])` where scipy gives `21`.
The per-subinterval arrays have length `limit` and only `[:last]` is filled;
the cycle arrays on the cos/sin route have length `limlst` and only `[:lst]` is
filled.

`quad` nests to any depth and is safe under `numba.prange`, the parallel loop
described in the [preamble](./index.md). An inner call cannot lose the outer
one, and parallel calls do not interfere. Integration over several variables is
that nesting, exposed as
[`nquad`](#multi-dimensional-integration-nquad-dblquad-tplquad).

## Multi-dimensional integration: `nquad`, `dblquad`, `tplquad`

`nquad` integrates over several axes by nesting `quad`. It takes one range per
axis, and its depth is the length of `ranges`. `ranges[0]` is the innermost
integral, as in scipy.

The integrand receives its coordinates as separate arguments, innermost first,
then the `args` entries: `func(x0, ..., *args)`. That is scipy's convention.

```python
import numpy as np
from numba import njit
from scijit.integrate import nquad

@njit
def f(x0, x1):                          # x0 innermost
    return x0 * x1

@njit
def run():
    return nquad(f, ((0.0, 1.0), (0.0, 2.0)))

run()          # -> (0.9999999999999999, 1.1102230246251564e-14)
```

An inner limit may depend on the axes outside it. scipy accepts two spellings
and so does this: a `(lo, hi)` pair whose members are callbacks, or one
callable returning both.

```python
@njit
def hi0(x1):                            # axis 0's upper limit
    return 1.0 - x1

@njit
def rng0(x1):                           # the same, as one callable
    return 0.0, 1.0 - x1

@njit
def tri():
    a = nquad(f, ((0.0, hi0), (0.0, 1.0)))
    b = nquad(f, (rng0, (0.0, 1.0)))
    return a[0], b[0]

tri()          # -> (0.04166666666666667, 0.04166666666666667)
```

A limit callback for axis `i` receives the already-fixed outer coordinates
`x_{i+1} ... x_{n-1}`, innermost of those first, then the `args` entries. That
is scipy's order.

`args` is scipy's tuple, forwarded to the integrand and the range callbacks
unpacked, one parameter each. Inside `@njit` write it as a tuple: an array
cannot be unpacked, because its type carries no length and the length is what
says how many parameters the integrand takes.

Per-axis options go in `opts`, innermost first, carrying `epsabs`, `epsrel`,
`limit`, `points`, `weight`, `wvar`, `maxp1` and `limlst`. One dict applies to
every axis; a tuple of dicts gives one per axis. From inside `@njit`:

- A single mixed-kind dict such as `{'weight': 'cos', 'wvar': 1.0}` works and
  applies to every axis. `weight` and `points` select the QUADPACK routine and
  are read when the call compiles, so a dict setting `weight` must also hold a
  number: `{'weight': 'cos'}` alone does not type.
- A dict of numbers only is read at run time and needs nothing special. It
  cannot hold a tuple or array value.
- A per-axis tuple of dicts carrying `weight` or `points` raises `TypingError`.
- An empty `{}` does not type. Pass `opts=None` instead.

Calling `nquad` from Python takes every spelling.

`full_output=True` adds a third value, scipy's `out_dict`, read as
`info['neval']`. Only the innermost integration is counted, which is scipy's
rule, and `abserr` is the largest estimate over every level, which is also
scipy's.

`dblquad` and `tplquad` are `nquad` at depth two and three, which is how scipy
builds them too. The integrand's variables run innermost first, `func(y, x)`
and `func(z, y, x)`, followed by any `args`.

```python
from scijit.integrate import dblquad

@njit
def g(y, x):                            # inner variable FIRST
    return x * y * y

@njit
def hfun(x):
    return 1.0 - x

@njit
def area():
    return dblquad(g, 0.0, 1.0, 0.0, hfun)[0]

area()         # -> 0.016666666666666666
```

The limit callbacks take their coordinates alone, `gfun(x)` for `dblquad` and
`qfun(x, y)`/`rfun(x, y)` for `tplquad`, because that is what scipy passes
them; a form taking the `args` entries after the coordinates is accepted as
well. Each inner limit is a constant or a callback, in any combination, and
either may be infinite.

`tplquad`'s `qfun` and `rfun` take their variables outermost first, the
opposite order from the integrand. That inconsistency is scipy's, reproduced so
calls port across unchanged.

## Fixed-order Gauss quadrature: `fixed_quad`

`fixed_quad` integrates by fixed-order Gauss-Legendre quadrature and returns
`(val, None)`, as scipy does. The integrand is a vector function: it maps a 1-D
array of abscissae to the array of values. `args` is a tuple splatted into the
integrand, so a parameterised integrand takes its parameters as ordinary
arguments.

```python
import numpy as np
from numba import njit
from scijit.integrate import fixed_quad

@njit
def integrand(t):                         # array -> array
    return np.cos(t)

@njit
def scaled(t, k):                         # parameters arrive splatted
    return np.cos(k * t)

@njit
def run():
    val, _ = fixed_quad(integrand, 0.0, np.pi/2, (), 5)   # int cos over [0, pi/2] -> ~1.0
    par, _ = fixed_quad(scaled, 0.0, np.pi/2, (2.0,), 5)  # int cos(2t) over [0, pi/2] -> ~0.0
    return val, par

run()          # -> (1.0000000000395655, 1.743934249004316e-16)
```

Inside `@njit` the `args` tuple has to be a tuple: a list or an array has no
compiled splat. `fixed_quad(cos, 0, pi/2, (), 5)` matched
`scipy.integrate.fixed_quad` to `4.4e-16`.

## Sampled-data quadrature: `trapezoid`, `simpson`, `romb`, `cumulative_*`

These routines integrate a sampled array rather than a function. They take no
callback and are safe to call from a `numba.prange` loop. `y` may be of any
rank, and `axis` selects the axis to integrate along, as in scipy. An integer
or boolean sample array is promoted to float64; a complex one stays complex.

```python
import numpy as np
from numba import njit
from scijit.integrate import (trapezoid, cumulative_trapezoid,
                              simpson, romb, cumulative_simpson)

x = np.linspace(0.0, np.pi, 33)
y = np.sin(x)

@njit
def integrate_samples(x, y):
    total = trapezoid(y, x)                       # composite trapezoid -> ~2.0
    cum   = cumulative_trapezoid(y, x, 1.0, -1, 0.0)     # running integral from 0.0
    return total, cum

@njit
def higher_order(x, y):
    total = simpson(y, x)                         # composite Simpson -> ~2.0
    cum   = cumulative_simpson(y, x=x, initial=0.0)
    return total, cum

@njit
def along_an_axis(Y):
    return simpson(Y, axis=0)                     # one value per column

@njit
def uniform_spacing(y):
    return (trapezoid(y, None, 0.5),              # x=None -> equal spacing dx
            cumulative_trapezoid(y, None, 0.5))   # length n-1, scipy's initial=None
```

`romb` needs `2**k + 1` equally spaced samples and the scalar spacing `dx`:

```python
@njit
def romberg(y, dx):
    return romb(y, dx)                            # -> 2.0000000000013216
```

`trapezoid`, `simpson` and `romb` are bit-identical to their scipy
counterparts on 65 samples of `sin` over `[0, pi]`, with `x` and with `dx`, and
`cumulative_trapezoid` and `cumulative_simpson` are bit-identical with and
without `initial`. The same holds at every axis of a rank-3 input.

Two behaviours differ from scipy. `cumulative_trapezoid` accepts a non-zero
`initial`, prepending it and adding it to every element, the convention
`cumulative_simpson` already uses; scipy raises ``ValueError: `initial` must be
`None` or `0`.`` there. `romb`'s `show` must be a literal constant inside
`@njit`, since it selects whether the Richardson table is built at all.

`trapezoid` duplicates what `np.trapz` already does inside `@njit`. It ships
anyway for parity, so `scipy.integrate.trapezoid` code ports across unchanged.

## Newton-Cotes weights: `newton_cotes`

`newton_cotes` returns the weights `an` and the error coefficient `B` for an
N-point rule. `rn` is either the order, as an int or a float, or the `N + 1`
sample positions as a 1-D array. `equal` forces equal spacing and replaces the
positions with `0..N`.

```python
import numpy as np
from numba import njit
from scijit.integrate import newton_cotes

@njit
def run():
    an, B = newton_cotes(4, 0)            # N = 4 -> Boole's rule
    pn, PB = newton_cotes(np.arange(5.0), 0)   # the same rule from positions
    return an, B, pn, PB
```

For `N = 1..14` the weights come from scipy's exact-rational table. Larger `N`
solves the Vandermonde system instead, which is ill-conditioned at high order.

High-order Newton-Cotes weights alternate in sign and grow, so a single rule
above about order 8 amplifies sample error. A composite low-order rule, `romb`
or `quad` is the accurate route on a fine grid.

## Initial value problems: `solve_ivp`

`solve_ivp` integrates an ODE system from a start state over a time span. The
right-hand side is a plain `@njit` `f(t, y)` returning the derivatives, with
`t` first as in scipy. `method` is a string.

```python
import numpy as np
from numba import njit
from scijit.integrate import solve_ivp

@njit
def rhs(t, y):
    dy = np.empty(2)
    dy[0] = y[1]                   # harmonic oscillator y'' = -4 y
    dy[1] = -4.0 * y[0]
    return dy

@njit
def run():
    res = solve_ivp(rhs, (0.0, 10.0), np.array([1.0, 0.0]), 'RK45',
                    np.linspace(0.0, 10.0, 200))
    return res.y                   # (2, 200): component-major, as in scipy
```

`res.y` has shape `(n_states, n_times)`, scipy's orientation. The result is a
namedtuple with scipy's field names, so `res.y`, `res.t`, `res.success`,
`res.nfev` and `res.message` all work. Field access by string does not:
`res['y']` is not valid on a namedtuple.

`'RK45'`, `'RK23'` and `'DOP853'` are pure-`@njit` ports of scipy's adaptive
explicit Runge-Kutta solvers. They hold no state anywhere, so they are the
methods to reach for under `prange`. `'LSODA'` reaches Fortran and switches
automatically between Adams and BDF, so it is the method for a stiff problem.
Every method integrates backwards in time, and every method supports
`dense_output=True`. `'Radau'` and `'BDF'` raise; both are pure Python in
scipy, so both are planned ports rather than blocked.

`scipy.integrate.RK45` is a stepper class advanced one step at a time. There is
no class equivalent here; `solve_ivp(..., method='RK45')` is the matching
spelling.

### Dense output

`dense_output=True` adds `res.sol`, a callable that evaluates the solution at
any time in the span.

```python
res = solve_ivp(rhs, (0.0, 10.0), np.array([1.0, 0.0]), 'RK45', None, True)
res.sol(0.5)                        # (2,)     one time
res.sol(np.array([0.25, 0.5]))      # (2, 2)   several
```

`res.sol(t)` takes a float or a float64 array. A Python list is not an array:
numba types it as a reflected list, so `res.sol([0.25, 0.5])` raises. Wrap it
in `np.array`. `isinstance(res.sol, OdeSolution)` is `True`.

Two rules apply to `res.sol` inside compiled code. Pass it in as an argument:
one reached as a module global or captured in a closure fails to compile.
Reading a shared one from a `prange` loop is safe and reproduces the serial
result exactly; writing a field from several threads is a data race.

### Events

`events` detects where a scalar function of the state crosses zero. It is a
plain `@njit` `g(t, y)` or `g(t, y, args)`. One function carries every event:
it returns a scalar for a single event and a float64 array of `ng` values for
several. scipy takes a callable or a list of them.

```python
from scijit.integrate import solve_ivp

@njit
def rhs(t, y):                     # y'' = -4y, so y = cos(2t)
    out = np.empty(2)
    out[0] = y[1]
    out[1] = -4.0 * y[0]
    return out

@njit
def event(t, y):                   # trigger when position crosses 0
    return y[0]

res = solve_ivp(rhs, (0.0, 10.0), np.array([1.0, 0.0]), 'LSODA',
                events=event, terminal=True)
# res.t_events[0][0] -> pi/4; res.t[-1] is the same point
```

`terminal=n` stops the integration at the `n`-th crossing and `terminal=True`
at the first. `direction` selects upward or downward crossings. All three match
scipy, which reads them as attributes on each event function where here they
arrive as arguments. `res.message` carries the text for `res.status`.

### LSODA time grid

`t_eval` gives the times at which the solution is reported. With `t_eval` set,
`'LSODA'` steps freely and interpolates onto those times, as
`scipy.integrate.solve_ivp` does, rather than stopping at each requested time as
`scipy.integrate.odeint` does. `dense_output` does not change this.

`t_eval=None` on `'LSODA'` reports the steps the solver actually took.
`npoints`, which has no scipy counterpart, asks for a uniform grid of that many
times instead.

## LSODA directly: `odeint`

`odeint` is scipy's LSODA driver with automatic stiff/nonstiff switching. The
right-hand side is a plain `@njit` `f(y, t, *args)`, with `y` first and one
parameter per entry of `args`.

```python
import numpy as np
from numba import njit
from scijit.integrate import odeint

@njit
def rhs(y, t, k):                  # harmonic oscillator y'' = -k y
    dy = np.empty(2)
    dy[0] = y[1]
    dy[1] = -k * y[0]
    return dy

@njit
def run():
    u0 = np.array([1.0, 0.0])      # initial [position, velocity]
    t_eval = np.linspace(0.0, 10.0, 200)
    return odeint(rhs, u0, t_eval, (4.0,))   # (200, 2)
```

`run()[-1]` is `[0.40808236, -1.82589087]`. `usol[i]` is the state at
`t_eval[i]`, time-major, which is the transpose of `solve_ivp`'s `res.y`.
`t_eval[0]` is the initial time.

`tfirst=True` switches the callback to `f(t, y, *args)`. `full_output=True`
returns `(y, infodict)`, and `success_out=True` appends a bool, which scipy has
no counterpart for. Each of `full_output`, `success_out` and `tfirst` changes
the number or shape of the returns, so each has to be a literal constant inside
`@njit`. An abnormal exit warns `ODEintWarning`, carrying scipy's text.

An analytic Jacobian goes through `Dfun`, a plain `@njit` function of the same
`(y, t, *args)` shape returning the derivative matrix. `ml` and `mu` select the
banded form, and `col_deriv` the transpose.

```python
@njit
def rhs(y, t, k):                  # y'' = -k y
    dy = np.empty(2)
    dy[0] = y[1]
    dy[1] = -k * y[0]
    return dy

@njit
def jac(y, t, k):                  # d(rhs)/dy
    J = np.empty((2, 2))
    J[0, 0] = 0.0; J[0, 1] = 1.0
    J[1, 0] = -k;  J[1, 1] = 0.0
    return J

@njit
def run_jac():
    return odeint(rhs, np.array([1.0, 0.0]),
                  np.array([0.0, 0.5, 1.0]), (4.0,), Dfun=jac)

run_jac()[-1]      # -> [-0.4161468 , -1.81859487]
```

### LSODA callback validation

A right-hand side that never writes `ydot` leaves it zero, and the integration
returns the initial condition with `success=True`. To catch that silent
failure, the right-hand side is called once before integrating with `ydot`
filled with a sentinel, and a `ValueError` is raised if the sentinel survives.
This check is always on and has no off switch.

Applies to `odeint` and `solve_ivp(method='LSODA')`. `quad` is unaffected,
since a QUADPACK integrand returns its value rather than writing through a
pointer.

## Differences from `scipy.integrate`

Every routine on this page is measured against its `scipy.integrate`
counterpart. Where a routine behaves differently, the difference is named here
and in that routine's docstring. Silence means the two were compared and
agreed.

| call | here | `scipy.integrate` |
|---|---|---|
| `quad(..., weight=, wopts=)` | `wopts` accepted and ignored | reuses Chebyshev moments between calls |
| `quad(..., full_output=1)` on a soft failure | 3 values; the code is `infodict.ier` | 4 values, the fourth a message string |
| the values in `quad`'s `infodict` | every one a 1-D float64 array: `info['neval']` is `array([21.])`, `iord` is float64, `chebmo` is flat of length `25 * maxp1` | an `int`, `int32` arrays and a two-dimensional `chebmo` |
| `nquad(..., opts=<callable>)` | `ValueError` | accepts a callable returning a dict |
| `nquad(..., opts=(dict, dict))` carrying `weight` or `points`, from inside `@njit` | `TypingError` | not applicable |
| `nquad(..., opts=[...])` of the wrong length | `ValueError` | indexes from the end, silently using the last `n` |
| `solve_ivp(..., vectorized=True)` | accepted on every method; the right-hand side must take an `(n, 1)` column and the answer is unchanged. Inside `@njit` the flag must be a literal | evaluates the right-hand side on a block of states |
| `solve_ivp(..., foo=1)` | `TypeError` | `UserWarning` naming `foo`, then integrates |
| `solve_ivp(...)` `nfev` after a terminal event | covers the whole span | stops at the event |
| `solve_ivp(..., method=<OdeSolver>)` | `ValueError`, strings only | accepts an `OdeSolver` subclass |
| `solve_ivp(..., method='Radau'\|'BDF')` | `ValueError` | implicit stiff solvers |
| the `infodict` from `full_output` | a namedtuple | a dict |
| `args` | one flat `float64` array | a tuple of objects |

The last two are package-wide rather than specific to one routine. numba's
string-keyed dict coerces its values to a single type, so a result object
holding an int count beside float arrays cannot be a dict. A compiled callback
cannot receive Python objects, so parameters travel as one array.

**`wopts` is accepted and ignored rather than rejected.** A scipy-shaped call
that passes it gets the right answer, computed without the moment reuse. The
cost is time, not accuracy.

**An `opts` length mismatch raises here and does not in scipy.** scipy indexes
`opts` from the end, so three entries against two ranges silently uses the last
two and ignores the first. That is a wrong answer with no signal, so this is a
deliberate divergence rather than an omission.

**`jac`, `lband` and `uband` on an RK method warn and are ignored**, which is
what scipy does, with the same `UserWarning` and the same text. They apply to
`method='LSODA'` only.

**`vectorized=True` is accepted.** With the flag set, the right-hand side
receives an `(n, 1)` column, as scipy passes it with `fun(t, y[:, None])`, and
must be written for that shape. The result matches the default `(n,)` call:
difference `0.000e+00` on RK45, RK23, DOP853 and LSODA. Inside `@njit` the flag
must be a literal.

**The warning and error classes are distinct objects from scipy's.**
`IntegrationWarning`, `ODEintWarning` and `ODEpackError` carry scipy's names and
message text, but `scijit.integrate.IntegrationWarning is
scipy.integrate.IntegrationWarning` is `False`. An `except` clause catches the
`scijit` class, not scipy's. `ODEpackError` has no scipy public counterpart.
