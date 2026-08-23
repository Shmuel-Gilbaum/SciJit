# Getting started

This page builds lookup-table splines, evaluates them inside compiled code, and
inverts them with a root find, all inside `@njit`, numba's decorator that
compiles a function to machine code. It runs in a few minutes on a fresh install.

## Install

```bash
pip install scijit
```

The wheels bundle the compiled Fortran. Details and the source path are in
[Install](install.md).

## A first workflow

Two quantities are tabulated on a `(rho, T)` grid, a pressure and an energy.
Build a spline for each, then use them inside compiled code.

```python
import numpy as np
from numba import njit
from scijit.interpolate import RectBivariateSpline

rho = np.linspace(1.0, 5.0, 9)
T   = np.linspace(1.0, 5.0, 9)
RR, TT = np.meshgrid(rho, T, indexing="ij")
pressure = RectBivariateSpline(rho, T, RR * TT)        # P ~ rho * T
energy   = RectBivariateSpline(rho, T, TT)             # U ~ T
```

A spline passed to an `@njit` function as an argument is called like scipy's,
`spl(r, t)`. For a bivariate spline that is the grid form, so a single point
comes back as a `(1, 1)` array.

```python
@njit
def sample(spl, r, t):
    return spl(r, t)[0, 0]

sample(pressure, 2.0, 3.0)           # -> 5.999999999999998
```

## Solving for a state with `fsolve`

Given target values `P = 6` and `U = 3`, solve for the state `(rho, T)` that
produces them. The residual is a plain `@njit` function and `fsolve` finds its
root. The two splines are built once, at the top level, and the residual calls
them directly.

`fsolve` runs inside a compiled function here, so the solved state feeds a
further calculation without leaving `@njit`. The isothermal sound speed
`sqrt(dP/drho)` at that state comes from the same pressure spline.

```python
from scijit.optimize import fsolve

target = np.array([6.0, 3.0])

@njit
def resid(x):
    p = pressure(x[0], x[1])[0, 0]
    u = energy(x[0], x[1])[0, 0]
    return np.array([p - target[0], u - target[1]])

@njit
def sound_speed_at_target():
    state = fsolve(resid, np.array([2.5, 2.5]))    # (rho, T) giving P = 6, U = 3
    rho, T = state[0], state[1]
    drho = 1e-4
    dP_drho = (pressure(rho + drho, T)[0, 0] - pressure(rho, T)[0, 0]) / drho
    return np.sqrt(dP_drho)                         # isothermal sound speed

sound_speed_at_target()          # -> 1.732050807564295
```

The first call to a compiled function pays numba's one-time compile cost; later
calls run the machine code.

## Next steps

- [Usage overview](usage/index.md) covers callback conventions and thread
  safety across the subpackages.
- The per-subpackage guides ([interpolate](usage/interpolate.md),
  [integrate](usage/integrate.md), [optimize](usage/optimize.md)) give one
  tested example per function.
