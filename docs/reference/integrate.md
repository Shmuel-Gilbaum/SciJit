# `scijit.integrate`

`scipy.integrate` equivalents callable from inside `numba.@njit` code: QUADPACK
adaptive quadrature, LSODA/`solve_ivp` ODE integration, and the sampled-data
rules. Each name below links to a page generated from its docstring: parameters,
returns, a runnable `@njit` example, and, where it applies, the ways it differs
from scipy.

```{eval-rst}
.. currentmodule:: scijit.integrate

.. autosummary::
   :toctree: generated
   :nosignatures:

   quad
   dblquad
   tplquad
   nquad
   odeint
   solve_ivp
   OdeSolution
   trapezoid
   simpson
   cumulative_simpson
   cumulative_trapezoid
   romb
   newton_cotes
   fixed_quad
   IntegrationWarning
   ODEintWarning
   ODEpackError
```
