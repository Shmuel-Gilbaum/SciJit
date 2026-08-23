# `scijit.optimize`

`scipy.optimize` equivalents callable from inside `numba.@njit` code: MINPACK
roots, L-BFGS-B / SLSQP / COBYLA / PRIMA minimizers, scalar root-finders and
minimizers, and the global optimizers. Each name below links to a page generated
from its docstring: parameters, returns, a runnable `@njit` example, and, where
it applies, the ways it differs from scipy.

```{eval-rst}
.. currentmodule:: scijit.optimize

.. autosummary::
   :toctree: generated
   :nosignatures:

   minimize
   minimize_scalar
   root
   root_scalar
   fsolve
   leastsq
   curve_fit
   nnls
   lsq_linear
   linear_sum_assignment
   fmin
   fmin_powell
   fmin_cg
   fmin_bfgs
   fmin_l_bfgs_b
   fmin_slsqp
   fmin_cobyla
   bisect
   brentq
   brenth
   ridder
   toms748
   newton
   fixed_point
   bracket
   golden
   brent
   fminbound
   brute
   basinhopping
   differential_evolution
   HessInv
   RootResults
   OptimizeResult
   OptimizeWarning
```
