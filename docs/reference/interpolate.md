# `scijit.interpolate`

`scipy.interpolate` equivalents callable from inside `numba.@njit` code, backed
by the same Dierckx FITPACK Fortran that SciPy wraps. Each name below links to a
page generated from its docstring: parameters, returns, a runnable `@njit`
example, and, where it applies, the ways it differs from scipy.

```{eval-rst}
.. currentmodule:: scijit.interpolate

.. autosummary::
   :toctree: generated
   :nosignatures:

   splrep
   splprep
   splev
   splder_ev
   splint
   sproot
   spalde
   bisplev
   bispev
   bispeu
   splder
   splantider
   UnivariateSpline
   InterpolatedUnivariateSpline
   LSQUnivariateSpline
   RectBivariateSpline
   SmoothBivariateSpline
   RectSphereBivariateSpline
   SmoothSphereBivariateSpline
   RegularGridInterpolator
   interpn
   CubicSpline
   PchipInterpolator
   interp1d
   BSpline
   make_interp_spline
   Akima1DInterpolator
```
