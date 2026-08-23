"""scijit.optimize, scipy.optimize equivalents inside numba @njit.

Layout mirrors scipy.optimize: private ``_module.py`` backends hold one
Fortran pack or one port each, and the public API is curated here.

Minimization, multivariate
    minimize            method='Nelder-Mead' | 'Powell' | 'CG' | 'BFGS' |
                        'L-BFGS-B' | 'SLSQP' | 'COBYLA', and additionally
                        'UOBYQA' | 'NEWUOA' | 'BOBYQA' | 'LINCOA'
    fmin                Nelder-Mead simplex
    fmin_powell         Powell's direction set
    fmin_cg             nonlinear conjugate gradient
    fmin_bfgs           BFGS
    fmin_l_bfgs_b       L-BFGS-B, with m, factr, pgtol and maxfun
    fmin_slsqp          SLSQP, equality and inequality constraints
    HessInv             the inverse-Hessian estimate an L-BFGS-B result
                        carries

Minimization, derivative-free (PRIMA)
    fmin_cobyla         linear and nonlinear inequality constraints

Global
    brute               grid search, with an optional local polish
    basinhopping        hopping between local minimizations
    differential_evolution

Scalar root finding and minimization
    bisect, brentq, brenth, ridder, toms748
                        bracketing root finders
    newton              secant, Newton-Raphson or Halley, selected by
                        which derivatives are supplied
    fixed_point         x = f(x) by iteration
    root_scalar         one front end over the bracketing solvers
    bracket             three points that bracket a minimum
    golden, brent       downhill bracket search, then a parabolic fit
    fminbound           minimization inside fixed bounds
    minimize_scalar     one front end over brent, golden and bounded
    RootResults         what bisect, brentq, brenth, ridder, toms748,
                        newton and root_scalar return

Root finding, systems of equations
    root                method='hybr' | 'lm'
    fsolve              MINPACK hybrd, or hybrj when fprime is given

Least squares
    leastsq             MINPACK lmdif, or lmder when Dfun is given
    curve_fit           nonlinear fit of a model to data
    nnls                non-negative least squares
    lsq_linear          bound-constrained linear least squares

Assignment
    linear_sum_assignment

Warnings
    OptimizeWarning

Callbacks
    Every routine here takes a plain ``@njit`` function. Where a Fortran
    pack needs a function pointer, the ``@cfunc`` is built and cached
    internally.

Each routine's own docstring records where it departs from scipy, under
Notes. A routine with no such entry does not depart.
"""
from ._minpack import fsolve, leastsq, root, OptimizeResult
from ._lbfgsb import fmin_l_bfgs_b
from ._slsqp import fmin_slsqp
from ._minimize import minimize, HessInv
from ._prima import fmin_cobyla

__all__ = [
    # minimization, gradient-based (plain @njit fg)
    'minimize', 'fmin_l_bfgs_b', 'fmin_slsqp',
    # the inverse-Hessian estimate a `minimize` result carries. scipy
    # exports its counterpart, LbfgsInvHessProduct, and isinstance works
    # against both.
    'HessInv',
    # minimization, derivative-free (PRIMA)
    'fmin_cobyla',
    # root finding
    'root', 'fsolve',
    # least squares
    'leastsq',
    # scalar root-finding + minimization (ports; f is a plain @njit fn)
    'bisect', 'brentq', 'brenth', 'ridder', 'toms748',
    'newton', 'fixed_point', 'root_scalar',
    'bracket', 'golden', 'brent', 'fminbound', 'minimize_scalar',
    # the result bisect, brentq, brenth, ridder, toms748, newton and
    # root_scalar return. scipy publishes the same name, so `isinstance`
    # against `scijit.optimize.RootResults` is the way to name the type.
    # The two classes are distinct: a result from here is not an instance
    # of `scipy.optimize.RootResults`.
    'RootResults',
    # what minimize, minimize_scalar, root, lsq_linear,
    # differential_evolution and basinhopping return. The field set is the
    # solver's: a field the call did not compute is absent, as in scipy.
    'OptimizeResult',
    # linear & nonlinear least squares (ports; nnls/lsq_linear prange-safe)
    'nnls', 'lsq_linear', 'curve_fit', 'OptimizeWarning',
    # multivariate minimizers (ports; plain @njit func/grad, prange-safe)
    'fmin', 'fmin_powell', 'fmin_cg', 'fmin_bfgs',
    # global / stochastic optimizers (ports; plain @njit func, prange-safe)
    'brute', 'basinhopping', 'differential_evolution',
    # assignment problem (port)
    'linear_sum_assignment',
]

from ._scalar import (
    bisect, brentq, brenth, ridder, toms748,
    newton, fixed_point, root_scalar,
    bracket, golden, brent, fminbound, minimize_scalar,
    RootResults,
)
from ._lsq import nnls, lsq_linear, curve_fit, OptimizeWarning
from ._minimize_py import fmin, fmin_powell, fmin_cg, fmin_bfgs
from ._global import (
    brute, basinhopping, differential_evolution,
)
from ._assignment import linear_sum_assignment
