"""scijit.integrate, adaptive integration and ODE solving in @njit.

Adaptive quadrature:
    quad       one name over seven QUADPACK routes: finite or infinite
               limits, break points, and the cos/sin/alg/cauchy weights.
               Takes a plain @njit f(x, args).
    nquad      integration over n variables; ranges[0] is innermost
    dblquad    double integral; each inner limit is a constant or a callback
    tplquad    triple integral; same

Sampled-data rules (integrate samples, take no callback):
    trapezoid  simpson  cumulative_simpson  cumulative_trapezoid  romb
    newton_cotes   weights and error coefficient of a rule
    fixed_quad     fixed-order Gauss-Legendre; takes a vectorized @njit fn

Initial value problems:
    solve_ivp      scipy's signature. method='RK45'/'RK23'/'DOP853' reaches
                   the pure port, 'LSODA' reaches Fortran. res.y is
                   (n_states, n_times), scipy's orientation, and the result
                   is a namedtuple, so res.y works and res['y'] does not.
                   dense_output=True adds a callable res.sol(t), on every
                   method. events=g reports the roots of g(t, y) and, with
                   terminal, stops there.
    odeint         LSODA over t_eval, auto stiff/nonstiff switching

Diagnostics:
    IntegrationWarning  what quad warns with when QUADPACK reports a soft
                   failure and full_output is off, as scipy does

Callbacks are plain @njit functions, everywhere. An integrand is written
`f(x, args)`, a right-hand side `f(t, y)` or scipy's `f(y, t)`. Parameters
reach a right-hand side either through a closure or, where the routine
accepts `args`, through a third parameter, `f(t, y, args)`. Everything is
passed as an ordinary argument:

    quad(f, 0.0, np.inf, args)          odeint(f, y0, t)
    dblquad(f, a, b, gfun, hfun)        solve_ivp(f, (t0, t1), y0, 'LSODA')
    nquad(f, ((0.0, 1.0), (0.0, 1.0)))

Where a routine reaches Fortran the @cfunc is built internally, when the
calling function compiles, so no caller needs to hold a `.address`. Every
public routine, `solve_ivp(method='LSODA')`, `odeint` and the quad family
included, takes a plain @njit function and rejects a `.address` or a raw
integer pointer in the callback slot, from Python and from inside @njit.

`nquad`, `dblquad` and `tplquad` take a plain function as the integrand and
reject an `.address` there too. The coordinates of a multi-dimensional
integrand cross as separate arguments, which the internal integrand callback
has no slot for.

Every routine in this subpackage is prange-safe. The Fortran packs reach
their callback through a module variable, and that slot is `!$omp
threadprivate`, so each thread reads its own copy.

Nesting needs nothing from the caller either. Each wrapper saves the slot on
entry and restores it on exit, so an integration inside another integration
is correct to any depth. That is what `nquad` is built on: an n-deep nest of
`quad`, its depth following the length of `ranges`. `dblquad` and `tplquad`
are `nquad` at depth two and three, which is how scipy implements them too,
so every QUADPACK route reaches them: infinite limits, break points, the
weights and `full_output`, per axis.

The names above are the whole public surface. Several array-level spellings
and result types live on in their modules and are reachable there, but a
caller reaching for one is reaching past the front end that covers it.
"""
from ._quadpack import quad, IntegrationWarning
from ._odeint_scipy import odeint, ODEintWarning, ODEpackError
from ._quadrature import (simpson, cumulative_simpson, romb, newton_cotes,
                          fixed_quad, trapezoid, cumulative_trapezoid)
from ._nquad import nquad, dblquad, tplquad
from ._solve_ivp import solve_ivp, OdeSolution

# Deliberately NOT exported, and each reachable by its module path.
#
#   OdeResult, OdeResultDense    the two result SHAPES solve_ivp returns,
#                                picked by `dense_output`. A caller reads
#                                fields off the instance and never names
#                                the type; scipy exports neither.
#   LsodaSolution                what `res.sol` is on 'LSODA'. A caller
#                                invokes `res.sol(t)` and never constructs
#                                one. `OdeSolution`, its RK counterpart, IS
#                                exported, because scipy publishes that name.
#   rk_dense_eval,               the array-level twins of `res.sol`.
#     lsoda_dense_eval
#   METHOD_RK45 .. METHOD_LSODA  int codes the private engines route on.
#
# `roots_legendre` belongs to `scijit.special`, which mirrors scipy's
# layout; `_quadrature` keeps it because `fixed_quad` needs it.

__all__ = [
    # adaptive quadrature
    'quad', 'dblquad', 'tplquad', 'nquad',
    # initial value problems (scipy's signature)
    'odeint', 'solve_ivp', 'OdeSolution',
    # fixed-sample / fixed-order quadrature
    'trapezoid', 'simpson', 'cumulative_simpson', 'cumulative_trapezoid',
    'romb', 'newton_cotes', 'fixed_quad',
    # warning and error classes a caller may need to name
    'IntegrationWarning', 'ODEintWarning', 'ODEpackError',
]
