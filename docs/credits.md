# Credits & provenance

scijit wraps established numerical libraries rather than reinventing them. The
hard numerical work is upstream; please credit it. Each subpackage carries its
upstream library's license in `src/<pack>/`, and the full citation list is in
[CITATION.cff](../CITATION.cff) (GitHub "Cite this repository").

scipy's classes are called like `spl(x)`, but a numba jitclass has no
`__call__`, so a compiled class cannot be called that way. To mirror scipy,
scijit depends on [scijitclass](https://github.com/Shmuel-Gilbaum/SciJitClass),
a companion package that adds `__call__` to numba jitclasses. This is what lets
scijit's spline and interpolator classes keep scipy's call syntax inside `@njit`
code. It is a standalone `@jitclass` replacement, usable in any numba project.

## Development

Much of scijit (the Fortran `bind(c)` wrappers, the pure-`@njit` ports, the
test suites, and this documentation) was written with Anthropic's Claude, via
Claude Code. The direction is the author's: the architecture, the decision of
what to wrap versus port, and the review of the output. Every routine is
verified against SciPy in the test suites. The numerical work itself belongs to
the upstream libraries credited below; scijit wraps and ports them, it does not
reinvent them.

## Nicholas Wogan

**Nicholas Wogan's** work runs through this project in three ways:

- **[NumbaMinpack](https://github.com/Nicholaswogan/NumbaMinpack)**: demonstrated
  the approach this package is built on (call a compiled Fortran pack from inside
  `@njit` via a `@cfunc` address + `bind(c)` wrapper). scijit keeps its low-level
  MINPACK signatures (`minpack_sig`, `hybrd`, `lmdif`); the public
  `fsolve`/`leastsq` follow scipy's API. His MIT notice is in
  `src/minpack/LICENSE_NumbaMinpack.txt`.
- **[odepack](https://github.com/Nicholaswogan/odepack)**: his thread-safe
  modern-Fortran ODEPACK is the library scijit vendored and wrapped for
  `scijit.integrate`'s LSODA/LSODAR.
- **[numbalsoda](https://github.com/Nicholaswogan/numbalsoda)**:
  `scijit.integrate`'s `lsoda_sig` cfunc signature matches numbalsoda's. scijit's
  `odeint` itself follows scipy's API.

A related project, not used here:
[NumbaQuadpack](https://github.com/Nicholaswogan/NumbaQuadpack) (also
Nicholaswogan). scijit's `quad` wraps `jacobwilliams/quadpack` with its own
wrappers; the only overlap is the natural integrand convention (return a value,
take an args pointer).

## Acknowledgements

- MINPACK is by Garbow, Hillstrom & Moré (Argonne, 1980); the sources in
  `src/minpack/` are the modernized
  [fortran-lang/minpack](https://github.com/fortran-lang/minpack) module
  (MIT + the University of Chicago MINPACK notice, `src/minpack/LICENSE.txt`).
- PRIMA is Zaikun Zhang's modern reference implementation of M. J. D. Powell's
  derivative-free solvers ([libprima/prima](https://github.com/libprima/prima),
  BSD-3-Clause; `src/prima/LICENCE.txt`).
- L-BFGS-B is by Nocedal, Zhu, Byrd & Morales; SLSQP by Dieter Kraft. The
  sources in `src/lbfgsb/` and `src/slsqp/` are Jacob Williams' modern-Fortran
  refactors ([jacobwilliams/lbfgsb](https://github.com/jacobwilliams/lbfgsb),
  [jacobwilliams/slsqp](https://github.com/jacobwilliams/slsqp), both BSD-3-style;
  license files carried alongside).
- QUADPACK is by Piessens, de Doncker-Kapenga, Überhuber & Kahaner; the sources
  in `src/quadpack/` are Jacob Williams'
  [jacobwilliams/quadpack](https://github.com/jacobwilliams/quadpack) double
  edition (BSD-3).
- ODEPACK (LSODA/LSODAR) is by Alan Hindmarsh (LLNL, public domain); the
  vendored modern edition is [Nicholaswogan/odepack](https://github.com/Nicholaswogan/odepack).
- Reference-LAPACK + BLAS ([Reference-LAPACK/lapack](https://github.com/Reference-LAPACK/lapack),
  Anderson et al., BSD-3) back `scijit.optimize`'s least-squares path
  (`curve_fit`, `lsq_linear`) through the shared `scijit/_lib/liblapackref`.
- FITPACK was written by Paul Dierckx (KU Leuven) and is described in his book
  *Curve and Surface Fitting with Splines* (Oxford University Press, 1993). The
  sources bundled in `src/fitpack/` are his public-domain originals, with one
  edit: the twelve smoothing fitters (`curfit` `percur` `parcur` `clocur`
  `concur` `regrid` `parsur` `pogrid` `spgrid` `sphere` `surfit` `polar`) take
  `tol` and `maxit` as trailing arguments instead of setting them as locals, so
  a caller can reach them. Each file carries a comment recording the change and
  the original values.
- scipy's `scipy.interpolate`, `scipy.optimize` and `scipy.integrate` wrap the
  same Fortran and are the references our test suites compare against.

## Upstream libraries & licenses

Each subpackage wraps one upstream numerical library (or is a pure `@njit` port
of scipy's pure-Python code); wrapped-library licenses are carried in
`src/<pack>/` and the full citation list is in [CITATION.cff](../CITATION.cff).

| subpackage | wraps | upstream | license |
|---|---|---|---|
| `interpolate` | FITPACK | Dierckx (netlib) | public domain |
| `optimize` (roots/lsq) | MINPACK | [fortran-lang/minpack](https://github.com/fortran-lang/minpack); low-level API from [Nicholaswogan/NumbaMinpack](https://github.com/Nicholaswogan/NumbaMinpack) | MIT |
| `optimize` (L-BFGS-B) | L-BFGS-B | Nocedal et al. / [jacobwilliams/lbfgsb](https://github.com/jacobwilliams/lbfgsb) | BSD-3 |
| `optimize` (SLSQP) | SLSQP | Kraft / [jacobwilliams/slsqp](https://github.com/jacobwilliams/slsqp) | BSD-3 |
| `optimize` (derivative-free) | PRIMA | Zhang, after Powell / [libprima/prima](https://github.com/libprima/prima) | BSD-3 |
| `optimize` (lsq LAPACK) | Reference-LAPACK | [Reference-LAPACK/lapack](https://github.com/Reference-LAPACK/lapack) (Anderson et al.) | BSD-3 |
| `optimize` (scalar roots) | n/a | pure `@njit` port of scipy | BSD-3 (scijit) |
| `integrate` (quad) | QUADPACK | Piessens et al. / [jacobwilliams/quadpack](https://github.com/jacobwilliams/quadpack) | BSD-3 |
| `integrate` (ODEs) | ODEPACK LSODA/LSODAR | Hindmarsh (LLNL); modern edition [Nicholaswogan/odepack](https://github.com/Nicholaswogan/odepack) | public domain |
| `integrate` (quadrature) | n/a | pure `@njit` ports of scipy | BSD-3 (scijit) |
