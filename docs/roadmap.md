# Roadmap and known gaps

The scipy surface scijit does not cover yet. Most gaps are future work, a few
larger because scipy implements them in C or C++, and a few are deliberate
choices.

## Subpackages

This release ships `interpolate`, `optimize` and `integrate`. Planned for later
releases: `fft`, `linalg`, `stats`, `special`, `spatial`, `signal`, `ndimage`,
`sparse` (`sparse.linalg`, `sparse.csgraph`), `cluster` and `constants`.

## Missing routines (future work)

Within the shipped subpackages, in roughly the order expected:

| subpackage | routines |
|---|---|
| `optimize` | `least_squares` (TRF/dogbox), `dual_annealing` |
| `integrate` | `solve_bvp`, implicit `Radau`/`BDF` |
| `interpolate` | `RBFInterpolator`, `griddata` (nearest mode) |

`least_squares` is the large one. `dual_annealing` is a loop over an existing
minimizer. An implicit `Radau`/`BDF` would be the first prange-safe stiff solver
in the package; the wrapped LSODA runs sequentially. Also future work: ODRPACK
(`scipy.odr`), a Fortran wrap the build already handles, and `KDTree`/`cKDTree`,
larger because a jitclass cannot reference its own type, so the tree has to be
flattened into index arrays and every traversal rewritten.

## Provided by numba and numpy

numba and numpy already run these inside `@njit`, so scijit does not duplicate
them: `np.linalg` (solve, inv, det, eig, svd, qr, cholesky), `np.interp`,
`np.convolve`, `np.trapz`, `np.roots`, `np.gradient`. numba's native coverage
grows release to release. As it does, more of this surface, including routines
scijit ports today, becomes available natively and moves out of scope.

## Backed by C or C++ in scipy

scipy implements these with C or C++ libraries. scijit wraps Fortran, so it does
not wrap those libraries directly. The algorithms are published, so an `@njit`
port or Fortran wrap is possible future work.

| routine | scipy library |
|---|---|
| `linprog`, `milp` | HiGHS |
| `direct` | DIRECT |
| `ConvexHull`, `Voronoi`, `Delaunay`, `griddata` (linear/cubic) | Qhull |
| `fmin_tnc`, trust-krylov | TNC, trlib |

`griddata`'s linear and cubic modes need Delaunay; only its nearest mode is
reachable without Qhull.

## Deliberate scope limits

Choices, not gaps.

- Randomized routines (`basinhopping`, `differential_evolution`) take an integer
  seed. They are reproducible per seed and independent across parallel calls, but
  do not return scipy's values, because numba cannot call `np.random.Generator`.
  Deterministic routines return scipy's values.
- Some wrapped routines are sequential-only, because the Fortran keeps its
  callback in a module variable. The thread-safety table in
  [compatibility.md](compatibility.md#thread-safety) says which.
