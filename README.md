# SciJIT (`scijit`)

📖 **Documentation:** <https://shmuel-gilbaum.github.io/SciJit/>

A scipy-equivalent library callable from inside numba `@njit` code. scijit
either wraps the same Fortran packs scipy wraps, or re-implements scipy's
pure-Python routines as `@njit`. Results match scipy, checked routine by
routine against it during development. Where a routine differs from scipy, its
Notes say so; an argument it does not implement raises rather than silently
returning something else. It pays off most when the same routine runs many
times inside a compiled loop, where scipy would pay Python's per-call overhead
on every iteration. Same namespaces, same names, same grouping. Not every scipy
subpackage is mirrored; see [docs/roadmap.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/roadmap.md) for the gaps.

scijit is not affiliated with, or endorsed by, the NumPy, SciPy or Numba
projects. It is also distinct from
[numba-scipy](https://github.com/numba/numba-scipy), which provides numba
support for a small part of `scipy.special`.

The approach comes from
[NumbaMinpack](https://github.com/Nicholaswogan/NumbaMinpack) (Nicholas
Wogan): reach a compiled Fortran pack from `@njit` through a `@cfunc` address
and a `bind(c)` wrapper. scijit applies it across many packs. See
[docs/credits.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/credits.md).

scipy's classes are called like `spl(x)`, but a numba jitclass has no
`__call__`, so a compiled class cannot be called that way. To mirror scipy,
scijit uses [scijitclass](https://github.com/Shmuel-Gilbaum/SciJitClass), a
companion package that adds `__call__` to numba jitclasses, so scijit's classes
keep scipy's call syntax inside `@njit`. It is a standalone `@jitclass`
replacement, usable in any numba project.

Anything numba already handles is out of scope: the
[numpy features numba supports](https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html),
and the `scipy.special` subset numba-scipy covers.

scijit was written mostly with Anthropic's Claude (via Claude Code) under my
direction. I set the architecture and the design choices, and every routine is
verified against SciPy during development.

## Subpackages

Three scipy subpackages, callable from inside `@njit`:

- **`scijit.interpolate`** — splines and the interpolator classes (`scipy.interpolate`).
- **`scijit.optimize`** — roots, least squares, and minimization (`scipy.optimize`).
- **`scijit.integrate`** — `quad`, `solve_ivp`, `odeint`, and quadrature (`scipy.integrate`).

Each routine's coverage, backing Fortran, and scipy parity are in the
[usage guides](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/usage/index.md) and [API reference](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/reference/index.md).
More subpackages (`fft`, `linalg`, `stats`, `special`, and others) are in
development; see [docs/roadmap.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/roadmap.md).

## Install

```bash
pip install scijit
```

The wheels bundle the compiled Fortran, so nothing extra is needed on that
path. A source install compiles each Fortran pack and requires **gfortran** on
the PATH. Full instructions and troubleshooting are in
[docs/install.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/install.md).

## Example

Build an interpolant once, then use it inside compiled code:

```python
import numpy as np
from numba import njit
from scijit.interpolate import CubicSpline
from scijit.integrate import simpson

# a coarse lookup table: opacity sampled at a few log-temperatures
logT   = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
logkap = np.array([0.1, 0.4, 1.2, 2.0, 1.5, 0.6])
opacity = CubicSpline(logT, logkap)        # build the interpolant once

@njit
def mean_opacity(opacity, T_lo, T_hi, n_cells=101):
    T = np.linspace(T_lo, T_hi, n_cells)
    return simpson(opacity(T), T) / (T_hi - T_lo)   # band-averaged, all in @njit

mean_opacity(opacity, 3.5, 7.5)            # -> 1.260677088
```

`opacity(...)` evaluates the spline, from Python and inside `@njit`, exactly as
in scipy, and `simpson` integrates it in the same compiled function. Build the
interpolant once and reuse it. A full walkthrough is in
[docs/getting_started.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/getting_started.md).

## Performance: when this helps

The gain comes from removing interpreter overhead, so it shows up in large or
nested loops. scipy cannot be called from `@njit`, so it pays the interpreter
on every iteration. Reproduce with `python benchmarks/<name>.py`:

| benchmark | workload | speedup |
|---|---|---|
| [`bench_pde_front.py`](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/benchmarks/bench_pde_front.py) | 1-D Burgers front (MOL + LSODA), a 2-var `fsolve` per gridpoint per adaptive step | **9-19x** |
| [`bench_ode_implicit.py`](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/benchmarks/bench_ode_implicit.py) | 400 stiff ODEs, a 3-var `fsolve` inside every LSODA step | **14-22x** |
| [`speed_grid.py`](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/benchmarks/speed_grid.py) | 18000-cell grid: root solve + opacity spline + `simpson` | **10-20x** |

Speedups are approximate and vary with machine, problem size and thread count.
Intel SVML has been unsupported by numba since 0.62, which costs some vectorized
performance on both sides of these comparisons.

The speedup tracks how much of each iteration is interpreter overhead, since
that is what the compiler removes. It is not a claim about the numerics: where
scijit calls Fortran, the same Fortran runs in both cases. numba also compiles on
first call, which can make a single isolated call slower.

Rule of thumb:

- **one large vectorized call** from Python
  ([`bench_where_scipy_wins.py`](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/benchmarks/bench_where_scipy_wins.py)) &rarr;
  use scipy. The shipped subpackages wrap the same Fortran, so a single big
  call ties: a 200k-point bivariate spline evaluation matched scipy within 4%,
  with identical results (max|diff| 0).
- **a small solve repeated many times**, or nested loops &rarr; use scijit.

## Documentation

The full documentation site is at
**<https://shmuel-gilbaum.github.io/SciJit/>** (Markdown sources under
[docs/](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/index.md)).

- [docs/getting_started.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/getting_started.md): a first end-to-end
  `@njit` workflow.
- [docs/install.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/install.md): wheel install, source install, and the
  import-failure note.
- [docs/usage/index.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/usage/index.md): task-by-task usage guides, one
  tested runnable example per function.
- [docs/compatibility.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/compatibility.md): supported versions, where
  agreement with scipy is bit-for-bit and where it is not, thread safety, and
  the `prange`-safety matrix.
- [docs/architecture.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/architecture.md): what a call passes through,
  from the entry points down through the callback adapter, the ctypes boundary,
  the return types, and the jitclass rules.
- [docs/roadmap.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/roadmap.md): what is not covered yet, and why.
- [docs/credits.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/credits.md): provenance and the upstream library and
  license table.

## How this project started

In 2021, during my PhD in astrophysics I was running an AGN accretion-disk simulation: a large coupled ODE system where every integration step had to solve 3 nonlinear equations at each of thousands of grid points. Both the root-solving and the variable extraction that followed depended on opacity tables, read through bivariate spline interpolation.

NumbaMinpack let me do the root-finding inside @njit code, but scipy's spline evaluators cannot be called there. So I followed NumbaMinpack's approach and wrapped the two FITPACK routines I needed for the table lookups, bispev and bispeu. The runtime dropped from unusable to a few hours.

scijit is the follow-up, built together with Claude: the whole of FITPACK instead of those two routines, then the other Fortran packs behind scipy, plus `@njit` versions of the routines scipy writes in Python.

## License & citation

scijit wraps established numerical libraries; each subpackage carries its
upstream license in `src/<pack>/`, and the full citation list is in
[CITATION.cff](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/CITATION.cff) (GitHub "Cite this repository"). Credit details
are in [docs/credits.md](https://github.com/Shmuel-Gilbaum/SciJit/blob/main/docs/credits.md).
