# SciJIT

[GitHub](https://github.com/shmuel-gilbaum/SciJit) &middot;
[PyPI](https://pypi.org/project/scijit/)

A scipy-equivalent library callable from inside numba `@njit` code. scijit
either wraps the same Fortran packs scipy wraps, or re-implements scipy's
pure-Python routines as `@njit`. Results match scipy, checked routine by
routine against it during development. It pays off most when the same routine
runs many times inside a compiled loop, where scipy would pay Python's
per-call overhead on every iteration.

scipy's classes are called like `spl(x)`, but a numba jitclass has no
`__call__`. To mirror scipy, scijit uses
[scijitclass](https://github.com/Shmuel-Gilbaum/SciJitClass), which adds
`__call__` to numba jitclasses, so scijit's classes keep scipy's call syntax
inside `@njit`. It is a standalone `@jitclass` replacement, usable in any numba
project.

```python
import numpy as np
from numba import njit
from scijit.integrate import quad

@njit
def decay(t):
    return np.exp(-t)

@njit
def area(a, b):
    return quad(decay, a, b)[0]

area(0.0, 1.0)                                 # 0.6321205588285578
```

## Getting started

- [Getting started](getting_started.md): a first end-to-end `@njit` workflow.
- [Install](install.md): wheel install, source install, and the import-failure
  note.

## Usage guides

- [Usage overview](usage/index.md): callback conventions and thread safety.
- [interpolate](usage/interpolate.md): FITPACK splines, `BSpline`,
  `make_interp_spline`, `Akima1DInterpolator`, and the interpolator classes.
- [integrate](usage/integrate.md): `quad`, `solve_ivp`, `odeint`, the nestable
  `nquad`/`dblquad`/`tplquad`, and the sampled-data quadrature routines.
- [optimize](usage/optimize.md): roots, least squares, minimization, PRIMA, and
  the scalar root-finders and minimizers.

## Reference

- [API reference](reference/index.md): every public name, generated from the
  docstrings.

## Explanation

- [Architecture](architecture.md): what a call passes through, the callback
  adapter, the ctypes boundary, the return types, and the jitclass rules.
- [Compatibility](compatibility.md): supported versions, where agreement with
  scipy is bit-for-bit, thread safety, and the `prange`-safety matrix.
- [Roadmap](roadmap.md): what is not covered yet, and why.
- [Credits](credits.md): provenance and the upstream library and license table.

```{toctree}
:hidden:
:caption: Getting started

getting_started
install
```

```{toctree}
:hidden:
:caption: Usage guides

usage/index
usage/interpolate
usage/integrate
usage/optimize
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:caption: Explanation

architecture
compatibility
roadmap
credits
```
