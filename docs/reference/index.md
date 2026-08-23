# Reference

The API reference is generated from the docstrings. Each public name carries a
numpydoc docstring with its parameters, returns, a runnable `@njit` example, and,
where it applies, the ways it differs from scipy.

## The scipy convention

A routine's docstring records only DIFFERENCES from scipy, under `Notes`. Silence
means the two were compared and found to agree. A routine with no scipy
counterpart says so.

Measured accuracy figures live in the docstring they describe, not in prose here.
The thread-safety and per-subpackage scope tables are in
[compatibility.md](../compatibility.md).

```{toctree}
:maxdepth: 1

interpolate
integrate
optimize
```
