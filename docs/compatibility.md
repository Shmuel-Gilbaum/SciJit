# Compatibility

What scijit is measured against, where it agrees with scipy exactly, where it
differs, and which routines are safe to call concurrently.

Last measured 2026-07-27 on **scipy 1.18.0, numba 0.66.0, numpy 2.4.6**.

---

## Requirements

Runtime dependencies, installed automatically with scijit:

| package | requirement | why |
|---|---|---|
| Python | `>=3.10` | |
| `numba` | `>=0.66` | 0.66 adds the multi-dimensional fancy indexing the package uses |
| `numpy` | numba's bound | scijit sets no bound of its own; the version is whatever the installed numba supports |
| `scijitclass` | `>=0.1.7` | makes the jitclasses callable (`spl(x)`); see [credits](credits.md) |

`scipy` is **not** a runtime dependency and is never imported by scijit. Parity
is measured against `scipy>=1.18` during development.

---

## Agreement with scipy

scijit wraps the same Fortran scipy has historically wrapped, so most routines
agree exactly. Where scipy has since moved a routine to a C translation or a
pure-Python port, scijit keeps the Fortran; the two then agree to a few ulp, or
a little more for the few where scipy runs a different algorithm entirely.
Argument names, defaults, return fields, status codes and error behaviour follow
scipy exactly.

Each routine's docstring names any difference from scipy under `Notes`; silence
means the two were compared and agreed. Per-routine accuracy figures are in the
docstrings.

---

## Degenerate inputs

On a few degenerate inputs where scipy is silent and returns a wrong or undefined
result, scijit raises instead: a wrong-length SLSQP gradient, a callback that
writes nothing, and `nnls` with zero equations (which reads uninitialised memory
in scipy). Each is noted in the routine's docstring.

---

## Distinct warning classes

`scijit.integrate.IntegrationWarning` and `scijit.optimize.OptimizeWarning` are
distinct class objects from the scipy warnings of the same name, sharing only the
name and the `UserWarning` base. A filter naming scipy's class does not catch this
one, and the reverse. Filter on this package's class or on `UserWarning`. This
keeps scipy out of the runtime dependencies. The same holds for
`ODEintWarning`/`ODEpackError` against scipy's private ODEPACK error type.

---

## Thread safety

Every routine in the three subpackages is safe to call from parallel code: inside
a `numba.prange` loop, a thread pool, or separate processes. "Parallel" here means
a `@njit(parallel=True)` function whose loop runs over `prange` instead of `range`.
One pattern keeps it safe:

**Build in serial, evaluate in parallel.** Construct each spline, interpolator or
solver result *before* the parallel loop; inside the loop, only evaluate it.
Evaluation is read-only, so many threads can share one object safely.

```python
from numba import njit, prange

@njit(parallel=True)
def eval_many(spl, xs, out):        # spl built in serial by the caller
    for i in prange(xs.shape[0]):
        out[i] = spl(xs[i])         # read-only, shared safely across threads
```

`multiprocessing` is always safe: each process loads its own copy of the
library. Measured: 32 concurrent threads reproduce the single-threaded result
exactly.

Two argument choices turn off the parallelism:

- **A Python `callback`, or `disp` output.** Passing a plain Python function as a
  `callback` to `fmin_*`, `basinhopping` or `differential_evolution`, or setting
  `disp=1`, forces those calls to run one at a time, even inside a `prange` loop.
  For the parallel path, pass the callback as an `@njit` function and leave
  `disp=0`.
- **A shared random generator.** In `basinhopping` and `differential_evolution`,
  leave `rng` at its default or pass an integer seed, so each call gets its own
  random state. Do not share one `numpy.random.Generator` object across threads.

---

## Compile-time constraints inside `@njit`

An argument that selects the SHAPE or PRESENCE of the result must be readable when
the call compiles: a literal, an omitted default, or a module-level constant. A
runtime value raises `TypingError`. From Python any value works.

- `full_output` (`bisect`, `brentq`, `brenth`, `ridder`, `toms748`, `newton`,
  `fixed_point`, `fsolve`, `leastsq`, `fminbound`, `brent`, `golden`),
  `brute`'s `full_output` and `workers` all select the return shape.
- A `method` string that selects the iteration or the return type
  (`root`, `minimize`, `minimize_scalar`, `fixed_point`), and whether `root`'s
  or `fsolve`'s `jac`/`fprime` is `None`, must be a compile-time constant.
- `romb`'s `show` and `nquad`'s per-axis weight `opts` are read at compile
  time.
- A `bracket`, `bounds`, `brack`, `ranges`, `args` or `opts` sequence passed
  into compiled code must be a tuple or array, not a Python list.

Passing a settings dict from inside `@njit` (`minimize`'s `options`, `nquad`'s
`opts`):

- A dict whose values are all numbers can hold only numbers. A tuple or array value
  in it fails to compile.
- An empty `{}` has no type. Pass `None` for the empty case.
- `nquad`'s `weight` and `points` settings work as a single `opts` dict applied to
  every axis, but not as a per-axis tuple of dicts. Call `nquad` from Python for the
  per-axis form.

---

## Result containers

Read a result the way you would in scipy. `res.x`, `res.fun`, `res.success`,
`res['x']`, `res.get(...)`, and `in` / `keys` / `values` / `items` all work, the
same from Python and inside `@njit`.

Two differences from scipy:

- The result is a namedtuple, not a `dict` subclass. `isinstance(res,
  scipy.optimize.OptimizeResult)` is False, and integer indexing or unpacking reads
  the field values (scipy raises, or yields field names).
- A field the solver did not compute is absent, not `None`. Reading it raises
  `AttributeError` from Python, `TypingError` inside `@njit`. The routine's `Notes`
  lists any field that is always absent.

**Constructor defaults are Python-only.** A jitclass constructor's defaults apply
from Python and are lost inside `@njit`, where every argument must be written out.
This affects `OdeSolution` and its LSODA/complex twins, and every
`scijit.interpolate` class constructed directly. The public `@njit` factory names
(`CubicSpline`, `make_interp_spline`, ...) keep their defaults in both worlds; the
private jitclass they wrap does not.

---

## Per-routine divergences

Where a routine differs from scipy, its docstring `Notes` says how; the
difference also shows on the routine's page in the
[API reference](reference/index.md). Silence there means it was compared
to scipy and agreed.

---

## Coverage of scipy arguments

A routine absent from this table matches SciPy on every argument it accepts.

Listed is each argument or mode a routine refuses where SciPy accepts it. An
argument that behaves differently but does not raise is not here; it is in the
routine's `Notes`. An argument scijit adds beyond scipy is not here either.

Exception class: from Python the refusal is a `ValueError` unless noted; a
refusal resolved when the call compiles is a `TypingError` inside `@njit`.
`BSpline` and `make_interp_spline` refuse complex coefficients with a
`TypingError` in both.

### interpolate

| Routine | Raises on |
|---|---|
| `RegularGridInterpolator` | `method` of `'slinear'`, `'cubic'`, `'quintic'` |
| `interpn` | `method` of `'slinear'`, `'cubic'`, `'quintic'` |
| `BSpline` | a complex `c` |
| `make_interp_spline` | a complex `y`; a derivative-pair `bc_type` |

### optimize

| Routine | Raises on | Routes to |
|---|---|---|
| `curve_fit` | `bounds`, `method`, `jac` | the nonlinear `least_squares` path |
| `minimize` | `method` outside {Nelder-Mead, Powell, CG, BFGS, L-BFGS-B, SLSQP, COBYLA}; `jac` of `'2-point'`, `'3-point'`, `'cs'`; a `constraints` list | |
| `root` | `method` outside {hybr, lm} | |
| `lsq_linear` | `lsq_solver='lsmr'` | the dense `'exact'` solver |
| `brute` | `finish` other than `None` or `fmin` | |
| `basinhopping` | `take_step`, `accept_test`; `minimizer_kwargs['method']` outside {Nelder-Mead, Powell, BFGS, CG} | |
| `differential_evolution` | `integrality`, a `constraints` list, `vectorized=True`, `workers != 1`, `updating='deferred'`, a callable `strategy` | |

### integrate

| Routine | Raises on |
|---|---|
| `solve_ivp` | `method` of `'Radau'`, `'BDF'`; `events` with a complex `y0` |

---

## Caching

Nothing here caches compiled code to disk. The first call to a routine in a process
pays numba's one-time compile cost; later calls in the same process are fast.

Build each spline, interpolator and solver result once, then reuse it inside
compiled code.

---

## Platforms

Wheels are built and tested for Linux, Windows and Apple Silicon (arm64) macOS.
Intel x86_64 macOS is not supported: numba moved it to Tier 2 in 0.63 and stopped
shipping binaries for it, so an x86_64 macOS wheel would have no numba to install
beside it.
