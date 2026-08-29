# Architecture

This page is for readers who want to understand how scijit is built internally.
Using scijit requires none of it; the usage guides and API reference are enough.

Every layer in this package exists to solve one problem. This page states the
problem, shows it in a few lines of plain numba, then shows how
`scijit.optimize.fsolve` solves it.

**Scope.** This page describes `scijit.optimize`. The other subpackages are
built from the same layers. Section 8 draws its examples from
`scijit.interpolate`.

---

## The call path

numba compiles a separate machine-code body for each combination of argument
types it is called with.

```python
from numba import njit

@njit
def add(a, b):
    return a + b

add(1, 2)                   # 3
add(1.0, 2.0)               # 3.0
add.signatures              # [(int64, int64), (float64, float64)]
```

Two calls, two compiled bodies. The types are fixed when the body is built, so
every type and every array shape has to be settled at compile time.

scipy's API breaks that rule constantly. One argument switches the return
between an array and a 4-tuple. Another is `None` or a float. A third accepts a
scalar, a list, a tuple or an array. A callback is a Python function where a
Fortran library needs a C pointer. A result carries an integer and four arrays
of different shapes under one name.

Each section below is one of those, with the fix.

Three of the fixes are the same mechanism, so it is worth naming once. An
`@overload` chooser is a second function registered against the public one. It
runs once per compiled signature, at compile time, and picks an implementation
from the numba types of the arguments rather than their values. Sections 1, 2
and 3 are three uses of it.

Every worked case on this page is the same function, `scijit.optimize.fsolve`,
which touches all eight layers. Its full signature, with every default:

```python
fsolve(func, x0, args=(), fprime=None, full_output=False, col_deriv=0,
       xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100.0,
       diag=None, mode=-1, validate=True, keep_shape=False)
```

The first twelve are scipy's twelve, name for name and in scipy's order.
`mode`, `validate` and `keep_shape` are additions, placed last so every position
ahead of them matches scipy.

Later sections quote whichever arguments they are about. Anything not named is
at the default above.

The whole path, from the two entry points to the result:

```
    fsolve(f, x0, full_output=True)
      │
      ├── called from Python ──────►  fsolve                    the def
      │
      └── called from @njit ───────►  _fsolve_ovl               the chooser
                                        │
                                        │  settle literals      section 1
                                        │  settle None          section 2
                                        │  normalise inputs     section 3
                                        │  build the callback   section 4
                                        ▼
                                      _core_hybrd               @njit, shared
                                        ▼
                                      _run_hybrd
                                        │  ctypes               section 5
                                        ▼
                                      libminpack.so             MINPACK hybrd
                                        │
                                        ▼
                                      InfoDict(...)             section 6
```

`fsolve` has a Python `def` and an `@overload` chooser. Both settle the
argument types and then call the same `@njit` core, so the two entries return
identical values.

## 1. Compile-time type resolution

**Problem.** A function cannot return a scalar in one branch and a tuple in
another, because the return type is part of the compiled body. `full_output` is
scipy's spelling of exactly that.

```python
@njit
def scale(x, full_output=False):
    if full_output:
        return 2.0 * x, 'doubled'
    return 2.0 * x

scale(3.0, True)
# numba.core.errors.TypingError: Failed in nopython mode pipeline
#   Can't unify return type from the following types:
#   Tuple(float64, Literal[str](doubled)), float64
```

`scale(3.0)` fails the same way. Both branches are compiled whatever the call
looks like, so one of them being unreachable does not help.

**Fix.** Write `scale` as a plain Python function, and register a chooser
against it holding one body per case.

```python
from numba import types
from numba.extending import overload

def scale(x, full_output=False):        # no @njit
    if full_output:
        return 2.0 * x, 'doubled'
    return 2.0 * x

def _literal_bool(v):
    if isinstance(v, bool):
        return v                        # argument omitted: numba passes the default
    if isinstance(v, types.Omitted):
        return bool(v.value)
    if isinstance(v, types.BooleanLiteral):
        return v.literal_value
    return None                         # a runtime value carries no literal

@overload(scale, prefer_literal=True)
def scale_ovl(x, full_output=False):
    fo = _literal_bool(full_output)
    if fo is None:
        return None                     # no body: numba raises

    if fo:
        def impl(x, full_output=False):
            return 2.0 * x, 'doubled'
    else:
        def impl(x, full_output=False):
            return 2.0 * x
    return impl
```

The `def` is not dead code. Called from Python it runs as written, so
`scale(3.0, True)` returns `(6.0, 'doubled')` in the interpreter. Called from
`@njit`, the chooser supplies the compiled body. One function, two entries.

The same calls now compile, and a runtime flag is refused:

```python
@njit
def use_default():
    return scale(3.0)

@njit
def use_literal():
    return scale(3.0, True)

@njit
def use_runtime(flag):
    return scale(3.0, flag)

use_default()                           # 6.0
use_literal()                           # (6.0, 'doubled')
use_runtime(True)
# numba.core.errors.TypingError: No implementation of function
#   Function(<function scale ...>) found for signature: ...
```

**In `fsolve`.** `full_output` is that argument, and `_lit_bool` is that helper.

```python
import numpy as np
from scijit.optimize import fsolve

@njit
def f(x):
    return np.array([x[0] + 2*x[1] - 2.0, x[0]**2 + 4*x[1]**2 - 4.0])

@njit
def solve():
    return fsolve(f, np.array([1.0, 2.0]))

@njit
def solve_full():
    x, info, ier, mesg = fsolve(f, np.array([1.0, 2.0]), full_output=True)
    return x, info.nfev, ier

solve()
# array([8.02802172e-17, 1.00000000e+00])
solve_full()
# (array([8.02802172e-17, 1.00000000e+00]), 17, 1)
```

The root is `(0, 1)`; the first component is the residual of a numerical solve.

`keep_shape` chooses between a flat result and one reshaped to the guess. It
must be a literal to get the reshape; a runtime value degrades to the flat
result rather than raising. Arguments that do not change the return type, such
as `xtol`, `maxfev` and `factor`, accept runtime variables.

## 2. Optional arguments and `None`

**Problem.** An optional argument whose presence changes the result. Supplying a
derivative adds a derivative-evaluation count to what comes back.

Half of this works without any help. numba removes an `is None` branch when
`None` is the value actually passed, so the call that omits the argument
compiles. The call that supplies it leaves both branches live, and their return
types do not unify.

```python
@njit
def solve_toy(x, deriv=None):
    if deriv is None:
        return 2.0 * x, 1               # value, nfev
    return 2.0 * x, 1, 1                # value, nfev, njev

solve_toy(3.0)                          # (6.0, 1)
solve_toy(3.0, 5.0)
# numba.core.errors.TypingError: Failed in nopython mode pipeline
#   Can't unify return type from the following types:
#   Tuple(float64, Literal[int](1)),
#   Tuple(float64, Literal[int](1), Literal[int](1))
```

An optional argument is only free while it leaves the return type alone.

**Fix.** The chooser again, branching on whether the argument is `None`.

```python
def solve_toy(x, deriv=None):           # no @njit
    if deriv is None:
        return 2.0 * x, 1
    return 2.0 * x, 1, 1

def _is_none(v):
    return (v is None                   # argument omitted: numba passes the default
            or isinstance(v, types.NoneType)
            or (isinstance(v, types.Omitted) and v.value is None))

@overload(solve_toy)
def solve_toy_ovl(x, deriv=None):
    if _is_none(deriv):
        def impl(x, deriv=None):
            return 2.0 * x, 1
    else:
        def impl(x, deriv=None):
            return 2.0 * x, 1, 1
    return impl
```

```python
@njit
def without():
    return solve_toy(3.0)

@njit
def with_deriv():
    return solve_toy(3.0, 5.0)

without()                               # (6.0, 1)
with_deriv()                            # (6.0, 1, 1)
```

**In `fsolve`.** `fprime` is `None` or a Jacobian, and the two reach different
MINPACK drivers returning different numbers of values. The extra one is `njev`,
the same field the toy above adds.

```python
@njit
def jac(x):
    J = np.empty((2, 2))
    J[0, 0] = 1.0;       J[0, 1] = 2.0
    J[1, 0] = 2.0*x[0];  J[1, 1] = 8.0*x[1]
    return J

@njit
def solve_jac():
    x, info, ier, mesg = fsolve(f, np.array([1.0, 2.0]),
                                fprime=jac, full_output=True)
    return x, info.nfev, info.njev

solve_jac()
# (array([-5.50632813e-18,  1.00000000e+00]), 13, 1)
```

Without `fprime` the solve took 17 function evaluations; with it, 13 evaluations
and 1 Jacobian.

## 3. Argument normalisation

**Problem.** scipy accepts a scalar, a list, a tuple or an array for the same
argument. Each is a different numba type, and Fortran needs one contiguous
float64 buffer.

**Fix.** Normalise every spelling to that buffer before the core sees it.
`_as_x0` does this, with a small `@overload` per input type.

```python
@njit
def from_list():
    return fsolve(f, [1.0, 2.0])

@njit
def from_tuple():
    return fsolve(f, (1.0, 2.0))

from_list()                             # array([8.02802172e-17, 1.00000000e+00])
from_tuple()                            # array([8.02802172e-17, 1.00000000e+00])
```

Any rank is flattened as scipy flattens it, so an `(n, m)` guess is one system
of `n*m` variables and not a batch.

Two constraints come out of the buffer being flat and float64.

**`args` must pack into one flat float64 buffer.** scipy's `args` is a tuple of
objects, each passed as a separate parameter of a Python callable. A compiled
callback receives a single `double*`. A heterogeneous or ragged tuple raises
`ValueError`, with the concatenation to use in the message.

**Arrays are passed through `np.ascontiguousarray`.** A numpy view's
`.ctypes.data` points into the base buffer, and Fortran reads contiguously, so
a strided array would reach the driver as different numbers. The call is a
no-op on an already contiguous array.

## 4. The callback adapter

**Problem.** MINPACK is Fortran from 1980. It searches for a root by trying a
guess, asking what the equations evaluate to there, and using the answer to pick
the next guess. It knows nothing about the equations themselves, so the caller
has to give it something to call.

What it accepts is a machine address: a number saying where in memory the
compiled equations begin. Fortran and C both call through such an address, and
neither has any way to invoke a Python function.

An `@njit` function has no address to give.

```python
@njit
def plain(x):
    return x * 2.0

plain.address
# AttributeError: 'CPUDispatcher' object has no attribute 'address'
```

`@cfunc` compiles to a C-callable signature and does have one.

```python
from numba import cfunc

@cfunc("float64(float64)")
def compiled(x):
    return x * 2.0

type(compiled.address)                  # int
```

So the objective has to be compiled as a `@cfunc`, and the question is who
writes it.

[NumbaMinpack](https://github.com/Nicholaswogan/NumbaMinpack) (the repo that
inspired this one) converts the incoming address to a Fortran procedure pointer
in a local variable and defines the adapter *inside* the wrapper. The caller
supplies the `@cfunc`, which means writing the equations in the shape the C
interface dictates:

```python
from numba import cfunc
from NumbaMinpack import minpack_sig    # void(double* x, double* fvec, double* args)

@cfunc(minpack_sig)
def f_c(x, fvec, args):
    fvec[0] = x[0] + 2*x[1] - 2.0
    fvec[1] = x[0]**2 + 4*x[1]**2 - 4.0     # written through a pointer, nothing returned

ADDR = f_c.address                          # taken by hand, at Python level
```

Nothing is returned, the results go out through a pointer, the lengths are not
passed so `2` has to be known by whoever writes the body, and the address is
taken by hand. Those equations are now written for MINPACK alone.

**Fix.** Take an ordinary `@njit` function and build the `@cfunc` from it. The
chooser does this at compile time, caches the result, and bakes its address into
the compiled body. `f` on this page is that ordinary function:

```python
@njit
def f(x):
    return np.array([x[0] + 2*x[1] - 2.0, x[0]**2 + 4*x[1]**2 - 4.0])
```

It takes an array and returns an array. Nothing about it is MINPACK-shaped, and
it can be called, tested and reused anywhere else.

**In `fsolve`.** `f` is passed directly, and `fsolve` builds the `@cfunc` from
it. A raw `@cfunc` `.address` is refused with a `ValueError` that names the
low-level drivers (`hybrd`, `lmdif`) which do take one.

```python
@njit
def solve_plain():
    return fsolve(f, np.array([1.0, 2.0]))

solve_plain()                           # array([8.02802172e-17, 1.00000000e+00])
```

Three consequences of building the `@cfunc` this way.

**`.address` is taken at Python level, never inside `@njit`.**

**The adapter costs 12% to 25% per call**, which amortises away over many
iterations. It is cheaper to compile, by roughly 65% on the first call, than the
hand-written `@cfunc`.

**The cache owns the `@cfunc`.** The address is baked into compiled code, so
dropping the last Python reference to it leaves the address dangling.

The adapter also needs the system size, which is not knowable from a pointer,
so the glue puts it in front of the caller's own parameters in the args buffer.
This applies to the adapter path only. A hand-written `@cfunc` receives `args`
untouched, because it indexes them itself.

**On the Fortran side the address lives in a module variable**, not in a local
read by an inner adapter. An inner procedure that reads its parent's locals is
not reachable by a plain address, so gfortran generates a stack trampoline and
the library is then marked as needing an executable stack, which hardened
distributions refuse to load. A module-level adapter has one fixed address and
needs none of that: `readelf -lW libminpack.so` reports `GNU_STACK RW`. Module
variables are shared, so `!$omp threadprivate` gives each thread its own copy.
Section 7 measures it.

**Two probes run before the solve, under `validate=True`.** A `@cfunc` address
carries no arity, so neither the residual count nor the length of `args` can be
read from it. One probe calls the callback once and raises `ValueError` if the
residual buffer was never written; the other probes 8 slots past each buffer end.
The first check is always on.

The failure they catch is silent. A `@cfunc` written with two arguments
transposed once returned the starting point reporting success, agreeing between
the Python and `@njit` paths, both wrong.

**The adapter reads only constants.** The `@cfunc` signature carries pointers
and nothing else, so a value the body needs beyond its pointer arguments is
baked into the compiled code as a compile-time constant. A NumPy array under
numba's constant-array limit and a scalar are baked in directly. A `scijit`
spline is a jitclass, which numba does not freeze on its own; `scijitclass`
supplies the rule that does (section 8). A spline that is fixed when the
objective compiles, held in a module global for instance, is therefore frozen
into the adapter, and the objective evaluates the spline object directly rather
than through its `tck` arrays. A spline whose values are known only at run time
cannot be baked in, and reaches the adapter through the `args` buffer instead.
[Getting started](getting_started.md) works this through.

## 5. The ctypes boundary

**Problem.** The vendored libraries take raw pointers and return nothing.

**Fix.** Bind each wrapper once at import, then pass `.ctypes.data`.

```python
def _sig(fn, nargs):
    fn.argtypes = [ct.c_void_p] * nargs
    fn.restype = None
    return fn
```

Scalars travel as 1-element arrays and integer workspaces are `np.int32`.

The pointer is only valid in the process that created it, and it is baked into
whichever compiled body holds it. An outer wrapper does not change that: the
address reaches every caller.

## 6. Return types

**Problem.** scipy returns a result object with named fields holding an integer
and several arrays of different shapes. A string-keyed dict inside `@njit`
holds one value type, and mixed values are silently coerced to it.

```python
@njit
def as_dict():
    return {'x': 1.5, 'nfev': 12}

as_dict()
# DictType[unicode_type,float64]({x: 1.5, nfev: 12.0})
```

`nfev` went in as `12` and came out as `12.0`. A dict mixing a scalar and an
array does not compile at all:

```python
@njit
def worse():
    return {'nfev': 12.0, 'fvec': np.zeros(3)}

worse()
# TypeError: cannot convert native LiteralStrKey[Dict]
#   ({Literal[str](nfev): float64, Literal[str](fvec): Array(float64, 1d, C)})
```

**Fix.** A namedtuple holds fields of different types and works inside `@njit`.

```python
from collections import namedtuple

Result = namedtuple('Result', ['x', 'nfev'])

@njit
def make():
    return Result(1.5, 12)

r = make()
r.x, r.nfev                             # (1.5, 12)
```

**In `fsolve`.** `full_output=True` returns `InfoDict`, and the `fprime` path
returns `InfoDictJ`, which adds `njev`.

```python
x, info, ier, mesg = fsolve(f, np.array([1.0, 2.0]), full_output=True)

type(info).__name__                     # 'InfoDict'
info._fields                            # ('nfev', 'fjac', 'r', 'qtf', 'fvec')
ier, mesg                               # (1, 'The solution converged.')
```

A namedtuple has a fixed field set, so a field scipy adds conditionally is
either always present or always absent. Where a field has no numba
representation it is absent, and the routine's `Notes` says so.

`nfev` counts every evaluation of the residual, including the ones the package
makes before the solver runs: one to read the residual count, one to check that
the callback writes its output buffer, and two more under `validate=True` for
the bounds probes. The `fprime` path skips the bounds probes.

## 7. Thread safety

**Problem.** A routine can run on several threads at once only if nothing inside
it is shared between calls. The library keeps the address of the callback in one
place, shared by every call. Two solves running at the same time overwrite each
other's entry, and each then evaluates the other's equations. The answers come
back wrong with nothing to indicate it.

**Fix.** Give every thread its own copy of that storage. Concurrent solves then
cannot see each other, and nothing is required of the caller. The Fortran
directive is `!$omp threadprivate`, named here so it can be found in the source
rather than because a caller needs it.

**In `fsolve`.** A `prange` loop over independent problems is safe.

```python
from numba import prange

@njit(parallel=True)
def many(starts):
    out = np.empty((starts.shape[0], 2))
    for i in prange(starts.shape[0]):
        out[i] = fsolve(f, starts[i])
    return out
```

Measured three ways. Running 32 solves concurrently reproduces the serial answer
exactly, `max|parallel - serial|` of `0.0`. Running 2000 independent solves
across 10 threads finished 8.71x faster than running them one after another.
Giving each thread its own storage costs between 4.6% and 9.7% on a
single-threaded call, paid whether or not threads are used.

Not every routine in the package is thread-safe. Each docstring says which, and
a routine that is not says so explicitly.

## 8. jitclasses

**Problem.** A jitclass constructor keeps its defaults in Python and loses them
inside `@njit`.

```python
import numba
from numba.experimental import jitclass

@jitclass([('size', numba.float64),
           ('rooms', numba.int64),
           ('bathrooms', numba.int64)])
class _House:
    def __init__(self, size, rooms, bathrooms=1):
        self.size = size
        self.rooms = rooms
        self.bathrooms = bathrooms

    def area_per_room(self):
        return self.size / self.rooms

_House(120.0, 3).bathrooms              # 1   from Python, the default applies

@njit
def build():
    return _House(120.0, 3).bathrooms

build()
# TypeError: invalid number of args: expected 4, got 3
```

The class is usable either way; what does not survive is the convenience.
Inside `@njit` every argument has to be written out at every call site.

**Fix.** Put the defaults on a thin `@njit` factory instead of on the class. A
plain `@njit` function keeps its defaults in both worlds, so the caller does not
have to supply them.

```python
@njit
def house(size, rooms, bathrooms=1):
    return _House(size, rooms, bathrooms)

@njit
def build():
    h = house(120.0, 3)                 # default applies here
    return h.bathrooms, h.area_per_room()

build()                                 # (1, 40.0)
house(120.0, 3).bathrooms               # 1   the same call from Python
```

The class keeps every argument explicit; the factory is the public name, and it
is where a default belongs.

**In `scijit.interpolate`.** Every public name is an `@njit` factory over a
private jitclass, so its defaults apply in both worlds and the same call
compiles either side of the boundary.

```python
from scijit.interpolate import CubicSpline

x = np.linspace(0.0, 6.0, 7)
y = np.sin(x)
CubicSpline(x, y)(2.5)                  # 0.5987436865244009   from Python

@njit
def spline_default(xx, yy):
    return CubicSpline(xx, yy)(2.5)

spline_default(x, y)                    # 0.5987436865244009   the same
```

Constructing the private class directly is the case that still needs every
argument written out, which is why the factory is the public name.

**A second jitclass limit** is that a plain jitclass cannot define `__call__`,
so a bare jitclass is evaluated through named methods: `.ev()` for an array,
`.ev_one()` for a scalar. `scijit` depends on `scijitclass`, a companion package
that registers `__call__` on a jitclass through numba's typing and lowering
hooks. The `scijit.interpolate` classes are built with it, so scipy's `spl(x)`
call works from Python and inside `@njit`, and the argument types choose the
method when the calling function compiles: a scalar reaches `.ev_one`, an array
reaches `.ev`.

```python
spl = CubicSpline(x, y)
spl(2.5)                                # 0.5987436865244009        scalar -> .ev_one
spl(np.array([2.5, 3.5]))               # [ 0.59874369 -0.35204928] array  -> .ev

@njit
def evaluate(xx, yy):
    return CubicSpline(xx, yy)(np.array([2.5, 3.5]))

evaluate(x, y)                          # [ 0.59874369 -0.35204928] the same, in @njit
```

The named methods remain available; `spl(x)` is the scipy-shaped spelling for
the same evaluation.

**A third `scijitclass` mechanism** freezes a registered instance as a
compile-time constant. numba lowers an `int`, a `float` and a NumPy array baked
into compiled code as a constant, but has no rule for a jitclass instance, so
one held in a module global or closed over inside a compiled function raised at
lowering. `scijitclass` registers that rule for a registered class, rebuilding
the instance from its fields, each of which numba can already lower as a
constant. This is what lets a spline appear inside the `@cfunc` a solver builds
from its objective (section 4): the spline is a constant baked into the
callback. The instance must be fixed when the code compiles; one passed in at
run time is not frozen, and its `tck` arrays travel through `args` instead.
Fields must themselves be constant-lowerable, so a nested jitclass or a
`typed.List` field cannot be frozen.

```python
from scijit.optimize import fsolve

table = CubicSpline(x, y)                # built once, a compile-time constant

@njit
def resid(v):
    return np.array([table(v[0]) - 0.5])

fsolve(resid, np.array([0.4]))           # array([0.49774486])   table frozen into the callback
```

One further jitclass limit. There is no inheritance, so classes scipy derives from a
common base are separate jitclasses here sharing one spec, with the shared
behaviour in a module-level `@njit` helper.
