"""Global and stochastic optimizers, in numba ``@njit``.

``scipy.optimize.brute``, ``basinhopping`` and ``differential_evolution``
are **pure Python** in scipy (they only orchestrate calls to a local
minimizer and to ``numpy.random``), so they are written here directly in
``@njit``, with no compiled backend and no module state.

Callback convention (shared with ``_minimize_py.py`` and ``_scalar.py``): the
objective is a PLAIN ``@njit`` function passed as a first-class
argument --

    ``func(x, *args) -> float64``    (x 1-D float64)

``args`` is a tuple unpacked into the argument list after ``x``, and its
elements may be of any type a compiled call accepts; an ndarray or a list
arrives as ONE argument instead.  No ``@cfunc``, no ``.address``, no captured
globals, no
module-level RNG -> **prange-safe** (verified: many independent global
searches at once).

--------------------------------------------------------------------------
ROUTINE LISTINGS

  brute                  grid search, optionally polished
  basinhopping           random restarts around a local minimizer
  differential_evolution population-based stochastic search

Signatures are not repeated here; each routine's own docstring carries
them, and a copy in this header cannot be kept honest.  (It was not: this
block previously listed `ranges_lo, ranges_hi`, `bounds_lo, bounds_hi`,
`mutation_lo, mutation_hi` and `args` in third position, none of which
had survived the retarget onto scipy's argument order.)

NAMES.  Three arguments select an algorithm, each by scipy's name for it.

  basinhopping ``minimizer_kwargs['method']``
                    'BFGS' (default, and scipy's own local minimizer),
                    'CG', 'Nelder-Mead', 'Powell'
  DE ``strategy``   'best1bin' (default), 'best1exp', 'rand1bin',
                    'rand1exp', 'randtobest1bin', 'randtobest1exp',
                    'currenttobest1bin', 'currenttobest1exp', 'best2bin',
                    'best2exp', 'rand2bin', 'rand2exp'
  DE ``init``       'latinhypercube' (default), 'random', 'sobol',
                    'halton'

  brute ``finish``  ``None``, or the minimiser itself.  scipy's default is
                    the callable ``finish=fmin``, and so is this one.
  DE ``polish``     a bool, defaulting to True as scipy's does. The polish
                    is L-BFGS-B over the same box, scipy's own.

--------------------------------------------------------------------------
DEVIATIONS from scipy:

  * THE RNG.  ``basinhopping`` and ``differential_evolution`` take
    scipy's ``rng``: ``None``, an integer, or a
    ``numpy.random.Generator``.  ``None`` selects an internal xorshift64*
    generator carried in a per-call state array, whose stream the
    additive integer ``seed`` selects and whose default, ``seed=None``,
    draws fresh entropy per call, as a bare scipy call does.  An integer
    ``rng`` builds a ``numpy.random.default_rng``, which is PCG64, where
    scipy's ``check_random_state`` builds the legacy MT19937
    ``RandomState`` that compiled code cannot hold.  Neither path
    reproduces scipy's trajectory at a given seed, so both are validated
    statistically: they find the known global minimum of Rastrigin /
    Ackley / six-hump-camel / Rosenbrock.  ``brute`` uses no RNG at all
    and IS compared to scipy exactly.
  * UNAVAILABLE: a ``grid`` whose ndim follows N.  ``brute`` returns it
    as a **(M, N)** array of the M = Ns**N
    grid points (last dimension varying fastest, i.e. C order) and
    ``Jout`` as a flat **(M,)** array, because a numba function cannot
    return an array whose ndim depends on a runtime value.  Reshape
    outside ``@njit`` to recover scipy's shapes:
    ``grid.T.reshape((N,) + (Ns,)*N)`` and ``Jout.reshape((Ns,)*N)``.
    The axis values reproduce ``numpy.mgrid``'s ``start + i*step``
    formula exactly (which is *not* identical to ``np.linspace`` in the
    last bit), so the grid is bit-identical to scipy's.
  * UNAVAILABLE: ``brute``'s per-axis ``slice`` objects with explicit
    steps.  ``ranges`` is one ``(N, 2)`` array of ``(lo, hi)`` pairs and
    ``Ns`` sets the count on every axis.  ``Ns`` defaults to 20, as
    scipy's does.  An INTEGER slice does reach a compiled function; a
    FLOAT slice, which is the common scipy spelling here, raises
    ``TypeError`` during unboxing before any compiled body runs
    (measured 2026-08-02).  A slice with an explicit step also gives each
    axis its own point count, which the shared ``Ns`` cannot express.
  * ``basinhopping`` takes scipy's ``minimizer_kwargs``, carrying
    ``'method'`` and ``'jac'``.  Inside ``@njit`` the dict must be written
    as a LITERAL at the call site: a literal is typed where it is written
    and can hold a callable, while a variable has to be unboxed and a numba
    dict cannot hold a function.  The default, ``None``, is BFGS with a
    forward-difference gradient, which is scipy's default quench.
  * NOT IMPLEMENTED: custom ``take_step`` / ``accept_test`` callables.
    Step-taking is scipy's ``RandomDisplacement`` +
    ``AdaptiveStepsize`` + ``Metropolis`` (all defaults reproduced:
    ``interval=50``, ``target_accept_rate=0.5``, ``stepwise_factor=0.9``).
  * NOT IMPLEMENTED, in ``differential_evolution``: ``constraints``,
    ``integrality``, ``vectorized`` and ``workers``.  Each is accepted
    only at its default.  Only ``updating='immediate'``, scipy's default,
    is implemented.  ``x0`` and ``callback`` are served.
  * ``mutation`` is a float or scipy's two-element dithering range,
    defaulting to ``(0.5, 1.0)``.
  * UNAVAILABLE: an ``OptimizeResult``.  Every routine returns a
    namedtuple, reached by attribute; numba has no result objects.

TOLERANCE: ``brute`` with ``finish=None`` is EXACT vs scipy (same grid,
same ``argmin`` tie-break); with the polish it agrees to the Nelder-Mead
tolerance (~1e-8 on the reported optimum).  ``basinhopping`` /
``differential_evolution`` are compared to the analytic global optimum,
not to scipy.
"""
import warnings
from collections import namedtuple

import numpy as np
from numba import njit, objmode, prange, types
from numba.typed import List
from numba.core.errors import TypingError
from numba.extending import overload

from ._minimize_py import (fmin, fmin_powell, fmin_cg, fmin_bfgs,
                            _powell_core, _powell_limits)
from ._lbfgsb import (_lbfgsb_run, _lbfgsb_bounds, _lbfgsb_flag, _ev_approx,
                      _call_args)
from ._minpack import _opt_result
from ._callback import (_bh_noop, _cb_noop, _cb_install, _cb_release,
                        _cb_resolve, _cb_resolve_ty,
                        _cb_halt_clear, _cb_halt_take)
# `_qmc` is a private vendored copy of `scijit.stats._qmc` (pure @njit, no
# other subpackage), so the public package ships interpolate/integrate/
# optimize without stats. FUTURE WORK: re-import from `..stats._qmc` when
# stats ships publicly, and delete `optimize/_qmc.py`.
from ._qmc import halton as _qmc_halton, sobol as _qmc_sobol


@njit
def _powell5(func, x0, args, xtol, ftol):
    """`fmin_powell`'s five reported values, at its own defaults.

    The local quenches inside `basinhopping` and `brute` run hundreds of
    times, so they reach the core directly: the public entry would print
    scipy's summary on every one of them at the scipy default `disp=1`.
    """
    n = x0.shape[0]
    mi, mf = _powell_limits(n, np.int64(0), np.int64(0), True, True)
    r = _powell_core(func, x0, args, xtol, ftol, mi, mf,
                     np.zeros((0, 0), np.float64), False, False)
    return r[0], r[1], r[3], r[4], r[5]

__all__ = ['brute', 'basinhopping', 'differential_evolution']


# =============================================================== RNG (local)
# xorshift64* -- tiny, fast, no global state.  The state lives in a length-1
# uint64 array owned by the caller, so every call is independent (and every
# prange iteration is independent) while still being seed-reproducible.

_U64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


@njit
def _rng_init(seed):
    """splitmix64 the user's integer seed into a non-zero xorshift state."""
    st = np.empty(1, dtype=np.uint64)
    z = np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    if z == np.uint64(0):
        z = np.uint64(0x9E3779B97F4A7C15)
    st[0] = z
    return st


@njit
def _rng_next(st):
    """One xorshift64* step: advance ``st`` and return 64 random bits.

    The whole generator is three shifts, three xors and a multiply, which
    matters because a global optimizer draws several numbers per
    objective evaluation. Keeping the state in the caller's array rather
    than in a module global is what makes concurrent runs independent.
    """
    x = st[0]
    x ^= x >> np.uint64(12)
    x ^= (x << np.uint64(25)) & _U64_MASK
    x ^= x >> np.uint64(27)
    st[0] = x
    return (x * np.uint64(0x2545F4914F6CDD1D)) & _U64_MASK


@njit
def _rng_u01(st):
    """Uniform in [0, 1) with 53 random bits."""
    return np.float64(_rng_next(st) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit
def _rng_uniform(st, lo, hi):
    """Uniform draw in ``[lo, hi)``.

    Written as ``lo + (hi - lo) * u`` because that is the order the
    basin-hopping displacement and the differential-evolution population
    both need; it consumes exactly one `_rng_u01` draw, which is what
    keeps a seeded run reproducible.
    """
    return lo + (hi - lo) * _rng_u01(st)


@njit
def _rng_randint(st, n):
    """Uniform integer in [0, n) (n > 0)."""
    return np.int64(_rng_u01(st) * np.float64(n)) % np.int64(n)


@njit
def _rng_shuffle(st, a):
    """Fisher-Yates in place (same sweep direction as numpy Generator)."""
    n = a.shape[0]
    for i in range(n - 1, 0, -1):
        j = _rng_randint(st, i + 1)
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp


# ------------------------------------------------ the seed, and `rng`
#: The numba type of a ``numpy.random.Generator``, named once so the
#: ``objmode`` block that builds one has something to declare.
_NP_GEN_T = types.NumPyRandomGeneratorType('Generator')

_RNG_MSG = (
    "rng must be None, an integer, or a numpy.random.Generator.")

#: The upper end of the entropy draw, exclusive, so the seed is a
#: non-negative int64.
_SEED_HI = 9223372036854775807


def _seed_value(seed):
    """`seed` as the integer `_rng_init` splits, drawing one when it is None.

    ``seed=None`` is scipy's default and means "draw fresh entropy", so two
    identical calls give different answers. The draw happens per call and at
    run time, from numba's own thread-local ``np.random`` state in compiled
    code and from numpy's global state in Python.
    """
    if seed is None:
        return int(np.random.randint(0, _SEED_HI))
    return int(seed)


@overload(_seed_value)
def _seed_value_ovl(seed):
    """@njit implementation of `_seed_value`.

    numba's ``np.random`` carries one state per thread, so the entropy draw
    is independent across a ``prange`` and across processes. Measured at 32
    threads: 100000 draws, 100000 distinct values.
    """
    s = seed.value if isinstance(seed, types.Omitted) else seed
    if s is None or isinstance(s, types.NoneType):
        def impl(seed):
            return np.random.randint(0, _SEED_HI)
        return impl

    def impl(seed):
        return np.int64(seed)
    return impl


def _rng_gen(rng):
    """scipy's `rng`: ``None``, an integer, or a ``numpy.random.Generator``.

    Returns the generator to draw from, or ``None`` for the internal
    xorshift64* path.
    """
    if rng is None:
        return None
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
        return np.random.default_rng(rng)
    raise ValueError(_RNG_MSG)


@overload(_rng_gen)
def _rng_gen_ovl(rng):
    """@njit implementation of `_rng_gen`.

    ``np.random.default_rng`` cannot be called from compiled code
    (``Unknown attribute 'default_rng'``), so the integer spelling builds its
    generator inside an ``objmode`` block, once per call rather than once per
    compilation: a generator built in the chooser would be shared by every
    call to the compiled function, which is both the wrong stream and the
    ``prange`` hazard.
    """
    r = rng.value if isinstance(rng, types.Omitted) else rng
    if r is None or isinstance(r, types.NoneType):
        def impl(rng):
            return None
        return impl
    if isinstance(r, types.NumPyRandomGeneratorType):
        def impl(rng):
            return rng
        return impl
    if isinstance(r, types.Integer) and not isinstance(r, types.Boolean):
        def impl(rng):
            with objmode(g=_NP_GEN_T):
                g = np.random.default_rng(rng)
            return g
        return impl

    def bad(rng):
        raise ValueError(_RNG_MSG)
    return bad


# The five draws the two optimizers make. Each is one function with two
# implementations chosen by the TYPE of `gen`: the xorshift64* state array
# when it is None, and the generator's own method otherwise, spelled the way
# scipy spells it so the order and the shape of the draws carry over.

def _dr_u01(st, gen):
    """One uniform in [0, 1)."""
    return _rng_u01(st) if gen is None else gen.uniform()


@overload(_dr_u01)
def _dr_u01_ovl(st, gen):
    if isinstance(gen, types.NoneType):
        def impl(st, gen):
            return _rng_u01(st)
        return impl

    def impl(st, gen):
        return gen.uniform()
    return impl


def _dr_uniform(st, gen, lo, hi):
    """One uniform in [lo, hi)."""
    return _rng_uniform(st, lo, hi) if gen is None else gen.uniform(lo, hi)


@overload(_dr_uniform)
def _dr_uniform_ovl(st, gen, lo, hi):
    if isinstance(gen, types.NoneType):
        def impl(st, gen, lo, hi):
            return _rng_uniform(st, lo, hi)
        return impl

    def impl(st, gen, lo, hi):
        return gen.uniform(lo, hi)
    return impl


def _dr_uvec(st, gen, lo, hi, out):
    """``out[:] = uniform(lo, hi, out.size)``, scipy's vector displacement."""
    if gen is None:
        for k in range(out.shape[0]):
            out[k] = _rng_uniform(st, lo, hi)
    else:
        out[:] = gen.uniform(lo, hi, out.shape[0])


@overload(_dr_uvec)
def _dr_uvec_ovl(st, gen, lo, hi, out):
    if isinstance(gen, types.NoneType):
        def impl(st, gen, lo, hi, out):
            for k in range(out.shape[0]):
                out[k] = _rng_uniform(st, lo, hi)
        return impl

    def impl(st, gen, lo, hi, out):
        out[:] = gen.uniform(lo, hi, out.shape[0])
    return impl


def _dr_randint(st, gen, n):
    """A uniform integer in [0, n)."""
    if gen is None:
        return _rng_randint(st, n)
    return np.int64(gen.uniform() * np.float64(n)) % np.int64(n)


@overload(_dr_randint)
def _dr_randint_ovl(st, gen, n):
    if isinstance(gen, types.NoneType):
        def impl(st, gen, n):
            return _rng_randint(st, n)
        return impl

    def impl(st, gen, n):
        return np.int64(gen.uniform() * np.float64(n)) % np.int64(n)
    return impl


def _dr_shuffle(st, gen, a):
    """Shuffle `a` in place."""
    if gen is None:
        _rng_shuffle(st, a)
    else:
        gen.shuffle(a)


@overload(_dr_shuffle)
def _dr_shuffle_ovl(st, gen, a):
    if isinstance(gen, types.NoneType):
        def impl(st, gen, a):
            _rng_shuffle(st, a)
        return impl

    def impl(st, gen, a):
        gen.shuffle(a)
    return impl


# ==================================================================== brute

@njit
def _mgrid_axis(lo, hi, Ns):
    """One axis of ``np.mgrid[lo:hi:Ns*1j]`` -- ``start + i*step``.

    NOT ``np.linspace`` (which snaps the endpoint), so the values are
    bit-identical to what scipy's ``brute`` feeds the objective.
    """
    g = np.empty(Ns, dtype=np.float64)
    if Ns == 1:
        g[0] = lo
        return g
    step = (hi - lo) / np.float64(Ns - 1)
    for i in range(Ns):
        g[i] = np.float64(i) * step + lo
    return g


from .._lib._typing import _lit_bool as _lit_bool_g   # noqa: E402


def _lit_int_g(v):
    """Compile-time int out of whatever numba hands the overload."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, types.Omitted):
        return int(v.value)
    if isinstance(v, (types.IntegerLiteral, types.BooleanLiteral)):
        return int(v.literal_value)
    return None


@njit
def _brute_grid(func, axes, args, N, Ns, M):
    """The grid evaluation in order, on one thread."""
    grid = np.empty((M, N), dtype=np.float64)
    Jout = np.empty(M, dtype=np.float64)
    pt = np.empty(N, dtype=np.float64)
    for m in range(M):
        rem = m
        for k in range(N - 1, -1, -1):
            idx = rem % Ns
            rem = rem // Ns
            pt[k] = axes[k, idx]
        grid[m, :] = pt
        if isinstance(args, tuple):
            Jout[m] = func(pt, *args)
        else:
            Jout[m] = func(pt, args)
    return grid, Jout


@njit(parallel=True)
def _brute_grid_par(func, axes, args, N, Ns, M):
    """The grid evaluation as a ``prange``.

    Independent by construction: iteration ``m`` decodes its own index and
    writes only row ``m``.  The scratch point is allocated INSIDE the loop
    body, which is what makes it private per iteration -- a single buffer
    hoisted outside would be shared across threads and is the whole hazard.
    """
    grid = np.empty((M, N), dtype=np.float64)
    Jout = np.empty(M, dtype=np.float64)
    # Per-axis divisor, so the body derives each digit from `m` DIRECTLY.
    # The serial loop carries a running remainder instead, which numba's
    # parfor pass rejects as "Overwrite of parallel loop index".
    div = np.empty(N, dtype=np.int64)
    for k in range(N):
        d = 1
        for _ in range(N - 1 - k):
            d = d * Ns
        div[k] = d
    for m in prange(M):
        pt = np.empty(N, dtype=np.float64)
        for k in range(N - 1, -1, -1):
            pt[k] = axes[k, (m // div[k]) % Ns]
        grid[m, :] = pt
        if isinstance(args, tuple):
            Jout[m] = func(pt, *args)
        else:
            Jout[m] = func(pt, args)
    return grid, Jout


_BRUTE_POLISH_WARN = (
    "Either final optimization did not succeed or `finish` does not return "
    "`statuscode` as its last argument.")


def _brute_polish_warn():
    warnings.warn(_BRUTE_POLISH_WARN, RuntimeWarning, stacklevel=2)


@njit
def _brute_warn_polish():
    """scipy's warning when the polish reports a non-zero status.

    Its own function because lowering an ``objmode`` block pickles the
    enclosing one, and the block goes on the failing branch so an ordinary
    polish never takes the GIL.
    """
    with objmode():
        _brute_polish_warn()


@njit
def _brute_finish(func, grid, Jout, args, finish, xtol, ftol, N, M, disp):
    """argmin over the raveled Jout, then the optional polish.

    The scan reproduces NUMPY's ``argmin``, which is what scipy applies to the
    raveled ``Jout``: a NaN compares as the minimum and the FIRST one wins, so
    a NaN-bearing objective reports that grid point with ``fval`` NaN.  A
    plain ``Jout[m] < fmin_val`` skips every NaN and returns a finite point
    instead, which hides the NaN.  Measured against scipy on a 9x9 grid whose
    objective is NaN for ``x[0] > 1``: both return ``x = [1.5, -2.0]``,
    ``fval = nan``.
    """
    imin = 0
    fmin_val = Jout[0]
    if not np.isnan(fmin_val):
        for m in range(1, M):
            v = Jout[m]
            if not (v >= fmin_val):      # true for v < fmin_val AND for NaN
                fmin_val = v
                imin = m
                if np.isnan(v):
                    break
    xmin = np.empty(N, dtype=np.float64)
    for k in range(N):
        xmin[k] = grid[imin, k]
    Jmin = fmin_val
    if finish == 1:
        # scipy passes `disp` straight into the `finish` minimiser's keyword
        # arguments, so a `disp=True` brute prints `fmin`'s summary, and it
        # warns when the polish reports a non-zero status.
        dp = 1 if disp else 0
        xo, fo, itn, ncalls, wflag = fmin(func, xmin, args, xtol, ftol,
                                          None, None, True, dp)
        xmin = xo
        Jmin = fo
        if wflag != 0 and disp:
            _brute_warn_polish()
    return xmin, Jmin


@njit
def _brute_axes(ranges_lo, ranges_hi, Ns):
    """Per-axis coordinates plus the grid size, shared by both loops.

    Holds the guards, so the serial and parallel paths and the python and
    @njit entries cannot disagree about what is rejected.
    """
    N = ranges_lo.size
    if N == 0:
        raise ValueError("brute: ranges must be non-empty")
    if N > 40:
        raise ValueError(
            "Brute Force not possible with more than 40 variables.")
    if Ns < 1:
        raise ValueError("brute: Ns must be >= 1")
    M = 1
    for k in range(N):
        M = M * Ns
        if M > 2147483647:
            raise ValueError("brute: grid has too many points (> 2**31)")
    axes = np.empty((N, Ns), dtype=np.float64)
    for k in range(N):
        axes[k, :] = _mgrid_axis(ranges_lo[k], ranges_hi[k], Ns)
    return axes, N, M


def _brute_split_ranges(ranges):
    """scipy's ``ranges`` -> (lo, hi).

    ``ranges`` is scipy's: ONE ``(min, max)`` pair per dimension, so an
    ``(N, 2)`` array.  scipy also takes ``slice`` objects, which numba
    cannot represent.

    Note there is no separate ``(lo, hi)`` spelling and no way to add one:
    at N = 2 a pair of length-2 arrays IS an ``(N, 2)`` array, so the two
    readings cannot be told apart by inspection.  Passing ``(lo, hi)``
    therefore reads as pairs, the same way scipy would read it.
    """
    r = np.ascontiguousarray(np.asarray(ranges, np.float64))
    if r.ndim != 2 or r.shape[1] != 2:
        raise ValueError(_BRUTE_RANGES_MSG)
    return np.ascontiguousarray(r[:, 0]), np.ascontiguousarray(r[:, 1])


@overload(_brute_split_ranges)
def _brute_split_ranges_ovl(ranges):
    """@njit implementation of `_brute_split_ranges`.

    `brute` is a Python body plus an ``@overload`` twin, so this shared
    validation step needs both as well: the function above is not jitted
    and cannot be reached from inside ``@njit``. The guard, its message
    and the returned pair are the same; only the dtype cast is spelled
    differently.
    """
    def impl(ranges):
        r = np.ascontiguousarray(np.asarray(ranges).astype(np.float64))
        if r.ndim != 2 or r.shape[1] != 2:
            raise ValueError(_BRUTE_RANGES_MSG)
        return (np.ascontiguousarray(r[:, 0]),
                np.ascontiguousarray(r[:, 1]))
    return impl


_BRUTE_RANGES_MSG = (
    "brute: `ranges` must be one (min, max) pair per dimension, an (N, 2) "
    "array_like, which is scipy's own spelling.")

_BRUTE_FINISH_MSG = (
    "brute: `finish` must be None (no polish) or scijit.optimize.fmin "
    "(polish with Nelder-Mead, scipy's default). scipy takes any minimiser "
    "callable; fmin is the one scipy defaults to, and it is the only polish "
    "this implements.")


def _finish_code(finish):
    """`finish` as the int code `_brute_finish` compares.

    scipy spells "no polish" as ``None`` and "polish" as the minimiser
    itself, defaulting to ``fmin``. Both are the argument's identity rather
    than a value a compiled comparison could reach, so both resolve here.
    """
    if finish is None:
        return 0
    if finish is fmin:
        return 1
    raise ValueError(_BRUTE_FINISH_MSG)


def _finish_code_t(finish):
    """`_finish_code` over numba TYPES, for the @overload.

    Returns the code, or ``None`` when the argument is neither spelling.

    The FIRST test is against the plain python object, not against a numba
    type, and it is the one that serves an omitted `finish`. An omitted
    default reaches a chooser as the RAW PYTHON VALUE, so `fmin` arrives
    here as the function itself; only an argument written at the call site
    arrives as ``types.Function``. Measured 2026-08-03: with that test
    removed, ``brute(f, ranges)`` inside ``@njit`` fails to type while
    ``brute(f, ranges, finish=fmin)`` compiles.
    """
    f = finish.value if isinstance(finish, types.Omitted) else finish
    if f is fmin:
        return 1
    if f is None or isinstance(f, types.NoneType):
        return 0
    if isinstance(f, types.Function) and f.typing_key is fmin:
        return 1
    return None

_BRUTE_FULL_OUTPUT_MSG = (
    "brute: full_output must be a compile-time constant. It selects between a "
    "bare x0 and a 4-tuple, and a numba function has one return type, so it "
    "cannot be a variable read while the code runs.")

_BRUTE_WORKERS_MSG = (
    "brute: workers must be an integer literal. scipy also accepts a "
    "map-like callable (multiprocessing.Pool.map); a compiled function "
    "cannot call one, so that spelling is unavailable. Use "
    "numba.set_num_threads(k) or NUMBA_NUM_THREADS to choose the count.")


def _brute_reshape_grid(grid, N, Ns):
    """The ``(M, N)`` sweep as scipy's `grid`.

    scipy builds `grid` with ``numpy.mgrid``, so it is ``(N,) + (Ns,)*N`` for
    ``N > 1`` and the bare ``(Ns,)`` axis for ``N == 1``. The sweep is the
    same points in the same order, one per row.

    Python only. A compiled function cannot return an array whose ndim
    follows a runtime value, so the ``@njit`` entry returns the ``(M, N)``
    form and the reshape recipe is in `brute`'s `Notes`.
    """
    if N == 1:
        return np.ascontiguousarray(grid[:, 0])
    return np.ascontiguousarray(grid.T).reshape((N,) + (Ns,) * N)


def brute(func, ranges, args=(), Ns=20, full_output=0, finish=fmin,
          disp=False, workers=1, xtol=1e-4, ftol=1e-4):
    """Brute-force grid search over a full Cartesian grid.

    Evaluates ``func`` on a full Cartesian grid and returns the best point,
    optionally polished by a local minimiser.  No randomness is involved: the
    same call gives the same answer every time.

    **Callback style A** -- ``func`` is a plain ``@njit``
    ``func(x, *args) -> float64``.

    Parameters
    ----------
    func : callable
        A plain ``@njit`` ``f(x, *args) -> float``.
    ranges : (N, 2) array_like
        One ``(min, max)`` pair per dimension. ``slice`` objects are not
        accepted; see Notes.
    args : tuple or ndarray, optional
        Extra parameters. A tuple is unpacked into separate arguments after
        `x`, so ``args=(a, b)`` calls ``func(x, a, b)``. Its elements may be
        of any type a compiled call accepts, arrays and strings included.
        Default ``()``, which calls ``func(x)``. An ndarray or a list
        arrives as ONE argument instead, ``func(x, args)``. See Notes.
    Ns : int, optional
        Points per axis. Default 20.
    full_output : bool, optional
        ``False`` (default) returns ``x0`` alone. ``True`` returns
        ``(x0, fval, grid, Jout)``. Compile-time constant inside ``@njit``.
    finish : callable or None, optional
        The minimiser that polishes the best grid point. ``None`` skips it;
        :func:`~scijit.optimize.fmin` (default) runs Nelder-Mead. Any other
        minimiser raises. Resolved when the call compiles, so it is a name at
        the call site rather than a variable. See Notes.
    disp : bool, optional
        Reaches the `finish` minimiser, which then prints its own summary, and
        turns on a ``RuntimeWarning`` for a polish that reports a non-zero
        status. Default ``False``.
    workers : int, optional
        ``1`` (default) evaluates the grid in order on one thread. Any
        other value evaluates it in a ``prange``; ``-1`` selects all cores.
        The result is BIT-IDENTICAL either way: the grid points are
        independent and nothing is reduced across them.

        The VALUE does not cap the thread count; numba's own
        ``NUMBA_NUM_THREADS`` or ``numba.set_num_threads(k)`` does.
        Compile-time literal, because ``prange`` needs ``parallel=True`` at
        compile time. A map-like callable in `workers` is not accepted; see
        Notes.

        Worth it only when the grid is large enough to cover the fixed cost
        of entering a parallel region: a few hundred microseconds. A
        20x20 grid of a cheap objective is slower in parallel.
    xtol, ftol : float, optional
        Passed to the ``finish`` minimiser. Both default to ``1e-4``. See
        Notes.

    Returns
    -------
    x0 : ndarray, shape (N,)
        Minimizer -- the best grid point, or the polished point.
    fval : float
        ``func`` at ``x0``. With ``full_output``.
    grid : ndarray, shape (M, N)
        Every grid point, C order. With ``full_output``. Reshape to
        ``(N,) + (Ns,)*N`` with ``grid.T.reshape((N,) + (Ns,)*N)``; see
        Notes.
    Jout : ndarray, shape (M,)
        ``func`` at every grid point. With ``full_output``. Reshape with
        ``Jout.reshape((Ns,)*N)``.

    Raises
    ------
    ValueError
        An empty `ranges`, `ranges` that is not ``(N, 2)``, more than 40
        variables, ``Ns < 1``, a grid of more than ``2**31`` points, a
        `finish` that is neither ``None`` nor
        :func:`~scijit.optimize.fmin`, or a non-integer `workers`. `ranges`,
        `finish` and `workers` raise from Python;
        inside ``@njit`` `finish` raises the same ``ValueError`` while
        `workers` and `full_output` raise ``TypingError`` carrying the same
        message, since both are read when the call compiles.

    See Also
    --------
    scipy.optimize.brute : The scipy routine this mirrors.
    scijit.optimize.differential_evolution : Stochastic, no grid.
    scijit.optimize.basinhopping : Random restarts around a local minimizer.
    scijit.optimize.fmin : The polish step ``finish=fmin`` runs.

    Notes
    -----
    `finish` takes two spellings, ``None`` and the minimiser itself,
    from Python and from inside ``@njit``. A minimiser other than
    :func:`~scijit.optimize.fmin` raises, and so does one chosen at RUN
    TIME, since the name is read when the call compiles.

    UNAVAILABLE: ``slice`` objects in `ranges`, which take ``(min, max)``
    pairs instead.

    UNAVAILABLE: a ``grid`` whose ndim follows ``N``. It and `Jout` are flat
    in C order, since a compiled function cannot return an array whose ndim
    depends on a runtime value.

    UNAVAILABLE: a map-like callable in `workers`. It is an int literal, and
    any other value raises.

    ``Ns < 1`` and a grid above ``2**31`` points raise before the grid is
    built. scipy builds a zero-point grid and fails later inside ``argmin``,
    with numpy's message.

    DELIBERATELY DIFFERENT: ``N == 1`` with ``finish=None`` returns a
    length-1 array where scipy returns a float. A numba function has one
    return type and ``N`` is a runtime value. With a polish scipy also
    returns an array, so the two agree there.

    `xtol` and `ftol` reach the `finish` minimiser and both default to
    ``1e-4``.

    `args` is unpacked into the objective's argument list, ``func(x, *args)``.

    DELIBERATELY DIFFERENT: an ndarray or a list `args` reaches `func` as ONE
    argument, ``func(x, args)``. scipy unpacks those two element by element.
    The arity of a compiled call is fixed when it compiles, so a sequence
    whose length is known only at run time cannot be unpacked:
    ``func(x, *args)`` on an ndarray inside ``@njit`` is a ``TypingError``.
    An objective written to the unpacked shape refuses the ndarray rather than
    reading it wrongly.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import brute
    >>> @njit
    ... def q(x):
    ...     return (x[0] - 1.5) ** 2 + (x[1] + 0.5) ** 2
    >>> ranges = np.array([[-3.0, 3.0], [-3.0, 3.0]])
    >>> @njit
    ... def run():
    ...     return brute(q, ranges)
    >>> np.round(run(), 4)
    array([ 1.5, -0.5])
    """
    fin = _finish_code(finish)
    if not isinstance(workers, (int, np.integer)) or isinstance(workers, bool):
        # scipy's map-like callable, or anything else that is not an int
        raise ValueError(_BRUTE_WORKERS_MSG)
    lo, hi = _brute_split_ranges(ranges)
    a = args
    axes, N, M = _brute_axes(lo, hi, Ns)
    if workers == 1:
        grid, Jout = _brute_grid(func, axes, a, N, Ns, M)
    else:
        grid, Jout = _brute_grid_par(func, axes, a, N, Ns, M)
    x0, fval = _brute_finish(func, grid, Jout, a, fin, xtol, ftol, N, M,
                             disp)
    if N == 1 and fin == 0:
        # scipy's `xmin = xmin[0]` at N == 1 without a polish. With one,
        # `xmin` is whatever the minimiser returned, which is an array.
        x0 = x0[0]
    if full_output:
        return x0, fval, _brute_reshape_grid(grid, N, Ns), Jout.reshape(
            (Ns,) * N)
    return x0


@overload(brute, prefer_literal=True)
def _brute_ovl(func, ranges, args=(), Ns=20, full_output=0, finish=fmin,
               disp=False, workers=1, xtol=1e-4, ftol=1e-4):
    """@njit implementation of `brute`, resolved at compile time.

    ``full_output`` selects between a bare ``x0`` and a 4-tuple, and
    ``workers`` selects the serial or the ``prange`` grid sweep -- a
    return type and a loop form, neither of which survives being a
    runtime variable. Returning ``None`` declines the call, which numba
    reports as a TypingError naming the argument that could not be
    served.
    """
    fo = _lit_bool_g(full_output)
    w = _lit_int_g(workers)
    # Declining by `return None` reports a failed dispatch that lists every
    # argument type and points at none of them.  Raising the python body's own
    # message names the argument, so the two entries give one reason.
    if w is None:
        raise TypingError(_BRUTE_WORKERS_MSG)
    if fo is None:
        raise TypingError(_BRUTE_FULL_OUTPUT_MSG)
    par = w != 1
    # `finish` names a minimiser, so it is resolved here rather than in the
    # body: `None` and `fmin` are types, not values a compiled comparison
    # can reach.
    FIN = _finish_code_t(finish)
    if FIN is None:
        # Refused HERE, beside the two refusals above, rather than by an impl
        # that raises: an impl whose body is only a `raise` TYPES AS RETURNING
        # `none`, so a caller that unpacks the `full_output` 4-tuple or writes
        # `r[0]` fails on the return type first and the message is never seen.
        # Measured: "failed to unpack none".
        raise TypingError(_BRUTE_FINISH_MSG)

    def impl(func, ranges, args=(), Ns=20, full_output=0, finish=fmin,
             disp=False, workers=1, xtol=1e-4, ftol=1e-4):
        lo, hi = _brute_split_ranges(ranges)
        a = args
        axes, N, M = _brute_axes(lo, hi, Ns)
        if par:
            grid, Jout = _brute_grid_par(func, axes, a, N, Ns, M)
        else:
            grid, Jout = _brute_grid(func, axes, a, N, Ns, M)
        x0, fval = _brute_finish(func, grid, Jout, a, FIN, xtol, ftol,
                                 N, M, disp)
        if fo:
            return x0, fval, grid, Jout
        return x0
    return impl


# ============================================================= basinhopping

@njit
def _bh_metropolis(st, gen, fq, energy, T):
    """scipy's ``Metropolis.accept_reject``.

    An RNG draw is consumed whether or not the move is downhill, exactly as
    in scipy (``w = exp(min(0, -(f_new - f_old) * beta)); return w >= u``).
    ``T == 0`` means ``beta = inf`` -> reject every uphill move.

    scipy spells the draw ``self.rng.uniform()``, so under a `rng` generator
    it is one ``uniform()`` here too.
    """
    u = _dr_u01(st, gen)
    if T == 0.0:
        return fq < energy
    prod = -(fq - energy) / T
    # scipy computes `w = exp(min(0, prod))`, and PYTHON's `min(0, nan)`
    # returns 0 -- `nan < 0` is False, so `min` keeps its first argument.
    # So a NaN difference gives w = 1.0 and the move is ACCEPTED. Writing
    # the branch as `prod > 0.0 -> 1.0 else exp(prod)` instead sends NaN
    # down the exp path, giving w = nan and `nan >= u` False, i.e. a
    # rejection. Spelled as `not (prod < 0.0)` the NaN lands on 1.0, as
    # scipy's does. Measured: an objective that is NaN for x[0] > 1, started
    # at (2, 2) -- scipy escapes the NaN region and reaches the minimum,
    # while rejecting left the walk pinned at x0 with fun = nan forever.
    if not (prod < 0.0):
        w = 1.0
    else:
        w = np.exp(prod)
    return w >= u


@njit
def _bh_adapt(stepsize, nstep, naccept, interval, target_accept_rate,
              stepwise_factor):
    """scipy's ``AdaptiveStepsize`` (cumulative counters, never reset).

    The test is spelled exactly as scipy spells it, `nstep % interval == 0`,
    and NOT as `interval > 0 and ...`. `interval` is a divisor, and Python's
    modulo returns 0 on an exact division whatever the divisor's sign, so
    `-k` fires on the same steps as `+k` and `-1` fires on every step. The
    `interval > 0` guard turned every negative value into "never adapt".
    Measured over 30 steps against scipy's own AdaptiveStepsize, now
    bit-identical at intervals -1, 1, -5, 5, -3, 3, -50 and 50; before, the
    four negative ones all returned the initial 0.5 where scipy returned
    0.021195579137608122, 0.2657205000000001, 0.17433922005000008 and 0.5.
    numba's `%` agrees with Python's for a negative divisor, so scipy's
    spelling is directly expressible. `interval == 0` is rejected by the
    callers, where the argument still has a name to put in the message.
    """
    if nstep % interval == 0:
        rate = np.float64(naccept) / np.float64(nstep)
        if rate > target_accept_rate:
            return stepsize / stepwise_factor
        return stepsize * stepwise_factor
    return stepsize


def _bh_step0_line(nstep, energy):
    """scipy's ``BasinHoppingRunner.__init__`` disp line."""
    print(f"basinhopping step {nstep}: f {energy:g}")


@njit
def _bh_print_step0(nstep, energy):
    with objmode():
        _bh_step0_line(nstep, energy)


def _bh_report_line(nstep, energy, energy_trial, accept, lowest):
    """scipy's ``BasinHoppingRunner.print_report``."""
    print(f"basinhopping step {nstep}: f {energy:g} "
          f"trial_f {energy_trial:g} accepted {accept} "
          f"lowest_f {lowest:g}")


@njit
def _bh_print_report(nstep, energy, energy_trial, accept, lowest):
    with objmode():
        _bh_report_line(nstep, energy, energy_trial, accept, lowest)


def _bh_newmin_line(nstep, energy):
    print(f"found new global minimum on step {nstep} with "
          f"function value {energy:g}")


@njit
def _bh_print_newmin(nstep, energy):
    with objmode():
        _bh_newmin_line(nstep, energy)


def _bh_adapt_line(accept_rate, target, new_ss, old_ss):
    """scipy's ``AdaptiveStepsize._adjust_step_size`` line."""
    print(f"adaptive stepsize: acceptance rate {accept_rate:f} target "
          f"{target:f} new stepsize {new_ss:g} old stepsize {old_ss:g}")


@njit
def _bh_print_adapt(accept_rate, target, new_ss, old_ss):
    with objmode():
        _bh_adapt_line(accept_rate, target, new_ss, old_ss)


def _bh_fail_line():
    print("warning: basinhopping: local minimization failure")


@njit
def _bh_print_fail():
    with objmode():
        _bh_fail_line()


#: scipy's ``basinhopping`` OptimizeResult, as a namedtuple.
#: ``lowest_optimization_result`` is absent: it is a NESTED OptimizeResult
#: whose key set varies with the local minimizer, which has no numba
#: representation.
#:
#: ``message`` is a ``numba.typed.List`` holding one string, which scipy's
#: is too. One container from both entry points, so ``res.message[0]`` is
#: scipy's sentence from Python and from inside ``@njit``.
#:
#: ``njev`` is ALWAYS present, and is ``0`` on the derivative-free
#: minimizers.  scipy's result carries the key only when the quench used a
#: gradient, and can, because its result is a dict.  A compiled function has
#: one return type.
BasinhoppingResult = _opt_result(
    ['x', 'fun', 'nit', 'nfev', 'njev', 'minimization_failures', 'success',
     'message'])

#: scipy's ``message`` is a LIST of strings, and so is this one; the strings
#: themselves are scipy's own.
_BH_MSG_OK = ("requested number of basinhopping iterations completed"
              " successfully")
_BH_MSG_SUCC = ("success condition satisfied")
#: scipy's own text, `_basinhopping.py:723-724`, whose two fragments are
#: concatenated without a space.
_BH_MSG_CB = ("callback function requested stop early byreturning True")

@njit
def _bh_msg_list(msg):
    """scipy's `message`: a LIST holding the one string.

    A ``numba.typed.List`` rather than a python ``list``, because the one
    body both entry points reach is compiled, so the same container crosses
    back to Python. Indexing and iteration serve ``res.message[0]`` on both
    sides.
    """
    out = List.empty_list(types.unicode_type)
    out.append(msg)
    return out


_BH_SCOPE_MSG = (
    "basinhopping: take_step and accept_test must be None. Both are Python "
    "callables; `stepsize`, `interval`, `target_accept_rate` and "
    "`stepwise_factor` cover scipy's own step taking and acceptance.")

_BH_MK_MSG = (
    "basinhopping: minimizer_kwargs takes 'method' and 'jac' only. scipy "
    "forwards the whole dict to `minimize`; here `args`, `xtol`, `ftol` and "
    "`gtol` are explicit arguments of `basinhopping` itself.")

_BH_METHOD_MSG = (
    "basinhopping: minimizer_kwargs['method'] must be 'Nelder-Mead', "
    "'Powell', 'BFGS' or 'CG'. 'BFGS' is the default and is scipy's.")

#: The four local minimizers, as the integer the core branches on.
_BH_METHODS = ('Nelder-Mead', 'Powell', 'BFGS', 'CG')


@njit
def _bh_quench(func, grad, x, args, mcode, xtol, ftol, gtol):
    """One local minimization, by whichever of the four `mcode` names.

    Returns ``(xq, fq, nfev, njev, warnflag)``, with ``njev`` zero on the
    derivative-free pair. Written once rather than per minimizer so the
    Monte-Carlo layer above cannot drift between them.

    `grad` is a plain ``@njit`` gradient or ``None``. ``None`` reaches
    `fmin_bfgs` and `fmin_cg` as their own ``fprime=None``, which is their
    forward-difference fallback and is scipy's default quench. The two
    derivative-free branches ignore `grad` entirely, as scipy's do.

    The quenches reach `fmin`'s and `fmin_powell`'s CORES rather than their
    public entries: those run hundreds of times per basin-hopping call and
    the public entries print scipy's summary at the scipy default ``disp=1``.
    """
    if mcode == 0:
        xq, fq, itn, nf, wf = fmin(func, x, args, xtol, ftol, None,
                                   None, True, 0)
        ng = 0
    elif mcode == 1:
        xq, fq, itn, nf, wf = _powell5(func, x, args, xtol, ftol)
        ng = 0
    elif mcode == 2:
        xq, fq, gq, Bq, nf, ng, wf = fmin_bfgs(
            func, x, grad, args, gtol, full_output=True, disp=0)
    else:
        xq, fq, nf, ng, wf = fmin_cg(
            func, x, grad, args, gtol, full_output=True, disp=0)
    return xq, fq, nf, ng, wf


@njit
def _bh_core(args, niter, T, stepsize, seed, interval, niter_success,
             target_accept_rate, stepwise_factor,
             f_first, x_first, ok_first, func, grad, mcode,
             xtol, ftol, gtol, disp, cb=_bh_noop, use_cb=False, gen=None):
    """Basin-hopping Monte-Carlo driver, over all four local minimizers.

    Returns ``(xlowest, flowest, nit, nfev, nfail, njev, lowest_ok,
    cb_stop)``.
    `niter_success` is the resolved iteration budget: the entry points turn
    scipy's ``None`` into ``niter + 2``, so every value here is a real
    threshold and ``-1`` means what scipy means by it.

    This was two drivers until 2026-08-03, one per minimizer style, because a
    single one branching at runtime was held to force the derivative-free
    path to hand a dummy gradient to `fmin_bfgs`. Measured on numba 0.66:
    ``None`` types there, so one driver serves all four.
    """
    n = x_first.shape[0]
    st = _rng_init(seed)

    x = x_first.copy()
    energy = f_first
    xlowest = x_first.copy()
    flowest = f_first
    lowest_ok = ok_first

    nstep = 0
    naccept = 0
    count = 0
    nit = 0
    nfev = 0                  # scipy counts EVERY objective call
    nfail = 0                 # scipy's `minimization_failures`
    njev = 0                  # zero unless the quench used a gradient
    nsucc = niter_success
    cb_stop = False

    xnew = np.empty(n, dtype=np.float64)

    for _it in range(niter):
        nit += 1
        nstep += 1
        old_ss = stepsize
        stepsize = _bh_adapt(stepsize, nstep, naccept, interval,
                             target_accept_rate, stepwise_factor)
        if disp and nstep % interval == 0:
            _bh_print_adapt(np.float64(naccept) / np.float64(nstep),
                            target_accept_rate, stepsize, old_ss)

        # scipy's `RandomDisplacement.__call__`:
        # `x += rng.uniform(-stepsize, stepsize, np.shape(x))`, one vector
        # draw rather than n scalar ones.
        _dr_uvec(st, gen, -stepsize, stepsize, xnew)
        for k in range(n):
            xnew[k] += x[k]

        xq, fq, nf, ng, wf = _bh_quench(func, grad, xnew, args, mcode,
                                        xtol, ftol, gtol)
        success = (wf == 0)
        nfev += nf
        njev += ng
        if not success:
            nfail += 1
            if disp:
                _bh_print_fail()

        accept = _bh_metropolis(st, gen, fq, energy, T)

        new_global_min = False
        if accept:
            naccept += 1
            energy = fq
            for k in range(n):
                x[k] = xq[k]
            if success and (fq < flowest or not lowest_ok):
                flowest = fq
                for k in range(n):
                    xlowest[k] = xq[k]
                lowest_ok = True
                new_global_min = True

        if disp:
            _bh_print_report(nstep, energy, fq, accept, flowest)
            if new_global_min:
                _bh_print_newmin(nstep, energy)

        # scipy calls the callback here, on the trial point rather than the
        # accepted one, and halts on a truthy return before the
        # niter_success test.
        if use_cb:
            if cb(xq.copy(), fq, accept):
                cb_stop = True
                break

        count += 1
        if new_global_min:
            count = 0
        elif count > nsucc:
            break

    return xlowest, flowest, nit, nfev, nfail, njev, lowest_ok, cb_stop


@njit
def _bh_mcode(method):
    """A local-minimizer NAME as the integer `_bh_core` branches on."""
    if method == 'Nelder-Mead':
        return 0
    elif method == 'Powell':
        return 1
    elif method == 'BFGS':
        return 2
    elif method == 'CG':
        return 3
    raise ValueError(_BH_METHOD_MSG)


@njit
def _bh_run(func, grad, x0, niter, T, stepsize, interval, niter_success,
            target_accept_rate, stepwise_factor, args, seed, mcode,
            xtol, ftol, gtol, disp, cb=_bh_noop, use_cb=False, rng=None):
    """Guards, the first quench, the driver, and the result. THE algorithm.

    Both entry points call only this, so there is no second implementation
    for them to disagree with. `rng` and `seed` are resolved here for the
    same reason.
    """
    xf = np.asarray(x0).astype(np.float64).copy()
    if xf.ndim != 1:
        raise ValueError("basinhopping: x0 must only have one dimension")
    n = xf.shape[0]
    if n == 0:
        raise ValueError("basinhopping: x0 must be non-empty")
    if niter < 0:
        raise ValueError("basinhopping: niter must be >= 0")
    if stepsize <= 0.0:
        raise ValueError("basinhopping: stepsize must be > 0")
    if target_accept_rate <= 0.0 or target_accept_rate >= 1.0:
        raise ValueError('target_accept_rate has to be in range (0, 1)')
    if stepwise_factor <= 0.0 or stepwise_factor >= 1.0:
        raise ValueError('stepwise_factor has to be in range (0, 1)')
    # `interval` divides the step counter, so scipy dies on
    # `nstep % interval`. The class is scipy's; the message names the argument
    # instead of repeating "integer modulo by zero". Every OTHER value,
    # negative included, reaches `_bh_adapt` and gets scipy's own arithmetic.
    if interval == 0:
        raise ZeroDivisionError(
            "basinhopping: interval must be nonzero; it is the period of the "
            "step-size adaptation and divides the iteration counter")

    x1, f1, nf, ng, wf = _bh_quench(func, grad, xf, args, mcode,
                                    xtol, ftol, gtol)
    # scipy counts the FIRST quench's failure too, in
    # `BasinHoppingRunner.__init__`, before any iteration runs.
    nfail0 = 0
    if wf != 0:
        nfail0 = 1
        if disp:
            _bh_print_fail()
    if disp:
        _bh_print_step0(0, f1)
    # scipy runs the callback once on the initial quench, before the loop,
    # and does NOT read its return there.
    if use_cb:
        cb(x1.copy(), f1, True)
    gen = _rng_gen(rng)
    xb, fb, nit, nfev, nfail, njev, low_ok, cb_stop = _bh_core(
        args, niter, T, stepsize, _seed_value(seed), interval, niter_success,
        target_accept_rate, stepwise_factor, f1, x1, wf == 0, func, grad,
        mcode, xtol, ftol, gtol, disp, cb, use_cb, gen)
    # scipy counts the initial quench too, and reports nit=1 for niter=0
    # because the local minimizer always runs once from x0.
    nfev += nf
    njev += ng
    nfail += nfail0
    if nit == 0:
        nit = 1
    msg = _BH_MSG_OK
    if cb_stop:
        msg = _BH_MSG_CB
    elif nit < niter:
        msg = _BH_MSG_SUCC
    # scipy's `res.success = res.lowest_optimization_result.success`, i.e. the
    # success flag of the minimization that produced the reported point.
    return BasinhoppingResult(xb, fb, nit, nfev, njev, nfail, low_ok,
                              _bh_msg_list(msg))


def _args_or_empty(args):
    """``None`` -> the empty tuple, anything else unchanged.

    Resolved when the call compiles, not when it runs.  A runtime
    ``() if args is None else args`` asks numba to unify an empty tuple with
    whatever `args` is, which fails for every array and every non-empty
    tuple.
    """
    return () if args is None else args


@overload(_args_or_empty)
def _args_or_empty_ovl(args):
    if isinstance(args, (types.NoneType, types.Omitted)):
        return lambda args: ()
    return lambda args: args


def _bh_mk_py(minimizer_kwargs):
    """scipy's ``minimizer_kwargs`` dict -> ``(mcode, jac)``. Python entry."""
    if minimizer_kwargs is None:
        return 2, None                       # scipy's default quench: BFGS
    for k in minimizer_kwargs:
        if k not in ('method', 'jac'):
            raise ValueError(_BH_MK_MSG)
    method = minimizer_kwargs.get('method', 'BFGS')
    if method not in _BH_METHODS:
        raise ValueError(_BH_METHOD_MSG)
    return _BH_METHODS.index(method), minimizer_kwargs.get('jac', None)


def _bh_mk_keys(mk):
    """The KEYS of a `minimizer_kwargs` argument, read at typing time.

    Returns the key list, or ``None`` when the argument is not a dict this
    can read. Two numba types arrive here and missing either one looks like
    the feature not working: a HETEROGENEOUS literal, ``{'method': 'BFGS',
    'jac': g}``, is a ``LiteralStrKeyDict`` and carries its keys as plain
    strings in ``.fields``; a HOMOGENEOUS one, ``{'method': 'Powell'}``, is
    an ordinary ``DictType`` whose keys are only known at run time.
    """
    if isinstance(mk, types.LiteralStrKeyDict):
        return list(mk.fields)
    if isinstance(mk, types.DictType):
        return []                            # keys unknown until run time
    return None


from .._lib._typing import _is_none as _is_none_g    # noqa: E402


def _bh_scope_ok(*vals):
    """True when every unimplemented hook given is at ``None``.

    Reads numba TYPES as well as plain values, so the one predicate serves
    the python entry and the chooser.
    """
    for v in vals:
        if isinstance(v, types.Omitted):
            v = v.value
        if not (v is None or isinstance(v, types.NoneType)):
            return False
    return True


def basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5,
                 minimizer_kwargs=None, take_step=None, accept_test=None,
                 callback=None, interval=50, disp=False, niter_success=None,
                 rng=None, target_accept_rate=0.5, stepwise_factor=0.9,
                 args=None, seed=None, xtol=1e-4, ftol=1e-4, gtol=1e-5):
    """Basin-hopping global optimization by random restarts.

    Alternates a random displacement with a local minimization, keeping or
    rejecting each new basin by a Metropolis test.

    **Callback style A** -- ``func`` and the gradient are plain ``@njit``
    functions passed as first-class arguments. No ``@cfunc``, no
    ``.address``.

    Parameters
    ----------
    func : @njit function ``func(x, *args) -> float64``
        Objective.
    x0 : float64 array, shape (N,)
        Starting point. Cast to float64 and copied.
    niter : int, optional
        Basin-hopping iterations. Default 100.
    T : float, optional
        Metropolis temperature. Default 1.0. ``T = 0`` makes beta infinite,
        so every uphill move is rejected. ``T < 0`` inverts the test.
    stepsize : float, optional
        Initial random-displacement size. Default 0.5. Must be positive.
    minimizer_kwargs : dict, optional
        The local minimizer and its gradient. ``'method'`` is one of
        ``'Nelder-Mead'``, ``'Powell'``, ``'BFGS'`` or ``'CG'``, defaulting to
        ``'BFGS'``; ``'jac'`` is a plain ``@njit``
        ``grad(x, *args) -> float64 array``. Omitting ``'jac'`` under a
        gradient method uses forward differences. Inside ``@njit`` the dict
        must be written as a LITERAL at the call site. See Notes.
    take_step, accept_test : None
        Accepted only at their defaults. See Notes.
    callback : callable, optional
        Called once on the initial quench and once per iteration, as
        ``callback(x, f, accept)`` on the TRIAL point, and halts the run on a
        truthy return. Two spellings are served: a plain Python callable, and
        a numba ``@njit`` ``callback(x, f, accept)`` returning a bool. See
        Notes.
    interval : int, optional
        Period of the step-size adaptation, in iterations. Default 50. The
        step is adapted whenever ``nstep % interval == 0``, so the sign
        carries no meaning. ``0`` raises.
    disp : bool, optional
        Print the per-step lines, the step-size adaptation line, the
        new-global-minimum line and the local-minimization-failure warning.
        Default ``False``.
    niter_success : int or None, optional
        Stop after this many consecutive iterations without improving the
        global minimum. ``None`` (default) uses ``niter + 2``, which the run
        can still reach when it breaks early. Every integer is a real
        threshold, ``-1`` included: it stops after the first iteration that
        does not improve the global minimum.
    rng : int or numpy.random.Generator, optional
        Source of the displacement and Metropolis draws. ``None`` (default)
        uses the internal xorshift64* generator; an integer builds a
        ``numpy.random.default_rng`` for the call; a ``Generator`` is drawn
        from directly and its state advances. See Notes.
    target_accept_rate : float, optional
        Acceptance rate the adaptive step size aims at. Default 0.5. Must lie
        in ``(0, 1)``.
    stepwise_factor : float, optional
        Multiplicative factor applied when adapting the step. Default 0.9.
        Must lie in ``(0, 1)``.
    args : tuple or ndarray, optional
        Extra parameters forwarded to `func` and to the gradient. A tuple is
        unpacked into separate arguments after `x`, so ``args=(a, b)`` calls
        ``func(x, a, b)``. Its elements may be of any type a compiled call
        accepts, arrays and strings included. ``None`` (default) calls
        ``func(x)``. An ndarray or a list arrives as ONE argument instead,
        ``func(x, args)``. See Notes.
    seed : int, optional
        BEYOND SCIPY. Seeds the internal xorshift64* generator, which is the
        `rng` at ``None`` path. ``None`` (default) draws fresh entropy per
        call, so two identical calls give different answers. See Notes.
    xtol, ftol : float, optional
        Tolerances handed to ``'Nelder-Mead'`` and ``'Powell'``. Both 1e-4.
    gtol : float, optional
        Gradient tolerance handed to ``'BFGS'`` and ``'CG'``. Default 1e-5.

    Returns
    -------
    res : BasinhoppingResult
        A namedtuple carrying ``x``, ``fun``, ``nit``, ``nfev``, ``njev``,
        ``minimization_failures``, ``success`` and ``message``. ``message``
        is a list holding one string, so ``res.message[0]`` is the sentence.
        ``njev`` is
        ``0`` under ``'Nelder-Mead'`` and ``'Powell'``. ``success`` is the
        success flag of the local minimization that produced ``x``.
        ``minimization_failures`` counts the quench from `x0` as well as the
        per-iteration ones.

    Raises
    ------
    ValueError
        A 2-D or empty `x0`, ``niter < 0``, ``stepsize <= 0``, a
        `target_accept_rate` or `stepwise_factor` outside ``(0, 1)``, an
        unknown ``'method'``, a key in `minimizer_kwargs` other than
        ``'method'`` and ``'jac'``, `take_step` or `accept_test` away from
        ``None``, a `rng` that is not ``None``, an integer or a
        ``numpy.random.Generator``, or a `callback` that is a ``@cfunc``, a
        raw function address or a non-callable.
    ZeroDivisionError
        ``interval == 0``. `interval` divides the step counter.

    See Also
    --------
    scipy.optimize.basinhopping : The scipy routine this mirrors.
    scijit.optimize.differential_evolution : Population-based, no start point.
    scijit.optimize.brute : Exhaustive grid search.
    scijit.optimize.minimize : One local minimization, no restarts.

    Notes
    -----
    `minimizer_kwargs` carries ``'method'`` and ``'jac'`` only. scipy forwards
    the whole dict to ``minimize``; here `args`, `xtol`, `ftol` and `gtol` are
    explicit arguments of ``basinhopping`` itself.

    ADDITIVE: `args` is a top-level parameter. scipy publishes none, and
    raises ``TypeError`` for one. A tuple is unpacked into the objective's
    argument list, ``func(x, *args)``; an ndarray or a list reaches `func` as
    ONE argument instead, because the arity of a compiled call is fixed when
    it compiles.

    Inside ``@njit`` the dict must be written as a LITERAL at the call site.
    A dict built elsewhere and passed in as a variable raises ``TypingError``:
    a literal is typed where it is written, carrying its keys and a callable
    value, while a variable has to be unboxed and a numba dict cannot hold a
    function.

    A Python `callback` is reached from compiled code through a module-level
    slot and a ``numba.objmode`` block, so it takes the GIL and pays an
    interpreter round trip once per iteration. It is also not ``prange``-safe,
    because the slot is module state that two concurrent runs share. Inside ``@njit`` only the
    ``@njit`` spelling is accepted, since a Python callable cannot cross into
    compiled code as an argument. The ``@njit`` spelling must RETURN a bool;
    a Python one may return ``None``, which scipy reads as no halt.

    NOT IMPLEMENTED: `take_step` and `accept_test`. Both are Python
    callables. Step taking, adaptation and acceptance reproduce scipy's
    ``RandomDisplacement`` + ``AdaptiveStepsize`` + ``Metropolis`` with all
    their defaults, including consuming an RNG draw on downhill moves.

    UNAVAILABLE: ``lowest_optimization_result``. It is a nested result object
    whose field set varies with the local minimizer.

    ADDITIVE: `njev` is always present and is ``0`` on the derivative-free
    minimizers. scipy carries the key only when the quench used a gradient,
    and can, because its result is a dict.

    With `rng` at ``None`` the draws come from an internal xorshift64*
    generator carried in a per-call state array. `seed` selects its stream
    and ``seed=None`` draws a fresh one per call.

    DELIBERATELY DIFFERENT: an integer `rng` builds a
    ``numpy.random.default_rng``, which is PCG64. scipy's
    ``check_random_state`` maps an integer to the legacy ``RandomState``,
    which is MT19937 and which compiled code cannot hold, so the two explore
    differently at the same integer. A ``Generator`` passed in is used
    directly, and its draws are numpy's own.

    A `rng` at ``None`` is safe to call from a ``numba.prange`` loop, because
    the generator state is per-call. One ``Generator`` SHARED across a
    ``prange`` is not: 70,499 of 100,000 draws were distinct across 32
    threads. An integer `rng` builds its generator inside the call, so it is
    per-call as well. A Python `callback` is not ``prange``-safe either, and
    serializes the loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import basinhopping
    >>> @njit
    ... def q(x):
    ...     return (x[0] - 1.5) ** 2 + (x[1] + 0.5) ** 2
    >>> @njit
    ... def run():
    ...     return basinhopping(q, np.array([2.0, 2.0]), 20)
    >>> res = run()
    >>> np.round(res.x, 4)
    array([ 1.5, -0.5])
    >>> res.njev > 0
    True
    """
    if not _bh_scope_ok(take_step, accept_test):
        raise ValueError(_BH_SCOPE_MSG)
    cbf, use_cb, pycb = _cb_resolve('basinhopping', callback, bh=True)
    mcode, grad = _bh_mk_py(minimizer_kwargs)
    a = _args_or_empty(args)
    ns = niter + 2 if niter_success is None else int(niter_success)
    prev = _cb_install(pycb)
    try:
        return _bh_run(func, grad, x0, niter, T, stepsize, interval,
                       ns, target_accept_rate, stepwise_factor,
                       a, seed, mcode, xtol, ftol, gtol, disp, cbf, use_cb,
                       _rng_gen(rng))
    finally:
        _cb_release(prev)


@overload(basinhopping, prefer_literal=True)
def _basinhopping_ovl(func, x0, niter=100, T=1.0, stepsize=0.5,
                      minimizer_kwargs=None, take_step=None, accept_test=None,
                      callback=None, interval=50, disp=False,
                      niter_success=None, rng=None, target_accept_rate=0.5,
                      stepwise_factor=0.9, args=None, seed=None, xtol=1e-4,
                      ftol=1e-4, gtol=1e-5):
    """@njit implementation of `basinhopping`, resolved at compile time.

    `minimizer_kwargs` is a type branch, not a value: whether ``'jac'`` is
    present decides whether a function or ``None`` reaches the quench, and
    those are different types. Everything algorithmic stays in `_bh_run`.

    Two things are decided here rather than there, and both have a python
    twin above that has to say the same thing: `niter_success` at ``None``
    becomes ``niter + 2``, and an unknown key in `minimizer_kwargs` is
    refused. A HOMOGENEOUS dict does not carry its keys at typing time, so
    that refusal is emitted as a run-time loop instead.
    """
    keys = _bh_mk_keys(minimizer_kwargs)
    mk_none = _bh_scope_ok(minimizer_kwargs, None, None, None)
    if keys is None and not mk_none:
        return None                          # not a dict -> TypingError
    if not _bh_scope_ok(take_step, accept_test):
        # Refused HERE rather than by an impl that raises: such an impl TYPES
        # AS RETURNING `none`, so a caller that reads `res.x` fails on the
        # return type first and the message is never seen. Measured:
        # "Unknown attribute 'x' of type none".
        raise TypingError(_BH_SCOPE_MSG)
    if keys is not None and any(k not in ('method', 'jac') for k in keys):
        # Same route as `bad_scope` above, and it was LIVE here: with a
        # LiteralStrKeyDict, which is the only dict whose keys this chooser
        # can read, `res.x` reported "Unknown attribute 'x' of type none" and
        # the message was lost. A HOMOGENEOUS dict is unaffected: its keys are
        # invisible at typing time, so its refusal is a run-time loop in the
        # body and already carried the ValueError.
        raise TypingError(_BH_MK_MSG)
    # A LiteralStrKeyDict reports its keys here; a plain DictType cannot, so
    # `keys == []` means "homogeneous, read the keys at RUN time". Such a
    # dict holds one value type and so cannot carry 'jac' at all.
    CBF, USE_CB = _cb_resolve_ty('basinhopping', callback, bh=True)
    rt_keys = keys == []
    has_method = mk_none is False and 'method' in (keys or ())
    has_jac = keys is not None and 'jac' in keys
    ns_none = _is_none_g(niter_success)

    def impl(func, x0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs=None,
             take_step=None, accept_test=None, callback=None, interval=50,
             disp=False, niter_success=None, rng=None, target_accept_rate=0.5,
             stepwise_factor=0.9, args=None, seed=None, xtol=1e-4, ftol=1e-4,
             gtol=1e-5):
        if rt_keys:
            for k in minimizer_kwargs:
                if k != 'method' and k != 'jac':
                    raise ValueError(_BH_MK_MSG)
            if 'method' in minimizer_kwargs:
                mcode = _bh_mcode(minimizer_kwargs['method'])
            else:
                mcode = 2                    # scipy's default quench: BFGS
        else:
            if has_method:
                mcode = _bh_mcode(minimizer_kwargs['method'])
            else:
                mcode = 2
        if has_jac:
            grad = minimizer_kwargs['jac']
        else:
            grad = None
        a = _args_or_empty(args)
        if ns_none:
            ns = niter + 2
        else:
            ns = niter_success
        return _bh_run(func, grad, x0, niter, T, stepsize, interval,
                       ns, target_accept_rate, stepwise_factor,
                       a, seed, mcode, xtol, ftol, gtol, disp, CBF, USE_CB,
                       rng)
    return impl


# ==================================================== differential evolution

@njit
def _de_scale(trial, arg1, arg2, out):
    """``_scale_parameters``: [0, 1] -> the actual parameter range."""
    for k in range(trial.shape[0]):
        out[k] = arg1[k] + (trial[k] - 0.5) * arg2[k]


@njit
def _de_select_samples(st, gen, pool, candidate, nsamp, out):
    """scipy's ``_select_samples``: shuffle the persistent index pool, take
    the first ``nsamp + 1``, drop ``candidate``, keep ``nsamp``."""
    _dr_shuffle(st, gen, pool)
    got = 0
    take = nsamp + 1
    if take > pool.shape[0]:
        take = pool.shape[0]
    for t in range(take):
        if pool[t] != candidate:
            out[got] = pool[t]
            got += 1
            if got == nsamp:
                break
    if got < nsamp:
        raise ValueError("differential_evolution: population too small for "
                         "the requested strategy")


@njit
def _de_core(func, bounds_lo, bounds_hi, args, maxiter=1000,
             popsize=15, tol=0.01, mutation_lo=0.5,
             mutation_hi=1.0, recombination=0.7, seed=0,
             strategy=0, atol=0.0, init=0, polish=0, disp=False,
             cb=_cb_noop, use_cb=False, rgen=None, x0=None):
    """The Storn-Price generation loop, on the box given as two vectors.

    This is the pre-retarget public body, kept unchanged. The scipy-shaped
    `differential_evolution` normalises the argument spellings and calls it,
    so the two are one core and one entry rather than two implementations.
    Group J in
    `tests/test_de_scipy.py` times this function DIRECTLY, in the same process
    and on the same problem as `differential_evolution`, which is why it must stay
    reachable: that comparison cannot drift with the machine or the toolchain,
    where a recorded number can.

    Returns ``(x, fval, nit, pop, energies, arg1, arg2, nfev)``. The middle four are
    what `differential_evolution` needs and a caller does not. `pop` is the
    population in
    UNIT-CUBE coordinates and `arg1`/`arg2` are the affine scaling back to the
    real box, so `differential_evolution` rescales row by row through
    `_de_scale` to build scipy's `population` field.

    The population holds ``max(5, popsize * max(1, N - n_equal_bounds))``
    members, where ``n_equal_bounds`` counts dimensions whose bounds coincide.
    The floor of 5 is scipy's. The ``rand2*`` strategies draw 5 distinct
    members plus the candidate, so a population under 6 raises here rather
    than reading past the pool.

    `mutation_lo` and `mutation_hi` arrive as two scalars because numba cannot
    type scipy's "float or 2-tuple" argument. They are sorted on entry, as
    scipy sorts the dither pair.

    Draws come from `rgen` when one is given, and otherwise from an internal
    xorshift64* generator seeded by `seed` and carried in a per-call state
    array. Per-call state is what makes this
    prange-safe: 12 concurrent runs
    at different seeds reproduced the serial answers to exactly 0.0. The
    stream is reproducible per seed and is not scipy's, so the suite validates
    against the known optimum instead. On 2-D Rastrigin over ``[-4, 4]^2``,
    ``f = 0.0`` exactly at ``|x| = 7.8e-10`` in 64 generations.

    A truthy `polish` runs L-BFGS-B over the same box and keeps the polished
    point on scipy's three conditions: an improvement, a converged polish, and
    the point inside the bounds. `nfev` counts the polish's evaluations too.

    `x0` replaces member 0 of the initial population, unscaled into the unit
    cube, which is where scipy places it.

    Implements scipy's default ``updating='immediate'`` scheme. Constraints,
    ``integrality``, ``vectorized``, ``workers`` and ``maxfun`` are not
    implemented.
    """
    N = bounds_lo.shape[0]
    if N == 0:
        raise ValueError("differential_evolution: bounds must be non-empty")
    if bounds_hi.shape[0] != N:
        raise ValueError("differential_evolution: bounds_lo and bounds_hi "
                         "must have equal length")
    for k in range(N):
        if not np.isfinite(bounds_lo[k]) or not np.isfinite(bounds_hi[k]):
            raise ValueError("differential_evolution: bounds must be finite")
    if popsize < 1:
        raise ValueError("differential_evolution: popsize must be >= 1")
    if maxiter < 0:
        raise ValueError("differential_evolution: maxiter must be >= 0")
    if tol < 0.0 or atol < 0.0:
        raise ValueError("differential_evolution: tol and atol must be >= 0")
    if recombination < 0.0 or recombination > 1.0:
        raise ValueError("differential_evolution: recombination must be in "
                         "[0, 1]")
    # scipy's test is `not all finite, or any >= 2, or any < 0`, so the upper
    # bound is EXCLUSIVE and a NaN is refused.  `2.0` was accepted here.
    if (not np.isfinite(mutation_lo) or not np.isfinite(mutation_hi) or
            mutation_lo < 0.0 or mutation_lo >= 2.0 or
            mutation_hi < 0.0 or mutation_hi >= 2.0):
        raise ValueError(
            "The mutation constant must be a float in U[0, 2), or specified "
            "as a tuple(min, max) where min < max and min, max are in "
            "U[0, 2).")
    if strategy < 0 or strategy > 11:
        raise ValueError("differential_evolution: strategy code must be 0..11")
    if init < 0 or init > 3:
        raise ValueError("differential_evolution: init must be 0 "
                         "(latinhypercube), 1 (random), 2 (sobol) or "
                         "3 (halton)")
    # `polish` is scipy's truthy flag: every non-zero value polishes.  It was
    # restricted to 0 and 1 here, where scipy accepts 2 and -1 as well.

    m_lo = mutation_lo
    m_hi = mutation_hi
    if m_hi < m_lo:                       # scipy sorts the dither pair
        m_lo = mutation_hi
        m_hi = mutation_lo

    # scaling between [0, 1] and the real parameter range
    arg1 = np.empty(N, dtype=np.float64)
    arg2 = np.empty(N, dtype=np.float64)
    eb_count = 0
    for k in range(N):
        arg1[k] = 0.5 * (bounds_lo[k] + bounds_hi[k])
        arg2[k] = abs(bounds_lo[k] - bounds_hi[k])
        if bounds_lo[k] == bounds_hi[k]:
            eb_count += 1

    nvary = N - eb_count
    if nvary < 1:
        nvary = 1
    npop = popsize * nvary
    if npop < 5:
        npop = 5
    if strategy >= 10 and npop < 6:       # rand2bin / rand2exp need 5 samples
        raise ValueError("differential_evolution: population too small for "
                         "the requested strategy")

    st = _rng_init(seed)
    binomial = (strategy % 2) == 0
    family = strategy // 2                # 0 best1, 1 rand1, 2 randtobest1,
    #                                       3 currenttobest1, 4 best2, 5 rand2
    if family == 0 or family == 3:
        nsamp = 2
    elif family == 1 or family == 2:
        nsamp = 3
    elif family == 4:
        nsamp = 4
    else:
        nsamp = 5

    # ---------------------------------------------------------- population
    pop = np.empty((npop, N), dtype=np.float64)
    if init == 0:
        # Latin hypercube, reproducing scipy's draw order
        segsize = 1.0 / np.float64(npop)
        samples = np.empty((npop, N), dtype=np.float64)
        for i in range(npop):
            for j in range(N):
                samples[i, j] = (segsize * _dr_u01(st, rgen)
                                 + np.float64(i) * segsize)
        order = np.empty(npop, dtype=np.int64)
        for j in range(N):
            for i in range(npop):
                order[i] = i
            _dr_shuffle(st, rgen, order)
            for i in range(npop):
                pop[i, j] = samples[order[i], j]
    elif init == 1:
        for i in range(npop):
            for j in range(N):
                pop[i, j] = _dr_u01(st, rgen)
    else:
        # scipy draws these from `scipy.stats.qmc`, scrambled with its own
        # `Generator`. The unscrambled sequences agree to the bit; the
        # scramble does not cross, so the seed comes off this solver's own
        # stream and the population differs from scipy's.
        qs = np.int64(_dr_u01(st, rgen) * 2147483647.0)
        if init == 2:
            qp = _qmc_sobol(N, npop, True, qs)
        else:
            qp = _qmc_halton(N, npop, True, qs)
        for i in range(npop):
            for j in range(N):
                pop[i, j] = qp[i, j]

    # scipy replaces member 0 with `x0`, unscaled into the unit cube, after
    # the initialisation and before any energy is known. A length-1 `x0` is
    # broadcast, as numpy broadcasts it there.
    if x0 is not None:
        nx = x0.shape[0]
        if nx != N and nx != 1:
            raise ValueError(
                "operands could not be broadcast together with shapes ("
                + str(nx) + ",) (" + str(N) + ",) ")
        for k in range(N):
            v = x0[0] if nx == 1 else x0[k]
            blo = bounds_lo[k]
            bhi = bounds_hi[k]
            if blo > bhi:
                blo = bounds_hi[k]
                bhi = bounds_lo[k]
            if v < blo or v > bhi:
                raise ValueError(
                    "Some entries in x0 lay outside the specified bounds")
            if arg2[k] != 0.0:
                pop[0, k] = (v - arg1[k]) / arg2[k] + 0.5
            else:
                pop[0, k] = 0.5

    params = np.empty(N, dtype=np.float64)
    energies = np.empty(npop, dtype=np.float64)
    nfev = 0                      # every evaluation of `func` this body makes
    for i in range(npop):
        _de_scale(pop[i], arg1, arg2, params)
        energies[i] = _call_args(func, params, args)
        nfev += 1

    # ``_promote_lowest_energy``: swap the best member into slot 0
    lbest = 0
    for i in range(1, npop):
        if energies[i] < energies[lbest]:
            lbest = i
    if lbest != 0:
        etmp = energies[0]
        energies[0] = energies[lbest]
        energies[lbest] = etmp
        for k in range(N):
            t = pop[0, k]
            pop[0, k] = pop[lbest, k]
            pop[lbest, k] = t

    pool = np.empty(npop, dtype=np.int64)
    for i in range(npop):
        pool[i] = i
    samp = np.empty(nsamp, dtype=np.int64)
    trial = np.empty(N, dtype=np.float64)
    bprime = np.empty(N, dtype=np.float64)
    cross = np.empty(N, dtype=np.bool_)

    nit = 0
    cb_stop = False
    # Drop whatever an earlier solve on this thread left behind, so the first
    # generation's read cannot see it.
    _cb_halt_clear()
    xcb = np.empty(N, dtype=np.float64)
    for gen in range(1, maxiter + 1):
        nit = gen
        scale = _dr_uniform(st, rgen, m_lo, m_hi)

        for candidate in range(npop):
            # ---------------------------------------------------- _mutate
            fill_point = _dr_randint(st, rgen, N)
            _de_select_samples(st, rgen, pool, candidate, nsamp, samp)
            for k in range(N):
                trial[k] = pop[candidate, k]

            if family == 0:                                   # best1
                r0 = samp[0]
                r1 = samp[1]
                for k in range(N):
                    bprime[k] = pop[0, k] + scale * (pop[r0, k] - pop[r1, k])
            elif family == 1:                                 # rand1
                r0 = samp[0]
                r1 = samp[1]
                r2 = samp[2]
                for k in range(N):
                    bprime[k] = pop[r0, k] + scale * (pop[r1, k] - pop[r2, k])
            elif family == 2:                                 # randtobest1
                r0 = samp[0]
                r1 = samp[1]
                r2 = samp[2]
                for k in range(N):
                    b = pop[r0, k]
                    b += scale * (pop[0, k] - b)
                    b += scale * (pop[r1, k] - pop[r2, k])
                    bprime[k] = b
            elif family == 3:                                 # currenttobest1
                r0 = samp[0]
                r1 = samp[1]
                for k in range(N):
                    bprime[k] = pop[candidate, k] + scale * (
                        pop[0, k] - pop[candidate, k] + pop[r0, k] - pop[r1, k])
            elif family == 4:                                 # best2
                r0 = samp[0]
                r1 = samp[1]
                r2 = samp[2]
                r3 = samp[3]
                for k in range(N):
                    bprime[k] = pop[0, k] + scale * (
                        pop[r0, k] + pop[r1, k] - pop[r2, k] - pop[r3, k])
            else:                                             # rand2
                r0 = samp[0]
                r1 = samp[1]
                r2 = samp[2]
                r3 = samp[3]
                r4 = samp[4]
                for k in range(N):
                    bprime[k] = pop[r0, k] + scale * (
                        pop[r1, k] + pop[r2, k] - pop[r3, k] - pop[r4, k])

            for k in range(N):
                cross[k] = _dr_u01(st, rgen) < recombination

            if binomial:
                cross[fill_point] = True
                for k in range(N):
                    if cross[k]:
                        trial[k] = bprime[k]
            else:
                cross[0] = True
                i = 0
                fp = fill_point
                while i < N and cross[i]:
                    trial[fp] = bprime[fp]
                    fp = (fp + 1) % N
                    i += 1

            # ------------------------------------------ _ensure_constraint
            for k in range(N):
                if trial[k] > 1.0 or trial[k] < 0.0:
                    trial[k] = _dr_u01(st, rgen)

            _de_scale(trial, arg1, arg2, params)
            energy = _call_args(func, params, args)
            nfev += 1

            if energy < energies[candidate]:
                energies[candidate] = energy
                for k in range(N):
                    pop[candidate, k] = trial[k]
                if energy < energies[0]:
                    # promote into slot 0
                    energies[candidate] = energies[0]
                    energies[0] = energy
                    for k in range(N):
                        t = pop[0, k]
                        pop[0, k] = pop[candidate, k]
                        pop[candidate, k] = t

        if disp:
            _de_print_step(gen, energies[0])

        # ------------------------------------------------------ converged?
        mean = 0.0
        for i in range(npop):
            mean += energies[i]
        mean /= np.float64(npop)
        var = 0.0
        for i in range(npop):
            d = energies[i] - mean
            var += d * d
        std = np.sqrt(var / np.float64(npop))

        # scipy calls the callback once per generation, after the
        # convergence quantity is known and before the test that would
        # end the run, so a callback can stop at the first generation.
        if use_cb:
            _de_scale(pop[0], arg1, arg2, xcb)
            cb(xcb.copy(), energies[0])
            # `_cb_halt_take` rather than `_cb_halt_get`: nothing downstream
            # of this core reads the slot, and the polish step below reaches
            # `minimize`, which does.
            if _cb_halt_take():
                cb_stop = True
                break

        if std <= atol + tol * abs(mean):
            break

    x = np.empty(N, dtype=np.float64)
    _de_scale(pop[0], arg1, arg2, x)
    fval = energies[0]
    # scipy attaches `jac` only when it ACCEPTS the polished point. A
    # namedtuple has a fixed field set, so the slot is always present and
    # holds zeros where scipy would carry no key at all.
    jac = np.zeros(N, dtype=np.float64)

    if polish:
        if disp:
            _de_print_polish()
        # scipy polishes with L-BFGS-B over the same box, at
        # `minimize(method='L-BFGS-B')`'s defaults, and adds the polish's own
        # `nfev`.  It accepts the polished point only on THREE conditions:
        # an improvement, `result.success`, and the point inside the bounds.
        plo = np.empty(N, dtype=np.float64)
        phi = np.empty(N, dtype=np.float64)
        for k in range(N):
            if bounds_lo[k] <= bounds_hi[k]:
                plo[k] = bounds_lo[k]
                phi[k] = bounds_hi[k]
            else:
                plo[k] = bounds_hi[k]
                phi[k] = bounds_lo[k]
        l, u, nbd = _lbfgsb_bounds(N, plo, phi)
        xo, fo, gout, conv, sr, pnf, pnit, which, sk, yk = _lbfgsb_run(
            _ev_approx, func, None, x, l, u, nbd, args, 10, 1e7, 1e-5,
            1e-8, 15000, 15000, -1, 20)
        nfev += pnf
        ok = _lbfgsb_flag(conv, sr) == 0
        inside = True
        for k in range(N):
            if xo[k] < plo[k] or xo[k] > phi[k]:
                inside = False
        if fo < fval and ok and inside:
            x = xo
            fval = fo
            jac = gout
            # scipy keeps its internal state consistent, so the reported
            # `population` and `population_energies` carry the polished member.
            energies[0] = fo
            for k in range(N):
                if arg2[k] != 0.0:
                    pop[0, k] = (xo[k] - arg1[k]) / arg2[k] + 0.5
                else:
                    pop[0, k] = 0.5

    return x, fval, nit, pop, energies, arg1, arg2, nfev, cb_stop, jac


#: ``differential_evolution``'s result, scipy's key set plus ``jac``.
#:
#: ``jac`` is always present here, holding the L-BFGS-B gradient at the
#: polished point and zeros where no polish was accepted.  scipy attaches it
#: on a RUN-TIME condition, `_differentialevolution.py`: the polished point
#: is kept only when it improves the objective, the polish reports success
#: AND the point lies inside the bounds.  Measured on scipy 1.18: an accepted
#: polish gives 9 keys, ``polish=False`` gives 8, and a polish that is
#: REJECTED also gives 8, on a piecewise-constant objective L-BFGS-B cannot
#: improve.  A compiled function has one return type per signature and that
#: condition is not known until the solve has run, so the key cannot come and
#: go here.
DEResult = _opt_result(
    ['x', 'fun', 'nit', 'nfev', 'population', 'population_energies',
     'success', 'message', 'jac'])

_DE_SCOPE_MSG = (
    "differential_evolution: integrality must be None, "
    "constraints empty, vectorized False, updating 'immediate' and workers 1. "
    "A callable `strategy` is a Python callable; constraints is a list of "
    "dicts; 'deferred' updating and workers are the parallel path.")

@njit
def _de_scope_ok(integrality, constraints, vectorized, workers,
                 updating):
    """True when every out-of-scope argument sits at its default.

    One helper so the two entry points cannot drift on what they refuse.
    `callback` is NOT here: it is served now, and its own shape check lives
    in `_cb_resolve` / `_cb_resolve_ty`.
    """
    if integrality is not None:
        return False
    if len(constraints) != 0 or vectorized or workers != 1:
        return False
    return updating == 'immediate'


_DE_MSG_OK = "Optimization terminated successfully."
_DE_MSG_MAX = "Maximum number of iterations has been exceeded."
_DE_MSG_CB = "callback function requested stop early"


@njit
def _de_strategy_code(strategy):
    """scipy's strategy name to the integer code `_de_core` compares.

    The twelve names are scipy's own spellings, in scipy's own order.
    """
    if strategy == 'best1bin':
        return 0
    elif strategy == 'best1exp':
        return 1
    elif strategy == 'rand1bin':
        return 2
    elif strategy == 'rand1exp':
        return 3
    elif strategy == 'randtobest1bin':
        return 4
    elif strategy == 'randtobest1exp':
        return 5
    elif strategy == 'currenttobest1bin':
        return 6
    elif strategy == 'currenttobest1exp':
        return 7
    elif strategy == 'best2bin':
        return 8
    elif strategy == 'best2exp':
        return 9
    elif strategy == 'rand2bin':
        return 10
    elif strategy == 'rand2exp':
        return 11
    raise ValueError(
        "differential_evolution: strategy must be one of scipy's names "
        "'best1bin', 'best1exp', 'rand1bin', 'rand1exp', 'randtobest1bin', "
        "'randtobest1exp', 'currenttobest1bin', 'currenttobest1exp', "
        "'best2bin', 'best2exp', 'rand2bin', 'rand2exp'")


@njit
def _de_init_code(init):
    """scipy's ``init`` name to the integer code `_de_core` compares."""
    if init == 'sobol':
        return 2
    elif init == 'halton':
        return 3
    if init == 'latinhypercube':
        return 0
    elif init == 'random':
        return 1
    raise ValueError("differential_evolution: init must be 'latinhypercube',"
                     " 'random', 'sobol' or 'halton'. An ndarray init is not "
                     "accepted here")


def _de_mutation_pair(mutation):
    """scipy's `mutation`: a float, or a ``(min, max)`` dithering pair.

    scipy leads its own docstring with the float spelling, which gives a
    CONSTANT mutation factor. `_de_core` takes the pair, so a scalar is
    widened to ``(m, m)`` here rather than in the loop.
    """
    if np.ndim(mutation) == 0:
        m = float(mutation)
        return m, m
    return float(mutation[0]), float(mutation[1])


@overload(_de_mutation_pair)
def _de_mutation_pair_ovl(mutation):
    """@njit implementation of `_de_mutation_pair`.

    A scalar and a sequence cannot be read by one expression, so the branch
    is taken over the numba TYPE. An omitted default arrives as
    ``types.Omitted`` carrying the raw python value.
    """
    m = mutation.value if isinstance(mutation, types.Omitted) else mutation
    if isinstance(m, (types.Float, types.Integer, float, int)):
        def impl(mutation):
            v = np.float64(mutation)
            return v, v
        return impl

    def impl(mutation):
        return np.float64(mutation[0]), np.float64(mutation[1])
    return impl


def _de_maxiter(maxiter):
    """scipy resolves ``maxiter=None`` to 1000, for backwards compatibility."""
    return 1000 if maxiter is None else int(maxiter)


@overload(_de_maxiter)
def _de_maxiter_ovl(maxiter):
    """@njit implementation of `_de_maxiter`, resolved at compile time."""
    m = maxiter.value if isinstance(maxiter, types.Omitted) else maxiter
    if m is None or isinstance(m, types.NoneType):
        def impl(maxiter):
            return 1000
        return impl

    def impl(maxiter):
        return np.int64(maxiter)
    return impl


def _de_step_line(nit, energy):
    """scipy's per-iteration `disp` line, formatted by scipy's own f-string."""
    print(f"differential_evolution step {nit}: f(x)="
          f" {energy}")


@njit
def _de_print_step(nit, energy):
    """`_de_step_line` from inside compiled code.

    Its own function because lowering an ``objmode`` block pickles the
    enclosing one, and `_de_core` reaches a ctypes entry point through the
    L-BFGS-B polish.
    """
    with objmode():
        _de_step_line(nit, energy)


def _de_polish_line():
    print("Polishing solution with 'L-BFGS-B'")


@njit
def _de_print_polish():
    """scipy's line before the polish, from inside compiled code."""
    with objmode():
        _de_polish_line()


@njit
def _de_bounds_to_lohi(bounds):
    """scipy's ``bounds`` -- an (n, 2) array_like -- to (lo, hi)."""
    b = np.ascontiguousarray(np.asarray(bounds, np.float64))
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("differential_evolution: bounds must be one "
                         "(min, max) pair per variable, an (n, 2) array_like")
    return (np.ascontiguousarray(b[:, 0]).copy(),
            np.ascontiguousarray(b[:, 1]).copy())


def differential_evolution(func, bounds, args=(),
                           strategy='best1bin',
                           maxiter=1000, popsize=15, tol=0.01,
                           mutation=(0.5, 1.0), recombination=0.7, rng=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube',
                           atol=0.0, updating='immediate', workers=1,
                           constraints=(), x0=None, integrality=None,
                           vectorized=False, seed=None):
    """Differential evolution, a population-based stochastic optimizer.

    **Callback style A** -- ``func`` is a plain ``@njit``
    ``func(x, *args) -> float64``.

    Parameters
    ----------
    func : callable
        A plain ``@njit`` ``f(x, *args) -> float``.
    bounds : (n, 2) array_like
        One ``(min, max)`` pair per variable. Must be finite.
    args : tuple, optional
        Extra parameters, unpacked into the argument list of `func` after
        `x`, so ``args=(a, b)`` calls ``func(x, a, b)``. Its elements may be
        of different types and shapes. Default ``()``, which calls
        ``func(x)``; ``None`` means the same. An ndarray or a list arrives as
        ONE argument instead, ``func(x, args)``. See Notes.
    strategy : str, optional
        Name of the mutation strategy. ``'best1bin'`` is the default. Full
        list in the module docstring. A callable `strategy` is not accepted.
    maxiter : int or None, optional
        Maximum number of generations. Default 1000. ``None`` resolves to
        1000.
    popsize : int, optional
        Population-size multiplier. Default 15. Must be ``>= 1``.
    tol : float, optional
        Relative convergence tolerance on the population energies. Default
        0.01. Must be ``>= 0``.
    mutation : float or 2-tuple, optional
        A constant mutation factor, or a ``(lo, hi)`` dithering range. Default
        ``(0.5, 1)``. Every entry must lie in ``[0, 2)``.
    recombination : float, optional
        Crossover probability. Default 0.7. Must lie in ``[0, 1]``.
    rng : int or numpy.random.Generator, optional
        Source of every draw the population and the generation loop make.
        ``None`` (default) uses the internal xorshift64* generator; an
        integer builds a ``numpy.random.default_rng`` for the call; a
        ``Generator`` is drawn from directly and its state advances. See
        Notes.
    callback : callable, optional
        Called once per generation, after the convergence quantity is known
        and before the test that would end the run. A plain Python
        ``callback(xk)`` or ``callback(intermediate_result)``, or an
        ``@njit`` ``callback(xk)``. Returning a truthy value or raising
        ``StopIteration`` halts the run; the result then reports
        ``success=False`` and ``message='callback function requested stop
        early'``, and the polish still runs.
    disp : bool, optional
        Print the per-iteration line, and the line before the polish. Default
        ``False``.
    polish : bool, optional
        Polish the best member with L-BFGS-B over the same box. ``True`` by
        default. The polished point replaces the best member only when it
        improves the objective, the polish converged, and it lies inside the
        bounds. A callable `polish` is not accepted; see Notes.
    init : str, optional
        ``'latinhypercube'`` (default), ``'random'``, ``'sobol'`` or
        ``'halton'``. An ndarray of starting points is not accepted.
    atol : float, optional
        Absolute convergence tolerance on the population energies. Default
        0.0. Must be ``>= 0``.
    updating, workers, constraints, x0, integrality, vectorized : optional
        Accepted only at their defaults. See Notes.
    seed : int, optional
        BEYOND SCIPY. Seeds the internal xorshift64* generator, which is the
        `rng` at ``None`` path. ``None`` (default) draws fresh entropy per
        call, so two identical calls give different answers. See Notes.

    Returns
    -------
    res : DEResult
        A namedtuple carrying ``x``, ``fun``, ``nit``, ``nfev``,
        ``population``, ``population_energies``, ``success``, ``message``
        and ``jac``. ``nfev`` is a count of the evaluations of `func` this
        routine makes, the polish's included. ``jac`` is the L-BFGS-B
        gradient at the polished point, and zeros where no polish was
        accepted.

    Raises
    ------
    ValueError
        Empty `bounds`, a non-finite bound, `bounds` that is not ``(n, 2)``,
        ``popsize < 1``, ``maxiter < 0``, a negative `tol` or `atol`, a
        `recombination` outside ``[0, 1]``, a `mutation` entry outside
        ``[0, 2)``, an unknown `strategy` or `init`, a `rng` that is not
        ``None``, an integer or a ``numpy.random.Generator``, or any of
        `callback`, `x0`, `integrality`, `constraints`, `vectorized`,
        `workers` and `updating` away from its default.

    See Also
    --------
    scipy.optimize.differential_evolution : The scipy routine this mirrors.
    scijit.optimize.brute : Exhaustive grid, deterministic.
    scijit.optimize.basinhopping : Random restarts around a local minimizer.

    Notes
    -----
    UNAVAILABLE: an ``OptimizeResult``. `res` is a namedtuple, so its field
    set is fixed. ``jac`` is always present where scipy adds the key only
    after a polish it accepted. ``constr``, ``constr_violation``, ``maxcv``
    and ``constr_penalty``, which scipy carries under constraints, are
    absent.

    NOT IMPLEMENTED: a callable `strategy`, a callable `polish`, and an
    ndarray in `init`.

    ``init='sobol'`` and ``init='halton'`` draw the initial population from
    the same quasi-random sequences scipy uses: unscrambled, the points from
    `scijit.stats.qmc` and ``scipy.stats.qmc`` agree to the bit. scipy
    scrambles them with a ``numpy.random.Generator``, which compiled code
    cannot hold, so the scramble seed comes off this solver's own stream and
    the population, the search path and the evaluation count are not scipy's
    at the same seed.

    `args` is unpacked into the objective's argument list, ``func(x, *args)``.

    DELIBERATELY DIFFERENT: an ndarray, a list or a bare scalar `args` reaches
    `func` as ONE argument, ``func(x, args)``. scipy unpacks the first two
    element by element and refuses the third. The arity of a compiled call is
    fixed when it compiles, so a sequence whose length is known only at run
    time cannot be unpacked: ``func(x, *args)`` on an ndarray inside ``@njit``
    is a ``TypingError``. An objective written to the unpacked shape refuses
    the ndarray rather than reading it wrongly.

    NOT IMPLEMENTED: `x0`, `integrality`, `constraints`, `vectorized`,
    `workers` and ``updating='deferred'``. Each raises ``ValueError`` away
    from its default rather than being ignored.

    DELIBERATELY DIFFERENT: scipy's legacy ``convergence`` value passed to
    ``callback(xk, convergence=val)`` is not carried here. Its value is
    ``tol / (std(energies) / |mean(energies)|)``, which this entry point does
    not compute.

    DELIBERATELY DIFFERENT: an ``intermediate_result`` here carries ``x`` and
    ``fun``. scipy's also carries ``convergence``, ``message``, ``nfev``,
    ``nit``, ``population``, ``population_energies`` and ``success``.

    An ``@njit`` `callback` halts by raising; its return value is not read.
    A plain Python `callback` also halts on a truthy return.

    With `rng` at ``None`` the draws come from an internal xorshift64*
    generator carried in a per-call state array. `seed` selects its stream
    and ``seed=None`` draws a fresh one per call.

    DELIBERATELY DIFFERENT: an integer `rng` builds a
    ``numpy.random.default_rng``, which is PCG64, where scipy's
    ``check_random_state`` builds the legacy MT19937 ``RandomState`` that
    compiled code cannot hold. Under any `rng` the population initialisation,
    the dither and the crossover consume draws in scipy's order and shapes,
    and the trial index is taken from one ``uniform()`` where scipy calls
    ``choice``, which numba does not implement. So the same integer explores
    a different population.

    A `rng` at ``None`` is safe to call from a ``numba.prange`` loop, because
    the generator state is per-call. One ``Generator`` SHARED across a
    ``prange`` is not: 70,499 of 100,000 draws were distinct across 32
    threads. An integer `rng` builds its generator inside the call, so it is
    per-call as well.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import differential_evolution
    >>> @njit
    ... def q(x, target):
    ...     return (x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2
    >>> bounds = np.array([[-3.0, 3.0], [-3.0, 3.0]])
    >>> @njit
    ... def run():
    ...     return differential_evolution(q, bounds,
    ...                                   (np.array([1.5, -0.5]),),
    ...                                   'best1bin', 200)
    >>> res = run()
    >>> np.round(res.x, 4)
    array([ 1.5, -0.5])
    >>> res.success
    True
    """
    if not _de_scope_ok(integrality, constraints, vectorized,
                        workers, updating):
        raise ValueError(_DE_SCOPE_MSG)
    cbf, use_cb, pycb = _cb_resolve('differential_evolution', callback,
                                    de=True)
    prev = _cb_install(pycb)
    try:
        return _de_run(func, bounds, _args_or_empty(args), strategy,
                       maxiter, popsize, tol,
                       mutation, recombination, disp, polish, init, atol,
                       seed, cbf, use_cb, _rng_gen(rng), x0)
    finally:
        _cb_release(prev)


@overload(differential_evolution, prefer_literal=True)
def _de_ovl(func, bounds, args=(), strategy='best1bin',
            maxiter=1000, popsize=15, tol=0.01,
            mutation=(0.5, 1.0), recombination=0.7, rng=None,
            callback=None, disp=False, polish=True,
            init='latinhypercube',
            atol=0.0, updating='immediate', workers=1,
            constraints=(), x0=None, integrality=None,
            vectorized=False, seed=None):
    """@njit implementation of `differential_evolution`.

    `callback` is a type branch, not a value: an absent one and an ``@njit``
    one reach different bodies, and a plain Python callable cannot cross into
    compiled code as an argument at all. Everything algorithmic stays in
    `_de_run`, so this entry and the Python one are one implementation.
    """
    CBF, USE_CB = _cb_resolve_ty('differential_evolution', callback, de=True)

    def impl(func, bounds, args=(), strategy='best1bin',
             maxiter=1000, popsize=15, tol=0.01,
             mutation=(0.5, 1.0), recombination=0.7, rng=None,
             callback=None, disp=False, polish=True,
             init='latinhypercube',
             atol=0.0, updating='immediate', workers=1,
             constraints=(), x0=None, integrality=None,
             vectorized=False, seed=None):
        if not _de_scope_ok(integrality, constraints, vectorized,
                            workers, updating):
            raise ValueError(_DE_SCOPE_MSG)
        return _de_run(func, bounds, _args_or_empty(args), strategy,
                       maxiter, popsize, tol,
                       mutation, recombination, disp, polish, init, atol,
                       seed, CBF, USE_CB, rng, x0)
    return impl


def _de_x0(x0):
    """scipy's `x0` as a contiguous float64 vector, or ``None``."""
    if x0 is None:
        return None
    return np.ascontiguousarray(np.asarray(x0, np.float64)).ravel()


@overload(_de_x0)
def _de_x0_ovl(x0):
    v = x0.value if isinstance(x0, types.Omitted) else x0
    if v is None or isinstance(v, types.NoneType):
        def impl(x0):
            return None
        return impl

    def impl(x0):
        return np.ascontiguousarray(np.asarray(x0).astype(np.float64)).ravel()
    return impl


@njit
def _de_run(func, bounds, args, strategy, maxiter, popsize, tol,
            mutation, recombination, disp, polish, init, atol, seed,
            cb, use_cb, rng=None, x0=None):
    """The one body both `differential_evolution` entry points reach.

    `rng` and `seed` are resolved here, so the two entry points cannot
    disagree about what a bare call draws.
    """
    # scipy's names for the twelve strategies and the two initializations,
    # resolved to the codes `_de_core` compares.
    scode = _de_strategy_code(strategy)
    icode = _de_init_code(init)
    lo, hi = _de_bounds_to_lohi(bounds)
    mlo, mhi = _de_mutation_pair(mutation)
    mit = _de_maxiter(maxiter)
    x, fval, nit, pop, energies, arg1, arg2, nfev, cb_stop, jac = _de_core(
        func, lo, hi, args, mit, popsize, tol, mlo, mhi,
        recombination, _seed_value(seed), scode, atol, icode, polish, disp,
        cb, use_cb, _rng_gen(rng), _de_x0(x0))
    npop = pop.shape[0]
    n = lo.size
    popx = np.empty((npop, n), np.float64)
    row = np.empty(n, np.float64)
    for i in range(npop):
        _de_scale(pop[i], arg1, arg2, row)
        for k in range(n):
            popx[i, k] = row[k]
    if cb_stop:
        ok = False
        msg = _DE_MSG_CB
    else:
        ok = nit < mit
        msg = _DE_MSG_OK if ok else _DE_MSG_MAX
    return DEResult(x, fval, np.int64(nit), np.int64(nfev), popx,
                    energies.copy(), ok, msg, jac)

