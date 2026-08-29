"""Scalar root-finders and scalar minimizers, in numba ``@njit``.

The *pure-algorithm* scalar solvers of ``scipy.optimize``, written so they
run inside ``@njit`` code.  The objective is always a PLAIN
``@njit`` function ``f(x) -> float`` passed as the first argument (numba
first-class functions), with scipy's ``args`` tuple beside it.  No
``full_output`` object and no callbacks-with-.address: just pass a jitted
``f``.

Where the algorithms come from: scipy's ``bisect``, ``brentq``, ``brenth``
and ``ridder`` are C (``scipy.optimize._zeros``) and ``toms748``, ``newton``,
``golden``, ``brent``, ``fminbound`` and ``fixed_point`` are pure Python.
Every routine here is a statement-for-statement transcription of the source
scipy runs, C or Python, variable names included.

Everything is ``@njit`` and pure (no module state) => prange-safe.

--------------------------------------------------------------------------
RETURN SHAPES are scipy's.  A bare call returns what scipy's bare call
returns, and `full_output` selects the fuller shape, exactly as scipy spells
it.

  bisect, brentq,   -> float, or (x, RootResults) under full_output
  brenth, ridder,      RootResults(root, iterations, function_calls,
  toms748, newton                  converged, flag, method)
  newton, array x0  -> ndarray, or ArrayNewtonResult(root, converged,
                       zero_der) under full_output, three arrays shaped
                       like x0
  brent             -> float, or (x, fun, nit, nfev)
  golden            -> float, or (x, fun, nfev)          THREE, not four
  fminbound         -> float, or (x, fun, status, nfev)  status THIRD
  fixed_point       -> float, where scipy returns a 0-d ``ndarray`` under
                       ``method='del2'`` and a ``numpy.float64`` under
                       ``method='iteration'``.  scipy publishes no
                       full_output here; this package adds one, ADDITIVELY,
                       giving
                       (x, FixedPointResults(x, iterations, function_calls,
                                             converged, flag, method))
  root_scalar       -> RootResults, always, as scipy's does
  minimize_scalar   -> MinimizeScalarResult(fun, message, nfev, nit,
                                            success, x) for ``'brent'`` and
                       ``'golden'``, and
                       MinimizeScalarResultBounded(fun, message, nfev, nit,
                                                   status, success, x) for
                       ``'bounded'``, which is the one method scipy gives a
                       ``status``

`full_output` selects the RETURN TYPE, so inside ``@njit`` it must be
readable when the call compiles: a literal, an omitted default or a
module-level constant.  A runtime variable raises ``TypingError`` naming the
constraint.  From Python a runtime value is fine.

`iterations` counts algorithm iterations; `function_calls` counts every
evaluation of the user function, with no path bypassing the counter.
Measured against scipy 1.18 on 2026-08-10, each figure equal to scipy's on
both fields.  On ``x*x - 2`` over ``[0, 2]``, starting the open methods at
``x0 = 1.0``: ``bisect`` (40, 42), ``brentq`` and ``brenth`` (8, 9),
``ridder`` (6, 14), ``toms748`` (5, 11), ``newton`` (5, 10), Halley
(4, 12), the secant loop (6, 7) and ``root_scalar`` (8, 9).  On
``(x - 1.5) ** 2 + 0.5``, with ``brent`` and ``golden`` seeded from their
own defaults, ``minimize_scalar`` from ``(0.0, 3.0)`` and ``fminbound``
bounded on ``[0, 3]``: ``brent`` (5, 8), ``golden`` 43 evaluations,
``fminbound`` 6 and ``minimize_scalar`` (5, 8).
``fixed_point`` has nothing to compare against, since scipy reports no
counter for it.

WHAT A COMPILED FUNCTION CANNOT DO is vary WHICH fields are present at RUN
time, since the field set is part of the return type.  `full_output` is
readable when the call compiles, so the two shapes scipy selects with it are
both reachable; what is not reachable is a field set that depends on a value
only known while the code runs.  ``minimize_scalar`` carries ``status`` on
``'bounded'`` alone, which is scipy's split, so its `method` has to be a
literal string inside ``@njit``; a `method` held in a variable is refused
when the call compiles.  From Python any string works.

--------------------------------------------------------------------------
METHOD NAMES.  Both dispatchers take scipy's method NAME.

  root_scalar(f, method=..., bracket=..., x0=..., x1=...,
              fprime=..., fprime2=...)
      'brentq'    bracket               (chosen for a bracket)
      'brenth'    bracket
      'bisect'    bracket
      'ridder'    bracket
      'toms748'   bracket
      'secant'    x0, x1                (chosen for x0 and x1)
      'newton'    x0, fprime            (chosen for x0, with or without
                                         a derivative)
      'halley'    x0, fprime, fprime2   (chosen for both derivatives)

  minimize_scalar(f, bracket=..., bounds=..., method=...)
      'brent'     bracket = seed points for the downhill bracket search,
                  or a validated 3-point bracket        (the default)
      'golden'    bracket, as above
      'bounded'   bounds = FIXED bounds  (fminbound)    (chosen for bounds)

--------------------------------------------------------------------------
DEVIATIONS from scipy:
  * ``newton`` selects its method from which derivatives are supplied,
    as scipy's does, and reports scipy's own name for the one it ran:
        newton(f, x0)                   -> the secant loop
        newton(f, x0, fprime)           -> Newton-Raphson
        newton(f, x0, fp, fprime2=fpp)  -> Halley
    The choice follows the TYPE of the derivative arguments, so it is
    made when the call compiles, not at run time.
  * ``args`` is scipy's, a tuple in scipy's own positional slot, unpacked
    into the call as ``f(x, *args)``.  It reaches ``fprime`` and ``fprime2``
    too.  scipy coerces a non-tuple to a 1-tuple in ``bisect``, ``brentq``,
    ``brenth``, ``ridder``, ``toms748``, ``root_scalar`` and
    ``minimize_scalar``, and refuses one in ``newton``, ``fixed_point``,
    ``brent``, ``golden`` and ``fminbound``; each name follows its own
    counterpart.
        bisect(f, 0.0, 2.0, (c,))       -> f(x, c)
        bisect(f, 0.0, 2.0, c)          -> f(x, c)
        newton(f, 1.0, args=c)          -> TypeError
    ``brent``, ``golden``, ``root_scalar`` and ``minimize_scalar`` take
    scipy's bracket SEQUENCE in scipy's own slot, so their whole signature
    lines up with scipy's.
  * Namedtuple instead of scipy's attribute-dict result object, wherever a
    result object is returned at all (see RETURN SHAPES above).  Every field
    is always present, where scipy's objects vary which keys they carry.
  * Bad input (bracket with f(a)*f(b) > 0, no valid downhill bracket,
    lower>upper bounds) raises ``ValueError`` as scipy does.
"""
import math
import operator
import warnings
from collections import namedtuple

import numpy as np
from numba import njit, objmode, types
from numba.core.errors import TypingError
from numba.extending import overload

from ._minpack import OptimizeResult, _opt_result

# scipy.optimize._zeros_py defaults
_EPS = 2.220446049250313e-16          # np.finfo(float).eps
_RTOL = 4.0 * _EPS                    # scipy _rtol
_XTOL = 2e-12                         # scipy _xtol
_EPSILON = 1.4901161193847656e-08     # sqrt(eps), scipy _epsilon
_NAN = float("nan")

#: scipy's ``RootResults``, as a namedtuple. Field names, order and the
#: ``flag`` strings are scipy's own (``scipy/optimize/_zeros_py.py``).
#: ``converged`` is stored rather than derived; scipy computes it from
#: ``flag`` and exposes it as an attribute either way.
RootResults = type(
    'RootResults',
    (OptimizeResult, namedtuple(
        'RootResults',
        ['root', 'iterations', 'function_calls', 'converged', 'flag',
         'method'])),
    {'__slots__': ()})

RootResults.__doc__ = """Result of a scalar root find.

Returned by `bisect`, `brentq`, `brenth`, `ridder`, `toms748`, `newton` and
`root_scalar`.

Attributes
----------
root : float
    The estimated root.
iterations : int
    Iterations taken.
function_calls : int
    Evaluations of `f`.
converged : bool
    ``True`` when the routine met its tolerance.
flag : str
    ``'converged'`` or ``'convergence error'``. See `Notes`.
method : str
    The routine that ran, by name.

Notes
-----
The `flag` set is two strings, ``'converged'`` and ``'convergence error'``.
``scipy.optimize.RootResults``'s ``flag_map`` also holds ``'sign error'``,
``'value error'`` and ``'No error'``, and its ``root_scalar`` puts an
exception message there on the NaN path.

The CONTAINER is a namedtuple, where
``scipy.optimize.RootResults`` subclasses ``OptimizeResult`` and so is a
``dict``. Attribute access reads the same on both. Three things do not, all
measured on the same root::

    isinstance(res, scipy.optimize.RootResults)     False
    res[0]                                          the root here,
                                                    KeyError: 0 in scipy
    tuple(res)                                      the six values here,
                                                    the six field NAMES in
                                                    scipy, which is a dict
                                                    iterating its keys

The third is silent, since both sides return a 6-tuple.

The CONSTRUCTOR differs too. This takes the six attributes above;
``scipy.optimize.RootResults`` takes five, ``(root, iterations,
function_calls, flag, method)``, with `flag` an integer code, and derives
`converged` and the flag string from it.
"""

#: `newton`'s third result shape, returned for an array `x0` under
#: ``full_output``. scipy builds the same namedtuple inside `_array_newton`
#: and calls the type ``result``, which is the name kept here; the three
#: members are arrays shaped like `x0`.
ArrayNewtonResult = namedtuple('result', ['root', 'converged', 'zero_der'])

ArrayNewtonResult.__doc__ = """Result of a `newton` root find over an array.

Attributes
----------
root : ndarray
    The estimated roots, shaped like `x0`.
converged : ndarray of bool
    ``True`` where that element met the tolerance.
zero_der : ndarray of bool
    ``True`` where the derivative was zero, so the element stopped being
    updated.
"""

#: `fixed_point`'s twin. scipy returns a bare float there and has no result
#: object to copy, so the members are this module's own with scipy's
#: diagnostics beside them. ADDITIVE.
FixedPointResults = type(
    'FixedPointResults',
    (OptimizeResult, namedtuple(
        'FixedPointResults',
        ['x', 'iterations', 'function_calls', 'converged', 'flag',
         'method'])),
    {'__slots__': ()})

#: scipy's ``minimize_scalar`` OptimizeResult for ``'brent'`` and
#: ``'golden'``, as a namedtuple. The field names are scipy's and the order
#: is alphabetical, which is neither the order scipy's dict carries
#: (``fun, x, nit, nfev, success, message``) nor the order its
#: ``_RichResult`` prints (``message, success, fun, x, nit, nfev``).
MinimizeScalarResult = _opt_result(
    ['fun', 'message', 'nfev', 'nit', 'success', 'x'])

#: scipy's ``minimize_scalar`` OptimizeResult for ``'bounded'``, which is the
#: one method that carries ``status``. Measured on scipy 1.18: ``'brent'`` 6
#: keys, ``'golden'`` 6, ``'bounded'`` 7.
MinimizeScalarResultBounded = _opt_result(
    ['fun', 'message', 'nfev', 'nit', 'status', 'success', 'x'])

#: scipy's flag strings, ``_zeros_py.flag_map``.
_FLAG_CONV = 'converged'
_FLAG_CONVERR = 'convergence error'

#: scipy's ``minimize_scalar`` messages. The ``'brent'`` and ``'golden'``
#: success text is scipy's first two lines. Its third, ``(using xtol = ...)``,
#: interpolates a float, and that number is rendered in PYTHON while the call
#: compiles rather than inside the compiled body. Measured 2026-08-03: the
#: run-time route through ``numba.objmode`` costs 1.225 us per call against
#: 0.292, and 176.592 ms against 0.040 ms over a 2000-iteration ``prange``.
_MS_MSG_OK = ("\nOptimization terminated successfully;\n"
              "The returned value satisfies the termination criteria")


def _ms_msg_ok(xtol):
    """scipy's ``'brent'``/``'golden'`` success message, tolerance included.

    scipy's third line is ``f"(using xtol = {xtol} )"``, which interpolates a
    float. numba's own ``str(<float64>)`` returns the literal text
    '<object type:float64>', so the number cannot be rendered inside compiled
    code without a ``numba.objmode`` block, and that block takes the GIL:
    measured 0.040 ms against 176.592 ms over a 2000-iteration ``prange``.

    It does not have to be rendered there. This runs in PYTHON, either at the
    python entry or in the ``@overload`` chooser while the call compiles, and
    the finished string is baked into the compiled body as a constant. Cost
    measured at 0.300 us/call against 0.292 for a bare literal.

    Returns ``None`` when `xtol` is not known at compile time, which is any
    explicitly-passed value: numba makes literals of ints, bools and strings
    but not of floats, so an explicit ``xtol=1e-9`` arrives typed ``float64``
    with the value gone.
    """
    if isinstance(xtol, types.Omitted):
        xtol = xtol.value
    if not isinstance(xtol, float):
        return None
    return _MS_MSG_OK + "\n(using xtol = %r )" % xtol


def _ms_or_plain(xtol):
    """`_ms_msg_ok`, falling back to the two-line message. Never ``None``."""
    m = _ms_msg_ok(xtol)
    return _MS_MSG_OK if m is None else m
_MS_MSG_MAXIT = "\nMaximum number of iterations exceeded"
_MS_MSG_NAN = "NaN result encountered."
_MS_MSG_BOUNDED_OK = "Solution found."
_MS_MSG_BOUNDED_MAX = "Maximum number of function calls reached."

_MS_METHOD_RUNTIME_MSG = (
    "minimize_scalar: inside @njit `method` must be a literal string written "
    "at the call site, not a variable. 'bounded' returns the seven fields "
    "scipy gives that method and 'brent' and 'golden' return six, `status` "
    "being scipy's on 'bounded' alone, and a compiled function has one return "
    "type per signature, so the method has to be known while the call "
    "compiles. Spell it literally, minimize_scalar(f, bounds=(0.0, 3.0), "
    "method='bounded'), or call from Python, where any string works. "
    "`minimize` and `root` refuse a runtime `method` for the same reason")

_MS_METHOD_TYPE_MSG = (
    "minimize_scalar: `method` must be a string or None. scipy reaches it "
    "through `method.lower()`, so anything else is refused there too")

_MS_UNKNOWN_MSG = "minimize_scalar: Unknown solver "

#: scipy's two `'bounded'` refusals, `_optimize.py:2320-2327`.
_MS_BOUNDS_MANDATORY_MSG = ('The `bounds` parameter is mandatory for '
                            'method `bounded`.')
_MS_BOUNDS_TWO_MSG = 'bounds must have two elements.'


# ======================================================================
# small helpers
# ======================================================================
@njit
def _sgn(x):
    """Sign of ``x`` as a float: ``-1.0``, ``0.0`` or ``1.0``.

    The bracketing solvers decide whether an interval still straddles a
    root by testing ``_sgn(fa) * _sgn(fb) > 0.0``, so an exact zero must
    report no sign at all. Folding the zero case into ``+1.0``, which is
    what ``math.copysign`` does, would hide a root that has been hit
    exactly.
    """
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    return 0.0


@njit
def _fc(f, x, cnt, args):
    """Evaluate ``f(x, *args)`` and count the call.

    ``function_calls`` is scipy's own counter and it counts EVERY evaluation,
    including the ones spent establishing a bracket, the ones a solver
    discards, and the DERIVATIVE evaluations: scipy's ``newton`` increments
    the same counter for ``fprime`` and ``fprime2``, so ``newton`` reports
    two per iteration and ``halley`` three. Routing every call through here is what keeps the count honest:
    an evaluation written directly as ``f(x)`` somewhere in a branch would be
    invisible to it, and nothing would fail.

    `cnt` is a length-1 ``int64`` array rather than a returned counter so the
    call sites stay expressions.

    `args` reproduces scipy's coercion, ``if not isinstance(args, tuple):
    args = (args,)``. ``isinstance`` against ``tuple`` resolves inside
    ``@njit`` on numba 0.66, measured 2026-08-05, so both spellings compile:
    a tuple is unpacked into the call and anything else is passed as one
    extra argument.
    """
    cnt[0] += 1
    if isinstance(args, tuple):
        return f(x, *args)
    return f(x, args)


@njit
def _fcr(f, x, cnt, args):
    """`_fc`, refusing a NaN function value.  Raises ``ValueError``.

    scipy passes `f` through ``_wrap_nan_raise`` before ``bisect``,
    ``brentq``, ``brenth``, ``ridder`` and ``toms748`` reach it, so a NaN
    value stops the run. Without that check every sign test is a product
    comparison and a NaN makes all of them false, so the solver runs to
    `maxiter` or returns a NaN root marked converged.

    The iterate IS interpolated, as scipy's is, through the same `objmode`
    formatter `toms748` uses.

    Applies to the five bracketing solvers only. ``newton``,
    ``fixed_point`` and the three minimizers use `_fc`, because scipy does
    not wrap `f` for those.
    """
    cnt[0] += 1
    if isinstance(args, tuple):
        fx = f(x, *args)
    else:
        fx = f(x, args)
    if math.isnan(fx):
        raise ValueError(_nanfval_msg(x))
    return fx


# --------------------------------------------------------------------------
# `args` in `newton`, `fixed_point`, `brent`, `golden` and `fminbound`.
#
# scipy writes `if not isinstance(args, tuple): args = (args,)` in `bisect`,
# `brentq`, `brenth`, `ridder`, `toms748`, `root_scalar` and
# `minimize_scalar`, and NOT in the other five, which reach the call with the
# value unchanged. Measured on scipy 1.18, `args=2.0`:
#
#   newton, fixed_point, fminbound   f(x, *args)
#       TypeError: Value after * must be an iterable, not float
#   brent, golden                    f(*((x,) + args))
#       TypeError: can only concatenate tuple (not "float") to tuple
#
# CPython names the value's type in both texts, so the guards below build the
# name from the value in Python and from its numba type while the call
# compiles, and one message serves both entry points.


def _ty_name(t):
    """CPython's ``type(x).__name__`` for a numba type."""
    if isinstance(t, types.Omitted):
        return type(t.value).__name__
    if isinstance(t, types.Boolean):
        return 'bool'
    if isinstance(t, types.Integer):
        return 'int'
    if isinstance(t, types.Float):
        return 'float'
    if isinstance(t, types.Complex):
        return 'complex'
    if isinstance(t, types.UnicodeType):
        return 'str'
    if isinstance(t, types.NoneType):
        return 'NoneType'
    if isinstance(t, types.Array):
        return 'ndarray'
    if isinstance(t, (types.List, types.ListType)):
        return 'list'
    return str(t)


def _args_star_msg(tname):
    return "Value after * must be an iterable, not " + tname


def _args_cat_msg(tname):
    return 'can only concatenate tuple (not "' + tname + '") to tuple'


def _args_tuple_star(args):
    """Refuse a non-tuple `args` for a routine calling ``f(x, *args)``."""
    if not isinstance(args, tuple):
        raise TypeError(_args_star_msg(type(args).__name__))


def _args_tuple_cat(args):
    """Refuse a non-tuple `args` for a routine calling ``f(*((x,) + args))``."""
    if not isinstance(args, tuple):
        raise TypeError(_args_cat_msg(type(args).__name__))


def _args_refusal(args, build):
    """The message a chooser must refuse `args` with, or ``None`` to proceed.

    The refusal is decided while the call compiles rather than raised from
    the compiled body, because the body would still have to TYPE: an
    ``args=None`` reaches ``f(x, args)`` in `_fc` and fails there first, with
    numba's error rather than scipy's.
    """
    if isinstance(args, types.Omitted):
        args = args.value
    if not isinstance(args, types.Type):
        # an omitted default reaches a chooser as the raw python value
        return None if isinstance(args, tuple) else build(
            type(args).__name__)
    if isinstance(args, types.BaseTuple):
        return None
    return build(_ty_name(args))


def _args_refusal_star(args):
    return _args_refusal(args, _args_star_msg)


def _args_refusal_cat(args):
    return _args_refusal(args, _args_cat_msg)


@njit
def _always():
    """``True`` at run time, and not foldable while the call compiles.

    An `@overload` arm whose body ONLY raises types as returning ``none``, so
    a caller writing ``root, res = f(...)`` fails on the RETURN TYPE before
    the body runs and reads numba's "failed to unpack none", which names
    neither the argument nor the mistake. Measured 2026-08-11 on
    ``full_output=True``: `toms748`, `newton`, `fixed_point` and `fminbound`
    all reported that instead of scipy's message.

    Guarding the raise with this keeps the raise at run time and leaves the
    arm free to declare a return type below it. The declaration is
    unreachable and exists so the caller's unpack types; it carries the SHAPE
    the routine returns, so a caller that then reads a field of the result
    still compiles and still reaches the raise. A bare ``raise`` followed by
    a ``return`` does NOT work: numba drops the unreachable return and the
    type is ``none`` again.

    THE MESSAGE MUST BE RAISED INLINE. Handing it to a shared ``@njit``
    raiser as an argument reports A DIFFERENT CALLER'S MESSAGE: measured
    2026-08-11 over seven callers and two messages, two callers were served
    the wrong one, with no warning.
    """
    return True


#: An unreachable `RootResults` for a refusing arm. See `_always`.
@njit
def _dead_root(name):
    return RootResults(0.0, 0, 0, False, _FLAG_CONVERR, name)


#: An unreachable `FixedPointResults` for a refusing arm. See `_always`.
@njit
def _dead_fp(name):
    return FixedPointResults(0.0, 0, 0, False, _FLAG_CONVERR, name)


# --------------------------------------------------------------------------
# `toms748` validates more than the four C solvers, and its messages carry the
# offending values. Text read off scipy 1.18 on 2026-08-05:
#
#   toms748(f, -inf, 2.0)          ValueError: a is not finite -inf
#   toms748(f, 0.0, 2.0, maxiter=0) ValueError: maxiter must be greater than 0
#   toms748(f_inf, 0.0, 2.0)       ValueError: Invalid function value:
#                                              f(1.000000) -> inf
#   toms748(x*x+1, 0.0, 2.0)       ValueError: f(a) and f(b) must have
#                                    different signs, but f(0.000000e+00)=
#                                    1.000000e+00, f(2.000000e+00)=...
#
# The values are formatted in an `objmode` block, entered only on the branch
# that is about to raise.


def _fmt_notfinite(which, v):
    return "%s is not finite %s" % (which, v)


def _fmt_badfval(x, fx):
    return "Invalid function value: f(%f) -> %s " % (x, fx)


def _fmt_nanfval(x):
    # scipy's `_wrap_nan_raise` text, which is a DIFFERENT message from
    # `_callf`'s: NaN is caught by the wrapper before the solver sees it,
    # every other non-finite value by the solver.
    return "The function value at x=%s is NaN; solver cannot continue." % (x,)


def _fmt_signerr(a, b, fa, fb):
    return ("f(a) and f(b) must have different signs, but "
            "f(%e)=%e, f(%e)=%e " % (a, fa, b, fb))


@njit
def _notfinite_msg(which, v):
    with objmode(msg='unicode_type'):
        msg = _fmt_notfinite(which, v)
    return msg


@njit
def _badfval_msg(x, fx):
    with objmode(msg='unicode_type'):
        msg = _fmt_badfval(x, fx)
    return msg


@njit
def _nanfval_msg(x):
    with objmode(msg='unicode_type'):
        msg = _fmt_nanfval(x)
    return msg


@njit
def _signerr_msg(a, b, fa, fb):
    with objmode(msg='unicode_type'):
        msg = _fmt_signerr(a, b, fa, fb)
    return msg


def _fmt_interval(a, b):
    return "a and b are not an interval [%s, %s]" % (a, b)


def _emit_k_warning():
    warnings.warn("toms748: Overriding k: ->100", RuntimeWarning,
                  stacklevel=3)


@njit
def _interval_msg(a, b):
    with objmode(msg='unicode_type'):
        msg = _fmt_interval(a, b)
    return msg


@njit
def _warn_k():
    """scipy's ``RuntimeWarning`` when `k` is clamped down to 100."""
    with objmode():
        _emit_k_warning()


@njit
def _fct(f, x, cnt, args):
    """`_fcr` for `toms748`, which refuses any NON-FINITE value, not just NaN.

    scipy's `TOMS748Solver._callf` tests `np.isfinite(fx)` on every call and
    raises `ValueError`, where the four C solvers only reject NaN through
    `_wrap_nan_raise`.
    """
    cnt[0] += 1
    if isinstance(args, tuple):
        fx = f(x, *args)
    else:
        fx = f(x, args)
    if math.isnan(fx):
        raise ValueError(_nanfval_msg(x))
    if not math.isfinite(fx):
        raise ValueError(_badfval_msg(x, fx))
    return fx


@njit
def _isclose(a, b, rtol, atol):
    """Closeness test ``|a - b| <= atol + rtol * |b|``.

    The convergence criterion the bracketing and Newton-family solvers
    share, written once so the call sites cannot drift apart. The test is
    asymmetric -- the relative term scales with ``b`` alone -- so the
    argument order is part of the contract.
    """
    return abs(a - b) <= atol + rtol * abs(b)


@njit
def _ge_zero(p):
    """``p >= 0`` under numpy's ordering, which also orders complex.

    The secant seed is ``p1 += eps if p1 >= 0 else -eps``
    (`optimize/_zeros_py.py:248`), and by then scipy's `p1` is a numpy
    scalar, so ``>=`` on a complex compares the real parts and then the
    imaginary ones. Python and numba both refuse to order a complex at all,
    so the rule is written out. On a real `p` this is ``p >= 0``.
    """
    return p.real > 0.0 or (p.real == 0.0 and p.imag >= 0.0)


# ======================================================================
# BRACKETING ROOT FINDERS
# ======================================================================
# --------------------------------------------------------------------------
# Non-convergence: raise by default, as scipy does.
#
# scipy spells this flag `disp` on bisect/brentq/brenth/ridder/toms748/newton
# and has no flag at all on fixed_point, which raises unconditionally. D5
# renamed the flag to `disp` on the six names scipy publishes it for and left
# it `validate` on `fixed_point` and `root_scalar`, where scipy publishes no
# such parameter. `validate=False` is the parameter-sweep setting, where an
# exception would strand a run over thousands of points.
#
# The message texts are scipy's, measured on 1.18 on 2026-08-10, and carry no
# routine-name prefix and no hint naming the escape flag.


@njit
def _nonconv_msg(n):
    """scipy's iteration-limit text for the four C bracketing solvers.

    ``"Failed to converge after 2 iterations."`` Measured on scipy 1.18 for
    `bisect`, `brentq`, `brenth` and `ridder`, which share one C string.
    `toms748` and `fixed_point` append their own state; see
    `_nonconv_bracket_msg` and `_nonconv_value_msg`.
    """
    return "Failed to converge after " + str(n) + " iterations."


def _fmt_nonconv_bracket(n, a, b):
    """`toms748`'s text, which names the surviving bracket.

    scipy builds it as ``"Failed to converge after %d iterations, bracket is
    %s" % (self.iterations + 1, self.ab)``, and `self.ab` is a list of
    ``numpy.float64``, so the rendering carries numpy's own repr.
    """
    return "Failed to converge after %d iterations, bracket is %s" % (
        n, [np.float64(a), np.float64(b)])


@njit
def _nonconv_bracket_msg(n, a, b):
    with objmode(msg='unicode_type'):
        msg = _fmt_nonconv_bracket(n, a, b)
    return msg


def _fmt_nonconv_value(n, x):
    """`fixed_point`'s text, which names the last iterate and has no period."""
    return "Failed to converge after %d iterations, value is %s" % (n, x)


@njit
def _nonconv_value_msg(n, x):
    with objmode(msg='unicode_type'):
        msg = _fmt_nonconv_value(n, x)
    return msg


_K_TYPE_MSG = "'float' object cannot be interpreted as an integer"


def _k_int_only(k):
    """Refuse a non-integer `k`, as scipy's ``range(2, self.k + 2)`` does.

    Runs AFTER the ``k >= 1`` test, which is the order on both sides:
    measured on scipy 1.18, ``k=0.5`` gives ``ValueError: k too small
    (0.5 < 1)`` and ``k=1.5`` gives ``TypeError: 'float' object cannot be
    interpreted as an integer``. `operator.index` is scipy's own route to
    that text, so a numpy integer passes and a numpy float is named by its
    own type.
    """
    if k >= 1:
        operator.index(k)


def _fmt_k_small(k):
    """scipy's text at `optimize/_zeros_py.py:1479`, value included."""
    return "k too small (%s < 1)" % (k,)


@njit
def _k_small_msg(k):
    with objmode(msg='unicode_type'):
        msg = _fmt_k_small(k)
    return msg


def _fmt_brent_tol(tol):
    """scipy's text at `optimize/_optimize.py:2729`."""
    return "tolerance should be >= 0, got %r" % (tol,)


@njit
def _brent_tol_msg(tol):
    with objmode(msg='unicode_type'):
        msg = _fmt_brent_tol(tol)
    return msg


def _fmt_tol_small(tol):
    """scipy's `newton` text at `optimize/_zeros_py.py:301`."""
    return "tol too small (%g <= 0)" % (tol,)


@njit
def _tol_small_msg(tol):
    with objmode(msg='unicode_type'):
        msg = _fmt_tol_small(tol)
    return msg




def _fmt_tol_reached(d):
    """scipy's `f"Tolerance of {p1 - p0} reached."`."""
    return "Tolerance of %s reached." % (d,)


@njit
def _tol_reached(d):
    with objmode(head='unicode_type'):
        head = _fmt_tol_reached(d)
    return head


@njit
def _nonconv_val_msg(n, x):
    """scipy's plain non-convergence text for the `newton` family.

    ``"Failed to converge after {itr+1} iterations, value is {p}."``
    (`optimize/_zeros_py.py:403-405`). The five bracketing solvers use
    `_nonconv_msg`, because scipy's C carries no value there.
    """
    return _iterate_msg("", n, x)


@njit
def _zeroder_msg(n, x):
    """scipy's zero-derivative text, carrying the iterate.

    scipy raises ``"Derivative was zero. Failed to converge after {itr+1}
    iterations, value is {p0}."`` (`optimize/_zeros_py.py:330-336`). The
    iterate is formatted in an `objmode` block, which is entered only here,
    on the branch that is about to raise.
    """
    return _iterate_msg("Derivative was zero.", n, x)


@njit
def _stall_msg(n, d, x):
    """scipy's coincident-iterates text, carrying ``p1 - p0`` and ``p1``.

    scipy raises ``"Tolerance of {p1 - p0} reached. Failed to converge after
    {itr+1} iterations, value is {p1}."`` (`optimize/_zeros_py.py:377-386`).
    """
    return _iterate_msg(_tol_reached(d), n, x)


# --------------------------------------------------------------------------
# Tolerance floors on the bracketing solvers.
#
# scipy raises ValueError for `xtol <= 0` and for an `rtol` below its floor.
# Measured 2026-08-02 on scipy 1.18:
#
#     brentq/brenth/bisect/ridder  rtol=0.0  -> rtol too small (0 < 8.88178e-16)
#     toms748                      rtol=0.0  -> rtol too small (0 < 2.22045e-16)
#     all five                     xtol=0.0  -> xtol too small (0 <= 0)
#
# The two floors DIFFER: toms748's is `eps`, the other four use `4 * eps`.
# `_check_tol` therefore takes the floor rather than assuming one.
#
# The offending value IS interpolated, as scipy's is. numba's
# `str(<float64>)` returns the literal text '<object type:float64>', but an
# `objmode` block runs in the interpreter and can format the number, and a
# runtime string can then be raised: measured 2026-08-05, `raise
# ValueError(msg)` with `msg` built in `objmode` gives
# `ValueError: xtol too small (0 <= 0)`. The block takes the GIL and is
# entered only on the branch that is about to raise.

def _fmt_xtol(xtol):
    """scipy's text at `optimize/_zeros_py.py:601`."""
    return "xtol too small (%g <= 0)" % (xtol,)


def _fmt_rtol(rtol, floor):
    """scipy's text at `optimize/_zeros_py.py:603`."""
    return "rtol too small (%g < %g)" % (rtol, floor)


def _fmt_iterate(head, n, x):
    """scipy's `"... Failed to converge after N iterations, value is X."`."""
    lead = "%s " % (head,) if head else ""
    return "%sFailed to converge after %d iterations, value is %s." % (
        lead, n, x)


@njit
def _xtol_msg(xtol):
    with objmode(msg='unicode_type'):
        msg = _fmt_xtol(xtol)
    return msg


@njit
def _rtol_msg(rtol, floor):
    with objmode(msg='unicode_type'):
        msg = _fmt_rtol(rtol, floor)
    return msg


@njit
def _iterate_msg(head, n, x):
    with objmode(msg='unicode_type'):
        msg = _fmt_iterate(head, n, x)
    return msg


@njit
def _check_tol(xtol, rtol, floor):
    """Reject a tolerance scipy rejects.  Raises ``ValueError``."""
    if xtol <= 0.0:
        raise ValueError(_xtol_msg(xtol))
    if rtol < floor:
        raise ValueError(_rtol_msg(rtol, floor))


# --------------------------------------------------------------------------
# The two RuntimeWarnings scipy's `newton` emits when it is not raising.
#
# Text read off scipy 1.18 `optimize/_zeros_py.py:331` and `:378` on
# 2026-08-05. scipy issues them on the `disp=False` exits only, so they are
# reached here on the `validate=False` exits only. It does NOT warn on plain
# non-convergence.
#
# `warnings.warn` is not typeable by numba; an `objmode` block runs its body
# in the interpreter, so the warning reaches the ordinary Python machinery and
# `catch_warnings`, `simplefilter`, `-W` and `PYTHONWARNINGS` all see it. The
# block takes the GIL and is entered only on the branch that warns.

_NEWTON_MSG_ZERODER = "Derivative was zero."


def _emit_zeroder_warning():
    warnings.warn(_NEWTON_MSG_ZERODER, RuntimeWarning, stacklevel=2)


def _emit_stall_warning(d):
    warnings.warn("Tolerance of %s reached." % (d,), RuntimeWarning,
                  stacklevel=2)


@njit
def _warn_zeroder():
    """scipy's zero-derivative ``RuntimeWarning``."""
    with objmode():
        _emit_zeroder_warning()


@njit
def _warn_stall(d):
    """scipy's coincident-iterates ``RuntimeWarning``, carrying ``p1 - p0``."""
    with objmode():
        _emit_stall_warning(d)


# --------------------------------------------------------------------------
# `fminbound`'s two `OptimizeWarning`s.
#
# scipy's `disp` defaults to 1 on `fminbound`, so a default-shaped call warns
# on both failure exits, through `_endprint` ->
# `_print_success_message_or_warn` (`optimize/_optimize.py:1529-1533`,
# `:3655-3669`). Text read off scipy 1.18 on 2026-08-05, backslash-n included.

_FB_MSG_MAXFUN = ("\nMaximum number of function evaluations exceeded --- "
                  "increase maxfun argument.\n")
_FB_MSG_NAN = "\nNaN result encountered."


def _emit_fminbound_warning(flag):
    from ._lsq import OptimizeWarning
    warnings.warn(_FB_MSG_MAXFUN if flag == 1 else _FB_MSG_NAN,
                  OptimizeWarning, stacklevel=2)


@njit
def _warn_fminbound(flag):
    """scipy's ``OptimizeWarning`` on a `fminbound` failure exit."""
    with objmode():
        _emit_fminbound_warning(flag)


#: scipy's text at `_minimize.py:1002-1006`, read off 1.18 on 2026-08-05.
#: It fires once per call, only when a tolerance is supplied to
#: ``method='bounded'``, and it is a `RuntimeWarning` rather than an
#: `OptimizeWarning`.
_MS_MSG_BOUNDED_RTOL = ("Method 'bounded' does not support relative tolerance"
                        " in x; defaulting to absolute tolerance.")


def _emit_bounded_tol_warning():
    warnings.warn(_MS_MSG_BOUNDED_RTOL, RuntimeWarning, stacklevel=2)


@njit
def _warn_bounded_tol():
    """scipy's ``RuntimeWarning`` for a tolerance given to ``'bounded'``."""
    with objmode():
        _emit_bounded_tol_warning()


@njit
def _check_newton_args(tol, maxiter):
    """Reject a tolerance or an iteration cap scipy's `newton` rejects.

    scipy raises ``ValueError(f"tol too small ({tol:g} <= 0)")`` and
    ``ValueError("maxiter must be greater than 0")`` before doing any work
    (``optimize/_zeros_py.py:300-304``).
    """
    if tol <= 0.0:
        raise ValueError(_tol_small_msg(tol))
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")


@njit
def _check_maxiter(maxiter):
    """Reject a negative iteration cap, as scipy's C front end does.

    Measured against scipy 1.18 on 2026-08-05: ``bisect(f, 0.0, 2.0,
    maxiter=-1)`` raises ``ValueError: maxiter must be >= 0``, and
    ``maxiter=0`` raises ``RuntimeError`` through the non-convergence path.
    Without this check a negative cap runs the loop zero times and reports a
    convergence failure, which is a different diagnosis of a bad argument.
    """
    if maxiter < 0:
        raise ValueError("maxiter must be >= 0")


#: Message for a `full_output` that is not known when the call compiles.
#: scipy's `full_output` selects the RETURN SHAPE, and a compiled function has
#: one return type per signature, so the flag has to be a constant the chooser
#: can read. A literal, an omitted default and a module-level constant all are;
#: a variable is not.
_FO_MSG = ("%s: full_output must be a compile-time constant inside @njit, "
           "because it selects the return shape. Write full_output=True or "
           "full_output=False literally at the call site, or call from Python "
           "where a runtime value is fine.")


#: `fminbound`'s `disp=3`. scipy prints a per-iteration table whose
#: ``Procedure`` column names the step type each iteration took, golden against
#: parabolic. `_bounded_min` does not report that, so the column cannot be
#: filled. Measured on scipy 1.18, converged solve of ``(x-1)**2`` on
#: ``[-5, 5]``: disp=0 and disp=1 print 0 characters, disp=2 prints 116 and
#: disp=3 prints 467.
_FB_DISP3_MSG = (
    "fminbound: disp=3 prints scipy's per-iteration Func-count table, whose "
    "Procedure column names the step type (golden or parabolic) each "
    "iteration took. The bounded-Brent core does not report it, so the table "
    "cannot be filled. Use disp=2 for scipy's termination block, or "
    "full_output=True for the counters.")


def _fb_print_ok_py(xtol):
    """scipy's `fminbound` termination block, verbatim, at ``disp >= 2``.

    ``_endprint`` (`optimize/_optimize.py`) prints scipy's two success lines
    and then ``(using xtol = <value> )``. numba's own ``str(<float64>)``
    returns the literal text '<object type:float64>', so the number can only
    come from a ``numba.objmode`` block, which takes the GIL.
    """
    print(_MS_MSG_OK)
    print("(using xtol = ", xtol, ")")


@njit
def _fb_print_ok(xtol):
    with objmode():
        _fb_print_ok_py(xtol)


# scipy reaches ``{'del2': True, 'iteration': False}[method]`` and lets the
# KeyError out, so its exception class is `KeyError` and its text is the repr
# of the bad value. Measured on scipy 1.18: ``fixed_point(f, 1.0,
# method='nope')`` raises ``KeyError: 'nope'``. Reproduced, class included.


def _fp_use_accel(method):
    """``method`` -> whether Aitken's del-squared acceleration runs."""
    if method == 'del2':
        return True
    if method == 'iteration':
        return False
    raise KeyError(method)


def _fp_use_accel_ty(method):
    """Compile-time twin of `_fp_use_accel`, for a chooser.

    A string ARGUMENT is `unicode_type` and carries no value, so only a
    literal or the omitted default can select the loop. A runtime string
    would need both loops compiled and a branch, which is reachable; it is
    not offered because scipy's `method` is a constant at every call site
    this package has seen, and adding it later breaks nothing.
    """
    if isinstance(method, str):                     # omitted default
        return _fp_use_accel(method)
    if isinstance(method, types.Omitted):
        return _fp_use_accel(method.value)
    if isinstance(method, types.StringLiteral):
        return _fp_use_accel(method.literal_value)
    raise TypingError(
        "fixed_point: method must be a compile-time constant inside @njit, "
        "because it selects the iteration. Write method='del2' or "
        "method='iteration' literally at the call site.")


def _brack_args(brack):
    """scipy's `brack` in, ``(xa0, xb0, xc0, use3)`` out.

    scipy's rule, `Brent.get_bracket_info`: ``None`` runs the downhill search
    from its own defaults, a 2-sequence runs it from those points, a
    3-sequence is used directly, and any other length raises. The defaults for
    the ``None`` case are `bracket`'s own ``xa=0.0, xb=1.0``.
    """
    if brack is None:
        return 0.0, 1.0, 0.0, False
    n = len(brack)
    if n == 2:
        return float(brack[0]), float(brack[1]), 0.0, False
    if n == 3:
        return float(brack[0]), float(brack[1]), float(brack[2]), True
    raise ValueError(_BRACK_LEN_MSG)


@overload(_brack_args)
def _brack_args_ovl(brack):
    """Compiled twin of `_brack_args`; the LENGTH is a typing-time question.

    A tuple's length is part of its numba type, so the three cases fold to
    constants in the chooser and the compiled body carries no branch. A
    wrong length is refused when the call compiles rather than when it runs,
    which is the only difference from the Python body.
    """
    if _is_none_ty(brack):
        def impl(brack):
            return 0.0, 1.0, 0.0, False
        return impl
    if isinstance(brack, types.BaseTuple):
        n = len(brack)
        if n == 2:
            def impl(brack):
                return np.float64(brack[0]), np.float64(brack[1]), 0.0, False
            return impl
        if n == 3:
            def impl(brack):
                return (np.float64(brack[0]), np.float64(brack[1]),
                        np.float64(brack[2]), True)
            return impl
        raise TypingError(_BRACK_LEN_MSG)
    if isinstance(brack, types.Array):
        def impl(brack):
            n = brack.size
            if n == 2:
                return np.float64(brack[0]), np.float64(brack[1]), 0.0, False
            if n == 3:
                return (np.float64(brack[0]), np.float64(brack[1]),
                        np.float64(brack[2]), True)
            raise ValueError(_BRACK_LEN_MSG)
        return impl
    raise TypingError(
        "brack must be None, a 2-tuple, a 3-tuple or an array of length 2 "
        "or 3. A python list is not typeable as an argument to compiled "
        "code; use a tuple.")


from .._lib._typing import _is_none as _is_none_ty   # noqa: E402


def _lit_fo(name, full_output):
    """Compile-time value of a `full_output`-style flag, for a chooser.

    Three spellings carry a value and one does not. An OMITTED default reaches
    the chooser as the raw Python ``False`` rather than as ``types.Omitted``,
    which is the trap `CLAUDE.md` records for optional arguments generally, so
    the plain ``bool`` test has to come first.
    """
    if isinstance(full_output, bool):               # omitted, raw python value
        return full_output
    if isinstance(full_output, types.Omitted):
        return bool(full_output.value)
    if isinstance(full_output, types.BooleanLiteral):
        return full_output.literal_value
    raise TypingError(_FO_MSG % name)


def _fo_or_false(full_output):
    """`_lit_fo` without the refusal, for an arm that is about to raise.

    A refusing arm still has to declare the return SHAPE the caller unpacks,
    and the shape follows `full_output`. The refusal it carries takes
    precedence over the one `_lit_fo` would raise, so a `full_output` that is
    a runtime variable reads as ``False`` here rather than ending the compile.
    """
    if isinstance(full_output, bool):
        return full_output
    if isinstance(full_output, types.Omitted):
        return bool(full_output.value)
    if isinstance(full_output, types.BooleanLiteral):
        return full_output.literal_value
    return False


@njit
def _bisect_core(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 disp=True):
    """Bisection engine behind `bisect`, returning the full `RootResults`.

    Private. The public `bisect` returns scipy's shape, a bare float or
    ``(x, RootResults)`` under `full_output`; this is where the object always
    exists, so `root_scalar` can reach the whole result without going through
    the public entry's return-shape choice.

    Parameters are `bisect`'s, with `disp` in place of `full_output`/`disp`:
    the object is always built and the caller decides what to keep.
    """
    _check_tol(xtol, rtol, _RTOL)
    _check_maxiter(maxiter)
    cnt = np.zeros(1, np.int64)
    xa = float(a)
    xb = float(b)
    fa = _fcr(f, xa, cnt, args)
    fb = _fcr(f, xb, cnt, args)
    if _sgn(fa) * _sgn(fb) > 0.0:
        raise ValueError("f(a) and f(b) must have different signs")
    if fa == 0.0:
        return RootResults(xa, 0, cnt[0], True, _FLAG_CONV, 'bisect')
    if fb == 0.0:
        return RootResults(xb, 0, cnt[0], True, _FLAG_CONV, 'bisect')
    # DELIBERATE DEVIATION, D-BISECT-1, D13's refusal extended to `bisect`.
    # Same cell, same reason, same position as `_brent_root`: an infinite
    # endpoint whose sign test passes runs scipy's full iteration budget and
    # then reports non-convergence. Measured on scipy 1.18,
    # `bisect(x*x - 2, 0.0, inf)`: RuntimeError after 100 iterations.
    if math.isinf(xa):
        raise ValueError(_notfinite_msg('a', xa))
    if math.isinf(xb):
        raise ValueError(_notfinite_msg('b', xb))
    dm = xb - xa
    xm = xa
    for i in range(maxiter):
        dm *= 0.5
        xm = xa + dm
        fm = _fcr(f, xm, cnt, args)
        if _sgn(fm) * _sgn(fa) >= 0.0:
            xa = xm
        if fm == 0.0 or abs(dm) < xtol + rtol * abs(xm):
            return RootResults(xm, i + 1, cnt[0], True, _FLAG_CONV, 'bisect')
    if disp:
        raise RuntimeError(_nonconv_msg(maxiter))
    # scipy's bisect.c returns the left endpoint of the surviving bracket
    # here, not the last midpoint. Measured 2026-08-05 on scipy 1.18:
    # bisect(x - 0.2, 0.0, 1.0, maxiter=2, disp=False) returns 0.0, which is
    # xa; the last midpoint was 0.25.
    return RootResults(xa, maxiter, cnt[0], False, _FLAG_CONVERR, 'bisect')


def bisect(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
           full_output=False, disp=True):
    """Bisection root-finder.

    **Callback style A**: ``f`` is a plain ``@njit`` function taking one
    float and returning one float.  No ``@cfunc``, no ``.address``, and
    with an ``args`` tuple beside it::

        @njit
        def f(x):
            return x * x - 2.0

        root = bisect(f, 0.0, 2.0)
        root, res = bisect(f, 0.0, 2.0, full_output=True)

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Continuous function whose root is wanted.
    a, b : float
        Bracket endpoints.  ``f(a)`` and ``f(b)`` must not have the same
        sign; if they do, ``ValueError`` is raised.  Either endpoint being
        an exact root returns immediately with ``iterations = 0``.
    args : tuple, optional
        Extra arguments for `f`, unpacked into every call as
        ``f(x, *args)``.  A non-tuple is taken as a single extra argument.
        Default ``()``.
    xtol : float, optional
        Absolute tolerance on the bracket width.  Default 2e-12.  Must be
        positive; ``ValueError`` otherwise.
    rtol : float, optional
        Relative tolerance; convergence when
        ``|dm| < xtol + rtol * |x|``.  Default ``4 * eps`` = 8.88e-16,
        which is also its floor.  A smaller value raises ``ValueError``.
    maxiter : int, optional
        Iteration cap.  Default 100.  Negative raises ``ValueError``.
    full_output : bool, optional
        ``False`` (default) returns the root alone.  ``True`` returns
        ``(x, RootResults)``.  Inside ``@njit`` it must be a compile-time
        constant; see `Notes`.
    disp : bool, optional
        ``True`` (default) raises ``RuntimeError`` when the iteration limit is
        reached.  ``False`` returns ``converged=False`` instead, with `root`
        set to the left endpoint of the surviving bracket.

    Returns
    -------
    x : float
        The estimated root, when ``full_output`` is False.
    (x, res) : tuple of (float, RootResults)
        When ``full_output`` is True.  `res` fields are reached by
        attribute, by index or by unpacking.

        root : float
            Best estimate of the root.
        iterations : int
            Iterations used.
        function_calls : int
            Evaluations of `f`. Counts every one, including the ones a
            solver discards.
        converged : bool
            True if a tolerance test was met before `maxiter`. Only ever
            False under ``disp=False``.
        flag : str
            ``'converged'``, or ``'convergence error'``.
        method : str
            The method that produced the result, by name.

    Raises
    ------
    ValueError
        If ``f(a)`` and ``f(b)`` have the same sign, so the interval is not
        known to bracket a root; if `a` or `b` is infinite and the sign test
        passes; if ``f`` returns NaN at any iterate; if ``xtol <= 0``; if
        `rtol` is below ``4 * eps``; or if ``maxiter < 0``.
    RuntimeError
        If `maxiter` is reached, unless ``disp=False``.
    numba.core.errors.TypingError
        From inside ``@njit``, if `full_output` is a runtime variable.

    See Also
    --------
    scipy.optimize.bisect : The scipy routine this mirrors.
    scijit.optimize.brentq : Faster on a well-behaved function; the usual
        choice.
    scijit.optimize.root_scalar : The bracketing methods behind a ``method``
        argument.

    Notes
    -----
    `full_output` selects the RETURN SHAPE, and a compiled function has one
    return type per signature, so inside ``@njit`` the flag has to be readable
    when the call compiles. A literal, an omitted default and a module-level
    constant all are; a variable is not, and raises `TypingError` naming the
    constraint. From Python a runtime value is fine.

    The result is a namedtuple, where scipy's is a ``dict`` subclass.
    See `RootResults`.

    ``iterations`` is 0 when `a` or `b` is an exact root. scipy's C returns
    before assigning that field and reports an indeterminate value read from
    uninitialised memory.

    An infinite `a` or `b` that passes the sign test raises ``ValueError``.
    scipy has no such check and runs to `maxiter`: measured on scipy 1.18,
    ``bisect(f, 0.0, inf)`` on ``x * x - 2`` reports ``RuntimeError: Failed
    to converge after 100 iterations``. An infinite endpoint that fails the
    sign test, is an exact root, or at which `f` returns NaN reaches the
    same outcome here as in scipy.

    Bisection uses only the SIGN of `f`, so no derivative or curvature
    estimate can mislead it, and the bracket halves every step: at most
    ``log2((b - a) / xtol)`` iterations. It is the slowest of the
    bracketing methods; prefer :func:`~scijit.optimize.brentq` unless `f`
    is badly behaved.

    Pure ``@njit``, no state and no callback slot, so it is safe to call from
    a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.bisect.html

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import bisect
    >>> @njit
    ... def f(x):
    ...     return x * x - 2.0
    >>> @njit
    ... def run():
    ...     return bisect(f, 0.0, 2.0)
    >>> round(run(), 10)
    1.4142135624
    >>> @njit
    ... def run_full():
    ...     return bisect(f, 0.0, 2.0, full_output=True)
    >>> x, res = run_full()
    >>> round(x, 10), res.converged, res.method
    (1.4142135624, True, 'bisect')
    """
    res = _bisect_core(f, a, b, args, xtol, rtol, maxiter, disp)
    if full_output:
        return res.root, res
    return res.root


@overload(bisect, prefer_literal=True)
def _bisect_ovl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                full_output=False, disp=True):
    if _lit_fo('bisect', full_output):
        def impl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
            res = _bisect_core(f, a, b, args, xtol, rtol, maxiter, disp)
            return res.root, res
    else:
        def impl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
            return _bisect_core(f, a, b, args, xtol, rtol, maxiter, disp).root
    return impl


@njit
def _brent_root(f, a, b, xtol, rtol, maxiter, hyperbolic, validate, name,
                cnt, args):
    """Brent's zeroin.  hyperbolic=False -> brentq (inverse quadratic),
    hyperbolic=True -> brenth (hyperbolic extrapolation)."""
    _check_tol(xtol, rtol, _RTOL)
    _check_maxiter(maxiter)
    xpre = float(a)
    xcur = float(b)
    fpre = _fcr(f, xpre, cnt, args)
    fcur = _fcr(f, xcur, cnt, args)
    if _sgn(fpre) * _sgn(fcur) > 0.0:
        raise ValueError("f(a) and f(b) must have different signs")
    if fpre == 0.0:
        return RootResults(xpre, 0, cnt[0], True, _FLAG_CONV, name)
    if fcur == 0.0:
        return RootResults(xcur, 0, cnt[0], True, _FLAG_CONV, name)
    # DELIBERATE DEVIATION, D13. scipy takes `a` and `b` as C `double` with no
    # finiteness test here, so an INFINITE endpoint whose sign test passes runs
    # the full iteration budget and then reports non-convergence. Measured on
    # scipy 1.18, `brentq(x*x - 2, 0.0, inf)`: RuntimeError after 100
    # iterations. The refusal is `toms748`'s, which is scipy's own text for the
    # one bracketing solver scipy does guard.
    #
    # THE TEST IS `isinf`, NOT `isfinite`, AND IT SITS HERE RATHER THAN ABOVE
    # THE TWO `_fcr` CALLS. Both placements refuse the same inputs; they
    # differ in which MESSAGE an infinite endpoint gets, and everything above
    # this line is a message the two libraries already agree on. Gridded
    # 2026-08-11 over three objectives crossed with three infinite-endpoint
    # cells: with the test above `_fcr`, ours differed from scipy in 9 cells
    # of 9; here it differs in the 2 that reach scipy's
    # wasted iterations, which is the cell D13 rules on. The other 7 are
    # scipy's sign-error and its NaN value message.
    if math.isinf(xpre):
        raise ValueError(_notfinite_msg('a', xpre))
    if math.isinf(xcur):
        raise ValueError(_notfinite_msg('b', xcur))
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0
    for i in range(maxiter):
        # scipy's brentq.c:67 is
        # `fpre != 0 && fcur != 0 && signbit(fpre) != signbit(fcur)`.
        # A product test loses this for two opposite-sign values whose
        # magnitudes multiply below the subnormal floor: the product is -0.0,
        # which is not < 0.0, and the bisection fallback is not recorded.
        if _sgn(fpre) * _sgn(fcur) < 0.0:
            xblk = xpre
            fblk = fpre
            spre = xcur - xpre
            scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre
            fpre = fcur
            fcur = fblk
            fblk = fpre
        delta = (xtol + rtol * abs(xcur)) / 2.0
        sbis = (xblk - xcur) / 2.0
        if fcur == 0.0 or abs(sbis) < delta:
            return RootResults(xcur, i + 1, cnt[0], True, _FLAG_CONV, name)
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                # interpolate (secant)
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                if hyperbolic:
                    stry = -fcur * (fblk - fpre) / (fblk * dpre - fpre * dblk)
                else:
                    stry = (-fcur * (fblk * dblk - fpre * dpre)
                            / (dblk * dpre * (fblk - fpre)))
            # C's `MIN(a, b)` is `((a) < (b) ? (a) : (b))`, which returns
            # the SECOND operand when the first is NaN; python's and numba's
            # `min` return the first (`Zeros/brentq.c:9,101`).
            _m1 = abs(spre)
            _m2 = 3.0 * abs(sbis) - delta
            if 2.0 * abs(stry) < (_m1 if _m1 < _m2 else _m2):
                spre = scur
                scur = stry
            else:
                spre = sbis
                scur = sbis
        else:
            spre = sbis
            scur = sbis
        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0.0 else -delta
        fcur = _fcr(f, xcur, cnt, args)
    if validate:
        raise RuntimeError(_nonconv_msg(maxiter))
    return RootResults(xcur, maxiter, cnt[0], False, _FLAG_CONVERR, name)


@njit
def _brentq_core(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
           disp=True):
    """Brentq engine behind `brentq`, returning the full `RootResults`.

    Private. The public `brentq` returns scipy's shape, a bare float or
    ``(x, RootResults)`` under `full_output`; the object always exists here,
    so `root_scalar` can reach every field without going through the public
    entry's return-shape choice.

    Parameters are `brentq`'s, with `disp` in place of `full_output`/`disp`.
    """
    return _brent_root(f, a, b, xtol, rtol, maxiter, False,
                       disp, 'brentq', np.zeros(1, np.int64), args)


def brentq(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
           full_output=False, disp=True):
    """Brent's method with inverse quadratic interpolation.

    The recommended general-purpose bracketing root-finder.

    **Callback style A**: ``f`` is a plain ``@njit`` ``f(x) -> float``.

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Continuous function whose root is wanted.
    a, b : float
        Bracket endpoints.  ``f(a)`` and ``f(b)`` must not have the same
        sign; if they do, ``ValueError`` is raised.  Either endpoint being
        an exact root returns immediately with ``iterations = 0``.
    args : tuple, optional
        Extra arguments for `f`, unpacked into every call as
        ``f(x, *args)``.  A non-tuple is taken as a single extra argument.
        Default ``()``.
    xtol : float, optional
        Absolute tolerance, default 2e-12.  Must be positive; ``ValueError``
        otherwise.
    rtol : float, optional
        Relative tolerance, default ``4 * eps`` = 8.88e-16, which is also
        its minimum allowed value.  A smaller value raises ``ValueError``.
    maxiter : int, optional
        Iteration cap.  Default 100.  Negative raises ``ValueError``.
    full_output : bool, optional
        ``False`` (default) returns the root alone.  ``True`` returns
        ``(x, RootResults)``.  Inside ``@njit`` it must be a compile-time
        constant; see `Notes`.
    disp : bool, optional
        ``True`` (default) raises ``RuntimeError`` when the iteration limit is
        reached. ``False`` returns ``converged=False`` instead.

    Returns
    -------
    x : float
        The estimated root, when ``full_output`` is False.
    (x, res) : tuple of (float, RootResults)
        When ``full_output`` is True.  `res` fields are reached by
        attribute, by index or by unpacking.

        root : float
            Root estimate.
        iterations : int
            Iterations used.
        function_calls : int
            Evaluations of `f`. Counts every one, including the ones a
            solver discards. For :func:`newton` with derivatives it also
            counts the derivative evaluations.
        converged : bool
            True if a tolerance test was met before `maxiter`.
        flag : str
            ``'converged'``, or ``'convergence error'``.
        method : str
            The method that produced the result, by name.

    Raises
    ------
    ValueError
        If ``f(a)`` and ``f(b)`` have the same sign; if `a` or `b` is
        infinite; if `f` returns NaN at any iterate; if ``xtol <= 0``; if
        `rtol` is below ``4 * eps``; or if ``maxiter < 0``.
    RuntimeError
        If `maxiter` is reached, unless ``disp=False``.
    numba.core.errors.TypingError
        From inside ``@njit``, if `full_output` is a runtime variable.

    See Also
    --------
    scipy.optimize.brentq : The scipy routine this mirrors.
    scijit.optimize.brenth : The same method with hyperbolic extrapolation.
    scijit.optimize.toms748 : Higher order, fewer evaluations on smooth `f`.
    scijit.optimize.bisect : Slower, and uses only the sign of `f`.

    Notes
    -----
    `full_output` selects the RETURN SHAPE, and a compiled function has one
    return type per signature, so inside ``@njit`` the flag has to be readable
    when the call compiles. A literal, an omitted default and a module-level
    constant all are; a variable is not, and raises `TypingError` naming the
    constraint. From Python a runtime value is fine.

    The result is a namedtuple, where scipy's is a ``dict`` subclass.
    See `RootResults`.

    ``iterations`` is 0 when `a` or `b` is an exact root. scipy's C returns
    before assigning that field and reports an indeterminate value read from
    uninitialised memory.

    An infinite `a` or `b` that passes the sign test raises ``ValueError``.
    scipy has no such check and runs to `maxiter`: measured on scipy 1.18,
    ``brentq(f, 0.0, inf)`` on ``x * x - 2`` reports ``RuntimeError: Failed
    to converge after 100 iterations``. An infinite endpoint that fails the
    sign test, is an exact root, or at which `f` returns NaN reaches the
    same outcome here as in scipy.

    Pure ``@njit``, safe to call from a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import brentq
    >>> @njit
    ... def f(x):
    ...     return x * x - 2.0
    >>> @njit
    ... def run():
    ...     return brentq(f, 0.0, 2.0)
    >>> round(run(), 12)
    1.414213562373
    >>> @njit
    ... def run_full():
    ...     return brentq(f, 0.0, 2.0, full_output=True)
    >>> x, res = run_full()
    >>> round(x, 12), res.converged, res.method
    (1.414213562373, True, 'brentq')
    """
    res = _brentq_core(f, a, b, args, xtol, rtol, maxiter, disp)
    if full_output:
        return res.root, res
    return res.root


@overload(brentq, prefer_literal=True)
def _brentq_ovl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                full_output=False, disp=True):
    if _lit_fo('brentq', full_output):
        def impl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
            res = _brentq_core(f, a, b, args, xtol, rtol, maxiter, disp)
            return res.root, res
    else:
        def impl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
            return _brentq_core(f, a, b, args, xtol, rtol, maxiter, disp).root
    return impl


@njit
def _brenth_core(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
           disp=True):
    """Brenth engine behind `brenth`, returning the full `RootResults`.

    Private. The public `brenth` returns scipy's shape, a bare float or
    ``(x, RootResults)`` under `full_output`; the object always exists here,
    so `root_scalar` can reach every field without going through the public
    entry's return-shape choice.

    Parameters are `brenth`'s, with `disp` in place of `full_output`/`disp`.
    """
    return _brent_root(f, a, b, xtol, rtol, maxiter, True,
                       disp, 'brenth', np.zeros(1, np.int64), args)


def brenth(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
           full_output=False, disp=True):
    """Brent's method with hyperbolic extrapolation.

    Identical to :func:`brentq` except that the interpolation step uses
    hyperbolic rather than inverse quadratic extrapolation.  Which one is
    faster depends on ``f``.  Convergence follows Bus and Dekker (1975).

    **Callback style A**: ``f`` is a plain ``@njit`` ``f(x) -> float``.

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Continuous function whose root is wanted.
    a, b : float
        Bracket endpoints.  ``f(a)`` and ``f(b)`` must not have the same
        sign; if they do, ``ValueError`` is raised.  Either endpoint being
        an exact root returns immediately with ``iterations = 0``.
    args : tuple, optional
        Extra arguments for `f`, unpacked into every call as
        ``f(x, *args)``.  A non-tuple is taken as a single extra argument.
        Default ``()``.
    xtol : float, optional
        Absolute tolerance, default 2e-12.  Must be positive; ``ValueError``
        otherwise.
    rtol : float, optional
        Relative tolerance, default ``4 * eps`` = 8.88e-16, which is also
        its floor.  A smaller value raises ``ValueError``.
    maxiter : int, optional
        Iteration cap.  Default 100.  Negative raises ``ValueError``.
    full_output : bool, optional
        ``False`` (default) returns the root alone.  ``True`` returns
        ``(x, RootResults)``.  Inside ``@njit`` it must be a compile-time
        constant; see `Notes`.
    disp : bool, optional
        ``True`` (default) raises ``RuntimeError`` when the iteration limit is
        reached. ``False`` returns ``converged=False`` instead.

    Returns
    -------
    x : float
        The estimated root, when ``full_output`` is False.
    (x, res) : tuple of (float, RootResults)
        When ``full_output`` is True.  `res` fields are reached by
        attribute, by index or by unpacking.

        root : float
            Root estimate.
        iterations : int
            Iterations used.
        function_calls : int
            Evaluations of `f`. Counts every one, including the ones a
            solver discards. For :func:`newton` with derivatives it also
            counts the derivative evaluations.
        converged : bool
            True if a tolerance test was met before `maxiter`.
        flag : str
            ``'converged'``, or ``'convergence error'``.
        method : str
            The method that produced the result, by name.

    Raises
    ------
    ValueError
        If ``f(a)`` and ``f(b)`` have the same sign; if `a` or `b` is
        infinite; if `f` returns NaN at any iterate; if ``xtol <= 0``; if
        `rtol` is below ``4 * eps``; or if ``maxiter < 0``.
    RuntimeError
        If `maxiter` is reached, unless ``disp=False``.
    numba.core.errors.TypingError
        From inside ``@njit``, if `full_output` is a runtime variable.

    See Also
    --------
    scipy.optimize.brenth : The scipy routine this mirrors.
    scijit.optimize.brentq : Inverse quadratic interpolation instead.
    scijit.optimize.toms748 : Higher order, fewer evaluations on smooth `f`.

    Notes
    -----
    `full_output` selects the RETURN SHAPE, and a compiled function has one
    return type per signature, so inside ``@njit`` the flag has to be readable
    when the call compiles. A literal, an omitted default and a module-level
    constant all are; a variable is not, and raises `TypingError` naming the
    constraint. From Python a runtime value is fine.

    The result is a namedtuple, where scipy's is a ``dict`` subclass.
    See `RootResults`.

    ``iterations`` is 0 when `a` or `b` is an exact root. scipy's C returns
    before assigning that field and reports an indeterminate value read from
    uninitialised memory.

    An infinite `a` or `b` that passes the sign test raises ``ValueError``.
    scipy has no such check and runs to `maxiter`: measured on scipy 1.18,
    ``brenth(f, 0.0, inf)`` on ``x * x - 2`` reports ``RuntimeError: Failed
    to converge after 100 iterations``. An infinite endpoint that fails the
    sign test, is an exact root, or at which `f` returns NaN reaches the
    same outcome here as in scipy.

    Pure ``@njit``, safe to call from a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brenth.html

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import brenth
    >>> @njit
    ... def f(x):
    ...     return x * x - 2.0
    >>> @njit
    ... def run():
    ...     return brenth(f, 0.0, 2.0)
    >>> round(run(), 12)
    1.414213562373
    >>> @njit
    ... def run_full():
    ...     return brenth(f, 0.0, 2.0, full_output=True)
    >>> x, res = run_full()
    >>> round(x, 12), res.converged, res.method
    (1.414213562373, True, 'brenth')
    """
    res = _brenth_core(f, a, b, args, xtol, rtol, maxiter, disp)
    if full_output:
        return res.root, res
    return res.root


@overload(brenth, prefer_literal=True)
def _brenth_ovl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                full_output=False, disp=True):
    if _lit_fo('brenth', full_output):
        def impl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
            res = _brenth_core(f, a, b, args, xtol, rtol, maxiter, disp)
            return res.root, res
    else:
        def impl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
            return _brenth_core(f, a, b, args, xtol, rtol, maxiter, disp).root
    return impl


@njit
def _ridder_core(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
           disp=True):
    """Ridder engine behind `ridder`, returning the full `RootResults`.

    Private. The public `ridder` returns scipy's shape, a bare float or
    ``(x, RootResults)`` under `full_output`; the object always exists here,
    so `root_scalar` can reach every field without going through the public
    entry's return-shape choice.

    Parameters are `ridder`'s, with `disp` in place of `full_output`/`disp`.
    """
    _check_tol(xtol, rtol, _RTOL)
    _check_maxiter(maxiter)
    cnt = np.zeros(1, np.int64)
    xl = float(a)
    xh = float(b)
    tol = xtol + rtol * min(abs(xl), abs(xh))
    fl = _fcr(f, xl, cnt, args)
    fh = _fcr(f, xh, cnt, args)
    if fl == 0.0:
        return RootResults(xl, 0, cnt[0], True, _FLAG_CONV, 'ridder')
    if fh == 0.0:
        return RootResults(xh, 0, cnt[0], True, _FLAG_CONV, 'ridder')
    if _sgn(fl) * _sgn(fh) > 0.0:
        raise ValueError("f(a) and f(b) must have different signs")
    # scipy declares `xn = 0.0` and returns it untouched when the loop never
    # runs (`Zeros/ridder.c:22,88`), so `maxiter=0` with `disp=False` gives
    # root 0.0 rather than the bracket midpoint. Every later iteration
    # overwrites this, so the seed is only observable on that exit.
    xnew = 0.0
    for i in range(maxiter):
        dm = 0.5 * (xh - xl)
        xm = xl + dm
        fm = _fcr(f, xm, cnt, args)
        if fm == 0.0:
            return RootResults(xm, i + 1, cnt[0], True, _FLAG_CONV, 'ridder')
        # Ridders' step, normalised by fl. The direct form
        # sqrt(fm*fm - fl*fh) underflows to zero once the function values
        # reach about 1e-180, which scipy fixed for 1.18 (PR #24462).
        ratio = fm / fl
        dn = dm * ratio / math.sqrt(ratio * ratio - fh / fl)
        # keep the new iterate clear of the bracket end
        step = min(abs(dn), abs(dm) - 0.5 * tol)
        xnew = xm + (step if dn > 0.0 else -step)
        fnew = _fcr(f, xnew, cnt, args)
        # keep a valid sign-changing bracket
        if _sgn(fm) != _sgn(fnew):
            xl = xnew
            fl = fnew
            xh = xm
            fh = fm
        elif _sgn(fl) != _sgn(fnew):
            xh = xnew
            fh = fnew
        else:
            xl = xnew
            fl = fnew
        tol = xtol + rtol * abs(xnew)
        if fnew == 0.0 or abs(xh - xl) < tol:
            return RootResults(xnew, i + 1, cnt[0], True, _FLAG_CONV,
                               'ridder')
    if disp:
        raise RuntimeError(_nonconv_msg(maxiter))
    return RootResults(xnew, maxiter, cnt[0], False, _FLAG_CONVERR, 'ridder')


# ----------------------------------------------------------------------
# TOMS 748 (Alefeld, Potra & Shi) -- transcribed from scipy's Python
# TOMS748Solver, scalar, k = 1.
# ----------------------------------------------------------------------


def ridder(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
           full_output=False, disp=True):
    """Ridders' method.

    Exponential-correction bracketer (the classic ``zriddr``): each
    iteration evaluates ``f`` twice and converges quadratically, so it
    often beats bisection while keeping a guaranteed bracket.

    **Callback style A**: ``f`` is a plain ``@njit`` ``f(x) -> float``.

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Continuous function whose root is wanted.
    a, b : float
        Bracket endpoints.  ``f(a)`` and ``f(b)`` must not have the same
        sign; if they do, ``ValueError`` is raised.  Either endpoint being
        an exact root returns immediately with ``iterations = 0``.
    args : tuple, optional
        Extra arguments for `f`, unpacked into every call as
        ``f(x, *args)``.  A non-tuple is taken as a single extra argument.
        Default ``()``.
    xtol : float, optional
        Absolute tolerance on the bracket width.  Default 2e-12.  Must be
        positive; ``ValueError`` otherwise.
    rtol : float, optional
        Relative tolerance; convergence when
        ``|b - a| < xtol + rtol * |x|``.  Default ``4 * eps``, which is
        also its floor.  A smaller value raises ``ValueError``.
    maxiter : int, optional
        Iteration cap.  Default 100.  Each iteration costs two `f`
        evaluations.  Negative raises ``ValueError``.
    full_output : bool, optional
        ``False`` (default) returns the root alone.  ``True`` returns
        ``(x, RootResults)``.  Inside ``@njit`` it must be a compile-time
        constant; see `Notes`.
    disp : bool, optional
        ``True`` (default) raises ``RuntimeError`` when the iteration limit is
        reached. ``False`` returns ``converged=False`` instead.

    Returns
    -------
    x : float
        The estimated root, when ``full_output`` is False.
    (x, res) : tuple of (float, RootResults)
        When ``full_output`` is True.  `res` fields are reached by
        attribute, by index or by unpacking.

        root : float
            Root estimate.
        iterations : int
            Iterations used.
        function_calls : int
            Evaluations of `f`. Counts every one, including the ones a
            solver discards. For :func:`newton` with derivatives it also
            counts the derivative evaluations.
        converged : bool
            True if a tolerance test was met before `maxiter`.
        flag : str
            ``'converged'``, or ``'convergence error'``.
        method : str
            The method that produced the result, by name.

    Raises
    ------
    ValueError
        If ``f(a)`` and ``f(b)`` have the same sign; if ``f`` returns NaN at
        any iterate; if ``xtol <= 0``; if `rtol` is below ``4 * eps``; or if
        ``maxiter < 0``.
    RuntimeError
        If `maxiter` is reached, unless ``disp=False``.
    numba.core.errors.TypingError
        From inside ``@njit``, if `full_output` is a runtime variable.

    See Also
    --------
    scipy.optimize.ridder : The scipy routine this mirrors.
    scijit.optimize.brentq : Usually fewer evaluations.
    scijit.optimize.bisect : Slower, and uses only the sign of `f`.

    Notes
    -----
    `full_output` selects the RETURN SHAPE, and a compiled function has one
    return type per signature, so inside ``@njit`` the flag has to be readable
    when the call compiles. A literal, an omitted default and a module-level
    constant all are; a variable is not, and raises `TypingError` naming the
    constraint. From Python a runtime value is fine.

    The result is a namedtuple, where scipy's is a ``dict`` subclass.
    See `RootResults`.

    ``iterations`` is 0 when `a` or `b` is an exact root. scipy's C returns
    before assigning that field and reports an indeterminate value read from
    uninitialised memory.

    The convergence test uses ``|x|`` where scipy uses ``x``. scipy's
    in-loop tolerance is ``xtol + rtol * xn`` with no absolute value, which
    is negative for a root at a large negative `x`: at ``xn = -1e10`` and the
    default `rtol` it is -4.4e-06, and ``|b - a|`` never falls below it, so
    scipy runs to `maxiter` unless some ``f(xn)`` is exactly zero. This
    converges there instead.

    Pure ``@njit``, **prange-safe**.

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import ridder
    >>> @njit
    ... def f(x):
    ...     return x * x - 2.0
    >>> @njit
    ... def run():
    ...     return ridder(f, 0.0, 2.0)
    >>> round(run(), 12)
    1.414213562372
    >>> @njit
    ... def run_full():
    ...     return ridder(f, 0.0, 2.0, full_output=True)
    >>> x, res = run_full()
    >>> round(x, 12), res.converged, res.method
    (1.414213562372, True, 'ridder')
    """
    res = _ridder_core(f, a, b, args, xtol, rtol, maxiter, disp)
    if full_output:
        return res.root, res
    return res.root


@overload(ridder, prefer_literal=True)
def _ridder_ovl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                full_output=False, disp=True):
    if _lit_fo('ridder', full_output):
        def impl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
            res = _ridder_core(f, a, b, args, xtol, rtol, maxiter, disp)
            return res.root, res
    else:
        def impl(f, a, b, args=(), xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
            return _ridder_core(f, a, b, args, xtol, rtol, maxiter, disp).root
    return impl


@njit(error_model='numpy')
def _t748_inv_poly_zero(a, b, c, d, fa, fb, fc, fd):
    """Inverse cubic interpolation: poly through (f, x) points, at f=0.

    Neville's algorithm on nodes = f-values, data = x-values, transcribed
    from scipy's ``_interpolated_poly`` (`optimize/_zeros_py.py`) INCLUDING
    its two tables and its final summation order.

    An in-place single-array Neville computes the same polynomial value and
    associates the arithmetic differently. Measured 2026-08-05 on
    ``x*x - 2`` over [0, 2]: the two agreed for seven evaluations and parted
    in the last two digits at the eighth, which changed the number of
    iterations left to run and made `iterations` and `function_calls` differ
    from scipy's, 4 and 9 against 5 and 11. The value was never wrong; the
    counters were. Keeping scipy's association removes both.
    """
    xv = np.empty(4)          # the NODES, which are the f-values
    xv[0] = fa
    xv[1] = fb
    xv[2] = fc
    xv[3] = fd
    Q = np.zeros((4, 4))
    D = np.zeros((4, 4))
    Q[0, 0] = a
    Q[1, 0] = b
    Q[2, 0] = c
    Q[3, 0] = d
    for i in range(4):
        D[i, 0] = Q[i, 0]
    for k in range(1, 4):
        for j in range(4 - k):
            alpha = D[k + j, k - 1] - Q[k - 1 + j, k - 1]
            diffik = xv[j] - xv[k + j]
            # scipy evaluates at x = 0, so `(xvals - x)` is just the node
            Q[k + j, k] = xv[k + j] / diffik * alpha
            D[k + j, k] = xv[j] / diffik * alpha
    # scipy returns `np.sum(Q[-1, 1:]) + Q[-1, 0]`, in that order
    return ((Q[3, 1] + Q[3, 2]) + Q[3, 3]) + Q[3, 0]


@njit(error_model='numpy')
def _t748_newton_quadratic(a, b, fa, fb, d, fd, k):
    """Newton-quadratic step: k Newton iterations on the quadratic through
    (a, fa), (b, fb), (d, fd)."""
    B = (fb - fa) / (b - a)
    fbd = (fd - fb) / (d - b)
    A = (fbd - B) / (d - a)
    if A == 0.0:
        return a - fa / B
    if _sgn(A) * _sgn(fa) > 0.0:
        r = a
    else:
        r = b
    for _ in range(k):
        P = (A * (r - b) + B) * (r - a) + fa
        Pp = B + A * (2.0 * r - a - b)
        r1 = r - P / Pp
        if not (a < r1 < b):
            if a < r < b:
                return r
            r = 0.5 * (a + b)
            break
        r = r1
    return r


@njit
def _t748_notclose(f0, f1, f2, f3, atol):
    """True when the four f-values are distinct enough to interpolate.

    TOMS 748 takes an inverse-cubic step through four points, and its
    divided differences blow up on a zero, a non-finite value, or any two
    ordinates within ``atol``. False sends the caller to the
    Newton-quadratic step instead. scipy's ``TOMS748Solver._notclose``.
    """
    fs = (f0, f1, f2, f3)
    for v in fs:
        if v == 0.0 or not math.isfinite(v):
            return False
    for i in range(4):
        for j in range(i + 1, 4):
            if abs(fs[i] - fs[j]) <= atol:
                return False
    return True


@njit(error_model='numpy')
def _toms748_core(f, a, b, args=(), k=1, xtol=_XTOL, rtol=_RTOL, maxiter=100,
            disp=True):
    """Engine behind `toms748`, returning the full `RootResults`.

    Private. The public `toms748` returns scipy's shape, a bare float or
    ``(x, RootResults)`` under `full_output`; the object always exists here,
    so `root_scalar` can reach every field without going through the public
    entry's return-shape choice.

    Parameters are `toms748`'s, with `disp` in place of `full_output`/`disp`.
    """
    _check_tol(xtol, rtol, _EPS)
    cnt = np.zeros(1, np.int64)
    eps = _EPS
    aa = float(a)
    bb = float(b)
    # scipy checks maxiter and the bracket's finiteness at the front end,
    # before any evaluation (`optimize/_zeros_py.py:1467-1473`, `:1196-1199`).
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if not math.isfinite(aa):
        raise ValueError(_notfinite_msg('a', aa))
    if not math.isfinite(bb):
        raise ValueError(_notfinite_msg('b', bb))
    if not (aa < bb):
        raise ValueError(_interval_msg(aa, bb))
    # scipy rejects k < 1 at the front end and clamps a high k in
    # `configure`, noisily (`optimize/_zeros_py.py:1478`, `:1160-1171`).
    if not k >= 1:
        raise ValueError(_k_small_msg(k))
    kk = k
    if kk > 100:
        _warn_k()
        kk = 100
    fa = _fct(f, aa, cnt, args)
    if fa == 0.0:
        return RootResults(aa, 0, cnt[0], True, _FLAG_CONV, 'toms748')
    fb = _fct(f, bb, cnt, args)
    if fb == 0.0:
        return RootResults(bb, 0, cnt[0], True, _FLAG_CONV, 'toms748')
    if _sgn(fa) * _sgn(fb) > 0.0:
        raise ValueError(_signerr_msg(aa, bb, fa, fb))

    iterations = 0
    # ----- first step: secant -----
    if fa == fb:
        c = 0.5 * (aa + bb)
    else:
        if abs(fb) > abs(fa):
            c = (-fa / fb * bb + aa) / (1.0 - fa / fb)
        else:
            c = (-fb / fa * aa + bb) / (1.0 - fb / fa)
    if not (aa < c < bb):
        c = 0.5 * (aa + bb)
    fc = _fct(f, c, cnt, args)
    if fc == 0.0:
        return RootResults(c, iterations, cnt[0], True, _FLAG_CONV,
                           'toms748')
    # update bracket, discarded endpoint -> d
    if _sgn(fa) * _sgn(fc) > 0.0:
        d = aa
        fd = fa
        aa = c
        fa = fc
    else:
        d = bb
        fd = fb
        bb = c
        fb = fc
    has_e = False
    e = 0.0
    fe = 0.0
    iterations += 1

    while True:
        iterations += 1
        ab_width = bb - aa
        # scipy sets `c = None` ONCE, before the loop, and tests `c is None`
        # INSIDE it (`optimize/_zeros_py.py:1287`, `:1298`). So from the
        # second pass on, a failed interpolation test REUSES the previous
        # pass's `c` instead of taking a Newton-quadratic step. Resetting per
        # pass is a different algorithm, and it is only reachable now that
        # `k` is a parameter: at k = 1 the loop runs once and the two agree.
        c_set = False
        c = 0.0
        # ----- k Newton-quadratic / inverse-poly steps -----
        for nsteps in range(2, kk + 2):
            if has_e and _t748_notclose(fa, fb, fd, fe, 32.0 * eps):
                c0 = _t748_inv_poly_zero(aa, bb, d, e, fa, fb, fd, fe)
                if aa < c0 < bb:
                    c = c0
                    c_set = True
            if not c_set:
                c = _t748_newton_quadratic(aa, bb, fa, fb, d, fd, nsteps)
                c_set = True
            fc = _fct(f, c, cnt, args)
            if fc == 0.0:
                return RootResults(c, iterations, cnt[0], True, _FLAG_CONV,
                           'toms748')
            e = d
            fe = fd
            has_e = True
            if _sgn(fa) * _sgn(fc) > 0.0:
                d = aa
                fd = fa
                aa = c
                fa = fc
            else:
                d = bb
                fd = fb
                bb = c
                fb = fc

        # ----- double-length secant step -----
        if abs(fa) < abs(fb):
            uix = 0
            u = aa
            fu = fa
        else:
            uix = 1
            u = bb
            fu = fb
        A = (fb - fa) / (bb - aa)
        c = u - 2.0 * fu / A
        if abs(c - u) > 0.5 * (bb - aa):
            c = 0.5 * (aa + bb)
        else:
            if _isclose(c, u, eps, 0.0):
                # f-values of vastly different magnitude, or root near u
                fu_e = math.frexp(fa)[1] if uix == 0 else math.frexp(fb)[1]
                fo_e = math.frexp(fb)[1] if uix == 0 else math.frexp(fa)[1]
                if fu_e < fo_e - 50:
                    other = bb if uix == 0 else aa
                    c = (31.0 * u + other) / 32.0
                else:
                    mm = 1.0 if uix == 0 else -1.0
                    # scipy forms the whole adjustment first and adds it
                    # once: `adj = mm*|c|*rtol + mm*xtol; c = u + adj`.
                    # Adding the two terms to `u` in sequence associates
                    # differently and moved the iterate by ~1e-16 at k=2.
                    adj = mm * abs(c) * rtol + mm * xtol
                    c = u + adj
                if not (aa < c < bb):
                    c = 0.5 * (aa + bb)
        fc = _fct(f, c, cnt, args)
        if fc == 0.0:
            return RootResults(c, iterations, cnt[0], True, _FLAG_CONV,
                           'toms748')
        e = d
        fe = fd
        has_e = True
        if _sgn(fa) * _sgn(fc) > 0.0:
            d = aa
            fd = fa
            aa = c
            fa = fc
        else:
            d = bb
            fd = fb
            bb = c
            fb = fc

        # ----- bisect if the interval didn't shrink enough -----
        if bb - aa > 0.5 * ab_width:
            e = d
            fe = fd
            has_e = True
            z = 0.5 * (aa + bb)
            fz = _fct(f, z, cnt, args)
            if fz == 0.0:
                return RootResults(z, iterations, cnt[0], True, _FLAG_CONV,
                           'toms748')
            if _sgn(fa) * _sgn(fz) > 0.0:
                d = aa
                fd = fa
                aa = z
                fa = fz
            else:
                d = bb
                fd = fb
                bb = z
                fb = fz

        # ----- status -----
        if _isclose(aa, bb, rtol, xtol):
            return RootResults(0.5 * (aa + bb), iterations, cnt[0], True, _FLAG_CONV,
                           'toms748')
        if iterations >= maxiter:
            if disp:
                raise RuntimeError(
                    _nonconv_bracket_msg(iterations + 1, aa, bb))
            return RootResults(0.5 * (aa + bb), iterations, cnt[0], False,
                           _FLAG_CONVERR, 'toms748')


# ======================================================================
# NEWTON FAMILY (open methods)
# ======================================================================


def toms748(f, a, b, args=(), k=1, xtol=_XTOL, rtol=_RTOL, maxiter=100,
            full_output=False, disp=True):
    """TOMS Algorithm 748 (Alefeld, Potra and Shi).

    Bracketing method using inverse cubic and Newton-quadratic
    interpolation; asymptotically the fastest of the bracketers for a
    smooth ``f``.

    **Callback style A**: ``f`` is a plain ``@njit`` ``f(x) -> float``.

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Continuous function whose root is wanted.
    a, b : float
        Bracket endpoints.  Unlike the other bracketers here,
        ``a < b`` is required, and ``ValueError`` otherwise.  ``f(a)`` and
        ``f(b)`` must not have the same sign; if they do, ``ValueError`` is
        raised.  Either endpoint being an exact root returns immediately.
    args : tuple, optional
        Extra arguments for `f`, unpacked into every call as
        ``f(x, *args)``.  A non-tuple is taken as a single extra argument.
        Default ``()``.
    k : int, optional
        Newton-quadratic steps per iteration, ``k >= 1``.  Default 1.
        Below 1 raises ``ValueError``; a value at or above 1 that is not an
        integer raises ``TypeError``; above 100 is clamped to 100 with a
        ``RuntimeWarning``.
        ``k = 2`` is asymptotically the most efficient choice on a four
        times continuously differentiable `f`.
    xtol : float, optional
        Absolute tolerance, default 2e-12.  Must be positive; ``ValueError``
        otherwise.
    rtol : float, optional
        Relative tolerance, default ``4 * eps`` = 8.88e-16.  The FLOOR
        here is ``eps`` = 2.22e-16, a quarter of the other bracketing
        solvers'.  Below it, ``ValueError``.
    maxiter : int, optional
        Iteration cap.  Default 100.  ``maxiter < 1`` raises
        ``ValueError``.
    full_output : bool, optional
        ``False`` (default) returns the root alone.  ``True`` returns
        ``(x, RootResults)``.  Inside ``@njit`` it must be a compile-time
        constant; see `Notes`.
    disp : bool, optional
        ``True`` (default) raises ``RuntimeError`` when the iteration limit is
        reached. ``False`` returns ``converged=False`` instead.

    Returns
    -------
    x : float
        The estimated root, when ``full_output`` is False.
    (x, res) : tuple of (float, RootResults)
        When ``full_output`` is True.  `res` fields are reached by
        attribute, by index or by unpacking.

        root : float
            Root estimate.
        iterations : int
            Iterations used.
        function_calls : int
            Evaluations of `f`. Counts every one, including the ones a
            solver discards. For :func:`newton` with derivatives it also
            counts the derivative evaluations.
        converged : bool
            True if a tolerance test was met before `maxiter`.
        flag : str
            ``'converged'``, or ``'convergence error'``.
        method : str
            The method that produced the result, by name.

    Raises
    ------
    ValueError
        If ``a >= b``; if either bound is not finite; if ``f`` returns a
        non-finite value at any iterate; if ``f(a)`` and ``f(b)`` have the
        same sign; if ``xtol <= 0``; if `rtol` is below ``eps``; if
        ``maxiter < 1``; or if ``k < 1``.
    TypeError
        If `k` is at least 1 and is not an integer.
    RuntimeError
        If `maxiter` is reached, unless ``disp=False``.
    numba.core.errors.TypingError
        From inside ``@njit``, if `full_output` is a runtime variable.

    Warns
    -----
    RuntimeWarning
        If ``k > 100``, which is clamped to 100.

    See Also
    --------
    scipy.optimize.toms748 : The scipy routine this mirrors.
    scijit.optimize.brentq : Fewer operations per iteration.
    scijit.optimize.root_scalar : The bracketing methods behind a ``method``
        argument.

    Notes
    -----
    `full_output` selects the RETURN SHAPE, and a compiled function has one
    return type per signature, so inside ``@njit`` the flag has to be readable
    when the call compiles. A literal, an omitted default and a module-level
    constant all are; a variable is not, and raises `TypingError` naming the
    constraint. From Python a runtime value is fine.

    The result is a namedtuple, where scipy's is a ``dict`` subclass.
    See `RootResults`.

    This solver validates more than the other four bracketing routines,
    because scipy's does: a non-finite bound, a non-finite value from `f` at
    any iterate, and ``maxiter < 1`` are all rejected here and are not by
    ``bisect``, ``brentq``, ``brenth`` or ``ridder``. The messages carry the
    offending values, as scipy's do.

    Write an integer power as a product. numba compiles ``x ** 3`` to
    repeated multiplication where CPython calls ``pow``, so an objective
    spelled that way is not the same function in the two languages, and the
    iteration sequence can then part company: measured on
    ``lambda x: x ** 3 - x - 2.0`` over ``[1, 2]`` at ``k=3``,
    `function_calls` is 8 here and 9 in scipy, at the same `iterations` and
    the same root. Spelled ``x * x * x`` the same call agrees on all three.

    Pure ``@njit``, **prange-safe**.

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import toms748
    >>> @njit
    ... def f(x):
    ...     return x * x - 2.0
    >>> @njit
    ... def run():
    ...     return toms748(f, 0.0, 2.0)
    >>> round(run(), 12)
    1.414213562373
    >>> @njit
    ... def run_full():
    ...     return toms748(f, 0.0, 2.0, full_output=True)
    >>> x, res = run_full()
    >>> round(x, 12), res.converged, res.method
    (1.414213562373, True, 'toms748')
    """
    _k_int_only(k)
    res = _toms748_core(f, a, b, args, k, xtol, rtol, maxiter, disp)
    if full_output:
        return res.root, res
    return res.root


@overload(toms748, prefer_literal=True)
def _toms748_ovl(f, a, b, args=(), k=1, xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
    kt = k.value if isinstance(k, types.Omitted) else k
    if isinstance(kt, types.Float):
        # scipy tests `k >= 1` first and only then reaches `range`.
        if _fo_or_false(full_output):
            def impl(f, a, b, args=(), k=1, xtol=_XTOL, rtol=_RTOL,
                     maxiter=100, full_output=False, disp=True):
                if not k >= 1:
                    raise ValueError(_k_small_msg(k))
                if _always():
                    raise TypeError(_K_TYPE_MSG)
                return 0.0, _dead_root('toms748')
        else:
            def impl(f, a, b, args=(), k=1, xtol=_XTOL, rtol=_RTOL,
                     maxiter=100, full_output=False, disp=True):
                if not k >= 1:
                    raise ValueError(_k_small_msg(k))
                if _always():
                    raise TypeError(_K_TYPE_MSG)
                return 0.0
        return impl
    if _lit_fo('toms748', full_output):
        def impl(f, a, b, args=(), k=1, xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
            res = _toms748_core(f, a, b, args, k, xtol, rtol, maxiter, disp)
            return res.root, res
    else:
        def impl(f, a, b, args=(), k=1, xtol=_XTOL, rtol=_RTOL, maxiter=100,
                 full_output=False, disp=True):
            return _toms748_core(f, a, b, args, k, xtol, rtol, maxiter, disp).root
    return impl


@njit
def _secant(f, x0, x1=_NAN, tol=1.48e-8, maxiter=50, rtol=0.0,
           validate=True, has_x1=False, args=()):
    """Secant method -- ``scipy.optimize.newton`` with ``fprime=None``.

    Split out under its own name because a numba first-class function
    argument cannot be ``None``: there is no way to write
    ``newton(f, x0, fprime=None)`` inside ``@njit``, so the
    derivative-free variant gets a separate entry point.  See also
    :func:`newton`, which selects this loop when no derivative is given.

    **Callback style A**: ``f`` is a plain ``@njit`` ``f(x) -> float``.
    No bracket is needed, and no convergence is guaranteed.

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Function whose root is wanted.
    x0 : float or complex
        First starting guess.
    x1 : float or complex, optional
        Second starting guess.  The default is NaN, which means "use
        scipy's seed", ``x1 = x0 * (1 + 1e-4) + (1e-4 with x0's sign)``
        -- the same expression scipy uses when it has to invent a
        second point.  Passing ``x1 == x0`` raises ``ValueError``.
    tol : float, optional
        Absolute step tolerance; convergence when
        ``|p - p1| <= tol + rtol * |p1|``.  Default 1.48e-8, scipy's
        ``newton`` default.
    maxiter : int, optional
        Iteration cap.  Default 50.
    rtol : float, optional
        Relative step tolerance.  Default 0.0.
    validate : bool, optional
        ``True`` (default) raises ``RuntimeError`` when the iteration limit is
        reached, and on the undefined-step conditions below. ``False`` returns
        ``converged=False`` instead.

    Returns
    -------
    res : RootResults
        A namedtuple whose fields are reached by attribute, by index or by
        unpacking.

        root : float
            Best estimate. On a flat secant, where ``f(p0) == f(p1)``,
            the midpoint is returned.
        iterations : int
            Iterations used.
        function_calls : int
            Evaluations of `f`, plus the derivative evaluations, counted
            on the same counter.
        converged : bool
            True if a tolerance test was met before `maxiter`.
        flag : str
            ``'converged'``, or ``'convergence error'``.
        method : str
            The method that produced the result, by name.

    Raises
    ------
    ValueError
        If ``x1 == x0``, leaving no secant to draw.
    RuntimeError
        If `maxiter` is reached, or the two iterates coincide so the secant
        step is undefined. Both are suppressed by ``validate=False``.

    Notes
    -----
    Follows the iteration order of scipy's pure-Python ``_zeros_py.newton``,
    including the initial swap that puts the smaller ``|f|`` in ``p1``.
    Reached through :func:`newton` with no derivative.
    """
    _check_newton_args(tol, maxiter)
    cnt = np.zeros(1, np.int64)
    # scipy's own cast, `np.asarray(x0)[()] * 1.0` (`_zeros_py.py:192`). It
    # promotes an integer start to float and KEEPS a complex one complex,
    # which `float(x0)` does not, and is what carries D8's complex route
    # through the same engine source.
    p0 = x0 * 1.0
    # `has_x1` and not `isnan(x1)`: scipy's second seed is optional through
    # `x1=None`, so an explicitly-passed `x1=nan` is a VALUE there and
    # produces a nan root. Testing the value instead would make nan mean
    # "not given" and silently invent the seed.
    if not has_x1:
        eps = 1e-4
        p1 = p0 * (1.0 + eps)
        p1 += eps if _ge_zero(p1) else -eps
    else:
        if x1 == p0:
            raise ValueError("x1 and x0 must be different")
        p1 = x1 * 1.0
    q0 = _fc(f, p0, cnt, args)
    q1 = _fc(f, p1, cnt, args)
    if abs(q1) < abs(q0):
        p0, p1 = p1, p0
        q0, q1 = q1, q0
    p = p1
    for itr in range(maxiter):
        if q1 == q0:
            # scipy nests the report inside `if p1 != p0`, so two coincident
            # iterates with coincident values return quietly, with no
            # exception and no warning, even under `disp=True`
            # (`optimize/_zeros_py.py:381-386`).
            if p1 != p0:
                if validate:
                    raise RuntimeError(
                        _stall_msg(itr + 1, p1 - p0, p1))
                _warn_stall(p1 - p0)
            p = 0.5 * (p1 + p0)
            return RootResults(p, itr + 1, cnt[0], False, _FLAG_CONVERR,
                               'secant')
        if abs(q1) > abs(q0):
            p = (-q0 / q1 * p1 + p0) / (1.0 - q0 / q1)
        else:
            p = (-q1 / q0 * p0 + p1) / (1.0 - q1 / q0)
        if _isclose(p, p1, rtol, tol):
            return RootResults(p, itr + 1, cnt[0], True, _FLAG_CONV,
                               'secant')
        p0 = p1
        q0 = q1
        p1 = p
        q1 = _fc(f, p1, cnt, args)
    if validate:
        raise RuntimeError(_nonconv_val_msg(maxiter, p))
    return RootResults(p, maxiter, cnt[0], False, _FLAG_CONVERR,
                       'secant')


@njit
def _newton_core(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
           fprime2=None, x1=None, rtol=0.0, disp=True):
    """Engine behind `newton`, returning the full `RootResults`.

    Private. The public `newton` returns scipy's shape, a bare float or
    ``(x, RootResults)`` under `full_output`; the object always exists here,
    so `root_scalar` can reach every field without going through the public
    entry's return-shape choice.

    Parameters are `newton`'s, with `disp` in place of `full_output`/`disp`.
    """
    _check_newton_args(tol, maxiter)
    cnt = np.zeros(1, np.int64)
    if fprime is None:
        if fprime2 is not None:
            raise ValueError(
                "newton: fprime2 was given without fprime; Halley's method "
                "needs both derivatives")
        if x1 is None:
            return _secant(f, x0, _NAN, tol, maxiter, rtol, disp, False,
                           args)
        return _secant(f, x0, x1, tol, maxiter, rtol, disp, True, args)
    if fprime2 is not None:
        return _halley(f, x0, fprime, fprime2, tol, maxiter, rtol, disp,
                       args)
    p0 = x0 * 1.0
    for itr in range(maxiter):
        fval = _fc(f, p0, cnt, args)
        if fval == 0.0:
            return RootResults(p0, itr, cnt[0], True, _FLAG_CONV,
                               'newton')
        fder = _fc(fprime, p0, cnt, args)
        if fder == 0.0:
            if disp:
                raise RuntimeError(_zeroder_msg(itr + 1, p0))
            _warn_zeroder()
            return RootResults(p0, itr + 1, cnt[0], False, _FLAG_CONVERR,
                               'newton')
        p = p0 - fval / fder
        if _isclose(p, p0, rtol, tol):
            return RootResults(p, itr + 1, cnt[0], True, _FLAG_CONV,
                               'newton')
        p0 = p
    if disp:
        raise RuntimeError(_nonconv_val_msg(maxiter, p0))
    return RootResults(p0, maxiter, cnt[0], False, _FLAG_CONVERR,
                       'newton')


def newton(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
           fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True):
    """Newton-Raphson, secant and Halley root-finder.

    Which derivatives are supplied selects the method: no `fprime` runs the
    secant method, `fprime` alone runs Newton-Raphson, `fprime` with
    `fprime2` runs Halley's method.

    **Callback style A** for every callback: plain ``@njit`` functions
    of one float, passed as first-class arguments::

        @njit
        def f(x):
            return x * x - 2.0

        @njit
        def fp(x):
            return 2.0 * x

        res = newton(f, 1.0, fp)
        res.root, res.converged

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Function whose root is wanted.
    x0 : float or complex
        Starting guess.  A complex `x0` runs the same iteration over
        ``complex128`` and returns a complex root, on all three methods.
        An ARRAY of any shape runs a different algorithm over every element
        at once, described under `Notes`; the callbacks then receive the
        whole array and return one of the same shape.  A 0-d array is a
        scalar start.
    fprime : @njit function ``fprime(x) -> float``, or None, optional
        First derivative.  ``None`` (default) runs the secant method with
        `x1` as the second starting guess.  Positional third.
    args : tuple, optional
        Extra arguments, unpacked into every call as ``f(x, *args)`` and
        into `fprime` and `fprime2` the same way.  Must be a tuple;
        anything else raises ``TypeError``.  Default ``()``.
    tol : float, optional
        Absolute step tolerance; convergence when
        ``|p - p0| <= tol + rtol * |p0|``.  Default 1.48e-8.
    maxiter : int, optional
        Iteration cap.  Default 50.  ``maxiter < 1`` raises ``ValueError``.
    fprime2 : @njit function ``fprime2(x) -> float``, or None, optional
        Second derivative.  Supplying it alongside `fprime` runs Halley's
        method.  Supplying it without `fprime` raises ``ValueError``.
    x1 : float, complex or None, optional
        Second starting guess for the secant method, used only when `fprime`
        is ``None``.  The default ``None`` seeds it from `x0`,
        ``x0 * (1 + 1e-4)`` moved a further ``1e-4`` away from zero.  Any
        float is taken as the seed, NaN included.  ``x1 == x0`` raises
        ``ValueError``.
    rtol : float, optional
        Relative step tolerance.  Default 0.0.
    full_output : bool, optional
        ``False`` (default) returns the root alone.  ``True`` returns
        ``(x, RootResults)``.  Inside ``@njit`` it must be a compile-time
        constant; see `Notes`.
    disp : bool, optional
        ``True`` (default) raises ``RuntimeError`` when the iteration limit is
        reached. ``False`` returns ``converged=False`` instead.

    Returns
    -------
    x : float or complex
        The estimated root, when ``full_output`` is False.  Complex when
        `x0` is.
    (x, res) : tuple of (float or complex, RootResults)
        When ``full_output`` is True.  `res` fields are reached by
        attribute, by index or by unpacking.

        root : float or complex
            Best estimate.
        iterations : int
            Iterations used. An immediate exact hit, where
            ``f(x0) == 0``, reports 0.
        function_calls : int
            Evaluations of `f`, plus the derivative evaluations, counted
            on the same counter.
        converged : bool
            True if a tolerance test was met before `maxiter`.
        flag : str
            ``'converged'``, or ``'convergence error'``.
        method : str
            The method that produced the result, by name.
    res : ArrayNewtonResult
        When `x0` is an array of one dimension or more, and `full_output` is
        True.  `root`, `converged` and `zero_der` are each shaped like `x0`.
        Under ``full_output=False`` that route returns the `root` member
        alone.

    Raises
    ------
    ValueError
        If `fprime2` is given without `fprime`; if ``tol <= 0``; if
        ``maxiter < 1``; or if ``x1 == x0``.
    RuntimeError
        If `maxiter` is reached, or the derivative is zero so the next step is
        undefined, or two secant iterates coincide. All three are suppressed
        by ``disp=False``.  On an array `x0` the iteration limit raises only
        when EVERY element failed, and `disp` does not suppress it.

    Warns
    -----
    RuntimeWarning
        Under ``disp=False``, on two exits:
        ``"Derivative was zero."`` and ``"Tolerance of {p1 - p0} reached."``.
        Plain non-convergence does not warn.

        An array `x0` warns from four more, whatever `disp` is:
        ``"all derivatives were zero"`` and ``"some derivatives were zero"``
        when a derivative was supplied, ``"RMS of {rms:g} reached"`` on the
        secant, and ``"some failed to converge after {maxiter} iterations"``
        when part of the array reached the limit.

    See Also
    --------
    scipy.optimize.newton : The scipy routine this mirrors.
    scijit.optimize.root_scalar : The same methods, by name.
    scijit.optimize.brentq : Bracketed, and guaranteed to converge.

    Notes
    -----
    `full_output` selects the RETURN SHAPE, and a compiled function has one
    return type per signature, so inside ``@njit`` the flag has to be readable
    when the call compiles. A literal, an omitted default and a module-level
    constant all are; a variable is not, and raises `TypingError` naming the
    constraint. From Python a runtime value is fine.

    The result is a namedtuple, where scipy's is a ``dict`` subclass.
    See `RootResults`.

    Which of the three methods runs follows the VALUE of `fprime` and
    `fprime2` from Python, as it does in scipy: one variable holding either a
    function or ``None`` reaches both methods and reports the one it ran.
    Inside ``@njit`` the choice follows the TYPE instead, so it is settled
    when the call compiles, and a variable that may hold either does not
    type. Both spellings work there written on their own.

    `fprime2` without `fprime` raises. scipy's Halley block sits inside
    ``if fprime is not None``, so scipy ignores a lone `fprime2`, runs the
    secant loop and reports ``method='secant'``.

    An array `x0` runs a different algorithm, not the scalar loop repeated.
    Every element advances together, an element whose derivative is zero
    stops being updated while the others continue, and the outcome is
    reported per element in `converged` and `zero_der`. `x1`, `rtol` and
    `disp` are not read on that route, and the result carries no
    ``iterations`` or ``function_calls``. Its secant seed is
    ``eps ** 0.33``, where the scalar secant seeds with ``1e-4``.

    A LENGTH-1 array takes the array route here. scipy switches on
    ``np.size(x0) > 1``, so a length-1 array takes its scalar route instead.
    Under ``full_output`` scipy returns ``(array, RootResults)`` for that one
    size and this returns `ArrayNewtonResult`. The size of an array is a run-time
    value while its number of dimensions is not, and a compiled function has
    one return type per signature. A 0-d array is the scalar route on both
    sides.

    An integer array `x0` returns float64. scipy returns the input integer
    dtype in the single case where no element is ever updated, which is an
    `x0` whose every element is already an exact root.

    Newton's method is not bracketed, so a bad start can diverge. There is no
    warning on divergence, only the iteration limit.

    Pure ``@njit``, safe to call from a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import newton
    >>> @njit
    ... def f(x):
    ...     return x * x - 2.0
    >>> @njit
    ... def fp(x):
    ...     return 2.0 * x
    >>> @njit
    ... def run():
    ...     return newton(f, 1.0, fp)
    >>> round(run(), 12)
    1.414213562373
    >>> @njit
    ... def run_full():
    ...     return newton(f, 1.0, fp, full_output=True)
    >>> x, res = run_full()
    >>> round(x, 12), res.converged, res.iterations
    (1.414213562373, True, 5)

    With no derivative, which runs the secant method:

    >>> @njit
    ... def run_secant():
    ...     return newton(f, 1.0)
    >>> round(run_secant(), 12)
    1.414213562373

    An array of starting points, solved together. The callbacks receive the
    whole array:

    >>> import numpy as np
    >>> @njit
    ... def run_vec():
    ...     return newton(f, np.array([1.0, 10.0, -3.0]), fp)
    >>> np.round(run_vec(), 12)
    array([ 1.41421356,  1.41421356, -1.41421356])
    >>> @njit
    ... def run_vec_full():
    ...     return newton(f, np.array([1.0, 10.0, -3.0]), fp,
    ...                   full_output=True)
    >>> res = run_vec_full()
    >>> res.converged, res.zero_der
    (array([ True,  True,  True]), array([False, False, False]))
    """
    _args_tuple_star(args)
    if isinstance(x0, np.ndarray) and x0.ndim > 0:
        ares = _array_newton(f, x0, fprime, args, tol, maxiter, fprime2)
        if full_output:
            return ares
        return ares.root
    if isinstance(x0, np.ndarray):
        x0 = x0[()]
    res = _newton_core(f, x0, fprime, args, tol, maxiter, fprime2, x1, rtol, disp)
    if full_output:
        return res.root, res
    return res.root


@overload(newton, prefer_literal=True)
def _newton_ovl(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
                fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True):
    bad = _args_refusal_star(args)
    if bad is not None:
        if _fo_or_false(full_output):
            def impl(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
                     fprime2=None, x1=None, rtol=0.0, full_output=False,
                     disp=True):
                if _always():
                    raise TypeError(bad)
                return 0.0, _dead_root('newton')
        else:
            def impl(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
                     fprime2=None, x1=None, rtol=0.0, full_output=False,
                     disp=True):
                if _always():
                    raise TypeError(bad)
                return 0.0
        return impl
    # An array `x0` runs a different algorithm and returns a third shape.
    # `ndim` is a property of the numba TYPE, so which arm runs is settled
    # while the call compiles; the SIZE is not, which is why a length-1
    # array takes this arm and scipy's `np.size(x0) > 1` sends it to the
    # scalar one.
    if isinstance(x0, types.Array) and x0.ndim > 0:
        if _lit_fo('newton', full_output):
            def impl(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
                     fprime2=None, x1=None, rtol=0.0, full_output=False,
                     disp=True):
                return _array_newton(f, x0, fprime, args, tol, maxiter,
                                     fprime2)
        else:
            def impl(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
                     fprime2=None, x1=None, rtol=0.0, full_output=False,
                     disp=True):
                return _array_newton(f, x0, fprime, args, tol, maxiter,
                                     fprime2).root
        return impl
    if isinstance(x0, types.Array):
        # A 0-d array is scipy's scalar route, and `[()]` is scipy's own
        # spelling for taking the value out of it.
        if _lit_fo('newton', full_output):
            def impl(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
                     fprime2=None, x1=None, rtol=0.0, full_output=False,
                     disp=True):
                res = _newton_core(f, x0[()], fprime, args, tol, maxiter,
                                   fprime2, x1, rtol, disp)
                return res.root, res
        else:
            def impl(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
                     fprime2=None, x1=None, rtol=0.0, full_output=False,
                     disp=True):
                return _newton_core(f, x0[()], fprime, args, tol, maxiter,
                                    fprime2, x1, rtol, disp).root
        return impl
    if _lit_fo('newton', full_output):
        def impl(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
                fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True):
            res = _newton_core(f, x0, fprime, args, tol, maxiter, fprime2, x1, rtol, disp)
            return res.root, res
    else:
        def impl(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
                fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True):
            return _newton_core(f, x0, fprime, args, tol, maxiter, fprime2, x1, rtol, disp).root
    return impl


@njit
def _halley(f, x0, fprime, fprime2, tol=1.48e-8, maxiter=50, rtol=0.0,
            validate=True, args=()):
    """Halley's method -- ``scipy.optimize.newton`` with both ``fprime``
    and ``fprime2`` given.

    Cubically convergent when the second derivative is available.

    **Callback style A** for all three callbacks: plain ``@njit``
    functions of one float.

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Function whose root is wanted.
    x0 : float or complex
        Starting guess.
    fprime : @njit function ``fprime(x) -> float``
        First derivative.  Required and positional, matching scipy's
        argument order.
    fprime2 : @njit function ``fprime2(x) -> float``
        Second derivative.  Required -- supplying only ``fprime`` means
        calling :func:`newton` instead.
    tol : float, optional
        Absolute step tolerance.  Default 1.48e-8, scipy's.
    maxiter : int, optional
        Iteration cap.  Default 50.
    rtol : float, optional
        Relative step tolerance.  Default 0.0.
    validate : bool, optional
        ``True`` (default) raises ``RuntimeError`` when the iteration limit is
        reached, and on the undefined-step conditions below. ``False`` returns
        ``converged=False`` instead.

    Returns
    -------
    res : RootResults
        A namedtuple whose fields are reached by attribute, by index or by
        unpacking.

        root : float
            Best estimate.
        iterations : int
            Iterations used.
        function_calls : int
            Evaluations of `f`, plus the derivative evaluations, counted
            on the same counter.
        converged : bool
            True if a tolerance test was met before `maxiter`.
        flag : str
            ``'converged'``, or ``'convergence error'``.
        method : str
            The method that produced the result, by name.

    Raises
    ------
    RuntimeError
        If `maxiter` is reached, or the first derivative is zero so the next
        step is undefined. Both are suppressed by ``validate=False``.

    Notes
    -----
    Keeps scipy's safeguard: the Halley correction is applied only when
    ``|step * f'' / f' / 2| < 1``, and the plain Newton step is taken
    otherwise. `method` reads ``'newton'`` on an exit taken before `fprime2`
    is first evaluated, which is where scipy overwrites the label. Reached
    through :func:`newton` with both derivatives.
    """
    _check_newton_args(tol, maxiter)
    cnt = np.zeros(1, np.int64)
    p0 = x0 * 1.0
    # scipy seeds `method = "newton"` and overwrites it on the line AFTER
    # `fprime2` is first called (`optimize/_zeros_py.py:316,345`), so an exit
    # taken before that reports `'newton'`.
    meth = 'newton'
    for itr in range(maxiter):
        fval = _fc(f, p0, cnt, args)
        if fval == 0.0:
            return RootResults(p0, itr, cnt[0], True, _FLAG_CONV, meth)
        fder = _fc(fprime, p0, cnt, args)
        if fder == 0.0:
            if validate:
                raise RuntimeError(_zeroder_msg(itr + 1, p0))
            _warn_zeroder()
            return RootResults(p0, itr + 1, cnt[0], False, _FLAG_CONVERR,
                               meth)
        step = fval / fder
        fder2 = _fc(fprime2, p0, cnt, args)
        meth = 'halley'
        adj = step * fder2 / fder / 2.0
        if abs(adj) < 1.0:
            step /= (1.0 - adj)
        p = p0 - step
        if _isclose(p, p0, rtol, tol):
            return RootResults(p, itr + 1, cnt[0], True, _FLAG_CONV, meth)
        p0 = p
    if validate:
        raise RuntimeError(_nonconv_val_msg(maxiter, p0))
    return RootResults(p0, maxiter, cnt[0], False, _FLAG_CONVERR, meth)


# ======================================================================
# NEWTON over an array `x0`
# ======================================================================
# scipy sends an `x0` of size greater than 1 to `_array_newton`
# (`optimize/_zeros_py.py`), which is a different algorithm from the scalar
# loop rather than the scalar loop run element by element: it carries a
# per-element convergence mask, stops updating an element whose derivative is
# zero, and reports per element. It reads neither `x1`, `rtol` nor `disp`,
# counts no function calls, and returns a three-array namedtuple.

#: scipy's array-secant seed step, ``np.finfo(float).eps ** 0.33``. Computed
#: rather than transcribed; it prints 6.8284993814695122e-06. The scalar
#: secant seeds with 1e-4 instead, which is why the two routes separate.
_ARRAY_DX = float(np.finfo(np.float64).eps ** 0.33)


@njit
def _ge_zero_arr(p):
    """numpy's ``p >= 0`` elementwise, defined for a complex array too.

    numpy orders a complex array lexicographically, so scipy's array-secant
    seed ``np.where(p >= 0, dx, -dx)`` is defined for a complex `x0`. numba
    refuses ``>=`` between a complex array and a number, so the order is
    written out. On a real array ``p.imag`` is zero and this reduces to
    ``p >= 0``. The scalar twin is `_ge_zero`.
    """
    return (p.real > 0) | ((p.real == 0) & (p.imag >= 0))


@njit
def _fca(f, p, args):
    """``f(p, *args)`` over the whole array. `_fc`'s coercion, no counter.

    scipy's `_array_newton` reports no ``function_calls``, so there is
    nothing to count and the call sites stay expressions.
    """
    if isinstance(args, tuple):
        return f(p, *args)
    return f(p, args)


def _fmt_rms(v):
    """scipy's ``f'RMS of {rms:g} reached'``."""
    return "RMS of %g reached" % (v,)


@njit
def _rms_msg(v):
    with objmode(msg='unicode_type'):
        msg = _fmt_rms(v)
    return msg


def _emit_runtime_warning(msg):
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


@njit
def _warn_runtime(msg):
    """A ``RuntimeWarning`` whose text is built while the call runs.

    `_array_newton`'s three warnings interpolate a count or a value, so
    unlike `_warn_zeroder` the text is not a compile-time constant and the
    string has to travel as an argument.
    """
    with objmode():
        _emit_runtime_warning(msg)


@njit(error_model='numpy')
def _array_newton(f, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
                  fprime2=None):
    """Engine behind `newton` for an array `x0`. scipy's `_array_newton`.

    Private. Statement for statement scipy's loop, with the masked
    subscripts rewritten as whole-array `numpy.where`: numba has no
    ``p[mask] = v``, and the two spellings select the same elements.

    Parameters are `newton`'s, less `x1`, `rtol` and `disp`, which scipy's
    array route does not read.

    Returns
    -------
    res : ArrayNewtonResult
        `root`, `converged` and `zero_der`, each shaped like `x0`.

    Raises
    ------
    ValueError
        If `fprime2` is given without `fprime`; if ``tol <= 0``; or if
        ``maxiter < 1``.
    RuntimeError
        If EVERY element failed to converge.
    """
    _check_newton_args(tol, maxiter)
    # `x0 * 1.0` is scipy's own cast and it COPIES, so the caller's array is
    # never written through. It promotes an integer start to float64 and
    # keeps a complex one complex.
    p = x0 * 1.0
    p1 = p
    failures = np.ones(p.shape, np.bool_)
    nz_der = np.ones(p.shape, np.bool_)
    if fprime is None:
        if fprime2 is not None:
            raise ValueError(
                "newton: fprime2 was given without fprime; Halley's method "
                "needs both derivatives")
        dx = _ARRAY_DX
        p1 = p * (1.0 + dx) + np.where(_ge_zero_arr(p), dx, -dx)
        q0 = _fca(f, p, args)
        q1 = _fca(f, p1, args)
        active = np.ones(p.shape, np.bool_)
        for _ in range(maxiter):
            nz_der = (q1 != q0)
            if not nz_der.any():
                p = (p1 + p) / 2.0
                break
            # scipy divides only at the surviving positions,
            # `(q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]`. Masking the
            # DENOMINATOR keeps a zero divisor out of the division and leaves
            # every surviving value the quotient scipy computes.
            den = np.where(nz_der, q1 - q0, 1.0)
            dp = q1 * (p1 - p) / den
            newp = np.where(nz_der, p1 - dp, p)
            active_zero_der = (~nz_der) & active
            newp = np.where(active_zero_der, (p1 + p) / 2.0, newp)
            active = active & nz_der
            failures = np.where(nz_der, np.abs(dp) >= tol, failures)
            p = newp
            if not (failures & nz_der).any():
                break
            p1, p = p, p1
            q0 = q1
            q1 = _fca(f, p1, args)
    else:
        for _ in range(maxiter):
            fval = _fca(f, p, args)
            if not (fval != 0).any():
                failures = (fval != 0)
                break
            fder = _fca(fprime, p, args)
            nz_der = (fder != 0)
            if not nz_der.any():
                break
            den = np.where(nz_der, fder, 1.0)
            dp = fval / den
            if fprime2 is not None:
                fder2 = _fca(fprime2, p, args)
                dp = dp / (1.0 - 0.5 * dp * fder2 / den)
            p = np.where(nz_der, p - dp, p)
            failures = np.where(nz_der, np.abs(dp) >= tol, failures)
            if not (failures & nz_der).any():
                break
    zero_der = (~nz_der) & failures
    if zero_der.any():
        if fprime is None:
            nonzero_dp = (p1 != p)
            rms_at = zero_der & nonzero_dp
            if rms_at.any():
                d = np.where(rms_at, p1 - p, 0.0)
                _warn_runtime(_rms_msg(np.sqrt(np.sum(d * d))))
        elif zero_der.all():
            _warn_runtime('all derivatives were zero')
        else:
            _warn_runtime('some derivatives were zero')
    elif failures.any():
        if failures.all():
            raise RuntimeError('all failed to converge after '
                               + str(maxiter) + ' iterations')
        _warn_runtime('some failed to converge after '
                      + str(maxiter) + ' iterations')
    return ArrayNewtonResult(p, ~failures, zero_der)


# ======================================================================
# FIXED POINT (Steffensen / Aitken del^2), scalar
# ======================================================================
@njit
def _fixed_point_core(f, x0, args=(), xtol=1e-8, maxiter=500, validate=True,
                      use_accel=True):
    """Engine behind `fixed_point`, returning the full `FixedPointResults`.

    Private. The public `fixed_point` returns scipy's bare value; the result
    object always exists here.
    """
    cnt = np.zeros(1, np.int64)
    # scipy runs `_asarray_validated(x0, as_inexact=True)`, whose
    # `check_finite` default refuses a non-finite start
    # (`optimize/_minpack_py.py:1186`).
    if not np.isfinite(x0):
        raise ValueError("array must not contain infs or NaNs")
    p0 = float(x0)
    p = p0
    # scipy's `_fixed_point_helper` runs ONE loop and switches the
    # acceleration on a `use_accel` bool (`optimize/_minpack_py.py`), so
    # `method='iteration'` is the same loop with `p = p1` and no Aitken
    # extrapolation. The name reported is scipy's method name either way.
    for i in range(maxiter):
        p1 = _fc(f, p0, cnt, args)
        if use_accel:
            p2 = _fc(f, p1, cnt, args)
            d = p2 - 2.0 * p1 + p0
            if d != 0.0:
                p = p0 - (p1 - p0) ** 2 / d
            else:
                p = p2
        else:
            p = p1
        if p0 != 0.0:
            relerr = (p - p0) / p0
        else:
            relerr = p
        if abs(relerr) < xtol:
            return FixedPointResults(p, i + 1, cnt[0], True, _FLAG_CONV,
                                     'del2' if use_accel else 'iteration')
        p0 = p
    if validate:
        raise RuntimeError(_nonconv_value_msg(maxiter, p))
    return FixedPointResults(p, maxiter, cnt[0], False, _FLAG_CONVERR,
                             'del2' if use_accel else 'iteration')


# ======================================================================
# SCALAR MINIMIZERS
# ======================================================================
class BracketError(RuntimeError):
    """Bracket-search failure, ``scipy.optimize._optimize.BracketError``.

    A `RuntimeError` subclass. scipy does not publish the name, so
    ``except RuntimeError`` is what catches it on both sides.
    """


_BRACKET_MSG = ("The algorithm terminated without finding a valid bracket. "
                "Consider trying different initial points.")


def fixed_point(f, x0, args=(), xtol=1e-8, maxiter=500, method='del2',
                validate=True, full_output=False):
    """Scalar fixed point of ``f``, a point where ``f(x) == x``.

    Uses Steffensen's method with Aitken's delta-squared acceleration under
    the default ``method='del2'``.

    **Callback style A**: ``f`` is a plain ``@njit`` ``f(x) -> float``.

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        The iteration map.  Note this is ``f(x) = x``, not ``f(x) = 0``
        -- for a root use :func:`brentq` or :func:`newton`.
    x0 : float
        Starting point.  Scalar only: one return type per function.
    args : tuple, optional
        Extra arguments for `f`, unpacked into every call as
        ``f(x, *args)``.  Must be a tuple; anything else raises
        ``TypeError``.  Default ``()``.
    xtol : float, optional
        Relative convergence tolerance on ``(p - p0) / p0``, falling
        back to the absolute value when ``p0 == 0``.  Default 1e-8.
    maxiter : int, optional
        Iteration cap.  Default 500.
    method : {'del2', 'iteration'}, optional
        ``'del2'`` (default) applies Aitken's del-squared acceleration;
        ``'iteration'`` runs the plain map.  Any other value raises
        `KeyError`.  Inside ``@njit`` it must be a compile-time constant.
    validate : bool, optional
        ``True`` (default) raises ``RuntimeError`` when the iteration limit is
        reached. ``False`` returns ``converged=False`` instead.  See `Notes`.
    full_output : bool, optional
        ``False`` (default) returns the fixed point alone.  ``True`` returns
        ``(x, FixedPointResults)``.  Inside ``@njit`` it must be a
        compile-time constant.  See `Notes`.

    Returns
    -------
    x : float
        The fixed point, when ``full_output`` is False.
    (x, res) : tuple of (float, FixedPointResults)
        When ``full_output`` is True.

    Raises
    ------
    ValueError
        If `x0` is not finite.
    TypeError
        If `args` is not a tuple.
    KeyError
        If `method` is neither ``'del2'`` nor ``'iteration'``.
    RuntimeError
        If `maxiter` is reached, unless ``validate=False``.

    See Also
    --------
    scipy.optimize.fixed_point : The scipy routine this mirrors.
    scijit.optimize.brentq : For ``f(x) == 0`` rather than ``f(x) == x``.

    Notes
    -----
    scipy's `fixed_point` returns the fixed point itself and publishes no
    result object and no `full_output`. The bare call here returns what
    scipy's returns. `full_output` is ADDITIVE: it exposes the iteration
    counts and the convergence flag that this package computes anyway, under
    a keyword no scipy-shaped call passes.

    `full_output` selects the RETURN SHAPE, and a compiled function has one
    return type per signature, so inside ``@njit`` the flag has to be readable
    when the call compiles. A literal, an omitted default and a module-level
    constant all are; a variable is not, and raises `TypingError` naming the
    constraint. From Python a runtime value is fine.

    `validate` has no counterpart in scipy's signature. scipy raises on
    non-convergence unconditionally, so the default reproduces scipy and
    ``validate=False`` is the additive escape, for a sweep where an exception
    would end a run over thousands of points.

    `x0` is scalar only. scipy also accepts an array, which is not expressible
    here: a numba function has one return type.

    The bare call returns a Python ``float``. scipy returns a 0-d
    ``ndarray`` under ``method='del2'`` and a ``numpy.float64`` under
    ``method='iteration'``, and neither boxes out of compiled code.

    `iterations` and `function_calls` have no scipy counterpart to agree or
    disagree with: scipy's ``fixed_point`` reports no counter of any kind.
    Under ``method='del2'`` they are related, ``function_calls ==
    2 * iterations``, because the accelerated map evaluates `f` twice per
    iteration; under ``method='iteration'`` they are equal.

    ``maxiter=0`` raises ``RuntimeError`` here. scipy raises
    ``UnboundLocalError`` on the same input, because it formats the iterate
    into the failure message and the loop never bound it.

    Pure ``@njit``, safe to call from a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fixed_point.html

    Examples
    --------
    The fixed point of cosine, where ``cos(x) == x``:

    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import fixed_point
    >>> @njit
    ... def g(x):
    ...     return np.cos(x)
    >>> @njit
    ... def run():
    ...     return fixed_point(g, 1.0)
    >>> round(run(), 10)
    0.7390851332
    >>> @njit
    ... def run_full():
    ...     return fixed_point(g, 1.0, full_output=True)
    >>> x, res = run_full()
    >>> round(x, 10), res.converged
    (0.7390851332, True)
    """
    _args_tuple_star(args)
    res = _fixed_point_core(f, x0, args, xtol, maxiter, validate,
                            _fp_use_accel(method))
    if full_output:
        return res.x, res
    return res.x


@overload(fixed_point, prefer_literal=True)
def _fixed_point_ovl(f, x0, args=(), xtol=1e-8, maxiter=500, method='del2',
                     validate=True, full_output=False):
    bad = _args_refusal_star(args)
    if bad is not None:
        if _fo_or_false(full_output):
            def impl(f, x0, args=(), xtol=1e-8, maxiter=500, method='del2',
                     validate=True, full_output=False):
                if _always():
                    raise TypeError(bad)
                return 0.0, _dead_fp('del2')
        else:
            def impl(f, x0, args=(), xtol=1e-8, maxiter=500, method='del2',
                     validate=True, full_output=False):
                if _always():
                    raise TypeError(bad)
                return 0.0
        return impl
    # `method` selects a loop body, so it is resolved here and baked in.
    accel = _fp_use_accel_ty(method)
    if _lit_fo('fixed_point', full_output):
        def impl(f, x0, args=(), xtol=1e-8, maxiter=500, method='del2',
                 validate=True, full_output=False):
            res = _fixed_point_core(f, x0, args, xtol, maxiter, validate,
                                    accel)
            return res.x, res
    else:
        def impl(f, x0, args=(), xtol=1e-8, maxiter=500, method='del2',
                 validate=True, full_output=False):
            return _fixed_point_core(f, x0, args, xtol, maxiter, validate,
                                     accel).x
    return impl


@njit
def _bracket(f, xa0, xb0, cnt, args, recover=False, grow_limit=110.0,
             maxiter=1000):
    """Downhill bracket search (scipy.optimize.bracket).  Returns
    (xa, xb, xc, fa, fb, fc, funcalls, ok) with fb <= fa, fb <= fc.

    `recover` decides what an INVALID bracket does, which is the one place
    scipy's three front ends disagree with each other. ``brent`` and
    ``golden`` are meant to raise; ``minimize_scalar`` is meant to return
    ``success=False`` with the partial state
    (`optimize/_optimize.py:3117-3131`). scipy carries that state on the
    exception object as `e.data` and intercepts it in
    `_recover_from_bracket_error`; a numba exception cannot carry a payload,
    so the flag travels down instead of the data travelling up.

    `grow_limit` and `maxiter` are scipy's two `bracket` parameters. The
    public `bracket` passes them through; `brent`, `golden` and
    `minimize_scalar` reach scipy's `Brent`/`golden`, which call `bracket`
    without them, so the defaults here are the values those three see.
    """
    _gold = 1.618034
    _vs = 1e-21
    xa = float(xa0)
    xb = float(xb0)
    fa = _fc(f, xa, cnt, args)
    fb = _fc(f, xb, cnt, args)
    if fa < fb:
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = xb + _gold * (xb - xa)
    fc = _fc(f, xc, cnt, args)
    funcalls = 3
    it = 0
    while fc < fb:
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if abs(val) < _vs:
            denom = 2.0 * _vs
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        if it > maxiter:
            # scipy raises a plain RuntimeError here
            # (`optimize/_optimize.py:3053-3055`).
            if recover:
                return xa, xb, xc, fa, fb, fc, funcalls, False
            raise RuntimeError(
                "No valid bracket was found before the iteration limit was "
                "reached. Consider trying different initial points or "
                "increasing `maxiter`.")
        it += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = _fc(f, w, cnt, args)
            funcalls += 1
            if fw < fc:
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif fw > fb:
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw = _fc(f, w, cnt, args)
            funcalls += 1
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = _fc(f, w, cnt, args)
            funcalls += 1
        elif (w - wlim) * (xc - w) > 0.0:
            fw = _fc(f, w, cnt, args)
            funcalls += 1
            if fw < fc:
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = _fc(f, w, cnt, args)
                funcalls += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = _fc(f, w, cnt, args)
            funcalls += 1
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw

    cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    cond2 = (xa < xb < xc) or (xc < xb < xa)
    cond3 = math.isfinite(xa) and math.isfinite(xb) and math.isfinite(xc)
    if not (cond1 and cond2 and cond3):
        if recover:
            return xa, xb, xc, fa, fb, fc, funcalls, False
        # scipy raises `BracketError`, a `RuntimeError` SUBCLASS
        # (`optimize/_optimize.py:3100-3113`), so `except RuntimeError`
        # around a scipy call has to catch this too.
        raise BracketError(_BRACKET_MSG)
    return xa, xb, xc, fa, fb, fc, funcalls, True


@njit
def _bracket_failed(xa, xb, xc, fa, fb, fc, funcalls):
    """scipy's `_recover_from_bracket_error` result, as our namedtuple.

    `x` is the best of the three bracket points, or NaN if any of the six
    values is NaN (`optimize/_optimize.py:3138-3147`).
    """
    if (math.isnan(xa) or math.isnan(xb) or math.isnan(xc)
            or math.isnan(fa) or math.isnan(fb) or math.isnan(fc)):
        return MinimizeScalarResult(np.nan, _BRACKET_MSG, funcalls, 0,
                                    False, np.nan)
    x, fv = xa, fa
    if fb < fv:
        x, fv = xb, fb
    if fc < fv:
        x, fv = xc, fc
    return MinimizeScalarResult(fv, _BRACKET_MSG, funcalls, 0, False, x)


def bracket(func, xa=0.0, xb=1.0, args=(), grow_limit=110.0, maxiter=1000):
    """Bracket a minimum of ``func``.

    Searches downhill from two initial points and returns three points
    that bracket a minimum, with the objective value at each.

    **Callback style A**: ``func`` is a plain ``@njit`` function taking one
    float and returning one float::

        @njit
        def f(x):
            return 10 * x ** 2 + 3 * x + 5

        xa, xb, xc, fa, fb, fc, funcalls = bracket(f, 0.1, 1.0)

    Parameters
    ----------
    func : @njit function ``func(x) -> float``
        Objective to bracket.
    xa, xb : float, optional
        Initial points, 0.0 and 1.0 by default.  They set the direction of
        the search and need not contain a minimum.
    args : tuple, optional
        Extra arguments for `func`, unpacked into every call as
        ``func(x, *args)``.  Must be a tuple; anything else raises
        ``TypeError``.  Default ``()``.
    grow_limit : float, optional
        Cap on how far one step may move the bracket, as a multiple of the
        current interval ``xc - xb``.  Default 110.0.
    maxiter : int, optional
        Iteration cap on the search.  Default 1000.

    Returns
    -------
    xa, xb, xc : float
        The three bracket points, ordered ``xa < xb < xc`` or
        ``xc < xb < xa``.
    fa, fb, fc : float
        ``func`` at those three points.
    funcalls : int
        Evaluations of `func` made.

    Raises
    ------
    RuntimeError
        If no valid bracket is found.  The subclass on that exit is
        `scijit.optimize._scalar.BracketError`; if the
        iteration limit is reached first the class is `RuntimeError`
        itself.
    TypeError
        If `args` is not a tuple.

    See Also
    --------
    scipy.optimize.bracket : The scipy routine this mirrors.
    scijit.optimize.brent : Minimises from a bracket found this way.
    scijit.optimize.golden : The same search behind a golden-section fit.

    Notes
    -----
    A valid bracket is three strictly ordered finite points with
    ``fb <= fa`` and ``fb <= fc``, one of the two strict.  The three
    returned points satisfy that, so a minimum lies inside them.

    `BracketError` is a `RuntimeError` subclass, so ``except RuntimeError``
    catches both exits and catches scipy's too.  scipy publishes the class
    only as the private ``scipy.optimize._optimize.BracketError``, so this
    package leaves it unexported as well.

    scipy attaches the seven values reached at the failure to the
    `BracketError` as ``e.data``.  A numba exception carries no payload, so
    the values are unavailable here.

    The six point and value returns are Python ``float``.  scipy returns
    ``numpy.float64``, and ``numpy.int64`` for `xa` and `xb` when both
    initial points are integers, which follows from its
    ``np.asarray([xa, xb])``.  Neither type boxes out of compiled code.

    Pure ``@njit``, no state and no callback slot, so it is safe to call
    from a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.bracket.html

    Examples
    --------
    Both initial points sit to the right of the minimum, so the third is
    found to the left:

    >>> from numba import njit
    >>> from scijit.optimize import bracket
    >>> @njit
    ... def f(x):
    ...     return 10 * x ** 2 + 3 * x + 5
    >>> @njit
    ... def run():
    ...     return bracket(f, 0.1, 1.0)
    >>> run()
    (1.0, 0.1, -1.3562306, 18.0, 5.4, 19.3249226037636, 3)
    """
    _args_tuple_cat(args)
    xa1, xb1, xc, fa, fb, fc, funcalls, ok = _bracket(
        func, xa, xb, np.zeros(1, np.int64), args, False, grow_limit,
        maxiter)
    return xa1, xb1, xc, fa, fb, fc, funcalls


@overload(bracket)
def _bracket_ovl(func, xa=0.0, xb=1.0, args=(), grow_limit=110.0,
                 maxiter=1000):
    bad = _args_refusal_cat(args)
    if bad is not None:
        def impl(func, xa=0.0, xb=1.0, args=(), grow_limit=110.0,
                 maxiter=1000):
            if _always():
                raise TypeError(bad)
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        return impl

    def impl(func, xa=0.0, xb=1.0, args=(), grow_limit=110.0, maxiter=1000):
        xa1, xb1, xc, fa, fb, fc, funcalls, ok = _bracket(
            func, xa, xb, np.zeros(1, np.int64), args, False, grow_limit,
            maxiter)
        return xa1, xb1, xc, fa, fb, fc, funcalls
    return impl


#: scipy's `brack` validation messages, `optimize/_optimize.py`
#: `Brent.get_bracket_info`, verbatim.
_BRACK_ORDER_MSG = ("Bracketing values (xa, xb, xc) do not fulfill this "
                    "requirement: (xa < xb) and (xb < xc)")
_BRACK_FVAL_MSG = ("Bracketing values (xa, xb, xc) do not fulfill this "
                   "requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))")
_BRACK_LEN_MSG = "Bracketing interval must be length 2 or 3 sequence."


@njit
def _brack3(f, xa, xb, xc, cnt, args):
    """scipy's 3-point `brack` path: swap, validate order, validate values.

    `Brent.get_bracket_info` swaps so ``xa < xc`` can be assumed, then makes
    two checks with two distinct messages, and counts exactly 3 evaluations.
    """
    if xa > xc:
        xc, xa = xa, xc
    if not ((xa < xb) and (xb < xc)):
        raise ValueError(_BRACK_ORDER_MSG)
    fa = _fc(f, xa, cnt, args)
    fb = _fc(f, xb, cnt, args)
    fc = _fc(f, xc, cnt, args)
    if not ((fb < fa) and (fb < fc)):
        raise ValueError(_BRACK_FVAL_MSG)
    return xa, xb, xc, fa, fb, fc


@njit
def _brent_min(f, xa0, xb0, tol, maxiter, cnt, msg_ok, args,
               recover=False, xc0=0.0, use3=False):
    """Brent's parabolic-interpolation minimizer (scipy Brent)."""
    # scipy checks this in `_minimize_scalar_brent`
    # (`optimize/_optimize.py:2728-2730`), so it covers `brent` and
    # `minimize_scalar(method='brent')` and NOT `golden`, which accepts a
    # negative tolerance on both sides. Measured on scipy 1.18, 2026-08-05:
    # `brent(f, tol=-1e-8)` raises, `golden(f, tol=-1e-8)` returns 1.4999999925.
    if tol < 0.0:
        raise ValueError(_brent_tol_msg(tol))
    if use3:
        xa, xb, xc, fa, fb, fc = _brack3(f, xa0, xb0, xc0, cnt, args)
    else:
        xa, xb, xc, fa, fb, fc, funcalls, ok = _bracket(f, xa0, xb0, cnt,
                                                        args, recover)
        if not ok:
            return _bracket_failed(xa, xb, xc, fa, fb, fc, cnt[0])
    _mintol = 1.0e-11
    _cg = 0.3819660
    x = xb
    w = xb
    v = xb
    fw = fb
    fv = fb
    fx = fb
    if xa < xc:
        a = xa
        b = xc
    else:
        a = xc
        b = xa
    deltax = 0.0
    rat = 0.0
    it = 0
    while it < maxiter:
        tol1 = tol * abs(x) + _mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        if abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break
        if abs(deltax) <= tol1:
            if x >= xmid:
                deltax = a - x
            else:
                deltax = b - x
            rat = _cg * deltax
        else:
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                p = -p
            tmp2 = abs(tmp2)
            dx_temp = deltax
            deltax = rat
            if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x))
                    and (abs(p) < abs(0.5 * tmp2 * dx_temp))):
                rat = p / tmp2
                u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    if xmid - x >= 0.0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if x >= xmid:
                    deltax = a - x
                else:
                    deltax = b - x
                rat = _cg * deltax
        if abs(rat) < tol1:
            if rat >= 0.0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu = _fc(f, u, cnt, args)
        funcalls += 1
        if fu > fx:
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        it += 1
    ok = it < maxiter and not (np.isnan(x) or np.isnan(fx))
    if ok:
        msg = msg_ok
    elif np.isnan(x) or np.isnan(fx):
        msg = _MS_MSG_NAN
    else:
        msg = _MS_MSG_MAXIT
    return MinimizeScalarResult(fx, msg, cnt[0], it, ok, x)


@njit
def _golden_min(f, xa0, xb0, tol, maxiter, cnt, msg_ok, args,
                recover=False, xc0=0.0, use3=False):
    """Golden-section minimizer (scipy golden)."""
    if use3:
        xa, xb, xc, fa, fb, fc = _brack3(f, xa0, xb0, xc0, cnt, args)
    else:
        xa, xb, xc, fa, fb, fc, funcalls, ok = _bracket(f, xa0, xb0, cnt,
                                                        args, recover)
        if not ok:
            return _bracket_failed(xa, xb, xc, fa, fb, fc, cnt[0])
    _gR = 0.61803399
    _gC = 1.0 - _gR
    x3 = xc
    x0 = xa
    if abs(xc - xb) > abs(xb - xa):
        x1 = xb
        x2 = xb + _gC * (xc - xb)
    else:
        x2 = xb
        x1 = xb - _gC * (xb - xa)
    f1 = _fc(f, x1, cnt, args)
    f2 = _fc(f, x2, cnt, args)
    nit = 0
    for i in range(maxiter):
        if abs(x3 - x0) <= tol * (abs(x1) + abs(x2)):
            break
        if f2 < f1:
            x0 = x1
            x1 = x2
            x2 = _gR * x1 + _gC * x3
            f1 = f2
            f2 = _fc(f, x2, cnt, args)
        else:
            x3 = x2
            x2 = x1
            x1 = _gR * x2 + _gC * x0
            f2 = f1
            f1 = _fc(f, x1, cnt, args)
        nit += 1
    if f1 < f2:
        xmin = x1
        fval = f1
    else:
        xmin = x2
        fval = f2
    ok = nit < maxiter and not (np.isnan(xmin) or np.isnan(fval))
    if ok:
        msg = msg_ok
    elif np.isnan(xmin) or np.isnan(fval):
        msg = _MS_MSG_NAN
    else:
        msg = _MS_MSG_MAXIT
    return MinimizeScalarResult(fval, msg, cnt[0], nit, ok, xmin)


@njit
def _bounded_min(f, x1, x2, xatol, maxfun, cnt, args, disp):
    """Bounded scalar minimizer on [x1, x2] (scipy fminbound).

    `disp` gates the failure-exit warning. scipy reaches `_endprint` only
    under ``disp > 0`` (`optimize/_optimize.py:2417`), and its
    `minimize_scalar` passes nothing, so the engine's own ``disp=0`` default
    applies there and no warning is emitted on that route.
    """
    # scipy checks finiteness before the ordering
    # (`optimize/_optimize.py:2314-2323`). Without it `x1 = -inf` makes
    # `fulc = a + golden_mean*(b - a)` equal -inf and the whole run nan.
    if not (np.isfinite(x1) and np.isfinite(x2)):
        raise ValueError("Optimization bounds must be finite scalars.")
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")
    sqrt_eps = math.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - math.sqrt(5.0))
    a = float(x1)
    b = float(x2)
    fulc = a + golden_mean * (b - a)
    nfc = fulc
    xf = fulc
    rat = 0.0
    e = 0.0
    x = xf
    fx = _fc(f, x, cnt, args)
    num = 1
    fu = np.inf
    ffulc = fx
    fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    flag = 0
    while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        do_golden = True
        if abs(e) > tol1:
            do_golden = False
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat
            if (abs(p) < abs(0.5 * q * r)) and (p > q * (a - xf)) \
                    and (p < q * (b - xf)):
                rat = p / q
                x = xf + rat
                if (x - a) < tol2 or (b - x) < tol2:
                    # scipy writes `np.sign(xm - xf) + ((xm - xf) == 0)`
                    # (`optimize/_optimize.py:2369`), which is nan for a nan
                    # argument and +1 at zero.
                    si = np.sign(xm - xf) + (
                        1.0 if (xm - xf) == 0.0 else 0.0)
                    rat = tol1 * si
            else:
                do_golden = True
        if do_golden:
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e
        # scipy writes `np.sign(rat) + (rat == 0)` (`:2379`).
        si = np.sign(rat) + (1.0 if rat == 0.0 else 0.0)
        x = xf + si * max(abs(rat), tol1)
        fu = _fc(f, x, cnt, args)
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc = nfc
            ffulc = fnfc
            nfc = xf
            fnfc = fx
            xf = x
            fx = fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc = nfc
                ffulc = fnfc
                nfc = x
                fnfc = fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc = x
                ffulc = fu
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxfun:
            flag = 1
            break
    # scipy tests `fu` too (`optimize/_optimize.py:2422-2423`). `fu` holds the
    # last trial value, which is nan whenever the final evaluation produced
    # one and it was rejected as worse than `fx`; without the term the run
    # reports status 0, success True and 'Solution found.' on a nan result.
    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2
    if flag == 0:
        msg = _MS_MSG_BOUNDED_OK
    elif flag == 1:
        msg = _MS_MSG_BOUNDED_MAX
    else:
        msg = _MS_MSG_NAN
    if flag != 0 and disp > 0:
        _warn_fminbound(flag)
    # scipy reports nit == nfev here: `fminbound`'s own counter is the
    # evaluation count, and its result sets both keys from it.
    return MinimizeScalarResultBounded(fx, msg, cnt[0], num, flag,
                                       flag == 0, xf)


@njit
def _fminbound_core(f, x1, x2, args=(), xtol=1e-5, maxfun=500, disp=1):
    """Bounded-Brent engine behind `fminbound`, returning the full result.

    Private. The public `fminbound` returns scipy's shape, a bare float or
    ``(x, fun, status, nfev)`` under `full_output`; the whole
    `MinimizeScalarResultBounded` exists here, so `minimize_scalar` can reach
    every field without going through the public entry's return-shape choice.
    """
    return _bounded_min(f, x1, x2, xtol, maxfun, np.zeros(1, np.int64), args,
                        disp)


# ======================================================================
# DISPATCHERS
# ======================================================================


def fminbound(f, x1, x2, args=(), xtol=1e-5, maxfun=500, full_output=False,
              disp=1):
    """Bounded scalar minimizer on a fixed interval.

    Unlike :func:`brent` and :func:`golden`, ``[x1, x2]`` is a hard
    box: the search never leaves it, so the answer is the constrained
    minimum, which may sit on an endpoint.

    **Callback style A**: ``f`` is a plain ``@njit`` ``f(x) -> float``.

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Function to minimize.
    x1, x2 : float
        Interval bounds, required (there is no default box).
        ``x1 > x2`` raises ``ValueError``.
    args : tuple, optional
        Extra arguments for `f`, unpacked into every call as
        ``f(x, *args)``.  Must be a tuple; anything else raises
        ``TypeError``.  Default ``()``.
    xtol : float, optional
        Absolute tolerance on the minimizer position.  Default 1e-5,
        looser than the other minimizers here.
    maxfun : int, optional
        Cap on **function evaluations**, not iterations.  Default 500.
    full_output : bool, optional
        ``False`` (default) returns `x` alone.  ``True`` returns
        ``(x, fun, status, nfev)``, with `status` THIRD.  Inside ``@njit``
        it must be a compile-time constant; see `Notes`.
    disp : int, optional
        Print level.  ``0`` and ``1`` print nothing on a converged solve,
        ``2`` prints the termination block.  Default ``1``.  ``3`` raises
        `NotImplementedError`; see `Notes`.

    Returns
    -------
    x : float
        The minimiser, when ``full_output`` is False.
    (x, fun, status, nfev) : tuple
        When ``full_output`` is True.  `status` is 0 on success and 1 when
        `maxfun` was reached, and it is the THIRD element, not the last.

    Raises
    ------
    ValueError
        If either bound is not finite, or if ``x1 > x2``.
    TypeError
        If `args` is not a tuple.
    NotImplementedError
        If ``disp >= 3``.

    Warns
    -----
    OptimizeWarning
        On either failure exit, when ``disp > 0``: ``"Maximum number of
        function evaluations exceeded --- increase maxfun argument."`` and
        ``"NaN result encountered."``, each wrapped in newlines.

    See Also
    --------
    scipy.optimize.fminbound : The scipy routine this mirrors.
    scijit.optimize.brent : Unbounded, seeded rather than bracketed.
    scijit.optimize.minimize_scalar : ``method='bounded'`` reaches this one.

    Notes
    -----
    `full_output` selects the RETURN SHAPE, and a compiled function has one
    return type per signature, so inside ``@njit`` the flag has to be readable
    when the call compiles. A literal, an omitted default and a module-level
    constant all are; a variable is not, and raises `TypingError` naming the
    constraint. From Python a runtime value is fine.

    `disp` is an int print level, not a bool. ``0`` and ``1`` print nothing
    on a converged solve, and signal a `maxfun` limit through a warning
    rather than a print. ``2`` prints the termination block. ``3`` would add
    a per-iteration Func-count table whose `Procedure` column names the step
    type (golden or parabolic) each iteration took; the bounded-Brent core
    does not report that, so ``3`` raises `NotImplementedError` rather than
    printing a table missing a column.

    Pure ``@njit``, safe to call from a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fminbound.html

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import fminbound
    >>> @njit
    ... def q(x):
    ...     return (x - 1.5) ** 2 + 0.5
    >>> @njit
    ... def run():
    ...     return fminbound(q, 0.0, 3.0)
    >>> round(run(), 8)
    1.5
    >>> @njit
    ... def run_full():
    ...     return fminbound(q, 0.0, 3.0, (), 1e-5, 500, True, 0)
    >>> run_full()
    (1.5, 0.5, 0, 6)
    """
    _args_tuple_star(args)
    if disp >= 3:
        raise NotImplementedError(_FB_DISP3_MSG)
    res = _fminbound_core(f, x1, x2, args, xtol, maxfun, disp)
    if disp >= 2 and res.status == 0:
        _fb_print_ok(xtol)
    if full_output:
        return res.x, res.fun, res.status, res.nfev
    return res.x


@overload(fminbound, prefer_literal=True)
def _fminbound_ovl(f, x1, x2, args=(), xtol=1e-5, maxfun=500,
                   full_output=False, disp=1):
    bad = _args_refusal_star(args)
    if bad is not None:
        if _fo_or_false(full_output):
            def impl(f, x1, x2, args=(), xtol=1e-5, maxfun=500,
                     full_output=False, disp=1):
                if _always():
                    raise TypeError(bad)
                return 0.0, 0.0, 0, 0
        else:
            def impl(f, x1, x2, args=(), xtol=1e-5, maxfun=500,
                     full_output=False, disp=1):
                if _always():
                    raise TypeError(bad)
                return 0.0
        return impl
    fo = _lit_fo('fminbound', full_output)
    if fo:
        def impl(f, x1, x2, args=(), xtol=1e-5, maxfun=500,
                 full_output=False, disp=1):
            if disp >= 3:
                raise NotImplementedError(_FB_DISP3_MSG)
            res = _fminbound_core(f, x1, x2, args, xtol, maxfun, disp)
            if disp >= 2 and res.status == 0:
                _fb_print_ok(xtol)
            return res.x, res.fun, res.status, res.nfev
    else:
        def impl(f, x1, x2, args=(), xtol=1e-5, maxfun=500,
                 full_output=False, disp=1):
            if disp >= 3:
                raise NotImplementedError(_FB_DISP3_MSG)
            res = _fminbound_core(f, x1, x2, args, xtol, maxfun, disp)
            if disp >= 2 and res.status == 0:
                _fb_print_ok(xtol)
            return res.x
    return impl


@njit(error_model='numpy')
def _approx_der(f, x, cnt, args):
    """Forward difference standing in for a missing `fprime`.

    ``root_scalar(method='newton')`` with no `fprime` builds one from
    ``approx_derivative(..., method='2-point')`` (``_root_scalar.py:307-317``)
    and hands it to the same Newton loop. The step is scipy's,
    ``sqrt(eps) * sign(x) * max(1, |x|)``, taken as ``dx = (x + h) - x`` so
    the divisor is the step the machine actually took.

    The DERIVATIVE is one evaluation on `cnt`, which is what scipy's counter
    reports: the two calls to `f` happen inside the derivative callback,
    below the loop that counts. `function_calls` therefore reads the same on
    this route as on the route where `fprime` is supplied.
    """
    cnt[0] += 1
    s = 1.0 if x >= 0.0 else -1.0
    h = _EPSILON * s * max(1.0, abs(x))
    xh = x + h
    dx = xh - x
    if isinstance(args, tuple):
        return (f(xh, *args) - f(x, *args)) / dx
    return (f(xh, args) - f(x, args)) / dx


@njit
def _newton_fd_core(f, x0, args, tol, maxiter, rtol, disp):
    """`_newton_core`'s Newton-Raphson loop over a finite-difference
    derivative, for ``root_scalar(method='newton')`` without `fprime`.

    Private. Statement for statement `_newton_core`'s `fprime` branch, with
    `_approx_der` in place of the derivative callback.
    """
    _check_newton_args(tol, maxiter)
    cnt = np.zeros(1, np.int64)
    p0 = float(x0)
    for itr in range(maxiter):
        fval = _fc(f, p0, cnt, args)
        if fval == 0.0:
            return RootResults(p0, itr, cnt[0], True, _FLAG_CONV, 'newton')
        fder = _approx_der(f, p0, cnt, args)
        if fder == 0.0:
            if disp:
                raise RuntimeError(_zeroder_msg(itr + 1, p0))
            _warn_zeroder()
            return RootResults(p0, itr + 1, cnt[0], False, _FLAG_CONVERR,
                               'newton')
        p = p0 - fval / fder
        if _isclose(p, p0, rtol, tol):
            return RootResults(p, itr + 1, cnt[0], True, _FLAG_CONV, 'newton')
        p0 = p
    if disp:
        raise RuntimeError(_nonconv_val_msg(maxiter, p0))
    return RootResults(p0, maxiter, cnt[0], False, _FLAG_CONVERR, 'newton')


def root_scalar(f, args=(), method=None, bracket=None, fprime=None,
                fprime2=None, x0=None, x1=None, xtol=None, rtol=None,
                maxiter=None, validate=False):
    """Scalar root-finder dispatcher.

    One entry point over the eight scalar root-finders, selected by name or
    by what was supplied.

    **Callback style A**: ``f``, `fprime` and `fprime2` are plain ``@njit``
    functions ``f(x) -> float``.

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Function whose root is wanted.
    args : tuple, optional
        Extra arguments for `f`, unpacked into every call as
        ``f(x, *args)``.  A non-tuple is taken as a single extra argument.
        Default ``()``.  Reaches `fprime` and `fprime2`
        too.
    method : str or None, optional
        ``'brentq'``, ``'brenth'``, ``'bisect'``, ``'ridder'``,
        ``'toms748'``, ``'secant'``, ``'newton'`` or ``'halley'``, matched
        case-insensitively.  ``None`` (default) selects from what was
        supplied: `bracket` gives ``'brentq'``; `x0` with `fprime` and
        `fprime2` gives ``'halley'``, with `fprime` alone ``'newton'``,
        with `x1` ``'secant'``, and on its own ``'newton'``.  Any other
        name raises ``ValueError``.
    bracket : sequence of two floats or None, optional
        Interval bracketing a root, for ``'brentq'``, ``'brenth'``,
        ``'bisect'``, ``'ridder'`` and ``'toms748'``.  ``f(a, *args)`` and
        ``f(b, *args)`` must not have the same sign.  A longer sequence
        uses its first two entries.
    fprime : @njit function ``fprime(x) -> float`` or None, optional
        First derivative, for ``'newton'`` and ``'halley'``.  ``'newton'``
        without one differences `f` forward instead.
    fprime2 : @njit function ``fprime2(x) -> float`` or None, optional
        Second derivative, required by ``'halley'``.
    x0 : float or None, optional
        Starting guess, for ``'secant'``, ``'newton'`` and ``'halley'``.
    x1 : float or None, optional
        Second starting guess, for ``'secant'``.  Without it the secant
        loop invents one from `x0`.
    xtol : float or None, optional
        Absolute tolerance.  ``None`` (default) gives each method its own:
        2e-12 for the five bracketing methods, and 1.48e-8 for
        ``'secant'``, ``'newton'`` and ``'halley'``, where it is that
        method's ``tol``.
    rtol : float or None, optional
        Relative tolerance.  ``None`` (default) gives ``4 * eps`` for the
        bracketing methods and 0.0 for the other three.
    maxiter : int or None, optional
        Iteration cap.  ``None`` (default) gives 100 for the bracketing
        methods and 50 for the other three.
    validate : bool, optional
        ``False`` (default) returns ``converged=False`` when the chosen
        method reaches `maxiter`. ``True`` raises ``RuntimeError`` instead,
        which is what the individual solvers in this module do.  See `Notes`.

    Returns
    -------
    res : RootResults
        A namedtuple whose fields are reached by attribute, by index or by
        unpacking.

        root : float
            Root estimate.
        iterations : int
            Iterations used.
        function_calls : int
            Evaluations of `f`. Counts every one, including the ones a
            solver discards. For :func:`newton` with derivatives it also
            counts the derivative evaluations.
        converged : bool
            True if a tolerance test was met before `maxiter`.
        flag : str
            ``'converged'``, or ``'convergence error'``.
        method : str
            The method that produced the result, by name.

    Raises
    ------
    ValueError
        If `method` is not one of the eight names; if neither `bracket` nor
        `x0` is given and `method` is ``None``; if a bracketing method gets
        no `bracket`, or an open method no `x0`, or ``'halley'`` no
        `fprime` or `fprime2`; or if the chosen method rejects its bracket.
    RuntimeError
        If the chosen method reaches `maxiter` and ``validate=True``. The
        default ``validate=False`` returns ``converged=False`` instead.

    See Also
    --------
    scipy.optimize.root_scalar : The scipy routine this mirrors.
    scijit.optimize.brentq : ``method='brentq'``, chosen for a `bracket`.
    scijit.optimize.newton : The three open methods, called directly.
    scijit.optimize.root : Systems of equations rather than one scalar.

    Notes
    -----
    `validate` has no counterpart in scipy's ``root_scalar`` signature.
    scipy overrides the chosen method's ``disp`` to ``False``
    (``optimize/_root_scalar.py:249``), so its ``root_scalar`` never raises
    on non-convergence whatever the method would do on its own. The default
    ``validate=False`` here reproduces that; ``validate=True`` is the
    additive setting, and it is what the individual solvers default to.

    A `RootResults` is returned always, as scipy's ``root_scalar`` does.
    scipy has no ``full_output`` on this routine. Its object is a ``dict``
    subclass that neither unpacks nor indexes by integer, where this
    namedtuple does both; the field names and their order are the same.

    `bracket` is a tuple or an array inside ``@njit``. A python list is not
    typeable as an argument to compiled code.

    scipy also accepts an `options` dict, which is how ``toms748``'s `k` is
    reachable through that front end. It is not available here; :func:`toms748`
    takes `k` directly.

    A NaN function value raises ``ValueError``, as it does in the underlying
    solvers. scipy catches that same exception inside ``root_scalar`` and
    returns a `RootResults` whose `iterations` is ``nan`` and whose `flag`
    is the exception text.

    Pure ``@njit``, safe to call from a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import root_scalar
    >>> @njit
    ... def f(x):
    ...     return x * x - 2.0
    >>> @njit
    ... def run():
    ...     return root_scalar(f, bracket=(0.0, 2.0))
    >>> res = run()
    >>> round(res.root, 12), res.converged, res.method
    (1.414213562373, True, 'brentq')

    A derivative selects Newton-Raphson, and both select Halley:

    >>> @njit
    ... def fp(x):
    ...     return 2.0 * x
    >>> @njit
    ... def run2():
    ...     return root_scalar(f, x0=1.0, fprime=fp)
    >>> res2 = run2()
    >>> round(res2.root, 12), res2.iterations, res2.method
    (1.414213562373, 5, 'newton')
    """
    return _root_scalar_core(f, args, method, _rs_seq(bracket),
                             bracket is not None, fprime, fprime2, x0, x1,
                             xtol, rtol, maxiter, validate)


def _rs_seq(bracket):
    """`bracket` as a tuple, or ``None`` when it is not a sequence at all.

    scipy tests ``isinstance(bracket, list | tuple | np.ndarray)`` and
    raises ``Bracket needed for <method>`` for anything else
    (``_root_scalar.py:280-282``), while the method CHOICE above it tests
    only ``bracket is not None``. The two questions are asked separately
    here for the same reason.
    """
    if bracket is None or isinstance(bracket, tuple):
        return bracket
    if isinstance(bracket, (list, np.ndarray)):
        return tuple(bracket)
    return None


@njit
def _root_scalar_core(f, args, method, bracket, has_bracket, fprime, fprime2,
                      x0, x1, xtol, rtol, maxiter, validate):
    """`root_scalar`'s dispatch, shared by both entry points.

    The Python body calls this and the ``@overload``'s ``impl`` calls it, so
    the two entry points run ONE source. Every ``is None`` test below is on
    an argument, so numba prunes the branch when the call compiles and the
    arm that is left is the arm scipy reaches at run time.

    `has_bracket` is ``bracket is not None`` before `_rs_seq` mapped a
    non-sequence to ``None``.
    """
    # `method=None` picks the "best" method from what was supplied,
    # `optimize/_root_scalar.py:253-268`. `name` keeps the caller's spelling,
    # which is what scipy's messages interpolate; `meth` is lower-cased,
    # which is what its `Unknown solver` message interpolates.
    if method is None:
        if has_bracket:
            meth = 'brentq'
        elif x0 is not None:
            if fprime is not None:
                if fprime2 is not None:
                    meth = 'halley'
                else:
                    meth = 'newton'
            elif x1 is not None:
                meth = 'secant'
            else:
                meth = 'newton'
        else:
            raise ValueError('Unable to select a solver as neither bracket '
                             'nor starting point provided.')
        name = meth
    else:
        name = method
        meth = method.lower()
    brk = (meth == 'brentq' or meth == 'brenth' or meth == 'bisect'
           or meth == 'ridder' or meth == 'toms748')
    # scipy passes a tolerance through only when the caller set one, "respect
    # solver-specific default tolerances" (`_root_scalar.py:238-243`), so each
    # method keeps its own: 2e-12 / 4*eps / 100 for the bracketing five, and
    # 1.48e-8 / 0.0 / 50 for `secant`, `newton` and `halley`.
    if xtol is None:
        xt = _XTOL if brk else 1.48e-8
    else:
        xt = xtol
    if rtol is None:
        rt = _RTOL if brk else 0.0
    else:
        rt = rtol
    if maxiter is None:
        mi = 100 if brk else 50
    else:
        mi = maxiter
    if brk:
        if bracket is None:
            raise ValueError('Bracket needed for ' + name)
        elif len(bracket) < 2:
            # scipy's `a, b = bracket[:2]` on a one-element sequence, whose
            # text is CPython's own.
            raise ValueError('not enough values to unpack (expected 2, got '
                             + str(len(bracket)) + ')')
        else:
            # scipy's `a, b = bracket[:2]`, so a longer sequence uses its
            # first two entries.
            a = np.float64(bracket[0])
            b = np.float64(bracket[1])
            if meth == 'brentq':
                return _brentq_core(f, a, b, args, xt, rt, mi, validate)
            elif meth == 'brenth':
                return _brenth_core(f, a, b, args, xt, rt, mi, validate)
            elif meth == 'bisect':
                return _bisect_core(f, a, b, args, xt, rt, mi, validate)
            elif meth == 'ridder':
                return _ridder_core(f, a, b, args, xt, rt, mi, validate)
            return _toms748_core(f, a, b, args, 1, xt, rt, mi, validate)
    elif meth == 'secant':
        if x0 is None:
            raise ValueError('x0 must not be None for ' + name)
        elif x1 is None:
            return _secant(f, np.float64(x0), _NAN, xt, mi, rt, validate,
                           False, args)
        else:
            return _secant(f, np.float64(x0), np.float64(x1), xt, mi, rt,
                           validate, True, args)
    elif meth == 'newton':
        if x0 is None:
            raise ValueError('x0 must not be None for ' + name)
        elif fprime is None:
            return _newton_fd_core(f, np.float64(x0), args, xt, mi, rt,
                                   validate)
        else:
            return _newton_core(f, np.float64(x0), fprime, args, xt, mi,
                                None, None, rt, validate)
    elif meth == 'halley':
        if x0 is None:
            raise ValueError('x0 must not be None for ' + name)
        elif fprime is None:
            raise ValueError('fprime must be specified for ' + name)
        elif fprime2 is None:
            raise ValueError('fprime2 must be specified for ' + name)
        else:
            return _halley(f, np.float64(x0), fprime, fprime2, xt, mi, rt,
                           validate, args)
    raise ValueError('Unknown solver ' + meth)


@overload(root_scalar, prefer_literal=True)
def _root_scalar_ovl(f, args=(), method=None, bracket=None, fprime=None,
                     fprime2=None, x0=None, x1=None, xtol=None, rtol=None,
                     maxiter=None, validate=False):
    """@njit implementation of `root_scalar`. See `_root_scalar_core`.

    The compiled entry reaches the core directly: a tuple and an array are
    both sequences, so `_rs_seq` has nothing left to do and `has_bracket` is
    ``bracket is not None``. A `bracket` that is neither is refused when the
    call compiles rather than when it runs.
    """
    if not (_is_none_ty(bracket)
            or isinstance(bracket, (types.BaseTuple, types.Array))):
        raise TypingError(
            "root_scalar: bracket must be None, a tuple of two floats or an "
            "array. A python list is not typeable as an argument to compiled "
            "code; use a tuple.")
    if _is_none_ty(bracket):
        def impl(f, args=(), method=None, bracket=None, fprime=None,
                 fprime2=None, x0=None, x1=None, xtol=None, rtol=None,
                 maxiter=None, validate=False):
            return _root_scalar_core(f, args, method, bracket, False, fprime,
                                     fprime2, x0, x1, xtol, rtol, maxiter,
                                     validate)
    else:
        def impl(f, args=(), method=None, bracket=None, fprime=None,
                 fprime2=None, x0=None, x1=None, xtol=None, rtol=None,
                 maxiter=None, validate=False):
            return _root_scalar_core(f, args, method, bracket, True, fprime,
                                     fprime2, x0, x1, xtol, rtol, maxiter,
                                     validate)
    return impl


def minimize_scalar(fun, bracket=None, bounds=None, args=(), method=None,
                    tol=None, maxiter=None):
    """Scalar minimizer dispatcher.

    One entry point over the three scalar minimizers, selected by name or by
    whether `bounds` was given.

    **Callback style A**: ``fun`` is a plain ``@njit`` ``f(x) -> float``.

    Parameters
    ----------
    fun : @njit function ``f(x) -> float``
        Function to minimize.
    bracket : sequence of two or three floats or None, optional
        For ``'brent'`` and ``'golden'``.  Two entries are *seed points* for
        a downhill bracket search and the minimum may land outside them.
        Three are a bracket used directly, and must satisfy
        ``xa < xb < xc`` and ``f(xb) < f(xa)``, ``f(xb) < f(xc)``.  ``None``
        (default) seeds the search from 0.0 and 1.0.
    bounds : sequence of two floats or None, optional
        Hard bounds for ``'bounded'``, which searches inside them.
        Mandatory for that method and refused by the other two.
    args : tuple, optional
        Extra arguments for `fun`, unpacked into every call as
        ``f(x, *args)``.  A non-tuple is taken as a single extra argument.
        Default ``()``.
    method : str or None, optional
        ``'brent'``, ``'golden'`` or ``'bounded'`` (:func:`fminbound`),
        matched case-insensitively.  ``None`` (default) is ``'brent'``, or
        ``'bounded'`` when `bounds` is given.  Anything else raises
        ``ValueError``.  Inside ``@njit`` it must be a literal string
        written at the call site, because it selects the return type.  See
        Notes.
    tol : float or None, optional
        Tolerance on the minimizer position, relative for ``'brent'`` and
        ``'golden'`` and absolute for ``'bounded'``.  ``None`` (default)
        gives each method its own default: 1.48e-8 for ``'brent'``,
        ``sqrt(eps)`` = 1.4901161193847656e-08 for ``'golden'``, 1e-5 for
        ``'bounded'``.  A value overrides all three, and supplying one for
        ``'bounded'`` warns.
    maxiter : int or None, optional
        ``None`` (default) gives each method its own default: 500 for
        ``'brent'``, 5000 for ``'golden'``, 500 for ``'bounded'``.  It counts
        iterations for ``'brent'`` and ``'golden'`` and **function
        evaluations** for ``'bounded'``, which passes it as ``maxfun``.

    Returns
    -------
    res : MinimizeScalarResult or MinimizeScalarResultBounded
        A namedtuple whose fields are reached by attribute, by index or by
        unpacking.  ``'brent'`` and ``'golden'`` give the six fields below;
        ``'bounded'`` gives those six and ``status``.

        fun : float
            ``f`` at the minimum.
        message : str
            Text for the outcome. See Notes.
        nfev : int
            Evaluations of `f`, including the ones spent bracketing for
            ``'brent'`` and ``'golden'``.  ``'bounded'`` does no bracketing.
        nit : int
            Iterations (``'brent'``, ``'golden'``) or function
            evaluations (``'bounded'``).
        status : int
            ``'bounded'`` only.  ``0`` on success, ``1`` for `maxfun`
            reached and ``2`` for NaN.
        success : bool
            False if the cap ran out or the result is NaN.
        x : float
            Position of the minimum.

    Raises
    ------
    ValueError
        Unknown ``method``; `bounds` given to ``'brent'`` or ``'golden'``;
        `bounds` missing for ``'bounded'``, or holding other than two
        elements, or not finite, or lower above upper; a `bracket` whose
        length is neither 2 nor 3, or whose three points are not ordered or
        do not bracket a minimum; or a negative `tol` on ``'brent'``, which
        ``'golden'`` accepts.
    TypeError
        If `args` is not a tuple, or `bracket` or `bounds` is not a
        sequence.
    numba.TypingError
        From inside ``@njit``: `method` held in a variable, `method` that is
        not a string, an unknown `method`, or a `bounds` that is missing or
        is a tuple of other than two elements.  See Notes.

    Warns
    -----
    RuntimeWarning
        If `tol` is supplied to ``method='bounded'``.

    See Also
    --------
    scipy.optimize.minimize_scalar : The scipy routine this mirrors.
    scijit.optimize.brent : ``method='brent'``, the default.
    scijit.optimize.golden : ``method='golden'``.
    scijit.optimize.fminbound : ``method='bounded'``.
    scijit.optimize.minimize : Several variables rather than one.

    Notes
    -----
    Three methods are named: ``'brent'``, ``'golden'`` and ``'bounded'``,
    matched case-insensitively. scipy accepts a CALLABLE in the `method`
    argument as a fourth dispatch, which is not expressible inside ``@njit``.

    Inside ``@njit``, `method` must be a literal string written at the call
    site. It selects the return type, since ``'bounded'`` carries ``status``
    and the other two do not, and a compiled function has one return type
    per signature. A `method` held in a variable raises
    `numba.TypingError` naming the constraint. From Python any string works.
    Two further refusals move to the same place inside ``@njit``, because
    both are decidable while the call compiles: an unknown `method`, and a
    `bounds` that is missing or is a tuple of other than two elements. A
    `bounds` ARRAY of the wrong length is not decidable there and raises
    ``ValueError`` while the code runs.

    A failed bracket search returns ``success=False`` with the best of the
    three bracket points, which is what scipy's ``minimize_scalar`` does.
    :func:`brent` and :func:`golden` raise on the same input, which is also
    what scipy does; the two front ends deliberately differ.

    `bracket` and `bounds` are tuples or arrays inside ``@njit``. A python
    list is not typeable as an argument to compiled code.

    `maxiter` is the one member of scipy's `options` dict that is reachable.
    scipy's `disp` travels in the same dict and has no counterpart here.

    `x` and `fun` are Python floats. scipy returns ``numpy.float64`` in
    both fields.

    `message` is scipy's, including its ``(using xtol = ...)`` line, whenever
    `tol` is known while the call compiles. It is known when the argument is
    omitted, which is every default call, and it is always known from Python.
    A `tol` written explicitly at an ``@njit`` call site is NOT: numba makes
    literals of ints, bools and strings but not of floats, so the value
    arrives typed ``float64`` with the number gone, and the message is
    scipy's first two lines without the third.

    Pure ``@njit``, safe to call from a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import minimize_scalar
    >>> @njit
    ... def q(x):
    ...     return (x - 1.5) ** 2 + 0.5
    >>> @njit
    ... def run():
    ...     return minimize_scalar(q, bracket=(0.0, 3.0))
    >>> res = run()
    >>> round(res.x, 8), round(res.fun, 8), res.success
    (1.5, 0.5, True)
    >>> @njit
    ... def run_bounded():
    ...     return minimize_scalar(q, bounds=(0.0, 3.0))
    >>> res = run_bounded()
    >>> round(res.x, 8), res.nfev, res.status
    (1.5, 6, 0)
    """
    # Two checks scipy makes BEFORE the search, in this order
    # (`_minimize.py:988-999`, `_optimize.py:2327`, `:2690`). They are here
    # rather than in the shared core because a length is a typing-time
    # question in compiled code, where the same two refusals arrive as
    # `TypingError`, and this entry point answers in Python.
    meth = (('brent' if bounds is None else 'bounded') if method is None
            else method.lower())
    # scipy's two messages interpolate the caller's own spelling.
    name = meth if method is None else method
    if bounds is not None and meth in ('brent', 'golden'):
        raise ValueError("Use of `bounds` is incompatible with 'method="
                         + method + "'.")
    brk = _ms_seq(bracket)
    if (meth in ('brent', 'golden') and brk is not None
            and len(brk) not in (2, 3)):
        raise ValueError(_BRACK_LEN_MSG)
    bnd = _ms_seq(bounds)
    if meth == 'bounded':
        # A tuple's length is a compile-time constant, so a core that tested
        # it would have no live return for a wrong one and would refuse with
        # a typing error rather than scipy's `ValueError`. scipy emits the
        # tolerance warning before either refusal (`_minimize.py:1002-1010`).
        if bnd is None or len(bnd) != 2:
            if tol is not None:
                _warn_bounded_tol()
            if bnd is None:
                raise ValueError(_MS_BOUNDS_MANDATORY_MSG)
            raise ValueError(_MS_BOUNDS_TWO_MSG)
        return _ms_core_bounded(fun, bnd, args, tol, maxiter)
    if meth == 'brent' or meth == 'golden':
        gold = meth == 'golden'
        return _ms_core_bg(
            fun, brk, False, name, args, gold, tol, maxiter,
            _ms_or_plain((_EPSILON if gold else 1.48e-8)
                         if tol is None else tol))
    raise ValueError("Unknown solver " + name)


def _ms_seq(seq):
    """`bracket` or `bounds` as a tuple, with scipy's ``len()`` refusal.

    scipy reaches both through ``len(brack)`` and ``len(bounds)``, so a
    value that is not a sequence raises ``TypeError`` from CPython itself
    (``optimize/_optimize.py:2690``, ``:2327``).
    """
    if seq is None or isinstance(seq, tuple):
        return seq
    if isinstance(seq, (list, np.ndarray)):
        return tuple(seq)
    raise TypeError("object of type '%s' has no len()" % type(seq).__name__)


@njit
def _ms_core_bg(f, bracket, has_bounds, name, args, golden, tol, maxiter,
                msg):
    """`minimize_scalar`'s ``'brent'`` and ``'golden'`` branch.

    The two share a result type, the six fields scipy carries for both, so
    one core serves both and `golden` selects the engine at run time. The
    success message is built in PYTHON, at the python entry or in the
    ``@overload`` chooser, because its third line interpolates the tolerance
    and numba cannot render a float.
    """
    if has_bounds:
        raise ValueError("Use of `bounds` is incompatible with 'method="
                         + name + "'.")
    # scipy passes NO tolerance and NO iteration cap through unless the caller
    # sets one (`_minimize.py:1002-1010`), so each engine keeps its own
    # default: `'brent'` xtol 1.48e-8 maxiter 500, `'golden'` xtol
    # sqrt(eps) = 1.4901161193847656e-08 maxiter 5000. `None` reproduces
    # that; a value overrides both.
    if tol is None:
        xt = _EPSILON if golden else 1.48e-8
    else:
        xt = tol
    if maxiter is None:
        mi = 5000 if golden else 500
    else:
        mi = maxiter
    xa0, xb0, xc0, use3 = _brack_args(bracket)
    if golden:
        return _golden_min(f, xa0, xb0, xt, mi, np.zeros(1, np.int64),
                           msg, args, True, xc0, use3)
    return _brent_min(f, xa0, xb0, xt, mi, np.zeros(1, np.int64),
                      msg, args, True, xc0, use3)


@njit
def _ms_core_bounded(f, bounds, args, tol, maxiter):
    """`minimize_scalar`'s ``'bounded'`` branch, the one that carries
    ``status``.

    Its defaults are the engine's own, xatol 1e-5 and maxfun 500, which is
    what scipy leaves in place when the caller sets neither.

    `bounds` arrives as a sequence. A missing one and a TUPLE of the wrong
    length are both compile-time facts, so both callers refuse them before
    this runs: a body whose only exit is a `raise` types as returning
    ``none``, and the caller then fails on the return type with
    ``Unknown attribute 'x' of type none`` instead of showing the message.
    An ARRAY's length is not a compile-time fact, which is what the test
    below is for.
    """
    if tol is None:
        xt = 1e-5
    else:
        xt = tol
        _warn_bounded_tol()
    if maxiter is None:
        mi = 500
    else:
        mi = maxiter
    if len(bounds) != 2:
        raise ValueError(_MS_BOUNDS_TWO_MSG)
    return _bounded_min(f, np.float64(bounds[0]), np.float64(bounds[1]),
                        xt, mi, np.zeros(1, np.int64), args, 0)


@overload(minimize_scalar, prefer_literal=True)
def _minimize_scalar_ovl(fun, bracket=None, bounds=None, args=(),
                         method=None, tol=None, maxiter=None):
    """@njit implementation of `minimize_scalar`.

    `method` is resolved HERE, because it selects the result type:
    ``'bounded'`` carries ``status`` and the other two do not. A `method`
    held in a variable is refused here, in the chooser, so the refusal is a
    compile-time `numba.TypingError`. It must NOT be an impl that raises:
    such an arm types as returning ``none`` and the caller fails on the
    return type first, so the message is never seen.

    The chooser also resolves `tol`'s per-method default, because the success
    message carries the tolerance and ``None`` cannot be rendered. A `bracket`
    or `bounds` that is not a sequence is refused when the call compiles
    rather than when it runs.
    """
    for arg, nm in ((bracket, 'bracket'), (bounds, 'bounds')):
        if not (_is_none_ty(arg)
                or isinstance(arg, (types.BaseTuple, types.Array))):
            raise TypingError(
                "minimize_scalar: %s must be None, a tuple of floats or an "
                "array. A python list is not typeable as an argument to "
                "compiled code; use a tuple." % nm)
    has_bounds = not _is_none_ty(bounds)
    mt = method.value if isinstance(method, types.Omitted) else method
    if isinstance(mt, types.StringLiteral):
        NAME = mt.literal_value
    elif isinstance(mt, str):
        NAME = mt
    elif mt is None or isinstance(mt, types.NoneType):
        NAME = 'bounded' if has_bounds else 'brent'
    elif isinstance(mt, types.UnicodeType):
        raise TypingError(_MS_METHOD_RUNTIME_MSG)
    else:
        raise TypingError(_MS_METHOD_TYPE_MSG)
    meth = NAME.lower()
    if meth not in ('brent', 'golden', 'bounded'):
        raise TypingError(_MS_UNKNOWN_MSG + NAME)

    if meth == 'bounded':
        # Both refusals are compile-time facts here and neither can live in
        # the core: an arm whose only exit is a `raise` types as returning
        # `none`, and the caller then fails on the return type instead of
        # showing the message. An ARRAY's length is not known here, and that
        # one stays a run-time `ValueError` in the core.
        if _is_none_ty(bounds):
            raise TypingError(_MS_BOUNDS_MANDATORY_MSG)
        if isinstance(bounds, types.BaseTuple) and bounds.count != 2:
            raise TypingError(_MS_BOUNDS_TWO_MSG)

        def impl(fun, bracket=None, bounds=None, args=(), method=None,
                 tol=None, maxiter=None):
            return _ms_core_bounded(fun, bounds, args, tol, maxiter)
        return impl

    t = tol
    if isinstance(t, types.Omitted):
        t = t.value
    GOLDEN = meth == 'golden'
    MSG = _ms_or_plain((_EPSILON if GOLDEN else 1.48e-8)
                       if t is None else t)

    def impl(fun, bracket=None, bounds=None, args=(), method=None, tol=None,
             maxiter=None):
        return _ms_core_bg(fun, bracket, has_bounds, NAME, args, GOLDEN, tol,
                           maxiter, MSG)
    return impl

def brent(f, args=(), brack=None, tol=1.48e-8, full_output=False,
          maxiter=500):
    """Brent's parabolic-interpolation minimizer.

    **Callback style A**: ``f`` is a plain ``@njit`` function taking one
    float and returning one float::

        @njit
        def f(x):
            return (x - 1.5) ** 2 + 0.5

        x = brent(f)
        x, fun, nit, nfev = brent(f, full_output=True)

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Objective to minimise.
    args : tuple, optional
        Extra arguments for `f`, unpacked into every call as ``f(x, *args)``.
        Must be a tuple; anything else raises ``TypeError``.  Default ``()``.
    brack : None or tuple, optional
        Bracket for the search.  ``None`` (default), a 2-sequence
        ``(xa, xb)`` giving the downhill search its starting points, or a
        3-sequence ``(xa, xb, xc)`` used directly.  See `Notes`.
    tol : float, optional
        Relative tolerance on the minimiser.  Default 1.48e-8.
    full_output : bool, optional
        ``False`` (default) returns `x` alone.  ``True`` returns
        (x, fun, nit, nfev).  Inside ``@njit`` it must be a compile-time constant; see
        `Notes`.
    maxiter : int, optional
        Iteration cap.  Default 500.

    Returns
    -------
    x : float
        The minimiser, when ``full_output`` is False.
    (x, fun, nit, nfev) : tuple
        When ``full_output`` is True.

    Raises
    ------
    ValueError
        If `brack` is a 3-sequence whose points are not ordered, or whose
        middle value is not below both ends; if `brack` has a length other
        than 2 or 3; or if `tol` is negative.
    TypeError
        If `args` is not a tuple.
    RuntimeError
        If the bracket search finds no valid bracket, or reaches its own
        iteration limit.
    numba.core.errors.TypingError
        From inside ``@njit``, if `full_output` is a runtime variable.

    See Also
    --------
    scipy.optimize.brent : The scipy routine this mirrors.
    scijit.optimize.minimize_scalar : The same engines behind a ``method``
        argument.
    scijit.optimize.fminbound : Minimises on a closed interval instead.

    Notes
    -----
    `full_output` selects the RETURN SHAPE, and a compiled function has one
    return type per signature, so inside ``@njit`` the flag has to be readable
    when the call compiles. A literal, an omitted default and a module-level
    constant all are; a variable is not, and raises `TypingError` naming the
    constraint. From Python a runtime value is fine.

    `brack` takes three spellings. ``None`` runs the downhill bracket search
    from ``(0.0, 1.0)``. A 2-sequence runs it from those two points. A
    3-sequence is used directly after two checks, ``(xa < xb) and (xb < xc)``
    after swapping so ``xa < xc``, and ``(f(xb) < f(xa)) and (f(xb) <
    f(xc))``; each raises `ValueError`. The 3-sequence path counts exactly 3
    evaluations. A sequence of any other length raises `ValueError`.

    Inside ``@njit`` `brack` must be a tuple or ``None``; a list is a
    heterogeneous-container question numba answers only for a literal, and a
    tuple is the spelling that types from both entry points.

    Pure ``@njit``, no state and no callback slot, so it is safe to call from
    a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brent.html

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import brent
    >>> @njit
    ... def q(x):
    ...     return (x - 1.5) ** 2 + 0.5
    >>> @njit
    ... def run():
    ...     return brent(q, full_output=True)
    >>> run()
    (1.5, 0.5, 5, 8)
    """
    _args_tuple_cat(args)
    xa0, xb0, xc0, use3 = _brack_args(brack)
    res = _brent_min(f, xa0, xb0, tol, maxiter, np.zeros(1, np.int64),
                     _ms_or_plain(tol), args, False, xc0, use3)
    if full_output:
        return res.x, res.fun, res.nit, res.nfev
    return res.x


@overload(brent, prefer_literal=True)
def _brent_ovl(f, args=(), brack=None, tol=1.48e-8, full_output=False,
               maxiter=500):
    bad = _args_refusal_cat(args)
    if bad is not None:
        if _fo_or_false(full_output):
            def impl(f, args=(), brack=None, tol=1.48e-8, full_output=False,
                     maxiter=500):
                if _always():
                    raise TypeError(bad)
                return 0.0, 0.0, 0, 0
        else:
            def impl(f, args=(), brack=None, tol=1.48e-8, full_output=False,
                     maxiter=500):
                if _always():
                    raise TypeError(bad)
                return 0.0
        return impl
    # `_ms_or_plain` runs in PYTHON while the call compiles, so scipy's
    # tolerance line is baked in as a constant. An explicitly-passed `tol` is
    # not knowable here and then the message is scipy's first two lines.
    msg = _ms_msg_ok(tol)
    if msg is None:
        msg = _MS_MSG_OK
    if _lit_fo('brent', full_output):
        def impl(f, args=(), brack=None, tol=1.48e-8, full_output=False,
                 maxiter=500):
            xa0, xb0, xc0, use3 = _brack_args(brack)
            res = _brent_min(f, xa0, xb0, tol, maxiter, np.zeros(1, np.int64),
                     msg, args, False, xc0, use3)
            return res.x, res.fun, res.nit, res.nfev
    else:
        def impl(f, args=(), brack=None, tol=1.48e-8, full_output=False,
                 maxiter=500):
            xa0, xb0, xc0, use3 = _brack_args(brack)
            res = _brent_min(f, xa0, xb0, tol, maxiter, np.zeros(1, np.int64),
                     msg, args, False, xc0, use3)
            return res.x
    return impl


def golden(f, args=(), brack=None, tol=_EPSILON, full_output=False,
           maxiter=5000):
    """Golden-section minimizer.

    **Callback style A**: ``f`` is a plain ``@njit`` function taking one
    float and returning one float::

        @njit
        def f(x):
            return (x - 1.5) ** 2 + 0.5

        x = golden(f)
        x, fun, nfev = golden(f, full_output=True)

    Parameters
    ----------
    f : @njit function ``f(x) -> float``
        Objective to minimise.
    args : tuple, optional
        Extra arguments for `f`, unpacked into every call as ``f(x, *args)``.
        Must be a tuple; anything else raises ``TypeError``.  Default ``()``.
    brack : None or tuple, optional
        Bracket for the search.  ``None`` (default), a 2-sequence
        ``(xa, xb)`` giving the downhill search its starting points, or a
        3-sequence ``(xa, xb, xc)`` used directly.  See `Notes`.
    tol : float, optional
        Relative tolerance on the minimiser.  Default
        ``sqrt(eps)`` = 1.4901161193847656e-08.
    full_output : bool, optional
        ``False`` (default) returns `x` alone.  ``True`` returns
        (x, fun, nfev).  Inside ``@njit`` it must be a compile-time constant; see
        `Notes`.
    maxiter : int, optional
        Iteration cap.  Default 5000.

    Returns
    -------
    x : float
        The minimiser, when ``full_output`` is False.
    (x, fun, nfev) : tuple
        When ``full_output`` is True.

    Raises
    ------
    ValueError
        If `brack` is a 3-sequence whose points are not ordered, or whose
        middle value is not below both ends; or if `brack` has a length other
        than 2 or 3.
    TypeError
        If `args` is not a tuple.
    RuntimeError
        If the bracket search finds no valid bracket, or reaches its own
        iteration limit.
    numba.core.errors.TypingError
        From inside ``@njit``, if `full_output` is a runtime variable.

    See Also
    --------
    scipy.optimize.golden : The scipy routine this mirrors.
    scijit.optimize.minimize_scalar : The same engines behind a ``method``
        argument.
    scijit.optimize.fminbound : Minimises on a closed interval instead.

    Notes
    -----
    `full_output` selects the RETURN SHAPE, and a compiled function has one
    return type per signature, so inside ``@njit`` the flag has to be readable
    when the call compiles. A literal, an omitted default and a module-level
    constant all are; a variable is not, and raises `TypingError` naming the
    constraint. From Python a runtime value is fine.

    `brack` takes three spellings. ``None`` runs the downhill bracket search
    from ``(0.0, 1.0)``. A 2-sequence runs it from those two points. A
    3-sequence is used directly after two checks, ``(xa < xb) and (xb < xc)``
    after swapping so ``xa < xc``, and ``(f(xb) < f(xa)) and (f(xb) <
    f(xc))``; each raises `ValueError`. The 3-sequence path counts exactly 3
    evaluations. A sequence of any other length raises `ValueError`.

    Inside ``@njit`` `brack` must be a tuple or ``None``; a list is a
    heterogeneous-container question numba answers only for a literal, and a
    tuple is the spelling that types from both entry points.

    Pure ``@njit``, no state and no callback slot, so it is safe to call from
    a ``numba.prange`` loop.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.golden.html

    Examples
    --------
    >>> from numba import njit
    >>> from scijit.optimize import golden
    >>> @njit
    ... def q(x):
    ...     return (x - 1.5) ** 2 + 0.5
    >>> @njit
    ... def run():
    ...     return golden(q, full_output=True)
    >>> x, fun, nfev = run()
    >>> round(x, 8), round(fun, 8), nfev
    (1.50000001, 0.5, 43)
    """
    _args_tuple_cat(args)
    xa0, xb0, xc0, use3 = _brack_args(brack)
    res = _golden_min(f, xa0, xb0, tol, maxiter, np.zeros(1, np.int64),
                      _ms_or_plain(tol), args, False, xc0, use3)
    if full_output:
        return res.x, res.fun, res.nfev
    return res.x


@overload(golden, prefer_literal=True)
def _golden_ovl(f, args=(), brack=None, tol=_EPSILON, full_output=False,
                maxiter=5000):
    bad = _args_refusal_cat(args)
    if bad is not None:
        if _fo_or_false(full_output):
            def impl(f, args=(), brack=None, tol=_EPSILON, full_output=False,
                     maxiter=5000):
                if _always():
                    raise TypeError(bad)
                return 0.0, 0.0, 0
        else:
            def impl(f, args=(), brack=None, tol=_EPSILON, full_output=False,
                     maxiter=5000):
                if _always():
                    raise TypeError(bad)
                return 0.0
        return impl
    # `_ms_or_plain` runs in PYTHON while the call compiles, so scipy's
    # tolerance line is baked in as a constant. An explicitly-passed `tol` is
    # not knowable here and then the message is scipy's first two lines.
    msg = _ms_msg_ok(tol)
    if msg is None:
        msg = _MS_MSG_OK
    if _lit_fo('golden', full_output):
        def impl(f, args=(), brack=None, tol=_EPSILON, full_output=False,
                 maxiter=5000):
            xa0, xb0, xc0, use3 = _brack_args(brack)
            res = _golden_min(f, xa0, xb0, tol, maxiter, np.zeros(1, np.int64),
                      msg, args, False, xc0, use3)
            return res.x, res.fun, res.nfev
    else:
        def impl(f, args=(), brack=None, tol=_EPSILON, full_output=False,
                 maxiter=5000):
            xa0, xb0, xc0, use3 = _brack_args(brack)
            res = _golden_min(f, xa0, xb0, tol, maxiter, np.zeros(1, np.int64),
                      msg, args, False, xc0, use3)
            return res.x
    return impl

