"""``scipy.integrate.solve_ivp``: initial value problem for a system of ODEs."""

import warnings
from collections import namedtuple

import inspect

import numpy as np
from numba import njit, objmode, types
from numba.core import types as nbtypes
from numba.core.errors import TypingError
from numba.experimental import jitclass
from numba.extending import overload

from scijitclass import scijitclass, sig, Scalar, ArrayOf

from ._ivp import (METHOD_RK45, METHOD_RK23, METHOD_DOP853, _rk_core,
                   rk_dense_eval, _NO_ARGS, _check_teval, _RTOL_SHAPE_MSG)
from ._odepack import (_adapter_rhs, _prepend_neq, _msg, _run_odeint,
                       _run_lsoda_dense, _lsoda_dense_work,
                       lsoda_dense_eval, _rhs_arity,
                       _py_func_of, _jac_route, _adapter_jac,
                       _jac_arity, _check_jac_shape_full,
                       _check_jac_shape_band)
from .._probe import wrote_ydot as _wrote_ydot_t
from ._events import (scan_events, split_events, truncate_at,
                      KIND_RK, KIND_LSODA)
from .._lib._typing import _is_none, _lit_bool, _lit_str

__all__ = ['solve_ivp', 'OdeResult', 'OdeResultDense', 'OdeSolution',
           'METHOD_LSODA']

#: Additive alias beside the three RK codes in :mod:`scijit.integrate._ivp`.
#: ``method='LSODA'`` is the spelling scipy uses and the one to prefer.
METHOD_LSODA = 3

#: scipy's ``OdeResult`` field names, minus ``sol``.  Returned when
#: ``dense_output=False``, which is the default.
OdeResult = namedtuple(
    'OdeResult',
    ['t', 'y', 't_events', 'y_events', 'nfev', 'njev', 'nlu', 'status',
     'message', 'success'])

#: scipy's ``OdeResult`` field names in full.  Returned when
#: ``dense_output=True``.  ``sol`` is an :class:`OdeSolution`.
OdeResultDense = namedtuple(
    'OdeResultDense',
    ['t', 'y', 'sol', 't_events', 'y_events', 'nfev', 'njev', 'nlu',
     'status', 'message', 'success'])

_NO_EVENTS = np.empty(0, np.float64)

#: Adapters turning a two-argument RK right-hand side into the three-argument
#: form the engine calls.  Keyed by the python function, and the cache OWNS
#: the dispatcher.
_RK3 = {}


#: `args` given to a right-hand side that has nowhere to put them. The RK
#: methods reached this by a route LSODA did not and were dropping the
#: parameters SILENTLY, returning a plausible answer computed without them.
#: scipy raises `TypeError: takes 2 positional arguments but 3 were given`.
_RK_ARGS_MSG = (
    "args= was given but the right-hand side takes only (t, y), so the "
    "parameters have nowhere to go. Write f(t, y, args) and read them as "
    "args[i], or close over them and drop args=")

#: The same mismatch on the LSODA route. `_odeint._ARGS_MSG` covers this
#: case for `odeint`, whose rhs is `f(y, t)`, so it names the arguments in the
#: opposite order and cannot be reused here.
_IVP_ARGS_MSG = _RK_ARGS_MSG

#: `args` NOT given to a right-hand side that declares them. Both engines pass
#: a length-1 zero buffer when `args is None`, so `args[0]` reads 0.0 and the
#: run returns a confident wrong answer with `success=True`: measured, a decay
#: rhs stayed flat at 1.0 instead of decaying to 0.368. scipy refuses the same
#: call with `TypeError: missing 1 required positional argument`.
_MISSING_ARGS_MSG = (
    "the right-hand side takes (t, y, args) but args= was not given, so the "
    "parameters would read as zeros. Pass args=, or write f(t, y)")


#: The same two mismatches on an EVENT function. scipy wraps every event the
#: way it wraps the right-hand side, `lambda t, x, event=event: event(t, x,
#: *args)`, so an event sees `args` whenever the call does.
_EV_ARGS_MSG = (
    "args= was given but the event function takes only (t, y), so the "
    "parameters have nowhere to go. Write g(t, y, args) and read them as "
    "args[i], or close over them and drop args=")

_EV_MISSING_MSG = (
    "the event function takes (t, y, args) but args= was not given, so the "
    "parameters would read as zeros. Pass args=, or write g(t, y)")


def _check_rhs_args(py, has_args, exc):
    """scipy's rhs arity contract, in BOTH directions.

    Measured on scipy 1.18, which raises `TypeError` either way::

        two(t, y)      + args=(1.0,)  TypeError: takes 2 positional
                                      arguments but 3 were given
        three(t, y, a) + no args      TypeError: missing 1 required
                                      positional argument: 'a'
        two(t, y)      + args=()      no error
        three(t, y, a) + args=(1.0,)  no error

    An EMPTY `args` is not a mismatch in either direction, so the caller
    folds it into `has_args`: scipy splats, and splatting nothing passes
    nothing. Callers reaching the engine by raw address are not checked,
    since a `@cfunc` address carries no arity.
    """
    arity = _rhs_arity(py)
    if arity == 2 and has_args:
        raise exc(_IVP_ARGS_MSG)
    if arity == 3 and not has_args:
        raise exc(_MISSING_ARGS_MSG)


def _check_event_args(py, has_args, exc):
    """The rhs arity contract of :func:`_check_rhs_args`, for an event.

    Measured on scipy 1.18: an event declared ``g(t, y, a)`` raises
    ``TypeError`` when ``args`` is absent, and one declared ``g(t, y)``
    raises ``TypeError`` when ``args`` is given, both from the wrapper
    ``_ivp/ivp.py:649`` builds.
    """
    arity = _rhs_arity(py)
    if arity == 2 and has_args:
        raise exc(_EV_ARGS_MSG)
    if arity == 3 and not has_args:
        raise exc(_EV_MISSING_MSG)


#: Adapters turning a two-argument event function into the three-argument
#: form the scan calls.  Keyed by the python function, as `_RK3`.
_EV3 = {}


def _ev3(py):
    """A three-argument ``g(t, y, args)`` returning ``(ng,)``, for any event.

    ``scan_events`` calls ``g(t, y, args)`` unconditionally and indexes the
    result, so a two-argument event is given the third parameter here, at
    TYPING time, and a SCALAR return is presented as a length-1 array.  Same
    shape as :func:`_rk_rhs3`, and for the same reason: threading ``args``
    through beats a closure over a runtime array.

    The scalar case is scipy's own spelling, where each event function
    returns one value.
    """
    hit = _EV3.get(py)
    if hit is not None:
        return hit
    inner = njit(py)
    if _rhs_arity(py) == 3:

        @njit
        def out(t, y, args):
            return np.asarray(inner(t, y, args)).astype(np.float64).ravel()
    else:

        @njit
        def out(t, y, args):
            return np.asarray(inner(t, y)).astype(np.float64).ravel()
    _EV3[py] = out
    return out


def _rk_rhs3(py):
    """A three-argument ``f(t, y, args)`` for a plain ``@njit`` RK rhs.

    ``_rk_core`` calls ``rhs(t, y, args)`` unconditionally, so a
    two-argument right-hand side is wrapped here, at TYPING time, where the
    inner dispatcher is a compile-time constant.

    The parameters cannot instead be captured in a closure: a closure defined
    inside ``@njit`` capturing a runtime array cannot be passed to another
    ``@njit`` function.  Measured, "TypingError: During: Pass
    make_function_op_code_to_jit_function".  Threading ``args`` through the
    engine is what works.
    """
    hit = _RK3.get(py)
    if hit is not None:
        return hit
    n = _rhs_arity(py)
    if n == 3:
        out = njit(py)
    else:
        inner = njit(py)

        @njit
        def out(t, y, args):
            return inner(t, y)
    _RK3[py] = out
    return out

#: Machine epsilon. scipy's `validate_tol` floors `rtol` at `100 * eps`,
#: which is 2.220446049250313e-14.
_EPS_IVP = np.finfo(np.float64).eps

_RADAU_BDF = ("method 'Radau' and 'BDF' are not implemented; use 'LSODA' "
              "for a stiff problem, or 'RK45'/'RK23'/'DOP853'")

#: scipy's wording for an argument the chosen solver does not use,
#: `_ivp/common.py:38-41`. It emits ONE `UserWarning` per call listing every
#: such argument, and integrates as if none had been passed. Measured on
#: `RK45`: the answer is identical with and without, maxdiff 0.000000e+00 at
#: nfev 38 either way.
_NO_EFFECT = "The following arguments have no effect for a chosen solver: %s."


def _no_effect_msg(names):
    """scipy's message for a list of ignored argument names."""
    return _NO_EFFECT % ", ".join("`%s`" % n for n in names)


def _emit_rk_ignored(min_step, mxstep, npoints, tail):
    """One `UserWarning` naming every argument an RK method will not use.

    scipy sees "was it passed", because it captures them in `**options`, so
    it warns even for `jac=None` and `min_step=0.0`. Named parameters cannot
    tell an omitted default from one written out, so the test here is
    "carries a value". Measured divergence, and the only one: scipy warns for
    `solve_ivp(..., 'RK45', jac=None)` and this does not.

    The first three are run-time tests; `tail` carries the ones the compiled
    path settles when the call compiles, comma separated. Both entry points
    reach this one builder, so the two cannot drift on the text or the order.
    """
    out = []
    if mxstep:
        out.append('mxstep')
    if npoints:
        out.append('npoints')
    if min_step:
        out.append('min_step')
    if tail:
        out.extend(tail.split(','))
    if out:
        # scipy's `warn_extraneous` passes `stacklevel=3`, which from here is
        # also the caller's frame: this helper, `solve_ivp`, the caller.
        warnings.warn(_no_effect_msg(out), stacklevel=3)


@njit
def _rk_warn(min_step, mxstep, npoints, tail):
    """:func:`_emit_rk_ignored` from inside `@njit`.

    Entered only when something was actually passed, so a scipy-shaped call
    never takes the GIL for it.
    """
    if min_step or mxstep or npoints or len(tail) > 0:
        with objmode():
            _emit_rk_ignored(min_step, mxstep, npoints, tail)


# ---------------------------------------------------------------------------
# the callable dense-output object
# ---------------------------------------------------------------------------
_SOL_SPEC = [
    ('t', nbtypes.float64[:]),
    ('h', nbtypes.float64[:]),
    ('ys', nbtypes.float64[:, :]),
    ('C', nbtypes.float64[:, ::1]),
    ('method', nbtypes.int64),
]


_HIST_T_MSG = "`t` must be strictly increasing or decreasing."
_HIST_N_MSG = "Numbers of time stamps and steps don't match."


@njit
def _check_hist_t(t):
    """Refuse a step history whose times are not strictly monotonic.

    A history is read by locating the step a time falls in, which needs an
    ordered ``t``.  Out of order, the search lands on the wrong step and the
    evaluation is wrong with no other signal: measured on a two-element swap
    of a ``y'' = -4y`` history, ``sol(0.5)`` read ``[0.99998491,
    -0.01098896]`` where the history it was built from gives ``[0.54038693,
    -1.68337531]``.

    A run over a zero-length span is the one exception, as it is in scipy: a
    two-entry ``t`` whose ends are equal is one step of size zero.

    The test is written as a loop rather than through ``np.diff``, which does
    not type for a non-contiguous array: the field is declared ``float64[:]``,
    so a strided view reaches here.
    """
    n = t.shape[0]
    if n == 2 and t[0] == t[1]:
        return
    up = True
    dn = True
    for i in range(n - 1):
        if not (t[i + 1] > t[i]):
            up = False
        if not (t[i + 1] < t[i]):
            dn = False
    if not (up or dn):
        raise ValueError(_HIST_T_MSG)


@scijitclass([('t', nbtypes.float64[:]),
              ('h', nbtypes.float64[:]),
              ('ys', nbtypes.float64[:, :]),
              ('C', nbtypes.float64[:, :, :]),
              ('method', nbtypes.int64)],
             dispatch=[('ev', sig(ArrayOf(ndim=1))),
                       ('ev_one', sig(Scalar))])
class OdeSolution:
    """Continuous solution over the integrated span, ``res.sol``.

    ``solve_ivp(..., dense_output=True)`` returns one as ``res.sol``, which is
    how most callers meet it.  Building one by hand is for a step history
    obtained some other way.

    ``sol(t)`` evaluates it, from Python and from inside ``@njit``.  A float
    gives ``(n_states,)``; a float64 array of ``k`` times gives
    ``(n_states, k)``.  A Python list raises; wrap it in ``np.array``.

    Parameters
    ----------
    t, h : float64 array, shape (s,)
        Start time and signed step size of each accepted step.
    ys : float64 array, shape (s, n)
        State at the start of each accepted step.
    C : float64 array, shape (s, 7, n)
        Interpolation coefficients per step.
    method : int
        ``METHOD_RK45``, ``METHOD_RK23`` or ``METHOD_DOP853``, the method
        the history was produced with.  A mismatch reads the coefficients
        under the wrong polynomial and returns wrong values silently.

    Raises
    ------
    ValueError
        If ``t`` is not strictly increasing or strictly decreasing, or if
        ``h``, ``ys`` and ``C`` do not all carry one entry per time stamp.
        A two-entry ``t`` whose ends are equal is a zero-length span and is
        accepted.

    See Also
    --------
    scipy.integrate._ivp.common.OdeSolution : The scipy routine this mirrors.

    Notes
    -----
    The constructor takes different arguments from scipy's.  scipy builds one
    as ``OdeSolution(ts, interpolants, alt_segment=False)`` from a list of
    ``DenseOutput`` objects; this one takes a step history as five arrays, and
    no parameter name is shared.

    **A constructor default is Python-only.**  Inside ``@njit`` every argument
    is passed explicitly, so a scipy-shaped ``OdeSolution(ts, interpolants)``,
    which leaves ``alt_segment`` to its default, has no compiled spelling.

    The attribute sets are disjoint.  A scipy instance carries ``ts``,
    ``interpolants``, ``n_segments``, ``t_min``, ``t_max``, ``ascending``,
    ``side`` and ``ts_sorted``.  This one carries ``t``, ``h``, ``ys``, ``C``
    and ``method``.

    ``sol(t)`` takes a float or a 1-D float64 array.  scipy runs
    ``np.asarray(t)`` and accepts any array_like, a list included.  On an
    empty array ``sol(np.array([]))`` returns shape ``(n, 0)`` and scipy
    raises ``ValueError('need at least one array to concatenate')``.

    One scipy class covers every method.  Here
    ``solve_ivp(method='LSODA', dense_output=True)`` returns a different type,
    ``_LsodaSolution`` built from a Nordsieck history, and a complex ``y0``
    returns a third, ``_OdeSolutionC``.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> from scijit.integrate import OdeSolution
    >>> @njit
    ... def rhs(t, y):
    ...     out = np.empty(2)
    ...     out[0] = y[1]
    ...     out[1] = -4.0 * y[0]
    ...     return out
    >>> res = si.solve_ivp(rhs, (0.0, 1.0), np.array([1.0, 0.0]),
    ...                    'RK45', None, True)
    >>> isinstance(res.sol, OdeSolution)
    True
    >>> res.sol(0.5)
    array([ 0.54038693, -1.68337531])

    Attributes
    ----------
    t : float64 array, shape (s,)
        Start time of each accepted step.
    h : float64 array, shape (s,)
        Signed step size of each accepted step.
    ys : float64 array, shape (s, n)
        State at the start of each accepted step.
    C : float64 array, shape (s, 7, n)
        Interpolation coefficients per step.
    method : int
        The method the history was produced with.
    """

    def __init__(self, t, h, ys, C, method):
        """Validate the step history and store it.  Every argument is
        required; see the class docstring."""
        _check_hist_t(t)
        n = t.shape[0]
        if h.shape[0] != n or ys.shape[0] != n or C.shape[0] != n:
            raise ValueError(_HIST_N_MSG)
        self.t = t
        self.h = h
        self.ys = ys
        self.C = C
        self.method = method

    def ev(self, tt):
        """Evaluate at an array of times.

        Parameters
        ----------
        tt : float64 array, shape (k,)
            Times to evaluate at.

        Returns
        -------
        y : float64 array, shape (n_states, k)
            State at each time, component-major.
        """
        return rk_dense_eval(tt, self.t, self.h, self.ys, self.C,
                             self.method).T.copy()

    def ev_one(self, tt):
        """Evaluate at a single time.

        Parameters
        ----------
        tt : float
            Time to evaluate at.

        Returns
        -------
        y : float64 array, shape (n_states,)
            State at ``tt``.
        """
        one = np.empty(1, np.float64)
        one[0] = tt
        return rk_dense_eval(one, self.t, self.h, self.ys, self.C,
                             self.method)[0]


@scijitclass([('t', nbtypes.float64[:]),
              ('h', nbtypes.float64[:]),
              ('ys', nbtypes.complex128[:, :]),
              ('C', nbtypes.complex128[:, :, :]),
              ('method', nbtypes.int64)],
             dispatch=[('ev', sig(ArrayOf(ndim=1))),
                       ('ev_one', sig(Scalar))])
class _OdeSolutionC:
    """Continuous solution over a complex state, ``res.sol``.

    The complex twin of :class:`OdeSolution`, with the same interface and the
    same evaluator.  A jitclass field carries one dtype, so the two states
    need two classes; ``solve_ivp`` picks between them from ``y0``'s dtype
    when the call compiles.

    Attributes
    ----------
    t : float64 array, shape (s,)
        Start time of each accepted step.
    h : float64 array, shape (s,)
        Signed step size of each accepted step.
    ys : complex128 array, shape (s, n)
        State at the start of each accepted step.
    C : complex128 array, shape (s, 7, n)
        Interpolation coefficients per step.
    method : int
        The method the history was produced with.
    """

    def __init__(self, t, h, ys, C, method):
        """Store the step history.  Every argument is required; see
        :class:`OdeSolution` on constructor defaults."""
        self.t = t
        self.h = h
        self.ys = ys
        self.C = C
        self.method = method

    def ev(self, tt):
        """Evaluate at an array of times, returning ``(n_states, len(tt))``."""
        return rk_dense_eval(tt, self.t, self.h, self.ys, self.C,
                             self.method).T.copy()

    def ev_one(self, tt):
        """Evaluate at a single time, returning ``(n_states,)``."""
        one = np.empty(1, np.float64)
        one[0] = tt
        return rk_dense_eval(one, self.t, self.h, self.ys, self.C,
                             self.method)[0]


@njit
def OdeSolutionC(t, h, ys, C, method):
    """Build an :class:`_OdeSolutionC` from a complex step history.

    ``solve_ivp(..., dense_output=True)`` on a complex ``y0`` calls this and
    returns the result as ``res.sol``.  The real counterpart is
    :class:`OdeSolution`; the two take the same arguments and differ only in
    the dtype of ``ys`` and ``C``.

    Parameters
    ----------
    t, h : float64 array, shape (s,)
        Start time and signed step size of each accepted step.
    ys : complex128 array, shape (s, n)
        State at the start of each accepted step.
    C : complex128 array, shape (s, 7, n)
        Interpolation coefficients per step.
    method : int
        ``METHOD_RK45``, ``METHOD_RK23`` or ``METHOD_DOP853``, the method the
        history was produced with.

    Returns
    -------
    sol : _OdeSolutionC
        Callable as ``sol(t)``.

    Raises
    ------
    ValueError
        If ``t`` is not strictly increasing or strictly decreasing, or if
        ``h``, ``ys`` and ``C`` do not all carry one entry per time stamp.
    """
    _check_hist_t(t)
    n = t.shape[0]
    if h.shape[0] != n or ys.shape[0] != n or C.shape[0] != n:
        raise ValueError(_HIST_N_MSG)
    return _OdeSolutionC(t, h, ys, C, method)


@scijitclass([('t', nbtypes.float64[:]),
              ('h', nbtypes.float64[:]),
              ('yh', nbtypes.float64[:, :, :])],
             dispatch=[('ev', sig(ArrayOf(ndim=1))),
                       ('ev_one', sig(Scalar))])
class _LsodaSolution:
    """Continuous solution over an LSODA span, ``res.sol``.

    Same surface as :class:`OdeSolution`: ``sol(t)`` evaluates it, from
    Python and from inside ``@njit``, a float giving ``(n_states,)`` and a
    float64 array of ``k`` times giving ``(n_states, k)``.  The two are
    separate classes because the stored history is different, a Nordsieck
    array here against Runge-Kutta interpolation coefficients there.

    **Constructor defaults are Python-only.**  Construct through the
    :func:`LsodaSolution` factory, whose defaults work in both worlds.
    Callers normally never construct one: ``solve_ivp(..., method='LSODA',
    dense_output=True)`` returns it.

    Attributes
    ----------
    t : float64 array, shape (s,)
        Time each accepted step reached.
    h : float64 array, shape (s,)
        Step size each Nordsieck block is scaled for.
    yh : float64 array, shape (s, n, 13)
        Nordsieck history per step; column ``j`` is ``h^j/j! * y^(j)``.
    """

    def __init__(self, t, h, yh):
        """Store the step history.  Every argument is required; see the class
        docstring on constructor defaults."""
        self.t = t
        self.h = h
        self.yh = yh

    def ev(self, tt):
        """Evaluate at an array of times, returning ``(n_states, len(tt))``."""
        return lsoda_dense_eval(tt, self.t, self.h, self.yh).T.copy()

    def ev_one(self, tt):
        """Evaluate at a single time, returning ``(n_states,)``."""
        one = np.empty(1, np.float64)
        one[0] = tt
        return lsoda_dense_eval(one, self.t, self.h, self.yh)[0]


@njit
def LsodaSolution(t, h, yh):
    """Build a :class:`_LsodaSolution` from an LSODA step history.

    ``solve_ivp(..., method='LSODA', dense_output=True)`` calls this and
    returns the result as ``res.sol``, which is how most callers meet it.

    A plain ``@njit`` factory rather than the bare class, because a class
    constructor's defaults are Python-only: this function's defaults work
    from Python and from inside ``@njit`` alike.  It takes no defaults
    today, so every argument is required in both worlds.

    Parameters
    ----------
    t, h : float64 array, shape (s,)
        Time each accepted step reached, and the step size its Nordsieck
        block is scaled for.
    yh : float64 array, shape (s, n, 13)
        Nordsieck history per step.

    Returns
    -------
    sol : _LsodaSolution
        Callable as ``sol(t)``.

    Raises
    ------
    ValueError
        If ``t`` is not strictly increasing or strictly decreasing, or if
        ``h`` and ``yh`` do not both carry one entry per time stamp.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> from scijit.integrate._solve_ivp import LsodaSolution
    >>> @njit
    ... def rhs(t, y):
    ...     out = np.empty(2)
    ...     out[0] = y[1]
    ...     out[1] = -4.0 * y[0]
    ...     return out
    >>> res = si.solve_ivp(rhs, (0.0, 1.0), np.array([1.0, 0.0]),
    ...                    'LSODA', None, True)
    >>> sol = LsodaSolution(res.sol.t, res.sol.h, res.sol.yh)
    >>> np.allclose(sol(0.5), res.sol(0.5))
    True
    """
    _check_hist_t(t)
    n = t.shape[0]
    if h.shape[0] != n or yh.shape[0] != n:
        raise ValueError(_HIST_N_MSG)
    return _LsodaSolution(t, h, yh)


# ---------------------------------------------------------------------------
# compile-time helpers
# ---------------------------------------------------------------------------
def _passed(v):
    """True when the caller WROTE this argument, whatever value it holds.

    An omitted default reaches an ``@overload`` chooser as the raw Python
    value and one written out reaches it as a numba type, so the two are
    distinguishable without a sentinel.  ``types.Omitted`` is a ``Type``
    subclass and stands for the omitted case, so it is excluded: an
    argument the caller wrote arrives as ``none``, ``int64`` or ``float64``,
    never as ``Omitted``.
    """
    return isinstance(v, types.Type) and not isinstance(v, types.Omitted)


#: The pre-flight probe's refusal, raised by both LSODA runs.
_YDOT_MSG = ("the right-hand side never wrote ydot. Check the @cfunc "
             "signature and argument order, or, for a plain @njit rhs, "
             "that it returns an array as long as y0")

#: `fun` must be a plain @njit function on every method. A @cfunc address or
#: raw integer pointer is no longer accepted; the solver-side @cfunc is built
#: from the plain function when the call compiles.
_RK_ADDR_MSG = (
    "fun must be a plain @njit function f(t, y) or f(t, y, args). A @cfunc "
    "address or raw integer pointer is not accepted: pass the @njit function "
    "itself and solve_ivp builds the callback")

#: `vectorized` selects which callback adapter is built, so it has to be
#: known when the call compiles. Inside `@njit` a literal `True`/`False` and
#: an omitted default all reach the chooser as values; a variable does not.
_VEC_MSG = ("vectorized must be a literal True or False inside @njit, since "
            "it selects the right-hand side adapter when the call compiles. "
            "Write the flag out at the call site, or call from python")

#: `method` takes scipy's strings and nothing else. The `METHOD_*` int codes
#: address the array-level entry points, which have no scipy counterpart.
_METHOD_MSG = "method must be 'RK45', 'RK23', 'DOP853' or 'LSODA'"

#: scipy's wording when a complex `y0` meets a real-only solver,
#: `_ivp/base.py:check_arguments`.  Measured on scipy 1.18: `'Radau'` and
#: `'LSODA'` raise it, `'RK45'`, `'RK23'`, `'DOP853'` and `'BDF'` do not.
_CPLX_Y0_MSG = ("`y0` is complex, but the chosen solver does not support "
                "integration in a complex domain.")

#: Events over a complex state.  The event machinery stores the state at each
#: root in a float64 array, so the root itself is real and the state beside it
#: is not expressible.  scipy carries both.
_CPLX_EVENTS_MSG = (
    "`events` are not supported with a complex `y0`; scipy accepts the "
    "combination and this package does not yet")


def _method_code(m, y0_complex=False):
    """scipy's method string, as the internal routing code.

    ``y0_complex`` selects the refusal a complex ``y0`` earns on a solver that
    does not carry one.  It is applied AFTER the name is recognised, which is
    scipy's order: an unknown ``method`` is reported before the state's dtype
    is looked at.
    """
    if not isinstance(m, str):
        raise ValueError(
            _METHOD_MSG + ", as a string. The METHOD_* int codes address "
            "_rk_core and rk_dense_eval, not this function")
    if m == 'RK45':
        return METHOD_RK45
    if m == 'RK23':
        return METHOD_RK23
    if m == 'DOP853':
        return METHOD_DOP853
    if m == 'LSODA':
        if y0_complex:
            raise ValueError(_CPLX_Y0_MSG)
        return METHOD_LSODA
    if m == 'Radau' or m == 'BDF':
        if y0_complex:
            raise ValueError(_CPLX_Y0_MSG)
        raise ValueError(_RADAU_BDF)
    raise ValueError(_METHOD_MSG)


def _is_complex_ty(t):
    """True when an argument's TYPE carries complex values.

    Reads the numba type an ``@overload`` chooser is handed, so the answer is
    available when the call compiles and never depends on a value.  Arrays,
    scalars, homogeneous and heterogeneous tuples and typed lists are all
    spellings a ``y0`` arrives in.
    """
    if isinstance(t, types.Omitted):
        t = t.value
    if isinstance(t, complex) and not isinstance(t, bool):
        return True
    if isinstance(t, np.ndarray):
        return bool(np.iscomplexobj(t))
    if isinstance(t, (list, tuple)):
        return any(_is_complex_ty(e) for e in t)
    if isinstance(t, types.Complex):
        return True
    if isinstance(t, types.Array):
        return isinstance(t.dtype, types.Complex)
    if isinstance(t, types.BaseTuple):
        return any(_is_complex_ty(e) for e in t.types)
    dt = getattr(t, 'dtype', None)
    if dt is not None and isinstance(dt, types.Complex):
        return True
    return False


@njit
def _as_state_f8(y0):
    """``y0`` as a contiguous 1-D float64 state vector."""
    a = np.asarray(y0).astype(np.float64)
    if a.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")
    return np.ascontiguousarray(a).ravel()


@njit
def _as_state_c16(y0):
    """``y0`` as a contiguous 1-D complex128 state vector."""
    a = np.asarray(y0).astype(np.complex128)
    if a.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")
    return np.ascontiguousarray(a).ravel()


# ---------------------------------------------------------------------------
# argument normalisation.  `None` is resolved HERE and never travels further.
# ---------------------------------------------------------------------------
@njit
def _teval_or_grid(t_eval, t0, t1, npoints):
    """``t_eval``, or a uniform grid when the caller supplied none."""
    if t_eval.size > 0:
        return t_eval
    if t1 == t0:
        # A uniform grid over a zero-length span is `npoints` copies of t0.
        # scipy reports the point twice there, which is what the RK branch
        # does as well.
        return np.full(2, t0)
    return np.linspace(t0, t1, npoints)


@njit
def _tol_arrays(rtol, atol, n):
    """``(rtol_buf, atol_buf, itol)`` for ODEPACK's scalar-or-vector selector."""
    rv = np.ascontiguousarray(np.asarray(rtol).astype(np.float64)).ravel()
    _a = np.asarray(atol).astype(np.float64)
    # scipy's `validate_tol` shape-tests `atol` and not `rtol`, on every
    # method: `atol.ndim > 0 and atol.shape != (n,)` raises. So a length-1
    # `atol` ARRAY is refused where n > 1, and a length-1 `rtol` broadcasts.
    if _a.ndim > 1 or (_a.ndim == 1 and _a.size != n):
        raise ValueError("`atol` has wrong shape.")
    av = np.ascontiguousarray(_a).ravel()
    if rv.size != 1 and rv.size != n:
        raise ValueError(_RTOL_SHAPE_MSG)
    # scipy's `solve_ivp` runs `validate_tol` (`_ivp/common.py:44-60`) before
    # the solver on EVERY method, LSODA included, so this branch validates the
    # same way the RK branch does: a negative `atol` is refused and a
    # too-small `rtol` is clamped and warned about. Measured in scipy on
    # `method='LSODA'`: `atol=-1` raises "`atol` must be positive.", while
    # `rtol=0`, `rtol=-1` and `rtol=1e-20` integrate successfully at nfev 35.
    #
    # `odeint` deliberately does NOT do this. scipy's `odeint` never calls
    # `validate_tol`, so a negative tolerance reaches ODEPACK and comes back
    # as "Illegal input detected (internal error)". Measured on both sides,
    # and this front end already matches it. The two front ends disagree
    # because scipy's two disagree.
    for i in range(av.size):
        if av[i] < 0.0:
            raise ValueError("`atol` must be positive.")
    clamped = False
    for i in range(rv.size):
        if rv[i] < 100.0 * _EPS_IVP:
            rv[i] = 100.0 * _EPS_IVP
            clamped = True
    if clamped:
        with objmode():
            warnings.warn(
                "At least one element of `rtol` is too small. "
                "Setting `rtol = np.maximum(rtol, 2.220446049250313e-14)`.")
    # ODEPACK itol: 1 both scalar, 2 atol vector, 3 rtol vector, 4 both
    if rv.size == 1 and av.size == 1:
        itol = 1
    elif rv.size == 1:
        itol = 2
    elif av.size == 1:
        itol = 3
    else:
        itol = 4
    return rv, av, itol


#: A nested or 2-D `t_span`.  scipy reports numpy's own
#: "only 0-dimensional arrays can be converted to Python scalars", which comes
#: out of `float()` and names neither the argument nor the rule; this names
#: both.  The class is scipy's.
_SPAN_RANK_MSG = ("each element of t_span must be a real number; a nested or "
                  "2-D span is not a pair of numbers")


@njit
def _teval_1d(t_eval):
    """``t_eval`` as a contiguous 1-D float64 array.

    The rank test precedes the ravel, so a 2-D ``t_eval`` is refused with
    scipy's text rather than flattened into a longer list of times.
    """
    a = np.asarray(t_eval).astype(np.float64)
    if a.ndim != 1:
        raise ValueError("`t_eval` must be 1-dimensional.")
    return np.ascontiguousarray(a).ravel()


@njit
def _span2(t_span):
    """``(t0, t1)`` from ``t_span``, refusing anything but two elements.

    The Python entry ravelled and length-checked; the compiled body read
    ``t_span[0]`` and ``t_span[1]`` and stopped there, so a three-element
    span integrated over its first two and a one-element one read past the
    end, which numba does not bounds-check.  Both now come through here.

    A nested or 2-D span is a ``TypeError``, ahead of the length test: scipy
    does ``t0, tf = map(float, t_span)`` and ``float()`` of a row raises.
    Ravelling first hid a ``(1, 2)`` span, which has two elements and is not
    a pair of numbers.
    """
    _ts = np.asarray(t_span).astype(np.float64)
    if _ts.ndim > 1:
        raise TypeError(_SPAN_RANK_MSG)
    ts = _ts.ravel()
    if ts.size != 2:
        raise ValueError("t_span must have exactly two elements")
    return ts[0], ts[1]


def _args_len_ty(args):
    """The length of ``args`` at TYPING time, or ``-1`` when only the run
    knows it.

    A numba array type carries rank, dtype and layout and no length, so a
    zero-length ``args`` array looks exactly like a full one to the chooser.
    A tuple carries its length in its type and a ``None`` carries it by
    being ``None``, so those two are decidable here.
    """
    if _is_none(args):
        return 0
    if isinstance(args, types.BaseTuple):
        return len(args.types)
    if isinstance(args, types.Array):
        return -1
    return 1


@njit
def _check_ivp_args(t0, t1, y0, max_step, fstep, has_fstep, min_step):
    """The guards scipy runs before the solver is chosen, for both branches.

    scipy validates in ``solve_ivp`` and in ``OdeSolver.__init__``, neither of
    which knows which method was asked for, so `'LSODA'` and the RK methods
    meet the same checks.  Here both entry points call this, so the compiled
    body cannot drift from the Python one.

    ``has_fstep`` says ``first_step`` was supplied; ``None`` means "choose
    one" and reaches this as ``fstep = 0.0``, which is a value scipy refuses
    when it is written out.
    """
    if np.isnan(t0) or np.isnan(t1):
        raise ValueError("t0 and t1 must not be nan")
    for i in range(y0.size):
        if not np.isfinite(y0[i]):
            raise ValueError(
                "All components of the initial state `y0` must be finite.")
    if max_step <= 0.0:
        raise ValueError("`max_step` must be positive.")
    if has_fstep:
        if fstep <= 0.0:
            raise ValueError("`first_step` must be positive.")
        if fstep > abs(t1 - t0):
            raise ValueError("`first_step` exceeds bounds.")
    if min_step < 0.0:
        raise ValueError("`min_step` must be nonnegative.")


# ---------------------------------------------------------------------------
# the two cores.  Both return everything; the entries only slice.
# ---------------------------------------------------------------------------
def _arity(py):
    """Positional parameter count, or ``None`` for ``*args``.

    A local copy of `_nquad`'s. Duplicated on purpose rather than imported:
    an import edge from the IVP half to the quadrature half for six lines is
    the trade `state/FIXES_NEEDED.md` F1 already declined once.
    """
    try:
        sig = inspect.signature(py)
    except (TypeError, ValueError):                          # noqa: BLE001
        return None
    n = 0
    for prm in sig.parameters.values():
        if prm.kind is prm.VAR_POSITIONAL:
            return None
        if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


#: Devectorised right-hand sides, keyed by the function they wrap.
_DEVEC = {}


def _devectorize(py):
    """A ``vectorized=True`` right-hand side, presented as a single-state one.

    scipy's contract, read off `OdeSolver.__init__` and confirmed by calling
    it: with ``vectorized=True`` the right-hand side is invoked as
    ``fun(t, y[:, None])``, shape ``(n, 1)``, and the result is ravelled.
    Measured on `scipy.integrate.solve_ivp`, the shapes seen by the callback
    are ``(2,)`` without the flag and ``(2, 1)`` with it, on RK45 and on
    LSODA alike, and the answer is the same either way.

    So this is an adapter rather than a solver change: nothing downstream
    needs to know. Built once per function at TYPING time, like `_swap2` in
    `_nquad`, because a numba closure over a function value is avoidable here
    and a generated one is not.

    scipy notes that vectorising buys nothing for an explicit RK method and
    nothing for LSODA; it pays off for the implicit solvers, which this
    package does not have yet. The flag is accepted for call compatibility,
    so a scipy-shaped call runs unchanged.
    """
    hit = _DEVEC.get(py)
    if hit is not None:
        return hit
    ar = _arity(py)
    if ar == 3:
        params, call = "t, y, args", "t, y.reshape(y.size, 1), args"
    else:
        params, call = "t, y", "t, y.reshape(y.size, 1)"
    ns = {"inner": njit(py), "np": np}
    exec("def single(%s):\n    return inner(%s).ravel()" % (params, call), ns)
    out = njit(ns["single"])
    _DEVEC[py] = out
    return out


@njit
def _core_rk(rhs, t0, t1, y0, code, rtol, atol, t_eval, max_step, want_dense,
             first_step=0.0, args=_NO_ARGS, teval_given=False):
    """RK branch. Returns ``(t, y_component_major, success, nfev, dt, dh, dy, dC)``."""
    r = _rk_core(rhs, t0, t1, y0, code, rtol, atol, t_eval, max_step,
                 want_dense, first_step, args, teval_given)
    return r[0], r[1].T.copy(), r[2], r[3], r[4], r[5], r[6], r[7]


@njit
def _report_nothing(n):
    """The empty report an EMPTY ``t_eval`` earns, in the RK branch's shapes.

    ``t_eval=[]`` and ``t_eval=None`` are different inputs: the first asks for
    a report at no times at all.  The RK engine already distinguishes them
    through its ``teval_given`` flag and returns ``(0,)`` and ``(n, 0)``; this
    gives the LSODA branch the same two shapes from the same place.
    """
    return np.zeros(0, np.float64), np.zeros((n, 0), np.float64)


@njit
def _core_lsoda(funcptr, y0, t, rtol, atol, mxstep, max_step, args, t0,
                first_step=0.0, min_step=0.0, jt=2, ml=0, mu=0, jacptr=0):
    """LSODA reporting run: values at requested times, no step history.

    Returns ``(t, y_component_major, nfev, njev, istate)``.

    The pre-flight probe runs here rather than at the entry point so it
    covers the Python entry and the compiled one from one place. Without
    it a right-hand side that writes no derivatives returns the initial
    condition, held constant, with ``success=True``.

    ``t0`` is separate from ``t`` because the Fortran driver takes the
    reporting grid as the integration span and reads its FIRST entry as the
    initial time. That is ``odeint``'s contract, not ``solve_ivp``'s: here
    ``y0`` belongs to ``t_span[0]`` and ``t_eval`` only says where to
    report, so a ``t_eval`` starting later than ``t_span[0]`` needs ``t0``
    prepended and the extra row dropped afterwards.
    """
    if not _wrote_ydot_t(funcptr, t[0], y0, args):
        raise ValueError(_YDOT_MSG)
    lead = t.size > 0 and t[0] != t0
    if lead:
        tin = np.empty(t.size + 1, np.float64)
        tin[0] = t0
        for i in range(t.size):
            tin[i + 1] = t[i]
    else:
        tin = t
    rv, av, itol = _tol_arrays(rtol, atol, y0.size)
    if max_step == np.inf:
        hmax = 0.0                       # LSODA reads 0 as "no limit"
    else:
        hmax = max_step
    # h0 and hmin are scipy's `first_step` and `min_step`. LSODA reads 0
    # as "choose one" for h0 and "no floor" for hmin, which is what the
    # defaults resolve to.
    #
    # h0 carries the SIGN of the direction of travel. scipy does
    # `first_step *= self.direction` straight after validating it
    # (`_ivp/lsoda.py:126-131`), so a backward run hands ODEPACK a negative
    # h0; an unsigned one asks for a first step out of the interval.
    if tin.size > 1 and tin[tin.size - 1] < tin[0]:
        h0 = -first_step
    else:
        h0 = first_step
    r = _run_odeint(funcptr, 1, y0, tin, rv, av, itol, 0, np.zeros(1, np.float64),
                    h0, hmax, min_step, 0, mxstep, 0, 12, 5, args,
                    jt, ml, mu, jacptr)
    y = r[0]                             # (ntime, neq)
    if lead:
        y = y[1:]                        # drop the prepended t0 row
    nfe = r[6]
    nje = r[7]
    if nfe.size > 0:
        nfev = nfe[nfe.size - 1]         # iwork(12) is cumulative
        njev = nje[nje.size - 1]
    else:
        nfev = 0
        njev = 0
    return t, y.T.copy(), nfev, njev, r[13]


@njit
def _core_lsoda_dense(funcptr, y0, t0, t1, rtol, atol, mxstep, args,
                      jt=2, ml=0, mu=0, jacptr=0, itol=1,
                      h0=0.0, hmax=0.0, hmin=0.0, cap0=4096):
    """LSODA step history for ``dense_output=True``.

    Returns ``(nsteps, t, h, yh, istate, nfev, njev)``.  The last two are
    LSODA's own cumulative counters for this run, so a caller that needs both
    a history and the work counts does not have to integrate twice.

    ``itol``, ``h0``, ``hmax`` and ``hmin`` are passed straight to
    :func:`_run_lsoda_dense` and carry ``first_step``, ``max_step``,
    ``min_step`` and a vector ``rtol``/``atol`` onto the history run, so this
    run and the reporting run beside it integrate the same problem.  The
    defaults are ODEPACK's own and reproduce what this function did before
    they existed.

    The buffer has to hold every accepted step of the whole run, and nothing
    knows that count in advance.  ``mxstep`` is the wrong size for it:
    ODEPACK counts ``mxstep`` internal steps BETWEEN TWO OUTPUT TIMES, not
    across the integration, so a stiff problem overruns it easily.  Measured
    on Robertson over ``[0, 40]`` at ``rtol = atol = 1e-10``, which takes 313
    accepted steps: a history truncated to 275 of them stops at ``t = 25.07``,
    and evaluating at ``t = 35`` EXTRAPOLATED, 5.625e-05 wrong while
    reporting success.

    So the buffer grows until the history actually reaches ``t1``.  A run
    that stops early for a real reason, ``istate < 0``, is returned as it
    stands for the caller to report.  Only a run that succeeded and ran out
    of room is continued.

    Each further buffer CONTINUES the integration rather than starting it
    again: LSODA's state is the four arrays :func:`_lsoda_dense_work`
    allocates, held here across the segments, and ODEPACK's own idiom is
    ``istate = 2`` with them untouched.  So the whole history costs one
    integration whatever the number of buffers, and the segments are joined
    end to end.  ``cap0`` is the first buffer's step count; the history stops
    growing at 4,194,304 steps, which is a step-count limit and not a memory
    one, since the cost per step is ``32 + 104 * neq`` bytes.
    """
    rwork, iwork, cs_r, cs_i = _lsoda_dense_work(y0.size, jt, ml, mu)
    cap = cap0
    r = _run_lsoda_dense(funcptr, y0, t0, t1, rtol, atol, mxstep, cap, args,
                         jt, ml, mu, jacptr, itol, h0, hmax, hmin,
                         rwork, iwork, cs_r, cs_i, 1)
    if t1 == t0:
        # LSODA takes no step over a zero-length span and reports h = 0,
        # which the evaluator divides by, so `res.sol(t)` came back as a bare
        # ZeroDivisionError. One constant block replaces it: column 0 holds
        # y0 and the rest are zero, making the polynomial y0 at every
        # argument. scipy's interpolant over the same span is also constant.
        n = y0.size
        yh = np.zeros((1, n, r[3].shape[2]), np.float64)
        for i in range(n):
            yh[0, i, 0] = y0[i]
        return 1, np.full(1, t0), np.ones(1), yh, r[4], r[5], r[6]
    direction = 1.0 if t1 >= t0 else -1.0
    total = r[0]
    t_all = r[1]
    h_all = r[2]
    yh_all = r[3]
    istate = r[4]
    nfev = r[5]
    njev = r[6]
    y_end = r[7]
    t_end = r[8]
    seg = r[0]
    while (istate >= 0 and seg == cap
           and direction * (t_all[total - 1] - t1) < 0.0
           and total < 4194304):
        cap *= 4
        r = _run_lsoda_dense(funcptr, y_end, t_end, t1, rtol, atol, mxstep,
                             cap, args, jt, ml, mu, jacptr, itol, h0, hmax,
                             hmin, rwork, iwork, cs_r, cs_i, 2)
        seg = r[0]
        istate = r[4]
        nfev = r[5]
        njev = r[6]
        y_end = r[7]
        t_end = r[8]
        if seg == 0:
            break
        t_all = np.concatenate((t_all, r[1]))
        h_all = np.concatenate((h_all, r[2]))
        yh_all = np.concatenate((yh_all, r[3]))
        total += seg
    return total, t_all, h_all, yh_all, istate, nfev, njev


@njit
def _hist_lsoda(funcptr, y0, t0, t1, rtol, atol, mxstep, args, jt, ml, mu,
                jacptr, max_step, first_step, min_step):
    """The step history, under the settings the reporting run takes.

    ``max_step``, ``first_step``, ``min_step`` and a vector ``rtol`` or
    ``atol`` each change the integration, so a history that answers for the
    whole call has to carry them.  They reach ODEPACK as ``hmax``, ``h0``,
    ``hmin`` and ``itol``, translated exactly as :func:`_core_lsoda`
    translates them: ``max_step = inf`` is LSODA's ``hmax = 0``, and ``h0``
    carries the sign of the direction of travel.

    The pre-flight probe runs here for the same reason it runs in
    :func:`_core_lsoda`: a right-hand side that writes no derivatives
    otherwise returns the initial condition, held constant, with
    ``success=True``.
    """
    if not _wrote_ydot_t(funcptr, t0, y0, args):
        raise ValueError(_YDOT_MSG)
    rv, av, itol = _tol_arrays(rtol, atol, y0.size)
    if max_step == np.inf:
        hmax = 0.0
    else:
        hmax = max_step
    if t1 < t0:
        h0 = -first_step
    else:
        h0 = first_step
    return _core_lsoda_dense(funcptr, y0, t0, t1, rv, av, mxstep, args,
                             jt, ml, mu, jacptr, itol, h0, hmax, min_step)


@njit
def _report_hist_steps(d, y0, t0, t1, ok):
    """The report ``t_eval=None`` earns: the solver's own steps.

    scipy reports ``t_span[0]``, every step the solver accepted inside the
    span, and ``t_span[1]``.  The history holds the same steps without the
    initial time, so the ends are supplied here: ``y0`` at ``t0``, and the
    state at ``t1`` read out of the history.  The last step lands on ``t1``,
    ODEPACK's ``TCRIT``, so that read returns the step's own state rather
    than an interpolation between two.

    A run that stopped early carries no endpoint.  The history reaches only
    as far as the solver got, and appending ``t1`` would extrapolate past
    it; scipy's loop stops appending at the same point.
    """
    n = d[0]
    t = d[1]
    yh = d[3]
    neq = y0.size
    if t1 >= t0:
        dirn = 1.0
    else:
        dirn = -1.0
    m = 0
    for k in range(n):
        if dirn * (t[k] - t0) > 0.0 and dirn * (t[k] - t1) < 0.0:
            m += 1
    if ok:
        size = m + 2
    else:
        size = m + 1
    rt = np.empty(size, np.float64)
    ry = np.empty((neq, size), np.float64)
    rt[0] = t0
    for i in range(neq):
        ry[i, 0] = y0[i]
    j = 1
    for k in range(n):
        if dirn * (t[k] - t0) > 0.0 and dirn * (t[k] - t1) < 0.0:
            rt[j] = t[k]
            for i in range(neq):
                ry[i, j] = yh[k, i, 0]
            j += 1
    if ok:
        one = np.empty(1, np.float64)
        one[0] = t1
        end = lsoda_dense_eval(one, t, d[2], yh)
        rt[j] = t1
        for i in range(neq):
            ry[i, j] = end[0, i]
    return rt, ry


@njit
def _report_hist_at(d, te):
    """The history evaluated at ``t_eval``, component-major."""
    return lsoda_dense_eval(te, d[1], d[2], d[3]).T.copy()


@njit
def _apply_events(g, ng, terminal, direction, kind, d_t, d_h, d_y, d_C,
                  d_yh, method, t0, t, y, args):
    """Scan a finished history for event roots and trim on a terminal one.

    Returns ``(t, y, t_events, y_events, status)``.  ``status`` is 1 when a
    terminal event fired and 0 otherwise, which is scipy's coding.
    """
    roots, idx, t_stop = scan_events(g, ng, terminal, direction, kind, d_t,
                                     d_h, d_y, d_C, d_yh, method, t0, args)
    te, ye = split_events(roots, idx, ng, t_stop, kind, d_t, d_h, d_y, d_C,
                          d_yh, method, y.shape[0])
    if np.isnan(t_stop):
        return t, y, te, ye, 0
    tt, yy = truncate_at(t, y, t_stop, kind, d_t, d_h, d_y, d_C, d_yh,
                         method)
    return tt, yy, te, ye, 1


@njit
def _event_spec(g, t0, y0, terminal, direction, args):
    """``(ng, terminal, direction)`` with scipy's defaults filled in.

    ``ng`` costs one evaluation of ``g``, which is what scipy pays to learn
    the same thing.  ``terminal=None`` means every event is non-terminal and
    ``direction=None`` means every crossing counts, both scipy's defaults.

    An entry of ``terminal`` is scipy's ``max_events``: ``n`` stops the
    integration at the ``n``-th occurrence and 0 never stops.  Anything
    negative or fractional is refused, as ``prepare_events`` refuses it.
    """
    ng = g(t0, y0, args).size
    if terminal is None:
        tm = np.zeros(ng, np.int64)
    else:
        tf = np.asarray(terminal).astype(np.float64).ravel()
        for i in range(tf.size):
            if tf[i] < 0.0 or tf[i] != np.floor(tf[i]):
                raise ValueError(_TERMINAL_MSG)
        tm = tf.astype(np.int64)
        if tm.size != ng:
            raise ValueError(
                "terminal must have one entry per event function")
    if direction is None:
        dr = np.zeros(ng, np.float64)
    else:
        dr = np.asarray(direction).astype(np.float64).ravel()
        if dr.size != ng:
            raise ValueError(
                "direction must have one entry per event function")
    return ng, tm, dr


#: scipy's wording for a `terminal` that is neither a boolean nor a positive
#: integer, `_ivp/ivp.py:40-41`.
_TERMINAL_MSG = ("The `terminal` attribute of each event must be a boolean "
                 "or positive integer.")

#: scipy's ``status`` messages, ``_ivp.ivp.MESSAGES``.
_MSG_DONE = ("The solver successfully reached the end of the integration "
             "interval.")
_MSG_EVENT = "A termination event occurred."
_MSG_FAIL = "Required step size is less than spacing between numbers."

_EMPTY_HIST_2 = np.zeros((0, 0), np.float64)
_EMPTY_HIST_3 = np.zeros((0, 0, 0), np.float64)


# ---------------------------------------------------------------------------
# the public front end
# ---------------------------------------------------------------------------
#: The four solver options this front end collects in ``**options`` rather
#: than naming, with their defaults. They are scipy's own ``**options``
#: members, and collecting them is what makes "was this passed" answerable:
#: a named parameter defaulting to ``None`` reads the same either way.
_SIVP_OPTIONS = {'min_step': 0.0, 'jac': None, 'lband': None, 'uband': None}


def solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
              events=None, vectorized=False, args=None, rtol=1e-3, atol=1e-6,
              max_step=np.inf, mxstep=0, npoints=100, terminal=None,
              direction=None, first_step=None, **options):
    """Solve an initial value problem for a system of ODEs.

    Integrates ``dy/dt = fun(t, y)`` from ``t_span[0]`` to ``t_span[1]``,
    starting at ``y0``.

    Parameters
    ----------
    fun : @njit function ``f(t, y)`` or ``f(t, y, args)``
        Right-hand side, t-first.  ``y`` is a 1-D float64 array and the return
        must be a new 1-D float64 array of the same length.  The three-argument
        form receives ``args``, on every method.  A plain ``@njit`` function,
        on every method.
    t_span : 2-tuple or length-2 array of float
        ``(t0, tf)``.  ``tf < t0`` integrates backwards, on every method.
        Without ``t_eval``, ``tf == t0`` returns the initial state twice; with
        ``t_eval`` it reports nothing.
    y0 : float64 or complex128 array, shape (n,)
        Initial state.  Must be 1-D.  A ``complex128`` ``y0`` integrates in the
        complex domain on ``'RK45'``, ``'RK23'`` and ``'DOP853'``, and the
        result's ``y`` and ``sol`` are ``complex128``.  ``'LSODA'`` raises
        there.  The dtype is read when the call compiles, so the result type
        never depends on a value.
    method : str, optional
        ``'RK45'`` (default), ``'RK23'``, ``'DOP853'`` or ``'LSODA'``.
        Case sensitive.  Inside ``@njit`` it must be a string literal, since it
        selects which solver is compiled in.  ``'Radau'`` and ``'BDF'`` raise.
    t_eval : float64 array or None, optional
        Times to report at, inside ``t_span`` and STRICTLY ordered in the
        direction of travel, so a repeat is refused.  ``None`` means the
        solver's own steps, on every method.  An empty ``t_eval`` reports
        nothing on every method, which is a different input from ``None``.
        Must be 1-D.
    dense_output : bool, optional
        Compile-time literal.  ``True`` adds a callable ``sol`` to the result
        and changes the return type.  Available on every method.
    events : @njit function ``g(t, y)`` or ``g(t, y, args)``, optional
        Event functions, evaluated together and returning a float64 array of
        ``ng`` values, one per event.  A root of any of them is reported.
        ``ng`` is learnt by calling the function once at ``(t_span[0], y0)``.
        A single event may return a scalar.  The three-argument form receives
        ``args``, and the arity must agree with ``args`` in both directions or
        the call raises ``TypeError``.
    vectorized : bool, optional
        Whether the right-hand side is written to take a block of states:
        ``True`` means it is called as ``fun(t, y[:, None])``, shape
        ``(n, 1)``, and its result ravelled.  Accepted on every method and on
        both callback spellings.

        Inside ``@njit`` it must be a literal, since it selects which
        adapter is built when the call compiles.  A variable raises.
    args : float64 array or None, optional
        Extra parameters, one flat float64 buffer.  Reaches ``fun``, ``jac``
        and ``events`` on every method.  The callback has to be able to
        receive it: a three-argument ``f(t, y, args)``.  A two-argument
        ``f(t, y)`` given ``args`` raises
        ``TypeError``, and a three-argument one given no ``args`` raises
        ``TypeError``, in both directions.  An EMPTY ``args`` carries no
        parameters, so it counts as absent in both: ``f(t, y)`` runs and
        ``f(t, y, args)`` raises, from Python and from inside ``@njit``.
    rtol, atol : float or float64 array, optional
        Tolerances, ``1e-3`` and ``1e-6``.  Either may be a scalar or an array
        with one entry per component of ``y0``, on every method.  ``atol`` is
        shape-checked and ``rtol`` is not, so a length-1 ``atol`` array raises
        where ``len(y0) > 1`` and a length-1 ``rtol`` array broadcasts.

        An ``rtol`` below ``100 * eps``, 2.220446049250313e-14, is raised to
        it and a ``UserWarning`` says so, on every method.  ``atol = 0`` is
        legal and makes the tolerance purely relative; a negative ``atol``
        raises.  A purely relative tolerance collapses the step size wherever
        a component passes through zero.
    max_step : float, optional
        Largest step allowed.  ``np.inf`` by default.
    mxstep : int, optional
        ``'LSODA'`` only: step limit per output interval.  ``0`` selects
        LSODA's own 500.  On the RK methods it is ignored with a
        ``UserWarning``.
    npoints : int, optional
        ``'LSODA'`` with ``t_eval=None`` only: how many uniformly spaced
        output times to produce, INSTEAD of the solver's own steps.  The
        default, 100, reports the steps; any other value reports the grid.
        On the RK methods it is ignored with a ``UserWarning``.
    terminal : int64 array, shape (ng,), optional
        ``n`` stops the integration at that event's ``n``-th occurrence and
        ``0`` never stops it.  ``None`` (the default) makes every event
        non-terminal.  A negative or fractional entry raises ``ValueError``.

        With two terminal events crossing in one step the earlier root in
        TIME stops the run, not the lower event index.
    direction : float64 array, shape (ng,), optional
        Positive keeps upward crossings only, negative downward only, zero
        keeps both.  ``None`` (the default) keeps both for every event.
    first_step : float or None, optional
        Initial step size.  ``None``, the default, lets the solver choose
        one.  Reaches all four methods.
    **options : dict, optional
        The keyword-only ``'LSODA'`` controls ``min_step``, ``lband``,
        ``uband`` and ``jac``, listed under Other Parameters.  Any other
        keyword raises ``TypeError``.

    Returns
    -------
    res : OdeResult, or OdeResultDense when ``dense_output=True``
        ``res.t`` shape ``(m,)``; ``res.y`` shape ``(n, m)``, component-major;
        ``res.nfev`` right-hand-side evaluations;
        ``res.njev`` Jacobian evaluations; ``res.nlu`` LU decompositions;
        ``res.status`` 0 when the end of ``t_span`` was reached and -1 when a
        step failed; ``res.message`` and ``res.success``.
        ``res.sol`` is present only under ``dense_output=True`` and is callable
        as ``res.sol(t)`` for a scalar or an array of times.
        Without ``events``, ``res.t_events`` and ``res.y_events`` are ``None``;
        with ``events`` both are lists of arrays, one entry per event, shapes
        ``(k_i,)`` and ``(k_i, n)``.

    Other Parameters
    ----------------
    min_step : float, optional
        ``'LSODA'`` only: smallest step allowed.  ``0.0``, the default, means
        no floor.  On the RK methods it is ignored with a ``UserWarning``.
    lband, uband : int or None, optional
        ``'LSODA'`` only: ignored with a ``UserWarning`` on the RK methods.
        Half-bandwidths of the Jacobian, counting the sub- and super-diagonals
        and excluding the main diagonal, so ``d f_i / d y_j`` is taken as zero
        outside ``i - lband <= j <= i + uband``.  Setting either one selects
        LSODA's banded Jacobian and the other becomes 0.  ``None`` on both,
        the default, keeps the full Jacobian.

        LSODA rebuilds the Jacobian by finite differences unless ``jac`` is
        given, and a full rebuild costs ``len(y0)`` right-hand-side
        evaluations against ``lband + uband + 1`` for a banded one.
        Measured on an 80-point heat equation by method of lines, whose
        Jacobian is tridiagonal, ``rtol=1e-6 atol=1e-9`` over ``[0, 0.5]``:
        6 rebuilds, ``nfev`` 597 without the bands and 135 with
        ``lband=uband=1``, values agreeing to 0.000e+00, and 2.85x less wall
        time over 20 solves.

        A value outside ``0 <= lband, uband <= len(y0) - 1`` reaches
        ODEPACK, which reports "Illegal input detected (internal error)"
        and returns ``success=False``.

        A band NARROWER than the Jacobian is legal and expensive.  ODEPACK
        takes it as an approximation, and the corrector then needs many more
        iterations.  Measured on the same 40-point problem, ``lband=1`` alone
        against the correct ``lband=uband=1``: ``nfev`` 4523 and ``njev`` 640
        against 265 and 17, landing 1.083e-06 away.  This can exhaust
        ``mxstep``, which ODEPACK counts PER OUTPUT INTERVAL, and the run then
        reports "Excess work done on this call".  Raise ``mxstep``: at
        ``mxstep=100000`` every narrow band above succeeds.
    jac : @njit function ``j(t, y)`` or ``j(t, y, args)``, optional
        ``'LSODA'`` only: ignored with a ``UserWarning`` on the RK methods.
        The Jacobian ``d f / d y``.  Without it LSODA builds one by finite
        differences, at the cost above.

        Returns ``(n, n)`` with ``jac[i, j] = d f_i / d y_j``.  With
        ``lband``/``uband`` it returns the packed banded form instead,
        ``(lband + uband + 1, n)`` with
        ``jac_packed[uband + i - j, j] = d f_i / d y_j``.  A wrong shape raises
        ``ValueError``, from one evaluation at ``(t_span[0], y0)``.

        The three-argument form receives ``args``, as ``fun``'s does.

        Measured on Robertson, 3 states, ``rtol=1e-8 atol=1e-10``:
        ``nfev`` 502 without ``jac`` and 424 with, ``njev`` 26 either way,
        values agreeing to 1.369777e-11.  The 78 evaluations removed are
        exactly the ``26 * 3`` the finite-difference rebuilds cost.

    Raises
    ------
    ValueError
        ``method`` outside ``'RK45'``, ``'RK23'``, ``'DOP853'``, ``'LSODA'``;
        ``'Radau'`` or ``'BDF'``; a ``t_span`` without exactly two elements;
        a ``nan`` in ``t_span``; a ``y0`` that is not 1-D or not finite; a
        ``t_eval`` that is not 1-D, leaves ``t_span`` or is not strictly
        ordered; a non-positive ``max_step``; a non-positive ``first_step``,
        or one exceeding the span; a negative ``min_step``; a negative
        ``atol`` or one of the wrong shape; an ``rtol`` of the wrong length;
        a ``terminal`` entry that is negative or fractional; a ``terminal``
        or ``direction`` whose length is not ``ng``; a ``jac`` of the wrong
        shape; a right-hand side returning the wrong length; a complex ``y0``
        on ``'LSODA'``, ``'Radau'`` or ``'BDF'``; a complex ``y0`` together
        with ``events``; and, on ``'LSODA'``, a right-hand side that never
        wrote ``ydot``.
    TypeError
        A nested or 2-D ``t_span``; an arity mismatch between ``args`` and
        ``fun`` or ``events``, in either direction.
    RuntimeError
        An event root that Brent's method did not bracket in 100 iterations.

    See Also
    --------
    scipy.integrate.solve_ivp : The scipy routine this mirrors.

    Notes
    -----
    Every method is ``prange``-safe.

    ``'RK45'``, ``'RK23'`` and ``'DOP853'`` hold no module state.  ``'LSODA'``
    reaches Fortran, and its callback slot is ``!$omp threadprivate``, so each
    thread reads its own copy.

    **The solution object.** ``res.sol(t)`` takes a float or a float64 array.
    A Python list is not an array: numba types it as a reflected list, which
    no array guard admits, so ``res.sol([0.1, 0.2])`` raises ``TypeError``.
    Wrap it, ``res.sol(np.array([0.1, 0.2]))``.

    A solution object reaches compiled code as an argument, as a module
    global, or captured in a closure; a registered jitclass instance is a
    compile-time constant.

    Reading a shared solution object from a ``prange`` loop is safe and
    reproduces the serial result exactly.  Writing a field from several
    threads is a data race.

    **Differences from scipy.**

    - ``sol`` is absent from the result unless ``dense_output=True``, where
      scipy always carries the field and sets it to ``None``.  So
      ``hasattr(res, 'sol')`` answers what ``res.sol is None`` answers in
      scipy.
    - ``events`` is one function returning ``ng`` values, where scipy takes
      a callable or a list of them, and ``terminal`` / ``direction`` are
      arrays passed alongside rather than attributes set on the function.
      A single event may return a scalar, as scipy's do.
    - ``nfev`` after a terminal event covers the WHOLE span.  scipy leaves
      its stepping loop at the event, so its count stops there; this
      integrates the span and trims the history afterwards.
    - ``jac``, ``lband``, ``uband``, ``min_step``, ``mxstep`` and ``npoints``
      on ``'RK45'``, ``'RK23'`` and ``'DOP853'`` are ignored with one
      ``UserWarning`` per call, "The following arguments have no effect for a
      chosen solver: `mxstep`, `npoints`, `min_step`, `jac`, `lband`,
      `uband`.", which is character for character scipy's on the same call.
      The answer is the one the same call gives without them.

      ``min_step``, ``jac``, ``lband`` and ``uband`` warn when the argument
      was PASSED, whatever its value, so ``solve_ivp(..., 'RK45', jac=None)``
      warns.  ``mxstep`` and ``npoints`` warn when they carry a value other
      than their default, so ``solve_ivp(..., 'RK45', mxstep=0)`` does not.
      Neither has a scipy counterpart.

      The names are listed in signature order.  scipy lists them in the
      order the caller passed them, which a compiled call site does not
      preserve.
    - An unrecognised keyword argument raises ``TypeError``.  scipy absorbs
      it into ``**options`` and names it in the same ``UserWarning``.
    - ``jac`` is a plain ``@njit`` function taking ``args`` as one flat
      float64 buffer, matching ``fun``, rather than scipy's callable with
      unpacked parameters.
    - ``args`` reaches the callback as one flat float64 buffer rather than
      unpacked into separate parameters.  scipy calls ``f(t, y, *args)``;
      this calls ``f(t, y, args)`` with everything packed into one array,
      because numba has no ``*args``.  Which methods accept it no longer
      differs: every one does, as in scipy.
    - An arity mismatch between ``fun`` and ``args`` is refused in both
      directions, as in scipy: ``args`` given to an ``f(t, y)``, and ``args``
      omitted for an ``f(t, y, args)``.  An empty ``args`` is not a mismatch
      either way.  The exception is ``TypeError``, scipy's, from python, and
      ``TypingError`` from inside ``@njit``.  One case differs there: an
      array's length is not known when the call compiles, so an empty
      ``args`` reads as given inside ``@njit`` and as absent from python.
    - ``tf == t0`` with a ``t_eval``, and an empty ``t_eval`` on any span,
      report nothing, as in scipy.  scipy returns ``t`` and ``y`` as empty
      Python lists there and this returns empty arrays, shapes ``(0,)`` and
      ``(n, 0)``.
    - ``events`` and a complex ``y0`` together raise.  scipy carries both:
      it stores the state at each event root, and here that store is
      ``float64``.  Every other combination of a complex ``y0`` with
      ``t_eval``, ``dense_output`` and ``args`` is supported.
    - ``'Radau'`` and ``'BDF'`` raise.  They are planned.
    - The result is a namedtuple with scipy's field names, so ``res.y``
      works and ``res['y']`` does not.
    - ``'LSODA'`` with a ``t_eval`` and neither ``dense_output`` nor
      ``events`` reports through a run that stops at each requested time,
      the route scipy's ``odeint`` takes, where scipy's ``solve_ivp`` steps
      freely and interpolates.  This returns scipy's ``odeint`` answer.
      Asking for ``dense_output`` or ``events`` takes the stepping route
      instead, matching scipy's ``solve_ivp``.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def rhs(t, y):                       # y'' = -4 y
    ...     out = np.empty(2)
    ...     out[0] = y[1]
    ...     out[1] = -4.0 * y[0]
    ...     return out
    >>> @njit
    ... def run():
    ...     return si.solve_ivp(rhs, (0.0, 1.0), np.array([1.0, 0.0]),
    ...                         'RK45', np.array([0.0, 0.5, 1.0]))
    >>> res = run()
    >>> res.y.shape                          # (n_states, n_times)
    (2, 3)
    >>> res.y[0]
    array([ 1.        ,  0.54038693, -0.41629169])
    >>> res.success
    True

    ``dense_output=True`` adds ``res.sol``, callable on a scalar or an
    array of times:

    >>> res = si.solve_ivp(rhs, (0.0, 1.0), np.array([1.0, 0.0]),
    ...                    'RK45', None, True)
    >>> res.sol(0.5)
    array([ 0.54038693, -1.68337531])
    >>> res.sol(np.array([0.25, 0.5]))
    array([[ 0.8775798 ,  0.54038693],
           [-0.95885456, -1.68337531]])

    ``events`` is one ``@njit`` function returning one value per event.  A
    root of any of them is reported in ``res.t_events``:

    >>> @njit
    ... def hits_zero(t, y):
    ...     out = np.empty(1)
    ...     out[0] = y[0]
    ...     return out
    >>> res = si.solve_ivp(rhs, (0.0, 4.0), np.array([1.0, 0.0]), 'RK45',
    ...                    events=hits_zero)
    >>> res.t_events[0]
    array([0.78528433, 2.35559604, 3.92597172])

    ``terminal`` stops the integration at the first root, where scipy sets
    a ``.terminal`` attribute on the event function:

    >>> res = si.solve_ivp(rhs, (0.0, 4.0), np.array([1.0, 0.0]), 'RK45',
    ...                    events=hits_zero, terminal=np.array([1]))
    >>> res.t_events[0]
    array([0.78528433])
    >>> res.status
    1
    >>> res.message
    'A termination event occurred.'
    """
    for _k in options:
        if _k not in _SIVP_OPTIONS:
            raise TypeError(
                "solve_ivp() got an unexpected keyword argument '%s'" % _k)
    # Presence, not value: `min_step` in `options` is True only when the
    # caller wrote it, whatever it was set to.
    _p_min_step = 'min_step' in options
    _p_jac = 'jac' in options
    _p_lband = 'lband' in options
    _p_uband = 'uband' in options
    min_step = options.get('min_step', 0.0)
    jac = options.get('jac', None)
    lband = options.get('lband', None)
    uband = options.get('uband', None)
    if vectorized:
        # scipy's contract: the right-hand side is called as
        # `fun(t, y[:, None])` and the result ravelled. Adapted here so
        # nothing downstream sees the flag. `fun` is a plain @njit function
        # (an int/address is rejected below), so `py_func` is present.
        _vpy = getattr(fun, 'py_func', None)
        if _vpy is not None:
            fun = _devectorize(_vpy)
    # The state's dtype decides which solvers are open and which result type
    # comes back, and it is read once, here, before the method is routed.
    y0_cplx = np.iscomplexobj(y0)
    code = _method_code(method, y0_cplx)
    if y0_cplx and events is not None:
        raise ValueError(_CPLX_EVENTS_MSG)
    dense = bool(dense_output)
    if code != METHOD_LSODA:
        # scipy warns and ignores rather than refusing; see `_no_effect_msg`.
        # `mxstep` and `npoints` reach only the LSODA branch and were being
        # dropped in silence, where scipy names any option the solver will
        # not use.
        _tail = []
        if _p_jac:
            _tail.append('jac')
        if _p_lband:
            _tail.append('lband')
        if _p_uband:
            _tail.append('uband')
        _emit_rk_ignored(_p_min_step, mxstep != 0, npoints != 100,
                         ','.join(_tail))
        jac = None
        lband = None
        uband = None
    banded = lband is not None or uband is not None
    ml = 0 if lband is None else int(lband)
    mu = 0 if uband is None else int(uband)
    _jt = _jac_route(jac is not None, banded)
    jac_np = 0 if jac is None else _jac_arity(_py_func_of(jac, "the jacobian"))

    # scipy does `t0, tf = map(float, t_span)`, so every element must be
    # float-convertible on its own and a nested span is a `TypeError` there.
    _tsa = np.asarray(t_span)
    if _tsa.ndim > 1:
        raise TypeError(_SPAN_RANK_MSG)
    ts = _tsa.astype(np.float64).ravel()
    if ts.size != 2:
        raise ValueError("t_span must have exactly two elements")
    t0 = float(ts[0])
    t1 = float(ts[1])
    fstep = 0.0 if first_step is None else float(first_step)

    # The rank test comes BEFORE `ascontiguousarray`, which promotes a 0-d
    # array to 1-d and so hid a scalar `y0`: scipy raises on one and this
    # integrated a one-component system instead.
    _ydt = np.complex128 if y0_cplx else np.float64
    _y0a = np.asarray(y0, dtype=_ydt)
    if _y0a.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")
    yy = np.ascontiguousarray(_y0a)
    _check_ivp_args(t0, t1, yy, float(max_step), fstep,
                    first_step is not None, float(min_step))

    if args is None:
        aa = np.zeros(1, np.float64)
        has_args = False
    else:
        aa = np.ascontiguousarray(np.asarray(args, dtype=np.float64)).ravel()
        # An EMPTY `args` carries no parameters, so it is not a mismatch
        # against a two-argument rhs. scipy accepts `args=()` there.
        has_args = aa.size > 0

    if t_eval is None:
        te = np.zeros(0, np.float64)
    else:
        _tev = np.asarray(t_eval, dtype=np.float64)
        if _tev.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")
        te = np.ascontiguousarray(_tev)
        # scipy validates `t_eval` in `solve_ivp`, before the solver is
        # chosen, so both branches reach the same two checks.
        _check_teval(te, t0, t1)

    if code == METHOD_LSODA:
        if isinstance(fun, (bool, int, np.integer)):
            raise ValueError(_RK_ADDR_MSG)
        _py = _py_func_of(fun)
        _check_rhs_args(_py, has_args, TypeError)
        ptr = _adapter_rhs(_py, True).address
        ab = _prepend_neq(aa, yy.size)
        jptr = 0
        if jac is not None:
            jptr = _adapter_jac(_py_func_of(jac, "the jacobian"), banded,
                                True, True).address
            jj = jac(t0, yy) if jac_np == 2 else jac(t0, yy, aa)
            if banded:
                _check_jac_shape_band(jj, yy.size, ml, mu)
            else:
                _check_jac_shape_full(jj, yy.size)
        # D44 and D46.  scipy integrates the span ONCE and reports the steps
        # the solver took, so the step history serves the report as well as
        # `dense_output` and `events`, and one run's counters cover the
        # whole call.  The reporting run below is what `t_eval` used to be
        # served by; it is now reached only when the history cannot serve
        # the report: an explicit `npoints`, which asks for a uniform grid
        # scipy has no counterpart for, and a history that stopped early
        # with a `t_eval` to answer, where evaluating it would extrapolate
        # past where the solver got.
        no_grid = t_eval is None and npoints == 100
        need_hist = dense or events is not None
        from_hist = no_grid
        if need_hist or no_grid or t_eval is not None:
            d = _hist_lsoda(ptr, yy, t0, t1, rtol, atol, mxstep, ab, _jt, ml,
                            mu, jptr, max_step, fstep, min_step)
            if t_eval is not None and d[4] >= 0:
                from_hist = True
        if from_hist:
            istate = d[4]
            nfev, njev = d[5], d[6]
            if t_eval is None:
                rt, ry = _report_hist_steps(d, yy, t0, t1, istate >= 0)
            elif te.size == 0:
                rt, ry = _report_nothing(yy.size)
            else:
                rt, ry = te, _report_hist_at(d, te)
        else:
            tt = _teval_or_grid(te, t0, t1, npoints)
            r = _core_lsoda(ptr, yy, tt, rtol, atol, mxstep, max_step, ab,
                            t0, fstep, min_step, _jt, ml, mu, jptr)
            istate = r[4]
            rt, ry = r[0], r[1]
            nfev, njev = r[2], r[3]
        ok = istate >= 0
        st = 0 if ok else -1
        msg = _msg(istate)
        # scipy leaves `t_events` and `y_events` as `None` when no events
        # were given, `_ivp/ivp.py:655-656`.
        te_ev, ye_ev = None, None
        if events is not None:
            _evpy = _py_func_of(events, "the event function")
            _check_event_args(_evpy, has_args, TypeError)
            ev_fun = _ev3(_evpy)
            ng, tm, dr = _event_spec(ev_fun, t0, yy, terminal, direction, aa)
            rt, ry, te_ev, ye_ev, ev_st = _apply_events(
                ev_fun, ng, tm, dr, KIND_LSODA, d[1], d[2], _EMPTY_HIST_2,
                _EMPTY_HIST_3, d[3], code, t0, rt, ry, aa)
            if ev_st == 1 and ok:
                st, msg = 1, _MSG_EVENT
        # D43. An EMPTY `t_eval` asks for a report at no times. The span is
        # still integrated, as in scipy, and `t_events` still fills; only the
        # report is empty. `_teval_or_grid` cannot see the difference, because
        # it tests `size > 0`, so the distinction is applied here.
        if t_eval is not None and te.size == 0:
            rt, ry = _report_nothing(yy.size)
        if dense:
            sol = LsodaSolution(d[1], d[2], d[3])
            return OdeResultDense(rt, ry, sol, te_ev, ye_ev,
                                  nfev, njev, njev, st, msg, ok)
        return OdeResult(rt, ry, te_ev, ye_ev, nfev, njev, njev, st, msg, ok)

    # The step history is needed by dense_output AND by events; asking for
    # it once serves both.
    want_hist = dense or events is not None
    if isinstance(fun, (bool, int, np.integer)):
        raise ValueError(_RK_ADDR_MSG)
    _rkpy = _py_func_of(fun)
    _check_rhs_args(_rkpy, has_args, TypeError)
    rk_fun = _rk_rhs3(_rkpy)
    r = _core_rk(rk_fun, t0, t1, yy, code, rtol, atol, te,
                 float(max_step), want_hist, fstep, aa, t_eval is not None)
    ok = r[2]
    status = 0 if ok else -1
    msg = _MSG_DONE if ok else _MSG_FAIL
    rt, ry = r[0], r[1]
    # scipy leaves `t_events` and `y_events` as `None` when no events were
    # given, `_ivp/ivp.py:655-656`.
    te_ev, ye_ev = None, None
    if events is not None:
        _evpy = _py_func_of(events, "the event function")
        _check_event_args(_evpy, has_args, TypeError)
        ev_fun = _ev3(_evpy)
        ng, tm, dr = _event_spec(ev_fun, t0, yy, terminal, direction, aa)
        rt, ry, te_ev, ye_ev, ev_st = _apply_events(
            ev_fun, ng, tm, dr, KIND_RK, r[4], r[5], r[6], r[7],
            _EMPTY_HIST_3, code, t0, rt, ry, aa)
        if ev_st == 1 and ok:
            status, msg = 1, _MSG_EVENT
    if dense:
        _mk = OdeSolutionC if y0_cplx else OdeSolution
        sol = _mk(r[4], r[5], r[6], r[7], code)
        return OdeResultDense(rt, ry, sol, te_ev, ye_ev,
                              r[3], 0, 0, status, msg, ok)
    return OdeResult(rt, ry, te_ev, ye_ev, r[3], 0, 0, status, msg, ok)


@overload(solve_ivp, prefer_literal=True)
def _solve_ivp_ovl(fun, t_span, y0, method='RK45', t_eval=None,
                   dense_output=False, events=None, vectorized=False,
                   args=None, rtol=1e-3, atol=1e-6, max_step=np.inf,
                   mxstep=0, npoints=100, terminal=None, direction=None,
                   first_step=None, min_step=0.0, jac=None, lband=None,
                   uband=None):
    """Compiled body for :func:`solve_ivp`, one per distinct call shape.

    ``method`` and ``dense_output`` are resolved here, not at run time: the
    first picks which solver is compiled in, the second picks between the
    two result types.  On the ``'LSODA'`` branch a plain ``@njit`` ``fun``
    also gets its ``@cfunc`` adapter built here, once, so both branches
    present one API to the caller.

    Returning ``None`` declines the overload and surfaces as a
    ``TypingError``, which is the intended answer for a runtime ``method``
    or ``dense_output``.
    """
    has_events = not _is_none(events)
    vec = _lit_bool(vectorized)
    if vec is None:
        # A runtime `vectorized` used to compile on with the flag DROPPED, so
        # a right-hand side written to scipy's vectorised contract was handed
        # an `(n,)` array and its result was not ravelled. The flag selects
        # which adapter is built, which happens while the call compiles.
        raise TypingError(_VEC_MSG)
    _vec_py = None
    if vec:
        # `fun` never reaches the compiled body: both branches build their
        # adapter from `fun.dispatcher.py_func` at typing time and close over
        # it. So the flag is applied by substituting THAT, not the argument.
        if isinstance(fun, types.Dispatcher):
            _vec_py = _devectorize(fun.dispatcher.py_func).py_func
        # else: a non-callable `fun`, refused below when the method routes.

    m = _lit_str(method)
    if m is None:
        if isinstance(method, types.Omitted):
            m = method.value             # the default, 'RK45'
        elif isinstance(method, str):
            m = method
        else:
            # An int code reaches _method_code, which names the two array
            # -level entry points it does belong to. Anything else is a
            # runtime value, and declining gives a TypingError.
            if isinstance(method, (types.Integer, int, np.integer)):
                _method_code(0)
            return None
    # `y0`'s dtype is a property of its TYPE, so which solvers are open, which
    # state arrays the engine allocates and which result type comes back are
    # all settled while the call compiles.  Nothing here reads a value.
    _cplx = _is_complex_ty(y0)
    code = _method_code(m, _cplx)
    if _cplx and has_events:
        raise ValueError(_CPLX_EVENTS_MSG)
    _cast_y0 = _as_state_c16 if _cplx else _as_state_f8
    _mk_sol = OdeSolutionC if _cplx else OdeSolution

    dense = _lit_bool(dense_output)
    if dense is None:
        return None                      # runtime flag -> TypingError, by design
    # the step history serves dense_output AND events; ask for it once
    need_hist = dense or has_events

    no_teval = _is_none(t_eval)
    teval_given = not no_teval
    no_args = _is_none(args)
    # An `args` ARRAY carries no length in its type, so the arity contract
    # splits: what is decidable here is decided here, and the rest becomes
    # two compile-time flags whose branches fold away when they are False.
    _alen = _args_len_ty(args)
    _cb_two = False
    _cb_three = False

    # The event function's `args` adapter is built HERE, at typing time, and
    # frozen into the body, exactly as the right-hand side's is. scipy wraps
    # events the same way it wraps `fun`, so the arity contract is the same
    # in both directions.
    _ev_fun = None
    _ev_two = False
    _ev_three = False
    if has_events:
        if not isinstance(events, types.Dispatcher):
            raise TypingError(
                "events must be a plain @njit function g(t, y) or "
                "g(t, y, args) returning a float64 array")
        _evpy = events.dispatcher.py_func
        if _alen >= 0:
            _check_event_args(_evpy, _alen > 0, TypingError)
        else:
            _ev_two = _rhs_arity(_evpy) == 2
            _ev_three = _rhs_arity(_evpy) == 3
        _ev_fun = _ev3(_evpy)
    # `first_step=None` is scipy's "choose one". Resolved here rather than
    # at run time, so the compiled body carries a plain float.
    no_fstep = _is_none(first_step)

    # The Jacobian route is settled here too, so the compiled body carries a
    # plain `jt` constant. The half-bandwidths stay runtime values; only
    # whether they were GIVEN decides the route, as in scipy.
    no_jac = _is_none(jac)
    no_lb = _is_none(lband)
    no_ub = _is_none(uband)
    _banded = not (no_lb and no_ub)
    _jt = _jac_route(not no_jac, _banded)

    # On an RK method these are ignored, with one `UserWarning` per call, as
    # in scipy. The chooser knows the names when the call compiles, so the
    # message is a constant in the compiled body and only the `warnings.warn`
    # itself crosses into the interpreter. Measured cost of that crossing:
    # 1.545 us against 0.222 us for the same body without it, and it is
    # reached only when an ignored argument was actually passed.
    #
    # PRESENCE, not value, which is scipy's own test: it collects these in
    # `**options` and warns for any key the solver will not read. An OMITTED
    # default reaches a chooser as the RAW PYTHON VALUE and one written out
    # reaches it as a numba TYPE, so `isinstance(v, types.Type)` is exactly
    # "the caller passed this". Measured 2026-08-10 on all four, keyword and
    # positional: omitted gives `None` / `0.0` (`is-numba-type=False`) and
    # written out gives `none` / `float64` (`is-numba-type=True`).
    _rk_tail = []
    if code != METHOD_LSODA:
        if _passed(jac):
            _rk_tail.append('jac')
        if _passed(lband):
            _rk_tail.append('lband')
        if _passed(uband):
            _rk_tail.append('uband')
    _rk_is_rk = code != METHOD_LSODA
    _rk_min_step = _passed(min_step)
    _rk_tail_str = ','.join(_rk_tail)

    if code == METHOD_LSODA:
        if isinstance(fun, types.Integer):
            # A raw @cfunc address / integer pointer is no longer accepted.
            raise TypingError(_RK_ADDR_MSG)
        if not isinstance(fun, types.Dispatcher):
            return None
        py = _vec_py if _vec_py is not None else fun.dispatcher.py_func
        # Both arity mismatches are refused rather than dropping the
        # parameters, or reading them as zeros, silently. An array's
        # LENGTH is not a typing-time property, so an `args` array defers
        # both directions to the compiled body.
        if _alen >= 0:
            _check_rhs_args(py, _alen > 0, TypingError)
            _cb_two = False
            _cb_three = False
        else:
            _cb_two = _rhs_arity(py) == 2
            _cb_three = _rhs_arity(py) == 3
        addr = _adapter_rhs(py, True).address
        pad = True

        # The jacobian's @cfunc is built HERE, at typing time, and its address
        # frozen into the body below, so `jac` stays a plain @njit function
        # for the caller.
        jaddr = 0
        jac_np = 0
        if not no_jac:
            if not isinstance(jac, types.Dispatcher):
                raise ValueError(
                    "jac must be a plain @njit function j(t, y) or "
                    "j(t, y, args)")
            jpy = jac.dispatcher.py_func
            jac_np = _jac_arity(jpy)
            jaddr = _adapter_jac(jpy, _banded, True, pad).address
        j3 = jac_np == 3

        def impl(fun, t_span, y0, method='RK45', t_eval=None,
                 dense_output=False, events=None, vectorized=False,
                 args=None, rtol=1e-3, atol=1e-6, max_step=np.inf,
                 mxstep=0, npoints=100, terminal=None, direction=None,
                 first_step=None, min_step=0.0, jac=None, lband=None,
                 uband=None):
            t0, t1 = _span2(t_span)
            fstep = 0.0 if no_fstep else np.float64(first_step)
            yy = _cast_y0(y0)
            _check_ivp_args(t0, t1, yy, np.float64(max_step), fstep,
                            not no_fstep, np.float64(min_step))
            if no_args:
                aa = np.zeros(1, np.float64)
            else:
                aa = np.ascontiguousarray(
                    np.asarray(args).astype(np.float64)).ravel()
                if _cb_two and aa.size > 0:
                    raise TypeError(_IVP_ARGS_MSG)
                if _cb_three and aa.size == 0:
                    raise TypeError(_MISSING_ARGS_MSG)
                if _ev_two and aa.size > 0:
                    raise TypeError(_EV_ARGS_MSG)
                if _ev_three and aa.size == 0:
                    raise TypeError(_EV_MISSING_MSG)
            if no_teval:
                te = np.zeros(0, np.float64)
            else:
                te = _teval_1d(t_eval)
                _check_teval(te, t0, t1)
            if pad:
                ab = _prepend_neq(aa, yy.size)
                ptr = addr
            else:
                ab = aa
                ptr = fun
            if no_lb:
                ml = 0
            else:
                ml = np.int64(lband)
            if no_ub:
                mu = 0
            else:
                mu = np.int64(uband)
            if no_jac:
                jptr = 0
            else:
                jptr = jaddr
                if j3:
                    jj = jac(t0, yy, aa)
                else:
                    jj = jac(t0, yy)
                if _banded:
                    _check_jac_shape_band(jj, yy.size, ml, mu)
                else:
                    _check_jac_shape_full(jj, yy.size)
            # D44 and D46; see the python entry for why the history serves
            # the report.
            no_grid = (not teval_given) and npoints == 100
            from_hist = no_grid
            if need_hist or no_grid or teval_given:
                d = _hist_lsoda(ptr, yy, t0, t1, rtol, atol, mxstep, ab, _jt,
                                ml, mu, jptr, max_step, fstep, min_step)
                if teval_given and d[4] >= 0:
                    from_hist = True
            if from_hist:
                istate = d[4]
                nfev = d[5]
                njev = d[6]
                if not teval_given:
                    rt, ry = _report_hist_steps(d, yy, t0, t1, istate >= 0)
                elif te.size == 0:
                    rt, ry = _report_nothing(yy.size)
                else:
                    rt = te
                    ry = _report_hist_at(d, te)
            else:
                tt = _teval_or_grid(te, t0, t1, npoints)
                r = _core_lsoda(ptr, yy, tt, rtol, atol, mxstep, max_step,
                                ab, t0, fstep, min_step, _jt, ml, mu, jptr)
                istate = r[4]
                rt = r[0]
                ry = r[1]
                nfev = r[2]
                njev = r[3]
            ok = istate >= 0
            if ok:
                st = 0
            else:
                st = -1
            msg = _msg(istate)
            if has_events:
                ng, tm, dr = _event_spec(_ev_fun, t0, yy, terminal,
                                         direction, aa)
                rt, ry, te_ev, ye_ev, ev_st = _apply_events(
                    _ev_fun, ng, tm, dr, KIND_LSODA, d[1], d[2],
                    _EMPTY_HIST_2, _EMPTY_HIST_3, d[3], code, t0, rt, ry, aa)
                if ev_st == 1 and ok:
                    st = 1
                    msg = _MSG_EVENT
                # D43; see the python entry for why this sits after the events
                # block rather than replacing the grid.
                if teval_given and te.size == 0:
                    rt, ry = _report_nothing(yy.size)
                if dense:
                    return OdeResultDense(rt, ry,
                                          LsodaSolution(d[1], d[2], d[3]),
                                          te_ev, ye_ev, nfev, njev,
                                          njev, st, msg, ok)
                return OdeResult(rt, ry, te_ev, ye_ev, nfev, njev,
                                 njev, st, msg, ok)
            if teval_given and te.size == 0:
                rt, ry = _report_nothing(yy.size)
            if dense:
                return OdeResultDense(rt, ry,
                                      LsodaSolution(d[1], d[2], d[3]),
                                      None, None, nfev, njev,
                                      njev, st, msg, ok)
            return OdeResult(rt, ry, None, None, nfev, njev,
                             njev, st, msg, ok)
        return impl

    # `args` reaches the RK right-hand side too, as it does in scipy. A
    # two-argument `f(t, y)` is adapted here; see `_rk_rhs3`.
    if not isinstance(fun, types.Dispatcher):
        # Declining here would report a failed dispatch listing every argument
        # type and pointing at none of them. The RK engine calls the
        # right-hand side directly, so an address cannot be used at all.
        raise TypingError(_RK_ADDR_MSG)
    _rkpy = _vec_py if _vec_py is not None else fun.dispatcher.py_func
    if _alen >= 0:
        _check_rhs_args(_rkpy, _alen > 0, TypingError)
    else:
        _cb_two = _rhs_arity(_rkpy) == 2
        _cb_three = _rhs_arity(_rkpy) == 3
    rk_fun = _rk_rhs3(_rkpy)

    def impl(fun, t_span, y0, method='RK45', t_eval=None,
             dense_output=False, events=None, vectorized=False,
             args=None, rtol=1e-3, atol=1e-6, max_step=np.inf,
             mxstep=0, npoints=100, terminal=None, direction=None,
             first_step=None, min_step=0.0, jac=None, lband=None,
             uband=None):
        if _rk_is_rk:
            _rk_warn(_rk_min_step, mxstep != 0, npoints != 100,
                     _rk_tail_str)
        t0, t1 = _span2(t_span)
        fstep = 0.0 if no_fstep else np.float64(first_step)
        yy = _cast_y0(y0)
        _check_ivp_args(t0, t1, yy, np.float64(max_step), fstep,
                        not no_fstep, np.float64(min_step))
        if no_teval:
            te = np.zeros(0, np.float64)
        else:
            te = _teval_1d(t_eval)
            _check_teval(te, t0, t1)
        if no_args:
            aa = _NO_ARGS
        else:
            aa = np.ascontiguousarray(
                np.asarray(args).astype(np.float64)).ravel()
            if _cb_two and aa.size > 0:
                raise TypeError(_RK_ARGS_MSG)
            if _cb_three and aa.size == 0:
                raise TypeError(_MISSING_ARGS_MSG)
            if _ev_two and aa.size > 0:
                raise TypeError(_EV_ARGS_MSG)
            if _ev_three and aa.size == 0:
                raise TypeError(_EV_MISSING_MSG)
        r = _core_rk(rk_fun, t0, t1, yy, code, rtol,
                     atol, te, np.float64(max_step), need_hist,
                     fstep, aa, not no_teval)
        ok = r[2]
        if ok:
            st = 0
            msg = _MSG_DONE
        else:
            st = -1
            msg = _MSG_FAIL
        rt = r[0]
        ry = r[1]
        if has_events:
            ng, tm, dr = _event_spec(_ev_fun, t0, yy, terminal, direction, aa)
            rt, ry, te_ev, ye_ev, ev_st = _apply_events(
                _ev_fun, ng, tm, dr, KIND_RK, r[4], r[5], r[6], r[7],
                _EMPTY_HIST_3, code, t0, rt, ry, aa)
            if ev_st == 1 and ok:
                st = 1
                msg = _MSG_EVENT
            if dense:
                sol = _mk_sol(r[4], r[5], r[6], r[7], code)
                return OdeResultDense(rt, ry, sol, te_ev, ye_ev,
                                      r[3], 0, 0, st, msg, ok)
            return OdeResult(rt, ry, te_ev, ye_ev, r[3], 0, 0, st, msg, ok)
        if dense:
            sol = _mk_sol(r[4], r[5], r[6], r[7], code)
            return OdeResultDense(rt, ry, sol, None, None,
                                  r[3], 0, 0, st, msg, ok)
        return OdeResult(rt, ry, None, None, r[3], 0, 0,
                         st, msg, ok)
    return impl
