"""Event detection for ``solve_ivp``: roots of ``g(t, y) = 0``.

Both engines produce a per-step history of the continuous solution, the
Runge-Kutta interpolation coefficients on one side and LSODA's Nordsieck
array on the other, so one scan serves both.  A step is examined at its two
ends, and a sign change there is resolved by root-finding on that step's
own polynomial, which is what scipy does.

Nothing here calls the solver.  It runs on a finished history, which is why
it needs no restart loop for a non-terminal event.
"""
import numpy as np
from numba import njit

from ._ivp import rk_dense_eval
from ._odepack import lsoda_dense_eval

#: History kinds, so one scan can read either engine's arrays.
KIND_RK = 0
KIND_LSODA = 1

#: scipy's root-finding tolerance in ``solve_event_equation``:
#: ``brentq(..., xtol=4 * EPS, rtol=4 * EPS)``.
_XTOL = 4.0 * np.finfo(np.float64).eps


@njit
def _eval_dense(tt, kind, d_t, d_h, d_y, d_C, d_yh, method):
    """The continuous solution at one time, from either engine's history.

    Returns ``(n,)``.  ``d_y``/``d_C`` carry the Runge-Kutta history and
    ``d_yh`` the Nordsieck one; the unused pair is a zero-sized array, so
    both are present in the signature and only one is read.
    """
    one = np.empty(1, np.float64)
    one[0] = tt
    if kind == KIND_RK:
        return rk_dense_eval(one, d_t, d_h, d_y, d_C, method)[0]
    return lsoda_dense_eval(one, d_t, d_h, d_yh)[0]


@njit
def _g_at(g, tt, kind, d_t, d_h, d_y, d_C, d_yh, method, args):
    """The event vector at one time, evaluated on the continuous solution."""
    return g(tt, _eval_dense(tt, kind, d_t, d_h, d_y, d_C, d_yh, method), args)


#: ``scipy.optimize.brentq`` runs at ``disp=True`` inside
#: ``solve_event_equation``, so exhausting its iteration budget raises rather
#: than returning the current iterate.  Measured on scipy 1.18:
#: ``brentq(lambda x: x - 0.3, 0.0, 1.0, maxiter=1)`` raises
#: ``RuntimeError: Failed to converge after 1 iterations.``
_MAXITER_EVENT = 100
_NOCONV_MSG = "Failed to converge after 100 iterations."


@njit
def _brentq_event(g, idx, ta, tb, ga, gb, kind, d_t, d_h, d_y, d_C, d_yh,
                  method, args):
    """Brent's method on ``g[idx](t, y(t))`` over a bracketing ``[ta, tb]``.

    Written here rather than calling ``scijit.optimize.brentq`` because the
    function to solve is the event composed with the dense evaluation, and
    that needs the whole step history.  ``brentq`` takes its parameters as
    one flat float64 buffer, and a numba closure cannot capture a runtime
    value, so the history has no way to reach it.  Same algorithm, and
    scipy's own ``xtol``.
    """
    a, b = ta, tb
    fa, fb = ga, gb
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    c, fc = a, fa
    d = b - a
    e = d
    for _ in range(_MAXITER_EVENT):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = b - a
            e = d
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2.0 * _XTOL * abs(b) + 0.5 * _XTOL
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return b
        if abs(e) < tol or abs(fa) <= abs(fb):
            d = m                       # bisect
            e = d
        else:
            s = fb / fa
            if a == c:                  # secant
                p = 2.0 * m * s
                q = 1.0 - s
            else:                       # inverse quadratic
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = m
                e = d
        a, fa = b, fb
        if abs(d) > tol:
            b += d
        elif m > 0.0:
            b += tol
        else:
            b -= tol
        fb = _g_at(g, b, kind, d_t, d_h, d_y, d_C, d_yh, method, args)[idx]
    raise RuntimeError(_NOCONV_MSG)


@njit
def _crossed(ga, gb, direction):
    """Whether ``g`` crosses zero between two samples.

    ``find_active_events`` is ``up = (g <= 0) & (g_new >= 0)`` and
    ``down = (g >= 0) & (g_new <= 0)``, non-strict at both ends, then
    ``direction`` selects: positive keeps upward crossings, negative
    downward, zero keeps either.

    Two consequences of the non-strict far end.  A root landing exactly on a
    step boundary is reported once per adjoining step.  An event that is
    identically zero is active on every step.
    """
    up = ga <= 0.0 and gb >= 0.0
    down = ga >= 0.0 and gb <= 0.0
    if direction > 0.0:
        return up
    if direction < 0.0:
        return down
    return up or down


@njit
def scan_events(g, ng, terminal, direction, kind, d_t, d_h, d_y, d_C, d_yh,
                method, t0, args):
    """Find every root of every event over a finished step history.

    Parameters
    ----------
    g : @njit function ``g(t, y, args) -> float64 array (ng,)``
        The event functions, evaluated together.
    ng : int
        How many values ``g`` returns.
    terminal : int64 array, shape (ng,)
        ``n`` stops the integration at that event's ``n``-th occurrence and
        0 never stops.
    direction : float64 array, shape (ng,)
        Positive keeps upward crossings only, negative downward only, zero
        keeps both.
    kind : int
        ``KIND_RK`` or ``KIND_LSODA``, which history to read.
    d_t, d_h, d_y, d_C, d_yh : float64 arrays
        The step history.  The pair the other engine uses is zero-sized.
    method : int
        The Runge-Kutta method code; ignored for ``KIND_LSODA``.
    t0 : float
        Start of the integration.  The first stored step ends at the first
        accepted step's time, so ``t0`` is where the first bracket opens.
    args : float64 array, shape (p,)
        Extra parameters passed to ``g`` on every evaluation.

    Returns
    -------
    t_roots : float64 array, shape (k,)
        Every root found, in the order encountered.
    idx : int64 array, shape (k,)
        Which event produced each root.
    t_stop : float
        Time of the terminating root, or ``nan`` when none fired.
    """
    s = d_t.size
    cap = 16
    t_roots = np.empty(cap, np.float64)
    idx = np.empty(cap, np.int64)
    nfound = 0
    t_stop = np.nan
    #: Occurrences so far, one counter per event.  scipy's ``event_count``,
    #: incremented once per step in which the event is active, and compared
    #: against ``max_events``.
    count = np.zeros(ng, np.int64)
    step_r = np.empty(ng, np.float64)
    step_i = np.empty(ng, np.int64)

    if s == 0:
        return t_roots[:0].copy(), idx[:0].copy(), t_stop

    # Step k spans (prev, d_t[k]].  For the RK history d_t[k] is the step's
    # START, so its span is [d_t[k], d_t[k] + d_h[k]]; for LSODA d_t[k] is
    # the time reached, so the span opens at the previous entry.
    for k in range(s):
        if kind == KIND_RK:
            ta = d_t[k]
            tb = d_t[k] + d_h[k]
        else:
            ta = t0 if k == 0 else d_t[k - 1]
            tb = d_t[k]
        if ta == tb:
            continue
        gv_a = _g_at(g, ta, kind, d_t, d_h, d_y, d_C, d_yh, method, args)
        gv_b = _g_at(g, tb, kind, d_t, d_h, d_y, d_C, d_yh, method, args)
        na = 0
        for i in range(ng):
            if not _crossed(gv_a[i], gv_b[i], direction[i]):
                continue
            step_r[na] = _brentq_event(g, i, ta, tb, gv_a[i], gv_b[i], kind,
                                       d_t, d_h, d_y, d_C, d_yh, method, args)
            step_i[na] = i
            na += 1
        if na == 0:
            continue

        # Every event active in this step occurs once, whether or not the
        # step is later cut short at an earlier root.  scipy increments the
        # whole `active_events` slice before `handle_events` truncates it.
        for q in range(na):
            count[step_i[q]] += 1

        # Within one step the roots are ordered by TIME, not by event index:
        # scipy sorts with `argsort(roots)` forward and `argsort(-roots)`
        # backward, so two terminal events crossing in the same step stop at
        # the earlier one.  Insertion sort, since ng is small.
        forward = tb > ta
        for a in range(1, na):
            rv = step_r[a]
            iv = step_i[a]
            b = a - 1
            while b >= 0 and ((step_r[b] > rv) if forward
                              else (step_r[b] < rv)):
                step_r[b + 1] = step_r[b]
                step_i[b + 1] = step_i[b]
                b -= 1
            step_r[b + 1] = rv
            step_i[b + 1] = iv

        # The first root, in that order, whose event has reached its cap.
        stop_q = -1
        for q in range(na):
            i = step_i[q]
            if terminal[i] != 0 and count[i] >= terminal[i]:
                stop_q = q
                break
        last = na - 1 if stop_q < 0 else stop_q

        for q in range(last + 1):
            if nfound >= cap:
                cap *= 2
                nt = np.empty(cap, np.float64)
                ni = np.empty(cap, np.int64)
                for w in range(nfound):
                    nt[w] = t_roots[w]
                    ni[w] = idx[w]
                t_roots = nt
                idx = ni
            t_roots[nfound] = step_r[q]
            idx[nfound] = step_i[q]
            nfound += 1

        if stop_q >= 0:
            t_stop = step_r[stop_q]
            break

    return t_roots[:nfound].copy(), idx[:nfound].copy(), t_stop


@njit
def split_events(t_roots, idx, ng, t_stop, kind, d_t, d_h, d_y, d_C, d_yh,
                 method, n):
    """Group the roots per event into ``t_events`` and ``y_events`` lists.

    Roots after a terminal one are dropped, since the integration is taken
    to have stopped there.

    Parameters
    ----------
    t_roots : float64 array, shape (k,)
        Every root found, as returned by :func:`scan_events`.
    idx : int64 array, shape (k,)
        Which event produced each root.
    ng : int
        Number of events.
    t_stop : float
        Time of the terminating root, or ``nan`` when none fired.
    kind : int
        ``KIND_RK`` or ``KIND_LSODA``, which history to read.
    d_t, d_h, d_y, d_C, d_yh : float64 arrays
        The step history.  The pair the other engine uses is zero-sized.
    method : int
        The Runge-Kutta method code; ignored for ``KIND_LSODA``.
    n : int
        Number of state components.

    Returns
    -------
    t_events : list of float64 arrays, one per event
        Root times of each event.
    y_events : list of float64 arrays, ``(k_i, n)`` per event
        State at each root.
    """
    keep = np.ones(t_roots.size, np.bool_)
    if not np.isnan(t_stop):
        for q in range(t_roots.size):
            if abs(t_roots[q] - t_stop) > 0.0 and _after(t_roots[q], t_stop,
                                                         d_t, kind):
                keep[q] = False

    t_events = [np.empty(0, np.float64) for _ in range(ng)]
    y_events = [np.empty((0, n), np.float64) for _ in range(ng)]
    for i in range(ng):
        c = 0
        for q in range(t_roots.size):
            if idx[q] == i and keep[q]:
                c += 1
        te = np.empty(c, np.float64)
        ye = np.empty((c, n), np.float64)
        j = 0
        for q in range(t_roots.size):
            if idx[q] == i and keep[q]:
                te[j] = t_roots[q]
                yv = _eval_dense(t_roots[q], kind, d_t, d_h, d_y, d_C,
                                 d_yh, method)
                for m in range(n):
                    ye[j, m] = yv[m]
                j += 1
        t_events[i] = te
        y_events[i] = ye
    return t_events, y_events


@njit
def _after(tr, t_stop, d_t, kind):
    """Whether ``tr`` lies past ``t_stop`` in the direction of travel."""
    if d_t.size < 2:
        forward = True
    else:
        forward = d_t[d_t.size - 1] >= d_t[0]
    if forward:
        return tr > t_stop
    return tr < t_stop


@njit
def truncate_at(t, y, t_stop, kind, d_t, d_h, d_y, d_C, d_yh, method):
    """Cut the reported solution at a terminal event and end on it.

    The whole span is integrated first; this trims the report to the times
    before ``t_stop`` and appends the event itself as the final point.

    Parameters
    ----------
    t : float64 array, shape (m,)
        Reported times.
    y : float64 array, shape (n, m)
        Reported states, component-major.
    t_stop : float
        Time of the terminating root.
    kind : int
        ``KIND_RK`` or ``KIND_LSODA``, which history to read.
    d_t, d_h, d_y, d_C, d_yh : float64 arrays
        The step history.  The pair the other engine uses is zero-sized.
    method : int
        The Runge-Kutta method code; ignored for ``KIND_LSODA``.

    Returns
    -------
    t : float64 array, shape (keep + 1,)
        Times up to and including ``t_stop``.
    y : float64 array, shape (n, keep + 1)
        States at those times, component-major.
    """
    n = y.shape[0]
    m = t.size
    if d_t.size < 2:
        forward = True
    else:
        forward = d_t[d_t.size - 1] >= d_t[0]

    keep = 0
    for q in range(m):
        if forward:
            if t[q] < t_stop:
                keep += 1
        else:
            if t[q] > t_stop:
                keep += 1

    tt = np.empty(keep + 1, np.float64)
    yy = np.empty((n, keep + 1), np.float64)
    for q in range(keep):
        tt[q] = t[q]
        for i in range(n):
            yy[i, q] = y[i, q]
    tt[keep] = t_stop
    yv = _eval_dense(t_stop, kind, d_t, d_h, d_y, d_C, d_yh, method)
    for i in range(n):
        yy[i, keep] = yv[i]
    return tt, yy
