"""Pure-``@njit`` adaptive explicit Runge-Kutta ODE integrators.

Port of scipy's ``scipy.integrate._ivp`` explicit RK solvers (RK45, RK23,
DOP853) to numba-compilable pure Python.  These are the engine behind
``scijit.integrate.solve_ivp`` for ``method='RK45'``, ``'RK23'`` and
``'DOP853'``; the names here are the array-level spelling, taking every
argument positionally and returning a plain tuple.

Public API (all ``@njit``, callable from inside other ``@njit`` code):

    _rk_core(rhs, t0, t1, y0, method, rtol, atol, t_eval, max_step)
    RK45(rhs, t0, t1, y0, rtol, atol, t_eval, max_step)
    RK23(rhs, t0, t1, y0, rtol, atol, t_eval, max_step)
    DOP853(rhs, t0, t1, y0, rtol, atol, t_eval, max_step)
    rk_dense_eval(tt, d_t, d_h, d_y, d_C, method)

``rhs`` is a plain ``@njit`` function ``f(t, y) -> ydot`` where ``t`` is a
float scalar and ``y`` is a 1-D ``float64`` array; it must return a 1-D
``float64`` array of the same length.  Extra parameters are passed by closure
(the user writes an ``@njit`` ``f(t, y)`` that closes over them).

Nothing here holds module state, so independent integrations run correctly
under ``prange``.

Coefficients (Butcher tableaux, error estimators, dense-output polynomials)
are copied bit-for-bit from scipy so results agree to ~rtol level.
"""

import warnings

import numpy as np
from numba import njit, objmode

# ---------------------------------------------------------------------------
# Step-size control constants (identical to scipy rk.py)
# ---------------------------------------------------------------------------
SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
_EPS = np.finfo(np.float64).eps

# ---------------------------------------------------------------------------
# Method codes
# ---------------------------------------------------------------------------
METHOD_RK45 = 0
METHOD_RK23 = 1
METHOD_DOP853 = 2

# ---------------------------------------------------------------------------
# RK23 (Bogacki-Shampine 3(2)) tableau
# ---------------------------------------------------------------------------
_RK23_C = np.array([0.0, 1.0 / 2.0, 3.0 / 4.0])
_RK23_A = np.array([
    [0.0, 0.0, 0.0],
    [1.0 / 2.0, 0.0, 0.0],
    [0.0, 3.0 / 4.0, 0.0],
])
_RK23_B = np.array([2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0])
_RK23_E = np.array([5.0 / 72.0, -1.0 / 12.0, -1.0 / 9.0, 1.0 / 8.0])
_RK23_P = np.array([
    [1.0, -4.0 / 3.0, 5.0 / 9.0],
    [0.0, 1.0, -2.0 / 3.0],
    [0.0, 4.0 / 3.0, -8.0 / 9.0],
    [0.0, -1.0, 1.0],
])

# ---------------------------------------------------------------------------
# RK45 (Dormand-Prince 5(4)) tableau
# ---------------------------------------------------------------------------
_RK45_C = np.array([0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0])
_RK45_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0],
    [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0],
    [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
])
_RK45_B = np.array([35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0,
                    -2187.0 / 6784.0, 11.0 / 84.0])
_RK45_E = np.array([-71.0 / 57600.0, 0.0, 71.0 / 16695.0, -71.0 / 1920.0,
                    17253.0 / 339200.0, -22.0 / 525.0, 1.0 / 40.0])
_RK45_P = np.array([
    [1.0, -8048581381.0 / 2820520608.0, 8663915743.0 / 2820520608.0,
     -12715105075.0 / 11282082432.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 131558114200.0 / 32700410799.0, -68118460800.0 / 10900136933.0,
     87487479700.0 / 32700410799.0],
    [0.0, -1754552775.0 / 470086768.0, 14199869525.0 / 1410260304.0,
     -10690763975.0 / 1880347072.0],
    [0.0, 127303824393.0 / 49829197408.0, -318862633887.0 / 49829197408.0,
     701980252875.0 / 199316789632.0],
    [0.0, -282668133.0 / 205662961.0, 2019193451.0 / 616988883.0,
     -1453857185.0 / 822651844.0],
    [0.0, 40617522.0 / 29380423.0, -110615467.0 / 29380423.0,
     69997945.0 / 29380423.0],
])

# ---------------------------------------------------------------------------
# DOP853 (8th order) extended tableau + error estimators + dense-output D
# (copied bit-for-bit from scipy.integrate._ivp.dop853_coefficients)
# ---------------------------------------------------------------------------
_DOP_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.05260015195876773, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0197250569845379, 0.0591751709536137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.02958758547680685, 0.0, 0.08876275643042054, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2413651341592667, 0.0, -0.8845494793282861, 0.924834003261792, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.037037037037037035, 0.0, 0.0, 0.17082860872947386, 0.12546768756682242, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.037109375, 0.0, 0.0, 0.17025221101954405, 0.06021653898045596, -0.017578125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.03709200011850479, 0.0, 0.0, 0.17038392571223998, 0.10726203044637328, -0.015319437748624402, 0.008273789163814023, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.6241109587160757, 0.0, 0.0, -3.3608926294469414, -0.868219346841726, 27.59209969944671, 20.154067550477894, -43.48988418106996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.47766253643826434, 0.0, 0.0, -2.4881146199716677, -0.590290826836843, 21.230051448181193, 15.279233632882423, -33.28821096898486, -0.020331201708508627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.9371424300859873, 0.0, 0.0, 5.186372428844064, 1.0914373489967295, -8.149787010746927, -18.52006565999696, 22.739487099350505, 2.4936055526796523, -3.0467644718982196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.273310147516538, 0.0, 0.0, -10.53449546673725, -2.0008720582248625, -17.9589318631188, 27.94888452941996, -2.8589982771350235, -8.87285693353063, 12.360567175794303, 0.6433927460157636, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.054293734116568765, 0.0, 0.0, 0.0, 0.0, 4.450312892752409, 1.8915178993145003, -5.801203960010585, 0.3111643669578199, -0.1521609496625161, 0.20136540080403034, 0.04471061572777259, 0.0, 0.0, 0.0, 0.0],
    [0.056167502283047954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25350021021662483, -0.2462390374708025, -0.12419142326381637, 0.15329179827876568, 0.00820105229563469, 0.007567897660545699, -0.008298, 0.0, 0.0, 0.0],
    [0.03183464816350214, 0.0, 0.0, 0.0, 0.0, 0.028300909672366776, 0.053541988307438566, -0.05492374857139099, 0.0, 0.0, -0.00010834732869724932, 0.0003825710908356584, -0.00034046500868740456, 0.1413124436746325, 0.0, 0.0],
    [-0.42889630158379194, 0.0, 0.0, 0.0, 0.0, -4.697621415361164, 7.683421196062599, 4.06898981839711, 0.3567271874552811, 0.0, 0.0, 0.0, -0.0013990241651590145, 2.9475147891527724, -9.15095847217987, 0.0]
])
_DOP_B = np.array([0.054293734116568765, 0.0, 0.0, 0.0, 0.0, 4.450312892752409, 1.8915178993145003, -5.801203960010585, 0.3111643669578199, -0.1521609496625161, 0.20136540080403034, 0.04471061572777259])
_DOP_C = np.array([0.0, 0.05260015195876773, 0.0789002279381516, 0.1183503419072274, 0.2816496580927726, 0.3333333333333333, 0.25, 0.3076923076923077, 0.6512820512820513, 0.6, 0.8571428571428571, 1.0, 1.0, 0.1, 0.2, 0.7777777777777778])
_DOP_E3 = np.array([-0.18980075407240762, 0.0, 0.0, 0.0, 0.0, 4.450312892752409, 1.8915178993145003, -5.801203960010585, -0.4226823213237919, -0.1521609496625161, 0.20136540080403034, 0.02265179219836082, 0.0])
_DOP_E5 = np.array([0.01312004499419488, 0.0, 0.0, 0.0, 0.0, -1.2251564463762044, -0.4957589496572502, 1.6643771824549864, -0.35032884874997366, 0.3341791187130175, 0.08192320648511571, -0.022355307863886294, 0.0])
_DOP_D = np.array([
    [-8.428938276109013, 0.0, 0.0, 0.0, 0.0, 0.5667149535193777, -3.0689499459498917, 2.38466765651207, 2.117034582445028, -0.871391583777973, 2.2404374302607883, 0.6315787787694688, -0.08899033645133331, 18.148505520854727, -9.194632392478356, -4.436036387594894],
    [10.427508642579134, 0.0, 0.0, 0.0, 0.0, 242.28349177525817, 165.20045171727028, -374.5467547226902, -22.113666853125306, 7.733432668472264, -30.674084731089398, -9.332130526430229, 15.697238121770845, -31.139403219565178, -9.35292435884448, 35.81684148639408],
    [19.985053242002433, 0.0, 0.0, 0.0, 0.0, -387.0373087493518, -189.17813819516758, 527.8081592054236, -11.57390253995963, 6.8812326946963, -1.0006050966910838, 0.7777137798053443, -2.778205752353508, -60.19669523126412, 84.32040550667716, 11.99229113618279],
    [-25.69393346270375, 0.0, 0.0, 0.0, 0.0, -154.18974869023643, -231.5293791760455, 357.6391179106141, 93.40532418362432, -37.45832313645163, 104.0996495089623, 29.8402934266605, -43.53345659001114, 96.32455395918828, -39.17726167561544, -149.72683625798564]
])

_DOP_N_STAGES = 12
_DOP_N_STAGES_EXTENDED = 16
_DOP_INTERP_POWER = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#: The empty parameter buffer a two-argument right-hand side is given.
#: `args` is threaded through the engine rather than captured in a
#: closure: a closure defined inside `@njit` that captures a runtime
#: array CANNOT be passed to another `@njit` function, measured,
#: "TypingError: During: Pass make_function_op_code_to_jit_function".
_NO_ARGS = np.zeros(0, np.float64)

#: A `rtol` array whose length is neither 1 nor ``n``. scipy shape-tests
#: `atol` only, so the same input reaches its error norm and fails there in
#: numpy's broadcast, "operands could not be broadcast together with shapes
#: (2,) (3,)". The class is `ValueError` on both sides; the text carries the
#: two shapes there and names the rule here, since numba raises with a
#: compile-time constant.
_RTOL_SHAPE_MSG = ("`rtol` must be a scalar, a length-1 array, or have one "
                   "entry per component of `y0`")


@njit
def _rms_norm(x):
    """RMS norm = ||x||_2 / sqrt(size), matching scipy common.norm.

    scipy's ``common.norm`` is ``np.linalg.norm(x) / sqrt(x.size)``, which is
    built on the modulus, so ``abs(x[i])**2`` is scipy's own formula rather
    than a widening of it.  For a real ``x`` the two spellings are equal in
    IEEE arithmetic, so the real path is unchanged bit for bit.
    """
    s = 0.0
    for i in range(x.size):
        a = abs(x[i])
        s += a * a
    return np.sqrt(s / x.size)


@njit
def _rk_step(rhs, t, y, f, h, A, B, C, K, n_stages, args):
    """One explicit RK step. Fills K[0..n_stages], returns (y_new, f_new).

    K has shape (n_stages + 1, n): stages in rows, last row = f_new.

    Every state array follows ``y.dtype``, so one source serves a real and a
    complex system; numba compiles one specialisation per dtype.
    """
    n = y.size
    K[0] = f
    for s in range(1, n_stages):
        dy = np.zeros(n, y.dtype)
        for j in range(s):
            a = A[s, j]
            if a != 0.0:
                for i in range(n):
                    dy[i] += a * K[j, i]
        for i in range(n):
            dy[i] *= h
        stage = rhs(t + C[s] * h, y + dy, args)
        for i in range(n):
            K[s, i] = stage[i]

    y_new = np.empty(n, y.dtype)
    for i in range(n):
        acc = y[0] * 0
        for s in range(n_stages):
            acc += B[s] * K[s, i]
        y_new[i] = y[i] + h * acc

    f_new = rhs(t + h, y_new, args)
    for i in range(n):
        K[n_stages, i] = f_new[i]
    return y_new, f_new


@njit
def _check_teval(t_eval, t0, t1):
    """scipy's two ``t_eval`` guards, ``_ivp/ivp.py:603-613``.

    scipy runs them inside ``solve_ivp`` itself, BEFORE the solver is chosen,
    so every method gets them.  This lives beside the engine and is called
    from both the front end and :func:`_rk_core`.

    The range test is ``t_eval < min(t0, tf)`` or ``t_eval > max(t0, tf)``.
    The order test is strict, ``np.any(d <= 0)`` forward and
    ``np.any(d >= 0)`` backward, so a repeated entry is refused; it does not
    run at all when ``t1 == t0``, because both of scipy's conjunctions are
    then False.
    """
    lo = t0 if t0 < t1 else t1
    hi = t1 if t0 < t1 else t0
    direction = -1.0 if t1 < t0 else 1.0
    for k in range(t_eval.size):
        if t_eval[k] < lo or t_eval[k] > hi:
            raise ValueError("Values in `t_eval` are not within `t_span`.")
        if t1 != t0 and k > 0 and \
                direction * (t_eval[k] - t_eval[k - 1]) <= 0.0:
            raise ValueError("Values in `t_eval` are not properly sorted.")


@njit
def _select_initial_step(rhs, t0, y0, tf, max_step, f0, order, rtol, atol,
                         direction, args):
    """scipy ``common.select_initial_step``.

    Returns an UNSIGNED step size. ``direction`` is +1 or -1 and enters
    only through the trial step used to estimate the second derivative:
    the probe has to move into the interval being integrated, not out of
    it.

    ``rtol`` and ``atol`` are ``(n,)`` arrays, one entry per component, as
    scipy's are after ``validate_tol``.
    """
    n = y0.size
    if n == 0:
        return np.inf
    interval_length = abs(tf - t0)
    if interval_length == 0.0:
        return 0.0

    # `scale` is real by construction (atol + |y0| * rtol); the difference
    # vectors follow `y0`.
    scale = np.empty(n)
    d0v = np.empty(n, y0.dtype)
    d1v = np.empty(n, y0.dtype)
    for i in range(n):
        scale[i] = atol[i] + abs(y0[i]) * rtol[i]
        d0v[i] = y0[i] / scale[i]
        d1v[i] = f0[i] / scale[i]
    d0 = _rms_norm(d0v)
    d1 = _rms_norm(d1v)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    if h0 > interval_length:
        h0 = interval_length

    y1 = np.empty(n, y0.dtype)
    for i in range(n):
        y1[i] = y0[i] + h0 * direction * f0[i]
    f1 = rhs(t0 + h0 * direction, y1, args)
    d2v = np.empty(n, y0.dtype)
    for i in range(n):
        d2v[i] = (f1[i] - f0[i]) / scale[i]
    d2 = _rms_norm(d2v) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1.0))

    out = 100.0 * h0
    if h1 < out:
        out = h1
    if interval_length < out:
        out = interval_length
    if max_step < out:
        out = max_step
    return out


@njit
def _rk_error_norm(K, h, E, n_stages, scale):
    """RMS error norm for RK23/RK45 (scale-weighted)."""
    n = scale.size
    s = 0.0
    for i in range(n):
        e = K[0, i] * 0
        for st in range(n_stages + 1):
            e += E[st] * K[st, i]
        e = e * h / scale[i]
        a = abs(e)
        s += a * a
    return np.sqrt(s / n)


@njit
def _dop_error_norm(K, h, scale):
    """DOP853 dual (5th/3rd) error norm (scipy DOP853._estimate_error_norm)."""
    n = scale.size
    err5_norm_2 = 0.0
    err3_norm_2 = 0.0
    for i in range(n):
        e5 = K[0, i] * 0
        e3 = K[0, i] * 0
        for st in range(13):
            e5 += _DOP_E5[st] * K[st, i]
            e3 += _DOP_E3[st] * K[st, i]
        e5 /= scale[i]
        e3 /= scale[i]
        a5 = abs(e5)
        a3 = abs(e3)
        err5_norm_2 += a5 * a5
        err3_norm_2 += a3 * a3
    if err5_norm_2 == 0.0 and err3_norm_2 == 0.0:
        return 0.0
    denom = err5_norm_2 + 0.01 * err3_norm_2
    return abs(h) * err5_norm_2 / np.sqrt(denom * n)


#: Rows in one stored interpolation-coefficient block.  DOP853 needs all
#: seven; RK45 uses four and RK23 three, and the unused rows stay zero, so a
#: single array shape serves every method.
_INTERP_ROWS = _DOP_INTERP_POWER


@njit
def _dense_eval_block(tt, t_old, h, y_old, Cf, method):
    """Evaluate one stored coefficient block at a single time ``tt``.

    ``Cf`` is ``(_INTERP_ROWS, n)``: the DOP853 ``F`` matrix, or the
    transpose of the RK ``Q`` matrix with the trailing rows zero.
    """
    n = y_old.size
    x = (tt - t_old) / h
    out = np.empty(n, y_old.dtype)
    if method == METHOD_DOP853:
        y = np.zeros(n, y_old.dtype)
        for k in range(_DOP_INTERP_POWER):
            fidx = _DOP_INTERP_POWER - 1 - k          # reversed(F)
            for i in range(n):
                y[i] += Cf[fidx, i]
            if k % 2 == 0:
                for i in range(n):
                    y[i] *= x
            else:
                for i in range(n):
                    y[i] *= (1.0 - x)
        for i in range(n):
            out[i] = y[i] + y_old[i]
        return out

    if method == METHOD_RK45:
        ncol = 4
    else:
        ncol = 3
    p = np.empty(ncol)
    acc = x
    for m in range(ncol):
        p[m] = acc
        acc *= x
    for i in range(n):
        s = 0.0
        for m in range(ncol):
            s += Cf[m, i] * p[m]
        out[i] = y_old[i] + h * s
    return out


@njit
def rk_dense_eval(tt, d_t, d_h, d_y, d_C, method):
    """Dense output from the step history, as plain arrays.

    The array-only counterpart of ``solve_ivp(..., dense_output=True).sol``.
    Every argument and the return value is an array.

    Parameters
    ----------
    tt : float64 array, shape (k,)
        Times to evaluate at.  Outside the integrated span the nearest step's
        polynomial is extrapolated, which is what scipy's ``OdeSolution`` does.
    d_t, d_h : float64 array, shape (s,)
        Start time and signed step size of each accepted step.
    d_y : float64 array, shape (s, n)
        State at the START of each accepted step, ``y(d_t[k])``.
    d_C : float64 array, shape (s, 7, n)
        Interpolation coefficients per step.
    method : int
        ``METHOD_RK45``, ``METHOD_RK23`` or ``METHOD_DOP853``, the method the
        history was produced with.

    Returns
    -------
    y : float64 array, shape (k, n)
        Solution at each ``tt``, time-major.  ``res.sol(tt)`` returns the
        transpose of this, ``(n, k)``, which is scipy's orientation.

    Raises
    ------
    ValueError
        ``d_t`` is empty, which is what an integration run without
        ``dense_output=True`` leaves behind.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> from scijit.integrate._ivp import rk_dense_eval, METHOD_RK45
    >>> @njit
    ... def rhs(t, y):
    ...     out = np.empty(2)
    ...     out[0] = y[1]
    ...     out[1] = -4.0 * y[0]
    ...     return out
    >>> res = si.solve_ivp(rhs, (0.0, 1.0), np.array([1.0, 0.0]),
    ...                    'RK45', None, True)
    >>> @njit
    ... def at(tt, d_t, d_h, d_y, d_C):
    ...     return rk_dense_eval(tt, d_t, d_h, d_y, d_C, METHOD_RK45)
    >>> at(np.array([0.25, 0.5]), res.sol.t, res.sol.h, res.sol.ys,
    ...    res.sol.C)
    array([[ 0.8775798 , -0.95885456],
           [ 0.54038693, -1.68337531]])
    """
    s = d_t.size
    if s == 0:
        raise ValueError(
            "no dense-output history; call solve_ivp with dense_output=True")
    n = d_y.shape[1]
    out = np.empty((tt.size, n), d_y.dtype)
    for j in range(tt.size):
        # locate the step whose span contains tt[j]; the history is ordered
        k = 0
        for m in range(s):
            if d_h[m] > 0.0:
                if tt[j] >= d_t[m]:
                    k = m
            else:
                if tt[j] <= d_t[m]:
                    k = m
        yi = _dense_eval_block(tt[j], d_t[k], d_h[k], d_y[k],
                               d_C[k], method)
        for i in range(n):
            out[j, i] = yi[i]
    return out


# ---------------------------------------------------------------------------
# Main integrator
# ---------------------------------------------------------------------------
@njit
def _rk_core(rhs, t0, t1, y0, method, rtol, atol, t_eval, max_step,
             want_dense, first_step=0.0, args=_NO_ARGS, teval_given=False):
    """The adaptive explicit Runge-Kutta algorithm, computing everything.

    ``solve_ivp(method='RK45'|'RK23'|'DOP853')``
    are both slices of this.  ``want_dense`` decides whether the per-step
    interpolation coefficients are kept; when it is False the history arrays
    come back empty and nothing extra is allocated.

    Returns ``(t, y, success, nfev, d_t, d_h, d_y, d_C)``.  ``y`` is time-major
    ``(m, n)``.  ``d_t[k]``, ``d_h[k]``, ``d_y[k]`` and ``d_C[k]`` describe accepted step
    ``k``: pass them to :func:`rk_dense_eval` to evaluate between the steps.

    ``teval_given`` says that ``t_eval`` was supplied.  An empty ``t_eval``
    is not the same input as no ``t_eval``: the first reports at no times at
    all and the second reports the solver's own steps.
    """
    n = y0.size

    # ---- input guards (numba has no bounds checking) ----
    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")
    for i in range(n):
        if not np.isfinite(y0[i]):
            raise ValueError(
                "All components of the initial state `y0` must be finite.")
    if method != METHOD_RK45 and method != METHOD_RK23 and method != METHOD_DOP853:
        raise ValueError("method must be 0 (RK45), 1 (RK23) or 2 (DOP853)")
    if np.isnan(t0) or np.isnan(t1):
        raise ValueError("t0 and t1 must not be nan")
    if max_step <= 0.0:
        raise ValueError("`max_step` must be positive.")

    # ---- tolerances, scalar or one entry per component ----
    # scipy's `validate_tol` (`_ivp/common.py:44-60`) shape-tests `atol` and
    # not `rtol`, so a length-1 `atol` array is refused where n > 1 and a
    # length-1 `rtol` array broadcasts. Measured on scipy 1.18 with n = 2:
    # `atol=np.array([1e-6])` raises "`atol` has wrong shape.",
    # `rtol=np.array([1e-3])` integrates.
    rv = np.asarray(rtol).astype(np.float64).ravel()
    _av = np.asarray(atol).astype(np.float64)
    if _av.ndim > 1 or (_av.ndim == 1 and _av.size != n):
        raise ValueError("`atol` has wrong shape.")
    av = _av.ravel()
    if rv.size != 1 and rv.size != n:
        raise ValueError(_RTOL_SHAPE_MSG)

    # A too-small `rtol` is CLAMPED and warned about, not refused. scipy's
    # `validate_tol` (`_ivp/common.py:47-51`) does exactly this, and measured,
    # `rtol=0`, `rtol=-1` and `rtol=1e-20` all integrate successfully there at
    # nfev 32. This routine used to raise on `rtol <= 0` before reaching the
    # clamp two lines down, so those three inputs were refused.
    _clamped = False
    for i in range(rv.size):
        if rv[i] < 100.0 * _EPS:
            rv[i] = 100.0 * _EPS
            _clamped = True
    if _clamped:
        with objmode():
            warnings.warn(
                "At least one element of `rtol` is too small. "
                "Setting `rtol = np.maximum(rtol, 2.220446049250313e-14)`.")

    # `atol = 0` is legal: the tolerance is then purely relative. scipy
    # accepts it (measured, nfev 20 on a decaying exponential, no warning) and
    # rejects only a NEGATIVE one, which is what `validate_tol` tests,
    # `_ivp/common.py:57`.
    for i in range(av.size):
        if av[i] < 0.0:
            raise ValueError("`atol` must be positive.")

    rtolv = np.empty(n)
    atolv = np.empty(n)
    for i in range(n):
        if rv.size == 1:
            rtolv[i] = rv[0]
        else:
            rtolv[i] = rv[i]
        if av.size == 1:
            atolv[i] = av[0]
        else:
            atolv[i] = av[i]

    # rhs sanity: length must match y0
    f0 = rhs(t0, y0, args)
    nfev = 1
    if f0.size != n:
        raise ValueError("rhs(t, y) must return an array the same length as y0")

    # ---- select tableau ----
    if method == METHOD_RK45:
        A = _RK45_A
        B = _RK45_B
        C = _RK45_C
        E = _RK45_E
        P = _RK45_P
        n_stages = 6
        err_est_order = 4
    elif method == METHOD_RK23:
        A = _RK23_A
        B = _RK23_B
        C = _RK23_C
        E = _RK23_E
        P = _RK23_P
        n_stages = 3
        err_est_order = 2
    else:  # DOP853
        A = _DOP_A          # 16x16 (only first 12x12 used for stepping)
        B = _DOP_B          # 12
        C = _DOP_C          # 16 (only first 12 used for stepping)
        E = _RK45_E         # dummy (unused for DOP853)
        P = _RK45_P         # dummy (unused for DOP853)
        n_stages = 12
        err_est_order = 7

    error_exponent = -1.0 / (err_est_order + 1.0)

    # Sign of the integration. scipy's `direction`; t1 == t0 counts as
    # forward, and the step loop then terminates before its first step.
    if t1 < t0:
        direction = -1.0
    else:
        direction = 1.0

    use_teval = teval_given or t_eval.size > 0
    if use_teval:
        _check_teval(t_eval, t0, t1)

    # ---- no components to advance ----
    # scipy integrates a shape-(0,) `y0` and reports one jump straight to
    # `t_span[1]`. Measured on 1.18, `solve_ivp(f, (0, 1), np.empty(0))`
    # returns `t = [0., 1.]`, `y.shape = (0, 2)`, `nfev = 1`, `status = 0` on
    # RK45, RK23 and DOP853 alike. The step loop cannot produce that, because
    # the error norm over no components is `0.0 / 0.0`.
    if n == 0:
        if want_dense:
            z_t = np.full(1, t0)
            z_h = np.full(1, t1 - t0 if t1 != t0 else 1.0)
            z_y = np.zeros((1, 0), y0.dtype)
            z_C = np.zeros((1, _INTERP_ROWS, 0), y0.dtype)
        else:
            z_t = np.zeros(0)
            z_h = np.zeros(0)
            z_y = np.zeros((0, 0), y0.dtype)
            z_C = np.zeros((0, _INTERP_ROWS, 0), y0.dtype)
        if use_teval:
            zt = np.empty(t_eval.size)
            for k in range(t_eval.size):
                zt[k] = t_eval[k]
            return (zt, np.zeros((t_eval.size, 0), y0.dtype), True, nfev,
                    z_t, z_h, z_y, z_C)
        zt = np.empty(2)
        zt[0] = t0
        zt[1] = t1
        return (zt, np.zeros((2, 0), y0.dtype), True, nfev,
                z_t, z_h, z_y, z_C)

    # ---- output storage ----
    if use_teval:
        m_out = t_eval.size
        out_t = np.empty(m_out)
        for k in range(m_out):
            out_t[k] = t_eval[k]
        out_y = np.empty((m_out, n), y0.dtype)
        eval_idx = 0
    else:
        cap = 64
        out_t = np.empty(cap)
        out_y = np.empty((cap, n), y0.dtype)
        out_t[0] = t0
        for i in range(n):
            out_y[0, i] = y0[i]
        count = 1

    # ---- degenerate span ----
    # t1 == t0 admits no step, and the step loop below would exit before
    # writing anything, so the initial state is emitted here instead. Every
    # t_eval entry has already been validated to equal t0.
    if t1 == t0:
        # A constant interpolant covers the zero-length span: every
        # coefficient is zero, so `_dense_eval_block` returns y_old whatever
        # time it is handed, and `h` is 1.0 only to keep its division finite.
        # scipy builds the same thing, as `ConstantDenseOutput`.
        if want_dense:
            g_t = np.full(1, t0)
            g_h = np.ones(1)
            g_y = np.empty((1, n), y0.dtype)
            for i in range(n):
                g_y[0, i] = y0[i]
            g_C = np.zeros((1, _INTERP_ROWS, n), y0.dtype)
        else:
            g_t = np.zeros(0)
            g_h = np.zeros(0)
            g_y = np.zeros((0, n), y0.dtype)
            g_C = np.zeros((0, _INTERP_ROWS, n), y0.dtype)
        if use_teval:
            # scipy reports NOTHING here. `tf > t0` is False for a zero-length
            # span, so `solve_ivp` takes the reversed-`t_eval` branch and
            # seeds `t_eval_i` past the end; every slice it then takes is
            # empty and `elif ts:` never converts the lists. Measured:
            # `solve_ivp(f, (0, 0), y0, t_eval=[0.0])` returns `res.t == []`
            # and `res.y == []`.
            return (np.zeros(0), np.zeros((0, n), y0.dtype), True, nfev,
                    g_t, g_h, g_y, g_C)
        # Without t_eval scipy reports the point TWICE: it seeds the output
        # with t0, then appends the state the solver finished on, which is
        # t0 again. Reporting it once read better and diverged, so the
        # duplicate is deliberate.
        out_t[1] = t0
        for i in range(n):
            out_y[1, i] = y0[i]
        return (out_t[:2].copy(), out_y[:2].copy(), True, nfev,
                g_t, g_h, g_y, g_C)

    # ---- solver state ----
    t = t0
    # The state follows `y0`'s dtype rather than being forced real, which is
    # what makes a complex system reach the same stepper.  `f0` is cast to it,
    # so a real-valued right-hand side over a complex state still fits.
    y = y0.copy()
    f = f0.astype(y0.dtype)

    if first_step > 0.0:
        # scipy's `first_step`: use it instead of choosing one. It is still
        # clipped by max_step below, as a chosen step would be.
        h_abs = first_step
    else:
        h_abs = _select_initial_step(rhs, t0, y, t1, max_step, f,
                                     err_est_order, rtolv, atolv, direction,
                                     args)
        nfev += 1

    # working stage storage: extended for DOP853 dense output
    K = np.empty((n_stages + 1, n), y0.dtype)
    if method == METHOD_DOP853:
        K_ext = np.empty((_DOP_N_STAGES_EXTENDED, n), y0.dtype)
    else:
        K_ext = np.empty((1, 1), y0.dtype)  # placeholder (unused)

    success = True

    # ---- dense-output history ----------------------------------------------
    # One (INTERP_ROWS, n) coefficient block per ACCEPTED step, plus the
    # (t_old, h) it is anchored on.  Sized 0 when not wanted, which keeps the
    # allocation off the default path while leaving the TYPE unchanged.
    if want_dense:
        dcap = 64
    else:
        dcap = 0
    d_t = np.empty(dcap)
    d_h = np.empty(dcap)
    d_y = np.empty((dcap, n), y0.dtype)
    d_C = np.empty((dcap, _INTERP_ROWS, n), y0.dtype)
    d_count = 0

    while True:
        if direction * (t - t1) >= 0.0:
            break

        # min_step (scipy uses 10*ulp toward the direction of travel)
        nxt = np.nextafter(t, direction * np.inf)
        min_step = 10.0 * abs(nxt - t)

        if h_abs > max_step:
            h_abs = max_step
        elif h_abs < min_step:
            h_abs = min_step

        step_accepted = False
        step_rejected = False
        t_new = t
        y_new = y
        f_new = f
        h = h_abs

        while not step_accepted:
            if h_abs < min_step:
                success = False
                break

            h = h_abs * direction
            t_new = t + h
            if direction * (t_new - t1) > 0.0:
                t_new = t1
            h = t_new - t
            h_abs = abs(h)

            y_new, f_new = _rk_step(rhs, t, y, f, h, A, B, C, K, n_stages, args)
            nfev += n_stages

            scale = np.empty(n)
            for i in range(n):
                ay = abs(y[i])
                ayn = abs(y_new[i])
                m = ay if ay > ayn else ayn
                scale[i] = atolv[i] + m * rtolv[i]

            if method == METHOD_DOP853:
                error_norm = _dop_error_norm(K, h, scale)
            else:
                error_norm = _rk_error_norm(K, h, E, n_stages, scale)

            if error_norm < 1.0:
                if error_norm == 0.0:
                    factor = MAX_FACTOR
                else:
                    factor = SAFETY * error_norm ** error_exponent
                    if factor > MAX_FACTOR:
                        factor = MAX_FACTOR
                if step_rejected and factor > 1.0:
                    factor = 1.0
                h_abs *= factor
                step_accepted = True
            else:
                factor = SAFETY * error_norm ** error_exponent
                if factor < MIN_FACTOR:
                    factor = MIN_FACTOR
                h_abs *= factor
                step_rejected = True

        if not success:
            break

        t_old = t
        y_old = y

        # ---- emit output for this accepted step ----
        # The interpolation coefficients are built when a t_eval point falls
        # in (t_old, t_new], and additionally on EVERY step when the caller
        # asked to keep the history.
        need_dense = want_dense
        if use_teval and eval_idx < m_out and \
                direction * (t_eval[eval_idx] - t_new) <= 0.0:
            need_dense = True
        Cf = np.zeros((_INTERP_ROWS, n), y0.dtype)
        if need_dense:
            if method == METHOD_DOP853:
                # build extended stages, then F, into the first
                # _DOP_INTERP_POWER rows of Cf
                for i in range(n):
                    for st in range(n_stages + 1):
                        K_ext[st, i] = K[st, i]
                for s in range(n_stages + 1, _DOP_N_STAGES_EXTENDED):
                    dy = np.zeros(n, y0.dtype)
                    for j in range(s):
                        a = _DOP_A[s, j]
                        if a != 0.0:
                            for i in range(n):
                                dy[i] += a * K_ext[j, i]
                    for i in range(n):
                        dy[i] *= h
                    stage = rhs(t_old + _DOP_C[s] * h, y_old + dy, args)
                    for i in range(n):
                        K_ext[s, i] = stage[i]
                nfev += _DOP_N_STAGES_EXTENDED - (n_stages + 1)
                for i in range(n):
                    delta = y_new[i] - y_old[i]
                    f_old_i = K_ext[0, i]
                    Cf[0, i] = delta
                    Cf[1, i] = h * f_old_i - delta
                    Cf[2, i] = 2.0 * delta - h * (f_new[i] + f_old_i)
                for jj in range(3, _DOP_INTERP_POWER):
                    drow = jj - 3
                    for i in range(n):
                        s = 0.0
                        for mm in range(_DOP_N_STAGES_EXTENDED):
                            s += _DOP_D[drow, mm] * K_ext[mm, i]
                        Cf[jj, i] = h * s
            else:
                # RK dense: Q = K.T @ P, stored TRANSPOSED as (ncol, n) so
                # both methods share one history array shape
                ncol = P.shape[1]
                for i in range(n):
                    for mcol in range(ncol):
                        s = 0.0
                        for st in range(n_stages + 1):
                            s += K[st, i] * P[st, mcol]
                        Cf[mcol, i] = s

        if use_teval:
            while eval_idx < m_out and \
                    direction * (t_eval[eval_idx] - t_new) <= 0.0:
                yi = _dense_eval_block(t_eval[eval_idx], t_old, h, y_old,
                                       Cf, method)
                for i in range(n):
                    out_y[eval_idx, i] = yi[i]
                eval_idx += 1
        else:
            if count >= out_t.shape[0]:
                newcap = out_t.shape[0] * 2
                nt = np.empty(newcap)
                ny = np.empty((newcap, n), y0.dtype)
                for k in range(count):
                    nt[k] = out_t[k]
                    for i in range(n):
                        ny[k, i] = out_y[k, i]
                out_t = nt
                out_y = ny
            out_t[count] = t_new
            for i in range(n):
                out_y[count, i] = y_new[i]
            count += 1

        if want_dense:
            if d_count >= d_t.shape[0]:
                ncap = d_t.shape[0] * 2
                nt = np.empty(ncap)
                nh = np.empty(ncap)
                ny = np.empty((ncap, n), y0.dtype)
                nC = np.empty((ncap, _INTERP_ROWS, n), y0.dtype)
                for k in range(d_count):
                    nt[k] = d_t[k]
                    nh[k] = d_h[k]
                    for i in range(n):
                        ny[k, i] = d_y[k, i]
                    for r in range(_INTERP_ROWS):
                        for i in range(n):
                            nC[k, r, i] = d_C[k, r, i]
                d_t = nt
                d_h = nh
                d_y = ny
                d_C = nC
            d_t[d_count] = t_old
            d_h[d_count] = h
            for i in range(n):
                d_y[d_count, i] = y_old[i]
            for r in range(_INTERP_ROWS):
                for i in range(n):
                    d_C[d_count, r, i] = Cf[r, i]
            d_count += 1

        # advance
        t = t_new
        y = y_new
        f = f_new

    # ---- assemble return ----
    d_t = d_t[:d_count].copy()
    d_h = d_h[:d_count].copy()
    d_y = d_y[:d_count].copy()
    d_C = d_C[:d_count].copy()

    if use_teval:
        if eval_idx == m_out:
            return out_t, out_y, success, nfev, d_t, d_h, d_y, d_C
        # truncate to covered points (e.g. on failure)
        rt = np.empty(eval_idx)
        ry = np.empty((eval_idx, n), y0.dtype)
        for k in range(eval_idx):
            rt[k] = out_t[k]
            for i in range(n):
                ry[k, i] = out_y[k, i]
        return rt, ry, success, nfev, d_t, d_h, d_y, d_C
    else:
        rt = np.empty(count)
        ry = np.empty((count, n), y0.dtype)
        for k in range(count):
            rt[k] = out_t[k]
            for i in range(n):
                ry[k, i] = out_y[k, i]
        return rt, ry, success, nfev, d_t, d_h, d_y, d_C
