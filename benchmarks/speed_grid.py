"""Speed benchmark: scijit (@njit) vs scipy (Python loop).

Mimics the workload scijit was built for (the AGN accretion-disk
origin story): a GRID of cells, each evolved over several timesteps, and
at every cell/step the expensive per-point work is

  1. solve a 3-variable nonlinear system   -> MINPACK  (optimize)
  2. look up an opacity from a bivariate spline table -> FITPACK (interpolate)
  3. integrate a small profile              -> simpson (integrate)

scijit runs the WHOLE grid loop inside ONE @njit function (the
solvers/lookups are called directly, no Python boundary per step). scipy
does the identical computation in an idiomatic Python loop. Same problem,
same tolerance, same result, the only difference is the interpreter /
per-call overhead the JIT removes.

Run:  PYTHONPATH=. python benchmarks/speed_grid.py
"""
import math
import time
import numpy as np
from numba import njit

KFREQ = 120       # frequency bins summed per residual eval (heavy inner loop)

import scipy.optimize
import scipy.interpolate
import scipy.integrate

from scijit.optimize import fsolve
from scijit.interpolate import RectBivariateSpline
from scijit.integrate import simpson

# ------------------------------------------------ problem setup (shared)
N = 1500          # grid cells
M = 12            # timesteps per cell   ->  N*M = 18000 nonlinear solves
NP = 21           # profile samples per simpson call (odd)

rng = np.random.default_rng(0)
# per-cell targets for the 3-var system; the residual sums KFREQ
# "frequency bins" (a smooth series), the root sits near (1,1,1) with
# the base constants below, perturbed per cell so each solve is genuinely
# different.
_ACC0 = sum(math.exp(-0.15 * k) * math.sin(0.1 * k)
            for k in range(1, KFREQ + 1))       # series value at (1,1,1)
A0 = (2.0 + 0.02 * _ACC0) + 0.2 * rng.standard_normal(N)
B0 = 1.0 + 0.2 * rng.standard_normal(N)
C0 = 2.0 + 0.2 * rng.standard_normal(N)

# opacity table kappa(T, rho) on a grid, fit as a bivariate spline
Tg = np.linspace(0.5, 4.0, 40)
Rg = np.linspace(0.5, 4.0, 40)
KAP = (np.exp(-0.3 * (Tg[:, None] - 2.0) ** 2 - 0.3 * (Rg[None, :] - 2.0) ** 2)
       + 0.5).copy()

sci_spl = scipy.interpolate.RectBivariateSpline(Tg, Rg, KAP)
nsi_spl = RectBivariateSpline(Tg, Rg, KAP)          # jitclass

PROF_X = np.linspace(0.0, np.pi, NP)
TLO, THI = Tg[0], Tg[-1]
RLO, RHI = Rg[0], Rg[-1]


@njit(inline='always')
def _clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ------------------------------------------------ the 3-var residual
# Every residual evaluation sums KFREQ "frequency bins" (a loop), this
# is the coupled-physics cost. MINPACK calls the residual ~15x per solve,
# so scipy pays (Python call + Python loop) hundreds of thousands of
# times, while scijit's @njit residual is compiled into the solver.

# scipy side: a Python closure over (a,b,c), with the inner loop
def F_py(x, a, b, c):
    acc = 0.0
    for k in range(1, KFREQ + 1):
        acc += math.exp(-0.15 * k * x[0]) * math.sin(0.1 * k * x[1])
    return [x[0] ** 2 + x[1] - a + 0.02 * acc,
            x[0] - x[1] ** 2 + x[2] - b,
            x[1] + x[2] ** 2 - c]


# scijit side: the identical residual as a plain @njit function. fsolve
# builds the @cfunc callback at typing time, so the function itself is
# passed, not an address.
@njit
def F_nb(x, a, b, c):
    acc = 0.0
    for k in range(1, KFREQ + 1):
        acc += math.exp(-0.15 * k * x[0]) * math.sin(0.1 * k * x[1])
    return np.array([x[0] ** 2 + x[1] - a + 0.02 * acc,
                     x[0] - x[1] ** 2 + x[2] - b,
                     x[1] + x[2] ** 2 - c])


# ------------------------------------------------ scipy driver (Python loop)
def run_scipy():
    total = 0.0
    for i in range(N):
        a, b, c = A0[i], B0[i], C0[i]
        x = np.array([1.0, 1.0, 1.0])
        for _ in range(M):
            x = scipy.optimize.fsolve(F_py, x, args=(a, b, c))     # 3-var root
            T = _clamp(x[0], TLO, THI)
            R = _clamp(x[1], RLO, RHI)
            # opacity + a 5-point stencil for its gradient (grid physics)
            # scipy 1.18's RectBivariateSpline.ev returns a 1-D array even for
            # scalar arguments, so float() raises; .item() works on both.
            kap = sci_spl.ev(T, R).item()
            gx = sci_spl.ev(_clamp(T + 0.05, TLO, THI), R).item() - \
                sci_spl.ev(_clamp(T - 0.05, TLO, THI), R).item()
            gy = sci_spl.ev(T, _clamp(R + 0.05, RLO, RHI)).item() - \
                sci_spl.ev(T, _clamp(R - 0.05, RLO, RHI)).item()
            grad = (gx * gx + gy * gy) ** 0.5
            prof = np.sin(PROF_X * kap)
            I = scipy.integrate.simpson(prof, dx=PROF_X[1] - PROF_X[0])
            total += I
            a += 0.01 * kap
            b += 0.005 * I + 0.003 * grad
            c += 0.008 * I + 0.004 * grad
    return total


# ------------------------------------------------ scijit driver (@njit)
@njit(cache=False)
def run_numba(a0, b0, c0, spl, prof_x):
    total = 0.0
    n = a0.shape[0]
    dx = prof_x[1] - prof_x[0]
    x0 = np.empty(3)
    for i in range(n):
        a = a0[i]; b = b0[i]; c = c0[i]
        x0[0] = 1.0; x0[1] = 1.0; x0[2] = 1.0
        for _ in range(M):
            # pass the @njit residual; fsolve builds the callback and
            # returns x directly
            x = fsolve(F_nb, x0, (a, b, c))
            T = _clamp(x[0], TLO, THI)
            R = _clamp(x[1], RLO, RHI)
            # opacity + a 5-point stencil for its gradient (grid physics)
            kap = spl.ev_one(T, R)
            gx = (spl.ev_one(_clamp(T + 0.05, TLO, THI), R)
                  - spl.ev_one(_clamp(T - 0.05, TLO, THI), R))
            gy = (spl.ev_one(T, _clamp(R + 0.05, RLO, RHI))
                  - spl.ev_one(T, _clamp(R - 0.05, RLO, RHI)))
            grad = (gx * gx + gy * gy) ** 0.5
            prof = np.sin(prof_x * kap)
            I = simpson(prof, dx=dx)                     # integrate
            total += I
            a += 0.01 * kap
            b += 0.005 * I + 0.003 * grad
            c += 0.008 * I + 0.004 * grad
            x0[0] = x[0]; x0[1] = x[1]; x0[2] = x[2]
    return total


if __name__ == "__main__":
    print(f"grid: N={N} cells x M={M} steps = {N*M} nonlinear 3-var solves")
    print("per step: MINPACK root (wrap) + bivariate spline (wrap) + "
          "simpson (port)\n")

    # warm up the JIT (compile), excluded from timing
    _ = run_numba(A0[:2].copy(), B0[:2].copy(), C0[:2].copy(), nsi_spl, PROF_X)

    t0 = time.perf_counter()
    r_nb = run_numba(A0.copy(), B0.copy(), C0.copy(), nsi_spl, PROF_X)
    t_nb = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_sp = run_scipy()
    t_sp = time.perf_counter() - t0

    print(f"scipy   (Python loop): {t_sp:8.3f} s   result={r_sp:.6f}")
    print(f"scijit (@njit)   : {t_nb:8.3f} s   result={r_nb:.6f}")
    print(f"agreement |diff|     : {abs(r_nb - r_sp):.3e}")
    print(f"\nSPEEDUP: {t_sp / t_nb:.1f}x   "
          f"({1e6*t_sp/(N*M):.1f} us/solve scipy vs "
          f"{1e6*t_nb/(N*M):.1f} us/solve scijit)")
