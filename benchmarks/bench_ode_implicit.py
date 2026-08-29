"""Benchmark: a nonlinear solve inside EVERY RHS eval of an adaptive ODE.

The hardest nesting there is, and a real pattern (differential-algebraic
systems / equilibrium chemistry / equation-of-state coupling): integrate
a stiff ODE whose right-hand side is only defined *implicitly*, at every
RHS evaluation you must solve a nonlinear algebraic system for auxiliary
variables z(y), then form dy/dt from z.

Nesting depth per ODE:  adaptive steps  x  RHS evals/step  x  Newton
iterations  x  residual evals. Times an ENSEMBLE of independent ODEs.

scijit runs the whole ensemble in ONE @njit driver: the right-hand side is
a plain @njit function that itself calls `fsolve` (MINPACK), and `odeint`
and `fsolve` each build their own callback internally, so the nest is
compiled end to end with no Python boundary anywhere in it. scipy does the
identical thing with a Python RHS calling `scipy.optimize.fsolve`, the
classic pattern that makes stiff DAE-style integration crawl in Python.

Run:  PYTHONPATH=. python benchmarks/bench_ode_implicit.py
"""
import math
import time
import numpy as np
from numba import njit

import scipy.integrate
import scipy.optimize

from scijit.integrate import odeint
from scijit.optimize import fsolve

D = 2             # ODE state dimension
Z = 3             # inner algebraic system dimension
E = 400           # ensemble: independent ODE systems
KFREQ = 300       # spectral opacity groups summed per residual eval (heavy loop)
T_EVAL = np.linspace(0.0, 4.0, 40)

rng = np.random.default_rng(0)
Y0 = 1.0 + 0.3 * rng.standard_normal((E, D))     # per-system initial state


# ---- inner algebraic system  H(z; y) = 0, contractive (converges fast)
# with a KFREQ-term "spectral opacity" sum in the first equation so every
# residual evaluation carries a real inner loop.
def inner_py(z, y):
    acc = 0.0
    for k in range(1, KFREQ + 1):
        acc += math.sin(k * z[0]) / (k * k)
    return [z[0] - 1.0 - 0.1 * math.sin(y[0]) - 0.05 * z[1] - 0.02 * acc,
            z[1] - 1.0 - 0.1 * math.sin(y[1]) - 0.05 * z[2],
            z[2] - 1.0 - 0.05 * z[0]]


@njit
def inner_nb(z, y):
    acc = 0.0
    for k in range(1, KFREQ + 1):
        acc += math.sin(k * z[0]) / (k * k)
    return np.array([z[0] - 1.0 - 0.1 * math.sin(y[0]) - 0.05 * z[1] - 0.02 * acc,
                     z[1] - 1.0 - 0.1 * math.sin(y[1]) - 0.05 * z[2],
                     z[2] - 1.0 - 0.05 * z[0]])


# ---- ODE right-hand side  dy/dt = g(y, z(y)) --------------------------
def rhs_py(y, t):
    z = scipy.optimize.fsolve(inner_py, np.array([1.0, 1.0, 1.0]), args=(y,))
    return [-0.10 * z[0] * y[0],
            -0.10 * z[1] * y[1] + 0.05 * z[2]]


@njit
def rhs_nb(y, t):
    z0 = np.array([1.0, 1.0, 1.0])
    yv = np.array([y[0], y[1]])                  # state -> fsolve args
    z = fsolve(inner_nb, z0, (yv,))              # inner nonlinear solve
    dy = np.empty(D)
    dy[0] = -0.10 * z[0] * y[0]
    dy[1] = -0.10 * z[1] * y[1] + 0.05 * z[2]
    return dy


# ---- drivers ----------------------------------------------------------
@njit
def run_numba(y0all, t_eval):
    total = 0.0
    for e in range(y0all.shape[0]):
        y0 = y0all[e].copy()
        usol = odeint(rhs_nb, y0, t_eval, ())
        total += usol[-1, 0] + usol[-1, 1]
    return total


def run_scipy(y0all, t_eval):
    total = 0.0
    for e in range(y0all.shape[0]):
        sol = scipy.integrate.odeint(rhs_py, y0all[e], t_eval)
        total += sol[-1, 0] + sol[-1, 1]
    return total


if __name__ == "__main__":
    print(f"ensemble: {E} stiff ODEs x {len(T_EVAL)} output pts")
    print("each RHS eval solves a 3-var nonlinear system (fsolve/MINPACK) "
          "inside LSODA's adaptive steps\n")

    _ = run_numba(Y0[:2].copy(), T_EVAL)         # warm up JIT (compile)

    t0 = time.perf_counter()
    r_nb = run_numba(Y0.copy(), T_EVAL)
    t_nb = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_sp = run_scipy(Y0.copy(), T_EVAL)
    t_sp = time.perf_counter() - t0

    print(f"scipy   (Python RHS + fsolve): {t_sp:8.3f} s   result={r_sp:.6f}")
    print(f"scijit (@njit, nested)   : {t_nb:8.3f} s   result={r_nb:.6f}")
    print(f"agreement |diff|             : {abs(r_nb - r_sp):.3e}")
    print(f"\nSPEEDUP: {t_sp / t_nb:.1f}x")
