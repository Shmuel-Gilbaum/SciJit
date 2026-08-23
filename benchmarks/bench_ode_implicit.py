"""Benchmark: a nonlinear solve inside EVERY RHS eval of an adaptive ODE.

The hardest nesting there is, and a real pattern (differential-algebraic
systems / equilibrium chemistry / equation-of-state coupling): integrate
a stiff ODE whose right-hand side is only defined *implicitly*, at every
RHS evaluation you must solve a nonlinear algebraic system for auxiliary
variables z(y), then form dy/dt from z.

Nesting depth per ODE:  adaptive steps  x  RHS evals/step  x  Newton
iterations  x  residual evals. Times an ENSEMBLE of independent ODEs.

scijit runs the whole ensemble in ONE @njit driver: the LSODA
integrator's RHS is a @cfunc that itself calls `fsolve` (MINPACK), all
compiled, no Python boundary anywhere in the nest. scipy does the
identical thing with a Python RHS calling `scipy.optimize.fsolve`, the
classic pattern that makes stiff DAE-style integration crawl in Python.

Run:  PYTHONPATH=. python benchmarks/bench_ode_implicit.py
"""
import math
import time
import numpy as np
from numba import njit, cfunc, carray

import scipy.integrate
import scipy.optimize

from scijit.integrate import odeint, lsoda_sig
from scijit.optimize import fsolve, minpack_sig

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


@cfunc(minpack_sig)
def inner_c(z_ptr, f_ptr, args_ptr):
    z = carray(z_ptr, Z)
    f = carray(f_ptr, Z)
    y = carray(args_ptr, D)
    acc = 0.0
    for k in range(1, KFREQ + 1):
        acc += math.sin(k * z[0]) / (k * k)
    f[0] = z[0] - 1.0 - 0.1 * math.sin(y[0]) - 0.05 * z[1] - 0.02 * acc
    f[1] = z[1] - 1.0 - 0.1 * math.sin(y[1]) - 0.05 * z[2]
    f[2] = z[2] - 1.0 - 0.05 * z[0]


_INNER = inner_c.address


# ---- ODE right-hand side  dy/dt = g(y, z(y)) --------------------------
def rhs_py(y, t):
    z = scipy.optimize.fsolve(inner_py, np.array([1.0, 1.0, 1.0]), args=(y,))
    return [-0.10 * z[0] * y[0],
            -0.10 * z[1] * y[1] + 0.05 * z[2]]


@cfunc(lsoda_sig)
def rhs_c(t, y_ptr, yd_ptr, args_ptr):
    y = carray(y_ptr, D)
    yd = carray(yd_ptr, D)
    z0 = np.array([1.0, 1.0, 1.0])
    yv = np.array([y[0], y[1]])                  # state -> fsolve args
    # scipy-shaped `fsolve` returns x directly (the legacy 4-tuple entry
    # was deleted 2026-07-27); a raw .address is still accepted.
    z = fsolve(_INNER, z0, args=yv)              # inner nonlinear solve
    yd[0] = -0.10 * z[0] * y[0]
    yd[1] = -0.10 * z[1] * y[1] + 0.05 * z[2]


_RHS = rhs_c.address


# ---- drivers ----------------------------------------------------------
@njit(cache=False)
def run_numba(y0all, t_eval):
    total = 0.0
    args = np.zeros(1)
    for e in range(y0all.shape[0]):
        y0 = y0all[e].copy()
        usol, ok = odeint(_RHS, y0, t_eval, args)
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
