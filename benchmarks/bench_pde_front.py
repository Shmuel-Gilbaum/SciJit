"""Benchmark: a stiff 1-D PDE front, with fsolve inside EVERY adaptive step.

This is the regime scijit was actually built for (the AGN accretion-disk
1-D PDE with steep gradients): a method-of-lines discretization of a viscous
Burgers equation with a stiff, *implicit* source term, integrated by an
adaptive-step solver (LSODA). At every right-hand-side evaluation, at every
interior grid point, we

  1. solve a small nonlinear algebraic system   -> fsolve (MINPACK, wrapped)
     for the local auxiliary/equilibrium variables z(u_i);
  2. evaluate a pure-Python "opacity series" of KFREQ terms  -> the kind of
     custom per-point physics that has NO vectorized scipy equivalent and so
     runs at interpreter speed in plain Python.

    u_t + u u_x = nu u_xx + S(u, z(u));   z solves H(z; u_i) = 0 per point.

Nesting per PDE solve:  adaptive steps x RHS evals/step x grid points x
(Newton iters x residual evals  +  KFREQ-term series). A steep initial front
forces the adaptive stepper to take many tiny substeps right where the gradient
is sharp -- exactly where interpreter overhead is multiplied hardest.

scijit runs the WHOLE thing in one @njit driver: the LSODA right-hand side is
a plain @njit function that itself calls fsolve per point and sums the series,
and `odeint` and `fsolve` each build their own callback internally, so the nest
is compiled end to end with no Python boundary in it. scipy does the identical
math with a Python RHS calling scipy.optimize.fsolve per point (same LSODA,
same MINPACK underneath), so the ONLY difference is the interpreter / per-call
overhead.

The sweep over KFREQ (per-point pure-Python work) shows the crux: the fsolve
core is the same Fortran at C speed in both, so with KFREQ=0 the ratio is just
the glue/loop overhead removed. As KFREQ grows, scipy pays interpreter cost per
term while scijit compiles it away -- and the speedup climbs. Your real
run, with cheap-but-countless per-point work and an exploding adaptive step
count in the steep region, lives at the high end of this curve.

Run:  PYTHONPATH=. python benchmarks/bench_pde_front.py
"""
import math
import time
import numpy as np
from numba import njit

import scipy.integrate
import scipy.optimize

from scijit.integrate import odeint
from scijit.optimize import fsolve

# ---------------------------------------------------------------- geometry
N = 90                                  # grid points
XL, XR = 0.0, 1.0
X = np.linspace(XL, XR, N)
DX = (XR - XL) / (N - 1)
INV2DX = 1.0 / (2.0 * DX)
INVDX2 = 1.0 / (DX * DX)
NU = 0.02                               # viscosity
Z = 2                                   # inner algebraic system dimension

DELTA = 0.03                            # front width (smaller = steeper)
U0 = 0.5 * (1.0 - np.tanh((X - 0.3) / DELTA))     # steep front initial state
T_EVAL = np.linspace(0.0, 0.15, 4)

KFREQ_SWEEP = [0, 10, 40, 120]         # per-point pure-Python work to sweep


# ---------------- inner algebraic system  H(z; u_i) = 0 (contractive) -----
# The local equilibrium couples to a KFREQ-term "opacity series" in z0 -- so
# the series lives INSIDE the residual, where the root-finder evaluates it
# ~15x per solve. In scipy that residual is a Python call summing KFREQ terms
# hundreds of thousands of times; in scijit the compiled callback absorbs it.
def inner_py(z, ui, kf):
    acc = 0.0
    for k in range(1, kf + 1):
        acc += math.exp(-0.1 * k * abs(ui)) * math.cos(0.2 * k * z[0]) / k
    return [z[0] - 1.0 - 0.2 * ui - 0.1 * z[1] - 0.01 * acc,
            z[1] - 0.5 * math.sin(z[0])]


@njit
def inner_nb(z, ui, kf):
    acc = 0.0
    for k in range(1, kf + 1):
        acc += math.exp(-0.1 * k * abs(ui)) * math.cos(0.2 * k * z[0]) / k
    return np.array([z[0] - 1.0 - 0.2 * ui - 0.1 * z[1] - 0.01 * acc,
                     z[1] - 0.5 * math.sin(z[0])])


# ---------------- method-of-lines RHS  du/dt = -u u_x + nu u_xx + S --------
def rhs_py(u, t, kf):
    yd = np.zeros(N)
    for i in range(1, N - 1):
        ui = u[i]
        ux = (u[i + 1] - u[i - 1]) * INV2DX
        uxx = (u[i + 1] - 2.0 * ui + u[i - 1]) * INVDX2
        z = scipy.optimize.fsolve(inner_py, [1.0, 0.5], args=(ui, kf))
        S = -0.5 * z[0] * (ui - 0.2)
        yd[i] = -ui * ux + NU * uxx + S
    return yd


@njit
def rhs_nb(u, t, kf):
    yd = np.zeros(N)
    for i in range(1, N - 1):
        ui = u[i]
        ux = (u[i + 1] - u[i - 1]) * INV2DX
        uxx = (u[i + 1] - 2.0 * ui + u[i - 1]) * INVDX2
        z0 = np.array([1.0, 0.5])
        z = fsolve(inner_nb, z0, (ui, kf))         # inner solve, per point
        S = -0.5 * z[0] * (ui - 0.2)
        yd[i] = -ui * ux + NU * uxx + S
    return yd


# ---------------- drivers -------------------------------------------------
@njit
def run_numba(u0, t_eval, kf):
    usol = odeint(rhs_nb, u0, t_eval, (kf,))
    return usol[-1].copy()


def run_scipy(u0, t_eval, kf):
    sol = scipy.integrate.odeint(rhs_py, u0, t_eval, args=(kf,))
    return sol[-1].copy()


if __name__ == "__main__":
    print(f"1-D viscous Burgers front, N={N} grid pts, method of lines + LSODA")
    print(f"steep front width DELTA={DELTA}; each RHS eval solves a {Z}-var "
          f"fsolve at every interior point + a KFREQ-term series\n")

    _ = run_numba(U0.copy(), T_EVAL, 1)            # warm up JIT (compile)

    print(f"{'KFREQ':>6} {'scipy (s)':>11} {'scijit (s)':>15} "
          f"{'speedup':>9} {'|diff|':>10}")
    print("-" * 56)
    for kf in KFREQ_SWEEP:
        t0 = time.perf_counter()
        r_nb = run_numba(U0.copy(), T_EVAL, kf)
        t_nb = time.perf_counter() - t0

        t0 = time.perf_counter()
        r_sp = run_scipy(U0.copy(), T_EVAL, kf)
        t_sp = time.perf_counter() - t0

        diff = np.max(np.abs(r_nb - r_sp))
        print(f"{kf:>6} {t_sp:>11.3f} {t_nb:>15.3f} "
              f"{t_sp / t_nb:>8.1f}x {diff:>10.2e}")

    print("\nKFREQ=0: fsolve/LSODA are the same Fortran in both, so the ratio is")
    print("purely the Python glue+loop overhead scijit removes. Adding per-point")
    print("custom physics (the series has no vectorized scipy form) that scipy must")
    print("run in the interpreter pushes the ratio up; it then saturates -- the")
    print("transcendental terms are real FLOPs BOTH must do, numba just untaxed.")
    print("This ratio is roughly INDEPENDENT of problem size: more gridpoints or")
    print("timesteps multiply both sides about equally. What sets it is the MIX of")
    print("per-iteration work (interpreted glue vs compiled numerics), not the scale.")
    print("A real problem can land somewhat above or below depending on that mix.")
