"""Benchmark: where scijit gives NO advantage (the honest other side).

scijit helps when a big Python LOOP of small heavy calls can be
compiled away (see speed_grid.py, bench_pde_front.py). For a SINGLE
large call made from ordinary Python there is no loop to remove, and the
shipped subpackages (interpolate / integrate / optimize) wrap the SAME
Fortran scipy uses, so the two tie. From Python, reach for scipy.

The only reason to reach for scijit is a call INSIDE @njit code, where
scipy cannot be called at all.

Run:  PYTHONPATH=. python benchmarks/bench_where_scipy_wins.py
"""
import time
import numpy as np

import scipy.interpolate

from scijit.interpolate import RectBivariateSpline


def _time(fn, *a, repeat=7):
    fn(*a)                                   # warm (JIT compile excluded)
    best = 1e30
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t0)
    return best


rng = np.random.default_rng(0)

# a smooth surface sampled on a grid, and a large set of query points
Tg = np.linspace(0.0, 10.0, 200)
Rg = np.linspace(0.0, 10.0, 200)
Z = np.exp(-0.1 * (Tg[:, None] - 5.0) ** 2 - 0.1 * (Rg[None, :] - 5.0) ** 2)
Z = np.ascontiguousarray(Z)

qx = np.ascontiguousarray(rng.uniform(0.0, 10.0, 200_000))
qy = np.ascontiguousarray(rng.uniform(0.0, 10.0, 200_000))

print("=== single large bivariate spline: scipy vs scijit (same FITPACK) ===\n")

# 1) construction (fit). scipy 1.18 runs a C translation of FITPACK; scijit
#    calls the Fortran directly.
t_sci = _time(lambda: scipy.interpolate.RectBivariateSpline(Tg, Rg, Z))
t_nsp = _time(lambda: RectBivariateSpline(Tg, Rg, Z))
print(f"construct (200x200):  scipy {1e3*t_sci:7.2f} ms   "
      f"scijit {1e3*t_nsp:7.2f} ms   ratio {t_nsp/t_sci:.2f}x")

# 2) evaluation at 200k points, one vectorized call each.
sci_spl = scipy.interpolate.RectBivariateSpline(Tg, Rg, Z)
nsi_spl = RectBivariateSpline(Tg, Rg, Z)
t_sci = _time(lambda: sci_spl.ev(qx, qy))
t_nsp = _time(lambda: nsi_spl.ev(qx, qy))
match = np.max(np.abs(sci_spl.ev(qx, qy) - nsi_spl.ev(qx, qy)))
print(f"evaluate (200k pts):  scipy {1e3*t_sci:7.2f} ms   "
      f"scijit {1e3*t_nsp:7.2f} ms   ratio {t_nsp/t_sci:.2f}x   "
      f"(max|diff|={match:.2e})")

print("\nTakeaway: one large call from Python is a wash (identical FITPACK "
      "underneath). scijit exists so the SAME call runs INSIDE @njit code, "
      "where scipy cannot be called at all.")
