"""scijit: scipy-equivalent numerical routines callable inside numba
@njit code, backed by the same Fortran libraries scipy itself wraps.

Subpackages
    interpolate     FITPACK (Dierckx) splines, all 31 public routines
                    plus scipy.interpolate-equivalent jitclasses;
                    results bit-for-bit identical to scipy.interpolate.

    optimize        MINPACK + L-BFGS-B + SLSQP + PRIMA, minimize
                    (unified, bound/constrained), root/fsolve/leastsq +
                    direct MINPACK drivers with the NumbaMinpack-
                    compatible cfunc API, and the Powell derivative-free
                    family (uobyqa/newuoa/bobyqa/lincoa/cobyla).

    integrate       QUADPACK + ODEPACK. quad over seven routes,
                    nquad/dblquad/tplquad over a nest of it, and
                    solve_ivp/odeint (LSODA/LSODAR).

scijit.optimize's least-squares path uses one vendored internal library,
scijit/_lib/liblapackref (full Reference-LAPACK + BLAS), so the
package needs no system BLAS/LAPACK, only gfortran at install time.
Each subpackage wraps one Fortran pack: sources in src/<pack>/ with a
bind(c) wrappers.f90, compiled at install into a shared library.

Originally inspired by Nicholas Wogan's NumbaMinpack (MIT), which
pioneered calling a compiled Fortran pack from inside @njit code:
    https://github.com/Nicholaswogan/NumbaMinpack
FITPACK is by Paul Dierckx (Curve and Surface Fitting with Splines, 1993).
"""

import os 
import sys

# On Windows (Python 3.8+), register all package directories containing .dll files
# so Windows can find sibling shared libraries (like liblapackref.dll)
if sys.platform == 'win32':
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for _root, _dirs, _files in os.walk(_pkg_dir):
        if any(_f.endswith('.dll') for _f in _files):
            try:
                os.add_dll_directory(_root)
            except OSError:
                pass

from . import interpolate
from . import optimize
from . import integrate

__version__ = '0.1.2'
