"""scijit.interpolate: scipy.interpolate, callable inside numba @njit code.

Two layers reach the same FITPACK (Dierckx) Fortran scipy wraps. A third is
written in numba directly and calls no Fortran at all, which is what makes it
safe to use inside a parallel loop.

Every class here is reached through a FACTORY: a small ``@njit`` function
carrying scipy's name and scipy's defaults, which builds and returns the
compiled class. The factory exists because a compiled class cannot supply its
own default arguments, so calling ``CubicSpline(x, y)`` without it would
require every argument to be written out.

Evaluating in a loop
    A bivariate spline object costs 693 ns per call and 315 ns per point when
    the query points arrive together, measured on a 200x500 table. Where a
    loop cannot batch, ``bispeu`` and ``bispev`` take the knot and coefficient
    arrays directly at 498 ns, and those arrays have to be bound to locals
    above the loop. Full measurement in docs/usage/interpolate.md, under
    "Evaluation cost per call".

scipy names, backed by FITPACK
    splrep, splprep, splev, splint, sproot, spalde, bisplev
    UnivariateSpline, InterpolatedUnivariateSpline, LSQUnivariateSpline,
    RectBivariateSpline, SmoothBivariateSpline,
    RectSphereBivariateSpline, SmoothSphereBivariateSpline

scipy names, pure @njit (no .so, so prange-safe)
    interp1d, CubicSpline, PchipInterpolator, Akima1DInterpolator
    BSpline, make_interp_spline, splder, splantider
    RegularGridInterpolator, interpn

Names scipy does not have
    splder_ev    splev(x, tck, der=nu) as its own name, since this package's
                 splev takes no `der` argument
    bispev, bispeu, evaluators, fitters   the raw FITPACK layer below

Low-level submodules (raw FITPACK argument conventions)
    evaluators   splev, splder, splint, spalde, sproot, fourco, insert,
                 curev, cualde, parder, pardeu, dblint, profil, surev,
                 bispev, bispeu
    fitters      curfit, percur, parcur, clocur, concur, cocosp, concon,
                 regrid, parsur, pogrid, spgrid, sphere, surfit, polar,
                 evapol

Where the code lives
    interpolate.py   the FITPACK-backed tck functions and the seven spline
                     classes above
    _interp1d.py _cubic.py _bspline.py _rgi.py   written in numba directly
    _ndaxis.py       shared helpers for an N-D `y` with an `axis`
"""
from . import evaluators
from . import fitters
from .evaluators import bispev, bispeu
from .interpolate import (
    splrep, splprep, splev, splder_ev, splint, sproot, spalde, bisplev,
    UnivariateSpline, InterpolatedUnivariateSpline, LSQUnivariateSpline,
    RectBivariateSpline, SmoothBivariateSpline,
    RectSphereBivariateSpline, SmoothSphereBivariateSpline,
)
# written in numba directly (prange-safe; no FITPACK .so)
from ._rgi import RegularGridInterpolator, interpn
from ._cubic import CubicSpline, PchipInterpolator
from ._interp1d import interp1d
from ._bspline import (
    BSpline, make_interp_spline, Akima1DInterpolator, splder, splantider,
)
# The flat names are free here because the raw FITPACK routines of the same
# name stay in `evaluators`, exactly as scipy keeps its own in a private
# module.

# The flat surface mirrors scipy.interpolate. The raw FITPACK layer is NOT
# flattened into it: scipy keeps its own equivalent private (`_fitpack`,
# `dfitpack`), and 26 Fortran-convention names would crowd the namespace for
# no gain. Nothing is hidden, though -- they stay reachable through the two
# submodules, which are exported here as part of the public API:
#     scijit.interpolate.evaluators.splder(...)
#     scijit.interpolate.fitters.surfit(...)
__all__ = [
    # submodules holding the raw FITPACK layer (argument-for-argument Dierckx)
    'evaluators', 'fitters',
    # tck-level functions (scipy.interpolate names)
    'splrep', 'splprep', 'splev', 'splder_ev', 'splint', 'sproot', 'spalde',
    'bisplev', 'bispev', 'bispeu',
    # spline classes. The scipy name is an @njit FACTORY returning the
    # jitclass instance, so its defaults work from Python and inside @njit
    # alike; the jitclass itself is the underscore-prefixed private name.
    'UnivariateSpline', 'InterpolatedUnivariateSpline', 'LSQUnivariateSpline',
    'RectBivariateSpline', 'SmoothBivariateSpline',
    'RectSphereBivariateSpline', 'SmoothSphereBivariateSpline',
    # written in numba directly (prange-safe, no FITPACK .so)
    'RegularGridInterpolator', 'interpn',
    'CubicSpline', 'PchipInterpolator',
    'interp1d',
    'BSpline', 'make_interp_spline', 'Akima1DInterpolator',
    'splder', 'splantider',
]
