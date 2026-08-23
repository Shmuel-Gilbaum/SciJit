"""Private VENDORED copy of scijit.stats._qmc for scijit.optimize.

`differential_evolution(init='sobol'|'halton')` needs `halton`/`sobol`, so this
pure-@njit copy lets the public package ship optimize without stats. FUTURE
WORK: when scijit.stats ships publicly, re-import from `..stats._qmc` and delete
this file.

scijit.stats._qmc -- quasi-Monte Carlo samplers ported to pure @njit code.

scipy counterpart: :mod:`scipy.stats.qmc`.  Everything here is a plain
``@njit`` function -- no compiled library, no callbacks, no module state --
so it runs inside ``@njit`` kernels and ``prange`` loops.

API map (scipy -> scijit)
-------------------------
=========================================  ==============================
``qmc.Sobol(d, scramble=...).random(n)``   ``sobol(d, n, scramble, seed)``
``qmc.Halton(d, scramble=...).random(n)``  ``halton(d, n, scramble, seed)``
``qmc.LatinHypercube(d).random(n)``        ``latin_hypercube(d, n, seed)``
``qmc.discrepancy(s, method="CD")``        ``discrepancy(s, 0)``
``qmc.scale(s, l, u)``                     ``scale(s, l, u)``
=========================================  ==============================

scipy's stateful ``QMCEngine`` classes (``.random()`` continuing where the
last call stopped, ``.fast_forward``, ``.reset``) collapse to stateless
functions that always start the sequence at index 0: ``sobol(d, n)`` is
``Sobol(d, scramble=False).random(n)`` on a *fresh* engine.

Agreement with scipy
--------------------
* ``sobol(d, n, scramble=False)``   -- **bit-identical** to
  ``scipy.stats.qmc.Sobol(d, scramble=False).random(n)``.  Same Joe-Kuo
  primitive polynomials / initial direction numbers, same Bratley-Fox
  recursion, same 30-bit fixed point, same Gray-code draw.
* ``halton(d, n, scramble=False)``  -- **bit-identical** to
  ``scipy.stats.qmc.Halton(d, scramble=False).random(n)``.
* ``discrepancy(s, 0|1|2)`` (CD / WD / MD) -- **bit-identical** to
  ``scipy.stats.qmc.discrepancy``.
* ``discrepancy(s, 3)`` (L2-star)   -- agrees to a few ulp *of the
  cancellation-amplified sum*: max relative difference 4.9e-14 measured
  over Sobol' and uniform-random samples with (n, d) in {(20,2), (64,3),
  (50,5)}, not bit-identical.  The three terms of the L2-star formula are
  individually O(10-100x) the final answer and cancel, so a last-bit
  difference in the summation order shows up magnified.  CD/WD/MD have no
  such cancellation.
* ``scale``                         -- **bit-identical** (same
  ``sample * (u - l) + l`` expression order).
* Anything randomized -- ``scramble=True`` for sobol/halton, and
  ``latin_hypercube`` unconditionally -- uses this module's own splitmix64
  generator, **not** ``numpy.random.Generator``.  It is reproducible from
  the integer ``seed`` but will **never** reproduce scipy's numbers.  It is
  a statistically valid scramble, not a port of scipy's stream.

Deviations forced by numba
---------------------------------------------------
* ``discrepancy`` takes ``method`` as an INT CODE, not a string:
      0 = "CD"       centered L2 discrepancy   (scipy default)
      1 = "WD"       wrap-around L2 discrepancy
      2 = "MD"       mixture L2 discrepancy
      3 = "L2-star"  star L2 discrepancy
* ``seed`` is an explicit integer (numba cannot hold a
  ``numpy.random.Generator``).  There is no "seed from OS entropy" mode:
  the same seed always gives the same sample.
* Sobol' direction numbers are EMBEDDED in this file (see
  ``SOBOL_MAXDIM``) rather than loaded from scipy's ``.npz`` at import.
  ``d > SOBOL_MAXDIM`` raises ValueError; scipy's own limit is 21201, and
  embedding all of it would add ~2 MB of literals.
* The Sobol' generator is fixed at 30 bits (scipy's default), so at most
  ``2**30`` points; ``n`` above that raises ValueError.
* No ``optimization=`` post-processing ("random-cd" / "lloyd"), no
  ``strength=2`` orthogonal-array LHS, no ``PoissonDisk``,
  ``MultinomialQMC``, ``MultivariateNormalQMC``, ``QMCEngine`` subclassing.
* Bad input raises ValueError (numba @njit has NO object-mode fallback and
  NO bounds checking, so every entry point validates explicitly).

Reentrancy: no callbacks and no module state -> **prange-safe**.  Distinct
``seed`` values give independent streams, so a prange loop may safely draw
a different randomized sample per iteration.

Verification (re-measured, not inherited): ``sobol`` unscrambled is
EXACTLY 0.0 max abs difference from scipy for (d, n) in {(1,16), (3,64),
(10,128), (50,64), (200,32)}; ``halton`` EXACTLY 0.0 for (1,20), (5,100),
(15,64); ``van_der_corput`` EXACTLY 0.0 for bases 2, 3, 7;
``discrepancy`` CD/WD/MD and ``scale`` (both directions) EXACTLY 0.0.

References
----------
I. M. Sobol'.  "The distribution of points in a cube and the approximate
evaluation of integrals."  USSR Comput. Math. Math. Phys., 1967.
P. Bratley & B. L. Fox.  "Algorithm 659: Implementing Sobol's quasirandom
sequence generator."  ACM Trans. Math. Softw. 14(1):88-100, 1988.
S. Joe & F. Y. Kuo.  "Constructing Sobol sequences with better
two-dimensional projections."  SIAM J. Sci. Comput. 30:2635-2654, 2008.
F. J. Hickernell.  "A generalized discrepancy and quadrature error bound."
Math. Comp. 67:299-322, 1998.
"""
import numpy as np
from numba import njit

__all__ = [
    "sobol", "halton", "latin_hypercube", "discrepancy", "scale",
    "van_der_corput", "primes_from_2_to", "SOBOL_MAXDIM", "SOBOL_BITS",
]

# ----------------------------------------------------------------------
# Sobol' direction numbers (Joe & Kuo, as shipped by scipy in
# scipy/stats/_sobol_direction_numbers.npz), truncated to the first
# SOBOL_MAXDIM dimensions and embedded as literals.
#
# _POLY_RAW[i]  : the i-th primitive polynomial, bit-packed.  Its degree is
#                 m_i = bit_length(_POLY_RAW[i]) - 1.
# _VINIT_RAW    : the initial direction numbers, flattened -- exactly m_i
#                 values for dimension i, concatenated in dimension order
#                 (all the trailing zeros of scipy's (d, 18) array dropped).
# ----------------------------------------------------------------------
SOBOL_MAXDIM = 200   # max `d` for sobol(); scipy's own limit is 21201
SOBOL_BITS = 30      # fixed-point bits; max 2**30 points (scipy default)
_MAXDEG = 18         # max polynomial degree in scipy's table

_POLY_RAW = (
    1, 3, 7, 11, 13, 19, 25, 37, 41, 47, 55, 59, 61, 67, 91, 97, 103, 109,
    115, 131, 137, 143, 145, 157, 167, 171, 185, 191, 193, 203, 211, 213,
    229, 239, 241, 247, 253, 285, 299, 301, 333, 351, 355, 357, 361, 369,
    391, 397, 425, 451, 463, 487, 501, 529, 539, 545, 557, 563, 601, 607,
    617, 623, 631, 637, 647, 661, 675, 677, 687, 695, 701, 719, 721, 731,
    757, 761, 787, 789, 799, 803, 817, 827, 847, 859, 865, 875, 877, 883,
    895, 901, 911, 949, 953, 967, 971, 973, 981, 985, 995, 1001, 1019, 1033,
    1051, 1063, 1069, 1125, 1135, 1153, 1163, 1221, 1239, 1255, 1267, 1279,
    1293, 1305, 1315, 1329, 1341, 1347, 1367, 1387, 1413, 1423, 1431, 1441,
    1479, 1509, 1527, 1531, 1555, 1557, 1573, 1591, 1603, 1615, 1627, 1657,
    1663, 1673, 1717, 1729, 1747, 1759, 1789, 1815, 1821, 1825, 1849, 1863,
    1869, 1877, 1881, 1891, 1917, 1933, 1939, 1969, 2011, 2035, 2041, 2053,
    2071, 2091, 2093, 2119, 2147, 2149, 2161, 2171, 2189, 2197, 2207, 2217,
    2225, 2255, 2257, 2273, 2279, 2283, 2293, 2317, 2323, 2341, 2345, 2363,
    2365, 2373, 2377, 2385, 2395, 2419, 2421, 2431, 2435, 2447, 2475, 2477,
    2489, 2503
)

_VINIT_RAW = (
    1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 3, 5, 13, 1, 1, 5, 5, 17, 1,
    1, 5, 5, 5, 1, 1, 7, 11, 19, 1, 1, 5, 1, 1, 1, 1, 1, 3, 11, 1, 3, 5, 5,
    31, 1, 3, 3, 9, 7, 49, 1, 1, 1, 15, 21, 21, 1, 3, 1, 13, 27, 49, 1, 1,
    1, 15, 7, 5, 1, 3, 1, 15, 13, 25, 1, 1, 5, 5, 19, 61, 1, 3, 7, 11, 23,
    15, 103, 1, 3, 7, 13, 13, 15, 69, 1, 1, 3, 13, 7, 35, 63, 1, 3, 5, 9, 1,
    25, 53, 1, 3, 1, 13, 9, 35, 107, 1, 3, 1, 5, 27, 61, 31, 1, 1, 5, 11,
    19, 41, 61, 1, 3, 5, 3, 3, 13, 69, 1, 1, 7, 13, 1, 19, 1, 1, 3, 7, 5,
    13, 19, 59, 1, 1, 3, 9, 25, 29, 41, 1, 3, 5, 13, 23, 1, 55, 1, 3, 7, 3,
    13, 59, 17, 1, 3, 1, 3, 5, 53, 69, 1, 1, 5, 5, 23, 33, 13, 1, 1, 7, 7,
    1, 61, 123, 1, 1, 7, 9, 13, 61, 49, 1, 3, 3, 5, 3, 55, 33, 1, 3, 1, 15,
    31, 13, 49, 245, 1, 3, 5, 15, 31, 59, 63, 97, 1, 3, 1, 11, 11, 11, 77,
    249, 1, 3, 1, 11, 27, 43, 71, 9, 1, 1, 7, 15, 21, 11, 81, 45, 1, 3, 7,
    3, 25, 31, 65, 79, 1, 3, 1, 1, 19, 11, 3, 205, 1, 1, 5, 9, 19, 21, 29,
    157, 1, 3, 7, 11, 1, 33, 89, 185, 1, 3, 3, 3, 15, 9, 79, 71, 1, 3, 7,
    11, 15, 39, 119, 27, 1, 1, 3, 1, 11, 31, 97, 225, 1, 1, 1, 3, 23, 43,
    57, 177, 1, 3, 7, 7, 17, 17, 37, 71, 1, 3, 1, 5, 27, 63, 123, 213, 1, 1,
    3, 5, 11, 43, 53, 133, 1, 3, 5, 5, 29, 17, 47, 173, 479, 1, 3, 3, 11, 3,
    1, 109, 9, 69, 1, 1, 1, 5, 17, 39, 23, 5, 343, 1, 3, 1, 5, 25, 15, 31,
    103, 499, 1, 1, 1, 11, 11, 17, 63, 105, 183, 1, 1, 5, 11, 9, 29, 97,
    231, 363, 1, 1, 5, 15, 19, 45, 41, 7, 383, 1, 3, 7, 7, 31, 19, 83, 137,
    221, 1, 1, 1, 3, 23, 15, 111, 223, 83, 1, 1, 5, 13, 31, 15, 55, 25, 161,
    1, 1, 3, 13, 25, 47, 39, 87, 257, 1, 1, 1, 11, 21, 53, 125, 249, 293, 1,
    1, 7, 11, 11, 7, 57, 79, 323, 1, 1, 5, 5, 17, 13, 81, 3, 131, 1, 1, 7,
    13, 23, 7, 65, 251, 475, 1, 3, 5, 1, 9, 43, 3, 149, 11, 1, 1, 3, 13, 31,
    13, 13, 255, 487, 1, 3, 3, 1, 5, 63, 89, 91, 127, 1, 1, 3, 3, 1, 19,
    123, 127, 237, 1, 1, 5, 7, 23, 31, 37, 243, 289, 1, 1, 5, 11, 17, 53,
    117, 183, 491, 1, 1, 1, 5, 1, 13, 13, 209, 345, 1, 1, 3, 15, 1, 57, 115,
    7, 33, 1, 3, 1, 11, 7, 43, 81, 207, 175, 1, 3, 1, 1, 15, 27, 63, 255,
    49, 1, 3, 5, 3, 27, 61, 105, 171, 305, 1, 1, 5, 3, 1, 3, 57, 249, 149,
    1, 1, 3, 5, 5, 57, 15, 13, 159, 1, 1, 1, 11, 7, 11, 105, 141, 225, 1, 3,
    3, 5, 27, 59, 121, 101, 271, 1, 3, 5, 9, 11, 49, 51, 59, 115, 1, 1, 7,
    1, 23, 45, 125, 71, 419, 1, 1, 3, 5, 23, 5, 105, 109, 75, 1, 1, 7, 15,
    7, 11, 67, 121, 453, 1, 3, 7, 3, 9, 13, 31, 27, 449, 1, 3, 1, 15, 19,
    39, 39, 89, 15, 1, 1, 1, 1, 1, 33, 73, 145, 379, 1, 3, 1, 15, 15, 43,
    29, 13, 483, 1, 1, 7, 3, 19, 27, 85, 131, 431, 1, 3, 3, 3, 5, 35, 23,
    195, 349, 1, 3, 3, 7, 9, 27, 39, 59, 297, 1, 1, 3, 9, 11, 17, 13, 241,
    157, 1, 3, 7, 15, 25, 57, 33, 189, 213, 1, 1, 7, 1, 9, 55, 73, 83, 217,
    1, 3, 3, 13, 19, 27, 23, 113, 249, 1, 3, 5, 3, 23, 43, 3, 253, 479, 1,
    1, 5, 5, 11, 5, 45, 117, 217, 1, 3, 3, 7, 29, 37, 33, 123, 147, 1, 3, 1,
    15, 5, 5, 37, 227, 223, 459, 1, 1, 7, 5, 5, 39, 63, 255, 135, 487, 1, 3,
    1, 7, 9, 7, 87, 249, 217, 599, 1, 1, 3, 13, 9, 47, 7, 225, 363, 247, 1,
    3, 7, 13, 19, 13, 9, 67, 9, 737, 1, 3, 5, 5, 19, 59, 7, 41, 319, 677, 1,
    1, 5, 3, 31, 63, 15, 43, 207, 789, 1, 1, 7, 9, 13, 39, 3, 47, 497, 169,
    1, 3, 1, 7, 21, 17, 97, 19, 415, 905, 1, 3, 7, 1, 3, 31, 71, 111, 165,
    127, 1, 1, 5, 11, 1, 61, 83, 119, 203, 847, 1, 3, 3, 13, 9, 61, 19, 97,
    47, 35, 1, 1, 7, 7, 15, 29, 63, 95, 417, 469, 1, 3, 1, 9, 25, 9, 71, 57,
    213, 385, 1, 3, 5, 13, 31, 47, 101, 57, 39, 341, 1, 1, 3, 3, 31, 57,
    125, 173, 365, 551, 1, 3, 7, 1, 13, 57, 67, 157, 451, 707, 1, 1, 1, 7,
    21, 13, 105, 89, 429, 965, 1, 1, 5, 9, 17, 51, 45, 119, 157, 141, 1, 3,
    7, 7, 13, 45, 91, 9, 129, 741, 1, 3, 7, 1, 23, 57, 67, 141, 151, 571, 1,
    1, 3, 11, 17, 47, 93, 107, 375, 157, 1, 3, 3, 5, 11, 21, 43, 51, 169,
    915, 1, 1, 5, 3, 15, 55, 101, 67, 455, 625, 1, 3, 5, 9, 1, 23, 29, 47,
    345, 595, 1, 3, 7, 7, 5, 49, 29, 155, 323, 589, 1, 3, 3, 7, 5, 41, 127,
    61, 261, 717, 1, 3, 7, 7, 17, 23, 117, 67, 129, 1009, 1, 1, 3, 13, 11,
    39, 21, 207, 123, 305, 1, 1, 3, 9, 29, 3, 95, 47, 231, 73, 1, 3, 1, 9,
    1, 29, 117, 21, 441, 259, 1, 3, 1, 13, 21, 39, 125, 211, 439, 723, 1, 1,
    7, 3, 17, 63, 115, 89, 49, 773, 1, 3, 7, 13, 11, 33, 101, 107, 63, 73,
    1, 1, 5, 5, 13, 57, 63, 135, 437, 177, 1, 1, 3, 7, 27, 63, 93, 47, 417,
    483, 1, 1, 3, 1, 23, 29, 1, 191, 49, 23, 1, 1, 3, 15, 25, 55, 9, 101,
    219, 607, 1, 3, 1, 7, 7, 19, 51, 251, 393, 307, 1, 3, 3, 3, 25, 55, 17,
    75, 337, 3, 1, 1, 1, 13, 25, 17, 65, 45, 479, 413, 1, 1, 7, 7, 27, 49,
    99, 161, 213, 727, 1, 3, 5, 1, 23, 5, 43, 41, 251, 857, 1, 3, 3, 7, 11,
    61, 39, 87, 383, 835, 1, 1, 3, 15, 13, 7, 29, 7, 505, 923, 1, 3, 7, 1,
    5, 31, 47, 157, 445, 501, 1, 1, 3, 7, 1, 43, 9, 147, 115, 605, 1, 3, 3,
    13, 5, 1, 119, 211, 455, 1001, 1, 1, 3, 5, 13, 19, 3, 243, 75, 843, 1,
    3, 7, 7, 1, 19, 91, 249, 357, 589, 1, 1, 1, 9, 1, 25, 109, 197, 279,
    411, 1, 3, 1, 15, 23, 57, 59, 135, 191, 75, 1, 1, 5, 15, 29, 21, 39,
    253, 383, 349, 1, 3, 3, 5, 19, 45, 61, 151, 199, 981, 1, 3, 5, 13, 9,
    61, 107, 141, 141, 1, 1, 3, 1, 11, 27, 25, 85, 105, 309, 979, 1, 3, 3,
    11, 19, 7, 115, 223, 349, 43, 1, 1, 7, 9, 21, 39, 123, 21, 275, 927, 1,
    1, 7, 13, 15, 41, 47, 243, 303, 437, 1, 1, 1, 7, 7, 3, 15, 99, 409, 719,
    1, 3, 3, 15, 27, 49, 113, 123, 113, 67, 469, 1, 3, 7, 11, 3, 23, 87,
    169, 119, 483, 199, 1, 1, 5, 15, 7, 17, 109, 229, 179, 213, 741, 1, 1,
    5, 13, 11, 17, 25, 135, 403, 557, 1433, 1, 3, 1, 1, 1, 61, 67, 215, 189,
    945, 1243, 1, 1, 7, 13, 17, 33, 9, 221, 429, 217, 1679, 1, 1, 3, 11, 27,
    3, 15, 93, 93, 865, 1049, 1, 3, 7, 7, 25, 41, 121, 35, 373, 379, 1547,
    1, 3, 3, 9, 11, 35, 45, 205, 241, 9, 59, 1, 3, 1, 7, 3, 51, 7, 177, 53,
    975, 89, 1, 1, 3, 5, 27, 1, 113, 231, 299, 759, 861, 1, 3, 3, 15, 25,
    29, 5, 255, 139, 891, 2031, 1, 3, 1, 1, 13, 9, 109, 193, 419, 95, 17, 1,
    1, 7, 9, 3, 7, 29, 41, 135, 839, 867, 1, 1, 7, 9, 25, 49, 123, 217, 113,
    909, 215, 1, 1, 7, 3, 23, 15, 43, 133, 217, 327, 901, 1, 1, 3, 3, 13,
    53, 63, 123, 477, 711, 1387, 1, 1, 3, 15, 7, 29, 75, 119, 181, 957, 247,
    1, 1, 1, 11, 27, 25, 109, 151, 267, 99, 1461, 1, 3, 7, 15, 5, 5, 53,
    145, 11, 725, 1501, 1, 3, 7, 1, 9, 43, 71, 229, 157, 607, 1835, 1, 3, 3,
    13, 25, 1, 5, 27, 471, 349, 127, 1, 1, 1, 1, 23, 37, 9, 221, 269, 897,
    1685, 1, 1, 3, 3, 31, 29, 51, 19, 311, 553, 1969, 1, 3, 7, 5, 5, 55, 17,
    39, 475, 671, 1529, 1, 1, 7, 1, 1, 35, 47, 27, 437, 395, 1635, 1, 1, 7,
    3, 13, 23, 43, 135, 327, 139, 389, 1, 3, 7, 3, 9, 25, 91, 25, 429, 219,
    513, 1, 1, 3, 5, 13, 29, 119, 201, 277, 157, 2043, 1, 3, 5, 3, 29, 57,
    13, 17, 167, 739, 1031, 1, 3, 3, 5, 29, 21, 95, 27, 255, 679, 1531, 1,
    3, 7, 15, 9, 5, 21, 71, 61, 961, 1201, 1, 3, 5, 13, 15, 57, 33, 93, 459,
    867, 223, 1, 1, 1, 15, 17, 43, 127, 191, 67, 177, 1073, 1, 1, 1, 15, 23,
    7, 21, 199, 75, 293, 1611, 1, 3, 7, 13, 15, 39, 21, 149, 65, 741, 319,
    1, 3, 7, 11, 23, 13, 101, 89, 277, 519, 711, 1, 3, 7, 15, 19, 27, 85,
    203, 441, 97, 1895, 1, 3, 1, 3, 29, 25, 21, 155, 11, 191, 197
)


def _build_sobol_tables():
    """Unpack the embedded literals into dense arrays (import time, once)."""
    poly = np.array(_POLY_RAW, dtype=np.int64)
    if poly.shape[0] != SOBOL_MAXDIM:
        raise RuntimeError("corrupt embedded Sobol' polynomial table")
    deg = np.empty(SOBOL_MAXDIM, dtype=np.int64)
    for i in range(SOBOL_MAXDIM):
        deg[i] = int(poly[i]).bit_length() - 1
    vinit = np.zeros((SOBOL_MAXDIM, _MAXDEG), dtype=np.int64)
    k = 0
    for i in range(SOBOL_MAXDIM):
        for j in range(int(deg[i])):
            vinit[i, j] = _VINIT_RAW[k]
            k += 1
    if k != len(_VINIT_RAW):
        raise RuntimeError("corrupt embedded Sobol' direction-number table")
    return poly, deg, vinit


_SOBOL_POLY, _SOBOL_DEG, _SOBOL_VINIT = _build_sobol_tables()


# ----------------------------------------------------------------------
# self-contained RNG (splitmix64) -- numba cannot carry a Generator
# ----------------------------------------------------------------------
_2P53_INV = 1.0 / 9007199254740992.0     # 2**-53


@njit
def _rng_init(seed):
    """One-element uint64 state array seeded from an integer."""
    st = np.empty(1, dtype=np.uint64)
    st[0] = np.uint64(seed)
    return st


@njit
def _rng_u64(st):
    """splitmix64: advance the state, return a uint64 draw."""
    st[0] = st[0] + np.uint64(0x9E3779B97F4A7C15)
    z = st[0]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit
def _rng_double(st):
    """Uniform double in [0, 1) with 53 random bits."""
    return np.float64(_rng_u64(st) >> np.uint64(11)) * _2P53_INV


@njit
def _rng_bit(st):
    """A single random bit (0 or 1)."""
    return np.int64(_rng_u64(st) >> np.uint64(63))


@njit
def _rng_below(st, k):
    """Uniform integer in [0, k).  Modulo bias is < 2**-58 for k < 2**6."""
    return np.int64(_rng_u64(st) % np.uint64(k))


# ----------------------------------------------------------------------
# Sobol'
# ----------------------------------------------------------------------
@njit
def _sobol_direction_numbers(d, bits):
    """The (d, bits) matrix of scaled direction numbers.

    Bratley & Fox (1988) section 2: seed each row with the tabulated
    initial values, extend by the primitive-polynomial recursion, then
    scale column j by 2**(bits-1-j) so each entry is a `bits`-bit fixed
    point fraction.
    """
    sv = np.zeros((d, bits), dtype=np.int64)
    # dimension 0: all direction numbers are 1
    for i in range(bits):
        sv[0, i] = 1
    for dd in range(1, d):
        p = _SOBOL_POLY[dd]
        m = _SOBOL_DEG[dd]
        for i in range(m):
            sv[dd, i] = _SOBOL_VINIT[dd, i]
        for j in range(m, bits):
            newv = sv[dd, j - m]
            pow2 = 1
            for k in range(m):
                pow2 = pow2 << 1
                if (p >> (m - 1 - k)) & 1:
                    newv = newv ^ (pow2 * sv[dd, j - k - 1])
            sv[dd, j] = newv
    # multiply column bits-1-b by 2**b
    pow2 = 1
    for b in range(bits):
        for i in range(d):
            sv[i, bits - 1 - b] *= pow2
        pow2 = pow2 << 1
    return sv


@njit
def _sobol_scramble(sv, shift, st, d, bits):
    """Linear matrix scramble + random digital shift (in place).

    For each dimension an independent lower-triangular (in digit order,
    unit diagonal) GF(2) matrix L is drawn and applied to the digits of
    every direction number; a random `bits`-bit digital shift seeds the
    Gray-code recursion.  Both operations are nonsingular, so the (t, m, s)
    net property of the sequence survives.

    NOT scipy's stream: scipy draws L and the shift from a
    ``numpy.random.Generator``.  Structure is equivalent, numbers are not.
    """
    for dd in range(d):
        # random digital shift
        acc = 0
        for b in range(bits):
            if _rng_bit(st) == 1:
                acc += 1 << b
        shift[dd] = acc
    ltm = np.zeros((bits, bits), dtype=np.int64)
    for dd in range(d):
        for i in range(bits):
            for j in range(i):
                ltm[i, j] = _rng_bit(st)
            ltm[i, i] = 1
        for col in range(bits):
            vdj = sv[dd, col]
            newv = 0
            for i in range(bits):
                # digit i (i=0 is the most significant fractional digit)
                t = 0
                for j in range(i + 1):
                    if ltm[i, j] == 1:
                        t ^= (vdj >> (bits - 1 - j)) & 1
                if t == 1:
                    newv += 1 << (bits - 1 - i)
            sv[dd, col] = newv


@njit
def sobol(d, n, scramble=False, seed=0):
    """Sobol' quasi-random sequence -- ``(n, d)`` float64 in ``[0, 1)``.

    scipy equivalent: ``scipy.stats.qmc.Sobol(d, scramble=scramble,
    rng=...).random(n)`` on a freshly constructed engine.

    Parameters
    ----------
    d : int
        Dimension, ``1 <= d <= SOBOL_MAXDIM`` (200).
    n : int
        Number of points, ``0 <= n <= 2**SOBOL_BITS``.  The balance
        properties of the sequence only hold for ``n`` a power of two
        (scipy warns; we do not, warnings are not available in @njit).
    scramble : bool, optional
        Apply a linear matrix scramble + digital shift.  Default False.
    seed : int, optional
        Seed for the internal splitmix64 generator.  Ignored unless
        ``scramble`` is True.

    Returns
    -------
    sample : ndarray of float64, shape (n, d)
        The first ``n`` points of the sequence, every coordinate in
        ``[0, 1)``.  ``n == 0`` returns an empty ``(0, d)`` array.

    Raises
    ------
    ValueError
        If ``d < 1``, ``d > SOBOL_MAXDIM``, ``n < 0``, or ``n > 2**30``.

    Notes
    -----
    Stateless: this always restarts at index 0, so it corresponds to a
    FRESHLY constructed ``scipy.stats.qmc.Sobol`` engine.  scipy's
    ``.fast_forward()`` / ``.reset()`` / continuation across ``.random()``
    calls have no equivalent.  The first point of the unscrambled sequence
    is the origin, exactly as in scipy.

    Measured against ``scipy.stats.qmc.Sobol(d, scramble=False).random(n)``
    for (d, n) in {(1,16), (3,64), (10,128), (50,64), (200,32)}: EXACTLY
    0.0 max absolute difference -- bit-identical, from the same Joe-Kuo
    direction numbers, Bratley-Fox recursion and 30-bit Gray-code draw.
    With ``scramble=True`` the linear matrix scramble is driven by this
    module's splitmix64 generator, NOT ``numpy.random.Generator`` (which
    numba cannot call at all): reproducible per ``seed``, statistically
    valid, but never equal to scipy's numbers.

    prange-safe; distinct ``seed`` values give independent streams.
    """
    if d < 1:
        raise ValueError("sobol: 'd' must be >= 1")
    if d > SOBOL_MAXDIM:
        raise ValueError(
            "sobol: 'd' exceeds the embedded direction-number table "
            "(SOBOL_MAXDIM = 200)"
        )
    if n < 0:
        raise ValueError("sobol: 'n' must be >= 0")
    if n > 1073741824:          # 2**SOBOL_BITS
        raise ValueError("sobol: at most 2**30 points can be generated")

    bits = SOBOL_BITS
    sv = _sobol_direction_numbers(d, bits)
    shift = np.zeros(d, dtype=np.int64)
    if scramble:
        st = _rng_init(seed)
        _sobol_scramble(sv, shift, st, d, bits)

    out = np.empty((n, d), dtype=np.float64)
    if n == 0:
        return out

    fscale = 1.0 / np.float64(1 << bits)
    quasi = shift.copy()
    for j in range(d):
        out[0, j] = np.float64(quasi[j]) * fscale
    num_gen = 0
    for i in range(1, n):
        # index of the rightmost zero bit of num_gen (Gray-code recursion)
        c = 0
        while (num_gen >> c) & 1:
            c += 1
        for j in range(d):
            quasi[j] = quasi[j] ^ sv[j, c]
            out[i, j] = np.float64(quasi[j]) * fscale
        num_gen += 1
    return out


# ----------------------------------------------------------------------
# Halton / van der Corput
# ----------------------------------------------------------------------
@njit
def primes_from_2_to(d):
    """The first `d` primes -- cf. ``scipy.stats._qmc.primes_from_2_to``.

    Parameters
    ----------
    d : int
        How many primes to return, ``>= 1``.  DEVIATION: scipy's helper of
        the same name takes an upper BOUND and returns however many primes
        fall below it; taking a count is what Halton actually needs and
        avoids guessing the bound.  There is no default.

    Returns
    -------
    ndarray of int64, shape (d,)
        The first ``d`` primes, ascending, starting at 2.

    Raises
    ------
    ValueError
        If ``d < 1``.

    Notes
    -----
    Trial division by the primes already found, stopping at ``sqrt(x)``;
    exact by construction, so there is no accuracy figure to quote.
    Verified against the known prefix ``2, 3, 5, 7, 11, 13, 17, 19, 23,
    29``.  prange-safe.
    """
    if d < 1:
        raise ValueError("primes_from_2_to: 'd' must be >= 1")
    p = np.empty(d, dtype=np.int64)
    cnt = 0
    x = 2
    while cnt < d:
        isp = True
        for i in range(cnt):
            q = p[i]
            if q * q > x:
                break
            if x % q == 0:
                isp = False
                break
        if isp:
            p[cnt] = x
            cnt += 1
        x += 1
    return p


@njit
def van_der_corput(n, base=2, scramble=False, seed=0):
    """Van der Corput sequence -- ``scipy.stats._qmc.van_der_corput``.

    The 1-D radical-inverse sequence Halton is built from.  Note the scipy
    counterpart is PRIVATE (``scipy.stats._qmc``), not part of the public
    ``scipy.stats.qmc`` namespace.

    Parameters
    ----------
    n : int
        Number of points, ``>= 0``.
    base : int, optional
        Radix, ``>= 2``.  Default 2, same as scipy.
    scramble : bool, optional
        Apply Owen-style random digit permutations.  Default False, same
        as scipy.
    seed : int, optional
        Seed for the internal splitmix64 generator.  Default 0.  Ignored
        unless ``scramble`` is True.

    Returns
    -------
    ndarray of float64, shape (n,)
        The radical inverses of ``0 .. n-1`` in ``base``, all in
        ``[0, 1)``.  ``n == 0`` returns an empty array.

    Raises
    ------
    ValueError
        If ``base < 2`` or ``n < 0``.

    Notes
    -----
    scipy's ``start_index`` and ``workers`` arguments are not supported --
    the sequence always starts at index 0.  Measured against
    ``scipy.stats._qmc.van_der_corput(n, base)`` for (n, base) in
    {(32,2), (50,3), (40,7)}: EXACTLY 0.0 max absolute difference.
    Scrambled output uses this module's splitmix64 permutations, not
    ``numpy.random`` -- reproducible per ``seed``, never equal to scipy's.

    prange-safe.
    """
    if base < 2:
        raise ValueError("van_der_corput: 'base' must be at least 2")
    if n < 0:
        raise ValueError("van_der_corput: 'n' must be >= 0")
    seq = np.empty(n, dtype=np.float64)
    if scramble:
        st = _rng_init(seed)
        perm = _vdc_permutations(base, st)
        _vdc_scrambled(seq, n, base, perm)
    else:
        _vdc_plain(seq, n, base)
    return seq


@njit
def _vdc_plain(out, n, base):
    """Radical inverse of 0..n-1 in `base`, written into `out`."""
    inv = 1.0 / np.float64(base)
    for i in range(n):
        q = i
        b2r = inv
        val = 0.0
        while q > 0:
            rem = q % base
            val += np.float64(rem) * b2r
            b2r /= np.float64(base)
            q //= base
        out[i] = val


@njit
def _vdc_permutations(base, st):
    """One random permutation of ``0..base-1`` per digit level (Owen)."""
    count = int(np.ceil(54.0 / np.log2(np.float64(base)))) - 1
    if count < 1:
        count = 1
    perm = np.empty((count, base), dtype=np.int64)
    for r in range(count):
        for c in range(base):
            perm[r, c] = c
        for c in range(base - 1, 0, -1):     # Fisher-Yates
            k = _rng_below(st, c + 1)
            tmp = perm[r, c]
            perm[r, c] = perm[r, k]
            perm[r, k] = tmp
    return perm


@njit
def _vdc_scrambled(out, n, base, perm):
    """Radical inverse with each digit level permuted independently."""
    count = perm.shape[0]
    inv = 1.0 / np.float64(base)
    for i in range(n):
        q = i
        b2r = inv
        val = 0.0
        for idx in range(count):
            rem = q % base
            q //= base
            val += np.float64(perm[idx, rem]) * b2r
            b2r /= np.float64(base)
        out[i] = val


@njit
def halton(d, n, scramble=False, seed=0):
    """Halton quasi-random sequence -- ``(n, d)`` float64 in ``[0, 1)``.

    scipy equivalent: ``scipy.stats.qmc.Halton(d, scramble=scramble,
    rng=...).random(n)`` on a freshly constructed engine.  Dimension ``j``
    is the van der Corput sequence in the ``j``-th prime base.

    Parameters
    ----------
    d : int
        Dimension, ``>= 1``.  There is no upper limit (the bases are the
        first `d` primes, computed on the fly), but Halton correlates badly
        in high dimensions -- prefer `sobol` above d ~ 10.
    n : int
        Number of points, ``>= 0``.
    scramble : bool, optional
        Apply Owen-style random digit permutations.  Default False.
    seed : int, optional
        Seed for the internal splitmix64 generator.  Ignored unless
        ``scramble`` is True.

    Returns
    -------
    sample : ndarray of float64, shape (n, d)
        The first ``n`` points, every coordinate in ``[0, 1)``.  ``n == 0``
        returns an empty ``(0, d)`` array.

    Raises
    ------
    ValueError
        If ``d < 1`` or ``n < 0``.

    Notes
    -----
    Stateless: always restarts at index 0, i.e. a FRESHLY constructed
    ``scipy.stats.qmc.Halton`` engine; scipy's ``.fast_forward()`` /
    ``.reset()`` have no equivalent.  The first point of the unscrambled
    sequence is the origin, exactly as in scipy.

    Measured against ``scipy.stats.qmc.Halton(d, scramble=False).random(n)``
    for (d, n) in {(1,20), (5,100), (15,64)}: EXACTLY 0.0 max absolute
    difference.  With ``scramble=True`` the digit permutations come from
    this module's splitmix64 generator, not ``numpy.random`` --
    reproducible per ``seed``, never equal to scipy's numbers.

    prange-safe.
    """
    if d < 1:
        raise ValueError("halton: 'd' must be >= 1")
    if n < 0:
        raise ValueError("halton: 'n' must be >= 0")
    bases = primes_from_2_to(d)
    out = np.empty((n, d), dtype=np.float64)
    if n == 0:
        return out
    col = np.empty(n, dtype=np.float64)
    if scramble:
        st = _rng_init(seed)
        for j in range(d):
            base = bases[j]
            perm = _vdc_permutations(base, st)
            _vdc_scrambled(col, n, base, perm)
            for i in range(n):
                out[i, j] = col[i]
    else:
        for j in range(d):
            _vdc_plain(col, n, bases[j])
            for i in range(n):
                out[i, j] = col[i]
    return out


# ----------------------------------------------------------------------
# Latin hypercube
# ----------------------------------------------------------------------
@njit
def latin_hypercube(d, n, seed=0, scramble=True):
    """Latin hypercube sample -- ``(n, d)`` float64 in ``(0, 1)``.

    scipy equivalent: ``scipy.stats.qmc.LatinHypercube(d, scramble=scramble,
    rng=...).random(n)`` (strength 1).

    Each column is a random permutation of the `n` equal-probability
    strata ``[i/n, (i+1)/n)``, so every stratum of every dimension holds
    exactly one point.  ``scramble=True`` (scipy's default) jitters the
    point uniformly inside its stratum; ``scramble=False`` centers it, i.e.
    each column is a permutation of ``(i + 0.5)/n``.

    Parameters
    ----------
    d : int
        Dimension, ``>= 1``.
    n : int
        Number of points, ``>= 0``.
    seed : int, optional
        Seed for the internal splitmix64 generator.  Default 0.  An
        explicit integer is REQUIRED by numba, which cannot hold a
        ``numpy.random.Generator``; there is no draw-from-OS-entropy mode,
        so the same seed always gives the same sample.
    scramble : bool, optional
        Jitter within the stratum (True, default -- scipy's default too)
        or center it (False, giving a permutation of ``(i + 0.5)/n``).

    Returns
    -------
    sample : ndarray of float64, shape (n, d)
        Every coordinate in ``(0, 1)``; each column is a permutation of the
        ``n`` strata, one point per stratum.  ``n == 0`` returns an empty
        ``(0, d)`` array.

    Raises
    ------
    ValueError
        If ``d < 1`` or ``n < 0``.

    Notes
    -----
    **This cannot be compared bit-for-bit with scipy.**  A Latin hypercube
    is randomized by construction (the strata permutations and the
    within-stratum offsets) and scipy draws them from
    ``numpy.random.Generator``, which numba cannot call at all; this uses
    splitmix64.  scipy's ``strength=2`` orthogonal-array variant and
    ``optimization=`` post-processing are not implemented.

    Verified by property instead of comparison, at d=4, n=20, seed=7: the
    same seed reproduces the sample EXACTLY, every value lies strictly
    inside (0, 1), each of the 4 columns hits all 20 strata exactly once,
    and ``scramble=False`` reproduces ``(i + 0.5)/n`` to within
    ``allclose``.

    prange-safe; distinct ``seed`` values give independent streams.
    """
    if d < 1:
        raise ValueError("latin_hypercube: 'd' must be >= 1")
    if n < 0:
        raise ValueError("latin_hypercube: 'n' must be >= 0")
    out = np.empty((n, d), dtype=np.float64)
    if n == 0:
        return out
    st = _rng_init(seed)
    perm = np.empty(n, dtype=np.int64)
    fn = np.float64(n)
    for j in range(d):
        for i in range(n):
            perm[i] = i + 1
        for i in range(n - 1, 0, -1):        # Fisher-Yates
            k = _rng_below(st, i + 1)
            tmp = perm[i]
            perm[i] = perm[k]
            perm[k] = tmp
        for i in range(n):
            if scramble:
                u = _rng_double(st)
            else:
                u = 0.5
            out[i, j] = (np.float64(perm[i]) - u) / fn
    return out


# ----------------------------------------------------------------------
# discrepancy
# ----------------------------------------------------------------------
@njit
def _check_unit_hypercube(s):
    n = s.shape[0]
    d = s.shape[1]
    if n < 1 or d < 1:
        raise ValueError("sample must have at least one point and one column")
    for i in range(n):
        for j in range(d):
            if s[i, j] > 1.0 or s[i, j] < 0.0:
                raise ValueError("Sample is not in unit hypercube")


@njit
def discrepancy(sample, method=0):
    """Discrepancy of a sample in the unit hypercube -- lower is better.

    scipy equivalent: ``scipy.stats.qmc.discrepancy(sample, method=...)``
    with ``iterative=False``.

    Parameters
    ----------
    sample : ndarray, shape (n, d)
        Points, all in ``[0, 1]``; values outside the unit hypercube raise.
    method : int, optional
        INT CODE for scipy's string flag (numba cannot dispatch on
        strings):

        * ``0`` -- ``"CD"``, centered L2 discrepancy.  Default, matching
          scipy's default ``method='CD'``.
        * ``1`` -- ``"WD"``, wrap-around L2 discrepancy.
        * ``2`` -- ``"MD"``, mixture L2 discrepancy.
        * ``3`` -- ``"L2-star"``, star L2 discrepancy.

    Returns
    -------
    disc : float
        The discrepancy -- lower is a more uniformly spread sample.

    Raises
    ------
    ValueError
        If ``method`` is outside 0..3, or if any sample value lies outside
        ``[0, 1]``.

    Notes
    -----
    scipy's ``iterative=True`` and ``workers`` are not supported.
    Measured against ``scipy.stats.qmc.discrepancy`` on Sobol' and
    uniform-random samples with (n, d) in {(20,2), (64,3), (50,5)}:
    CD, WD and MD EXACTLY 0.0 relative difference (bit-identical);
    L2-star max relative difference 4.9e-14, because its three terms are
    each far larger than their difference and cancel, amplifying any
    last-bit summation-order difference.

    prange-safe.
    """
    if method < 0 or method > 3:
        raise ValueError(
            "discrepancy: 'method' must be 0 (CD), 1 (WD), 2 (MD) or "
            "3 (L2-star)"
        )
    s = np.ascontiguousarray(np.asarray(sample).astype(np.float64))
    _check_unit_hypercube(s)
    n = s.shape[0]
    d = s.shape[1]

    disc1 = 0.0
    if method != 1:
        for i in range(n):
            prod = 1.0
            for j in range(d):
                if method == 0:
                    a = abs(s[i, j] - 0.5)
                    prod *= 1.0 + 0.5 * a - 0.5 * a * a
                elif method == 2:
                    a = abs(s[i, j] - 0.5)
                    prod *= 5.0 / 3.0 - 0.25 * a - 0.25 * a * a
                else:
                    prod *= 1.0 - s[i, j] * s[i, j]
            disc1 += prod

    disc2 = 0.0
    for i in range(n):
        for k in range(n):
            prod = 1.0
            for j in range(d):
                if method == 0:
                    prod *= (1.0 + 0.5 * abs(s[i, j] - 0.5)
                             + 0.5 * abs(s[k, j] - 0.5)
                             - 0.5 * abs(s[i, j] - s[k, j]))
                elif method == 1:
                    x = abs(s[i, j] - s[k, j])
                    prod *= 1.5 - x + x * x
                elif method == 2:
                    a = abs(s[i, j] - 0.5)
                    b = abs(s[k, j] - 0.5)
                    c = abs(s[i, j] - s[k, j])
                    prod *= (15.0 / 8.0 - 0.25 * a - 0.25 * b
                             - 0.75 * c + 0.5 * c * c)
                else:
                    prod *= 1.0 - max(s[i, j], s[k, j])
            disc2 += prod

    fn = np.float64(n)
    # NOTE: the exponent must be a FLOAT.  numba compiles `x ** int64` to a
    # repeated-multiplication powi that is several ulp off libm's pow(),
    # which is what scipy's C code (and CPython's float.__pow__) calls;
    # `x ** float64(d)` reproduces scipy bit-for-bit.
    fd = np.float64(d)
    if method == 0:
        return ((13.0 / 12.0) ** fd - 2.0 / fn * disc1
                + 1.0 / (fn * fn) * disc2)
    if method == 1:
        return -(4.0 / 3.0) ** fd + 1.0 / (fn * fn) * disc2
    if method == 2:
        return ((19.0 / 12.0) ** fd - 2.0 / fn * disc1
                + 1.0 / (fn * fn) * disc2)
    one_div_n = 1.0 / fn
    return np.sqrt(3.0 ** (-fd)
                   - one_div_n * 2.0 ** (1.0 - fd) * disc1
                   + one_div_n * one_div_n * disc2)


# ----------------------------------------------------------------------
# scale
# ----------------------------------------------------------------------
@njit
def scale(sample, l_bounds, u_bounds, reverse=False):
    """Map a unit-hypercube sample onto ``[l_bounds, u_bounds]``.

    scipy equivalent: ``scipy.stats.qmc.scale(sample, l_bounds, u_bounds,
    reverse=reverse)``; bit-identical (same ``sample * (u - l) + l``).

    Parameters
    ----------
    sample : ndarray, shape (n, d)
        Points in ``[0, 1]`` (or in ``[l, u]`` when ``reverse`` is True).
    l_bounds, u_bounds : ndarray, shape (d,)
        Lower / upper bounds, one entry per column of ``sample``;
        ``l_bounds[j] < u_bounds[j]`` required.  Arrays only -- scipy also
        accepts scalars, which broadcast; here pass a full-length array.
    reverse : bool, optional
        ``False`` (default, same as scipy) maps the unit hypercube onto
        ``[l, u]``; ``True`` maps ``[l, u]`` back to the unit hypercube.

    Returns
    -------
    sample : ndarray of float64, shape (n, d)
        The transformed sample.

    Raises
    ------
    ValueError
        If ``l_bounds`` or ``u_bounds`` does not have one entry per column
        of ``sample``, or if ``l_bounds[j] >= u_bounds[j]`` for any ``j``.

    Notes
    -----
    Measured against ``scipy.stats.qmc.scale``, forward and reverse, on a
    32-point 3-D Sobol' sample with mixed-sign bounds: EXACTLY 0.0 max
    absolute difference in both directions -- the expression order
    ``sample * (u - l) + l`` is the same.  prange-safe.
    """
    s = np.ascontiguousarray(np.asarray(sample).astype(np.float64))
    lo = np.ascontiguousarray(np.asarray(l_bounds).astype(np.float64)).ravel()
    hi = np.ascontiguousarray(np.asarray(u_bounds).astype(np.float64)).ravel()
    n = s.shape[0]
    d = s.shape[1]
    if lo.shape[0] != d or hi.shape[0] != d:
        raise ValueError(
            "scale: 'l_bounds' and 'u_bounds' must have one entry per "
            "column of 'sample'"
        )
    for j in range(d):
        if not (lo[j] < hi[j]):
            raise ValueError("scale: bounds are not consistent, l_bounds < "
                             "u_bounds is required")
    out = np.empty((n, d), dtype=np.float64)
    if not reverse:
        for i in range(n):
            for j in range(d):
                if s[i, j] > 1.0 or s[i, j] < 0.0:
                    raise ValueError("Sample is not in unit hypercube")
        for j in range(d):
            span = hi[j] - lo[j]
            for i in range(n):
                out[i, j] = s[i, j] * span + lo[j]
    else:
        for i in range(n):
            for j in range(d):
                if s[i, j] < lo[j] or s[i, j] > hi[j]:
                    raise ValueError("Sample is out of bounds")
        for j in range(d):
            span = hi[j] - lo[j]
            for i in range(n):
                out[i, j] = (s[i, j] - lo[j]) / span
    return out
