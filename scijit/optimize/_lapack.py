"""Private LAPACK plumbing for scijit.optimize.

Vendors ``qr_pivot`` and ``solve_triangular`` (the ``trf``/``bvls`` machinery
behind :func:`lsq_linear` needs them) together with their ctypes binding, so
``scijit.optimize`` imports nothing from ``scijit.linalg``. The three routines
are backed by ``liboptlapack`` (built from ``src/optlapack``), a wrappers-only
shared library holding only ``dtrtrs_c`` / ``dgeqp3_c`` / ``dorgqr_c`` and
linking the shared ``liblapackref``.

Both functions are vendored verbatim from ``scijit.linalg`` (``_tier2.qr_pivot``,
``_linalg.solve_triangular``). FUTURE WORK: when ``scijit.linalg`` ships in the
public package, re-import them from ``..linalg`` and delete this module and
``src/optlapack``.
"""
from .._lib._load import load

import numpy as np
from numba import njit

_lib, _sig = load(__file__, "liboptlapack")


_dtrtrs = _sig(_lib.dtrtrs_c, 10)
_dgeqp3 = _sig(_lib.dgeqp3_c, 7)
_dorgqr = _sig(_lib.dorgqr_c, 7)

# char codes (match src/optlapack/wrappers.f90)
_UP = 0
_LO = 1
_N = 0
_T = 1


@njit
def _fortran(a):
    """Column-major (Fortran-order) float64 copy of a 2-D array."""
    return np.ascontiguousarray(np.asarray(a, np.float64).T)


@njit
def qr_pivot(a):
    """Economy-size QR with column pivoting (dgeqp3 + dorgqr).

    Returns ``(Q, R, piv)`` with ``a[:, piv] == Q @ R`` and ``piv`` 0-based.
    """
    a = np.asarray(a, np.float64)
    m = np.int32(a.shape[0])
    n = np.int32(a.shape[1])
    k = np.int32(min(m, n))
    af = _fortran(a)
    jpvt = np.zeros(n, np.int32)
    tau = np.zeros(k, np.float64)
    info = np.zeros(1, np.int32)
    m_ = np.array(m, np.int32)
    n_ = np.array(n, np.int32)
    lda = np.array(m, np.int32)
    _dgeqp3(m_.ctypes.data, n_.ctypes.data, af.ctypes.data, lda.ctypes.data,
            jpvt.ctypes.data, tau.ctypes.data, info.ctypes.data)
    if info[0] < 0:
        raise ValueError("qr_pivot: illegal argument to dgeqp3")
    R = np.zeros((k, n), np.float64)
    for i in range(k):
        for j in range(n):
            if i <= j:
                R[i, j] = af[j, i]
    kk = np.array(k, np.int32)
    _dorgqr(m_.ctypes.data, kk.ctypes.data, kk.ctypes.data, af.ctypes.data,
            lda.ctypes.data, tau.ctypes.data, info.ctypes.data)
    if info[0] < 0:
        raise ValueError("qr_pivot: illegal argument to dorgqr")
    Q = np.zeros((m, k), np.float64)
    for j in range(k):
        for i in range(m):
            Q[i, j] = af[j, i]
    piv = np.zeros(n, np.int64)
    for j in range(n):
        piv[j] = jpvt[j] - 1
    return Q, R, piv


@njit
def solve_triangular(a, b, lower=False, trans=0, unit_diagonal=False):
    """Solve a triangular system ``A x = b`` (dtrtrs), ``b`` 1-D."""
    a = np.asarray(a, np.float64)
    n = np.int32(a.shape[0])
    if a.shape[1] != n:
        raise ValueError("solve_triangular: matrix must be square")
    af = _fortran(a)
    bb = np.ascontiguousarray(np.asarray(b, np.float64)).copy()
    if bb.size != n:
        raise ValueError("solve_triangular: b length must equal dimension")
    uplo = np.array(_LO if lower else _UP, np.int32)
    tr = np.array(trans, np.int32)
    diag = np.array(1 if unit_diagonal else 0, np.int32)
    n_ = np.array(n, np.int32)
    nrhs = np.array(1, np.int32)
    lda = np.array(n, np.int32)
    ldb = np.array(n, np.int32)
    info = np.zeros(1, np.int32)
    _dtrtrs(uplo.ctypes.data, tr.ctypes.data, diag.ctypes.data,
            n_.ctypes.data, nrhs.ctypes.data, af.ctypes.data,
            lda.ctypes.data, bb.ctypes.data, ldb.ctypes.data,
            info.ctypes.data)
    if info[0] < 0:
        raise ValueError("solve_triangular: illegal argument to dtrtrs")
    if info[0] > 0:
        raise ValueError("solve_triangular: matrix is singular")
    return bb
