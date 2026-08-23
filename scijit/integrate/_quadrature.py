"""Sampled-data quadrature, the rules that take samples rather than a callback.

Every routine here is a plain ``@njit`` function reaching no shared library,
so each is callable inside other ``@njit`` code and each is prange-safe: no
module state, no Python objects.

Covered, under scipy's own names: ``trapezoid``, ``cumulative_trapezoid``,
``simpson``, ``cumulative_simpson``, ``romb``, ``newton_cotes``,
``fixed_quad``, plus the ``scipy.special.roots_legendre`` helper
(Golub-Welsch).

``y`` may be of any rank and ``axis`` selects the axis to integrate along,
as in scipy. An integer or boolean sample array is promoted to float64 and a
complex one stays complex, which is scipy's ``force_floating`` promotion.
Every reduction runs in numpy's own summation order, so a result is bit for
bit what scipy returns wherever the algorithm agrees.

Deviations from scipy:

* ``cumulative_trapezoid`` accepts a non-zero ``initial``, prepending it and
  adding it to every element. scipy accepts a non-zero ``initial`` on
  ``cumulative_simpson`` and rejects it here.
* ``fixed_quad``'s ``args`` must be a tuple inside ``@njit``. From the
  interpreter a list or an array is splatted, as scipy's own ``*args``
  splats it, and a compiled call has no equivalent because the length of an
  array is not part of its type.
* ``newton_cotes`` raises for ``N < 1``, where scipy warns and then raises
  ``ValueError: math domain error``, and for an ``rn`` of rank 2 or more,
  where scipy reads its order from the first axis alone.
* ``romb``'s ``show`` must be a compile-time constant inside ``@njit``.
* Where a routine differs from scipy its docstring says so under ``Notes``,
  and silence means the two were compared and agreed.
"""
import math

import numpy as np
from numba import njit, objmode, types
from numba.core.errors import TypingError
from numba.extending import overload


# ---------------------------------------------------------------------
# compile-time predicates, shared by the choosers below
# ---------------------------------------------------------------------
def _lit_bool(v):
    """The value of a boolean-ish argument at TYPING time, or None when
    it is a runtime variable and no compiled body can be chosen."""
    if isinstance(v, bool):
        return v                                  # omitted default
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, types.Omitted):
        return bool(v.value)
    if isinstance(v, types.BooleanLiteral):
        return v.literal_value
    if isinstance(v, types.IntegerLiteral):
        return bool(v.literal_value)
    return None


def _is_none(v):
    """True when an argument is ``None`` at TYPING time."""
    return (v is None or isinstance(v, types.NoneType)
            or (isinstance(v, types.Omitted) and v.value is None))


def _fn_arity(v):
    """Number of positional parameters of a first-class ``@njit``
    function, from a Dispatcher object (python entry) or a
    ``types.Dispatcher`` (chooser)."""
    if isinstance(v, types.Dispatcher):
        py = v.dispatcher.py_func
    else:
        py = getattr(v, 'py_func', v)
    return py.__code__.co_argcount


# ---------------------------------------------------------------------
# summation order
# ---------------------------------------------------------------------
_PW_BLOCKSIZE = 128


@njit
def _pairwise_sum(a, off, n):
    """``sum(a[off:off + n])`` in numpy's summation order.

    numpy does not accumulate a reduction sequentially: below 8 elements it
    runs a plain loop, up to 128 it carries eight partial accumulators and
    combines them as a balanced tree, and above 128 it splits in half at a
    multiple of 8 and recurses.  Every scipy routine in this module reduces
    with ``xp.sum``, so a sequential loop lands one or more bits away.
    """
    if n < 8:
        res = a.dtype.type(0)
        for i in range(n):
            res += a[off + i]
        return res
    if n <= _PW_BLOCKSIZE:
        r0 = a[off]
        r1 = a[off + 1]
        r2 = a[off + 2]
        r3 = a[off + 3]
        r4 = a[off + 4]
        r5 = a[off + 5]
        r6 = a[off + 6]
        r7 = a[off + 7]
        i = 8
        m = n - (n % 8)
        while i < m:
            r0 += a[off + i]
            r1 += a[off + i + 1]
            r2 += a[off + i + 2]
            r3 += a[off + i + 3]
            r4 += a[off + i + 4]
            r5 += a[off + i + 5]
            r6 += a[off + i + 6]
            r7 += a[off + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[off + i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum(a, off, n2) + _pairwise_sum(a, off + n2, n - n2)


@njit
def _pairwise_sum_c(a, off, n):
    """:func:`_pairwise_sum` for a complex array.

    numpy counts DOUBLES rather than elements here, so the same three
    constants become four accumulators, a block of 64 and a threshold of 4.
    Measured: bit-identical to ``np.sum`` at all 203 lengths tried, where the
    real blocking mismatched 129 of them.
    """
    if n < 4:
        res = a.dtype.type(0)
        for i in range(n):
            res += a[off + i]
        return res
    if n <= _PW_BLOCKSIZE // 2:
        r0 = a[off]
        r1 = a[off + 1]
        r2 = a[off + 2]
        r3 = a[off + 3]
        i = 4
        m = n - (n % 4)
        while i < m:
            r0 += a[off + i]
            r1 += a[off + i + 1]
            r2 += a[off + i + 2]
            r3 += a[off + i + 3]
            i += 4
        res = (r0 + r1) + (r2 + r3)
        while i < n:
            res += a[off + i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 4
    return _pairwise_sum_c(a, off, n2) + _pairwise_sum_c(a, off + n2, n - n2)


def _psum(a):
    """``sum(a)`` in numpy's summation order, whatever ``a``'s dtype."""
    return np.sum(a)


@overload(_psum)
def _psum_ovl(a):
    """Compiled body for :func:`_psum`, chosen on the dtype."""
    if isinstance(a.dtype, types.Complex):
        def impl(a):
            return _pairwise_sum_c(a, 0, a.shape[0])
        return impl

    def impl(a):
        return _pairwise_sum(a, 0, a.shape[0])
    return impl


# ---------------------------------------------------------------------
# N-D samples and scipy's `axis`
# ---------------------------------------------------------------------
# scipy indexes ``y.shape[axis]``, so an out-of-range axis surfaces as
# numpy's own tuple message rather than a validated one.
_AXIS_MSG = 'tuple index out of range'


def _samples(y):
    """``y`` as a contiguous array of floating dtype.

    scipy runs ``xp_promote(..., force_floating=True)``, which leaves an
    inexact array alone and lifts everything else to the default float.
    """
    a = np.ascontiguousarray(np.asarray(y))
    if a.dtype == np.complex64 or a.dtype == np.complex128:
        return a.astype(np.complex128)
    return a.astype(np.float64)


def _positions(x):
    """``x`` as a contiguous float64 array."""
    return np.ascontiguousarray(np.asarray(x)).astype(np.float64)


def _cast_of(ty):
    """The dtype :func:`_samples` produces, spelled for generated source."""
    return 'np.complex128' if isinstance(ty.dtype, types.Complex) else 'np.float64'


def _x_kind(x, nd):
    """How ``x`` reaches a compiled body: 0 absent, 1 one-dimensional,
    2 the same rank as ``y``.  ``None`` when it is neither."""
    if _is_none(x):
        return 0
    if not isinstance(x, types.Array):
        return None
    if x.ndim == 1:
        return 1
    if x.ndim == nd:
        return 2
    return None


def _axis_lines(nd, tail, ind):
    """Source choosing the output shape once ``ax`` is known.

    A tuple cannot be sliced by a value numba only knows at compile time, so
    the shape is written out per axis and the branch is resolved at run time.
    All branches build a tuple of the same length, so they share one type.
    """
    def shp(ax):
        dims = ['ys[%d]' % k for k in range(nd) if k != ax] + list(tail)
        return ', '.join(dims)

    if nd == 1 and not tail:
        return ['%sout = res' % ind]
    out = []
    for ax in range(nd - 1):
        kw = 'if' if ax == 0 else 'elif'
        out.append('%s%s ax == %d:' % (ind, kw, ax))
        out.append('%s    out = res.reshape((%s))' % (ind, shp(ax)))
    if out:
        out.append('%selse:' % ind)
        out.append('%s    out = res.reshape((%s))' % (ind, shp(nd - 1)))
    else:
        out.append('%sout = res.reshape((%s))' % (ind, shp(0)))
    return out


def _nd_body(sig, call, nd, xk, cast, klen, glb, pre=()):
    """Compile a body applying a 1-D core along ``axis`` of an N-D ``y``.

    ``call`` names the core and is handed ``(yrow, xrow, has_x)`` plus
    whatever else the routine's own arguments supply.  A reduction writes one
    scalar per row; a cumulative routine writes one row of ``k`` values, which
    is why the output rank differs.
    """
    L = ['def impl(%s):' % sig,
         '    yc = np.ascontiguousarray(np.asarray(y)).astype(%s)' % cast,
         '    ax = axis + %d if axis < 0 else axis' % nd,
         '    if ax < 0 or ax >= %d:' % nd,
         '        raise IndexError(_AXIS_MSG)',
         '    ys = yc.shape',
         '    ym = np.ascontiguousarray(np.moveaxis(yc, ax, -1))',
         '    n = ym.shape[%d]' % (nd - 1)]
    L += ['    ' + s for s in pre]
    L.append('    M = 1')
    for k in range(nd - 1):
        L.append('    M *= ym.shape[%d]' % k)
    L.append('    flat = ym.reshape(M, n)')
    if xk == 0:
        L.append('    xr = np.empty(0, np.float64)')
        row = 'xr, False'
    elif xk == 1:
        L.append('    xr = np.ascontiguousarray(np.asarray(x))'
                 '.astype(np.float64)')
        row = 'xr, True'
    else:
        L += ['    xm = np.ascontiguousarray(np.moveaxis('
              'np.ascontiguousarray(np.asarray(x)).astype(np.float64), ax, -1))',
              '    xf = xm.reshape(M, n)']
        row = 'xf[i], True'
    if klen is not None:
        L += ['    k = %s' % klen,
              '    res = np.empty((M, k), %s)' % cast,
              '    for i in range(M):',
              '        res[i] = %s' % call.format(y='flat[i]', x=row)]
        L += _axis_lines(nd, ['k'], '    ')
        L.append('    return np.ascontiguousarray(np.moveaxis(out, -1, ax))')
    else:
        L += ['    res = np.empty(M, %s)' % cast,
              '    for i in range(M):',
              '        res[i] = %s' % call.format(y='flat[i]', x=row)]
        L += _axis_lines(nd, [], '    ')
        L.append('    return out')
    ns = dict(glb)
    exec('\n'.join(L), ns)                                   # noqa: S102
    return ns['impl']


# ---------------------------------------------------------------------
# trapezoid
# ---------------------------------------------------------------------
# scipy's `trapezoid` assigns into a LIST of slices, so its out-of-range
# axis message is not the tuple one the rest of the module reaches.
_TZ_AXIS_MSG = 'list assignment index out of range'


@njit
def _bcast_msg(m, n):
    """numpy's message for two 1-D operands that do not broadcast."""
    return ('operands could not be broadcast together with shapes ('
            + str(m) + ',) (' + str(n) + ',) ')


@njit
def _trapz_1d(y, x, has_x, dx):
    """Composite trapezoid over one row of samples.

    scipy multiplies the spacings against the sample pairs and lets numpy
    broadcast, so the two lengths do not have to agree: either may be 1.
    """
    n = y.shape[0]
    ny = n - 1
    if ny < 0:
        ny = 0
    if has_x:
        m = x.shape[0] - 1
        if m < 0:
            m = 0
    else:
        m = ny
    if m == ny or m == 1:
        L = ny
    elif ny == 1:
        L = m
    else:
        raise ValueError(_bcast_msg(m, ny))
    terms = np.empty(L, y.dtype)
    for i in range(L):
        j = i if ny > 1 else 0
        k = i if m > 1 else 0
        if has_x:
            d = x[k + 1] - x[k]
        else:
            d = dx
        terms[i] = d * (y[j + 1] + y[j]) / 2.0
    return _psum(terms)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Composite trapezoidal rule over samples.

    Takes NO callback of either style: it integrates *samples*, not a
    function.  For a function use :func:`fixed_quad` or
    :func:`~scijit.integrate.quad`.

    Parameters
    ----------
    y : array_like
        Samples to integrate, of any rank.  A list or a tuple is converted,
        an integer or boolean array is promoted to float64, and a complex
        one stays complex.
    x : array_like or None, optional
        Sample positions, which may be non-uniform.  Either 1-D along
        ``axis``, or of ``y``'s shape.  ``None`` (the default) means equal
        spacing ``dx``.
    dx : float, optional
        Spacing used when ``x`` is None.  Default 1.0.
    axis : int, optional
        Axis to integrate along.  Default -1, the last axis.

    Returns
    -------
    total : float or ndarray
        The integral.  A scalar for a 1-D ``y``, otherwise an array of
        ``y``'s rank less one.  Fewer than two samples gives 0.0.

    Raises
    ------
    IndexError
        ``axis`` outside ``y``'s rank.
    ValueError
        The spacings and the sample pairs do not broadcast along ``axis``,
        with numpy's own message; ``x`` neither 1-D nor of ``y``'s shape.

    See Also
    --------
    scipy.integrate.trapezoid : The scipy routine this mirrors.

    Notes
    -----
    The number of spacings and the number of sample pairs do not have to
    agree.  The spacings ``x[1:] - x[:-1]`` are multiplied against the pairs
    ``y[1:] + y[:-1]`` and broadcast, so either may be 1 and repeat against
    the other: ``len(x) == 2`` with ``len(y) == 4`` integrates with one
    repeated spacing, and ``len(y) == 1`` with ``len(x) == 3`` raises.

    ``np.trapezoid`` computes the same value inside ``@njit`` but takes no
    ``axis`` argument.

    Pure ``@njit``, no state, so **prange-safe**.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> x = np.linspace(0.0, np.pi, 65)
    >>> @njit
    ... def run(x):
    ...     return si.trapezoid(np.sin(x), x)
    >>> run(x)
    1.9995983886400375
    """
    yc = _samples(y)
    nd = yc.ndim
    ax = axis + nd if axis < 0 else axis
    if ax < 0 or ax >= nd:
        raise IndexError(_TZ_AXIS_MSG)
    if x is not None and np.ndim(x) != 1 and np.shape(x) != yc.shape:
        raise ValueError(_XSHAPE_MSG)
    if nd == 1:
        xr = _positions(x) if x is not None else _EMPTY_X
        return _trapz_1d(yc, xr, x is not None, dx)
    flat, xf, M, n = _nd_rows(yc, x, ax)
    res = np.empty(M, yc.dtype)
    for i in range(M):
        if xf is None:
            xr = _EMPTY_X
        elif xf.ndim == 1:
            xr = xf
        else:
            xr = xf[i]
        res[i] = _trapz_1d(flat[i], xr, xf is not None, dx)
    return res.reshape(yc.shape[:ax] + yc.shape[ax + 1:])


@overload(trapezoid)
def _trapezoid_ovl(y, x=None, dx=1.0, axis=-1):
    """Compiled body for :func:`trapezoid`, one per rank of ``y``."""
    if not isinstance(y, types.Array):
        return None
    xk = _x_kind(x, y.ndim)
    if xk is None:
        raise TypingError(_XSHAPE_MSG)
    cast = _cast_of(y)
    if y.ndim == 1:
        CAST = np.complex128 if cast == 'np.complex128' else np.float64
        if xk == 0:
            def impl(y, x=None, dx=1.0, axis=-1):
                if axis != -1 and axis != 0:
                    raise IndexError(_TZ_AXIS_MSG)
                yc = np.ascontiguousarray(np.asarray(y)).astype(CAST)
                return _trapz_1d(yc, _EMPTY_X, False, dx)
        else:
            def impl(y, x=None, dx=1.0, axis=-1):
                if axis != -1 and axis != 0:
                    raise IndexError(_TZ_AXIS_MSG)
                yc = np.ascontiguousarray(np.asarray(y)).astype(CAST)
                xc = np.ascontiguousarray(np.asarray(x)).astype(np.float64)
                return _trapz_1d(yc, xc, True, dx)
        return impl
    return _nd_body('y, x=None, dx=1.0, axis=-1',
                    '_trapz_1d({y}, {x}, dx)',
                    y.ndim, xk, cast, None,
                    {'np': np, '_AXIS_MSG': _TZ_AXIS_MSG,
                     '_trapz_1d': _trapz_1d})

# ---------------------------------------------------------------------
# roots_legendre  (Golub-Welsch)
# ---------------------------------------------------------------------
@njit
def _roots_legendre_core(n):
    """Golub-Welsch Gauss-Legendre nodes and weights. THE algorithm; both
    entry points below only slice its return."""
    if n < 1:
        raise ValueError("n must be a positive integer.")
    J = np.zeros((n, n))
    for k in range(1, n):
        b = k / np.sqrt(4.0 * k * k - 1.0)
        J[k - 1, k] = b
        J[k, k - 1] = b
    w, v = np.linalg.eigh(J)
    nodes = w.copy()
    weights = np.empty(n)
    for i in range(n):
        weights[i] = 2.0 * v[0, i] * v[0, i]
    return nodes, weights


# The integral of the Gauss-Legendre weight function w(x) = 1 over
# [-1, 1].  scipy computes it as 2**(a+b+1) * B(a+1, b+1) with
# a = b = 0, which is 2.0 for every n; measured equal to 2.0 exactly at
# n = 1, 2, 5, 12, 32.
_LEGENDRE_MU = 2.0


def roots_legendre(n, mu=False):
    """Gauss-Legendre nodes and weights on ``[-1, 1]``.

    Lives in ``integrate`` rather than ``special`` because
    :func:`fixed_quad` needs it; it is re-exported, not duplicated, by
    ``scijit.special``.

    Takes NO callback of either style.

    Parameters
    ----------
    n : int
        Number of quadrature points; must be >= 1.  The rule is exact
        for polynomials up to degree ``2n - 1``.
    mu : bool, optional
        When true, also return the integral of the weight function.
        Default False.  Inside ``@njit`` it must be a compile-time
        constant (a literal, or omitted): it selects the number of
        return values, so a runtime variable raises ``TypingError``.

    Returns
    -------
    nodes : float64 array, shape (n,)
        Abscissae in ``[-1, 1]``, ascending.
    weights : float64 array, shape (n,)
        Corresponding weights, summing to 2.
    mu : float
        Present only when ``mu=True``.  The integral of the weight
        function ``w(x) = 1`` over ``[-1, 1]``, so 2.0 for every ``n``.

    Raises
    ------
    ValueError
        ``n < 1``.

    See Also
    --------
    scipy.special.roots_legendre : The scipy routine this mirrors.

    Notes
    -----
    Golub-Welsch: eigen-decompose the symmetric tridiagonal Jacobi
    matrix with zero diagonal and off-diagonal
    ``beta_k = k / sqrt(4k^2 - 1)``.  Nodes are the eigenvalues,
    weights are ``2 * (first component of the normalized
    eigenvector)^2``.  scipy instead refines Newton iterations on the
    Legendre polynomial, so agreement is to rounding rather than bit
    for bit.  Measured against ``scipy.special.roots_legendre``:
    nodes 0.0 / 1.1e-16 / 2.2e-16 / 4.4e-16 and weights 0.0 / 3.3e-16 /
    1.7e-15 / 4.5e-15 at ``n`` = 1, 5, 12, 32.

    ``mu`` must be a compile-time constant inside ``@njit``.  It selects
    the number of return values, which numba fixes when the caller
    compiles, so a runtime variable raises ``TypingError``.

    Pure ``@njit`` on top of ``np.linalg.eigh`` -- **prange-safe**.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> from scijit.integrate._quadrature import roots_legendre
    >>> @njit
    ... def run():
    ...     return roots_legendre(3)
    >>> nodes, weights = run()
    >>> nodes
    array([-0.77459667,  0.        ,  0.77459667])
    >>> weights
    array([0.55555556, 0.88888889, 0.55555556])
    """
    x, w = _roots_legendre_core(n)
    if mu:
        return x, w, _LEGENDRE_MU
    return x, w


@overload(roots_legendre, prefer_literal=True)
def _roots_legendre_ovl(n, mu=False):
    """Compiled body for :func:`roots_legendre`, one per value of ``mu``.

    ``mu`` selects the number of return values, so it cannot be a runtime
    variable.  Declining the overload is how that surfaces to the caller, as
    a ``TypingError``.
    """
    lit = _lit_bool(mu)
    if lit is None:
        return None                     # runtime flag -> TypingError, by design
    if lit:
        def impl(n, mu=False):
            x, w = _roots_legendre_core(n)
            return x, w, _LEGENDRE_MU
    else:
        def impl(n, mu=False):
            x, w = _roots_legendre_core(n)
            return x, w
    return impl


# ---------------------------------------------------------------------
# fixed_quad
# ---------------------------------------------------------------------
_FQ_INF_MSG = ('Gaussian quadrature is only available for '
               'finite limits.')
_FQ_CALLABLE_MSG = (
    'fixed_quad: func must be a plain @njit function called as '
    'func(x_array, *args). A @cfunc address is an int and is not callable.')
_FQ_ARGS_MSG = (
    'fixed_quad: inside @njit, args must be a tuple. A list or an array is '
    'splatted by the interpreter and has no compiled equivalent.')
_FQ_NONE_ARGS_MSG = 'Value after * must be an iterable, not NoneType'


def _fq_sum(w, fy):
    """``sum(w * fy, axis=-1)``, in numpy's summation order.

    An integrand may return shape ``(..., len(x))``, so the reduction is
    over the last axis and the result carries the leading axes.
    """
    return np.sum(w * fy, axis=-1)


@overload(_fq_sum)
def _fq_sum_ovl(w, fy):
    """Compiled body for :func:`_fq_sum`, one per rank of the return."""
    if not isinstance(fy, types.Array):
        raise TypingError('fixed_quad: func must return an array')
    if fy.ndim == 1:
        def impl(w, fy):
            return _psum(w * fy)
        return impl
    nd = fy.ndim

    def impl(w, fy):
        z = np.ascontiguousarray(w * fy)
        n = z.shape[nd - 1]
        M = z.size // n
        flat = z.reshape(M, n)
        res = np.empty(M, z.dtype)
        for i in range(M):
            res[i] = _psum(flat[i])
        return res.reshape(fy.shape[:-1])
    return impl


def fixed_quad(func, a, b, args=(), n=5):
    """Definite integral by fixed-order Gauss-Legendre quadrature.

    Non-adaptive: it evaluates ``func`` at exactly ``n`` points and returns,
    with no error estimate and no refinement.  Use
    :func:`~scijit.integrate.quad` when the integrand is not smooth or the
    accuracy matters.

    ``func`` is a plain ``@njit`` function passed as a first-class argument,
    and it is **vectorized**: it receives the whole array of abscissae at
    once::

        @njit
        def f(x):                  # x is a 1-D float64 array
            return np.exp(-x * x)  # an array of the same length

        val, _ = fixed_quad(f, 0.0, 1.0, (), 5)

    Parameters bound to the integrand travel in ``args`` and arrive as
    separate arguments::

        @njit
        def g(x, alpha, beta):
            return beta * np.exp(-alpha * x * x)

        val, _ = fixed_quad(g, 0.0, 1.0, (2.0, 3.0), 5)

    Parameters
    ----------
    func : @njit function
        Integrand, vectorized over its first argument, called as
        ``func(x_array, *args)``.  A scalar-in/scalar-out function will not
        work.  A vector-valued integrand returns shape ``(..., len(x))`` and
        the result carries the leading axes.
    a, b : float
        Finite integration limits.  Infinite limits raise ``ValueError``.
    args : tuple, optional
        Extra arguments, splatted into ``func``.  Default ``()``.
    n : int, optional
        Number of Gauss-Legendre points, so the rule is exact for
        polynomials up to degree ``2n - 1``.  Default 5.

    Returns
    -------
    val : float or ndarray
        The integral.
    none : None
        Always ``None``, present as the second element so a caller can
        unpack two names.

    Raises
    ------
    TypeError
        ``func`` not callable; ``args`` not iterable.
    ValueError
        ``n < 1``, from the Gauss-Legendre roots; infinite ``a`` or ``b``.
        The roots are resolved first, so ``n < 1`` is reported ahead of an
        infinite limit.

    See Also
    --------
    scipy.integrate.fixed_quad : The scipy routine this mirrors.

    Notes
    -----
    ``args`` must be a tuple inside ``@njit``.  From the interpreter a list
    or an array is also accepted and splatted.

    Pure ``@njit``, no state, so **prange-safe**.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def f(x, alpha):                # x is the whole abscissa array
    ...     return alpha * np.exp(-x * x)
    >>> @njit
    ... def run():
    ...     return si.fixed_quad(f, 0.0, 1.0, (2.0,), 5)
    >>> val, none = run()
    >>> val
    1.4936482535324966
    >>> none is None
    True
    """
    if not callable(func):
        raise TypeError(_FQ_CALLABLE_MSG)
    if args is None:
        raise TypeError(_FQ_NONE_ARGS_MSG)
    x, w = _roots_legendre_core(n)             # scipy resolves the roots first
    af = float(a)
    bf = float(b)
    if math.isinf(af) or math.isinf(bf):
        raise ValueError(_FQ_INF_MSG)
    y = (bf - af) * (x + 1.0) / 2.0 + af
    return (bf - af) / 2.0 * np.sum(w * func(y, *args), axis=-1), None


@overload(fixed_quad)
def _fixed_quad_ovl(func, a, b, args=(), n=5):
    """Compiled body for :func:`fixed_quad`.

    ``args`` is splatted into the call exactly as the interpreter splats it,
    so nothing about the integrand's arity has to be sniffed and a
    parameterised integrand keeps scipy's own shape.
    """
    if not isinstance(func, types.Dispatcher):
        raise TypingError(_FQ_CALLABLE_MSG)
    if _is_none(args):
        raise TypingError(_FQ_NONE_ARGS_MSG)
    # an OMITTED default reaches a chooser as the raw python value, so the
    # plain `tuple` belongs in this test beside the two numba types
    if not isinstance(args, (types.BaseTuple, types.Omitted, tuple)):
        raise TypingError(_FQ_ARGS_MSG)

    def impl(func, a, b, args=(), n=5):
        x, w = _roots_legendre_core(n)
        af = np.float64(a)
        bf = np.float64(b)
        if math.isinf(af) or math.isinf(bf):
            raise ValueError(_FQ_INF_MSG)
        y = (bf - af) * (x + 1.0) / 2.0 + af
        return (bf - af) / 2.0 * _fq_sum(w, func(y, *args)), None
    return impl



# ---------------------------------------------------------------------
# simpson
# ---------------------------------------------------------------------
_SQUEEZE_MSG = ('cannot select an axis to squeeze out which has size not '
                'equal to one')
_EMPTY_IDX_MSG = 'index -1 is out of bounds for axis 0 with size 0'


@njit
def _div0(a, b):
    """``a / b`` elementwise, and 0.0 wherever ``b`` is zero.

    scipy guards every division a zero spacing can make singular with
    ``xpx.apply_where(<den> != 0, ..., xp.divide, fill_value=0.)``.
    """
    out = np.empty(b.shape[0], np.float64)
    for i in range(b.shape[0]):
        if b[i] != 0.0:
            out[i] = a[i] / b[i]
        else:
            out[i] = 0.0
    return out


@njit
def _div0s(a, b):
    """:func:`_div0` for one value."""
    if b != 0.0:
        return a / b
    return 0.0


@njit
def _basic_simpson_uniform(y, start, stop, dx):
    """Composite Simpson over the paired intervals of ``y[start:stop+2]`` at
    constant spacing.  The inner loop of :func:`simpson`, split out so the
    even-``N`` correction can be applied to what is left over."""
    if start >= stop:
        return y.dtype.type(0)
    y0 = y[start:stop:2]
    y1 = y[start + 1:stop + 1:2]
    y2 = y[start + 2:stop + 2:2]
    return dx / 3.0 * _psum(y0 + 4.0 * y1 + y2)


@njit
def _basic_simpson_nonuniform(y, start, stop, x):
    """:func:`_basic_simpson_uniform` for samples at arbitrary positions.

    Kept as a separate function rather than a branch because the two use
    different weights throughout, not a different spacing in one place.
    """
    if start >= stop:
        return y.dtype.type(0)
    h = np.diff(x)
    h0 = h[start:stop:2]
    h1 = h[start + 1:stop + 1:2]
    y0 = y[start:stop:2]
    y1 = y[start + 1:stop + 1:2]
    y2 = y[start + 2:stop + 2:2]
    hsum = h0 + h1
    hprod = h0 * h1
    h0divh1 = _div0(h0, h1)
    ones = np.ones(h0divh1.shape[0])
    tmp = hsum / 6.0 * (y0 * (2.0 - _div0(ones, h0divh1)) +
                        y1 * (hsum * _div0(hsum, hprod)) +
                        y2 * (2.0 - h0divh1))
    return _psum(tmp)


@njit
def _simpson_1d(y, x, has_x, dx):
    """Composite Simpson over one row of samples."""
    N = y.shape[0]
    if has_x and x.shape[0] != N:
        raise ValueError(_XLEN_MSG)
    if N == 0:
        if has_x:
            raise ValueError(_SQUEEZE_MSG)
        raise IndexError(_EMPTY_IDX_MSG)
    if N % 2 == 0:
        if N == 2:
            if has_x:
                last_dx = x[N - 1] - x[N - 2]
            else:
                last_dx = dx
            return 0.5 * last_dx * (y[N - 1] + y[N - 2])
        if has_x:
            result = _basic_simpson_nonuniform(y, 0, N - 3, x)
            h0 = x[N - 2] - x[N - 3]
            h1 = x[N - 1] - x[N - 2]
        else:
            result = _basic_simpson_uniform(y, 0, N - 3, dx)
            h0 = dx
            h1 = dx
        alpha = _div0s(2.0 * h1 * h1 + 3.0 * h0 * h1, 6.0 * (h1 + h0))
        beta = _div0s(h1 * h1 + 3.0 * h0 * h1, 6.0 * h0)
        eta = _div0s(h1 * h1 * h1, 6.0 * h0 * (h0 + h1))
        return result + (alpha * y[N - 1] + beta * y[N - 2] - eta * y[N - 3])
    if has_x:
        return _basic_simpson_nonuniform(y, 0, N - 2, x)
    return _basic_simpson_uniform(y, 0, N - 2, dx)


def simpson(y, x=None, *, dx=1.0, axis=-1):
    """Composite Simpson's rule over samples.

    Takes NO callback of either style: it integrates *samples*.

    Parameters
    ----------
    y : array_like
        Samples to integrate, of any rank.  An integer or boolean array is
        promoted to float64 and a complex one stays complex.  Copied to a
        contiguous buffer, so strided views are safe.
    x : array_like or None, optional
        Sample positions, which may be non-uniform.  Either 1-D along
        ``axis``, or of ``y``'s rank.  ``None`` (the default) means equal
        spacing ``dx``.
    dx : float, optional
        Spacing used when ``x`` is None.  Default 1.0.  Keyword-only, from
        both entry points.
    axis : int, optional
        Axis to integrate along.  Default -1, the last axis.  Keyword-only.

    Returns
    -------
    total : float or ndarray
        The integral.  A scalar for a 1-D ``y``, otherwise an array of
        ``y``'s rank less one.

    Raises
    ------
    IndexError
        ``axis`` outside ``y``'s rank; or no samples along ``axis`` with no
        ``x``.
    ValueError
        ``x`` neither 1-D nor of ``y``'s rank; ``x`` and ``y`` of different
        lengths along ``axis``; no samples along ``axis`` with an ``x``.

    See Also
    --------
    scipy.integrate.simpson : The scipy routine this mirrors.

    Notes
    -----
    An odd ``N`` uses plain composite Simpson.  An even ``N``, where the
    intervals cannot be paired, uses the Cartwright correction on the last
    interval; ``N == 2`` degenerates to the trapezoid.

    A zero spacing leaves six divisions singular, and each substitutes 0.0
    for the quotient.  A repeated sample is not rejected, so this is the path
    a non-increasing ``x`` reaches.

    Pure ``@njit``, no state, so **prange-safe**.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> x = np.linspace(0.0, np.pi, 65)
    >>> @njit
    ... def run(x):
    ...     return si.simpson(np.sin(x), x)
    >>> run(x)
    2.000000064530002
    """
    yc = _samples(y)
    nd = yc.ndim
    yc.shape[axis]                            # a bad axis raises IndexError
    ax = axis + nd if axis < 0 else axis
    if x is not None and np.ndim(x) != 1 and np.ndim(x) != nd:
        raise ValueError(_XSHAPE_MSG)
    if nd == 1:
        xr = _positions(x) if x is not None else _EMPTY_X
        return _simpson_1d(yc, xr, x is not None, dx)
    flat, xf, M, n = _nd_rows(yc, x, ax)
    res = np.empty(M, yc.dtype)
    for i in range(M):
        if xf is None:
            xr = _EMPTY_X
        elif xf.ndim == 1:
            xr = xf
        else:
            xr = xf[i]
        res[i] = _simpson_1d(flat[i], xr, xf is not None, dx)
    return res.reshape(yc.shape[:ax] + yc.shape[ax + 1:])


@overload(simpson, prefer_literal=True)
def _simpson_ovl(y, x=None, *, dx=1.0, axis=-1):
    """Compiled body for :func:`simpson`, one per rank of ``y``.

    The ``*`` belongs in this signature and not in the body's: numba binds
    the call against the chooser, so a positional ``dx`` is refused here as
    it is from the interpreter, while an impl carrying the marker fails to
    lower.
    """
    if not isinstance(y, types.Array):
        return None
    xk = _x_kind(x, y.ndim)
    if xk is None:
        raise TypingError(_XSHAPE_MSG)
    cast = _cast_of(y)
    if y.ndim == 1:
        CAST = np.complex128 if cast == 'np.complex128' else np.float64
        if xk == 0:
            def impl(y, x=None, dx=1.0, axis=-1):
                if axis != -1 and axis != 0:
                    raise IndexError(_AXIS_MSG)
                yc = np.ascontiguousarray(np.asarray(y)).astype(CAST)
                return _simpson_1d(yc, _EMPTY_X, False, dx)
        else:
            def impl(y, x=None, dx=1.0, axis=-1):
                if axis != -1 and axis != 0:
                    raise IndexError(_AXIS_MSG)
                yc = np.ascontiguousarray(np.asarray(y)).astype(CAST)
                xc = np.ascontiguousarray(np.asarray(x)).astype(np.float64)
                return _simpson_1d(yc, xc, True, dx)
        return impl
    return _nd_body('y, x=None, dx=1.0, axis=-1',
                    '_simpson_1d({y}, {x}, dx)',
                    y.ndim, xk, cast, None,
                    {'np': np, '_AXIS_MSG': _AXIS_MSG,
                     '_simpson_1d': _simpson_1d})



# ---------------------------------------------------------------------
# cumulative_simpson
# ---------------------------------------------------------------------
_CS_XSHAPE_MSG = ('If given, shape of `x` must be the same as `y` or 1-D '
                  'with the same length as `y` along `axis`.')
_CS_INCR_MSG = 'Input x must be strictly increasing.'
_CS_DX_MSG = ('If provided, `dx` must either be a scalar or have the same '
              'shape as `y` but with only 1 point along `axis`.')
_CS_INIT_MSG = ('If provided, `initial` must either be a scalar or have the '
                'same shape as `y` but with only 1 point along `axis`.')


@njit
def _cs_axis_msg(axis, nd):
    """scipy's message for an axis its ``y`` does not have."""
    return ('`axis=' + str(axis) + '` is not valid for `y` with `y.ndim='
            + str(nd) + '`.')


@njit
def _simpson_subintegrals(y, dxa, unequal):
    """Simpson integral over every h1 sub-interval (eqns (8)/(10) of
    Cartwright). Reverse the inputs to get the h2 sub-integrals."""
    m = y.shape[0] - 2
    out = np.empty(m, y.dtype)
    for i in range(m):
        f1 = y[i]
        f2 = y[i + 1]
        f3 = y[i + 2]
        if unequal:
            x21 = dxa[i]
            x32 = dxa[i + 1]
            x31 = x21 + x32
            x21_x31 = x21 / x31
            x21_x32 = x21 / x32
            p = x21_x31 * x21_x32
            out[i] = x21 / 6.0 * ((3.0 - x21_x31) * f1 +
                                  (3.0 + p + x21_x31) * f2 +
                                  (-p) * f3)
        else:
            d = dxa[i]
            out[i] = d / 3.0 * (5.0 * f1 / 4.0 + 2.0 * f2 - f3 / 4.0)
    return out


@njit
def _cumulatively_sum_simpson(y, dxa, unequal):
    """Running Simpson integral, by interleaving two sets of sub-integrals.

    Simpson needs three samples per estimate, so a running integral cannot
    use one direction alone.  This takes the forward sub-integrals for even
    positions and the reversed ones for odd, which is scipy's construction,
    then accumulates.
    """
    sub_h1 = _simpson_subintegrals(y, dxa, unequal)
    yr = np.ascontiguousarray(y[::-1])
    dr = np.ascontiguousarray(dxa[::-1])
    sub_h2 = _simpson_subintegrals(yr, dr, unequal)[::-1]
    m = sub_h1.shape[0]
    sub = np.empty(m + 1, y.dtype)
    for i in range(m):
        if i % 2 == 0:
            sub[i] = sub_h1[i]
        else:
            sub[i] = sub_h2[i - 1]
    sub[m] = sub_h2[m - 1]
    return np.cumsum(sub)


@njit
def _cumsimp_1d(y, x, has_x, dxv, off, init_v):
    """Running Simpson over one row of samples.

    ``off`` is 1 when a leading value is prepended.  scipy adds ``initial``
    to every element as well, which is what the prepended value means here.
    """
    n = y.shape[0]
    if n < 3:
        res = _cumtrapz_1d(y, x, has_x, dxv, 0, 0.0)
    else:
        if has_x:
            dxa = np.diff(x)
            for i in range(dxa.shape[0]):
                if dxa[i] <= 0.0:
                    raise ValueError(_CS_INCR_MSG)
            res = _cumulatively_sum_simpson(y, dxa, True)
        else:
            dxa = np.empty(n - 1)
            for i in range(n - 1):
                dxa[i] = dxv
            res = _cumulatively_sum_simpson(y, dxa, False)
    if off == 0:
        return res
    out = np.empty(res.shape[0] + 1, y.dtype)
    iv = y.dtype.type(init_v)
    out[0] = iv
    for i in range(res.shape[0]):
        out[i + 1] = res[i] + iv
    return out


def _cs_perrow(v, shape, ax, M, msg):
    """A per-row value from a scipy ``dx`` or ``initial``.

    A scalar applies to every row; an array must carry ``y``'s shape with
    length one along ``axis``, and its rows line up with ``y``'s.
    """
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 0:
        return np.full(M, float(a))
    want = list(shape)
    want[ax] = 1
    if a.shape != tuple(want):
        raise ValueError(msg)
    return np.ascontiguousarray(np.moveaxis(a, ax, -1)).reshape(M)


def cumulative_simpson(y, *, x=None, dx=1.0, axis=-1, initial=None):
    """Running integral by composite Simpson's 1/3 rule.

    Takes NO callback of either style: it integrates *samples*.

    Parameters
    ----------
    y : array_like
        Samples to integrate, of any rank.  At least one point is required
        along ``axis``.  Two or fewer fall back to
        :func:`cumulative_trapezoid`.
    x : array_like or None, optional
        Sample positions, **strictly increasing** along ``axis``.  Either of
        ``y``'s shape, or 1-D with ``y``'s length along ``axis``.  ``None``
        (the default) means equal spacing ``dx``.  Keyword-only.
    dx : float or array_like, optional
        Spacing used when ``x`` is None.  Either a scalar or an array of
        ``y``'s shape with length one along ``axis``.  Default 1.0.
        Keyword-only.
    axis : int, optional
        Axis to integrate along.  Default -1, the last axis.  Keyword-only.
    initial : float or array_like or None, optional
        Value to prepend, and to add to every other element.  Either a
        scalar or an array of ``y``'s shape with length one along ``axis``.
        ``None`` (the default) omits it.  Keyword-only.  The addition is a
        common surprise, since ``initial`` shifts the whole curve rather
        than only inserting a point.

    Returns
    -------
    res : ndarray
        Running integral, of ``y``'s rank.  Along ``axis`` its length is
        ``N - 1``, the integral evaluated at the samples after the first, or
        ``N`` when ``initial`` is given.

    Raises
    ------
    ValueError
        ``axis`` outside ``y``'s rank; no samples along ``axis``; ``x``
        neither of ``y``'s shape nor 1-D along ``axis``; ``x`` not strictly
        increasing; ``dx`` or ``initial`` neither a scalar nor of ``y``'s
        shape with length one along ``axis``.

    See Also
    --------
    scipy.integrate.cumulative_simpson : The scipy routine this mirrors.

    Notes
    -----
    Pure ``@njit``, no state, so **prange-safe**.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> x = np.linspace(0.0, np.pi, 65)
    >>> @njit
    ... def run(x):
    ...     return si.cumulative_simpson(np.sin(x), x=x, initial=0.0)[-1]
    >>> run(x)
    2.0000000645300022
    """
    yc = _samples(y)
    nd = yc.ndim
    ax = axis + nd if axis < 0 else axis
    if ax < 0 or ax >= nd:
        raise ValueError(_cs_axis_msg(axis, nd))
    n = yc.shape[ax]
    if n == 0:
        raise ValueError(_ONEPT_MSG)
    if n >= 3 and x is not None:
        xs = np.shape(x)
        if not (xs == yc.shape or (len(xs) == 1 and xs[0] == n)):
            raise ValueError(_CS_XSHAPE_MSG)

    flat, xf, M, _n = _nd_rows(yc, x if n >= 3 else None, ax)
    if n < 3 and x is not None:
        # scipy hands the short case straight to cumulative_trapezoid, whose
        # own guards and messages are the ones a caller then sees
        _, xf, _, _ = _nd_rows(yc, x, ax)
    dxr = _cs_perrow(dx, yc.shape, ax, M, _CS_DX_MSG)
    off = 0 if initial is None else 1
    ivr = (np.zeros(M) if initial is None
           else _cs_perrow(initial, yc.shape, ax, M, _CS_INIT_MSG))

    res = np.empty((M, n - 1 + off), yc.dtype)
    for i in range(M):
        if xf is None:
            xr = _EMPTY_X
        elif xf.ndim == 1:
            xr = xf
        else:
            xr = xf[i]
        res[i] = _cumsimp_1d(flat[i], xr, xf is not None, dxr[i], off, ivr[i])
    out = res.reshape(yc.shape[:ax] + yc.shape[ax + 1:] + (n - 1 + off,))
    return np.ascontiguousarray(np.moveaxis(out, -1, ax))


@overload(cumulative_simpson, prefer_literal=True)
def _cumulative_simpson_ovl(y, *, x=None, dx=1.0, axis=-1, initial=None):
    """Compiled body for :func:`cumulative_simpson`, one per rank of ``y``.

    The ``*`` belongs in this signature and not in the body's: numba binds
    the call against the chooser, so scipy's keyword-only parameters are
    keyword-only here too, while an impl carrying the marker fails to lower.
    """
    if not isinstance(y, types.Array):
        return None
    nd = y.ndim
    xk = _x_kind(x, nd)
    if xk is None:
        raise TypingError(_CS_XSHAPE_MSG)
    dx_arr = isinstance(dx, types.Array)
    iv_none = _is_none(initial)
    iv_arr = isinstance(initial, types.Array)
    if dx_arr and dx.ndim != nd:
        raise TypingError(_CS_DX_MSG)
    if iv_arr and initial.ndim != nd:
        raise TypingError(_CS_INIT_MSG)
    cast = _cast_of(y)

    L = ['def impl(y, x=None, dx=1.0, axis=-1, initial=None):',
         '    yc = np.ascontiguousarray(np.asarray(y)).astype(%s)' % cast,
         '    ax = axis + %d if axis < 0 else axis' % nd,
         '    if ax < 0 or ax >= %d:' % nd,
         '        raise ValueError(_cs_axis_msg(axis, %d))' % nd,
         '    ys = yc.shape',
         '    n = ys[ax]',
         '    if n == 0:',
         '        raise ValueError(_ONEPT_MSG)',
         '    ym = np.ascontiguousarray(np.moveaxis(yc, ax, -1))',
         '    M = 1']
    for k in range(nd - 1):
        L.append('    M *= ym.shape[%d]' % k)
    L.append('    flat = ym.reshape(M, n)')
    if xk == 0:
        L.append('    xr = np.empty(0, np.float64)')
        row = 'xr, False'
    elif xk == 1:
        L += ['    xr = np.ascontiguousarray(np.asarray(x)).astype(np.float64)',
              '    if n >= 3 and xr.shape[0] != n:',
              '        raise ValueError(_CS_XSHAPE_MSG)']
        row = 'xr, True'
    else:
        L += ['    if n >= 3:',
              '        for _k in range(%d):' % nd,
              '            if x.shape[_k] != ys[_k]:',
              '                raise ValueError(_CS_XSHAPE_MSG)',
              '    xm = np.ascontiguousarray(np.moveaxis('
              'np.ascontiguousarray(np.asarray(x)).astype(np.float64), '
              'ax, -1))',
              '    xf = xm.reshape(M, n)']
        row = 'xf[i], True'
    if dx_arr:
        L += ['    for _k in range(%d):' % nd,
              '        if dx.shape[_k] != (1 if _k == ax else ys[_k]):',
              '            raise ValueError(_CS_DX_MSG)',
              '    dxr = np.ascontiguousarray(np.moveaxis('
              'dx.astype(np.float64), ax, -1)).reshape(M)']
    else:
        L.append('    dxr = np.full(M, np.float64(dx))')
    if iv_none:
        L += ['    off = 0', '    ivr = np.zeros(M)']
    elif iv_arr:
        L += ['    off = 1',
              '    for _k in range(%d):' % nd,
              '        if initial.shape[_k] != (1 if _k == ax else ys[_k]):',
              '            raise ValueError(_CS_INIT_MSG)',
              '    ivr = np.ascontiguousarray(np.moveaxis('
              'initial.astype(np.float64), ax, -1)).reshape(M)']
    else:
        L += ['    off = 1', '    ivr = np.full(M, np.float64(initial))']
    L += ['    res = np.empty((M, n - 1 + off), %s)' % cast,
          '    for i in range(M):',
          '        res[i] = _cumsimp_1d(flat[i], %s, dxr[i], off, ivr[i])'
          % row]
    L += _axis_lines(nd, ['n - 1 + off'], '    ')
    L.append('    return np.ascontiguousarray(np.moveaxis(out, -1, ax))')
    ns = {'np': np, '_cumsimp_1d': _cumsimp_1d, '_cs_axis_msg': _cs_axis_msg,
          '_ONEPT_MSG': _ONEPT_MSG, '_CS_XSHAPE_MSG': _CS_XSHAPE_MSG,
          '_CS_DX_MSG': _CS_DX_MSG, '_CS_INIT_MSG': _CS_INIT_MSG}
    exec('\n'.join(L), ns)                                   # noqa: S102
    return ns['impl']

# ---------------------------------------------------------------------
# romb
# ---------------------------------------------------------------------
_ROMB_MSG = ('Number of samples must be one plus a '
             'non-negative power of 2.')
_ROMB_NDMSG = ('*** Printing table only supported for integrals'
               ' of a single data set.')
_ROMB_TITLE = 'Richardson Extrapolation Table for Romberg Integration'


@njit
def _romb_table(y, dx):
    """The Richardson extrapolation table for one row of samples."""
    Nsamps = y.shape[0]
    Ninterv = Nsamps - 1
    n = 1
    k = 0
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError(_ROMB_MSG)

    R = np.zeros((k + 1, k + 1), y.dtype)
    h = Ninterv * np.float64(dx)
    R[0, 0] = (y[0] + y[Nsamps - 1]) / 2.0 * h
    start = Ninterv
    stop = Ninterv
    step = Ninterv
    for i in range(1, k + 1):
        start = start >> 1
        m = 0
        idx = start
        while idx < stop:
            m += 1
            idx += step
        gath = np.empty(m, y.dtype)
        idx = start
        for t in range(m):
            gath[t] = y[idx]
            idx += step
        step = step >> 1
        R[i, 0] = 0.5 * (R[i - 1, 0] + h * _psum(gath))
        for j in range(1, i + 1):
            prev = R[i, j - 1]
            R[i, j] = prev + (prev - R[i - 1, j - 1]) / ((1 << (2 * j)) - 1)
        h /= 2.0
    return R, k


@njit
def _romb_1d(y, x, has_x, dx):
    """Romberg integral of one row.  ``x`` is unused and is there so the
    N-D driver can hand every core the same arguments."""
    R, k = _romb_table(y, dx)
    return R[k, k]


def _romb_print(R, k, precis, width):
    """scipy's Richardson table, formatted as scipy formats it."""
    formstr = "%%%d.%df" % (width, precis)
    print(_ROMB_TITLE, "=" * len(_ROMB_TITLE), sep="\n", end="\n")
    for i in range(k + 1):
        for j in range(i + 1):
            print(formstr % R[i, j], end=" ")
        print()
    print("=" * len(_ROMB_TITLE))


@njit
def _romb_show(R, k, precis, width):
    """:func:`_romb_print` from compiled code.

    The block lives in its own module-level ``@njit`` because lowering an
    ``objmode`` block pickles the enclosing function.
    """
    with objmode():
        _romb_print(R, k, precis, width)


def _romb_showargs(show):
    """scipy's ``(precision, width)``, from a ``show`` that may be a bool."""
    try:
        precis = show[0]
    except (TypeError, IndexError):
        precis = 5
    try:
        width = show[1]
    except (TypeError, IndexError):
        width = 8
    return precis, width


def _romb_show_flag(show):
    """Whether ``show`` is on at TYPING time, or None when it is a runtime
    variable and no compiled body can be chosen."""
    if isinstance(show, types.Omitted):
        show = show.value
    if isinstance(show, types.BaseTuple):
        return True
    if isinstance(show, tuple):
        return bool(show)
    if isinstance(show, (bool, int, np.integer)):
        return bool(show)
    if isinstance(show, types.BooleanLiteral):
        return show.literal_value
    if isinstance(show, types.IntegerLiteral):
        return bool(show.literal_value)
    return None


def _romb_show_len(show):
    """How many values a ``show`` sequence carries, 0 for a bool."""
    if isinstance(show, types.Omitted):
        show = show.value
    if isinstance(show, types.BaseTuple):
        return len(show.types)
    if isinstance(show, tuple):
        return len(show)
    return 0


def romb(y, dx=1.0, axis=-1, show=False):
    """Romberg integration of equally-spaced samples.

    Repeated Richardson extrapolation of the trapezoid rule.  Very accurate
    on smooth data, but the sample count is constrained.

    Takes NO callback of either style: it integrates *samples*.

    Parameters
    ----------
    y : array_like
        Samples on an equally-spaced grid.  The length along ``axis`` must
        be one plus a non-negative power of two, otherwise ``ValueError``.
    dx : float, optional
        Sample spacing, a scalar.  Default 1.0.
    axis : int, optional
        Axis to integrate along.  Default -1, the last axis.
    show : bool or sequence, optional
        Print the Richardson extrapolation table.  Default False.  A
        sequence supplies ``(precision, width)``, defaulting to 5 and 8.
        Only a single data set is printed; an ``y`` of rank 2 or more prints
        a notice instead.  Inside ``@njit`` it must be a compile-time
        constant, since it selects whether the table is built at all.

    Returns
    -------
    total : float or ndarray
        The integral, the top-right corner of the Romberg table.  A scalar
        for a 1-D ``y``, otherwise an array of ``y``'s rank less one.

    Raises
    ------
    IndexError
        ``axis`` outside ``y``'s rank.
    ValueError
        The number of samples along ``axis`` is not one plus a non-negative
        power of two.

    See Also
    --------
    scipy.integrate.romb : The scipy routine this mirrors.

    Notes
    -----
    ``scipy.integrate.romb`` documents ``show`` as a bool but reads
    ``(precision, width)`` from a sequence when one is passed; a sequence is
    accepted here as well.

    Pure ``@njit``, no state, so **prange-safe**.  Printing the table takes
    the GIL for the duration, so it serializes a ``prange`` loop.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> x = np.linspace(0.0, np.pi, 65)
    >>> @njit
    ... def run(x):
    ...     return si.romb(np.sin(x), x[1] - x[0])
    >>> run(x)
    1.9999999999999996
    """
    yc = _samples(y)
    nd = yc.ndim
    ax = axis + nd if axis < 0 else axis
    if ax < 0 or ax >= nd:
        raise IndexError(_AXIS_MSG)
    if nd == 1:
        R, k = _romb_table(yc, dx)
        if show:
            precis, width = _romb_showargs(show)
            _romb_print(R, k, precis, width)
        return R[k, k]
    flat, _xf, M, _n = _nd_rows(yc, None, ax)
    res = np.empty(M, yc.dtype)
    for i in range(M):
        res[i] = _romb_1d(flat[i], _EMPTY_X, False, dx)
    if show:
        print(_ROMB_NDMSG)
    return res.reshape(yc.shape[:ax] + yc.shape[ax + 1:])


@overload(romb, prefer_literal=True)
def _romb_ovl(y, dx=1.0, axis=-1, show=False):
    """Compiled body for :func:`romb`, one per rank of ``y``.

    ``show`` is resolved here rather than at run time: it decides whether
    the table is printed at all, and the ``objmode`` block that prints it is
    never reached on a call that does not ask for it.
    """
    if not isinstance(y, types.Array):
        return None
    cast = _cast_of(y)
    show_on = _romb_show_flag(show)
    if show_on is None:
        raise TypingError('romb: show must be a compile-time constant')
    if y.ndim == 1:
        CAST = np.complex128 if cast == 'np.complex128' else np.float64
        ns = _romb_show_len(show)
        if show_on and ns >= 2:
            def impl(y, dx=1.0, axis=-1, show=False):
                if axis != -1 and axis != 0:
                    raise IndexError(_AXIS_MSG)
                yc = np.ascontiguousarray(np.asarray(y)).astype(CAST)
                R, k = _romb_table(yc, dx)
                _romb_show(R, k, show[0], show[1])
                return R[k, k]
        elif show_on and ns == 1:
            def impl(y, dx=1.0, axis=-1, show=False):
                if axis != -1 and axis != 0:
                    raise IndexError(_AXIS_MSG)
                yc = np.ascontiguousarray(np.asarray(y)).astype(CAST)
                R, k = _romb_table(yc, dx)
                _romb_show(R, k, show[0], 8)
                return R[k, k]
        elif show_on:
            def impl(y, dx=1.0, axis=-1, show=False):
                if axis != -1 and axis != 0:
                    raise IndexError(_AXIS_MSG)
                yc = np.ascontiguousarray(np.asarray(y)).astype(CAST)
                R, k = _romb_table(yc, dx)
                _romb_show(R, k, 5, 8)
                return R[k, k]
        else:
            def impl(y, dx=1.0, axis=-1, show=False):
                if axis != -1 and axis != 0:
                    raise IndexError(_AXIS_MSG)
                yc = np.ascontiguousarray(np.asarray(y)).astype(CAST)
                R, k = _romb_table(yc, dx)
                return R[k, k]
        return impl
    return _nd_body('y, dx=1.0, axis=-1, show=False',
                    '_romb_1d({y}, {x}, dx)',
                    y.ndim, 0, cast, None,
                    {'np': np, '_AXIS_MSG': _AXIS_MSG, '_romb_1d': _romb_1d})

# ---------------------------------------------------------------------
# newton_cotes
# ---------------------------------------------------------------------
@njit
def _newton_cotes_builtin(N):
    """Exact-rational Newton-Cotes weights & error coeff for N=1..14. The
    common factor is applied in a different grouping from scipy's
    ``_builtincoeffs``, so the weights agree to rounding (worst 3.6e-15) and
    ``B`` exactly. Returns ``(weights, B, True)``; ``(empty, 0, False)`` for
    N outside 1..14."""
    if N == 1:
        v = np.array([1.0, 1.0]); return v * (1.0 / 2.0), -1.0 / 12.0, True
    if N == 2:
        v = np.array([1.0, 4.0, 1.0]); return v * (1.0 / 3.0), -1.0 / 90.0, True
    if N == 3:
        v = np.array([1.0, 3.0, 3.0, 1.0]); return v * (3.0 / 8.0), -3.0 / 80.0, True
    if N == 4:
        v = np.array([7.0, 32.0, 12.0, 32.0, 7.0]); return v * (2.0 / 45.0), -8.0 / 945.0, True
    if N == 5:
        v = np.array([19.0, 75.0, 50.0, 50.0, 75.0, 19.0]); return v * (5.0 / 288.0), -275.0 / 12096.0, True
    if N == 6:
        v = np.array([41.0, 216.0, 27.0, 272.0, 27.0, 216.0, 41.0]); return v * (1.0 / 140.0), -9.0 / 1400.0, True
    if N == 7:
        v = np.array([751.0, 3577.0, 1323.0, 2989.0, 2989.0, 1323.0, 3577.0, 751.0]); return v * (7.0 / 17280.0), -8183.0 / 518400.0, True
    if N == 8:
        v = np.array([989.0, 5888.0, -928.0, 10496.0, -4540.0, 10496.0, -928.0, 5888.0, 989.0]); return v * (4.0 / 14175.0), -2368.0 / 467775.0, True
    if N == 9:
        v = np.array([2857.0, 15741.0, 1080.0, 19344.0, 5778.0, 5778.0, 19344.0, 1080.0, 15741.0, 2857.0]); return v * (9.0 / 89600.0), -4671.0 / 394240.0, True
    if N == 10:
        v = np.array([16067.0, 106300.0, -48525.0, 272400.0, -260550.0, 427368.0, -260550.0, 272400.0, -48525.0, 106300.0, 16067.0]); return v * (5.0 / 299376.0), -673175.0 / 163459296.0, True
    if N == 11:
        v = np.array([2171465.0, 13486539.0, -3237113.0, 25226685.0, -9595542.0, 15493566.0, 15493566.0, -9595542.0, 25226685.0, -3237113.0, 13486539.0, 2171465.0]); return v * (11.0 / 87091200.0), -2224234463.0 / 237758976000.0, True
    if N == 12:
        v = np.array([1364651.0, 9903168.0, -7587864.0, 35725120.0, -51491295.0, 87516288.0, -87797136.0, 87516288.0, -51491295.0, 35725120.0, -7587864.0, 9903168.0, 1364651.0]); return v * (1.0 / 5255250.0), -3012.0 / 875875.0, True
    if N == 13:
        v = np.array([8181904909.0, 56280729661.0, -31268252574.0, 156074417954.0, -151659573325.0, 206683437987.0, -43111992612.0, -43111992612.0, 206683437987.0, -151659573325.0, 156074417954.0, -31268252574.0, 56280729661.0, 8181904909.0]); return v * (13.0 / 402361344000.0), -2639651053.0 / 344881152000.0, True
    if N == 14:
        v = np.array([90241897.0, 710986864.0, -770720657.0, 3501442784.0, -6625093363.0, 12630121616.0, -16802270373.0, 19534438464.0, -16802270373.0, 12630121616.0, -6625093363.0, 3501442784.0, -770720657.0, 710986864.0, 90241897.0]); return v * (7.0 / 2501928000.0), -3740727473.0 / 1275983280000.0, True
    return np.empty(0), 0.0, False


_NC_POS_MSG = 'The sample positions must start at 0 and end at N'
_NC_ORDER_MSG = 'Order N must be a positive integer'
_NC_EMPTY_MSG = 'index 0 is out of bounds for axis 0 with size 0'
_NC_RANK_MSG = ('The truth value of an array with more than one element is '
                'ambiguous. Use a.any() or a.all()')


@njit
def _nc_builtin_order(N):
    """True when the exact-rational table covers ``N``."""
    return N >= 1.0 and N <= 14.0 and N == np.floor(N)


@njit
def _nc_core(pos, N, equal):
    """scipy's flow from the builtin lookup onward.

    ``pos`` is the sample positions after the caller's ``equal`` has been
    applied, ``N`` the order, and ``equal`` the flag scipy holds at that
    point, which the auto-detection may have set.
    """
    if equal != 0 and _nc_builtin_order(N):
        an_b, B_b, ok = _newton_cotes_builtin(int(N))
        if ok:
            return an_b, B_b

    n = pos.shape[0]
    if n == 0:
        raise IndexError(_NC_EMPTY_MSG)
    if pos[0] != 0.0 or pos[n - 1] != N:
        raise ValueError(_NC_POS_MSG)
    Ni = int(N)
    if Ni < 1:
        raise ValueError(_NC_ORDER_MSG)

    yi = np.empty(Ni + 1)
    for i in range(Ni + 1):
        yi[i] = pos[i] / float(Ni)
    ti = 2.0 * yi - 1.0

    # C[i, j] = ti[j] ** i
    C = np.empty((Ni + 1, Ni + 1))
    for j in range(Ni + 1):
        acc = 1.0
        for i in range(Ni + 1):
            C[i, j] = acc
            acc *= ti[j]
    # numba's inv returns an F-layout array where numpy's returns C, and the
    # refinement below cannot allocate the 'A' layout that mixing them
    # produces.  Making it C matches numpy and removes the mixed dot.
    Cinv = np.ascontiguousarray(np.linalg.inv(C))
    # improve precision (2 Newton-Schulz refinements, as in scipy)
    for _ in range(2):
        prod = np.ascontiguousarray(Cinv.dot(C)).dot(Cinv)
        Cinv = 2.0 * Cinv - prod

    # ai = Cinv[:, ::2] . vec * (N/2),  vec[t] = 2/(2t+1)
    ncols = (Ni + 1 + 1) // 2  # number of even columns 0,2,4,...
    ai = np.empty(Ni + 1)
    for r in range(Ni + 1):
        acc = 0.0
        for t in range(ncols):
            acc += Cinv[r, 2 * t] * (2.0 / (2 * t + 1))
        ai[r] = acc * (Ni / 2.0)

    if Ni % 2 == 0 and equal != 0:
        BN = Ni / (Ni + 3.0)
        power = Ni + 2
    else:
        BN = Ni / (Ni + 2.0)
        power = Ni + 1

    # `yi ** power` must go through libm `pow`, as numpy's does: numba
    # compiles an INTEGER exponent to repeated multiplication (gotcha 11).
    yp = np.empty(Ni + 1)
    pf = np.float64(power)
    for i in range(Ni + 1):
        yp[i] = yi[i] ** pf
    BN = BN - np.dot(yp, ai)
    p1 = power + 1
    fac = math.exp(power * math.log(Ni) - math.lgamma(p1))
    return ai, BN * fac


def newton_cotes(rn, equal=0):
    """Weights and error coefficient of a Newton-Cotes rule.

    Takes NO callback of either style: it returns the rule, it does not
    apply it.  For an equally-spaced rule, integrate with
    ``dx * sum(an * f(a + arange(N+1) * dx))`` where ``dx = (b - a) / N``.

    Parameters
    ----------
    rn : int, float or 1-D array_like
        Either the order ``N``, the number of intervals, so the rule uses
        ``N + 1`` equally-spaced points; or the ``N + 1`` sample positions,
        which must start at 0 and end at ``N``.  A float order is accepted
        and used by value.
    equal : int, optional
        Set to 1 to treat the samples as equally spaced whatever ``rn``
        holds, which also replaces the positions with ``0..N``.  Default 0,
        and then equal spacing is detected from ``rn`` itself when every gap
        is exactly 1, in which case the positions are kept.

    Returns
    -------
    an : float64 array, shape (N+1,)
        Weights, scaled so the integral is ``dx * sum(an * f_i)``.
    B : float
        Error coefficient.  The error term is
        ``B * dx**(N+2) * f**(N+1)(xi)``, and
        ``B * dx**(N+3) * f**(N+2)(xi)`` when the samples are equally spaced
        and ``N`` is even.

    Raises
    ------
    IndexError
        An empty ``rn``.
    ValueError
        The sample positions do not start at 0 and end at ``N``; ``N < 1``;
        ``rn`` of rank 2 or more.

    See Also
    --------
    scipy.integrate.newton_cotes : The scipy routine this mirrors.

    Notes
    -----
    For ``N = 1..14`` equally spaced the weights match ``scipy.integrate.
    newton_cotes`` to rounding and the error coefficient ``B`` is exact.

    Above 14, and for unequally spaced samples, the weights are built through
    the Vandermonde path and lose accuracy as the system grows
    ill-conditioned.  High-order Newton-Cotes weights alternate in sign and
    grow, so rules above about ``N = 8`` are numerically unusable regardless
    of who computes them.  Prefer composite low-order rules, :func:`romb`, or
    :func:`~scijit.integrate.quad`.

    ``N < 1`` raises here.  scipy reaches ``rn / float(N)``, emits a numpy
    ``RuntimeWarning: invalid value encountered in divide``, and then raises
    ``ValueError: math domain error`` from ``math.log(0)``.

    An ``rn`` of rank 2 or more raises here.  scipy reads its order from
    ``len(rn)``, which is the first axis alone, so the rest of the array
    selects nothing and the rule returned describes an order the caller did
    not ask for.

    Pure ``@njit``, no state, so **prange-safe**.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> @njit
    ... def run():
    ...     return si.newton_cotes(4, 0)
    >>> an, B = run()
    >>> an
    array([0.31111111, 1.42222222, 0.53333333, 1.42222222, 0.31111111])
    >>> B
    -0.008465608465608466
    """
    if np.ndim(rn) == 0:                  # scipy: `len(rn)` raised, so
        N = float(rn)                     # `rn` is the order itself
        pos = np.arange(N + 1.0)
        equal = 1
    else:
        a = np.ascontiguousarray(np.asarray(rn, dtype=np.float64))
        if a.ndim != 1:
            raise ValueError(_NC_RANK_MSG)
        N = float(a.shape[0] - 1)
        if equal:
            pos = np.arange(N + 1.0)
        else:
            pos = a
            if a.size and np.all(np.diff(a) == 1):
                equal = 1
    return _nc_core(pos, N, int(equal))


@overload(newton_cotes)
def _newton_cotes_ovl(rn, equal=0):
    """Compiled body for :func:`newton_cotes`, chosen on ``rn``'s type.

    A scalar is the ORDER and is expanded to ``0..N``; an array is already
    the sample positions.  The core takes positions either way, so there is
    one algorithm.
    """
    if isinstance(rn, (types.Integer, types.Boolean, types.Float)):
        def impl(rn, equal=0):
            N = np.float64(rn)
            pos = np.arange(N + 1.0)
            return _nc_core(pos, N, 1)
        return impl
    if isinstance(rn, types.Array):
        if rn.ndim != 1:
            raise TypingError(_NC_RANK_MSG)

        def impl(rn, equal=0):
            a = np.ascontiguousarray(rn.astype(np.float64))
            N = np.float64(a.shape[0] - 1)
            eq = int(equal)
            if eq != 0:
                pos = np.arange(N + 1.0)
            else:
                pos = a
                if a.size > 0:
                    same = True
                    for i in range(a.shape[0] - 1):
                        if a[i + 1] - a[i] != 1.0:
                            same = False
                            break
                    if same:
                        eq = 1
            return _nc_core(pos, N, eq)
        return impl
    return None

# ---------------------------------------------------------------------
# cumulative_trapezoid
# ---------------------------------------------------------------------
_XSHAPE_MSG = 'If given, shape of x must be 1-D or the same as y.'
_XLEN_MSG = 'If given, length of x along axis must be the same as y.'
_ONEPT_MSG = 'At least one point is required along `axis`.'
_INITSCALAR_MSG = '`initial` parameter should be a scalar.'


@njit
def _cumtrapz_1d(y, x, has_x, dx, off, init_v):
    """Running trapezoid integral of one row.

    ``off`` is 1 when a leading value is prepended and 0 otherwise;
    ``init_v`` is that value, and is also added to every element.
    """
    n = y.shape[0]
    if has_x and x.shape[0] != n:
        raise ValueError(_XLEN_MSG)
    out = np.empty(n - 1 + off, y.dtype)
    iv = y.dtype.type(init_v)
    if off == 1:
        out[0] = iv
    acc = y.dtype.type(0)
    for i in range(n - 1):
        if has_x:
            d = x[i + 1] - x[i]
        else:
            d = dx
        acc += d * (y[i + 1] + y[i]) / 2.0
        out[off + i] = acc + iv
    return out


def _ct_initial(initial):
    """The ``initial`` guard: ``None``, or a scalar.

    scipy validates after the integral, but nothing between the two can
    raise, so the order a caller sees is the same.
    """
    if initial is None:
        return False
    if not np.isscalar(initial):
        raise ValueError(_INITSCALAR_MSG)
    return True


def _nd_rows(yc, x, ax):
    """``(rows of y with axis last, matching rows of x or None, M, n)``.

    The interpreter-side half of :func:`_nd_body`, so both entry points run
    one implementation of the 1-D core rather than two.
    """
    ym = np.ascontiguousarray(np.moveaxis(yc, ax, -1))
    n = ym.shape[-1]
    M = 1
    for s in ym.shape[:-1]:
        M *= s
    flat = ym.reshape(M, n)
    if x is None:
        xf = None
    elif np.ndim(x) == 1:
        xf = _positions(x)
    else:
        xf = np.ascontiguousarray(
            np.moveaxis(_positions(x), ax, -1)).reshape(M, n)
    return flat, xf, M, n


_EMPTY_X = np.empty(0)


def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """Running integral by the composite trapezoid rule.

    Takes NO callback of either style: it integrates *samples*, not a
    function.

    Parameters
    ----------
    y : array_like
        Samples to integrate, of any rank.  An integer or boolean array is
        promoted to float64 and a complex one stays complex.  Copied to a
        contiguous buffer, so strided views are safe.
    x : array_like or None, optional
        Sample positions, which may be non-uniform.  Either 1-D along
        ``axis``, or of ``y``'s rank.  ``None`` (the default) means equal
        spacing ``dx``.
    dx : float, optional
        Spacing used when ``x`` is None.  Default 1.0.
    axis : int, optional
        Axis to integrate along.  Default -1, the last axis.
    initial : float or None, optional
        Value prepended to the result along ``axis``, which then has the
        same length there as ``y``, and added to every element.  ``None``
        (the default) returns the ``n - 1`` running integrals.  Inside
        ``@njit`` the choice between ``None`` and a float fixes the length,
        so it must not be a variable that is sometimes one and sometimes
        the other.

    Returns
    -------
    res : ndarray
        Running integral, of ``y``'s rank.  Along ``axis`` its length is
        ``n - 1``, the integral evaluated at the samples after the first,
        or ``n`` when a leading value is prepended.

    Raises
    ------
    IndexError
        ``axis`` outside ``y``'s rank, from indexing ``y.shape``.
    ValueError
        No samples along ``axis``; ``x`` neither 1-D nor of ``y``'s rank;
        ``x`` and ``y`` of different lengths along ``axis``; ``initial``
        neither ``None`` nor a scalar.

    See Also
    --------
    scipy.integrate.cumulative_trapezoid : The scipy routine this mirrors.

    Notes
    -----
    A non-zero ``initial`` is accepted, prepended and added to every element.
    ``scipy.integrate.cumulative_trapezoid`` raises ``ValueError: `initial`
    must be `None` or `0`.`` and accepts a non-zero ``initial`` only on
    ``cumulative_simpson``.

    Pure ``@njit``, no state, so **prange-safe**.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> import scijit.integrate as si
    >>> x = np.linspace(0.0, np.pi, 65)
    >>> @njit
    ... def run(x):
    ...     return si.cumulative_trapezoid(np.sin(x), x, 1.0, -1, 0.0)[-1]
    >>> run(x)
    1.9995983886400375

    A 2-D ``y``, integrated down the rows:

    >>> si.cumulative_trapezoid(np.arange(6.0).reshape(2, 3), None, 1.0, 0)
    array([[1.5, 2.5, 3.5]])
    """
    yc = _samples(y)
    nd = yc.ndim
    if yc.shape[axis] == 0:                   # a bad axis raises IndexError
        raise ValueError(_ONEPT_MSG)
    ax = axis + nd if axis < 0 else axis
    if x is not None and np.ndim(x) != 1 and np.ndim(x) != nd:
        raise ValueError(_XSHAPE_MSG)
    if _ct_initial(initial):
        off, init_v = 1, float(initial)
    else:
        off, init_v = 0, 0.0

    flat, xf, M, n = _nd_rows(yc, x, ax)
    res = np.empty((M, n - 1 + off), yc.dtype)
    for i in range(M):
        if xf is None:
            xr = _EMPTY_X
        elif xf.ndim == 1:
            xr = xf
        else:
            xr = xf[i]
        res[i] = _cumtrapz_1d(flat[i], xr, xf is not None, dx, off, init_v)
    out = res.reshape(yc.shape[:ax] + yc.shape[ax + 1:] + (n - 1 + off,))
    return np.ascontiguousarray(np.moveaxis(out, -1, ax))


@overload(cumulative_trapezoid)
def _cumulative_trapezoid_ovl(y, x=None, dx=1.0, axis=-1, initial=None):
    """Compiled body for :func:`cumulative_trapezoid`, one per rank of ``y``.

    A tuple cannot be sliced by a compile-time constant, so the output shape
    is written out per axis; the branch itself is resolved at run time and
    ``axis`` stays an ordinary argument.
    """
    if not isinstance(y, types.Array):
        return None
    xk = _x_kind(x, y.ndim)
    if xk is None:
        raise TypingError(_XSHAPE_MSG)
    glb = {'np': np, '_positions': _positions, '_AXIS_MSG': _AXIS_MSG,
           '_ONEPT_MSG': _ONEPT_MSG,
           '_cumtrapz_1d': _cumtrapz_1d,
           'HAS_INIT': not _is_none(initial)}
    pre = ['if n == 0:',
           '    raise ValueError(_ONEPT_MSG)',
           'if HAS_INIT:',
           '    off = 1',
           '    init_v = np.float64(initial)',
           'else:',
           '    off = 0',
           '    init_v = 0.0']
    return _nd_body('y, x=None, dx=1.0, axis=-1, initial=None',
                    '_cumtrapz_1d({y}, {x}, dx, off, init_v)',
                    y.ndim, xk, _cast_of(y), 'n - 1 + off', glb, pre)
