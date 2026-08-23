"""Linear sum assignment (rectangular LSAP), in numba ``@njit``.

``scipy.optimize.linear_sum_assignment`` is a C++ extension
(``scipy/optimize/rectangular_lsap/rectangular_lsap.cpp``, exposed as
``scipy.optimize._lsap``), which compiled code cannot call.  The algorithm
inside needs no C library: it is Crouse's modified Jonker-Volgenant
shortest-augmenting-path method (D. F. Crouse, "On implementing 2D
rectangular assignment algorithms", *IEEE Trans. Aerospace and Electronic
Systems* **52** (4), 1679-1696, 2016), about 120 lines of integer and float
bookkeeping.  This module is a line-for-line transcription of it into
``@njit``, including its two deliberate tie-breaking devices

  * the ``remaining`` work list is filled in **reverse** order (scipy
    gh-11602: makes a constant cost matrix return the identity), and
  * among equal shortest-path costs the candidate that yields a **new
    sink** (an unmatched column) wins,

which is why the returned index pairs, and not only the optimal total
cost, are fully determined across square, rectangular, integer, float,
massively tied, ``maximize=True`` and infinite-entry matrices.

--------------------------------------------------------------------------
PUBLIC API

  linear_sum_assignment(cost_matrix, maximize=False) -> (row_ind, col_ind)

``cost_matrix`` is 2-D.  From the interpreter it is anything numpy converts
to a 2-D numeric array, matching what scipy's
``PyArray_ContiguousFromAny(obj, NPY_DOUBLE, 0, 0)`` accepts: a list of
lists, a tuple of tuples, an integer or boolean array, a masked array's
data.  Inside ``@njit`` it is an array already, of any numeric dtype.
Strided views and Fortran-ordered arrays work from both; a float64
C-contiguous matrix that needs neither transposing nor negating is read in
place, as in scipy, and every other case gets a working copy.  ``row_ind``
is always sorted ascending; ``cost_matrix[row_ind, col_ind].sum()`` is the
minimal (or, with ``maximize=True``, the maximal) total.  Both outputs are
``int64``, of length ``min(nr, nc)``, exactly as in scipy.

--------------------------------------------------------------------------
DEVIATIONS from scipy:

  * A rank other than 2, and any container that is not an array, are
    compile-time ``TypingError`` refusals inside ``@njit`` where the
    interpreter raises ``ValueError`` and converts respectively.  Rank and
    container are properties of an argument's type, which is fixed when the
    call is compiled.

An empty cost matrix (``nr == 0`` or ``nc == 0``) is legal and returns a
pair of empty ``int64`` arrays.  (This block previously recorded it as a
deviation that raises ``ValueError``; measured, both sides return the
empty pair.)

Error messages are scipy's: ``"matrix contains invalid numeric entries"``
for a NaN or ``-inf``, checked after the optional negation for
``maximize``, and ``"cost matrix is infeasible"`` when no finite perfect
matching of the short side exists.

Safe to call from a ``numba.prange`` loop: no module state, no callbacks
and no globals, so many independent assignment problems can be solved
concurrently.
"""
import numpy as np
from numba import njit, types
from numba.core.errors import TypingError
from numba.extending import overload

__all__ = ['linear_sum_assignment']

_RANK_MSG = "expected a matrix (2-D array), got a %d array"
_NOT_ARRAY_MSG = (
    "linear_sum_assignment: cost_matrix must be a 2-D array inside @njit. "
    "A list of lists or a tuple of tuples is accepted from the interpreter "
    "only; in compiled code build the array first, for example "
    "np.asarray(((4.0, 1.0), (2.0, 0.0))).")


# --------------------------------------------------------------- internals


def _as_cost_matrix(cost_matrix):
    """Reproduce ``PyArray_ContiguousFromAny(obj, NPY_DOUBLE, 0, 0)``.

    An ndarray is cast under numpy's ``'safe'`` rule, so a complex, object
    or string array raises ``TypeError`` with numpy's own text, which is
    what scipy's C binding surfaces.  Anything else goes through
    ``np.asarray(obj, dtype=np.float64)``, which is the conversion scipy
    performs for a list, a tuple or ``None``.  The rank is checked after
    the conversion, in scipy's order.
    """
    if isinstance(cost_matrix, np.ndarray):
        a = cost_matrix
        if a.dtype != np.float64:
            a = a.astype(np.float64, casting='safe')
        a = np.asarray(a)                 # drop ndarray subclasses
    else:
        a = np.asarray(cost_matrix, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(_RANK_MSG % a.ndim)
    return a

@njit
def _augmenting_path(nc, cost, u, v, path, row4col, shortest_path_costs,
                     i, SR, SC, remaining):
    """Find the shortest augmenting path from row ``i``.

    Direct transcription of ``augmenting_path`` in scipy's
    ``rectangular_lsap.cpp``.  ``cost`` is the flat (nr*nc,) row-major
    working matrix.  Returns ``(sink, min_val)``; ``sink < 0`` flags an
    infeasible (all-inf) matrix.
    """
    min_val = 0.0

    # Crouse's pseudocode uses set complements to track remaining nodes; a
    # vector is faster.  Filling it in REVERSE order makes a constant cost
    # matrix produce the identity assignment (scipy gh-11602).
    num_remaining = nc
    for it in range(nc):
        remaining[it] = nc - it - 1

    for k in range(SR.shape[0]):
        SR[k] = False
    for k in range(nc):
        SC[k] = False
        shortest_path_costs[k] = np.inf

    sink = -1
    while sink == -1:
        index = -1
        lowest = np.inf
        SR[i] = True

        for it in range(num_remaining):
            j = remaining[it]

            r = min_val + cost[i * nc + j] - u[i] - v[j]
            if r < shortest_path_costs[j]:
                path[j] = i
                shortest_path_costs[j] = r

            # When several nodes tie for the minimum cost, prefer one that
            # gives a NEW sink node.  Matters a lot for integer cost
            # matrices with small coefficients.
            if (shortest_path_costs[j] < lowest or
                    (shortest_path_costs[j] == lowest and row4col[j] == -1)):
                lowest = shortest_path_costs[j]
                index = it

        min_val = lowest
        if min_val == np.inf:          # infeasible cost matrix
            return -1, min_val

        j = remaining[index]
        if row4col[j] == -1:
            sink = j
        else:
            i = row4col[j]

        SC[j] = True
        num_remaining -= 1
        remaining[index] = remaining[num_remaining]

    return sink, min_val


@njit
def _argsort_stable(a):
    """Argsort of a 1-D int64 array, used on the transposed path.

    ``a`` is ``col4row``, which holds one distinct column index per row of
    a matching, so no two keys compare equal and the choice of sort is not
    observable.  scipy's ``argsort_iter`` uses ``std::sort``, which is not
    stable; this one is.
    """
    return np.argsort(a, kind='mergesort')


def _flat_f64(a):
    """Row-major flat float64 view of `a`, copying only when it must."""
    r = a.ravel()
    return r if r.dtype == np.float64 else r.astype(np.float64)


@overload(_flat_f64)
def _flat_f64_ovl(a):
    if a.dtype == types.float64:
        def impl(a):
            return a.ravel()
    else:
        def impl(a):
            return a.ravel().astype(np.float64)

    return impl


# ------------------------------------------------------------------ engine

@njit
def _lsa(cost_matrix, maximize):
    """Crouse's algorithm on an already-validated 2-D numeric array.

    Both entry points of ``linear_sum_assignment`` reach this, so the
    interpreter and compiled code run one implementation.  ``cost_matrix``
    has rank 2 and any numeric dtype; ``maximize`` is a boolean.
    """
    nr0 = cost_matrix.shape[0]
    nc0 = cost_matrix.shape[1]
    if nr0 == 0 or nc0 == 0:
        # scipy returns two empty int64 arrays here.  A runtime branch, not
        # an @overload: both paths return the same type.
        return (np.empty(0, np.int64), np.empty(0, np.int64))

    # Tall (nr > nc) matrices are transposed so that nr <= nc.
    transpose = nc0 < nr0
    if transpose:
        nr = nc0
        nc = nr0
    else:
        nr = nr0
        nc = nc0

    # Build the flat, row-major, float64 working matrix, transposing and/or
    # negating in the same pass.  scipy allocates only when it has to
    # transpose or negate (rectangular_lsap.cpp:139-165) and otherwise reads
    # the caller's buffer; _flat_f64 is that second case.  Nothing below
    # writes to `work`, so the alias cannot reach the caller's array.
    if transpose or maximize:
        work = np.empty(nr * nc, dtype=np.float64)
        if transpose:
            if maximize:
                for i in range(nr0):
                    for j in range(nc0):
                        work[j * nr0 + i] = -np.float64(cost_matrix[i, j])
            else:
                for i in range(nr0):
                    for j in range(nc0):
                        work[j * nr0 + i] = np.float64(cost_matrix[i, j])
        else:
            for i in range(nr0):
                for j in range(nc0):
                    work[i * nc0 + j] = -np.float64(cost_matrix[i, j])
    else:
        work = _flat_f64(cost_matrix)

    # NaN / -inf test, on the (possibly negated) working matrix -- scipy
    # checks after negation too, so +inf with maximize=True is rejected.
    for k in range(nr * nc):
        w = work[k]
        if np.isnan(w) or w == -np.inf:
            raise ValueError("matrix contains invalid numeric entries")

    u = np.zeros(nr, dtype=np.float64)
    v = np.zeros(nc, dtype=np.float64)
    shortest_path_costs = np.empty(nc, dtype=np.float64)
    path = np.full(nc, -1, dtype=np.int64)
    col4row = np.full(nr, -1, dtype=np.int64)
    row4col = np.full(nc, -1, dtype=np.int64)
    SR = np.zeros(nr, dtype=np.bool_)
    SC = np.zeros(nc, dtype=np.bool_)
    remaining = np.zeros(nc, dtype=np.int64)

    for cur_row in range(nr):
        sink, min_val = _augmenting_path(
            nc, work, u, v, path, row4col, shortest_path_costs,
            cur_row, SR, SC, remaining)
        if sink < 0:
            raise ValueError("cost matrix is infeasible")

        # update dual variables
        u[cur_row] += min_val
        for i in range(nr):
            if SR[i] and i != cur_row:
                u[i] += min_val - shortest_path_costs[col4row[i]]
        for j in range(nc):
            if SC[j]:
                v[j] -= min_val - shortest_path_costs[j]

        # augment the previous solution
        j = sink
        while True:
            i = path[j]
            row4col[j] = i
            tmp = col4row[i]
            col4row[i] = j
            j = tmp
            if i == cur_row:
                break

    a = np.empty(nr, dtype=np.int64)
    b = np.empty(nr, dtype=np.int64)
    if transpose:
        order = _argsort_stable(col4row)
        for k in range(nr):
            w = order[k]
            a[k] = col4row[w]
            b[k] = w
    else:
        for i in range(nr):
            a[i] = i
            b[i] = col4row[i]

    return a, b


# ------------------------------------------------------------------ public

def linear_sum_assignment(cost_matrix, maximize=False):
    """Solve the linear sum assignment problem.

    Choose one entry from each row and each column so that the total is as
    small as possible: the least-cost way to give ``nr`` jobs to ``nc``
    workers, one each.

    Parameters
    ----------
    cost_matrix : array_like, shape (nr, nc)
        Cost matrix.  From the interpreter, anything numpy converts to a
        2-D numeric array: a list of lists, a tuple of tuples, an integer
        or boolean array, a Fortran-ordered array, a strided view.  A
        masked array contributes its data and its mask is ignored.  Inside
        ``@njit`` it has to be a 2-D array already, of any numeric dtype;
        strided and Fortran-ordered arrays are accepted there too.  An
        empty matrix is legal and yields two empty index arrays.
    maximize : bool, optional
        If True maximize the total instead of minimizing it. Default False.
        Implemented by negating the matrix.

    Returns
    -------
    (row_ind, col_ind) : two int64 arrays of length ``min(nr, nc)``
        The assignment, as one entry per chosen row and column. ``row_ind``
        is sorted ascending, and ``cost_matrix[row_ind, col_ind].sum()`` is
        its total cost.

    Raises
    ------
    ValueError
        If `cost_matrix` has a rank other than 2 once converted; if it holds
        a NaN or a ``-inf``, checked after the optional negation for
        `maximize`; or if no finite perfect matching of the short side
        exists.
    TypeError
        If an array is passed whose dtype does not cast to float64 under
        numpy's ``'safe'`` rule, such as a complex, object or string array.

    See Also
    --------
    scipy.optimize.linear_sum_assignment : The scipy routine this mirrors.

    Notes
    -----
    Solved by the modified Jonker-Volgenant shortest augmenting path
    algorithm. The tie-breaking rule is fixed so the index pairs are
    determined, not only the total cost: the work list is filled in reverse
    order, so a constant cost matrix returns the identity assignment, and
    among equal shortest-path costs a candidate that opens a new unmatched
    column wins.

    Rank and container are compile-time properties of an argument, so
    inside ``@njit`` a rank other than 2 is a ``TypingError`` where the
    interpreter raises ``ValueError``, and a list or a tuple is a
    ``TypingError`` where the interpreter converts it.  The rank refusal
    carries the rank it was given; the container refusal names
    `cost_matrix` and the spelling to use instead.

    No module state, no callback and no globals, so many independent
    assignment problems can be solved concurrently inside a
    ``numba.prange`` loop.

    Examples
    --------
    >>> import numpy as np
    >>> from numba import njit
    >>> from scijit.optimize import linear_sum_assignment
    >>> cost = np.array([[4.0, 1.0, 3.0],
    ...                  [2.0, 0.0, 5.0],
    ...                  [3.0, 2.0, 2.0]])
    >>> @njit
    ... def run():
    ...     return linear_sum_assignment(cost)
    >>> row_ind, col_ind = run()
    >>> row_ind
    array([0, 1, 2])
    >>> col_ind
    array([1, 0, 2])

    From the interpreter the same problem can be written as a list of
    lists:

    >>> linear_sum_assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])[1]
    array([1, 0, 2])

    numba rejects indexing one array with two index arrays, so the total is
    summed in a loop rather than as ``cost[row_ind, col_ind].sum()``:

    >>> @njit
    ... def total():
    ...     r, c = linear_sum_assignment(cost)
    ...     s = 0.0
    ...     for k in range(r.size):
    ...         s += cost[r[k], c[k]]
    ...     return s
    >>> total()
    5.0
    """
    return _lsa(_as_cost_matrix(cost_matrix), bool(maximize))


@overload(linear_sum_assignment)
def _linear_sum_assignment_ovl(cost_matrix, maximize=False):
    if not isinstance(cost_matrix, types.Array):
        raise TypingError(_NOT_ARRAY_MSG)
    if cost_matrix.ndim != 2:
        raise TypingError(_RANK_MSG % cost_matrix.ndim)

    def impl(cost_matrix, maximize=False):
        # `bool()` here and not in `_lsa`: the interpreter reaches `_lsa`
        # through the same coercion, so the two entry points run one
        # normalisation rather than two spellings of it.
        return _lsa(cost_matrix, bool(maximize))

    return impl
