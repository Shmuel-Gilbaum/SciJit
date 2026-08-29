"""scijit._lib, internal shared code and libraries.

Holds liblapackref (the vendored full-precision Reference-LAPACK + BLAS,
BSD-3) that packs such as scijit.sparse.linalg (ARPACK) link against
at build time and resolve at import via a relocatable rpath, and
``_typing``, the typing-time predicates the ``@overload`` front ends share.
Not a public API.
"""
