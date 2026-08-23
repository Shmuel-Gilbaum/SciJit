"""scijit._lib, internal shared Fortran libraries.

Holds liblapackref (the vendored full-precision Reference-LAPACK + BLAS,
BSD-3) that packs such as scijit.sparse.linalg (ARPACK) link against
at build time and resolve at import via a relocatable rpath. Not a
public API.
"""
