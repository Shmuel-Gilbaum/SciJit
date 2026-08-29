#!/usr/bin/env bash
# Build liblapackref.so from the vendored full-precision Reference-LAPACK.
# Compiles objects in parallel (LAPACK is ~2200 tiny files); the two
# module files (la_constants, la_xisnan) must build first so the .f90
# routines that `use` them find the .mod.
set -e
cd "$(dirname "$0")"                       # src/lapack
OUT="${1:-../../scijit/_lib/liblapackref.so}"
OBJ="$(mktemp -d)"
FLAGS="-fPIC -O2 -fallow-argument-mismatch"

# 1. modules first (sequential; emit .mod into $OBJ). Guarded by existence:
# the public release ships a trimmed closure that need not contain them.
for m in la_constants.f90 la_xisnan.F90; do
  [ -f "$m" ] && gfortran $FLAGS -J"$OBJ" -I"$OBJ" -c "$m" -o "$OBJ/$m.o" || :
done

# 2. everything else in parallel
ls *.f *.f90 *.F *.F90 2>/dev/null \
  | grep -vE '^(la_constants\.f90|la_xisnan\.F90)$' \
  | xargs -P "$(nproc)" -I{} gfortran $FLAGS -J"$OBJ" -I"$OBJ" -c {} -o "$OBJ/{}.o"

# 3. link one shared library
gfortran -shared -Wl,-z,noexecstack "$OBJ"/*.o -o "$OUT"
rm -rf "$OBJ"
echo "built $OUT"
