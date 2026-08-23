# Install

## Wheels (primary)

```bash
pip install scijit
```

The wheels bundle the compiled Fortran runtime, so no toolchain is needed on
this path. Wheels are built for Linux, macOS and Windows.

## From source

A source install compiles each Fortran pack in `src/<pack>/` into a shared
library and places it in the matching subpackage. This path requires
**gfortran** on the PATH.

```bash
pip install .
```

Install gfortran first:

- Debian/Ubuntu: `apt-get install gfortran`
- Fedora/RHEL: `dnf install gcc-gfortran`
- macOS: `brew install gcc`
- Windows: the MinGW-w64 gfortran, for example through MSYS2 or WinLibs.

## If import fails loading a compiled library

This applies on any platform. A source build links against the gfortran runtime
on the build machine. On a machine where gfortran is absent, `import scijit` can
fail while loading a compiled library (the error comes from `ctypes.CDLL`, the
cross-platform loader), with a message naming a library and "or one of its
dependencies". The missing gfortran runtime is the real cause. Install gfortran,
or use the wheels, which carry the runtime.

Diagnose the dependency directly, per platform:

- Linux: `objdump -p lib<pack>.so | grep NEEDED`
- macOS: `strings lib<pack>.dylib | grep dylib`
- Windows: `objdump -p lib<pack>.dll | grep "DLL Name"`
