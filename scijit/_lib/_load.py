"""The ctypes loader the Fortran-backed glue modules share.

Each pack's shared library ships inside its own subpackage directory, so the
directory to search is the CALLING module's, not this one's. The caller passes
its ``__file__``. Not a public API.
"""

import ctypes as ct
import os
import platform

__all__ = ['load']

# liblinalg, liboptlapack, libarpack and libpropack link against liblapackref,
# which lives here in scijit/_lib. On Linux and macOS an rpath resolves it. On
# Windows (Python 3.8+) the loader no longer searches PATH for a dependent DLL,
# so that directory is registered once, here, where its path is known without
# counting levels up from a caller.
if platform.uname()[0] == "Windows":
    _libdir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(_libdir):
        os.add_dll_directory(_libdir)


def _sig(fn, nargs):
    """Declare a ``bind(c)`` wrapper as ``nargs`` opaque pointers, void return.

    Every ``*_wrapper`` in ``src/<pack>/wrappers.f90`` takes its arguments by
    reference, so each one is a bare pointer. The count must match the Fortran
    argument list: a count that disagrees with the call site raises
    ``ExternalFunctionPointer``, and a count wrong in the same way on both
    sides raises nothing and runs into undefined behaviour.
    """
    fn.argtypes = [ct.c_void_p] * nargs
    fn.restype = None
    return fn


def load(caller_file, libname):
    """The pack's shared library and the ``_sig`` helper, as ``(lib, sig)``.

    Parameters
    ----------
    caller_file : str
        The calling module's ``__file__``. The library sits beside it.
    libname : str
        The library's base name with no extension, such as ``'libfitpack'``.
    """
    rootdir = os.path.dirname(os.path.abspath(caller_file))
    system = platform.uname()[0]
    if system == "Windows":
        ext = ".dll"
    elif system == "Linux":
        ext = ".so"
    else:
        ext = ".dylib"
    return ct.CDLL(os.path.join(rootdir, libname + ext)), _sig
