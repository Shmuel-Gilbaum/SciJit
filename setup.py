"""OS-aware installer for scijit, a multi-pack collection of
scipy-equivalent Fortran libraries callable inside numba @njit code.

Each entry in PACKS wraps one Fortran pack: its sources live in
src/<pack>/ (F77 .f files plus a bind(c) wrappers.f90) and are compiled
into one shared library named per platform
    Linux   -> lib<pack>.so
    Windows -> lib<pack>.dll
    macOS   -> lib<pack>.dylib
placed inside the matching scijit subpackage, where the glue modules
pick it up with the same platform switch.

Currently shipped packs
    fitpack  -> scijit.interpolate   (Dierckx splines, 31 routines)
    minpack  -> scijit.optimize      (hybrd/hybrj, lmdif/lmder)
    lbfgsb   -> scijit.optimize      (L-BFGS-B bound-constrained min)
    slsqp    -> scijit.optimize      (SLSQP constrained min)
    optlapack-> scijit.optimize      (LAPACK QR/triangular for lsq)
    prima    -> scijit.optimize      (Powell derivative-free family)
    quadpack -> scijit.integrate     (adaptive integration / quad)
    odepack  -> scijit.integrate     (LSODA/LSODAR ODE solving)
To add another pack: PACKS row + sources in src/<pack>/ (a SOURCES.txt
there fixes the compile order when alphabetical is not enough). The
subpackage may be dotted ('sparse.linalg').

Requires a Fortran compiler (gfortran) on PATH.
"""
import glob
import os
import platform
import subprocess
import tempfile
from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py

TOP_PACKAGE = 'scijit'
ROOT = os.path.dirname(os.path.abspath(__file__))


# Tell setuptools this distribution contains native compiled shared libraries.
# NOT redundant with `root_is_pure = False` in the bdist_wheel subclass below:
# root_is_pure sets the wheel TAG, has_ext_modules() decides whether files land
# in platlib or in .data/purelib.  Removing this in 0.1.3 produced a
# py3-none wheel whose .so files sat in purelib, and auditwheel refused it:
#   RuntimeError: Invalid binary wheel, found the following shared
#   library/libraries in purelib folder
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True



# (fortran pack dir under src/, python subpackage, library base name)
PACKS = [
    ('fitpack', 'interpolate', 'libfitpack'),
    ('minpack', 'optimize', 'libminpack'),
    ('lbfgsb', 'optimize', 'liblbfgsb'),
    ('slsqp', 'optimize', 'libslsqp'),
    ('optlapack', 'optimize', 'liboptlapack'),
    ('prima', 'optimize', 'libprima'),
    ('quadpack', 'integrate', 'libquadpack'),
    ('odepack', 'integrate', 'liblsoda'),
]

# Internal SHARED libraries built BEFORE the packs and reused across
# them, rather than re-vendored per pack. Each entry:
#   (src dir under src/, output subpackage, library base, n_prelude)
# n_prelude = how many leading SOURCES.txt entries define Fortran
# modules and must compile sequentially (emitting .mod) before the rest
# compile in parallel. Reference-LAPACK: la_constants + la_xisnan.
SHARED_LIBS = [
    ('lapack', '_lib', 'liblapackref', 2),
]

# packs that link one of the SHARED_LIBS instead of vendoring BLAS/LAPACK.
#   pack -> (shared-lib output subpackage, library base)
# The consuming .so gets an $ORIGIN/@loader_path-relative rpath to the
# shared lib so it resolves at import with no system BLAS/LAPACK needed.
LINK_SHARED = {
    'optlapack': ('_lib', 'liblapackref'),
}

# Packs whose wrappers hold the user callback in a MODULE VARIABLE and mark
# it `!$omp threadprivate`.  gfortran lowers that directive to thread-local
# storage, which is what makes concurrent solves independent, and it reads
# the directive only under -fopenmp.  WITHOUT this flag the line is an
# ordinary comment: the build succeeds, single-threaded results are
# unchanged, and every concurrent call silently shares one callback slot
# again.  Measured on the shared-slot build, 32 concurrent 2-variable
# fsolve problems: 4 runs in 5 aborted with `double free or corruption`,
# the fifth returned 15 wrong rows.
#
# No OpenMP construct other than threadprivate is used, so no libgomp
# dependency appears; the only new dynamic symbol is
# __tls_get_addr@GLIBC_2.3 from ld-linux.
OPENMP_PACKS = {'minpack', 'prima', 'quadpack', 'odepack'}


def lib_name(base):
    system = platform.system()
    if system == 'Windows':
        return base + '.dll'
    if system == 'Linux':
        return base + '.so'
    return base + '.dylib'             # macOS / other, matches the glue code


def _rpath_flag(from_dir, to_dir):
    """Linker rpath making a library find a sibling shared lib by a
    path relative to its own location (so an installed wheel is
    relocatable). $ORIGIN on Linux, @loader_path on macOS."""
    rel = os.path.relpath(to_dir, from_dir)
    system = platform.system()
    if system == 'Darwin':
        return ['-Wl,-rpath,' + os.path.join('@loader_path', rel)]
    if system == 'Windows':
        return []                        # no rpath; DLL copied alongside
    return ['-Wl,-rpath,' + os.path.join('$ORIGIN', rel)]


def runtime_flags():
    """Link the gfortran runtime INTO each library instead of against it.

    Without this every built library keeps a dynamic dependency on the
    BUILD machine's toolchain, libgfortran.so.5 on Linux, libgfortran-5.dll
    on Windows, /opt/homebrew/.../libgfortran.5.dylib on macOS, and
    importing the installed wheel where gfortran is absent dies inside the
    ctypes.CDLL call with

        FileNotFoundError: Could not find module '...libfitpack.dll'
        (or one of its dependencies)

    -static-libgfortran alone is NOT enough: libgfortran.a itself references
    libquadmath symbols, which gfortran's spec then resolves from the shared
    (or import) library, so libquadmath survives in the dependency table.
    There is no -static-libquadmath, hence the per-platform work-arounds.

    LINUX IS DELIBERATELY EXCLUDED: distro libgfortran.a is built WITHOUT
    -fPIC, so linking it into a shared object fails outright with
        relocation R_X86_64_TPOFF32 against hidden symbol `thread_unit'
        can not be used when making a shared object
    The Linux answer is to bundle libgfortran.so.5 into the wheel after the
    build instead -- `auditwheel repair` copies it into scijit.libs/ and
    rewrites RPATH, which is what every scientific manylinux wheel does.
    """
    system = platform.system()
    if system == 'Linux':
        return []
    flags = ['-static-libgfortran', '-static-libgcc']
    if system == 'Windows':
        # MinGW: -static makes the linker prefer .a over .dll.a for every
        # library, which is what finally drops libquadmath-0.dll (and
        # libwinpthread-1.dll). It composes with -shared.
        flags.insert(0, '-static')
    elif system == 'Darwin':
        # No usable -static on macOS (there is no static libSystem), and
        # gfortran links libquadmath.0.dylib by absolute Homebrew path,
        # so name the archive explicitly. -print-file-name echoes its
        # argument back unchanged when the file does not exist.
        quad = subprocess.run(['gfortran', '-print-file-name=libquadmath.a'],
                              capture_output=True, text=True).stdout.strip()
        if os.path.isabs(quad) and os.path.isfile(quad):
            flags.append(quad)
    if system != 'Darwin':
        # Hide the static runtime's symbols (GNU ld only, and it applies to
        # archives only, so our own wrapper objects stay exported) so that
        # several scijit libraries in one process do not each re-export
        # libgfortran internals.
        flags.append('-Wl,--exclude-libs,ALL')
    return flags


def install_name_flag(out):
    """macOS: stamp the library's own id as @rpath/<name> so that consumers
    record a relocatable dependency instead of the absolute build path, and
    the @loader_path rpath from _rpath_flag() is actually consulted."""
    if platform.system() == 'Darwin':
        return ['-Wl,-install_name,@rpath/' + os.path.basename(out)]
    return []


def build_shared_lib(srcdir, out, n_prelude):
    """Compile a large Fortran source tree (e.g. all of Reference-LAPACK)
    into one shared library: the n_prelude module files first
    (sequential, to emit .mod), the rest as parallel per-file objects,
    then a single link. One monolithic gfortran call over ~2000 files is
    far too slow / memory-heavy, hence the object fan-out."""
    import concurrent.futures
    with open(os.path.join(srcdir, 'SOURCES.txt'), encoding='utf-8') as fh:
        srcs = [ln.strip() for ln in fh if ln.strip()]
    objdir = tempfile.mkdtemp(prefix='nsp_lapack_obj_')
    base = ['gfortran', '-fPIC', '-O2', '-fallow-argument-mismatch',
            '-J', objdir, '-I', objdir]

    def compile_one(src):
        obj = os.path.join(objdir, src + '.o')
        subprocess.run(base + ['-c', os.path.join(srcdir, src), '-o', obj],
                       check=True)
        return obj

    objs = [compile_one(s) for s in srcs[:n_prelude]]     # modules first
    rest = srcs[n_prelude:]
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count() or 4) as ex:
        objs += list(ex.map(compile_one, rest))

    # Link via a RESPONSE FILE (@file): ~2200 object paths on one command
    # line overflow Windows' 32 KB CreateProcess limit.
    rsp = os.path.join(objdir, 'link_objs.rsp')
    with open(rsp, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join('"%s"' % o.replace('\\', '/') for o in objs))
    link = ['gfortran', '-shared', '-o', out, '@' + rsp]

    system = platform.system()
    link += runtime_flags()            # no libgfortran/libquadmath dependency
    link += install_name_flag(out)     # macOS: id = @rpath/liblapackref.dylib

    if system == 'Linux':
        link.append('-Wl,-z,noexecstack')
        # The source list is a COMPUTED closure (lapack_closure.py), not the
        # whole pack, so a walker bug drops a file.  Without this flag that
        # links cleanly and the symbol binds at load time to whatever else in
        # the process exports it -- numpy and scipy both bundle their own
        # LAPACK/BLAS -- giving a plausible wrong answer.  --no-undefined
        # makes it a build error naming the symbol instead.  Every symbol
        # this library legitimately leaves undefined is libc or libgfortran,
        # resolved by runtime_flags() above; a probe link confirmed it passes.
        link.append('-Wl,--no-undefined')

    if system == 'Windows':
        # emit a MinGW import library so consuming packs can -l against it
        link.append('-Wl,--out-implib,' + out + '.a')
    subprocess.run(link, check=True)
    for f in glob.glob(os.path.join(objdir, '*')):
        os.remove(f)
    os.rmdir(objdir)

class BuildFortran(build_py):
    """Compile every Fortran pack before the normal python build."""

    def run(self):
        # 1. internal shared libraries (BLAS/LAPACK) consumed by packs
        for pack, subpkg, libbase, n_prelude in SHARED_LIBS:
            srcdir = os.path.join(ROOT, 'src', pack)
            out = os.path.join(ROOT, TOP_PACKAGE, *subpkg.split('.'),
                               lib_name(libbase))
            print('>>> Compiling shared lib', libbase, '(parallel) ->', out)
            build_shared_lib(srcdir, out, n_prelude)

        # 2. the packs (each their own .so; some link a shared lib above)
        for pack, subpkg, libbase in PACKS:
            srcdir = os.path.join(ROOT, 'src', pack)
            order = os.path.join(srcdir, 'SOURCES.txt')
            if os.path.exists(order):
                with open(order, encoding='utf-8') as fh:
                    sources = [os.path.join(srcdir, line.strip())
                               for line in fh if line.strip()]
            else:
                sources = sorted(glob.glob(os.path.join(srcdir, '*.f')))
                sources += sorted(glob.glob(os.path.join(srcdir, '*.f90')))
            out = os.path.join(ROOT, TOP_PACKAGE, *subpkg.split('.'),
                               lib_name(libbase))

            moddir = tempfile.mkdtemp(prefix='nsp_mod_')
            cmd = ['gfortran'] + sources + [
                '-shared', '-O3', '-fallow-argument-mismatch',
                '-o', out, '-J', moddir, '-I', moddir
            ]
            cmd.append('-I' + srcdir)
            if pack in OPENMP_PACKS:
                cmd.append('-fopenmp')

            system = platform.system()
            cmd += runtime_flags()         # no libgfortran/libquadmath dependency
            cmd += install_name_flag(out)  # macOS: id = @rpath/lib<pack>.dylib

            if system != 'Windows':
                cmd.insert(1, '-fPIC')
            if system == 'Windows':
                # duplicate gfortran runtime symbols across the statically
                # linked archives are expected, not an error
                cmd.append('-Wl,--allow-multiple-definition')
            if system == 'Linux':
                cmd.append('-Wl,-z,noexecstack')

            if pack in LINK_SHARED:
                shsub, shbase = LINK_SHARED[pack]
                shdir = os.path.join(ROOT, TOP_PACKAGE, *shsub.split('.'))
                if system == 'Windows':
                    # The -static in runtime_flags() puts ld in -Bstatic mode,
                    # where a -l search refuses a MinGW *import* library:
                    #   cannot find -llapackref: have you installed the
                    #   static version of the lapackref library ?
                    # Naming liblapackref.dll.a as an explicit file operand
                    # bypasses the -l search rules, so lapackref stays an
                    # ordinary DLL dependency (resolved at import time by the
                    # os.add_dll_directory() walk in scijit/__init__.py).
                    # mirrors the --out-implib name in build_shared_lib()
                    cmd.append(os.path.join(shdir, lib_name(shbase) + '.a'))
                else:
                    cmd += ['-L' + shdir, '-l' + shbase[3:]]
                cmd += _rpath_flag(os.path.dirname(out), shdir)

            print('>>> Compiling', pack, ': gfortran', len(sources),
                  'sources ->', out)
            try :
                subprocess.run(cmd, check=True, cwd=ROOT,capture_output=True,text=True)
            except subprocess.CalledProcessError as e:
                print("\n" + "=" * 80)
                print("GFORTRAN COMPILATION ERROR IN PACK:", pack)
                print(e.stderr)
                print("=" * 80 + "\n")
                raise
            for f in glob.glob(os.path.join(moddir, '*')):
                os.remove(f)
            os.rmdir(moddir)

        super().run()
        
  
        

# Tag wheel as 'py3-none-<platform>' so it works across all Python 3.x versions
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
    def get_tag(self):
        python, abi, plat = super().get_tag()
        # macOS: setuptools reports the tag of the PYTHON BUILD, and the
        # python.org installer is universal2 -> we would claim a fat
        # wheel. gfortran only ever emits the HOST architecture, so the
        # libraries inside are single-arch and the claim is false:
        #   delocate.libsana.DelocationError: Failed to find any binary
        #   with the required architecture: 'x86_64'
        # Narrow the tag to the arch we actually built.
        if plat.startswith('macosx') and 'universal2' in plat:
            plat = plat.replace('universal2', platform.machine())
        return 'py3', 'none', plat


custom_cmdclass = {'build_py': BuildFortran, 'bdist_wheel': bdist_wheel}

# several packs may share one subpackage (optimize = minpack + lbfgsb +
# slsqp), dedupe packages and AGGREGATE package_data per subpackage (a
# dict comprehension would silently keep only the last pack's library)
# DISCOVER the packages from the source tree rather than deriving them from
# PACKS. Many scijit subpackages are pure-@njit ports with no Fortran pack at
# all (ndimage, signal, spatial, cluster, constants, sparse.csgraph), so a
# PACKS-derived list silently omits them from the wheel, and the failure
# surfaces far from the cause, as
#     ImportError: cannot import name 'ndimage' from partially initialized
#     module 'scijit' (most likely due to a circular import)
# which really just means "that directory was never installed". The tests run
# with PYTHONPATH=. against the source tree, so they never catch it. This bug
# shipped twice (spatial/signal/constants, then ndimage) with a hand-kept
# list; find_packages removes the possibility.
_packages = sorted(find_packages(include=[TOP_PACKAGE, TOP_PACKAGE + '.*']))
_package_data = {}
# Ship ONLY the extension this OS builds.  Listing all three was harmless on
# CI, where a fresh checkout holds one platform's artifacts and .gitignore
# excludes them anyway, but on a development machine that has built for more
# than one platform it put the Windows .dll set into the Linux wheel as well
# (28 shared libraries instead of 14, roughly double the size, and a
# linux_x86_64 wheel carrying Windows binaries).
_LIBEXT = lib_name('')         # '.so' / '.dll' / '.dylib' for this platform
for _, subpkg, libbase in PACKS:
    _package_data.setdefault('%s.%s' % (TOP_PACKAGE, subpkg), []).append(
        libbase + _LIBEXT)
for _, subpkg, libbase, _ in SHARED_LIBS:
    _package_data.setdefault('%s.%s' % (TOP_PACKAGE, subpkg), []).append(
        libbase + _LIBEXT)


   
    
    
setup(
    name='scijit',
    version='0.1.5',
    author='Shmuel Gilbaum',
    author_email='s.gilbaum@gmail.com',
    url='https://github.com/shmuel-gilbaum/SciJit',
    project_urls={
        'Documentation': 'https://shmuel-gilbaum.github.io/SciJit/',
        'Source': 'https://github.com/shmuel-gilbaum/SciJit',
    },
    license='BSD-3-Clause',
    description=('scipy-equivalent numerical routines callable inside numba '
                 '@njit code, backed by the same Fortran libraries scipy '
                 'wraps: FITPACK splines (scijit.interpolate); MINPACK, '
                 'L-BFGS-B, SLSQP, PRIMA (scijit.optimize); QUADPACK + '
                 'ODEPACK LSODA (scijit.integrate).'),
    long_description=open(os.path.join(ROOT, 'README.md'),
                          encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    # Only terms describing what the RELEASE actually ships. The dev tree has
    # fft, special, sparse.linalg and more; advertising them here would send a
    # PyPI searcher to a package that does not contain them. Add each back as
    # it ships. Kept in step with the GitHub topics.
    keywords=[
        'numba', 'scipy', 'njit', 'jit', 'numba-scipy',
        'scientific-computing', 'numerical-methods', 'fortran',
        'ode', 'ode-solver', 'solve_ivp', 'odeint', 'quadrature', 'integration',
        'fsolve', 'root-finding', 'least-squares', 'curve-fitting',
        'optimization', 'minimization',
        'interpolation', 'splines',
        'minpack', 'lapack', 'fitpack', 'quadpack', 'odepack',
        'lbfgsb', 'slsqp', 'prima',
    ],
    classifiers=[
        # No 'License ::' classifier. setuptools deprecated them in favour of
        # an SPDX expression, which `license='BSD-3-Clause'` above already is,
        # and emits SetuptoolsDeprecationWarning while both are present.
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Programming Language :: Fortran',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
    ],
    packages=_packages,
    package_data=_package_data,
    # numba floor, in order of what actually forces it:
    #   numpy 2.4 needs numba >= 0.64   (0.61 tops out at numpy 2.1)
    #   the package is measured on 0.66
    #   0.66 adds multi-dimensional fancy indexing (PR #10432), which
    #   docs/usage/optimize.md uses; older numba raises a TypingError
    # Dropping to 0.64 costs only that one example. Dropping to 0.61 would
    # also require pinning numpy <= 2.1.
    install_requires=['numpy', 'numba>=0.66', 'scijitclass>=0.1.7'],
    extras_require={'docs': ['sphinx', 'myst-parser', 'numpydoc', 'furo']},
    # >=3.10 because every dependency requires it: numba >= 0.61
    # dropped 3.9, current numpy requires 3.10, and scijitclass
    # declares requires-python >=3.10. A 3.9 install fails while
    # resolving a dependency, which names the dependency and not
    # scijit. It is also the floor CI builds and tests: cp310.
    python_requires='>=3.10',
    cmdclass=custom_cmdclass,
    distclass=BinaryDistribution,   # platlib, not purelib
    zip_safe=False,                 # the shared libraries must be real files
)
