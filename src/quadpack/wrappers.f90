! bind(c) wrappers exposing QUADPACK (jacobwilliams/quadpack modern
! edition, double precision) to ctypes/numba.
!
! QUADPACK's integrand interface is a bare scalar function f(x) with no
! parameter slot. To pass args (like NumbaQuadpack's f(x, data)) we use
! the same module-variable adapter as the minpack pack: the numba
! @cfunc pointer and the args pointer live in module variables, and the
! function handed to QUADPACK is an ordinary module function -> no
! nested procedure, no trampoline, GNU_STACK stays RW.
!
! The slot is marked `!$omp threadprivate`, so gfortran lowers it to
! thread-local storage and concurrent integrations on different threads
! do not share it.  The directive is read only under -fopenmp; without
! that flag it is an ordinary comment, the build succeeds, and every
! concurrent call silently shares one slot again.  See
! setup.py OPENMP_PACKS.
!
! Nesting on ONE thread is still forbidden: an inner integration
! overwrites the outer one's slot.
!
! numba @cfunc ABI (returns the integrand value; x by value, args by
! reference):
!     quadpack_sig = float64(float64 x, float64* args)
module quadpack_wrappers
    use iso_c_binding, only: c_double, c_int, c_funptr, c_f_procpointer
    use quadpack_double, only: dqags, dqagi, dqagp, dqawo, dqawf, &
                               dqaws, dqawc
    implicit none
    private
    public :: dqags_wrapper, dqagi_wrapper, dqagp_wrapper, &
              dqawo_wrapper, dqawf_wrapper, dqaws_wrapper, dqawc_wrapper

    abstract interface
        function numba_f(x, args) bind(c)          ! quadpack_sig
            import :: c_double
            implicit none
            real(c_double), intent(in), value :: x
            real(c_double), intent(in) :: args(*)
            real(c_double) :: numba_f
        end function numba_f
    end interface

    procedure(numba_f), pointer :: cb => null()
    real(c_double), pointer :: cb_args(:) => null()
    !$omp threadprivate(cb, cb_args)

contains

    ! ordinary module function matching QUADPACK's func(x).
    ! RECURSIVE because a nested integration re-enters it: the user
    ! callback may itself call a wrapper, which calls this again.
    recursive real(c_double) function f_adapter(x)
        real(c_double), intent(in) :: x
        f_adapter = cb(x, cb_args)
    end function f_adapter

    ! finite interval, general (handles singularities via extrapolation).
    ! scipy.integrate.quad's finite-bound engine. 15 C arguments.
    !
    ! `last` is the number of subintervals in the final partition; the
    ! caller reads work(1:last), work(limit+1:limit+last), ... as
    ! alist/blist/rlist/elist and iwork(1:last) as iord, which is what
    ! scipy's full_output infodict holds.
    recursive subroutine dqags_wrapper(cfcn, a, b, epsabs, epsrel, result, &
                             abserr, neval, ier, limit, iwork, work, &
                             args, nargs, last) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        real(c_double), intent(in) :: a, b, epsabs, epsrel
        real(c_double), intent(out) :: result, abserr
        integer(c_int), intent(out) :: neval, ier
        integer(c_int), intent(in) :: limit
        integer(c_int), intent(inout) :: iwork(limit)
        real(c_double), intent(inout) :: work(4*limit)
        real(c_double), intent(in), target :: args(nargs)
        integer(c_int), intent(in) :: nargs
        integer(c_int), intent(out) :: last

        procedure(numba_f), pointer :: cb_save
        real(c_double), pointer :: cb_args_save(:)

        cb_save => cb
        cb_args_save => cb_args
        call c_f_procpointer(cfcn, cb)
        cb_args => args
        call dqags(f_adapter, a, b, epsabs, epsrel, result, abserr, &
                   neval, ier, limit, 4*limit, last, iwork, work)
        cb => cb_save
        cb_args => cb_args_save
    end subroutine dqags_wrapper

    ! semi-infinite / infinite interval. inf = 1 -> (bound, +inf),
    ! -1 -> (-inf, bound), 2 -> (-inf, +inf). scipy uses this for
    ! infinite bounds. 16 C arguments.
    !
    ! alist/blist here are in the TRANSFORMED variable, not the user's.
    recursive subroutine dqagi_wrapper(cfcn, bound, inf, epsabs, epsrel, result, &
                             abserr, neval, ier, limit, iwork, work, &
                             args, nargs, last) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        real(c_double), intent(in) :: bound, epsabs, epsrel
        integer(c_int), intent(in) :: inf
        real(c_double), intent(out) :: result, abserr
        integer(c_int), intent(out) :: neval, ier
        integer(c_int), intent(in) :: limit
        integer(c_int), intent(inout) :: iwork(limit)
        real(c_double), intent(inout) :: work(4*limit)
        real(c_double), intent(in), target :: args(nargs)
        integer(c_int), intent(in) :: nargs
        integer(c_int), intent(out) :: last

        procedure(numba_f), pointer :: cb_save
        real(c_double), pointer :: cb_args_save(:)

        cb_save => cb
        cb_args_save => cb_args
        call c_f_procpointer(cfcn, cb)
        cb_args => args
        call dqagi(f_adapter, bound, inf, epsabs, epsrel, result, &
                   abserr, neval, ier, limit, 4*limit, last, iwork, work)
        cb => cb_save
        cb_args => cb_args_save
    end subroutine dqagi_wrapper

    ! finite interval with user-supplied break points (known
    ! singularities/discontinuities interior to (a,b)). points has
    ! npts2 = (#break points)+2 entries; the last two are set by the
    ! routine, so pass an array of that length. 18 C arguments.
    recursive subroutine dqagp_wrapper(cfcn, a, b, npts2, points, epsabs, &
                             epsrel, result, abserr, neval, ier, &
                             leniw, lenw, iwork, work, args, nargs, &
                             last) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        real(c_double), intent(in) :: a, b, epsabs, epsrel
        integer(c_int), intent(in) :: npts2
        real(c_double), intent(inout) :: points(npts2)
        real(c_double), intent(out) :: result, abserr
        integer(c_int), intent(out) :: neval, ier
        integer(c_int), intent(in) :: leniw, lenw
        integer(c_int), intent(inout) :: iwork(leniw)
        real(c_double), intent(inout) :: work(lenw)
        real(c_double), intent(in), target :: args(nargs)
        integer(c_int), intent(in) :: nargs
        integer(c_int), intent(out) :: last

        procedure(numba_f), pointer :: cb_save
        real(c_double), pointer :: cb_args_save(:)

        cb_save => cb
        cb_args_save => cb_args
        call c_f_procpointer(cfcn, cb)
        cb_args => args
        call dqagp(f_adapter, a, b, npts2, points, epsabs, epsrel, &
                   result, abserr, neval, ier, leniw, lenw, last, &
                   iwork, work)
        cb => cb_save
        cb_args => cb_args_save
    end subroutine dqagp_wrapper

    ! ------------------------------------------------------------------
    ! the weight family: scipy's quad(..., weight=...)
    ! ------------------------------------------------------------------

    ! oscillatory weight on a FINITE interval:
    !     integral of f(x)*cos(omega*x)  (integr = 1)
    !     integral of f(x)*sin(omega*x)  (integr = 2)
    ! scipy: weight='cos' / 'sin' with finite a and b.
    ! dqawo splits its own workspace as limit = leniw/2, and requires
    ! leniw >= 2, maxp1 >= 1, lenw >= 2*leniw + 25*maxp1.
    !
    ! `momcom` is the number of chebyshev moment sets dqawoe computed, and
    ! is the one entry of scipy's infodict on this route that no other
    ! output carries.  dqawo's Fortran 77 driver declares it a local and
    ! throws it away; 01_quadpack_double.f90 carries a LOCAL ADDITION making
    ! it an OPTIONAL intent(out) argument, four lines, so the driver's own
    ! validity check and its xerror call are untouched.  Passing it here is
    ! the only use.
    ! 20 C arguments.
    recursive subroutine dqawo_wrapper(cfcn, a, b, omega, integr, epsabs, epsrel, &
                             result, abserr, neval, ier, leniw, maxp1, &
                             lenw, iwork, work, args, nargs, last, &
                             momcom) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        real(c_double), intent(in) :: a, b, omega, epsabs, epsrel
        integer(c_int), intent(in) :: integr
        real(c_double), intent(out) :: result, abserr
        integer(c_int), intent(out) :: neval, ier
        integer(c_int), intent(in) :: leniw, maxp1, lenw
        integer(c_int), intent(inout) :: iwork(leniw)
        real(c_double), intent(inout) :: work(lenw)
        real(c_double), intent(in), target :: args(nargs)
        integer(c_int), intent(in) :: nargs
        integer(c_int), intent(out) :: last
        integer(c_int), intent(out) :: momcom

        procedure(numba_f), pointer :: cb_save
        real(c_double), pointer :: cb_args_save(:)
        integer :: mcom

        cb_save => cb
        cb_args_save => cb_args
        call c_f_procpointer(cfcn, cb)
        cb_args => args
        call dqawo(f_adapter, a, b, omega, integr, epsabs, epsrel, &
                   result, abserr, neval, ier, leniw, maxp1, lenw, &
                   last, iwork, work, mcom)
        momcom = mcom
        cb => cb_save
        cb_args => cb_args_save
    end subroutine dqawo_wrapper

    ! Fourier integral over the SEMI-INFINITE interval (a, +inf):
    !     integral of f(x)*cos(omega*x)  (integr = 1)
    !     integral of f(x)*sin(omega*x)  (integr = 2)
    ! scipy: weight='cos' / 'sin' with b = +inf.  epsrel has no meaning
    ! here, dqawf takes epsabs only.  `lst` is the number of cycles used,
    ! the dqawf analogue of `last`.
    ! dqawf requires limlst >= 3, leniw >= limlst + 2, maxp1 >= 1,
    ! lenw >= 2*leniw + 25*maxp1, and splits limit = (leniw-limlst)/2.
    ! 18 C arguments.
    recursive subroutine dqawf_wrapper(cfcn, a, omega, integr, epsabs, result, &
                             abserr, neval, ier, limlst, leniw, maxp1, &
                             lenw, iwork, work, args, nargs, lst) &
                             bind(c)
        type(c_funptr), intent(in), value :: cfcn
        real(c_double), intent(in) :: a, omega, epsabs
        integer(c_int), intent(in) :: integr
        real(c_double), intent(out) :: result, abserr
        integer(c_int), intent(out) :: neval, ier
        integer(c_int), intent(in) :: limlst, leniw, maxp1, lenw
        integer(c_int), intent(inout) :: iwork(leniw)
        real(c_double), intent(inout) :: work(lenw)
        real(c_double), intent(in), target :: args(nargs)
        integer(c_int), intent(in) :: nargs
        integer(c_int), intent(out) :: lst

        procedure(numba_f), pointer :: cb_save
        real(c_double), pointer :: cb_args_save(:)

        cb_save => cb
        cb_args_save => cb_args
        call c_f_procpointer(cfcn, cb)
        cb_args => args
        call dqawf(f_adapter, a, omega, integr, epsabs, result, abserr, &
                   neval, ier, limlst, lst, leniw, maxp1, lenw, iwork, &
                   work)
        cb => cb_save
        cb_args => cb_args_save
    end subroutine dqawf_wrapper

    ! algebraico-logarithmic end-point singularities on (a, b):
    !     w(x) = (x-a)**alfa * (b-x)**beta * v(x),  alfa > -1, beta > -1
    !     integr = 1  v(x) = 1
    !            = 2  v(x) = log(x-a)
    !            = 3  v(x) = log(b-x)
    !            = 4  v(x) = log(x-a)*log(b-x)
    ! scipy: weight='alg' / 'alg-loga' / 'alg-logb' / 'alg-log'.
    ! dqaws requires limit >= 2 and lenw >= 4*limit.  19 C arguments.
    recursive subroutine dqaws_wrapper(cfcn, a, b, alfa, beta, integr, epsabs, &
                             epsrel, result, abserr, neval, ier, limit, &
                             lenw, iwork, work, args, nargs, last) &
                             bind(c)
        type(c_funptr), intent(in), value :: cfcn
        real(c_double), intent(in) :: a, b, alfa, beta, epsabs, epsrel
        integer(c_int), intent(in) :: integr
        real(c_double), intent(out) :: result, abserr
        integer(c_int), intent(out) :: neval, ier
        integer(c_int), intent(in) :: limit, lenw
        integer(c_int), intent(inout) :: iwork(limit)
        real(c_double), intent(inout) :: work(lenw)
        real(c_double), intent(in), target :: args(nargs)
        integer(c_int), intent(in) :: nargs
        integer(c_int), intent(out) :: last

        procedure(numba_f), pointer :: cb_save
        real(c_double), pointer :: cb_args_save(:)

        cb_save => cb
        cb_args_save => cb_args
        call c_f_procpointer(cfcn, cb)
        cb_args => args
        call dqaws(f_adapter, a, b, alfa, beta, integr, epsabs, epsrel, &
                   result, abserr, neval, ier, limit, lenw, last, &
                   iwork, work)
        cb => cb_save
        cb_args => cb_args_save
    end subroutine dqaws_wrapper

    ! Cauchy principal value of f(x)/(x-c) over (a, b), c /= a, c /= b.
    ! scipy: weight='cauchy', wvar = c.
    ! dqawc requires limit >= 1 and lenw >= 4*limit.  18 C arguments.
    recursive subroutine dqawc_wrapper(cfcn, a, b, c, epsabs, epsrel, result, &
                             abserr, neval, ier, limit, lenw, iwork, &
                             work, args, nargs, last) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        real(c_double), intent(in) :: a, b, c, epsabs, epsrel
        real(c_double), intent(out) :: result, abserr
        integer(c_int), intent(out) :: neval, ier
        integer(c_int), intent(in) :: limit, lenw
        integer(c_int), intent(inout) :: iwork(limit)
        real(c_double), intent(inout) :: work(lenw)
        real(c_double), intent(in), target :: args(nargs)
        integer(c_int), intent(in) :: nargs
        integer(c_int), intent(out) :: last

        procedure(numba_f), pointer :: cb_save
        real(c_double), pointer :: cb_args_save(:)

        cb_save => cb
        cb_args_save => cb_args
        call c_f_procpointer(cfcn, cb)
        cb_args => args
        call dqawc(f_adapter, a, b, c, epsabs, epsrel, result, abserr, &
                   neval, ier, limit, lenw, last, iwork, work)
        cb => cb_save
        cb_args => cb_args_save
    end subroutine dqawc_wrapper

end module quadpack_wrappers
