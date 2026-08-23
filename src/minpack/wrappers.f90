! bind(c) wrapper module exposing MINPACK (fortran-lang/minpack modern
! module, src/minpack.f90) to ctypes/numba, following the NumbaFitpack
! architecture: every argument is passed by reference through a c_void_p
! slot, except the callback address which arrives by value.
!
! Callback design, module-variable adapters, NO nested procedures.
! ----------------------------------------------------------------
! The original NumbaMinpack (and fortran-lang's own minpack_capi.f90)
! pass a *contained* procedure to the solver, capturing the user cfunc
! and args by host association. gfortran implements that with a stack
! trampoline, which forces GNU_STACK RWE on the whole library: hardened
! distros (Fedora/RHEL, glibc >= 2.41) then refuse to dlopen it, and
! stripping the flag afterwards segfaults at the first callback.
!
! Here the cfunc pointer and the args pointer live in module variables,
! and the adapter passed to MINPACK is an ordinary module procedure:
! no trampoline is ever generated, the stack stays RW everywhere.
!
! Trade-off (document, do not "fix"): module state makes solver calls
! non-reentrant, no two simultaneous solves in one process (numba
! prange users beware). Same practical situation as most F77 wrappers.
!
! User callback ABIs (numba @cfunc side, all args are pointers because
! Fortran passes by reference):
!   fcn only    (hybrd/lmdif) : void(x*, fvec*, args*)          = minpack_sig
!   fcn + jac   (hybrj/lmder) : void(x*, fvec*, fjac*, iflag*, args*)
!   fcn + rows  (lmstr)       : void(x*, fvec*, fjrow*, iflag*, args*)
! For the jacobian variants MINPACK drives a two-phase protocol:
!   iflag = 1 -> fill fvec, do not touch fjac
!   iflag = 2 -> fill fjac (COLUMN-major, flattened, ld = n for hybrj,
!                ld = m for lmder), do not touch fvec
!   lmstr: iflag >= 2 -> fill fjrow with row (iflag-1) of the jacobian
module minpack_wrappers
    use iso_c_binding, only: c_double, c_int, c_funptr, c_f_procpointer
    use iso_fortran_env, only: wp => real64   ! same kind minpack_module uses
    use minpack_module, only: hybrd, lmdif, hybrj1, lmder1, lmstr1, &
                              chkder, enorm
    implicit none
    private
    public :: hybrd_wrapper, lmdif_wrapper, hybrj1_wrapper, &
              lmder1_wrapper, lmstr1_wrapper, chkder_wrapper, enorm_wrapper

    abstract interface
        subroutine numba_cb(x, fvec, args)          ! minpack_sig
            import :: c_double
            implicit none
            real(c_double), intent(in) :: x(*), args(*)
            real(c_double), intent(out) :: fvec(*)
        end subroutine numba_cb

        subroutine numba_cb_jac(x, fvec, fjac, iflag, args)
            import :: c_double, c_int
            implicit none
            real(c_double), intent(in) :: x(*), args(*)
            real(c_double), intent(inout) :: fvec(*), fjac(*)
            integer(c_int), intent(in) :: iflag
        end subroutine numba_cb_jac
    end interface
    ! lmstr's row callback has the identical shape as numba_cb_jac with
    ! fjrow in place of fjac, one interface serves both.

    ! module-scope state read by the adapters (set per solver call)
    procedure(numba_cb), pointer :: cb => null()
    procedure(numba_cb_jac), pointer :: cb_jac => null()
    real(c_double), pointer :: cb_args(:) => null()
    !$omp threadprivate(cb, cb_jac, cb_args)

contains

    ! ------------------------------------------------------------------
    ! adapters, ordinary module procedures, signatures matching the
    ! abstract interfaces in minpack_module exactly (default integer,
    ! real(wp)); no trampoline is ever created.
    ! ------------------------------------------------------------------

    subroutine fcn_hybrd(n, x, fvec, iflag)         ! matches func
        integer, intent(in) :: n
        real(wp), intent(in) :: x(n)
        real(wp), intent(out) :: fvec(n)
        integer, intent(inout) :: iflag
        call cb(x, fvec, cb_args)
    end subroutine fcn_hybrd

    subroutine fcn_lmdif(m, n, x, fvec, iflag)      ! matches func2
        integer, intent(in) :: m, n
        real(wp), intent(in) :: x(n)
        real(wp), intent(out) :: fvec(m)
        integer, intent(inout) :: iflag
        call cb(x, fvec, cb_args)
    end subroutine fcn_lmdif

    subroutine fcn_hybrj(n, x, fvec, fjac, ldfjac, iflag)  ! matches fcn_hybrj
        integer, intent(in) :: n, ldfjac
        real(wp), dimension(n), intent(in) :: x
        real(wp), dimension(n), intent(inout) :: fvec
        real(wp), dimension(ldfjac, n), intent(inout) :: fjac
        integer, intent(inout) :: iflag
        call cb_jac(x, fvec, fjac, iflag, cb_args)
    end subroutine fcn_hybrj

    subroutine fcn_lmder(m, n, x, fvec, fjac, ldfjac, iflag)  ! matches fcn_lmder
        integer, intent(in) :: m, n, ldfjac
        real(wp), intent(in) :: x(n)
        real(wp), intent(inout) :: fvec(m)
        real(wp), intent(inout) :: fjac(ldfjac, n)
        integer, intent(inout) :: iflag
        call cb_jac(x, fvec, fjac, iflag, cb_args)
    end subroutine fcn_lmder

    subroutine fcn_lmstr(m, n, x, fvec, fjrow, iflag)  ! matches fcn_lmstr
        integer, intent(in) :: m, n
        real(wp), intent(in) :: x(n)
        real(wp), intent(inout) :: fvec(m)
        real(wp), intent(inout) :: fjrow(n)
        integer, intent(inout) :: iflag
        call cb_jac(x, fvec, fjrow, iflag, cb_args)
    end subroutine fcn_lmstr

    ! ------------------------------------------------------------------
    ! bind(c) wrappers
    ! ------------------------------------------------------------------

    ! hybrd1 semantics with a maxfev passthrough (the *1 driver hardcodes
    ! maxfev = 200*(n+1); NumbaMinpack patched its F77 to expose it, we
    ! replicate the hybrd1 body instead, same defaults, same info map).
    ! 11 args.
    subroutine hybrd_wrapper(cfcn, n, x, fvec, tol, maxfev, info, &
                             wa, lwa, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: n, maxfev, lwa, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: fvec(n)
        real(c_double), intent(in) :: tol
        integer(c_int), intent(out) :: info
        real(c_double), intent(inout) :: wa(lwa)
        real(c_double), intent(in), target :: args(nargs)
        integer(c_int) :: index, j, lr, ml, mode, mu, nfev, nprint
        real(c_double) :: epsfcn
        real(c_double), parameter :: factor = 100.0_c_double

        call c_f_procpointer(cfcn, cb)
        cb_args => args

        info = 0
        if (n > 0 .and. tol >= 0.0_c_double .and. &
            lwa >= (n*(3*n + 13))/2) then
            ml = n - 1
            mu = n - 1
            epsfcn = 0.0_c_double
            mode = 2
            do j = 1, n
                wa(j) = 1.0_c_double
            end do
            nprint = 0
            lr = (n*(n + 1))/2
            index = 6*n + lr
            call hybrd(fcn_hybrd, n, x, fvec, tol, maxfev, ml, mu, epsfcn, &
                       wa(1), mode, factor, nprint, info, nfev, &
                       wa(index + 1), n, wa(6*n + 1), lr, &
                       wa(n + 1), wa(2*n + 1), wa(3*n + 1), wa(4*n + 1), &
                       wa(5*n + 1))
            if (info == 5) info = 4
        end if
    end subroutine hybrd_wrapper

    ! lmdif1 semantics with a maxfev passthrough (same rationale). 13 args.
    subroutine lmdif_wrapper(cfcn, m, n, x, fvec, tol, maxfev, info, &
                             iwa, wa, lwa, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: m, n, maxfev, lwa, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: fvec(m)
        real(c_double), intent(in) :: tol
        integer(c_int), intent(out) :: info
        integer(c_int), intent(inout) :: iwa(n)
        real(c_double), intent(inout) :: wa(lwa)
        real(c_double), intent(in), target :: args(nargs)
        integer(c_int) :: mode, mp5n, nfev, nprint
        real(c_double) :: epsfcn, ftol, gtol, xtol
        real(c_double), parameter :: factor = 100.0_c_double

        call c_f_procpointer(cfcn, cb)
        cb_args => args

        info = 0
        if (n > 0 .and. m >= n .and. tol >= 0.0_c_double .and. &
            lwa >= m*n + 5*n + m) then
            ftol = tol
            xtol = tol
            gtol = 0.0_c_double
            epsfcn = 0.0_c_double
            mode = 1
            nprint = 0
            mp5n = m + 5*n
            call lmdif(fcn_lmdif, m, n, x, fvec, ftol, xtol, gtol, maxfev, &
                       epsfcn, wa(1), mode, factor, nprint, info, nfev, &
                       wa(mp5n + 1), m, iwa, &
                       wa(n + 1), wa(2*n + 1), wa(3*n + 1), wa(4*n + 1), &
                       wa(5*n + 1))
            if (info == 8) info = 4
        end if
    end subroutine lmdif_wrapper

    ! 12 args.
    subroutine hybrj1_wrapper(cfcn, n, x, fvec, fjac, ldfjac, tol, info, &
                              wa, lwa, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: n, ldfjac, lwa, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: fvec(n)
        real(c_double), intent(inout) :: fjac(ldfjac, n)
        real(c_double), intent(in) :: tol
        integer(c_int), intent(out) :: info
        real(c_double), intent(inout) :: wa(lwa)
        real(c_double), intent(in), target :: args(nargs)

        call c_f_procpointer(cfcn, cb_jac)
        cb_args => args
        call hybrj1(fcn_hybrj, n, x, fvec, fjac, ldfjac, tol, info, wa, lwa)
    end subroutine hybrj1_wrapper

    ! 14 args.
    subroutine lmder1_wrapper(cfcn, m, n, x, fvec, fjac, ldfjac, tol, &
                              info, ipvt, wa, lwa, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: m, n, ldfjac, lwa, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: fvec(m)
        real(c_double), intent(inout) :: fjac(ldfjac, n)
        real(c_double), intent(in) :: tol
        integer(c_int), intent(out) :: info
        integer(c_int), intent(inout) :: ipvt(n)
        real(c_double), intent(inout) :: wa(lwa)
        real(c_double), intent(in), target :: args(nargs)

        call c_f_procpointer(cfcn, cb_jac)
        cb_args => args
        call lmder1(fcn_lmder, m, n, x, fvec, fjac, ldfjac, tol, info, &
                    ipvt, wa, lwa)
    end subroutine lmder1_wrapper

    ! 14 args.
    subroutine lmstr1_wrapper(cfcn, m, n, x, fvec, fjac, ldfjac, tol, &
                              info, ipvt, wa, lwa, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: m, n, ldfjac, lwa, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: fvec(m)
        real(c_double), intent(inout) :: fjac(ldfjac, n)
        real(c_double), intent(in) :: tol
        integer(c_int), intent(out) :: info
        integer(c_int), intent(inout) :: ipvt(n)
        real(c_double), intent(inout) :: wa(lwa)
        real(c_double), intent(in), target :: args(nargs)

        call c_f_procpointer(cfcn, cb_jac)
        cb_args => args
        call lmstr1(fcn_lmstr, m, n, x, fvec, fjac, ldfjac, tol, info, &
                    ipvt, wa, lwa)
    end subroutine lmstr1_wrapper

    ! gradient checker, no callback: caller supplies both evaluations.
    ! 10 args.
    subroutine chkder_wrapper(m, n, x, fvec, fjac, ldfjac, xp, fvecp, &
                              mode, err) bind(c)
        integer(c_int), intent(in) :: m, n, ldfjac, mode
        real(c_double), intent(in) :: x(n), fvec(m), fjac(ldfjac, n), fvecp(m)
        real(c_double), intent(out) :: xp(n), err(m)

        call chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, mode, err)
    end subroutine chkder_wrapper

    ! euclidean norm (Fortran function -> res out-arg). 3 args.
    subroutine enorm_wrapper(n, x, res) bind(c)
        integer(c_int), intent(in) :: n
        real(c_double), intent(in) :: x(n)
        real(c_double), intent(out) :: res

        res = enorm(n, x)
    end subroutine enorm_wrapper

end module minpack_wrappers
