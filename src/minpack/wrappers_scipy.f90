! bind(c) wrappers exposing the FULL hybrd/hybrj drivers with the same
! argument choices scipy.optimize.fsolve makes, so the two agree bit for bit.
!
! Why this file exists alongside wrappers.f90:
!   hybrd_wrapper there replicates the hybrd1 SIMPLE driver -- mode = 2 with
!   diag = 1 (i.e. no scaling) and info 5 remapped to 4.  scipy's _minpack
!   layer instead passes mode = 1 when diag is absent (MINPACK derives the
!   scaling from the jacobian column norms and keeps updating it) and reports
!   info 5 unchanged.  On well-scaled systems the two paths coincide; on
!   badly-scaled or higher-dimensional ones they take different steps and land
!   ~1e-10 apart.  These wrappers take mode/diag/factor/epsfcn/ml/mu as real
!   arguments and return nfev/fjac/r/qtf, which is everything scipy's
!   `infodict` reports.
!
!   lmdif_wrapper there replicates the lmdif1 SIMPLE driver -- ftol = xtol =
!   tol, gtol = 0, epsfcn = 0, mode = 1, factor = 100, info 8 remapped to 4,
!   and it returns neither ipvt nor qtf nor fjac nor nfev.  scipy's
!   `leastsq` passes ftol, xtol, gtol, maxfev, epsfcn, factor and diag
!   independently and reports fjac/ipvt/qtf/nfev in its `infodict`, from which
!   `cov_x` is built.  lmdif_sp_wrapper / lmder_sp_wrapper expose all of it.
!
! Symbols: hybrd_sp_wrapper (20 args), hybrj_sp_wrapper (18 args),
!          lmdif_sp_wrapper (21 args), lmder_sp_wrapper (21 args).
! Callback ABI is unchanged: minpack_sig / minpack_jac_sig, reached through a
! module variable (never a nested procedure -- those need an executable stack).
module minpack_scipy_wrappers
    use iso_c_binding, only: c_double, c_int, c_funptr, c_f_procpointer
    use iso_fortran_env, only: wp => real64
    use minpack_module, only: hybrd, hybrj, lmdif, lmder
    implicit none
    private
    public :: hybrd_sp_wrapper, hybrj_sp_wrapper, &
              lmdif_sp_wrapper, lmder_sp_wrapper

    abstract interface
        subroutine numba_cb(x, fvec, args)              ! minpack_sig
            import :: c_double
            implicit none
            real(c_double), intent(in) :: x(*), args(*)
            real(c_double), intent(out) :: fvec(*)
        end subroutine numba_cb

        subroutine numba_cb_jac(x, fvec, fjac, iflag, args)  ! minpack_jac_sig
            import :: c_double, c_int
            implicit none
            real(c_double), intent(in) :: x(*), args(*)
            real(c_double), intent(inout) :: fvec(*), fjac(*)
            integer(c_int), intent(in) :: iflag
        end subroutine numba_cb_jac
    end interface

    procedure(numba_cb), pointer :: cb => null()
    procedure(numba_cb_jac), pointer :: cb_jac => null()
    real(c_double), pointer :: cb_args(:) => null()
    !$omp threadprivate(cb, cb_jac, cb_args)

contains

    subroutine fcn_hybrd(n, x, fvec, iflag)     ! intents must match `func`
        integer, intent(in) :: n
        real(wp), intent(in) :: x(n)
        real(wp), intent(out) :: fvec(n)
        integer, intent(inout) :: iflag
        call cb(x, fvec, cb_args)
    end subroutine fcn_hybrd

    subroutine fcn_hybrj(n, x, fvec, fjac, ldfjac, iflag)
        integer, intent(in) :: n, ldfjac
        real(wp), dimension(n), intent(in) :: x
        real(wp), dimension(n), intent(inout) :: fvec
        real(wp), dimension(ldfjac, n), intent(inout) :: fjac
        integer, intent(inout) :: iflag
        call cb_jac(x, fvec, fjac, iflag, cb_args)
    end subroutine fcn_hybrj

    subroutine fcn_lmdif(m, n, x, fvec, iflag)      ! matches func2
        integer, intent(in) :: m, n
        real(wp), intent(in) :: x(n)
        real(wp), intent(out) :: fvec(m)
        integer, intent(inout) :: iflag
        call cb(x, fvec, cb_args)
    end subroutine fcn_lmdif

    subroutine fcn_lmder(m, n, x, fvec, fjac, ldfjac, iflag)
        integer, intent(in) :: m, n, ldfjac
        real(wp), intent(in) :: x(n)
        real(wp), intent(inout) :: fvec(m)
        real(wp), intent(inout) :: fjac(ldfjac, n)
        integer, intent(inout) :: iflag
        call cb_jac(x, fvec, fjac, iflag, cb_args)
    end subroutine fcn_lmder

    ! Full hybrd, forward-difference jacobian.  Mirrors scipy _minpack._hybrd.
    ! ml/mu negative -> n-1 (scipy passes -10 for "no band").  20 args.
    subroutine hybrd_sp_wrapper(cfcn, n, x, fvec, xtol, maxfev, ml, mu, &
                                epsfcn, mode, diag, factor, info, nfev, &
                                fjac, r, lr, qtf, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: n, maxfev, ml, mu, mode, lr, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: fvec(n)
        real(c_double), intent(in) :: xtol, epsfcn, factor
        real(c_double), intent(inout) :: diag(n)
        integer(c_int), intent(out) :: info, nfev
        real(c_double), intent(out) :: fjac(n, n), r(lr), qtf(n)
        real(c_double), intent(in), target :: args(nargs)
        integer :: ml_, mu_, nprint
        real(wp), allocatable :: wa1(:), wa2(:), wa3(:), wa4(:)

        call c_f_procpointer(cfcn, cb)
        cb_args => args

        info = 0
        nfev = 0
        if (n <= 0 .or. xtol < 0.0_c_double .or. maxfev <= 0 .or. &
            factor <= 0.0_c_double .or. lr < (n*(n + 1))/2) return

        ml_ = ml
        mu_ = mu
        if (ml_ < 0) ml_ = n - 1
        if (mu_ < 0) mu_ = n - 1
        nprint = 0

        allocate (wa1(n), wa2(n), wa3(n), wa4(n))
        call hybrd(fcn_hybrd, n, x, fvec, xtol, maxfev, ml_, mu_, epsfcn, &
                   diag, mode, factor, nprint, info, nfev, fjac, n, r, lr, &
                   qtf, wa1, wa2, wa3, wa4)
        deallocate (wa1, wa2, wa3, wa4)
        ! NOTE: no `if (info == 5) info = 4` here.  hybrd1 does that remap;
        ! scipy does not, and reports 5 for "not making good progress".
    end subroutine hybrd_sp_wrapper

    ! Full hybrj, analytic jacobian.  Mirrors scipy _minpack._hybrj.  18 args.
    subroutine hybrj_sp_wrapper(cfcn, n, x, fvec, fjac, xtol, maxfev, mode, &
                                diag, factor, info, nfev, njev, r, lr, qtf, &
                                args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: n, maxfev, mode, lr, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: fvec(n)
        real(c_double), intent(out) :: fjac(n, n), r(lr), qtf(n)
        real(c_double), intent(in) :: xtol, factor
        real(c_double), intent(inout) :: diag(n)
        integer(c_int), intent(out) :: info, nfev, njev
        real(c_double), intent(in), target :: args(nargs)
        integer :: nprint
        real(wp), allocatable :: wa1(:), wa2(:), wa3(:), wa4(:)

        call c_f_procpointer(cfcn, cb_jac)
        cb_args => args

        info = 0
        nfev = 0
        njev = 0
        if (n <= 0 .or. xtol < 0.0_c_double .or. maxfev <= 0 .or. &
            factor <= 0.0_c_double .or. lr < (n*(n + 1))/2) return

        nprint = 0
        allocate (wa1(n), wa2(n), wa3(n), wa4(n))
        call hybrj(fcn_hybrj, n, x, fvec, fjac, n, xtol, maxfev, diag, mode, &
                   factor, nprint, info, nfev, njev, r, lr, qtf, &
                   wa1, wa2, wa3, wa4)
        deallocate (wa1, wa2, wa3, wa4)
    end subroutine hybrj_sp_wrapper

    ! Full lmdif, forward-difference jacobian.  Mirrors scipy _minpack._lmdif,
    ! which is what scipy.optimize.leastsq(Dfun=None) calls.  21 args.
    ! fjac is (ldfjac, n) column-major with ldfjac = m, i.e. the same bytes as
    ! the (n, m) C-order array scipy reports in infodict['fjac'].
    subroutine lmdif_sp_wrapper(cfcn, m, n, x, fvec, ftol, xtol, gtol, &
                                maxfev, epsfcn, mode, diag, factor, info, &
                                nfev, fjac, ldfjac, ipvt, qtf, args, &
                                nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: m, n, maxfev, mode, ldfjac, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: fvec(m)
        real(c_double), intent(in) :: ftol, xtol, gtol, epsfcn, factor
        real(c_double), intent(inout) :: diag(n)
        integer(c_int), intent(out) :: info, nfev
        real(c_double), intent(out) :: fjac(ldfjac, n), qtf(n)
        integer(c_int), intent(out) :: ipvt(n)
        real(c_double), intent(in), target :: args(nargs)
        integer :: nprint, i
        integer, allocatable :: ipvt_(:)
        real(wp), allocatable :: wa1(:), wa2(:), wa3(:), wa4(:)

        call c_f_procpointer(cfcn, cb)
        cb_args => args

        info = 0
        nfev = 0
        if (n <= 0 .or. m < n .or. ldfjac < m .or. maxfev <= 0 .or. &
            ftol < 0.0_c_double .or. xtol < 0.0_c_double .or. &
            gtol < 0.0_c_double .or. factor <= 0.0_c_double) return

        nprint = 0
        allocate (wa1(n), wa2(n), wa3(n), wa4(m), ipvt_(n))
        call lmdif(fcn_lmdif, m, n, x, fvec, ftol, xtol, gtol, maxfev, &
                   epsfcn, diag, mode, factor, nprint, info, nfev, fjac, &
                   ldfjac, ipvt_, qtf, wa1, wa2, wa3, wa4)
        do i = 1, n
            ipvt(i) = int(ipvt_(i), c_int)
        end do
        deallocate (wa1, wa2, wa3, wa4, ipvt_)
        ! NOTE: no `if (info == 8) info = 4` here.  lmdif1 does that remap;
        ! scipy does not, and reports 8 with its own message.
    end subroutine lmdif_sp_wrapper

    ! Full lmder, analytic jacobian.  Mirrors scipy _minpack._lmder, which is
    ! what scipy.optimize.leastsq(Dfun=callable) calls.  21 args.
    subroutine lmder_sp_wrapper(cfcn, m, n, x, fvec, fjac, ldfjac, ftol, &
                                xtol, gtol, maxfev, mode, diag, factor, &
                                info, nfev, njev, ipvt, qtf, args, &
                                nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: m, n, ldfjac, maxfev, mode, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: fvec(m)
        real(c_double), intent(inout) :: fjac(ldfjac, n)
        real(c_double), intent(in) :: ftol, xtol, gtol, factor
        real(c_double), intent(inout) :: diag(n)
        integer(c_int), intent(out) :: info, nfev, njev
        integer(c_int), intent(out) :: ipvt(n)
        real(c_double), intent(out) :: qtf(n)
        real(c_double), intent(in), target :: args(nargs)
        integer :: nprint, i
        integer, allocatable :: ipvt_(:)
        real(wp), allocatable :: wa1(:), wa2(:), wa3(:), wa4(:)

        call c_f_procpointer(cfcn, cb_jac)
        cb_args => args

        info = 0
        nfev = 0
        njev = 0
        if (n <= 0 .or. m < n .or. ldfjac < m .or. maxfev <= 0 .or. &
            ftol < 0.0_c_double .or. xtol < 0.0_c_double .or. &
            gtol < 0.0_c_double .or. factor <= 0.0_c_double) return

        nprint = 0
        allocate (wa1(n), wa2(n), wa3(n), wa4(m), ipvt_(n))
        call lmder(fcn_lmder, m, n, x, fvec, fjac, ldfjac, ftol, xtol, gtol, &
                   maxfev, diag, mode, factor, nprint, info, nfev, njev, &
                   ipvt_, qtf, wa1, wa2, wa3, wa4)
        do i = 1, n
            ipvt(i) = int(ipvt_(i), c_int)
        end do
        deallocate (wa1, wa2, wa3, wa4, ipvt_)
    end subroutine lmder_sp_wrapper

end module minpack_scipy_wrappers
