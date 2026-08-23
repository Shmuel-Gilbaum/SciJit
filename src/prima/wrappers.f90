! bind(c) wrappers exposing PRIMA (libprima/prima, Zaikun Zhang's
! modern-Fortran reference implementation of Powell's derivative-free
! solvers) to ctypes/numba.
!
! Callback architecture: same module-variable adapter design as the
! minpack pack (no nested procedures -> no trampolines -> GNU_STACK
! stays RW). The user objective is a numba @cfunc; its address arrives
! by value, the adapters below are ordinary module procedures matching
! PRIMA's OBJ/OBJCON abstract interfaces (assumed-shape x, real64).
! Trade-off as in minpack: module state -> solver calls are
! NON-REENTRANT (no two simultaneous solves; prange forbidden).
!
! cfunc ABIs (all pointers, Fortran passes by reference):
!   objective only  (uobyqa/newuoa/bobyqa/lincoa):
!       void(x*, f*, args*)                      = prima_sig
!   objective + nonlinear constraints (cobyla):
!       void(x*, f*, constr*, args*)             = prima_con_sig
!       constraints are cstrv-style: PRIMA requires constr(x) <= 0
!
! The python glue always passes every "optional" explicitly (rhobeg,
! rhoend, maxfun, npt, iprint=0...), with PRIMA's documented defaults
! computed on the numba side. info success codes (infos.f90):
!   0 = SMALL_TR_RADIUS (normal convergence), 1 = FTARGET_ACHIEVED;
!   3 = MAXFUN_REACHED, 20 = MAXTR_REACHED, negatives = NaN/Inf trouble.
module prima_wrappers
    use iso_c_binding, only: c_double, c_int, c_funptr, c_f_procpointer
    use consts_mod, only: RP, IK
    use uobyqa_mod, only: uobyqa
    use newuoa_mod, only: newuoa
    use bobyqa_mod, only: bobyqa
    use lincoa_mod, only: lincoa
    use cobyla_mod, only: cobyla
    implicit none
    private
    public :: uobyqa_wrapper, newuoa_wrapper, bobyqa_wrapper, &
              lincoa_wrapper, cobyla_wrapper

    abstract interface
        subroutine numba_obj(x, f, args)             ! prima_sig
            import :: c_double
            implicit none
            real(c_double), intent(in) :: x(*), args(*)
            real(c_double), intent(out) :: f
        end subroutine numba_obj

        subroutine numba_objcon(x, f, constr, args)  ! prima_con_sig
            import :: c_double
            implicit none
            real(c_double), intent(in) :: x(*), args(*)
            real(c_double), intent(out) :: f
            real(c_double), intent(out) :: constr(*)
        end subroutine numba_objcon
    end interface

    ! module-scope state read by the adapters (set per solver call)
    procedure(numba_obj), pointer :: cb_obj => null()
    procedure(numba_objcon), pointer :: cb_objcon => null()
    real(c_double), pointer :: cb_args(:) => null()
    !$omp threadprivate(cb_obj, cb_objcon, cb_args)

contains

    ! adapters, ordinary module procedures matching OBJ / OBJCON
    subroutine calfun_adapter(x, f)
        real(RP), intent(in) :: x(:)
        real(RP), intent(out) :: f
        call cb_obj(x, f, cb_args)
    end subroutine calfun_adapter

    subroutine calcfc_adapter(x, f, constr)
        real(RP), intent(in) :: x(:)
        real(RP), intent(out) :: f
        real(RP), intent(out) :: constr(:)
        call cb_objcon(x, f, constr, cb_args)
        ! scijit convention (as scipy / minimize_slsqp): user returns
        ! c(x) with c >= 0 feasible; PRIMA wants constr <= 0, negate.
        constr = -constr
    end subroutine calcfc_adapter

    ! ------------------------------------------------------------------
    ! 11 args.
    subroutine uobyqa_wrapper(cfcn, n, x, f, rhobeg, rhoend, maxfun, &
                              nf, info, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: n, maxfun, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: f
        real(c_double), intent(in) :: rhobeg, rhoend
        integer(c_int), intent(out) :: nf, info
        real(c_double), intent(in), target :: args(nargs)
        integer(IK) :: nf_, info_

        call c_f_procpointer(cfcn, cb_obj)
        cb_args => args
        call uobyqa(calfun_adapter, x, f, nf=nf_, &
                    rhobeg=real(rhobeg, RP), rhoend=real(rhoend, RP), &
                    maxfun=int(maxfun, IK), iprint=0_IK, info=info_)
        nf = int(nf_, c_int)
        info = int(info_, c_int)
    end subroutine uobyqa_wrapper

    ! 12 args.
    subroutine newuoa_wrapper(cfcn, n, x, f, rhobeg, rhoend, maxfun, &
                              npt, nf, info, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: n, maxfun, npt, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out) :: f
        real(c_double), intent(in) :: rhobeg, rhoend
        integer(c_int), intent(out) :: nf, info
        real(c_double), intent(in), target :: args(nargs)
        integer(IK) :: nf_, info_

        call c_f_procpointer(cfcn, cb_obj)
        cb_args => args
        call newuoa(calfun_adapter, x, f, nf=nf_, &
                    rhobeg=real(rhobeg, RP), rhoend=real(rhoend, RP), &
                    maxfun=int(maxfun, IK), npt=int(npt, IK), &
                    iprint=0_IK, info=info_)
        nf = int(nf_, c_int)
        info = int(info_, c_int)
    end subroutine newuoa_wrapper

    ! 14 args.
    subroutine bobyqa_wrapper(cfcn, n, x, xl, xu, f, rhobeg, rhoend, &
                              maxfun, npt, nf, info, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: n, maxfun, npt, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(in) :: xl(n), xu(n)
        real(c_double), intent(out) :: f
        real(c_double), intent(in) :: rhobeg, rhoend
        integer(c_int), intent(out) :: nf, info
        real(c_double), intent(in), target :: args(nargs)
        integer(IK) :: nf_, info_

        call c_f_procpointer(cfcn, cb_obj)
        cb_args => args
        call bobyqa(calfun_adapter, x, f, xl=xl, xu=xu, nf=nf_, &
                    rhobeg=real(rhobeg, RP), rhoend=real(rhoend, RP), &
                    maxfun=int(maxfun, IK), npt=int(npt, IK), &
                    iprint=0_IK, info=info_)
        nf = int(nf_, c_int)
        info = int(info_, c_int)
    end subroutine bobyqa_wrapper

    ! 21 args. Aineq is (m_ineq, n) column-major, Aeq is (m_eq, n);
    ! zero-size arrays mean "no such constraints".
    subroutine lincoa_wrapper(cfcn, n, x, m_ineq, Aineq, bineq, m_eq, &
                              Aeq, beq, xl, xu, f, cstrv, rhobeg, &
                              rhoend, maxfun, npt, nf, info, args, &
                              nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: n, m_ineq, m_eq, maxfun, npt, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(in) :: Aineq(m_ineq, n), bineq(m_ineq)
        real(c_double), intent(in) :: Aeq(m_eq, n), beq(m_eq)
        real(c_double), intent(in) :: xl(n), xu(n)
        real(c_double), intent(out) :: f, cstrv
        real(c_double), intent(in) :: rhobeg, rhoend
        integer(c_int), intent(out) :: nf, info
        real(c_double), intent(in), target :: args(nargs)
        integer(IK) :: nf_, info_

        call c_f_procpointer(cfcn, cb_obj)
        cb_args => args
        call lincoa(calfun_adapter, x, f, cstrv=cstrv, &
                    Aineq=Aineq, bineq=bineq, Aeq=Aeq, beq=beq, &
                    xl=xl, xu=xu, nf=nf_, &
                    rhobeg=real(rhobeg, RP), rhoend=real(rhoend, RP), &
                    maxfun=int(maxfun, IK), npt=int(npt, IK), &
                    iprint=0_IK, info=info_)
        nf = int(nf_, c_int)
        info = int(info_, c_int)
    end subroutine lincoa_wrapper

    ! 23 args. Nonlinear constraints: PRIMA convention constr(x) <= 0.
    ! ctol is the constraint-violation tolerance PRIMA uses when it selects
    ! the returned point.  scipy 1.18 passes fmin_cobyla's `catol` here
    ! (default 2e-4) and minimize(method='COBYLA')'s, which defaults to
    ! sqrt(eps); before this argument existed the solver always used
    ! CTOL_DFT = sqrt(EPS) and `catol` was only a post-check on maxcv.
    subroutine cobyla_wrapper(cfcn, m_nlcon, n, x, m_ineq, Aineq, &
                              bineq, m_eq, Aeq, beq, xl, xu, f, cstrv, &
                              nlconstr, rhobeg, rhoend, ctol, maxfun, nf, &
                              info, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: m_nlcon, n, m_ineq, m_eq
        integer(c_int), intent(in) :: maxfun, nargs
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(in) :: Aineq(m_ineq, n), bineq(m_ineq)
        real(c_double), intent(in) :: Aeq(m_eq, n), beq(m_eq)
        real(c_double), intent(in) :: xl(n), xu(n)
        real(c_double), intent(out) :: f, cstrv
        real(c_double), intent(out) :: nlconstr(m_nlcon)
        real(c_double), intent(in) :: rhobeg, rhoend, ctol
        integer(c_int), intent(out) :: nf, info
        real(c_double), intent(in), target :: args(nargs)
        integer(IK) :: nf_, info_

        call c_f_procpointer(cfcn, cb_objcon)
        cb_args => args
        call cobyla(calcfc_adapter, int(m_nlcon, IK), x, f, &
                    cstrv=cstrv, nlconstr=nlconstr, &
                    Aineq=Aineq, bineq=bineq, Aeq=Aeq, beq=beq, &
                    xl=xl, xu=xu, nf=nf_, &
                    rhobeg=real(rhobeg, RP), rhoend=real(rhoend, RP), &
                    ctol=real(ctol, RP), &
                    maxfun=int(maxfun, IK), iprint=0_IK, info=info_)
        nf = int(nf_, c_int)
        info = int(info_, c_int)
    end subroutine cobyla_wrapper

end module prima_wrappers
