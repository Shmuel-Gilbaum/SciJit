! bind(c) wrapper exposing ODEPACK's LSODA / LSODAR (Nicholaswogan's
! thread-safe modern-Fortran edition) to ctypes/numba.
!
! LSODA = Adams/BDF with automatic stiffness detection and switching =
! scipy.integrate.odeint / solve_ivp(method='LSODA'). LSODAR adds root
! finding = event detection (solve_ivp events).
!
! This edition threads all solver state through an odepack_common_data
! derived type, so the SOLVER is reentrant; each call holds its own.
! lsoda_dense_wrapper takes that state from the caller instead, because a
! history longer than one buffer continues rather than starting again; see
! its own header. The only shared state is the module-variable callback
! slot (the user cfunc + args pointer), exactly as in the minpack pack ,
! because ODEPACK's f(neq,t,y,ydot,ierr) interface has no user-data
! slot. That slot is one per THREAD (`!$omp threadprivate` below), and
! each wrapper saves it on entry and restores it on exit, so an
! integration started from inside another integration's callback leaves
! the outer one's slot as it found it.
!
! numba @cfunc ABI (t by value, the rest pointers; Fortran passes by
! reference):
!     lsoda_sig = void(float64 t, float64* y, float64* ydot, float64* args)
!
! The whole t_eval loop runs inside the wrapper (common_data must
! persist across the internal LSODA steps), returning the solution at
! every requested time.
module odepack_wrappers
    use iso_c_binding, only: c_double, c_int, c_funptr, c_f_procpointer
    use iso_fortran_env, only: dp => real64
    use odepack_common, only: odepack_common_data
    use odepack_interface, only: dlsoda, dlsodar
    implicit none
    private
    public :: lsoda_wrapper, lsoda_dense_wrapper, lsodar_wrapper

    ! LSODA's state lives in two places: RWORK/IWORK, and the three former
    ! COMMON blocks, which this edition carries in odepack_common_data.  A
    ! call that continues an integration needs BOTH as the previous call left
    ! them, so lsoda_dense_wrapper takes the work arrays from the caller and
    ! copies the common blocks in and out through two flat buffers.
    !
    ! The lengths are 01_odepack_common.f90's own declarations.  Every copy
    ! below is a whole-array assignment between two constant shapes, so a
    ! wrong number here is a compile error rather than a silent short read.
    integer, parameter :: N_DLS001_R = 218, N_DLS001_I = 37
    integer, parameter :: N_DLSA01_R = 22, N_DLSA01_I = 9
    integer, parameter :: N_DLSR01_R = 5, N_DLSR01_I = 9
    integer, parameter :: NCS_R = N_DLS001_R + N_DLSA01_R + N_DLSR01_R
    integer, parameter :: NCS_I = N_DLS001_I + N_DLSA01_I + N_DLSR01_I

    abstract interface
        subroutine numba_rhs(t, y, ydot, args) bind(c)      ! lsoda_sig
            import :: c_double
            implicit none
            real(c_double), intent(in), value :: t
            real(c_double), intent(in) :: y(*), args(*)
            real(c_double), intent(out) :: ydot(*)
        end subroutine numba_rhs

        subroutine numba_grt(t, y, gout, args) bind(c)      ! event fn
            import :: c_double
            implicit none
            real(c_double), intent(in), value :: t
            real(c_double), intent(in) :: y(*), args(*)
            real(c_double), intent(out) :: gout(*)
        end subroutine numba_grt

        ! jacobian callback, jt = 1 and jt = 4.  Same ABI as the one in
        ! wrappers_scipy.f90; the two modules hold separate slots because they
        ! hold separate right-hand-side slots.
        subroutine numba_jac(t, y, pd, neq, ml, mu, nrowpd, args) bind(c)
            import :: c_double, c_int
            implicit none
            real(c_double), intent(in), value :: t
            real(c_double), intent(in) :: y(*), args(*)
            real(c_double), intent(inout) :: pd(*)
            integer(c_int), intent(in), value :: neq, ml, mu, nrowpd
        end subroutine numba_jac
    end interface

    procedure(numba_rhs), pointer :: cb => null()
    procedure(numba_grt), pointer :: cb_g => null()
    procedure(numba_jac), pointer :: cb_jac_d => null()
    real(c_double), pointer :: cb_args(:) => null()

    ! `cb_jac_d`, not `cb_jac`: gfortran gives a threadprivate PROCEDURE
    ! pointer an UNMANGLED global symbol, so the name is shared across every
    ! module in the library.  wrappers_scipy.f90 has its own `cb_jac` and the
    ! two collide at link time,
    !     ld: multiple definition of `cb_jac'
    ! which is why `cb`/`cb_g` here and `cb_t`/`cb_y` there do not overlap
    ! either.  A threadprivate DATA pointer is mangled normally, which is how
    ! both modules can hold a `cb_args`.
    !
    ! One callback slot per THREAD rather than per process.  gfortran lowers
    ! this to thread-local storage and reads the directive only under
    ! -fopenmp, so `odepack` must be in setup.py's OPENMP_PACKS.  WITHOUT
    ! that flag the line is an ordinary comment: the build succeeds,
    ! single-threaded results are unchanged, and concurrent calls silently
    ! share one slot again.  Measured on the shared-slot build, 32
    ! concurrent 3-state LSODA integrations: 3 runs in 3 aborted the
    ! process (double free or corruption, SIGSEGV).
    !$omp threadprivate(cb, cb_g, cb_jac_d, cb_args)

contains

    ! adapters matching ODEPACK's odepack_f / odepack_g / odepack_jac.
    ! RECURSIVE because the user callback may itself start an integration,
    ! which re-enters them one frame down.
    recursive subroutine f_adapter(neq, t, y, ydot, ierr)
        integer, intent(in) :: neq
        real(dp), intent(in) :: t, y(neq)
        real(dp), intent(out) :: ydot(neq)
        integer, intent(out) :: ierr
        call cb(t, y, ydot, cb_args)
        ierr = 0
    end subroutine f_adapter

    recursive subroutine g_adapter(neq, t, y, ng, gout, ierr)
        integer, intent(in) :: neq, ng
        real(dp), intent(in) :: t, y(neq)
        real(dp), intent(out) :: gout(ng)
        integer, intent(out) :: ierr
        call cb_g(t, y, gout, cb_args)
        ierr = 0
    end subroutine g_adapter

    ! jt = 2 or 5 -> internally generated Jacobian; this is never called.
    ! Still used by lsoda_wrapper and lsodar_wrapper, which are jt = 2 only.
    recursive subroutine jac_dummy(neq, t, y, ml, mu, pd, nrowpd, ierr)
        integer, intent(in) :: neq, ml, mu, nrowpd
        real(dp), intent(in) :: t
        real(dp), intent(inout) :: y(neq)
        real(dp), intent(out) :: pd(nrowpd, neq)
        integer, intent(out) :: ierr
        ierr = 0
    end subroutine jac_dummy

    ! jt = 1 or 4 -> the caller's jacobian, reached through the thread-local
    ! slot.  The `associated` test makes a cleared slot a no-op instead of a
    ! jump through a null pointer.
    recursive subroutine jac_adapter(neq, t, y, ml, mu, pd, nrowpd, ierr)
        integer, intent(in) :: neq, ml, mu, nrowpd
        real(dp), intent(in) :: t
        real(dp), intent(inout) :: y(neq)
        real(dp), intent(out) :: pd(nrowpd, neq)
        integer, intent(out) :: ierr
        if (associated(cb_jac_d)) then
            call cb_jac_d(t, y, pd, neq, ml, mu, nrowpd, cb_args)
        end if
        ierr = 0
    end subroutine jac_adapter

    ! LSODA. Integrates u0 at t_eval(1) to each later t_eval(i),
    ! filling yout(neq, ntime). 12 args.
    recursive subroutine lsoda_wrapper(cfcn, neq, u0, t_eval, ntime, yout, &
                                       rtol, atol, mxstep, istate_out, &
                                       args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn
        integer(c_int), intent(in) :: neq, ntime, mxstep, nargs
        real(c_double), intent(in) :: u0(neq), t_eval(ntime)
        real(c_double), intent(out) :: yout(neq, ntime)
        real(c_double), intent(in) :: rtol, atol
        integer(c_int), intent(out) :: istate_out
        real(c_double), intent(in), target :: args(nargs)

        type(odepack_common_data) :: cdat
        real(dp) :: y(neq), t, atol1(1)
        real(dp), allocatable :: rwork(:)
        integer, allocatable :: iwork(:)
        integer :: lrw, liw, istate, i

        ! the thread-local slots, saved on entry and restored on exit
        procedure(numba_rhs), pointer :: cb_save
        real(c_double), pointer :: cb_args_save(:)

        cb_save => cb
        cb_args_save => cb_args
        call c_f_procpointer(cfcn, cb)
        cb_args => args

        ! LSODA needs the MAX over the Adams (16 cols) and BDF (neq+9
        ! cols) methods: lrw >= 22 + neq*max(16, neq+9). The plain
        ! 22+9*neq+neq^2 undersizes small systems (neq<7) -> "illegal
        ! input".
        lrw = 22 + neq*max(16, neq + 9)
        liw = 20 + neq
        allocate (rwork(lrw), iwork(liw))
        rwork = 0.0_dp
        iwork = 0
        ! iopt=1 optional inputs, hmax must be > 0 (0 is illegal here);
        ! the max orders / step-message counts must be valid.
        rwork(6) = huge(1.0_dp)           ! hmax
        iwork(6) = mxstep                 ! MXSTEP
        iwork(7) = 10                     ! MXHNIL
        iwork(8) = 12                     ! MXORDN (Adams)
        iwork(9) = 5                      ! MXORDS (BDF)
        cdat%iprint = 0                   ! silence internal messages
        atol1(1) = atol

        y = u0
        t = t_eval(1)
        yout(:, 1) = u0
        istate = 1
        do i = 2, ntime
            call dlsoda(f_adapter, neq, y, t, t_eval(i), 1, rtol, &
                        atol1, 1, istate, 1, rwork, lrw, iwork, liw, &
                        jac_dummy, 2, cdat)
            if (istate < 0) exit
            yout(:, i) = y
        end do
        istate_out = istate
        deallocate (rwork, iwork)

        cb => cb_save
        cb_args => cb_args_save
    end subroutine lsoda_wrapper

    ! LSODA driven ONE STEP AT A TIME (itask=2), snapshotting the
    ! Nordsieck history after every accepted step.  This is what makes a
    ! callable dense solution possible for method='LSODA'.
    !
    ! DINTDY interpolates only within [TCUR - HU, TCUR], one step, so a
    ! solution callable over the whole span needs the history kept per
    ! step rather than one call at the end.  The columns of YH are
    ! h^j/j! * y^(j), so evaluation is a plain polynomial in
    ! (t - t_step)/h and needs no Fortran at all; the caller does it.
    !
    ! What each step records, with the RWORK/IWORK slots it comes from,
    ! matching what scipy's LSODA._dense_output_impl reads:
    !     t_hist   TCUR   rwork(13), the time reached
    !     h_hist   H      rwork(12), the step to be attempted NEXT, which
    !                     is the one YH is currently scaled for
    !     hu_hist  HU     rwork(11), the step just used
    !     nq_hist  NQU    iwork(14), the order of the step just taken
    !     nqn_hist NQ     iwork(15), the order to attempt next; when it
    !                     is LOWER than NQU the last column of YH was
    !                     never updated and the caller rescales it
    !     yh_hist  YH     rwork(21...), (neq, nqu+1) column-major
    !
    ! Stops at maxsteps and reports nsteps_out, so a caller can size the
    ! buffers and detect truncation rather than read uninitialised rows.
    !
    ! jt / ml / mu are LSODA's jacobian-type indicator and half-bandwidths,
    ! carried here as well as in wrappers_scipy.f90 because the reporting run
    ! and this history run are two separate integrations of the same problem:
    ! a caller who asks for a banded jacobian and dense_output would otherwise
    ! get the band on one of them.  LYH is 21 whatever jt is (odepack.f:1247),
    ! so the yh_hist read below is unaffected.  cjac is read only when jt is
    ! 1 or 4, so the other two routes may pass a null funptr.
    !
    ! itol, h0, hmax and hmin are LSODA's, and they are arguments rather than
    ! the constants they were because the history run and the reporting run
    ! are two integrations of the same problem: a caller who sets max_step,
    ! first_step, min_step or a vector rtol/atol and asks for dense output
    ! would otherwise get those settings on one of the two runs only.
    !
    !   itol   1 rtol scalar atol scalar   2 rtol scalar atol vector
    !          3 rtol vector atol scalar   4 rtol vector atol vector
    !   h0     rwork(5), the first step to attempt; 0 -> solver-determined
    !   hmax   rwork(6), the largest allowed step; 0 -> no limit
    !          (odepack.f:1226-1229 refuses only HMAX < 0, and sets
    !          HMXI = 0 when HMAX = 0, which is the iopt = 0 default at
    !          odepack.f:1200)
    !   hmin   rwork(7), the smallest allowed step; 0 -> no limit
    !
    ! RESUMING.  The history buffers are the caller's and are sized before
    ! the run, while the number of steps a problem takes is known only after
    ! it.  A caller whose buffer fills continues with `istate_in = 2` rather
    ! than starting again: ODEPACK's own documented idiom (odepack.f:414-417),
    ! `istate = 1` on the first call and 2 to carry on, with RWORK and IWORK
    ! untouched between calls (odepack.f:568-569).  That needs every piece of
    ! solver state to outlive the call, so the caller owns it:
    !
    !   istate_in         1 to start, 2 to continue
    !   rwork, lrw        RWORK and its declared length
    !   iwork, liw        IWORK and its declared length
    !   cstate_r,         the three former COMMON blocks, flattened; the
    !   cstate_i          state that is NOT in RWORK/IWORK
    !   y_out, t_out      where the segment stopped, which the next call
    !                     passes back as u0 and t0
    !
    ! A short RWORK is memory corruption, not an exception, so the lengths
    ! are recomputed here from ODEPACK's own rule and a caller who passes a
    ! short buffer is refused with istate_out = -3, ODEPACK's own code for
    ! illegal input, before anything is written.  The cstate lengths are
    ! constants of 01_odepack_common.f90 and are declared, not passed.
    !
    ! On `istate_in = 1` RWORK and IWORK are zeroed and the optional inputs
    ! written, so a buffer may be reused for an unrelated problem.  On
    ! `istate_in = 2` the only slot written is RWORK(1), TCRIT, which ITASK 5
    ! reads on every call (odepack.f:1435).
    !
    ! 39 args.
    recursive subroutine lsoda_dense_wrapper(cfcn, neq, u0, t0, tf, itol, &
                                   rtol, atol, h0, hmax, hmin, &
                                   mxstep, maxsteps, maxord, nsteps_out, &
                                   t_hist, h_hist, hu_hist, nq_hist, &
                                   nqn_hist, yh_hist, istate_out, &
                                   nfev_out, njev_out, args, &
                                   nargs, jt, ml, mu, cjac, istate_in, &
                                   rwork, lrw, iwork, liw, cstate_r, &
                                   cstate_i, y_out, t_out) bind(c)
        type(c_funptr), intent(in), value :: cfcn, cjac
        integer(c_int), intent(in) :: neq, mxstep, maxsteps, maxord, nargs
        integer(c_int), intent(in) :: jt, ml, mu, itol
        integer(c_int), intent(in) :: istate_in, lrw, liw
        real(c_double), intent(in) :: u0(neq), t0, tf
        real(c_double), intent(in) :: rtol(*), atol(*)
        real(c_double), intent(in) :: h0, hmax, hmin
        real(c_double), intent(inout) :: rwork(lrw), cstate_r(NCS_R)
        integer(c_int), intent(inout) :: iwork(liw), cstate_i(NCS_I)
        real(c_double), intent(out) :: y_out(neq), t_out
        integer(c_int), intent(out) :: nsteps_out, istate_out
        ! IWORK(12) and IWORK(13), LSODA's own cumulative counts of
        ! right-hand-side and jacobian evaluations.  They are reported here
        ! so this history run can serve a caller that also needs the work
        ! counters, instead of a second reporting run beside it.
        integer(c_int), intent(out) :: nfev_out, njev_out
        real(c_double), intent(out) :: t_hist(maxsteps), h_hist(maxsteps)
        real(c_double), intent(out) :: hu_hist(maxsteps)
        integer(c_int), intent(out) :: nq_hist(maxsteps)
        integer(c_int), intent(out) :: nqn_hist(maxsteps)
        real(c_double), intent(out) :: yh_hist(neq, maxord + 1, maxsteps)
        real(c_double), intent(in), target :: args(nargs)

        type(odepack_common_data) :: cdat
        real(dp) :: y(neq), t, dirn
        integer :: istate, k, nqu, j, base
        integer(c_int) :: lrw_need, liw_need

        ! the thread-local slots, saved on entry and restored on exit
        procedure(numba_rhs), pointer :: cb_save
        procedure(numba_jac), pointer :: cb_jac_save
        real(c_double), pointer :: cb_args_save(:)

        ! Before the slots are touched, so the refusal leaves nothing behind.
        call lsoda_dense_worksize(neq, jt, ml, mu, lrw_need, liw_need)
        if (lrw < lrw_need .or. liw < liw_need) then
            nsteps_out = 0
            istate_out = -3
            nfev_out = 0
            njev_out = 0
            y_out = u0
            t_out = t0
            return
        end if

        cb_save => cb
        cb_jac_save => cb_jac_d
        cb_args_save => cb_args
        call c_f_procpointer(cfcn, cb)
        cb_args => args
        if (jt == 1 .or. jt == 4) then
            call c_f_procpointer(cjac, cb_jac_d)
        else
            cb_jac_d => null()
        end if

        ! sign of travel, so the endpoint test works in both directions
        if (tf < t0) then
            dirn = -1.0_dp
        else
            dirn = 1.0_dp
        end if

        istate = istate_in
        if (istate == 1) then
            rwork = 0.0_dp
            iwork = 0
            rwork(5) = h0                 ! first step to attempt
            rwork(6) = hmax               ! largest step allowed
            rwork(7) = hmin               ! smallest step allowed
            iwork(6) = mxstep
            iwork(7) = 10                 ! MXHNIL
            iwork(8) = 12                 ! MXORDN (Adams)
            iwork(9) = 5                  ! MXORDS (BDF)
            ! ML and MU, read by LSODA only when jt is 4 or 5
            ! (odepack.f:553-563).  Left alone otherwise, so the jt = 2 path
            ! is what it was.
            if (jt >= 4) then
                iwork(1) = ml
                iwork(2) = mu
            end if
        else
            ! Continuing: RWORK and IWORK arrive as the previous call left
            ! them and are not written, and the common blocks are restored
            ! from the caller's copy.  Without this INIT is 0 and ODEPACK
            ! refuses the continuation at odepack.f:1167 with istate = -3.
            cdat%DLS001%reals = cstate_r(1:N_DLS001_R)
            cdat%DLSA01%reals = cstate_r(N_DLS001_R + 1:N_DLS001_R + N_DLSA01_R)
            cdat%DLSR01%reals = cstate_r(N_DLS001_R + N_DLSA01_R + 1:NCS_R)
            cdat%DLS001%ints = cstate_i(1:N_DLS001_I)
            cdat%DLSA01%ints = cstate_i(N_DLS001_I + 1:N_DLS001_I + N_DLSA01_I)
            cdat%DLSR01%ints = cstate_i(N_DLS001_I + N_DLSA01_I + 1:NCS_I)
        end if
        rwork(1) = tf                     ! TCRIT, read on every itask 5 call
        cdat%iprint = 0                   ! silence internal messages

        y = u0
        t = t0
        nsteps_out = 0
        nfev_out = 0
        njev_out = 0
        t_hist = 0.0_dp
        h_hist = 0.0_dp
        hu_hist = 0.0_dp
        nq_hist = 0
        nqn_hist = 0
        yh_hist = 0.0_dp

        do k = 1, maxsteps
            ! dlsoda(f, neq, y, t, tout, ITOL, rtol, atol, ITASK, istate,
            !        iopt, rwork, lrw, iwork, liw, jac, jt, common)
            ! itask=5 takes ONE step and does not step past TCRIT, which is
            ! rwork(1) and is tf here.  Approaching it, ODEPACK truncates the
            ! step to H = (TCRIT - TN)*(1 - 4*UROUND) so the run lands on the
            ! endpoint exactly.
            ! `rtol(1)` rather than `rtol`: the interface at
            ! 02_odepack_interface.f90:58 declares RTOL scalar, and ODEPACK
            ! itself declares it RTOL(*), so the element is passed and the
            ! solver reads on from it, which is what odeint_scipy_wrapper
            ! does at wrappers_scipy.f90 too.
            !
            ! itask=5 with rwork(1) = t_bound is what scipy's LSODA does,
            ! read out of the installed scipy 1.18 rather than recalled:
            ! `_ivp/lsoda.py` injects `solver._integrator.rwork[0] =
            ! self.t_bound` at construction and sets `call_args[2] = 5` for
            ! the duration of every `_step_impl`.  Under itask=2 the last
            ! step overshot tf and the endpoint was interpolated from it, so
            ! the final state differed from scipy's by 6.740e-05 at rtol
            ! 1e-3 and 1.978e-11 at rtol 1e-10 (7 cells, decay, backward,
            ! sho and Robertson; every earlier step already agreed at
            ! 0.000e+00).
            !
            ! THIS LINE PREVIOUSLY CARRIED THE OPPOSITE VERDICT, and the
            ! claim is kept rather than dropped: it said itask=5 leaves the
            ! Nordsieck array inconsistent with the step size recorded
            ! beside it, "measured on Robertson, the interpolant was good to
            ! 1e-15 early in the span and 4.3e-05 near the end".  scipy
            ! reads the same two slots the same way (`h = rwork[11]`, YH
            ! from rwork[20:]) and drives with itask=5, so that objection,
            ! if general, would describe scipy too.  Measured on this tree
            ! before the change, over the last two steps against a
            ! rtol 1e-12 reference: Robertson ours 4.329e-06 against scipy's
            ! 4.330e-06, sho ours 4.729e-08 against 4.729e-08.
            !
            ! THE VERDICT IS ANSWERED, on the rebuilt library:
            ! implementation/integrate/fix_wave2/laneX/tcrit_pre.log against
            ! tcrit_post.log, the `sol@end` column.  The interpolant near the
            ! endpoint did not degrade; it moved ONTO scipy's own number on
            ! every cell where the two differed -- decay 8.391e-04 ->
            ! 7.280e-04, sho at rtol 1e-10 6.081e-10 -> 6.003e-10, Robertson
            ! 4.329e-06 -> 4.330e-06, each equal to scipy's -- and reached
            ! 4.3e-05 on no cell, including the Robertson problem the old
            ! verdict names.
            !
            ! WHERE 4.3e-05 DOES COME FROM, measured, and short of proof
            ! because the build the old verdict describes is gone: a history
            ! buffer that runs out.  `_core_lsoda_dense`'s docstring records
            ! that number for the same problem, evaluating at t = 35 past a
            ! history that stopped at t = 25.2.  On this tree a Robertson
            ! history truncated to stop at t = 25.07 reads 5.625e-05 at
            ! t = 35 against the untruncated run, and the figure tracks how
            ! far past the end the read is: 2.057e-05 at 8.1 past, 3.241e-07
            ! at 4.3, 1.902e-10 at 0.4.
            ! implementation/integrate/fix_wave2/laneAD/t16_4e5_probe.py.
            call dlsoda(f_adapter, neq, y, t, tf, itol, rtol(1), atol, 5, &
                        istate, 1, rwork, lrw, iwork, liw, jac_adapter, &
                        jt, cdat)
            if (istate < 0) exit
            nsteps_out = k
            t_hist(k) = t
            h_hist(k) = rwork(12)
            hu_hist(k) = rwork(11)
            nqu = iwork(14)
            nq_hist(k) = nqu
            nqn_hist(k) = iwork(15)
            ! YH is column-major (neq, nqu+1) from rwork(21).
            if (nqu + 1 <= maxord + 1) then
                do j = 1, nqu + 1
                    base = 20 + (j - 1)*neq
                    yh_hist(1:neq, j, k) = rwork(base + 1:base + neq)
                end do
            end if
            if (dirn*(t - tf) >= 0.0_dp) exit
        end do
        istate_out = istate
        nfev_out = iwork(12)
        njev_out = iwork(13)
        ! Where this segment stopped, and the state a continuation needs.
        y_out = y
        t_out = t
        cstate_r(1:N_DLS001_R) = cdat%DLS001%reals
        cstate_r(N_DLS001_R + 1:N_DLS001_R + N_DLSA01_R) = cdat%DLSA01%reals
        cstate_r(N_DLS001_R + N_DLSA01_R + 1:NCS_R) = cdat%DLSR01%reals
        cstate_i(1:N_DLS001_I) = cdat%DLS001%ints
        cstate_i(N_DLS001_I + 1:N_DLS001_I + N_DLSA01_I) = cdat%DLSA01%ints
        cstate_i(N_DLS001_I + N_DLSA01_I + 1:NCS_I) = cdat%DLSR01%ints

        cb => cb_save
        cb_jac_d => cb_jac_save
        cb_args => cb_args_save
    end subroutine lsoda_dense_wrapper

    ! The RWORK and IWORK lengths lsoda_dense_wrapper needs, so its guard on
    ! a short buffer reads them from ODEPACK's own rule rather than from the
    ! caller.  NOT bind(c) and NOT public: a new exported symbol would make
    ! the package fail to IMPORT wherever the shipped library predates this
    ! file, which on Windows it does, and that is worse than the argument
    ! count disagreeing on one call.  The glue holds the same expression and
    ! tests/integrate/test_solve_ivp_scipy.py pins them equal through this guard.
    !
    ! odepack.f:499-517 gives the RWORK requirement as MAX(LRN, LRS) with
    !     LRN = 20 + 16*NEQ
    !     LRS = 22 + 9*NEQ + NEQ**2            if JT = 1 or 2
    !     LRS = 22 + 10*NEQ + (2*ML + MU)*NEQ  if JT = 4 or 5
    ! and the IWORK requirement as 20 + NEQ (odepack.f:542).  LSODA switches
    ! between the nonstiff and stiff methods on its own, so the fixed-length
    ! case is the one to allocate: both regimes at once, not the current one.
    pure subroutine lsoda_dense_worksize(neq, jt, ml, mu, lrw, liw)
        integer(c_int), intent(in) :: neq, jt, ml, mu
        integer(c_int), intent(out) :: lrw, liw

        if (jt >= 4) then
            lrw = 22 + neq*max(16, 10 + 2*ml + mu)
        else
            lrw = 22 + neq*max(16, neq + 9)
        end if
        liw = 20 + neq
    end subroutine lsoda_dense_worksize

    ! LSODAR: LSODA + root finding. Stops at the first t where any of
    ! the ng event functions g(t,y) crosses zero; returns that t and the
    ! state there. 16 args.
    recursive subroutine lsodar_wrapper(cfcn, gfcn, neq, u0, t0, tout, ng, &
                              y_out, t_root, jroot, rtol, atol, mxstep, &
                              istate_out, args, nargs) bind(c)
        type(c_funptr), intent(in), value :: cfcn, gfcn
        integer(c_int), intent(in) :: neq, ng, mxstep, nargs
        real(c_double), intent(in) :: u0(neq), t0, tout
        real(c_double), intent(out) :: y_out(neq), t_root
        integer(c_int), intent(out) :: jroot(ng)
        real(c_double), intent(in) :: rtol, atol
        integer(c_int), intent(out) :: istate_out
        real(c_double), intent(in), target :: args(nargs)

        type(odepack_common_data) :: cdat
        real(dp) :: y(neq), t, atol1(1)
        real(dp), allocatable :: rwork(:)
        integer, allocatable :: iwork(:), jr(:)
        integer :: lrw, liw, istate

        ! the thread-local slots, saved on entry and restored on exit
        procedure(numba_rhs), pointer :: cb_save
        procedure(numba_grt), pointer :: cb_g_save
        real(c_double), pointer :: cb_args_save(:)

        cb_save => cb
        cb_g_save => cb_g
        cb_args_save => cb_args
        call c_f_procpointer(cfcn, cb)
        call c_f_procpointer(gfcn, cb_g)
        cb_args => args

        lrw = 22 + neq*max(16, neq + 9) + 3*ng
        liw = 20 + neq
        allocate (rwork(lrw), iwork(liw), jr(ng))
        rwork = 0.0_dp
        iwork = 0
        rwork(6) = huge(1.0_dp)           ! hmax
        iwork(6) = mxstep
        iwork(7) = 10                     ! MXHNIL
        iwork(8) = 12                     ! MXORDN
        iwork(9) = 5                      ! MXORDS
        cdat%iprint = 0
        atol1(1) = atol
        jr = 0

        y = u0
        t = t0
        istate = 1
        call dlsodar(f_adapter, neq, y, t, tout, 1, rtol, atol1, 1, &
                     istate, 1, rwork, lrw, iwork, liw, jac_dummy, 2, &
                     g_adapter, ng, jr, cdat)
        y_out = y
        t_root = t                        ! istate==3 -> a root was found
        jroot = jr
        istate_out = istate
        deallocate (rwork, iwork, jr)

        cb => cb_save
        cb_g => cb_g_save
        cb_args => cb_args_save
    end subroutine lsodar_wrapper

end module odepack_wrappers
