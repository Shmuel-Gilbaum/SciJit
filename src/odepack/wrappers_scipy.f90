! bind(c) wrapper exposing LSODA with the argument choices
! scipy.integrate.odeint makes, and with the full optional-output set its
! `full_output=True` infodict reports.
!
! Why this file exists alongside wrappers.f90:
!   lsoda_wrapper there hardcodes itol = 1 (scalar tolerances), itask = 1
!   (no critical point), rwork(6) = huge (hmax), rwork(5) = rwork(7) = 0,
!   iwork(5) = 0, iwork(7,8,9) = 10, 12, 5, and copies out nothing but
!   istate.  scipy passes itol 1/2/3/4 from the shapes of rtol and atol,
!   itask 4 when tcrit is given, h0/hmax/hmin/ixpr/mxstep/mxhnil/mxordn/
!   mxords straight through, and reports 13 optional outputs.  All 13
!   already sit in the work arrays; this wrapper copies them out per
!   integration leg.
!
!   It also carries the SECOND callback ABI.  scipy's default is
!   func(y, t, *args); scijit's published lsoda_sig is func(t, y, ...),
!   i.e. scipy's tfirst=True.  Both are reachable here: `tfirst` selects
!   which of the two abstract interfaces the adapter calls.
!
!   It also carries LSODA's four Jacobian treatments.  `jt` is an argument
!   rather than the hardcoded 2 it was, and `ml`/`mu` reach IWORK(1)/IWORK(2):
!
!       jt = 1   user-supplied full jacobian
!       jt = 2   internally generated full        (the only route before)
!       jt = 4   user-supplied banded
!       jt = 5   internally generated banded
!
!   scipy's routing, _ode.py:1300-1319: a jacobian selects 1 or 4, and
!   lband/uband select 4 or 5, with the unset half becoming 0.  The RWORK
!   length changes with it; see the LRW comment in the body.
!
! Symbol: odeint_scipy_wrapper (40 args).  wrappers.f90's lsoda_wrapper and
! lsodar_wrapper are untouched.
!
! The cfunc addresses, the args pointer and the tfirst flag live in module
! variables, ONE SLOT PER THREAD (see the `!$omp threadprivate` below).  The
! wrapper saves all five on entry and restores them on exit, so an
! integration started from inside another integration's callback leaves the
! outer one's slots as it found them.
module odepack_scipy_wrappers
    use iso_c_binding, only: c_double, c_int, c_funptr, c_f_procpointer
    use iso_fortran_env, only: dp => real64
    use odepack_common, only: odepack_common_data
    use odepack_interface, only: dlsoda
    implicit none
    private
    public :: odeint_scipy_wrapper

    abstract interface
        ! scipy tfirst=True  == scijit lsoda_sig == numbalsoda lsoda_sig
        subroutine numba_rhs_t(t, y, ydot, args) bind(c)
            import :: c_double
            implicit none
            real(c_double), intent(in), value :: t
            real(c_double), intent(in) :: y(*), args(*)
            real(c_double), intent(out) :: ydot(*)
        end subroutine numba_rhs_t

        ! scipy tfirst=False (its default) == scijit lsoda_yfirst_sig
        subroutine numba_rhs_y(y, t, ydot, args) bind(c)
            import :: c_double
            implicit none
            real(c_double), intent(in) :: y(*), args(*)
            real(c_double), intent(in), value :: t
            real(c_double), intent(out) :: ydot(*)
        end subroutine numba_rhs_y

        ! The jacobian callback, jt = 1 and jt = 4.  ONE argument order,
        ! unlike the right-hand side: a y-first user jacobian is handled
        ! inside the @cfunc adapter the glue builds, not here.
        !
        ! neq, ml, mu and nrowpd are LSODA's, forwarded by value because the
        ! adapter is built before any of them is known.  Storage: the full
        ! case (jt = 1) has nrowpd = neq and wants d f(i)/d y(j) at
        ! pd(i + j*nrowpd), zero-based; the banded case (jt = 4) has
        ! nrowpd = 2*ml + mu + 1 and wants it at pd(i - j + mu + j*nrowpd).
        subroutine numba_jac(t, y, pd, neq, ml, mu, nrowpd, args) bind(c)
            import :: c_double, c_int
            implicit none
            real(c_double), intent(in), value :: t
            real(c_double), intent(in) :: y(*), args(*)
            real(c_double), intent(inout) :: pd(*)
            integer(c_int), intent(in), value :: neq, ml, mu, nrowpd
        end subroutine numba_jac
    end interface

    procedure(numba_rhs_t), pointer :: cb_t => null()
    procedure(numba_rhs_y), pointer :: cb_y => null()
    procedure(numba_jac), pointer :: cb_jac => null()
    real(c_double), pointer :: cb_args(:) => null()
    logical :: t_first = .false.

    ! One callback slot per THREAD.  See the same directive in wrappers.f90
    ! for what it costs and what happens without -fopenmp.  `t_first`
    ! belongs here too: it selects which of the two callback ABIs the
    ! adapter calls, so a shared copy makes one thread invoke another
    ! thread's callback through the wrong argument order.  `cb_jac` is the
    ! same requirement one level down: two concurrent runs would otherwise
    ! evaluate each other's jacobian, which is the silent-wrong bucket
    ! because a jacobian only steers the Newton iteration and a wrong one
    ! still converges to a plausible answer.
    !$omp threadprivate(cb_t, cb_y, cb_jac, cb_args, t_first)

contains

    ! adapter matching ODEPACK's odepack_f abstract interface.
    ! RECURSIVE because the user callback may itself start an integration,
    ! which re-enters this adapter one frame down.
    recursive subroutine f_adapter(neq, t, y, ydot, ierr)
        integer, intent(in) :: neq
        real(dp), intent(in) :: t, y(neq)
        real(dp), intent(out) :: ydot(neq)
        integer, intent(out) :: ierr
        if (t_first) then
            call cb_t(t, y, ydot, cb_args)
        else
            call cb_y(y, t, ydot, cb_args)
        end if
        ierr = 0
    end subroutine f_adapter

    ! adapter matching ODEPACK's odepack_jac abstract interface.  Reached only
    ! when jt is 1 or 4; LSODA never calls it for 2 or 5.  The `associated`
    ! test makes a stale thread-local slot a no-op rather than a jump through
    ! a null pointer, and LSODA presets pd to zero either way (odepack.f:609).
    recursive subroutine jac_adapter(neq, t, y, ml, mu, pd, nrowpd, ierr)
        integer, intent(in) :: neq, ml, mu, nrowpd
        real(dp), intent(in) :: t
        real(dp), intent(inout) :: y(neq)
        real(dp), intent(out) :: pd(nrowpd, neq)
        integer, intent(out) :: ierr
        if (associated(cb_jac)) then
            call cb_jac(t, y, pd, neq, ml, mu, nrowpd, cb_args)
        end if
        ierr = 0
    end subroutine jac_adapter

    ! LSODA over t(1) .. t(ntime), one dlsoda call per leg, filling
    ! yout(neq, ntime) and the nine per-leg optional-output vectors
    ! (length ntime-1) plus the three scalars.
    !
    ! tfirst   1 -> the callback is (t, y, ydot, args); 0 -> (y, t, ydot, args)
    ! itol     1..4, ODEPACK's rtol/atol scalar-or-array selector
    ! usetcrit 1 -> itask 4 from the first leg with rwork(1) = tcrit(1).  On
    !               each leg whose tout passes the current critical time the
    !               index advances by one and rwork(1) follows it; itask drops
    !               to 1 once the array is exhausted and stays there.  scipy's
    !               own walk, _odepackmodule.c:757-781.  The test is strictly
    !               tout > tcrit(ic) and is not direction-aware, as scipy's is
    !               not.
    ! ntcrit   how many critical times tcrit holds; read only when usetcrit
    !               is non-zero.  scipy reads tcrit(ic) once more after the
    !               final advance, one past the end, and then sets itask = 1
    !               so LSODA never looks at it; the bound below is the same
    !               behaviour without the out-of-range read.
    ! jt       1, 2, 4 or 5, LSODA's jacobian-type indicator
    ! ml, mu   half-bandwidths, read only when jt is 4 or 5
    ! cjac     the jacobian callback, read only when jt is 1 or 4; ignored
    !          otherwise, so jt = 2 and jt = 5 may pass a null funptr
    recursive subroutine odeint_scipy_wrapper(cfcn, tfirst, neq, y0, t, &
                                              ntime, yout, &
                                              itol, rtol, atol, usetcrit, &
                                              tcrit, ntcrit, &
                                              h0, hmax, hmin, ixpr, mxstep, &
                                              mxhnil, mxordn, mxords, &
                                              istate_out, &
                                              hu, tcur, tolsf, tsw, nst, &
                                              nfe, nje, nqu, mused, imxer, &
                                              lenrw, leniw, &
                                              args, nargs, jt, ml, mu, &
                                              cjac) bind(c)
        type(c_funptr), intent(in), value :: cfcn, cjac
        integer(c_int), intent(in) :: tfirst, neq, ntime, itol, usetcrit
        integer(c_int), intent(in) :: ntcrit
        integer(c_int), intent(in) :: ixpr, mxstep, mxhnil, mxordn, mxords
        integer(c_int), intent(in) :: nargs, jt, ml, mu
        real(c_double), intent(in) :: y0(neq), t(ntime)
        real(c_double), intent(out) :: yout(neq, ntime)
        real(c_double), intent(in) :: rtol(*), atol(*)
        real(c_double), intent(in) :: tcrit(*), h0, hmax, hmin
        integer(c_int), intent(out) :: istate_out
        real(c_double), intent(out) :: hu(*), tcur(*), tolsf(*), tsw(*)
        integer(c_int), intent(out) :: nst(*), nfe(*), nje(*), nqu(*)
        integer(c_int), intent(out) :: mused(*)
        integer(c_int), intent(out) :: imxer, lenrw, leniw
        real(c_double), intent(in), target :: args(nargs)

        type(odepack_common_data) :: cdat
        real(dp) :: y(neq), tt
        real(dp), allocatable :: rwork(:)
        integer, allocatable :: iwork(:)
        integer :: lrw, liw, istate, itask, i, k, ic

        ! The five thread-local slots, saved on entry and restored on exit.
        ! Without this an integration started from inside another
        ! integration's callback leaves the outer LSODA running against the
        ! INNER callback and the inner `args` buffer, which the inner call no
        ! longer owns: the outer run then returns a plausible wrong answer
        ! rather than failing.  `src/quadpack/wrappers.f90` has done this
        ! since it was written; this file did not.
        procedure(numba_rhs_t), pointer :: cb_t_save
        procedure(numba_rhs_y), pointer :: cb_y_save
        procedure(numba_jac), pointer :: cb_jac_save
        real(c_double), pointer :: cb_args_save(:)
        logical :: t_first_save

        cb_t_save => cb_t
        cb_y_save => cb_y
        cb_jac_save => cb_jac
        cb_args_save => cb_args
        t_first_save = t_first

        t_first = (tfirst /= 0)
        if (t_first) then
            call c_f_procpointer(cfcn, cb_t)
        else
            call c_f_procpointer(cfcn, cb_y)
        end if
        cb_args => args
        ! jt 2 and 5 generate the jacobian internally, so the slot is cleared
        ! rather than left holding whatever a previous call on this thread put
        ! there.
        if (jt == 1 .or. jt == 4) then
            call c_f_procpointer(cjac, cb_jac)
        else
            cb_jac => null()
        end if

        ! LSODA needs the max over the Adams and BDF methods, and the BDF
        ! half depends on jt.  odepack.f:499-517:
        !     LRN = 20 + 16*NEQ
        !     LRS = 22 + 9*NEQ + NEQ**2           if JT = 1 or 2
        !     LRS = 22 + 10*NEQ + (2*ML+MU)*NEQ   if JT = 4 or 5
        ! and the fixed-length case wants MAX(LRN, LRS).  Writing each as
        ! 22 + neq*max(16, ...) covers LRN too, since 22 > 20.
        !
        ! The full form 22+9*neq+neq**2 on its own undersizes small systems
        ! (neq < 7) -> istate = -3, "illegal input", with no other symptom.
        ! odepack.f:1216-1224 clamps MXORDN to 12 and MXORDS to 5, so the
        ! default-order formula is an upper bound for every legal input.
        if (jt >= 4) then
            lrw = 22 + neq*max(16, 10 + 2*ml + mu)
        else
            lrw = 22 + neq*max(16, neq + 9)
        end if
        ! odepack.f:542 -- LIS = 20 + NEQ and LIN = 20, and IWORK(1)/IWORK(2)
        ! sit inside the first 20 words, so jt does not change this.
        liw = 20 + neq
        allocate (rwork(lrw), iwork(liw))
        rwork = 0.0_dp
        iwork = 0

        ! IWORK(1)/IWORK(2) are ML and MU, "required if JT is 4 or 5, and
        ! ignored otherwise" (odepack.f:553-563).  Written only on the banded
        ! routes, so the jt = 2 path allocates and fills exactly what it did
        ! before this argument existed.
        if (jt >= 4) then
            iwork(1) = ml
            iwork(2) = mu
        end if

        ! iopt = 1: every optional input below is set explicitly, so the
        ! defaults must be spelled out.  scipy's own defaults are
        ! h0 = hmax = hmin = 0, ixpr = 0, mxstep = 0, mxhnil = 0,
        ! mxordn = 12, mxords = 5, and LSODA reads 0 as "use my default".
        rwork(5) = h0
        rwork(6) = hmax
        rwork(7) = hmin
        iwork(5) = ixpr
        iwork(6) = mxstep
        iwork(7) = mxhnil
        iwork(8) = mxordn
        iwork(9) = mxords
        ! ODEPACK's message channel writes to Fortran unit 6, which a caller
        ! can neither catch nor redirect, and under a prange loop the threads
        ! interleave on it.  So it stays off unless ixpr asks for the
        ! method-switch messages it exists to produce.
        !
        ! scipy 1.18 cannot produce them at all: its C translation of ODEPACK
        ! dropped the message routines, so `strings` finds SIX "LSODA-" texts
        ! in this library and ZERO in scipy's _odepack extension, while its
        ! _odepackmodule.c:728 still writes iwork[4] = ixpr.  Its own
        ! docstring documents ixpr as "extra printing at method switches",
        ! so the argument is documented and inert there.  Deliberate
        ! deviation: an accepted-and-ignored argument is the one shape a
        ! caller cannot detect.
        if (ixpr /= 0) then
            cdat%iprint = 1
        else
            cdat%iprint = 0
        end if

        imxer = 0
        lenrw = 0
        leniw = 0
        ! IWORK(16) is written by LSODA only on an error- or convergence-test
        ! failure (odepack.f label 560).  scipy reports -1 whenever LSODA ran
        ! and left it alone, and 0 when LSODA was never called (ntime == 1),
        ! so the slot is seeded to -1 exactly when a call is about to happen.
        if (ntime >= 2) iwork(16) = -1

        y = y0
        tt = t(1)
        yout(:, 1) = y0
        istate = 1
        ic = 1
        if (usetcrit /= 0 .and. ntcrit >= 1) then
            itask = 4
            rwork(1) = tcrit(1)
        else
            itask = 1
        end if
        do i = 2, ntime
            if (itask == 4) then
                if (t(i) > tcrit(ic)) then
                    ic = ic + 1
                    if (ic <= ntcrit) rwork(1) = tcrit(ic)
                end if
                if (ic > ntcrit) itask = 1
            end if
            call dlsoda(f_adapter, neq, y, tt, t(i), itol, rtol(1), atol, &
                        itask, istate, 1, rwork, lrw, iwork, liw, &
                        jac_adapter, jt, cdat)
            k = i - 1
            hu(k) = rwork(11)
            tcur(k) = rwork(13)
            tolsf(k) = rwork(14)
            tsw(k) = rwork(15)
            nst(k) = iwork(11)
            nfe(k) = iwork(12)
            nje(k) = iwork(13)
            nqu(k) = iwork(14)
            mused(k) = iwork(19)
            ! scipy stores the state LSODA reached even on the failing leg,
            ! then stops; the rows after it are never written.
            yout(:, i) = y
            if (istate < 0) exit
        end do
        if (ntime >= 2) then
            imxer = iwork(16)
            lenrw = iwork(17)
            leniw = iwork(18)
        end if
        istate_out = istate
        deallocate (rwork, iwork)

        cb_t => cb_t_save
        cb_y => cb_y_save
        cb_jac => cb_jac_save
        cb_args => cb_args_save
        t_first = t_first_save
    end subroutine odeint_scipy_wrapper

end module odepack_scipy_wrappers
