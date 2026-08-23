! bind(c) wrapper exposing Kraft's SLSQP (jacobwilliams/slsqp modern
! refactor, slsqp_core) to ctypes/numba.
!
! slsqp is REVERSE COMMUNICATION via mode: on return mode==1 asks the
! caller for f and c (constraints), mode==-1 for g and a (gradients);
! |mode| /= 1 means finished. No callbacks, no cfunc.
!
! The modern refactor moved the F77 SAVE state into two derived types
! (slsqpb_data: 10 reals + 8 ints; linmin_data: 18 reals). They cross
! the C boundary as flat arrays owned by the caller and are packed/
! unpacked here each call, NO module state, concurrent solves safe.
!
! Extra tuning arguments of the refactor are passed through; the numba
! glue supplies the defaults that reproduce the classic scipy-wrapped
! behavior (alphamin=0.1, alphamax=1.0, tolf/toldf/toldx=-1 disabled,
! max_iter_ls=0 -> internal default, nnls_mode=1 original algorithm,
! infinite_bound=huge).
module slsqp_wrappers
    use iso_c_binding, only: c_double, c_int
    use slsqp_core, only: slsqp, slsqpb_data, linmin_data
    implicit none
    private
    public :: slsqp_wrapper

contains

    ! 26 args.
    subroutine slsqp_wrapper(m, meq, la, n, x, xl, xu, f, c, g, a, acc, &
                             iter, mode, w, l_w, sdat_r, sdat_i, ldat_r, &
                             alphamin, alphamax, tolf, toldf, toldx, &
                             max_iter_ls, nnls_mode) bind(c)
        integer(c_int), intent(in) :: m, meq, la, n, l_w
        integer(c_int), intent(in) :: max_iter_ls, nnls_mode
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(in) :: xl(n), xu(n)
        real(c_double), intent(in) :: f
        real(c_double), intent(in) :: c(la)
        real(c_double), intent(in) :: g(n + 1)
        real(c_double), intent(in) :: a(la, n + 1)
        real(c_double), intent(inout) :: acc
        integer(c_int), intent(inout) :: iter, mode
        real(c_double), intent(inout) :: w(l_w)
        real(c_double), intent(inout) :: sdat_r(10)
        integer(c_int), intent(inout) :: sdat_i(8)
        real(c_double), intent(inout) :: ldat_r(18)
        real(c_double), intent(in) :: alphamin, alphamax, tolf, toldf, toldx

        type(slsqpb_data) :: sdat
        type(linmin_data) :: ldat

        ! unpack caller-held state
        sdat%t = sdat_r(1);  sdat%f0 = sdat_r(2);  sdat%h1 = sdat_r(3)
        sdat%h2 = sdat_r(4); sdat%h3 = sdat_r(5);  sdat%h4 = sdat_r(6)
        sdat%t0 = sdat_r(7); sdat%gs = sdat_r(8);  sdat%tol = sdat_r(9)
        sdat%alpha = sdat_r(10)
        sdat%line = sdat_i(1);   sdat%iexact = sdat_i(2)
        sdat%incons = sdat_i(3); sdat%ireset = sdat_i(4)
        sdat%itermx = sdat_i(5); sdat%n1 = sdat_i(6)
        sdat%n2 = sdat_i(7);     sdat%n3 = sdat_i(8)

        ldat%a = ldat_r(1);   ldat%b = ldat_r(2);   ldat%d = ldat_r(3)
        ldat%e = ldat_r(4);   ldat%p = ldat_r(5);   ldat%q = ldat_r(6)
        ldat%r = ldat_r(7);   ldat%u = ldat_r(8);   ldat%v = ldat_r(9)
        ldat%w = ldat_r(10);  ldat%x = ldat_r(11);  ldat%m = ldat_r(12)
        ldat%fu = ldat_r(13); ldat%fv = ldat_r(14); ldat%fw = ldat_r(15)
        ldat%fx = ldat_r(16); ldat%tol1 = ldat_r(17)
        ldat%tol2 = ldat_r(18)

        call slsqp(m, meq, la, n, x, xl, xu, f, c, g, a, acc, iter, mode, &
                   w, l_w, sdat, ldat, alphamin, alphamax, tolf, toldf, &
                   toldx, max_iter_ls, nnls_mode, huge(1.0_c_double))

        ! pack state back for the next reverse-communication call
        sdat_r(1) = sdat%t;  sdat_r(2) = sdat%f0;  sdat_r(3) = sdat%h1
        sdat_r(4) = sdat%h2; sdat_r(5) = sdat%h3;  sdat_r(6) = sdat%h4
        sdat_r(7) = sdat%t0; sdat_r(8) = sdat%gs;  sdat_r(9) = sdat%tol
        sdat_r(10) = sdat%alpha
        sdat_i(1) = sdat%line;   sdat_i(2) = sdat%iexact
        sdat_i(3) = sdat%incons; sdat_i(4) = sdat%ireset
        sdat_i(5) = sdat%itermx; sdat_i(6) = sdat%n1
        sdat_i(7) = sdat%n2;     sdat_i(8) = sdat%n3

        ldat_r(1) = ldat%a;   ldat_r(2) = ldat%b;   ldat_r(3) = ldat%d
        ldat_r(4) = ldat%e;   ldat_r(5) = ldat%p;   ldat_r(6) = ldat%q
        ldat_r(7) = ldat%r;   ldat_r(8) = ldat%u;   ldat_r(9) = ldat%v
        ldat_r(10) = ldat%w;  ldat_r(11) = ldat%x;  ldat_r(12) = ldat%m
        ldat_r(13) = ldat%fu; ldat_r(14) = ldat%fv; ldat_r(15) = ldat%fw
        ldat_r(16) = ldat%fx; ldat_r(17) = ldat%tol1
        ldat_r(18) = ldat%tol2
    end subroutine slsqp_wrapper

end module slsqp_wrappers
