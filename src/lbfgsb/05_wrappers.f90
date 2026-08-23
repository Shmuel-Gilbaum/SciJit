! bind(c) wrapper exposing L-BFGS-B 3.0 (jacobwilliams/lbfgsb modern
! refactor) to ctypes/numba, following the scijit architecture:
! every argument arrives by reference through a c_void_p slot.
!
! setulb is REVERSE COMMUNICATION: the caller drives the loop and
! supplies f and g when asked, no callbacks, no cfunc, and (unlike the
! minpack pack) NO module state: every piece of solver state lives in
! caller-owned arrays, so concurrent solves are safe.
!
! The one impedance mismatch is Fortran's character(len=60) task/csave
! strings, which carry the RC protocol ('START' -> 'FG...'/'NEW_X' ->
! 'CONV...'/'STOP...'/'ABNO'/'ERROR'). They cross the C boundary as
! 60-byte integer(int8) arrays, converted here on entry/exit, so the
! full string state round-trips losslessly between calls. The numba
! side initialises the buffer to 'START' padded with BLANKS (32), and
! branches on the byte prefix. Same treatment for the logical lsave(4)
! (as int32 0/1).
module lbfgsb_wrappers
    use iso_c_binding, only: c_double, c_int, c_int8_t
    use lbfgsb_module, only: setulb
    implicit none
    private
    public :: setulb_wrapper

contains

    ! 19 args.
    subroutine setulb_wrapper(n, m, x, l, u, nbd, f, g, factr, pgtol, &
                              wa, iwa, task_b, iprint, csave_b, lsave, &
                              isave, dsave, maxls) bind(c)
        integer(c_int), intent(in) :: n, m, iprint, maxls
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(in) :: l(n), u(n)
        integer(c_int), intent(in) :: nbd(n)
        real(c_double), intent(inout) :: f
        real(c_double), intent(inout) :: g(n)
        real(c_double), intent(in) :: factr, pgtol
        real(c_double), intent(inout) :: wa(2*m*n + 5*n + 11*m*m + 8*m)
        integer(c_int), intent(inout) :: iwa(3*n)
        integer(c_int8_t), intent(inout) :: task_b(60), csave_b(60)
        integer(c_int), intent(inout) :: lsave(4)
        integer(c_int), intent(inout) :: isave(44)
        real(c_double), intent(inout) :: dsave(29)

        character(len=60) :: task, csave
        logical :: lsave_l(4)
        integer :: i

        do i = 1, 60
            task(i:i) = achar(int(task_b(i)))
            csave(i:i) = achar(int(csave_b(i)))
        end do
        lsave_l = (lsave /= 0)

        call setulb(n, m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, &
                    task, iprint, csave, lsave_l, isave, dsave, &
                    Maxls=int(maxls))

        do i = 1, 60
            task_b(i) = int(iachar(task(i:i)), c_int8_t)
            csave_b(i) = int(iachar(csave(i:i)), c_int8_t)
        end do
        lsave = merge(1_c_int, 0_c_int, lsave_l)
    end subroutine setulb_wrapper

end module lbfgsb_wrappers
