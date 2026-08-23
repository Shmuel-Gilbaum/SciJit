! Minimal bind(c) LAPACK wrappers for scijit.optimize (private).
!
! Contains ONLY the three routines scijit.optimize needs: dtrtrs_c
! (triangular solve, behind solve_triangular) and dgeqp3_c + dorgqr_c
! (pivoted QR, behind qr_pivot). Vendors NO Fortran of its own: every LAPACK
! routine called below lives in the shared liblapackref (the full
! Reference-LAPACK built once into scijit/_lib); setup.py links this
! wrappers-only .so against it, exactly like src/linalg and src/arpack.
! No callbacks, no state -> prange-safe.
!
! Extracted verbatim from src/linalg/wrappers.f90 (dtrtrs_c, dgeqp3_c,
! dorgqr_c and the char-code helpers). FUTURE WORK: when scijit.linalg ships
! in the public package, optimize re-imports qr_pivot/solve_triangular from
! there and this pack (src/optlapack, scijit/optimize/_lapack.py) is removed.
!
! ABI: all arguments by reference; character arguments arrive as small
! integer codes and are rebuilt locally (uplo 0->'U' 1->'L'; trans 0->'N'
! 1->'T' 2->'C'; diag 0->'N' 1->'U'). LAPACK is column-major; the @njit glue
! passes Fortran-ordered buffers.

module optlapack_wrappers
    use iso_c_binding, only: c_int, c_double
    implicit none
    private
    public :: dtrtrs_c, dgeqp3_c, dorgqr_c

contains

    function uplo_char(code) result(c)
        integer(c_int), intent(in) :: code
        character(len=1) :: c
        if (code == 0) then; c = 'U'; else; c = 'L'; end if
    end function uplo_char

    function trans_char(code) result(c)
        integer(c_int), intent(in) :: code
        character(len=1) :: c
        select case (code)
        case (0); c = 'N'
        case (1); c = 'T'
        case default; c = 'C'
        end select
    end function trans_char

    function diag_char(code) result(c)
        integer(c_int), intent(in) :: code
        character(len=1) :: c
        if (code == 0) then; c = 'N'; else; c = 'U'; end if
    end function diag_char

    ! ---- triangular solve: dtrtrs (behind solve_triangular) ----
    subroutine dtrtrs_c(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info) &
                        bind(c, name="dtrtrs_c")
        integer(c_int), intent(in)    :: uplo, trans, diag, n, nrhs, lda, ldb
        real(c_double), intent(in)    :: a(lda*n)
        real(c_double), intent(inout) :: b(ldb*nrhs)
        integer(c_int), intent(inout) :: info
        call dtrtrs(uplo_char(uplo), trans_char(trans), diag_char(diag), &
                    n, nrhs, a, lda, b, ldb, info)
    end subroutine dtrtrs_c

    ! ---- pivoted QR factorisation: dgeqp3 (behind qr_pivot). Internal
    !      workspace query. ----
    subroutine dgeqp3_c(m, n, a, lda, jpvt, tau, info) &
                        bind(c, name="dgeqp3_c")
        integer(c_int), intent(in)    :: m, n, lda
        real(c_double), intent(inout) :: a(lda*n), tau(*)
        integer(c_int), intent(inout) :: jpvt(n)
        integer(c_int), intent(inout) :: info
        real(c_double) :: wq(1)
        real(c_double), allocatable :: work(:)
        integer :: lwork
        call dgeqp3(m, n, a, lda, jpvt, tau, wq, -1, info)
        lwork = int(wq(1))
        allocate(work(lwork))
        call dgeqp3(m, n, a, lda, jpvt, tau, work, lwork, info)
        deallocate(work)
    end subroutine dgeqp3_c

    ! ---- form Q from the reflectors: dorgqr (behind qr_pivot) ----
    subroutine dorgqr_c(m, n, k, a, lda, tau, info) bind(c, name="dorgqr_c")
        integer(c_int), intent(in)    :: m, n, k, lda
        real(c_double), intent(inout) :: a(lda*n), tau(*)
        integer(c_int), intent(inout) :: info
        real(c_double) :: wq(1)
        real(c_double), allocatable :: work(:)
        integer :: lwork
        call dorgqr(m, n, k, a, lda, tau, wq, -1, info)
        lwork = int(wq(1))
        allocate(work(lwork))
        call dorgqr(m, n, k, a, lda, tau, work, lwork, info)
        deallocate(work)
    end subroutine dorgqr_c

end module optlapack_wrappers
