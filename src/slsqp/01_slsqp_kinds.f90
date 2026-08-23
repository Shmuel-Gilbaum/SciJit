

!*****************************************************************************************
!> author: Jacob Williams
!  date: 12/22/2015
!  license: BSD
!
!  Numeric kind definitions.
!
!@note The default real kind (`wp`) can be
!      changed using optional preprocessor flags.
!      This library was built with real kind:







!      `real(kind=real64)` [8 bytes]


    module slsqp_kinds

    use, intrinsic :: iso_fortran_env

    implicit none

    private








    integer, parameter, public :: slsqp_rk = real64   !! real kind used by this module [8 bytes]


    integer,parameter,public :: wp = slsqp_rk  !! copy of `slsqp_rk` with a shorter name

    end module slsqp_kinds
!*****************************************************************************************
