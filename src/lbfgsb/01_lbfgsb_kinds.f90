

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


    module lbfgsb_kinds_module

    use, intrinsic :: iso_fortran_env

    implicit none

    private








    integer,parameter,public :: lbfgsb_wp = real64   !! real kind used by this module [8 bytes]


    end module lbfgsb_kinds_module
!*****************************************************************************************
