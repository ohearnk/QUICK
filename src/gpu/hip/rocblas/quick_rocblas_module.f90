#include "util.fh"
!---------------------------------------------------------------------!
! Created by Madu Manathunga on 11/19/2021                            !
!                                                                     ! 
! Copyright (C) 2021-2022 Merz lab                                    !
! Copyright (C) 2021-2022 Götz lab                                    !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
!_____________________________________________________________________!

!---------------------------------------------------------------------!
! This module contains drivers required for utilizing rocBLAS in QUICK!
!_____________________________________________________________________!   
 
module quick_rocblas_module

    use iso_c_binding

    implicit none

    private
    public :: rocBlasInit, rocDGEMM, rocDGEMV, rocBlasFinalize
    public :: handle, hinfo, dinfo, dA, dB, dC, hA, hB, hC ! Required for rocSolver
   
    ! global variables
    type(c_ptr), target :: handle ! rocblas handle, responsible for temp work space
    integer(c_int), dimension(:), allocatable, target :: hinfo ! pointer to a rocblas_int type
    type(c_ptr), target :: dinfo    ! device pointer to a rocblas_int type
    type(c_ptr), target :: dA     ! device matrix A
    type(c_ptr), target :: dB     ! device matrix B
    type(c_ptr), target :: dC     ! device matrix C
    real(8), dimension(:), allocatable, target :: hA ! host matrix A
    real(8), dimension(:), allocatable, target :: hB ! host matrix B
    real(8), dimension(:), allocatable, target :: hC ! host matrix C

    ! interfaces
    interface rocDGEMM
        module procedure quick_rocblas_dgemm
    end interface rocDGEMM

    interface rocDGEMV
        module procedure quick_rocblas_dgemv
    end interface rocDGEMV

    interface rocBlasInit
        module procedure rocblas_init
    end interface rocBlasInit

    interface rocBlasFinalize
        module procedure rocblas_finalize
    end interface rocBlasFinalize

    


contains

    ! Initializes rocblas handle, allocates nabsis*nbasis*8 bytes of host & device 
    ! memory for each matrix. Note that we assume the largest matrix would be nbasis x nbasis.
    ! This must be changed if one wants to use rocDGEMM for larger matrices. 
    subroutine rocblas_init(maxlda)
        use iso_c_binding
        use rocblas
        use rocblas_enums
        use rocblas_extra

        implicit none

        integer, intent(in) :: maxlda
        integer :: rb_size
        
        rb_size=maxlda*maxlda

        ! Allocate host-side memory
!        if(.not. allocated(hinfo)) allocate(hinfo(1))
        if(.not. allocated(hA)) allocate(hA(rb_size))
        if(.not. allocated(hB)) allocate(hB(rb_size))
        if(.not. allocated(hC)) allocate(hC(rb_size))

        ! Allocate device-side memory
!        call HIP_CHECK(hipMalloc(c_loc(dinfo), int(1, c_size_t) * 4))
        call HIP_CHECK(hipMalloc(c_loc(dA), int(rb_size, c_size_t) * 8))
        call HIP_CHECK(hipMalloc(c_loc(dB), int(rb_size, c_size_t) * 8))
        call HIP_CHECK(hipMalloc(c_loc(dC), int(rb_size, c_size_t) * 8))

        ! Create rocBLAS handle
 !       call ROCBLAS_CHECK(rocblas_create_handle(c_loc(handle)))
        
        ! Set handle and call rocblas_dgemm
!        call ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, 0))

    end subroutine rocblas_init




    subroutine rocblas_finalize()

        use iso_c_binding
        use rocblas
        use rocblas_enums
        use rocblas_extra

        implicit none

        ! Cleanup
        call HIP_CHECK(hipFree(dA))
        call HIP_CHECK(hipFree(dB))
        call HIP_CHECK(hipFree(dC))
!        call ROCBLAS_CHECK(rocblas_destroy_handle(handle))
!        call HIP_CHECK(hipFree(dinfo))

        if(allocated(hA)) deallocate(hA)
        if(allocated(hB)) deallocate(hB)
        if(allocated(hC)) deallocate(hC)      
!        if(allocated(hinfo)) deallocate(hinfo)

    end subroutine rocblas_finalize


    ! MM: Wrapper function for rocblas_dgemm. 
    subroutine quick_rocblas_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        use iso_c_binding
        use rocblas
        use rocblas_enums
        use rocblas_extra

        implicit none

        character, intent(in) :: transa, transb
        integer, intent(in) :: m, n, k, lda, ldb, ldc
        double precision, intent(in) :: alpha, beta
        double precision, dimension(lda,*), intent(in) :: A
        double precision, dimension(ldb,*), intent(in) :: B
        double precision, dimension(ldc,n), intent(out) :: C

        !internal variables
        integer(c_int) :: rb_n, rb_m, rb_k, rb_lda, rb_ldb, rb_ldc, rb_sizea, rb_sizeb, rb_sizec
        real(c_double), target :: rb_alpha, rb_beta
        
        integer(kind(rocblas_operation_none)) :: rb_transa
        integer(kind(rocblas_operation_transpose))  :: rb_transb 

        integer :: i, j, lidx

        ! Initialize internal variables
        rb_transa = rocblas_operation_none
        rb_transb = rocblas_operation_none

        rb_m = m
        rb_n = n
        rb_k = k
        rb_lda = lda
        rb_ldb = ldb
        rb_ldc = ldc

        if(transa .eq. 'n') then
          rb_lda = rb_m
          rb_sizea = rb_k * rb_lda
        else
          rb_lda = rb_k
          rb_sizea = rb_m * rb_lda
          rb_transa = rocblas_operation_transpose
        endif


        if(transb .eq. 'n') then
          rb_ldb = rb_k
          rb_sizeb = rb_n * rb_ldb
        else
          rb_ldb = rb_n
          rb_sizeb = rb_k * rb_ldb
          rb_transb = rocblas_operation_transpose
        endif

        rb_ldc = rb_m
        rb_sizec = rb_n * rb_ldc

        rb_alpha = alpha
        rb_beta = beta

        call rocBlasInit(rb_lda)

        ! Initialize host memory
        hA = reshape(A(1:lda,1:rb_lda), (/rb_sizea/))
        hB = reshape(B(1:ldb,1:rb_ldb), (/rb_sizeb/))

        ! Copy memory from host to device
        call HIP_CHECK(hipMemcpy(dA, c_loc(hA), int(rb_sizea, c_size_t) * 8, 1))
        call HIP_CHECK(hipMemcpy(dB, c_loc(hB), int(rb_sizeb, c_size_t) * 8, 1))
        call HIP_CHECK(hipMemset(dC, 0, int(rb_sizec, c_size_t) * 8))

        ! Create rocBLAS handle
        call ROCBLAS_CHECK(rocblas_create_handle(c_loc(handle)))

        ! Set handle and call rocblas_dgemm
        call ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, 0))        

        call ROCBLAS_CHECK(rocblas_dgemm(handle, rb_transa, rb_transb, rb_m, rb_n, rb_k, c_loc(rb_alpha), &
                                         dA, rb_lda, dB, rb_ldb, c_loc(rb_beta), dC, rb_ldc))

        call HIP_CHECK(hipDeviceSynchronize())

        ! Copy output from device to host
        call HIP_CHECK(hipMemcpy(c_loc(hC), dC, int(rb_sizec, c_size_t) * 8, 2))

        ! Transfer result
        C = reshape(hC, (/ldc, n/))

        ! Destroy rockblas handle
        call ROCBLAS_CHECK(rocblas_destroy_handle(handle))

        call rocBlasFinalize()

    end subroutine quick_rocblas_dgemm


    ! MM: Wrapper function for rocblas_dgemv. 
    subroutine quick_rocblas_dgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
        use iso_c_binding
        use rocblas
        use rocblas_enums
        use rocblas_extra

        implicit none

        character, intent(in) :: trans
        integer, intent(in) :: m, n, lda, incx, incy
        double precision, intent(in) :: alpha, beta
        double precision, dimension(lda,*), intent(in) :: A
        double precision, dimension(*), intent(in) :: x
        double precision, dimension(*), intent(inout) :: y

        !internal variables
        integer(c_int) :: rb_n, rb_m, rb_lda, rb_incx, rb_incy, rb_sizea, rb_sizex, rb_sizey
        real(c_double), target :: rb_alpha, rb_beta
        
        integer(kind(rocblas_operation_none)) :: rb_trans

        type(c_ptr), target :: dA     ! device matrix A
        type(c_ptr), target :: dx     ! device vector x
        type(c_ptr), target :: dy     ! device vector y
        real(8), dimension(:), allocatable, target :: hA ! host matrix A
        real(8), dimension(:), allocatable, target :: hx ! host vector x
        real(8), dimension(:), allocatable, target :: hy ! host vector y

        ! Initialize internal variables
        rb_m = m
        rb_n = n
        rb_lda = lda
        rb_incx = incx
        rb_incx = incy

        if(trans .eq. 'n') then
          rb_lda = rb_m
          rb_sizea = rb_n * rb_lda
          rb_sizex = rb_n
          rb_trans = rocblas_operation_none
        else
          rb_lda = rb_n
          rb_sizea = rb_m * rb_lda
          rb_sizex = rb_m
          rb_trans = rocblas_operation_transpose
        endif

        rb_sizey = rb_lda

        rb_alpha = alpha
        rb_beta = beta

        ! Allocate host-side memory
        if(.not. allocated(hA)) allocate(hA(rb_sizea))
        if(.not. allocated(hx)) allocate(hx(rb_sizex))
        if(.not. allocated(hy)) allocate(hy(rb_sizey))

        ! Allocate device-side memory
        call HIP_CHECK(hipMalloc(c_loc(dA), int(rb_sizea, c_size_t) * 8))
        call HIP_CHECK(hipMalloc(c_loc(dx), int(rb_sizex, c_size_t) * 8))
        call HIP_CHECK(hipMalloc(c_loc(dy), int(rb_sizey, c_size_t) * 8))

        ! Initialize host memory
        hA = reshape(A(1:lda,1:rb_lda), (/rb_sizea/))
        hx = x(1:rb_sizex)
        hy = y(1:rb_sizey)

        ! Copy memory from host to device
        call HIP_CHECK(hipMemcpy(dA, c_loc(hA), int(rb_sizea, c_size_t) * 8, 1))
        call HIP_CHECK(hipMemcpy(dx, c_loc(hx), int(rb_sizex, c_size_t) * 8, 1))
        call HIP_CHECK(hipMemcpy(dy, c_loc(hy), int(rb_sizey, c_size_t) * 8, 1))

        ! Create rocBLAS handle
        call ROCBLAS_CHECK(rocblas_create_handle(c_loc(handle)))

        ! Set handle and call rocblas_dgemm
        call ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, 0))        

        call ROCBLAS_CHECK(rocblas_dgemv(handle, rb_trans, rb_m, rb_n, c_loc(rb_alpha), &
                                         dA, rb_lda, dx, rb_incx, c_loc(rb_beta), dy, rb_incy))

        call HIP_CHECK(hipDeviceSynchronize())

        ! Copy output from device to host
        call HIP_CHECK(hipMemcpy(c_loc(hy), dy, int(rb_sizey, c_size_t) * 8, 2))

        ! Transfer result
        y = hy(1:rb_sizey)

        ! Destroy rockblas handle
        call ROCBLAS_CHECK(rocblas_destroy_handle(handle))

        ! Cleanup
        call HIP_CHECK(hipFree(dA))
        call HIP_CHECK(hipFree(dx))
        call HIP_CHECK(hipFree(dy))

        if(allocated(hA)) deallocate(hA)
        if(allocated(hx)) deallocate(hx)
        if(allocated(hy)) deallocate(hy)      

    end subroutine quick_rocblas_dgemv

end module quick_rocblas_module

