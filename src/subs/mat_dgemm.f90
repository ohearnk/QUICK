#include "util.fh"
!
!       mat_dgemm.f90
!
!       Architecture agnostic wrapper for dense 
!       matrix-matrix multiplications via LAPACK-style 
!       DGEMM interface.
!
!-----------------------------------------------------------

subroutine MAT_DGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, &
                C, LDC)
    implicit none

    double precision ALPHA, BETA
    integer K, LDA, LDB, LDC, M, N
    character TRANSA, TRANSB
    double precision A(LDA,*), B(LDB,*), C(LDC,*)

#if defined(HIP) || defined(HIP_MPIV)
    use quick_rocblas_module, only: rocDGEMM
#endif

#if defined(GPU) || defined(MPIV_GPU)
    call GPU_DGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
    ! CPU-based multiplication via LAPACK-style libraries (bundled or external)
    call DGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif

end subroutine MAT_DGEMM
