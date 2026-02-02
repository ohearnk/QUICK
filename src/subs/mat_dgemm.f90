#include "util.fh"
!
!       mat_dgemm.f90
!
!       Architecture agnostic wrapper for dense 
!       matrix-matrix multiplications via LAPACK-style 
!       DGEMM interface.
!
!-----------------------------------------------------------

subroutine MAT_DGEMM(M, N, K, A, LDA, B, LDB, C, LDC)
#if defined(HIP) || defined(HIP_MPIV)
    use quick_rocblas_module, only: rocDGEMM
#endif

#if defined(GPU) || defined(MPIV_GPU)
    call GPU_DGEMM('n', 'n', M, N, K, 1.0d0, A, LDA, B, LDB, 0.0d0, C, LDC)
#else
    ! CPU-based multiplication via LAPACK-style libraries (bundled or external)
    call DGEMM('n', 'n', M, N, K, 1.0d0, A, LDA, B, LDB, 0.0d0, C, LDC)
#endif

end subroutine MAT_DGEMM
