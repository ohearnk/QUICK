/*
 *  gpu_get2e.cpp
 *  new_quick
 *
 *  Created by Yipu Miao on 6/17/11.
 *  Copyright 2011 University of Florida.All rights reserved.
 *  
 *  Yipu Miao 9/15/11:  the first draft is released. And the GPUGP QM compuation can 
 *                      achieve as much as 15x faster at double precision level compared with CPU.
 */

#include "gpu.h"
#include <cuda.h>

//#define USE_TEXTURE

#if defined(USE_TEXTURE)
  #define USE_TEXTURE_CUTMATRIX
  #define USE_TEXTURE_YCUTOFF
  #define USE_TEXTURE_XCOEFF
#endif

#ifdef USE_TEXTURE_CUTMATRIX
texture <int2, cudaTextureType1D, cudaReadModeElementType> tex_cutMatrix;
#endif
#ifdef USE_TEXTURE_YCUTOFF
texture <int2, cudaTextureType1D, cudaReadModeElementType> tex_YCutoff;
#endif
#ifdef USE_TEXTURE_XCOEFF
texture <int2, cudaTextureType1D, cudaReadModeElementType> tex_Xcoeff;
#endif

//#define USE_ERI_GRAD_STOREADD
//#ifdef USE_ERI_GRAD_STOREADD
//  #define STORE_OPERATOR +=
//#else
//  #define STORE_OPERATOR =  
//#endif

#include "../gpu_get2e_subs_hrr.h"
#if defined(COMPILE_GPU_AOINT)
  #include "../gpu_eri_vertical_int.h"
#endif

#define int_sp
#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_assembler_sp.h"
#include "../gpu_get2e_subs.h"
#include "../gpu_eri_grad_assembler_sp.h"
#include "../gpu_get2e_subs_grad.h"

#undef int_sp
#define int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_assembler_spd.h"
#include "../gpu_get2e_subs.h"
#include "../gpu_eri_grad_assembler_spd.h"
#include "../gpu_get2e_subs_grad.h"


//===================================

#undef int_spd
#define int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_grad_vrr_dddd_1.h"
#include "../gpu_get2e_subs_grad.h"

#undef int_spd
#undef int_spdf
#define int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_grad_vrr_dddd_2.h"
#include "../gpu_get2e_subs_grad.h"


/*
#undef int_spd
#undef int_spdf
#undef int_spdf2
#define int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs_grad.h"
*/

#ifdef GPU_SPDF
//===================================

#undef int_spd
#define int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_assembler_spdf_1.h"
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#define int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_assembler_spdf_2.h"
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#define int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_assembler_spdf_3.h"
#include "../gpu_get2e_subs.h"

#include "../gpu_eri_assembler_spdf_1_2.h"
#include "../gpu_eri_assembler_spdf_2_2.h"
#include "../gpu_eri_assembler_spdf_3_2.h"
#include "../gpu_eri_assembler_spdf_4_2.h"
#include "../gpu_eri_assembler_spdf_5_2.h"
#include "../gpu_eri_assembler_spdf_6_2.h"
#include "../gpu_eri_assembler_spdf_7_2.h"
#include "../gpu_eri_assembler_spdf_8_2.h"

#include "../gpu_eri_grad_assembler_spd_2.h"
#include "../gpu_eri_grad_assembler_spdf_1.h" 
#include "../gpu_eri_grad_assembler_spdf_2.h"
#include "../gpu_eri_grad_assembler_spdf_3.h"
#include "../gpu_eri_grad_assembler_spdf_4.h"
#include "../gpu_eri_grad_assembler_spdf_5.h"
#include "../gpu_eri_grad_assembler_spdf_6.h"
//#include "../gpu_eri_grad_assembler_spdf_7_1.h"
//#include "../gpu_eri_grad_assembler_spdf_7_2.h"
//#include "../gpu_eri_grad_assembler_spdf_7_3.h"
#include "../gpu_get2e_subs_grad.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#define int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_assembler_spdf_4.h"
//#include "../gpu_eri_grad_vrr_ffff.h"
#include "../gpu_get2e_subs.h"
//#include "../gpu_get2e_subs_grad.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#define int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_assembler_spdf_5.h"
#include "../gpu_get2e_subs.h"


#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#define int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_assembler_spdf_6.h"
#include "../gpu_get2e_subs.h"


#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#define int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_assembler_spdf_7.h"
#include "../gpu_get2e_subs.h"


#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#define int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_eri_assembler_spdf_8.h"
#include "../gpu_get2e_subs.h"


#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#define int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#define int_spdf10
#include "../gpu_get2e_subs.h"

#endif

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10

//Include the kernels for open shell eri calculations
#define OSHELL

#define int_sp
#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#undef new_quick_2_gpu_get2e_subs_h
#include "../gpu_get2e_subs.h"
#include "../gpu_get2e_subs_grad.h"

#undef int_sp
#define int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"
#include "../gpu_get2e_subs_grad.h"

//===================================

#undef int_spd
#define int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs_grad.h"

#undef int_spd
#undef int_spdf
#define int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs_grad.h"

/*
#undef int_spd
#undef int_spdf
#undef int_spdf2
#define int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs_grad.h"
*/

#ifdef GPU_SPDF
//===================================

#undef int_spd
#define int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#define int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#define int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#define int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#define int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#define int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#define int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#define int_spdf8
#undef int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"


#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#define int_spdf9
#undef int_spdf10
#include "../gpu_get2e_subs.h"

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#define int_spdf10
#include "../gpu_get2e_subs.h"

#endif

#undef int_spd
#undef int_spdf
#undef int_spdf2
#undef int_spdf3
#undef int_spdf4
#undef int_spdf5
#undef int_spdf6
#undef int_spdf7
#undef int_spdf8
#undef int_spdf9
#undef int_spdf10

#undef OSHELL


#if defined(USE_TEXTURE)
static void bind_eri_texture(_gpu_type gpu)
{
  #if defined(USE_TEXTURE_CUTMATRIX)
    cudaBindTexture(NULL, tex_cutMatrix, gpu->gpu_sim.cutMatrix, sizeof(QUICKDouble) * gpu->nshell * gpu->nshell);
  #endif
  #if defined(USE_TEXTURE_YCUTOFF)
    cudaBindTexture(NULL, tex_YCutoff, gpu->gpu_sim.YCutoff, sizeof(QUICKDouble) * gpu->nshell * gpu->nshell);
  #endif
  #if defined(USE_TEXTURE_XCOEFF)
    cudaBindTexture(NULL, tex_Xcoeff, gpu->gpu_sim.Xcoeff, sizeof(QUICKDouble) * 4 * gpu->jbasis * gpu->jbasis);
  #endif
}


static void unbind_eri_texture()
{
  #if defined(USE_TEXTURE_CUTMATRIX)
    cudaUnbindTexture(tex_cutMatrix);
  #endif
  #if defined(USE_TEXTURE_YCUTOFF)
    cudaUnbindTexture(tex_YCutoff);
  #endif
  #if defined(USE_TEXTURE_XCOEFF)
    cudaUnbindTexture(tex_Xcoeff);    
  #endif
}
#endif


// totTime is the timer for GPU 2e time. Only on under debug mode
#if defined(DEBUG) || defined(DEBUGTIME)
static float totTime;
#endif


#ifdef COMPILE_GPU_AOINT
// =======   INTERFACE SECTION ===========================
// interface to call Kernel subroutine
void getAOInt(_gpu_type gpu, QUICKULL intStart, QUICKULL intEnd, cudaStream_t streamI, int streamID, ERI_entry* aoint_buffer)
{
    QUICK_SAFE_CALL((getAOInt_kernel <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
#ifdef GPU_SPDF
    // Part f-1
    QUICK_SAFE_CALL((getAOInt_kernel_spdf <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
    // Part f-2
    QUICK_SAFE_CALL((getAOInt_kernel_spdf2 <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
    // Part f-3
    QUICK_SAFE_CALL((getAOInt_kernel_spdf3 <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
    // Part f-4
    QUICK_SAFE_CALL((getAOInt_kernel_spdf4 <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
    // Part f-5
    QUICK_SAFE_CALL((getAOInt_kernel_spdf5 <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
    // Part f-6
    QUICK_SAFE_CALL((getAOInt_kernel_spdf6 <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
    // Part f-7
    QUICK_SAFE_CALL((getAOInt_kernel_spdf7 <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
    // Part f-8
    QUICK_SAFE_CALL((getAOInt_kernel_spdf8 <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
    // Part f-9
    QUICK_SAFE_CALL((getAOInt_kernel_spdf9 <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
    // Part f-10
    QUICK_SAFE_CALL((getAOInt_kernel_spdf10 <<<gpu->blocks, gpu->twoEThreadsPerBlock, 0, streamI>>> (intStart, intEnd, aoint_buffer, streamID)));
#endif
}
#endif


// interface to call Kernel subroutine
void get2e(_gpu_type gpu)
{
    // Part spd
//    nvtxRangePushA("SCF 2e");

#if defined(USE_TEXTURE)
    bind_eri_texture(gpu);
#endif

    QUICK_SAFE_CALL((k_eri_cshell_sp <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.dense,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

    QUICK_SAFE_CALL((k_eri_cshell_spd <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.dense,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
 
#ifdef GPU_SPDF
    if (gpu->maxL >= 3) {
        // Part f-1
        QUICK_SAFE_CALL((k_eri_cshell_spdf <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.dense,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-2
        QUICK_SAFE_CALL((k_eri_cshell_spdf2 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.dense,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-3
        QUICK_SAFE_CALL((k_eri_cshell_spdf3 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.dense,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-4
        QUICK_SAFE_CALL((k_eri_cshell_spdf4 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.dense,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-5
        QUICK_SAFE_CALL((k_eri_cshell_spdf5 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.dense,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-6
        QUICK_SAFE_CALL((k_eri_cshell_spdf6 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.dense,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-7
        QUICK_SAFE_CALL((k_eri_cshell_spdf7 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.dense,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-8
        QUICK_SAFE_CALL((k_eri_cshell_spdf8 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.dense,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-9
//        QUICK_SAFE_CALL((k_eri_cshell_spdf9 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
//                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
//                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
//                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
//                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
//                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
//                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
//                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
//                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
//#if defined(USE_LEGACY_ATOMICS)
//                 gpu->gpu_sim.oULL,
//#else
//                 gpu->gpu_sim.o,
//#endif
//                 gpu->gpu_sim.dense,
//                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
//                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
//                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
//                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
//                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
//#if defined(MPIV_GPU)
//                 gpu->gpu_sim.mpi_bcompute,
//#endif
//                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-10
//        QUICK_SAFE_CALL((k_eri_cshell_spdf10 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
//                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
//                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
//                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
//                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
//                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
//                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
//                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
//                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
//#if defined(USE_LEGACY_ATOMICS)
//                 gpu->gpu_sim.oULL,
//#else
//                 gpu->gpu_sim.o,
//#endif
//                 gpu->gpu_sim.dense,
//                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
//                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
//                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
//                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
//                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
//#if defined(MPIV_GPU)
//                 gpu->gpu_sim.mpi_bcompute,
//#endif
//                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
    }
#endif 

#if defined(USE_TEXTURE)
    unbind_eri_texture();
#endif

//    nvtxRangePop();
}


// interface to call Kernel subroutine for uscf
void get_oshell_eri(_gpu_type gpu)
{
    // Part spd
//    nvtxRangePushA("SCF 2e");

#if defined(USE_TEXTURE)
    bind_eri_texture(gpu);
#endif

    QUICK_SAFE_CALL((k_eri_oshell_sp <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>> 
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
#else
                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
#endif
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

    QUICK_SAFE_CALL((k_eri_oshell_spd <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>> 
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
#else
                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
#endif
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

#ifdef GPU_SPDF
    if (gpu->maxL >= 3) {
        // Part f-1
        QUICK_SAFE_CALL((k_eri_oshell_spdf <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>> 
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
#else
                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
#endif
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

        // Part f-2
        QUICK_SAFE_CALL((k_eri_oshell_spdf2 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
#else
                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
#endif
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-3
        QUICK_SAFE_CALL((k_eri_oshell_spdf3 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
#else
                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
#endif
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-4
        QUICK_SAFE_CALL((k_eri_oshell_spdf4 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
#else
                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
#endif
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-5
        QUICK_SAFE_CALL((k_eri_oshell_spdf5 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
#else
                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
#endif
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-6
        QUICK_SAFE_CALL((k_eri_oshell_spdf6 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
#else
                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
#endif
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-7
        QUICK_SAFE_CALL((k_eri_oshell_spdf7 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
#else
                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
#endif
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-8
        QUICK_SAFE_CALL((k_eri_oshell_spdf8 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
#else
                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
#endif
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-9
//        QUICK_SAFE_CALL((k_eri_oshell_spdf9 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
//                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)>>>
//                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
//                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
//                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
//                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
//                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
//                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
//                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
//#if defined(USE_LEGACY_ATOMICS)
//                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
//#else
//                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
//#endif
//                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
//                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
//                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
//                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
//                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
//                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
//#if defined(MPIV_GPU)
//                 gpu->gpu_sim.mpi_bcompute,
//#endif
//                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        // Part f-10
//        QUICK_SAFE_CALL((k_eri_oshell_spdf10 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
//                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u + 3 * gpu->nbasis)>>>
//                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
//                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
//                 gpu->gpu_sim.xyz, gpu->gpu_sim.fStart, gpu->gpu_sim.ffStart, gpu->gpu_sim.kstart,
//                 gpu->gpu_sim.katom, gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart,
//                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
//                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
//                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
//#if defined(USE_LEGACY_ATOMICS)
//                 gpu->gpu_sim.oULL, gpu->gpu_sim.obULL,
//#else
//                 gpu->gpu_sim.o, gpu->gpu_sim.ob,
//#endif
//                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
//                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
//                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
//                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
//                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.integralCutoff,
//                 gpu->gpu_sim.primLimit, gpu->gpu_sim.maxIntegralCutoff, gpu->gpu_sim.leastIntegralCutoff,
//#if defined(MPIV_GPU)
//                 gpu->gpu_sim.mpi_bcompute,
//#endif
//                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
    }
#endif

#if defined(USE_TEXTURE)
    unbind_eri_texture();
#endif

//    nvtxRangePop();
}


#ifdef COMPILE_GPU_AOINT
// interface to call Kernel subroutine
void getAddInt(_gpu_type gpu, uint32_t bufferSize, ERI_entry* aoint_buffer)
{
    QUICK_SAFE_CALL((k_get_add_int <<<gpu->blocks, gpu->twoEThreadsPerBlock>>> 
                (bufferSize, aoint_buffer, gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.nbasis,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
  #if defined(OSHELL)
                 gpu->gpu_sim.obULL,
  #endif
#else
                 gpu->gpu_sim.o,
  #if defined(OSHELL)
                 gpu->gpu_sim.ob,
  #endif
#endif
                 gpu->gpu_sim.dense
#if defined(OSHELL)
                 , gpu->gpu_sim.denseb
#endif
                 )));
}
#endif


// interface to call Kernel subroutine
void getGrad(_gpu_type gpu)
{
//   nvtxRangePushA("Gradient 2e");
    QUICK_SAFE_CALL((k_get_grad_cshell_sp <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
#if defined(USE_LEGACY_ATOMICS)
                + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                 gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
                 gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
                 gpu->gpu_sim.dense, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
                 gpu->gpu_sim.gradCutoff,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.gradULL,
#else
                 gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.store2,
                 gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
                 gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

    QUICK_SAFE_CALL((k_get_grad_cshell_spd <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
#if defined(USE_LEGACY_ATOMICS)
                + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                 gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
                 gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
                 gpu->gpu_sim.dense, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
                 gpu->gpu_sim.gradCutoff,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.gradULL,
#else
                 gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.store2,
                 gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
                 gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

    // compute one electron gradients in the meantime
    //get_oneen_grad_();

    if (gpu->maxL >= 2) {
        // Part f-1
        QUICK_SAFE_CALL((k_get_grad_cshell_spdf <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                    sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
#if defined(USE_LEGACY_ATOMICS)
                    + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                    + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                    (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                     gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                     gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                     gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                     gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                     gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
                     gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
                     gpu->gpu_sim.dense, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                     gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                     gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                     gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
                     gpu->gpu_sim.gradCutoff,
#if defined(USE_LEGACY_ATOMICS)
                     gpu->gpu_sim.gradULL,
#else
                     gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                     gpu->gpu_sim.mpi_bcompute,
#endif
                     gpu->gpu_sim.store, gpu->gpu_sim.store2,
                     gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
                     gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
        
        // Part f-2
        QUICK_SAFE_CALL((k_get_grad_cshell_spdf2 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                    sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
#if defined(USE_LEGACY_ATOMICS)
                    + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                    + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                    (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                     gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                     gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                     gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                     gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                     gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
                     gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
                     gpu->gpu_sim.dense, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                     gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                     gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                     gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
                     gpu->gpu_sim.gradCutoff,
#if defined(USE_LEGACY_ATOMICS)
                     gpu->gpu_sim.gradULL,
#else
                     gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                     gpu->gpu_sim.mpi_bcompute,
#endif
                     gpu->gpu_sim.store, gpu->gpu_sim.store2,
                     gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
                     gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

        if (gpu->maxL >= 3) {
#ifdef GPU_SPDF
            // Part f-3
            QUICK_SAFE_CALL((k_get_grad_cshell_spdf3 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                        sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
#if defined(USE_LEGACY_ATOMICS)
                        + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                        + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                        (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                         gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                         gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                         gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                         gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                         gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
                         gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
                         gpu->gpu_sim.dense, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                         gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                         gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                         gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
                         gpu->gpu_sim.gradCutoff,
#if defined(USE_LEGACY_ATOMICS)
                         gpu->gpu_sim.gradULL,
#else
                         gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                         gpu->gpu_sim.mpi_bcompute,
#endif
                         gpu->gpu_sim.store, gpu->gpu_sim.store2,
                         gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
                         gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

//            QUICK_SAFE_CALL((k_get_grad_cshell_spdf4 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
//                        sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
//#if defined(USE_LEGACY_ATOMICS)
//                        + sizeof(QUICKULL) * 3u * gpu->natom>>>
//#else
//                        + sizeof(QUICKDouble) * 3u * gpu->natom>>>
//#endif
//                        (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
//                         gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
//                         gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
//                         gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
//                         gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
//                         gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
//                         gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
//                         gpu->gpu_sim.dense, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
//                         gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
//                         gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
//                         gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
//                         gpu->gpu_sim.gradCutoff,
//#if defined(USE_LEGACY_ATOMICS)
//                         gpu->gpu_sim.gradULL,
//#else
//                         gpu->gpu_sim.grad,
//#endif
//#if defined(MPIV_GPU)
//                         gpu->gpu_sim.mpi_bcompute,
//#endif
//                         gpu->gpu_sim.store, gpu->gpu_sim.store2,
//                         gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
//                         gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
#endif
        }
    }

//    nvtxRangePop();
}


// interface to call uscf gradient Kernels
void get_oshell_eri_grad(_gpu_type gpu)
{
//    nvtxRangePushA("Gradient 2e");
    QUICK_SAFE_CALL((k_get_grad_oshell_sp <<<gpu->blocks, gpu->gradThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
#if defined(USE_LEGACY_ATOMICS)
                + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                 gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
                 gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
                 gpu->gpu_sim.gradCutoff,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.gradULL,
#else
                 gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.store2,
                 gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
                 gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

    QUICK_SAFE_CALL((k_get_grad_oshell_spd <<<gpu->blocks, gpu->gradThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
#if defined(USE_LEGACY_ATOMICS)
                + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                 gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
                 gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
                 gpu->gpu_sim.gradCutoff,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.gradULL,
#else
                 gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.store2,
                 gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
                 gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

    // compute one electron gradients in the meantime
    //get_oneen_grad_();

    if (gpu->maxL >= 2) {
//#ifdef GPU_SPDF
        // Part f-1
        QUICK_SAFE_CALL((k_get_grad_oshell_spdf <<<gpu->blocks, gpu->gradThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
#if defined(USE_LEGACY_ATOMICS)
                + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                 gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
                 gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
                 gpu->gpu_sim.gradCutoff,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.gradULL,
#else
                 gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.store2,
                 gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
                 gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

        // Part f-2
        QUICK_SAFE_CALL((k_get_grad_oshell_spdf2 <<<gpu->blocks, gpu->gradThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
#if defined(USE_LEGACY_ATOMICS)
                + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
                 gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                 gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
                 gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
                 gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
                 gpu->gpu_sim.dense, gpu->gpu_sim.denseb, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
                 gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
                 gpu->gpu_sim.gradCutoff,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.gradULL,
#else
                 gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.store2,
                 gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
                 gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

        // Part f-3
	if (gpu->maxL >= 3) {
//            QUICK_SAFE_CALL((k_get_grad_oshell_spdf3 <<<gpu->blocks, gpu->gradThreadsPerBlock,
//                    sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u)
//#if defined(USE_LEGACY_ATOMICS)
//                    + sizeof(QUICKULL) * 3u * gpu->natom>>>
//#else
//                    + sizeof(QUICKDouble) * 3u * gpu->natom>>>
//#endif
//                    (gpu->gpu_sim.hyb_coeff, gpu->gpu_sim.natom,
//                     gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
//                     gpu->gpu_sim.xyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
//                     gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
//                     gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber,
//                     gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo,
//                     gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
//                     gpu->gpu_sim.dense, gpu->gpu_sim.denseb, gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
//                     gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
//                     gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ, gpu->gpu_sim.cutMatrix,
//                     gpu->gpu_sim.YCutoff, gpu->gpu_sim.cutPrim, gpu->gpu_sim.primLimit,
//                     gpu->gpu_sim.gradCutoff,
//#if defined(USE_LEGACY_ATOMICS)
//                     gpu->gpu_sim.gradULL,
//#else
//                     gpu->gpu_sim.grad,
//#endif
//#if defined(MPIV_GPU)
//                     gpu->gpu_sim.mpi_bcompute,
//#endif
//                     gpu->gpu_sim.store, gpu->gpu_sim.store2,
//                     gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB, gpu->gpu_sim.storeCC,
//                     gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
//#endif
	}
    }

//    nvtxRangePop();
}


#ifdef COMPILE_GPU_AOINT
// =======   KERNEL SECTION ===========================
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_get_add_int(uint32_t bufferSize, ERI_entry* aoint_buffer,
        QUICKDouble hyb_coeff, uint32_t nbasis,
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL * const oULL,
  #if defined(OSHELL)
        QUICKULL * const obULL,
  #endif
#else
        QUICKDouble * const o,
  #if defined(OSHELL)
        QUICKDouble * const ob,
  #endif
#endif
        QUICKDouble * const dense
#if defined(OSHELL)
        , QUICKDouble * const denseb
#endif
        )
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    uint32_t const batchSize = 20;
    ERI_entry a[batchSize];
    uint32_t j = 0;
    QUICKDouble temp;
#if defined(OSHELL)
    QUICKDouble temp2;
#endif
 
    QUICKULL myInt = (QUICKULL) (bufferSize) / totalThreads;
    if ((bufferSize - myInt * totalThreads) > offset) myInt++;
    
    for (QUICKULL i = 1; i <= myInt; i++) {
        QUICKULL currentInt = totalThreads * (i - 1) + offset;
        a[j] = aoint_buffer[currentInt];
        j++;

        if (j == batchSize || i == myInt) {
            for (uint32_t k = 0; k < j; k++) {
                uint32_t III = a[k].IJ / nbasis;
                uint32_t JJJ = a[k].IJ % nbasis;
                uint32_t KKK = a[k].KL / nbasis;
                uint32_t LLL = a[k].KL % nbasis;
                
                if (III < nbasis && III >= 0 && JJJ < nbasis && JJJ >= 0
                        && KKK < nbasis && KKK >= 0 && LLL < nbasis && LLL >= 0) {
//                    QUICKDouble hybrid_coeff = 0.0;
//                    if (method == HF) {
//                        hybrid_coeff = 1.0;
//                    } else if (method == B3LYP) {
//                        hybrid_coeff = 0.2;
//                    } else if (method == DFT) {
//                        hybrid_coeff = 0.0;
//                    } else if( method == LIBXC) {
//			hybrid_coeff = hyb_coeff;			
//		    }

#if defined(OSHELL)
                    QUICKDouble DENSELK = (QUICKDouble) (LOC2(dense, LLL, KKK, nbasis, nbasis)
                            + LOC2(denseb, LLL, KKK, nbasis, nbasis));
                    QUICKDouble DENSEJI = (QUICKDouble) (LOC2(dense, JJJ, III, nbasis, nbasis)
                            + LOC2(denseb, JJJ, III, nbasis, nbasis));

                    QUICKDouble DENSEKIA = (QUICKDouble) LOC2(dense, KKK, III, nbasis, nbasis);
                    QUICKDouble DENSEKJA = (QUICKDouble) LOC2(dense, KKK, JJJ, nbasis, nbasis);
                    QUICKDouble DENSELJA = (QUICKDouble) LOC2(dense, LLL, JJJ, nbasis, nbasis);
                    QUICKDouble DENSELIA = (QUICKDouble) LOC2(dense, LLL, III, nbasis, nbasis);

                    QUICKDouble DENSEKIB = (QUICKDouble) LOC2(denseb, KKK, III, nbasis, nbasis);
                    QUICKDouble DENSEKJB = (QUICKDouble) LOC2(denseb, KKK, JJJ, nbasis, nbasis);
                    QUICKDouble DENSELJB = (QUICKDouble) LOC2(denseb, LLL, JJJ, nbasis, nbasis);
                    QUICKDouble DENSELIB = (QUICKDouble) LOC2(denseb, LLL, III, nbasis, nbasis);
#else
                    QUICKDouble DENSEKI = (QUICKDouble) LOC2(dense, KKK, III, nbasis, nbasis);
                    QUICKDouble DENSEKJ = (QUICKDouble) LOC2(dense, KKK, JJJ, nbasis, nbasis);
                    QUICKDouble DENSELJ = (QUICKDouble) LOC2(dense, LLL, JJJ, nbasis, nbasis);
                    QUICKDouble DENSELI = (QUICKDouble) LOC2(dense, LLL, III, nbasis, nbasis);
                    QUICKDouble DENSELK = (QUICKDouble) LOC2(dense, LLL, KKK, nbasis, nbasis);
                    QUICKDouble DENSEJI = (QUICKDouble) LOC2(dense, JJJ, III, nbasis, nbasis);
#endif

                    // ATOMIC ADD VALUE 1
                    temp = (KKK == LLL) ? DENSELK * a[k].value : 2.0 * DENSELK * a[k].value;
                    o_JI += temp;
#if defined(OSHELL)
                    ob_JI += temp;
#endif

                    // ATOMIC ADD VALUE 2
                    if (LLL != JJJ || III != KKK) {
                        temp = (III == JJJ) ? DENSEJI * a[k].value : 2.0 * DENSEJI * a[k].value;
#  if defined(USE_LEGACY_ATOMICS)
                        GPUATOMICADD(&LOC2(oULL, LLL, KKK, nbasis, nbasis), temp, OSCALE);
#  else
                        atomicAdd(&LOC2(o, LLL, KKK, nbasis, nbasis), temp);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                        GPUATOMICADD(&LOC2(obULL, LLL, KKK, nbasis, nbasis), temp, OSCALE);
#  else
                        atomicAdd(&LOC2(ob, LLL, KKK, nbasis, nbasis), temp);
#  endif
#endif
                    }

                    // ATOMIC ADD VALUE 3
#if defined(OSHELL)
                    temp = (III == KKK && III < JJJ && JJJ < LLL)
                        ? -2.0 * hyb_coeff * DENSELJA * a[k].value : -(hyb_coeff * DENSELJA * a[k].value);
                    temp2 = (III == KKK && III < JJJ && JJJ < LLL)
                        ? -2.0 * hyb_coeff * DENSELJB * a[k].value : -(hyb_coeff * DENSELJB * a[k].value);
                    o_KI += temp;
                    ob_KI += temp2;
#else
                    temp = (III == KKK && III < JJJ && JJJ < LLL)
                        ? -(hyb_coeff * DENSELJ * a[k].value) : -0.5 * hyb_coeff * DENSELJ * a[k].value;
                    o_KI += temp;
#endif

                    // ATOMIC ADD VALUE 4
                    if (KKK != LLL) {
#if defined(OSHELL)
                        temp = -(hyb_coeff * DENSEKJA * a[k].value);
                        temp2 = -(hyb_coeff * DENSEKJB * a[k].value);
#  if defined(USE_LEGACY_ATOMICS)
                        GPUATOMICADD(&LOC2(oULL, LLL, III, nbasis, nbasis), temp, OSCALE);
                        GPUATOMICADD(&LOC2(obULL, LLL, III, nbasis, nbasis), temp2, OSCALE);
#  else
                        atomicAdd(&LOC2(o, LLL, III, nbasis, nbasis), temp);
                        atomicAdd(&LOC2(ob, LLL, III, nbasis, nbasis), temp2);
#  endif
#else
                        temp = -0.5 * hyb_coeff * DENSEKJ * a[k].value;
#  if defined(USE_LEGACY_ATOMICS)
                        GPUATOMICADD(&LOC2(oULL, LLL, III, nbasis, nbasis), temp, OSCALE);
#  else
                        atomicAdd(&LOC2(o, LLL, III, nbasis, nbasis), temp);
#  endif
#endif
                    }

                    // ATOMIC ADD VALUE 5
#if defined(OSHELL)
                    temp = -(hyb_coeff * DENSELIA * a[k].value);
                    temp2 = -(hyb_coeff * DENSELIB * a[k].value);
#else
                    temp = -0.5 * hyb_coeff * DENSELI * a[k].value;
#endif
                    if ((III != JJJ && III < KKK)
                            || (III == JJJ && III == KKK && III < LLL)
                            || (III == KKK && III < JJJ && JJJ < LLL)) {
                        o_JK_MM += temp;
#if defined(OSHELL)
                        ob_JK_MM += temp2;
#endif
                    }

                    // ATOMIC ADD VALUE 5 - 2
                    if (III != JJJ && JJJ == KKK) {
                        o_JK += temp;
#if defined(OSHELL)
                        ob_JK += temp2;
#endif
                    }

                    // ATOMIC ADD VALUE 6
                    if (III != JJJ && KKK != LLL) {
#if defined(OSHELL)
                        temp = -(hyb_coeff * DENSEKIA * a[k].value);
                        temp2 = -(hyb_coeff * DENSEKIB * a[k].value);
#else
                        temp = -0.5 * hyb_coeff * DENSEKI * a[k].value;
#endif
#  if defined(USE_LEGACY_ATOMICS)
                        GPUATOMICADD(&LOC2(oULL, MAX(JJJ, LLL), MIN(JJJ, LLL), nbasis, nbasis), temp, OSCALE);
#  else
                        atomicAdd(&LOC2(o, MAX(JJJ, LLL), MIN(JJJ, LLL), nbasis, nbasis), temp);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                        GPUATOMICADD(&LOC2(obULL, MAX(JJJ, LLL), MIN(JJJ, LLL), nbasis, nbasis), temp2, OSCALE);
#  else
                        atomicAdd(&LOC2(ob, MAX(JJJ, LLL), MIN(JJJ, LLL), nbasis, nbasis), temp2);
#  endif
#endif

                        // ATOMIC ADD VALUE 6 - 2
                        if (JJJ == LLL && III != KKK) {
#  if defined(USE_LEGACY_ATOMICS)
                            GPUATOMICADD(&LOC2(oULL, LLL, JJJ, nbasis, nbasis), temp, OSCALE);
#  else
                            atomicAdd(&LOC2(o, LLL, JJJ, nbasis, nbasis), temp);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                            GPUATOMICADD(&LOC2(obULL, LLL, JJJ, nbasis, nbasis), temp2, OSCALE);
#  else
                            atomicAdd(&LOC2(ob, LLL, JJJ, nbasis, nbasis), temp2);
#  endif
#endif
                        }
                    }
                }
            }

            j = 0;
        }
    }
}
#endif
