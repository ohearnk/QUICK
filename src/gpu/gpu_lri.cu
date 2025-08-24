/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 06/17/2021                            !
  !                                                                     !
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains functions required for computing 3 center !
  ! integrals necessary for CEW method.                                 !
  !---------------------------------------------------------------------!
*/

#if defined(CEW)

#if defined(CUDA) || defined(CUDA_MPIV)
  #include "cuda/gpu.h"
#elif defined(HIP) || defined(HIP_MPIV)
  #include "hip/gpu.h"
#endif


#include "gpu_lri_subs_hrr.h"

namespace lri {
#include "gpu_lri_vertical_int.h"
}

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
#include "gpu_lri_subs.h"
#include "gpu_lri_subs_grad.h"
//===================================
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
#include "gpu_lri_subs_grad.h"
//===================================
#if defined(GPU_SPDF)
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
  #include "gpu_lri_subs.h"
#endif
//===================================
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


// totTime is the timer for GPU lri time. Only on under debug mode
#if defined(DEBUG) || defined(DEBUGTIME)
static float totTime;
#endif

// =======   INTERFACE SECTION ===========================


// interface to call Kernel subroutine
void get_lri(_gpu_type gpu)
{
    // Part spd
//    nvtxRangePushA("SCF lri");
    QUICK_SAFE_CALL((k_get_lri <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u + 3u * gpu->nbasis)>>>
                (gpu->gpu_sim.natom, gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.allxyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                 gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart, gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis,
                 gpu->gpu_sim.sorted_Qnumber, gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY,
                 gpu->gpu_sim.weightedCenterZ, sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
 
#if defined(GPU_SPDF)
    if (gpu->maxL >= 3) {
        // Part f-2
        QUICK_SAFE_CALL((k_get_lri_spdf2 <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                    sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u + 3u * gpu->nbasis)>>>
                    (gpu->gpu_sim.natom, gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                     gpu->gpu_sim.xyz, gpu->gpu_sim.allxyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                     gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart, gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis,
                     gpu->gpu_sim.sorted_Qnumber, gpu->gpu_sim.sorted_Q, gpu->gpu_sim.cons, gpu->gpu_sim.KLMN,
                     gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                     gpu->gpu_sim.oULL,
#else
                     gpu->gpu_sim.o,
#endif
                     gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                     gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY,
                     gpu->gpu_sim.weightedCenterZ, sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ,
#if defined(MPIV_GPU)
                     gpu->gpu_sim.mpi_bcompute,
#endif
                     gpu->gpu_sim.store, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
    }
#endif 

//    nvtxRangePop();
}


// interface to call Kernel subroutine
void get_lri_grad(_gpu_type gpu)
{
//   nvtxRangePushA("Gradient lri");
    QUICK_SAFE_CALL((k_get_lri_grad <<<gpu->blocks, gpu->gradThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u + 3u * gpu->nbasis)
#if defined(USE_LEGACY_ATOMICS)
                + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                (gpu->gpu_sim.natom, gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                 gpu->gpu_sim.xyz, gpu->gpu_sim.allxyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                 gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                 gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis,
                 gpu->gpu_sim.sorted_Qnumber, gpu->gpu_sim.sorted_Q,
                 gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo, gpu->gpu_sim.KLMN,
                 gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start, gpu->gpu_sim.dense,
#if defined(OSHELL)
                 gpu->gpu_sim.denseb,
#endif
                 gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY,
                 gpu->gpu_sim.weightedCenterZ, gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.gradULL,
#else
                 gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_bcompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.store2, gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB,
                 gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));

    if (gpu->maxL >= 2) {
//#if defined(GPU_SPDF)
        // Part f-2
        QUICK_SAFE_CALL((k_get_lri_grad_spdf2 <<<gpu->blocks, gpu->gradThreadsPerBlock,
                    sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u + 3u * gpu->nbasis)
#if defined(USE_LEGACY_ATOMICS)
                    + sizeof(QUICKULL) * 3u * gpu->natom>>>
#else
                    + sizeof(QUICKDouble) * 3u * gpu->natom>>>
#endif
                    (gpu->gpu_sim.natom, gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis,
                     gpu->gpu_sim.xyz, gpu->gpu_sim.allxyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                     gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart,
                     gpu->gpu_sim.Qsbasis, gpu->gpu_sim.Qfbasis,
                     gpu->gpu_sim.sorted_Qnumber, gpu->gpu_sim.sorted_Q,
                     gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo, gpu->gpu_sim.KLMN,
                     gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start, gpu->gpu_sim.dense,
#if defined(OSHELL)
                     gpu->gpu_sim.denseb,
#endif
                     gpu->gpu_sim.Xcoeff, gpu->gpu_sim.expoSum,
                     gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY,
                     gpu->gpu_sim.weightedCenterZ, gpu->gpu_sim.sqrQshell, gpu->gpu_sim.sorted_YCutoffIJ,
#if defined(USE_LEGACY_ATOMICS)
                     gpu->gpu_sim.gradULL,
#else
                     gpu->gpu_sim.grad,
#endif
#if defined(MPIV_GPU)
                     gpu->gpu_sim.mpi_bcompute,
#endif
                     gpu->gpu_sim.store, gpu->gpu_sim.store2, gpu->gpu_sim.storeAA, gpu->gpu_sim.storeBB,
                     gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
//#endif
    }

//    nvtxRangePop();
}
#endif
