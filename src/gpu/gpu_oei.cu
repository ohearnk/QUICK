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
   ! This source file contains functions required for QUICK one electron !
   ! integral computation.                                               !
   !---------------------------------------------------------------------!
 */

#if defined(CUDA) || defined(CUDA_MPIV)
  #include "cuda/gpu.h"
#elif defined(HIP) || defined(HIP_MPIV)
  #include "hip/gpu.h"
#endif
#include "gpu_common.h"


#define STOREDIM (20)
#define REG_PF
#define REG_FP
#define REG_SF
#define REG_FS
//#define USE_PARTIAL_DP
//#define USE_PARTIAL_PF
//#define USE_PARTIAL_FP

#include "gpu_oei_classes.h"
#include "gpu_oei_definitions.h"
#include "gpu_oei_assembler.h"
#include "gpu_oei.h"

#include "gpu_oei_grad_assembler.h"
#include "gpu_oei_grad.h"


#if defined(DEBUG) || defined(DEBUGTIME)
static float totTime;
#endif


// interface for kernel launching
void getOEI(_gpu_type gpu) {
    QUICK_SAFE_CALL((k_oei <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u + 3u * gpu->nbasis)>>>
                (gpu->gpu_sim.natom, gpu->gpu_sim.nextatom, gpu->gpu_sim.nbasis, gpu->gpu_sim.nshell,
                 gpu->gpu_sim.jbasis, gpu->gpu_sim.Qshell, gpu->gpu_sim.allchg,
                 gpu->gpu_sim.allxyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                 gpu->gpu_sim.kprim, gpu->gpu_sim.Qstart, gpu->gpu_sim.Qsbasis,
                 gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber, gpu->gpu_sim.sorted_Q,
                 gpu->gpu_sim.cons, gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total, gpu->gpu_sim.prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.oULL,
#else
                 gpu->gpu_sim.o,
#endif
                 gpu->gpu_sim.Xcoeff_oei, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.coreIntegralCutoff, gpu->gpu_sim.sorted_OEICutoffIJ,
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_boeicompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.store2,
                 gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
}


void get_oei_grad(_gpu_type gpu) {
    QUICK_SAFE_CALL((k_oei_grad <<<gpu->blocks, gpu->twoEThreadsPerBlock,
                sizeof(uint32_t) * (TRANSDIM * TRANSDIM * TRANSDIM + 10u + 3u * gpu->nbasis)
#if defined(USE_LEGACY_ATOMICS)
                + sizeof(QUICKULL) * 3u * (gpu->natom + gpu->nextatom)>>>
#else
                + sizeof(QUICKDouble) * 3u * (gpu->natom + gpu->nextatom)>>>
#endif
                (gpu->gpu_sim.is_oshell, gpu->gpu_sim.natom, gpu->gpu_sim.nextatom, gpu->gpu_sim.nbasis,
                 gpu->gpu_sim.nshell, gpu->gpu_sim.jbasis, gpu->gpu_sim.Qshell, gpu->gpu_sim.allchg,
                 gpu->gpu_sim.allxyz, gpu->gpu_sim.kstart, gpu->gpu_sim.katom,
                 gpu->gpu_sim.kprim, gpu->gpu_sim.Ksumtype, gpu->gpu_sim.Qstart, gpu->gpu_sim.Qsbasis,
                 gpu->gpu_sim.Qfbasis, gpu->gpu_sim.sorted_Qnumber, gpu->gpu_sim.sorted_Q,
                 gpu->gpu_sim.cons, gpu->gpu_sim.gcexpo, gpu->gpu_sim.KLMN, gpu->gpu_sim.prim_total,
                 gpu->gpu_sim.prim_start, gpu->gpu_sim.dense, gpu->gpu_sim.denseb,
                 gpu->gpu_sim.Xcoeff_oei, gpu->gpu_sim.expoSum,
                 gpu->gpu_sim.weightedCenterX, gpu->gpu_sim.weightedCenterY, gpu->gpu_sim.weightedCenterZ,
                 gpu->gpu_sim.coreIntegralCutoff, gpu->gpu_sim.sorted_OEICutoffIJ,
#if defined(USE_LEGACY_ATOMICS)
                 gpu->gpu_sim.gradULL, gpu->gpu_sim.ptchg_gradULL,
#else
                 gpu->gpu_sim.grad, gpu->gpu_sim.ptchg_grad,
#endif
#if defined(MPIV_GPU)
                 gpu->gpu_sim.mpi_boeicompute,
#endif
                 gpu->gpu_sim.store, gpu->gpu_sim.store2, gpu->gpu_sim.storeAA,
                 gpu->gpu_sim.storeBB, gpu->gpu_sim.trans, gpu->gpu_sim.Sumindex)));
}

