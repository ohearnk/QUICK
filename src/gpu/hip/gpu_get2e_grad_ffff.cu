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

#include <hip/hip_runtime.h>

#include <iostream>
#include <algorithm>

#include "../gpu_common.h"
#include "gpu_type.h"
#include "gpu_get2e_grad_ffff.h"

/*
 * Constant Memory in GPU is fast but quite limited and hard to operate, usually not allocatable and
 * readonly. So we put the following variables into constant memory:
 * devTrans: arrays to save the mapping index, will be elimited by hand writing unrolling code.
 */
static __constant__ uint8_t devTrans[TRANSDIM * TRANSDIM * TRANSDIM];


//#define USE_TEXTURE
#if defined(USE_TEXTURE)
  #define USE_TEXTURE_CUTMATRIX
  #define USE_TEXTURE_YCUTOFF
  #define USE_TEXTURE_XCOEFF
#endif

#if defined(USE_TEXTURE_CUTMATRIX)
  texture <int2, hipTextureType1D, hipReadModeElementType> tex_cutMatrix;
#endif
#if defined(USE_TEXTURE_YCUTOFF)
  texture <int2, hipTextureType1D, hipReadModeElementType> tex_YCutoff;
#endif
#if defined(USE_TEXTURE_XCOEFF)
  texture <int2, hipTextureType1D, hipReadModeElementType> tex_Xcoeff;
#endif

//#define USE_ERI_GRAD_STOREADD
//#ifdef USE_ERI_GRAD_STOREADD
//  #define STORE_OPERATOR +=
//#else
//  #define STORE_OPERATOR =
//#endif

#define ERI_GRAD_FFFF_TPB (32)
#define ERI_GRAD_FFFF_BPSM (8)

#define ERI_GRAD_FFFF_SMEM_UINT8_SIZE (512)
#define ERI_GRAD_FFFF_SMEM_UINT32_SIZE (5)
#define ERI_GRAD_FFFF_SMEM_UINT32_PTR_SIZE (11)
#define ERI_GRAD_FFFF_SMEM_DBL_SIZE (3)
#define ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE (18)
#define ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE (1)
#define ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE (1)

#define ERI_GRAD_FFFF_SMEM_PTR_SIZE (1)

#define DEV_SIM_UINT32_PTR_KATOM smem_uint32_ptr[threadIdx.x]
#define DEV_SIM_UINT32_PTR_KPRIM smem_uint32_ptr[ERI_GRAD_FFFF_TPB + threadIdx.x]
#define DEV_SIM_UINT32_PTR_KSTART smem_uint32_ptr[ERI_GRAD_FFFF_TPB * 2 + threadIdx.x]
#define DEV_SIM_UINT32_PTR_KSUMTYPE smem_uint32_ptr[ERI_GRAD_FFFF_TPB * 3 + threadIdx.x]
#define DEV_SIM_UINT32_PTR_PRIM_START smem_uint32_ptr[ERI_GRAD_FFFF_TPB * 4 + threadIdx.x]
#define DEV_SIM_UINT32_PTR_QFBASIS smem_uint32_ptr[ERI_GRAD_FFFF_TPB * 5 + threadIdx.x]
#define DEV_SIM_UINT32_PTR_QSBASIS smem_uint32_ptr[ERI_GRAD_FFFF_TPB * 6 + threadIdx.x]
#define DEV_SIM_UINT32_PTR_QSTART smem_uint32_ptr[ERI_GRAD_FFFF_TPB * 7 + threadIdx.x]
#define DEV_SIM_UINT32_PTR_SORTED_QNUMBER smem_uint32_ptr[ERI_GRAD_FFFF_TPB * 8 + threadIdx.x]
#define DEV_SIM_UINT32_PTR_SORTED_Q smem_uint32_ptr[ERI_GRAD_FFFF_TPB * 9 + threadIdx.x]
#define DEV_SIM_UINT32_PTR_KLMN smem_uint32_ptr[ERI_GRAD_FFFF_TPB + 10 * threadIdx.x]
#define DEV_SIM_INT2_PTR_SORTED_YCUTOFFIJ smem_int2_ptr[threadIdx.x]
#define DEV_SIM_CHAR_PTR_MPI_BCOMPUTE smem_char_ptr[threadIdx.x]
#define DEV_SIM_DBL_PTR_CONS smem_dbl_ptr[threadIdx.x]
#define DEV_SIM_DBL_PTR_CUTMATRIX smem_dbl_ptr[ERI_GRAD_FFFF_TPB + threadIdx.x]
#define DEV_SIM_DBL_PTR_CUTPRIM smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 2 + threadIdx.x]
#define DEV_SIM_DBL_PTR_DENSE smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 3 + threadIdx.x]
#define DEV_SIM_DBL_PTR_DENSEB smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 4 + threadIdx.x]
#define DEV_SIM_DBL_PTR_EXPOSUM smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 5 + threadIdx.x]
#define DEV_SIM_DBL_PTR_GCEXPO smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 6 + threadIdx.x]
#define DEV_SIM_DBL_PTR_STORE smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 7 + threadIdx.x]
#define DEV_SIM_DBL_PTR_STORE2 smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 8 + threadIdx.x]
#define DEV_SIM_DBL_PTR_STOREAA smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 9 + threadIdx.x]
#define DEV_SIM_DBL_PTR_STOREBB smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 10 + threadIdx.x]
#define DEV_SIM_DBL_PTR_STORECC smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 11 + threadIdx.x]
#define DEV_SIM_DBL_PTR_WEIGHTEDCENTERX smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 12 + threadIdx.x]
#define DEV_SIM_DBL_PTR_WEIGHTEDCENTERY smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 13 + threadIdx.x]
#define DEV_SIM_DBL_PTR_WEIGHTEDCENTERZ smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 14 + threadIdx.x]
#define DEV_SIM_DBL_PTR_XCOEFF smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 15 + threadIdx.x]
#define DEV_SIM_DBL_PTR_XYZ smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 16 + threadIdx.x]
#define DEV_SIM_DBL_PTR_YCUTOFF smem_dbl_ptr[ERI_GRAD_FFFF_TPB * 17 + threadIdx.x]
#define DEV_SIM_DBL_PRIMLIMIT smem_dbl[threadIdx.x]
#define DEV_SIM_DBL_GRADCUTOFF smem_dbl[ERI_GRAD_FFFF_TPB + threadIdx.x]
#define DEV_SIM_DBL_HYB_COEFF smem_dbl[ERI_GRAD_FFFF_TPB * 2 + threadIdx.x]
#define DEV_SIM_UINT32_NATOM smem_uint32[threadIdx.x]
#define DEV_SIM_UINT32_NBASIS smem_uint32[ERI_GRAD_FFFF_TPB + threadIdx.x]
#define DEV_SIM_UINT32_NSHELL smem_uint32[ERI_GRAD_FFFF_TPB * 2 + threadIdx.x]
#define DEV_SIM_UINT32_JBASIS smem_uint32[ERI_GRAD_FFFF_TPB * 3 + threadIdx.x]
#define DEV_SIM_UINT32_PRIM_TOTAL smem_uint32[ERI_GRAD_FFFF_TPB * 4 + threadIdx.x]

#define DEV_SIM_PTR_GRAD smem_grad_ptr[threadIdx.x]

#define DEV_SIM_UINT8_TRANS smem_uint8

#if defined(GPU_SPDF)
  #define int_spdf4
  #include "../gpu_eri_grad_vrr_ffff.h"
  #include "gpu_get2e_grad_ffff.cuh"
#endif
#undef int_spdf4

//Include the kernels for open shell eri calculations
#define OSHELL
#if defined(GPU_SPDF)
  #define int_spdf4
//  #include "gpu_get2e_grad_ffff.cuh"
  #endif
#undef OSHELL


// totTime is the timer for GPU 2e time. Only on under debug mode
#if defined(DEBUG) || defined(DEBUGTIME)
  static float totTime;
#endif


struct Partial_ERI {
    int32_t YCutoffIJ_x;
    int32_t YCutoffIJ_y;
    uint32_t Qnumber_x;
    uint32_t Qnumber_y;
    uint32_t kprim_x;
    uint32_t kprim_y;
    uint32_t Q_x;
    uint32_t Q_y;
    uint32_t kprim_score;
};


bool ComparePrimNum(Partial_ERI p1, Partial_ERI p2) {
    return p1.kprim_score > p2.kprim_score;
}


void ResortERIs(_gpu_type gpu) {
    int2 eri_type_order[] = {{0,0}, {0,1}, {1,0}, {1,1},
        {0,2}, {2,0}, {1,2}, {2,1},
        {0,3}, {3,0}, {2,2}, {1,3},
        {3,1}, {2,3}, {3,2}, {3,3}};
    unsigned char eri_type_order_map[] = {0, 1, 3, 6, 10, 13, 15, 16};
    uint32_t eri_type_block_map[17];
    int2 *resorted_YCutoffIJ = (int2 *) malloc(sizeof(int2) * gpu->gpu_cutoff->sqrQshell);
    bool ffset = false;

    // Step 1: sort according sum of angular momentum of a partial ERI. (ie. i+j of <ij| ).
    // Step 2: sort according to type order specified in eri_type_order array. This ensures that eri vector follows the order we
    // want.
    uint32_t idx1 = 0;
    uint32_t idx2 = 0;
    uint32_t ffStart = 0;

    for (uint32_t ij_sum = 0; ij_sum <= 6; ij_sum++) {
        for (uint32_t ieto = eri_type_order_map[ij_sum]; ieto < eri_type_order_map[ij_sum + 1]; ieto++) {
            int2 lbl_t = eri_type_order[ieto];
            eri_type_block_map[idx2] = idx1;
            for (uint32_t i = 0; i < gpu->gpu_cutoff->sqrQshell; i++) {
                if ((int) gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x] == lbl_t.x
                        && (int) gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y] == lbl_t.y) {
                    resorted_YCutoffIJ[idx1].x = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x;
                    resorted_YCutoffIJ[idx1].y = gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y;
                    idx1++;
                }
            }
            idx2++;
        }
    }

    eri_type_block_map[idx2]=idx1;

    for (uint32_t i = 0; i < gpu->gpu_cutoff->sqrQshell; i++) {
        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x=resorted_YCutoffIJ[i].x;
        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y=resorted_YCutoffIJ[i].y;

        if (ffset == false
                && gpu->gpu_basis->sorted_Qnumber->_hostData[resorted_YCutoffIJ[i].x]
                + gpu->gpu_basis->sorted_Qnumber->_hostData[resorted_YCutoffIJ[i].y] == 6) {
            ffStart = i;
            ffset = true;
        }
    }

    // create an array of structs
    Partial_ERI *partial_eris = (Partial_ERI *) malloc(sizeof(Partial_ERI) * gpu->gpu_cutoff->sqrQshell);

    for (uint32_t i = 0; i < gpu->gpu_cutoff->sqrQshell; i++) {
        uint32_t kprim1
            = gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x]];
        uint32_t kprim2
            = gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y]];
        uint32_t kprim_score = 10 * std::max(kprim1, kprim2) + std::min(kprim1, kprim2) + (kprim1 + kprim2);
        partial_eris[i] = {gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x,
            gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y,
            gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x],
            gpu->gpu_basis->sorted_Qnumber->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y],
            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x]],
            gpu->gpu_basis->kprim->_hostData[gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y]],
            gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x],
            gpu->gpu_basis->sorted_Q->_hostData[gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y],
            kprim_score};
    }

    for (uint32_t i = 0; i < 16; i++) {
        std::sort(partial_eris + eri_type_block_map[i], partial_eris + eri_type_block_map[i + 1], ComparePrimNum);
    }

    for (uint32_t i = 0; i < gpu->gpu_cutoff->sqrQshell; i++) {
        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].x = partial_eris[i].YCutoffIJ_x;
        gpu->gpu_cutoff->sorted_YCutoffIJ->_hostData[i].y = partial_eris[i].YCutoffIJ_y;
    }

    gpu->gpu_cutoff->sorted_YCutoffIJ->Upload();
    gpu->gpu_sim.sorted_YCutoffIJ = gpu->gpu_cutoff->sorted_YCutoffIJ->_devData;
    gpu->gpu_sim.ffStart = ffStart;

    free(resorted_YCutoffIJ);
    free(partial_eris);
}


void getGrad_ffff(_gpu_type gpu)
{
    ResortERIs(gpu);

    uint8_t trans[TRANSDIM * TRANSDIM * TRANSDIM] = {};
    uint32_t *uint32_buffer = (uint32_t *) malloc(sizeof(uint32_t) * ERI_GRAD_FFFF_SMEM_UINT32_SIZE * ERI_GRAD_FFFF_TPB);
    uint32_t **uint32_ptr_buffer = (uint32_t **) malloc(sizeof(uint32_t *) * ERI_GRAD_FFFF_SMEM_UINT32_PTR_SIZE * ERI_GRAD_FFFF_TPB);
    QUICKDouble *dbl_buffer = (QUICKDouble*) malloc(sizeof(QUICKDouble) * ERI_GRAD_FFFF_SMEM_DBL_SIZE*ERI_GRAD_FFFF_TPB);
    QUICKDouble **dbl_ptr_buffer = (QUICKDouble **) malloc(sizeof(QUICKDouble *) * ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE * ERI_GRAD_FFFF_TPB);
    int2 **int2_ptr_buffer = (int2 **) malloc(sizeof(int2 *) * ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE * ERI_GRAD_FFFF_TPB);
    unsigned char **char_ptr_buffer = (unsigned char **) malloc(sizeof(unsigned char *) * ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE * ERI_GRAD_FFFF_TPB);
    QUICKAtomicType **grad_ptr_buffer = (QUICKAtomicType **) malloc(sizeof(QUICKAtomicType *) * ERI_GRAD_FFFF_SMEM_PTR_SIZE * ERI_GRAD_FFFF_TPB);

    for (uint32_t i = 0; i < ERI_GRAD_FFFF_TPB; i++) {
        uint32_buffer[i] = gpu->gpu_sim.natom;
        uint32_buffer[ERI_GRAD_FFFF_TPB + i] = gpu->gpu_sim.nbasis;
        uint32_buffer[ERI_GRAD_FFFF_TPB * 2 + i] = gpu->gpu_sim.nshell;
        uint32_buffer[ERI_GRAD_FFFF_TPB * 3 + i] = gpu->gpu_sim.jbasis;
        uint32_buffer[ERI_GRAD_FFFF_TPB * 4 + i] = gpu->gpu_sim.prim_total;
    }

    for (uint32_t i = 0; i < ERI_GRAD_FFFF_TPB; i++) {
        uint32_ptr_buffer[i] = gpu->gpu_sim.katom;
        uint32_ptr_buffer[ERI_GRAD_FFFF_TPB + i] = gpu->gpu_sim.kprim;
        uint32_ptr_buffer[ERI_GRAD_FFFF_TPB * 2 + i] = gpu->gpu_sim.kstart;
        uint32_ptr_buffer[ERI_GRAD_FFFF_TPB * 3 + i] = gpu->gpu_sim.Ksumtype;
        uint32_ptr_buffer[ERI_GRAD_FFFF_TPB * 4 + i] = gpu->gpu_sim.prim_start;
        uint32_ptr_buffer[ERI_GRAD_FFFF_TPB * 5 + i] = gpu->gpu_sim.Qfbasis;
        uint32_ptr_buffer[ERI_GRAD_FFFF_TPB * 6 + i] = gpu->gpu_sim.Qsbasis;
        uint32_ptr_buffer[ERI_GRAD_FFFF_TPB * 7 + i] = gpu->gpu_sim.Qstart;
        uint32_ptr_buffer[ERI_GRAD_FFFF_TPB * 8 + i] = gpu->gpu_sim.sorted_Q;
        uint32_ptr_buffer[ERI_GRAD_FFFF_TPB * 9 + i] = gpu->gpu_sim.sorted_Qnumber;
        uint32_ptr_buffer[ERI_GRAD_FFFF_TPB * 10 + i] = gpu->gpu_sim.KLMN;
    }

    for (uint32_t i = 0; i < ERI_GRAD_FFFF_TPB; i++) {
        dbl_buffer[i] = gpu->gpu_sim.primLimit;
        dbl_buffer[ERI_GRAD_FFFF_TPB + i] = gpu->gpu_sim.gradCutoff;
        dbl_buffer[ERI_GRAD_FFFF_TPB * 2 + i] = gpu->gpu_sim.hyb_coeff;
    }

    for (uint32_t i = 0; i < ERI_GRAD_FFFF_TPB; i++) {
        dbl_ptr_buffer[i] = gpu->gpu_sim.cons;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB + i] = gpu->gpu_sim.cutMatrix;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 2 + i] = gpu->gpu_sim.cutPrim;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 3 + i] = gpu->gpu_sim.dense;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 4 + i] = gpu->gpu_sim.denseb;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 5 + i] = gpu->gpu_sim.expoSum;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 6 + i] = gpu->gpu_sim.gcexpo;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 7 + i] = gpu->gpu_sim.store;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 8 + i] = gpu->gpu_sim.store2;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 9 + i] = gpu->gpu_sim.storeAA;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 10 + i] = gpu->gpu_sim.storeBB;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 11 + i] = gpu->gpu_sim.storeCC;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 12 + i] = gpu->gpu_sim.weightedCenterX;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 13 + i] = gpu->gpu_sim.weightedCenterY;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 14 + i] = gpu->gpu_sim.weightedCenterZ;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 15 + i] = gpu->gpu_sim.Xcoeff;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 16 + i] = gpu->gpu_sim.xyz;
        dbl_ptr_buffer[ERI_GRAD_FFFF_TPB * 17 + i] = gpu->gpu_sim.YCutoff;
    }

    for (uint32_t i = 0; i < ERI_GRAD_FFFF_TPB; i++) {
        int2_ptr_buffer[i] = gpu->gpu_sim.sorted_YCutoffIJ;
    }

    for (uint32_t i = 0; i < ERI_GRAD_FFFF_TPB; i++) {
        char_ptr_buffer[i] = gpu->gpu_sim.mpi_bcompute;
    }

    for (uint32_t i = 0; i < ERI_GRAD_FFFF_TPB; i++) {
#if defined(USE_LEGACY_ATOMICS)
        grad_ptr_buffer[i] = gpu->gpu_sim.gradULL;
#else
        grad_ptr_buffer[i] = gpu->gpu_sim.grad;
#endif
    }

    LOC3(trans, 0, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 0;
    LOC3(trans, 0, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 3;
    LOC3(trans, 0, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 9;
    LOC3(trans, 0, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 19;
    LOC3(trans, 0, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 34;
    LOC3(trans, 0, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 55;
    LOC3(trans, 0, 0, 6, TRANSDIM, TRANSDIM, TRANSDIM) = 83;
    LOC3(trans, 0, 0, 7, TRANSDIM, TRANSDIM, TRANSDIM) = 119;
    LOC3(trans, 0, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 2;
    LOC3(trans, 0, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 5;
    LOC3(trans, 0, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 16;
    LOC3(trans, 0, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 31;
    LOC3(trans, 0, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 47;
    LOC3(trans, 0, 1, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 66;
    LOC3(trans, 0, 1, 6, TRANSDIM, TRANSDIM, TRANSDIM) = 99;
    LOC3(trans, 0, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 8;
    LOC3(trans, 0, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 15;
    LOC3(trans, 0, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 22;
    LOC3(trans, 0, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 41;
    LOC3(trans, 0, 2, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 72;
    LOC3(trans, 0, 2, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 105;
    LOC3(trans, 0, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 18;
    LOC3(trans, 0, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 30;
    LOC3(trans, 0, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 42;
    LOC3(trans, 0, 3, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 78;
    LOC3(trans, 0, 3, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 111;
    LOC3(trans, 0, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 33;
    LOC3(trans, 0, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 48;
    LOC3(trans, 0, 4, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 73;
    LOC3(trans, 0, 4, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 112;
    LOC3(trans, 0, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 54;
    LOC3(trans, 0, 5, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 67;
    LOC3(trans, 0, 5, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 106;
    LOC3(trans, 0, 6, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 82;
    LOC3(trans, 0, 6, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 100;
    LOC3(trans, 0, 7, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 118;
    LOC3(trans, 1, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 1;
    LOC3(trans, 1, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 6;
    LOC3(trans, 1, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 14;
    LOC3(trans, 1, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 27;
    LOC3(trans, 1, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 49;
    LOC3(trans, 1, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 68;
    LOC3(trans, 1, 0, 6, TRANSDIM, TRANSDIM, TRANSDIM) = 101;
    LOC3(trans, 1, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 4;
    LOC3(trans, 1, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 10;
    LOC3(trans, 1, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 25;
    LOC3(trans, 1, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 40;
    LOC3(trans, 1, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 58;
    LOC3(trans, 1, 1, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 86;
    LOC3(trans, 1, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 12;
    LOC3(trans, 1, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 24;
    LOC3(trans, 1, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 35;
    LOC3(trans, 1, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 59;
    LOC3(trans, 1, 2, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 87;
    LOC3(trans, 1, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 29;
    LOC3(trans, 1, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 39;
    LOC3(trans, 1, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 60;
    LOC3(trans, 1, 3, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 93;
    LOC3(trans, 1, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 51;
    LOC3(trans, 1, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 57;
    LOC3(trans, 1, 4, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 88;
    LOC3(trans, 1, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 70;
    LOC3(trans, 1, 5, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 85;
    LOC3(trans, 1, 6, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 103;
    LOC3(trans, 2, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 7;
    LOC3(trans, 2, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 13;
    LOC3(trans, 2, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 21;
    LOC3(trans, 2, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 43;
    LOC3(trans, 2, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 74;
    LOC3(trans, 2, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 109;
    LOC3(trans, 2, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 11;
    LOC3(trans, 2, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 23;
    LOC3(trans, 2, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 36;
    LOC3(trans, 2, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 61;
    LOC3(trans, 2, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 89;
    LOC3(trans, 2, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 20;
    LOC3(trans, 2, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 37;
    LOC3(trans, 2, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 65;
    LOC3(trans, 2, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 98;
    LOC3(trans, 2, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 45;
    LOC3(trans, 2, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 63;
    LOC3(trans, 2, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 97;
    LOC3(trans, 2, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 76;
    LOC3(trans, 2, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 91;
    LOC3(trans, 2, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 109;
    LOC3(trans, 3, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 17;
    LOC3(trans, 3, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 26;
    LOC3(trans, 3, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 44;
    LOC3(trans, 3, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 79;
    LOC3(trans, 3, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 113;
    LOC3(trans, 3, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 28;
    LOC3(trans, 3, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 38;
    LOC3(trans, 3, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 62;
    LOC3(trans, 3, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 94;
    LOC3(trans, 3, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 46;
    LOC3(trans, 3, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 64;
    LOC3(trans, 3, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 96;
    LOC3(trans, 3, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 80;
    LOC3(trans, 3, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 95;
    LOC3(trans, 3, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 115;
    LOC3(trans, 4, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 32;
    LOC3(trans, 4, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 50;
    LOC3(trans, 4, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 75;
    LOC3(trans, 4, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 114;
    LOC3(trans, 4, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 52;
    LOC3(trans, 4, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 56;
    LOC3(trans, 4, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 90;
    LOC3(trans, 4, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 77;
    LOC3(trans, 4, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 92;
    LOC3(trans, 4, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 116;
    LOC3(trans, 5, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 53;
    LOC3(trans, 5, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 69;
    LOC3(trans, 5, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 108;
    LOC3(trans, 5, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 71;
    LOC3(trans, 5, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 84;
    LOC3(trans, 5, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 110;
    LOC3(trans, 6, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 81;
    LOC3(trans, 6, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 102;
    LOC3(trans, 6, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 104;
    LOC3(trans, 7, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 117;

    uint8_t *dev_uint8_buffer;
    uint32_t *dev_uint32_buffer;
    uint32_t **dev_uint32_ptr_buffer;
    QUICKDouble *dev_dbl_buffer;
    QUICKDouble **dev_dbl_ptr_buffer;
    int2 **dev_int2_ptr_buffer;
    unsigned char **dev_char_ptr_buffer;
    QUICKAtomicType **dev_grad_ptr_buffer;

    gpuMalloc((void **) &dev_uint8_buffer, sizeof(uint8_t) * ERI_GRAD_FFFF_SMEM_UINT8_SIZE);
    gpuMalloc((void **) &dev_uint32_buffer, sizeof(uint32_t) * ERI_GRAD_FFFF_SMEM_UINT32_SIZE * ERI_GRAD_FFFF_TPB);
    gpuMalloc((void **) &dev_uint32_ptr_buffer, sizeof(uint32_t *) * ERI_GRAD_FFFF_SMEM_UINT32_PTR_SIZE * ERI_GRAD_FFFF_TPB);
    gpuMalloc((void **) &dev_dbl_buffer, sizeof(QUICKDouble) * ERI_GRAD_FFFF_SMEM_DBL_SIZE * ERI_GRAD_FFFF_TPB);
    gpuMalloc((void **) &dev_dbl_ptr_buffer, sizeof(QUICKDouble *) * ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE * ERI_GRAD_FFFF_TPB);
    gpuMalloc((void **) &dev_int2_ptr_buffer, sizeof(int2 *) * ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE * ERI_GRAD_FFFF_TPB);
    gpuMalloc((void **) &dev_char_ptr_buffer, sizeof(unsigned char *) * ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE * ERI_GRAD_FFFF_TPB);
    gpuMalloc((void **) &dev_grad_ptr_buffer, sizeof(QUICKAtomicType *) * ERI_GRAD_FFFF_SMEM_PTR_SIZE * ERI_GRAD_FFFF_TPB);

    gpuMemcpy(dev_uint8_buffer, &trans, sizeof(uint8_t) * ERI_GRAD_FFFF_SMEM_UINT8_SIZE, hipMemcpyHostToDevice);
    gpuMemcpy(dev_uint32_buffer, uint32_buffer, sizeof(uint32_t) * ERI_GRAD_FFFF_SMEM_UINT32_SIZE * ERI_GRAD_FFFF_TPB, hipMemcpyHostToDevice);
    gpuMemcpy(dev_uint32_ptr_buffer, uint32_ptr_buffer, sizeof(uint32_t *) * ERI_GRAD_FFFF_SMEM_UINT32_PTR_SIZE * ERI_GRAD_FFFF_TPB, hipMemcpyHostToDevice);
    gpuMemcpy(dev_dbl_buffer, dbl_buffer, sizeof(QUICKDouble) * ERI_GRAD_FFFF_SMEM_DBL_SIZE * ERI_GRAD_FFFF_TPB, hipMemcpyHostToDevice);
    gpuMemcpy(dev_dbl_ptr_buffer, dbl_ptr_buffer, sizeof(QUICKDouble *) * ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE * ERI_GRAD_FFFF_TPB, hipMemcpyHostToDevice);
    gpuMemcpy(dev_int2_ptr_buffer, int2_ptr_buffer, sizeof(int2 *) * ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE * ERI_GRAD_FFFF_TPB, hipMemcpyHostToDevice);
    gpuMemcpy(dev_char_ptr_buffer, char_ptr_buffer, sizeof(unsigned char *) * ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE * ERI_GRAD_FFFF_TPB, hipMemcpyHostToDevice);
    gpuMemcpy(dev_grad_ptr_buffer, grad_ptr_buffer, sizeof(QUICKAtomicType *) * ERI_GRAD_FFFF_SMEM_PTR_SIZE * ERI_GRAD_FFFF_TPB, hipMemcpyHostToDevice);

    // Part f-3
    if (gpu->maxL >= 3) {
#ifdef GPU_SPDF
        QUICK_SAFE_CALL((getGrad_kernel_ffff <<<gpu->blocks * ERI_GRAD_FFFF_BPSM, ERI_GRAD_FFFF_TPB,
                    (sizeof(uint8_t) * ERI_GRAD_FFFF_SMEM_UINT8_SIZE
                     + sizeof(uint32_t) * ERI_GRAD_FFFF_SMEM_UINT32_SIZE
                     + sizeof(QUICKDouble) * ERI_GRAD_FFFF_SMEM_DBL_SIZE
                     + sizeof(QUICKDouble *) * ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE
                     + sizeof(uint32_t *) * ERI_GRAD_FFFF_SMEM_UINT32_PTR_SIZE
                     + sizeof(int2 *) * ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE
                     + sizeof(unsigned char *) * ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE
                     + sizeof(QUICKAtomicType *) * ERI_GRAD_FFFF_SMEM_PTR_SIZE) * ERI_GRAD_FFFF_TPB>>>
                    (dev_uint8_buffer, dev_uint8_ptr_buffer, dev_uint32_buffer, dev_uint32_ptr_buffer, dev_dbl_buffer,
                     dev_dbl_ptr_buffer, dev_int2_ptr_buffer, dev_char_ptr_buffer, dev_grad_ptr_buffer,
                     gpu->gpu_sim.ffStart, gpu->gpu_sim.sqrQshell)));

#endif
    }

    free(uint32_buffer);
    free(uint32_ptr_buffer);
    free(dbl_buffer);
    free(dbl_ptr_buffer);
    free(int2_ptr_buffer);
    free(char_ptr_buffer);
    free(grad_ptr_buffer);

    gpuFree(dev_uint8_buffer);
    gpuFree(dev_uint32_buffer);
    gpuFree(dev_uint32_ptr_buffer);
    gpuFree(dev_dbl_buffer);
    gpuFree(dev_dbl_ptr_buffer);
    gpuFree(dev_int2_ptr_buffer);
    gpuFree(dev_char_ptr_buffer);
    gpuFree(dev_grad_ptr_buffer);
}


// interface to call uscf gradient Kernels
void get_oshell_eri_grad_ffff(_gpu_type gpu)
{
//   nvtxRangePushA("Gradient 2e");

    // compute one electron gradients in the meantime
//    get_oneen_grad_();

    // Part f-3
//    if (gpu->maxL >= 3) {
//        QUICK_SAFE_CALL((getGrad_oshell_kernel_ffff <<<gpu->blocks, gpu->gradThreadsPerBlock>>> ()))
//#endif
//    }

//    nvtxRangePop();
}


void upload_para_to_const_ffff()
{
    uint8_t trans[TRANSDIM * TRANSDIM * TRANSDIM] = {};

    LOC3(trans, 0, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 0;
    LOC3(trans, 0, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 3;
    LOC3(trans, 0, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 9;
    LOC3(trans, 0, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 19;
    LOC3(trans, 0, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 34;
    LOC3(trans, 0, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 55;
    LOC3(trans, 0, 0, 6, TRANSDIM, TRANSDIM, TRANSDIM) = 83;
    LOC3(trans, 0, 0, 7, TRANSDIM, TRANSDIM, TRANSDIM) = 119;
    LOC3(trans, 0, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 2;
    LOC3(trans, 0, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 5;
    LOC3(trans, 0, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 16;
    LOC3(trans, 0, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 31;
    LOC3(trans, 0, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 47;
    LOC3(trans, 0, 1, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 66;
    LOC3(trans, 0, 1, 6, TRANSDIM, TRANSDIM, TRANSDIM) = 99;
    LOC3(trans, 0, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 8;
    LOC3(trans, 0, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 15;
    LOC3(trans, 0, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 22;
    LOC3(trans, 0, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 41;
    LOC3(trans, 0, 2, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 72;
    LOC3(trans, 0, 2, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 105;
    LOC3(trans, 0, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 18;
    LOC3(trans, 0, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 30;
    LOC3(trans, 0, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 42;
    LOC3(trans, 0, 3, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 78;
    LOC3(trans, 0, 3, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 111;
    LOC3(trans, 0, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 33;
    LOC3(trans, 0, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 48;
    LOC3(trans, 0, 4, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 73;
    LOC3(trans, 0, 4, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 112;
    LOC3(trans, 0, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 54;
    LOC3(trans, 0, 5, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 67;
    LOC3(trans, 0, 5, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 106;
    LOC3(trans, 0, 6, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 82;
    LOC3(trans, 0, 6, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 100;
    LOC3(trans, 0, 7, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 118;
    LOC3(trans, 1, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 1;
    LOC3(trans, 1, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 6;
    LOC3(trans, 1, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 14;
    LOC3(trans, 1, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 27;
    LOC3(trans, 1, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 49;
    LOC3(trans, 1, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 68;
    LOC3(trans, 1, 0, 6, TRANSDIM, TRANSDIM, TRANSDIM) = 101;
    LOC3(trans, 1, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 4;
    LOC3(trans, 1, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 10;
    LOC3(trans, 1, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 25;
    LOC3(trans, 1, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 40;
    LOC3(trans, 1, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 58;
    LOC3(trans, 1, 1, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 86;
    LOC3(trans, 1, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 12;
    LOC3(trans, 1, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 24;
    LOC3(trans, 1, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 35;
    LOC3(trans, 1, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 59;
    LOC3(trans, 1, 2, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 87;
    LOC3(trans, 1, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 29;
    LOC3(trans, 1, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 39;
    LOC3(trans, 1, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 60;
    LOC3(trans, 1, 3, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 93;
    LOC3(trans, 1, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 51;
    LOC3(trans, 1, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 57;
    LOC3(trans, 1, 4, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 88;
    LOC3(trans, 1, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 70;
    LOC3(trans, 1, 5, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 85;
    LOC3(trans, 1, 6, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 103;
    LOC3(trans, 2, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 7;
    LOC3(trans, 2, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 13;
    LOC3(trans, 2, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 21;
    LOC3(trans, 2, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 43;
    LOC3(trans, 2, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 74;
    LOC3(trans, 2, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 109;
    LOC3(trans, 2, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 11;
    LOC3(trans, 2, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 23;
    LOC3(trans, 2, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 36;
    LOC3(trans, 2, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 61;
    LOC3(trans, 2, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 89;
    LOC3(trans, 2, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 20;
    LOC3(trans, 2, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 37;
    LOC3(trans, 2, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 65;
    LOC3(trans, 2, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 98;
    LOC3(trans, 2, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 45;
    LOC3(trans, 2, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 63;
    LOC3(trans, 2, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 97;
    LOC3(trans, 2, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 76;
    LOC3(trans, 2, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 91;
    LOC3(trans, 2, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 109;
    LOC3(trans, 3, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 17;
    LOC3(trans, 3, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 26;
    LOC3(trans, 3, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 44;
    LOC3(trans, 3, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 79;
    LOC3(trans, 3, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 113;
    LOC3(trans, 3, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 28;
    LOC3(trans, 3, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 38;
    LOC3(trans, 3, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 62;
    LOC3(trans, 3, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 94;
    LOC3(trans, 3, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 46;
    LOC3(trans, 3, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 64;
    LOC3(trans, 3, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 96;
    LOC3(trans, 3, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 80;
    LOC3(trans, 3, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 95;
    LOC3(trans, 3, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 115;
    LOC3(trans, 4, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 32;
    LOC3(trans, 4, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 50;
    LOC3(trans, 4, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 75;
    LOC3(trans, 4, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 114;
    LOC3(trans, 4, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 52;
    LOC3(trans, 4, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 56;
    LOC3(trans, 4, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 90;
    LOC3(trans, 4, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 77;
    LOC3(trans, 4, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 92;
    LOC3(trans, 4, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 116;
    LOC3(trans, 5, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 53;
    LOC3(trans, 5, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 69;
    LOC3(trans, 5, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 108;
    LOC3(trans, 5, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 71;
    LOC3(trans, 5, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 84;
    LOC3(trans, 5, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 110;
    LOC3(trans, 6, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 81;
    LOC3(trans, 6, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 102;
    LOC3(trans, 6, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 104;
    LOC3(trans, 7, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 117;

    gpuMemcpyToSymbol((const void *) devTrans, (const void *) trans,
            sizeof(uint8_t) * TRANSDIM * TRANSDIM * TRANSDIM);
}
