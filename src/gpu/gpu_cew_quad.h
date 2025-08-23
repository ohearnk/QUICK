/*
   !---------------------------------------------------------------------!
   ! Written by Madu Manathunga on 09/29/2021                            !
   !                                                                     !
   ! Copyright (C) 2020-2021 Merz lab                                    !
   ! Copyright (C) 2020-2021 Götz lab                                    !
   !                                                                     !
   ! This Source Code Form is subject to the terms of the Mozilla Public !
   ! License, v. 2.0. If a copy of the MPL was not distributed with this !
   ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
   !_____________________________________________________________________!

   !---------------------------------------------------------------------!
   ! This source file contains preprocessable functions required for     !
   ! QUICK GPU version.                                                  !
   !---------------------------------------------------------------------!
   */

#if !defined(OSHELL)
__global__ void k_getcew_quad(uint32_t nbasis, int npoints,
        QUICKDouble const * const gridx, QUICKDouble const * const gridy, QUICKDouble const * const gridz,
        QUICKDouble const * const weight, int const * const basf, int const * const primf,
        int const * const basf_locator, int const * const primf_locator, int const * const bin_locator,
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL * const oULL,
#else
        QUICKDouble * const o,
#endif
        QUICKDouble DMCutoff, QUICKDouble const * const cew_vrecip)
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (QUICKULL gid = offset; gid < npoints; gid += totalThreads) {
        int bin_id = bin_locator[gid];
        int bfloc_st = basf_locator[bin_id];
        int bfloc_end = basf_locator[bin_id + 1];

        QUICKDouble gridx = gridx[gid];
        QUICKDouble gridy = gridy[gid];
        QUICKDouble gridz = gridz[gid];

        QUICKDouble weight = weight[gid];

        QUICKDouble dfdr = cew_vrecip[gid];

        for (int i = bfloc_st; i < bfloc_end; ++i) {
            int ibas = basf[i];
            QUICKDouble phi, dphidx, dphidy, dphidz;

            pteval_new(gridx, gridy, gridz,
                    &phi, &dphidx, &dphidy, &dphidz,
                    primf, primf_locator, ibas, i);

            if (abs(phi + dphidx + dphidy + dphidz) > DMCutoff) {
                for (int j = bfloc_st; j < bfloc_end; j++) {
                    int jbas = basf[j];
                    QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

                    pteval_new(gridx, gridy, gridz,
                            &phi2, &dphidx2, &dphidy2, &dphidz2,
                            primf, primf_locator, jbas, j);

                    QUICKDouble _tmp = phi * phi2 * dfdr * weight;

#if defined(USE_LEGACY_ATOMICS)
                    GPUATOMICADD(&LOC2(oULL, jbas, ibas, nbasis, nbasis), _tmp, OSCALE);
#else
                    atomicAdd(&LOC2(o, jbas, ibas, nbasis, nbasis), _tmp);
#endif
                }
            }
        }
    }
}
#endif


#if defined(OSHELL)
__global__ void k_getcew_quad_grad_oshell
#else
__global__ void k_getcew_quad_grad_cshell
#endif
    (uint32_t natom, uint32_t nbasis, int npoints,
     QUICKDouble const * const gridx, QUICKDouble const * const gridy, QUICKDouble const * const gridz,
     QUICKDouble const * const sswt, QUICKDouble const * const weight, QUICKDouble const * const densa,
#if defined(OSHELL)
     QUICKDouble const * const densb,
#endif
     QUICKDouble * const exc, int const * const gatm, int * const dweight_ssd,
     int const * const basf, int const * const primf,
     int const * const basf_locator, int const * const primf_locator, int const * const bin_locator,
     int const * const ncenter,
     QUICKDouble const * const dense,
#if defined(OSHELL)
     QUICKDouble const * const denseb,
#endif
     QUICKDouble DMCutoff, 
#if defined(USE_LEGACY_ATOMICS)
     QUICKULL * const gradULL,
#else
     QUICKDouble * const grad,
#endif
     QUICKDouble const * const cew_vrecip)

{
#if defined(USE_LEGACY_ATOMICS)
    //declare smem grad vector
    extern __shared__ QUICKULL smem_buffer[];
    QUICKULL* smemGrad = (QUICKULL *) smem_buffer;

    // initialize smem grad
    for (int i = threadIdx.x; i < natom * 3; i += blockDim.x) {
        smemGrad[i] = 0ull;
    }
#else
    //declare smem grad vector
    extern __shared__ QUICKDouble smem_buffer[];
    QUICKDouble* smemGrad = (QUICKDouble *) smem_buffer;

    // initialize smem grad
    for (int i = threadIdx.x; i < natom * 3; i += blockDim.x) {
        smemGrad[i] = 0.0;
    }
#endif

    __syncthreads();

    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (QUICKULL gid = offset; gid < npoints; gid += totalThreads) {
        int bin_id = bin_locator[gid];
        int bfloc_st = basf_locator[bin_id];
        int bfloc_end = basf_locator[bin_id + 1];

        QUICKDouble gridx = gridx[gid];
        QUICKDouble gridy = gridy[gid];
        QUICKDouble gridz = gridz[gid];
        QUICKDouble weight = weight[gid];
#if defined(OSHELL)
        QUICKDouble densitysum = densa[gid] + densb[gid];
#else
        QUICKDouble densitysum = 2 * densa[gid];
#endif

        QUICKDouble dfdr = cew_vrecip[gid];

        if (densitysum > DMCutoff) {
            QUICKDouble _tmp = (QUICKDouble) (dfdr * densitysum);

            exc[gid] = _tmp;

            QUICKDouble sumGradx = 0.0;
            QUICKDouble sumGrady = 0.0;
            QUICKDouble sumGradz = 0.0;

            for (int i = bfloc_st; i < bfloc_end; i++) {
                int ibas = basf[i];
                QUICKDouble phi, dphidx, dphidy, dphidz;
                pteval_new(gridx, gridy, gridz,
                        &phi, &dphidx, &dphidy, &dphidz,
                        primf, primf_locator, ibas, i);

                if (abs(phi + dphidx + dphidy + dphidz) > DMCutoff) {
                    //QUICKDouble dxdx, dxdy, dxdz, dydy, dydz, dzdz;
                    //pt2der_new(gridx, gridy, gridz, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz, primf, primf_locator, ibas, i);

                    int Istart = (ncenter[ibas] - 1) * 3;

                    for (int j = bfloc_st; j < bfloc_end; j++) {
                        int jbas = basf[j];
                        QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

                        pteval_new(gridx, gridy, gridz,
                                &phi2, &dphidx2, &dphidy2, &dphidz2,
                                primf, primf_locator, jbas, j);

                        QUICKDouble denseij = (QUICKDouble) LOC2(dense, ibas, jbas, nbasis, nbasis);

#if defined(OSHELL)
                        denseij += (QUICKDouble) LOC2(denseb, ibas, jbas, nbasis, nbasis);
#endif

                        QUICKDouble Gradx = -2.0 * denseij * weight * (dfdr * dphidx * phi2);
                        QUICKDouble Grady = -2.0 * denseij * weight * (dfdr * dphidy * phi2);
                        QUICKDouble Gradz = -2.0 * denseij * weight * (dfdr * dphidz * phi2);
                        //printf("test quad grad %f %f %f %f %f %f %f %f %f %f\n", gridx, gridy, gridz, denseij, weight, dfdr, dphidx, dphidy, dphidz, phi2);

                        GPUATOMICADD(&smemGrad[Istart], Gradx, GRADSCALE);
                        GPUATOMICADD(&smemGrad[Istart + 1], Grady, GRADSCALE);
                        GPUATOMICADD(&smemGrad[Istart + 2], Gradz, GRADSCALE);
                        sumGradx += Gradx;
                        sumGrady += Grady;
                        sumGradz += Gradz;
                    }
                }
            }

            int Istart = (gatm[gid] - 1) * 3;

            GPUATOMICADD(&smemGrad[Istart], -sumGradx, GRADSCALE);
            GPUATOMICADD(&smemGrad[Istart + 1], -sumGrady, GRADSCALE);
            GPUATOMICADD(&smemGrad[Istart + 2], -sumGradz, GRADSCALE);
        }

        //Set weights for sswder calculation
        if (densitysum < DMCutoff) {
            dweight_ssd[gid] = 0;
        }

        if (sswt[gid] == 1) {
            dweight_ssd[gid] = 0;
        }
    }

    __syncthreads();

    // update gmem grad vector
    for (int i = threadIdx.x; i < natom * 3; i += blockDim.x) {
#if defined(USE_LEGACY_ATOMICS)
        atomicAdd(&gradULL[i], smemGrad[i]);
#else
        atomicAdd(&grad[i], smemGrad[i]);
#endif
    }

    __syncthreads();
}
