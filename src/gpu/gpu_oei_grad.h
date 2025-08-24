/*
   !---------------------------------------------------------------------!
   ! Written by Madu Manathunga on 07/29/2021                            !
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

#if !defined(__QUICK_GPU_OEI_GRAD_H_)
#define __QUICK_GPU_OEI_GRAD_H_

#undef VY
#define VY(a,b,c) (YVerticalTemp[(c)])

// support up to d functions (refactor if OEI f func support added and/or specialized for sp, spd, spdf, etc.)
#define PRIM_INT_OEI_GRAD_LEN (7)


__device__ static inline void iclass_oei_grad(uint8_t I, uint8_t J, uint32_t II, uint32_t JJ,
        uint32_t iatom, bool is_oshell, uint32_t natom, uint32_t nextatom, uint32_t nbasis,
        uint32_t nshell, uint32_t jbasis,
        QUICKDouble const * const allchg, QUICKDouble const * const allxyz,
        uint32_t const * const kstart, uint32_t const * const katom,
        uint32_t const * const kprim, uint32_t const * const Ksumtype, uint32_t const * const Qstart,
        uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
        QUICKDouble const * const cons, QUICKDouble const * const gcexpo, uint8_t const * const KLMN,
        uint32_t prim_total, uint32_t const * const prim_start,
        QUICKDouble const * const dense, QUICKDouble const * const denseb, 
        QUICKDouble const * const Xcoeff_oei, QUICKDouble const * const expoSum,
        QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
        QUICKDouble const * const weightedCenterZ, QUICKDouble coreIntegralCutoff,
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL * const gradULL, QUICKULL * const ptchg_gradULL,
#else
        QUICKDouble * const grad, QUICKDouble * const ptchg_grad,
#endif
        QUICKDouble * const store, QUICKDouble * const store2,
        QUICKDouble * const storeAA, QUICKDouble * const storeBB)
{
    /*
       kAtom A, B  is the coresponding atom for shell II, JJ
       and be careful with the index difference between Fortran and C++,
       Fortran starts array index with 1 and C++ starts 0.
       Ai, Bi, Ci are the coordinates for atom katomA, katomB, katomC,
       which means they are corrosponding coorinates for shell II, JJ and nuclei.
    */
    const QUICKDouble Ax = LOC2(allxyz, 0, katom[II], 3, natom + nextatom);
    const QUICKDouble Ay = LOC2(allxyz, 1, katom[II], 3, natom + nextatom);
    const QUICKDouble Az = LOC2(allxyz, 2, katom[II], 3, natom + nextatom);
    const QUICKDouble Bx = LOC2(allxyz, 0, katom[JJ], 3, natom + nextatom);
    const QUICKDouble By = LOC2(allxyz, 1, katom[JJ], 3, natom + nextatom);
    const QUICKDouble Bz = LOC2(allxyz, 2, katom[JJ], 3, natom + nextatom);

    /*
       kPrimI and kPrimJ indicates the number of primitives in shell II and JJ.
       kStartI, J indicates the starting guassian function for shell II, JJ.
       We retrieve from global memory and save them to register to avoid multiple retrieve.
    */
    const uint32_t kPrimI = kprim[II];
    const uint32_t kPrimJ = kprim[JJ];
    const uint32_t kStartI = kstart[II];
    const uint32_t kStartJ = kstart[JJ];

    /*
       At this point, we will need 3 arrays. The first, store2, will keep the sum of primitive integral
       values as in oei code. Gradient calculation also requires scaling certain primitive integral values
       by the exponents on each center. It is possible to eliminate the second and third arrays, by looping
       through the primitives and updating the grad vector during each cycle. But this incurs a huge performance
       penalty.
    */

    /*
       initialize the region of store2 array that we will be using. This region is determined by looking at the
       Sumindex array with angular momentums of the shells.
    */
    for (uint8_t i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint8_t i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
        for (uint8_t j = Sumindex[I + 1]; j < Sumindex[I + 3]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(&storeAA[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint8_t i = Sumindex[J + 1]; i < Sumindex[J + 3]; ++i) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(&storeBB[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint32_t i = 0; i < kPrimI * kPrimJ ; ++i) {
        const uint32_t JJJ = (uint32_t) i / kPrimI;
        const uint32_t III = (uint32_t) i - kPrimI * JJJ;

        /*
           In the following comments, we have I, J, K, L denote the primitive gaussian function we use, and
           for example, expo(III, ksumtype(II)) stands for the expo for the IIIth primitive guassian function for II shell,
           we use I to express the corresponding index.
           Zeta = expo(I)+expo(J)
           --->                --->
           ->     expo(I) * xyz (I) + expo(J) * xyz(J)
           P  = ---------------------------------------
           expo(I) + expo(J)
           Those two are pre-calculated in CPU stage.
        */

        uint32_t ii_start = prim_start[II];
        uint32_t jj_start = prim_start[JJ];

        const QUICKDouble Zeta = LOC2(expoSum, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        const QUICKDouble Px = LOC2(weightedCenterX, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        const QUICKDouble Py = LOC2(weightedCenterY, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        const QUICKDouble Pz = LOC2(weightedCenterZ, ii_start + III, jj_start + JJJ, prim_total, prim_total);

        const QUICKDouble AA = LOC2(gcexpo, III, Ksumtype[II], MAXPRIM, nbasis);
        const QUICKDouble BB = LOC2(gcexpo, JJJ, Ksumtype[JJ], MAXPRIM, nbasis);

        // get Xcoeff, which is a product of overlap prefactor and contraction coefficients
        const QUICKDouble Xcoeff = LOC4(Xcoeff_oei, kStartI + III, kStartJ + JJJ,
                I - Qstart[II], J - Qstart[JJ], jbasis, jbasis, 2, 2);

        if (abs(Xcoeff) > coreIntegralCutoff) {
            const QUICKDouble Cx = LOC2(allxyz, 0, iatom, 3, natom + nextatom);
            const QUICKDouble Cy = LOC2(allxyz, 1, iatom, 3, natom + nextatom);
            const QUICKDouble Cz = LOC2(allxyz, 2, iatom, 3, natom + nextatom);
            const QUICKDouble chg = -1.0 * allchg[iatom];

            // compute boys function values, the third term of OS A20
            double YVerticalTemp[PRIM_INT_OEI_GRAD_LEN];
            FmT(I + J + 2, Zeta * (SQR(Px - Cx) + SQR(Py - Cy) + SQR(Pz - Cz)),
                    YVerticalTemp);

            // compute all auxilary integrals and store
            for (uint32_t n = 0; n <= I + J + 2; n++) {
                VY(0, 0, n) = VY(0, 0, n) * Xcoeff * chg;
                //printf("aux: %d %f \n", i, VY(0, 0, i));
            }

            // decompose all attraction integrals to their auxilary integrals through VRR scheme.
            oei_grad_vertical(I, J, 
#if defined(DEBUG_OEI)
                    II, JJ,
#endif
                    Px - Ax, Py - Ay, Pz - Az,
                    Px - Bx, Py - By, Pz - Bz,
                    Px - Cx, Py - Cy, Pz - Cz,
                    1.0 / (2.0 * Zeta),
                    &store[blockIdx.x * blockDim.x + threadIdx.x],
                    YVerticalTemp);

            // sum up primitive integral values into store array
            for (uint8_t i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
                for (uint8_t j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
                    if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM)
                            += LOCSTORE(&store[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM);
                    }
                }
            }

            // scale primitive integral values with exponent of the first center and add up into storeAA
            for (uint8_t i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
                for (uint8_t j = Sumindex[I + 1]; j < Sumindex[I + 3]; ++j) {
                    if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(&storeAA[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM)
                            += LOCSTORE(&store[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM) * AA * 2.0;
                    }
                }
            }

            // scale primitive integral values with exponent of the second center and add up into storeBB
            for (uint8_t i = Sumindex[J + 1]; i < Sumindex[J + 3]; ++i) {
                for (uint8_t j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
                    if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(&storeBB[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM)
                            += LOCSTORE(&store[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM) * BB * 2.0;
                    }
                }
            }
        }
    }

    // retrive computed integral values from store array and update the Fock matrix
    
    // obtain the start and final basis function indices for given shells II and JJ. They will help us to save the integral
    // contribution into correct location in Fock matrix.
    const uint32_t III1 = LOC2(Qsbasis, II, I, nshell, 4);
    const uint32_t III2 = LOC2(Qfbasis, II, I, nshell, 4);
    const uint32_t JJJ1 = LOC2(Qsbasis, JJ, J, nshell, 4);
    const uint32_t JJJ2 = LOC2(Qfbasis, JJ, J, nshell, 4);

    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;

    for (uint32_t III = III1; III <= III2; III++) {
        const uint8_t i = LOC3(devTrans,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM);

        for (uint32_t JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            QUICKDouble DENSEJI = (QUICKDouble) LOC2(dense, JJJ, III, nbasis, nbasis);

            if (is_oshell)
                DENSEJI += (QUICKDouble) LOC2(denseb, JJJ, III, nbasis, nbasis);

            if (III != JJJ)
                DENSEJI *= 2.0;

            const QUICKDouble constant = cons[III] * cons[JJJ] * DENSEJI;

            // devTrans maps a basis function with certain angular momentum to store2 array. Get the correct indices now.
            uint8_t j = LOC3(devTrans,
                    LOC2(KLMN, 0, JJJ, 3, nbasis),
                    LOC2(KLMN, 1, JJJ, 3, nbasis),
                    LOC2(KLMN, 2, JJJ, 3, nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM);

            // sum up gradient wrt x-coordinate of first center
            uint8_t itemp = LOC3(devTrans,
                    LOC2(KLMN, 0, III, 3, nbasis) + 1,
                    LOC2(KLMN, 1, III, 3, nbasis),
                    LOC2(KLMN, 2, III, 3, nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM);

            AGradx += constant * LOCSTORE(&storeAA[blockIdx.x * blockDim.x + threadIdx.x], itemp, j, STOREDIM, STOREDIM);

            if (LOC2(KLMN, 0, III, 3, nbasis) >= 1) {
                itemp = LOC3(devTrans,
                        LOC2(KLMN, 0, III, 3, nbasis) - 1,
                        LOC2(KLMN, 1, III, 3, nbasis),
                        LOC2(KLMN, 2, III, 3, nbasis),
                        TRANSDIM, TRANSDIM, TRANSDIM);

                AGradx -= constant * LOC2(KLMN, 0, III, 3, nbasis)
                    * LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], itemp, j, STOREDIM, STOREDIM);
            }

            // sum up gradient wrt y-coordinate of first center
            itemp = LOC3(devTrans,
                    LOC2(KLMN, 0, III, 3, nbasis),
                    LOC2(KLMN, 1, III, 3, nbasis) + 1,
                    LOC2(KLMN, 2, III, 3, nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM);

            AGrady += constant * LOCSTORE(&storeAA[blockIdx.x * blockDim.x + threadIdx.x], itemp, j, STOREDIM, STOREDIM);

            if (LOC2(KLMN, 1, III, 3, nbasis) >= 1) {
                itemp = LOC3(devTrans,
                        LOC2(KLMN, 0, III, 3, nbasis),
                        LOC2(KLMN, 1, III, 3, nbasis) - 1,
                        LOC2(KLMN, 2, III, 3, nbasis),
                        TRANSDIM, TRANSDIM, TRANSDIM);

                AGrady -= constant * LOC2(KLMN, 1, III, 3, nbasis)
                    * LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], itemp, j, STOREDIM, STOREDIM);
            }

            // sum up gradient wrt z-coordinate of first center
            itemp = LOC3(devTrans,
                    LOC2(KLMN, 0, III, 3, nbasis),
                    LOC2(KLMN, 1, III, 3, nbasis),
                    LOC2(KLMN, 2, III, 3, nbasis) + 1,
                    TRANSDIM, TRANSDIM, TRANSDIM);

            AGradz += constant * LOCSTORE(&storeAA[blockIdx.x * blockDim.x + threadIdx.x], itemp, j, STOREDIM, STOREDIM);

            if (LOC2(KLMN, 2, III, 3, nbasis) >= 1) {
                itemp = LOC3(devTrans,
                        LOC2(KLMN, 0, III, 3, nbasis),
                        LOC2(KLMN, 1, III, 3, nbasis),
                        LOC2(KLMN, 2, III, 3, nbasis) - 1,
                        TRANSDIM, TRANSDIM, TRANSDIM);

                AGradz -= constant * LOC2(KLMN, 2, III, 3, nbasis)
                    * LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], itemp, j, STOREDIM, STOREDIM);
            }

            // sum up gradient wrt x-coordinate of second center
            j = LOC3(devTrans,
                    LOC2(KLMN, 0, JJJ, 3, nbasis) + 1,
                    LOC2(KLMN, 1, JJJ, 3, nbasis),
                    LOC2(KLMN, 2, JJJ, 3, nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM);

            BGradx += constant * LOCSTORE(&storeBB[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM);

            if (LOC2(KLMN, 0, JJJ, 3, nbasis) >= 1) {
                j = LOC3(devTrans,
                        LOC2(KLMN, 0, JJJ, 3, nbasis) - 1,
                        LOC2(KLMN, 1, JJJ, 3, nbasis),
                        LOC2(KLMN, 2, JJJ, 3, nbasis),
                        TRANSDIM, TRANSDIM, TRANSDIM);

                BGradx -= constant * LOC2(KLMN, 0, JJJ, 3, nbasis)
                    * LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM);
            }

            // sum up gradient wrt y-coordinate of second center
            j = LOC3(devTrans,
                    LOC2(KLMN, 0, JJJ, 3, nbasis),
                    LOC2(KLMN, 1, JJJ, 3, nbasis) + 1,
                    LOC2(KLMN, 2, JJJ, 3, nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM);

            BGrady += constant * LOCSTORE(&storeBB[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM);

            if (LOC2(KLMN, 1, JJJ, 3, nbasis) >= 1) {
                j = LOC3(devTrans,
                        LOC2(KLMN, 0, JJJ, 3, nbasis),
                        LOC2(KLMN, 1, JJJ, 3, nbasis) - 1,
                        LOC2(KLMN, 2, JJJ, 3, nbasis),
                        TRANSDIM, TRANSDIM, TRANSDIM);

                BGrady -= constant * LOC2(KLMN, 1, JJJ, 3, nbasis)
                    * LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM);
            }

            // sum up gradient wrt z-coordinate of second center
            j = LOC3(devTrans,
                    LOC2(KLMN, 0, JJJ, 3, nbasis),
                    LOC2(KLMN, 1, JJJ, 3, nbasis),
                    LOC2(KLMN, 2, JJJ, 3, nbasis) + 1,
                    TRANSDIM, TRANSDIM, TRANSDIM);

            BGradz += constant * LOCSTORE(&storeBB[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM);

            if (LOC2(KLMN, 2, JJJ, 3, nbasis) >= 1) {
                j = LOC3(devTrans,
                        LOC2(KLMN, 0, JJJ, 3, nbasis),
                        LOC2(KLMN, 1, JJJ, 3, nbasis),
                        LOC2(KLMN, 2, JJJ, 3, nbasis) - 1,
                        TRANSDIM, TRANSDIM, TRANSDIM);

                BGradz -= constant * LOC2(KLMN, 2, JJJ, 3, nbasis)
                    * LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM);
            }
        }
    }

    const uint32_t AStart = katom[II] * 3;
    const uint32_t BStart = katom[JJ] * 3;
    const uint32_t CStart = (iatom < natom) ? iatom * 3 : (iatom - natom) * 3;

#if defined(USE_LEGACY_ATOMICS)
    GPUATOMICADD(&gradULL[AStart], AGradx, GRADSCALE);
    GPUATOMICADD(&gradULL[AStart + 1], AGrady, GRADSCALE);
    GPUATOMICADD(&gradULL[AStart + 2], AGradz, GRADSCALE);

    GPUATOMICADD(&gradULL[BStart], BGradx, GRADSCALE);
    GPUATOMICADD(&gradULL[BStart + 1], BGrady, GRADSCALE);
    GPUATOMICADD(&gradULL[BStart + 2], BGradz, GRADSCALE);

    if (iatom < natom) {
        GPUATOMICADD(&gradULL[CStart], -AGradx - BGradx, GRADSCALE);
        GPUATOMICADD(&gradULL[CStart + 1], -AGrady - BGrady, GRADSCALE);
        GPUATOMICADD(&gradULL[CStart + 2], -AGradz - BGradz, GRADSCALE);
    } else {
        GPUATOMICADD(&ptchg_gradULL[CStart], -AGradx - BGradx, GRADSCALE);
        GPUATOMICADD(&ptchg_gradULL[CStart + 1], -AGrady - BGrady, GRADSCALE);
        GPUATOMICADD(&ptchg_gradULL[CStart + 2], -AGradz - BGradz, GRADSCALE);
    }
#else
    atomicAdd(&grad[AStart], AGradx);
    atomicAdd(&grad[AStart + 1], AGrady);
    atomicAdd(&grad[AStart + 2], AGradz);

    atomicAdd(&grad[BStart], BGradx);
    atomicAdd(&grad[BStart + 1], BGrady);
    atomicAdd(&grad[BStart + 2], BGradz);

    if (iatom < natom) {
        atomicAdd(&grad[CStart], -AGradx - BGradx);
        atomicAdd(&grad[CStart + 1], -AGrady - BGrady);
        atomicAdd(&grad[CStart + 2], -AGradz - BGradz);
    } else {
        atomicAdd(&ptchg_grad[CStart], -AGradx - BGradx);
        atomicAdd(&ptchg_grad[CStart + 1], -AGrady - BGrady);
        atomicAdd(&ptchg_grad[CStart + 2], -AGradz - BGradz);
    }
#endif
}


__global__ void k_oei_grad(bool is_oshell, uint32_t natom, uint32_t nextatom, uint32_t nbasis,
        uint32_t nshell, uint32_t jbasis, uint32_t Qshell,
        QUICKDouble const * const allchg, QUICKDouble const * const allxyz,
        uint32_t const * const kstart, uint32_t const * const katom,
        uint32_t const * const kprim, uint32_t const * const Ksumtype, uint32_t const * const Qstart,
        uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
        uint8_t const * const sorted_Qnumber, uint32_t const * const sorted_Q,
        QUICKDouble const * const cons, QUICKDouble const * const gcexpo, uint8_t const * const KLMN,
        uint32_t prim_total, uint32_t const * const prim_start,
        QUICKDouble const * const dense, QUICKDouble const * const denseb, 
        QUICKDouble const * const Xcoeff_oei, QUICKDouble const * const expoSum,
        QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
        QUICKDouble const * const weightedCenterZ, QUICKDouble coreIntegralCutoff,
        int2 const * const sorted_OEICutoffIJ,
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL * const gradULL, QUICKULL * const ptchg_gradULL,
#else
        QUICKDouble * const grad, QUICKDouble * const ptchg_grad,
#endif
#if defined(MPIV_GPU)
        unsigned char const * const mpi_boeicompute,
#endif
        QUICKDouble * const store, QUICKDouble * const store2,
        QUICKDouble * const storeAA, QUICKDouble * const storeBB)
{
    const QUICKULL jshell = (QUICKULL) Qshell;
#if defined(USE_LEGACY_ATOMICS)
    extern __shared__ QUICKULL smem[];
    QUICKULL *sgradULL = smem;
    QUICKULL *sptchg_gradULL = &sgradULL[3u * natom];

    for (int i = threadIdx.x; i < 3u * natom; i += blockDim.x) {
      sgradULL[i] = 0ull;
    }
    for (int i = threadIdx.x; i < 3u * nextatom; i += blockDim.x) {
      sptchg_gradULL[i] = 0ull;
    }
#else
    extern __shared__ QUICKDouble smem[];
    QUICKDouble *sgrad = smem;
    QUICKDouble *sptchg_grad = &sgrad[3u * natom];

    for (int i = threadIdx.x; i < 3u * natom; i += blockDim.x) {
        sgrad[i] = 0.0;
    }
    for (int i = threadIdx.x; i < 3u * nextatom; i += blockDim.x) {
        sptchg_grad[i] = 0.0;
    }
#endif

    __syncthreads();

    for (QUICKULL i = blockIdx.x * blockDim.x + threadIdx.x;
            i < jshell * jshell * (natom + nextatom); i += blockDim.x * gridDim.x) {
        // use the global index to obtain shell pair. Note that here we obtain a couple of indices that helps us to obtain
        // shell number (ii and jj) and quantum numbers (iii, jjj).
        const uint32_t iatom = i / (jshell * jshell);
        const uint32_t idx = i - iatom * jshell * jshell;

#ifdef MPIV_GPU
        if (mpi_boeicompute[idx] > 0) {
#endif
        const int II = sorted_OEICutoffIJ[idx].x;
        const int JJ = sorted_OEICutoffIJ[idx].y;

        // get the shell numbers of selected shell pair
        const uint32_t ii = sorted_Q[II];
        const uint32_t jj = sorted_Q[JJ];

        // get the quantum number (or angular momentum of shells, s=0, p=1 and so on.)
        const uint8_t iii = sorted_Qnumber[II];
        const uint8_t jjj = sorted_Qnumber[JJ];

        // compute coulomb attraction for the selected shell pair.
        iclass_oei_grad(iii, jjj, ii, jj, iatom, is_oshell, natom, nextatom, nbasis, nshell, jbasis,
                allchg, allxyz, kstart, katom, kprim, Ksumtype, Qstart, Qsbasis, Qfbasis,
                cons, gcexpo, KLMN, prim_total, prim_start, dense, denseb,
                Xcoeff_oei, expoSum, weightedCenterX, weightedCenterY, weightedCenterZ,
                coreIntegralCutoff,
#if defined(USE_LEGACY_ATOMICS)
                sgradULL, sptchg_gradULL,
#else
                sgrad, sptchg_grad,
#endif
                store, store2, storeAA, storeBB);
#ifdef MPIV_GPU
        }
#endif
    }

    __syncthreads();

    for (int i = threadIdx.x; i < 3u * natom; i += blockDim.x) {
#if defined(USE_LEGACY_ATOMICS)
        atomicAdd(&gradULL[i], sgradULL[i]);
#else
        atomicAdd(&grad[i], sgrad[i]);
#endif
    }
    for (int i = threadIdx.x; i < 3u * nextatom; i += blockDim.x) {
#if defined(USE_LEGACY_ATOMICS)
        atomicAdd(&ptchg_gradULL[i], sptchg_gradULL[i]);
#else
        atomicAdd(&ptchg_grad[i], sptchg_grad[i]);
#endif
    }
}


#endif
