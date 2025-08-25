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

#if !defined(__QUICK_GPU_OEI_H_)
#define __QUICK_GPU_OEI_H_

#undef FMT_NAME
#define FMT_NAME FmT
#include "gpu_fmt.h"

#undef VY
#define VY(a,b,c) (YVerticalTemp[(c)])

// support up to d functions (refactor if OEI f func support added and/or specialized for sp, spd, spdf, etc.)
#define PRIM_INT_OEI_LEN (5)


__device__ static inline void iclass_oei(uint32_t I, uint32_t J, uint32_t II, uint32_t JJ,
        uint32_t iatom, uint32_t natom, uint32_t nextatom, uint32_t nbasis,
        uint32_t nshell, uint32_t jbasis,
        QUICKDouble const * const allchg, QUICKDouble const * const allxyz,
        uint32_t const * const kstart, uint32_t const * const katom,
        uint32_t const * const kprim, uint32_t const * const Qstart,
        uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
        QUICKDouble const * const cons, uint32_t const * const KLMN,
        uint32_t prim_total, uint32_t const * const prim_start,
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL * const oULL,
#else
        QUICKDouble * const o,
#endif
        QUICKDouble const * const Xcoeff_oei, QUICKDouble const * const expoSum,
        QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
        QUICKDouble const * const weightedCenterZ, QUICKDouble coreIntegralCutoff,
        QUICKDouble * const store, QUICKDouble * const store2,
        uint32_t const * const trans, uint32_t const * const Sumindex)
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
       Store array holds contracted integral values computed using VRR algorithm.
       See J. Chem. Phys. 1986, 84, 3963−3974 for theoretical details.
    */
    for (uint32_t i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
        for (uint32_t j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(&store[blockIdx.x * blockDim.x + threadIdx.x],
                        j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint32_t i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
        for (uint32_t j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x],
                        j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    const uint32_t ii_start = prim_start[II];
    const uint32_t jj_start = prim_start[JJ];

    for (uint32_t i = 0; i < kPrimI * kPrimJ; ++i) {
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
        const QUICKDouble Zeta = LOC2(expoSum, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        const QUICKDouble Px = LOC2(weightedCenterX, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        const QUICKDouble Py = LOC2(weightedCenterY, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        const QUICKDouble Pz = LOC2(weightedCenterZ, ii_start + III, jj_start + JJJ, prim_total, prim_total);

        // get Xcoeff, which is a product of overlap prefactor and contraction coefficients
        const QUICKDouble Xcoeff = LOC4(Xcoeff_oei, kStartI + III, kStartJ + JJJ,
                I - Qstart[II], J - Qstart[JJ], jbasis, jbasis, 2, 2);

        if (abs(Xcoeff) > coreIntegralCutoff) {
            const QUICKDouble Cx = LOC2(allxyz, 0, iatom, 3, natom + nextatom);
            const QUICKDouble Cy = LOC2(allxyz, 1, iatom, 3, natom + nextatom);
            const QUICKDouble Cz = LOC2(allxyz, 2, iatom, 3, natom + nextatom);
            const QUICKDouble chg = -1.0 * allchg[iatom];

            // compute boys function values, the third term of OS A20
            double YVerticalTemp[PRIM_INT_OEI_LEN];
            FmT(I + J, Zeta * (SQR(Px - Cx) + SQR(Py - Cy) + SQR(Pz - Cz)),
                    YVerticalTemp);

            // compute all auxilary integrals and store
            for (uint32_t n = 0; n <= I + J; n++) {
                YVerticalTemp[n] *= Xcoeff * chg;
                //printf("aux: %d %f \n", i, VY(0, 0, i));
            }

            // decompose all attraction integrals to their auxilary integrals through VRR scheme.
            OEint_vertical(I, J, 
#if defined(DEBUG_OEI)
                    II, JJ, 
#endif
                    Px - Ax, Py - Ay, Pz - Az,
                    Px - Bx, Py - By, Pz - Bz,
                    Px - Cx, Py - Cy, Pz - Cz,
                    1.0 / (2.0 * Zeta), &store[blockIdx.x * blockDim.x + threadIdx.x],
                    YVerticalTemp);

            // sum up primitive integral contributions
            for (uint32_t i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
                for (uint32_t j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
                    if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM)
                            += LOCSTORE(&store[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM);
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

    for (uint32_t III = III1; III <= III2; III++) {
        // trans maps a basis function with certain angular momentum to store2 array. Get the correct indices now.
        const uint32_t i = LOC3(trans,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM);

        for (uint32_t JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            // trans maps a basis function with certain angular momentum to store2 array. Get the correct indices now.
            const uint32_t j = LOC3(trans,
                    LOC2(KLMN, 0, JJJ, 3, nbasis),
                    LOC2(KLMN, 1, JJJ, 3, nbasis),
                    LOC2(KLMN, 2, JJJ, 3, nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM);

            // multiply the integral value by normalization constants.
            const QUICKDouble Y = cons[III] * cons[JJJ]
                * LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM);

//            if (III == 10 && JJJ == 50) {
//                printf("OEI debug: III JJJ I J iatm i j c1 c2 store2 Y %d %d %d %d %d %d %d %f %f %f %f\n",
//                        III, JJJ, I, J, iatom, i,
//                        j, cons[III], cons[JJJ],
//                        LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM), Y);
//                printf("OEI debug: dt1 dt2 dt3 dt4 dt5 dt6:  %d %d %d %d %d %d \n",
//                        LOC2(KLMN, 0, III, 3, nbasis),
//                        LOC2(KLMN, 1, III, 3, nbasis),
//                        LOC2(KLMN, 2, III, 3, nbasis),
//                        LOC2(KLMN, 0, JJJ, 3, nbasis),
//                        LOC2(KLMN, 1, JJJ, 3, nbasis),
//                        LOC2(KLMN, 2, JJJ, 3, nbasis));
//            }

            // Now add the contribution into Fock matrix.
#if defined(USE_LEGACY_ATOMICS)
            GPUATOMICADD(&LOC2(oULL, JJJ, III, nbasis, nbasis), Y, OSCALE);
#else
            atomicAdd(&LOC2(o, JJJ, III, nbasis, nbasis), Y);
#endif

//            printf("addint_oei: %d %d %f %f %f \n", III, JJJ, cons[III], cons[JJJ],
//                    LOCSTORE(&store2[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM));
        }
    }
}


__global__ void k_oei(uint32_t natom, uint32_t nextatom, uint32_t nbasis,
        uint32_t nshell, uint32_t jbasis, uint32_t Qshell,
        QUICKDouble const * const allchg, QUICKDouble const * const allxyz,
        uint32_t const * const kstart, uint32_t const * const katom,
        uint32_t const * const kprim, uint32_t const * const Qstart,
        uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
        uint32_t const * const sorted_Qnumber, uint32_t const * const sorted_Q,
        QUICKDouble const * const cons, uint32_t const * const KLMN,
        uint32_t prim_total, uint32_t const * const prim_start,
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL * const oULL,
#else
        QUICKDouble * const o,
#endif
        QUICKDouble const * const Xcoeff_oei, QUICKDouble const * const expoSum,
        QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
        QUICKDouble const * const weightedCenterZ, QUICKDouble coreIntegralCutoff,
        int2 const * const sorted_OEICutoffIJ,
#if defined(MPIV_GPU)
        unsigned char const * const mpi_boeicompute,
#endif
        QUICKDouble * const store, QUICKDouble * const store2,
        uint32_t const * const trans, uint32_t const * const Sumindex)
{
    const QUICKULL jshell = (QUICKULL) Qshell;
    extern __shared__ uint32_t smem[];
    uint32_t *strans = smem;
    uint32_t *sSumindex = &strans[TRANSDIM * TRANSDIM * TRANSDIM];

    for (int i = threadIdx.x; i < TRANSDIM * TRANSDIM * TRANSDIM; i += blockDim.x) {
        strans[i] = trans[i];
    }
    for (int i = threadIdx.x; i < 10; i += blockDim.x) {
        sSumindex[i] = Sumindex[i];
    }

    __syncthreads();

    for (QUICKULL i = blockIdx.x * blockDim.x + threadIdx.x;
            i < jshell * jshell * (natom + nextatom); i += blockDim.x * gridDim.x) {
        // use the global index to obtain shell pair. Note that here we obtain a couple of indices that helps us to obtain
        // shell number (II and JJ) and quantum numbers (I, J).
        const uint32_t iatom = i / (jshell * jshell);
        const uint32_t idx = i - iatom * jshell * jshell;

#if defined(MPIV_GPU)
        if (mpi_boeicompute[idx] > 0) {
#endif
        const int II = sorted_OEICutoffIJ[idx].x;
        const int JJ = sorted_OEICutoffIJ[idx].y;

        // get the shell numbers of selected shell pair
        const uint32_t ii = sorted_Q[II];
        const uint32_t jj = sorted_Q[JJ];

        // get the quantum number (or angular momentum of shells, s=0, p=1 and so on.)
        const uint32_t iii = sorted_Qnumber[II];
        const uint32_t jjj = sorted_Qnumber[JJ];

        // compute coulomb attraction for the selected shell pair.
        iclass_oei(iii, jjj, ii, jj, iatom, natom, nextatom, nbasis, nshell, jbasis,
                allchg, allxyz, kstart, katom, kprim, Qstart, Qsbasis, Qfbasis,
                cons, KLMN, prim_total, prim_start,
#if defined(USE_LEGACY_ATOMICS)
                oULL,
#else
                o,
#endif
                Xcoeff_oei, expoSum, weightedCenterX, weightedCenterY, weightedCenterZ,
                coreIntegralCutoff, store, store2, strans, sSumindex);
#if defined(MPIV_GPU)
        }
#endif
    }
}


#endif
