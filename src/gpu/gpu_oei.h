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


__device__ static inline void iclass_oei(uint8_t I, uint8_t J, uint32_t II, uint32_t JJ,
        uint32_t iatom) {
    /*
       kAtom A, B  is the coresponding atom for shell II, JJ
       and be careful with the index difference between Fortran and C++,
       Fortran starts array index with 1 and C++ starts 0.
       Ai, Bi, Ci are the coordinates for atom katomA, katomB, katomC,
       which means they are corrosponding coorinates for shell II, JJ and nuclei.
    */
    const QUICKDouble Ax = LOC2(devSim.allxyz, 0, devSim.katom[II], 3, devSim.natom + devSim.nextatom);
    const QUICKDouble Ay = LOC2(devSim.allxyz, 1, devSim.katom[II], 3, devSim.natom + devSim.nextatom);
    const QUICKDouble Az = LOC2(devSim.allxyz, 2, devSim.katom[II], 3, devSim.natom + devSim.nextatom);
    const QUICKDouble Bx = LOC2(devSim.allxyz, 0, devSim.katom[JJ], 3, devSim.natom + devSim.nextatom);
    const QUICKDouble By = LOC2(devSim.allxyz, 1, devSim.katom[JJ], 3, devSim.natom + devSim.nextatom);
    const QUICKDouble Bz = LOC2(devSim.allxyz, 2, devSim.katom[JJ], 3, devSim.natom + devSim.nextatom);

    /*
       kPrimI and kPrimJ indicates the number of primitives in shell II and JJ.
       kStartI, J indicates the starting guassian function for shell II, JJ.
       We retrieve from global memory and save them to register to avoid multiple retrieve.
    */
    const uint32_t kPrimI = devSim.kprim[II];
    const uint32_t kPrimJ = devSim.kprim[JJ];
    const uint32_t kStartI = devSim.kstart[II];
    const uint32_t kStartJ = devSim.kstart[JJ];

    /*
       Store array holds contracted integral values computed using VRR algorithm.
       See J. Chem. Phys. 1986, 84, 3963−3974 for theoretical details.
    */
    for (uint8_t i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                        j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint8_t i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(&devSim.store2[blockIdx.x * blockDim.x + threadIdx.x],
                        j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    const uint32_t ii_start = devSim.prim_start[II];
    const uint32_t jj_start = devSim.prim_start[JJ];

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
        const QUICKDouble Zeta = LOC2(devSim.expoSum, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        const QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        const QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        const QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);

        // get Xcoeff, which is a product of overlap prefactor and contraction coefficients
        const QUICKDouble Xcoeff_oei = LOC4(devSim.Xcoeff_oei, kStartI + III, kStartJ + JJJ,
                I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);

        if (abs(Xcoeff_oei) > devSim.coreIntegralCutoff) {
            const QUICKDouble Cx = LOC2(devSim.allxyz, 0, iatom, 3, devSim.natom + devSim.nextatom);
            const QUICKDouble Cy = LOC2(devSim.allxyz, 1, iatom, 3, devSim.natom + devSim.nextatom);
            const QUICKDouble Cz = LOC2(devSim.allxyz, 2, iatom, 3, devSim.natom + devSim.nextatom);
            const QUICKDouble chg = -1.0 * devSim.allchg[iatom];

            // compute boys function values, the third term of OS A20
            double YVerticalTemp[PRIM_INT_OEI_LEN];
            FmT(I + J, Zeta * (SQR(Px - Cx) + SQR(Py - Cy) + SQR(Pz - Cz)),
                    YVerticalTemp);

            // compute all auxilary integrals and store
            for (uint32_t n = 0; n <= I + J; n++) {
                YVerticalTemp[n] *= Xcoeff_oei * chg;
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
                    1.0 / (2.0 * Zeta), &devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                    YVerticalTemp);

            // sum up primitive integral contributions
            for (uint8_t i = Sumindex[J]; i < Sumindex[J + 2]; ++i) {
                for (uint8_t j = Sumindex[I]; j < Sumindex[I + 2]; ++j) {
                    if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(&devSim.store2[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM)
                            += LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x], j, i, STOREDIM, STOREDIM);
                    }
                }
            }
        }
    }

    // retrive computed integral values from store array and update the Fock matrix
    
    // obtain the start and final basis function indices for given shells II and JJ. They will help us to save the integral
    // contribution into correct location in Fock matrix.
    const uint32_t III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    const uint32_t III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    const uint32_t JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    const uint32_t JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

    for (uint32_t III = III1; III <= III2; III++) {
        // devTrans maps a basis function with certain angular momentum to store2 array. Get the correct indices now.
        const uint8_t i = LOC3(devTrans,
                LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM);

        for (uint32_t JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            // devTrans maps a basis function with certain angular momentum to store2 array. Get the correct indices now.
            const uint8_t j = LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM);

            // multiply the integral value by normalization constants.
            const QUICKDouble Y = devSim.cons[III] * devSim.cons[JJJ]
                * LOCSTORE(&devSim.store2[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM);

//            if (III == 10 && JJJ == 50) {
//                printf("OEI debug: III JJJ I J iatm i j c1 c2 store2 Y %d %d %d %d %d %d %d %f %f %f %f\n",
//                        III, JJJ, I, J, iatom, i,
//                        j, devSim.cons[III], devSim.cons[JJJ],
//                        LOCSTORE(&devSim.store2[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM), Y);
//                printf("OEI debug: dt1 dt2 dt3 dt4 dt5 dt6:  %d %d %d %d %d %d \n",
//                        LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
//                        LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
//                        LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
//                        LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
//                        LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
//                        LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis));
//            }

            // Now add the contribution into Fock matrix.
#if defined(USE_LEGACY_ATOMICS)
            GPUATOMICADD(&LOC2(devSim.oULL, JJJ, III, devSim.nbasis, devSim.nbasis), Y, OSCALE);
#else
            atomicAdd(&LOC2(devSim.o, JJJ, III, devSim.nbasis, devSim.nbasis), Y);
#endif

//            printf("addint_oei: %d %d %f %f %f \n", III, JJJ, devSim.cons[III], devSim.cons[JJJ],
//                    LOCSTORE(&devSim.store2[blockIdx.x * blockDim.x + threadIdx.x], i, j, STOREDIM, STOREDIM));
        }
    }
}


__global__ void k_oei() {
    const QUICKULL jshell = (QUICKULL) devSim.Qshell;

    for (QUICKULL i = blockIdx.x * blockDim.x + threadIdx.x;
            i < jshell * jshell * (devSim.natom + devSim.nextatom); i += blockDim.x * gridDim.x) {
        // use the global index to obtain shell pair. Note that here we obtain a couple of indices that helps us to obtain
        // shell number (ii and jj) and quantum numbers (iii, jjj).
        const uint32_t iatom = i / (jshell * jshell);
        const uint32_t idx = i - iatom * jshell * jshell;

#if defined(MPIV_GPU)
        if (devSim.mpi_boeicompute[idx] > 0) {
#endif
        const int II = devSim.sorted_OEICutoffIJ[idx].x;
        const int JJ = devSim.sorted_OEICutoffIJ[idx].y;

        // get the shell numbers of selected shell pair
        const uint32_t ii = devSim.sorted_Q[II];
        const uint32_t jj = devSim.sorted_Q[JJ];

        // get the quantum number (or angular momentum of shells, s=0, p=1 and so on.)
        const uint8_t iii = devSim.sorted_Qnumber[II];
        const uint8_t jjj = devSim.sorted_Qnumber[JJ];

        // compute coulomb attraction for the selected shell pair.
        iclass_oei(iii, jjj, ii, jj, iatom);
#if defined(MPIV_GPU)
        }
#endif
    }
}


#endif
