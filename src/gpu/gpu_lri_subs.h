/*
   !---------------------------------------------------------------------!
   ! Written by Madu Manathunga on 08/31/2021                            !
   !                                                                     !
   ! Copyright (C) 2020-2021 Merz lab                                    !
   ! Copyright (C) 2020-2021 Götz lab                                    !
   !                                                                     !
   ! This Source Code Form is subject to the terms of the Mozilla Public !
   ! License, v. 2.0. If a copy of the MPL was not distributed with this !
   ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
   !_____________________________________________________________________!

   !---------------------------------------------------------------------!
   ! This source file contains driver functions required for computing 3 !
   ! center integrals necessary for CEW method.                          !
   !---------------------------------------------------------------------!
   */

#include "gpu_common.h"

#undef STOREDIM
#undef VY
#if defined(int_spd)
  #define STOREDIM STOREDIM_S
#else
  #define STOREDIM STOREDIM_L
#endif
#define VY(a,b,c) (YVerticalTemp[(c)])

#undef FMT_NAME
#define FMT_NAME FmT
#include "gpu_fmt.h"

#undef PRIM_INT_LRI_LEN
#if defined(int_spd)
  #define PRIM_INT_LRI_LEN (5)
#elif defined(int_spdf2)
  #define PRIM_INT_LRI_LEN (7)
#endif


/*
   iclass_lri subroutine is to generate 3 center intergrals using HRR and VRR method.
*/
#if defined(int_spd)
__device__ static inline void iclass_lri
#elif defined(int_spdf2)
__device__ static inline void iclass_lri_spdf2
#endif
    (uint8_t I, uint8_t J, uint32_t II, uint32_t JJ, uint32_t iatom,
     uint32_t totalatom, uint32_t natom, uint32_t nbasis,
     uint32_t nshell, uint32_t jbasis, QUICKDouble const * const xyz,
     QUICKDouble const * const allxyz, uint32_t const * const kstart, uint32_t const * const katom,
     uint32_t const * const kprim, uint32_t const * const Qstart,
     uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
     uint8_t const * const sorted_Qnumber, uint32_t const * const sorted_Q,
     QUICKDouble const * const cons, uint8_t const * const KLMN,
     uint32_t prim_total, uint32_t const * const prim_start,
#if defined(USE_LEGACY_ATOMICS)
     QUICKULL * const oULL,
#else
     QUICKDouble * const o,
#endif
     QUICKDouble const * const Xcoeff, QUICKDouble const * const expoSum,
     QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
     QUICKDouble const * const weightedCenterZ, uint32_t sqrQshell, int2 const * const sorted_YCutoffIJ,
#if defined(MPIV_GPU)
     unsigned char const * const mpi_bcompute,
#endif
     QUICKDouble * const store)
{
    /*
       kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
       and be careful with the index difference between Fortran and C++,
       Fortran starts array index with 1 and C++ starts 0.


       RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
       which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
       And we don't need the coordinates now, so we will not retrieve the data now.
   */
    QUICKDouble RAx = LOC2(xyz, 0, katom[II], 3, natom);
    QUICKDouble RAy = LOC2(xyz, 1, katom[II], 3, natom);
    QUICKDouble RAz = LOC2(xyz, 2, katom[II], 3, natom);

    QUICKDouble RCx = LOC2(allxyz, 0, iatom, 3, totalatom);
    QUICKDouble RCy = LOC2(allxyz, 1, iatom, 3, totalatom);
    QUICKDouble RCz = LOC2(allxyz, 2, iatom, 3, totalatom);

    /*
       kPrimI, J, K and L indicates the primtive gaussian function number
       kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
       We retrieve from global memory and save them to register to avoid multiple retrieve.
       */
    uint32_t kPrimI = kprim[II];
    uint32_t kPrimJ = kprim[JJ];

    uint32_t kStartI = kstart[II];
    uint32_t kStartJ = kstart[JJ];

    /*
       store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
       of GPU limitation, we can not do that now.

       See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
       */
    /*
       Initial the neccessary element for
    */
#if defined(int_spd)
    for (uint8_t i = Sumindex[1] + 1; i <= Sumindex[2]; i++) {
        for (uint8_t j = Sumindex[I + 1] + 1; j <= Sumindex[I + J + 2]; j++) {
            if (i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j - 1, i - 1, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }
#elif(defined int_spdf2)
    for (uint8_t i = Sumindex[1] + 1; i <= Sumindex[2]; i++) {
        for (uint8_t j = Sumindex[I + 1] + 1; j <= Sumindex[I + J + 2]; j++) {
            if (i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j - 1, i - 1, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }
#endif

    for (uint32_t i = 0; i < kPrimI * kPrimJ; i++) {
        uint32_t JJJ = (uint32_t) i / kPrimI;
        uint32_t III = (uint32_t) i - kPrimI * JJJ;
        /*
           In the following comments, we have I, J, K, L denote the primitive gaussian function we use, and
           for example, expo(III, ksumtype(II)) stands for the expo for the IIIth primitive guassian function for II shell,
           we use I to express the corresponding index.
           AB = expo(I)+expo(J)
           --->                --->
           ->     expo(I) * xyz (I) + expo(J) * xyz(J)
           P  = ---------------------------------------
           expo(I) + expo(J)
           Those two are pre-calculated in CPU stage.

        */
        uint32_t ii_start = prim_start[II];
        uint32_t jj_start = prim_start[JJ];

        QUICKDouble AB = LOC2(expoSum, ii_start+III, jj_start+JJJ, prim_total, prim_total);
        QUICKDouble Px = LOC2(weightedCenterX, ii_start+III, jj_start+JJJ, prim_total, prim_total);
        QUICKDouble Py = LOC2(weightedCenterY, ii_start+III, jj_start+JJJ, prim_total, prim_total);
        QUICKDouble Pz = LOC2(weightedCenterZ, ii_start+III, jj_start+JJJ, prim_total, prim_total);

        /*
           X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
           cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
           */
        // QUICKDouble cutoffPrim = DNMax * LOC2(cutPrim, kStartI+III, kStartJ+JJJ, jbasis, jbasis);
        QUICKDouble X1 = LOC4(Xcoeff, kStartI+III, kStartJ+JJJ,
                I - Qstart[II], J - Qstart[JJ], jbasis, jbasis, 2, 2);

        QUICKDouble CD = lri_zeta;

        QUICKDouble ABCD = 1.0 / (AB + CD);

        /*
           X2 is the multiplication of four indices normalized coeffecient
           */
        QUICKDouble X2 = sqrt(ABCD) * X1 * X0 * (1.0 / lri_zeta)
            * lri_cc[iatom] * pow(lri_zeta / PI, 1.5);

        /*
           Q' is the weighting center of K and L
           --->           --->
           ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
           Q = P'(K,L)  = ------------------------------
           expo(K) + expo(L)

           W' is the weight center for I, J, K, L

           --->             --->             --->            --->
           ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
           W = -------------------------------------------------------------------
           expo(I) + expo(J) + expo(K) + expo(L)
           ->  ->  2
           RPQ =| P - Q |

           ->  -> 2
           T = ROU * | P - Q|
       */
        QUICKDouble Qx = RCx;
        QUICKDouble Qy = RCy;
        QUICKDouble Qz = RCz;

        double YVerticalTemp[PRIM_INT_LRI_LEN];
        FmT(I + J, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)), YVerticalTemp);

        for (uint32_t i = 0; i <= I + J; i++) {
            YVerticalTemp[i] *= X2;
        }

#if defined(int_spd)
        lri::vertical(I, J, 0, 0, YVerticalTemp, store,
                Px - RAx, Py - RAy, Pz - RAz,
                (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                Qx - RCx, Qy - RCy, Qz - RCz,
                (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif(defined int_spdf2)
        lri::vertical_spdf2(I, J, 0, 0, YVerticalTemp, store,
                Px - RAx, Py - RAy, Pz - RAz,
                (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                Qx - RCx, Qy - RCy, Qz - RCz,
                (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#endif
    }

    QUICKDouble RBx, RBy, RBz;

    RBx = LOC2(xyz, 0, katom[JJ], 3, natom);
    RBy = LOC2(xyz, 1, katom[JJ], 3, natom);
    RBz = LOC2(xyz, 2, katom[JJ], 3, natom);

    uint32_t III1 = LOC2(Qsbasis, II, I, nshell, 4);
    uint32_t III2 = LOC2(Qfbasis, II, I, nshell, 4);
    uint32_t JJJ1 = LOC2(Qsbasis, JJ, J, nshell, 4);
    uint32_t JJJ2 = LOC2(Qfbasis, JJ, J, nshell, 4);

    /*QUICKDouble hybrid_coeff = 0.0;
      if (method == HF){
      hybrid_coeff = 1.0;
      }else if (method == B3LYP){
      hybrid_coeff = 0.2;
      }else if (method == DFT){
      hybrid_coeff = 0.0;
      }else if(method == LIBXC){
      hybrid_coeff = hyb_coeff;
      }*/

    for (uint32_t III = III1; III <= III2; III++) {
        for (uint32_t JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
#if defined(int_spd)
            QUICKDouble Y = (QUICKDouble) hrrwhole_lri
#elif(defined int_spdf2)
            QUICKDouble Y = (QUICKDouble) hrrwhole_lri_2_2
#else
            QUICKDouble Y = (QUICKDouble) hrrwhole_lri_2
#endif
                (I, J, 0, 0, III, JJJ, 0, 0,
                 RAx, RAy, RAz, RBx, RBy, RBz,
                 RCx, RCy, RCz, 0.0, 0.0, 0.0,
                 nbasis, cons, KLMN, store);

            //printf("II JJ III JJJ Y %d %d %d %d %f \n", II, JJ, III, JJJ, Y);
#if defined(int_spd)
            if (abs(Y) > 0.0e0)
#else
            if (abs(Y) > coreIntegralCutoff)
#endif
            {
#if defined(USE_LEGACY_ATOMICS)
                GPUATOMICADD(&LOC2(oULL, JJJ, III, nbasis, nbasis), Y, OSCALE);    
#else
                atomicAdd(&LOC2(o, JJJ, III, nbasis, nbasis), Y);
#endif
            }
        }
    }
}


/*
   Note that this driver implementations are very similar to the ones implemented by Yipu Miao in gpu_get2e_subs.h.
   To understand the following comments better, please refer to Figure 2(b) and 2(d) in Miao and Merz 2015 paper.

   In the following kernel, we treat f orbital into 5 parts.

  type:   ss sp ps sd ds pp dd sf pf | df ff |
  ss                                 |       |
  sp                                 |       |
  ps                                 | zone  |
  sd                                 |  2    |
  ds         zone 0                  |       |
  pp                                 |       |
  dd                                 |       |
  sf                                 |       |
  pf                                 |       |
  -------------------------------------------
  df         zone 1                  | z | z |
  ff                                 | 3 | 4 |
  -------------------------------------------
  
  
  because the single f orbital kernel is impossible to compile completely, we treat VRR as:
  
  
  I+J  0 1 2 3 4 | 5 | 6 |
  0 ----------------------
  1|             |       |
  2|   Kernel    |  K2   |
  3|     0       |       |
  4|             |       |
  -----------------------|
  5|   Kernel    | K | K |
  6|     1       | 3 | 4 |
  ------------------------
  
  Their responses for
  I+J          K+L
  Kernel 0:   0-4           0-4
  Kernel 1:   0-4           5,6
  Kernel 2:   5,6           0-4
  Kernel 3:   5             5,6
  Kernel 4:   6             5,6
  
  Integrals in zone need kernel:
  zone 0: kernel 0
  zone 1: kernel 0,1
  zone 2: kernel 0,2
  zone 3: kernel 0,1,2,3
  zone 4: kernel 0,1,2,3,4
  
  so first, kernel 0: zone 0,1,2,3,4 (k_get_lri()), if no f, then that's it.
  second,   kernel 1: zone 1,3,4(k_get_lri_spdf())
  then,     kernel 2: zone 2,3,4(k_get_lri_spdf2())
  then,     kernel 3: zone 3,4(k_get_lri_spdf3())
  finally,  kernel 4: zone 4(k_get_lri_spdf4())
*/
#if defined(int_spd)
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_get_lri
#elif defined(int_spdf2)
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_get_lri_spdf2
#endif
    (uint32_t natom, uint32_t nbasis, uint32_t nshell, uint32_t jbasis,
     QUICKDouble const * const xyz, QUICKDouble const * const allxyz,
     uint32_t const * const kstart, uint32_t const * const katom,
     uint32_t const * const kprim, uint32_t const * const Qstart,
     uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
     uint8_t const * const sorted_Qnumber, uint32_t const * const sorted_Q,
     QUICKDouble const * const cons, uint8_t const * const KLMN,
     uint32_t prim_total, uint32_t const * const prim_start,
#if defined(USE_LEGACY_ATOMICS)
     QUICKULL * const oULL,
#else
     QUICKDouble * const o,
#endif
     QUICKDouble const * const Xcoeff, QUICKDouble const * const expoSum,
     QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
     QUICKDouble const * const weightedCenterZ, uint32_t sqrQshell, int2 const * const sorted_YCutoffIJ,
#if defined(MPIV_GPU)
     unsigned char const * const mpi_bcompute,
#endif
     QUICKDouble * const store)
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    // jshell and jshell2 defines the regions in i+j and k+l axes respectively.
    // sqrQshell= Qshell x Qshell; where Qshell is the number of sorted shells (see gpu_upload_basis_ in gpu.cu)
    // for details on sorting.
#if defined(int_spd)
    /*
       Here we walk through full cutoff matrix.

       --sqrQshell --
       _______________
       |             |  |
       |             |  |
       |             | sqrQshell
       |             |  |
       |             |  |
       |_____________|  |

*/
    QUICKULL jshell = (QUICKULL) sqrQshell;
#elif defined(int_spdf2)
    QUICKULL jshell = (QUICKULL) sqrQshell;
#endif

    uint32_t totalatom = natom + nextatom;

    for (QUICKULL i = offset; i < jshell * totalatom; i+= totalThreads) {
#if defined(int_spd)
        // Zone 0
        QUICKULL iatom = (QUICKULL) i / jshell;
        QUICKULL b = (QUICKULL) (i - iatom * jshell);
#elif defined(int_spdf2)
        // Zone 2
        QUICKULL iatom = (QUICKULL) i / jshell;
        QUICKULL b = (QUICKULL) (i - iatom * jshell);
        //a = a + fStart;
#endif

#if defined(MPIV_GPU)
        if (mpi_bcompute[b] > 0) {
#endif
            int II = sorted_YCutoffIJ[b].x;
            int JJ = sorted_YCutoffIJ[b].y;

            uint32_t ii = sorted_Q[II];
            uint32_t jj = sorted_Q[JJ];

//            printf("b II JJ ii jj %lu %lu %d %d %d %d \n", jshell, b, II, JJ, ii, jj);

            uint8_t iii = sorted_Qnumber[II];
            uint8_t jjj = sorted_Qnumber[JJ];

            // assign values to dummy variables, to be cleaned up eventually
#if defined(int_spd)
            {
                iclass_lri
#elif defined(int_spdf2)
            if (iii + jjj > 4 && iii + jjj <= 6) {
                iclass_lri_spdf2
#endif
                (iii, jjj, ii, jj, iatom, totalatom, natom, nbasis, nshell, jbasis,
                 xyz, allxyz, kstart, katom, kprim, Qstart, Qsbasis, Qfbasis,
                 cons, KLMN, prim_total, prim_start,
#if defined(USE_LEGACY_ATOMICS)
                 oULL,
#else
                 o,
#endif
                 Xcoeff, expoSum, weightedCenterX, weightedCenterY,
                 weightedCenterZ, store + offset);
            }
#if defined(MPIV_GPU)
        }
#endif
    }
}


#undef STOREDIM
