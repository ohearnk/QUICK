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
 uint32_t totalatom, QUICKDouble * const store)
{
    /*
       kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
       and be careful with the index difference between Fortran and C++,
       Fortran starts array index with 1 and C++ starts 0.


       RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
       which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
       And we don't need the coordinates now, so we will not retrieve the data now.
   */
    QUICKDouble RAx = LOC2(devSim.xyz, 0, devSim.katom[II], 3, devSim.natom);
    QUICKDouble RAy = LOC2(devSim.xyz, 1, devSim.katom[II], 3, devSim.natom);
    QUICKDouble RAz = LOC2(devSim.xyz, 2, devSim.katom[II], 3, devSim.natom);

    QUICKDouble RCx = LOC2(devSim.allxyz, 0, iatom, 3, totalatom);
    QUICKDouble RCy = LOC2(devSim.allxyz, 1, iatom, 3, totalatom);
    QUICKDouble RCz = LOC2(devSim.allxyz, 2, iatom, 3, totalatom);

    /*
       kPrimI, J, K and L indicates the primtive gaussian function number
       kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
       We retrieve from global memory and save them to register to avoid multiple retrieve.
       */
    uint32_t kPrimI = devSim.kprim[II];
    uint32_t kPrimJ = devSim.kprim[JJ];

    uint32_t kStartI = devSim.kstart[II];
    uint32_t kStartJ = devSim.kstart[JJ];

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
        uint32_t ii_start = devSim.prim_start[II];
        uint32_t jj_start = devSim.prim_start[JJ];

        QUICKDouble AB = LOC2(devSim.expoSum, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);

        /*
           X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
           cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
           */
        // QUICKDouble cutoffPrim = DNMax * LOC2(devSim.cutPrim, kStartI+III, kStartJ+JJJ, devSim.jbasis, devSim.jbasis);
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI+III, kStartJ+JJJ,
                I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);

        QUICKDouble CD = devSim.lri_zeta;

        QUICKDouble ABCD = 1.0 / (AB + CD);

        /*
           X2 is the multiplication of four indices normalized coeffecient
           */
        QUICKDouble X2 = sqrt(ABCD) * X1 * X0 * (1.0 / devSim.lri_zeta)
            * devSim.lri_cc[iatom] * pow(devSim.lri_zeta / PI, 1.5);

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

    RBx = LOC2(devSim.xyz, 0, devSim.katom[JJ], 3, devSim.natom);
    RBy = LOC2(devSim.xyz, 1, devSim.katom[JJ], 3, devSim.natom);
    RBz = LOC2(devSim.xyz, 2, devSim.katom[JJ], 3, devSim.natom);

    uint32_t III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    uint32_t III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    uint32_t JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    uint32_t JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);

    /*QUICKDouble hybrid_coeff = 0.0;
      if (devSim.method == HF){
      hybrid_coeff = 1.0;
      }else if (devSim.method == B3LYP){
      hybrid_coeff = 0.2;
      }else if (devSim.method == DFT){
      hybrid_coeff = 0.0;
      }else if(devSim.method == LIBXC){
      hybrid_coeff = devSim.hyb_coeff;
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
                (I, J, 0, 0,
                 III, JJJ, 0, 0, store,
                 RAx, RAy, RAz, RBx, RBy, RBz,
                 RCx, RCy, RCz, 0.0, 0.0, 0.0);

            //printf("II JJ III JJJ Y %d %d %d %d %f \n", II, JJ, III, JJJ, Y);
#if defined(int_spd)
            if (abs(Y) > 0.0e0)
#else
            if (abs(Y) > devSim.coreIntegralCutoff)
#endif
            {
#if defined(USE_LEGACY_ATOMICS)
                GPUATOMICADD(&LOC2(devSim.oULL, JJJ, III, devSim.nbasis, devSim.nbasis), Y, OSCALE);    
#else
                atomicAdd(&LOC2(devSim.o, JJJ, III, devSim.nbasis, devSim.nbasis), Y);
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
  
  so first, kernel 0: zone 0,1,2,3,4 (get_lri_kernel()), if no f, then that's it.
  second,   kernel 1: zone 1,3,4(get_lri_kernel_spdf())
  then,     kernel 2: zone 2,3,4(get_lri_kernel_spdf2())
  then,     kernel 3: zone 3,4(get_lri_kernel_spdf3())
  finally,  kernel 4: zone 4(get_lri_kernel_spdf4())
*/
#if defined(int_spd)
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_lri_kernel()
#elif defined(int_spdf2)
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_lri_kernel_spdf2()
#endif
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
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
#elif defined(int_spdf2)
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
#endif

    uint32_t totalatom = devSim.natom + devSim.nextatom;

    for (QUICKULL i = offset; i < jshell * totalatom; i+= totalThreads) {
#if defined(int_spd)
        // Zone 0
        QUICKULL iatom = (QUICKULL) i / jshell;
        QUICKULL b = (QUICKULL) (i - iatom * jshell);
#elif defined(int_spdf2)
        // Zone 2
        QUICKULL iatom = (QUICKULL) i / jshell;
        QUICKULL b = (QUICKULL) (i - iatom * jshell);
        //a = a + devSim.fStart;
#endif

#if defined(MPIV_GPU)
        if (devSim.mpi_bcompute[b] > 0) {
#endif
            int II = devSim.sorted_YCutoffIJ[b].x;
            int JJ = devSim.sorted_YCutoffIJ[b].y;

            uint32_t ii = devSim.sorted_Q[II];
            uint32_t jj = devSim.sorted_Q[JJ];

//            printf("b II JJ ii jj %lu %lu %d %d %d %d \n", jshell, b, II, JJ, ii, jj);

            uint8_t iii = devSim.sorted_Qnumber[II];
            uint8_t jjj = devSim.sorted_Qnumber[JJ];

            // assign values to dummy variables, to be cleaned up eventually
#if defined(int_spd)
            iclass_lri(iii, jjj, ii, jj, iatom, totalatom,
                    devSim.store + offset);
#elif defined(int_spdf2)
            if (iii + jjj > 4 && iii + jjj <= 6) {
                iclass_lri_spdf2(iii, jjj, ii, jj, iatom, totalatom,
                        devSim.store + offset);
            }
#endif
#if defined(MPIV_GPU)
        }
#endif
    }
}


#undef STOREDIM
