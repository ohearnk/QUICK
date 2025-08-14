//
//  gpu_get2e_subs.h
//  new_quick 2
//
//  Created by Yipu Miao on 6/18/13.
//
//

#include "gpu_common.h"

#undef STOREDIM
#undef VDIM3
#undef VY
#undef LOCSTORE
#if defined(int_sp)
  #define STOREDIM STOREDIM_T
  #define VDIM3 VDIM3_T
#elif defined(int_spd)
  #define STOREDIM STOREDIM_S
  #define VDIM3 VDIM3_S
#else
  #define STOREDIM STOREDIM_L
  #define VDIM3 VDIM3_L
#endif
#define LOCSTORE(A,i1,i2,d1,d2) ((A)[((i2) * (d1) + (i1)) * gridDim.x * blockDim.x])
#define VY(a,b,c) LOCVY(&devSim.YVerticalTemp[blockIdx.x * blockDim.x + threadIdx.x], (a), (b), (c), VDIM1, VDIM2, VDIM3)

#undef FMT_NAME
#define FMT_NAME FmT
#include "gpu_fmt.h"


#if !defined(__gpu_get2e_subs_h_)
  #define __gpu_get2e_subs_h_
  #if !defined(OSHELL)
__device__ static inline bool check_iclass(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t II, uint32_t JJ, uint32_t KK, uint32_t LL)
{
    bool ret = false;

    for (uint32_t III = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
            III <= LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4); III++) {
        for (uint32_t JJJ = MAX(III, LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4));
                JJJ <= LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4); JJJ++) {
            for (uint32_t KKK = MAX(III, LOC2(devSim.Qsbasis, KK, K, devSim.nshell, 4));
                    KKK <= LOC2(devSim.Qfbasis, KK, K, devSim.nshell, 4); KKK++) {
                for (uint32_t LLL = MAX(KKK, LOC2(devSim.Qsbasis, LL, L, devSim.nshell, 4));
                        LLL <= LOC2(devSim.Qfbasis, LL, L, devSim.nshell, 4); LLL++) {
                    if ((III < JJJ && III < KKK && KKK < LLL)
                            || (III < KKK || JJJ <= LLL)) {
                       ret = true;
                       break;
                    }
                }
            }
        }
    }

    return ret;
}
  #endif
#endif


/*
 iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
 performance algrithem for electron intergral evaluation. See description below for details
 */
#if defined(OSHELL)
  #if defined(int_sp)
__device__ static inline void iclass_oshell_sp
  #elif defined(int_spd)
__device__ static inline void iclass_oshell_spd
  #elif defined(int_spdf)
__device__ static inline void iclass_oshell_spdf
  #elif defined(int_spdf2)
__device__ static inline void iclass_oshell_spdf2
  #elif defined(int_spdf3)
__device__ static inline void iclass_oshell_spdf3
  #elif defined(int_spdf4)
__device__ static inline void iclass_oshell_spdf4
  #elif defined(int_spdf5)
__device__ static inline void iclass_oshell_spdf5
  #elif defined(int_spdf6)
__device__ static inline void iclass_oshell_spdf6
  #elif defined(int_spdf7)
__device__ static inline void iclass_oshell_spdf7
  #elif defined(int_spdf8)
__device__ static inline void iclass_oshell_spdf8
  #elif defined(int_spdf9)
__device__ static inline void iclass_oshell_spdf9
  #elif defined(int_spdf10)
__device__ static inline void iclass_oshell_spdf10
  #endif
#else
  #if defined(int_sp)
__device__ static inline void iclass_cshell_sp
  #elif defined(int_spd)
__device__ static inline void iclass_cshell_spd
  #elif defined(int_spdf)
__device__ static inline void iclass_cshell_spdf
  #elif defined(int_spdf2)
__device__ static inline void iclass_cshell_spdf2
  #elif defined(int_spdf3)
__device__ static inline void iclass_cshell_spdf3
  #elif defined(int_spdf4)
__device__ static inline void iclass_cshell_spdf4
  #elif defined(int_spdf5)
__device__ static inline void iclass_cshell_spdf5
  #elif defined(int_spdf6)
__device__ static inline void iclass_cshell_spdf6
  #elif defined(int_spdf7)
__device__ static inline void iclass_cshell_spdf7
  #elif defined(int_spdf8)
__device__ static inline void iclass_cshell_spdf8
  #elif defined(int_spdf9)
__device__ static inline void iclass_cshell_spdf9
  #elif defined(int_spdf10)
__device__ static inline void iclass_cshell_spdf10
  #endif
#endif
      (uint8_t I, uint8_t J, uint8_t K, uint8_t L, uint32_t II, uint32_t JJ, uint32_t KK, uint32_t LL,
       QUICKDouble DNMax)
{
    QUICKDouble temp;
#if defined(OSHELL)
    QUICKDouble temp2;
#endif

    /*
     kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
     and be careful with the index difference between Fortran and C++,
     Fortran starts array index with 1 and C++ starts 0.
     
     RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
     which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
     And we don't need the coordinates now, so we will not retrieve the data now.
     */
    const QUICKDouble RAx = LOC2(devSim.xyz, 0, devSim.katom[II], 3, devSim.natom);
    const QUICKDouble RAy = LOC2(devSim.xyz, 1, devSim.katom[II], 3, devSim.natom);
    const QUICKDouble RAz = LOC2(devSim.xyz, 2, devSim.katom[II], 3, devSim.natom);
    const QUICKDouble RCx = LOC2(devSim.xyz, 0, devSim.katom[KK], 3, devSim.natom);
    const QUICKDouble RCy = LOC2(devSim.xyz, 1, devSim.katom[KK], 3, devSim.natom);
    const QUICKDouble RCz = LOC2(devSim.xyz, 2, devSim.katom[KK], 3, devSim.natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    const uint32_t kPrimI = devSim.kprim[II];
    const uint32_t kPrimJ = devSim.kprim[JJ];
    const uint32_t kPrimK = devSim.kprim[KK];
    const uint32_t kPrimL = devSim.kprim[LL];
    
    const uint32_t kStartI = devSim.kstart[II];
    const uint32_t kStartJ = devSim.kstart[JJ];
    const uint32_t kStartK = devSim.kstart[KK];
    const uint32_t kStartL = devSim.kstart[LL];
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of GPU limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
    */
    for (uint8_t i = Sumindex[K + 1] + 1; i <= Sumindex[K + L + 2]; i++) {
        for (uint8_t j = Sumindex[I + 1] + 1; j <= Sumindex[I + J + 2]; j++) {
            if (i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                        j - 1, i - 1, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }
    
    for (uint32_t i = 0; i < kPrimI * kPrimJ; i++) {
        const uint32_t JJJ = i / kPrimI;
        const uint32_t III = i - kPrimI * JJJ;

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
        const uint32_t ii_start = devSim.prim_start[II];
        const uint32_t jj_start = devSim.prim_start[JJ];
        
        const QUICKDouble AB = LOC2(devSim.expoSum, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        const QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        const QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        const QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        const QUICKDouble cutoffPrim = DNMax * LOC2(devSim.cutPrim, kStartI + III, kStartJ + JJJ, devSim.jbasis, devSim.jbasis);
        const QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI + III, kStartJ + JJJ, I - devSim.Qstart[II], J - devSim.Qstart[JJ],
                devSim.jbasis, devSim.jbasis, 2, 2);
        
        for (uint32_t j = 0; j < kPrimK * kPrimL; j++) {
            const uint32_t LLL = j / kPrimK;
            const uint32_t KKK = j - kPrimK * LLL;
            
            if (cutoffPrim * LOC2(devSim.cutPrim, kStartK + KKK, kStartL + LLL, devSim.jbasis, devSim.jbasis) > devSim.primLimit) {
                /*
                 CD = expo(L)+expo(K)
                 ABCD = 1/ (AB + CD) = 1 / (expo(I)+expo(J)+expo(K)+expo(L))
                 AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
                 Rou(Greek Letter) =   ----------- = ------------------------------------
                 AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
                 
                 expo(I)+expo(J)                        expo(K)+expo(L)
                 ABcom = --------------------------------  CDcom = --------------------------------
                 expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)
                 
                 ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))
                 */
                const uint32_t kk_start = devSim.prim_start[KK];
                const uint32_t ll_start = devSim.prim_start[LL];
                const QUICKDouble CD = LOC2(devSim.expoSum, kk_start + KKK, ll_start + LLL, devSim.prim_total, devSim.prim_total);
                const QUICKDouble ABCD = 1.0 / (AB + CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
#if defined(USE_TEXTURE) && defined(USE_TEXTURE_XCOEFF)
                const int2 XcoeffInt2 = tex1Dfetch(tex_Xcoeff, L - devSim.Qstart[LL] +
                        (K - devSim.Qstart[KK] + ((kStartL + LLL) + (kStartK + KKK) * devSim.jbasis) * 2) * 2);
                const QUICKDouble X2 = sqrt(ABCD) * X1 * __hiloint2double(XcoeffInt2.y, XcoeffInt2.x);
#else
                const QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(devSim.Xcoeff, kStartK + KKK, kStartL + LLL,
                        K - devSim.Qstart[KK], L - devSim.Qstart[LL], devSim.jbasis, devSim.jbasis, 2, 2);
#endif                

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
                const QUICKDouble Qx = LOC2(devSim.weightedCenterX, kk_start + KKK, ll_start + LLL, devSim.prim_total, devSim.prim_total);
                const QUICKDouble Qy = LOC2(devSim.weightedCenterY, kk_start + KKK, ll_start + LLL, devSim.prim_total, devSim.prim_total);
                const QUICKDouble Qz = LOC2(devSim.weightedCenterZ, kk_start + KKK, ll_start + LLL, devSim.prim_total, devSim.prim_total);
                
                FmT(I + J + K + L, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)),
                        &devSim.YVerticalTemp[blockIdx.x * blockDim.x + threadIdx.x]);

                for (uint32_t i = 0; i <= I + J + K + L; i++) {
                    VY(0, 0, i) *= X2;
                }

#if defined(int_sp)
                ERint_vertical_sp
#elif defined(int_spd)
                ERint_vertical_spd
#elif defined(int_spdf)
                ERint_vertical_spdf_1
#elif defined(int_spdf2)
                ERint_vertical_spdf_2
#elif defined(int_spdf3)
                ERint_vertical_spdf_3
#elif defined(int_spdf4)
                ERint_vertical_spdf_4
#elif defined(int_spdf5)
                ERint_vertical_spdf_5
#elif defined(int_spdf6)
                ERint_vertical_spdf_6
#elif defined(int_spdf7)
                ERint_vertical_spdf_7
#elif defined(int_spdf8)
                ERint_vertical_spdf_8
#elif defined(int_spdf9)
                ERint_vertical_spdf_8
#elif defined(int_spdf10)
                ERint_vertical_spdf_8
#endif
                    (I, J, K, L,
                     Px - RAx, Py - RAy, Pz - RAz, (Px * AB + Qx * CD) * ABCD - Px,
                     (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                     Qx - RCx, Qy - RCy, Qz - RCz, (Px * AB + Qx * CD) * ABCD - Qx,
                     (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                     0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD,
                     &devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                     &devSim.YVerticalTemp[blockIdx.x * blockDim.x + threadIdx.x]);
            }
        }
    }
    
    const QUICKDouble RBx = LOC2(devSim.xyz, 0, devSim.katom[JJ], 3, devSim.natom);
    const QUICKDouble RBy = LOC2(devSim.xyz, 1, devSim.katom[JJ], 3, devSim.natom);
    const QUICKDouble RBz = LOC2(devSim.xyz, 2, devSim.katom[JJ], 3, devSim.natom);
    const QUICKDouble RDx = LOC2(devSim.xyz, 0, devSim.katom[LL], 3, devSim.natom);
    const QUICKDouble RDy = LOC2(devSim.xyz, 1, devSim.katom[LL], 3, devSim.natom);
    const QUICKDouble RDz = LOC2(devSim.xyz, 2, devSim.katom[LL], 3, devSim.natom);
    
    const uint32_t III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    const uint32_t III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    const uint32_t JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    const uint32_t JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);
    const uint32_t KKK1 = LOC2(devSim.Qsbasis, KK, K, devSim.nshell, 4);
    const uint32_t KKK2 = LOC2(devSim.Qfbasis, KK, K, devSim.nshell, 4);
    const uint32_t LLL1 = LOC2(devSim.Qsbasis, LL, L, devSim.nshell, 4);
    const uint32_t LLL2 = LOC2(devSim.Qfbasis, LL, L, devSim.nshell, 4);
    
//    QUICKDouble hybrid_coeff = 0.0;
//    if (devSim.method == HF) {
//        hybrid_coeff = 1.0;
//    } else if (devSim.method == B3LYP) {
//        hybrid_coeff = 0.2;
//    } else if (devSim.method == DFT) {
//        hybrid_coeff = 0.0;
//    } else if (devSim.method == LIBXC) {
//        hybrid_coeff = devSim.hyb_coeff;                        
//    }
    
    for (uint32_t III = III1; III <= III2; III++) {
        for (uint32_t JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            QUICKDouble o_JI = 0.0;
#if defined(OSHELL)
            QUICKDouble ob_JI = 0.0;
#endif
            for (uint32_t KKK = MAX(III, KKK1); KKK <= KKK2; KKK++) {
                QUICKDouble o_KI = 0.0;
                QUICKDouble o_JK = 0.0;
                QUICKDouble o_JK_MM = 0.0;
#if defined(OSHELL)
                QUICKDouble ob_KI = 0.0;
                QUICKDouble ob_JK = 0.0;
                QUICKDouble ob_JK_MM = 0.0;
#endif

                for (uint32_t LLL = MAX(KKK, LLL1); LLL <= LLL2; LLL++) {
                    if (III < KKK
                            || (III == JJJ && III == LLL)
                            || (III == JJJ && III < LLL)
                            || (JJJ == LLL && III < JJJ)
                            || (III == KKK && III < JJJ && JJJ < LLL)) {
#if defined(int_sp)
                        QUICKDouble Y = (QUICKDouble) hrrwhole_sp
#elif defined(int_spd)
                        QUICKDouble Y = (QUICKDouble) hrrwhole
#elif defined(int_spdf1)
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_1
#elif defined(int_spdf2)
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_2
#elif defined(int_spdf3)
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_3
#elif defined(int_spdf4)
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_4
#elif defined(int_spdf5)
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_5
#elif defined(int_spdf6)
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_6
#elif defined(int_spdf7)
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_7
#elif defined(int_spdf8)
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_8
#elif defined(int_spdf9)
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_9
#elif defined(int_spdf10)
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_10
#else
                        QUICKDouble Y = (QUICKDouble) hrrwhole2
#endif
                           (I, J, K, L, III, JJJ, KKK, LLL,
                           RAx, RAy, RAz, RBx, RBy, RBz,
                           RCx, RCy, RCz, RDx, RDy, RDz);

                        if (abs(Y) > devSim.integralCutoff)
                        {
#if defined(OSHELL)
                            QUICKDouble DENSELK = (QUICKDouble) (LOC2(devSim.dense, LLL, KKK, devSim.nbasis, devSim.nbasis)
                                    + LOC2(devSim.denseb, LLL, KKK, devSim.nbasis, devSim.nbasis));
                            QUICKDouble DENSEJI = (QUICKDouble) (LOC2(devSim.dense, JJJ, III, devSim.nbasis, devSim.nbasis)
                                    + LOC2(devSim.denseb, JJJ, III, devSim.nbasis, devSim.nbasis));

                            QUICKDouble DENSEKIA = (QUICKDouble) LOC2(devSim.dense, KKK, III, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSEKJA = (QUICKDouble) LOC2(devSim.dense, KKK, JJJ, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSELJA = (QUICKDouble) LOC2(devSim.dense, LLL, JJJ, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSELIA = (QUICKDouble) LOC2(devSim.dense, LLL, III, devSim.nbasis, devSim.nbasis);

                            QUICKDouble DENSEKIB = (QUICKDouble) LOC2(devSim.denseb, KKK, III, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSEKJB = (QUICKDouble) LOC2(devSim.denseb, KKK, JJJ, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSELJB = (QUICKDouble) LOC2(devSim.denseb, LLL, JJJ, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSELIB = (QUICKDouble) LOC2(devSim.denseb, LLL, III, devSim.nbasis, devSim.nbasis);
#else
                            QUICKDouble DENSEKI = (QUICKDouble) LOC2(devSim.dense, KKK, III, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSEKJ = (QUICKDouble) LOC2(devSim.dense, KKK, JJJ, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSELJ = (QUICKDouble) LOC2(devSim.dense, LLL, JJJ, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSELI = (QUICKDouble) LOC2(devSim.dense, LLL, III, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSELK = (QUICKDouble) LOC2(devSim.dense, LLL, KKK, devSim.nbasis, devSim.nbasis);
                            QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ, III, devSim.nbasis, devSim.nbasis);
#endif

                            // ATOMIC ADD VALUE 1
                            temp = (KKK == LLL) ? DENSELK * Y : 2.0 * DENSELK * Y;
                            o_JI += temp;
#if defined(OSHELL)
                            ob_JI += temp;
#endif

                            // ATOMIC ADD VALUE 2
                            if (LLL != JJJ || III != KKK) {
                                temp = (III == JJJ) ? DENSEJI * Y : 2.0 * DENSEJI * Y;
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(devSim.oULL, LLL, KKK, devSim.nbasis, devSim.nbasis), temp, OSCALE);
#  else
                                atomicAdd(&LOC2(devSim.o, LLL, KKK, devSim.nbasis, devSim.nbasis), temp);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(devSim.obULL, LLL, KKK, devSim.nbasis, devSim.nbasis), temp, OSCALE);
#  else
                                atomicAdd(&LOC2(devSim.ob, LLL, KKK, devSim.nbasis, devSim.nbasis), temp);
#  endif
#endif
                            }

                            // ATOMIC ADD VALUE 3
#if defined(OSHELL)
                            temp = (III == KKK && III < JJJ && JJJ < LLL)
                                ? -2.0 * devSim.hyb_coeff * DENSELJA * Y : -(devSim.hyb_coeff * DENSELJA * Y);
                            temp2 = (III == KKK && III < JJJ && JJJ < LLL)
                                ? -2.0 * devSim.hyb_coeff * DENSELJB * Y : -(devSim.hyb_coeff * DENSELJB * Y);
                            o_KI += temp;
                            ob_KI += temp2;
#else
                            temp = (III == KKK && III < JJJ && JJJ < LLL)
                                ? -(devSim.hyb_coeff * DENSELJ * Y) : -0.5 * devSim.hyb_coeff * DENSELJ * Y;
                            o_KI += temp;
#endif

                            // ATOMIC ADD VALUE 4
                            if (KKK != LLL) {
#if defined(OSHELL)
                                temp = -(devSim.hyb_coeff * DENSEKJA * Y);
                                temp2 = -(devSim.hyb_coeff * DENSEKJB * Y);
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(devSim.oULL, LLL, III, devSim.nbasis, devSim.nbasis), temp, OSCALE);
                                GPUATOMICADD(&LOC2(devSim.obULL, LLL, III, devSim.nbasis, devSim.nbasis), temp2, OSCALE);
#  else
                                atomicAdd(&LOC2(devSim.o, LLL, III, devSim.nbasis, devSim.nbasis), temp);
                                atomicAdd(&LOC2(devSim.ob, LLL, III, devSim.nbasis, devSim.nbasis), temp2);
#  endif
#else
                                temp = -0.5 * devSim.hyb_coeff * DENSEKJ * Y;
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(devSim.oULL, LLL, III, devSim.nbasis, devSim.nbasis), temp, OSCALE);
#  else
                                atomicAdd(&LOC2(devSim.o, LLL, III, devSim.nbasis, devSim.nbasis), temp);
#  endif
#endif
                            }

                            // ATOMIC ADD VALUE 5
#if defined(OSHELL)
                            temp = -(devSim.hyb_coeff * DENSELIA * Y);
                            temp2 = -(devSim.hyb_coeff * DENSELIB * Y);
#else
                            temp = -0.5 * devSim.hyb_coeff * DENSELI * Y;
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
                                temp = -(devSim.hyb_coeff * DENSEKIA * Y);
                                temp2 = -(devSim.hyb_coeff * DENSEKIB * Y);
#else
                                temp = -0.5 * devSim.hyb_coeff * DENSEKI * Y;
#endif
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(devSim.oULL, MAX(JJJ, LLL), MIN(JJJ, LLL), devSim.nbasis, devSim.nbasis), temp, OSCALE);
#  else
                                atomicAdd(&LOC2(devSim.o, MAX(JJJ, LLL), MIN(JJJ, LLL), devSim.nbasis, devSim.nbasis), temp);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(devSim.obULL, MAX(JJJ, LLL), MIN(JJJ, LLL), devSim.nbasis, devSim.nbasis), temp2, OSCALE);
#  else
                                atomicAdd(&LOC2(devSim.ob, MAX(JJJ, LLL), MIN(JJJ, LLL), devSim.nbasis, devSim.nbasis), temp2);
#  endif
#endif

                                // ATOMIC ADD VALUE 6 - 2
                                if (JJJ == LLL && III != KKK) {
#  if defined(USE_LEGACY_ATOMICS)
                                    GPUATOMICADD(&LOC2(devSim.oULL, LLL, JJJ, devSim.nbasis, devSim.nbasis), temp, OSCALE);
#  else
                                    atomicAdd(&LOC2(devSim.o, LLL, JJJ, devSim.nbasis, devSim.nbasis), temp);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                                    GPUATOMICADD(&LOC2(devSim.obULL, LLL, JJJ, devSim.nbasis, devSim.nbasis), temp2, OSCALE);
#  else
                                    atomicAdd(&LOC2(devSim.ob, LLL, JJJ, devSim.nbasis, devSim.nbasis), temp2);
#  endif
#endif
                                }
                            }
                        }
                    }
                }

#  if defined(USE_LEGACY_ATOMICS)
                GPUATOMICADD(&LOC2(devSim.oULL, KKK, III, devSim.nbasis, devSim.nbasis), o_KI, OSCALE);
                GPUATOMICADD(&LOC2(devSim.oULL, MAX(JJJ, KKK), MIN(JJJ, KKK), devSim.nbasis, devSim.nbasis), o_JK_MM, OSCALE);
                GPUATOMICADD(&LOC2(devSim.oULL, JJJ, KKK, devSim.nbasis, devSim.nbasis), o_JK, OSCALE);
#  else
                atomicAdd(&LOC2(devSim.o, KKK, III, devSim.nbasis, devSim.nbasis), o_KI);
                atomicAdd(&LOC2(devSim.o, MAX(JJJ, KKK), MIN(JJJ, KKK), devSim.nbasis, devSim.nbasis), o_JK_MM);
                atomicAdd(&LOC2(devSim.o, JJJ, KKK, devSim.nbasis, devSim.nbasis), o_JK);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                GPUATOMICADD(&LOC2(devSim.obULL, KKK, III, devSim.nbasis, devSim.nbasis), ob_KI, OSCALE);
                GPUATOMICADD(&LOC2(devSim.obULL, MAX(JJJ, KKK), MIN(JJJ, KKK), devSim.nbasis, devSim.nbasis), ob_JK_MM, OSCALE);
                GPUATOMICADD(&LOC2(devSim.obULL, JJJ, KKK, devSim.nbasis, devSim.nbasis), ob_JK, OSCALE);
#  else
                atomicAdd(&LOC2(devSim.ob, KKK, III, devSim.nbasis, devSim.nbasis), ob_KI);
                atomicAdd(&LOC2(devSim.ob, MAX(JJJ, KKK), MIN(JJJ, KKK), devSim.nbasis, devSim.nbasis), ob_JK_MM);
                atomicAdd(&LOC2(devSim.ob, JJJ, KKK, devSim.nbasis, devSim.nbasis), ob_JK);
#  endif
#endif
            }

#  if defined(USE_LEGACY_ATOMICS)
            GPUATOMICADD(&LOC2(devSim.oULL, JJJ, III, devSim.nbasis, devSim.nbasis), o_JI, OSCALE);
#  else
            atomicAdd(&LOC2(devSim.o, JJJ, III, devSim.nbasis, devSim.nbasis), o_JI);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
            GPUATOMICADD(&LOC2(devSim.obULL, JJJ, III, devSim.nbasis, devSim.nbasis), ob_JI, OSCALE);
#  else
            atomicAdd(&LOC2(devSim.ob, JJJ, III, devSim.nbasis, devSim.nbasis), ob_JI);
#  endif
#endif
        }
    }
}


/*
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
 
 so first, kernel 0: zone 0,1,2,3,4 (get2e_kernel()), if no f, then that's it.
 second,   kernel 1: zone 1,3,4(get2e_kernel_spdf())
 then,     kernel 2: zone 2,3,4(get2e_kernel_spdf2())
 then,     kernel 3: zone 3,4(get2e_kernel_spdf3())
 finally,  kernel 4: zone 4(get2e_kernel_spdf4())

 */
#if defined(OSHELL)
  #if defined(int_sp)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_sp()
  #elif defined(int_spd)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spd()
  #elif defined(int_spdf)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf()
  #elif defined(int_spdf2)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf2()
  #elif defined(int_spdf3)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf3()
  #elif defined(int_spdf4)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf4()
  #elif defined(int_spdf5)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf5()
  #elif defined(int_spdf6)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf6()
  #elif defined(int_spdf7)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf7()
  #elif defined(int_spdf8)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf8()
  #elif defined(int_spdf9)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf9()
  #elif defined(int_spdf10)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf10()
  #endif
#else
  #if defined(int_sp)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_sp()
  #elif defined(int_spd)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spd()
  #elif defined(int_spdf)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf()
  #elif defined(int_spdf2)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf2()
  #elif defined(int_spdf3)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf3()
  #elif defined(int_spdf4)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf4()
  #elif defined(int_spdf5)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf5()
  #elif defined(int_spdf6)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf6()
  #elif defined(int_spdf7)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf7()
  #elif defined(int_spdf8)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf8()
  #elif defined(int_spdf9)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf9()
  #elif defined(int_spdf10)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf10()
  #endif
#endif
{
    // jshell and jshell2 defines the regions in i+j and k+l axes respectively.    
    // sqrQshell= Qshell x Qshell; where Qshell is the number of sorted shells (see gpu_upload_basis_ in gpu.cu)
    // for details on sorting. 
#if defined(int_sp)
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell;
#elif defined(int_spd)
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
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell;
#elif defined(int_spdf)
/*  
 Here we walk through following region of the cutoff matrix.

 --sqrQshell --
 _______________ 
 |             |  
 |             |  
 |             |  
 |_____________|  
 |             |  | sqrQshell - fStart
 |_____________|  |

*/
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - devSim.fStart;
#elif defined(int_spdf2)
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - devSim.fStart;
#elif defined(int_spdf3)
    const QUICKULL jshell0 = (QUICKULL) devSim.fStart;
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell0;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell0;
#elif defined(int_spdf4)
    const QUICKULL jshell0 = (QUICKULL) devSim.fStart;
    const QUICKULL jshell00 = (QUICKULL) devSim.ffStart;
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell00;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell0;
#elif defined(int_spdf5)
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - devSim.ffStart;
#elif defined(int_spdf6)
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - devSim.ffStart;
#elif defined(int_spdf7)
    const QUICKULL jshell0 = (QUICKULL) devSim.fStart;
    const QUICKULL jshell00 = (QUICKULL) devSim.ffStart;
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell0;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell00;
#elif defined(int_spdf8)
    const QUICKULL jshell0 = (QUICKULL) devSim.ffStart;
    const QUICKULL jshell00 = (QUICKULL) devSim.ffStart;
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell00;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell0;
#elif defined(int_spdf9)
    const QUICKULL jshell0 = (QUICKULL) devSim.ffStart;
    const QUICKULL jshell00 = (QUICKULL) devSim.ffStart;
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell00;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell0;
#elif defined(int_spdf10)
    const QUICKULL jshell0 = (QUICKULL) devSim.ffStart;
    const QUICKULL jshell00 = (QUICKULL) devSim.ffStart;
    const QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell00;
    const QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell0;
#endif

    for (QUICKULL i = blockIdx.x * blockDim.x + threadIdx.x; i < jshell * jshell2; i += blockDim.x * gridDim.x) {
#if defined(int_sp) || defined(int_spd)
        // Zone 0
        const QUICKULL a = i / jshell;
        const QUICKULL b = i - a * jshell;
#elif defined(int_spdf)
        // Zone 1
        QUICKULL b = i / jshell;
        const QUICKULL a = i - b * jshell;
        b += (QUICKULL) devSim.fStart;
#elif defined(int_spdf2)
        // Zone 2
        QUICKULL a = i / jshell;
        const QUICKULL b = i - a * jshell;
        a += (QUICKULL) devSim.fStart;
#elif defined(int_spdf3)
        // Zone 3
        QUICKULL a, b;
        if (jshell != 0) {
            a = i / jshell;
            b = i - a * jshell + jshell0;
            a += jshell0;
        } else {
            a = 0;
            b = 0;
        }
#elif defined(int_spdf4)
        // Zone 4
        QUICKULL a, b;
        if (jshell2 != 0) {
            a = i / jshell2;
            b = i - a * jshell2 + jshell0;
            a += jshell00;
        } else {
            a = 0;
            b = 0;
        }
#elif defined(int_spdf5)
        // Zone 5
        QUICKULL b = i / jshell;
        const QUICKULL a = i - b * jshell;
        b += (QUICKULL) devSim.ffStart;
#elif defined(int_spdf6)
        // Zone 2
        QUICKULL a = i / jshell;
        const QUICKULL b = i - a * jshell;
        a += (QUICKULL) devSim.ffStart;
#elif defined(int_spdf7)
        // Zone 3
        QUICKULL a, b;
        if (jshell != 0) {
            a = i / jshell;
            b = i - a * jshell + jshell00;
            a += jshell0;
        } else {
            a = 0;
            b = 0;
        }
#elif defined(int_spdf8) || defined(int_spdf9) || defined(int_spdf10)
        // Zone 4
        QUICKULL a, b;
        if (jshell2 != 0) {
            a = i / jshell2;
            b = i - a * jshell2 + jshell0;
            a += jshell00;
        } else {
            a = 0;
            b = 0;
        }
#endif

#if defined(MPIV_GPU)
        if (devSim.mpi_bcompute[a] > 0) {
#endif 
        const int II = devSim.sorted_YCutoffIJ[a].x;
        const int KK = devSim.sorted_YCutoffIJ[b].x;        
        const uint32_t ii = devSim.sorted_Q[II];
        const uint32_t kk = devSim.sorted_Q[KK];
        
        if (ii <= kk) {
            const int JJ = devSim.sorted_YCutoffIJ[a].y;            
            const int LL = devSim.sorted_YCutoffIJ[b].y;

            const uint8_t iii = devSim.sorted_Qnumber[II];
            const uint8_t jjj = devSim.sorted_Qnumber[JJ];
            const uint8_t kkk = devSim.sorted_Qnumber[KK];
            const uint8_t lll = devSim.sorted_Qnumber[LL];

#if defined(int_sp)
            if (iii < 2 && jjj < 2 && kkk < 2 && lll < 2) {
#endif
#if defined(int_spd)
            if (!(iii < 2 && jjj < 2 && kkk < 2 && lll < 2)) {
#endif
            const uint32_t jj = devSim.sorted_Q[JJ];
            const uint32_t ll = devSim.sorted_Q[LL];
            const uint32_t nshell = devSim.nshell;
#if defined(USE_TEXTURE)
            int2 tmpInt2Val;
            QUICKDouble val_ii_jj;
            QUICKDouble val_kk_ll;
#endif
#if defined(USE_TEXTURE) && defined(USE_TEXTURE_CUTMATRIX)
            tmpInt2Val = tex1Dfetch(tex_cutMatrix, ii + jj * nshell);
            val_ii_jj = __hiloint2double(tmpInt2Val.y, tmpInt2Val.x);

            tmpInt2Val = tex1Dfetch(tex_cutMatrix, kk + ll * nshell);
            val_kk_ll = __hiloint2double(tmpInt2Val.y, tmpInt2Val.x);

            tmpInt2Val = tex1Dfetch(tex_cutMatrix, ii + ll * nshell);
            QUICKDouble val_ii_ll = __hiloint2double(tmpInt2Val.y, tmpInt2Val.x);

            tmpInt2Val = tex1Dfetch(tex_cutMatrix, ii + kk * nshell);
            QUICKDouble val_ii_kk = __hiloint2double(tmpInt2Val.y, tmpInt2Val.x);

            tmpInt2Val = tex1Dfetch(tex_cutMatrix, jj + kk * nshell);
            QUICKDouble val_jj_kk = __hiloint2double(tmpInt2Val.y, tmpInt2Val.x);

            tmpInt2Val = tex1Dfetch(tex_cutMatrix, jj + ll * nshell);
            QUICKDouble val_jj_ll = __hiloint2double(tmpInt2Val.y, tmpInt2Val.x);

            QUICKDouble DNMax = MAX(MAX(4.0 * val_ii_jj, 4.0 * val_kk_ll),
                    MAX(MAX(val_ii_ll, val_ii_kk), MAX(val_jj_kk, val_jj_ll)));
#else
            const QUICKDouble DNMax = MAX(
                    MAX(4.0 * LOC2(devSim.cutMatrix, ii, jj, nshell, nshell), 4.0 * LOC2(devSim.cutMatrix, kk, ll, nshell, nshell)),
                    MAX(MAX(LOC2(devSim.cutMatrix, ii, ll, nshell, nshell), LOC2(devSim.cutMatrix, ii, kk, nshell, nshell)),
                        MAX(LOC2(devSim.cutMatrix, jj, kk, nshell, nshell), LOC2(devSim.cutMatrix, jj, ll, nshell, nshell))));
#endif

#if defined(USE_TEXTURE) && defined(USE_TEXTURE_YCUTOFF)
            tmpInt2Val = tex1Dfetch(tex_YCutoff, kk + ll * nshell);
            val_kk_ll = __hiloint2double(tmpInt2Val.y, tmpInt2Val.x);

            tmpInt2Val = tex1Dfetch(tex_YCutoff, ii + jj * nshell);
            val_ii_jj = __hiloint2double(tmpInt2Val.y, tmpInt2Val.x);

            if ((val_kk_ll * val_ii_jj) > devSim.integralCutoff
                    && (val_kk_ll * val_ii_jj * DNMax) > devSim.integralCutoff) {

#else
            if ((LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell))
                    > devSim.integralCutoff
                && (LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell) * DNMax)
                    > devSim.integralCutoff) {
#endif                
#if defined(OSHELL)
  #if defined(int_sp)
                iclass_oshell_sp(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
  #elif defined(int_spd)
                iclass_oshell_spd(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
  #elif defined(int_spdf)
                if (kkk + lll <= 6 && kkk + lll > 4) {
                    iclass_oshell_spdf(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf2)
                if (iii + jjj > 4 && iii + jjj <= 6 ) {
                    iclass_oshell_spdf2(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf3)
                if (iii + jjj >= 5 && iii + jjj <= 6 && kkk + lll <= 6 && kkk + lll >= 5) {
                    iclass_oshell_spdf3(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf4)
                if (iii + jjj == 6 && kkk + lll <= 6 && kkk + lll >= 5) {
                    iclass_oshell_spdf4(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf5)
                if (kkk + lll == 6 && iii + jjj >= 4 && iii + jjj <= 6) {
                    iclass_oshell_spdf5(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf6)
                if (iii + jjj == 6 && kkk + lll <= 6 && kkk + lll >= 4) {
                    iclass_oshell_spdf6(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf7)
                if (iii + jjj >= 5 && iii + jjj <= 6 && kkk + lll == 6) {
                    iclass_oshell_spdf7(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf8)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_oshell_spdf8(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf9)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_oshell_spdf9(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf10)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_oshell_spdf10(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #endif
#else          
  #if defined(int_sp)
                if (check_iclass(iii, jjj, kkk, lll, ii, jj, kk, ll) == true)
                    iclass_cshell_sp(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
  #elif defined(int_spd)
                if (check_iclass(iii, jjj, kkk, lll, ii, jj, kk, ll) == true)
                    iclass_cshell_spd(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
  #elif defined(int_spdf)
                if (kkk + lll <= 6 && kkk + lll > 4) {
                    iclass_cshell_spdf(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf2)
                if (iii + jjj > 4 && iii + jjj <= 6 ) {
                    iclass_cshell_spdf2(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf3)
                if (iii + jjj >= 5 && iii + jjj <= 6 && kkk + lll <= 6 && kkk + lll >= 5) {
                    iclass_cshell_spdf3(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf4)
                if (iii + jjj == 6 && kkk + lll <= 6 && kkk + lll >= 5) {
                    iclass_cshell_spdf4(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf5)
                if (kkk + lll == 6 && iii + jjj >= 4 && iii + jjj <= 6) {
                    iclass_cshell_spdf5(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf6)
                if (iii + jjj == 6 && kkk + lll <= 6 && kkk + lll >= 4) {
                    iclass_cshell_spdf6(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf7)
                if (iii + jjj >= 5 && iii + jjj <= 6 && kkk + lll == 6) {
                    iclass_cshell_spdf7(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf8)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_cshell_spdf8(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf9)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_cshell_spdf9(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #elif defined(int_spdf10)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_cshell_spdf10(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax);
                }
  #endif
#endif
            }
#if defined(int_sp) || defined(int_spd)
        }
#endif
        }
#if defined(MPIV_GPU)
        }      
#endif        
    }
}


#if defined(COMPILE_GPU_AOINT)
  #if !defined(OSHELL) && !defined(int_sp)
    #if defined(int_spd)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel
    #elif defined(int_spdf)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf
    #elif defined(int_spdf2)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf2
    #elif defined(int_spdf3)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf3
    #elif defined(int_spdf4)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf4
    #elif defined(int_spdf5)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf5
    #elif defined(int_spdf6)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf6
    #elif defined(int_spdf7)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf7
    #elif defined(int_spdf8)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf8
    #elif defined(int_spdf9)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf9
    #elif defined(int_spdf10)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf10
    #endif
    (QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
{
    unsigned int offside = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    QUICKULL myInt = (QUICKULL) ((intEnd - intStart + 1) / totalThreads);
    
    if (intEnd - intStart + 1 - myInt * totalThreads > offside)
        myInt++;
    
    for (QUICKULL i = 1; i <= myInt; i++) {
        QUICKULL currentInt = (QUICKULL) totalThreads * (i - 1) + (QUICKULL) offside + intStart;
        QUICKULL a = currentInt / jshell;
        QUICKULL b = currentInt - a * jshell;
        
        int II = devSim.sorted_YCutoffIJ[a].x;
        int JJ = devSim.sorted_YCutoffIJ[a].y;
        int KK = devSim.sorted_YCutoffIJ[b].x;
        int LL = devSim.sorted_YCutoffIJ[b].y;
        
        uint32_t ii = devSim.sorted_Q[II];
        uint32_t jj = devSim.sorted_Q[JJ];
        uint32_t kk = devSim.sorted_Q[KK];
        uint32_t ll = devSim.sorted_Q[LL];
        
        if (ii <= kk) {
            uint32_t nshell = devSim.nshell;
            
            if ((LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell))
                    > devSim.leastIntegralCutoff) {
                const uint8_t iii = devSim.sorted_Qnumber[II];
                const uint8_t jjj = devSim.sorted_Qnumber[JJ];
                const uint8_t kkk = devSim.sorted_Qnumber[KK];
                const uint8_t lll = devSim.sorted_Qnumber[LL];
    #if defined(int_spd)
                iclass_AOInt(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                        devSim.YVerticalTemp + offside, devSim.store + offside);
    #elif defined(int_spdf)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            devSim.YVerticalTemp + offside, devSim.store + offside);
                }
    #elif defined(int_spdf2)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf2(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            devSim.YVerticalTemp + offside, devSim.store + offside);
                }
    #elif defined(int_spdf3)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf3(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            devSim.YVerticalTemp + offside, devSim.store + offside);
                }
    #elif defined(int_spdf4)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf4(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            devSim.YVerticalTemp + offside, devSim.store + offside);
                }
    #elif defined(int_spdf5)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf5(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            devSim.YVerticalTemp + offside, devSim.store + offside);
                }
    #elif defined(int_spdf6)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf6(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            devSim.YVerticalTemp + offside, devSim.store + offside);
                }
    #elif defined(int_spdf7)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf7(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            devSim.YVerticalTemp + offside, devSim.store + offside);
                }
    #elif defined(int_spdf8)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf8(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            devSim.YVerticalTemp + offside, devSim.store + offside);
                }
    #elif defined(int_spdf9)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf9(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            devSim.YVerticalTemp + offside, devSim.store + offside);
                }
    #elif defined(int_spdf10)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf10(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            devSim.YVerticalTemp + offside, devSim.store + offside);
                }
    #endif
            }
        }
    }
}


/*
 iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
 performance algrithem for electron intergral evaluation. See description below for details
 */
    #if defined(int_spd)
__device__ static inline void iclass_AOInt
    #elif defined(int_spdf)
__device__ static inline void iclass_AOInt_spdf
    #elif defined(int_spdf2)
__device__ static inline void iclass_AOInt_spdf2
    #elif defined(int_spdf3)
__device__ static inline void iclass_AOInt_spdf3
    #elif defined(int_spdf4)
__device__ static inline void iclass_AOInt_spdf4
    #elif defined(int_spdf5)
__device__ static inline void iclass_AOInt_spdf5
    #elif defined(int_spdf6)
__device__ static inline void iclass_AOInt_spdf6
    #elif defined(int_spdf7)
__device__ static inline void iclass_AOInt_spdf7
    #elif defined(int_spdf8)
__device__ static inline void iclass_AOInt_spdf8
    #elif defined(int_spdf9)
__device__ static inline void iclass_AOInt_spdf9
    #elif defined(int_spdf10)
__device__ static inline void iclass_AOInt_spdf10
    #endif
    (uint8_t I, uint8_t J, uint8_t K, uint8_t L, uint32_t II, uint32_t JJ, uint32_t KK, uint32_t LL,
     QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID, QUICKDouble* YVerticalTemp, QUICKDouble* store)
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
    
    QUICKDouble RCx = LOC2(devSim.xyz, 0, devSim.katom[KK], 3, devSim.natom);
    QUICKDouble RCy = LOC2(devSim.xyz, 1, devSim.katom[KK], 3, devSim.natom);
    QUICKDouble RCz = LOC2(devSim.xyz, 2, devSim.katom[KK], 3, devSim.natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    uint32_t kPrimI = devSim.kprim[II];
    uint32_t kPrimJ = devSim.kprim[JJ];
    uint32_t kPrimK = devSim.kprim[KK];
    uint32_t kPrimL = devSim.kprim[LL];
    
    uint32_t kStartI = devSim.kstart[II];
    uint32_t kStartJ = devSim.kstart[JJ];
    uint32_t kStartK = devSim.kstart[KK];
    uint32_t kStartL = devSim.kstart[LL];
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of GPU limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */

    /*
     Initial the neccessary element for
     */
    for (uint8_t i = Sumindex[K + 1] + 1; i <= Sumindex[K + L + 2]; i++) {
        for (uint8_t j = Sumindex[I + 1] + 1; j <= Sumindex[I + J + 2]; j++) {
            if (i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j - 1, i - 1, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }
    
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
        
        QUICKDouble AB = LOC2(devSim.expoSum, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start + III, jj_start + JJJ, devSim.prim_total, devSim.prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        QUICKDouble cutoffPrim = DNMax * LOC2(devSim.cutPrim, kStartI + III, kStartJ + JJJ, devSim.jbasis, devSim.jbasis);
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI + III, kStartJ + JJJ,
                I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);
        
        for (uint32_t j = 0; j < kPrimK * kPrimL; j++) {
            uint32_t LLL = (uint32_t) j / kPrimK;
            uint32_t KKK = (uint32_t) j - kPrimK * LLL;
            
            if (cutoffPrim * LOC2(devSim.cutPrim, kStartK + KKK, kStartL + LLL, devSim.jbasis, devSim.jbasis) > devSim.primLimit) {
                /*
                 CD = expo(L)+expo(K)
                 ABCD = 1/ (AB + CD) = 1 / (expo(I)+expo(J)+expo(K)+expo(L))
                 
                 `````````````````````````AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
                 Rou(Greek Letter) =   ----------- = ------------------------------------
                 `````````````````````````AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
                 
                 ```````````````````expo(I)+expo(J)                        expo(K)+expo(L)
                 ABcom = --------------------------------  CDcom = --------------------------------
                 `````````expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)
                 
                 ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))
                 */
                uint32_t kk_start = devSim.prim_start[KK];
                uint32_t ll_start = devSim.prim_start[LL];
                QUICKDouble CD = LOC2(devSim.expoSum, kk_start + KKK, ll_start + LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble ABCD = 1.0 / (AB + CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
                QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(devSim.Xcoeff, kStartK + KKK, kStartL + LLL,
                        K - devSim.Qstart[KK], L - devSim.Qstart[LL], devSim.jbasis, devSim.jbasis, 2, 2);
                
                /*
                 Q' is the weighting center of K and L
                 ```````````````````````````--->           --->
                 ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
                 Q = P'(K,L)  = ------------------------------
                 `````````````````````````expo(K) + expo(L)
                 
                 W' is the weight center for I, J, K, L
                 
                 ```````````````--->             --->             --->            --->
                 ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
                 W = -------------------------------------------------------------------
                 `````````````````````````expo(I) + expo(J) + expo(K) + expo(L)
                 ``````->  ->  2
                 RPQ =| P - Q |
                 
                 ```````````->  -> 2
                 T = ROU * | P - Q|
                 */
                QUICKDouble Qx = LOC2(devSim.weightedCenterX, kk_start + KKK, ll_start + LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qy = LOC2(devSim.weightedCenterY, kk_start + KKK, ll_start + LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qz = LOC2(devSim.weightedCenterZ, kk_start + KKK, ll_start + LLL, devSim.prim_total, devSim.prim_total);
                
                FmT(I + J + K + L, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)),
                        YVerticalTemp);

                for (uint32_t i = 0; i <= I + J + K + L; i++) {
                    VY(0, 0, i) *= X2;
                }
                
    #if defined(int_spd)
                vertical
    #elif defined(int_spdf)
                vertical_spdf
    #elif defined(int_spdf2)
                vertical_spdf2
    #elif defined(int_spdf3)
                vertical_spdf3
    #elif defined(int_spdf4)
                vertical_spdf4
    #elif defined(int_spdf5)
                vertical_spdf5
    #elif defined(int_spdf6)
                vertical_spdf6
    #elif defined(int_spdf7)
                vertical_spdf7
    #elif defined(int_spdf8)
                vertical_spdf8
    #elif defined(int_spdf9)
                vertical_spdf9
    #elif defined(int_spdf10)
                vertical_spdf10
    #endif
                    (I, J, K, L, YVerticalTemp, store,
                     Px - RAx, Py - RAy, Pz - RAz, (Px * AB + Qx * CD) * ABCD - Px,
                     (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                     Qx - RCx, Qy - RCy, Qz - RCz, (Px * AB + Qx * CD) * ABCD - Qx,
                     (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                     0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
            }
        }
    }
    
    QUICKDouble RBx = LOC2(devSim.xyz, 0, devSim.katom[JJ], 3, devSim.natom);
    QUICKDouble RBy = LOC2(devSim.xyz, 1, devSim.katom[JJ], 3, devSim.natom);
    QUICKDouble RBz = LOC2(devSim.xyz, 2, devSim.katom[JJ], 3, devSim.natom);
    QUICKDouble RDx = LOC2(devSim.xyz, 0, devSim.katom[LL], 3, devSim.natom);
    QUICKDouble RDy = LOC2(devSim.xyz, 1, devSim.katom[LL], 3, devSim.natom);
    QUICKDouble RDz = LOC2(devSim.xyz, 2, devSim.katom[LL], 3, devSim.natom);
    
    uint32_t III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    uint32_t III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    uint32_t JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    uint32_t JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);
    uint32_t KKK1 = LOC2(devSim.Qsbasis, KK, K, devSim.nshell, 4);
    uint32_t KKK2 = LOC2(devSim.Qfbasis, KK, K, devSim.nshell, 4);
    uint32_t LLL1 = LOC2(devSim.Qsbasis, LL, L, devSim.nshell, 4);
    uint32_t LLL2 = LOC2(devSim.Qfbasis, LL, L, devSim.nshell, 4);
    
    // Store generated ERI to buffer
    for (uint32_t III = III1; III <= III2; III++) {
        for (uint32_t JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            for (uint32_t KKK = MAX(III, KKK1); KKK <= KKK2; KKK++) {
                for (uint32_t LLL = MAX(KKK, LLL1); LLL <= LLL2; LLL++) {
                    if ((III < JJJ && III < KKK && KKK < LLL)
                            || (III < KKK || JJJ <= LLL)) {
    #if defined(int_spd)
                        QUICKDouble Y = (QUICKDouble) hrrwhole
    #else
                        QUICKDouble Y = (QUICKDouble) hrrwhole2
    #endif
                            (I, J, K, L,
                              III, JJJ, KKK, LLL, store,
                              RAx, RAy, RAz, RBx, RBy, RBz,
                              RCx, RCy, RCz, RDx, RDy, RDz);
                        
                        if (abs(Y) > devSim.maxIntegralCutoff) {
                            ERI_entry a;
                            a.value = Y;
                            a.IJ = III * devSim.nbasis + JJJ;
                            a.KL = KKK * devSim.nbasis + LLL;
                            
                            aoint_buffer[atomicAdd(&devSim.intCount[streamID], 1)] = a;
                        }
                    }
                }
            }
        }
    }
}
  #endif
#endif
