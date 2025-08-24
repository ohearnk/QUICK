//
//  gpu_get2e_subs.h
//  new_quick 2
//
//  Created by Yipu Miao on 6/18/13.
//
//

#include "gpu_common.h"

#undef STOREDIM
#undef VY
#undef LOCSTORE
#if defined(int_sp)
  #define STOREDIM STOREDIM_T
#elif defined(int_spd)
  #define STOREDIM STOREDIM_S
#else
  #define STOREDIM STOREDIM_L
#endif
#define LOCSTORE(A,i1,i2,d1,d2) ((A)[((i2) * (d1) + (i1)) * gridDim.x * blockDim.x])
#define VY(a,b,c) (YVerticalTemp[(c)])

#undef FMT_NAME
#define FMT_NAME FmT
#include "gpu_fmt.h"

#undef PRIM_INT_ERI_LEN
#if defined(int_sp)
  #define PRIM_INT_ERI_LEN (5)
#elif defined(int_spd)
  #define PRIM_INT_ERI_LEN (9)
#elif defined(int_spdf) || defined(int_spdf2) || defined(int_spdf3) || defined(int_spdf4) \
    || defined(int_spdf5) || defined(int_spdf6) || defined(int_spdf7) || defined(int_spdf8) \
    || defined(int_spdf9) || defined(int_spdf10)
  #define PRIM_INT_ERI_LEN (13)
#endif


#if !defined(__gpu_get2e_subs_h_)
  #define __gpu_get2e_subs_h_
  #if !defined(OSHELL)
__device__ static inline bool check_iclass(uint32_t I, uint32_t J, uint32_t K, uint32_t L,
        uint32_t II, uint32_t JJ, uint32_t KK, uint32_t LL, uint32_t nshell,
        uint32_t const * const Qsbasis, uint32_t const * const Qfbasis)
{
    bool ret = false;

    for (uint32_t III = LOC2(Qsbasis, II, I, nshell, 4);
            III <= LOC2(Qfbasis, II, I, nshell, 4); III++) {
        for (uint32_t JJJ = MAX(III, LOC2(Qsbasis, JJ, J, nshell, 4));
                JJJ <= LOC2(Qfbasis, JJ, J, nshell, 4); JJJ++) {
            for (uint32_t KKK = MAX(III, LOC2(Qsbasis, KK, K, nshell, 4));
                    KKK <= LOC2(Qfbasis, KK, K, nshell, 4); KKK++) {
                for (uint32_t LLL = MAX(KKK, LOC2(Qsbasis, LL, L, nshell, 4));
                        LLL <= LOC2(Qfbasis, LL, L, nshell, 4); LLL++) {
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
      (uint32_t I, uint32_t J, uint32_t K, uint32_t L, uint32_t II, uint32_t JJ, uint32_t KK, uint32_t LL,
       QUICKDouble DNMax, QUICKDouble hyb_coeff, uint32_t natom, uint32_t nbasis,
        uint32_t nshell, uint32_t jbasis, QUICKDouble const * const xyz,
        uint32_t const * const kstart, uint32_t const * const katom,
        uint32_t const * const kprim, uint32_t const * const Qstart,
        uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
        QUICKDouble const * const cons, uint32_t const * const KLMN,
        uint32_t prim_total, uint32_t const * const prim_start,
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL * const oULL,
  #if defined(OSHELL)
        QUICKULL * const obULL,
  #endif
#else
        QUICKDouble * const o,
  #if defined(OSHELL)
        QUICKDouble * const ob,
  #endif
#endif
        QUICKDouble * const dense,
#if defined(OSHELL)
        QUICKDouble * const denseb,
#endif
        QUICKDouble const * const Xcoeff, QUICKDouble const * const expoSum,
        QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
        QUICKDouble const * const weightedCenterZ, QUICKDouble const * const cutPrim,
        QUICKDouble integralCutoff, QUICKDouble primLimit, QUICKDouble * const store,
        uint32_t const * const trans, uint32_t const * const Sumindex)
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
    const QUICKDouble RAx = LOC2(xyz, 0, katom[II], 3, natom);
    const QUICKDouble RAy = LOC2(xyz, 1, katom[II], 3, natom);
    const QUICKDouble RAz = LOC2(xyz, 2, katom[II], 3, natom);
    const QUICKDouble RCx = LOC2(xyz, 0, katom[KK], 3, natom);
    const QUICKDouble RCy = LOC2(xyz, 1, katom[KK], 3, natom);
    const QUICKDouble RCz = LOC2(xyz, 2, katom[KK], 3, natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    const uint32_t kPrimI = kprim[II];
    const uint32_t kPrimJ = kprim[JJ];
    const uint32_t kPrimK = kprim[KK];
    const uint32_t kPrimL = kprim[LL];
    
    const uint32_t kStartI = kstart[II];
    const uint32_t kStartJ = kstart[JJ];
    const uint32_t kStartK = kstart[KK];
    const uint32_t kStartL = kstart[LL];
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of GPU limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
    */
    for (uint32_t i = Sumindex[K + 1] + 1; i <= Sumindex[K + L + 2]; i++) {
        for (uint32_t j = Sumindex[I + 1] + 1; j <= Sumindex[I + J + 2]; j++) {
            if (i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(&store[blockIdx.x * blockDim.x + threadIdx.x],
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
        const uint32_t ii_start = prim_start[II];
        const uint32_t jj_start = prim_start[JJ];
        
        const QUICKDouble AB = LOC2(expoSum, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        const QUICKDouble Px = LOC2(weightedCenterX, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        const QUICKDouble Py = LOC2(weightedCenterY, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        const QUICKDouble Pz = LOC2(weightedCenterZ, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        const QUICKDouble cutoffPrim = DNMax * LOC2(cutPrim, kStartI + III, kStartJ + JJJ, jbasis, jbasis);
        const QUICKDouble X1 = LOC4(Xcoeff, kStartI + III, kStartJ + JJJ, I - Qstart[II], J - Qstart[JJ],
                jbasis, jbasis, 2, 2);
        
        for (uint32_t j = 0; j < kPrimK * kPrimL; j++) {
            const uint32_t LLL = j / kPrimK;
            const uint32_t KKK = j - kPrimK * LLL;
            
            if (cutoffPrim * LOC2(cutPrim, kStartK + KKK, kStartL + LLL, jbasis, jbasis) > primLimit) {
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
                const uint32_t kk_start = prim_start[KK];
                const uint32_t ll_start = prim_start[LL];
                const QUICKDouble CD = LOC2(expoSum, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                const QUICKDouble ABCD = 1.0 / (AB + CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
#if defined(USE_TEXTURE) && defined(USE_TEXTURE_XCOEFF)
                const int2 XcoeffInt2 = tex1Dfetch(tex_Xcoeff, L - Qstart[LL] +
                        (K - Qstart[KK] + ((kStartL + LLL) + (kStartK + KKK) * jbasis) * 2) * 2);
                const QUICKDouble X2 = sqrt(ABCD) * X1 * __hiloint2double(XcoeffInt2.y, XcoeffInt2.x);
#else
                const QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(Xcoeff, kStartK + KKK, kStartL + LLL,
                        K - Qstart[KK], L - Qstart[LL], jbasis, jbasis, 2, 2);
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
                const QUICKDouble Qx = LOC2(weightedCenterX, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                const QUICKDouble Qy = LOC2(weightedCenterY, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                const QUICKDouble Qz = LOC2(weightedCenterZ, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                
                double YVerticalTemp[PRIM_INT_ERI_LEN];
                FmT(I + J + K + L, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)),
                        YVerticalTemp);

                for (uint32_t i = 0; i <= I + J + K + L; i++) {
                    YVerticalTemp[i] *= X2;
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
                     &store[blockIdx.x * blockDim.x + threadIdx.x],
                     YVerticalTemp);
            }
        }
    }
    
    const QUICKDouble RBx = LOC2(xyz, 0, katom[JJ], 3, natom);
    const QUICKDouble RBy = LOC2(xyz, 1, katom[JJ], 3, natom);
    const QUICKDouble RBz = LOC2(xyz, 2, katom[JJ], 3, natom);
    const QUICKDouble RDx = LOC2(xyz, 0, katom[LL], 3, natom);
    const QUICKDouble RDy = LOC2(xyz, 1, katom[LL], 3, natom);
    const QUICKDouble RDz = LOC2(xyz, 2, katom[LL], 3, natom);
    
    const uint32_t III1 = LOC2(Qsbasis, II, I, nshell, 4);
    const uint32_t III2 = LOC2(Qfbasis, II, I, nshell, 4);
    const uint32_t JJJ1 = LOC2(Qsbasis, JJ, J, nshell, 4);
    const uint32_t JJJ2 = LOC2(Qfbasis, JJ, J, nshell, 4);
    const uint32_t KKK1 = LOC2(Qsbasis, KK, K, nshell, 4);
    const uint32_t KKK2 = LOC2(Qfbasis, KK, K, nshell, 4);
    const uint32_t LLL1 = LOC2(Qsbasis, LL, L, nshell, 4);
    const uint32_t LLL2 = LOC2(Qfbasis, LL, L, nshell, 4);
    
//    QUICKDouble hybrid_coeff = 0.0;
//    if (method == HF) {
//        hybrid_coeff = 1.0;
//    } else if (method == B3LYP) {
//        hybrid_coeff = 0.2;
//    } else if (method == DFT) {
//        hybrid_coeff = 0.0;
//    } else if (method == LIBXC) {
//        hybrid_coeff = hyb_coeff;                        
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
                           RCx, RCy, RCz, RDx, RDy, RDz,
                           nbasis, cons, KLMN, store, trans);

                        if (abs(Y) > integralCutoff) {
#if defined(OSHELL)
                            QUICKDouble DENSELK = (QUICKDouble) (LOC2(dense, LLL, KKK, nbasis, nbasis)
                                    + LOC2(denseb, LLL, KKK, nbasis, nbasis));
                            QUICKDouble DENSEJI = (QUICKDouble) (LOC2(dense, JJJ, III, nbasis, nbasis)
                                    + LOC2(denseb, JJJ, III, nbasis, nbasis));

                            QUICKDouble DENSEKIA = (QUICKDouble) LOC2(dense, KKK, III, nbasis, nbasis);
                            QUICKDouble DENSEKJA = (QUICKDouble) LOC2(dense, KKK, JJJ, nbasis, nbasis);
                            QUICKDouble DENSELJA = (QUICKDouble) LOC2(dense, LLL, JJJ, nbasis, nbasis);
                            QUICKDouble DENSELIA = (QUICKDouble) LOC2(dense, LLL, III, nbasis, nbasis);

                            QUICKDouble DENSEKIB = (QUICKDouble) LOC2(denseb, KKK, III, nbasis, nbasis);
                            QUICKDouble DENSEKJB = (QUICKDouble) LOC2(denseb, KKK, JJJ, nbasis, nbasis);
                            QUICKDouble DENSELJB = (QUICKDouble) LOC2(denseb, LLL, JJJ, nbasis, nbasis);
                            QUICKDouble DENSELIB = (QUICKDouble) LOC2(denseb, LLL, III, nbasis, nbasis);
#else
                            QUICKDouble DENSEKI = (QUICKDouble) LOC2(dense, KKK, III, nbasis, nbasis);
                            QUICKDouble DENSEKJ = (QUICKDouble) LOC2(dense, KKK, JJJ, nbasis, nbasis);
                            QUICKDouble DENSELJ = (QUICKDouble) LOC2(dense, LLL, JJJ, nbasis, nbasis);
                            QUICKDouble DENSELI = (QUICKDouble) LOC2(dense, LLL, III, nbasis, nbasis);
                            QUICKDouble DENSELK = (QUICKDouble) LOC2(dense, LLL, KKK, nbasis, nbasis);
                            QUICKDouble DENSEJI = (QUICKDouble) LOC2(dense, JJJ, III, nbasis, nbasis);
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
                                GPUATOMICADD(&LOC2(oULL, LLL, KKK, nbasis, nbasis), temp, OSCALE);
#  else
                                atomicAdd(&LOC2(o, LLL, KKK, nbasis, nbasis), temp);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(obULL, LLL, KKK, nbasis, nbasis), temp, OSCALE);
#  else
                                atomicAdd(&LOC2(ob, LLL, KKK, nbasis, nbasis), temp);
#  endif
#endif
                            }

                            // ATOMIC ADD VALUE 3
#if defined(OSHELL)
                            temp = (III == KKK && III < JJJ && JJJ < LLL)
                                ? -2.0 * hyb_coeff * DENSELJA * Y : -(hyb_coeff * DENSELJA * Y);
                            temp2 = (III == KKK && III < JJJ && JJJ < LLL)
                                ? -2.0 * hyb_coeff * DENSELJB * Y : -(hyb_coeff * DENSELJB * Y);
                            o_KI += temp;
                            ob_KI += temp2;
#else
                            temp = (III == KKK && III < JJJ && JJJ < LLL)
                                ? -(hyb_coeff * DENSELJ * Y) : -0.5 * hyb_coeff * DENSELJ * Y;
                            o_KI += temp;
#endif

                            // ATOMIC ADD VALUE 4
                            if (KKK != LLL) {
#if defined(OSHELL)
                                temp = -(hyb_coeff * DENSEKJA * Y);
                                temp2 = -(hyb_coeff * DENSEKJB * Y);
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(oULL, LLL, III, nbasis, nbasis), temp, OSCALE);
                                GPUATOMICADD(&LOC2(obULL, LLL, III, nbasis, nbasis), temp2, OSCALE);
#  else
                                atomicAdd(&LOC2(o, LLL, III, nbasis, nbasis), temp);
                                atomicAdd(&LOC2(ob, LLL, III, nbasis, nbasis), temp2);
#  endif
#else
                                temp = -0.5 * hyb_coeff * DENSEKJ * Y;
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(oULL, LLL, III, nbasis, nbasis), temp, OSCALE);
#  else
                                atomicAdd(&LOC2(o, LLL, III, nbasis, nbasis), temp);
#  endif
#endif
                            }

                            // ATOMIC ADD VALUE 5
#if defined(OSHELL)
                            temp = -(hyb_coeff * DENSELIA * Y);
                            temp2 = -(hyb_coeff * DENSELIB * Y);
#else
                            temp = -0.5 * hyb_coeff * DENSELI * Y;
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
                                temp = -(hyb_coeff * DENSEKIA * Y);
                                temp2 = -(hyb_coeff * DENSEKIB * Y);
#else
                                temp = -0.5 * hyb_coeff * DENSEKI * Y;
#endif
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(oULL, MAX(JJJ, LLL), MIN(JJJ, LLL), nbasis, nbasis), temp, OSCALE);
#  else
                                atomicAdd(&LOC2(o, MAX(JJJ, LLL), MIN(JJJ, LLL), nbasis, nbasis), temp);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                                GPUATOMICADD(&LOC2(obULL, MAX(JJJ, LLL), MIN(JJJ, LLL), nbasis, nbasis), temp2, OSCALE);
#  else
                                atomicAdd(&LOC2(ob, MAX(JJJ, LLL), MIN(JJJ, LLL), nbasis, nbasis), temp2);
#  endif
#endif

                                // ATOMIC ADD VALUE 6 - 2
                                if (JJJ == LLL && III != KKK) {
#  if defined(USE_LEGACY_ATOMICS)
                                    GPUATOMICADD(&LOC2(oULL, LLL, JJJ, nbasis, nbasis), temp, OSCALE);
#  else
                                    atomicAdd(&LOC2(o, LLL, JJJ, nbasis, nbasis), temp);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                                    GPUATOMICADD(&LOC2(obULL, LLL, JJJ, nbasis, nbasis), temp2, OSCALE);
#  else
                                    atomicAdd(&LOC2(ob, LLL, JJJ, nbasis, nbasis), temp2);
#  endif
#endif
                                }
                            }
                        }
                    }
                }

#  if defined(USE_LEGACY_ATOMICS)
                GPUATOMICADD(&LOC2(oULL, KKK, III, nbasis, nbasis), o_KI, OSCALE);
                GPUATOMICADD(&LOC2(oULL, MAX(JJJ, KKK), MIN(JJJ, KKK), nbasis, nbasis), o_JK_MM, OSCALE);
                GPUATOMICADD(&LOC2(oULL, JJJ, KKK, nbasis, nbasis), o_JK, OSCALE);
#  else
                atomicAdd(&LOC2(o, KKK, III, nbasis, nbasis), o_KI);
                atomicAdd(&LOC2(o, MAX(JJJ, KKK), MIN(JJJ, KKK), nbasis, nbasis), o_JK_MM);
                atomicAdd(&LOC2(o, JJJ, KKK, nbasis, nbasis), o_JK);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
                GPUATOMICADD(&LOC2(obULL, KKK, III, nbasis, nbasis), ob_KI, OSCALE);
                GPUATOMICADD(&LOC2(obULL, MAX(JJJ, KKK), MIN(JJJ, KKK), nbasis, nbasis), ob_JK_MM, OSCALE);
                GPUATOMICADD(&LOC2(obULL, JJJ, KKK, nbasis, nbasis), ob_JK, OSCALE);
#  else
                atomicAdd(&LOC2(ob, KKK, III, nbasis, nbasis), ob_KI);
                atomicAdd(&LOC2(ob, MAX(JJJ, KKK), MIN(JJJ, KKK), nbasis, nbasis), ob_JK_MM);
                atomicAdd(&LOC2(ob, JJJ, KKK, nbasis, nbasis), ob_JK);
#  endif
#endif
            }

#  if defined(USE_LEGACY_ATOMICS)
            GPUATOMICADD(&LOC2(oULL, JJJ, III, nbasis, nbasis), o_JI, OSCALE);
#  else
            atomicAdd(&LOC2(o, JJJ, III, nbasis, nbasis), o_JI);
#  endif
#if defined(OSHELL)
#  if defined(USE_LEGACY_ATOMICS)
            GPUATOMICADD(&LOC2(obULL, JJJ, III, nbasis, nbasis), ob_JI, OSCALE);
#  else
            atomicAdd(&LOC2(ob, JJJ, III, nbasis, nbasis), ob_JI);
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
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_sp
  #elif defined(int_spd)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spd
  #elif defined(int_spdf)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf
  #elif defined(int_spdf2)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf2
  #elif defined(int_spdf3)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf3
  #elif defined(int_spdf4)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf4
  #elif defined(int_spdf5)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf5
  #elif defined(int_spdf6)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf6
  #elif defined(int_spdf7)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf7
  #elif defined(int_spdf8)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf8
  #elif defined(int_spdf9)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf9
  #elif defined(int_spdf10)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_oshell_spdf10
  #endif
#else
  #if defined(int_sp)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_sp
  #elif defined(int_spd)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spd
  #elif defined(int_spdf)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf
  #elif defined(int_spdf2)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf2
  #elif defined(int_spdf3)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf3
  #elif defined(int_spdf4)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf4
  #elif defined(int_spdf5)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf5
  #elif defined(int_spdf6)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf6
  #elif defined(int_spdf7)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf7
  #elif defined(int_spdf8)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf8
  #elif defined(int_spdf9)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf9
  #elif defined(int_spdf10)
__global__ void __launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) k_eri_cshell_spdf10
  #endif
#endif
    (QUICKDouble hyb_coeff, uint32_t natom, uint32_t nbasis,
        uint32_t nshell, uint32_t jbasis, QUICKDouble const * const xyz, uint32_t fStart,
        uint32_t ffStart, uint32_t const * const kstart, uint32_t const * const katom,
        uint32_t const * const kprim, uint32_t const * const Qstart,
        uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
        uint32_t const * const sorted_Qnumber, uint32_t const * const sorted_Q,
        QUICKDouble const * const cons, uint32_t const * const KLMN,
        uint32_t prim_total, uint32_t const * const prim_start,
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL * const oULL,
  #if defined(OSHELL)
        QUICKULL * const obULL,
  #endif
#else
        QUICKDouble * const o,
  #if defined(OSHELL)
        QUICKDouble * const ob,
  #endif
#endif
        QUICKDouble * const dense,
#if defined(OSHELL)
        QUICKDouble * const denseb,
#endif
        QUICKDouble const * const Xcoeff, QUICKDouble const * const expoSum,
        QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
        QUICKDouble const * const weightedCenterZ, uint32_t sqrQshell, int2 const * const sorted_YCutoffIJ,
        QUICKDouble const * const cutMatrix, QUICKDouble const * const YCutoff,
        QUICKDouble const * const cutPrim, QUICKDouble integralCutoff, QUICKDouble primLimit,
        QUICKDouble maxIntegralCutoff, QUICKDouble leastIntegralCutoff,
#if defined(MPIV_GPU)
        unsigned char const * const mpi_bcompute,
#endif
        QUICKDouble * const store, uint32_t const * const trans, uint32_t const * const Sumindex)
{
    // jshell and jshell2 defines the regions in i+j and k+l axes respectively.    
    // sqrQshell= Qshell x Qshell; where Qshell is the number of sorted shells (see gpu_upload_basis_ in gpu.cu)
    // for details on sorting. 
#if defined(int_sp)
    const QUICKULL jshell = (QUICKULL) sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell;
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
    const QUICKULL jshell = (QUICKULL) sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell;
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
    const QUICKULL jshell = (QUICKULL) sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell - fStart;
#elif defined(int_spdf2)
    const QUICKULL jshell = (QUICKULL) sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell - fStart;
#elif defined(int_spdf3)
    const QUICKULL jshell0 = (QUICKULL) fStart;
    const QUICKULL jshell = (QUICKULL) sqrQshell - jshell0;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell - jshell0;
#elif defined(int_spdf4)
    const QUICKULL jshell0 = (QUICKULL) fStart;
    const QUICKULL jshell00 = (QUICKULL) ffStart;
    const QUICKULL jshell = (QUICKULL) sqrQshell - jshell00;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell - jshell0;
#elif defined(int_spdf5)
    const QUICKULL jshell = (QUICKULL) sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell - ffStart;
#elif defined(int_spdf6)
    const QUICKULL jshell = (QUICKULL) sqrQshell;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell - ffStart;
#elif defined(int_spdf7)
    const QUICKULL jshell0 = (QUICKULL) fStart;
    const QUICKULL jshell00 = (QUICKULL) ffStart;
    const QUICKULL jshell = (QUICKULL) sqrQshell - jshell0;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell - jshell00;
#elif defined(int_spdf8)
    const QUICKULL jshell0 = (QUICKULL) ffStart;
    const QUICKULL jshell00 = (QUICKULL) ffStart;
    const QUICKULL jshell = (QUICKULL) sqrQshell - jshell00;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell - jshell0;
#elif defined(int_spdf9)
    const QUICKULL jshell0 = (QUICKULL) ffStart;
    const QUICKULL jshell00 = (QUICKULL) ffStart;
    const QUICKULL jshell = (QUICKULL) sqrQshell - jshell00;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell - jshell0;
#elif defined(int_spdf10)
    const QUICKULL jshell0 = (QUICKULL) ffStart;
    const QUICKULL jshell00 = (QUICKULL) ffStart;
    const QUICKULL jshell = (QUICKULL) sqrQshell - jshell00;
    const QUICKULL jshell2 = (QUICKULL) sqrQshell - jshell0;
#endif
    extern __shared__ uint32_t smem[];
    uint32_t *strans = smem;
    uint32_t *sSumindex = &strans[TRANSDIM * TRANSDIM * TRANSDIM];
    uint32_t *sKLMN = &sSumindex[10];

    for (int i = threadIdx.x; i < TRANSDIM * TRANSDIM * TRANSDIM; i += blockDim.x) {
        strans[i] = trans[i];
    }
    for (int i = threadIdx.x; i < 10; i += blockDim.x) {
        sSumindex[i] = Sumindex[i];
    }
    for (int i = threadIdx.x; i < 3 * (int) nbasis; i += blockDim.x) {
        sKLMN[i] = KLMN[i];
    }

    __syncthreads();

    for (QUICKULL i = blockIdx.x * blockDim.x + threadIdx.x; i < jshell * jshell2; i += blockDim.x * gridDim.x) {
#if defined(int_sp) || defined(int_spd)
        // Zone 0
        const QUICKULL a = i / jshell;
        const QUICKULL b = i - a * jshell;
#elif defined(int_spdf)
        // Zone 1
        QUICKULL b = i / jshell;
        const QUICKULL a = i - b * jshell;
        b += (QUICKULL) fStart;
#elif defined(int_spdf2)
        // Zone 2
        QUICKULL a = i / jshell;
        const QUICKULL b = i - a * jshell;
        a += (QUICKULL) fStart;
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
        b += (QUICKULL) ffStart;
#elif defined(int_spdf6)
        // Zone 2
        QUICKULL a = i / jshell;
        const QUICKULL b = i - a * jshell;
        a += (QUICKULL) ffStart;
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
        if (mpi_bcompute[a] > 0) {
#endif 
        const int II = sorted_YCutoffIJ[a].x;
        const int KK = sorted_YCutoffIJ[b].x;        
        const uint32_t ii = sorted_Q[II];
        const uint32_t kk = sorted_Q[KK];
        
        if (ii <= kk) {
            const int JJ = sorted_YCutoffIJ[a].y;            
            const int LL = sorted_YCutoffIJ[b].y;

            const uint32_t iii = sorted_Qnumber[II];
            const uint32_t jjj = sorted_Qnumber[JJ];
            const uint32_t kkk = sorted_Qnumber[KK];
            const uint32_t lll = sorted_Qnumber[LL];

#if defined(int_sp)
            if (iii < 2 && jjj < 2 && kkk < 2 && lll < 2) {
#endif
#if defined(int_spd)
            if (!(iii < 2 && jjj < 2 && kkk < 2 && lll < 2)) {
#endif
            const uint32_t jj = sorted_Q[JJ];
            const uint32_t ll = sorted_Q[LL];
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
                    MAX(4.0 * LOC2(cutMatrix, ii, jj, nshell, nshell), 4.0 * LOC2(cutMatrix, kk, ll, nshell, nshell)),
                    MAX(MAX(LOC2(cutMatrix, ii, ll, nshell, nshell), LOC2(cutMatrix, ii, kk, nshell, nshell)),
                        MAX(LOC2(cutMatrix, jj, kk, nshell, nshell), LOC2(cutMatrix, jj, ll, nshell, nshell))));
#endif

#if defined(USE_TEXTURE) && defined(USE_TEXTURE_YCUTOFF)
            tmpInt2Val = tex1Dfetch(tex_YCutoff, kk + ll * nshell);
            val_kk_ll = __hiloint2double(tmpInt2Val.y, tmpInt2Val.x);

            tmpInt2Val = tex1Dfetch(tex_YCutoff, ii + jj * nshell);
            val_ii_jj = __hiloint2double(tmpInt2Val.y, tmpInt2Val.x);

            if ((val_kk_ll * val_ii_jj) > integralCutoff
                    && (val_kk_ll * val_ii_jj * DNMax) > integralCutoff) {

#else
            if ((LOC2(YCutoff, kk, ll, nshell, nshell) * LOC2(YCutoff, ii, jj, nshell, nshell))
                    > integralCutoff
                && (LOC2(YCutoff, kk, ll, nshell, nshell) * LOC2(YCutoff, ii, jj, nshell, nshell) * DNMax)
                    > integralCutoff) {
#endif                
#if defined(OSHELL)
  #if defined(int_sp)
                {
                    iclass_oshell_sp
  #elif defined(int_spd)
                {
                    iclass_oshell_spd
  #elif defined(int_spdf)
                if (kkk + lll <= 6 && kkk + lll > 4) {
                    iclass_oshell_spdf
  #elif defined(int_spdf2)
                if (iii + jjj > 4 && iii + jjj <= 6 ) {
                    iclass_oshell_spdf2
  #elif defined(int_spdf3)
                if (iii + jjj >= 5 && iii + jjj <= 6 && kkk + lll <= 6 && kkk + lll >= 5) {
                    iclass_oshell_spdf3
  #elif defined(int_spdf4)
                if (iii + jjj == 6 && kkk + lll <= 6 && kkk + lll >= 5) {
                    iclass_oshell_spdf4
  #elif defined(int_spdf5)
                if (kkk + lll == 6 && iii + jjj >= 4 && iii + jjj <= 6) {
                    iclass_oshell_spdf5
  #elif defined(int_spdf6)
                if (iii + jjj == 6 && kkk + lll <= 6 && kkk + lll >= 4) {
                    iclass_oshell_spdf6
  #elif defined(int_spdf7)
                if (iii + jjj >= 5 && iii + jjj <= 6 && kkk + lll == 6) {
                    iclass_oshell_spdf7
  #elif defined(int_spdf8)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_oshell_spdf8
  #elif defined(int_spdf9)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_oshell_spdf9
  #elif defined(int_spdf10)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_oshell_spdf10
  #endif
#else          
  #if defined(int_sp)
                if (check_iclass(iii, jjj, kkk, lll, ii, jj, kk, ll, nshell, Qsbasis, Qfbasis) == true) {
                    iclass_cshell_sp
  #elif defined(int_spd)
                if (check_iclass(iii, jjj, kkk, lll, ii, jj, kk, ll, nshell, Qsbasis, Qfbasis) == true) {
                    iclass_cshell_spd
  #elif defined(int_spdf)
                if (kkk + lll <= 6 && kkk + lll > 4) {
                    iclass_cshell_spdf
  #elif defined(int_spdf2)
                if (iii + jjj > 4 && iii + jjj <= 6) {
                    iclass_cshell_spdf2
  #elif defined(int_spdf3)
                if (iii + jjj >= 5 && iii + jjj <= 6 && kkk + lll <= 6 && kkk + lll >= 5) {
                    iclass_cshell_spdf3
  #elif defined(int_spdf4)
                if (iii + jjj == 6 && kkk + lll <= 6 && kkk + lll >= 5) {
                    iclass_cshell_spdf4
  #elif defined(int_spdf5)
                if (kkk + lll == 6 && iii + jjj >= 4 && iii + jjj <= 6) {
                    iclass_cshell_spdf5
  #elif defined(int_spdf6)
                if (iii + jjj == 6 && kkk + lll <= 6 && kkk + lll >= 4) {
                    iclass_cshell_spdf6
  #elif defined(int_spdf7)
                if (iii + jjj >= 5 && iii + jjj <= 6 && kkk + lll == 6) {
                    iclass_cshell_spdf7
  #elif defined(int_spdf8)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_cshell_spdf8
  #elif defined(int_spdf9)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_cshell_spdf9
  #elif defined(int_spdf10)
                if (iii + jjj == 6 && kkk + lll == 6) {
                    iclass_cshell_spdf10
  #endif
#endif
                        (iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, hyb_coeff, natom, nbasis,
                        nshell, jbasis, xyz, kstart, katom, kprim, Qstart, Qsbasis, Qfbasis,
                        cons, sKLMN, prim_total, prim_start,
#if defined(USE_LEGACY_ATOMICS)
                        oULL,
  #if defined(OSHELL)
                        obULL,
  #endif
#else
                        o,
  #if defined(OSHELL)
                        ob,
  #endif
#endif
                        dense,
#if defined(OSHELL)
                        denseb,
#endif
                        Xcoeff, expoSum, weightedCenterX, weightedCenterY,
                        weightedCenterZ, cutPrim, integralCutoff, primLimit,
                        store, strans, sSumindex);
                }
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
    
    QUICKULL jshell = (QUICKULL) sqrQshell;
    QUICKULL myInt = (QUICKULL) ((intEnd - intStart + 1) / totalThreads);
    
    if (intEnd - intStart + 1 - myInt * totalThreads > offside)
        myInt++;
    
    for (QUICKULL i = 1; i <= myInt; i++) {
        QUICKULL currentInt = (QUICKULL) totalThreads * (i - 1) + (QUICKULL) offside + intStart;
        QUICKULL a = currentInt / jshell;
        QUICKULL b = currentInt - a * jshell;
        
        int II = sorted_YCutoffIJ[a].x;
        int JJ = sorted_YCutoffIJ[a].y;
        int KK = sorted_YCutoffIJ[b].x;
        int LL = sorted_YCutoffIJ[b].y;
        
        uint32_t ii = sorted_Q[II];
        uint32_t jj = sorted_Q[JJ];
        uint32_t kk = sorted_Q[KK];
        uint32_t ll = sorted_Q[LL];
        
        if (ii <= kk) {
            uint32_t nshell = nshell;
            
            if ((LOC2(YCutoff, kk, ll, nshell, nshell) * LOC2(YCutoff, ii, jj, nshell, nshell))
                    > leastIntegralCutoff) {
                const uint32_t iii = sorted_Qnumber[II];
                const uint32_t jjj = sorted_Qnumber[JJ];
                const uint32_t kkk = sorted_Qnumber[KK];
                const uint32_t lll = sorted_Qnumber[LL];
    #if defined(int_spd)
                iclass_AOInt(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                        store + offside);
    #elif defined(int_spdf)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            store + offside);
                }
    #elif defined(int_spdf2)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf2(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            store + offside);
                }
    #elif defined(int_spdf3)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf3(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            store + offside);
                }
    #elif defined(int_spdf4)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf4(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            store + offside);
                }
    #elif defined(int_spdf5)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf5(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            store + offside);
                }
    #elif defined(int_spdf6)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf6(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            store + offside);
                }
    #elif defined(int_spdf7)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf7(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            store + offside);
                }
    #elif defined(int_spdf8)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf8(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            store + offside);
                }
    #elif defined(int_spdf9)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf9(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            store + offside);
                }
    #elif defined(int_spdf10)
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf10(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID,
                            store + offside);
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
    (uint32_t I, uint32_t J, uint32_t K, uint32_t L, uint32_t II, uint32_t JJ, uint32_t KK, uint32_t LL,
     QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID, QUICKDouble* store)
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
    
    QUICKDouble RCx = LOC2(xyz, 0, katom[KK], 3, natom);
    QUICKDouble RCy = LOC2(xyz, 1, katom[KK], 3, natom);
    QUICKDouble RCz = LOC2(xyz, 2, katom[KK], 3, natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    uint32_t kPrimI = kprim[II];
    uint32_t kPrimJ = kprim[JJ];
    uint32_t kPrimK = kprim[KK];
    uint32_t kPrimL = kprim[LL];
    
    uint32_t kStartI = kstart[II];
    uint32_t kStartJ = kstart[JJ];
    uint32_t kStartK = kstart[KK];
    uint32_t kStartL = kstart[LL];
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of GPU limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */

    /*
     Initial the neccessary element for
     */
    for (uint32_t i = Sumindex[K + 1] + 1; i <= Sumindex[K + L + 2]; i++) {
        for (uint32_t j = Sumindex[I + 1] + 1; j <= Sumindex[I + J + 2]; j++) {
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
        uint32_t ii_start = prim_start[II];
        uint32_t jj_start = prim_start[JJ];
        
        QUICKDouble AB = LOC2(expoSum, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        QUICKDouble Px = LOC2(weightedCenterX, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        QUICKDouble Py = LOC2(weightedCenterY, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        QUICKDouble Pz = LOC2(weightedCenterZ, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        QUICKDouble cutoffPrim = DNMax * LOC2(cutPrim, kStartI + III, kStartJ + JJJ, jbasis, jbasis);
        QUICKDouble X1 = LOC4(Xcoeff, kStartI + III, kStartJ + JJJ,
                I - Qstart[II], J - Qstart[JJ], jbasis, jbasis, 2, 2);
        
        for (uint32_t j = 0; j < kPrimK * kPrimL; j++) {
            uint32_t LLL = (uint32_t) j / kPrimK;
            uint32_t KKK = (uint32_t) j - kPrimK * LLL;
            
            if (cutoffPrim * LOC2(cutPrim, kStartK + KKK, kStartL + LLL, jbasis, jbasis) > primLimit) {
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
                uint32_t kk_start = prim_start[KK];
                uint32_t ll_start = prim_start[LL];
                QUICKDouble CD = LOC2(expoSum, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                QUICKDouble ABCD = 1.0 / (AB + CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
                QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(Xcoeff, kStartK + KKK, kStartL + LLL,
                        K - Qstart[KK], L - Qstart[LL], jbasis, jbasis, 2, 2);
                
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
                QUICKDouble Qx = LOC2(weightedCenterX, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                QUICKDouble Qy = LOC2(weightedCenterY, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                QUICKDouble Qz = LOC2(weightedCenterZ, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                
                double YVerticalTemp[PRIM_INT_ERI_LEN];
                FmT(I + J + K + L, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)),
                        YVerticalTemp);

                for (uint32_t i = 0; i <= I + J + K + L; i++) {
                    YVerticalTemp[i] *= X2;
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
    
    QUICKDouble RBx = LOC2(xyz, 0, katom[JJ], 3, natom);
    QUICKDouble RBy = LOC2(xyz, 1, katom[JJ], 3, natom);
    QUICKDouble RBz = LOC2(xyz, 2, katom[JJ], 3, natom);
    QUICKDouble RDx = LOC2(xyz, 0, katom[LL], 3, natom);
    QUICKDouble RDy = LOC2(xyz, 1, katom[LL], 3, natom);
    QUICKDouble RDz = LOC2(xyz, 2, katom[LL], 3, natom);
    
    uint32_t III1 = LOC2(Qsbasis, II, I, nshell, 4);
    uint32_t III2 = LOC2(Qfbasis, II, I, nshell, 4);
    uint32_t JJJ1 = LOC2(Qsbasis, JJ, J, nshell, 4);
    uint32_t JJJ2 = LOC2(Qfbasis, JJ, J, nshell, 4);
    uint32_t KKK1 = LOC2(Qsbasis, KK, K, nshell, 4);
    uint32_t KKK2 = LOC2(Qfbasis, KK, K, nshell, 4);
    uint32_t LLL1 = LOC2(Qsbasis, LL, L, nshell, 4);
    uint32_t LLL2 = LOC2(Qfbasis, LL, L, nshell, 4);
    
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
                        
                        if (abs(Y) > maxIntegralCutoff) {
                            ERI_entry a;
                            a.value = Y;
                            a.IJ = III * nbasis + JJJ;
                            a.KL = KKK * nbasis + LLL;
                            
                            aoint_buffer[atomicAdd(&intCount[streamID], 1)] = a;
                        }
                    }
                }
            }
        }
    }
}
  #endif
#endif
