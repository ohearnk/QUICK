//
//  gpu_get2e_subs_grad.h
//  new_quick
//
//  Created by Yipu Miao on 1/22/14.
//
//

#include "gpu_common.h"


#undef FMT_NAME
#define FMT_NAME FmT
#include "gpu_fmt.h"


#ifndef gpu_get2e_subs_grad_h
  #define gpu_get2e_subs_grad_h
  #undef STOREDIM
  #define STOREDIM STOREDIM_GRAD_T


__device__ static inline void hrrwholegrad_sp(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        QUICKDouble * const Yccx, QUICKDouble * const Yccy, QUICKDouble * const Yccz,
        uint8_t J, uint8_t L, uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble * const store, QUICKDouble * const storeAA,
        QUICKDouble * const storeBB, QUICKDouble * const storeCC,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz,
        uint32_t nbasis, QUICKDouble const * const cons, uint8_t const * const KLMN)
{
    uint8_t angularL[4], angularR[4];
    QUICKDouble coefAngularL[4], coefAngularR[4];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;

    uint8_t numAngularL, numAngularR;

    numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    //  Part A - x
    numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis) + 1, LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis) + 1, LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis),
            LOC2(KLMN,1,III,3,nbasis),
            LOC2(KLMN,2,III,3,nbasis) + 1,
            LOC2(KLMN,0,JJJ,3,nbasis),
            LOC2(KLMN,1,JJJ,3,nbasis),
            LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis) + 1, LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis) + 1, LOC2(KLMN,2,JJJ,3,nbasis),
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis) + 1,
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,0,III,3,nbasis) >= 1) {
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis) - 1, LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yaax = *Yaax - LOC2(KLMN,0,III,3,nbasis) * coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,1,III,3,nbasis) >= 1) {
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis) - 1, LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yaay = *Yaay - LOC2(KLMN,1,III,3,nbasis) * coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,2,III,3,nbasis) >= 1) {
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis) - 1,
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yaaz = *Yaaz - LOC2(KLMN,2,III,3,nbasis) * coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,0,JJJ,3,nbasis) >= 1) {
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis) - 1, LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J - 1, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Ybbx = *Ybbx - LOC2(KLMN,0,JJJ,3,nbasis) * coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,1,JJJ,3,nbasis) >= 1) {
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis) - 1, LOC2(KLMN,2,JJJ,3,nbasis),
                J - 1, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Ybby = *Ybby - LOC2(KLMN,1,JJJ,3,nbasis) * coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,2,JJJ,3,nbasis) >= 1) {
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis) - 1,
                J - 1, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Ybbz = *Ybbz - LOC2(KLMN,2,JJJ,3,nbasis) * coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    // KET PART =====================================
    // Part C - x
    numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis) + 1, LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,0,KKK,3,nbasis) >= 1) {
        numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(KLMN,0,KKK,3,nbasis) - 1, LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis),
                LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
                L, coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yccx = *Yccx - LOC2(KLMN,0,KKK,3,nbasis) * coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    // Part C - y
    numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis) + 1, LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,1,KKK,3,nbasis) >= 1) {
        numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis) - 1, LOC2(KLMN,2,KKK,3,nbasis),
                LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
                L, coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yccy = *Yccy - LOC2(KLMN,1,KKK,3,nbasis) * coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    // Part C - z
    numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis) + 1,
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,2,KKK,3,nbasis) >= 1) {
        numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis) - 1,
                LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
                L, coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yccz = *Yccz - LOC2(KLMN,2,KKK,3,nbasis) * coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    QUICKDouble constant = cons[III] * cons[JJJ] * cons[KKK] * cons[LLL];

    *Yaax *= constant;
    *Yaay *= constant;
    *Yaaz *= constant;

    *Ybbx *= constant;
    *Ybby *= constant;
    *Ybbz *= constant;

    *Yccx *= constant;
    *Yccy *= constant;
    *Yccz *= constant;
}


  #undef STOREDIM
  #define STOREDIM STOREDIM_S
__device__ static inline void hrrwholegrad(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        QUICKDouble * const Yccx, QUICKDouble * const Yccy, QUICKDouble * const Yccz,
        uint8_t J, uint8_t L, uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble * const store, QUICKDouble * const storeAA,
        QUICKDouble * const storeBB, QUICKDouble * const storeCC,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz,
        uint32_t nbasis, QUICKDouble const * const cons, uint8_t const * const KLMN)
{
    uint8_t angularL[8], angularR[8];
    QUICKDouble coefAngularL[8], coefAngularR[8];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;

    uint8_t numAngularL, numAngularR;

    numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    //  Part A - x
    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis) + 1, LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis) + 1, LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis) + 1,
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis) + 1, LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis) + 1, LOC2(KLMN,2,JJJ,3,nbasis),
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis) + 1,
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,0,III,3,nbasis) >= 1) {
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis) - 1, LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yaax = *Yaax - LOC2(KLMN,0,III,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,1,III,3,nbasis) >= 1) {
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis) - 1, LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yaay = *Yaay - LOC2(KLMN,1,III,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,2,III,3,nbasis) >= 1) {
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis) - 1,
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yaaz = *Yaaz - LOC2(KLMN,2,III,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,0,JJJ,3,nbasis) >= 1) {
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis) - 1, LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J - 1, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Ybbx = *Ybbx - LOC2(KLMN,0,JJJ,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,1,JJJ,3,nbasis) >= 1) {
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis) - 1, LOC2(KLMN,2,JJJ,3,nbasis),
                J - 1, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Ybby = *Ybby - LOC2(KLMN,1,JJJ,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,2,JJJ,3,nbasis) >= 1) {
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis) - 1,
                J - 1, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Ybbz = *Ybbz - LOC2(KLMN,2,JJJ,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    // KET PART =====================================
    // Part C - x
    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis) + 1, LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,0,KKK,3,nbasis) >= 1) {
        numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(KLMN,0,KKK,3,nbasis) - 1, LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis),
                LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
                L, coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yccx = *Yccx - LOC2(KLMN,0,KKK,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    // Part C - y
    numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis) + 1, LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,1,KKK,3,nbasis) >= 1) {
        numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis) - 1, LOC2(KLMN,2,KKK,3,nbasis),
                LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
                L, coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yccy = *Yccy - LOC2(KLMN,1,KKK,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    // Part C - z
    numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis) + 1,
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,2,KKK,3,nbasis) >= 1) {
        numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis) - 1,
                LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
                L, coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yccz = *Yccz - LOC2(KLMN,2,KKK,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    QUICKDouble constant = cons[III] * cons[JJJ] * cons[KKK] * cons[LLL];

    *Yaax *= constant;
    *Yaay *= constant;
    *Yaaz *= constant;

    *Ybbx *= constant;
    *Ybby *= constant;
    *Ybbz *= constant;

    *Yccx *= constant;
    *Yccy *= constant;
    *Yccz *= constant;
}


  #undef STOREDIM
  #define STOREDIM STOREDIM_XL
__device__ static inline void hrrwholegrad2(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        QUICKDouble * const Yccx, QUICKDouble * const Yccy, QUICKDouble * const Yccz,
        uint8_t J, uint8_t L, uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble * const store, QUICKDouble * const storeAA,
        QUICKDouble * const storeBB, QUICKDouble * const storeCC,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz,
        uint32_t nbasis, QUICKDouble const * const cons, uint8_t const * const KLMN)
{
    uint8_t angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;

    QUICKDouble constant = cons[III] * cons[JJJ] * cons[KKK] * cons[LLL];
    uint8_t numAngularL, numAngularR;

    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    //  Part A - x
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis) + 1, LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis) + 1, LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis) + 1,
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis) + 1, LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis) + 1, LOC2(KLMN,2,JJJ,3,nbasis),
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis) + 1,
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,0,III,3,nbasis) >= 1) {
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis) - 1, LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yaax = *Yaax - LOC2(KLMN,0,III,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,1,III,3,nbasis) >= 1) {
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis) - 1, LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yaay = *Yaay - LOC2(KLMN,1,III,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,2,III,3,nbasis) >= 1) {
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis) - 1,
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yaaz = *Yaaz - LOC2(KLMN,2,III,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,0,JJJ,3,nbasis) >= 1) {
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis) - 1, LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
                J - 1, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Ybbx = *Ybbx - LOC2(KLMN,0,JJJ,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,1,JJJ,3,nbasis) >= 1) {
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis) - 1, LOC2(KLMN,2,JJJ,3,nbasis),
                J - 1, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Ybby = *Ybby - LOC2(KLMN,1,JJJ,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    if (LOC2(KLMN,2,JJJ,3,nbasis) >= 1) {
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
                LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis) - 1,
                J - 1, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Ybbz = *Ybbz - LOC2(KLMN,2,JJJ,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    // KET PART =====================================
    // Part C - x
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis) + 1, LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,0,KKK,3,nbasis) >= 1) {
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(KLMN,0,KKK,3,nbasis) - 1, LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis),
                LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
                L, coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yccx = *Yccx - LOC2(KLMN,0,KKK,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    // Part C - y
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis) + 1, LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,1,KKK,3,nbasis) >= 1) {
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis) - 1, LOC2(KLMN,2,KKK,3,nbasis),
                LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
                L, coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yccy = *Yccy - LOC2(KLMN,1,KKK,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    // Part C - z
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis) + 1,
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,2,KKK,3,nbasis) >= 1) {
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis) - 1,
                LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
                L, coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    *Yccz = *Yccz - LOC2(KLMN,2,KKK,3,nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;

    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;

    *Yccx = *Yccx * constant;
    *Yccy = *Yccy * constant;
    *Yccz = *Yccz * constant;
}


  #undef STOREDIM
  #define STOREDIM STOREDIM_GRAD_S
__device__ static inline void hrrwholegrad2_1(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        QUICKDouble * const Yccx, QUICKDouble * const Yccy, QUICKDouble * const Yccz,
        uint8_t J, uint8_t L, uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble * const store, QUICKDouble * const storeAA,
        QUICKDouble * const storeBB, QUICKDouble * const storeCC,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz,
        uint32_t nbasis, QUICKDouble const * const cons, uint8_t const * const KLMN)
{
    uint8_t angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;

    uint8_t numAngularL, numAngularR;

    // KET PART =====================================
    // Part C - x
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis), LOC2(KLMN,1,III,3,nbasis), LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis), LOC2(KLMN,1,JJJ,3,nbasis), LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL);

    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis) + 1, LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    // Part C - y
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis) + 1, LOC2(KLMN,2,KKK,3,nbasis),
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    // Part C - z
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN,0,KKK,3,nbasis), LOC2(KLMN,1,KKK,3,nbasis), LOC2(KLMN,2,KKK,3,nbasis) + 1,
            LOC2(KLMN,0,LLL,3,nbasis), LOC2(KLMN,1,LLL,3,nbasis), LOC2(KLMN,2,LLL,3,nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    QUICKDouble constant = cons[III] * cons[JJJ] * cons[KKK] * cons[LLL];

    *Yaax *= constant;
    *Yaay *= constant;
    *Yaaz *= constant;

    *Ybbx *= constant;
    *Ybby *= constant;
    *Ybbz *= constant;

    *Yccx *= constant;
    *Yccy *= constant;
    *Yccz *= constant;
}


__device__ static inline void hrrwholegrad2_2(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        QUICKDouble * const Yccx, QUICKDouble * const Yccy, QUICKDouble * const Yccz,
        uint8_t J, uint8_t L, uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble * const store, QUICKDouble * const storeAA,
        QUICKDouble * const storeBB, QUICKDouble * const storeCC,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz,
        uint32_t nbasis, QUICKDouble const * const cons, uint8_t const * const KLMN)
{
    uint8_t angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;

    uint8_t numAngularL, numAngularR;

    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN, 0, KKK, 3, nbasis),
            LOC2(KLMN, 1, KKK, 3, nbasis),
            LOC2(KLMN, 2, KKK, 3, nbasis),
            LOC2(KLMN, 0, LLL, 3, nbasis),
            LOC2(KLMN, 1, LLL, 3, nbasis),
            LOC2(KLMN, 2, LLL, 3, nbasis),
            L, coefAngularR, angularR);

    //  Part A - x
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis) + 1,
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis) + 1,
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis) + 1,
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeAA, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis) + 1,
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis) + 1,
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis) + 1,
            J + 1, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(storeBB, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    QUICKDouble constant = cons[III] * cons[JJJ] * cons[KKK] * cons[LLL];

    *Yaax *= constant;
    *Yaay *= constant;
    *Yaaz *= constant;

    *Ybbx *= constant;
    *Ybby *= constant;
    *Ybbz *= constant;

    *Yccx *= constant;
    *Yccy *= constant;
    *Yccz *= constant;
}
#endif


#undef STOREDIM
#undef PRIM_INT_ERI_GRAD_LEN
#if defined(int_sp)
  #undef LOCSTORE
  #undef VY
  #define STOREDIM STOREDIM_GRAD_T
  #define LOCSTORE(A,i1,i2,d1,d2) (A[((i2) * (d1) + (i1)) * gridDim.x * blockDim.x])
  #define VY(a,b,c) (YVerticalTemp[(c)])
  #define PRIM_INT_ERI_GRAD_LEN (6)
#elif defined(int_spd)
  #undef VY
  #undef LOCSTORE
  #define STOREDIM STOREDIM_S
  #define LOCSTORE(A,i1,i2,d1,d2) (A[((i2) * (d1) + (i1)) * gridDim.x * blockDim.x])
  #define VY(a,b,c) (YVerticalTemp[(c)])
  #define PRIM_INT_ERI_GRAD_LEN (10)
#elif defined(int_spdf) || defined(int_spdf2)
  #define STOREDIM STOREDIM_GRAD_S
  #define PRIM_INT_ERI_GRAD_LEN (15)
#elif defined(int_spdf3) || defined(int_spdf4)
  #define STOREDIM STOREDIM_XL
  #define PRIM_INT_ERI_GRAD_LEN (15)
#else
  #define STOREDIM STOREDIM_L
  #define PRIM_INT_ERI_GRAD_LEN (15)
#endif


/*
   iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
   performance algrithem for electron intergral evaluation. See description below for details
   */
#if defined(int_sp)
  #if defined(OSHELL)
__device__ static inline void iclass_grad_oshell_sp
  #else
__device__ static inline void iclass_grad_cshell_sp
  #endif
#elif defined(int_spd)
  #if defined(OSHELL)
__device__ static inline void iclass_grad_oshell_spd
  #else
__device__ static inline void iclass_grad_cshell_spd
  #endif
#endif
#if defined(int_sp) || defined(int_spd)
    (uint8_t I, uint8_t J, uint8_t K, uint8_t L,
     uint32_t II, uint32_t JJ, uint32_t KK, uint32_t LL, QUICKDouble DNMax,
     QUICKDouble hyb_coeff, uint32_t natom, uint32_t nbasis,
     uint32_t nshell, uint32_t jbasis, QUICKDouble const * const xyz,
     uint32_t const * const kstart, uint32_t const * const katom,
     uint32_t const * const kprim, uint32_t const * const Ksumtype, uint32_t const * const Qstart,
     uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
     QUICKDouble const * const cons, QUICKDouble const * const gcexpo, uint8_t const * const KLMN,
     uint32_t prim_total, uint32_t const * const prim_start, QUICKDouble * const dense,
#if defined(OSHELL)
     QUICKDouble * const denseb,
#endif
     QUICKDouble const * const Xcoeff, QUICKDouble const * const expoSum,
     QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
     QUICKDouble const * const weightedCenterZ, QUICKDouble const * const cutPrim, QUICKDouble primLimit,
#if defined(USE_LEGACY_ATOMICS)
     QUICKULL * const gradULL,
#else
     QUICKDouble * const grad,
#endif
     QUICKDouble * const store, QUICKDouble * const store2,
     QUICKDouble * const storeAA, QUICKDouble * const storeBB, QUICKDouble * const storeCC)
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
    for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 2]; i++) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 2]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(store, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 3]; i++) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 3]; i++) {
        for (uint8_t j = Sumindex[I + 1]; j < Sumindex[I + J + 3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 3]; i++) {
        for (uint8_t j = Sumindex[I + 1]; j < Sumindex[I + J + 3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint8_t i = Sumindex[K + 1]; i < Sumindex[K + L + 3]; i++) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(storeCC, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint32_t i = 0; i < kPrimI * kPrimJ; i++) {
        uint32_t JJJ = i / kPrimI;
        uint32_t III = i - kPrimI * JJJ;
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

        QUICKDouble AA = LOC2(gcexpo, III, Ksumtype[II], MAXPRIM, nbasis);
        QUICKDouble BB = LOC2(gcexpo, JJJ, Ksumtype[JJ], MAXPRIM, nbasis);

        QUICKDouble AB = LOC2(expoSum, ii_start+III, jj_start+JJJ, prim_total, prim_total);
        QUICKDouble Px = LOC2(weightedCenterX, ii_start+III, jj_start+JJJ, prim_total, prim_total);
        QUICKDouble Py = LOC2(weightedCenterY, ii_start+III, jj_start+JJJ, prim_total, prim_total);
        QUICKDouble Pz = LOC2(weightedCenterZ, ii_start+III, jj_start+JJJ, prim_total, prim_total);

        /*
           X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
           cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
           */
        QUICKDouble cutoffPrim = DNMax
            * LOC2(cutPrim, kStartI + III, kStartJ + JJJ, jbasis, jbasis);
        QUICKDouble X1 = LOC4(Xcoeff, kStartI + III, kStartJ + JJJ,
                I - Qstart[II], J - Qstart[JJ],
                jbasis, jbasis, 2, 2);

        for (uint32_t j = 0; j < kPrimK * kPrimL; j++) {
            uint32_t LLL = j / kPrimK;
            uint32_t KKK = j - kPrimK * LLL;

            if (cutoffPrim * LOC2(cutPrim, kStartK+KKK, kStartL+LLL, jbasis, jbasis) > primLimit) {
                QUICKDouble CC = LOC2(gcexpo, KKK, Ksumtype[KK], MAXPRIM, nbasis);
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

                uint32_t kk_start = prim_start[KK];
                uint32_t ll_start = prim_start[LL];

                QUICKDouble CD = LOC2(expoSum, kk_start+KKK, ll_start+LLL, prim_total, prim_total);

                QUICKDouble ABCD = 1/(AB+CD);

                /*
                   X2 is the multiplication of four indices normalized coeffecient
                */
                QUICKDouble X2 = sqrt(ABCD) * X1
                    * LOC4(Xcoeff, kStartK + KKK, kStartL + LLL,
                            K - Qstart[KK], L - Qstart[LL],
                            jbasis, jbasis, 2, 2);

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
                QUICKDouble Qx = LOC2(weightedCenterX, kk_start+KKK, ll_start+LLL, prim_total, prim_total);
                QUICKDouble Qy = LOC2(weightedCenterY, kk_start+KKK, ll_start+LLL, prim_total, prim_total);
                QUICKDouble Qz = LOC2(weightedCenterZ, kk_start+KKK, ll_start+LLL, prim_total, prim_total);

                double YVerticalTemp[PRIM_INT_ERI_GRAD_LEN];
                FmT(I + J + K + L + 1, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)),
                        YVerticalTemp);

                for (uint32_t i = 0; i <= I + J + K + L + 1; i++) {
                    YVerticalTemp[i] *= X2;
                }

#if defined(int_sp)
                ERint_grad_vertical_sp
#elif defined(int_spd)
                ERint_grad_vertical_spd
#endif
                (I, J + 1, K, L + 1,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 2]; i++) {
                    for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(store, j, i , STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM);
                        }
                    }
                }

                for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 2]; i++) {
                    for (uint8_t j = Sumindex[I + 1]; j < Sumindex[I + J + 3]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * AA * 2 ;
                            LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * BB * 2 ;
                        }
                    }
                }

                for (uint8_t i = Sumindex[K + 1]; i < Sumindex[K + L + 3]; i++) {
                    for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(storeCC, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * CC * 2 ;
                        }
                    }
                }
            }
        }
    }

    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;
    QUICKDouble CGradx = 0.0;
    QUICKDouble CGrady = 0.0;
    QUICKDouble CGradz = 0.0;

    uint32_t AStart = katom[II] * 3;
    uint32_t BStart = katom[JJ] * 3;
    uint32_t CStart = katom[KK] * 3;
    uint32_t DStart = katom[LL] * 3;

    QUICKDouble RBx, RBy, RBz;
    QUICKDouble RDx, RDy, RDz;

    RBx = LOC2(xyz, 0, katom[JJ], 3, natom);
    RBy = LOC2(xyz, 1, katom[JJ], 3, natom);
    RBz = LOC2(xyz, 2, katom[JJ], 3, natom);
    RDx = LOC2(xyz, 0, katom[LL], 3, natom);
    RDy = LOC2(xyz, 1, katom[LL], 3, natom);
    RDz = LOC2(xyz, 2, katom[LL], 3, natom);

    uint32_t III1 = LOC2(Qsbasis, II, I, nshell, 4);
    uint32_t III2 = LOC2(Qfbasis, II, I, nshell, 4);
    uint32_t JJJ1 = LOC2(Qsbasis, JJ, J, nshell, 4);
    uint32_t JJJ2 = LOC2(Qfbasis, JJ, J, nshell, 4);
    uint32_t KKK1 = LOC2(Qsbasis, KK, K, nshell, 4);
    uint32_t KKK2 = LOC2(Qfbasis, KK, K, nshell, 4);
    uint32_t LLL1 = LOC2(Qsbasis, LL, L, nshell, 4);
    uint32_t LLL2 = LOC2(Qfbasis, LL, L, nshell, 4);

    for (uint32_t III = III1; III <= III2; III++) {
        for (uint32_t JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
            for (uint32_t KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
                for (uint32_t LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                    if (III < KKK
                            || (III == JJJ && III == LLL)
                            || (III == JJJ && III < LLL)
                            || (JJJ == LLL && III < JJJ)
                            || (III == KKK && III < JJJ && JJJ < LLL)) {
                        QUICKDouble Yaax, Yaay, Yaaz;
                        QUICKDouble Ybbx, Ybby, Ybbz;
                        QUICKDouble Yccx, Yccy, Yccz;
#if defined(int_sp)
                        hrrwholegrad_sp
#elif defined(int_spd)
                        hrrwholegrad
#endif
                            (&Yaax, &Yaay, &Yaaz, &Ybbx, &Ybby, &Ybbz, &Yccx, &Yccy, &Yccz,
                             J, L, III, JJJ, KKK, LLL,
                             store, storeAA, storeBB, storeCC,
                             RAx, RAy, RAz, RBx, RBy, RBz, RCx, RCy, RCz, RDx, RDy, RDz,
                             nbasis, cons, KLMN);

                        //printf("Y   %d   %d   %d   %d   %d   %d   %d   %d   %d   %d   %d   %d   %.9f   %.9f   %.9f   %.9f   %.9f   %.9f   %.9f   %.9f   %.9f\n",II, JJ, KK, LL, I, J, K, L, III, JJJ, KKK, LLL, Yaax, Yaay, Yaaz, Ybbx, Ybby, Ybbz, Yccx, Yccy, Yccz);

                        QUICKDouble constant = 0.0;
#if defined(OSHELL)
                        QUICKDouble DENSELJ = (QUICKDouble) (LOC2(dense, LLL, JJJ, nbasis, nbasis)
                                + LOC2(denseb, LLL, JJJ, nbasis, nbasis));
                        QUICKDouble DENSELI = (QUICKDouble) (LOC2(dense, LLL, III, nbasis, nbasis)
                                + LOC2(denseb, LLL, III, nbasis, nbasis));
                        QUICKDouble DENSELK = (QUICKDouble) (LOC2(dense, LLL, KKK, nbasis, nbasis)
                                + LOC2(denseb, LLL, KKK, nbasis, nbasis));
                        QUICKDouble DENSEJI = (QUICKDouble) (LOC2(dense, JJJ, III, nbasis, nbasis)
                                + LOC2(denseb, JJJ, III, nbasis, nbasis));

                        QUICKDouble DENSEKIA = (QUICKDouble) LOC2(dense, KKK, III, nbasis, nbasis);
                        QUICKDouble DENSEKJA = (QUICKDouble) LOC2(dense, KKK, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELJA = (QUICKDouble) LOC2(dense, LLL, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELIA = (QUICKDouble) LOC2(dense, LLL, III, nbasis, nbasis);
                        QUICKDouble DENSEJIA = (QUICKDouble) LOC2(dense, JJJ, III, nbasis, nbasis);

                        QUICKDouble DENSEKIB = (QUICKDouble) LOC2(denseb, KKK, III, nbasis, nbasis);
                        QUICKDouble DENSEKJB = (QUICKDouble) LOC2(denseb, KKK, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELJB = (QUICKDouble) LOC2(denseb, LLL, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELIB = (QUICKDouble) LOC2(denseb, LLL, III, nbasis, nbasis);
                        QUICKDouble DENSEJIB = (QUICKDouble) LOC2(denseb, JJJ, III, nbasis, nbasis);
#else
                        QUICKDouble DENSEKI = (QUICKDouble) LOC2(dense, KKK, III, nbasis, nbasis);
                        QUICKDouble DENSEKJ = (QUICKDouble) LOC2(dense, KKK, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELJ = (QUICKDouble) LOC2(dense, LLL, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELI = (QUICKDouble) LOC2(dense, LLL, III, nbasis, nbasis);
                        QUICKDouble DENSELK = (QUICKDouble) LOC2(dense, LLL, KKK, nbasis, nbasis);
                        QUICKDouble DENSEJI = (QUICKDouble) LOC2(dense, JJJ, III, nbasis, nbasis);
#endif

                        if (II < JJ && II < KK && KK < LL
                                || (III < KKK && III < JJJ && KKK < LLL)) {
#if defined(OSHELL)
                            constant = 4.0 * DENSEJI * DENSELK
                                    - 2.0 * hyb_coeff * (DENSEKIA * DENSELJA
                                            + DENSELIA * DENSEKJA
                                            + DENSEKIB * DENSELJB
                                            + DENSELIB * DENSEKJB);
#else
                            constant = 4.0 * DENSEJI * DENSELK
                                    - hyb_coeff * (DENSEKI * DENSELJ + DENSELI * DENSEKJ);
#endif
                        } else {
                            if (III < KKK) {
                                if (III == JJJ && KKK == LLL) {
#if defined(OSHELL)
                                    constant = DENSEJI * DENSELK
                                            - hyb_coeff * (DENSEKIA * DENSEKIA + DENSEKIB * DENSEKIB);
#else
                                    constant = DENSEJI * DENSELK - 0.5 * hyb_coeff * DENSEKI * DENSEKI;
#endif
                                } else if (JJJ == KKK && JJJ == LLL) {
#if defined(OSHELL)
                                    constant = 2.0 * (DENSELJ * DENSEJI
                                        - hyb_coeff * (DENSELJA * DENSEJIA + DENSELJB * DENSEJIB));
#else
                                    constant = (2.0 - hyb_coeff) * DENSELJ * DENSEJI;
#endif
                                } else if (KKK == LLL && III < JJJ && JJJ != KKK) {
#if defined(OSHELL)
                                    constant = 2.0 * (DENSEJI * DENSELK
                                            - hyb_coeff * (DENSEKIA * DENSEKJA + DENSEKIB * DENSEKJB));
#else
                                    constant = 2.0 * DENSEJI * DENSELK - hyb_coeff * DENSEKI * DENSEKJ;
#endif
                                } else if (III == JJJ && KKK < LLL) {
#if defined(OSHELL)
                                    constant = 2.0 * (DENSELK * DENSEJI
                                            - hyb_coeff * (DENSEKIA * DENSELIA + DENSEKIB * DENSELIB));
#else
                                    constant = 2.0 * DENSELK * DENSEJI - hyb_coeff * DENSEKI * DENSELI;
#endif
                                }
                            }
                            else {
                                if (JJJ <= LLL) {
                                    if (III == JJJ && III == KKK && III == LLL) {
                                        ; // Do nothing
                                    } else if (III == JJJ && III == KKK && III < LLL) {
#if defined(OSHELL)
                                        constant = 2.0 * (DENSELI * DENSEJI
                                                - hyb_coeff * (DENSELIA * DENSEJIA + DENSELIB * DENSEJIB));
#else
                                        constant = (2.0 - hyb_coeff) * DENSELI * DENSEJI;
#endif
                                    } else if (III == KKK && JJJ == LLL && III < JJJ) {
#if defined(OSHELL)
                                        constant = (2.0 * DENSEJI * DENSEJI
                                                - hyb_coeff * (DENSEJIA * DENSEJIA
                                                    + DENSELJA * DENSEKIA
                                                    + DENSEJIB * DENSEJIB
                                                    + DENSELJB * DENSEKIB));
#else
                                        constant = 2.0 * DENSEJI * DENSEJI
                                                - 0.5 * hyb_coeff * (DENSEJI * DENSEJI + DENSELJ * DENSEKI);
#endif
                                    } else if (III== KKK && III < JJJ && JJJ < LLL) {
#if defined(OSHELL)
                                        constant = 4.0 * DENSEJI * DENSELI
                                                - 2.0 * hyb_coeff * (DENSEJIA * DENSELIA
                                                    + DENSELJA * DENSEKIA
                                                    + DENSEJIB * DENSELIB
                                                    + DENSELJB * DENSEKIB);
#else
                                        constant = 4.0 * DENSEJI * DENSELI
                                                - hyb_coeff * (DENSEJI * DENSELI + DENSELJ * DENSEKI);
#endif
                                    }
                                }
                            }
                        }

                        AGradx += constant * Yaax;
                        AGrady += constant * Yaay;
                        AGradz += constant * Yaaz;

                        BGradx += constant * Ybbx;
                        BGrady += constant * Ybby;
                        BGradz += constant * Ybbz;

                        CGradx += constant * Yccx;
                        CGrady += constant * Yccy;
                        CGradz += constant * Yccz;
                    }
                }
            }
        }
    }

#if defined(USE_LEGACY_ATOMICS)
    GPUATOMICADD(&gradULL[AStart], AGradx, GRADSCALE);
    GPUATOMICADD(&gradULL[AStart + 1], AGrady, GRADSCALE);
    GPUATOMICADD(&gradULL[AStart + 2], AGradz, GRADSCALE);
    
    GPUATOMICADD(&gradULL[BStart], BGradx, GRADSCALE);
    GPUATOMICADD(&gradULL[BStart + 1], BGrady, GRADSCALE);
    GPUATOMICADD(&gradULL[BStart + 2], BGradz, GRADSCALE);
    
    GPUATOMICADD(&gradULL[CStart], CGradx, GRADSCALE);
    GPUATOMICADD(&gradULL[CStart + 1], CGrady, GRADSCALE);
    GPUATOMICADD(&gradULL[CStart + 2], CGradz, GRADSCALE);
    
    GPUATOMICADD(&gradULL[DStart], -AGradx - BGradx - CGradx, GRADSCALE);
    GPUATOMICADD(&gradULL[DStart + 1], -AGrady - BGrady - CGrady, GRADSCALE);
    GPUATOMICADD(&gradULL[DStart + 2], -AGradz - BGradz - CGradz, GRADSCALE);
#else
    atomicAdd(&grad[AStart], AGradx);
    atomicAdd(&grad[AStart + 1], AGrady);
    atomicAdd(&grad[AStart + 2], AGradz);
    
    atomicAdd(&grad[BStart], BGradx);
    atomicAdd(&grad[BStart + 1], BGrady);
    atomicAdd(&grad[BStart + 2], BGradz);
    
    atomicAdd(&grad[CStart], CGradx);
    atomicAdd(&grad[CStart + 1], CGrady);
    atomicAdd(&grad[CStart + 2], CGradz);
    
    atomicAdd(&grad[DStart], -AGradx - BGradx - CGradx);
    atomicAdd(&grad[DStart + 1], -AGrady - BGrady - CGrady);
    atomicAdd(&grad[DStart + 2], -AGradz - BGradz - CGradz);
#endif
}


#else
/*
   iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
   performance algrithem for electron intergral evaluation. See description below for details
   */
#if defined(OSHELL)
  #if defined(int_spdf)
__device__ static inline void iclass_grad_oshell_spdf
  #elif defined(int_spdf2)
__device__ static inline void iclass_grad_oshell_spdf2
  #elif defined(int_spdf3)
__device__ static inline void iclass_grad_oshell_spdf3
  #elif defined(int_spdf4)
__device__ static inline void iclass_grad_oshell_spdf4
  #elif defined(int_spdf5)
__device__ static inline void iclass_grad_oshell_spdf5
  #elif defined(int_spdf6)
__device__ static inline void iclass_grad_oshell_spdf6
  #elif defined(int_spdf7)
__device__ static inline void iclass_grad_oshell_spdf7
  #elif defined(int_spdf8)
__device__ static inline void iclass_grad_oshell_spdf8
  #endif
#else
  #if defined(int_spdf)
__device__ static inline void iclass_grad_spdf
  #elif defined(int_spdf2)
__device__ static inline void iclass_grad_spdf2
  #elif defined(int_spdf3)
__device__ static inline void iclass_grad_spdf3
  #elif defined(int_spdf4)
__device__ static inline void iclass_grad_spdf4
  #elif defined(int_spdf5)
__device__ static inline void iclass_grad_spdf5
  #elif defined(int_spdf6)
__device__ static inline void iclass_grad_spdf6
  #elif defined(int_spdf7)
__device__ static inline void iclass_grad_spdf7
  #elif defined(int_spdf8)
__device__ static inline void iclass_grad_spdf8
  #endif
#endif
    (uint8_t I, uint8_t J, uint8_t K, uint8_t L,
     uint32_t II, uint32_t JJ, uint32_t KK, uint32_t LL, QUICKDouble DNMax,
     QUICKDouble hyb_coeff, uint32_t natom, uint32_t nbasis,
     uint32_t nshell, uint32_t jbasis, QUICKDouble const * const xyz,
     uint32_t const * const kstart, uint32_t const * const katom,
     uint32_t const * const kprim, uint32_t const * const Ksumtype, uint32_t const * const Qstart,
     uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
     QUICKDouble const * const cons, QUICKDouble const * const gcexpo, uint8_t const * const KLMN,
     uint32_t prim_total, uint32_t const * const prim_start, QUICKDouble * const dense,
#if defined(OSHELL)
     QUICKDouble * const denseb,
#endif
     QUICKDouble const * const Xcoeff, QUICKDouble const * const expoSum,
     QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
     QUICKDouble const * const weightedCenterZ, QUICKDouble const * const cutPrim, QUICKDouble primLimit,
#if defined(USE_LEGACY_ATOMICS)
     QUICKULL * const gradULL,
#else
     QUICKDouble * const grad,
#endif
     QUICKDouble * const store, QUICKDouble * const store2,
     QUICKDouble * const storeAA, QUICKDouble * const storeBB, QUICKDouble * const storeCC)
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
    for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 3]; i++) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 3]; i++) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 3]; j++) {
            if (j < Sumindex[I + J + 2] && i < Sumindex[K + L + 2]) {
                LOCSTORE(store, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 3]; i++) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 3]; j++) {
            if (j >= Sumindex[I + 1]) {
                LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) = 0.0;
                LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 3]; i++) {
        for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 3]; j++) {
            if (i >= Sumindex[K + 1]) {
                LOCSTORE(storeCC, j, i, STOREDIM, STOREDIM) = 0.0;
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

        QUICKDouble AA = LOC2(gcexpo, III, Ksumtype[II], MAXPRIM, nbasis);
        QUICKDouble BB = LOC2(gcexpo, JJJ, Ksumtype[JJ], MAXPRIM, nbasis);

        QUICKDouble AB = LOC2(expoSum, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        QUICKDouble Px = LOC2(weightedCenterX, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        QUICKDouble Py = LOC2(weightedCenterY, ii_start + III, jj_start + JJJ, prim_total, prim_total);
        QUICKDouble Pz = LOC2(weightedCenterZ, ii_start + III, jj_start + JJJ, prim_total, prim_total);

        /*
           X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
           cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
           */
        QUICKDouble cutoffPrim = DNMax
            * LOC2(cutPrim, kStartI+III, kStartJ+JJJ, jbasis, jbasis);
        QUICKDouble X1 = LOC4(Xcoeff, kStartI + III, kStartJ + JJJ,
                I - Qstart[II], J - Qstart[JJ],
                jbasis, jbasis, 2, 2);

        for (uint32_t j = 0; j < kPrimK * kPrimL; j++) {
            uint32_t LLL = (uint32_t) j / kPrimK;
            uint32_t KKK = (uint32_t) j - kPrimK * LLL;

            if (cutoffPrim * LOC2(cutPrim, kStartK + KKK, kStartL + LLL, jbasis, jbasis)
                    > primLimit) {

                QUICKDouble CC = LOC2(gcexpo, KKK, Ksumtype[KK], MAXPRIM, nbasis);
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
                uint32_t kk_start = prim_start[KK];
                uint32_t ll_start = prim_start[LL];

                QUICKDouble CD = LOC2(expoSum, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                QUICKDouble ABCD = 1.0 / (AB + CD);
                /*
                   X2 is the multiplication of four indices normalized coeffecient
                */
                QUICKDouble X2 = sqrt(ABCD) * X1
                    * LOC4(Xcoeff, kStartK + KKK, kStartL + LLL,
                        K - Qstart[KK], L - Qstart[LL],
                        jbasis, jbasis, 2, 2);

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
                QUICKDouble Qx = LOC2(weightedCenterX, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                QUICKDouble Qy = LOC2(weightedCenterY, kk_start + KKK, ll_start + LLL, prim_total, prim_total);
                QUICKDouble Qz = LOC2(weightedCenterZ, kk_start + KKK, ll_start + LLL, prim_total, prim_total);

                double YVerticalTemp[PRIM_INT_ERI_GRAD_LEN];
                FmT(I + J + K + L + 2, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)),
                        YVerticalTemp);

                for (uint32_t i = 0; i <= I + J + K + L + 2; i++) {
                    YVerticalTemp[i] *= X2;
                }

#if defined(int_spdf)
                ERint_grad_vertical_dddd_1(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
#elif defined(int_spdf2)
                ERint_grad_vertical_dddd_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
#elif defined(int_spdf3)
                ERint_vertical_spdf_1_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_2_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_3_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_4_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_5_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_6_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_7_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_8_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spd_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_1(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_2(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_3(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_4(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_5(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_6(I, J+1, K, L+1,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
#elif defined(int_spdf4)
                ERint_grad_vrr_ffff_1(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_2(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_3(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_4(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_5(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_6(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_7(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_8(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_9(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_10(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_11(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_12(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_13(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_14(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_15(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_16(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_17(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_18(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_19(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_20(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_21(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_22(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_23(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_24(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_25(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_26(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_27(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_28(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_29(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_30(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_31(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_32(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
#endif

                for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 2]; i++) {
                    for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(store, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM);
                        }
                    }
                }

                for (uint8_t i = Sumindex[K]; i < Sumindex[K + L + 2]; i++) {
                    for (uint8_t j = Sumindex[I + 1]; j < Sumindex[I + J + 3]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * AA * 2.0;
                            LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * BB * 2.0;
                        }
                    }
                }

                for (uint8_t i = Sumindex[K + 1]; i < Sumindex[K + L + 3]; i++) {
                    for (uint8_t j = Sumindex[I]; j < Sumindex[I + J + 2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(storeCC, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * CC * 2.0;
                        }
                    }
                }
            }
        }
    }

    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;
    QUICKDouble CGradx = 0.0;
    QUICKDouble CGrady = 0.0;
    QUICKDouble CGradz = 0.0;

    uint32_t AStart = katom[II] * 3;
    uint32_t BStart = katom[JJ] * 3;
    uint32_t CStart = katom[KK] * 3;
    uint32_t DStart = katom[LL] * 3;

    QUICKDouble RBx, RBy, RBz;
    QUICKDouble RDx, RDy, RDz;

    RBx = LOC2(xyz, 0, katom[JJ], 3, natom);
    RBy = LOC2(xyz, 1, katom[JJ], 3, natom);
    RBz = LOC2(xyz, 2, katom[JJ], 3, natom);
    RDx = LOC2(xyz, 0, katom[LL], 3, natom);
    RDy = LOC2(xyz, 1, katom[LL], 3, natom);
    RDz = LOC2(xyz, 2, katom[LL], 3, natom);

    uint32_t III1 = LOC2(Qsbasis, II, I, nshell, 4);
    uint32_t III2 = LOC2(Qfbasis, II, I, nshell, 4);
    uint32_t JJJ1 = LOC2(Qsbasis, JJ, J, nshell, 4);
    uint32_t JJJ2 = LOC2(Qfbasis, JJ, J, nshell, 4);
    uint32_t KKK1 = LOC2(Qsbasis, KK, K, nshell, 4);
    uint32_t KKK2 = LOC2(Qfbasis, KK, K, nshell, 4);
    uint32_t LLL1 = LOC2(Qsbasis, LL, L, nshell, 4);
    uint32_t LLL2 = LOC2(Qfbasis, LL, L, nshell, 4);

    for (uint32_t III = III1; III <= III2; III++) {
        for (uint32_t JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
            for (uint32_t KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
                for (uint32_t LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                    if (III < KKK
                            || (III == JJJ && III == LLL)
                            || (III == JJJ && III < LLL)
                            || (JJJ == LLL && III < JJJ)
                            || (III == KKK && III < JJJ && JJJ < LLL)) {
                        QUICKDouble Yaax, Yaay, Yaaz;
                        QUICKDouble Ybbx, Ybby, Ybbz;
                        QUICKDouble Yccx, Yccy, Yccz;

#if defined(int_spdf)
                        hrrwholegrad2_1
#elif defined(int_spdf2)
                        hrrwholegrad2_2
#else
                        hrrwholegrad2
#endif
                            (&Yaax, &Yaay, &Yaaz,
                             &Ybbx, &Ybby, &Ybbz,
                             &Yccx, &Yccy, &Yccz,
                             J, L, III, JJJ, KKK, LLL, 
                             store, storeAA, storeBB, storeCC,
                             RAx, RAy, RAz, RBx, RBy, RBz,
                             RCx, RCy, RCz, RDx, RDy, RDz,
                             nbasis, cons, KLMN);

                        QUICKDouble constant = 0.0;
#if defined(OSHELL)
                        QUICKDouble DENSELJ = (QUICKDouble) (LOC2(dense, LLL, JJJ, nbasis, nbasis)
                                + LOC2(denseb, LLL, JJJ, nbasis, nbasis));
                        QUICKDouble DENSELI = (QUICKDouble) (LOC2(dense, LLL, III, nbasis, nbasis)
                                + LOC2(denseb, LLL, III, nbasis, nbasis));
                        QUICKDouble DENSELK = (QUICKDouble) (LOC2(dense, LLL, KKK, nbasis, nbasis)
                                + LOC2(denseb, LLL, KKK, nbasis, nbasis));
                        QUICKDouble DENSEJI = (QUICKDouble) (LOC2(dense, JJJ, III, nbasis, nbasis)
                                + LOC2(denseb, JJJ, III, nbasis, nbasis));

                        QUICKDouble DENSEKIA = (QUICKDouble) LOC2(dense, KKK, III, nbasis, nbasis);
                        QUICKDouble DENSEKJA = (QUICKDouble) LOC2(dense, KKK, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELJA = (QUICKDouble) LOC2(dense, LLL, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELIA = (QUICKDouble) LOC2(dense, LLL, III, nbasis, nbasis);
                        QUICKDouble DENSEJIA = (QUICKDouble) LOC2(dense, JJJ, III, nbasis, nbasis);

                        QUICKDouble DENSEKIB = (QUICKDouble) LOC2(denseb, KKK, III, nbasis, nbasis);
                        QUICKDouble DENSEKJB = (QUICKDouble) LOC2(denseb, KKK, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELJB = (QUICKDouble) LOC2(denseb, LLL, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELIB = (QUICKDouble) LOC2(denseb, LLL, III, nbasis, nbasis);
                        QUICKDouble DENSEJIB = (QUICKDouble) LOC2(denseb, JJJ, III, nbasis, nbasis);
#else
                        QUICKDouble DENSEKI = (QUICKDouble) LOC2(dense, KKK, III, nbasis, nbasis);
                        QUICKDouble DENSEKJ = (QUICKDouble) LOC2(dense, KKK, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELJ = (QUICKDouble) LOC2(dense, LLL, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELI = (QUICKDouble) LOC2(dense, LLL, III, nbasis, nbasis);
                        QUICKDouble DENSELK = (QUICKDouble) LOC2(dense, LLL, KKK, nbasis, nbasis);
                        QUICKDouble DENSEJI = (QUICKDouble) LOC2(dense, JJJ, III, nbasis, nbasis);
#endif

                        if (II < JJ && II < KK && KK < LL
                                || (III < KKK && III < JJJ && KKK < LLL)) {
#if defined(OSHELL)
                            constant = 4.0 * DENSEJI * DENSELK
                                    - 2.0 * hyb_coeff * (DENSEKIA * DENSELJA
                                        + DENSELIA * DENSEKJA
                                        + DENSEKIB * DENSELJB
                                        + DENSELIB * DENSEKJB);
#else
                            constant = 4.0 * DENSEJI * DENSELK
                                    - hyb_coeff * (DENSEKI * DENSELJ + DENSELI * DENSEKJ);
#endif
                        } else {
                            if (III < KKK) {
                                if (III == JJJ && KKK == LLL) {
#if defined(OSHELL)
                                    constant = DENSEJI * DENSELK
                                            - hyb_coeff * (DENSEKIA * DENSEKIA + DENSEKIB * DENSEKIB);
#else
                                    constant = DENSEJI * DENSELK - 0.5 * hyb_coeff * DENSEKI * DENSEKI;
#endif

                                } else if (JJJ == KKK && JJJ == LLL) {
#if defined(OSHELL)
                                    constant = 2.0 * (DENSELJ * DENSEJI
                                        - hyb_coeff * (DENSELJA * DENSEJIA + DENSELJB * DENSEJIB));
#else
                                    constant = (2.0 - hyb_coeff) * DENSELJ * DENSEJI;
#endif
                                } else if (KKK == LLL && III < JJJ && JJJ != KKK) {
#if defined(OSHELL)
                                    constant = 2.0 * (DENSEJI * DENSELK
                                            - hyb_coeff * (DENSEKIA * DENSEKJA + DENSEKIB * DENSEKJB));
#else
                                    constant = 2.0 * DENSEJI * DENSELK - hyb_coeff * DENSEKI * DENSEKJ;
#endif
                                } else if (III == JJJ && KKK < LLL) {
#if defined(OSHELL)
                                    constant = 2.0 * (DENSELK * DENSEJI
                                            - hyb_coeff * (DENSEKIA * DENSELIA + DENSEKIB * DENSELIB));
#else
                                    constant = 2.0 * DENSELK * DENSEJI - hyb_coeff * DENSEKI * DENSELI;
#endif
                                }
                            }
                            else {
                                if (JJJ <= LLL) {
                                    if (III == JJJ && III == KKK && III == LLL) {
                                        ; // Do nothing
                                    } else if (III == JJJ && III == KKK && III < LLL) {
#if defined(OSHELL)
                                        constant = 2.0 * (DENSELI * DENSEJI
                                            - hyb_coeff * (DENSELIA * DENSEJIA + DENSELIB * DENSEJIB));
#else
                                        constant = (2.0 - hyb_coeff) * DENSELI * DENSEJI;
#endif
                                    } else if (III == KKK && JJJ == LLL && III < JJJ) {
#if defined(OSHELL)
                                        constant = 2.0 * DENSEJI * DENSEJI
                                                - hyb_coeff * (DENSEJIA * DENSEJIA
                                                        + DENSELJA * DENSEKIA
                                                        + DENSEJIB * DENSEJIB
                                                        + DENSELJB * DENSEKIB);
#else
                                        constant = (2.0 - 0.5 * hyb_coeff) * DENSEJI * DENSEJI
                                                - 0.5 * hyb_coeff * DENSELJ * DENSEKI;
#endif

                                    } else if (III == KKK && III < JJJ && JJJ < LLL) {
#if defined(OSHELL)
                                        constant = 4.0 * DENSEJI * DENSELI
                                                - 2.0 * hyb_coeff * (DENSEJIA * DENSELIA
                                                        + DENSELJA * DENSEKIA
                                                        + DENSEJIB * DENSELIB
                                                        + DENSELJB * DENSEKIB);
#else
                                        constant = (4.0 - hyb_coeff) * DENSEJI * DENSELI
                                            - hyb_coeff * DENSELJ * DENSEKI;
#endif
                                    }
                                }
                            }
                        }

                        AGradx += constant * Yaax;
                        AGrady += constant * Yaay;
                        AGradz += constant * Yaaz;

                        BGradx += constant * Ybbx;
                        BGrady += constant * Ybby;
                        BGradz += constant * Ybbz;

                        CGradx += constant * Yccx;
                        CGrady += constant * Yccy;
                        CGradz += constant * Yccz;
                    }
                }
            }
        }
    }

#if defined(DEBUG)
    //printf("FILE: %s, LINE: %d, FUNCTION: %s, hyb_coeff \n", __FILE__, __LINE__, __func__);
#endif

#if defined(USE_LEGACY_ATOMICS)
    GPUATOMICADD(&gradULL[AStart], AGradx, GRADSCALE);
    GPUATOMICADD(&gradULL[AStart + 1], AGrady, GRADSCALE);
    GPUATOMICADD(&gradULL[AStart + 2], AGradz, GRADSCALE);
    
    GPUATOMICADD(&gradULL[BStart], BGradx, GRADSCALE);
    GPUATOMICADD(&gradULL[BStart + 1], BGrady, GRADSCALE);
    GPUATOMICADD(&gradULL[BStart + 2], BGradz, GRADSCALE);
    
    GPUATOMICADD(&gradULL[CStart], CGradx, GRADSCALE);
    GPUATOMICADD(&gradULL[CStart + 1], CGrady, GRADSCALE);
    GPUATOMICADD(&gradULL[CStart + 2], CGradz, GRADSCALE);
    
    GPUATOMICADD(&gradULL[DStart], -AGradx - BGradx - CGradx, GRADSCALE);
    GPUATOMICADD(&gradULL[DStart + 1], -AGrady - BGrady - CGrady, GRADSCALE);
    GPUATOMICADD(&gradULL[DStart + 2], -AGradz - BGradz - CGradz, GRADSCALE);
#else
    atomicAdd(&grad[AStart], AGradx);
    atomicAdd(&grad[AStart + 1], AGrady);
    atomicAdd(&grad[AStart + 2], AGradz);

    atomicAdd(&grad[BStart], BGradx);
    atomicAdd(&grad[BStart + 1], BGrady);
    atomicAdd(&grad[BStart + 2], BGradz);

    atomicAdd(&grad[CStart], CGradx);
    atomicAdd(&grad[CStart + 1], CGrady);
    atomicAdd(&grad[CStart + 2], CGradz);

    atomicAdd(&grad[DStart], -AGradx - BGradx - CGradx);
    atomicAdd(&grad[DStart + 1], -AGrady - BGrady - CGrady);
    atomicAdd(&grad[DStart + 2], -AGradz - BGradz - CGradz);
#endif
}
#endif


#if defined(OSHELL)
  #if defined(int_sp)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_oshell_sp
  #elif defined(int_spd)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_oshell_spd
  #elif defined(int_spdf)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_oshell_spdf
  #elif defined(int_spdf2)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_oshell_spdf2
  #elif defined(int_spdf3)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_oshell_spdf3
  #elif defined(int_spdf4)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_oshell_spdf4
  #elif defined(int_spdf5)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_oshell_spdf5
  #elif defined(int_spdf6)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_oshell_spdf6
  #elif defined(int_spdf7)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_oshell_spdf7
  #elif defined(int_spdf8)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_oshell_spdf8
  #endif
#else
  #if defined(int_sp)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_cshell_sp
  #elif defined(int_spd)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_cshell_spd
  #elif defined(int_spdf)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_cshell_spdf
  #elif defined(int_spdf2)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_cshell_spdf2
  #elif defined(int_spdf3)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_cshell_spdf3
  #elif defined(int_spdf4)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_cshell_spdf4
  #elif defined(int_spdf5)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_cshell_spdf5
  #elif defined(int_spdf6)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_cshell_spdf6
  #elif defined(int_spdf7)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_cshell_spdf7
  #elif defined(int_spdf8)
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) k_get_grad_cshell_spdf8
  #endif
#endif
    (QUICKDouble hyb_coeff, uint32_t natom, uint32_t nbasis,
        uint32_t nshell, uint32_t jbasis, QUICKDouble const * const xyz,
        uint32_t const * const kstart, uint32_t const * const katom,
        uint32_t const * const kprim, uint32_t const * const Ksumtype, uint32_t const * const Qstart,
        uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
        uint8_t const * const sorted_Qnumber, uint32_t const * const sorted_Q,
        QUICKDouble const * const cons, QUICKDouble const * const gcexpo, uint8_t const * const KLMN,
        uint32_t prim_total, uint32_t const * const prim_start, QUICKDouble * const dense,
#if defined(OSHELL)
        QUICKDouble * const denseb,
#endif
        QUICKDouble const * const Xcoeff, QUICKDouble const * const expoSum,
        QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
        QUICKDouble const * const weightedCenterZ, uint32_t sqrQshell, int2 const * const sorted_YCutoffIJ,
        QUICKDouble const * const cutMatrix, QUICKDouble const * const YCutoff,
        QUICKDouble const * const cutPrim, QUICKDouble primLimit, QUICKDouble gradCutoff,
#if defined(USE_LEGACY_ATOMICS)
        QUICKULL * const gradULL,
#else
        QUICKDouble * const grad,
#endif
#if defined(MPIV_GPU)
        unsigned char const * const mpi_bcompute,
#endif
        QUICKDouble * const store, QUICKDouble * const store2,
        QUICKDouble * const storeAA, QUICKDouble * const storeBB, QUICKDouble * const storeCC)
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    QUICKULL jshell = (QUICKULL) sqrQshell;
    QUICKULL jshell2 = (QUICKULL) sqrQshell;
#if defined(USE_LEGACY_ATOMICS)
    extern __shared__ QUICKULL smem[];
    QUICKULL *sgradULL = smem;

    for (int i = threadIdx.x; i < 3u * natom; i += blockDim.x) {
      sgradULL[i] = 0ull;
    }
#else
    extern __shared__ QUICKDouble smem[];
    QUICKDouble *sgrad = smem;

    for (int i = threadIdx.x; i < 3u * natom; i += blockDim.x) {
        sgrad[i] = 0.0;
    }
#endif

    __syncthreads();

    for (QUICKULL i = offset; i < jshell2 * jshell; i += totalThreads) {
        QUICKULL a = (QUICKULL) i / jshell;
        QUICKULL b = (QUICKULL) (i - a * jshell);

#if defined(MPIV_GPU)
        if (mpi_bcompute[a] > 0) {
#endif
        int II = sorted_YCutoffIJ[a].x;
        int KK = sorted_YCutoffIJ[b].x;

        uint32_t ii = sorted_Q[II];
        uint32_t kk = sorted_Q[KK];

        if (ii <= kk) {
            int JJ = sorted_YCutoffIJ[a].y;
            int LL = sorted_YCutoffIJ[b].y;

            uint8_t iii = sorted_Qnumber[II];
            uint8_t jjj = sorted_Qnumber[JJ];
            uint8_t kkk = sorted_Qnumber[KK];
            uint8_t lll = sorted_Qnumber[LL];

#if defined(int_sp)
            if (iii < 2 && jjj < 2 && kkk < 2 && lll < 2) {
#elif defined(int_spd)
            if (!(iii < 2 && jjj < 2 && kkk < 2 && lll < 2)) {
#endif
                uint32_t jj = sorted_Q[JJ];
                uint32_t ll = sorted_Q[LL];

                // In case 4 indices are in the same atom
                if (!(katom[ii] == katom[jj]
                            && katom[ii] == katom[kk]
                            && katom[ii] == katom[ll])) {

                    QUICKDouble DNMax =
                        MAX(MAX(4.0 * LOC2(cutMatrix, ii, jj, nshell, nshell),
                                    4.0 * LOC2(cutMatrix, kk, ll, nshell, nshell)),
                                MAX(MAX(LOC2(cutMatrix, ii, ll, nshell, nshell),
                                        LOC2(cutMatrix, ii, kk, nshell, nshell)),
                                    MAX(LOC2(cutMatrix, jj, kk, nshell, nshell),
                                        LOC2(cutMatrix, jj, ll, nshell, nshell))));

                    if ((LOC2(YCutoff, kk, ll, nshell, nshell) * LOC2(YCutoff, ii, jj, nshell, nshell))
                            > gradCutoff
                        && (LOC2(YCutoff, kk, ll, nshell, nshell) * LOC2(YCutoff, ii, jj, nshell, nshell) * DNMax)
                            > gradCutoff) {
#if defined(OSHELL)
  #if defined(int_sp)
                        {
                            iclass_grad_oshell_sp
  #elif defined(int_spd)
                        {
                            iclass_grad_oshell_spd
  #elif defined(int_spdf)
                        if (kkk + lll >= 4) {
                            iclass_grad_oshell_spdf
  #elif defined(int_spdf2)
                        if (iii + jjj >= 4) {
                            iclass_grad_oshell_spdf2
  #elif defined(int_spdf3)
//                        {
//                            iclass_grad_oshell_spdf3
  #elif defined(int_spdf4)
                        {
                            iclass_grad_oshell_spdf4
  #elif defined(int_spdf5)
                        {
                            iclass_grad_oshell_spdf5
  #elif defined(int_spdf6)
                        {
                            iclass_grad_oshell_spdf6
  #elif defined(int_spdf7)
                        {
                            iclass_grad_oshell_spdf7
  #elif defined(int_spdf8)
                        {
                            iclass_grad_oshell_spdf8
  #endif
#else
  #if defined(int_sp)
                        if (iii != 3 && jjj != 3 && kkk != 3 && lll != 3) {
                            iclass_grad_cshell_sp
  #elif defined(int_spd)
                        if (iii != 3 && jjj != 3 && kkk != 3 && lll != 3) {
                            iclass_grad_cshell_spd
  #elif defined(int_spdf)
                        if (kkk + lll >= 4 && iii != 3 && jjj != 3 && kkk != 3 && lll != 3) {
                            iclass_grad_spdf
  #elif defined(int_spdf2)
                        if (iii + jjj >= 4 && iii != 3 && jjj != 3 && kkk != 3 && lll != 3) {
                            iclass_grad_spdf2
  #elif defined(int_spdf3)
                        if ((iii == 3 || jjj == 3 || kkk == 3 || lll == 3)
                                && (iii != 3 || jjj != 3 || kkk != 3 || lll != 3)) {
                                iclass_grad_spdf3
  #elif defined(int_spdf4)
                        if (iii == 3 && jjj == 3 && kkk == 3 && lll == 3) {
                            iclass_grad_spdf4
  #elif defined(int_spdf5)
                        {
                            iclass_grad_spdf5
  #elif(defined int_spdf6)
                        {
                            iclass_grad_spdf6
  #elif(defined int_spdf7)
                        {
                            iclass_grad_spdf7
  #elif(defined int_spdf8)
                        {
                            iclass_grad_spdf8
  #endif
#endif
                            (iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax,
                             hyb_coeff, natom, nbasis, nshell, jbasis, xyz,
                             kstart, katom, kprim, Ksumtype, Qstart, Qsbasis, Qfbasis,
                             cons, gcexpo, KLMN, prim_total, prim_start, dense,
#if defined(OSHELL)
                             denseb,
#endif
                             Xcoeff, expoSum, weightedCenterX, weightedCenterY, weightedCenterZ,
                             cutPrim, primLimit,
#if defined(USE_LEGACY_ATOMICS)
                             sgradULL,
#else
                             sgrad,
#endif
                             store + offset,
                             store2 + offset,
                             storeAA + offset,
                             storeBB + offset,
                             storeCC + offset);
                        }
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

    __syncthreads();

    for (int i = threadIdx.x; i < 3u * natom; i += blockDim.x) {
#if defined(USE_LEGACY_ATOMICS)
        atomicAdd(&gradULL[i], sgradULL[i]);
#else
        atomicAdd(&grad[i], sgrad[i]);
#endif
    }
}
