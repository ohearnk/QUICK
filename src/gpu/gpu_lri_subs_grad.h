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

#undef PRIM_INT_LRI_GRAD_LEN
#if defined(int_spd)
  #define PRIM_INT_LRI_GRAD_LEN (6)
#elif defined(int_spdf2)
  #define PRIM_INT_LRI_GRAD_LEN (9)
#endif


#if !defined(__gpu_get2e_subs_grad_h_)
  #define __gpu_get2e_subs_grad_h_
  #undef STOREDIM
  #define STOREDIM STOREDIM_S
__device__ static inline void hrrwholegrad_lri(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        uint32_t J, uint32_t III, uint32_t JJJ,
        QUICKDouble * const store, QUICKDouble * const storeAA, QUICKDouble * const storeBB,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint32_t nbasis, QUICKDouble const * const cons, uint32_t const * const KLMN,
        uint32_t const * const trans)
{
    uint32_t angularL[12];
    QUICKDouble coefAngularL[12];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;

    QUICKDouble constant = cons[III] * cons[JJJ];
    uint32_t numAngularL;

    //  Part A - x
    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis) + 1,
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Yaax = *Yaax + coefAngularL[i] * LOCSTORE(storeAA, angularL[i], 0, STOREDIM, STOREDIM);
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis) + 1,
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Yaay = *Yaay + coefAngularL[i] * LOCSTORE(storeAA, angularL[i], 0, STOREDIM, STOREDIM);
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis) + 1,
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Yaaz = *Yaaz + coefAngularL[i] * LOCSTORE(storeAA, angularL[i], 0, STOREDIM, STOREDIM);
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis) + 1,
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J + 1, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Ybbx = *Ybbx + coefAngularL[i] * LOCSTORE(storeBB, angularL[i], 0, STOREDIM, STOREDIM);
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis) + 1,
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J + 1, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Ybby = *Ybby + coefAngularL[i] * LOCSTORE(storeBB, angularL[i], 0, STOREDIM, STOREDIM);
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis) + 1,
            J + 1, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Ybbz = *Ybbz + coefAngularL[i] * LOCSTORE(storeBB, angularL[i], 0, STOREDIM, STOREDIM);
        }
    }

    if (LOC2(KLMN, 0, III, 3, nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis) - 1,
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis),
                LOC2(KLMN, 0, JJJ, 3, nbasis),
                LOC2(KLMN, 1, JJJ, 3, nbasis),
                LOC2(KLMN, 2, JJJ, 3, nbasis),
                J, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Yaax = *Yaax - LOC2(KLMN, 0, III, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN, 1, III, 3, nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis) - 1,
                LOC2(KLMN, 2, III, 3, nbasis),
                LOC2(KLMN, 0, JJJ, 3, nbasis),
                LOC2(KLMN, 1, JJJ, 3, nbasis),
                LOC2(KLMN, 2, JJJ, 3, nbasis),
                J, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Yaay = *Yaay - LOC2(KLMN, 1, III, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN, 2, III, 3, nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis) - 1,
                LOC2(KLMN, 0, JJJ, 3, nbasis),
                LOC2(KLMN, 1, JJJ, 3, nbasis),
                LOC2(KLMN, 2, JJJ, 3, nbasis),
                J, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Yaaz = *Yaaz - LOC2(KLMN, 2, III, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,0,JJJ,3,nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis),
                LOC2(KLMN, 0, JJJ, 3, nbasis) - 1,
                LOC2(KLMN, 1, JJJ, 3, nbasis),
                LOC2(KLMN, 2, JJJ, 3, nbasis),
                J - 1, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Ybbx = *Ybbx - LOC2(KLMN, 0, JJJ, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN, 1, JJJ, 3, nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis),
                LOC2(KLMN, 0, JJJ, 3, nbasis),
                LOC2(KLMN, 1, JJJ, 3, nbasis) - 1,
                LOC2(KLMN, 2, JJJ, 3, nbasis),
                J - 1, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Ybby = *Ybby - LOC2(KLMN, 1, JJJ, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN, 2, JJJ, 3, nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis),
                LOC2(KLMN, 0, JJJ, 3, nbasis),
                LOC2(KLMN, 1, JJJ, 3, nbasis),
                LOC2(KLMN, 2, JJJ, 3, nbasis) - 1,
                J - 1, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Ybbz = *Ybbz - LOC2(KLMN, 2, JJJ, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;

    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
}


  #undef STOREDIM
  #define STOREDIM STOREDIM_L
__device__ static inline void hrrwholegrad_lri2(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        uint32_t J, uint32_t III, uint32_t JJJ,
        QUICKDouble * const store, QUICKDouble AA, QUICKDouble BB,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint32_t nbasis, QUICKDouble const * const cons, uint32_t const * const KLMN,
        uint32_t const * const trans)
{
    uint32_t angularL[12];
    QUICKDouble coefAngularL[12];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;

    QUICKDouble constant = cons[III] * cons[JJJ];
    uint32_t numAngularL;

    //  Part A - x
    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN,0,III,3,nbasis) + 1,
            LOC2(KLMN,1,III,3,nbasis),
            LOC2(KLMN,2,III,3,nbasis),
            LOC2(KLMN,0,JJJ,3,nbasis),
            LOC2(KLMN,1,JJJ,3,nbasis),
            LOC2(KLMN,2,JJJ,3,nbasis),
            J, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Yaax = *Yaax + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis) + 1,
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Yaay = *Yaay + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis) + 1,
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Yaaz = *Yaaz + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis) + 1,
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J + 1, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Ybbx = *Ybbx + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis) + 1,
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J + 1, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Ybby = *Ybby + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis) + 1,
            J + 1, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Ybbz = *Ybbz + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    if (LOC2(KLMN,0,III,3,nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis) - 1,
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis),
                LOC2(KLMN, 0, JJJ, 3, nbasis),
                LOC2(KLMN, 1, JJJ, 3, nbasis),
                LOC2(KLMN, 2, JJJ, 3, nbasis),
                J, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Yaax = *Yaax - LOC2(KLMN, 0, III, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN,1,III,3,nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis) - 1,
                LOC2(KLMN, 2, III, 3, nbasis),
                LOC2(KLMN, 0, JJJ, 3, nbasis),
                LOC2(KLMN, 1, JJJ, 3, nbasis),
                LOC2(KLMN, 2, JJJ, 3, nbasis),
                J, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Yaay = *Yaay - LOC2(KLMN, 1, III, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN, 2, III, 3, nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis) - 1,
                LOC2(KLMN, 0, JJJ, 3, nbasis),
                LOC2(KLMN, 1, JJJ, 3, nbasis),
                LOC2(KLMN, 2, JJJ, 3, nbasis),
                J, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Yaaz = *Yaaz - LOC2(KLMN, 2, III, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN, 0, JJJ, 3, nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis), 
                LOC2(KLMN, 2, III, 3, nbasis),
                LOC2(KLMN, 0, JJJ, 3, nbasis) - 1,
                LOC2(KLMN, 1, JJJ, 3, nbasis),
                LOC2(KLMN, 2, JJJ, 3, nbasis),
                J - 1, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Ybbx = *Ybbx - LOC2(KLMN, 0, JJJ, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN, 1, JJJ, 3, nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis),
                LOC2(KLMN, 0, JJJ, 3, nbasis),
                LOC2(KLMN, 1, JJJ, 3, nbasis) - 1,
                LOC2(KLMN, 2, JJJ, 3, nbasis),
                J - 1, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Ybby = *Ybby - LOC2(KLMN, 1, JJJ, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
            }
        }
    }

    if (LOC2(KLMN, 2, JJJ, 3, nbasis) >= 1) {
        numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(KLMN, 0, III, 3, nbasis),
                LOC2(KLMN, 1, III, 3, nbasis),
                LOC2(KLMN, 2, III, 3, nbasis),
                LOC2(KLMN, 0, JJJ, 3, nbasis),
                LOC2(KLMN, 1, JJJ, 3, nbasis),
                LOC2(KLMN, 2, JJJ, 3, nbasis) - 1,
                J - 1, coefAngularL, angularL, trans);

        for (uint32_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                *Ybbz = *Ybbz - LOC2(KLMN, 2, JJJ, 3, nbasis) * coefAngularL[i]
                    * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);

            }
        }
    }

    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;

    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
}


__device__ static inline void hrrwholegrad_lri2_2(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        uint32_t J, uint32_t III, uint32_t JJJ,
        QUICKDouble * const store, QUICKDouble AA, QUICKDouble BB,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint32_t nbasis, QUICKDouble const * const cons, uint32_t const * const KLMN,
        uint32_t const * const trans)
{
    uint32_t angularL[12];
    QUICKDouble coefAngularL[12];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;

    QUICKDouble constant = cons[III] * cons[JJJ];
    uint32_t numAngularL;

    //  Part A - x
    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis) + 1,
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Yaax = *Yaax + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis) + 1,
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Yaay = *Yaay + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis) + 1,
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Yaaz = *Yaaz + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * AA;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis) + 1,
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J + 1, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Ybbx = *Ybbx + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis) + 1,
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J + 1, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Ybby = *Ybby + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis) + 1,
            J + 1, coefAngularL, angularL, trans);

    for (uint32_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            *Ybbz = *Ybbz + coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM) * 2 * BB;
        }
    }

    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;

    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
}
#endif


#undef STOREDIM
#if defined(int_spd)
  #define STOREDIM STOREDIM_S
#else
  #define STOREDIM STOREDIM_L
#endif


/*
   iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
   performance algrithem for electron intergral evaluation. See description below for details
   */
#if defined(int_spd)
  #if defined(OSHELL)
__device__ static inline void iclass_oshell_lri_grad
  #else
__device__ static inline void iclass_lri_grad
  #endif
(uint32_t I, uint32_t J, uint32_t II, uint32_t JJ, uint32_t iatom, uint32_t totalatom,
 uint32_t natom, uint32_t nbasis, uint32_t nshell, uint32_t jbasis,
 QUICKDouble const * const xyz, QUICKDouble const * const allxyz,
 uint32_t const * const kstart, uint32_t const * const katom,
 uint32_t const * const kprim, uint32_t const * const Ksumtype, uint32_t const * const Qstart,
 uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
 QUICKDouble const * const cons, QUICKDouble const * const gcexpo, uint32_t const * const KLMN,
 uint32_t prim_total, uint32_t const * const prim_start, QUICKDouble * const dense,
#if defined(OSHELL)
 QUICKDouble * const denseb,
#endif
 QUICKDouble const * const Xcoeff, QUICKDouble const * const expoSum,
 QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
 QUICKDouble const * const weightedCenterZ,
#if defined(USE_LEGACY_ATOMICS)
 QUICKULL * const gradULL,
#else
 QUICKDouble * const grad,
#endif
 QUICKDouble * const store, QUICKDouble * const store2,
 QUICKDouble * const storeAA, QUICKDouble * const storeBB)
}
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
    for (uint32_t i = Sumindex[0]; i < Sumindex[2]; i++) {
        for (uint32_t j = Sumindex[I]; j < Sumindex[I + J + 2]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(store, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint32_t i = Sumindex[0]; i < Sumindex[3]; i++) {
        for (uint32_t j = Sumindex[I + 1]; j < Sumindex[I + J + 3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) = 0.0;
            }
        }
    }

    for (uint32_t i = Sumindex[0]; i < Sumindex[3]; i++) {
        for (uint32_t j = Sumindex[I + 1]; j < Sumindex[I + J + 3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {
                LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) = 0.0;
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
        QUICKDouble X1 = LOC4(Xcoeff, kStartI + III, kStartJ + JJJ,
                I - Qstart[II], J - Qstart[JJ], jbasis, jbasis, 2, 2);

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
        QUICKDouble CD = lri_zeta;
        QUICKDouble ABCD = 1.0 / (AB + CD);

        /*
           X2 is the multiplication of four indices normalized coeffecient
        */
        QUICKDouble X2 = sqrt(ABCD) * X1 * X0 * (1.0 / lri_zeta)
            * lri_cc[iatom] * pow(lri_zeta / PI, 1.5);
        
        //printf("lngr grad itt, x0,xcoeff1,x2,xcoeff2,x44: %f %f %f %f %f %f \n", X0, X1, lri_zeta, lri_cc[iatom], (1/lri_zeta) * lri_cc[iatom] * pow(lri_zeta/PI, 1.5), X1 * X0 * (1/lri_zeta) * lri_cc[iatom] * pow(lri_zeta/PI, 1.5));

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

        double YVerticalTemp[PRIM_INT_LRI_GRAD_LEN];
        FmT(I + J + 1, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)), YVerticalTemp);

        for (uint32_t i = 0; i <= I + J + 1; i++) {
            YVerticalTemp[i] *= X2;
        }

        lri::vertical2(I, J + 1, 0, 1, YVerticalTemp, store2,
                Px - RAx, Py - RAy, Pz - RAz,
                (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                Qx - RCx, Qy - RCy, Qz - RCz,
                (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);

        for (uint32_t i = Sumindex[0]; i < Sumindex[2]; i++) {
            for (uint32_t j = Sumindex[I]; j < Sumindex[I + J + 2]; j++) {
                if (i < STOREDIM && j < STOREDIM) {
                    LOCSTORE(store, j, i , STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM);
                    //printf("store2 %d %d %f \n", i, j, LOCSTORE(store2, j, i, STOREDIM, STOREDIM));
                }
            }
        }

        for (uint32_t i = Sumindex[0]; i < Sumindex[2]; i++) {
            for (uint32_t j = Sumindex[I + 1]; j < Sumindex[I + J + 3]; j++) {
                if (i < STOREDIM && j < STOREDIM) {
                    LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * AA * 2.0;
                    LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * BB * 2.0;
                    //printf("storeAA storeBB %d %d %f %f \n", j, i, LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM), LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM));
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

    uint32_t AStart = katom[II] * 3;
    uint32_t BStart = katom[JJ] * 3;
    uint32_t CStart = iatom * 3;

    QUICKDouble RBx, RBy, RBz;

    RBx = LOC2(xyz, 0, katom[JJ], 3, natom);
    RBy = LOC2(xyz, 1, katom[JJ], 3, natom);
    RBz = LOC2(xyz, 2, katom[JJ], 3, natom);

    uint32_t III1 = LOC2(Qsbasis, II, I, nshell, 4);
    uint32_t III2 = LOC2(Qfbasis, II, I, nshell, 4);
    uint32_t JJJ1 = LOC2(Qsbasis, JJ, J, nshell, 4);
    uint32_t JJJ2 = LOC2(Qfbasis, JJ, J, nshell, 4);

    for (uint32_t III = III1; III <= III2; III++) {
        for (uint32_t JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            QUICKDouble Yaax, Yaay, Yaaz;
            QUICKDouble Ybbx, Ybby, Ybbz;

            hrrwholegrad_lri(&Yaax, &Yaay, &Yaaz, &Ybbx, &Ybby, &Ybbz,
                    J, III, JJJ, store, storeAA, storeBB,
                    RAx, RAy, RAz, RBx, RBy, RBz,
                    nbasis, cons, KLMN, trans);

  #if defined(OSHELL)
            QUICKDouble DENSEJI = (QUICKDouble) (LOC2(dense, JJJ, III, nbasis, nbasis)
                    + LOC2(denseb, JJJ, III, nbasis, nbasis));
  #else
            QUICKDouble DENSEJI = (QUICKDouble) LOC2(dense, JJJ, III, nbasis, nbasis);
  #endif

            QUICKDouble constant;
            if (III != JJJ) {
                constant = 2.0 * DENSEJI;
            } else {
                constant = DENSEJI;
            }

//            printf("iatom %d %d %d %d %d dmx: %f Y: %f %f %f %f %f %f %f %f %f \n",
//                    iatom, II, JJ, III, JJJ, constant, RAx, RBx, RCx, Yaax, Yaay, Yaaz, Ybbx, Ybby, Ybbz);

            AGradx += constant * Yaax;
            AGrady += constant * Yaay;
            AGradz += constant * Yaaz;

            BGradx += constant * Ybbx;
            BGrady += constant * Ybby;
            BGradz += constant * Ybbz;
        }
    }

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
        CStart = (iatom - natom) * 3;
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
        CStart = (iatom - natom) * 3;
        atomicAdd(&ptchg_grad[CStart], -AGradx - BGradx);
        atomicAdd(&ptchg_grad[CStart + 1], -AGrady - BGrady);
        atomicAdd(&ptchg_grad[CStart + 2], -AGradz - BGradz);
    }    
#endif
}


#else
/*
   iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
   performance algrithem for electron intergral evaluation. See description below for details
*/
  #if defined(OSHELL)
    #if defined(int_spdf2)
__device__ static inline void iclass_oshell_lri_grad_spdf2
    #endif
  #else
    #if defined(int_spdf2)
__device__ static inline void iclass_lri_grad_spdf2
    #endif
  #endif
    (uint32_t I, uint32_t J, uint32_t II, uint32_t JJ, uint32_t iatom, uint32_t totalatom,
     uint32_t natom, uint32_t nbasis, uint32_t nshell, uint32_t jbasis,
     QUICKDouble const * const xyz, QUICKDouble const * const allxyz,
     uint32_t const * const kstart, uint32_t const * const katom,
     uint32_t const * const kprim, uint32_t const * const Ksumtype, uint32_t const * const Qstart,
     uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
     QUICKDouble const * const cons, QUICKDouble const * const gcexpo, uint32_t const * const KLMN,
     uint32_t prim_total, uint32_t const * const prim_start, QUICKDouble * const dense,
#if defined(OSHELL)
     QUICKDouble * const denseb,
#endif
     QUICKDouble const * const Xcoeff, QUICKDouble const * const expoSum,
     QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
     QUICKDouble const * const weightedCenterZ,
#if defined(USE_LEGACY_ATOMICS)
     QUICKULL * const gradULL,
#else
     QUICKDouble * const grad,
#endif
     QUICKDouble * const store, QUICKDouble * const store2,
     QUICKDouble * const storeAA, QUICKDouble * const storeBB,
     uint32_t const * const trans, uint32_t const * const Sumindex)
{
    /*
       kAtom A, B, C, D is the coresponding atom for shell ii, jj, kk, ll
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
    uint32_t kPrimK = 1;
    uint32_t kPrimL = 1;

    uint32_t kStartI = kstart[II];
    uint32_t kStartJ = kstart[JJ];

    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;

    uint32_t AStart = katom[II] * 3;
    uint32_t BStart = katom[JJ] * 3;
    uint32_t CStart = iatom * 3;

    /*
       store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
       of GPU limitation, we can not do that now.

       See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
   */

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
        QUICKDouble X1 = LOC4(Xcoeff, kStartI + III, kStartJ + JJJ,
                I - Qstart[II], J - Qstart[JJ], jbasis, jbasis, 2, 2);

        for (uint32_t j = 0; j < kPrimK * kPrimL; j++) {
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
            QUICKDouble CD = lri_zeta;
            QUICKDouble ABCD = 1.0 / (AB + CD);

            /*
               X2 is the multiplication of four indices normalized coeffecient
               */
            QUICKDouble X2 = sqrt(ABCD) * X1 * X0 * (1/lri_zeta) * lri_cc[iatom] * pow(lri_zeta/PI, 1.5);

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

            double YVerticalTemp[PRIM_INT_LRI_GRAD_LEN];
            FmT(I + J + 2, AB * CD * ABCD * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)), YVerticalTemp);

            for (uint32_t i = 0; i <= I + J + 2; i++) {
                YVerticalTemp[i] *= X2;
            }

            for (uint32_t i = Sumindex[0]; i < Sumindex[3]; i++) {
                for (uint32_t j = Sumindex[I]; j < Sumindex[I + J + 3]; j++) {
                    if (i < STOREDIM && j < STOREDIM && !(i >= Sumindex[I + J + 2] && j >= Sumindex[2])) {
                        LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0.0;
                    }
                }
            }

  #if defined(int_spdf2)
            lri::vertical2_spdf2(I, J + 1, 0, 1, YVerticalTemp, store2,
                    Px - RAx, Py - RAy, Pz - RAz,
                    (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                    Qx - RCx, Qy - RCy, Qz - RCz,
                    (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
  #endif

            QUICKDouble RBx, RBy, RBz;

            RBx = LOC2(xyz, 0, katom[JJ], 3, natom);
            RBy = LOC2(xyz, 1, katom[JJ], 3, natom);
            RBz = LOC2(xyz, 2, katom[JJ], 3, natom);

            uint32_t III1 = LOC2(Qsbasis, II, I, nshell, 4);
            uint32_t III2 = LOC2(Qfbasis, II, I, nshell, 4);
            uint32_t JJJ1 = LOC2(Qsbasis, JJ, J, nshell, 4);
            uint32_t JJJ2 = LOC2(Qfbasis, JJ, J, nshell, 4);

            uint32_t nbasis = nbasis;

            for (uint32_t III = III1; III <= III2; III++) {
                for (uint32_t JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
                    QUICKDouble Yaax, Yaay, Yaaz;
                    QUICKDouble Ybbx, Ybby, Ybbz;

  #if defined(int_spdf2)
                    hrrwholegrad_lri2_2
  #else
                    hrrwholegrad_lri2
  #endif
                    (&Yaax, &Yaay, &Yaaz, &Ybbx, &Ybby, &Ybbz,
                     J, III, JJJ, store2, AA, BB,
                     RAx, RAy, RAz, RBx, RBy, RBz,
                     nbasis, cons, KLMN, trans);

  #if defined(OSHELL)
                    QUICKDouble DENSEJI = (QUICKDouble) (LOC2(dense, JJJ, III, nbasis, nbasis)
                            + LOC2(denseb, JJJ, III, nbasis, nbasis));
  #else
                    QUICKDouble DENSEJI = (QUICKDouble) LOC2(dense, JJJ, III, nbasis, nbasis);
  #endif

                    QUICKDouble constant;
                    if (III != JJJ) {
                        constant = 2.0 * DENSEJI;
                    } else {
                        constant = DENSEJI;
                    }

                    AGradx += constant * Yaax;
                    AGrady += constant * Yaay;
                    AGradz += constant * Yaaz;

                    BGradx += constant * Ybbx;
                    BGrady += constant * Ybby;
                    BGradz += constant * Ybbz;
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

    if (iatom < natom) {
        GPUATOMICADD(&gradULL[CStart], -AGradx - BGradx, GRADSCALE);
        GPUATOMICADD(&gradULL[CStart + 1], -AGrady - BGrady, GRADSCALE);
        GPUATOMICADD(&gradULL[CStart + 2], -AGradz - BGradz, GRADSCALE);
    } else {
        CStart = (iatom - natom) * 3;
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
        CStart = (iatom - natom) * 3;
        atomicAdd(&ptchg_grad[CStart], -AGradx - BGradx);
        atomicAdd(&ptchg_grad[CStart + 1], -AGrady - BGrady);
        atomicAdd(&ptchg_grad[CStart + 2], -AGradz - BGradz);
    }
  #endif
}
#endif


#if defined(int_spd)
__global__ void
//__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1)
k_get_lri_grad
#elif defined(int_spdf2)
__global__ void
//__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1)
k_get_lri_grad_spdf2
#endif
    (uint32_t natom, uint32_t nbasis, uint32_t nshell, uint32_t jbasis,
     QUICKDouble const * const xyz, QUICKDouble const * const allxyz,
     uint32_t const * const kstart, uint32_t const * const katom,
     uint32_t const * const kprim, uint32_t const * const Ksumtype, uint32_t const * const Qstart,
     uint32_t const * const Qsbasis, uint32_t const * const Qfbasis,
     uint32_t const * const sorted_Qnumber, uint32_t const * const sorted_Q,
     QUICKDouble const * const cons, QUICKDouble const * const gcexpo, uint32_t const * const KLMN,
     uint32_t prim_total, uint32_t const * const prim_start, QUICKDouble * const dense,
#if defined(OSHELL)
     QUICKDouble * const denseb,
#endif
     QUICKDouble const * const Xcoeff, QUICKDouble const * const expoSum,
     QUICKDouble const * const weightedCenterX, QUICKDouble const * const weightedCenterY,
     QUICKDouble const * const weightedCenterZ, uint32_t sqrQshell, int2 const * const sorted_YCutoffIJ,
#if defined(USE_LEGACY_ATOMICS)
     QUICKULL * const gradULL,
#else
     QUICKDouble * const grad,
#endif
#if defined(MPIV_GPU)
     unsigned char const * const mpi_bcompute,
#endif
     QUICKDouble * const store, QUICKDouble * const store2,
     QUICKDouble * const storeAA, QUICKDouble * const storeBB,
     uint32_t const * const trans, uint32_t const * const Sumindex)
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    uint32_t totalatom = natom + nextatom;
    QUICKULL jshell = (QUICKULL) sqrQshell;
#if defined(USE_LEGACY_ATOMICS)
    extern __shared__ QUICKULL smem2[];
    QUICKULL *sgradULL = smem2;
    uint32_t *strans = (uint32_t *) &sgradULL[3u * natom];
    uint32_t *sSumindex = &strans[TRANSDIM * TRANSDIM * TRANSDIM];

    for (int i = threadIdx.x; i < 3 * (int) natom; i += blockDim.x) {
      sgradULL[i] = 0ull;
    }
    for (int i = threadIdx.x; i < TRANSDIM * TRANSDIM * TRANSDIM; i += blockDim.x) {
        strans[i] = trans[i];
    }
    for (int i = threadIdx.x; i < 10; i += blockDim.x) {
        sSumindex[i] = Sumindex[i];
    }
#else
    extern __shared__ QUICKDouble smem2[];
    QUICKDouble *sgrad = smem2;
    uint32_t *strans = (uint32_t *) &sgrad[3u * natom];
    uint32_t *sSumindex = &strans[TRANSDIM * TRANSDIM * TRANSDIM];

    for (int i = threadIdx.x; i < 3 * (int) natom; i += blockDim.x) {
        sgrad[i] = 0.0;
    }
    for (int i = threadIdx.x; i < TRANSDIM * TRANSDIM * TRANSDIM; i += blockDim.x) {
        strans[i] = trans[i];
    }
    for (int i = threadIdx.x; i < 10; i += blockDim.x) {
        sSumindex[i] = Sumindex[i];
    }
#endif

    __syncthreads();

    for (QUICKULL i = offset; i < totalatom * jshell; i += totalThreads) {
        QUICKULL iatom = (QUICKULL) i / jshell;
        QUICKULL b = (QUICKULL) (i - iatom * jshell);

#if defined(MPIV_GPU)
        if (mpi_bcompute[b] > 0) {
#endif
        int II = sorted_YCutoffIJ[b].x;
        int JJ = sorted_YCutoffIJ[b].y;

        uint32_t ii = sorted_Q[II];
        uint32_t jj = sorted_Q[JJ];

        uint32_t iii = sorted_Qnumber[II];
        uint32_t jjj = sorted_Qnumber[JJ];

#if defined(int_spd)
        {
            iclass_lri_grad
#elif defined(int_spdf2)
        if (iii + jjj >= 4) {
            iclass_lri_grad_spdf2
#endif
                (iii, jjj, ii, jj, iatom, totalatom,
                 natom, nbasis, nshell, jbasis, xyz, allxyz,
                 kstart, katom, kprim, Ksumtype, Qstart, Qsbasis, Qfbasis,
                 cons, gcexpo, KLMN, prim_total, prim_start, dense,
#if defined(OSHELL)
                 denseb,
#endif
                 Xcoeff, expoSum, weightedCenterX, weightedCenterY, weightedCenterZ,
#if defined(USE_LEGACY_ATOMICS)
                 sgradULL,
#else
                 sgrad,
#endif
                 store + offset, store2 + offset,
                 storeAA + offset, storeBB + offset,
                 strans, sSumindex);
        }
#if defined(MPIV_GPU)
        }
#endif
    }

    __syncthreads();

    for (int i = threadIdx.x; i < 3 * (int) natom; i += blockDim.x) {
#if defined(USE_LEGACY_ATOMICS)
        atomicAdd(&gradULL[i], sgradULL[i]);
#else
        atomicAdd(&grad[i], sgrad[i]);
#endif
    }
}
