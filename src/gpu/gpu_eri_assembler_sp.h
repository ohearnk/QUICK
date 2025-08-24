/*
   !---------------------------------------------------------------------!
   ! Written by QUICK-GenInt code generator on 03/27/2023                !
   !                                                                     !
   ! Copyright (C) 2023-2024 Merz lab                                    !
   ! Copyright (C) 2023-2024 Götz lab                                    !
   !                                                                     !
   ! This Source Code Form is subject to the terms of the Mozilla Public !
   ! License, v. 2.0. If a copy of the MPL was not distributed with this !
   ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
   !_____________________________________________________________________!
   */

#undef STOREDIM
#undef VY
#undef LOCSTORE
#define STOREDIM STOREDIM_T
#define LOCSTORE(A,i1,i2,d1,d2) (A[((i2) * (d1) + (i1)) * gridDim.x * blockDim.x])
#define VY(a,b,c) (YVerticalTemp[c])


__device__ static inline void ERint_vertical_sp(uint32_t I, uint32_t J, uint32_t K, uint32_t L,
        QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,
        QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz,
        QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,
        QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz,
        QUICKDouble ABCDtemp, QUICKDouble ABtemp, QUICKDouble CDtemp,
        QUICKDouble ABcom, QUICKDouble CDcom,
        QUICKDouble * const store, QUICKDouble * const YVerticalTemp) {
#include "iclass_ssss.h"
    if (K + L >= 1) {
        if (K <= 1 && I == 0) {
#include "iclass_ssps.h"
        }
        if (K + L >= 2) {
            if (K <= 2 && I == 0) {
#include "iclass_ssds.h"
            }
        }
    }
    if (I + J >= 1) {
        if (I <= 1) {
#include "iclass_psss.h"
        }
        if (K + L >= 1) {
            if (K <= 1 && I <= 1) {
#include "iclass_psps.h"
            }
            if (K + L >= 2) {
                if (K <= 2 && I <= 1) {
#include "iclass_psds.h"
                }
            }
            if (I + J >= 2) {
                if (K <= 1 && I <= 2) {
#include "iclass_dsps.h"
                }
                if (K + L >= 2) {
                    if (K <= 2 && I <= 2) {
#include "iclass_dsds.h"
                    }
                }
            }
        }
        if (I + J >= 2) {
            if (K == 0 && I <= 2) {
#include "iclass_dsss.h"
            }
        }
    }
}
