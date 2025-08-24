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
#define STOREDIM STOREDIM_XL
#define LOCSTORE(A,i1,i2,d1,d2) (A[((i2) * (d1) + (i1)) * gridDim.x * blockDim.x])
#define VY(a,b,c) (YVerticalTemp[(c)])


__device__ static inline void ERint_grad_vertical_spdf_7_1(uint32_t I, uint32_t J, uint32_t K, uint32_t L,
        const QUICKDouble Ptempx, const QUICKDouble Ptempy, const QUICKDouble Ptempz,
        const QUICKDouble WPtempx, const QUICKDouble WPtempy, const QUICKDouble WPtempz,
        const QUICKDouble Qtempx, const QUICKDouble Qtempy, const QUICKDouble Qtempz,
        const QUICKDouble WQtempx, const QUICKDouble WQtempy, const QUICKDouble WQtempz,
        const QUICKDouble ABCDtemp, const QUICKDouble ABtemp, const QUICKDouble CDtemp,
        const QUICKDouble ABcom, const QUICKDouble CDcom,
        QUICKDouble * const store, QUICKDouble * const YVerticalTemp)
{
    if (I + J >= 7 && K + L >= 7) {
#include "iclass_ksks_p1.h"
    }
}
