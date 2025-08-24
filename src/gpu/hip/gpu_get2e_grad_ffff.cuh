#include "hip/hip_runtime.h"
//
//  gpu_get2e_subs_grad.h
//  new_quick
//
//  Created by Yipu Miao on 1/22/14.
//
//

#if !defined(__QUICK_GPU_GET2E_GRAD_FFFF_H_)
#define __QUICK_GPU_GET2E_GRAD_FFFF_H_

#undef STOREDIM
#if defined int_spdf4
  #define STOREDIM STOREDIM_XL
  #define STORE_INIT (4)
  #define STORE_DIM (80)
  #define STORE_INIT_I_AA (4)
  #define STORE_INIT_J_AA (10)
  #define STORE_DIM_I_AA (80)
  #define STORE_DIM_J_AA (110)
  
  #define STORE_INIT_I_CC (10)
  #define STORE_INIT_J_CC (4)
  #define STORE_DIM_I_CC (110)
  #define STORE_DIM_J_CC (80)
#endif

#undef STOREDIM
#define STOREDIM STOREDIM_XL

// support up to f functions
#define PRIM_INT_ERI_GRAD_FFFF_LEN (15)


#if !defined(OSHELL)
  #define FMT_NAME FmT
  #include "../gpu_fmt.h"


__device__ static inline uint32_t lefthrr_2(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint32_t KLMNAx, uint32_t KLMNAy, uint32_t KLMNAz,
        uint32_t KLMNBx, uint32_t KLMNBy, uint32_t KLMNBz,
        QUICKDouble * const coefAngularL, uint32_t * const angularL, uint32_t * const smem_uint8)
{
    uint32_t numAngularL;

    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + KLMNBx,
            KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    if (KLMNBx == 2 || KLMNBy == 2 || KLMNBz == 2) {
        numAngularL = 3;
        QUICKDouble tmp;

        if (KLMNBx == 2) {
            tmp = RAx - RBx;
            angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else if(KLMNBy == 2) {
            tmp = RAy - RBy;
            angularL[1] =  LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else if (KLMNBz == 2) {
            tmp = RAz - RBz;
            angularL[1] =  LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        coefAngularL[1] = 2 * tmp;
        coefAngularL[2]= tmp * tmp;

        angularL[numAngularL - 1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

        return numAngularL;
    } else {
        numAngularL = 4;
        QUICKDouble tmp, tmp2;

        if (KLMNBx == 1 && KLMNBy == 1) {
            tmp = RAx - RBx;
            tmp2 = RAy - RBy;
            angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else if (KLMNBx == 1 && KLMNBz == 1) {
            tmp = RAx - RBx;
            tmp2 = RAz - RBz;
            angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else if (KLMNBy == 1 && KLMNBz == 1) {
            tmp = RAy - RBy;
            tmp2 = RAz - RBz;
            angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        coefAngularL[1] = tmp;
        coefAngularL[2] = tmp2;
        coefAngularL[3] = tmp * tmp2;

        angularL[numAngularL - 1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

        return numAngularL;
    }
}


__device__ static inline uint32_t lefthrr(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint32_t KLMNAx, uint32_t KLMNAy, uint32_t KLMNAz,
        uint32_t KLMNBx, uint32_t KLMNBy, uint32_t KLMNBz,
        QUICKDouble * const coefAngularL, uint32_t * const angularL, uint32_t * const smem_uint8)
{
    uint32_t numAngularL;

    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    if (KLMNBx == 3 || KLMNBy == 3 || KLMNBz == 3) {
        numAngularL = 4;
        QUICKDouble tmp;

        if (KLMNBx == 3) {
            tmp = RAx - RBx;
            angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else if (KLMNBy == 3) {
            tmp = RAy - RBy;
            angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else if (KLMNBz == 3) {
            tmp = RAz - RBz;
            angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        coefAngularL[1] = 3 * tmp;
        coefAngularL[2] = 3 * tmp * tmp;
        coefAngularL[3] = tmp * tmp * tmp;

        angularL[numAngularL - 1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBx == 1 && KLMNBy == 1) {
        numAngularL = 8;
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAy - RBy;
        QUICKDouble tmp3 = RAz - RBz;

        coefAngularL[1] = tmp;
        coefAngularL[2] = tmp2;
        coefAngularL[3] = tmp3;
        coefAngularL[4] = tmp * tmp2;
        coefAngularL[5] = tmp * tmp3;
        coefAngularL[6] = tmp2 * tmp3;
        coefAngularL[7] = tmp * tmp2 * tmp3;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

        angularL[numAngularL - 1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else {
        numAngularL = 6;
        QUICKDouble tmp;
        QUICKDouble tmp2;

        if (KLMNBx == 1) {
            tmp = RAx - RBx;
        } else if (KLMNBy == 1) {
            tmp = RAy - RBy;
        } else if (KLMNBz == 1) {
            tmp = RAz - RBz;
        }

        if (KLMNBx == 2) {
            tmp2 = RAx - RBx;
        } else if (KLMNBy == 2) {
            tmp2 = RAy - RBy;
        } else if (KLMNBz == 2) {
            tmp2 = RAz - RBz;
        }

        coefAngularL[1] = tmp;
        coefAngularL[2] = 2 * tmp2;
        coefAngularL[3] = 2 * tmp * tmp2;
        coefAngularL[4] = tmp2 * tmp2;
        coefAngularL[5] = tmp * tmp2 * tmp2;

        if (KLMNBx == 2) {
            angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        if (KLMNBy == 2) {
            angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        if (KLMNBz == 2) {
            angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        if (KLMNBx == 1) {
            // 120
            if (KLMNBy == 2) {
                angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            // 102
            } else {
                angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }

        if (KLMNBy == 1) {
            // 210
            if (KLMNBx == 2) {
                angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            // 012
            } else {
                angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }

        if (KLMNBz == 1) {
            // 201
            if (KLMNBx == 2) {
                angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            // 021
            } else {
                angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }

        if (KLMNBx == 1) {
            angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        if (KLMNBy == 1) {
            angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        if (KLMNBz == 1) {
            angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        angularL[numAngularL - 1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    }

    return numAngularL;
}


__device__ static inline uint32_t lefthrr_4(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint32_t KLMNAx, uint32_t KLMNAy, uint32_t KLMNAz,
        uint32_t KLMNBx, uint32_t KLMNBy, uint32_t KLMNBz,
        QUICKDouble * const coefAngularL, uint32_t * const angularL, uint32_t * const smem_uint8)
{
    uint32_t numAngularL;

    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + KLMNBx,
            KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    if (KLMNBx == 4) {
        numAngularL = 5;
        QUICKDouble tmp = RAx - RBx;

        coefAngularL[1] = 4 * tmp;
        coefAngularL[2] = 6 * tmp * tmp;
        coefAngularL[3] = 4 * tmp * tmp * tmp;
        coefAngularL[4] = tmp * tmp * tmp * tmp;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 3, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBy == 4) {
        numAngularL = 5;
        QUICKDouble tmp = RAy - RBy;
        coefAngularL[1] = 4 * tmp;
        coefAngularL[2] = 6 * tmp * tmp;
        coefAngularL[3] = 4 * tmp * tmp * tmp;
        coefAngularL[4] = tmp * tmp * tmp * tmp;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 3, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBz == 4) {
        numAngularL = 5;

        QUICKDouble tmp = RAz - RBz;
        coefAngularL[1] = 4 * tmp;
        coefAngularL[2] = 6 * tmp * tmp;
        coefAngularL[3] = 4 * tmp * tmp * tmp;
        coefAngularL[4] = tmp * tmp * tmp * tmp;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 3, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBx == 1 && KLMNBy == 3) {
        numAngularL = 8;
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAy - RBy;

        coefAngularL[1] = tmp;
        coefAngularL[2] = 3 * tmp2;
        coefAngularL[3] = 3 * tmp * tmp2;
        coefAngularL[4] = 3 * tmp2 * tmp2;
        coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
        coefAngularL[6] = tmp2 * tmp2 * tmp2;
        coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 3, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBx == 3 && KLMNBy == 1) {
        numAngularL = 8;
        QUICKDouble tmp = RAy - RBy;
        QUICKDouble tmp2 = RAx - RBx;

        coefAngularL[1] = tmp;
        coefAngularL[2] = 3 * tmp2;
        coefAngularL[3] = 3 * tmp * tmp2;
        coefAngularL[4] = 3 * tmp2 * tmp2;
        coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
        coefAngularL[6] = tmp2 * tmp2 * tmp2;
        coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 3, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBx == 1 && KLMNBz == 3) {
        numAngularL = 8;
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAz - RBz;

        coefAngularL[1] = tmp;
        coefAngularL[2] = 3 * tmp2;
        coefAngularL[3] = 3 * tmp * tmp2;
        coefAngularL[4] = 3 * tmp2 * tmp2;
        coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
        coefAngularL[6] = tmp2 * tmp2 * tmp2;
        coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 3, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

    } else if (KLMNBx == 3 && KLMNBz == 1) {
        numAngularL = 8;
        QUICKDouble tmp = RAz - RBz;
        QUICKDouble tmp2 = RAx - RBx;

        coefAngularL[1] = tmp;
        coefAngularL[2] = 3 * tmp2;
        coefAngularL[3] = 3 * tmp * tmp2;
        coefAngularL[4] = 3 * tmp2 * tmp2;
        coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
        coefAngularL[6] = tmp2 * tmp2 * tmp2;
        coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 3, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBy == 1 && KLMNBz == 3) {
        numAngularL = 8;
        QUICKDouble tmp = RAy - RBy;
        QUICKDouble tmp2 = RAz - RBz;

        coefAngularL[1] = tmp;
        coefAngularL[2] = 3 * tmp2;
        coefAngularL[3] = 3 * tmp * tmp2;
        coefAngularL[4] = 3 * tmp2 * tmp2;
        coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
        coefAngularL[6] = tmp2 * tmp2 * tmp2;
        coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx,   KLMNAy,   KLMNAz+3, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx,   KLMNAy+1, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx,   KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBy == 3 && KLMNBz == 1) {
        numAngularL = 8;
        QUICKDouble tmp = RAz - RBz;
        QUICKDouble tmp2 = RAy - RBy;

        coefAngularL[1] = tmp;
        coefAngularL[2] = 3 * tmp2;
        coefAngularL[3] = 3 * tmp * tmp2;
        coefAngularL[4] = 3 * tmp2 * tmp2;
        coefAngularL[5] = 3 * tmp * tmp2 * tmp2;
        coefAngularL[6] = tmp2 * tmp2 * tmp2;
        coefAngularL[7] = tmp * tmp2 * tmp2 * tmp2;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 3, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBx == 2 && KLMNBy == 2) {
        numAngularL = 9;
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAy - RBy;

        coefAngularL[1] = 2 * tmp;
        coefAngularL[2] = 2 * tmp2;
        coefAngularL[3] = 4 * tmp * tmp2;
        coefAngularL[4] = tmp * tmp;
        coefAngularL[5] = tmp2 * tmp2;
        coefAngularL[6] = 2 * tmp * tmp2 * tmp2;
        coefAngularL[7] = 2 * tmp * tmp * tmp2;
        coefAngularL[8] = tmp * tmp * tmp2 * tmp2;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[7] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBx == 2 && KLMNBz == 2) {
        numAngularL = 9;
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAz - RBz;

        coefAngularL[1] = 2 * tmp;
        coefAngularL[2] = 2 * tmp2;
        coefAngularL[3] = 4 * tmp * tmp2;
        coefAngularL[4] = tmp * tmp;
        coefAngularL[5] = tmp2 * tmp2;
        coefAngularL[6] = 2 * tmp * tmp2 * tmp2;
        coefAngularL[7] = 2 * tmp * tmp * tmp2;
        coefAngularL[8] = tmp * tmp * tmp2 * tmp2;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[7] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBy == 2 && KLMNBz == 2) {
        numAngularL = 9;
        QUICKDouble tmp = RAy - RBy;
        QUICKDouble tmp2 = RAz - RBz;

        coefAngularL[1] = 2 * tmp;
        coefAngularL[2] = 2 * tmp2;
        coefAngularL[3] = 4 * tmp * tmp2;
        coefAngularL[4] = tmp * tmp;
        coefAngularL[5] = tmp2 * tmp2;
        coefAngularL[6] = 2 * tmp * tmp2 * tmp2;
        coefAngularL[7] = 2 * tmp * tmp * tmp2;
        coefAngularL[8] = tmp * tmp * tmp2 * tmp2;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[7] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBx == 1 && KLMNBy == 1 && KLMNBz == 2) {
        numAngularL = 12;
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAy - RBy;
        QUICKDouble tmp3 = RAz - RBz;

        coefAngularL[1] = tmp;
        coefAngularL[2] = tmp2;
        coefAngularL[3] = 2 * tmp3;
        coefAngularL[4] = tmp * tmp2;
        coefAngularL[5] = 2 * tmp * tmp3;
        coefAngularL[6] = 2 * tmp2 * tmp3;
        coefAngularL[7] = tmp3 * tmp3;
        coefAngularL[8] = 2 * tmp * tmp2 * tmp3;
        coefAngularL[9] = tmp * tmp3 * tmp3;
        coefAngularL[10] = tmp2 * tmp3 * tmp3;
        coefAngularL[11] = tmp * tmp2 * tmp3 * tmp3;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy+1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[7] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[8] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[9] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[10] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBx == 1 && KLMNBz == 1 && KLMNBy == 2) {
        numAngularL = 12;
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAz - RBz;
        QUICKDouble tmp3 = RAy - RBy;

        coefAngularL[1] = tmp;
        coefAngularL[2] = tmp2;
        coefAngularL[3] = 2 * tmp3;
        coefAngularL[4] = tmp * tmp2;
        coefAngularL[5] = 2 * tmp * tmp3;
        coefAngularL[6] = 2 * tmp2 * tmp3;
        coefAngularL[7] = tmp3 * tmp3;
        coefAngularL[8] = 2 * tmp * tmp2 * tmp3;
        coefAngularL[9] = tmp * tmp3 * tmp3;
        coefAngularL[10] = tmp2 * tmp3 * tmp3;
        coefAngularL[11] = tmp * tmp2 * tmp3 * tmp3;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[7] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[8] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[9] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[10] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (KLMNBy == 1 && KLMNBz == 1 && KLMNBx == 2) {
        numAngularL = 12;
        QUICKDouble tmp = RAy - RBy;
        QUICKDouble tmp2 = RAz - RBz;
        QUICKDouble tmp3 = RAx - RBx;

        coefAngularL[1] = tmp;
        coefAngularL[2] = tmp2;
        coefAngularL[3] = 2 * tmp3;
        coefAngularL[4] = tmp * tmp2;
        coefAngularL[5] = 2 * tmp * tmp3;
        coefAngularL[6] = 2 * tmp2 * tmp3;
        coefAngularL[7] = tmp3 * tmp3;
        coefAngularL[8] = 2 * tmp * tmp2 * tmp3;
        coefAngularL[9] = tmp * tmp3 * tmp3;
        coefAngularL[10] = tmp2 * tmp3 * tmp3;
        coefAngularL[11] = tmp * tmp2 * tmp3 * tmp3;

        angularL[1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[4] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[5] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[6] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[7] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[8] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[9] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[10] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    }

    angularL[numAngularL - 1] = LOC3(DEV_SIM_UINT8_TRANS, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

    return numAngularL;
}


__device__ static inline void hrrwholegrad2_ffff(QUICKDouble * const Yaax, QUICKDouble * const Yaay, QUICKDouble * const Yaaz,
        QUICKDouble * const Ybbx, QUICKDouble * const Ybby, QUICKDouble * const Ybbz,
        QUICKDouble * const Yccx, QUICKDouble * const Yccy, QUICKDouble * const Yccz,
        const uint32_t III, uint32_t JJJ, const uint32_t KKK, const uint32_t LLL,
        QUICKDouble * const store, QUICKDouble * const storeAA, QUICKDouble * const storeBB, QUICKDouble * const storeCC,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz,
        uint32_t ** const smem_uint8_ptr, uint32_t * const smem_uint32,
        uint32_t ** const smem_uint32_ptr, QUICKDouble ** const smem_dbl_ptr,
        unsigned char ** const smem_char_ptr, uint32_t * const smem_uint8)
{
    unsigned char angularL[12], angularR[12];
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

    QUICKDouble constant = DEV_SIM_DBL_PTR_CONS[III] * DEV_SIM_DBL_PTR_CONS[JJJ]
        * DEV_SIM_DBL_PTR_CONS[KKK] * DEV_SIM_DBL_PTR_CONS[LLL];
    uint32_t numAngularL, numAngularR;

    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, KKK, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, KKK, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, KKK, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, LLL, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, LLL, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, LLL, 3, DEV_SIM_UINT32_NBASIS),
            coefAngularR, angularR, smem_uint8);

    //  Part A - x
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS) + 1,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            coefAngularL, angularL, smem_uint8);

    for (uint32_t i = 0; i < numAngularL; i++) {
        for (uint32_t j = 0; j < numAngularR; j++) {
//            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
            *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j]
                * LOCSTORE(storeAA, angularL[i] - STORE_INIT_J_AA, angularR[j] - STORE_INIT_I_AA, STORE_DIM_J_AA, STORE_DIM_I_AA);

//            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS) + 1,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            coefAngularL, angularL, smem_uint8);

    for (uint32_t i = 0; i < numAngularL; i++) {
        for (uint32_t j = 0; j < numAngularR; j++) {
//            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
            *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j]
                * LOCSTORE(storeAA, angularL[i] - STORE_INIT_J_AA, angularR[j] - STORE_INIT_I_AA, STORE_DIM_J_AA, STORE_DIM_I_AA);
//            }
        }
    }

    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS) + 1,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            coefAngularL, angularL, smem_uint8);

    for (uint32_t i = 0; i < numAngularL; i++) {
        for (uint32_t j = 0; j < numAngularR; j++) {
//            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
            *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j]
                * LOCSTORE(storeAA, angularL[i] - STORE_INIT_J_AA, angularR[j] - STORE_INIT_I_AA, STORE_DIM_J_AA, STORE_DIM_I_AA);
//            }
        }
    }

    numAngularL = lefthrr_4(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS) + 1,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            coefAngularL, angularL, smem_uint8);

    for (uint32_t i = 0; i < numAngularL; i++) {
        for (uint32_t j = 0; j < numAngularR; j++) {
//            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
            *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j]
                * LOCSTORE(storeBB, angularL[i] - STORE_INIT_J_AA, angularR[j] - STORE_INIT_I_AA, STORE_DIM_J_AA, STORE_DIM_I_AA);
//            }
        }
    }

    numAngularL = lefthrr_4(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS) + 1,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            coefAngularL, angularL, smem_uint8);

    for (uint32_t i = 0; i < numAngularL; i++) {
        for (uint32_t j = 0; j < numAngularR; j++) {
//            if (angularL[i] <= STOREDIM && angularR[j] < STOREDIM) {
            *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j]
                * LOCSTORE(storeBB, angularL[i] - STORE_INIT_J_AA, angularR[j] - STORE_INIT_I_AA, STORE_DIM_J_AA, STORE_DIM_I_AA);
//            }
        }
    }

    numAngularL = lefthrr_4(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS) + 1,
            coefAngularL, angularL, smem_uint8);

    for (uint32_t i = 0; i < numAngularL; i++) {
        for (uint32_t j = 0; j < numAngularR; j++) {
//            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
            *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j]
                * LOCSTORE(storeBB, angularL[i] - STORE_INIT_J_AA, angularR[j] - STORE_INIT_I_AA, STORE_DIM_J_AA, STORE_DIM_I_AA);
//            }
        }
    }

    if (LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS) >= 1) {
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS) - 1,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                coefAngularL, angularL, smem_uint8);

        for (uint32_t i = 0; i < numAngularL; i++) {
            for (uint32_t j = 0; j < numAngularR; j++) {
//                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaax = *Yaax - LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS) * coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i] - STORE_INIT, angularR[j] - STORE_INIT, STORE_DIM, STORE_DIM);
//                }
            }
        }
    }

    if (LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS) >= 1) {
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS) - 1,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                coefAngularL, angularL, smem_uint8);

        for (uint32_t i = 0; i < numAngularL; i++) {
            for (uint32_t j = 0; j < numAngularR; j++) {
//                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaay = *Yaay - LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS) * coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i] - STORE_INIT, angularR[j] - STORE_INIT, STORE_DIM, STORE_DIM);
//                }
            }
        }
    }

    if (LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS) >= 1) {
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS) - 1,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                coefAngularL, angularL, smem_uint8);

        for (uint32_t i = 0; i < numAngularL; i++) {
            for (uint32_t j = 0; j < numAngularR; j++) {
//                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yaaz = *Yaaz - LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS) * coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i] - STORE_INIT, angularR[j] - STORE_INIT, STORE_DIM, STORE_DIM);
//                }
            }
        }
    }

    if (LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS) >= 1) {
        numAngularL = lefthrr_2(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS) - 1,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                coefAngularL, angularL, smem_uint8);

        for (uint32_t i = 0; i < numAngularL; i++) {
            for (uint32_t j = 0; j < numAngularR; j++) {
//                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybbx = *Ybbx - LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS) * coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i] - STORE_INIT, angularR[j] - STORE_INIT, STORE_DIM, STORE_DIM);
//                }
            }
        }
    }

    if (LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS) >= 1) {
        numAngularL = lefthrr_2(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS) - 1,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                coefAngularL, angularL, smem_uint8);

        for (uint32_t i = 0; i < numAngularL; i++) {
            for (uint32_t j = 0; j < numAngularR; j++) {
//                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybby = *Ybby - LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS) * coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i] - STORE_INIT, angularR[j] - STORE_INIT, STORE_DIM, STORE_DIM);
//                }
            }
        }
    }

    if (LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS) >= 1) {
        numAngularL = lefthrr_2(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS) - 1,
                coefAngularL, angularL, smem_uint8);

        for (uint32_t i = 0; i<numAngularL; i++) {
            for (uint32_t j = 0; j<numAngularR; j++) {
//                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Ybbz = *Ybbz - LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS) * coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i] - STORE_INIT, angularR[j] - STORE_INIT, STORE_DIM, STORE_DIM);
//                }
            }
        }
    }

    // KET PART =====================================
    // Part C - x
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, III, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, JJJ, 3, DEV_SIM_UINT32_NBASIS),
            coefAngularL, angularL, smem_uint8);

    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, KKK, 3, DEV_SIM_UINT32_NBASIS) + 1,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, KKK, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, KKK, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, LLL, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, LLL, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, LLL, 3, DEV_SIM_UINT32_NBASIS),
            coefAngularR, angularR, smem_uint8);

    for (uint32_t i = 0; i < numAngularL; i++) {
        for (uint32_t j = 0; j < numAngularR; j++) {
//            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
            *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j]
                * LOCSTORE(storeCC, angularL[i] - STORE_INIT_J_CC, angularR[j] - STORE_INIT_I_CC, STORE_DIM_J_CC, STORE_DIM_I_CC);
//            }
        }
    }

    if (LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, KKK, 3, DEV_SIM_UINT32_NBASIS) >= 1) {
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, KKK, 3, DEV_SIM_UINT32_NBASIS) - 1,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, KKK, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, KKK, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, LLL, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, LLL, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, LLL, 3, DEV_SIM_UINT32_NBASIS),
                coefAngularR, angularR, smem_uint8);

        for (uint32_t i = 0; i < numAngularL; i++) {
            for (uint32_t j = 0; j < numAngularR; j++) {
//                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccx = *Yccx - LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, KKK, 3, DEV_SIM_UINT32_NBASIS) * coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i] - STORE_INIT, angularR[j] - STORE_INIT, STORE_DIM, STORE_DIM);
//                }
            }
        }
    }

    // Part C - y
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, KKK, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, KKK, 3, DEV_SIM_UINT32_NBASIS) + 1,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, KKK, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, LLL, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, LLL, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, LLL, 3, DEV_SIM_UINT32_NBASIS),
            coefAngularR, angularR, smem_uint8);

    for (uint32_t i = 0; i < numAngularL; i++) {
        for (uint32_t j = 0; j < numAngularR; j++) {
//            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
            *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j]
                * LOCSTORE(storeCC, angularL[i] - STORE_INIT_J_CC, angularR[j] - STORE_INIT_I_CC, STORE_DIM_J_CC, STORE_DIM_I_CC);
//            }
        }
    }

    if (LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, KKK, 3, DEV_SIM_UINT32_NBASIS) >= 1) {
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, KKK, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, KKK, 3, DEV_SIM_UINT32_NBASIS) - 1,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, KKK, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, LLL, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, LLL, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, LLL, 3, DEV_SIM_UINT32_NBASIS),
                coefAngularR, angularR, smem_uint8);

        for (uint32_t i = 0; i < numAngularL; i++) {
            for (uint32_t j = 0; j < numAngularR; j++) {
//                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccy = *Yccy - LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, KKK, 3, DEV_SIM_UINT32_NBASIS) * coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i] - STORE_INIT, angularR[j] - STORE_INIT, STORE_DIM, STORE_DIM);
//                }
            }
        }
    }

    // Part C - z
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, KKK, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, KKK, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, KKK, 3, DEV_SIM_UINT32_NBASIS) + 1,
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, LLL, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, LLL, 3, DEV_SIM_UINT32_NBASIS),
            LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, LLL, 3, DEV_SIM_UINT32_NBASIS),
            coefAngularR, angularR, smem_uint8);

    for (uint32_t i = 0; i < numAngularL; i++) {
        for (uint32_t j = 0; j < numAngularR; j++) {
//            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
            *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j]
                * LOCSTORE(storeCC, angularL[i] - STORE_INIT_J_CC, angularR[j] - STORE_INIT_I_CC, STORE_DIM_J_CC, STORE_DIM_I_CC);
//            }
        }
    }

    if (LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, KKK, 3, DEV_SIM_UINT32_NBASIS) >= 1) {
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, KKK, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, KKK, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, KKK, 3, DEV_SIM_UINT32_NBASIS) - 1,
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 0, LLL, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 1, LLL, 3, DEV_SIM_UINT32_NBASIS),
                LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, LLL, 3, DEV_SIM_UINT32_NBASIS),
                coefAngularR, angularR, smem_uint8);

        for (uint32_t i = 0; i < numAngularL; i++) {
            for (uint32_t j = 0; j < numAngularR; j++) {
//                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                *Yccz = *Yccz - LOC2(DEV_SIM_UINT8_PTR_KLMN, 2, KKK, 3, DEV_SIM_UINT32_NBASIS) * coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i] - STORE_INIT, angularR[j] - STORE_INIT, STORE_DIM, STORE_DIM);
//                }
            }
        }
    }

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


/*
   iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
   performance algrithem for electron intergral evaluation. See description below for details
   */
#if defined(OSHELL)
  #if defined(int_spdf4)
__device__ static inline void iclass_oshell_grad_ffff
  #endif
#else
  #if defined(int_spdf4)
__device__ static inline void iclass_grad_ffff
  #endif
#endif
(uint32_t I, uint32_t J, uint32_t K, uint32_t L,
        uint32_t II, uint32_t JJ, uint32_t KK, uint32_t LL,
        QUICKDouble DNMax,
        QUICKDouble * const store, QUICKDouble * const store2,
        QUICKDouble * const storeAA, QUICKDouble * const storeBB, QUICKDouble * const storeCC,
        uint32_t * const smem_uint32,
        QUICKDouble * const smem_dbl, uint32_t ** const smem_uint32_ptr,
        QUICKDouble ** const smem_dbl_ptr, unsigned char ** const smem_char_ptr,
        uint8_t * const smem_uint8, QUICKAtomicType** const smem_grad_ptr)
{
    /*
       kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
       and be careful with the index difference between Fortran and C++,
       Fortran starts array index with 1 and C++ starts 0.


       RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
       which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
       And we don't need the coordinates now, so we will not retrieve the data now.
    */
    const QUICKDouble RAx = LOC2(DEV_SIM_DBL_PTR_XYZ, 0, DEV_SIM_UINT32_PTR_KATOM[II], 3, DEV_SIM_UINT32_NATOM);
    const QUICKDouble RAy = LOC2(DEV_SIM_DBL_PTR_XYZ, 1, DEV_SIM_UINT32_PTR_KATOM[II], 3, DEV_SIM_UINT32_NATOM);
    const QUICKDouble RAz = LOC2(DEV_SIM_DBL_PTR_XYZ, 2, DEV_SIM_UINT32_PTR_KATOM[II], 3, DEV_SIM_UINT32_NATOM);

    const QUICKDouble RCx = LOC2(DEV_SIM_DBL_PTR_XYZ, 0, DEV_SIM_UINT32_PTR_KATOM[KK], 3, DEV_SIM_UINT32_NATOM);
    const QUICKDouble RCy = LOC2(DEV_SIM_DBL_PTR_XYZ, 1, DEV_SIM_UINT32_PTR_KATOM[KK], 3, DEV_SIM_UINT32_NATOM);
    const QUICKDouble RCz = LOC2(DEV_SIM_DBL_PTR_XYZ, 2, DEV_SIM_UINT32_PTR_KATOM[KK], 3, DEV_SIM_UINT32_NATOM);

    /*
       kPrimI, J, K and L indicates the primtive gaussian function number
       kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
       We retrieve from global memory and save them to register to avoid multiple retrieve.
    */
    const uint32_t kPrimI = DEV_SIM_UINT32_PTR_KPRIM[II];
    const uint32_t kPrimJ = DEV_SIM_UINT32_PTR_KPRIM[JJ];
    const uint32_t kPrimK = DEV_SIM_UINT32_PTR_KPRIM[KK];
    const uint32_t kPrimL = DEV_SIM_UINT32_PTR_KPRIM[LL];

    const uint32_t kStartI = DEV_SIM_UINT32_PTR_KSTART[II];
    const uint32_t kStartJ = DEV_SIM_UINT32_PTR_KSTART[JJ];
    const uint32_t kStartK = DEV_SIM_UINT32_PTR_KSTART[KK];
    const uint32_t kStartL = DEV_SIM_UINT32_PTR_KSTART[LL];

    /*
       store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
       of GPU limitation, we can not do that now.

       See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
    */

    for (uint32_t i = 4; i < 84; i++) {
        for (uint32_t j = 4; j < 84; j++) {
            LOCSTORE(store, j - STORE_INIT, i - STORE_INIT, STORE_DIM, STORE_DIM) = 0.0;
        }
    }

    for (uint32_t i = 4; i < 84; i++) {
        for (uint32_t j = 10; j < 120; j++) {
            LOCSTORE(storeAA, j - STORE_INIT_J_AA, i - STORE_INIT_I_AA, STORE_DIM_J_AA, STORE_DIM_I_AA) = 0.0;
        }
    }

    for (uint32_t i = 4; i < 84; i++) {
        for (uint32_t j = 10; j < 120; j++) {
            LOCSTORE(storeBB, j - STORE_INIT_J_AA, i - STORE_INIT_I_AA, STORE_DIM_J_AA, STORE_DIM_I_AA) = 0.0;
        }
    }

    for (uint32_t i = 10; i < 120; i++) {
        for (uint32_t j = 4; j < 84; j++) {
            LOCSTORE(storeCC, j - STORE_INIT_J_CC, i - STORE_INIT_I_CC, STORE_DIM_J_CC, STORE_DIM_I_CC) = 0.0;
        }
    }

    for (uint32_t i = 4; i < 120; i++) {
        for (uint32_t j = 4; j < 120; j++) {
//          if (i < STOREDIM && j < STOREDIM ) {
            LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0.0;
//          }
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
        const uint32_t ii_start = DEV_SIM_UINT32_PTR_PRIM_START[II];
        const uint32_t jj_start = DEV_SIM_UINT32_PTR_PRIM_START[JJ];

        QUICKDouble AA = LOC2(DEV_SIM_DBL_PTR_GCEXPO, III, DEV_SIM_UINT32_PTR_KSUMTYPE[II], MAXPRIM, DEV_SIM_UINT32_NBASIS);
        QUICKDouble BB = LOC2(DEV_SIM_DBL_PTR_GCEXPO, JJJ, DEV_SIM_UINT32_PTR_KSUMTYPE[JJ], MAXPRIM, DEV_SIM_UINT32_NBASIS);

        QUICKDouble AB = LOC2(DEV_SIM_DBL_PTR_EXPOSUM, ii_start + III, jj_start + JJJ,
                DEV_SIM_UINT32_PRIM_TOTAL, DEV_SIM_UINT32_PRIM_TOTAL);
        QUICKDouble Px = LOC2(DEV_SIM_DBL_PTR_WEIGHTEDCENTERX, ii_start + III, jj_start + JJJ,
                DEV_SIM_UINT32_PRIM_TOTAL, DEV_SIM_UINT32_PRIM_TOTAL);
        QUICKDouble Py = LOC2(DEV_SIM_DBL_PTR_WEIGHTEDCENTERY, ii_start + III, jj_start + JJJ,
                DEV_SIM_UINT32_PRIM_TOTAL, DEV_SIM_UINT32_PRIM_TOTAL);
        QUICKDouble Pz = LOC2(DEV_SIM_DBL_PTR_WEIGHTEDCENTERZ, ii_start + III, jj_start + JJJ,
                DEV_SIM_UINT32_PRIM_TOTAL, DEV_SIM_UINT32_PRIM_TOTAL);

        /*
           X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
           cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
        */
        QUICKDouble cutoffPrim = DNMax
            * LOC2(DEV_SIM_DBL_PTR_CUTPRIM, kStartI + III, kStartJ + JJJ, DEV_SIM_UINT32_JBASIS, DEV_SIM_UINT32_JBASIS);
        QUICKDouble X1 = LOC4(DEV_SIM_DBL_PTR_XCOEFF, kStartI + III, kStartJ + JJJ,
                I - DEV_SIM_UINT32_PTR_QSTART[II], J - DEV_SIM_UINT32_PTR_QSTART[JJ],
                DEV_SIM_UINT32_JBASIS, DEV_SIM_UINT32_JBASIS, 2, 2);

        for (uint32_t j = 0; j < kPrimK * kPrimL; j++) {
            const uint32_t LLL = j / kPrimK;
            const uint32_t KKK = j - kPrimK * LLL;

            if (cutoffPrim * LOC2(DEV_SIM_DBL_PTR_CUTPRIM, kStartK + KKK, kStartL + LLL, DEV_SIM_UINT32_JBASIS, DEV_SIM_UINT32_JBASIS)
                    > DEV_SIM_DBL_PRIMLIMIT) {
                QUICKDouble CC = LOC2(DEV_SIM_DBL_PTR_GCEXPO, KKK, DEV_SIM_UINT32_PTR_KSUMTYPE[KK], MAXPRIM, DEV_SIM_UINT32_NBASIS);
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

                const uint32_t kk_start = DEV_SIM_UINT32_PTR_PRIM_START[KK];
                const uint32_t ll_start = DEV_SIM_UINT32_PTR_PRIM_START[LL];

                const QUICKDouble CD = LOC2(DEV_SIM_DBL_PTR_EXPOSUM, kk_start + KKK, ll_start + LLL,
                        DEV_SIM_UINT32_PRIM_TOTAL, DEV_SIM_UINT32_PRIM_TOTAL);

                const QUICKDouble ABCD = 1.0 / (AB + CD);

                /*
                   X2 is the multiplication of four indices normalized coeffecient
                */
                const QUICKDouble X2 = sqrt(ABCD) * X1
                    * LOC4(DEV_SIM_DBL_PTR_XCOEFF, kStartK + KKK, kStartL + LLL,
                            K - DEV_SIM_UINT32_PTR_QSTART[KK], L - DEV_SIM_UINT32_PTR_QSTART[LL],
                            DEV_SIM_UINT32_JBASIS, DEV_SIM_UINT32_JBASIS, 2, 2);

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

                const QUICKDouble Qx = LOC2(DEV_SIM_DBL_PTR_WEIGHTEDCENTERX, kk_start + KKK, ll_start + LLL,
                        DEV_SIM_UINT32_PRIM_TOTAL, DEV_SIM_UINT32_PRIM_TOTAL);
                const QUICKDouble Qy = LOC2(DEV_SIM_DBL_PTR_WEIGHTEDCENTERY, kk_start + KKK, ll_start + LLL,
                        DEV_SIM_UINT32_PRIM_TOTAL, DEV_SIM_UINT32_PRIM_TOTAL);
                const QUICKDouble Qz = LOC2(DEV_SIM_DBL_PTR_WEIGHTEDCENTERZ, kk_start + KKK, ll_start + LLL,
                        DEV_SIM_UINT32_PRIM_TOTAL, DEV_SIM_UINT32_PRIM_TOTAL);

                double YVerticalTemp[PRIM_INT_ERI_GRAD_FFFF_LEN];
                FmT(I + J + K + L + 2, AB * CD * ABCD
                        * (SQR(Px - Qx) + SQR(Py - Qy) + SQR(Pz - Qz)), YVerticalTemp);

                for (uint32_t i = 0; i <= I + J + K + L + 2; i++) {
                    YVerticalTemp[i] *= X2;
                }

#if defined(int_spdf4)
                ERint_grad_vrr_ffff_1(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_2(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_3(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_4(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_5(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_6(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_7(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_8(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_9(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_10(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_11(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_12(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_13(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_14(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_15(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_16(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_17(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_18(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_19(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_20(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_21(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_22(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_23(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_24(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_25(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_26(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_27(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_28(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_29(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_30(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_31(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vrr_ffff_32(I, J, K, L,
                        Px - RAx, Py - RAy, Pz - RAz,
                        (Px * AB + Qx * CD) * ABCD - Px, (Py * AB + Qy * CD) * ABCD - Py, (Pz * AB + Qz * CD) * ABCD - Pz,
                        Qx - RCx, Qy - RCy, Qz - RCz,
                        (Px * AB + Qx * CD) * ABCD - Qx, (Py * AB + Qy * CD) * ABCD - Qy, (Pz * AB + Qz * CD) * ABCD - Qz,
                        0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
#endif

                for (uint32_t i = 4; i < 84; i++) {
                    for (uint32_t j = 4; j < 84; j++) {
//                        if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(store, j - STORE_INIT, i - STORE_INIT, STORE_DIM, STORE_DIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM);
//                        }
                    }
                }

                for (uint32_t i = 4; i < 84; i++) {
                    for (uint32_t j = 10; j < 120; j++) {
//                        if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(storeAA, j - STORE_INIT_J_AA, i - STORE_INIT_I_AA, STORE_DIM_J_AA, STORE_DIM_I_AA)
                            += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * AA * 2;
//                        }
                    }
                }

                for (uint32_t i = 4; i < 84; i++) {
                    for (uint32_t j = 10; j < 120; j++) {
//                        if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(storeBB, j - STORE_INIT_J_AA, i - STORE_INIT_I_AA, STORE_DIM_J_AA, STORE_DIM_I_AA)
                            += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * BB * 2;
//                        }
                    }
                }

                for (uint32_t i = 10; i < 120; i++) {
                    for (uint32_t j = 4; j < 84; j++) {
//                        if (i < STOREDIM && j < STOREDIM) {
                        LOCSTORE(storeCC, j - STORE_INIT_J_CC, i - STORE_INIT_I_CC, STORE_DIM_J_CC, STORE_DIM_I_CC)
                            += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * CC * 2;
//                        }
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

    uint32_t AStart = DEV_SIM_UINT32_PTR_KATOM[II] * 3;
    uint32_t BStart = DEV_SIM_UINT32_PTR_KATOM[JJ] * 3;
    uint32_t CStart = DEV_SIM_UINT32_PTR_KATOM[KK] * 3;
    uint32_t DStart = DEV_SIM_UINT32_PTR_KATOM[LL] * 3;

    QUICKDouble RBx, RBy, RBz;
    QUICKDouble RDx, RDy, RDz;

    RBx = LOC2(DEV_SIM_DBL_PTR_XYZ, 0, DEV_SIM_UINT32_PTR_KATOM[JJ], 3, DEV_SIM_UINT32_NATOM);
    RBy = LOC2(DEV_SIM_DBL_PTR_XYZ, 1, DEV_SIM_UINT32_PTR_KATOM[JJ], 3, DEV_SIM_UINT32_NATOM);
    RBz = LOC2(DEV_SIM_DBL_PTR_XYZ, 2, DEV_SIM_UINT32_PTR_KATOM[JJ], 3, DEV_SIM_UINT32_NATOM);

    RDx = LOC2(DEV_SIM_DBL_PTR_XYZ, 0, DEV_SIM_UINT32_PTR_KATOM[LL], 3, DEV_SIM_UINT32_NATOM);
    RDy = LOC2(DEV_SIM_DBL_PTR_XYZ, 1, DEV_SIM_UINT32_PTR_KATOM[LL], 3, DEV_SIM_UINT32_NATOM);
    RDz = LOC2(DEV_SIM_DBL_PTR_XYZ, 2, DEV_SIM_UINT32_PTR_KATOM[LL], 3, DEV_SIM_UINT32_NATOM);

    uint32_t III1 = LOC2(DEV_SIM_UINT32_PTR_QSBASIS, II, I, DEV_SIM_UINT32_NSHELL, 4);
    uint32_t III2 = LOC2(DEV_SIM_UINT32_PTR_QFBASIS, II, I, DEV_SIM_UINT32_NSHELL, 4);
    uint32_t JJJ1 = LOC2(DEV_SIM_UINT32_PTR_QSBASIS, JJ, J, DEV_SIM_UINT32_NSHELL, 4);
    uint32_t JJJ2 = LOC2(DEV_SIM_UINT32_PTR_QFBASIS, JJ, J, DEV_SIM_UINT32_NSHELL, 4);
    uint32_t KKK1 = LOC2(DEV_SIM_UINT32_PTR_QSBASIS, KK, K, DEV_SIM_UINT32_NSHELL, 4);
    uint32_t KKK2 = LOC2(DEV_SIM_UINT32_PTR_QFBASIS, KK, K, DEV_SIM_UINT32_NSHELL, 4);
    uint32_t LLL1 = LOC2(DEV_SIM_UINT32_PTR_QSBASIS, LL, L, DEV_SIM_UINT32_NSHELL, 4);
    uint32_t LLL2 = LOC2(DEV_SIM_UINT32_PTR_QFBASIS, LL, L, DEV_SIM_UINT32_NSHELL, 4);

    uint32_t nbasis = DEV_SIM_UINT32_NBASIS;

    for (uint32_t III = III1; III <= III2; III++) {
        for (uint32_t JJJ = MAX(III, JJJ1); JJJ <= JJJ2; JJJ++) {
            for (uint32_t KKK = MAX(III, KKK1); KKK <= KKK2; KKK++) {
                for (uint32_t LLL = MAX(KKK, LLL1); LLL <= LLL2; LLL++) {
                    if (III < KKK
                            || (III == JJJ && III == LLL)
                            || (III == JJJ && III < LLL)
                            || (JJJ == LLL && III < JJJ)
                            || (III == KKK && III < JJJ && JJJ < LLL)) {
                        QUICKDouble Yaax, Yaay, Yaaz;
                        QUICKDouble Ybbx, Ybby, Ybbz;
                        QUICKDouble Yccx, Yccy, Yccz;

                        hrrwholegrad2_ffff(&Yaax, &Yaay, &Yaaz, &Ybbx, &Ybby, &Ybbz, &Yccx, &Yccy, &Yccz,
                             III, JJJ, KKK, LLL,
                             store, storeAA, storeBB, storeCC, RAx, RAy, RAz, RBx, RBy, RBz,
                             RCx, RCy, RCz, RDx, RDy, RDz,
                             smem_uint8_ptr, smem_uint32, smem_uint32_ptr, smem_dbl_ptr, smem_char_ptr, smem_uint8);

                        QUICKDouble constant = 0.0;
#if defined(OSHELL)
                        QUICKDouble DENSELJ = (QUICKDouble) (LOC2(DEV_SIM_DBL_PTR_DENSE, LLL, JJJ, nbasis, nbasis)
                                + LOC2(DEV_SIM_DBL_PTR_DENSEb, LLL, JJJ, nbasis, nbasis));
                        QUICKDouble DENSELI = (QUICKDouble) (LOC2(DEV_SIM_DBL_PTR_DENSE, LLL, III, nbasis, nbasis)
                                + LOC2(DEV_SIM_DBL_PTR_DENSEb, LLL, III, nbasis, nbasis));
                        QUICKDouble DENSELK = (QUICKDouble) (LOC2(DEV_SIM_DBL_PTR_DENSE, LLL, KKK, nbasis, nbasis)
                                + LOC2(DEV_SIM_DBL_PTR_DENSEb, LLL, KKK, nbasis, nbasis));
                        QUICKDouble DENSEJI = (QUICKDouble) (LOC2(DEV_SIM_DBL_PTR_DENSE, JJJ, III, nbasis, nbasis)
                                + LOC2(DEV_SIM_DBL_PTR_DENSEb, JJJ, III, nbasis, nbasis));

                        QUICKDouble DENSEKIA = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, KKK, III, nbasis, nbasis);
                        QUICKDouble DENSEKJA = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, KKK, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELJA = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, LLL, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELIA = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, LLL, III, nbasis, nbasis);
                        QUICKDouble DENSEJIA = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, JJJ, III, nbasis, nbasis);

                        QUICKDouble DENSEKIB = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSEb, KKK, III, nbasis, nbasis);
                        QUICKDouble DENSEKJB = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSEb, KKK, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELJB = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSEb, LLL, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELIB = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSEb, LLL, III, nbasis, nbasis);
                        QUICKDouble DENSEJIB = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSEb, JJJ, III, nbasis, nbasis);

#else
                        QUICKDouble DENSEKI = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, KKK, III, nbasis, nbasis);
                        QUICKDouble DENSEKJ = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, KKK, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELJ = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, LLL, JJJ, nbasis, nbasis);
                        QUICKDouble DENSELI = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, LLL, III, nbasis, nbasis);
                        QUICKDouble DENSELK = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, LLL, KKK, nbasis, nbasis);
                        QUICKDouble DENSEJI = (QUICKDouble) LOC2(DEV_SIM_DBL_PTR_DENSE, JJJ, III, nbasis, nbasis);
#endif

                        if (II < JJ && II < KK && KK < LL
                                || (III < KKK && III < JJJ && KKK < LLL)) {
                            //constant = ( 4.0 * DENSEJI * DENSELK - DENSEKI * DENSELJ - DENSELI * DENSEKJ);
#if defined(OSHELL)
                            constant = ( 4.0 * DENSEJI * DENSELK - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSEKIA * DENSELJA - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSELIA * DENSEKJA
                                    - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSEKIB * DENSELJB - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSELIB * DENSEKJB);
#else
                            constant = ( 4.0 * DENSEJI * DENSELK - DEV_SIM_DBL_HYB_COEFF * DENSEKI * DENSELJ - DEV_SIM_DBL_HYB_COEFF * DENSELI * DENSEKJ);
#endif
                        } else {
                            if (III < KKK) {
                                if (III == JJJ && KKK == LLL) {
                                    //constant = (DENSEJI * DENSELK - 0.5 * DENSEKI * DENSEKI);
#if defined(OSHELL)
                                    constant = (DENSEJI * DENSELK - DEV_SIM_DBL_HYB_COEFF * DENSEKIA * DENSEKIA - DEV_SIM_DBL_HYB_COEFF * DENSEKIB * DENSEKIB);
#else
                                    constant = (DENSEJI * DENSELK - 0.5 * DEV_SIM_DBL_HYB_COEFF * DENSEKI * DENSEKI);
#endif

                                } else if (JJJ == KKK && JJJ == LLL) {
                                    //constant = DENSELJ * DENSEJI;
#if defined(OSHELL)
                                    constant = 2.0 * DENSELJ * DENSEJI - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSELJA * DENSEJIA - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSELJB * DENSEJIB;
#else
                                    constant = 2.0 * DENSELJ * DENSEJI - DEV_SIM_DBL_HYB_COEFF * DENSELJ * DENSEJI;
#endif
                                } else if (KKK == LLL && III < JJJ && JJJ != KKK) {
                                    //constant = (2.0* DENSEJI * DENSELK - DENSEKI * DENSEKJ);
#if defined(OSHELL)
                                    constant = (2.0* DENSEJI * DENSELK - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSEKIA * DENSEKJA - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSEKIB * DENSEKJB);
#else
                                    constant = (2.0* DENSEJI * DENSELK - DEV_SIM_DBL_HYB_COEFF * DENSEKI * DENSEKJ);
#endif
                                } else if (III == JJJ && KKK < LLL) {
                                    //constant = (2.0* DENSELK * DENSEJI - DENSEKI * DENSELI);
#if defined(OSHELL)
                                    constant = (2.0* DENSELK * DENSEJI - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSEKIA * DENSELIA - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSEKIB * DENSELIB);
#else
                                    constant = (2.0* DENSELK * DENSEJI - DEV_SIM_DBL_HYB_COEFF * DENSEKI * DENSELI);
#endif
                                }
                            } else {
                                if (JJJ <= LLL) {
                                    if (III == JJJ && III == KKK && III == LLL) {
                                        // Do nothing
                                    } else if (III == JJJ && III == KKK && III < LLL) {
                                        //constant = DENSELI * DENSEJI;
#if defined(OSHELL)
                                        constant = 2.0 * DENSELI * DENSEJI - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSELIA * DENSEJIA - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSELIB * DENSEJIB;
#else
                                        constant = 2.0 * DENSELI * DENSEJI - DEV_SIM_DBL_HYB_COEFF * DENSELI * DENSEJI;
#endif
                                    } else if (III == KKK && JJJ == LLL && III < JJJ) {
                                        //constant = (1.5 * DENSEJI * DENSEJI - 0.5 * DENSELJ * DENSEKI);
#if defined(OSHELL)
                                        constant = (2.0 * DENSEJI * DENSEJI - DEV_SIM_DBL_HYB_COEFF * DENSEJIA * DENSEJIA - DEV_SIM_DBL_HYB_COEFF * DENSELJA * DENSEKIA
                                                - DEV_SIM_DBL_HYB_COEFF * DENSEJIB * DENSEJIB - DEV_SIM_DBL_HYB_COEFF * DENSELJB * DENSEKIB);
#else
                                        constant = (2.0 * DENSEJI * DENSEJI - 0.5 * DEV_SIM_DBL_HYB_COEFF * DENSEJI * DENSEJI - 0.5 * DEV_SIM_DBL_HYB_COEFF * DENSELJ * DENSEKI);
#endif

                                    } else if (III == KKK && III < JJJ && JJJ < LLL) {
                                        //constant = (3.0 * DENSEJI * DENSELI - DENSELJ * DENSEKI);
#if defined(OSHELL)
                                        constant = (4.0 * DENSEJI * DENSELI - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSEJIA * DENSELIA - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSELJA * DENSEKIA
                                                - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSEJIB * DENSELIB - 2.0 * DEV_SIM_DBL_HYB_COEFF * DENSELJB * DENSEKIB);
#else
                                        constant = (4.0 * DENSEJI * DENSELI - DEV_SIM_DBL_HYB_COEFF * DENSEJI * DENSELI - DEV_SIM_DBL_HYB_COEFF * DENSELJ * DENSEKI);
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

//    if (abs(AGradx) > 0.0 || abs(AGrady) > 0.0 || abs(AGradz) > 0.0
//            || abs(BGradx) > 0.0 || abs(BGrady) > 0.0 || abs(BGradz) > 0.0
//            || abs(CGradx) > 0.0 || abs(CGrady) > 0.0 || abs(CGradz) > 0.0) {
//       printf("%i %i %i %i %i %i %i %i %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e \n",
//               II, JJ, KK, LL, I, J, K, L, AGradx, AGrady, AGradz,
//               BGradx, BGrady, BGradz, CGradx, CGrady, CGradz);
//   }

#ifdef DEBUG
//    printf("FILE: %s, LINE: %d, FUNCTION: %s, DEV_SIM_DBL_HYB_COEFF \n", __FILE__, __LINE__, __func__);
#endif

    GPUATOMICADD(&DEV_SIM_PTR_GRAD[AStart], AGradx, GRADSCALE);
    GPUATOMICADD(&DEV_SIM_PTR_GRAD[AStart + 1], AGrady, GRADSCALE);
    GPUATOMICADD(&DEV_SIM_PTR_GRAD[AStart + 2], AGradz, GRADSCALE);

    GPUATOMICADD(&DEV_SIM_PTR_GRAD[BStart], BGradx, GRADSCALE);
    GPUATOMICADD(&DEV_SIM_PTR_GRAD[BStart + 1], BGrady, GRADSCALE);
    GPUATOMICADD(&DEV_SIM_PTR_GRAD[BStart + 2], BGradz, GRADSCALE);

    GPUATOMICADD(&DEV_SIM_PTR_GRAD[CStart], CGradx, GRADSCALE);
    GPUATOMICADD(&DEV_SIM_PTR_GRAD[CStart + 1], CGrady, GRADSCALE);
    GPUATOMICADD(&DEV_SIM_PTR_GRAD[CStart + 2], CGradz, GRADSCALE);

    GPUATOMICADD(&DEV_SIM_PTR_GRAD[DStart], -AGradx - BGradx - CGradx, GRADSCALE);
    GPUATOMICADD(&DEV_SIM_PTR_GRAD[DStart + 1], -AGrady - BGrady - CGrady, GRADSCALE);
    GPUATOMICADD(&DEV_SIM_PTR_GRAD[DStart + 2], -AGradz - BGradz - CGradz, GRADSCALE);
}


#if defined(OSHELL)
  #if defined(int_spdf4)
__global__ void __launch_bounds__(ERI_GRAD_FFFF_TPB, ERI_GRAD_FFFF_BPSM) getGrad_oshell_kernel_ffff()
  #endif
#else
  #if defined(int_spdf4)
__global__ void __launch_bounds__(ERI_GRAD_FFFF_TPB, ERI_GRAD_FFFF_BPSM) getGrad_kernel_ffff(uint8_t * dev_uint8_data,
        uint32_t * dev_uint32_data, uint32_t ** dev_uint32_ptr_data,
        QUICKDouble * dev_dbl_data, QUICKDouble ** dev_dbl_ptr_data,
        int2 ** dev_int2_ptr_data, unsigned char ** dev_char_ptr_data, 
        QUICKAtomicType ** dev_grad_ptr_data, uint32_t ffStart, uint32_t sqrQshell)
  #endif
#endif
{
    extern __shared__ QUICKDouble smem_buffer[];

    QUICKDouble *smem_dbl = smem_buffer;
    QUICKDouble **smem_dbl_ptr = (QUICKDouble **) &smem_dbl[ERI_GRAD_FFFF_SMEM_DBL_SIZE * ERI_GRAD_FFFF_TPB];
    uint32_t **smem_uint32_ptr = (uint32_t **) &smem_dbl_ptr[ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE * ERI_GRAD_FFFF_TPB];
    int2 **smem_int2_ptr = (int2 **) &smem_uint32_ptr[ERI_GRAD_FFFF_SMEM_UINT32_PTR_SIZE * ERI_GRAD_FFFF_TPB];
    unsigned char **smem_char_ptr = (unsigned char **) &smem_int2_ptr[ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE * ERI_GRAD_FFFF_TPB];
    uint32_t *smem_uint32 = (uint32_t *) &smem_char_ptr[ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE * ERI_GRAD_FFFF_TPB];
    uint8_t *smem_uint8 = (uint8_t *) &smem_uint32[ERI_GRAD_FFFF_SMEM_UINT32_SIZE * ERI_GRAD_FFFF_TPB];
    QUICKAtomicType **smem_grad_ptr = (QUICKAtomicType **) &smem_uint8[ERI_GRAD_FFFF_SMEM_UINT8_SIZE];

    for (int i = threadIdx.x; i < ERI_GRAD_FFFF_TPB * ERI_GRAD_FFFF_SMEM_DBL_SIZE; i += blockDim.x)
        smem_dbl[i] = dev_dbl_data[i];

    for (int i = threadIdx.x; i < ERI_GRAD_FFFF_TPB * ERI_GRAD_FFFF_SMEM_DBL_PTR_SIZE; i += blockDim.x)
        smem_dbl_ptr[i] = dev_dbl_ptr_data[i];

    for (int i = threadIdx.x; i < ERI_GRAD_FFFF_TPB * ERI_GRAD_FFFF_SMEM_UINT8_PTR_SIZE; i += blockDim.x)
        smem_uint8_ptr[i] = dev_uint8_ptr_data[i];

    for (int i = threadIdx.x; i < ERI_GRAD_FFFF_TPB * ERI_GRAD_FFFF_SMEM_UINT32_SIZE; i += blockDim.x)
        smem_uint32[i] = dev_uint32_data[i];

    for (int i = threadIdx.x; i < ERI_GRAD_FFFF_TPB * ERI_GRAD_FFFF_SMEM_UINT32_PTR_SIZE; i += blockDim.x)
        smem_uint32_ptr[i] = dev_uint32_ptr_data[i];

    for (int i = threadIdx.x; i < ERI_GRAD_FFFF_TPB * ERI_GRAD_FFFF_SMEM_INT2_PTR_SIZE; i += blockDim.x)
        smem_int2_ptr[i] = dev_int2_ptr_data[i];

    for (int i = threadIdx.x; i < ERI_GRAD_FFFF_TPB * ERI_GRAD_FFFF_SMEM_CHAR_PTR_SIZE; i += blockDim.x)
        smem_char_ptr[i] = dev_char_ptr_data[i];

    for (int i = threadIdx.x; i < ERI_GRAD_FFFF_SMEM_UINT8_SIZE; i += blockDim.x)
        smem_uint8[i] = dev_uint8_data[i];

    for (int i = threadIdx.x; i < ERI_GRAD_FFFF_TPB * ERI_GRAD_FFFF_SMEM_PTR_SIZE; i += blockDim.x)
        smem_grad_ptr[i] = dev_grad_ptr_data[i];

    __syncthreads();


    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalThreads = blockDim.x * gridDim.x;

    QUICKULL jshell0 = (QUICKULL) ffStart;
    QUICKULL jshell00 = (QUICKULL) ffStart;
    QUICKULL jshell = (QUICKULL) sqrQshell - jshell00;
    QUICKULL jshell2 = (QUICKULL) sqrQshell - jshell0;

    for (QUICKULL i = offset; i < jshell2 * jshell; i += totalThreads) {
        QUICKULL a, b;
        if (jshell2 != 0) {
            a = (QUICKULL) i / jshell2;
            b = (QUICKULL) (i - a * jshell2);
            a = a + jshell00;
            b = b + jshell0;
        } else {
            a = 0;
            b = 0;
        }

#if defined(MPIV_GPU)
        if (DEV_SIM_CHAR_PTR_MPI_BCOMPUTE[a] > 0) {
#endif
            int II = DEV_SIM_INT2_PTR_SORTED_YCUTOFFIJ[a].x;
            int KK = DEV_SIM_INT2_PTR_SORTED_YCUTOFFIJ[b].x;

            uint32_t ii = DEV_SIM_UINT32_PTR_SORTED_Q[II];
            uint32_t kk = DEV_SIM_UINT32_PTR_SORTED_Q[KK];

            if (ii <= kk) {
                int JJ = DEV_SIM_INT2_PTR_SORTED_YCUTOFFIJ[a].y;
                int LL = DEV_SIM_INT2_PTR_SORTED_YCUTOFFIJ[b].y;

                uint32_t iii = DEV_SIM_UINT8_PTR_SORTED_QNUMBER[II];
                uint32_t jjj = DEV_SIM_UINT8_PTR_SORTED_QNUMBER[JJ];
                uint32_t kkk = DEV_SIM_UINT8_PTR_SORTED_QNUMBER[KK];
                uint32_t lll = DEV_SIM_UINT8_PTR_SORTED_QNUMBER[LL];

                uint32_t jj = DEV_SIM_UINT32_PTR_SORTED_Q[JJ];
                uint32_t ll = DEV_SIM_UINT32_PTR_SORTED_Q[LL];

                // In case 4 indices are in the same atom
                if (!((DEV_SIM_UINT32_PTR_KATOM[ii] == DEV_SIM_UINT32_PTR_KATOM[jj])
                            && (DEV_SIM_UINT32_PTR_KATOM[ii] == DEV_SIM_UINT32_PTR_KATOM[kk])
                            && (DEV_SIM_UINT32_PTR_KATOM[ii] == DEV_SIM_UINT32_PTR_KATOM[ll]))) {
                    uint32_t nshell = DEV_SIM_UINT32_NSHELL;

                    QUICKDouble DNMax = MAX(MAX(4.0 * LOC2(DEV_SIM_DBL_PTR_CUTMATRIX, ii, jj, nshell, nshell),
                                4.0 * LOC2(DEV_SIM_DBL_PTR_CUTMATRIX, kk, ll, nshell, nshell)),
                            MAX(MAX(LOC2(DEV_SIM_DBL_PTR_CUTMATRIX, ii, ll, nshell, nshell),
                                    LOC2(DEV_SIM_DBL_PTR_CUTMATRIX, ii, kk, nshell, nshell)),
                                MAX(LOC2(DEV_SIM_DBL_PTR_CUTMATRIX, jj, kk, nshell, nshell),
                                    LOC2(DEV_SIM_DBL_PTR_CUTMATRIX, jj, ll, nshell, nshell))));

                    if ((LOC2(DEV_SIM_DBL_PTR_YCUTOFF, kk, ll, nshell, nshell)
                                * LOC2(DEV_SIM_DBL_PTR_YCUTOFF, ii, jj, nshell, nshell))
                            > DEV_SIM_DBL_GRADCUTOFF
                            && (LOC2(DEV_SIM_DBL_PTR_YCUTOFF, kk, ll, nshell, nshell)
                                * LOC2(DEV_SIM_DBL_PTR_YCUTOFF, ii, jj, nshell, nshell) * DNMax)
                            > DEV_SIM_DBL_GRADCUTOFF) {
#if defined(OSHELL)
  #if defined(int_spdf4)
                        if (iii == 3 && jjj == 3 && kkk == 3 && lll == 3) {
                            iclass_oshell_grad_ffff(iii, jjj, kkk, lll, ii, jj, kk, ll,
                                    DNMax,
                                    DEV_SIM_DBL_PTR_STORE + offset, DEV_SIM_DBL_PTR_STORE2 + offset,
                                    DEV_SIM_DBL_PTR_STOREAA + offset, DEV_SIM_DBL_PTR_STOREBB + offset,
                                    DEV_SIM_DBL_PTR_STORECC + offset,
                                    smem_uint8_ptr, smem_uint32, smem_dbl, smem_uint32_ptr,
                                    smem_dbl_ptr, smem_char_ptr, smem_uint8, smem_grad_ptr);
                        }
  #endif
#else
  #if defined(int_spdf4)
                        if (iii == 3 && jjj == 3 && kkk == 3 && lll == 3) {
                            iclass_grad_ffff(iii, jjj, kkk, lll, ii, jj, kk, ll,
                                    DNMax,
                                    DEV_SIM_DBL_PTR_STORE + offset, DEV_SIM_DBL_PTR_STORE2 + offset,
                                    DEV_SIM_DBL_PTR_STOREAA + offset, DEV_SIM_DBL_PTR_STOREBB + offset,
                                    DEV_SIM_DBL_PTR_STORECC + offset,
                                    smem_uint8_ptr, smem_uint32, smem_dbl, smem_uint32_ptr,
                                    smem_dbl_ptr, smem_char_ptr, smem_uint8, smem_grad_ptr);
                        }
  #endif
#endif
                    }
                }
            }
#if defined(MPIV_GPU)
        }
#endif
    }
}


#endif
