//
//  gpu_get2e_subs_hrr.h
//  new_quick
//
//  Created by Yipu Miao on 3/18/14.
//
//

#if !defined(gpu_lri_subs_hrr_h)
#define gpu_lri_subs_hrr_h

#undef STOREDIM
#define STOREDIM STOREDIM_S


__device__ static inline QUICKDouble quick_pow(QUICKDouble a, uint8_t power)
{
    QUICKDouble ret;

    /*
       notice 0^0 = 1 for this subroutine but is invalid mathmatically
    */
    if (power == 0u) {
        ret = 1.0;
    } else if (power == 1u) {
        ret = a;
    } else if (power == 2u) {
        ret = a * a;
    } else if (power == 3u) {
        ret = a * a * a;
    } else {
        ret = 0.0;
    }

    return ret;
}


__device__ static inline uint8_t lefthrr_lri(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint8_t KLMNAx, uint8_t KLMNAy, uint8_t KLMNAz,
        uint8_t KLMNBx, uint8_t KLMNBy, uint8_t KLMNBz,
        uint8_t IJTYPE, QUICKDouble * const coefAngularL, uint8_t * angularL)
{
    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    if (IJTYPE == 0) {
        return 1;
    } else if (IJTYPE == 1) {
        if (KLMNBx != 0) {
            coefAngularL[1] = RAx - RBx;
        } else if (KLMNBy != 0) {
            coefAngularL[1] = RAy - RBy;
        } else if (KLMNBz != 0) {
            coefAngularL[1] = RAz - RBz;
        }

        angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

        return 2;
    } else if (IJTYPE == 2) {
        if (KLMNBx == 2 || KLMNBy == 2 || KLMNBz == 2) {
            QUICKDouble tmp;

            if (KLMNBx == 2) {
                tmp = RAx - RBx;
                angularL[1] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBy == 2) {
                tmp = RAy - RBy;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBz == 2){
                tmp = RAz - RBz;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            coefAngularL[1] = 2 * tmp;
            coefAngularL[2]= tmp * tmp;

            angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

            return 3;

        } else {
            QUICKDouble tmp, tmp2;

            if (KLMNBx == 1 && KLMNBy == 1) {
                tmp = RAx - RBx;
                tmp2 = RAy - RBy;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

            } else if (KLMNBx == 1 && KLMNBz == 1) {
                tmp = RAx - RBx;
                tmp2 = RAz - RBz;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBy == 1 && KLMNBz == 1) {
                tmp = RAy - RBy;
                tmp2 = RAz - RBz;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            coefAngularL[1] = tmp;
            coefAngularL[2] = tmp2;
            coefAngularL[3] = tmp * tmp2;

            angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

            return 4;
        }
    } else if (IJTYPE == 3) {
        if (KLMNBx == 3 || KLMNBy == 3 || KLMNBz == 3) {
            QUICKDouble tmp;

            if (KLMNBx == 3) {
                tmp = RAx - RBx;
                angularL[1] = LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBy == 3) {
                tmp = RAy - RBy;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBz == 3) {
                tmp = RAz - RBz;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            coefAngularL[1] = 3 * tmp;
            coefAngularL[2] = 3 * tmp * tmp;
            coefAngularL[3] = tmp * tmp * tmp;

            angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

            return 4;
        } else if (KLMNBx == 1 && KLMNBy == 1) {
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

            angularL[1] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);

            angularL[7] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

            return 8;
        } else {
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
                angularL[1] = LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBy == 2) {
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBz == 2) {
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBx == 1) {
                if (KLMNBy == 2) {  //120
                    angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                } else {              //102
                    angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }

            if (KLMNBy == 1) {
                if (KLMNBx == 2) {  // 210
                    angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                } else {              // 012
                    angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }

            if (KLMNBz == 1) {
                if (KLMNBx == 2) {  // 201
                    angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                } else {              // 021
                    angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }

            if (KLMNBx == 1) {
                angularL[4] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBy == 1) {
                angularL[4] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBz == 1) {
                angularL[4] = LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            angularL[5] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

            return 6;
        }
    }

    return 0;
}


__device__ static inline uint8_t lefthrr_lri23(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint8_t KLMNAx, uint8_t KLMNAy, uint8_t KLMNAz,
        uint8_t KLMNBx, uint8_t KLMNBy, uint8_t KLMNBz,
        uint8_t IJTYPE, QUICKDouble * const coefAngularL, uint8_t * angularL)
{
    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    /*
       if this subroutine is called, (ij|kl) for (k+l)>=5 is computed, but (k+l)>=5 entering this subroutine
       here ijtype is the value of l
    */
    if (KLMNBx == 3 || KLMNBy == 3 || KLMNBz == 3) {
        QUICKDouble tmp;

        if (KLMNBx == 3) {
            tmp = RAx - RBx;
            angularL[1] = LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else if (KLMNBy == 3) {
            tmp = RAy - RBy;
            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else if (KLMNBz == 3) {
            tmp = RAz - RBz;
            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        coefAngularL[1] = 3 * tmp;

        return 2;
    } else if (KLMNBx == 1 && KLMNBy == 1) {
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAy - RBy;
        QUICKDouble tmp3 = RAz - RBz;

        coefAngularL[1] = tmp;
        coefAngularL[2] = tmp2;
        coefAngularL[3] = tmp3;

        angularL[1] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);

        return 4;
    } else {
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

        if (KLMNBx == 2) {
            angularL[1] = LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
        }

        if (KLMNBy == 2) {
            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
        }

        if (KLMNBz == 2) {
            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        if (KLMNBx == 1) {
            if (KLMNBy == 2) {  //120
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else {              //102
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }

        if (KLMNBy == 1) {
            if (KLMNBx == 2) {  // 210
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else {              // 012
                angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }

        if (KLMNBz == 1) {
            if (KLMNBx == 2) {  // 201
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            } else {              // 021
                angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }

        return 3;
    //}
    }

    //return 0;
}


__device__ static inline uint8_t lefthrr_lri23_new(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint8_t KLMNAx, uint8_t KLMNAy, uint8_t KLMNAz,
        uint8_t KLMNBx, uint8_t KLMNBy, uint8_t KLMNBz,
        uint8_t IJTYPE, QUICKDouble * const coefAngularL, uint8_t * angularL)
{

    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    /*
       if this subroutine is called, (ij|kl) for (k+l)>=5 is computed, but (k+l)>=5 entering this subroutine
       here ijtype is the value of l
    */
    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    QUICKDouble tmp4 = 1.0;

    QUICKDouble tmpx = RAx - RBx;
    QUICKDouble tmpy = RAy - RBy;
    QUICKDouble tmpz = RAz - RBz;

    if (KLMNBx > 0) {
        tmp4 = tmp4 * quick_pow(tmpx, KLMNBx);
    }

    if (KLMNBy > 0) {
        tmp4 = tmp4 * quick_pow(tmpy, KLMNBy);
    }

    if (KLMNBz > 0) {
        tmp4 = tmp4 * quick_pow(tmpz, KLMNBz);
    }

    uint8_t numAngularL = 1;

    if (KLMNBx >= 2 && tmpx != 0) {
        angularL[numAngularL] = LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBx * (KLMNBx - 1) / 2 * tmp4 / (tmpx * tmpx);
        numAngularL++;
    }

    if (KLMNBy >= 2 && tmpy != 0) {
        angularL[numAngularL] = LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBy * (KLMNBy - 1) / 2 * tmp4 / (tmpy * tmpy);
        numAngularL++;
    }

    if (KLMNBz >= 2 && tmpz != 0) {
        angularL[numAngularL] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBz * (KLMNBz - 1) / 2 * tmp4 / (tmpz * tmpz);
        numAngularL++;
    }

    if (KLMNBx >= 1 && KLMNBy >= 1 && tmpx != 0 && tmpy != 0) {
        angularL[numAngularL] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBx * KLMNBy * tmp4 / (tmpx * tmpy);
        numAngularL++;
    }

    if (KLMNBx >= 1 && KLMNBz >= 1 && tmpx != 0 && tmpz != 0) {
        angularL[numAngularL] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBx * KLMNBz * tmp4 / (tmpx * tmpz);
        numAngularL++;
    }

    if (KLMNBy >= 1 && KLMNBz >= 1 && tmpy != 0 && tmpz != 0) {
        angularL[numAngularL] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        coefAngularL[numAngularL] = (QUICKDouble) KLMNBy * KLMNBz * tmp4 / (tmpy * tmpz);
        numAngularL++;
    }

    /* the last case is IJTYPE = 3 */
    return numAngularL;

    //return 0;
}


__device__ static inline QUICKDouble hrrwhole_lri(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz,
        uint32_t nbasis, QUICKDouble const * const cons, uint8_t const * const KLMN,
        QUICKDouble * const store)
{
    QUICKDouble Y = 0.0;
    uint8_t angularL[12];
    QUICKDouble coefAngularL[12];

    uint8_t numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL);
    for (uint8_t i = 0; i < numAngularL; i++) {
        if (angularL[i] < STOREDIM) {
            Y += coefAngularL[i] * LOCSTORE(store, angularL[i], 0, STOREDIM, STOREDIM);
        }
    }
    Y = Y * cons[III] * cons[JJJ];
    //#endif

    return Y;
}


#undef STOREDIM
#define STOREDIM STOREDIM_L
__device__ static inline QUICKDouble hrrwhole_lri_2(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz,
        uint32_t nbasis, QUICKDouble const * const cons, uint8_t const * const KLMN,
        QUICKDouble * const store)
{
    QUICKDouble Y = 0.0;
    uint8_t angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];

    uint8_t numAngularL = lefthrr_lri(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL);
    uint8_t numAngularR = lefthrr_lri(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN, 0, KKK, 3, nbasis),
            LOC2(KLMN, 1, KKK, 3, nbasis),
            LOC2(KLMN, 2, KKK, 3, nbasis),
            LOC2(KLMN, 0, LLL, 3, nbasis),
            LOC2(KLMN, 1, LLL, 3, nbasis),
            LOC2(KLMN, 2, LLL, 3, nbasis),
            L, coefAngularR, angularR);
    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                Y += coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    Y = Y * cons[III] * cons[JJJ] * cons[KKK] * cons[LLL];
    //#endif

    return Y;
}


__device__ static inline QUICKDouble hrrwhole_lri_2_2(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz,
        uint32_t nbasis, QUICKDouble const * const cons, uint8_t const * const KLMN,
        QUICKDouble * const store)
{
    QUICKDouble Y = 0.0;
    uint8_t angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];

    uint8_t numAngularR = lefthrr_lri(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(KLMN, 0, KKK, 3, nbasis),
            LOC2(KLMN, 1, KKK, 3, nbasis),
            LOC2(KLMN, 2, KKK, 3, nbasis),
            LOC2(KLMN, 0, LLL, 3, nbasis),
            LOC2(KLMN, 1, LLL, 3, nbasis),
            LOC2(KLMN, 2, LLL, 3, nbasis),
            L, coefAngularR, angularR);

    if (J == 2) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularR[j] < STOREDIM) {
                Y += coefAngularR[j] * LOCSTORE(store,
                        LOC3(devTrans,
                            LOC2(KLMN, 0, III, 3, nbasis) + LOC2(KLMN, 0, JJJ, 3, nbasis),
                            LOC2(KLMN, 1, III, 3, nbasis) + LOC2(KLMN, 1, JJJ, 3, nbasis),
                            LOC2(KLMN, 2, III, 3, nbasis) + LOC2(KLMN, 2, JJJ, 3, nbasis),
                            TRANSDIM, TRANSDIM, TRANSDIM),
                        angularR[j], STOREDIM, STOREDIM);
            }
        }

        Y = Y * cons[III] * cons[JJJ] * cons[KKK] * cons[LLL];

        return Y;
    }

    uint8_t numAngularL = lefthrr_lri23(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(KLMN, 0, III, 3, nbasis),
            LOC2(KLMN, 1, III, 3, nbasis),
            LOC2(KLMN, 2, III, 3, nbasis),
            LOC2(KLMN, 0, JJJ, 3, nbasis),
            LOC2(KLMN, 1, JJJ, 3, nbasis),
            LOC2(KLMN, 2, JJJ, 3, nbasis),
            J, coefAngularL, angularL);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                Y += coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(store, angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    Y = Y * cons[III] * cons[JJJ] * cons[KKK] * cons[LLL];
    //#endif

    return Y;
}


#endif
