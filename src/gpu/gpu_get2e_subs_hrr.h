//
//  gpu_get2e_subs_hrr.h
//  new_quick
//
//  Created by Yipu Miao on 3/18/14.
//
//

#if !defined(gpu_get2e_subs_hrr_h)
#define gpu_get2e_subs_hrr_h

#undef STOREDIM
#undef LOCSTORE
#define STOREDIM STOREDIM_T
#define LOCSTORE(A,i1,i2,d1,d2) ((A)[((i2) * (d1) + (i1)) * gridDim.x * blockDim.x])


__device__ static inline uint8_t lefthrr(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint8_t KLMNAx, uint8_t KLMNAy, uint8_t KLMNAz,
        uint8_t KLMNBx, uint8_t KLMNBy, uint8_t KLMNBz,
        uint8_t IJTYPE, QUICKDouble * const coefAngularL, uint8_t * const angularL)
{
    uint8_t numAngularL;

    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    if (IJTYPE == 0u) {
        numAngularL = 1;
    } else if (IJTYPE == 1u) {
        numAngularL = 2;

        if (KLMNBx != 0) {
            coefAngularL[1] = RAx - RBx;
        } else if (KLMNBy != 0) {
            coefAngularL[1] = RAy - RBy;
        } else if (KLMNBz != 0) {
            coefAngularL[1] = RAz - RBz;
        }

        angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (IJTYPE == 2u) {
        if (KLMNBx == 2 || KLMNBy == 2 || KLMNBz == 2) {
            numAngularL = 3;
            QUICKDouble tmp;

            if (KLMNBx == 2) {
                tmp = RAx - RBx;
                angularL[1] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBy == 2) {
                tmp = RAy - RBy;
                angularL[1] =  LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBz == 2) {
                tmp = RAz - RBz;
                angularL[1] =  LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            coefAngularL[1] = 2 * tmp;
            coefAngularL[2]= tmp * tmp;

            angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else {
            numAngularL = 4;
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

            angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }
    } else if (IJTYPE == 3u) {
        if (KLMNBx == 3 || KLMNBy == 3 || KLMNBz == 3) {
            numAngularL = 4;
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

            angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);

            angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
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
                //120
                if (KLMNBy == 2) {
                    angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                //102
                } else {
                    angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }

            if (KLMNBy == 1) {
                // 210
                if (KLMNBx == 2) {
                    angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                // 012
                } else {
                    angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }

            if (KLMNBz == 1) {
                // 201
                if (KLMNBx == 2) {
                    angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                // 021
                } else {
                    angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }

            if (KLMNBx == 1) {
                angularL[4] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBy == 1) {
                angularL[4] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBz == 1) {
                angularL[4] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }
    } else if (IJTYPE == 4u) {
        if (KLMNBx == 4) {
            numAngularL = 5;
            QUICKDouble tmp = RAx - RBx;

            coefAngularL[1] = 4 * tmp;
            coefAngularL[2] = 6 * tmp * tmp;
            coefAngularL[3] = 4 * tmp * tmp * tmp;
            coefAngularL[4] = tmp * tmp * tmp * tmp;

            angularL[1] = LOC3(devTrans, KLMNAx + 3, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else if (KLMNBy == 4) {
            numAngularL = 5;
            QUICKDouble tmp = RAy - RBy;
            coefAngularL[1] = 4 * tmp;
            coefAngularL[2] = 6 * tmp * tmp;
            coefAngularL[3] = 4 * tmp * tmp * tmp;
            coefAngularL[4] = tmp * tmp * tmp * tmp;

            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy + 3, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else if (KLMNBz == 4) {
            numAngularL = 5;

            QUICKDouble tmp = RAz - RBz;
            coefAngularL[1] = 4 * tmp;
            coefAngularL[2] = 6 * tmp * tmp;
            coefAngularL[3] = 4 * tmp * tmp * tmp;
            coefAngularL[4] = tmp * tmp * tmp * tmp;

            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 3, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy + 3, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx + 1, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx+3, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx+2, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx+2, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+3, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx+3, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+3, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy+3, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx+1, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx+2, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx,   KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx+2, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[7] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx+2, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[7] = LOC3(devTrans, KLMNAx,   KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[7] = LOC3(devTrans, KLMNAx, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[7] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[8] = LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[9] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[10] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx,   KLMNAy+2, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx,   KLMNAy+2, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[7] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[8] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[9] = LOC3(devTrans, KLMNAx,   KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[10] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx + 2, KLMNAy, KLMNAz + 1,TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx + 2, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx + 1, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[7] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[8] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[9] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[10] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }

        angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    }

    return numAngularL;
}


/*
   if this subroutine is called, (ij|kl) for (k+l)>=5 is computed, but (k+l)>=5 entering this subroutine
   here ijtype is the value of l
*/
__device__ static inline uint8_t lefthrr23(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint8_t KLMNAx, uint8_t KLMNAy, uint8_t KLMNAz,
        uint8_t KLMNBx, uint8_t KLMNBy, uint8_t KLMNBz,
        QUICKDouble * const coefAngularL, uint8_t * const angularL)
{
    uint8_t numAngularL;

    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    if (KLMNBx == 3 || KLMNBy == 3 || KLMNBz == 3) {
        numAngularL = 2;
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
    } else if (KLMNBx == 1 && KLMNBy == 1) {
        numAngularL = 4;
        QUICKDouble tmp = RAx - RBx;
        QUICKDouble tmp2 = RAy - RBy;
        QUICKDouble tmp3 = RAz - RBz;

        coefAngularL[1] = tmp;
        coefAngularL[2] = tmp2;
        coefAngularL[3] = tmp3;

        angularL[1] = LOC3(devTrans, KLMNAx,   KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy,   KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
        angularL[3] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz,   TRANSDIM, TRANSDIM, TRANSDIM);
    } else {
        numAngularL = 3;
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
            //120
            if (KLMNBy == 2) {
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            //102
            } else {
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }

        if (KLMNBy == 1) {
            // 210
            if (KLMNBx == 2) {
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            // 012
            } else {
                angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }

        if (KLMNBz == 1) {
            // 201
            if (KLMNBx == 2) {
                angularL[2] = LOC3(devTrans, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            // 021
            } else {
                angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
            }
        }
    }

    return numAngularL;
}


__device__ static inline uint8_t lefthrr_sp(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint8_t KLMNAx, uint8_t KLMNAy, uint8_t KLMNAz,
        uint8_t KLMNBx, uint8_t KLMNBy, uint8_t KLMNBz,
        uint8_t IJTYPE, QUICKDouble * const coefAngularL, uint8_t * const angularL)
{
    uint8_t numAngularL;

    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    if (IJTYPE == 0u) {
        numAngularL = 1;
    } else if (IJTYPE == 1u) {
        numAngularL = 2;

        if (KLMNBx != 0) {
            coefAngularL[1] = RAx - RBx;
        } else if (KLMNBy != 0) {
            coefAngularL[1] = RAy - RBy;
        } else if (KLMNBz != 0) {
            coefAngularL[1] = RAz - RBz;
        }

        angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else {
        numAngularL = 0;
    }

    return numAngularL;
}


__device__ static inline QUICKDouble hrrwhole_sp(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    QUICKDouble Y = 0.0;
    QUICKDouble coefAngularL[2], coefAngularR[2];
    uint8_t angularL[2], angularR[2];

    uint8_t numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
            J, coefAngularL, angularL);
    uint8_t numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                Y += coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                            angularL[i], angularR[j], STOREDIM, STOREDIM);

//                printf("I %d J %d K %d L %d III %d JJJ %d KKK %d LLL %d i %d j %d store: %f \n",
//                        I, J, K, L, III, JJJ, KKK, LLL,
//                        angularL[i], angularR[j],
//                        LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
//                            angularL[i], angularR[j], STOREDIM, STOREDIM));
            }
        }
    }

    Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];

    return Y;
}


#undef STOREDIM
#define STOREDIM STOREDIM_S
#undef LOCSTORE
#define LOCSTORE(A,i1,i2,d1,d2) (A)[((i1) + (i2) * (d1)) * gridDim.x * blockDim.x]
__device__ static inline uint8_t lefthrr_spd(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        uint8_t KLMNAx, uint8_t KLMNAy, uint8_t KLMNAz,
        uint8_t KLMNBx, uint8_t KLMNBy, uint8_t KLMNBz,
        uint8_t IJTYPE, QUICKDouble * const coefAngularL, uint8_t * const angularL)
{
    uint8_t numAngularL;

    coefAngularL[0] = 1.0;
    angularL[0] = LOC3(devTrans, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);

    if (IJTYPE == 0u) {
        numAngularL = 1;
    } else if (IJTYPE == 1u) {
        numAngularL = 2;

        if (KLMNBx != 0) {
            coefAngularL[1] = RAx - RBx;
        } else if (KLMNBy != 0) {
            coefAngularL[1] = RAy - RBy;
        } else if (KLMNBz != 0) {
            coefAngularL[1] = RAz - RBz;
        }

        angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
    } else if (IJTYPE == 2u) {
        if (KLMNBx == 2 || KLMNBy == 2 || KLMNBz == 2) {
            numAngularL = 3;
            QUICKDouble tmp;

            if (KLMNBx == 2) {
                tmp = RAx - RBx;
                angularL[1] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if(KLMNBy == 2) {
                tmp = RAy - RBy;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBz == 2) {
                tmp = RAz - RBz;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            coefAngularL[1] = 2 * tmp;
            coefAngularL[2]= tmp * tmp;

            angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        } else {
            numAngularL = 4;
            QUICKDouble tmp, tmp2;

            if (KLMNBx == 1 && KLMNBy == 1) {
                tmp = RAx - RBx;
                tmp2 = RAy - RBy;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

            } else if (KLMNBx == 1 && KLMNBz == 1) {
                tmp = RAx - RBx;
                tmp2 = RAz - RBz;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBy == 1 && KLMNBz == 1) {
                tmp = RAy - RBy;
                tmp2 = RAz - RBz;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            coefAngularL[1] = tmp;
            coefAngularL[2] = tmp2;
            coefAngularL[3] = tmp * tmp2;

            angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }
    } else if (IJTYPE == 3u) {
        if (KLMNBx == 3 || KLMNBy == 3 || KLMNBz == 3) {
            numAngularL = 4;
            QUICKDouble tmp;

            if (KLMNBx == 3) {
                tmp = RAx - RBx;
                angularL[1] = LOC3(devTrans, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBy == 3) {
                tmp = RAy - RBy;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            } else if (KLMNBz == 3) {
                tmp = RAz - RBz;
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            coefAngularL[1] = 3 * tmp;
            coefAngularL[2] = 3 * tmp * tmp;
            coefAngularL[3] = tmp * tmp * tmp;

            angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
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

            angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[2] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[3] = LOC3(devTrans, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[4] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[5] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[6] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);

            angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
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
                angularL[1] = LOC3(devTrans, KLMNAx + 2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBy == 2) {
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy + 2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBz == 2) {
                angularL[1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 2, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBx == 1) {
                //120
                if (KLMNBy == 2) {
                    angularL[2] = LOC3(devTrans, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                //102
                } else {
                    angularL[2] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }

            if (KLMNBy == 1) {
                // 210
                if (KLMNBx == 2) {
                    angularL[2] = LOC3(devTrans, KLMNAx + 1, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                // 012
                } else {
                    angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }

            if (KLMNBz == 1) {
                // 201
                if (KLMNBx == 2) {
                    angularL[2] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
                // 021
                } else {
                    angularL[2] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
                }
            }

            if (KLMNBx == 1) {
                angularL[4] = LOC3(devTrans, KLMNAx + 1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBy == 1) {
                angularL[4] = LOC3(devTrans, KLMNAx, KLMNAy + 1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            if (KLMNBz == 1) {
                angularL[4] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz + 1, TRANSDIM, TRANSDIM, TRANSDIM);
            }

            angularL[numAngularL - 1] = LOC3(devTrans, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
        }
    }

    return numAngularL;
}


__device__ static inline QUICKDouble hrrwhole(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    QUICKDouble Y = 0.0;
    QUICKDouble coefAngularL[8], coefAngularR[8];
    uint8_t angularL[8], angularR[8];

    uint8_t numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
            J, coefAngularL, angularL);
    uint8_t numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                Y += coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                            angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];

    return Y;
}


#undef STOREDIM
#define STOREDIM STOREDIM_L
__device__ static inline QUICKDouble hrrwhole2(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    QUICKDouble Y = 0.0;
    QUICKDouble coefAngularL[12], coefAngularR[12];
    uint8_t angularL[12], angularR[12];

    uint8_t numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
            J, coefAngularL, angularL);
    uint8_t numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
            L, coefAngularR, angularR);

    for (uint8_t i = 0; i < numAngularL; i++) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                Y += coefAngularL[i] * coefAngularR[j]
                    * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                            angularL[i], angularR[j], STOREDIM, STOREDIM);
            }
        }
    }

    Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];

    return Y;
}


__device__ static inline QUICKDouble hrrwhole2_1(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    QUICKDouble Y = 0.0;
    QUICKDouble coefAngularL[12], coefAngularR[12];
    uint8_t angularL[12], angularR[12];

    uint8_t numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
            LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
            J, coefAngularL, angularL);

    if (L == 2) {
        for (uint8_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM ) {
                Y += coefAngularL[i] * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                        angularL[i],
                        LOC3(devTrans,
                            LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                            TRANSDIM, TRANSDIM, TRANSDIM),
                        STOREDIM, STOREDIM);
            }
        }
    } else {
        uint8_t numAngularR = lefthrr23(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    Y += coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                                angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];

    return Y;
}


__device__ static inline QUICKDouble hrrwhole2_2(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    QUICKDouble Y = 0.0;
    QUICKDouble coefAngularL[12], coefAngularR[12];
    uint8_t angularL[12], angularR[12];

    uint8_t numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
            LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
            LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
            L, coefAngularR, angularR);

    if (J == 2) {
        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularR[j] < STOREDIM) {
                Y += coefAngularR[j] * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                        LOC3(devTrans,
                            LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                            TRANSDIM, TRANSDIM, TRANSDIM),
                        angularR[j], STOREDIM, STOREDIM);
            }
        }
    } else {
        uint8_t numAngularL = lefthrr23(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    Y += coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                                angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }
    }

    Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];

    return Y;
}


/*
   When this subroutine is called, (ij|kl) where i+j = 4 and k+l = 6 is computed, but (i+j) >=4 and k+l = 6 entering this subroutine
   therefore, k = 3 and l = 3 is confirmed.
*/
__device__ static inline QUICKDouble hrrwhole2_5(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    QUICKDouble Y;

    if (K + L == 6 && I + J == 4) {
        Y = LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM),
                LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM),
                STOREDIM, STOREDIM) * devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    } else {
        // else case, j can be 2 or 3, and k = 3 and l = 3
        uint8_t angularL[12];
        QUICKDouble coefAngularL[12];
        Y = 0.0;

        uint8_t numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                J, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                Y += coefAngularL[i] * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                        angularL[i],
                        LOC3(devTrans,
                            LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                            TRANSDIM, TRANSDIM, TRANSDIM),
                        STOREDIM, STOREDIM);
            }
        }

        Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    }

    return Y;
}


/*
   When this subroutine is called, (ij|kl) where i+j = 6 and k+l = 4 is computed, but (i+j) ==6 and k+l >= 4 entering this subroutine
   therefore, i = 3 and j = 3 is confirmed.
*/
__device__ static inline QUICKDouble hrrwhole2_6(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    QUICKDouble Y;

    if (K + L == 4 && I + J == 6) {
        Y = LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM),
                LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM),
                STOREDIM, STOREDIM) * devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    } else {
        uint8_t angularR[12];
        QUICKDouble coefAngularR[12];
        Y = 0.0;

        // For hrr, only k+l need hrr, but can be simplified to only consider k+l=5 contibution
        uint8_t numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                L, coefAngularR, angularR);

        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularR[j] < STOREDIM) {
                Y += coefAngularR[j] * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                        LOC3(devTrans,
                            LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                            TRANSDIM, TRANSDIM, TRANSDIM),
                        angularR[j], STOREDIM, STOREDIM);
            }
        }

        Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    }

    return Y;
}


/*
   when this subroutine is called, only (ij|kl) k+l = 5 and i+j = 5 is computed, but (k+l)>=5 and (i+J)>=5 is entering this subroutine
*/
__device__ static inline QUICKDouble hrrwhole2_3(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    QUICKDouble Y;

    if (K + L == 5 && I + J == 5) {
        Y = LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM),
                LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM),
                STOREDIM, STOREDIM) * devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    } else if (K + L == 5 && I + J == 6) {
        uint8_t angularL[12];
        QUICKDouble coefAngularL[12];
        Y = 0.0;

        uint8_t numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                3, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                Y += coefAngularL[i] * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                        angularL[i],
                        LOC3(devTrans,
                            LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                            TRANSDIM, TRANSDIM, TRANSDIM),
                        STOREDIM, STOREDIM);
            }
        }

        Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    } else if (I + J == 5 && K + L == 6) {
        uint8_t angularR[12];
        QUICKDouble coefAngularR[12];
        Y = 0.0;

        uint8_t numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                3, coefAngularR, angularR);

        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularR[j] < STOREDIM) {
                Y += coefAngularR[j] * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                        LOC3(devTrans,
                            LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                            TRANSDIM, TRANSDIM, TRANSDIM),
                        angularR[j], STOREDIM, STOREDIM);
            }
        }

        Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    } else if (I + J == 6 && K + L == 6) {
        uint8_t angularL[12], angularR[12];
        QUICKDouble coefAngularL[12], coefAngularR[12];
        Y = 0.0;

        uint8_t numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                3, coefAngularL, angularL);
        uint8_t numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                3, coefAngularR, angularR);

        for (uint8_t i = 0; i < numAngularL; i++) {
            for (uint8_t j = 0; j < numAngularR; j++) {
                if (angularL[i] < STOREDIM && angularR[j] < STOREDIM) {
                    Y += coefAngularL[i] * coefAngularR[j]
                        * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                                angularL[i], angularR[j], STOREDIM, STOREDIM);
                }
            }
        }

        Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    } else {
        Y = 0.0;
    }

    return Y;
}


/*
   When this subroutine is called, only (ij|kl) k+l=5 and i+j=6 integral is computed, but (k+l)>=5 and i+j=6 is entering this subroutine
   since i+j = 6, i=3 and j= 3
   so if (k+l) = 5, then, the highest integral is used, and no selection hrr.
   if (k+l) = 6, then, k=3 and l=3
*/
__device__ static inline QUICKDouble hrrwhole2_4(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    QUICKDouble Y;

    if (K + L == 5) {
        Y = LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM),
                LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM),
                STOREDIM, STOREDIM) * devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    } else {
        //k=3 and l = 3, for i and j , i = 3 and j = 3
        uint8_t angularR[12];
        QUICKDouble coefAngularR[12];
        Y = 0.0;

        // For hrr, only k+l need hrr, but can be simplified to only consider k+l=5 contibution
        uint8_t numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                3, coefAngularR, angularR);

        for (uint8_t j = 0; j < numAngularR; j++) {
            if (angularR[j] < STOREDIM) {
                Y += coefAngularR[j] * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                        LOC3(devTrans,
                            LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis), 
                            LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                            TRANSDIM, TRANSDIM, TRANSDIM),
                        angularR[j], STOREDIM, STOREDIM);
            }
        }

        Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    }

    return Y;
}


// For this subroutine, the basic idea is the same with hrrwhole2_4, just swap i to k and j to l.
__device__ static inline QUICKDouble hrrwhole2_7(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    QUICKDouble Y;

    if (I + J == 5) {
        Y = LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM),
                LOC3(devTrans,
                    LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                    LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                    TRANSDIM, TRANSDIM, TRANSDIM),
                STOREDIM, STOREDIM) * devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    } else {
        uint8_t angularL[12];
        QUICKDouble coefAngularL[12];
        Y = 0.0;

        uint8_t numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                3, coefAngularL, angularL);

        for (uint8_t i = 0; i < numAngularL; i++) {
            if (angularL[i] < STOREDIM) {
                Y += coefAngularL[i] * LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
                        angularL[i],
                        LOC3(devTrans,
                            LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                            LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                            TRANSDIM, TRANSDIM, TRANSDIM),
                        STOREDIM, STOREDIM);
            }
        }

        Y *= devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
    }

    return Y;
}


// For hrrwhole2_8,9,10, the situation is much simple, i=3, j=3, k=3, l=3
__device__ static inline QUICKDouble hrrwhole2_8(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    return LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
            LOC3(devTrans,
                LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM),
            LOC3(devTrans,
                LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM),
            STOREDIM, STOREDIM) * devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
}


__device__ static inline QUICKDouble hrrwhole2_9(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    return LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
            LOC3(devTrans,
                LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM),
            LOC3(devTrans,
                LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM),
            STOREDIM, STOREDIM) * devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
}


__device__ static inline QUICKDouble hrrwhole2_10(uint8_t I, uint8_t J, uint8_t K, uint8_t L,
        uint32_t III, uint32_t JJJ, uint32_t KKK, uint32_t LLL,
        QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz,
        QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
        QUICKDouble RCx, QUICKDouble RCy, QUICKDouble RCz,
        QUICKDouble RDx, QUICKDouble RDy, QUICKDouble RDz)
{
    return LOCSTORE(&devSim.store[blockIdx.x * blockDim.x + threadIdx.x],
            LOC3(devTrans,
                LOC2(devSim.KLMN, 0, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, JJJ, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, III, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, JJJ, 3, devSim.nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM),
            LOC3(devTrans,
                LOC2(devSim.KLMN, 0, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 0, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 1, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 1, LLL, 3, devSim.nbasis),
                LOC2(devSim.KLMN, 2, KKK, 3, devSim.nbasis) + LOC2(devSim.KLMN, 2, LLL, 3, devSim.nbasis),
                TRANSDIM, TRANSDIM, TRANSDIM),
            STOREDIM, STOREDIM) * devSim.cons[III] * devSim.cons[JJJ] * devSim.cons[KKK] * devSim.cons[LLL];
}
#endif
