/*
   !---------------------------------------------------------------------!
   ! Written by Madu Manathunga on 12/03/2020                            !
   !                                                                     !
   ! Copyright (C) 2020-2021 Merz lab                                    !
   ! Copyright (C) 2020-2021 Götz lab                                    !
   !                                                                     !
   ! This Source Code Form is subject to the terms of the Mozilla Public !
   ! License, v. 2.0. If a copy of the MPL was not distributed with this !
   ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
   !_____________________________________________________________________!

   !---------------------------------------------------------------------!
   ! This source file contains preprocessable functions required for     !
   ! QUICK GPU version.                                                  !
   !---------------------------------------------------------------------!
   */

#if defined(OSHELL)
  #define NSPIN 2
#else
  #define NSPIN 1
#endif


//-----------------------------------------------
// Calculate the density and gradients of density at
// each grid point.
//-----------------------------------------------
#if defined(OSHELL)
__global__ void k_get_density_oshell
#else
__global__ void k_get_density_cshell
#endif
    (uint32_t natom, uint32_t nbasis, int npoints, int32_t maxcontract,
    QUICKDouble const * const gridx, QUICKDouble const * const gridy, QUICKDouble const * const gridz,
    QUICKDouble * const densa, QUICKDouble * const densb,
    QUICKDouble * const gax, QUICKDouble * const gbx,
    QUICKDouble * const gay, QUICKDouble * const gby,
    QUICKDouble * const gaz, QUICKDouble * const gbz,
    int const * const basf, int const * const primf, int const * const basf_locator,
    int const * const primf_locator, int const * const bin_locator,
    int const * const itype, QUICKDouble const * const aexp, QUICKDouble const * const dcoeff,
    QUICKDouble const * const xyz, int const * const ncenter, QUICKDouble const * const dense,
#if defined(OSHELL)
    QUICKDouble const * const denseb,
#endif
    QUICKDouble XCCutoff)
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (QUICKULL gid = offset; gid < npoints; gid += totalThreads) {
        int bin_id = bin_locator[gid];
        int bfloc_st = basf_locator[bin_id];
        int bfloc_end = basf_locator[bin_id + 1];

        QUICKDouble gridxg = gridx[gid];
        QUICKDouble gridyg = gridy[gid];
        QUICKDouble gridzg = gridz[gid];

        QUICKDouble density = 0.0;
        QUICKDouble gaxg = 0.0;
        QUICKDouble gayg = 0.0;
        QUICKDouble gazg = 0.0;
#if defined(OSHELL)
        QUICKDouble densityb = 0.0;
        QUICKDouble gbxg = 0.0;
        QUICKDouble gbyg = 0.0;
        QUICKDouble gbzg = 0.0;
#endif

        for (int i = bfloc_st; i < bfloc_end; i++) {
            int ibas = basf[i];
            QUICKDouble phi, dphidx, dphidy, dphidz;

            pteval_new(natom, nbasis, maxcontract,
                    gridxg, gridyg, gridzg, &phi, &dphidx, &dphidy, &dphidz,
                    primf, primf_locator, ibas, i,
                    itype, aexp, dcoeff, xyz, ncenter);

            if (abs(phi + dphidx + dphidy + dphidz) >= XCCutoff) {
                QUICKDouble denseii = LOC2(dense, ibas, ibas, nbasis, nbasis) * phi;
#if defined(OSHELL)
                QUICKDouble densebii = LOC2(denseb, ibas, ibas, nbasis, nbasis) * phi;
#endif

#if defined(OSHELL)
                density += denseii * phi;
                densityb += densebii * phi;
#else
                density += denseii * phi / 2.0;
#endif
                gaxg += denseii * dphidx;
                gayg += denseii * dphidy;
                gazg += denseii * dphidz;
#if defined(OSHELL)
                gbxg += densebii * dphidx;
                gbyg += densebii * dphidy;
                gbzg += densebii * dphidz;
#endif

                for (int j = i + 1; j < bfloc_end; j++) {
                    int jbas = basf[j];
                    QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

                    pteval_new(natom, nbasis, maxcontract,
                            gridxg, gridyg, gridzg, &phi2, &dphidx2, &dphidy2, &dphidz2,
                            primf, primf_locator, jbas, j,
                            itype, aexp, dcoeff, xyz, ncenter);

                    QUICKDouble denseij = LOC2(dense, ibas, jbas, nbasis, nbasis);
#if defined(OSHELL)
                    QUICKDouble densebij = LOC2(denseb, ibas, jbas, nbasis, nbasis);
#endif

#if defined(OSHELL)
                    density += 2.0 * denseij * phi * phi2;
                    densityb += 2.0 * densebij * phi * phi2;
#else
                    density += denseij * phi * phi2;
#endif
                    gaxg += denseij * (phi * dphidx2 + phi2 * dphidx);
                    gayg += denseij * (phi * dphidy2 + phi2 * dphidy);
                    gazg += denseij * (phi * dphidz2 + phi2 * dphidz);
#if defined(OSHELL)
                    gbxg += densebij * (phi * dphidx2 + phi2 * dphidx);
                    gbyg += densebij * (phi * dphidy2 + phi2 * dphidy);
                    gbzg += densebij * (phi * dphidz2 + phi2 * dphidz);
#endif
                }
            }
        }
#if defined(OSHELL)
        densa[gid] = density;
        densb[gid] = densityb;
        gax[gid] = 2.0 * gaxg;
        gbx[gid] = 2.0 * gbxg;
        gay[gid] = 2.0 * gayg;
        gby[gid] = 2.0 * gbyg;
        gaz[gid] = 2.0 * gazg;
        gbz[gid] = 2.0 * gbzg;
#else
        densa[gid] = density;
        densb[gid] = density;
        gax[gid] = gaxg;
        gbx[gid] = gaxg;
        gay[gid] = gayg;
        gby[gid] = gayg;
        gaz[gid] = gazg;
        gbz[gid] = gazg;
#endif
    }
}


#if defined(OSHELL)
__global__ void k_getxc_oshell
#else
__global__ void k_getxc_cshell
#endif
    (QUICK_METHOD method, DFT_calculated_type * const DFT_calculated,
     uint32_t natom, uint32_t nbasis, int npoints, int32_t maxcontract,
     QUICKDouble const * const gridx, QUICKDouble const * const gridy, QUICKDouble const * const gridz,
     QUICKDouble const * const weight, QUICKDouble * const densa, QUICKDouble * const densb,
     QUICKDouble * const gax, QUICKDouble * const gbx,
     QUICKDouble * const gay, QUICKDouble * const gby,
     QUICKDouble * const gaz, QUICKDouble * const gbz,
     int const * const basf, int const * const primf, int const * const basf_locator,
     int const * const primf_locator, int const * const bin_locator,
     int const * const itype, QUICKDouble const * const aexp, QUICKDouble const * const dcoeff,
     QUICKDouble const * const xyz, int const * const ncenter,
     gpu_libxc_info ** const glinfo, int nauxfunc,
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
     QUICKDouble const * const dense,
#if defined(OSHELL)
     QUICKDouble const * const denseb,
#endif
     QUICKDouble DMCutoff, QUICKDouble XCCutoff)
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (QUICKULL gid = offset; gid < npoints; gid += totalThreads) {
        int bin_id = bin_locator[gid];
        int bfloc_st = basf_locator[bin_id];
        int bfloc_end = basf_locator[bin_id+1];

        QUICKDouble gridxg = gridx[gid];
        QUICKDouble gridyg = gridy[gid];
        QUICKDouble gridzg = gridz[gid];

        QUICKDouble weightg = weight[gid];
        QUICKDouble density = densa[gid];
        QUICKDouble densityb = densb[gid];
        QUICKDouble gaxg = gax[gid];
        QUICKDouble gayg = gay[gid];
        QUICKDouble gazg = gaz[gid];
        QUICKDouble gbxg = gbx[gid];
        QUICKDouble gbyg = gby[gid];
        QUICKDouble gbzg = gbz[gid];

        if (density > DMCutoff) {
            QUICKDouble dfdr;
            QUICKDouble xdot, ydot, zdot;
            QUICKDouble _tmp ;
#if defined(OSHELL)
            QUICKDouble dfdrb;
            QUICKDouble xdotb, ydotb, zdotb;

            QUICKDouble gaa = (gaxg * gaxg + gayg * gayg + gazg * gazg);
            QUICKDouble gab = (gaxg * gbxg + gayg * gbyg + gazg * gbzg);
            QUICKDouble gbb = (gbxg * gbxg + gbyg * gbyg + gbzg * gbzg);
#else
            QUICKDouble dot;
            QUICKDouble sigma = 4.0 * (gaxg * gaxg + gayg * gayg + gazg * gazg);

            if (method == B3LYP) {
                _tmp = b3lyp_e(2.0 * density, sigma) * weightg;
            } else if (method == BLYP) {
                _tmp = (becke_e(density, densityb, gaxg, gayg, gazg, gbxg, gbyg, gbzg)
                        + lyp_e(density, densityb, gaxg, gayg, gazg, gbxg, gbyg, gbzg)) * weightg;
            }

            if (method == B3LYP) {
                dot = b3lypf(2.0 * density, sigma, &dfdr);
                xdot = dot * gaxg;
                ydot = dot * gayg;
                zdot = dot * gazg;
            } else if (method == BLYP) {
                QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
                QUICKDouble dfdr2;

                becke(density, gaxg, gayg, gazg, gbxg, gbyg, gbzg, &dfdr, &dfdgaa, &dfdgab);
                lyp(density, densityb, gaxg, gayg, gazg, gbxg, gbyg, gbzg, &dfdr2, &dfdgaa2, &dfdgab2);
                dfdr += dfdr2;
                dfdgaa += dfdgaa2;
                dfdgab += dfdgab2;
                // Calculate the first term in the dot product shown above, i.e.:
                // (2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
                xdot = 2.0 * dfdgaa * gaxg + dfdgab * gbxg;
                ydot = 2.0 * dfdgaa * gayg + dfdgab * gbyg;
                zdot = 2.0 * dfdgaa * gazg + dfdgab * gbzg;
            } else if (method == LIBXC) {
#endif
                //Prepare in/out for libxc call
                double d_rhoa = (double) density;
                double d_rhob = (double) densityb;

                // array d_sigma stores gaa, gab and gbb respectively
                QUICKDouble d_sigma[3] = {0.0, 0.0, 0.0};
                // array d_vrho stores dfdra and dfdrb respectively
                QUICKDouble d_vrho[2] = {0.0, 0.0};
                // array d_vsigma carries dfdgaa, dfdgab and dfdgbb respectively
                QUICKDouble d_vsigma[3] = {0.0, 0.0, 0.0};
                QUICKDouble d_zk = 0.0;

#if defined(OSHELL)
                d_sigma[0] = gaa;
                d_sigma[1] = gab;
                d_sigma[2] = gbb;
#else
                d_sigma[0] = sigma;
#endif

                for (int i = 0; i < nauxfunc; i++) {
                    QUICKDouble tmp_d_zk = 0.0;
                    QUICKDouble tmp_d_vrho[2] = {0.0, 0.0};
                    QUICKDouble tmp_d_vsigma[3] = {0.0, 0.0, 0.0};

                    gpu_libxc_info *tmp_glinfo = glinfo[i];

                    switch (tmp_glinfo->gpu_worker) {
                        case GPU_WORK_LDA:
                            gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, (QUICKDouble *) &tmp_d_vrho, NSPIN);
                            break;

                        case GPU_WORK_GGA_X:
                            gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, (QUICKDouble *) &d_sigma, &tmp_d_zk,
                                    (QUICKDouble *) &tmp_d_vrho, (QUICKDouble *) &tmp_d_vsigma, NSPIN);
                            break;

                        case GPU_WORK_GGA_C:
                            gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, (QUICKDouble *) &d_sigma, &tmp_d_zk,
                                    (QUICKDouble *) &tmp_d_vrho, (QUICKDouble *) &tmp_d_vsigma, NSPIN);
                            break;

                        default:
                            break;
                    }

                    d_zk += tmp_d_zk * tmp_glinfo->mix_coeff;
                    d_vrho[0] += tmp_d_vrho[0] * tmp_glinfo->mix_coeff;
                    d_vsigma[0] += tmp_d_vsigma[0] * tmp_glinfo->mix_coeff;
#if defined(OSHELL)
                    d_vrho[1] += tmp_d_vrho[1] * tmp_glinfo->mix_coeff;
                    d_vsigma[1] += tmp_d_vsigma[1] * tmp_glinfo->mix_coeff;
                    d_vsigma[2] += tmp_d_vsigma[2] * tmp_glinfo->mix_coeff;
#endif
                }

                _tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)) * weightg);
                dfdr = (QUICKDouble) d_vrho[0];
#if defined(OSHELL)
                dfdrb= (QUICKDouble) d_vrho[1];

                xdot = 2.0 * d_vsigma[0] * gaxg + d_vsigma[1] * gbxg;
                ydot = 2.0 * d_vsigma[0] * gayg + d_vsigma[1] * gbyg;
                zdot = 2.0 * d_vsigma[0] * gazg + d_vsigma[1] * gbzg;

                xdotb = 2.0 * d_vsigma[2] * gbxg + d_vsigma[1] * gaxg;
                ydotb = 2.0 * d_vsigma[2] * gbyg + d_vsigma[1] * gayg;
                zdotb = 2.0 * d_vsigma[2] * gbzg + d_vsigma[1] * gazg;
#else
                xdot = 4.0 * d_vsigma[0] * gaxg;
                ydot = 4.0 * d_vsigma[0] * gayg;
                zdot = 4.0 * d_vsigma[0] * gazg;
#endif
#ifndef OSHELL
            }
#endif

#if defined(USE_LEGACY_ATOMICS)
            GPUATOMICADD(&DFT_calculated[0].Eelxc, _tmp, OSCALE);
            GPUATOMICADD(&DFT_calculated[0].aelec, weightg * density, OSCALE);
            GPUATOMICADD(&DFT_calculated[0].belec, weightg * densityb, OSCALE);
#else
            atomicAdd(&DFT_calculated[0].Eelxc, _tmp);
            atomicAdd(&DFT_calculated[0].aelec, weightg * density);
            atomicAdd(&DFT_calculated[0].belec, weightg * densityb);
#endif

            for (int i = bfloc_st; i < bfloc_end; ++i) {
                int ibas = basf[i];
                QUICKDouble phi, dphidx, dphidy, dphidz;

                pteval_new(natom, nbasis, maxcontract,
                        gridxg, gridyg, gridzg, &phi, &dphidx, &dphidy, &dphidz,
                        primf, primf_locator, ibas, i,
                        itype, aexp, dcoeff, xyz, ncenter);

                if (abs(phi + dphidx + dphidy + dphidz) > XCCutoff) {
                    for (int j = bfloc_st; j < bfloc_end; j++) {
                        int jbas = basf[j];
                        QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

                        pteval_new(natom, nbasis, maxcontract,
                                gridxg, gridyg, gridzg, &phi2, &dphidx2, &dphidy2, &dphidz2,
                                primf, primf_locator, jbas, j,
                                itype, aexp, dcoeff, xyz, ncenter);

                        QUICKDouble _tmp = (phi * phi2 * dfdr + xdot * (phi * dphidx2 + phi2 * dphidx)
                                + ydot * (phi * dphidy2 + phi2 * dphidy) + zdot * (phi * dphidz2 + phi2 * dphidz)) * weightg;

#if defined(USE_LEGACY_ATOMICS)
                        GPUATOMICADD(&LOC2(oULL, jbas, ibas, nbasis, nbasis), _tmp, OSCALE);
#else
                        atomicAdd(&LOC2(o, jbas, ibas, nbasis, nbasis), _tmp);
#endif

#if defined(OSHELL)
                        QUICKDouble _tmpb = (phi * phi2 * dfdrb + xdotb * (phi * dphidx2 + phi2 * dphidx)
                                + ydotb * (phi * dphidy2 + phi2 * dphidy) + zdotb * (phi * dphidz2 + phi2 * dphidz)) * weightg;

#if defined(USE_LEGACY_ATOMICS)
                        GPUATOMICADD(&LOC2(obULL, jbas, ibas, nbasis, nbasis), _tmpb, OSCALE);
#else
                        atomicAdd(&LOC2(ob, jbas, ibas, nbasis, nbasis), _tmpb);
#endif
#endif
                    }
                }
            }
        }
    }
}


#if defined(OSHELL)
__global__ void k_getxc_grad_oshell
#else
__global__ void k_getxc_grad_cshell
#endif
    (QUICK_METHOD method, uint32_t natom, uint32_t nbasis, int npoints, int32_t maxcontract,
    QUICKDouble const * const gridx, QUICKDouble const * const gridy, QUICKDouble const * const gridz,
    QUICKDouble const * const sswt, QUICKDouble const * const weight,
    QUICKDouble * const densa, QUICKDouble * const densb,
    QUICKDouble * const gax, QUICKDouble * const gbx,
    QUICKDouble * const gay, QUICKDouble * const gby,
    QUICKDouble * const gaz, QUICKDouble * const gbz,
    QUICKDouble * const exc, int const * const gatm, int * const dweight_ssd,
    int const * const basf, int const * const primf, int const * const basf_locator,
    int const * const primf_locator, int const * const bin_locator,
    gpu_libxc_info ** const glinfo, int nauxfunc, int const * const ncenter,
    int const * const itype, QUICKDouble const * const aexp, QUICKDouble const * const dcoeff,
    QUICKDouble const * const xyz, QUICKDouble const * const dense,
#if defined(OSHELL)
    QUICKDouble const * const denseb,
#endif
    QUICKDouble DMCutoff, QUICKDouble XCCutoff,
#if defined(USE_LEGACY_ATOMICS)
    QUICKULL * const gradULL,
#else
    QUICKDouble * const grad,
#endif
    QUICKDouble const * const cew_vrecip, bool use_cew)
{
#if defined(USE_LEGACY_ATOMICS)
    //declare smem grad vector
    extern __shared__ QUICKULL smem_buffer[];
    QUICKULL *smemGrad = (QUICKULL *) smem_buffer;

    // initialize smem grad
    for (int i = threadIdx.x; i < natom * 3; i += blockDim.x) {
        smemGrad[i] = 0ull;
    }
#else
    //declare smem grad vector
    extern __shared__ QUICKDouble smem_buffer[];
    QUICKDouble *smemGrad = (QUICKDouble *) smem_buffer;

    // initialize smem grad
    for (int i = threadIdx.x; i < natom * 3; i += blockDim.x) {
        smemGrad[i] = 0.0;
    }
#endif

    __syncthreads();

    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (QUICKULL gid = offset; gid < npoints; gid += totalThreads) {
        int bin_id = bin_locator[gid];
        int bfloc_st = basf_locator[bin_id];
        int bfloc_end = basf_locator[bin_id+1];

        QUICKDouble gridxg = gridx[gid];
        QUICKDouble gridyg = gridy[gid];
        QUICKDouble gridzg = gridz[gid];
        QUICKDouble weightg = weight[gid];
        QUICKDouble density = densa[gid];
        QUICKDouble densityb = densb[gid];
        QUICKDouble gaxg = gax[gid];
        QUICKDouble gayg = gay[gid];
        QUICKDouble gazg = gaz[gid];
        QUICKDouble gbxg = gbx[gid];
        QUICKDouble gbyg = gby[gid];
        QUICKDouble gbzg = gbz[gid];

#if defined(CEW)
        QUICKDouble dfdr_cew = 0.0;
        if (use_cew) {
            dfdr_cew = cew_vrecip[gid];
        }
#endif

        if (density > DMCutoff) {
            QUICKDouble dfdr;
            QUICKDouble xdot, ydot, zdot;
            QUICKDouble _tmp ;

#if defined(OSHELL)
            QUICKDouble dfdrb;
            QUICKDouble xdotb, ydotb, zdotb;

            QUICKDouble gaa = (gaxg * gaxg + gayg * gayg + gazg * gazg);
            QUICKDouble gab = (gaxg * gbxg + gayg * gbyg + gazg * gbzg);
            QUICKDouble gbb = (gbxg * gbxg + gbyg * gbyg + gbzg * gbzg);
#else
            QUICKDouble dot;
            QUICKDouble sigma = 4.0 * (gaxg * gaxg + gayg * gayg + gazg * gazg);

            if (method == B3LYP) {
                _tmp = b3lyp_e(2.0 * density, sigma);
            } else if (method == BLYP) {
                _tmp = (becke_e(density, densityb, gaxg, gayg, gazg, gbxg, gbyg, gbzg)
                        + lyp_e(density, densityb, gaxg, gayg, gazg, gbxg, gbyg, gbzg));
            }


            if (method == B3LYP) {
                dot = b3lypf(2.0 * density, sigma, &dfdr);
                xdot = dot * gaxg;
                ydot = dot * gayg;
                zdot = dot * gazg;
            } else if (method == BLYP) {
                QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
                QUICKDouble dfdr2;

                becke(density, gaxg, gayg, gazg, gbxg, gbyg, gbzg, &dfdr, &dfdgaa, &dfdgab);
                lyp(density, densityb, gaxg, gayg, gazg, gbxg, gbyg, gbzg, &dfdr2, &dfdgaa2, &dfdgab2);
                dfdr += dfdr2;
                dfdgaa += dfdgaa2;
                dfdgab += dfdgab2;

                //Calculate the first term in the dot product shown above,i.e.:
                //(2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
                xdot = 2.0 * dfdgaa * gaxg + dfdgab * gbxg;
                ydot = 2.0 * dfdgaa * gayg + dfdgab * gbyg;
                zdot = 2.0 * dfdgaa * gazg + dfdgab * gbzg;

            } else if (method == LIBXC) {
#endif
                //Prepare in/out for libxc call
                QUICKDouble d_rhoa = density;
                QUICKDouble d_rhob = densityb;
                // array d_sigma stores gaa, gab and gbb respectively
                QUICKDouble d_sigma[3] = {0.0, 0.0, 0.0};
                // array d_vrho stores dfdra and dfdrb respectively
                QUICKDouble d_vrho[2] = {0.0, 0.0};
                // array d_vsigma carries dfdgaa, dfdgab and dfdgbb respectively
                QUICKDouble d_vsigma[3] = {0.0, 0.0, 0.0};
                QUICKDouble d_zk = 0.0;

#if defined(OSHELL)
                d_sigma[0] = gaa;
                d_sigma[1] = gab;
                d_sigma[2] = gbb;
#else
                d_sigma[0] = sigma;
#endif

                for (int i = 0; i < nauxfunc; i++) {
                    QUICKDouble tmp_d_zk = 0.0;
                    QUICKDouble tmp_d_vrho[2] = {0.0, 0.0};
                    QUICKDouble tmp_d_vsigma[3] = {0.0, 0.0, 0.0};

                    gpu_libxc_info *tmp_glinfo = glinfo[i];

                    switch (tmp_glinfo->gpu_worker) {
                        case GPU_WORK_LDA:
                            gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, (QUICKDouble *) &tmp_d_vrho, NSPIN);
                            break;

                        case GPU_WORK_GGA_X:
                            gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, (QUICKDouble *) &d_sigma, &tmp_d_zk,
                                    (QUICKDouble *) &tmp_d_vrho, (QUICKDouble *) &tmp_d_vsigma, NSPIN);
                            break;

                        case GPU_WORK_GGA_C:
                            gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, (QUICKDouble *) &d_sigma, &tmp_d_zk,
                                    (QUICKDouble *) &tmp_d_vrho, (QUICKDouble *) &tmp_d_vsigma, NSPIN);
                            break;

                        default:
                            break;
                    }

                    d_zk += (tmp_d_zk * tmp_glinfo->mix_coeff);
                    d_vrho[0] += (tmp_d_vrho[0] * tmp_glinfo->mix_coeff);
                    d_vsigma[0] += (tmp_d_vsigma[0] * tmp_glinfo->mix_coeff);
#if defined(OSHELL)
                    d_vrho[1] += (tmp_d_vrho[1] * tmp_glinfo->mix_coeff);
                    d_vsigma[1]+= (tmp_d_vsigma[1] * tmp_glinfo->mix_coeff);
                    d_vsigma[2] += (tmp_d_vsigma[2] * tmp_glinfo->mix_coeff);
#endif
                }

                _tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)));
                dfdr = (QUICKDouble) d_vrho[0];

#if defined(OSHELL)
                dfdrb= (QUICKDouble) d_vrho[1];

                xdot = 2.0 * d_vsigma[0] * gaxg + d_vsigma[1] * gbxg;
                ydot = 2.0 * d_vsigma[0] * gayg + d_vsigma[1] * gbyg;
                zdot = 2.0 * d_vsigma[0] * gazg + d_vsigma[1] * gbzg;

                xdotb = 2.0 * d_vsigma[2] * gbxg + d_vsigma[1] * gaxg;
                ydotb = 2.0 * d_vsigma[2] * gbyg + d_vsigma[1] * gayg;
                zdotb = 2.0 * d_vsigma[2] * gbzg + d_vsigma[1] * gazg;
#else
                xdot = 4.0 * d_vsigma[0] * gaxg;
                ydot = 4.0 * d_vsigma[0] * gayg;
                zdot = 4.0 * d_vsigma[0] * gazg;
#endif
#ifndef OSHELL
            }
#endif

#if defined(CEW)
            exc[gid] = _tmp + (dfdr_cew * (density + densityb));
#else
            exc[gid] = _tmp;
#endif

            QUICKDouble sumGradx = 0.0, sumGrady = 0.0, sumGradz = 0.0;

            for (int i = bfloc_st; i < bfloc_end; i++) {
                int ibas = basf[i];
                QUICKDouble phi, dphidx, dphidy, dphidz;
                pteval_new(natom, nbasis, maxcontract,
                        gridxg, gridyg, gridzg, &phi, &dphidx, &dphidy, &dphidz,
                        primf, primf_locator, ibas, i,
                        itype, aexp, dcoeff, xyz, ncenter);

                if (abs(phi + dphidx + dphidy + dphidz) > XCCutoff) {
                    QUICKDouble dxdx, dxdy, dxdz, dydy, dydz, dzdz;

                    pt2der_new(natom, nbasis, maxcontract,
                            gridxg, gridyg, gridzg, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz,
                            primf, primf_locator, ibas, i,
                            itype, aexp, dcoeff, xyz, ncenter);

                    int Istart = (ncenter[ibas] - 1) * 3;

                    for (int j = bfloc_st; j < bfloc_end; j++) {
                        int jbas = basf[j];
                        QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

                        pteval_new(natom, nbasis, maxcontract,
                                gridxg, gridyg, gridzg, &phi2, &dphidx2, &dphidy2, &dphidz2,
                                primf, primf_locator, jbas, j,
                                itype, aexp, dcoeff, xyz, ncenter);

                        QUICKDouble denseij = (QUICKDouble) LOC2(dense, ibas, jbas, nbasis, nbasis);

                        QUICKDouble Gradx = -2.0 * denseij * weightg * (dfdr * dphidx * phi2
                                + xdot * (dxdx * phi2 + dphidx * dphidx2)
                                + ydot * (dxdy * phi2 + dphidx * dphidy2)
                                + zdot * (dxdz * phi2 + dphidx * dphidz2));

                        QUICKDouble Grady = -2.0 * denseij * weightg * (dfdr * dphidy * phi2
                                + xdot * (dxdy * phi2 + dphidy * dphidx2)
                                + ydot * (dydy * phi2 + dphidy * dphidy2)
                                + zdot * (dydz * phi2 + dphidy * dphidz2));

                        QUICKDouble Gradz = -2.0 * denseij * weightg * (dfdr * dphidz * phi2
                                + xdot * (dxdz * phi2 + dphidz * dphidx2)
                                + ydot * (dydz * phi2 + dphidz * dphidy2)
                                + zdot * (dzdz * phi2 + dphidz * dphidz2));
#if defined(OSHELL)
                        QUICKDouble densebij = (QUICKDouble) LOC2(denseb, ibas, jbas, nbasis, nbasis);

                        Gradx += -2.0 * densebij * weightg * (dfdrb * dphidx * phi2
                                + xdotb * (dxdx * phi2 + dphidx * dphidx2)
                                + ydotb * (dxdy * phi2 + dphidx * dphidy2)
                                + zdotb * (dxdz * phi2 + dphidx * dphidz2));

                        Grady += -2.0 * densebij * weightg * (dfdrb * dphidy * phi2
                                + xdotb * (dxdy * phi2 + dphidy * dphidx2)
                                + ydotb * (dydy * phi2 + dphidy * dphidy2)
                                + zdotb * (dydz * phi2 + dphidy * dphidz2));

                        Gradz += -2.0 * densebij * weightg * (dfdrb * dphidz * phi2
                                + xdotb * (dxdz * phi2 + dphidz * dphidx2)
                                + ydotb * (dydz * phi2 + dphidz * dphidy2)
                                + zdotb * (dzdz * phi2 + dphidz * dphidz2));
#endif

#if defined(CEW)
                        if (use_cew) {
#if defined(OSHELL)
                            denseij += densebij;
#endif

                            Gradx -= 2.0 * denseij * weightg * dfdr_cew * dphidx * phi2;
                            Grady -= 2.0 * denseij * weightg * dfdr_cew * dphidy * phi2;
                            Gradz -= 2.0 * denseij * weightg * dfdr_cew * dphidz * phi2;
                        }
#endif

                        GPUATOMICADD(&smemGrad[Istart], Gradx, GRADSCALE);
                        GPUATOMICADD(&smemGrad[Istart + 1], Grady, GRADSCALE);
                        GPUATOMICADD(&smemGrad[Istart + 2], Gradz, GRADSCALE);
                        sumGradx += Gradx;
                        sumGrady += Grady;
                        sumGradz += Gradz;
                    }
                }
            }

            int Istart = (gatm[gid] - 1) * 3;

            GPUATOMICADD(&smemGrad[Istart], -sumGradx, GRADSCALE);
            GPUATOMICADD(&smemGrad[Istart + 1], -sumGrady, GRADSCALE);
            GPUATOMICADD(&smemGrad[Istart + 2], -sumGradz, GRADSCALE);
        }
        //Set weights for sswder calculation
        if (density < DMCutoff) {
            dweight_ssd[gid] = 0;
        }

        if (sswt[gid] == 1) {
            dweight_ssd[gid] = 0;
        }
    }

    __syncthreads();

    // update gmem grad vector
    for (int i = threadIdx.x; i < natom * 3; i += blockDim.x) {
#if defined(USE_LEGACY_ATOMICS)
        atomicAdd(&gradULL[i], smemGrad[i]);
#else
        atomicAdd(&grad[i], smemGrad[i]);
#endif
    }

    __syncthreads();
}

#undef NSPIN
