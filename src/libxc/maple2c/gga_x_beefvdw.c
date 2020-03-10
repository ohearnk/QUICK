/*    
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.   
  Do not edit this file directly as it can be overwritten!!   
   
  This Source Code Form is subject to the terms of the Mozilla Public   
  License, v. 2.0. If a copy of the MPL was not distributed with this   
  file, You can obtain one at http://mozilla.org/MPL/2.0/.   
   
  Maple version     : Maple 2016 (X86 64 LINUX)   
  Maple source      : ../maple/gga_x_beefvdw.mpl   
  Type of functional: work_gga_x   
*/   
   
#ifdef DEVICE   
__device__ void xc_gga_x_beefvdw_enhance_kernel
  (const void *p,  xc_gga_work_x_t *r)   
#else   
void xc_gga_x_beefvdw_enhance   
  (const xc_func_type *p,  xc_gga_work_x_t *r)   
#endif   
{   
  double t1, t2, t3, t4, t5, t6, t7, t10;   
  double t11, t13, t15, t16, t17, t18, t19, t20;   
  double t21, t23, t24, t27, t28, t30, t31, t32;   
  double t34, t36, t37, t39, t40, t42, t44, t45;   
  double t47, t49, t52, t54, t56, t58, t60, t74;   
  double t76, t78, t81, t82, t83, t84, t86, t90;   
  double t91, t93, t95, t97, t99, t103, t105, t107;   
  double t109, t111, t113, t115, t117, t118, t120, t122;   
  double t124, t126, t128, t130, t132, t134, t136, t138;   
  double t140, t142, t144, t146, t148, t149, t152, t154;   
  double t157, t158, t159, t162, t163, t165, t190, t221;   
  double t253, t282, t285, t288, t292, t296, t298, t341;   
  double t347, t364, t387, t433, t457, t479;   
   
   
  t1 = M_CBRT6;   
  t2 = 0.31415926535897932385e1 * 0.31415926535897932385e1;   
  t3 = cbrt(t2);   
  t4 = t3 * t3;   
  t5 = 0.1e1 / t4;   
  t6 = t1 * t5;   
  t7 = r->x * r->x;   
  t10 = 0.4e1 + t6 * t7 / 0.24e2;   
  t11 = 0.1e1 / t10;   
  t13 = t6 * t7 * t11;   
  t15 = t13 / 0.12e2 - 0.1e1;   
  t16 = t15 * t15;   
  t17 = t16 * t16;   
  t18 = t17 * t17;   
  t19 = t18 * t17;   
  t20 = t18 * t18;   
  t21 = t20 * t19;   
  t23 = t17 * t15;   
  t24 = t18 * t23;   
  t27 = t18 * t16;   
  t28 = t20 * t27;   
  t30 = t16 * t15;   
  t31 = t18 * t30;   
  t32 = t20 * t31;   
  t34 = t20 * t18;   
  t36 = t18 * t15;   
  t37 = t20 * t36;   
  t39 = t17 * t30;   
  t40 = t20 * t39;   
  t42 = t20 * t23;   
  t44 = t17 * t16;   
  t45 = t20 * t44;   
  t47 = t20 * t30;   
  t49 = t20 * t17;   
  t52 = t20 * t15;   
  t54 = t20 * t16;   
  t56 = t18 * t39;   
  t58 = -0.54277774626371860320e4 * t21 + 0.41355861880146538750e4 * t20 * t24 + 0.40074935854432390114e5 * t28 - 0.29150193011493262292e5 * t32 - 0.13204466182182150467e6 * t34 + 0.90365611108522808258e5 * t37 - 0.16114215399846280595e6 * t40 + 0.18078200670879145336e6 * t42 + 0.25589479526235334610e6 * t45 - 0.12981481812794983922e6 * t47 - 0.32352403136049329184e6 * t49 + 0.37534251004296526981e-1 * t13 + 0.56174007979372666951e5 * t52 + 0.27967048856303053872e6 * t54 - 0.10276426607863824397e5 * t56;   
  t60 = t18 * t44;   
  t74 = 0.11313514630621233134e1 - 0.16837084139014120539e6 * t20 + 0.70504541869034010051e5 * t60 - 0.20148245175625047025e5 * t19 - 0.2810240180568462990e4 * t24 + 0.37835396407252402359e4 * t27 + 0.22748997850816485208e4 * t31 - 0.44233229018433803622e3 * t18 - 0.61754786104528599731e3 * t36 + 0.30542034959315850168e2 * t44 + 0.86005730499279641299e2 * t39 - 0.69459735177638985466e0 * t17 - 0.72975787893717136018e1 * t23 + 0.52755620115589800943e0 * t30 - 0.38916037779196815969e0 * t16;   
  r->f = t58 + t74;   
   
  if(r->order < 1) return;   
   
  t76 = t6 * r->x * t11;   
  t78 = t1 * t1;   
  t81 = t78 / t3 / t2;   
  t82 = t7 * r->x;   
  t83 = t10 * t10;   
  t84 = 0.1e1 / t83;   
  t86 = t81 * t82 * t84;   
  t90 = t76 / 0.6e1 - t86 / 0.144e3;   
  t91 = t40 * t90;   
  t93 = t34 * t90;   
  t95 = t37 * t90;   
  t97 = t28 * t90;   
  t99 = t32 * t90;   
  t103 = t44 * t90;   
  t105 = t39 * t90;   
  t107 = t18 * t90;   
  t109 = t36 * t90;   
  t111 = t27 * t90;   
  t113 = t31 * t90;   
  t115 = t19 * t90;   
  t117 = 0.75068502008593053962e-1 * t76 - 0.31278542503580439151e-2 * t86 - 0.31690718837237161121e7 * t91 + 0.22591402777130702064e7 * t93 + 0.10419483322152421430e7 * t95 - 0.78705521131031808188e6 * t97 - 0.15197776895384120890e6 * t99 + 0.11993199945242496238e6 * t21 * t90 + 0.60204011349495748909e3 * t103 - 0.35386583214747042898e4 * t105 - 0.55579307494075739758e4 * t107 + 0.37835396407252402359e5 * t109 + 0.25023897635898133729e5 * t111 - 0.24177894210750056430e6 * t113 - 0.36533122347390018870e5 * t115;   
  t118 = t24 * t90;   
  t120 = t60 * t90;   
  t122 = t56 * t90;   
  t124 = t20 * t90;   
  t126 = t52 * t90;   
  t128 = t54 * t90;   
  t130 = t47 * t90;   
  t132 = t49 * t90;   
  t134 = t42 * t90;   
  t136 = t45 * t90;   
  t138 = t15 * t90;   
  t140 = t16 * t90;   
  t142 = t30 * t90;   
  t144 = t17 * t90;   
  t146 = t23 * t90;   
  t148 = 0.98706358616647614071e6 * t118 - 0.15414639911795736596e6 * t120 - 0.26939334622422592862e7 * t122 + 0.95495813564933533817e6 * t124 + 0.50340687941345496970e7 * t126 - 0.24664815444310469452e7 * t128 - 0.64704806272098658368e7 * t130 + 0.37964221408846205206e7 * t132 + 0.56296854957717736142e7 * t134 - 0.37062695419646445368e7 * t136 - 0.77832075558393631938e0 * t138 + 0.15826686034676940283e1 * t140 - 0.27783894071055594186e1 * t142 - 0.36487893946858568009e2 * t144 + 0.18325220975589510101e3 * t146;   
  r->dfdx = t117 + t148;   
   
  if(r->order < 2) return;   
   
  t149 = t90 * t90;   
  t152 = t81 * t7 * t84;   
  t154 = t6 * t11;   
  t157 = t2 * t2;   
  t158 = 0.1e1 / t157;   
  t159 = t7 * t7;   
  t162 = 0.1e1 / t83 / t10;   
  t163 = t158 * t159 * t162;   
  t165 = t154 / 0.6e1 - 0.5e1 / 0.144e3 * t152 + t163 / 0.144e3;   
  t190 = -0.77832075558393631938e0 * t149 - 0.15639271251790219576e-1 * t152 - 0.24664815444310469452e7 * t54 * t165 - 0.64704806272098658368e7 * t47 * t165 + 0.37964221408846205206e7 * t49 * t165 + 0.56296854957717736142e7 * t42 * t165 - 0.37062695419646445368e7 * t45 * t165 - 0.31690718837237161121e7 * t40 * t165 + 0.22591402777130702064e7 * t34 * t165 + 0.10419483322152421430e7 * t37 * t165 - 0.78705521131031808188e6 * t28 * t165 - 0.15197776895384120890e6 * t32 * t165 + 0.11993199945242496238e6 * t21 * t165 - 0.77832075558393631938e0 * t15 * t165;   
  t221 = 0.15826686034676940283e1 * t16 * t165 - 0.27783894071055594186e1 * t30 * t165 - 0.36487893946858568009e2 * t17 * t165 + 0.18325220975589510101e3 * t23 * t165 + 0.60204011349495748909e3 * t44 * t165 - 0.35386583214747042898e4 * t39 * t165 - 0.55579307494075739758e4 * t18 * t165 + 0.37835396407252402359e5 * t36 * t165 + 0.25023897635898133729e5 * t27 * t165 - 0.24177894210750056430e6 * t31 * t165 - 0.36533122347390018870e5 * t19 * t165 + 0.98706358616647614071e6 * t24 * t165 - 0.15414639911795736596e6 * t60 * t165 - 0.26939334622422592862e7 * t56 * t165 + 0.95495813564933533817e6 * t20 * t165;   
  t253 = 0.50340687941345496970e7 * t52 * t165 + 0.12831826620164189829e8 * t19 * t149 - 0.21580495876514031234e7 * t24 * t149 - 0.40409001933633889293e8 * t60 * t149 + 0.15279330170389365411e8 * t56 * t149 + 0.85579169500287344849e8 * t20 * t149 - 0.44396667799758845014e8 * t52 * t149 - 0.12293913191698745090e9 * t54 * t149 + 0.75928442817692410412e8 * t47 * t149 + 0.11822339541120724590e9 * t49 * t149 - 0.81537929923222179810e8 * t42 * t149 - 0.72888653325645470578e8 * t45 * t149 + 0.54219366665113684954e8 * t40 * t149 + 0.26048708305381053575e8 * t34 * t149 - 0.20463435494068270129e8 * t37 * t149;   
  t282 = -0.41033997617537126403e7 * t28 * t149 + 0.33580959846678989466e7 * t32 * t149 + 0.31653372069353880566e1 * t15 * t149 - 0.83351682213166782558e1 * t16 * t149 - 0.14595157578743427204e3 * t30 * t149 + 0.91626104877947550505e3 * t17 * t149 + 0.36122406809697449345e4 * t23 * t149 - 0.24770608250322930029e5 * t44 * t149 - 0.44463445995260591806e5 * t39 * t149 + 0.34051856766527162123e6 * t18 * t149 + 0.25023897635898133729e6 * t36 * t149 - 0.26595683631825062073e7 * t27 * t149 - 0.43839746816868022644e6 * t31 * t149 + 0.75068502008593053962e-1 * t154 + 0.31278542503580439151e-2 * t163;   
  r->d2fdx2 = t190 + t221 + t253 + t282;   
   
  if(r->order < 3) return;   
   
  t285 = t81 * t84 * r->x;   
  t288 = t158 * t82 * t162;   
  t292 = t83 * t83;   
  t296 = t158 * t159 * r->x / t292 * t1 * t5;   
  t298 = -t285 / 0.12e2 + t288 / 0.16e2 - t296 / 0.576e3;   
  t341 = -0.24461378976966653943e9 * t134 * t165 - 0.21866595997693641174e9 * t136 * t165 - 0.64741487629542093702e7 * t118 * t165 - 0.12122700580090166788e9 * t120 * t165 + 0.45837990511168096233e8 * t122 * t165 + 0.25673750850086203455e9 * t124 * t165 - 0.13319000339927653504e9 * t126 * t165 - 0.13339033798578177542e6 * t105 * t165 + 0.10215557029958148637e7 * t107 * t165 + 0.75071692907694401187e6 * t109 * t165 - 0.79787050895475186219e7 * t111 * t165;   
  t347 = t149 * t90;   
  t364 = -0.13151924045060406793e7 * t113 * t165 + 0.38495479860492569487e8 * t115 * t165 + 0.31653372069353880566e1 * t347 - 0.78196356258951097878e-3 * t296 - 0.24177894210750056430e6 * t31 * t298 - 0.36533122347390018870e5 * t19 * t298 + 0.98706358616647614071e6 * t24 * t298 - 0.15414639911795736596e6 * t60 * t298 - 0.26939334622422592862e7 * t56 * t298 + 0.95495813564933533817e6 * t20 * t298 + 0.50340687941345496970e7 * t52 * t298;   
  t387 = -0.24664815444310469452e7 * t54 * t298 - 0.64704806272098658368e7 * t47 * t298 + 0.37964221408846205206e7 * t49 * t298 + 0.56296854957717736142e7 * t42 * t298 - 0.37062695419646445368e7 * t45 * t298 - 0.31690718837237161121e7 * t40 * t298 + 0.22591402777130702064e7 * t34 * t298 + 0.10419483322152421430e7 * t37 * t298 - 0.78705521131031808188e6 * t28 * t298 - 0.15197776895384120890e6 * t32 * t298 - 0.10668839380559652865e9 * t37 * t347;   
  t433 = 0.37835396407252402359e5 * t36 * t298 + 0.25023897635898133729e5 * t27 * t298 + 0.14426404135361557978e10 * t54 * t347 + 0.23644679082241449180e10 * t47 * t347 - 0.17122965283876657760e10 * t49 * t347 - 0.16035503731642003527e10 * t42 * t347 + 0.12470454332976147539e10 * t45 * t347 + 0.62516899932914528580e9 * t40 * t347 - 0.51158588735170675322e9 * t34 * t347 - 0.48223721498554824908e7 * t27 * t347 + 0.15398191944197027795e9 * t31 * t347;   
  t457 = -0.28054644639468240604e8 * t19 * t347 - 0.56572602707087445010e9 * t24 * t347 + 0.22918995255584048116e9 * t60 * t347 + 0.13692667120045975176e10 * t56 * t347 - 0.75474335259590036524e9 * t20 * t347 - 0.22129043745057741162e10 * t52 * t347 + 0.36650441951179020202e4 * t30 * t347 + 0.18061203404848724672e5 * t17 * t347 - 0.14862364950193758017e6 * t23 * t347 - 0.31124412196682414264e6 * t44 * t347 + 0.27241485413221729698e7 * t39 * t347;   
  t479 = 0.22521507872308320356e7 * t18 * t347 - 0.26595683631825062073e8 * t36 * t347 - 0.16670336442633356512e2 * t15 * t347 - 0.43785472736230281612e3 * t16 * t347 - 0.25005504663950034768e2 * t140 * t165 - 0.43785472736230281612e3 * t142 * t165 + 0.27487831463384265152e4 * t144 * t165 + 0.10836722042909234804e5 * t146 * t165 - 0.74311824750968790087e5 * t103 * t165 + 0.28150688253222395236e-1 * t288 + 0.94960116208061641698e1 * t138 * t165;   
  r->d3fdx3 = -0.35386583214747042898e4 * t39 * t298 - 0.55579307494075739758e4 * t18 * t298 + 0.90668591586033271558e8 * t28 * t347 - 0.23349622667518089582e1 * t90 * t165 - 0.77832075558393631938e0 * t15 * t298 + 0.15826686034676940283e1 * t16 * t298 - 0.27783894071055594186e1 * t30 * t298 - 0.36487893946858568009e2 * t17 * t298 + 0.18325220975589510101e3 * t23 * t298 + 0.60204011349495748909e3 * t44 * t298 + 0.11993199945242496238e6 * t21 * t298 + 0.16265809999534105486e9 * t91 * t165 + 0.78146124916143160725e8 * t93 * t165 - 0.61390306482204810387e8 * t95 * t165 - 0.12310199285261137921e8 * t97 * t165 + 0.10074287954003696840e8 * t99 * t165 - 0.36881739575096235270e9 * t128 * t165 + 0.22778532845307723123e9 * t130 * t165 + 0.35467018623362173770e9 * t132 * t165 + t479 + t457 + t433 + t387 + t364 + t341 - 0.37534251004296526982e-1 * t285;   
   
  if(r->order < 4) return;   
   
   
}   
   
#ifndef DEVICE   
#define maple2c_order 3   
#define maple2c_func  xc_gga_x_beefvdw_enhance   
#define kernel_id 17 
#endif