/*
 !---------------------------------------------------------------------!
 ! Written by QUICK-GenInt code generator on 03/25/2023                !
 !                                                                     !
 ! Copyright (C) 2023-2024 Merz lab                                    !
 ! Copyright (C) 2023-2024 Götz lab                                    !
 !                                                                     !
 ! This Source Code Form is subject to the terms of the Mozilla Public !
 ! License, v. 2.0. If a copy of the MPL was not distributed with this !
 ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
 !_____________________________________________________________________!
*/

{

    // [IS|PS] integral - Start
    QUICKDouble VY_1 = VY(0, 0, 1);
    QUICKDouble VY_2 = VY(0, 0, 2);
    QUICKDouble x_3_0_1 = Ptempz * VY_1 + WPtempz * VY_2;
    QUICKDouble VY_3 = VY(0, 0, 3);
    QUICKDouble x_3_0_2 = Ptempz * VY_2 + WPtempz * VY_3;
    QUICKDouble VY_4 = VY(0, 0, 4);
    QUICKDouble x_3_0_3 = Ptempz * VY_3 + WPtempz * VY_4;
    QUICKDouble VY_5 = VY(0, 0, 5);
    QUICKDouble x_3_0_4 = Ptempz * VY_4 + WPtempz * VY_5;
    QUICKDouble VY_6 = VY(0, 0, 6);
    QUICKDouble x_3_0_5 = Ptempz * VY_5 + WPtempz * VY_6;
    QUICKDouble VY_0 = VY(0, 0, 0);
    QUICKDouble x_3_0_0 = Ptempz * VY_0 + WPtempz * VY_1;
    QUICKDouble VY_7 = VY(0, 0, 7);
    QUICKDouble x_3_0_6 = Ptempz * VY_6 + WPtempz * VY_7;
    QUICKDouble x_2_0_1 = Ptempy * VY_1 + WPtempy * VY_2;
    QUICKDouble x_2_0_2 = Ptempy * VY_2 + WPtempy * VY_3;
    QUICKDouble x_2_0_3 = Ptempy * VY_3 + WPtempy * VY_4;
    QUICKDouble x_2_0_4 = Ptempy * VY_4 + WPtempy * VY_5;
    QUICKDouble x_2_0_5 = Ptempy * VY_5 + WPtempy * VY_6;
    QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
    QUICKDouble x_5_0_2 = Ptempy * x_3_0_2 + WPtempy * x_3_0_3;
    QUICKDouble x_5_0_3 = Ptempy * x_3_0_3 + WPtempy * x_3_0_4;
    QUICKDouble x_5_0_4 = Ptempy * x_3_0_4 + WPtempy * x_3_0_5;
    QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
    QUICKDouble x_5_0_5 = Ptempy * x_3_0_5 + WPtempy * x_3_0_6;
    QUICKDouble x_4_0_1 = Ptempx * x_2_0_1 + WPtempx * x_2_0_2;
    QUICKDouble x_4_0_2 = Ptempx * x_2_0_2 + WPtempx * x_2_0_3;
    QUICKDouble x_4_0_3 = Ptempx * x_2_0_3 + WPtempx * x_2_0_4;
    QUICKDouble x_4_0_4 = Ptempx * x_2_0_4 + WPtempx * x_2_0_5;
    QUICKDouble x_10_0_1 = Ptempx * x_5_0_1 + WPtempx * x_5_0_2;
    QUICKDouble x_10_0_2 = Ptempx * x_5_0_2 + WPtempx * x_5_0_3;
    QUICKDouble x_10_0_3 = Ptempx * x_5_0_3 + WPtempx * x_5_0_4;
    QUICKDouble x_10_0_0 = Ptempx * x_5_0_0 + WPtempx * x_5_0_1;
    QUICKDouble x_10_0_4 = Ptempx * x_5_0_4 + WPtempx * x_5_0_5;
    QUICKDouble x_11_0_1 = Ptempx * x_4_0_1 + WPtempx * x_4_0_2 + ABtemp * (x_2_0_1 - CDcom * x_2_0_2);
    QUICKDouble x_11_0_2 = Ptempx * x_4_0_2 + WPtempx * x_4_0_3 + ABtemp * (x_2_0_2 - CDcom * x_2_0_3);
    QUICKDouble x_11_0_3 = Ptempx * x_4_0_3 + WPtempx * x_4_0_4 + ABtemp * (x_2_0_3 - CDcom * x_2_0_4);
    QUICKDouble x_23_0_1 = Ptempx * x_10_0_1 + WPtempx * x_10_0_2 + ABtemp * (x_5_0_1 - CDcom * x_5_0_2);
    QUICKDouble x_23_0_2 = Ptempx * x_10_0_2 + WPtempx * x_10_0_3 + ABtemp * (x_5_0_2 - CDcom * x_5_0_3);
    QUICKDouble x_23_0_0 = Ptempx * x_10_0_0 + WPtempx * x_10_0_1 + ABtemp * (x_5_0_0 - CDcom * x_5_0_1);
    QUICKDouble x_23_0_3 = Ptempx * x_10_0_3 + WPtempx * x_10_0_4 + ABtemp * (x_5_0_3 - CDcom * x_5_0_4);
    QUICKDouble x_28_0_1 = Ptempx * x_11_0_1 + WPtempx * x_11_0_2 + 2.000000 * ABtemp * (x_4_0_1 - CDcom * x_4_0_2);
    QUICKDouble x_28_0_2 = Ptempx * x_11_0_2 + WPtempx * x_11_0_3 + 2.000000 * ABtemp * (x_4_0_2 - CDcom * x_4_0_3);
    QUICKDouble x_38_0_1 = Ptempx * x_23_0_1 + WPtempx * x_23_0_2 + 2.000000 * ABtemp * (x_10_0_1 - CDcom * x_10_0_2);
    QUICKDouble x_38_0_0 = Ptempx * x_23_0_0 + WPtempx * x_23_0_1 + 2.000000 * ABtemp * (x_10_0_0 - CDcom * x_10_0_1);
    QUICKDouble x_38_0_2 = Ptempx * x_23_0_2 + WPtempx * x_23_0_3 + 2.000000 * ABtemp * (x_10_0_2 - CDcom * x_10_0_3);
    QUICKDouble x_52_0_1 = Ptempx * x_28_0_1 + WPtempx * x_28_0_2 + 3.000000 * ABtemp * (x_11_0_1 - CDcom * x_11_0_2);
    QUICKDouble x_56_0_0 = Ptempx * x_38_0_0 + WPtempx * x_38_0_1 + 3.000000 * ABtemp * (x_23_0_0 - CDcom * x_23_0_1);
    QUICKDouble x_56_0_1 = Ptempx * x_38_0_1 + WPtempx * x_38_0_2 + 3.000000 * ABtemp * (x_23_0_1 - CDcom * x_23_0_2);
    LOCSTORE(store, 56, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_56_0_0 + WQtempz * x_56_0_1 + ABCDtemp * x_52_0_1;
    QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
    QUICKDouble x_6_0_2 = Ptempx * x_3_0_2 + WPtempx * x_3_0_3;
    QUICKDouble x_6_0_3 = Ptempx * x_3_0_3 + WPtempx * x_3_0_4;
    QUICKDouble x_6_0_4 = Ptempx * x_3_0_4 + WPtempx * x_3_0_5;
    QUICKDouble x_13_0_1 = Ptempx * x_6_0_1 + WPtempx * x_6_0_2 + ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
    QUICKDouble x_13_0_2 = Ptempx * x_6_0_2 + WPtempx * x_6_0_3 + ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
    QUICKDouble x_13_0_3 = Ptempx * x_6_0_3 + WPtempx * x_6_0_4 + ABtemp * (x_3_0_3 - CDcom * x_3_0_4);
    QUICKDouble x_26_0_1 = Ptempx * x_13_0_1 + WPtempx * x_13_0_2 + 2.000000 * ABtemp * (x_6_0_1 - CDcom * x_6_0_2);
    QUICKDouble x_26_0_2 = Ptempx * x_13_0_2 + WPtempx * x_13_0_3 + 2.000000 * ABtemp * (x_6_0_2 - CDcom * x_6_0_3);
    QUICKDouble x_50_0_1 = Ptempx * x_26_0_1 + WPtempx * x_26_0_2 + 3.000000 * ABtemp * (x_13_0_1 - CDcom * x_13_0_2);
    LOCSTORE(store, 56, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_56_0_0 + WQtempy * x_56_0_1 + ABCDtemp * x_50_0_1;
    LOCSTORE(store, 56, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_56_0_0 + WQtempx * x_56_0_1 + 4.000000 * ABCDtemp * x_38_0_1;
    QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
    QUICKDouble x_8_0_2 = Ptempy * x_2_0_2 + WPtempy * x_2_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
    QUICKDouble x_8_0_3 = Ptempy * x_2_0_3 + WPtempy * x_2_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
    QUICKDouble x_8_0_4 = Ptempy * x_2_0_4 + WPtempy * x_2_0_5 + ABtemp * (VY_4 - CDcom * VY_5);
    QUICKDouble x_15_0_1 = Ptempy * x_5_0_1 + WPtempy * x_5_0_2 + ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
    QUICKDouble x_15_0_2 = Ptempy * x_5_0_2 + WPtempy * x_5_0_3 + ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
    QUICKDouble x_15_0_3 = Ptempy * x_5_0_3 + WPtempy * x_5_0_4 + ABtemp * (x_3_0_3 - CDcom * x_3_0_4);
    QUICKDouble x_15_0_0 = Ptempy * x_5_0_0 + WPtempy * x_5_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
    QUICKDouble x_15_0_4 = Ptempy * x_5_0_4 + WPtempy * x_5_0_5 + ABtemp * (x_3_0_4 - CDcom * x_3_0_5);
    QUICKDouble x_18_0_1 = Ptempy * x_8_0_1 + WPtempy * x_8_0_2 + 2.000000 * ABtemp * (x_2_0_1 - CDcom * x_2_0_2);
    QUICKDouble x_18_0_2 = Ptempy * x_8_0_2 + WPtempy * x_8_0_3 + 2.000000 * ABtemp * (x_2_0_2 - CDcom * x_2_0_3);
    QUICKDouble x_18_0_3 = Ptempy * x_8_0_3 + WPtempy * x_8_0_4 + 2.000000 * ABtemp * (x_2_0_3 - CDcom * x_2_0_4);
    QUICKDouble x_30_0_1 = Ptempy * x_15_0_1 + WPtempy * x_15_0_2 + 2.000000 * ABtemp * (x_5_0_1 - CDcom * x_5_0_2);
    QUICKDouble x_30_0_2 = Ptempy * x_15_0_2 + WPtempy * x_15_0_3 + 2.000000 * ABtemp * (x_5_0_2 - CDcom * x_5_0_3);
    QUICKDouble x_30_0_0 = Ptempy * x_15_0_0 + WPtempy * x_15_0_1 + 2.000000 * ABtemp * (x_5_0_0 - CDcom * x_5_0_1);
    QUICKDouble x_30_0_3 = Ptempy * x_15_0_3 + WPtempy * x_15_0_4 + 2.000000 * ABtemp * (x_5_0_3 - CDcom * x_5_0_4);
    QUICKDouble x_33_0_1 = Ptempy * x_18_0_1 + WPtempy * x_18_0_2 + 3.000000 * ABtemp * (x_8_0_1 - CDcom * x_8_0_2);
    QUICKDouble x_33_0_2 = Ptempy * x_18_0_2 + WPtempy * x_18_0_3 + 3.000000 * ABtemp * (x_8_0_2 - CDcom * x_8_0_3);
    QUICKDouble x_48_0_1 = Ptempy * x_30_0_1 + WPtempy * x_30_0_2 + 3.000000 * ABtemp * (x_15_0_1 - CDcom * x_15_0_2);
    QUICKDouble x_48_0_0 = Ptempy * x_30_0_0 + WPtempy * x_30_0_1 + 3.000000 * ABtemp * (x_15_0_0 - CDcom * x_15_0_1);
    QUICKDouble x_48_0_2 = Ptempy * x_30_0_2 + WPtempy * x_30_0_3 + 3.000000 * ABtemp * (x_15_0_2 - CDcom * x_15_0_3);
    QUICKDouble x_51_0_1 = Ptempx * x_33_0_1 + WPtempx * x_33_0_2;
    QUICKDouble x_57_0_0 = Ptempx * x_48_0_0 + WPtempx * x_48_0_1;
    QUICKDouble x_57_0_1 = Ptempx * x_48_0_1 + WPtempx * x_48_0_2;
    LOCSTORE(store, 57, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_57_0_0 + WQtempz * x_57_0_1 + ABCDtemp * x_51_0_1;
    QUICKDouble x_39_0_1 = Ptempx * x_30_0_1 + WPtempx * x_30_0_2;
    LOCSTORE(store, 57, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_57_0_0 + WQtempy * x_57_0_1 + 4.000000 * ABCDtemp * x_39_0_1;
    LOCSTORE(store, 57, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_57_0_0 + WQtempx * x_57_0_1 + ABCDtemp * x_48_0_1;
    QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
    QUICKDouble x_9_0_2 = Ptempz * x_3_0_2 + WPtempz * x_3_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
    QUICKDouble x_9_0_3 = Ptempz * x_3_0_3 + WPtempz * x_3_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
    QUICKDouble x_9_0_4 = Ptempz * x_3_0_4 + WPtempz * x_3_0_5 + ABtemp * (VY_4 - CDcom * VY_5);
    QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
    QUICKDouble x_9_0_5 = Ptempz * x_3_0_5 + WPtempz * x_3_0_6 + ABtemp * (VY_5 - CDcom * VY_6);
    QUICKDouble x_19_0_1 = Ptempz * x_9_0_1 + WPtempz * x_9_0_2 + 2.000000 * ABtemp * (x_3_0_1 - CDcom * x_3_0_2);
    QUICKDouble x_19_0_2 = Ptempz * x_9_0_2 + WPtempz * x_9_0_3 + 2.000000 * ABtemp * (x_3_0_2 - CDcom * x_3_0_3);
    QUICKDouble x_19_0_3 = Ptempz * x_9_0_3 + WPtempz * x_9_0_4 + 2.000000 * ABtemp * (x_3_0_3 - CDcom * x_3_0_4);
    QUICKDouble x_19_0_0 = Ptempz * x_9_0_0 + WPtempz * x_9_0_1 + 2.000000 * ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
    QUICKDouble x_19_0_4 = Ptempz * x_9_0_4 + WPtempz * x_9_0_5 + 2.000000 * ABtemp * (x_3_0_4 - CDcom * x_3_0_5);
    QUICKDouble x_34_0_1 = Ptempz * x_19_0_1 + WPtempz * x_19_0_2 + 3.000000 * ABtemp * (x_9_0_1 - CDcom * x_9_0_2);
    QUICKDouble x_34_0_2 = Ptempz * x_19_0_2 + WPtempz * x_19_0_3 + 3.000000 * ABtemp * (x_9_0_2 - CDcom * x_9_0_3);
    QUICKDouble x_34_0_0 = Ptempz * x_19_0_0 + WPtempz * x_19_0_1 + 3.000000 * ABtemp * (x_9_0_0 - CDcom * x_9_0_1);
    QUICKDouble x_34_0_3 = Ptempz * x_19_0_3 + WPtempz * x_19_0_4 + 3.000000 * ABtemp * (x_9_0_3 - CDcom * x_9_0_4);
    QUICKDouble x_31_0_1 = Ptempy * x_19_0_1 + WPtempy * x_19_0_2;
    QUICKDouble x_31_0_2 = Ptempy * x_19_0_2 + WPtempy * x_19_0_3;
    QUICKDouble x_47_0_1 = Ptempy * x_34_0_1 + WPtempy * x_34_0_2;
    QUICKDouble x_47_0_0 = Ptempy * x_34_0_0 + WPtempy * x_34_0_1;
    QUICKDouble x_47_0_2 = Ptempy * x_34_0_2 + WPtempy * x_34_0_3;
    QUICKDouble x_40_0_1 = Ptempx * x_31_0_1 + WPtempx * x_31_0_2;
    QUICKDouble x_58_0_0 = Ptempx * x_47_0_0 + WPtempx * x_47_0_1;
    QUICKDouble x_58_0_1 = Ptempx * x_47_0_1 + WPtempx * x_47_0_2;
    LOCSTORE(store, 58, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_58_0_0 + WQtempz * x_58_0_1 + 4.000000 * ABCDtemp * x_40_0_1;
    QUICKDouble x_49_0_1 = Ptempx * x_34_0_1 + WPtempx * x_34_0_2;
    LOCSTORE(store, 58, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_58_0_0 + WQtempy * x_58_0_1 + ABCDtemp * x_49_0_1;
    LOCSTORE(store, 58, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_58_0_0 + WQtempx * x_58_0_1 + ABCDtemp * x_47_0_1;
    QUICKDouble x_16_0_1 = Ptempy * x_9_0_1 + WPtempy * x_9_0_2;
    QUICKDouble x_16_0_2 = Ptempy * x_9_0_2 + WPtempy * x_9_0_3;
    QUICKDouble x_16_0_3 = Ptempy * x_9_0_3 + WPtempy * x_9_0_4;
    QUICKDouble x_31_0_0 = Ptempy * x_19_0_0 + WPtempy * x_19_0_1;
    QUICKDouble x_31_0_3 = Ptempy * x_19_0_3 + WPtempy * x_19_0_4;
    QUICKDouble x_22_0_1 = Ptempy * x_16_0_1 + WPtempy * x_16_0_2 + ABtemp * (x_9_0_1 - CDcom * x_9_0_2);
    QUICKDouble x_22_0_2 = Ptempy * x_16_0_2 + WPtempy * x_16_0_3 + ABtemp * (x_9_0_2 - CDcom * x_9_0_3);
    QUICKDouble x_41_0_1 = Ptempy * x_31_0_1 + WPtempy * x_31_0_2 + ABtemp * (x_19_0_1 - CDcom * x_19_0_2);
    QUICKDouble x_41_0_0 = Ptempy * x_31_0_0 + WPtempy * x_31_0_1 + ABtemp * (x_19_0_0 - CDcom * x_19_0_1);
    QUICKDouble x_41_0_2 = Ptempy * x_31_0_2 + WPtempy * x_31_0_3 + ABtemp * (x_19_0_2 - CDcom * x_19_0_3);
    QUICKDouble x_35_0_1 = Ptempx * x_22_0_1 + WPtempx * x_22_0_2;
    QUICKDouble x_59_0_0 = Ptempx * x_41_0_0 + WPtempx * x_41_0_1;
    QUICKDouble x_59_0_1 = Ptempx * x_41_0_1 + WPtempx * x_41_0_2;
    LOCSTORE(store, 59, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_59_0_0 + WQtempz * x_59_0_1 + 3.000000 * ABCDtemp * x_35_0_1;
    LOCSTORE(store, 59, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_59_0_0 + WQtempy * x_59_0_1 + 2.000000 * ABCDtemp * x_40_0_1;
    LOCSTORE(store, 59, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_59_0_0 + WQtempx * x_59_0_1 + ABCDtemp * x_41_0_1;
    QUICKDouble x_16_0_0 = Ptempy * x_9_0_0 + WPtempy * x_9_0_1;
    QUICKDouble x_16_0_4 = Ptempy * x_9_0_4 + WPtempy * x_9_0_5;
    QUICKDouble x_22_0_0 = Ptempy * x_16_0_0 + WPtempy * x_16_0_1 + ABtemp * (x_9_0_0 - CDcom * x_9_0_1);
    QUICKDouble x_22_0_3 = Ptempy * x_16_0_3 + WPtempy * x_16_0_4 + ABtemp * (x_9_0_3 - CDcom * x_9_0_4);
    QUICKDouble x_42_0_1 = Ptempy * x_22_0_1 + WPtempy * x_22_0_2 + 2.000000 * ABtemp * (x_16_0_1 - CDcom * x_16_0_2);
    QUICKDouble x_42_0_0 = Ptempy * x_22_0_0 + WPtempy * x_22_0_1 + 2.000000 * ABtemp * (x_16_0_0 - CDcom * x_16_0_1);
    QUICKDouble x_42_0_2 = Ptempy * x_22_0_2 + WPtempy * x_22_0_3 + 2.000000 * ABtemp * (x_16_0_2 - CDcom * x_16_0_3);
    QUICKDouble x_60_0_0 = Ptempx * x_42_0_0 + WPtempx * x_42_0_1;
    QUICKDouble x_60_0_1 = Ptempx * x_42_0_1 + WPtempx * x_42_0_2;
    LOCSTORE(store, 60, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_60_0_0 + WQtempz * x_60_0_1 + 2.000000 * ABCDtemp * x_39_0_1;
    LOCSTORE(store, 60, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_60_0_0 + WQtempy * x_60_0_1 + 3.000000 * ABCDtemp * x_35_0_1;
    LOCSTORE(store, 60, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_60_0_0 + WQtempx * x_60_0_1 + ABCDtemp * x_42_0_1;
    QUICKDouble x_25_0_1 = Ptempx * x_16_0_1 + WPtempx * x_16_0_2;
    QUICKDouble x_25_0_2 = Ptempx * x_16_0_2 + WPtempx * x_16_0_3;
    QUICKDouble x_40_0_0 = Ptempx * x_31_0_0 + WPtempx * x_31_0_1;
    QUICKDouble x_40_0_2 = Ptempx * x_31_0_2 + WPtempx * x_31_0_3;
    QUICKDouble x_36_0_1 = Ptempx * x_25_0_1 + WPtempx * x_25_0_2 + ABtemp * (x_16_0_1 - CDcom * x_16_0_2);
    QUICKDouble x_61_0_0 = Ptempx * x_40_0_0 + WPtempx * x_40_0_1 + ABtemp * (x_31_0_0 - CDcom * x_31_0_1);
    QUICKDouble x_61_0_1 = Ptempx * x_40_0_1 + WPtempx * x_40_0_2 + ABtemp * (x_31_0_1 - CDcom * x_31_0_2);
    LOCSTORE(store, 61, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_61_0_0 + WQtempz * x_61_0_1 + 3.000000 * ABCDtemp * x_36_0_1;
    QUICKDouble x_27_0_1 = Ptempx * x_19_0_1 + WPtempx * x_19_0_2;
    QUICKDouble x_27_0_2 = Ptempx * x_19_0_2 + WPtempx * x_19_0_3;
    QUICKDouble x_43_0_1 = Ptempx * x_27_0_1 + WPtempx * x_27_0_2 + ABtemp * (x_19_0_1 - CDcom * x_19_0_2);
    LOCSTORE(store, 61, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_61_0_0 + WQtempy * x_61_0_1 + ABCDtemp * x_43_0_1;
    LOCSTORE(store, 61, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_61_0_0 + WQtempx * x_61_0_1 + 2.000000 * ABCDtemp * x_40_0_1;
    QUICKDouble x_25_0_0 = Ptempx * x_16_0_0 + WPtempx * x_16_0_1;
    QUICKDouble x_25_0_3 = Ptempx * x_16_0_3 + WPtempx * x_16_0_4;
    QUICKDouble x_36_0_0 = Ptempx * x_25_0_0 + WPtempx * x_25_0_1 + ABtemp * (x_16_0_0 - CDcom * x_16_0_1);
    QUICKDouble x_36_0_2 = Ptempx * x_25_0_2 + WPtempx * x_25_0_3 + ABtemp * (x_16_0_2 - CDcom * x_16_0_3);
    QUICKDouble x_62_0_0 = Ptempx * x_36_0_0 + WPtempx * x_36_0_1 + 2.000000 * ABtemp * (x_25_0_0 - CDcom * x_25_0_1);
    QUICKDouble x_62_0_1 = Ptempx * x_36_0_1 + WPtempx * x_36_0_2 + 2.000000 * ABtemp * (x_25_0_1 - CDcom * x_25_0_2);
    LOCSTORE(store, 62, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_62_0_0 + WQtempz * x_62_0_1 + 2.000000 * ABCDtemp * x_38_0_1;
    QUICKDouble x_14_0_1 = Ptempx * x_9_0_1 + WPtempx * x_9_0_2;
    QUICKDouble x_14_0_2 = Ptempx * x_9_0_2 + WPtempx * x_9_0_3;
    QUICKDouble x_14_0_3 = Ptempx * x_9_0_3 + WPtempx * x_9_0_4;
    QUICKDouble x_21_0_1 = Ptempx * x_14_0_1 + WPtempx * x_14_0_2 + ABtemp * (x_9_0_1 - CDcom * x_9_0_2);
    QUICKDouble x_21_0_2 = Ptempx * x_14_0_2 + WPtempx * x_14_0_3 + ABtemp * (x_9_0_2 - CDcom * x_9_0_3);
    QUICKDouble x_44_0_1 = Ptempx * x_21_0_1 + WPtempx * x_21_0_2 + 2.000000 * ABtemp * (x_14_0_1 - CDcom * x_14_0_2);
    LOCSTORE(store, 62, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_62_0_0 + WQtempy * x_62_0_1 + ABCDtemp * x_44_0_1;
    LOCSTORE(store, 62, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_62_0_0 + WQtempx * x_62_0_1 + 3.000000 * ABCDtemp * x_36_0_1;
    QUICKDouble x_29_0_1 = Ptempx * x_18_0_1 + WPtempx * x_18_0_2;
    QUICKDouble x_29_0_2 = Ptempx * x_18_0_2 + WPtempx * x_18_0_3;
    QUICKDouble x_39_0_0 = Ptempx * x_30_0_0 + WPtempx * x_30_0_1;
    QUICKDouble x_39_0_2 = Ptempx * x_30_0_2 + WPtempx * x_30_0_3;
    QUICKDouble x_45_0_1 = Ptempx * x_29_0_1 + WPtempx * x_29_0_2 + ABtemp * (x_18_0_1 - CDcom * x_18_0_2);
    QUICKDouble x_63_0_0 = Ptempx * x_39_0_0 + WPtempx * x_39_0_1 + ABtemp * (x_30_0_0 - CDcom * x_30_0_1);
    QUICKDouble x_63_0_1 = Ptempx * x_39_0_1 + WPtempx * x_39_0_2 + ABtemp * (x_30_0_1 - CDcom * x_30_0_2);
    LOCSTORE(store, 63, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_63_0_0 + WQtempz * x_63_0_1 + ABCDtemp * x_45_0_1;
    QUICKDouble x_24_0_1 = Ptempx * x_15_0_1 + WPtempx * x_15_0_2;
    QUICKDouble x_24_0_2 = Ptempx * x_15_0_2 + WPtempx * x_15_0_3;
    QUICKDouble x_37_0_1 = Ptempx * x_24_0_1 + WPtempx * x_24_0_2 + ABtemp * (x_15_0_1 - CDcom * x_15_0_2);
    LOCSTORE(store, 63, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_63_0_0 + WQtempy * x_63_0_1 + 3.000000 * ABCDtemp * x_37_0_1;
    LOCSTORE(store, 63, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_63_0_0 + WQtempx * x_63_0_1 + 2.000000 * ABCDtemp * x_39_0_1;
    QUICKDouble x_12_0_1 = Ptempx * x_8_0_1 + WPtempx * x_8_0_2;
    QUICKDouble x_12_0_2 = Ptempx * x_8_0_2 + WPtempx * x_8_0_3;
    QUICKDouble x_12_0_3 = Ptempx * x_8_0_3 + WPtempx * x_8_0_4;
    QUICKDouble x_24_0_0 = Ptempx * x_15_0_0 + WPtempx * x_15_0_1;
    QUICKDouble x_24_0_3 = Ptempx * x_15_0_3 + WPtempx * x_15_0_4;
    QUICKDouble x_20_0_1 = Ptempx * x_12_0_1 + WPtempx * x_12_0_2 + ABtemp * (x_8_0_1 - CDcom * x_8_0_2);
    QUICKDouble x_20_0_2 = Ptempx * x_12_0_2 + WPtempx * x_12_0_3 + ABtemp * (x_8_0_2 - CDcom * x_8_0_3);
    QUICKDouble x_37_0_0 = Ptempx * x_24_0_0 + WPtempx * x_24_0_1 + ABtemp * (x_15_0_0 - CDcom * x_15_0_1);
    QUICKDouble x_37_0_2 = Ptempx * x_24_0_2 + WPtempx * x_24_0_3 + ABtemp * (x_15_0_2 - CDcom * x_15_0_3);
    QUICKDouble x_46_0_1 = Ptempx * x_20_0_1 + WPtempx * x_20_0_2 + 2.000000 * ABtemp * (x_12_0_1 - CDcom * x_12_0_2);
    QUICKDouble x_64_0_0 = Ptempx * x_37_0_0 + WPtempx * x_37_0_1 + 2.000000 * ABtemp * (x_24_0_0 - CDcom * x_24_0_1);
    QUICKDouble x_64_0_1 = Ptempx * x_37_0_1 + WPtempx * x_37_0_2 + 2.000000 * ABtemp * (x_24_0_1 - CDcom * x_24_0_2);
    LOCSTORE(store, 64, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_64_0_0 + WQtempz * x_64_0_1 + ABCDtemp * x_46_0_1;
    LOCSTORE(store, 64, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_64_0_0 + WQtempy * x_64_0_1 + 2.000000 * ABCDtemp * x_38_0_1;
    LOCSTORE(store, 64, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_64_0_0 + WQtempx * x_64_0_1 + 3.000000 * ABCDtemp * x_37_0_1;
    QUICKDouble x_35_0_0 = Ptempx * x_22_0_0 + WPtempx * x_22_0_1;
    QUICKDouble x_35_0_2 = Ptempx * x_22_0_2 + WPtempx * x_22_0_3;
    QUICKDouble x_65_0_0 = Ptempx * x_35_0_0 + WPtempx * x_35_0_1 + ABtemp * (x_22_0_0 - CDcom * x_22_0_1);
    QUICKDouble x_65_0_1 = Ptempx * x_35_0_1 + WPtempx * x_35_0_2 + ABtemp * (x_22_0_1 - CDcom * x_22_0_2);
    LOCSTORE(store, 65, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_65_0_0 + WQtempz * x_65_0_1 + 2.000000 * ABCDtemp * x_37_0_1;
    LOCSTORE(store, 65, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_65_0_0 + WQtempy * x_65_0_1 + 2.000000 * ABCDtemp * x_36_0_1;
    LOCSTORE(store, 65, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_65_0_0 + WQtempx * x_65_0_1 + 2.000000 * ABCDtemp * x_35_0_1;
    QUICKDouble x_55_0_1 = Ptempz * x_34_0_1 + WPtempz * x_34_0_2 + 4.000000 * ABtemp * (x_19_0_1 - CDcom * x_19_0_2);
    QUICKDouble x_55_0_0 = Ptempz * x_34_0_0 + WPtempz * x_34_0_1 + 4.000000 * ABtemp * (x_19_0_0 - CDcom * x_19_0_1);
    QUICKDouble x_55_0_2 = Ptempz * x_34_0_2 + WPtempz * x_34_0_3 + 4.000000 * ABtemp * (x_19_0_2 - CDcom * x_19_0_3);
    QUICKDouble x_66_0_0 = Ptempy * x_55_0_0 + WPtempy * x_55_0_1;
    QUICKDouble x_66_0_1 = Ptempy * x_55_0_1 + WPtempy * x_55_0_2;
    LOCSTORE(store, 66, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_66_0_0 + WQtempz * x_66_0_1 + 5.000000 * ABCDtemp * x_47_0_1;
    LOCSTORE(store, 66, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_66_0_0 + WQtempy * x_66_0_1 + ABCDtemp * x_55_0_1;
    LOCSTORE(store, 66, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_66_0_0 + WQtempx * x_66_0_1;
    QUICKDouble x_54_0_1 = Ptempy * x_33_0_1 + WPtempy * x_33_0_2 + 4.000000 * ABtemp * (x_18_0_1 - CDcom * x_18_0_2);
    QUICKDouble x_67_0_0 = Ptempy * x_48_0_0 + WPtempy * x_48_0_1 + 4.000000 * ABtemp * (x_30_0_0 - CDcom * x_30_0_1);
    QUICKDouble x_67_0_1 = Ptempy * x_48_0_1 + WPtempy * x_48_0_2 + 4.000000 * ABtemp * (x_30_0_1 - CDcom * x_30_0_2);
    LOCSTORE(store, 67, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_67_0_0 + WQtempz * x_67_0_1 + ABCDtemp * x_54_0_1;
    LOCSTORE(store, 67, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_67_0_0 + WQtempy * x_67_0_1 + 5.000000 * ABCDtemp * x_48_0_1;
    LOCSTORE(store, 67, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_67_0_0 + WQtempx * x_67_0_1;
    QUICKDouble x_68_0_0 = Ptempx * x_55_0_0 + WPtempx * x_55_0_1;
    QUICKDouble x_68_0_1 = Ptempx * x_55_0_1 + WPtempx * x_55_0_2;
    LOCSTORE(store, 68, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_68_0_0 + WQtempz * x_68_0_1 + 5.000000 * ABCDtemp * x_49_0_1;
    LOCSTORE(store, 68, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_68_0_0 + WQtempy * x_68_0_1;
    LOCSTORE(store, 68, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_68_0_0 + WQtempx * x_68_0_1 + ABCDtemp * x_55_0_1;
    QUICKDouble x_1_0_1 = Ptempx * VY_1 + WPtempx * VY_2;
    QUICKDouble x_1_0_2 = Ptempx * VY_2 + WPtempx * VY_3;
    QUICKDouble x_1_0_3 = Ptempx * VY_3 + WPtempx * VY_4;
    QUICKDouble x_1_0_4 = Ptempx * VY_4 + WPtempx * VY_5;
    QUICKDouble x_1_0_5 = Ptempx * VY_5 + WPtempx * VY_6;
    QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
    QUICKDouble x_6_0_5 = Ptempx * x_3_0_5 + WPtempx * x_3_0_6;
    QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2 + ABtemp * (VY_1 - CDcom * VY_2);
    QUICKDouble x_7_0_2 = Ptempx * x_1_0_2 + WPtempx * x_1_0_3 + ABtemp * (VY_2 - CDcom * VY_3);
    QUICKDouble x_7_0_3 = Ptempx * x_1_0_3 + WPtempx * x_1_0_4 + ABtemp * (VY_3 - CDcom * VY_4);
    QUICKDouble x_7_0_4 = Ptempx * x_1_0_4 + WPtempx * x_1_0_5 + ABtemp * (VY_4 - CDcom * VY_5);
    QUICKDouble x_13_0_0 = Ptempx * x_6_0_0 + WPtempx * x_6_0_1 + ABtemp * (x_3_0_0 - CDcom * x_3_0_1);
    QUICKDouble x_13_0_4 = Ptempx * x_6_0_4 + WPtempx * x_6_0_5 + ABtemp * (x_3_0_4 - CDcom * x_3_0_5);
    QUICKDouble x_17_0_1 = Ptempx * x_7_0_1 + WPtempx * x_7_0_2 + 2.000000 * ABtemp * (x_1_0_1 - CDcom * x_1_0_2);
    QUICKDouble x_17_0_2 = Ptempx * x_7_0_2 + WPtempx * x_7_0_3 + 2.000000 * ABtemp * (x_1_0_2 - CDcom * x_1_0_3);
    QUICKDouble x_17_0_3 = Ptempx * x_7_0_3 + WPtempx * x_7_0_4 + 2.000000 * ABtemp * (x_1_0_3 - CDcom * x_1_0_4);
    QUICKDouble x_26_0_0 = Ptempx * x_13_0_0 + WPtempx * x_13_0_1 + 2.000000 * ABtemp * (x_6_0_0 - CDcom * x_6_0_1);
    QUICKDouble x_26_0_3 = Ptempx * x_13_0_3 + WPtempx * x_13_0_4 + 2.000000 * ABtemp * (x_6_0_3 - CDcom * x_6_0_4);
    QUICKDouble x_32_0_1 = Ptempx * x_17_0_1 + WPtempx * x_17_0_2 + 3.000000 * ABtemp * (x_7_0_1 - CDcom * x_7_0_2);
    QUICKDouble x_32_0_2 = Ptempx * x_17_0_2 + WPtempx * x_17_0_3 + 3.000000 * ABtemp * (x_7_0_2 - CDcom * x_7_0_3);
    QUICKDouble x_50_0_0 = Ptempx * x_26_0_0 + WPtempx * x_26_0_1 + 3.000000 * ABtemp * (x_13_0_0 - CDcom * x_13_0_1);
    QUICKDouble x_50_0_2 = Ptempx * x_26_0_2 + WPtempx * x_26_0_3 + 3.000000 * ABtemp * (x_13_0_2 - CDcom * x_13_0_3);
    QUICKDouble x_53_0_1 = Ptempx * x_32_0_1 + WPtempx * x_32_0_2 + 4.000000 * ABtemp * (x_17_0_1 - CDcom * x_17_0_2);
    QUICKDouble x_69_0_0 = Ptempx * x_50_0_0 + WPtempx * x_50_0_1 + 4.000000 * ABtemp * (x_26_0_0 - CDcom * x_26_0_1);
    QUICKDouble x_69_0_1 = Ptempx * x_50_0_1 + WPtempx * x_50_0_2 + 4.000000 * ABtemp * (x_26_0_1 - CDcom * x_26_0_2);
    LOCSTORE(store, 69, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_69_0_0 + WQtempz * x_69_0_1 + ABCDtemp * x_53_0_1;
    LOCSTORE(store, 69, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_69_0_0 + WQtempy * x_69_0_1;
    LOCSTORE(store, 69, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_69_0_0 + WQtempx * x_69_0_1 + 5.000000 * ABCDtemp * x_50_0_1;
    QUICKDouble x_2_0_0 = Ptempy * VY_0 + WPtempy * VY_1;
    QUICKDouble x_2_0_6 = Ptempy * VY_6 + WPtempy * VY_7;
    QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
    QUICKDouble x_8_0_5 = Ptempy * x_2_0_5 + WPtempy * x_2_0_6 + ABtemp * (VY_5 - CDcom * VY_6);
    QUICKDouble x_18_0_0 = Ptempy * x_8_0_0 + WPtempy * x_8_0_1 + 2.000000 * ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
    QUICKDouble x_18_0_4 = Ptempy * x_8_0_4 + WPtempy * x_8_0_5 + 2.000000 * ABtemp * (x_2_0_4 - CDcom * x_2_0_5);
    QUICKDouble x_33_0_0 = Ptempy * x_18_0_0 + WPtempy * x_18_0_1 + 3.000000 * ABtemp * (x_8_0_0 - CDcom * x_8_0_1);
    QUICKDouble x_33_0_3 = Ptempy * x_18_0_3 + WPtempy * x_18_0_4 + 3.000000 * ABtemp * (x_8_0_3 - CDcom * x_8_0_4);
    QUICKDouble x_54_0_0 = Ptempy * x_33_0_0 + WPtempy * x_33_0_1 + 4.000000 * ABtemp * (x_18_0_0 - CDcom * x_18_0_1);
    QUICKDouble x_54_0_2 = Ptempy * x_33_0_2 + WPtempy * x_33_0_3 + 4.000000 * ABtemp * (x_18_0_2 - CDcom * x_18_0_3);
    QUICKDouble x_70_0_0 = Ptempx * x_54_0_0 + WPtempx * x_54_0_1;
    QUICKDouble x_70_0_1 = Ptempx * x_54_0_1 + WPtempx * x_54_0_2;
    LOCSTORE(store, 70, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_70_0_0 + WQtempz * x_70_0_1;
    LOCSTORE(store, 70, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_70_0_0 + WQtempy * x_70_0_1 + 5.000000 * ABCDtemp * x_51_0_1;
    LOCSTORE(store, 70, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_70_0_0 + WQtempx * x_70_0_1 + ABCDtemp * x_54_0_1;
    QUICKDouble x_4_0_0 = Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
    QUICKDouble x_4_0_5 = Ptempx * x_2_0_5 + WPtempx * x_2_0_6;
    QUICKDouble x_11_0_0 = Ptempx * x_4_0_0 + WPtempx * x_4_0_1 + ABtemp * (x_2_0_0 - CDcom * x_2_0_1);
    QUICKDouble x_11_0_4 = Ptempx * x_4_0_4 + WPtempx * x_4_0_5 + ABtemp * (x_2_0_4 - CDcom * x_2_0_5);
    QUICKDouble x_28_0_0 = Ptempx * x_11_0_0 + WPtempx * x_11_0_1 + 2.000000 * ABtemp * (x_4_0_0 - CDcom * x_4_0_1);
    QUICKDouble x_28_0_3 = Ptempx * x_11_0_3 + WPtempx * x_11_0_4 + 2.000000 * ABtemp * (x_4_0_3 - CDcom * x_4_0_4);
    QUICKDouble x_52_0_0 = Ptempx * x_28_0_0 + WPtempx * x_28_0_1 + 3.000000 * ABtemp * (x_11_0_0 - CDcom * x_11_0_1);
    QUICKDouble x_52_0_2 = Ptempx * x_28_0_2 + WPtempx * x_28_0_3 + 3.000000 * ABtemp * (x_11_0_2 - CDcom * x_11_0_3);
    QUICKDouble x_71_0_0 = Ptempx * x_52_0_0 + WPtempx * x_52_0_1 + 4.000000 * ABtemp * (x_28_0_0 - CDcom * x_28_0_1);
    QUICKDouble x_71_0_1 = Ptempx * x_52_0_1 + WPtempx * x_52_0_2 + 4.000000 * ABtemp * (x_28_0_1 - CDcom * x_28_0_2);
    LOCSTORE(store, 71, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_71_0_0 + WQtempz * x_71_0_1;
    LOCSTORE(store, 71, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_71_0_0 + WQtempy * x_71_0_1 + ABCDtemp * x_53_0_1;
    LOCSTORE(store, 71, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_71_0_0 + WQtempx * x_71_0_1 + 5.000000 * ABCDtemp * x_52_0_1;
    QUICKDouble x_72_0_0 = Ptempy * x_47_0_0 + WPtempy * x_47_0_1 + ABtemp * (x_34_0_0 - CDcom * x_34_0_1);
    QUICKDouble x_72_0_1 = Ptempy * x_47_0_1 + WPtempy * x_47_0_2 + ABtemp * (x_34_0_1 - CDcom * x_34_0_2);
    LOCSTORE(store, 72, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_72_0_0 + WQtempz * x_72_0_1 + 4.000000 * ABCDtemp * x_41_0_1;
    LOCSTORE(store, 72, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_72_0_0 + WQtempy * x_72_0_1 + 2.000000 * ABCDtemp * x_47_0_1;
    LOCSTORE(store, 72, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_72_0_0 + WQtempx * x_72_0_1;
    QUICKDouble x_73_0_0 = Ptempy * x_42_0_0 + WPtempy * x_42_0_1 + 3.000000 * ABtemp * (x_22_0_0 - CDcom * x_22_0_1);
    QUICKDouble x_73_0_1 = Ptempy * x_42_0_1 + WPtempy * x_42_0_2 + 3.000000 * ABtemp * (x_22_0_1 - CDcom * x_22_0_2);
    LOCSTORE(store, 73, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_73_0_0 + WQtempz * x_73_0_1 + 2.000000 * ABCDtemp * x_48_0_1;
    LOCSTORE(store, 73, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_73_0_0 + WQtempy * x_73_0_1 + 4.000000 * ABCDtemp * x_42_0_1;
    LOCSTORE(store, 73, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_73_0_0 + WQtempx * x_73_0_1;
    QUICKDouble x_49_0_0 = Ptempx * x_34_0_0 + WPtempx * x_34_0_1;
    QUICKDouble x_49_0_2 = Ptempx * x_34_0_2 + WPtempx * x_34_0_3;
    QUICKDouble x_74_0_0 = Ptempx * x_49_0_0 + WPtempx * x_49_0_1 + ABtemp * (x_34_0_0 - CDcom * x_34_0_1);
    QUICKDouble x_74_0_1 = Ptempx * x_49_0_1 + WPtempx * x_49_0_2 + ABtemp * (x_34_0_1 - CDcom * x_34_0_2);
    LOCSTORE(store, 74, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_74_0_0 + WQtempz * x_74_0_1 + 4.000000 * ABCDtemp * x_43_0_1;
    LOCSTORE(store, 74, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_74_0_0 + WQtempy * x_74_0_1;
    LOCSTORE(store, 74, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_74_0_0 + WQtempx * x_74_0_1 + 2.000000 * ABCDtemp * x_49_0_1;
    QUICKDouble x_14_0_0 = Ptempx * x_9_0_0 + WPtempx * x_9_0_1;
    QUICKDouble x_14_0_4 = Ptempx * x_9_0_4 + WPtempx * x_9_0_5;
    QUICKDouble x_21_0_0 = Ptempx * x_14_0_0 + WPtempx * x_14_0_1 + ABtemp * (x_9_0_0 - CDcom * x_9_0_1);
    QUICKDouble x_21_0_3 = Ptempx * x_14_0_3 + WPtempx * x_14_0_4 + ABtemp * (x_9_0_3 - CDcom * x_9_0_4);
    QUICKDouble x_44_0_0 = Ptempx * x_21_0_0 + WPtempx * x_21_0_1 + 2.000000 * ABtemp * (x_14_0_0 - CDcom * x_14_0_1);
    QUICKDouble x_44_0_2 = Ptempx * x_21_0_2 + WPtempx * x_21_0_3 + 2.000000 * ABtemp * (x_14_0_2 - CDcom * x_14_0_3);
    QUICKDouble x_75_0_0 = Ptempx * x_44_0_0 + WPtempx * x_44_0_1 + 3.000000 * ABtemp * (x_21_0_0 - CDcom * x_21_0_1);
    QUICKDouble x_75_0_1 = Ptempx * x_44_0_1 + WPtempx * x_44_0_2 + 3.000000 * ABtemp * (x_21_0_1 - CDcom * x_21_0_2);
    LOCSTORE(store, 75, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_75_0_0 + WQtempz * x_75_0_1 + 2.000000 * ABCDtemp * x_50_0_1;
    LOCSTORE(store, 75, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_75_0_0 + WQtempy * x_75_0_1;
    LOCSTORE(store, 75, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_75_0_0 + WQtempx * x_75_0_1 + 4.000000 * ABCDtemp * x_44_0_1;
    QUICKDouble x_51_0_0 = Ptempx * x_33_0_0 + WPtempx * x_33_0_1;
    QUICKDouble x_51_0_2 = Ptempx * x_33_0_2 + WPtempx * x_33_0_3;
    QUICKDouble x_76_0_0 = Ptempx * x_51_0_0 + WPtempx * x_51_0_1 + ABtemp * (x_33_0_0 - CDcom * x_33_0_1);
    QUICKDouble x_76_0_1 = Ptempx * x_51_0_1 + WPtempx * x_51_0_2 + ABtemp * (x_33_0_1 - CDcom * x_33_0_2);
    LOCSTORE(store, 76, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_76_0_0 + WQtempz * x_76_0_1;
    LOCSTORE(store, 76, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_76_0_0 + WQtempy * x_76_0_1 + 4.000000 * ABCDtemp * x_45_0_1;
    LOCSTORE(store, 76, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_76_0_0 + WQtempx * x_76_0_1 + 2.000000 * ABCDtemp * x_51_0_1;
    QUICKDouble x_12_0_0 = Ptempx * x_8_0_0 + WPtempx * x_8_0_1;
    QUICKDouble x_12_0_4 = Ptempx * x_8_0_4 + WPtempx * x_8_0_5;
    QUICKDouble x_20_0_0 = Ptempx * x_12_0_0 + WPtempx * x_12_0_1 + ABtemp * (x_8_0_0 - CDcom * x_8_0_1);
    QUICKDouble x_20_0_3 = Ptempx * x_12_0_3 + WPtempx * x_12_0_4 + ABtemp * (x_8_0_3 - CDcom * x_8_0_4);
    QUICKDouble x_46_0_0 = Ptempx * x_20_0_0 + WPtempx * x_20_0_1 + 2.000000 * ABtemp * (x_12_0_0 - CDcom * x_12_0_1);
    QUICKDouble x_46_0_2 = Ptempx * x_20_0_2 + WPtempx * x_20_0_3 + 2.000000 * ABtemp * (x_12_0_2 - CDcom * x_12_0_3);
    QUICKDouble x_77_0_0 = Ptempx * x_46_0_0 + WPtempx * x_46_0_1 + 3.000000 * ABtemp * (x_20_0_0 - CDcom * x_20_0_1);
    QUICKDouble x_77_0_1 = Ptempx * x_46_0_1 + WPtempx * x_46_0_2 + 3.000000 * ABtemp * (x_20_0_1 - CDcom * x_20_0_2);
    LOCSTORE(store, 77, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_77_0_0 + WQtempz * x_77_0_1;
    LOCSTORE(store, 77, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_77_0_0 + WQtempy * x_77_0_1 + 2.000000 * ABCDtemp * x_52_0_1;
    LOCSTORE(store, 77, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_77_0_0 + WQtempx * x_77_0_1 + 4.000000 * ABCDtemp * x_46_0_1;
    QUICKDouble x_78_0_0 = Ptempy * x_41_0_0 + WPtempy * x_41_0_1 + 2.000000 * ABtemp * (x_31_0_0 - CDcom * x_31_0_1);
    QUICKDouble x_78_0_1 = Ptempy * x_41_0_1 + WPtempy * x_41_0_2 + 2.000000 * ABtemp * (x_31_0_1 - CDcom * x_31_0_2);
    LOCSTORE(store, 78, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_78_0_0 + WQtempz * x_78_0_1 + 3.000000 * ABCDtemp * x_42_0_1;
    LOCSTORE(store, 78, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_78_0_0 + WQtempy * x_78_0_1 + 3.000000 * ABCDtemp * x_41_0_1;
    LOCSTORE(store, 78, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_78_0_0 + WQtempx * x_78_0_1;
    QUICKDouble x_27_0_0 = Ptempx * x_19_0_0 + WPtempx * x_19_0_1;
    QUICKDouble x_27_0_3 = Ptempx * x_19_0_3 + WPtempx * x_19_0_4;
    QUICKDouble x_43_0_0 = Ptempx * x_27_0_0 + WPtempx * x_27_0_1 + ABtemp * (x_19_0_0 - CDcom * x_19_0_1);
    QUICKDouble x_43_0_2 = Ptempx * x_27_0_2 + WPtempx * x_27_0_3 + ABtemp * (x_19_0_2 - CDcom * x_19_0_3);
    QUICKDouble x_79_0_0 = Ptempx * x_43_0_0 + WPtempx * x_43_0_1 + 2.000000 * ABtemp * (x_27_0_0 - CDcom * x_27_0_1);
    QUICKDouble x_79_0_1 = Ptempx * x_43_0_1 + WPtempx * x_43_0_2 + 2.000000 * ABtemp * (x_27_0_1 - CDcom * x_27_0_2);
    LOCSTORE(store, 79, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_79_0_0 + WQtempz * x_79_0_1 + 3.000000 * ABCDtemp * x_44_0_1;
    LOCSTORE(store, 79, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_79_0_0 + WQtempy * x_79_0_1;
    LOCSTORE(store, 79, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_79_0_0 + WQtempx * x_79_0_1 + 3.000000 * ABCDtemp * x_43_0_1;
    QUICKDouble x_29_0_0 = Ptempx * x_18_0_0 + WPtempx * x_18_0_1;
    QUICKDouble x_29_0_3 = Ptempx * x_18_0_3 + WPtempx * x_18_0_4;
    QUICKDouble x_45_0_0 = Ptempx * x_29_0_0 + WPtempx * x_29_0_1 + ABtemp * (x_18_0_0 - CDcom * x_18_0_1);
    QUICKDouble x_45_0_2 = Ptempx * x_29_0_2 + WPtempx * x_29_0_3 + ABtemp * (x_18_0_2 - CDcom * x_18_0_3);
    QUICKDouble x_80_0_0 = Ptempx * x_45_0_0 + WPtempx * x_45_0_1 + 2.000000 * ABtemp * (x_29_0_0 - CDcom * x_29_0_1);
    QUICKDouble x_80_0_1 = Ptempx * x_45_0_1 + WPtempx * x_45_0_2 + 2.000000 * ABtemp * (x_29_0_1 - CDcom * x_29_0_2);
    LOCSTORE(store, 80, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_80_0_0 + WQtempz * x_80_0_1;
    LOCSTORE(store, 80, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_80_0_0 + WQtempy * x_80_0_1 + 3.000000 * ABCDtemp * x_46_0_1;
    LOCSTORE(store, 80, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_80_0_0 + WQtempx * x_80_0_1 + 3.000000 * ABCDtemp * x_45_0_1;
    QUICKDouble x_1_0_0 = Ptempx * VY_0 + WPtempx * VY_1;
    QUICKDouble x_1_0_6 = Ptempx * VY_6 + WPtempx * VY_7;
    QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1 + ABtemp * (VY_0 - CDcom * VY_1);
    QUICKDouble x_7_0_5 = Ptempx * x_1_0_5 + WPtempx * x_1_0_6 + ABtemp * (VY_5 - CDcom * VY_6);
    QUICKDouble x_17_0_0 = Ptempx * x_7_0_0 + WPtempx * x_7_0_1 + 2.000000 * ABtemp * (x_1_0_0 - CDcom * x_1_0_1);
    QUICKDouble x_17_0_4 = Ptempx * x_7_0_4 + WPtempx * x_7_0_5 + 2.000000 * ABtemp * (x_1_0_4 - CDcom * x_1_0_5);
    QUICKDouble x_32_0_0 = Ptempx * x_17_0_0 + WPtempx * x_17_0_1 + 3.000000 * ABtemp * (x_7_0_0 - CDcom * x_7_0_1);
    QUICKDouble x_32_0_3 = Ptempx * x_17_0_3 + WPtempx * x_17_0_4 + 3.000000 * ABtemp * (x_7_0_3 - CDcom * x_7_0_4);
    QUICKDouble x_53_0_0 = Ptempx * x_32_0_0 + WPtempx * x_32_0_1 + 4.000000 * ABtemp * (x_17_0_0 - CDcom * x_17_0_1);
    QUICKDouble x_53_0_2 = Ptempx * x_32_0_2 + WPtempx * x_32_0_3 + 4.000000 * ABtemp * (x_17_0_2 - CDcom * x_17_0_3);
    QUICKDouble x_81_0_0 = Ptempx * x_53_0_0 + WPtempx * x_53_0_1 + 5.000000 * ABtemp * (x_32_0_0 - CDcom * x_32_0_1);
    QUICKDouble x_81_0_1 = Ptempx * x_53_0_1 + WPtempx * x_53_0_2 + 5.000000 * ABtemp * (x_32_0_1 - CDcom * x_32_0_2);
    LOCSTORE(store, 81, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_81_0_0 + WQtempz * x_81_0_1;
    LOCSTORE(store, 81, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_81_0_0 + WQtempy * x_81_0_1;
    LOCSTORE(store, 81, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_81_0_0 + WQtempx * x_81_0_1 + 6.000000 * ABCDtemp * x_53_0_1;
    QUICKDouble x_82_0_0 = Ptempy * x_54_0_0 + WPtempy * x_54_0_1 + 5.000000 * ABtemp * (x_33_0_0 - CDcom * x_33_0_1);
    QUICKDouble x_82_0_1 = Ptempy * x_54_0_1 + WPtempy * x_54_0_2 + 5.000000 * ABtemp * (x_33_0_1 - CDcom * x_33_0_2);
    LOCSTORE(store, 82, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_82_0_0 + WQtempz * x_82_0_1;
    LOCSTORE(store, 82, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_82_0_0 + WQtempy * x_82_0_1 + 6.000000 * ABCDtemp * x_54_0_1;
    LOCSTORE(store, 82, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_82_0_0 + WQtempx * x_82_0_1;
    QUICKDouble x_83_0_0 = Ptempz * x_55_0_0 + WPtempz * x_55_0_1 + 5.000000 * ABtemp * (x_34_0_0 - CDcom * x_34_0_1);
    QUICKDouble x_83_0_1 = Ptempz * x_55_0_1 + WPtempz * x_55_0_2 + 5.000000 * ABtemp * (x_34_0_1 - CDcom * x_34_0_2);
    LOCSTORE(store, 83, 3, STOREDIM, STOREDIM) STORE_OPERATOR Qtempz * x_83_0_0 + WQtempz * x_83_0_1 + 6.000000 * ABCDtemp * x_55_0_1;
    LOCSTORE(store, 83, 2, STOREDIM, STOREDIM) STORE_OPERATOR Qtempy * x_83_0_0 + WQtempy * x_83_0_1;
    LOCSTORE(store, 83, 1, STOREDIM, STOREDIM) STORE_OPERATOR Qtempx * x_83_0_0 + WQtempx * x_83_0_1;
    // [IS|PS] integral - End 

}
