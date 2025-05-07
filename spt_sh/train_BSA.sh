#!/bin/bash

# List all filenames directly
for filename in \
    "BSA_200mM_kcl_BSA_0_2_0019_Ldp" \
    "BSA_200mM_kcl_BSA_0_2_0021_Ldp" \
    "BSA_200mM_kcl_bsa_qd_30_0014_Ldp" \
    "BSA_200mM_kcl_bsa_qd_30_res_Ldp" \
    "BSA_200mM_kcl_BSA_QD_resonant_0001_Ldp" \
    "BSA_200mM_kcl_BSA_QD_resonant_0004_Ldp" \
    "BSA_200mM_kcl_BSA_QD_resonant_0005_Ldp" \
    "BSA_200mM_kcl_U_BSA_U_QD_Ldp" \
    "BSA_400mM_kcl_ac_1_1_BSA_ac_1_1_400kcl_0003_Ldp" \
    "BSA_400mM_kcl_ac_1_1_BSA_ac_1_1_400kcl_0006_Ldp" \
    "BSA_400mM_kcl_BSA_dex_0_400kcl_0006_Ldp" \
    "BSA_400mM_kcl_BSA_dex_0_400kcl_0010_Ldp" \
    "BSA_400mM_kcl_BSA_dex_0_400kcl_0011_Ldp" \
    "BSA_400mM_kcl_BSA_dex_0_400kcl_0015_Ldp" \
    "BSA_400mM_kcl_BSA_dex_0_400kcl_0016_Ldp" \
    "BSA_400mM_kcl_bsa_qd_30_kcl_0006_Ldp" \
    "BSA_400mM_kcl_bsa_qd_30_kcl_0007_Ldp"; do
    
    echo "Running: $filename"
    nohup python main_spt_flow_diffL.py \
        --filename "$filename" \
        --epochs 50 \
        --beta_diffL 1.0 \
        --beta_photo 1.0 \
        --is_spt_pt \
        --is_train \
        --sample_rate 1.0 \
        > "output.log" 2>&1
    
    echo "Completed: $filename"
done