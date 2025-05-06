#!/bin/bash

# List all filenames directly
for filename in \
    "BSA_400mM_kcl_5mg_F_BSA_KCl_400mM_F_res_0005_Ldp" \
    "BSA_400mM_kcl_5mg_F_BSA_KCl_400mM_F_res_0004_Ldp" \
    "BSA_400mM_kcl_5mg_F_BSA_KCl_400mM_F_res_0003_Ldp" \
    "BSA_400mM_kcl_5mg_F_BSA_KCl_400mM_F_res_0002_Ldp" \
    "BSA_400mM_kcl_5mg_F_BSA_KCl_400mM_F_res_0001_Ldp" \
    "BSA_400mM_kcl_5e-1mg_HA_BSA_KCl_400mM_HA_res_0002_Ldp" \
    "BSA_200mM_kcl_5mg_F_BSA_KCl_200mM_F_res_Ldp" \
    "BSA_200mM_kcl_5mg_F_BSA_KCl_200mM_F_res_0006_Ldp" \
    "BSA_200mM_kcl_5mg_F_BSA_KCl_200mM_F_res_0005_Ldp" \
    "BSA_200mM_kcl_5mg_F_BSA_KCl_200mM_F_res_0004_Ldp" \
    "BSA_200mM_kcl_5mg_F_BSA_KCl_200mM_F_res_0003_Ldp" \
    "BSA_200mM_kcl_5mg_F_BSA_KCl_200mM_F_res_0002_Ldp" \
    "BSA_200mM_kcl_5mg_F_BSA_KCl_200mM_F_res_0001_Ldp" \
    "BSA_50mM_kcl_100mM_NaAc_BSA_NaAc_50_mM_res02_Ldp" \
    "BSA_50mM_kcl_100mM_NaAc_BSA_NaAc_50_mM_res_Ldp" \
    "BSA_50mM_kcl_100mM_NaAc_BSA_NaAc_50_mM_res_0002_Ldp" \
    "BSA_50mM_kcl_100mM_NaAc_BSA_NaAc_50_mM_res_0001_Ldp" \
    "BSA_0mM_kcl_100mM_NaAc_BSA_NaAc_Kcl_0mM_res_Ldp"; do
    
    echo "Running: $filename"
    nohup python main_spt_flow_diffL.py \
        --filename "$filename" \
        --epochs 50 \
        --beta_diffL 1.0 \
        --beta_photo 1.0 \
        --is_spt_pt \
        --link_method 'lap'\
        > "output.log" 2>&1
    
    echo "Completed: $filename"
done