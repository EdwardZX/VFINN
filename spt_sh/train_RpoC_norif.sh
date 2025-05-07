#!/bin/bash

# List all filenames directly
for filename in \
    "norif_DACP_2_30mW_spt_1_MMStack_Pos0_Ldp" \
    "norif_DACP_2_30mW_spt_2_MMStack_Pos0_Ldp" \
    "norif_DACP_3_30mW_spt_1_MMStack_Pos0_Ldp" \
    "norif_DACP_3_30mW_spt_2_MMStack_Pos0_Ldp" \
    "norif_DACP_3_60mW_spt_1_MMStack_Pos0_Ldp" \
    "norif_DACP_3_60mW_spt_2_MMStack_Pos0_Ldp" \
    "norif_DACP_14_30mW_spt_1_MMStack_Pos0_Ldp" \
    "norif_DACP_14_30mW_spt_2_MMStack_Pos0_Ldp"; do
    
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