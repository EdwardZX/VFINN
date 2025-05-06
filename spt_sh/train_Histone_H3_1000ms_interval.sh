#!/bin/bash

# For Histone_H3_1000ms_interval
for stack in 1 2 4 5 6 8 10 12 14; do
    filename="Histone_H3_1000ms_interval_${stack}_MMStack_Ldp"
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