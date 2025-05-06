#!/bin/bash

for density in 1 5 10 25 50; do
    for dfree in 1 10 100; do
        filename="nocutoff_singlePop_densMult10_${density}_locum2_DfreeMult10_${dfree}_um2s_Ldp"
        echo "Running: $filename"
        nohup python main_spt_flow_diffL.py \
            --filename "$filename" \
            --epochs 50 \
            --beta_diffL 1.0 \
            --beta_photo 1.0 \
            --is_spt_pt \
            --is_train \
            --sample_rate 0.2 \
            > "output.log" 2>&1
        
        echo "Completed: $filename"
    done
done