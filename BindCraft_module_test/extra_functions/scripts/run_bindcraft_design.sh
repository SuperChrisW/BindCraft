#!/bin/bash

# ====== Set CUDA Device and Paths ======
export CUDA_VISIBLE_DEVICES=2
BINDCRAFT_DIR=/home/lwang/models/BindCraft/BindCraft_fork/IL23_pepBinder2
mode='bindcraft_module' #bindcraft_module(whole pipeline)/bindcraft_mpnnScores(input pdb)/bindcraft_Filters(input csv sequence)

cd /home/lwang/models/BindCraft/BindCraft_fork || { echo "Failed to change directory"; exit 5; }
#-i "$BINDCRAFT_DIR/dplm2_designs.csv" \

python /home/lwang/models/BindCraft/BindCraft_fork/${mode}.py \
    -s "$BINDCRAFT_DIR/../extra_settings/IL23_pepBinder.json" \
    -f /home/lwang/models/BindCraft/BindCraft_fork/extra_settings/no_PB_filters.json \
    -a $BINDCRAFT_DIR/../extra_settings/peptide_3stage_multimer_revised.json \
    --cuda "$CUDA_VISIBLE_DEVICES"
ret=$?
if [ $ret -ne 0 ]; then
    echo "Error: BindCraft filter failed." >&2
    exit 6
fi

echo "Pipeline finished successfully."