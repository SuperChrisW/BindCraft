#!bin/bash

# run bindcraft design
#python bindcraft_module.py -s ./extra_settings/IL23_traj1.json -f ./extra_settings/no_filters.json -a ./extra_settings/advance_4stage_traj.json --cuda 3

# run bindcraft mpnn score + MPNN-FR redesign
#python bindcraft_mpnnScore.py \
# -i /home/lwang/models/BindCraft/BindCraft_fork/IL23_traj_test/Trajectory \
# -s /home/lwang/models/BindCraft/BindCraft_fork/extra_settings/IL23_traj1.json \
# -f /home/lwang/models/BindCraft/BindCraft_fork/extra_settings/MPNN_redesign_filters.json \
# -a /home/lwang/models/BindCraft/BindCraft_fork/extra_settings/advance_MPNN-FR.json \
# --cuda 2

python /home/lwang/models/BindCraft/BindCraft_fork/bindcraft_Filters.py \
 -i /home/lwang/models/BindCraft/BindCraft_fork/8OE4_RF_scaffold/traj1/8OE4_XMPNN_selected_slice_1.csv \
 -s /home/lwang/models/BindCraft/BindCraft_fork/8OE4_RF_scaffold/8OE4_scaffold.json \
 -f /home/lwang/models/BindCraft/BindCraft_fork/extra_settings/no_PB_filters.json \
 -a /home/lwang/models/BindCraft/BindCraft_fork/IL23_BC_XMPNN_top300/advance_XMPNN.json \
 --cuda 3