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

bindcraft_dir=$1
CUDA_VISIBLE_DEVICES=$2

cd /mnt/idd_intern/liyao.wang/BindCraft
python /mnt/idd_intern/liyao.wang/BindCraft/bindcraft_Filters.py \
 -i ${bindcraft_dir}/XMPNN_redesign_seqs_1.csv \
 -s ${bindcraft_dir}/IL23_BC_XMPNN.json \
 -f /mnt/idd_intern/liyao.wang/BindCraft/extra_settings/no_filters.json \
 -a /mnt/idd_intern/liyao.wang/BindCraft/IL23_BC_traj/IL23_BC_XMPNN/advance_XMPNN.json \
 --cuda ${CUDA_VISIBLE_DEVICES}