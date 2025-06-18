#!bin/bash

# run bindcraft design
#python bindcraft_module.py -s ./extra_settings/IL23_traj1.json -f ./extra_settings/no_filters.json -a ./extra_settings/advance_4stage_traj.json --cuda 3

# run bindcraft mpnn score
python bindcraft_mpnnScore.py -i /home/lwang/models/BindCraft/BindCraft_fork/IL23_traj_test/Trajectory \
 -s /home/lwang/models/BindCraft/BindCraft_fork/extra_settings/IL23_traj1.json -f /home/lwang/models/BindCraft/BindCraft_fork/extra_settings/no_filters.json -a /home/lwang/models/BindCraft/BindCraft_fork/extra_settings/advance_4stage_traj.json --cuda 2