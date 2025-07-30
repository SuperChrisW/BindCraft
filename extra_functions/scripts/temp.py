import pandas as pd
import os
import shutil

input_dir = "/mnt/idd_intern/liyao.wang/BindCraft/IL23_BC_traj/5mj4_dplm_scaffold/XMPNN_redesign_filter/Trajectory"
input_df = "/mnt/idd_intern/liyao.wang/BindCraft/IL23_BC_traj/5mj4_dplm_scaffold/XMPNN_redesign_filter/XMPNN_redesign_seqs.csv"
df = pd.read_csv(input_df)

#targets = [file.split('.')[0] for file in os.listdir(input_dir)]
targets = df['batch'].unique().tolist()
print(targets)

for i in range(1,7):
    dirpath = f"/mnt/idd_intern/liyao.wang/BindCraft/IL23_BC_traj/5mj4_dplm_scaffold/dplm_filter/traj{i}/Accepted"
    for target in targets:
        dst_fpath = os.path.join(input_dir, target+".pdb")
        src_fpath = os.path.join(dirpath, target+".pdb")
        if os.path.exists(src_fpath):
            shutil.copy(src_fpath, dst_fpath)
