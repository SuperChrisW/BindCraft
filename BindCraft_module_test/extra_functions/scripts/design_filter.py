import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

def convert_to_float(value):
    if value is not None and isinstance(value, str):
        try:
            value = float(value) if any(char.isdigit() for char in value.replace('.', '', 1)) else value
        except ValueError:
            pass
    return value

def check_filters(mpnn_data, filters):
    # check mpnn_data against labels
    mpnn_dict = mpnn_data

    unmet_conditions = []

    # check filters against thresholds
    for label, conditions in filters.items():
        # special conditions for interface amino acid counts
        try:
            if label == 'Average_InterfaceAAs' or label == '1_InterfaceAAs' or label == '2_InterfaceAAs' or label == '3_InterfaceAAs' or label == '4_InterfaceAAs' or label == '5_InterfaceAAs':
                continue    
            else:
                # if no threshold, then skip
                value = mpnn_dict.get(label)
                # Convert value to float if it looks like a number
                value = convert_to_float(value)

                if value is None or conditions["threshold"] is None:
                    continue
                if conditions["higher"]:
                    if value < conditions["threshold"]:
                        unmet_conditions.append(label)
                else:
                    if value > conditions["threshold"]:
                        unmet_conditions.append(label)
        except Exception as e:
            print(f"Error processing {label}: {e}")
            print(value, conditions["threshold"])
            raise e

    # if all filters are passed then return True
    if len(unmet_conditions) == 0:
        return True
    # if some filters were unmet, print them out
    else:
        return False

root_dir = f"/home/lwang/models/Data_pass/IL23_RF_XMPNN"
save_dir = f"/home/lwang/models/BindCraft/BindCraft_fork/IL23_RF_XMPNN"
filters = f"/home/lwang/models/BindCraft/BindCraft_fork/extra_settings/no_PB_filters.json"

os.makedirs(save_dir, exist_ok=True)

total_dfs = []
filter_dfs = []
for subfolder in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, subfolder)):
        data_fp = f"{root_dir}/{subfolder}/final_design_stats.csv"
    else:
        continue
    
    data_df = pd.read_csv(data_fp)
    total_dfs.append(data_df)

    with open(filters, "r") as f:
        filters_dict = json.load(f)
    if 'tqdm' in globals():
        mask = []
        for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Filtering designs"):
            mask.append(check_filters(row.to_dict(), filters_dict))
        mask = pd.Series(mask, index=data_df.index)
    else:
        mask = data_df.apply(lambda row: check_filters(row.to_dict(), filters_dict), axis=1)
    filtered_design = data_df[mask].reset_index(drop=True)
    filter_dfs.append(filtered_design)
    print(subfolder)
    print("filtered_design shape:", filtered_design.shape)
    print("original data shape:", data_df.shape)

total_df = pd.concat(total_dfs)
filtered_df = pd.concat(filter_dfs)
filtered_df = filtered_df.drop_duplicates(subset=['Sequence'])

print("total_df shape:", total_df.shape)
print("filtered_df shape:", filtered_df.shape)
print("success rate:", len(filtered_df) / len(total_df))
total_df.to_csv(f"{save_dir}/final_design_stats_total.csv", index=False)
filtered_df.to_csv(f"{save_dir}/final_design_stats_filtered.csv", index=False)