import os
import shutil

def copy_folder_contents(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for item in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item)
        dst_item = os.path.join(dst_folder, item)
        if os.path.isdir(src_item):
            # Recursively copy subfolder
            copy_folder_contents(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

patch_dir = './extra_functions'
conda_dir = os.environ['CONDA_PREFIX']

destination_map = {
    'bindcraft_patch': './functions',
    'colabdesign_patch/af': f'{conda_dir}/lib/python3.10/site-packages/colabdesign/af',
    'colabdesign_patch/shared': f'{conda_dir}/lib/python3.10/site-packages/colabdesign/shared',
    'colabdesign_patch/confidence.py': f'{conda_dir}/lib/python3.10/site-packages/colabdesign/af/alphafold/common/confidence.py',
}

for relative_src, dest in destination_map.items():
    src = os.path.join(patch_dir, relative_src)
    dst = dest

    if not os.path.exists(src):
        print(f"Patch source not found: {src}")
        continue

    if os.path.isdir(src):
        print(f"Copying contents of directory {src} -> {dst}")
        copy_folder_contents(src, dst)
    else:
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"Copying file {src} -> {dst}")
        shutil.copy2(src, dst)

print("Patching complete.")
