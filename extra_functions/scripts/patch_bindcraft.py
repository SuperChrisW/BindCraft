import os
import shutil

# Define the patch directory and the mapping to destination directories
patch_dir = './extra_functions'
conda_dir = os.environ['CONDA_PREFIX']

# Map: filename -> destination directory (absolute path)
destination_map = {
    'colabdesign_utils.py': './functions',
    'confidence.py': f'{conda_dir}/lib/python3.10/site-packages/colabdesign/af/alphafold/common/confidence.py',
    'loss.py': f'{conda_dir}/lib/python3.10/site-packages/colabdesign/af/loss.py',
}

for filename, dest_dir in destination_map.items():
    src = os.path.join(patch_dir, filename)
    dst = os.path.join(dest_dir, filename)
    if not os.path.exists(src):
        print(f"Patch file not found: {src}")
        continue
    if not os.path.exists(dest_dir):
        print(f"Destination directory does not exist: {dest_dir}")
        continue
    print(f"Copying {src} -> {dst}")
    shutil.copy2(src, dst)

print("Patching complete.")