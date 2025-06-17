import os
import sys
import shutil
import argparse
import glob
import pandas as pd
from typing import List, Dict, Any
from functions.pyrosetta_utils import *

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

from bindcraft_module import (
    Initialization, Workspace, Scorer, BinderDesign, ArtifactHandler, pr_relax, insert_data
)

def get_pdb_files(input_path: str) -> List[str]:
    """
    Accepts a directory or a comma-separated list of files and returns a list of PDB file paths.
    """
    if os.path.isdir(input_path):
        pdbs = sorted(glob.glob(os.path.join(input_path, '*.pdb')))
        if not pdbs:
            raise FileNotFoundError(f"No .pdb files found in directory: {input_path}")
        return pdbs
    else:
        # Assume comma-separated list
        pdbs = [p.strip() for p in input_path.split(',') if p.strip()]
        for p in pdbs:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"PDB file not found: {p}")
        return pdbs

def infer_length_from_pdb(pdb_path: str, chain: str = 'B') -> int:
    """
    Infer the length of the binder chain from the PDB file.
    """
    length = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[21].strip() == chain:
                resseq = line[22:26].strip()
                length = max(length, int(resseq))
    return length

def get_sequence_from_pdb(pdb_path: str, chain: str = 'B') -> str:
    """
    Extract the sequence of the specified chain from a PDB file.
    """
    from Bio.PDB import PDBParser, PPBuilder
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('binder', pdb_path)
    for model in structure:
        for ch in model:
            if ch.id == chain:
                ppb = PPBuilder()
                seqs = [str(pp.get_sequence()) for pp in ppb.build_peptides(ch)]
                return ''.join(seqs)
    return ''

def main():
    parser = argparse.ArgumentParser(description='BindCraft pipeline from pre-generated PDBs (skip hallucination).')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory containing PDBs or comma-separated list of PDB files.')
    parser.add_argument('--settings', '-s', type=str, required=True,
                        help='Path to the basic settings.json file. Required.')
    parser.add_argument('--filters', '-f', type=str,
                        default='./settings_filters/default_filters.json',
                        help='Path to the filters.json file used to filter designs.')
    parser.add_argument('--advanced', '-a', type=str,
                        default='./settings_advanced/default_4stage_multimer.json',
                        help='Path to the advanced.json file with additional design settings.')
    parser.add_argument('--cuda', type=str, default='0',
                        help='define the GPU devices')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # Use Initialization to load settings and models
    initializer = Initialization()
    # Patch the args for Initialization
    class Args:
        pass
    patched_args = Args()
    patched_args.settings = args.settings
    patched_args.filters = args.filters
    patched_args.advanced = args.advanced
    patched_args.cuda = args.cuda
    initializer.parse_args = lambda: patched_args
    cfg = initializer.run()

    # Get input PDBs
    pdb_files = get_pdb_files(args.input)
    logger.info(f"Parsed PDB files: {len(pdb_files)}")
    
    ws = Workspace(cfg)
    ws.setup()
    ws.init_dataframes()
    logger.info(f"Workspace setup complete.")

    designer = BinderDesign(cfg["design_models"], ws.settings, ws.design_paths, ws.csv_paths)
    logger.info(f"BinderDesign initialized.")

    artifact = ArtifactHandler(ws.design_paths, ws.settings, ws.csv_paths)
    logger.info(f"ArtifactHandler initialized.")
    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {ws.advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
    accepted = 0

    
    for pdb_path in pdb_files:
        logger.info(f"Processing PDB file: {pdb_path}")
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        dest_pdb = os.path.join(ws.design_paths["Trajectory"], f"{name}.pdb")
        shutil.copy(pdb_path, dest_pdb)

        # Infer length and sequence
        binder_chain = 'B'
        length = infer_length_from_pdb(dest_pdb, binder_chain)
        seq = get_sequence_from_pdb(dest_pdb, binder_chain)

        ws.traj_info = {
            "name": name,
            "length": length,
            "seed": 0,
            "helicity_value": None,
            "traj_time_text": "",
            "binder_chain": binder_chain,
            "prediction_models": cfg["prediction_models"],
            "mutlimer_validation": cfg["multimer_validation"],
            "design_models": cfg["design_models"]
        }
        scorer = Scorer(ws)
        logger.info(f"Scorer refreshed.")

        # Relax
        relaxed_pdb = os.path.join(ws.design_paths["Trajectory/Relaxed"], f"{name}.pdb")
        pr_relax(dest_pdb, relaxed_pdb)
        logger.info(f"Finished relaxing trajectory")

        # Score
        traj_metrics = {}  # No AF2 metrics, so leave empty
        ws.traj_data = scorer.score_traj(seq, dest_pdb, relaxed_pdb, traj_metrics, binder_chain)
        logger.info(f"Finished scoring trajectory")

        insert_data(ws.csv_paths["trajectory_csv"], ws.traj_data.values())

        # MPNN redesign
        if ws.advanced_settings["enable_mpnn"]:
            logger.info(f"Starting MPNN redesign")
            accepted_mpnn = designer.mpnn_design(ws, scorer, dest_pdb)
            accepted += accepted_mpnn
            logger.info(f"Finished MPNN redesign: {accepted_mpnn} designs accepted")

    # Optionally rerank
    artifact.rerank_designs()
    print(f"Pipeline complete. Total accepted MPNN designs: {accepted}")

if __name__ == "__main__":
    main() 