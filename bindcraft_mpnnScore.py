import os
import shutil
import argparse
import glob
import pandas as pd
import numpy as np

from typing import List, Dict, Any
from functions.pyrosetta_utils import *
from Bio.PDB import PDBParser, PPBuilder

from colabdesign import clear_mem
from colabdesign.mpnn import mk_mpnn_model

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

from bindcraft_module import (
    Initialization, Workspace, Scorer, BinderDesign, ArtifactHandler, pr_relax, insert_data
)

class PDBHandler:
    def __init__(self):
        pass

    def get_pdb_files(self, input_path: str) -> List[str]:
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

    def infer_length_from_pdb(self, pdb_path: str, chain: str = 'B') -> int:
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

    def get_sequence_from_pdb(self, pdb_path: str, chain: str = 'B') -> str:
        """
        Extract the sequence of the specified chain from a PDB file.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('binder', pdb_path)
        for model in structure:
            for ch in model:
                if ch.id == chain:
                    ppb = PPBuilder()
                    seqs = [str(pp.get_sequence()) for pp in ppb.build_peptides(ch)]
                    return ''.join(seqs)
        return ''

class MPNNHandler:
    def __init__(self):
        pass

    def get_mpnn_score(self, pdb_path: str, ws, pdb_handler, binder_chain='B') -> float:
        clear_mem()

        logger.info(f"Processing PDB file: {pdb_path}")
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        dest_pdb = os.path.join(ws.design_paths["Trajectory"], f"{name}.pdb")
        #shutil.copy(pdb_path, dest_pdb)

        length = pdb_handler.infer_length_from_pdb(dest_pdb, binder_chain)
        # skip relax
        mpnn_model = mk_mpnn_model(backbone_noise=ws.advanced_settings["backbone_noise"], model_name=ws.advanced_settings["model_path"], weights=ws.advanced_settings["mpnn_weights"])
        design_chains = 'A,B'

        traj_interface_scores, traj_interface_AA, traj_interface_residues = score_interface(pdb_path, binder_chain=binder_chain)
        if ws.advanced_settings["mpnn_fix_interface"]:
            fixed_positions = 'A,' + traj_interface_residues
            fixed_positions = fixed_positions.rstrip(",")
            print("Fixing interface residues: "+traj_interface_residues)
        else:
            fixed_positions = 'A'

        # prepare inputs for MPNN
        mpnn_model.prep_inputs(pdb_filename=dest_pdb, chain=design_chains, fix_pos=fixed_positions, rm_aa=ws.advanced_settings["omit_AAs"])

        # sample MPNN sequences in parallel
        mpnn_sequences = mpnn_model.sample(temperature=ws.advanced_settings["sampling_temp"], num=1, batch=ws.advanced_settings["num_seqs"])
        mpnn_seq_data = {
            'seq': [mpnn_sequences['seq'][i][-length:] for i in range(ws.advanced_settings["num_seqs"])], 
            'score': [mpnn_sequences['score'][i] for i in range(ws.advanced_settings["num_seqs"])], 
            }
        
        return np.mean(mpnn_seq_data['score'])

class CustomPipeline:
    def __init__(self):
        self.cfg = None
    
    def parse_args(self):
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
        return args
    
    def init_pipeline(self, args):
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
        self.cfg = initializer.run()
        
        ws = Workspace(self.cfg)
        ws.setup()
        ws.init_dataframes()
        logger.info(f"Workspace setup complete.")

        designer = BinderDesign(self.cfg["design_models"], ws.settings, ws.design_paths, ws.csv_paths)
        logger.info(f"BinderDesign initialized.")

        artifact = ArtifactHandler(ws.design_paths, ws.settings, ws.csv_paths)
        logger.info(f"ArtifactHandler initialized.")
        pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {ws.advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')

        return ws, designer, artifact
    
    def update_traj_info(self, ws, name, binder_chain, length, seed=0, helicity_value=None, traj_time_text=""):
        ws.traj_info = {
            "name": name,
            "length": length,
            "seed": seed,
            "helicity_value": helicity_value,
            "traj_time_text": traj_time_text,
            "binder_chain": binder_chain,
            "prediction_models": self.cfg["prediction_models"],
            "mutlimer_validation": self.cfg["multimer_validation"],
            "design_models": self.cfg["design_models"]
        }
        scorer = Scorer(ws)
        return scorer

    def run_PDB_redesign(self, pdb_path, ws, designer, pdb_handler):
        '''
        This function is used to redesign a PDB file using BindCraft pipeline.
        It will:
        1. Copy the PDB file to the trajectory directory
        2. Relax the trajectory
        3. Score the trajectory
        4. MPNN redesign the trajectory (keep interface residues while redesign the rest)
        return the number of accepted designs
        '''
        logger.info(f"Processing PDB file: {pdb_path}")
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        dest_pdb = os.path.join(ws.design_paths["Trajectory"], f"{name}.pdb")
        shutil.copy(pdb_path, dest_pdb)

        # Relax
        relaxed_pdb = os.path.join(ws.design_paths["Trajectory/Relaxed"], f"{name}.pdb")
        pr_relax(dest_pdb, relaxed_pdb)
        logger.info(f"Finished relaxing trajectory")

        # Score
        binder_chain = 'B'
        length = pdb_handler.infer_length_from_pdb(dest_pdb, binder_chain)
        seq = pdb_handler.get_sequence_from_pdb(dest_pdb, binder_chain)
        scorer = self.update_traj_info(ws, name, binder_chain, length)
        traj_metrics = {}  # No AF2 metrics, so leave empty
        ws.traj_data = scorer.score_traj(seq, dest_pdb, relaxed_pdb, traj_metrics, binder_chain)
        logger.info(f"Finished scoring trajectory")
        insert_data(ws.csv_paths["trajectory_csv"], ws.traj_data.values())

        # MPNN redesign
        accepted_mpnn = 0
        if ws.advanced_settings["enable_mpnn"]:
            logger.info(f"Starting MPNN redesign")
            accepted_mpnn = designer.mpnn_design(ws, scorer, dest_pdb)
            accepted += accepted_mpnn
            logger.info(f"Finished MPNN redesign: {accepted_mpnn} designs accepted")
        return accepted_mpnn

def main():
    # Initialize pipeline
    pipeline = CustomPipeline()
    args = pipeline.parse_args()
    ws, designer, artifact = pipeline.init_pipeline(args)

    # Get input PDBs
    pdb_handler = PDBHandler()
    pdb_files = pdb_handler.get_pdb_files(args.input)
    logger.info(f"Parsed PDB files: {len(pdb_files)}")
    accepted = 0

    mpnn_handler = MPNNHandler()
    # customize loop
    for pdb_path in pdb_files:
        #accepted += pipeline.run_PDB_redesign(pdb_path, ws, designer, pdb_handler)
        score = mpnn_handler.get_mpnn_score(pdb_path, ws, pdb_handler)
        print(f"for PDB {os.path.basename(pdb_path)}, MPNN score: {score}")

    #artifact.rerank_designs()
    #print(f"Pipeline complete. Total accepted MPNN designs: {accepted}")

if __name__ == "__main__":
    main() 