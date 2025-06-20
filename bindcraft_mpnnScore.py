import os
import shutil
import argparse
import time
import glob
import pandas as pd
import numpy as np

from typing import List, Dict, Any
from functions.pyrosetta_utils import *
from Bio.PDB import PDBParser, PPBuilder

from colabdesign import clear_mem
from colabdesign.mpnn import mk_mpnn_model
from functions.generic_utils import create_dataframe, insert_data
from functions.biopython_utils import hotspot_residues

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

from bindcraft_module import (
    Initialization, Workspace, Scorer, BinderDesign, ArtifactHandler, TerminationCriteria, pr_relax, insert_data
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

# MPNN functions 
class MPNNHandler:
    def __init__(self):
        pass

    def get_mpnn_score(self, pdb_path: str, ws: Workspace, pdb_handler: PDBHandler, binder_chain='B') -> float:
        score_start_time = time.time()
        clear_mem()

        logger.info(f"Processing PDB file: {pdb_path}")
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        dest_pdb = os.path.join(ws.design_paths["Trajectory"], f"{name}.pdb")
        #shutil.copy(pdb_path, dest_pdb)

        length = pdb_handler.infer_length_from_pdb(dest_pdb, binder_chain)
        # skip relax
        mpnn_model = mk_mpnn_model(backbone_noise=ws.advanced_settings["backbone_noise"], model_name=ws.advanced_settings["model_path"], weights=ws.advanced_settings["mpnn_weights"])
        design_chains = 'A,B'

        traj_interface_residues = []
        interface_residues_set = hotspot_residues(pdb_path, binder_chain=binder_chain, atom_distance_cutoff=4.0)
        for pdb_res_num, aa_type in interface_residues_set.items():
            traj_interface_residues.append(f"{binder_chain}{pdb_res_num}")
        traj_interface_residues = ','.join(traj_interface_residues)

        if ws.advanced_settings["mpnn_fix_interface"]:
            fixed_positions = 'A,' + traj_interface_residues
            fixed_positions = fixed_positions.rstrip(",")
            print("Fixing interface residues: "+ traj_interface_residues)
        else:
            fixed_positions = 'A'
        
        mpnn_model.prep_inputs(pdb_filename=dest_pdb, chain=design_chains, fix_pos=fixed_positions, rm_aa=ws.advanced_settings["omit_AAs"])

        # sample MPNN sequences in parallel
        mpnn_sequences = mpnn_model.sample(temperature=ws.advanced_settings["sampling_temp"], num=1, batch=ws.advanced_settings["num_seqs"])
        for i in range(ws.advanced_settings["num_seqs"]):
            row_data = {
                "Design": name,
                "MPNN_id": f"_mpnn{i}",
                "Sequence": mpnn_sequences['seq'][i][-length:],
                "MPNN_score": mpnn_sequences['score'][i],
                "InterfaceResidues": traj_interface_residues,
            }
            insert_data(ws.csv_paths["mpnn_bb_score_stats"], row_data)
        logger.info(f"MPNN scoring time: {time.time() - score_start_time:.2f} seconds")
        return np.mean([mpnn_sequences['score'][i] for i in range(ws.advanced_settings["num_seqs"])])

    def select_topBB_design(self, ws, top_N: int = 200, pick_strategy: str = 'top', num_pick_per_batch: int = 1) -> pd.DataFrame:
        '''
        Select the top-ranked designs from the MPNN scoring results.
        Args:
            ws: Workspace object
            top_N: Number of top-ranked designs to select
            pick_strategy: Strategy to pick the representative design from each group
            num_pick_per_batch: Number of designs to pick from each group
        Returns:
            pd.DataFrame: DataFrame containing (top_N * num_pick_per_batch) designs
        '''
        BB_df = pd.read_csv(ws.csv_paths["mpnn_bb_score_stats"])
        BB_df['Length'] = BB_df['Sequence'].apply(len)
        #BB_grouped = BB_df.groupby("Design")["MPNN_score"].mean().reset_index()
        #BB_grouped_sorted = BB_grouped.sort_values(by="MPNN_score", ascending=True)
        BB_GS_L50 = BB_df[BB_df["Length"] <= 50].groupby("Design")["MPNN_score"].mean().reset_index()
        BB_GS_M50 = BB_df[BB_df["Length"] > 50].groupby("Design")["MPNN_score"].mean().reset_index()
        #top_designs = BB_grouped_sorted.head(top_N)
        top_designs = pd.concat([BB_GS_L50.head(top_N//2), BB_GS_M50.head(top_N//2)])

        representative_designs = BB_df[BB_df["Design"].isin(top_designs["Design"])]
        # For each group, pick the top-ranked entry as the representative design
        if pick_strategy == 'top':
            final_grouped_table = representative_designs.groupby("Design").apply(
                lambda x: x.nsmallest(num_pick_per_batch, "MPNN_score")
            ).reset_index(drop=True)
        elif pick_strategy == 'random':
            final_grouped_table = representative_designs.groupby("Design").apply(
                lambda x: x.sample(n=num_pick_per_batch, random_state=42)
            ).reset_index(drop=True)
        else:
            raise ValueError(f"Invalid pick_strategy: {pick_strategy}")
        
        final_grouped_table.to_csv(ws.csv_paths["mpnn_bb_score_stats"].replace(".csv", "_selected.csv"), index=False)
        return final_grouped_table

class CustomPipeline:
    def __init__(self):
        self.cfg = None
        self.pipeline_start_time = time.time()
    
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
        parser.add_argument('--opt_cycles', type=int, default=1,
                            help='Number of relax-MPNN optimization cycles to perform (default: 1)')
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
        
        # initiate a new bb score csv file
        ws.settings["mpnn_bb_labels"] = ['Design', 'MPNN_id', 'Sequence', 'MPNN_score', 'InterfaceResidues']
        mpnn_bb_score_csv = os.path.join(ws.target_settings["design_path"], "mpnn_bb_score_stats.csv")
        ws.csv_paths["mpnn_bb_score_stats"] = mpnn_bb_score_csv
        create_dataframe(ws.csv_paths["mpnn_bb_score_stats"], ws.settings["mpnn_bb_labels"])
        logger.info(f"Workspace setup complete.")

        designer = BinderDesign(self.cfg["design_models"], ws.settings, ws.design_paths, ws.csv_paths)
        logger.info(f"BinderDesign initialized.")

        artifact = ArtifactHandler(ws.design_paths, ws.settings, ws.csv_paths)
        criteria = TerminationCriteria(ws.settings['advanced_settings'])
        logger.info(f"ArtifactHandler initialized.")
        pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {ws.advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')

        return ws, designer, artifact, criteria
    
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

    def select_best_mpnn_design(self, cycle_results):
        """
        Select the best MPNN design from the current cycle for the next iteration.
        """
        best_designs = None
        best_score = float('-inf')
        for key, mpnn_dict in cycle_results['mpnn_data'].items():
            plddt_scores = mpnn_dict.get('plddt_score', [])
            if not plddt_scores:
                continue
            max_score = max(plddt_scores)
            indices = [i for i, s in enumerate(plddt_scores) if s == max_score]
            if max_score > best_score:
                best_score = max_score
                best_designs = [(key, i) for i in indices]

        if best_designs:
            key, idx = best_designs[0]
            return os.path.abspath(cycle_results['mpnn_data'][key]['relaxed_pdb'][idx])
        else:
            return None

    def run_PMPNN_redesign(self, pdb_path, ws, designer, pdb_handler, criteria, opt_cycles=1, num_mpnn_sample = 20, binder_chain = 'B'):
        '''
        This function is used to redesign a PDB file using BindCraft pipeline with iterative optimization.
        
        The workflow performs multiple relax-MPNN cycles:
        1. Copy the PDB file to the trajectory directory
        2. For each optimization cycle:
           a. Relax the trajectory (structure optimization)
           b. Score the trajectory (evaluate current state)
           c. MPNN redesign the trajectory (sequence optimization while keeping interface residues fixed)
           d. Select best design for next cycle (if available)
        
        Args:
            pdb_path: Path to input PDB file, defaultly use the top-ranked backbone design from the initial MPNN sampling
            ws: Workspace object
            designer: BinderDesign object
            pdb_handler: PDBHandler object
            opt_cycles: Number of relax-MPNN optimization cycles to perform (default: 1)
        
        Returns:
            int: Total number of accepted MPNN designs across all cycles
        '''
        logger.info(f"Processing PDB file: {pdb_path} with {opt_cycles} optimization cycles")
        name = os.path.splitext(os.path.basename(pdb_path))[0]
        if not pdb_path.endswith('.pdb'):
            pdb_path += '.pdb'
        dest_pdb = os.path.join(ws.design_paths["Trajectory"], f"{name}.pdb")
        
        if os.path.abspath(pdb_path) != os.path.abspath(dest_pdb):  # Check to avoid copying the same file
            shutil.copy(pdb_path, dest_pdb)

        length = pdb_handler.infer_length_from_pdb(dest_pdb, binder_chain)
        seq = pdb_handler.get_sequence_from_pdb(dest_pdb, binder_chain)
        scorer = self.update_traj_info(ws, name, binder_chain, length)
        
        accepted_mpnn = 0
        current_pdb = dest_pdb
        cycle_results = []

        logger.info(f"Starting iterative optimization with {opt_cycles} cycles")
        for cycle in range(opt_cycles):
            # apply an early stopping criterion to prevent contact <3 aa | ca clashes model
            if not criteria.quality_check(current_pdb, ws.csv_paths["failure_csv"]):
                logger.info(f"✓ Early stopping: {current_pdb} failed quality check")
                break
            else:
                logger.info(f"✓ {current_pdb} passed quality check")

            logger.info(f"=== Optimization Cycle {cycle + 1}/{opt_cycles} ===")
            scorer = self.update_traj_info(ws, f"{name}_cycle{cycle}", binder_chain, length, seed=0, helicity_value=None, traj_time_text=f"Cycle {cycle + 1}")

            # Step 1: Relax - Structure optimization
            relaxed_pdb = os.path.join(ws.design_paths["Trajectory/Relaxed"], f"{name}_cycle{cycle}.pdb")
            logger.info(f"Step 1: Relaxing structure from {os.path.basename(current_pdb)}")
            pr_relax(current_pdb, relaxed_pdb)
            logger.info(f"✓ Relaxation complete: {os.path.basename(relaxed_pdb)}")

            # Step 2: Score - Evaluate current state
            logger.info(f"Step 2: Scoring relaxed structure")
            traj_metrics = {}  # No AF2 metrics, so leave empty
            ws.traj_data = scorer.score_traj(seq, current_pdb, relaxed_pdb, traj_metrics, binder_chain)
            insert_data(ws.csv_paths["trajectory_csv"], ws.traj_data.values())
            logger.info(f"✓ Scoring complete")

            # Step 3: MPNN redesign - Sequence optimization
            cycle_accepted = 0
            if ws.advanced_settings["enable_mpnn"]:
                logger.info(f"Step 3: MPNN sequence redesign")
                ws.advanced_settings['max_mpnn_sequences'] = num_mpnn_sample if ws.advanced_settings['num_seqs'] > num_mpnn_sample else ws.advanced_settings['num_seqs']
                cycle_accepted, mpnn_dict = designer.mpnn_design(ws, scorer, relaxed_pdb)
                #note: relaxed traj pdb will be removed in set in advanced_settings
                accepted_mpnn += cycle_accepted
                logger.info(f"✓ MPNN redesign complete: {cycle_accepted} designs accepted")
                
                # Store cycle results for potential selection of best design
                cycle_results.append({
                    'cycle': cycle + 1,
                    'relaxed_pdb': relaxed_pdb,
                    'accepted_designs': cycle_accepted,
                    'mpnn_data': mpnn_dict,
                })
                
                # Update current_pdb to the best MPNN design for next cycle
                if cycle_accepted > 0:
                    # if multiple mpnn designs returned, pick the best one based on plddt score and return pdb file path
                    best_design = self.select_best_mpnn_design(cycle_results[-1]) # pick design from the last cycle
                    if best_design:
                        current_pdb = best_design
                        logger.info(f"✓ Selected best design for next cycle: {os.path.basename(best_design)}")
                    else:
                        # Fallback to relaxed structure if no best design selected
                        current_pdb = relaxed_pdb
                        logger.info(f"✓ Using relaxed structure for next cycle: {os.path.basename(relaxed_pdb)}")
                else:
                    # If no MPNN designs were accepted, highly possible that this backbone is not suitable 
                    logger.info(f"✓ No MPNN designs accepted, skipping this design")
                    break
            else:
                current_pdb = relaxed_pdb
                cycle_results.append({
                    'cycle': cycle + 1,
                    'relaxed_pdb': relaxed_pdb,
                    'accepted_designs': 0,
                    'traj_data': ws.traj_data.copy() if hasattr(ws.traj_data, 'copy') else ws.traj_data
                })
                logger.info(f"✓ MPNN disabled, using relaxed structure: {os.path.basename(relaxed_pdb)}")

        logger.info(f"=== Optimization Complete ===")
        logger.info(f"Total accepted MPNN designs across all cycles: {accepted_mpnn}")
        
        # Log summary of all cycles
        logger.info("Cycle Summary:")
        for result in cycle_results:
            logger.info(f"  Cycle {result['cycle']}: {result['accepted_designs']} designs accepted")
        
        return accepted_mpnn

def main():
    # Initialize pipeline
    pipeline = CustomPipeline()
    args = pipeline.parse_args()
    ws, designer, artifact, criteria = pipeline.init_pipeline(args)

    # Get input PDBs
    pdb_handler = PDBHandler()
    pdb_files = pdb_handler.get_pdb_files(args.input)
    logger.info(f"Parsed PDB files: {len(pdb_files)}")
    accepted = 0
    
    # Initial MPNN sampling
    # Evaluate Backbone designability: MPNN score
    mpnn_handler = MPNNHandler()
    for i, pdb_fpath in enumerate(pdb_files):
        basename = os.path.basename(pdb_fpath).replace('.pdb', '')
        score = mpnn_handler.get_mpnn_score(pdb_fpath, ws, pdb_handler) # sample 20 sequences and get the mean score
        logger.info(f"MPNN score for {basename}: {score}")
    
    # Select top 200 BB designs based on MPNN score
    # pick the top-ranked design from each BB group as representative design for MPNN redesign
    mpnn_struct_seeds = mpnn_handler.select_topBB_design(ws, top_N=300) #50% for length <= 50, 50% for length > 50
    logger.info(f"Selected {len(mpnn_struct_seeds)} designs for MPNN redesign")

    # MPNN redesign cycles
    for file in mpnn_struct_seeds["Design"].unique():
        # single MPNN design or multiple FR-MPNN design cycles
        pdb_fpath = os.path.join(ws.design_paths["Trajectory"], f"{file}.pdb")
        accepted += pipeline.run_PMPNN_redesign(pdb_fpath, ws, designer, pdb_handler, criteria, opt_cycles = 5, 
                                                num_mpnn_sample=10, binder_chain='B')
        # total designs = top_N Backbone * opt_cycles * num_mpnn_sample
        # 300 * 5 * 10 = 15000 designs

    artifact.rerank_designs()
    pipeline.end_time = time.time()
    logger.info(f"Pipeline complete. Total accepted MPNN designs: {accepted}")
    logger.info(f"Total time: {pipeline.end_time - pipeline.pipeline_start_time:.2f} seconds")

if __name__ == "__main__":
    main() 