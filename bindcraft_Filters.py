import os
import shutil
import argparse
import time
import glob
import pandas as pd
import numpy as np
import re
from colabdesign.shared.utils import copy_dict
from typing import List, Dict, Any, Tuple
from functions.pyrosetta_utils import *
from Bio.PDB import PDBParser, PPBuilder

from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from functions.generic_utils import create_dataframe, insert_data, update_failures, check_filters
from functions.biopython_utils import hotspot_residues, validate_design_sequence

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
    
    def predict_binder(self, binder_seq: str, ws: Workspace, traj_pdb: str) -> Dict:
        clear_mem()
        binder_stats = {}
        prediction_models = ws.traj_info.get("prediction_models", [0])

        model = mk_afdesign_model(protocol="hallucination", use_templates=False, initial_guess=False, 
                                                    use_initial_atom_pos=False, num_recycles=ws.advanced_settings["num_recycles_validation"], 
                                                    data_dir=ws.advanced_settings["af_params_dir"], use_multimer=ws.traj_info['multimer_validation'])
        model.prep_inputs(length=len(binder_seq))
        model.set_seq(binder_seq)
        
        for model_num_to_use in prediction_models:
            binder_alone_pdb = os.path.join(ws.design_paths["MPNN/Binder"], f"{ws.traj_info['name']}_model{model_num_to_use+1}.pdb")
        
            model.predict(models=[model_num_to_use],
                          num_recycles = ws.advanced_settings['num_recycles_validation'],
                          verbose = False)
            model.save_pdb(binder_alone_pdb)
            logger.info(f"Predicted binder structure saved to {binder_alone_pdb}")

            prediction_metrics = copy_dict(model.aux["log"])
            stats = {
                'output_pdb': binder_alone_pdb,
                'plddt': round(prediction_metrics.get('plddt', 0), 2),
                'ptm': round(prediction_metrics.get('ptm',0), 2), 
                'pae': round(prediction_metrics.get('pae',0), 2)
            }
            binder_stats[model_num_to_use+1] = stats
            align_pdbs(traj_pdb, binder_alone_pdb, ws.traj_info["binder_chain"], 'A')

        return binder_stats

    def predict_complex(self, binder_seq: str, ws: Workspace) -> Tuple[Dict, bool]:
        """
        Predicts the complex structure from a binder sequence and a target PDB.

        This function uses the 'binder' protocol from colabdesign. It takes the
        target structure from ws.target_settings['starting_pdb'] and models the
        provided 'binder_seq' in complex with it.

        Args:
            binder_seq (str): The amino acid sequence of the binder.
            ws (Workspace): The workspace object containing paths and settings.
            cfg (dict): The main configuration dictionary containing model settings.

        Returns:
            - A dictionary with prediction metrics (pLDDT, pTM, etc.) for each model.
        """
        clear_mem()
        prediction_stats = {}
        filter_failures = {}
        pass_af2_filters = True

        # Get model parameters from cfg and advanced_settings
        prediction_models = ws.traj_info.get("prediction_models", [0])
        multimer_validation = ws.traj_info.get("multimer_validation", ws.advanced_settings.get("use_multimer_design", False))
        
        # 1. Initialize the AFDesign model for binder prediction
        model = mk_afdesign_model(protocol="binder", 
                                num_recycles=ws.advanced_settings["num_recycles_validation"], 
                                data_dir=ws.advanced_settings["af_params_dir"], 
                                use_multimer=ws.traj_info["multimer_validation"], 
                                use_initial_guess=ws.advanced_settings["predict_initial_guess"], 
                                use_initial_atom_pos=ws.advanced_settings["predict_bigbang"])

        # 2. Prepare inputs using only the target PDB as a template
        model.prep_inputs(
            pdb_filename=ws.target_settings["starting_pdb"],
            chain=ws.target_settings["chains"],
            binder_len=len(binder_seq),
            rm_target_seq=ws.advanced_settings.get("rm_template_seq_predict", False),
            rm_target_sc=ws.advanced_settings.get("rm_template_sc_predict", False)
        )

        # Clean the sequence
        binder_seq = re.sub("[^A-Z]", "", binder_seq.upper())

         # 3. Run prediction
        for model_num_to_use in prediction_models:
            design_name = ws.traj_info["name"]
            output_pdb_path = os.path.join(ws.design_paths["MPNN"], f"{design_name}_model{model_num_to_use+1}.pdb")
            #if os.path.exists(output_pdb_path):
            #    continue

            model.predict(
                seq=binder_seq,
                models=[model_num_to_use],
                num_recycles=ws.advanced_settings.get("num_recycles_validation", 3),
                verbose=False
            )
            model.save_pdb(output_pdb_path)
            logger.info(f"Predicted structure saved to {output_pdb_path}")

            prediction_metrics = copy_dict(model.aux["log"])
            stats = {
                'output_pdb': output_pdb_path,
                'plddt': round(prediction_metrics.get('plddt', 0), 2),
                'ptm': round(prediction_metrics.get('ptm', 0), 2),
                'i_ptm': round(prediction_metrics.get('i_ptm', 0), 2),
                'pae': round(prediction_metrics.get('pae', 0), 2),
                'i_pae': round(prediction_metrics.get('i_pae', 0), 2)
            }

            prediction_stats[model_num_to_use+1] = stats

            # List of filter conditions and corresponding keys
            filter_conditions = [
                (f"{model_num_to_use+1}_pLDDT", 'plddt', '>='),
                (f"{model_num_to_use+1}_pTM", 'ptm', '>='),
                (f"{model_num_to_use+1}_i_pTM", 'i_ptm', '>='),
                (f"{model_num_to_use+1}_pAE", 'pae', '<='),
                (f"{model_num_to_use+1}_i_pAE", 'i_pae', '<='),
            ]

            # perform initial AF2 values filtering to determine whether to skip relaxation and interface scoring
            filters = ws.filters_settings        
            for filter_name, metric_key, comparison in filter_conditions:
                threshold = filters.get(filter_name, {}).get("threshold")
                if threshold is not None:
                    if comparison == '>=' and prediction_metrics[metric_key] < threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1
                    elif comparison == '<=' and prediction_metrics[metric_key] > threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1

            if not pass_af2_filters:
                break
        if filter_failures:
            update_failures(ws.csv_paths["failure_csv"], filter_failures)

        return prediction_stats, pass_af2_filters

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
            fixed_positions = fixed_positions.rstrip(",") # FIXME: why do we need to remove the last comma?
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
        self.statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                            'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                            'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                            'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

    def parse_args(self):
        parser = argparse.ArgumentParser(description='BindCraft pipeline from pre-generated PDBs (skip hallucination).')
        parser.add_argument('--input', '-i', type=str, required=True,
                            help='A csv file containing generated designs')
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

        #os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
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
        #ws.settings["mpnn_bb_labels"] = ['Design', 'MPNN_id', 'Sequence', 'MPNN_score', 'InterfaceResidues']
        ws.csv_paths["final_designs"] = args.input
        #create_dataframe(ws.csv_paths["mpnn_bb_score_stats"], ws.settings["mpnn_bb_labels"])
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
            "multimer_validation": self.cfg["multimer_validation"],
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

    def score_MPNN_designs(self, binder_seq, mpnn_score, ws, pdb_handler, scorer, criteria):
        '''
        This function is used to score a sequence using BindCraft pipeline.
        It will:
        1. Predict the initial structure (default 2 models and pick the best one) and filter by af2 filters
        2. Relax the structure
        3. Score the relaxed structure
        4. Evaluate the confidence and pyrosetta metrics
        5. Check if the design passes the filters
        '''
        name = ws.traj_info['name']
        length = len(binder_seq)

        # Step 1: Structure Prediction
        # predict intial structure and filter by af2 filters        
        complex_stats, pass_af2_filteres = pdb_handler.predict_complex(binder_seq=binder_seq, ws=ws)
        # Find the traj with max ipTM value
        best_key = max(complex_stats, key=lambda k: complex_stats[k].get('i_ptm', 0))
        best_traj = complex_stats[best_key]['output_pdb']

        # early-stop clash check
        if not pass_af2_filteres or not criteria.quality_check(best_traj, ws.csv_paths["failure_csv"]):
            logger.info(f"✓ Early stopping: {best_traj} failed quality check")            
            return False

        # Step 2: Relax - Structure optimization
        relaxed_pdb = os.path.join(ws.design_paths["MPNN/Relaxed"], f"{name}_relaxed.pdb")
        logger.info(f"Step 1: Relaxing structure from {os.path.basename(best_traj)}")
        pr_relax(best_traj, relaxed_pdb)
        logger.info(f"✓ Relaxation complete: {os.path.basename(relaxed_pdb)}")

        # Step 3: Score - Evaluate current state
        logger.info(f"Step 2: Scoring relaxed structure")
        traj_metrics = complex_stats[best_key]
        ws.traj_data = scorer.score_traj(binder_seq, best_traj, relaxed_pdb, traj_metrics, ws.traj_info["binder_chain"])
        
        #FIXME: for rfdiffusion the binder chain is 'A', while the redesigned chain is 'B'
        #SKIP for now
        ref_pdb = os.path.join(ws.design_paths['Trajectory'], f"{name}.pdb")
        ws.traj_data['Hotspot_RMSD'] = 0 # delete if issue fixed
        if os.path.exists(ref_pdb):
            rmsd_site = unaligned_rmsd(ref_pdb, relaxed_pdb, ws.traj_info["binder_chain"], ws.traj_info["binder_chain"])
            ws.traj_data['Hotspot_RMSD'] = rmsd_site
        else:
            logger.info(f"✓ Reference PDB not found: {ref_pdb}, skip Hotspot_RMSD calculation")

        complex_stats[best_key].update(ws.traj_data)

        binder_stats = pdb_handler.predict_binder(binder_seq, ws, best_traj)
        best_binder_key = max(binder_stats, key=lambda k: complex_stats[k].get('plddt', 0))
        best_binder_traj = binder_stats[best_binder_key]['output_pdb']
        binder_stats[best_binder_key]['Binder_RMSD'] = unaligned_rmsd(best_traj, best_binder_traj, ws.traj_info['binder_chain'], 'A')

        seq_notes = validate_design_sequence(binder_seq, complex_stats[best_key].get('Relaxed_Clashes', None), ws.advanced_settings)
        
        mpnn_data = [os.path.basename(relaxed_pdb).replace(".pdb", ""), ws.advanced_settings["design_algorithm"], length, ws.traj_info["seed"], ws.traj_info["helicity_value"], 
                    ws.target_settings["target_hotspot_residues"], binder_seq, ws.traj_data["InterfaceResidues"], mpnn_score, '']

        model_numbers = range(1, 6)
        for label in self.statistics_labels:
            mpnn_data.append(complex_stats[best_key].get(label, None))
            mpnn_data.extend([None for _ in model_numbers]) # skip separate score
                
        for label in ['plddt', 'ptm', 'pae', 'Binder_RMSD']:
            mpnn_data.append(binder_stats[best_binder_key].get(label, None))
            mpnn_data.extend([None for _ in model_numbers]) # skip separate score
        
        elapsed_mpnn_text = ""
        mpnn_data.extend([elapsed_mpnn_text, seq_notes, ws.settings_path, ws.filters_path, ws.advanced_path])
        insert_data(ws.csv_paths["mpnn_csv"], mpnn_data)
        logger.info(f"✓ Scoring complete")

        # Step 3: evaluation confidence and pyrosetta metrics
        pass_filters = check_filters(mpnn_data, ws.settings['design_labels'], ws.settings['filters_settings'])
        if pass_filters == True:
            logger.info(f"✓ {name} passed filters")
            insert_data(ws.csv_paths["final_csv"], ['']+mpnn_data)
            shutil.copy(relaxed_pdb, ws.design_paths['Accepted'])
            return True
        else:
            logger.info(f"✓ {name} failed filters")
            return False

def main():
    # Initialize pipeline
    pipeline = CustomPipeline()
    args = pipeline.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError('Cannot find the csv input file')    
    ws, designer, artifact, criteria = pipeline.init_pipeline(args)

    design_df = pd.read_csv(args.input)

    # Get input PDBs
    pdb_handler = PDBHandler()
    logger.info(f"Parsed sequences: {len(design_df)}")
    accepted = 0

    # parse bb_selected.csv
    for i, row in design_df.iterrows():
        start_time = time.time()
        logger.info(f"Processing {i+1} / {len(design_df)}")
        mpnn_score = row['score']
        binder_seq = row['seq']
        name = row['design']
        if os.path.exists(os.path.join(ws.design_paths['Accepted'], f'{name}_relaxed.pdb')):
            accepted += 1
            continue
        
        scorer = pipeline.update_traj_info(ws, name = name, binder_chain='B', length=len(binder_seq))
        if pipeline.score_MPNN_designs(binder_seq, mpnn_score, ws, pdb_handler, scorer, criteria):
            accepted += 1
        logger.info(f"Success rate {accepted} / {i+1}")
        logger.info(f"Time taken for {name}: {time.time() - start_time:.2f} seconds")

    artifact.rerank_designs()
    pipeline.end_time = time.time()
    logger.info(f"Pipeline complete. Total accepted designs: {accepted} / {len(design_df)}")
    logger.info(f"Total time: {pipeline.end_time - pipeline.pipeline_start_time:.2f} seconds")

if __name__ == "__main__":
    main() 