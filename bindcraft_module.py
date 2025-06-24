#####################################################
## rewrite BindCraft pipeline into functional Modules
## WANG LIYAO, 2025
#####################################################
### Import dependencies
import os, shutil, time, gc
import pandas as pd
import numpy as np
import argparse

from typing import Any, Dict, Tuple

from functions.biopython_utils import *
from functions.pyrosetta_utils import *
from functions.generic_utils import *
from functions.colabdesign_utils import *

import logging

logger = logging.getLogger(__name__)
init_logger = logging.getLogger(f"{__name__}.Initialization")
design_logger = logging.getLogger(f"{__name__}.BinderDesign")
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
init_logger.setLevel(logging.INFO)
design_logger.setLevel(logging.INFO)

def build_design_name(target_settings: Dict[str, Any], length: int, seed: int) -> str:
    """
    Construct a unique design identifier based on settings, trajectory length, and seed.
    """
    name = f"{target_settings.get('binder_name', 'design')}_l{length}_s{seed}"
    return name

def print_summary(total_attempts: int, accepted: int) -> None:
    """
    Print a brief summary of the pipeline results.
    """
    logger.info(f"Total trajectories attempted: {total_attempts}")
    logger.info(f"Total designs accepted: {accepted}")

def sample_trajectory_params(settings: Dict[str, Any]) -> Tuple[int, int]:
    """
    Choose a random seed and trajectory length for the next design.
    """
    target_settings = settings['target_settings']

    seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])
    # Example: pick length from a list in target_settings
    lengths = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1)
    length = np.random.choice(lengths)
    return seed, length

def convert_to_float(value):
    if value is not None and isinstance(value, str):
        try:
            value = float(value) if any(char.isdigit() for char in value.replace('.', '', 1)) else value
        except ValueError:
            pass
    return value

class Initialization:
    """
    Handles argument parsing, input validation, and model loading.
    """
    def __init__(self):
        self.settings_path: str = ''
        self.filters_path: str = ''
        self.advanced_path: str = ''
        self.target_settings: Dict[str, Any] = {}
        self.filters: Dict[str, Any] = {}
        self.advanced_settings: Dict[str, Any] = {}
        self.design_models: Any = None
        self.prediction_models: Any = None
        self.multimer_validation: Any = None

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description='Script to run BindCraft binder design.'
        )
        parser.add_argument(
            '--settings', '-s', type=str, required=True,
            help='Path to the basic settings.json file. Required.'
        )
        parser.add_argument(
            '--filters', '-f', type=str,
            default='./settings_filters/default_filters.json',
            help='Path to the filters.json file used to filter designs.'
        )
        parser.add_argument(
            '--advanced', '-a', type=str,
            default='./settings_advanced/default_4stage_multimer.json',
            help='Path to the advanced.json file with additional design settings.'
        )
        parser.add_argument(
            "--cuda", type=str, default="0",
            help="define the GPU devices"
        )
        args = parser.parse_args()
        if not args.settings:
            init_logger.error("No settings file provided. Use the --settings option to specify the path.")
            parser.print_help()
            exit(1)
        init_logger.debug(f"Parsed arguments: settings={args.settings}, filters={args.filters}, advanced={args.advanced}")
        
        return args

    def run(self) -> Dict[str, Any]:
        # 1. Parse command-line arguments
        args = self.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

        # 2. Validate input files
        (
            self.settings_path,
            self.filters_path,
            self.advanced_path
        ) = perform_input_check(args)

        # 3. Load JSON settings
        (
            self.target_settings,
            self.advanced_settings,
            self.filters
        ) = load_json_settings(
            self.settings_path,
            self.filters_path,
            self.advanced_path
        )

        # 4. Derive simple file-base names (optional)
        self.settings_file = os.path.basename(self.settings_path).split('.')[0]
        self.filters_file = os.path.basename(self.filters_path).split('.')[0]
        self.advanced_file = os.path.basename(self.advanced_path).split('.')[0]

        # 5. Load AF2 model architectures
        (
            self.design_models,
            self.prediction_models,
            self.multimer_validation
        ) = load_af2_models(
            self.advanced_settings.get("use_multimer_design", False)
        )
        init_logger.debug(
            f"Loaded models: design_models={self.design_models}, \
            prediction_models={self.prediction_models}, multimer_validation={self.multimer_validation}"
        )

        # 6. Sanity-check advanced_settings
        bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
        self.advanced_settings = perform_advanced_settings_check(
            self.advanced_settings,
            bindcraft_folder
        )

        return {
            "target_settings": self.target_settings,
            "filters_settings": self.filters,
            "advanced_settings": self.advanced_settings,
            "settings_path": self.settings_path,
            "filters_path": self.filters_path,
            "advanced_path": self.advanced_path,
            "design_models": self.design_models,
            "prediction_models": self.prediction_models,
            "multimer_validation": self.multimer_validation,
        }

class Workspace:
    """
    Sets up output directories, CSV/statistics file paths, and settings for downstream modules.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.target_settings = cfg["target_settings"]
        self.filters_settings = cfg["filters_settings"]
        self.advanced_settings = cfg["advanced_settings"]
        self.settings_path = cfg["settings_path"]
        self.filters_path = cfg["filters_path"]
        self.advanced_path = cfg["advanced_path"]

        self.design_paths: Dict[str, str] = {}
        self.csv_paths: Dict[str, str] = {}
        self.settings: Dict[str, Any] = {}
        self.traj_info: Dict[str, Any] = {}
        self.traj_data: Dict[str, Any] = {}

    def setup(self) -> None:
        # Directories
        self.design_paths = generate_directories(self.target_settings["design_path"])

        # CSV/statistics file paths
        base_dir = self.target_settings["design_path"]
        self.csv_paths = {
            "trajectory_csv": os.path.join(base_dir, 'trajectory_stats.csv'),
            "mpnn_csv": os.path.join(base_dir, 'mpnn_design_stats.csv'),
            "final_csv": os.path.join(base_dir, 'final_design_stats.csv'),
            "failure_csv": os.path.join(base_dir, 'failure_csv.csv'),
        }

        # Generate column labels for each stats CSV
        trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

        # Settings for downstream modules
        self.settings = {
            "target_settings": self.target_settings,
            "advanced_settings": self.advanced_settings,
            "filters_settings": self.filters_settings,
            "settings_path": self.settings_path,
            "filters_path": self.filters_path,
            "advanced_path": self.advanced_path,
            "trajectory_labels": trajectory_labels,
            "design_labels": design_labels,
            "final_labels": final_labels,
        }

    def init_dataframes(self) -> None:
        # Create initial CSV files with headers
        create_dataframe(self.csv_paths["trajectory_csv"], self.settings["trajectory_labels"])
        create_dataframe(self.csv_paths["mpnn_csv"], self.settings["design_labels"])
        create_dataframe(self.csv_paths["final_csv"], self.settings["final_labels"])
        generate_filter_pass_csv(self.csv_paths["failure_csv"], self.filters_path)

class TerminationCriteria:
    def __init__(self, advanced_settings: Dict[str, Any]):
        self.advanced = advanced_settings

    def max_trajectories_reached(self, attempted: int) -> bool:
        max_traj = self.advanced.get("max_trajectories", 100)
        return attempted >= max_traj
    
    def pdb_final_designs_reached(self, design_paths: Dict[str, str], target_settings: Dict[str, Any]) -> bool:
        """
        Check if the number of accepted designs has reached the target.
        """
        accepted_binders = [f for f in os.listdir(design_paths["Accepted"]) if f.endswith('.pdb')]
        max_designs = target_settings.get("number_of_final_designs", 100)
        return len(accepted_binders) >= max_designs

    def check_success_rate(self, accepted: int, attempted: int) -> bool:
        success_rate = self.advanced.get("acceptance_rate", 0.5)
        return accepted / attempted < success_rate

    def quality_check(self, model_pdb_path: str, failure_csv: str) -> bool:
        '''
        Check the quality of the trajectory
        adapted from bindercraft.colabdesign_utils.binder_hallucination
        '''
        ca_clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)

        if ca_clashes > 0:
            update_failures(failure_csv, 'Trajectory_Clashes')
            logger.info("Severe clashes detected, skipping analysis and MPNN optimisation")
            return False
        else:
            # does it have enough contacts to consider?
            binder_contacts = hotspot_residues(model_pdb_path)
            binder_contacts_n = len(binder_contacts.items())

            # if less than 3 contacts then protein is floating above and is not binder
            if binder_contacts_n < 3:
                update_failures(failure_csv, 'Trajectory_Contacts')
                logger.info("Too few contacts at the interface, skipping analysis and MPNN optimisation")
                return False
            else:
                # phew, trajectory is okay! We can continue
                return True

class Scorer:
    def __init__(self, ws: Workspace):
        self.target_settings = ws.settings['target_settings']
        self.advanced_settings = ws.settings['advanced_settings']
        self.filters_settings = ws.settings['filters_settings']
        self.settings_path = ws.settings['settings_path']
        self.filters_path = ws.settings['filters_path']
        self.advanced_path = ws.settings['advanced_path']
        self.design_paths = ws.design_paths
        self.traj_info = ws.traj_info

    def score_traj(self, traj_seq, traj_pdb, traj_relaxed, traj_metrics, binder_chain="B"):
        # Calculate clashes before and after relaxation
        num_clashes_trajectory = calculate_clash_score(traj_pdb)
        num_clashes_relaxed = calculate_clash_score(traj_relaxed)

        # secondary structure content of starting trajectory binder and interface
        traj_alpha, traj_beta, traj_loops, traj_alpha_interface, traj_beta_interface, \
            traj_loops_interface, traj_i_plddt, traj_ss_plddt = \
            calc_ss_percentage(traj_pdb, self.advanced_settings, binder_chain)

        # analyze interface scores for relaxed af2 trajectory
        traj_interface_scores, traj_interface_AA, traj_interface_residues = score_interface(traj_relaxed, binder_chain)

        # starting binder sequence
        #traj_seq = traj.get_seq(get_best=True)[0]

        # analyze sequence
        traj_seq_notes = validate_design_sequence(traj_seq, num_clashes_relaxed, self.advanced_settings)

        # target structure RMSD compared to input PDB
        traj_target_rmsd = target_pdb_rmsd(traj_pdb, self.target_settings["starting_pdb"], self.target_settings["chains"])

        return {
                "design_name": self.traj_info["name"],
                "design_algorithm": self.advanced_settings["design_algorithm"],
                "length": self.traj_info["length"],
                "seed": self.traj_info["seed"],
                "helicity_value": self.traj_info["helicity_value"],
                "target_hotspot_residues": self.target_settings["target_hotspot_residues"],
                "Sequence": traj_seq,
                "InterfaceResidues": traj_interface_residues,

                # following keys are used in mpnn statistics
                "pLDDT": traj_metrics.get('plddt', None),
                "pTM": traj_metrics.get('ptm', None),
                "i_pTM": traj_metrics.get('i_ptm', None),
                "pAE": traj_metrics.get('pae', None),
                "i_pAE": traj_metrics.get('i_pae', None),

                "i_pLDDT": traj_i_plddt,
                "ss_pLDDT": traj_ss_plddt,
                "Unrelaxed_Clashes": num_clashes_trajectory,
                "Relaxed_Clashes": num_clashes_relaxed,
                "Binder_Energy_Score": traj_interface_scores['binder_score'],
                "Surface_Hydrophobicity": traj_interface_scores['surface_hydrophobicity'],
                "ShapeComplementarity": traj_interface_scores['interface_sc'],
                "PackStat": traj_interface_scores['interface_packstat'],
                "dG": traj_interface_scores['interface_dG'],
                "dSASA": traj_interface_scores['interface_dSASA'],
                "dG/dSASA": traj_interface_scores['interface_dG_SASA_ratio'],
                "Interface_SASA_%": traj_interface_scores['interface_fraction'],
                "Interface_Hydrophobicity": traj_interface_scores['interface_hydrophobicity'],
                "n_InterfaceResidues": traj_interface_scores['interface_nres'],
                "n_InterfaceHbonds": traj_interface_scores['interface_interface_hbonds'],
                "InterfaceHbondsPercentage": traj_interface_scores['interface_hbond_percentage'],
                "n_InterfaceUnsatHbonds": traj_interface_scores['interface_delta_unsat_hbonds'],
                "InterfaceUnsatHbondsPercentage": traj_interface_scores['interface_delta_unsat_hbonds_percentage'],
                "Interface_Helix%": traj_alpha_interface,
                "Interface_BetaSheet%": traj_beta_interface,
                "Interface_Loop%": traj_loops_interface,
                "Binder_Helix%": traj_alpha,
                "Binder_BetaSheet%": traj_beta,
                "Binder_Loop%": traj_loops,
                "InterfaceAAs": traj_interface_AA,
                "Target_RMSD": traj_target_rmsd,
                # end

                "trajectory_time_text": self.traj_info["traj_time_text"],
                "seq_notes": traj_seq_notes,
                "settings_file": self.settings_path,
                "filters_file": self.filters_path,
                "advanced_file": self.advanced_path
            }
        
class BinderDesign:
    def __init__(self, design_models: Any, settings: Dict[str, Any], design_paths: Dict[str, str], csv_paths: Dict[str,str]):
        self.design_models = design_models
        self.advanced_settings = settings['advanced_settings']
        self.target_settings = settings['target_settings']
        self.design_paths = design_paths
        self.failure_csv = csv_paths['failure_csv']

        # trajectory design
        self.traj_start_time = None
        self.traj_time_text = None

        # MPNN redesign of starting binder
        self.mpnn_n = 1
        self.accepted_mpnn = 0
        self.mpnn_dict = {}
        self.design_start_time = None

        self.statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                                            'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                                            'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%',
                                            'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD']

    def hallucinate(self, design_name: str, length: int, seed: int, helicity_value) -> Any:
        """
        Generate a binder design trajectory.
        Returns metadata including PDB file paths.
        """
        #design_logger.debug(f"Entering hallucinate: design_name={design_name}, length={length}, seed={seed}, helicity_value={helicity_value}")
        trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
        trajectory_exists = any(os.path.exists(os.path.join(self.design_paths[trajectory_dir], design_name + ".pdb")) for trajectory_dir in trajectory_dirs)

        if not trajectory_exists:
            design_logger.info(f"Starting trajectory: {design_name}")

            ### Begin binder hallucination
            trajectory = binder_hallucination(design_name, self.target_settings["starting_pdb"], self.target_settings["chains"],
                                                self.target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                                self.design_models, self.advanced_settings, self.design_paths, self.failure_csv)
            # time trajectory
            trajectory_time = time.time() - self.traj_start_time
            self.traj_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
            design_logger.info(f"Starting trajectory took: {self.traj_time_text}")
            return trajectory
        else:
            design_logger.info(f"Trajectory {design_name} already exists, skipping...")
            return None
    
    def mpnn_design(self, ws: Workspace, scorer: Scorer, traj_pdb: str) -> Any:
        """
        Generate a binder design using MPNN.
        """
        design_logger.debug(f"Entering mpnn_design for {ws.traj_info['name']}")
        self.mpnn_n = 1
        self.accepted_mpnn = 0
        self.design_start_time = time.time()
        self.mpnn_dict = {}
        length = ws.traj_info["length"]

        ### MPNN redesign of starting binder
        mpnn_traj = mpnn_gen_sequence(traj_pdb, ws.traj_info['binder_chain'], ws.traj_data["InterfaceResidues"], ws.advanced_settings)
        existing_mpnn_sequences = set(pd.read_csv(ws.csv_paths["mpnn_csv"], usecols=['Sequence'])['Sequence'].values)
        
        # create set of MPNN sequences with allowed amino acid composition
        restricted_AAs = set(aa.strip().upper() for aa in ws.advanced_settings["omit_AAs"].split(',')) if ws.advanced_settings["force_reject_AA"] else set()

        mpnn_sequences = sorted({ 
            mpnn_traj['seq'][n][-length:]: {
                'seq': mpnn_traj['seq'][n][-length:],
                'score': mpnn_traj['score'][n],
                'seqid': mpnn_traj['seqid'][n]
            } for n in range(ws.advanced_settings["num_seqs"])
            if (not restricted_AAs or not any(aa in mpnn_traj['seq'][n][-length:].upper() for aa in restricted_AAs))
            and mpnn_traj['seq'][n][-length:] not in existing_mpnn_sequences
        }.values(), key=lambda x: x['score'])

        del existing_mpnn_sequences

        if mpnn_sequences:
            design_logger.debug(f"MPNN sequences generated: {len(mpnn_sequences)}")
            complex_prediction_model, binder_prediction_model = self.prediction_models_prep(ws, traj_pdb)

            # iterate over designed sequences
            for mpnn_sequence in mpnn_sequences:
                design_logger.debug(f"Processing MPNN sequence {self.mpnn_n} for {ws.traj_info['name']}")
                mpnn_time = time.time()

                # generate mpnn design name numbering
                mpnn_design_name = ws.traj_data["design_name"] + "_mpnn" + str(self.mpnn_n)
                mpnn_score = round(mpnn_sequence['score'],2)
                mpnn_seqid = round(mpnn_sequence['seqid'],2)

                self.mpnn_dict[mpnn_design_name] = {'seq': mpnn_sequence['seq'], 'score': mpnn_score, 'seqid': mpnn_seqid, 
                                                'relaxed_pdb':[], "plddt_score": []}
                
                # save fasta sequence
                if ws.advanced_settings["save_mpnn_fasta"] is True:
                    save_fasta(mpnn_design_name, mpnn_sequence['seq'], ws.design_paths)
                
                mpnn_complex_statistics, pass_af2_filters = predict_binder_complex(complex_prediction_model,
                                                                                    mpnn_sequence['seq'], mpnn_design_name, ws.target_settings['starting_pdb'], 
                                                                                    ws.target_settings["chains"],ws.traj_info["length"], traj_pdb, ws.traj_info["prediction_models"], ws.advanced_settings,
                                                                                    ws.filters_settings, ws.design_paths, ws.csv_paths["failure_csv"])

                # if AF2 filters are not passed then skip the scoring
                if not pass_af2_filters:
                    design_logger.info(f"Base AF2 filters not passed for {mpnn_design_name}, skipping interface scoring")
                    self.mpnn_n += 1
                    continue    

                # calculate statistics for each model individually
                for model_num in ws.traj_info["prediction_models"]:
                    mpnn_design_pdb = os.path.join(ws.design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
                    mpnn_design_relaxed = os.path.join(ws.design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
                    self.mpnn_dict[mpnn_design_name]['relaxed_pdb'].append(mpnn_design_relaxed)

                    # score the design
                    mpnn_design_scores = scorer.score_traj(mpnn_sequence['seq'], mpnn_design_pdb, mpnn_design_relaxed, 
                                                    {}, ws.traj_info["binder_chain"]) # leave empty dict here, generate None value for ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE'] keys
                    
                    rmsd_site = unaligned_rmsd(traj_pdb, mpnn_design_pdb, ws.traj_info["binder_chain"], ws.traj_info["binder_chain"])
                    mpnn_design_scores['Hotspot_RMSD'] = rmsd_site
                    mpnn_design_scores = {k:v for k,v in mpnn_design_scores.items() if k in self.statistics_labels[5:]} # first 5 keys will updated later ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE']
                    design_logger.debug(f'MPNN design {mpnn_design_name} model {model_num+1} scores: {mpnn_design_scores}')
                    
                    # add the additional statistics to the mpnn_complex_statistics dictionary
                    mpnn_complex_statistics[model_num+1].update(mpnn_design_scores)
                    
                    if ws.advanced_settings["remove_unrelaxed_complex"]:
                        os.remove(mpnn_design_pdb)
                
                # calculate and update mpnn statistics
                mpnn_complex_averages = calculate_averages(mpnn_complex_statistics, handle_aa=True)
                  
                #predict binder alone
                binder_statistics = predict_binder_alone(binder_prediction_model, mpnn_sequence['seq'], mpnn_design_name, length,
                                                        traj_pdb, ws.traj_info["binder_chain"], ws.traj_info["prediction_models"], ws.advanced_settings,
                                                        ws.design_paths)
                
                # extract RMSDs of binder to the original trajectory
                for model_num in ws.traj_info["prediction_models"]:
                    mpnn_binder_pdb = os.path.join(ws.design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")
                    
                    if os.path.exists(mpnn_binder_pdb):
                        rmsd_binder = unaligned_rmsd(traj_pdb, mpnn_binder_pdb, ws.traj_info["binder_chain"], "A")

                    # append to statistics
                    binder_statistics[model_num+1].update({
                            'Binder_RMSD': rmsd_binder
                        })
                    
                    # save space by removing binder monomer models?
                    if ws.advanced_settings["remove_binder_monomer"]:
                        os.remove(mpnn_binder_pdb)
                
                # calculate binder averages
                binder_averages = calculate_averages(binder_statistics)
                # analyze sequence to make sure there are no cysteins and it contains residues that absorb UV for detection
                seq_notes = validate_design_sequence(mpnn_sequence['seq'], mpnn_complex_averages.get('Relaxed_Clashes', None), ws.advanced_settings)

                mpnn_end_time = time.time() - mpnn_time
                elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(mpnn_end_time // 3600), int((mpnn_end_time % 3600) // 60), int(mpnn_end_time % 60))}"

                model_numbers = range(1, 6)
                mpnn_data = [mpnn_design_name, ws.advanced_settings["design_algorithm"], length, ws.traj_info["seed"], ws.traj_info["helicity_value"], 
                             ws.target_settings["target_hotspot_residues"], mpnn_sequence['seq'], ws.traj_data["InterfaceResidues"], mpnn_score, mpnn_seqid]

                for label in self.statistics_labels:
                    mpnn_data.append(mpnn_complex_averages.get(label, None))
                    for model in model_numbers:
                        mpnn_data.append(mpnn_complex_statistics.get(model, {}).get(label, None))
                
                for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:
                    mpnn_data.append(binder_averages.get(label, None))
                    for model in model_numbers:
                        mpnn_data.append(binder_statistics.get(model, {}).get(label, None))

                mpnn_data.extend([elapsed_mpnn_text, seq_notes, ws.settings_path, ws.filters_path, ws.advanced_path])
                insert_data(ws.csv_paths["mpnn_csv"], mpnn_data)

                # find best model number by pLDDT
                plddt_values = {i: mpnn_data[i] for i in range(11, 15) if mpnn_data[i] is not None}
                self.mpnn_dict[mpnn_design_name]['plddt_score'] = plddt_values
                highest_plddt_key = int(max(plddt_values, key=plddt_values.get))
                best_model_number = highest_plddt_key - 10
                best_model_pdb = os.path.join(ws.design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{best_model_number}.pdb")

                # run design data against filter thresholds
                filter_conditions = check_filters(mpnn_data, ws.settings["design_labels"], ws.filters_settings)
                if filter_conditions == True:
                    design_logger.info(f"MPNN design {mpnn_design_name} passed all filters")
                    self.accepted_mpnn += 1

                    # copy designs to accepted folder
                    shutil.copy(best_model_pdb, ws.design_paths["Accepted"])
                    
                    # insert data into final csv
                    final_data = [''] + mpnn_data
                    insert_data(ws.csv_paths["final_csv"], final_data)

                    # copy animation from accepted trajectory
                    if ws.advanced_settings["save_design_animations"]:
                        accepted_animation = os.path.join(ws.design_paths["Accepted/Animation"], f"{ws.traj_info['name']}.html")
                        if not os.path.exists(accepted_animation):
                            shutil.copy(os.path.join(ws.design_paths["Trajectory/Animation"], f"{ws.traj_info['name']}.html"), accepted_animation)

                    # copy plots of accepted trajectory
                    plot_files = os.listdir(ws.design_paths["Trajectory/Plots"])
                    plots_to_copy = [f for f in plot_files if f.startswith(ws.traj_info['name']) and f.endswith('.png')]
                    for accepted_plot in plots_to_copy:
                        source_plot = os.path.join(ws.design_paths["Trajectory/Plots"], accepted_plot)
                        target_plot = os.path.join(ws.design_paths["Accepted/Plots"], accepted_plot)
                        if not os.path.exists(target_plot):
                            shutil.copy(source_plot, target_plot)
                else:
                    design_logger.info(f"Unmet filter conditions for {mpnn_design_name}")
                    failure_df = pd.read_csv(ws.csv_paths["failure_csv"])
                    special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
                    incremented_columns = set()
                    
                    for column in filter_conditions:
                        base_column = column
                        for prefix in special_prefixes:
                            if column.startswith(prefix):
                                base_column = column.split('_', 1)[1]

                        if base_column not in incremented_columns:
                            failure_df[base_column] = failure_df[base_column] + 1
                            incremented_columns.add(base_column)
                    
                    failure_df.to_csv(ws.csv_paths["failure_csv"], index=False)
                    shutil.copy(best_model_pdb, ws.design_paths["Rejected"])

                self.mpnn_n += 1

                if self.accepted_mpnn >= ws.advanced_settings["max_mpnn_sequences"]:
                    design_logger.debug(f"Max MPNN designs reached for {ws.traj_info['name']}, skipping to next trajectory.")
                    break

            if self.accepted_mpnn >= 1:
                design_logger.info(f"Found {self.accepted_mpnn} MPNN designs passing filters\n")
            else:
                design_logger.info("No accepted MPNN designs found for this trajectory.\n")
        
        else:
            design_logger.info("Duplicate MPNN designs sampled with different trajectory, skipping current trajectory optimisation\n")

        if ws.advanced_settings["remove_unrelaxed_trajectory"]:
            os.remove(traj_pdb)
        design_time = time.time() - self.design_start_time
        design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
        design_logger.info(f"Design and validation of trajectory {ws.traj_info['name']} took: {design_time_text}")

        return self.accepted_mpnn, self.mpnn_dict

    def prediction_models_prep(self, ws: Workspace, traj_pdb: str) -> Tuple[Any, Any]:
        # add optimisation for increasing recycles if trajectory is beta sheeted
        if ws.advanced_settings["optimise_beta"] and float(ws.traj_data["Binder_BetaSheet%"]) > 15:
            ws.advanced_settings["num_recycles_validation"] = ws.advanced_settings["optimise_beta_recycles_valid"]
        
        clear_mem()

        # compile complex prediction model
        complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=ws.advanced_settings["num_recycles_validation"], data_dir=ws.advanced_settings["af_params_dir"], 
                                                    use_multimer=ws.traj_info["mutlimer_validation"], use_initial_guess=ws.advanced_settings["predict_initial_guess"], use_initial_atom_pos=ws.advanced_settings["predict_bigbang"])
        
        if ws.advanced_settings["predict_initial_guess"] or ws.advanced_settings["predict_bigbang"]:
            complex_prediction_model.prep_inputs(pdb_filename=traj_pdb, chain='A', binder_chain=ws.traj_info["binder_chain"], binder_len=ws.traj_info["length"], use_binder_template=True, rm_target_seq=ws.advanced_settings["rm_template_seq_predict"],
                                                rm_target_sc=ws.advanced_settings["rm_template_sc_predict"], rm_template_ic=True)
        else:
            complex_prediction_model.prep_inputs(pdb_filename=ws.target_settings["starting_pdb"], chain=ws.target_settings["chains"], binder_len=ws.traj_info["length"], rm_target_seq=ws.advanced_settings["rm_template_seq_predict"],
                                                rm_target_sc=ws.advanced_settings["rm_template_sc_predict"])
        
        # compile binder monomer prediction model
        binder_prediction_model = mk_afdesign_model(protocol="hallucination", use_templates=False, initial_guess=False, 
                                                    use_initial_atom_pos=False, num_recycles=ws.advanced_settings["num_recycles_validation"], 
                                                    data_dir=ws.advanced_settings["af_params_dir"], use_multimer=ws.traj_info["mutlimer_validation"])
        binder_prediction_model.prep_inputs(length=ws.traj_info["length"])
        
        return complex_prediction_model, binder_prediction_model
    
    def rfdiffusion_design(self, mpnn_data, design_labels, filters):
        raise NotImplementedError("RFdiffusion design not implemented yet")
    
    def dlpm_design(self, mpnn_data, design_labels, filters):
        raise NotImplementedError("DLPM design not implemented yet")
    
    def hallucinate_scaffolding(self, mpnn_data, design_labels, filters):
        raise NotImplementedError("Scaffolding design not implemented yet")

    def rfdiffusion_scaffolding(self, mpnn_data, design_labels, filters):
        raise NotImplementedError("RFdiffusion scaffolding design not implemented yet")
    
    def dlpm_scaffolding(self, mpnn_data, design_labels, filters):
        raise NotImplementedError("DLPM scaffolding design not implemented yet")

class ArtifactHandler:
    def __init__(self, design_paths: Dict[str, str], settings: Dict[str, Any], csv_paths: Dict[str, str]) -> None:
        self.design_paths = design_paths
        self.settings = settings
        self.csv_paths = csv_paths

    def rerank_designs(self) -> None:
        """
        Rerank accepted designs and update the final CSV and Ranked folder.
        """
        design_logger.debug("Reranking accepted designs and updating final CSV.")
        design_paths = self.design_paths
        mpnn_csv = self.csv_paths["mpnn_csv"]
        final_labels = self.settings["final_labels"]
        final_csv = self.csv_paths["final_csv"]
        advanced_settings = self.settings["advanced_settings"]
        target_settings = self.settings["target_settings"]
        design_labels = self.settings["design_labels"]

        accepted_binders = [f for f in os.listdir(design_paths["Accepted"]) if f.endswith('.pdb')]

        # Clear the Ranked folder
        for f in os.listdir(design_paths["Accepted/Ranked"]):
            os.remove(os.path.join(design_paths["Accepted/Ranked"], f))

        # Load dataframe of designed binders
        design_df = pd.read_csv(mpnn_csv)
        design_df = design_df.sort_values('Average_i_pTM', ascending=False)

        # Create final csv dataframe to copy matched rows, initialize with the column labels
        final_df = pd.DataFrame(columns=final_labels)

        # Check the ranking of the designs and copy them with new ranked IDs to the folder
        rank = 1
        for _, row in design_df.iterrows():
            for binder in accepted_binders:
                binder_name, model = binder.rsplit('_model', 1)
                if binder_name == row['Design']:
                    # rank and copy into ranked folder
                    row_data = {'Rank': rank, **{label: row[label] for label in design_labels}}
                    final_df = pd.concat([final_df, pd.DataFrame([row_data])], ignore_index=True)
                    old_path = os.path.join(design_paths["Accepted"], binder)
                    new_path = os.path.join(design_paths["Accepted/Ranked"], f"{rank}_{target_settings['binder_name']}_model{model.rsplit('.', 1)[0]}.pdb")
                    shutil.copyfile(old_path, new_path)
                    rank += 1
                    break

        # Save the final_df to final_csv
        final_df.to_csv(final_csv, index=False)

        # Zip large folders to save space
        if advanced_settings.get("zip_animations", False):
            zip_and_empty_folder(design_paths["Trajectory/Animation"], '.html')
        if advanced_settings.get("zip_plots", False):
            zip_and_empty_folder(design_paths["Trajectory/Plots"], '.png')

class BindCraftPipeline:
    def __init__(self, cfg: Dict[str, Any]):        
        # unused but remain
        self.design_models = cfg["design_models"]
        self.prediction_models = cfg["prediction_models"]
        self.multimer_validation = cfg["multimer_validation"]
        self.script_dir = os.getcwd()

        # Initialize statistics
        self.script_start_time = time.time()
        self.trajectory_n = 1
        self.accepted = 0

        self.ws = Workspace(cfg)
        self.ws.setup()
        self.ws.init_dataframes()

    def run(self) -> None:
        # Workspace
        ws = self.ws

        # Components
        criteria = TerminationCriteria(ws.settings['advanced_settings'])
        artifact = ArtifactHandler(ws.design_paths, ws.settings, ws.csv_paths)
        designer = BinderDesign(self.design_models, ws.settings, ws.design_paths, ws.csv_paths)
        # initialize pyrosetta
        pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {ws.advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1')
        design_logger.info(f"Running binder design for target {ws.settings_path}")
        design_logger.info(f"Design settings used: {ws.advanced_path}")
        design_logger.info(f"Filtering designs based on {ws.filters_path}")
        design_logger.debug("Starting main pipeline run loop.")

        # Main Loop
        while True:
            design_logger.debug(f"Trajectory loop: trajectory_n={self.trajectory_n}, accepted={self.accepted}")

            if criteria.pdb_final_designs_reached(ws.design_paths, ws.settings['target_settings']):
                design_logger.debug("Final designs reached, reranking and exiting loop.")
                artifact.rerank_designs()
                break
            if criteria.max_trajectories_reached(self.trajectory_n):
                design_logger.debug("Max trajectories reached, exiting loop.")
                break
            
            designer.traj_start_time = time.time()

            seed, length = sample_trajectory_params(ws.settings)
            name = build_design_name(ws.settings['target_settings'], length, seed)
            helicity_value = load_helicity(ws.settings['advanced_settings'])
            design_logger.debug(f"Sampled trajectory params: name={name}, length={length}, seed={seed}, helicity_value={helicity_value}")

            # Design step
            traj = designer.hallucinate(name, length, seed, helicity_value)
            if traj is None: # trajectory already exists
                design_logger.debug(f"Trajectory {name} already exists, skipping to next.")
                break

            # update ws.traj info
            ws.traj_info = {
                "name": name,
                "length": length,
                "seed": seed,
                "helicity_value": helicity_value,
                "traj_time_text": designer.traj_time_text,
                "binder_chain": "B",
                "prediction_models": self.prediction_models,
                "mutlimer_validation": self.multimer_validation,
                "design_models": self.design_models
            }
            traj_scorer = Scorer(ws) # refresh scorer.traj_info in every trajectory

            traj_metrics = copy_dict(traj._tmp["best"]["aux"]["log"]) # contains plddt, ptm, i_ptm, pae, i_pae
            traj_pdb = os.path.join(ws.design_paths["Trajectory"], name + ".pdb")
            # round the metrics to two decimal places
            traj_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in traj_metrics.items()}
            
            # Proceed if there is no trajectory termination signal
            if traj.aux["log"]["terminate"] == "":
                # Relax binder to calculate statistics
                traj_relaxed = os.path.join(ws.design_paths["Trajectory/Relaxed"], name + ".pdb")

                design_logger.debug(f"Relaxed trajectory {name}, proceeding to scoring.")                
                pr_relax(traj_pdb, traj_relaxed)

                design_logger.debug(f"Scoring trajectory {name}.")
                ws.traj_data = traj_scorer.score_traj(traj.get_seq(get_best=True)[0], traj_pdb, traj_relaxed, traj_metrics, ws.traj_info["binder_chain"])
                
                insert_data(ws.csv_paths["trajectory_csv"], ws.traj_data.values())
                design_logger.debug(f"Scored trajectory {name}, proceeding to MPNN if enabled.")

                if ws.advanced_settings["enable_mpnn"]:
                    accepted_mpnn, _ = designer.mpnn_design(ws, traj_scorer, traj_pdb)
                    self.accepted += accepted_mpnn
                    design_logger.debug(f"MPNN design for {name} complete. accepted_mpnn={accepted_mpnn}, total accepted={self.accepted}")

                if self.trajectory_n >= ws.advanced_settings["start_monitoring"] and ws.advanced_settings["enable_rejection_check"]:
                    if criteria.check_success_rate(self.accepted, self.trajectory_n):
                        design_logger.info("The ratio of successful designs is lower than defined acceptance rate! Consider changing your design settings!")
                        design_logger.info("Script execution stopping...")
                        break
            else:
                design_logger.debug(f"Trajectory {name} terminated early due to {traj.aux['log']['terminate']}, skipping.")

            design_logger.debug(f"End of trajectory loop iteration. trajectory_n={self.trajectory_n}")
            self.trajectory_n += 1
            gc.collect()

def main():
    initializer = Initialization()
    cfg = initializer.run()
    pipeline = BindCraftPipeline(cfg)
    pipeline.run()

if __name__ == "__main__":
    main()