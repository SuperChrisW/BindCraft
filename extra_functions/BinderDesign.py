from typing import Any, Dict, Tuple
import os
import shutil
import time
import pandas as pd
from functions.colabdesign_utils import *
from functions.generic_utils import *
from functions.pyrosetta_utils import *

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

        self.statistics_labels = statistics_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
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

                self.mpnn_dict[mpnn_design_name] = {'seq': mpnn_sequence['seq'], 'score': mpnn_score, 'seqid': mpnn_seqid}
                
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

        return self.accepted_mpnn

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