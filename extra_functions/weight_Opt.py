import os
import json
import sys
import numpy as np
import ultraimport
from functions.generic_utils import load_json_settings

bindcraft_pipeline = ultraimport('__dir__/bindcraft_customize.py', 'bindcraft_pipeline')

def opt_weight(weight='pTMEnergy', test_values=None, **kwargs):
    """
    Optimize the weight for a given metric by testing various values and selecting the best one.
    
    Args:
        weight (str): The name of the weight to optimize.
        test_values (list, optional): A list of values to test for the weight. If None, defaults to [0.01, 0.1, 1.0].
    
    Returns:
        float: The best weight value found.
    """
    if test_values is None:
        test_values = np.arange(0, 1.1, 0.1)  # Default test values from 1 to 10
    
    best_value = None
    best_score = float('-inf')
    traj_scores = []
    binder_name = kwargs['settings']['binder_name']
    for value in test_values:
        # Simulate a scoring function that evaluates the performance with the given weight
        print(f"Running BindCraft with weight '{weight}' set to {value} and sample number {kwargs.get('sample_num', 20)}.")
        score = run_bindcraft(weight, value, **{**kwargs, 'binder_name': binder_name})
        traj_scores.append(score)
        if score > best_score:
            best_score = score
            best_value = value
    
    return best_value, traj_scores

def run_bindcraft(weight, value, **kwargs):
    """
    Run the BindCraft optimization with the specified weight and value.
    
    Args:
        weight (str): The name of the weight to optimize.
        value (float): The value of the weight to test.
        sample_num (int): The number of samples to use for the optimization.
    
    Returns:
        dict: A dictionary containing the results of the optimization.
    """
    if 'settings' not in kwargs or 'filters' not in kwargs or 'advanced' not in kwargs:
        raise ValueError("Missing required keys in kwargs: 'settings', 'filters', 'advanced'")

    # modify settings 
    sample_num = kwargs.get('sample_num', 20)
    kwargs['settings']['design_path'] = f"./weight_Opt/{kwargs['binder_name']}_{weight}_{value}"
    kwargs['settings']['number_of_final_designs'] = sample_num
    kwargs['advanced'][f'weights_{weight}'] = value

    # run pipeline
    try:
        bindcraft_pipeline(
            target_settings=kwargs['settings'],
            filters=kwargs['filters'],
            advanced_settings=kwargs['advanced'],
            args= kwargs['args']
        )
    except Exception as e:
        print(f"Error running BindCraft pipeline: {e}")
        raise RuntimeError("BindCraft pipeline failed to run.")

    # compute score
    #score = compute_success_rate(kwargs['settings']['design_path'])

    return 0

def compute_success_rate(design_path):

    return 0

if __name__ == "__main__":
    # Load settings from a JSON file
    import argparse
    parser = argparse.ArgumentParser(description='Script to run BindCraft binder design.')
    parser.add_argument('--settings', '-s', type=str, default='./settings_target/IL23.json',
                        help='Path to the basic settings.json file. Required.')
    parser.add_argument('--filters', '-f', type=str, default='./settings_filters/no_filters.json',
                        help='Path to the filters.json file used to filter design. If not provided, default will be used.')
    parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_multimer_pTME.json',
                        help='Path to the advanced.json file with additional design settings. If not provided, default will be used.')
    args = parser.parse_args()

    target_settings, advanced_settings, filters = load_json_settings(args.settings, args.filters, args.advanced)

    # Example usage of opt_weight
    test_values = np.arange(0.55, 1.05, 0.05)
    settings = {
        'settings': target_settings,
        'filters': filters,
        'advanced': advanced_settings,
        'sample_num': 20,
        'args': args,
    }
    best_weight, traj_scores = opt_weight(weight='pTMEnergy', test_values=test_values, **settings)
    print(f"Best weight value: {best_weight}")
