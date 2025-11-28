import numpy as np
from pulp import *
from copy import deepcopy
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
# import scipy # Make sure scipy is imported if used in the original score logic
import torch
import torch.nn as nn
# Assume BIT_WIDTHS is defined globally or passed appropriately
from SMI.utils.mqbench_layer_to_block import conv_to_block_output_dict


def model_size_wrt_bitwidth(model, bit_width, quant_dict_base, last_layer_bw = 8):
    total_params_in_specified_layers = 0
    found_layers_params = {}
    # Iterate through all named modules in the model
    for name, module in model.named_modules():
        if name in quant_dict_base:
            layer_params = 0
            # Iterate through parameters of the current module
            for param in module.parameters():
                if param.requires_grad:
                    layer_params += param.numel()
            found_layers_params[name] = layer_params
            total_params_in_specified_layers += layer_params

    #add fully connected layer called as fc or classifier

    # Calculate the size in MB in dict
    size_in_mb = (total_params_in_specified_layers * bit_width) / (8 * 1024 * 1024)  # Convert bits to MB


    if hasattr(model, 'fc'):
        layer_params = 0
        for param in model.fc.parameters():
            if param.requires_grad:
                layer_params += param.numel()
        found_layers_params['fc'] = layer_params
        total_params_in_specified_layers += layer_params
        size_in_mb += (layer_params * last_layer_bw) / (8 * 1024 * 1024)  # Add the last layer size in MB
    elif hasattr(model, 'classifier'):
        layer_params = 0
        for param in model.classifier.parameters():
            if param.requires_grad:
                layer_params += param.numel()
        found_layers_params['classifier'] = layer_params
        total_params_in_specified_layers += layer_params
        size_in_mb += (layer_params * last_layer_bw) / (8 * 1024 * 1024)  # Add the last layer size in MB
        
    for layer_name, params in found_layers_params.items():
        if layer_name == 'fc' or layer_name == 'classifier':
            # For the last layer, use the specified bit width
            found_layers_params[layer_name] = (params * last_layer_bw) / (8 * 1024 * 1024)
        else:
            found_layers_params[layer_name] = (params * bit_width) / (8 * 1024 * 1024)

    return size_in_mb, found_layers_params

def checkpoint_sizes(params):
    FP_size = sum((np.array(params) * 4) / (1024*1024)) #32 bit
    eight_bit_size = FP_size / 4
    four_bit_size = FP_size / 8
    seven_bit_size = (FP_size / (32))*7
    six_bit_size = (FP_size / (32))*6
    five_bit_size = (FP_size / (32))*5
    three_bit_size = (FP_size / (32))*3
    two_half_bit_size = (FP_size / (32))*2.5
    two_bit_size = FP_size / 16
    x12_compression_size = FP_size / 12
    print(f"FP size: {FP_size} MB")
    print(f"8 bit size: {eight_bit_size} MB")
    print(f"7 bit size: {seven_bit_size} MB")
    print(f"6 bit size: {six_bit_size} MB")
    print(f"5 bit size: {five_bit_size} MB")
    print(f"4 bit size: {four_bit_size} MB")
    print(f"3 bit size: {three_bit_size} MB")
    print(f"2.5 bit size: {two_half_bit_size} MB")
    print(f"2 bit size: {two_bit_size} MB")
    print(f"X12 compression size: {x12_compression_size} MB")

    return {
        '32': FP_size,
        '8': eight_bit_size,
        '7': seven_bit_size,
        '6': six_bit_size,
        '5': five_bit_size,
        '4': four_bit_size,
        '3': three_bit_size,
        '2.5': two_half_bit_size,
        '2': two_bit_size,
        'x12_compression_size': x12_compression_size
    }



def total_params(model, conv_layer_names):
    print(f"{'Layer Name':<30} | {'Parameters':<15}")
    print("-" * 48)

    total_params_in_specified_layers = 0
    found_layers_params = {}

    # Iterate through all named modules in the model
    for name, module in model.named_modules():
        if name in conv_layer_names:
            layer_params = 0
            # Iterate through parameters of the current module
            for param in module.parameters():
                if param.requires_grad: # Usually, we count trainable parameters
                    layer_params += param.numel()
            
            found_layers_params[name] = layer_params
            total_params_in_specified_layers += layer_params

    # Print parameters for the specified layers in the requested order
    for layer_name in conv_layer_names:
        if layer_name in found_layers_params:
            print(f"{layer_name:<30} | {found_layers_params[layer_name]:<15,}")
        else:
            print(f"{layer_name:<30} | Layer not found or no parameters")

    print("-" * 48)
    print(f"{'Total params in specified layers':<30} | {total_params_in_specified_layers:<15,}")

    # For verification, calculate total model parameters
    total_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'Total model parameters':<30} | {total_model_params:<15,}")

    return found_layers_params


def get_param_size(model: nn.Module, layer_name: str, bit_width: int) -> int:
    """
    Calculate the memory size required for parameters of a specific layer.
    
    Args:
        model: The SqueezeNet model
        layer_name: Name of the layer (e.g., 'features.3.squeeze')
        bit_width: Number of bits used to represent each parameter
        
    Returns:
        int: Total memory size in bits (number of parameters * bit_width)
    """
    # Parse the layer name to navigate the model hierarchy
    parts = layer_name.split('.')
    
    # Navigate to the specified module
    current_module = model
    for part in parts:
        current_module = getattr(current_module, part)
    
    # Count parameters in the module
    num_params = sum(p.numel() for p in current_module.parameters())
    
    # Calculate total bits
    total_bits = num_params * bit_width
    
    return total_bits

# ---------------------------------------------
# 1. Function to calculate scores (Remains the same)
# ---------------------------------------------
def get_smi_scores(layer_bit_dict, column_names, score_df, XL_CORR=None, LY_CORR=None,model_name='mobilenetv2', dataset='imagenet', BIT_WIDTHS=[8, 4, 2]):

    layers = list(layer_bit_dict.keys())
    scores = {} # {layer_index: {bit_width: score}}

    # Fallback logic for 8-bit 'all' layer
    eight_bit_row = score_df[
        (score_df['layer'] == 'all') & (score_df['bw'] == 8)
    ]

    # Score column filtering (adapt as needed)
    score_columns = []
    for col in column_names:
        if 'mi_ly' in col or 'mi_xl' in col:
            score_columns.append(col)
    if not score_columns:
        print("Warning: No score columns identified. Using 'acc' if available.")
        if 'acc' in score_df.columns:
            score_columns = ['acc']
        else:
             print("Warning: 'acc' column not found either. Scores might be zero.")
             score_columns = []


    for i, layer in enumerate(layers):
        scores[i] = {}
        for b in BIT_WIDTHS:
            row_to_use = score_df[
                (score_df['layer'] == layer) & (score_df['bw'] == b)
            ]
            if row_to_use.empty:
                #print(f"Warning: No score data for layer {layer}, bw {b}. Assigning zero score.")
                scores[i][b] = 0.0
                continue

            current_row_data = row_to_use.iloc[0]
            eight_bit_data = eight_bit_row.iloc[0]

            # --- Detailed score calculation logic (same as before) ---
            XL, XL_8, LY, LY_8 = 0.0, 0.0, 0.0, 0.0 # Initialize as floats
            # Add other variables needed: ss, j_denom, ss_xl, j_xl_denom, etc.
            ss, j_denom, ss_xl, j_xl_denom = 0.0, 0.0, 0.0, 0.0
            total_ly, eight_bit_ly, total_xl, eight_bit_xl = 0.0, 0.0, 0.0, 0.0

            start_col_index = 0 # Default start index
            try:
                if model_name == 'mobilenetv2_x0_5' and dataset == 'cifar10':
                    layer_base_parts = layer.split('.')[:2]
                    layer_prefix_search = '.'.join(layer_base_parts) + '_'
                elif model_name == 'resnet20' and dataset == 'cifar10':
                    layer_base_parts = layer.split('.')[:2]
                    layer_prefix_search = '.'.join(layer_base_parts) + '_'
                elif model_name == 'resnet18_gluon' and dataset == 'imagenet_256x256':
                    layer_base_parts = layer.split('.')[:2]
                    layer_prefix_search = '.'.join(layer_base_parts) + '_'
                elif model_name == 'resnet34_gluon' and dataset == 'imagenet_256x256':
                    layer_base_parts = layer.split('.')[:2]
                    layer_prefix_search = '.'.join(layer_base_parts) + '_'
                elif model_name == 'resnet50_gluon' and dataset == 'imagenet_256x256':
                    layer_base_parts = layer.split('.')[:2]
                    layer_prefix_search = '.'.join(layer_base_parts) + '_'
                elif model_name == 'mobilenetv2' and dataset == 'imagenet_256x256':
                    if layer == 'features.0':
                        layer_prefix_search =  'features.2_'
                    elif layer == 'conv.0':
                        layer_prefix_search =  'avgpool_'
                    else:
                        layer_base_parts = layer.split('.')[:2]
                       
                        layer_prefix_search = '.'.join(layer_base_parts) + '_'
                        if layer_prefix_search == 'features.1_' or layer_prefix_search == 'features.0_':
                            layer_prefix_search = 'features.2_'
                    #print(f"Layer prefix search for {layer}: {layer_prefix_search}")
                start_col_index = -1
                for idx, col_name in enumerate(score_columns):
                    if col_name.startswith(layer_prefix_search):
                        start_col_index = idx
                        break
                if start_col_index == -1: start_col_index = 0
            except Exception as e: 
                print(f"Error finding start column index for layer {layer}: {e}")
                start_col_index = 0
            #print(f"Start column index for layer {layer}: {start_col_index}")
            layer_to_block_dict = conv_to_block_output_dict(model_name, dataset)
            start_col_index = score_columns.index(layer_to_block_dict[layer]+'_mi_ly')
#[6, 4, 5, 6, 6, 4, 4, 5, 3, 4, 4, 5, 5, 4, 3, 4, 4, 4, 5, 4, 4, 2, 3, 3, 4, 5, 5, 6, 2, 5, 4, 3, 3, 4, 5, 5, 5, 4, 5, 6, 4, 4, 6, 3, 4, 4, 5, 5, 4, 6, 6, 5]
#[6, 4, 5, 6, 6, 4, 4, 5, 3, 4, 4, 5, 5, 4, 3, 4, 4, 4, 5, 4, 4, 2, 3, 3, 4, 5, 5, 6, 2, 5, 4, 3, 3, 4, 5, 5, 5, 4, 5, 6, 4, 4, 6, 3, 4, 4, 5, 5, 4, 6, 6, 5]       
            # --------- XL Score calculation ---------- #
            if model_name == 'mobilenetv2_x0_5' and dataset == 'cifar10':
                START_XL = 4
                END_XL = 26
            elif model_name == 'resnet20' and dataset == 'cifar10':
                START_XL = 6
                END_XL = 12
            elif model_name == 'resnet18_gluon' and dataset == 'imagenet_256x256':
                START_XL = 8
                END_XL = 14
            elif model_name == 'resnet34_gluon' and dataset == 'imagenet_256x256':
                START_XL = 2
                END_XL = 6
            elif model_name == 'resnet50_gluon' and dataset == 'imagenet_256x256':
                START_XL = 2
                END_XL = 14
            elif model_name == 'mobilenetv2' and dataset == 'imagenet_256x256':
                START_XL = 2
                END_XL = 32 
        
            range_start_xl = max(START_XL, start_col_index) # Original logic
            range_end_xl = min(END_XL, len(score_columns)) # Original logic, ensure bounds
            def clamp(n, min_value, max_value):
                return max(min_value, min(n, max_value))
            for l_idx in range(range_start_xl, range_end_xl, 2):
                corr_idx = l_idx // 2
                lay = score_columns[l_idx]
                # Safely get xl values
                current_xl_val = current_row_data.get(lay.replace('_mi_ly', '_mi_xl'), 0.0)
                eight_bit_xl_val = eight_bit_data.get(lay.replace('_mi_ly', '_mi_xl'), 0.0)

                XL += abs(current_xl_val - eight_bit_xl_val)#clamp(current_xl_val - eight_bit_xl_val, 0, float('inf')) * abs(XL_CORR[corr_idx]) # Difference weighted by XL_CORR
                XL_8 += eight_bit_xl_val #* abs(XL_CORR[corr_idx]) # Weighted 8-bit value

            # --------- LY Score calculation ---------- #
            if model_name == 'mobilenetv2_x0_5':
                START_LY = 26
            elif model_name == 'resnet20':
                START_LY = 16
            elif model_name == 'resnet18_gluon':
                START_LY = 14
            elif model_name == 'resnet34_gluon':
                START_LY = 26
            elif model_name == 'resnet50_gluon':
                START_LY = 26
            elif model_name == 'mobilenetv2':
                START_LY = 34
            range_start_ly = max(start_col_index, START_LY) # Original logic
            range_end_ly = len(score_columns) # Original logic

            for l_idx in range(range_start_ly, range_end_ly, 2):
                corr_idx = l_idx // 2

                lay = score_columns[l_idx]
                current_val = current_row_data.get(lay, 0.0)
                eight_bit_val = eight_bit_data.get(lay, 0.0)

                LY += abs(eight_bit_val-current_val) #clamp(eight_bit_val-current_val, 0, float('inf')) #* LY_CORR[corr_idx] #abs(eight_bit_val-current_val) * LY_CORR[corr_idx]#
                LY_8 += eight_bit_val #* LY_CORR[corr_idx]


            # Final Score Calculation
            denom_ibopt = XL_8 + LY_8
            if denom_ibopt != 0:
                # Ensure calculations result in float
                #use minus when we maximize the scores, + when we minimize
                #we recommend, change the signs after function is called, not here!!
            
                score_value = -(float(XL) + float(LY)) / float(denom_ibopt)
                # Add checks for NaN or Inf if divisions can produce them
                if np.isnan(score_value) or np.isinf(score_value):
                        print(f"Warning: Score for layer {layer} bw {b} is NaN or Inf. Setting to 0.")
                        scores[i][b] = 0.0
                else:
                        scores[i][b] = score_value * (1/b)

            else:
                # print(f"Warning: Division by zero avoided in IBOPT calc for layer {layer}, bw {b}. Score set to 0.")
                scores[i][b] = 0.0
    print('SMI Scores Calculated')
    return scores


# ---------------------------------------------
# 2. Function to calculate parameter sizes (Remains the same)
# ---------------------------------------------
def get_param_constraint(layer_bit_dict, params, BIT_WIDTHS=[8, 4, 2]):
    """
    Calculates the parameter size for each layer and bit width.
    (Implementation is the same as the previous version)
    """
    layers = list(layer_bit_dict.keys())
    param_sizes = {} # {layer_index: {bit_width: size}}

    #ensure number of layers is same as params
    if len(layers) != len(params):
        raise ValueError("Mismatch between number of layers and parameter sizes.")
    params = [(i*4)/(1024*1024) for i in params]
    
    for i, layer in enumerate(layers):
        param_sizes[i] = {}
        for b in BIT_WIDTHS:
            # Ensure the result is a float for PuLP
            #convert to MB
            param_sizes[i][b] = float((params[i] * (b/32) ))
            # Ensure float for PuLP
    return param_sizes

# ---------------------------------------------
# 3. Modified main optimization function (Now takes scores/params directly)
# ---------------------------------------------
def solve_bit_width_optimization_single(
    quant_dict_base,         # Still needed for layer names/indices
    scores,                  # Pre-calculated scores passed in
    param_sizes,             # Pre-calculated sizes passed in
    max_param_size=None,
    min_score=None,
    existing_prob=None,
    constraints_to_add=None,
    BIT_WIDTHS=[8, 4, 2]  # Default bit widths, can be overridden
):
    """
    Solves a single instance of the bit-width optimization problem.
    Takes pre-calculated scores and param_sizes dictionaries as input.
    """
    layers = list(quant_dict_base.keys()) # Needed for mapping indices
    n_layers = len(layers)

    if existing_prob is None:
        # --- Initial Setup ---
        if max_param_size is not None:
            prob = LpProblem("BitWidthOptimization", LpMaximize)
            objective_type = "score"
        elif min_score is not None:
            prob = LpProblem("BitWidthOptimization", LpMinimize)
            objective_type = "param_size"
        else:
            raise ValueError("Either max_param_size or min_score must be provided")

        # Create binary decision variables
        x = {}
        for i in range(n_layers):
            x[i] = {}
            for b in BIT_WIDTHS:
                x[i][b] = LpVariable(f"x_{i}_{b}", cat=LpBinary)

        # Set up the objective function using the PASSED scores and param_sizes
        if objective_type == "score":
            prob += lpSum([scores[i][b] * x[i][b] for i in range(n_layers) for b in BIT_WIDTHS]), "Total_Score"
        else: # Minimize param_size
            prob += lpSum([param_sizes[i][b] * x[i][b] for i in range(n_layers) for b in BIT_WIDTHS]), "Total_Param_Size"

        # Constraint 1: Each layer must be assigned exactly one bit width
        for i in range(n_layers):
            prob += lpSum([x[i][b] for b in BIT_WIDTHS]) == 1, f"Layer_{i}_One_Bitwidth"

        # Constraint 2: Parameter size constraint
        if objective_type == "score" and max_param_size is not None:
            prob += lpSum([param_sizes[i][b] * x[i][b] for i in range(n_layers) for b in BIT_WIDTHS]) <= max_param_size, "Max_Param_Constraint"

        # Constraint 3: Score constraint
        if objective_type == "param_size" and min_score is not None:
            prob += lpSum([scores[i][b] * x[i][b] for i in range(n_layers) for b in BIT_WIDTHS]) >= min_score, "Min_Score_Constraint"

        # Store references needed for iterative solving
        prob.x_vars = x
        prob.layers = layers # Store layers for mapping indices if needed later
        # Store the passed scores/sizes on the problem object for consistency/potential reuse
        prob.scores = scores
        prob.param_sizes = param_sizes
        prob.objective_type = objective_type

    else:
        # Use the existing problem
        prob = existing_prob
        x = prob.x_vars
        # layers = prob.layers # Already have n_layers from quant_dict_base
        # objective_type = prob.objective_type # Not strictly needed here
        # We use the scores and param_sizes passed into *this function call*,
        # not necessarily the ones stored on prob from the first iteration,
        # although in the current find_top_n structure they will be the same.

    # Add any new constraints (like solution exclusion)
    if constraints_to_add:
        for constraint in constraints_to_add:
            prob += constraint

    # Solve the problem
    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # Check if a solution was found
    if LpStatus[prob.status] != 'Optimal':
        return None, None, None, None, prob

    # Extract the solution
    bit_width_assignment = []
    current_solution_indices = []
    
    for i in range(n_layers): # Iterate using index
        for b in BIT_WIDTHS:
            
            if value(x[i][b]) > 0.99:
                bit_width_assignment.append(b)
                current_solution_indices.append((i, b))
                break
    
    # Calculate total parameter size and score for the optimal solution
    # Use the 'param_sizes' and 'scores' passed into this function
    total_param = sum(param_sizes[i][b] for i, b in current_solution_indices) # More direct calculation
    total_score = sum(scores[i][b] for i, b in current_solution_indices) # More direct calculation

    return bit_width_assignment, total_param, total_score, current_solution_indices, prob


# ---------------------------------------------
# 4. Top N configurations function (Now takes scores/params, no model info)
# ---------------------------------------------
def find_top_n_configs(
    n,
    quant_dict_base,         # Still needed for layer info/count
    scores,                  # Pre-calculated scores passed in
    param_sizes,             # Pre-calculated sizes passed in
    max_param_size=None,
    min_score=None,
    BIT_WIDTHS=[8, 4, 2]     # Default bit widths, can be overridden
):
    """
    Finds the top N distinct bit-width configurations based on the objective.
    Requires pre-calculated scores and param_sizes dictionaries.
    """
    top_solutions = []
    prob = None # Initialize problem state for iterative solving
    layers = list(quant_dict_base.keys()) # Get layers for count/indices
    n_layers = len(layers)

    # Determine objective based on provided constraints
    if max_param_size is not None and min_score is None:
        objective_type = "score"
        print(f"Finding top {n} configs maximizing score (Param Size <= {max_param_size})...")
    elif min_score is not None and max_param_size is None:
        objective_type = "param_size"
        print(f"Finding top {n} configs minimizing param size (Score >= {min_score})...")
    else:
         raise ValueError("Provide *either* max_param_size (to maximize score) *or* min_score (to minimize param size), but not both or neither.")


    for k in range(n):
        # print(f"Finding configuration #{k+1}...") # Moved print outside loop
        # Pass the pre-calculated scores and param_sizes in each iteration
        bit_assignment, total_param, total_score, solution_indices, current_prob = \
            solve_bit_width_optimization_single(
                quant_dict_base=quant_dict_base,
                scores=scores,                 # Pass calculated scores
                param_sizes=param_sizes,       # Pass calculated sizes
                max_param_size=max_param_size, # Pass constraint
                min_score=min_score,           # Pass constraint
                existing_prob=prob,            # Pass current problem state
                constraints_to_add=None,        # Exclusion added below
                BIT_WIDTHS=BIT_WIDTHS           # Pass bit widths
            )

        if bit_assignment is None:
            print(f"Could not find configuration #{k+1} (No more optimal solutions found under constraints). Stopping.")
            break

        top_solutions.append({
            'rank': k + 1,
            'bit_assignment': bit_assignment,
            'param_size': total_param,
            'score': total_score,
            'solution_indices': solution_indices
        })

        if prob is None:
             prob = current_prob # Store problem state after first solve

        # Add exclusion constraint for the found solution to the 'prob' object
        # This ensures the next call to solve finds a *different* solution
        if solution_indices:
             exclude_constraint = lpSum(prob.x_vars[i][b] for i, b in solution_indices) <= n_layers - 1
             constraint_name = f"Exclude_Solution_{k+1}"
             prob += (exclude_constraint, constraint_name)
        else:
             # This should not happen if a solution was found, but as a safeguard:
             print("Warning: solution_indices were empty after finding a solution. Cannot exclude. Stopping.")
             break

    return pd.DataFrame(top_solutions)




import torch
import torch.nn as nn
from collections import defaultdict
def activation_sizes(model, layer_bit_dict, BIT_WIDTHS=[8, 4, 2]):
    # This dictionary will store the final results.
    activation_sizes_result = defaultdict(dict)
    
    # This dictionary will temporarily store the captured sizes from the hooks.
    # {layer_name: size}
    captured_sizes = {}
    hooks = []

    # Define the hook function that will be registered to each layer.
    def hook_fn(module, input_tensors, output_tensors):
        # 'input_tensors' is a tuple. For a conv layer, the activation is the first element.
        input_tensor = input_tensors[0]
        
        # Find the name of the module that this hook is attached to.
        for name, layer_module in model.named_modules():
            if layer_module is module:
                # Store the number of elements in the input tensor.
                captured_sizes[name] = input_tensor.numel()
                break

    # Register the forward hook for each layer specified in the input dictionary.
    for layer_name in layer_bit_dict.keys():
        try:
            layer = model.get_submodule(layer_name)
            hooks.append(layer.register_forward_hook(hook_fn))
        except AttributeError:
            print(f"Warning: Layer '{layer_name}' not found in the model and will be skipped.")
            continue

    # Create a dummy input tensor with the specified size (1x3x224x224).
    # The values can be random as we only need to propagate the shape.
    dummy_input = torch.randn(1, 3, 224, 224).to('cuda')

    # Perform a forward pass in evaluation mode to trigger the hooks.
    # We use torch.no_grad() for efficiency as we don't need to compute gradients.
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    # Important: Remove the hooks after use to prevent memory leaks and
    # unexpected behavior in subsequent model operations.
    for hook in hooks:
        hook.remove()

    # Populate the final result dictionary in the requested format.
    # The activation size (number of elements) is the same for all bit widths.
    i = 0
    for layer_name, size in captured_sizes.items():
        for bit in BIT_WIDTHS:
            activation_sizes_result[i][bit] = (size * bit) / (8 * 1024 * 1024)  # Convert bits to MB
        i += 1
            
    return dict(activation_sizes_result)
            


        
def get_smi_scores_ACT(layer_bit_dict, column_names, score_df, XL_CORR=None, LY_CORR=None,model_name='mobilenetv2', dataset='imagenet', BIT_WIDTHS=[8, 4, 2]):

    layers = list(layer_bit_dict.keys())
    scores = {} # {layer_index: {bit_width: score}}

    # Fallback logic for 8-bit 'all' layer
    eight_bit_row = score_df[
        (score_df['layer'] == 'all') & (score_df['bw'] == 8)
    ]

    # Score column filtering (adapt as needed)
    score_columns = []
    for col in column_names:
        if 'mi_ly' in col or 'mi_xl' in col:
            score_columns.append(col)
    if not score_columns:
        print("Warning: No score columns identified. Using 'acc' if available.")
        if 'acc' in score_df.columns:
            score_columns = ['acc']
        else:
             print("Warning: 'acc' column not found either. Scores might be zero.")
             score_columns = []


    for i, layer in enumerate(layers):
        scores[i] = {}
        for b in BIT_WIDTHS:
            row_to_use = score_df[
                (score_df['layer'] == layer) & (score_df['bw'] == b)
            ]
            if row_to_use.empty:
                #print(f"Warning: No score data for layer {layer}, bw {b}. Assigning zero score.")
                scores[i][b] = 0.0
                continue

            current_row_data = row_to_use.iloc[0]
            eight_bit_data = eight_bit_row.iloc[0]

            # --- Detailed score calculation logic (same as before) ---
            XL, XL_8, LY, LY_8 = 0.0, 0.0, 0.0, 0.0 # Initialize as floats
            # Add other variables needed: ss, j_denom, ss_xl, j_xl_denom, etc.
            ss, j_denom, ss_xl, j_xl_denom = 0.0, 0.0, 0.0, 0.0
            total_ly, eight_bit_ly, total_xl, eight_bit_xl = 0.0, 0.0, 0.0, 0.0

            start_col_index = 0 # Default start index
            try:
                if model_name == 'mobilenetv2_x0_5' and dataset == 'cifar10':
                    layer_base_parts = layer.split('.')[:2]
                    layer_prefix_search = '.'.join(layer_base_parts) + '_'
                elif model_name == 'resnet20' and dataset == 'cifar10':
                    layer_base_parts = layer.split('.')[:2]
                    layer_prefix_search = '.'.join(layer_base_parts) + '_'
                elif model_name == 'resnet18_gluon' and dataset == 'imagenet_256x256':
                    layer_base_parts = layer.split('.')[:2]
                    layer_prefix_search = '.'.join(layer_base_parts) + '_'
                elif model_name == 'resnet34_gluon' and dataset == 'imagenet_256x256':
                    layer_base_parts = layer.split('.')[:2]
                    layer_prefix_search = '.'.join(layer_base_parts) + '_'
                elif model_name == 'resnet50_gluon' and dataset == 'imagenet_256x256':
                    layer_base_parts = layer.split('.')[:2]
                    layer_prefix_search = '.'.join(layer_base_parts) + '_'
                elif model_name == 'mobilenetv2' and dataset == 'imagenet_256x256':
                    if layer == 'features.0':
                        layer_prefix_search =  'features.2_'
                    elif layer == 'conv.0':
                        layer_prefix_search =  'avgpool_'
                    else:
                        layer_base_parts = layer.split('.')[:2]
                       
                        layer_prefix_search = '.'.join(layer_base_parts) + '_'
                        if layer_prefix_search == 'features.1_' or layer_prefix_search == 'features.0_':
                            layer_prefix_search = 'features.2_'
                    #print(f"Layer prefix search for {layer}: {layer_prefix_search}")
                start_col_index = -1
                for idx, col_name in enumerate(score_columns):
                    if col_name.startswith(layer_prefix_search):
                        start_col_index = idx
                        break
                if start_col_index == -1: start_col_index = 0
            except Exception as e: 
                print(f"Error finding start column index for layer {layer}: {e}")
                start_col_index = 0
            #print(f"Start column index for layer {layer}: {start_col_index}")
            layer_to_block_dict = conv_to_block_output_dict(model_name, dataset)
            start_col_index = score_columns.index(layer_to_block_dict[layer]+'_mi_ly')

        
            # --------- XL Score calculation ---------- #
            if model_name == 'mobilenetv2_x0_5' and dataset == 'cifar10':
                START_XL = 4
                END_XL = 26
            elif model_name == 'resnet20' and dataset == 'cifar10':
                START_XL = 6
                END_XL = 12
            elif model_name == 'resnet18_gluon' and dataset == 'imagenet_256x256':
                START_XL = 8
                END_XL = 14
            elif model_name == 'resnet34_gluon' and dataset == 'imagenet_256x256':
                START_XL = 2
                END_XL = 6
            elif model_name == 'resnet50_gluon' and dataset == 'imagenet_256x256':
                START_XL = 2
                END_XL = 14
            elif model_name == 'mobilenetv2' and dataset == 'imagenet_256x256':
                START_XL = 2
                END_XL = 32
        
            range_start_xl = max(START_XL, start_col_index) # Original logic
            range_end_xl = min(END_XL, len(score_columns)) # Original logic, ensure bounds
            def clamp(n, min_value, max_value):
                return max(min_value, min(n, max_value))
            for l_idx in range(range_start_xl, range_end_xl, 2):
                corr_idx = l_idx // 2
                lay = score_columns[l_idx]
                # Safely get xl values
                current_xl_val = current_row_data.get(lay.replace('_mi_ly', '_mi_xl'), 0.0)
                eight_bit_xl_val = eight_bit_data.get(lay.replace('_mi_ly', '_mi_xl'), 0.0)

                XL += abs(current_xl_val - eight_bit_xl_val)#clamp(current_xl_val - eight_bit_xl_val, 0, float('inf')) * abs(XL_CORR[corr_idx]) # Difference weighted by XL_CORR
                XL_8 += eight_bit_xl_val #* abs(XL_CORR[corr_idx]) # Weighted 8-bit value

            # --------- LY Score calculation ---------- #
            if model_name == 'mobilenetv2_x0_5':
                START_LY = 26
            elif model_name == 'resnet20':
                START_LY = 16
            elif model_name == 'resnet18_gluon':
                START_LY = 14
            elif model_name == 'resnet34_gluon':
                START_LY = 26
            elif model_name == 'resnet50_gluon':
                START_LY = 26
            elif model_name == 'mobilenetv2':
                START_LY = 34
            range_start_ly = max(start_col_index, START_LY) # Original logic
            range_end_ly = len(score_columns) # Original logic

            for l_idx in range(range_start_ly, range_end_ly, 2):
                corr_idx = l_idx // 2

                lay = score_columns[l_idx]
                current_val = current_row_data.get(lay, 0.0)
                eight_bit_val = eight_bit_data.get(lay, 0.0)

                LY += abs(eight_bit_val-current_val) #clamp(eight_bit_val-current_val, 0, float('inf')) #* LY_CORR[corr_idx] #abs(eight_bit_val-current_val) * LY_CORR[corr_idx]#
                LY_8 += eight_bit_val #* LY_CORR[corr_idx]


            # Final Score Calculation
            denom_ibopt = XL_8 + LY_8
            if denom_ibopt != 0:
                # Ensure calculations result in float
                score_value = -(float(XL) + float(LY)) / float(denom_ibopt)
                # Add checks for NaN or Inf if divisions can produce them
                if np.isnan(score_value) or np.isinf(score_value):
                        print(f"Warning: Score for layer {layer} bw {b} is NaN or Inf. Setting to 0.")
                        scores[i][b] = 0.0
                else:
                        scores[i][b] = score_value * (1/b)

            else:
                # print(f"Warning: Division by zero avoided in IBOPT calc for layer {layer}, bw {b}. Score set to 0.")
                scores[i][b] = 0.0
    print('SMI Scores Calculated')
    return scores

