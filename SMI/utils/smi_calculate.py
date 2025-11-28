import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import gc # Garbage Collector
import os # Still needed for os.path stuff if helper functions used it before, but not for saving

from SMI.utils.basic_functions import compute_ksmi_optimized_parallel

# --- Hook related globals and functions (Keep these as they manage activation capture) ---
hooks = []
activation_outputs_current_batch = defaultdict(lambda: None)

def get_activation(name):
    """Hook function to capture the output FOR THE CURRENT BATCH"""
    def hook(model, input, output):
        global activation_outputs_current_batch
        # Store the detached CPU tensor directly for the current batch.
        if isinstance(output, torch.Tensor):
            # Ensure it's on CPU to avoid GPU memory buildup if not needed immediately
            activation_outputs_current_batch[name] = output.detach().cpu()
        elif isinstance(output, tuple): # Handle layers returning tuples
             if len(output) > 0 and isinstance(output[0], torch.Tensor):
                 activation_outputs_current_batch[name] = output[0].detach().cpu()
             else: # Cannot extract tensor, store None or skip
                  activation_outputs_current_batch[name] = None
        else: # Non-tensor, store None or skip
             activation_outputs_current_batch[name] = None
    return hook

def register_hooks(model: nn.Module, module_names: list):
    """Registers forward hooks on specified modules"""
    global hooks
    # Remove any existing hooks before registering new ones
    for h in hooks:
        h.remove()
    hooks = []
    registered_count = 0
    for name, module in model.named_modules():
        if name in module_names:
            hook_handle = module.register_forward_hook(get_activation(name))
            hooks.append(hook_handle)
            registered_count += 1

    print(f"Registered {registered_count}/{len(module_names)} hooks.")
    if registered_count < len(module_names):
         # Find which names weren't found in model.named_modules()
         found_names = {name for name, module in model.named_modules() if name in module_names}
         missing = set(module_names) - found_names
         print(f"Warning: Could not find all target modules. Missing: {missing}")

import torch
import numpy as np
from collections import defaultdict
import gc

activation_outputs_current_batch = {}
hooks = []
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# --- Modified calculate_smi function ---
def calculate_smi(
    model,          # The model itself
    eval_dataloader,# The dataloader
    targets,        # Pre-computed targets (numpy array)
    input_data,     # Pre-computed input data (numpy array, assumed N x Features)
    number_of_class,
    device = 'cuda', # Device ('cuda' or 'cpu')
    m=1000,         # Parameter for MI calculation
    target_modules=[], # List of layer names to analyze
    layer_needed=None, # Optional: only process this specific layer
    image_column='image', # Key for image tensor in dataloader batches
    # label_column='label' # Not strictly needed if targets are pre-provided
    XL_OBSERVERS = [],
    YL_OBSERVERS = [],
):
    """
    Calculates SMI by processing one layer at a time to conserve memory.
    Activations are collected in memory during a pass over the dataloader for each layer.

    Args:
        model (nn.Module): The neural network model.
        eval_dataloader (DataLoader): DataLoader for evaluation data.
        targets (np.ndarray): Numpy array of target labels (N, 1) or (N, C) if one-hot.
        input_data (np.ndarray): Numpy array of the input data (N, Features).
        device (torch.device): Device ('cuda' or 'cpu').
        m (int): Parameter for the KSMI estimator.
        target_modules (list): List of layer names (str) to extract activations from.
        layer_needed (str, optional): If set, only calculates MI for this specific layer.
        image_column (str): Dictionary key for images in the dataloader batch.

    Returns:
        dict: Dictionary mapping layer names to (MI(X,L), MI(L,Y)) tuples.
              e.g., {'layer1': (mi_x_l_value, mi_l_y_value), ...}
    """
    global activation_outputs_current_batch, hooks # Declare usage of global variables

    mi_values = {} # Initialize results dictionary

    # --- Prepare Targets (once) ---
    if len(targets.shape) == 1:
        targets_flat = targets.reshape(-1, 1)
    else:
        if targets.shape[1] > 1 and np.all(np.sum(targets, axis=1) == 1) and not np.any(targets < 0): # Check for one-hot
             print("Targets appear one-hot encoded. Converting to class indices for MI(L,Y).")
             targets_flat = np.argmax(targets, axis=1).reshape(-1, 1)
        else:
             targets_flat = targets # Assume already (N, 1) or compute_ksmi handles (N, C) or other formats

    print(f"Using targets with shape: {targets_flat.shape}")

    # --- Prepare Input Data (once) ---
    print(f"Using input data with shape: {input_data.shape}")

    # --- Setup Model (once) ---
    model.eval()
    model.to(device)
    needed_layer_found = False
    for layer_name in target_modules:
        if not needed_layer_found:
          if layer_needed is not None and layer_name != layer_needed:
              continue
          else:
            needed_layer_found = True

        print(f"\n--- Processing layer: {layer_name} ---")
        if layer_name not in XL_OBSERVERS and layer_name not in YL_OBSERVERS:
            ##assign zero , not used 
            mi_values[layer_name] = (0.0, 0.0)
            print(f"Skipping layer {layer_name} as it is not in XL_OBSERVERS or YL_OBSERVERS.")
            continue

        # --- Clear and Register Hooks for the CURRENT LAYER ONLY ---
        if hooks: # Remove any hooks from a previous iteration (should be empty if managed correctly)
            print(f"Warning: Hooks list was not empty before processing layer {layer_name}. Clearing {len(hooks)} hooks.")
            for h in hooks: h.remove()
        hooks = [] # Ensure hooks list is empty
        activation_outputs_current_batch.clear() # Clear any stale activations

        # Register hook for the current single layer
        # Assumes register_hooks populates the global 'hooks' list
        # and sets up the hook to populate 'activation_outputs_current_batch[layer_name]'
        current_layer_target_list = [layer_name]
        register_hooks(model, current_layer_target_list)

        if not any(hook_is_for_layer(h, layer_name, model) for h in hooks): # Heuristic check
             print(f"Warning: No hook seems to be registered for layer {layer_name}. Skipping MI calculation for this layer.")
             # Clean up any hooks that might have been spuriously registered if register_hooks had issues
             for h_rem in list(hooks): # Iterate over a copy for safe removal
                 h_rem.remove()
                 hooks.remove(h_rem)
             gc.collect()
             continue

        # --- Accumulate Activations in Memory for the CURRENT LAYER ---
        current_layer_batch_activations = [] # List to hold CPU tensors for the current layer
        print(f"Running inference on {len(eval_dataloader)} batches to collect activations for layer: {layer_name}...")

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                try:
                    x = batch[image_column].to(device)
                except KeyError as e:
                    print(f"\nError: Column key '{e}' not found in batch {i} keys: {list(batch.keys())}")
                    for h_rem in list(hooks): h_rem.remove()
                    hooks = []
                    gc.collect()
                    raise e
                except AttributeError as e:
                     print(f"\nError: Failed to process batch {i}. Check batch content. Type: {type(batch)}. Error: {e}")
                     for h_rem in list(hooks): h_rem.remove()
                     hooks = []
                     gc.collect()
                     raise e
                except Exception as e:
                     print(f"\nError processing batch {i} data: {e}")
                     for h_rem in list(hooks): h_rem.remove()
                     hooks = []
                     gc.collect()
                     raise e

                _ = model(x) # Forward pass triggers the hook for layer_name

                # Collect activation captured by the hook for THIS BATCH for THIS LAYER
                tensor_batch_gpu = activation_outputs_current_batch.get(layer_name)

                if tensor_batch_gpu is not None:
                    current_layer_batch_activations.append(tensor_batch_gpu.cpu()) # Move to CPU and store
                    activation_outputs_current_batch[layer_name] = None # Clear for next batch
                # else: # Optional debug
                    # print(f"Debug: No activation found for {layer_name} in batch {i}")

                if (i + 1) % 50 == 0:
                     gc.collect()
                     print(f"  Processed batch {i+1}/{len(eval_dataloader)} for layer {layer_name}")

        print(f"Inference complete for layer {layer_name}. Activations collected.")

        # --- Remove Hooks for the CURRENT LAYER ---
        print(f"Removing {len(hooks)} hooks for layer {layer_name}...")
        for h in hooks:
            h.remove()
        hooks = [] # Clear the global list
        activation_outputs_current_batch.clear() # Clear any remaining data
        gc.collect()
        # --- --- --- --- ---

        # --- Process Accumulated Activations for the CURRENT LAYER ---
        if not current_layer_batch_activations:
            print(f"Warning: No activations collected for layer {layer_name}. Skipping MI calculation.")
            continue

        try:
            # Ensure all items are tensors before cat
            valid_batches = [b for b in current_layer_batch_activations if isinstance(b, torch.Tensor)]
            if not valid_batches:
                print(f"Warning: No valid activation tensors found for layer {layer_name} after filtering.")
                del current_layer_batch_activations # Clean up list
                gc.collect()
                continue
            layer_output = torch.cat(valid_batches, dim=0)
            print(f"Layer '{layer_name}': Concatenated shape: {layer_output.shape}")
        except Exception as e:
            print(f"Error concatenating activations for layer {layer_name}: {e}")
            del current_layer_batch_activations # Clean up list
            gc.collect()
            continue # Skip this layer

        # --- Free the list of batch activations for this layer ---
        del current_layer_batch_activations # Explicitly delete the list of tensor batches
        del valid_batches
        gc.collect()

        # --- Reshape the layer output to 2D (samples, features) ---
        if len(layer_output.shape) == 4:  # Conv layer
            layer_data = layer_output.reshape(layer_output.shape[0], -1)
        elif len(layer_output.shape) == 2:  # Fully connected layer
            layer_data = layer_output
        else:
            print(f"Unsupported layer output shape: {layer_output.shape} for layer {layer_name}. Skipping.")
            del layer_output
            gc.collect()
            continue

        # --- Convert to NumPy and free Torch tensor ---
        L_compressed = layer_data.cpu().numpy() # Ensure it's on CPU before numpy conversion
        del layer_output # Free GPU/CPU memory held by the torch tensor
        del layer_data
        gc.collect()

        print(f"Layer '{layer_name}': Reshaped to NumPy array: {L_compressed.shape}")

        # --- Ensure consistent number of samples ---
        if L_compressed.shape[0] != input_data.shape[0]:
             print(f"Mismatch in number of samples for layer {layer_name}: Activations {L_compressed.shape[0]}, Input {input_data.shape[0]}. Skipping MI.")
             del L_compressed
             gc.collect()
             continue
        if L_compressed.shape[0] != targets_flat.shape[0]:
             print(f"Mismatch in number of samples for layer {layer_name}: Activations {L_compressed.shape[0]}, Targets {targets_flat.shape[0]}. Skipping MI.")
             del L_compressed
             gc.collect()
             continue

        # --- Compute MI ---
        mi_estimate_xl, mi_estimate_ly = np.nan, np.nan # Initialize
     
        print(f"Computing MI(X, L) for layer {layer_name}...")
        try:
             if layer_name in XL_OBSERVERS:
                mi_estimate_xl = compute_ksmi_optimized_parallel(
                    L_compressed,
                    input_data,
                    discrete=False,
                    m=m # Original code used m=100 here, not the function parameter m
                )
                print(f"  MI estimate (X, L) for layer {layer_name}: {mi_estimate_xl:.4f}")
             else:
                mi_estimate_xl = 0
        except Exception as e:
            print(f"Error computing MI(X,L) for layer {layer_name}: {e}")
            # mi_estimate_xl remains np.nan
       
        print(f"Computing MI(L, Y) for layer {layer_name}...")
        import time
        start_time = time.time() # Start timing
        try: 
             if layer_name in YL_OBSERVERS:  
                mi_estimate_ly = compute_ksmi_optimized_parallel(
                    L_compressed,
                    targets_flat,
                    discrete=True,
                    m=m, # Uses the function parameter m
                    number_of_class=number_of_class
                )
             else:
                mi_estimate_ly = 0
             end_time = time.time() # End timing
             #total time in seconds
             elapsed_time = end_time - start_time
             print(f"  MI estimate (L, Y) for layer {layer_name}: {mi_estimate_ly:.4f} (computed in {elapsed_time:.2f} seconds)")
             print(f"  MI estimate (L, Y) for layer {layer_name}: {mi_estimate_ly:.4f}")
        except Exception as e:
            print(f"Error computing MI(L,Y) for layer {layer_name}: {e}")
            # mi_estimate_ly remains np.nan

        # Store the MI estimates for the current layer
        if not (np.isnan(mi_estimate_xl) and np.isnan(mi_estimate_ly)): # Store if at least one is not nan
            mi_values[layer_name] = (mi_estimate_xl, mi_estimate_ly)
        else:
            print(f"Both MI(X,L) and MI(L,Y) failed for layer {layer_name}.")


        # --- Clean up large NumPy array for the current layer ---
        del L_compressed
        gc.collect()
        print(f"--- Finished processing layer: {layer_name} ---")

    print("\nFinished all MI calculations for all specified layers.")
    return mi_values

# Helper function to check if a hook is for a specific layer (example heuristic)
# This is for the warning message and might need adjustment based on your actual hook objects.
def hook_is_for_layer(hook, layer_name, model):
    # This is a placeholder. How you check this depends on how your `register_hooks`
    # and hook objects are structured.
    # If hook objects store the layer name or can be introspected, use that.
    # For now, we assume `register_hooks([layer_name])` only registers for that layer.
    # If `hooks` list only contains hooks for the current layer_name after `register_hooks`,
    # then checking `if not hooks:` would suffice, but the original code structure
    # used a global `hooks` list.
    # A simple check if any layer module associated with a hook matches the name:
    try:
        # This is a bit of a hack and relies on internal structure of hook handle
        # It might not be universally reliable.
        # PyTorch hook handles have a 'id' and refer to a 'hook' function,
        # and are attached to a module. We need to find which module.
        for name, mod in model.named_modules():
            if name == layer_name:
                # Check if any of the module's forward hooks match the hook object
                # This is difficult to check directly without knowing hook internals
                # or how register_hooks links them.
                # A simpler check: if register_hooks exclusively adds hooks for layer_name,
                # then len(hooks) > 0 is enough.
                pass # Cannot reliably check this way without more hook info
    except:
        pass
    return True # Default to true to avoid false negatives without a robust check.
                # The main check is `if not current_layer_batch_activations:` later.
