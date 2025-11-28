from copy import deepcopy
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import importlib
import torch
#import mqbench_functions
image_column = 'image'
label_column = 'label'
# Reload the module
#importlib.reload(mqbench_functions)

# Now try importing the specific function again
#from mqbench_functions import evaluate_quantized, quantize_ptq_mixed_precision, mpq_quantize, finetune_quantized_model

print("Successfully reloaded mqbench_functions and imported evaluate_quantized!")



from typing import Optional
import torch
import torch.nn as nn
import torch.utils.data
import copy # To avoid modifying the original model
import logging # Optional: for better feedback

# MQBench imports
# Assuming these imports are correct based on your previous success
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.fake_quantize import FixedFakeQuantize
#import disable observer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# May potentially need QuantizeScheme if constructing it manually, but let's try dicts first
# from mqbench.scheme import QuantizeScheme

# Setup basic logging (optional)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def evaluate_quantized(model: nn.Module,
                       test_dataloader: torch.utils.data.DataLoader,
                       device: str = None) -> float:
    """
    Evaluates the top-1 accuracy of a PyTorch model on a given test dataset.

    Args:

        model: The PyTorch model (nn.Module) to evaluate (can be FP32 or quantized).
        test_dataloader: DataLoader providing the test data. Assumes batches yield
                         (inputs, labels).
        device: The device to run evaluation on ('cuda', 'cpu', or None).
                If None, defaults to 'cuda' if available, otherwise 'cpu'.

    Returns:
        The top-1 accuracy of the model on the test set (float, range 0.0 to 100.0).
    """
    # --- Device Setup ---
    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    print(f"Evaluating on device: {resolved_device}")

    model.to(resolved_device)
    model.eval()  # Set model to evaluation mode (important!)

    correct_predictions = 0
    total_samples = 0
    evaluation_batches = len(test_dataloader)

    # --- Evaluation Loop ---
    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # Move data to the evaluation device
            inputs = batch['image'].to(resolved_device)
            labels = batch['label'].to(resolved_device)
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Update correct predictions and total samples
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            # Optional: Log progress
            if i % 10 == 0:  # Log every 10 batches
                print(f"Batch {i}/{evaluation_batches}: "
                             f"Correct: {correct_predictions}, "
                             f"Total: {total_samples}")
    # --- Final Logging ---
    print(f"Final Evaluation: Correct: {correct_predictions}, Total: {total_samples}")

    # --- Calculate and Log Accuracy ---
    if total_samples == 0:
        logging.warning("No samples found in the test dataloader.")
        return 0.0

    accuracy = 100.0 * correct_predictions / total_samples
    print(
        f'Evaluation Complete: Accuracy: {accuracy:.2f}% '
        f'({correct_predictions}/{total_samples})'
    )

    return accuracy
def getModuleByName(model,moduleName):
    '''
        replace module with name modelName.moduleName with newModule
    '''
    tokens = moduleName.split('.')
    m = model
    for tok in tokens:
        m = getattr(m,tok)
    return m

def mpq_quantize(
    layer_bitwidth_dict: dict,
    eight_bit_quantized_model: nn.Module = None,
    four_bit_quantized_model: nn.Module = None,
    two_bit_quantized_model: nn.Module = None,
    num_calibration_batches: int = 100,
    input_shape: tuple = (1, 3, 32, 32), # Default MobileNetV2 input shape,
    calib_dataloader: Optional[torch.utils.data.DataLoader] = None,
):
    copy_base_model = copy.deepcopy(eight_bit_quantized_model)
    four_bit_copy_base_model = copy.deepcopy(four_bit_quantized_model)
    two_bit_copy_base_model = copy.deepcopy(two_bit_quantized_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ll in layer_bitwidth_dict.keys():
        bit = layer_bitwidth_dict[ll]
        if bit == 8:
            continue
        elif bit == 4:
            #copy_base_model.ll.weight_fake_quant = four_bit_quantized_model.ll.weight_fake_quant
            #get module ll
            base_module = getModuleByName(copy_base_model, ll)
            target_module = getModuleByName(four_bit_copy_base_model, ll)
            base_module.weight_fake_quant = target_module.weight_fake_quant

        elif bit == 2:
            #copy_base_model.ll.weight_fake_quant = two_bit_quantized_model.ll.weight_fake_quant
            #get module ll
            base_module = getModuleByName(copy_base_model, ll)
            target_module = getModuleByName(two_bit_copy_base_model, ll)
            base_module.weight_fake_quant = target_module.weight_fake_quant
        else:
            logging.warning(f"Invalid bit-width {bit} for layer {ll}. Skipping override.")
            continue
    enable_quantization(copy_base_model) # Enable fake quantizers using calibrated stats
    #copy_base_model.eval()
    #return copy_base_model

    # ---- CALIBRATION ------
    logging.info(f"Starting calibration using {num_calibration_batches} batches...")
    enable_calibration(copy_base_model) # Enable observers
    device = next(copy_base_model.parameters()).device # Get model device (CPU/GPU)
    calib_count = 0
    with torch.no_grad():
        for i, batch in enumerate(calib_dataloader):
            data = batch[image_column].to(device)
            copy_base_model(data)
            calib_count += 1
            if calib_count >= num_calibration_batches:
                logging.info(f"Calibration completed after {calib_count} batches.")
                break
        else: # Executed if the loop finishes without break
             logging.warning(f"Calibration finished after {calib_count} batches (fewer than requested {num_calibration_batches}).")

    # --- 3. Convert for Deployment ---
    logging.info("Converting model for deployment...")
    enable_quantization(copy_base_model) # Enable fake quantizers using calibrated stats
    copy_base_model.eval()


    return copy_base_model.to(device)

#import Optional


def quantize_ptq_mixed_precision(
    model_to_quantize: nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    num_calibration_batches: int = 100,
    input_shape: tuple = (1, 3, 32, 32), # Default MobileNetV2 input shape,
    BIT = 8,
    no_act_quantization: bool = False
) -> nn.Module:
    """
    Performs Post-Training Quantization (PTQ) using MQBench with mixed-precision weights
    and 8-bit activations, following documentation structure.

    Args:
        model_to_quantize: The PyTorch model (nn.Module) to be quantized.
        layer_bitwidth_dict: A dictionary mapping specific layer names (str) to
                             their desired weight bit-width (int).
                             Example: {'features.1.conv.0.0': 8, 'features.17.conv.2': 2}
        eval_dataloader: DataLoader providing calibration data.
        num_calibration_batches: The number of batches from eval_dataloader to use for calibration.
        input_shape: The shape of a single input sample (e.g., (1, 3, 224, 224)).
                     Required for the deployment conversion step.

    Returns:
        A quantized model ready for deployment/evaluation.
    """

    logging.info("--- Starting PTQ Mixed-Precision Quantization ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Quantizing on device: {device}")

    # --- 1. Prepare Model & Configuration ---
    model = copy.deepcopy(model_to_quantize) # Work on a copy
    model.eval()

    backend = BackendType.Academic

    # --- NEW CONFIGURATION based on Documentation ---
    # Pass this dictionary directly to prepare_by_platform
    custom_config = {
        'extra_qconfig_dict': {
            'w_observer': 'MSEObserver',                              # custom weight observer
            'a_observer': 'EMAMSEObserver',                              # custom activation observer
            'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
            'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
            'w_qscheme': {
                'bit': BIT,                                             # custom bitwidth for weight,
                'symmetry': True,                                    # custom whether quant is symmetric for weight,
                'per_channel': False,#True,                                  # custom whether quant is per-channel or per-tensor for weight,
                'pot_scale': False,                                   # custom whether scale is power of two for weight.
            },
            'a_qscheme': {
                'bit': 8,                                             # custom bitwidth for activation,
                'symmetry': False,                                    # custom whether quant is symmetric for activation,
                'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
                'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        },

        # Nested dictionary for layer-specific overrides, using the key from docs
        'extra_quantizer_dict': {
            # Populated below
        }
        # Other keys like 'extra_quantizer_dict', 'extra_fuse_dict' are omitted for now
    }

    logging.info(f"Preparing model using backend: {backend.name} with documentation-based config.")
    # Pass the new custom_config directly
    model_prepared = prepare_by_platform(model, backend, custom_config)

    model_prepared = model_prepared.to(device)
    logging.info("Model preparation complete.")
    # print(model_prepared) # Optional: Inspect the prepared model structure

    # --- 2. Calibration ---
    logging.info(f"Starting calibration using {num_calibration_batches} batches...")
    enable_calibration(model_prepared) # Enable observers

    device = next(model_prepared.parameters()).device # Get model device (CPU/GPU)
    calib_count = 0 # Initialize counter
    logging.info(f"Starting calibration using {num_calibration_batches} batches...")

    with torch.no_grad():
        # Iterate over the dataloader, getting one dictionary batch at a time
        for i, batch in enumerate(eval_dataloader):
            try:
                # --- Extract the image tensor from the batch dictionary ---
                # Use the correct key for your image data (e.g., 'image')
                data = batch[image_column].to(device)
                # --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            except KeyError as e:
                logging.error(f"Error: Image column key '{image_column}' not found in calibration batch {i}. Available keys: {list(batch.keys())}")
                # Decide how to handle this - stop calibration, skip batch?
                raise e # Re-raise the error to stop execution
            except AttributeError as e:
                logging.error(f"Error: Failed to move data to device for calibration batch {i}. Check batch content for key '{image_column}'. Type: {type(batch.get(image_column))}")
                raise e # Re-raise the error

            # Now 'data' is the tensor on the correct device
            model_prepared(data) # Run the forward pass for calibration
            calib_count += 1

            # Log progress periodically (optional)
            if (calib_count % 20 == 0) and calib_count > 0 :
                logging.info(f"  Calibration progress: {calib_count}/{num_calibration_batches} batches processed.")

            # Check if enough batches have been processed
            if calib_count >= num_calibration_batches:
                logging.info(f"Calibration completed after reaching target of {calib_count} batches.")
                break # Exit the loop

        # This 'else' block executes if the loop completes *without* hitting the 'break'
        # (i.e., if the dataloader runs out of batches before reaching num_calibration_batches)
        else:
            logging.warning(f"Calibration finished after processing all {calib_count} batches available in the dataloader, which is fewer than the requested {num_calibration_batches}.")

    # You might want to add a final log message regardless of how the loop finished
    logging.info(f"Calibration phase finished. Total batches processed: {calib_count}")

    # --- 3. Convert for Deployment ---
    logging.info("Converting model for deployment...")
    enable_quantization(model_prepared) # Enable fake quantizers using calibrated stats
    model_prepared.eval()

    return model_prepared.to(device)



# --- Example Usage --- (Code remains the same)


import torch
import torch.nn as nn
import torch.optim as optim # Import optimizer
import torch.utils.data
import copy
import logging
import os # For checkpoint saving
import time # For timing epochs

# Assuming MQBench imports are available as in your previous code
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization, disable_all # Added disable_all
from mqbench.fake_quantize import FixedFakeQuantize # Or whichever class you use

# Assuming evaluate_quantized and getModuleByName functions are defined as before
# --- [Insert evaluate_quantized function here] ---
# --- [Insert getModuleByName function here] ---
# --- [Insert mpq_quantize function here] --- # Needed if you want to create the mixed-precision model first
# --- [Insert quantize_ptq_mixed_precision function here] --- # Needed by mpq_quantize

# Setup basic logging (optional)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')

def finetune_quantized_model(
    model_prepared: nn.Module, # Model AFTER prepare_by_platform
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 1e-5, # Use a small LR for fine-tuning
    optimizer_class=optim.Adam, # Or optim.SGD
    criterion=nn.CrossEntropyLoss(),
    device: str = None,
    early_stopping_patience: int = 3, # Stop if no improvement after N epochs
    checkpoint_dir: str = './quant_finetune_checkpoints' # Directory to save best model
) -> nn.Module:
    """
    Fine-tunes a prepared quantized model (QAT) to potentially recover accuracy.

    Args:
        model_prepared: The model prepared by MQBench (with fake quantizers).
                       MUST NOT be the deployed model. enable_quantization will be
                       called on this model.
        train_dataloader: DataLoader for training data.
        eval_dataloader: DataLoader for validation data (used for early stopping
                         and reporting).
        num_epochs: Maximum number of fine-tuning epochs.
        learning_rate: Learning rate for the optimizer.
        optimizer_class: The optimizer class to use (e.g., torch.optim.Adam).
        criterion: The loss function (e.g., nn.CrossEntropyLoss).
        device: The device to run training on ('cuda', 'cpu', or None).
        early_stopping_patience: Number of epochs to wait for validation accuracy
                                 improvement before stopping. If <= 0, disabled.
        checkpoint_dir: Directory to save the best performing model checkpoint.
                        If None, checkpointing is disabled.

    Returns:
        The fine-tuned model (with weights loaded from the best checkpoint if
        checkpointing was enabled and improvement was seen).
    """
    logging.info("--- Starting Quantization-Aware Fine-tuning ---")

    # --- Device Setup ---
    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    logging.info(f"Fine-tuning on device: {resolved_device}")

    model_prepared.to(resolved_device)

    # --- CRITICAL: Enable Quantization ---
    # This activates the fake quantizers using calibrated stats and enables STE
    logging.info("Enabling quantization for fine-tuning (activates fake quantizers).")
    enable_quantization(model_prepared)

    # --- Optimizer and Loss ---
    # Fine-tune all parameters. Adjust if you only want to fine-tune specific layers.
    optimizer = optimizer_class(model_prepared.parameters(), lr=learning_rate)
    logging.info(f"Using Optimizer: {optimizer_class.__name__} with LR: {learning_rate}")

    # --- Checkpointing and Early Stopping Setup ---
    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    best_model_path = None

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, 'best_quantized_model.pth')
        logging.info(f"Best model checkpoint will be saved to: {best_model_path}")

    # --- Initial Evaluation (Before Fine-tuning) ---
    logging.info("Evaluating model performance BEFORE fine-tuning...")
    initial_accuracy = evaluate_quantized(
        model=model_prepared,
        test_dataloader=eval_dataloader,
        device=resolved_device
    )
    logging.info(f"Initial Quantized Accuracy: {initial_accuracy:.2f}%")
    best_val_accuracy = initial_accuracy # Start tracking from initial accuracy

    # --- Fine-tuning Loop ---
    for epoch in range(num_epochs):
        start_time = time.time()
        logging.info(f"--- Epoch {epoch+1}/{num_epochs} ---")

        # --- Training Phase ---
        model_prepared.train() # Set model to training mode
        running_loss = 0.0
        batches_processed = 0
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(resolved_device), labels.to(resolved_device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass (quantization simulation happens here)
            outputs = model_prepared(inputs)
            loss = criterion(outputs, labels)

            # Backward pass (gradients flow via STE)
            loss.backward()

            # Optimizer step
            optimizer.step()

            running_loss += loss.item()
            batches_processed += 1

            # Optional: Log training progress periodically
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_dataloader):
                 logging.debug(
                     f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], '
                     f'Avg Batch Loss: {loss.item():.4f}'
                 )

        epoch_loss = running_loss / batches_processed
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1} Training Finished. Average Loss: {epoch_loss:.4f}. Time: {epoch_time:.2f}s")

        # --- Validation Phase ---
        logging.info(f"Epoch {epoch+1} Evaluating on validation set...")
        # Evaluation uses the existing evaluate_quantized function
        val_accuracy = evaluate_quantized(
            model=model_prepared, # Already on correct device, eval mode set inside function
            test_dataloader=eval_dataloader,
            device=resolved_device
        )
        logging.info(f"Epoch {epoch+1} Validation Accuracy: {val_accuracy:.2f}%")

        # --- Checkpointing & Early Stopping ---
        if val_accuracy > best_val_accuracy:
            logging.info(
                f"Validation accuracy improved ({best_val_accuracy:.2f}% -> {val_accuracy:.2f}%)."
            )
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            if best_model_path:
                logging.info(f"Saving best model checkpoint to {best_model_path}")
                # Save the state dict of the model *as it is* (prepared, with quantizers)
                torch.save(model_prepared.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            logging.info(
                f"Validation accuracy did not improve for {epochs_without_improvement} epoch(s). "
                f"Best accuracy remains {best_val_accuracy:.2f}%."
            )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            logging.warning(
                f"Early stopping triggered after {epoch+1} epochs due to no validation "
                f"accuracy improvement for {early_stopping_patience} epochs."
            )
            break # Exit the training loop

    # --- Post-Training ---
    logging.info("--- Fine-tuning Finished ---")

    # Load the best checkpoint if saved
    if best_model_path and os.path.exists(best_model_path) and best_val_accuracy > initial_accuracy:
        logging.info(f"Loading best model weights from {best_model_path}")
        model_prepared.load_state_dict(torch.load(best_model_path, map_location=resolved_device))
        logging.info("Evaluating final model loaded from best checkpoint...")
        final_accuracy = evaluate_quantized(
            model=model_prepared,
            test_dataloader=eval_dataloader,
            device=resolved_device
        )
        logging.info(f"Final Accuracy (Best Checkpoint): {final_accuracy:.2f}%")
    else:
        logging.info("Using model state from the end of training (no better checkpoint found or saving disabled).")
        logging.info("Evaluating final model state...")
        final_accuracy = evaluate_quantized(
            model=model_prepared,
            test_dataloader=eval_dataloader,
            device=resolved_device
        )
        logging.info(f"Final Accuracy (End of Training): {final_accuracy:.2f}%")


    # Optional: Disable quantization stubs if you want to inspect the FP32 weights again
    # disable_all(model_prepared)
    # model_prepared.eval() # Set back to eval mode if you disabled stubs

    return model_prepared


