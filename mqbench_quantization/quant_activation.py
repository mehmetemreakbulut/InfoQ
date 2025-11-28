from typing import Optional

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import copy # To avoid modifying the original model
import logging # Optional: for better feedback

# MQBench imports
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.fake_quantize import FixedFakeQuantize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_column = 'image'
label_column = 'label'
# May potentially need QuantizeScheme if constructing it manually, but let's try dicts first
# from mqbench.scheme import QuantizeScheme

# Setup basic logging (optional)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def getModuleByName(model,moduleName):
    '''
        replace module with name modelName.moduleName with newModule
    '''
    tokens = moduleName.split('.')
    m = model
    for tok in tokens:
        m = getattr(m,tok)
    return m

# ---- HELPER FUNCTION TO UPDATE BATCHNORM STATS ---
def set_bn_to_train_mode(model):
    """
    Sets all BatchNorm layers in the model to training mode.
    This is useful for updating running stats during calibration.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()

def uniform_quantized_models(bitwidths: Optional[list] = [8, 4, 2],
                            model_to_quantize: Optional[nn.Module] = None,
                            num_calibration_batches: int = 100,
                            calib_dataloader: Optional[torch.utils.data.DataLoader] = None,
                            per_channel: bool = False,
                            activation_bit: int = 8,  # Default activation bitwidth
):
    if 8 not in bitwidths:
        bitwidths.append(8)
    bit_model_dict = {}
    for bit in bitwidths:
        bit_model_dict[bit] = {}
        uniform_quantized = uniform_quantize(
            model_to_quantize,
            calib_dataloader,
            num_calibration_batches=num_calibration_batches,
            BIT=bit,
            per_channel=per_channel,
            activation_bit=activation_bit, 
        )
        bit_model_dict[bit] = copy.deepcopy(uniform_quantized)

    return bit_model_dict

def mpq_quantize_ACT(
    layer_bitwidth_dict: dict,
    num_calibration_batches: int = 100,
    calib_dataloader: Optional[torch.utils.data.DataLoader] = None,
    bit_model_dict: Optional[dict] = None,
    layer_input_map: Optional[dict] = None,  # Optional mapping for layer inputs
):
    #append 8 to bitwidths if not in list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    copy_base_model = copy.deepcopy(bit_model_dict[8]) # 8-bit model as base

    # --- SET MIXED PRECISION LAYERS ---
    # Because we calibrate the activation observers again, changing layer by layer is same as initial MPQ construction. See similar implementation https://github.com/JamesTuna/CLADO_MPQ

    for ll in layer_bitwidth_dict.keys():
        bit = layer_bitwidth_dict[ll]
        if bit == 8:
            continue
        print(f"Setting layer {ll} to {bit}-bit quantization")
        target = bit_model_dict[bit]
        a_cmd = f'copy_base_model.{layer_input_map[ll]}=target.{layer_input_map[ll]}'
        print(a_cmd)
        exec(a_cmd)
        continue
        '''
        base_module = getModuleByName(copy_base_model, ll)
        target_module = getModuleByName(bit_model_dict[bit], ll)
        base_module.weight_fake_quant = target_module.weight_fake_quant
        '''

    copy_base_model.eval() # Set the model to evaluation mode
    set_bn_to_train_mode(copy_base_model) # to calculate batch-norm statistics again, not so important if dist is not changing too much
    copy_base_model.to(device) # Move the model to the correct device

    # ---- CALIBRATION ------
    logging.info(f"Starting calibration using {num_calibration_batches} batches...")
    enable_calibration(copy_base_model) # Enable observers
    device = next(copy_base_model.parameters()).device # Get model device (CPU/GPU)
    calib_count = 0
    with torch.no_grad():
        for batch in tqdm(calib_dataloader, desc="Calibrating model", total=num_calibration_batches):
            data = batch['image'] 
            data = data.to(device)
            copy_base_model(data)
            calib_count += 1
            if calib_count >= num_calibration_batches:
                logging.info(f"Calibration completed after {calib_count} batches.")
                break
        else: # Executed if the loop finishes without break
             logging.warning(f"Calibration finished after {calib_count} batches (fewer than requested {num_calibration_batches}).")



    enable_quantization(copy_base_model) # Enable fake quantizers using calibrated stats
    return copy_base_model.eval().to(device) # Ensure model is in eval mode and on the correct device



def uniform_quantize(
    model_to_quantize: nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    num_calibration_batches: int = 100,
    BIT = 8,
    activation_bit: int = 8,
    per_channel: bool = False,
) -> nn.Module:
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
                'per_channel': per_channel,                                  # custom whether quant is per-channel or per-tensor for weight,
                'pot_scale': False,                                   # custom whether scale is power of two for weight.
            },
            'a_qscheme': {
                'bit': activation_bit,                                             # custom bitwidth for activation,
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


    model_prepared = prepare_by_platform(model, backend, custom_config)

    model_prepared = model_prepared.to(device)

    model_prepared.eval() # Ensure the model is in eval mode
    set_bn_to_train_mode(model_prepared) # Set BatchNorm layers to train mode for calibration
    model_prepared.to(device)
    # print(model_prepared) # Optional: Inspect the prepared model structure

    # --- 2. Calibration ---
    logging.info(f"Starting calibration using {num_calibration_batches} batches...")
    enable_calibration(model_prepared) # Enable observers

    calib_count = 0 # Initialize counter
    logging.info(f"Starting calibration using {num_calibration_batches} batches...")

    image_column='image'
    with torch.no_grad():
        # Iterate over the dataloader, getting one dictionary batch at a time
        for batch in tqdm(eval_dataloader, desc="Calibrating model", total=num_calibration_batches):
 
            data = batch[image_column].to(device)

            model_prepared(data) # Run the forward pass for calibration
            calib_count += 1

            if (calib_count % 20 == 0) and calib_count > 0 :
                logging.info(f"  Calibration progress: {calib_count}/{num_calibration_batches} batches processed.")

            if calib_count >= num_calibration_batches:
                logging.info(f"Calibration completed after reaching target of {calib_count} batches.")
                break # Exit the loop
        else:
            logging.warning(f"Calibration finished after processing all {calib_count} batches available in the dataloader, which is fewer than the requested {num_calibration_batches}.")

    enable_quantization(model_prepared) # Enable fake quantizers using calibrated stats
    return model_prepared.eval().to(device)