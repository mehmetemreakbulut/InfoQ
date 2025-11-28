import argparse
import yaml
import os
import numpy as np
from models.models import create_model
import pandas as pd
#import get_dataloader, get_targets from dataloader.py

from SMI.dataloaders import get_dataloader, get_targets, load_imagenet_256x256_ds

from SMI.input_compressor import get_X, imagenet_256x256_X
from copy import deepcopy
import torch
from mqbench_quantization.quant import mpq_quantize, uniform_quantized_models
from mqbench_quantization.quant_activation import mpq_quantize_ACT
from SMI.utils.smi_calculate import calculate_smi
from SMI.utils.general_functions import evaluate_model
from SMI.utils.mqbench_layer_to_block import conv_to_block_output_dict

import logging
import sys

original_stdout = sys.stdout
original_stderr = sys.stderr


#reload imports
import importlib

from .compress_models import Autoencoder,AdditiveGaussianNoise, CIFAR10_ConvEncoder, CIFAR10_ConvDecoder, get_outputs
class LoggerWriter:
    """
    A class that acts like a stream (e.g., sys.stdout) and redirects
    any 'write' calls to a logger. This is the new destination for `print()`.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        # Only log messages that are not just newline characters
        if message.rstrip() != "":
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        # This flush method is needed for compatibility with stream-like objects.
        pass

# --- Configure logging manually to avoid circular dependencies ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 1. Handler for writing to the log file
file_handler = logging.FileHandler("log.txt", mode='w')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# 2. Handler for writing to the original console
stream_handler = logging.StreamHandler(original_stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# --- Replace stdout and stderr with our logger ---
# After this point, any `print` statement will go to our LoggerWriter
sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.ERROR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt
def main(config):

    
    print(f"Model Name: {config['model_name']}")
    print(f"Dataset: {config['dataset']}")

    model = create_model(config['model_name'], config['dataset'])
    # print model to model_name.txt file
    model_file = f"{config['model_name']}.txt"

    # We temporarily restore original_stdout to write the model file,
    # to avoid this file content from appearing in our log.
    _stdout = sys.stdout
    sys.stdout = original_stdout
    with open(model_file, 'w') as f:
        f.write(str(model))
    sys.stdout = _stdout # a
 
    train_dataloader, _, _ = get_dataloader(config['dataset'], config['num_target_classes'], config['num_samples_per_class'], config['cifar_subset_size'], num_workers=64)
    _, _, train_dataloader_shuffled = get_dataloader(config['dataset'], subset=False, batch_size = 64, num_workers=64)
    
    
    if config['dataset'] == 'imagenet_256x256':
        ds = load_imagenet_256x256_ds(int(config['num_target_classes']), int(config['num_samples_per_class']))
        X_compressed = np.array(imagenet_256x256_X(ds))
    else:
        X_compressed = np.array(get_X(config['dataset'], train_dataloader))
    
    targets = get_targets(train_dataloader, config['dataset'], config['num_target_classes'], config['num_samples_per_class'])


    target_modules = config.get('target_modules', None)

    
    MODEL_TO_QUANTIZE = deepcopy(model).to(device)

    quant_dict_base = {}
    for name,a in MODEL_TO_QUANTIZE.named_modules():
        if isinstance(a, torch.nn.Conv2d):
            quant_dict_base[name] = 8
    print(quant_dict_base)



    BWS = config.get('bitwidths', [4, 2])
    #copy bws
    uniform_models = {}
    eight_act_model = uniform_quantized_models(
            bitwidths= [],
            model_to_quantize=MODEL_TO_QUANTIZE,
            num_calibration_batches=config['num_calibration_batches'],
            calib_dataloader=train_dataloader_shuffled,
            per_channel=config['per_channel'],
            activation_bit=8,
        )[8]
    uniform_models[8] = eight_act_model  # 8-bit model as base
    for bw in BWS:
        m = uniform_quantized_models(
            bitwidths= [],
            model_to_quantize=MODEL_TO_QUANTIZE,
            num_calibration_batches=config['num_calibration_batches'],
            calib_dataloader=train_dataloader_shuffled,
            per_channel=config['per_channel'],
            activation_bit=bw,
        )[8]
        uniform_models[bw] = m

    eight_bit_model = uniform_models[8]  # 8-bit model as base
    #evaluate 8-bit model

    layer_input_map = {}
    def getModuleByName(model,moduleName):
        '''
        replace module with name modelName.moduleName with newModule
        '''
        tokens = moduleName.split('.')
        m = model
        for tok in tokens:
            m = getattr(m,tok)
        return m
    mod = deepcopy(eight_bit_model)
    types_to_quant = (torch.nn.Conv2d,torch.nn.Linear)
    for node in eight_bit_model.graph.nodes:
        try:
            node_target = getModuleByName(mod,node.target)
            if isinstance(node_target,types_to_quant):
                node_args = node.args[0]
                #print('input of ',node.target,' is ',node_args)
                layer_input_map[node.target] = str(node_args.target)
        except:
            continue
    print("Layer input map:", layer_input_map)
   
  
    #print the 8-bit model to file
    eight_bit_model_file = f"{config['model_name']}_8bit_ACT.txt"
    two_bit_model_file = f"{config['model_name']}_2bit_ACT.txt"
    _stdout = sys.stdout
    sys.stdout = original_stdout
    with open(eight_bit_model_file, 'w') as f:
        f.write(str(eight_bit_model))
    with open(two_bit_model_file, 'w') as f:
        f.write(str(uniform_models[2]))
    sys.stdout = _stdout # restore stdout
    
    records = []
    XL_OBSERVERS = config.get('XL_OBSERVERS', [])
    YL_OBSERVERS = config.get('YL_OBSERVERS', [])
    for bw in BWS:
        for layer, quant in quant_dict_base.items():
            copy_quant_dict = quant_dict_base.copy()
            copy_quant_dict[layer] = bw
            model_q = mpq_quantize_ACT(
                layer_bitwidth_dict=copy_quant_dict,
                num_calibration_batches=config['num_calibration_batches'],
                calib_dataloader=train_dataloader_shuffled,
                bit_model_dict=uniform_models,
                layer_input_map=layer_input_map
            )

                
            acc = evaluate_model(model_q, train_dataloader)
            print(f"Accuracy for {layer} with {bw} bits: {acc:.4f}")

            layer_needed = conv_to_block_output_dict(model=config['model_name'], dataset=config['dataset'])[layer]
            smi_results = calculate_smi(
                model_q,
                train_dataloader,
                targets,
                X_compressed,
                number_of_class=config['num_target_classes'],
                m=config['smi_m'],
                target_modules=target_modules,
                layer_needed=layer_needed,
                XL_OBSERVERS=XL_OBSERVERS,
                YL_OBSERVERS=YL_OBSERVERS,
            )
            records.append({
                'layer': layer,
                'bw': bw,
                'acc': acc,
            }
            )
            for k, v in smi_results.items():
                records[-1][k+'_mi_ly'] = v[1]
                records[-1][k+'_mi_xl'] = v[0]
    
    ##EIGHT BIT RECORD
    model_q = uniform_models[8]  # 8-bit model as base

    acc = evaluate_model(model_q, train_dataloader)
    print(f"Accuracy for 8 bit model {acc:.4f}")
    model_q.eval()
    smi_results = calculate_smi(
        model_q,
        train_dataloader,
        targets,
        X_compressed,
        number_of_class=config['num_target_classes'],
        m=config['smi_m'],
        target_modules=target_modules,
        XL_OBSERVERS=XL_OBSERVERS,
        YL_OBSERVERS=YL_OBSERVERS,
        layer_needed=None,  # Process all layers
    )
    records.append({
        'layer': 'all',
        'bw': 8,
        'acc': acc,
    }
    )
    for k, v in smi_results.items():
        records[-1][k+'_mi_ly'] = v[1]
        records[-1][k+'_mi_xl'] = v[0]

    results_smi_df = pd.DataFrame(records)

    #save results and config to a folders, two files: results.yaml and config.yaml, SMI/RUNS/modelname_dataset
    import os
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    results_dir = os.path.join('RUNS', 'ACT', config['model_name'], config['dataset'], timestamp)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'results.csv')
    results_smi_df.to_csv(results_file)
    config_file = os.path.join(results_dir, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    parser = argparse.ArgumentParser(description="SMI Calculation Script using YAML config")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the YAML configuration file. Defaults to config.yaml in the same directory.')
    args = parser.parse_args()

    try:
        file_path = os.path.join(os.path.dirname(__file__), 'configs', args.config)
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{file_path}'")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)

    # Run the main function with the loaded config
    main(config)