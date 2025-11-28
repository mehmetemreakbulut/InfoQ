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
#from SMI.utils.smi_calculate import calculate_smi
from SMI.utils.general_functions import evaluate_model
from SMI.utils.mqbench_layer_to_block import conv_to_block_output_dict

from SMI.dataloaders import get_dataloader
from SMI.utils.general_functions import evaluate_model
from SMI_score.other_algorithms.algorithm_scores import get_hawqv2_sensitivities, get_mpqco_sensitivities, get_ompq_sensitivities, get_limpq_sensitivities
from SMI_score.score_utils import checkpoint_sizes, find_top_n_configs, total_params
from SMI_score.smi_score import smi_score
from SMI_score.smi_score_ACT import smi_score_act
from SMI_score.score_utils import total_params, model_size_wrt_bitwidth
from models.models import create_model
import copy
import torch # type: ignore


#from mqbench_quantization.quant_old import quantize_ptq_mixed_precision




import pickle
#from SMI_score.loader_benchmark import loader_highres_train
def test():

    model = create_model('resnet18_gluon', 'imagenet_256x256') # resnet18(weights="IMAGENET1K_V1")#
    modelname = 'resnet18_gluon'
    #print(f"Benchmarking {modelname} on {dataset} with sizes {sizes}")


    _, test_loader , train_dataloader= get_dataloader('imagenet_256x256',batch_size=64, num_workers=32, subset=False)
    
    #evaluate the model

    #print('BASE MODEL', evaluate_model(model, test_loader))
    #exit()
    quant_dict_base = {}
    for name,a in model.named_modules():
        if isinstance(a, torch.nn.Conv2d):
            quant_dict_base[name] = 8

    size_in_mb, _ = model_size_wrt_bitwidth(model, 32 , quant_dict_base, last_layer_bw=32)
    print('Size of model in MB:', size_in_mb)

    size_in_mb = size_in_mb / 10.66
    
    _, found_l = model_size_wrt_bitwidth(model, 4, quant_dict_base, last_layer_bw=8)
    #print('Found layers:', found_l['classifier'])
    if 'fc' in found_l:
        conv_size_constrainted = size_in_mb - found_l['fc'] # remove the size of the last layer, as it is not quantized in this case
    elif 'classifier' in found_l:
        conv_size_constrainted = size_in_mb - found_l['classifier']
    print('Constrainted size of conv layers:', conv_size_constrainted)
    #floor the closest size to 2 decimal places
    closest_size = conv_size_constrainted
    #closest_size = round(conv_size_constrainted, 1)
    print('Closest size to 1 decimal places:', closest_size)
    with open('logs/test_log.txt', 'w') as f:
        f.write(f"Model size in MB: {size_in_mb}\n")
        f.write(f"Found layers: {found_l}\n")
    


        
    #exit()
    
    '''
    uniform_models = uniform_quantized_models(
        bitwidths= [2,3,4,5,6,7,8].copy(),
        model_to_quantize=copy.deepcopy(model),
        num_calibration_batches=1,
        calib_dataloader=train_dataloader,
        per_channel=False,
        activation_bit=8,
    )
 
    '''
    smi_sensitivities, param_constraint, _ = smi_score(modelname, dataset,BIT_WIDTHS=[2,3,4,5,6,7,8])
    smi_sensitivities[0] = {2:-1e8,3:-1e8,4:-1e8,5:-1e8,6:-1e8,7:-1e8,8:0} # set the first layer to -inf for all bitwidths
    S = 4.1 #resnet18 3bit weight
    S = closest_size ## 3.7 for resnet18
    print(closest_size)
    allocations = []
    top_configs = find_top_n_configs(
                5,
                quant_dict_base,
                smi_sensitivities,
                param_constraint,
                max_param_size=S,
                BIT_WIDTHS=[2, 3, 4, 5, 6, 7, 8],
    )
    for config in top_configs.iterrows():
        allocations.append(config[1]['bit_assignment'])

    for allocation in allocations:
        print(f"Allocation: {allocation}")
        exit()
        copy_quant_dict = copy.deepcopy(quant_dict_base)
        for j, b in enumerate(copy_quant_dict.keys()):
            copy_quant_dict[b] = allocation[j]
        model_q = mpq_quantize(
            layer_bitwidth_dict=copy_quant_dict,
            num_calibration_batches=5,
            calib_dataloader=train_dataloader,
            bit_model_dict=uniform_models,
        )
        print(f"Evaluating model with allocation {allocation}")
        print(evaluate_model(model_q, test_loader))
        
if __name__ == "__main__":

    
    modelname ='mobilenetv2'
    dataset = 'imagenet_256x256'
    #Write the print statements in this file to a file called logs/benchmarking_imagenet_{modelname}_{dataset}.txt
    import os
    import sys
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging to redirect print statements to file
    log_filename = f'logs/FINAL_QAT/imagenet_{modelname}.txt'
    
    # Create a custom print function that writes to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # Ensure immediate writing
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        
        def close(self):
            self.log.close()
    
    # Redirect stdout to our logger
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print(f"Logging output to: {log_filename}")
    print(f"Starting benchmarking for {modelname} on {dataset}")
    print("=" * 50) 

    test()

