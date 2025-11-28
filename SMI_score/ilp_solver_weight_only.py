import argparse
import yaml
import os
import numpy as np
from models.models import create_model
import pandas as pd
#import get_dataloader, get_targets from dataloader.py


from copy import deepcopy
import torch
from SMI_score.score_utils import checkpoint_sizes, find_top_n_configs, total_params
from SMI_score.smi_score import smi_score
from SMI_score.smi_score_ACT import smi_score_act
from SMI_score.score_utils import total_params, model_size_wrt_bitwidth
from models.models import create_model
import copy
import torch # type: ignore


#from mqbench_quantization.quant_old import quantize_ptq_mixed_precision
import pickle

 

def solve(modelname, last_layer_bw=8, compression=10.66):
    model = create_model(modelname, 'imagenet_256x256') 

    quant_dict_base = {}
    for name,a in model.named_modules():
        if isinstance(a, torch.nn.Conv2d):
            quant_dict_base[name] = 8

    size_in_mb, _ = model_size_wrt_bitwidth(model, 32 , quant_dict_base, last_layer_bw=32)
    print('Size of model in MB:', size_in_mb)

    size_in_mb = size_in_mb / compression
    
    _, found_l = model_size_wrt_bitwidth(model, 32, quant_dict_base, last_layer_bw=last_layer_bw)
    #print('Found layers:', found_l['classifier'])
    if 'fc' in found_l:
        conv_size_constrainted = size_in_mb - found_l['fc'] # remove the size of the last layer, as it is not quantized in this case
    elif 'classifier' in found_l:
        conv_size_constrainted = size_in_mb - found_l['classifier']
    print('Constrainted size of conv layers:', conv_size_constrainted)
    #floor the closest size to 2 decimal places
    closest_size = conv_size_constrainted
    #closest_size = round(conv_size_constrainted, 1)
    print('Closest size', closest_size)


    smi_sensitivities, param_constraint, _ = smi_score(modelname, 'imagenet_256x256',BIT_WIDTHS=[2,3,4,5,6,7,8], folder_name = '2345678_bit_results_sota_experiments')
    smi_sensitivities[0] = {2:-1e8,3:-1e8,4:-1e8,5:-1e8,6:-1e8,7:-1e8,8:0} # set the first layer to -inf for all bitwidths

    S = closest_size
    allocations = []
    top_configs = find_top_n_configs(
                1,
                quant_dict_base,
                smi_sensitivities,
                param_constraint,
                max_param_size=S,
                BIT_WIDTHS=[2, 3, 4, 5, 6, 7, 8],
    )
    for config in top_configs.iterrows():
        allocations.append((config[1]['bit_assignment'], config[1]['param_size']))

    for allocation in allocations:
        print(f"Allocation: {allocation[0][1:]} with size {allocation[1]}")
        print(f'First Layer is 8 bit and FC Layer is {last_layer_bw} bit')

        
if __name__ == "__main__":

    #get the model name and dataset from command line arguments
    parser = argparse.ArgumentParser(description='Benchmarking script for SMI')
    parser.add_argument('--modelname', type=str, default='resnet18_gluon',
                        help='Name of the model to benchmark')
    parser.add_argument('--last_layer_bw', type=int, default=8,
                        help='Bitwidth of the last layer')  
    parser.add_argument('--compression', type=float, default=10.66,
                        help='Compression ratio for the model size')

    args = parser.parse_args()
    modelname = args.modelname
    last_layer_bw = args.last_layer_bw
    compression = args.compression

    solve(modelname, last_layer_bw=last_layer_bw, compression=compression)

