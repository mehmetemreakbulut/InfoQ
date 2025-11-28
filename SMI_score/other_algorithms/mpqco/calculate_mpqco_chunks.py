import argparse

import os
import pdb,time
import torch,torchvision,time,pickle
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.utils.state import disable_all
from copy import deepcopy
import torch.nn.functional as F
import argparse

from models.models import create_model
from SMI.dataloaders import get_dataloader

torch.manual_seed(42)
np.random.seed(42)


def MPQCOCalculate(modelname, dataset, batch_size_train=64, MPQ_scheme=(2,4,8), per_channel=False, activation_bit=8, START_BATCH=0, END_BATCH=16):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mn = dataset.lower()+ '_' + modelname


    model = create_model(modelname, dataset)
    model = deepcopy(model) 


    _, _, train = get_dataloader(dataset=dataset, batch_size=batch_size_train, subset=False)


    # calibration data used to calibrate PTQ and MPQ
    calib_data = []
    stacked_tensor = []
    calib_fp_output = []
    i = 0
    START_BATCH = START_BATCH
    END_BATCH = END_BATCH
    with torch.no_grad():
        for batch in train:
            img, label = batch['image'], batch['label']
            # stacked_tensor is to calibrate the model (1024 samples)
            if i< 1024/batch_size_train: # always use 1024 samples to calibrate
                stacked_tensor.append(img)
            # calib_data is the sensitivity set (number of samples specified by user)
            if i>= START_BATCH and i< END_BATCH:
                img = img.to(device)
                label = label.to(device)
                calib_data.append((img,label))
            if i >= END_BATCH:
                break
            i += 1
    model.eval()
    # configuration


    for b in MPQ_scheme:
        mqb_fp_model = deepcopy(model)

        # MSE calibration on model parameters
        backend = BackendType.Academic
        extra_config = {
            'extra_qconfig_dict': {
                'w_observer': 'MSEObserver',
                'a_observer': 'EMAMSEObserver',
                'w_fakequantize': 'FixedFakeQuantize',
                'a_fakequantize': 'FixedFakeQuantize',
                'w_qscheme': {
                    'bit': b,
                    'symmetry': True,
                    'per_channel': per_channel,
                    'pot_scale': False,
                },
                'a_qscheme': {
                    'bit': activation_bit,
                    'symmetry': False,
                    'per_channel': False,
                    'pot_scale': False,
                }
            }
        }
        print(f'Prepare {b}bits model using MQBench')

        exec(f'mqb_{b}bits_model=prepare_by_platform(mqb_fp_model, backend,extra_config).cuda(device)')

        # calibration loop
        enable_calibration(eval(f'mqb_{b}bits_model'))
        print(f'Calibration with {len(stacked_tensor)} batch')
        for img in stacked_tensor:
            
            eval(f'mqb_{b}bits_model')(img.cuda(device))

    for b in MPQ_scheme:
        disable_all(eval(f'mqb_{b}bits_model'))
        # evaluation loop
        enable_quantization(eval(f'mqb_{b}bits_model'))
        eval(f'mqb_{b}bits_model').eval()
        print(f'Evaluate {b}bits model using MQBench')


    mqb_fp_model = deepcopy(eval('mqb_8bits_model'))
    disable_all(mqb_fp_model)
    mqb_mix_model = deepcopy(mqb_fp_model)

    def getModuleByName(model,moduleName):
        '''
            replace module with name modelName.moduleName with newModule
        '''
        tokens = moduleName.split('.')
        m = model
        for tok in tokens:
            m = getattr(m,tok)
        return m

    def perturb(perturb_scheme):
        # perturb_scheme: {layer_name:(act_bits,weight_bits)}
        for layer_name in perturb_scheme:
            a_bits,w_bits = perturb_scheme[layer_name]

            if w_bits is not None:
                mix_module = getModuleByName(mqb_mix_model,layer_name)
                tar_module = getModuleByName(eval(f'mqb_{w_bits}bits_model'),layer_name)
                # replace weight quant to use a_bits quantization
                w_cmd = f'mix_module.weight_fake_quant=tar_module.weight_fake_quant'
                exec(w_cmd)

            if a_bits is not None:

                # replace act quant to use w_bits quantization
                a_cmd = f'mqb_mix_model.{layer_input_map[layer_name]}=mqb_{a_bits}bits_model.{layer_input_map[layer_name]}'
                exec(a_cmd)

    # 1. record all modules we want to quantize
    types_to_quant = (torch.nn.Conv2d,)

    layer_input_map = {}

    first,last = None,None
    for node in eval('mqb_8bits_model').graph.nodes:
        try:
            node_target = getModuleByName(mqb_mix_model,node.target)
            if isinstance(node_target,types_to_quant):
                node_args = node.args[0]
                print('input of ',node.target,' is ',node_args)
                layer_input_map[node.target] = str(node_args.target)
                if first is None:
                    first = node.target
                last = node.target
        except:
            continue
    #print(f'first conv layer and last conv layer are',first,last)

    #aw_scheme = [(8,2),(8,4),(8,8)]

    #del layer_input_map[first] # usually the first conv layer and the last fc layers (already excluded) are kept 8 bits

    

    def estimate_deltaL(eval_data,wbit_choices=[2,4,8], models = {}):
        for b in wbit_choices:
            exec(f'mqb_{b}bits_model=models[{b}]')

        tot_batches = len(eval_data)

        processed_batches = 0

        print(f'MPQCO Ltilde: {processed_batches}/{tot_batches} batch of data processed')

        for batch_img,batch_label in eval_data:

            s_time = time.time()
            # init deltaL dictionary
            deltaL = {}
            for layer in layer_input_map:
                if layer in ('fc'):
                    continue
                deltaL[layer] = {}
                for wbit in wbit_choices:
                    deltaL[layer][wbit] = 0

            for i in range(batch_img.size(0)):

                img,label = batch_img[i].unsqueeze(0),batch_label[i]
                model.zero_grad()
                logits = model(img.cuda(device))
                logits[0][label].backward()

                with torch.no_grad():
                    for layer_name in layer_input_map:
                        if layer_name in ('fc'):
                            continue
                        for w_bits in wbit_choices:
                            tar_module = getModuleByName(eval(f'mqb_{w_bits}bits_model'),layer_name)
                            dw = tar_module.weight_fake_quant(tar_module.weight) - tar_module.weight
                            dl = (dw * getModuleByName(model,layer_name).weight.grad).sum()
                            dl /= logits[0][label]
                            dl = dl ** 2
                            deltaL[layer_name][w_bits] += dl.cpu().numpy()

            for layer in layer_input_map:
                if layer in ('fc'):
                    continue
                for wbit in wbit_choices:
                    deltaL[layer][wbit] /= 2 * batch_img.size(0)

            deltaL['n_samples'] = batch_img.size(0)

            if not os.path.exists(f'SMI_score/other_algorithms/mpqco/MPQCO_DELTAL_{dataset}_{modelname}'):
                os.mkdir(f'SMI_score/other_algorithms/mpqco/MPQCO_DELTAL_{dataset}_{modelname}')

            with open(f'SMI_score/other_algorithms/mpqco/MPQCO_DELTAL_{dataset}_{modelname}/MPQCO_DELTAL_{dataset}_{modelname}_batch{START_BATCH+processed_batches}(size{batch_size_train}).pkl','wb') as f:
                pickle.dump(deltaL,f)

            processed_batches += 1

            print(f'MPPCO Ltilde: {processed_batches+START_BATCH} {processed_batches}/{tot_batches} batch of data processed')
            print(f'batch cost:{time.time()-s_time:.2f} seconds')

    import time

    # MPQCO sensitivities
    bit_choices = list(MPQ_scheme)
    models = {}
    for b in bit_choices:
        models[b] = eval(f'mqb_{b}bits_model')
    estimate_deltaL(calib_data,wbit_choices=bit_choices, models=models)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPQCO Sensitivity Calculation')
    parser.add_argument('--modelname', type=str, default='resnet18', help='Model name')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--batch_size_train', type=int, default=64, help='Batch size for training')
    parser.add_argument('--MPQ_scheme', type=int, nargs='+', default=[2, 4, 8], help='MPQ bit scheme')
    parser.add_argument('--per_channel', action='store_true', help='Use per-channel quantization') #false default
    parser.add_argument('--activation_bit', type=int, default=8, help='Activation bit width')
    parser.add_argument('--start_batch', type=int, default=0, help='Starting batch index for calibration.')
    parser.add_argument('--end_batch', type=int, default=16, help='Ending batch index for calibration.')

    args = parser.parse_args()

    print('Per channel:', args.per_channel)


    MPQCOCalculate(args.modelname, args.dataset, args.batch_size_train, args.MPQ_scheme, args.per_channel, args.activation_bit, args.start_batch, args.end_batch)
