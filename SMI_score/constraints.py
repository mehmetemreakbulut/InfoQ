import torch

import math
import pickle
from pulp import *
import time
import argparse
from models.models import create_model
import copy
from SMI_score.smi_score import smi_score
from SMI_score.smi_score_ACT import smi_score_act

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18_gluon', help='model name')
parser.add_argument('--b', type=int, nargs="+", help='bit-width candidates')
parser.add_argument('--wb', type=int, default=3, help='bit-width of weight')
parser.add_argument('--ab', type=int, default=4, help='bit-width of activation')
parser.add_argument('--cr', type=float, default=1, help='compress ratio constraint (enable: >0)')
parser.add_argument('--bops', type=bool, default=True, help='bitops constraint')
parser.add_argument('--alpha', type=float, default=2., help='the hyper-parameter to balance the weight and activation')

args = parser.parse_args()
print(args)

# --------------- model and bit-width settings ---------------
model_name = args.model
model = create_model(model_name, 'imagenet_256x256') 

alpha = args.alpha
avg_bit_w = args.wb
avg_bit_a = args.ab
bit_width_list = args.b

# --------------- searh constraints ---------------
bitops_constraint = args.bops # bitops constraint is calculated through "avg_bit_w" and "avg_bit_a"

compression_rate_constraint = args.cr > 0
compression_ratio = args.cr

# --------------- units ---------------
MB = 1024*1024*8
GBITOPS = 1e9

AVG_BITS_CONS = False

# --------------- calculate the complexity ---------------
def compute_ops_hook(self, input):
    x = input[0]
    w_h = int(((x.shape[2]+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride[0] + 1)
              * ((x.shape[3]+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)/self.stride[1] + 1))
    lw_bitops = (self.in_channels*self.out_channels*w_h *
                 (self.kernel_size[0]**2))//self.groups
    bit_ops.append(lw_bitops)

bit_ops = []
number_of_tensor = []

extra_bitops = 0

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        module.register_forward_pre_hook(compute_ops_hook)

        number_of_tensor.append(
            (module.in_channels*module.out_channels*module.kernel_size[0]*module.kernel_size[1])//module.groups)
    elif isinstance(module, torch.nn.Linear):
        linear_number = module.in_features * module.out_features

model(torch.randn((1, 3, 224, 224)).to('cuda'))
first_layer_bitops = bit_ops[0]
extra_bitops += bit_ops[0] + linear_number

bit_ops = bit_ops[1:]

first_layer_size = number_of_tensor[0]
number_of_tensor = number_of_tensor[1:]

# --------------- layer-wise importance indicators ---------------
layer_wise_act = {}
layer_wise_weight = {}
layer_name_list = []
wired_point = []

#assert os.path.exists(args.i)
#with open(args.i, 'rb') as f:
#    importance_indicators = pickle.load(f)

#layer_wise_weight = importance_indicators['weight']
#layer_wise_act = importance_indicators['act']
#layer_name_list = importance_indicators['layer_name']
#print('weight', len(layer_wise_weight))
#print('Act', layer_wise_act)
#print('layer_name_list', layer_name_list)
#exit()
layer_wise_weight,_, _ = smi_score(model_name, 'imagenet_256x256', BIT_WIDTHS=bit_width_list + [8] , folder_name='2345678_bit_results_sota_experiments')
layer_wise_act, _, _ = smi_score_act(model_name, 'imagenet_256x256', BIT_WIDTHS=bit_width_list + [8])

print('layer_wise_weight', layer_wise_weight)
print('layer_wise_act', layer_wise_act)

layer_name_list = []
for name,a in model.named_modules():
    if isinstance(a, torch.nn.Conv2d):
       layer_name_list.append('module.'+name)
for i, name in enumerate(layer_name_list):
    if name not in layer_wise_weight:
        val = layer_wise_weight[i]
        #make keys in val a string
        val = {str(k): -v for k, v in val.items()}
        layer_wise_weight[name] = val
        del layer_wise_weight[i]
    if name not in layer_wise_act:
        val = layer_wise_act[i]
        #make keys in val a string
        val = {str(k): -v for k, v in val.items()}
        layer_wise_act[name] = val
        del layer_wise_act[i]

print('layer_wise_weight', layer_wise_weight)
print('layer_wise_act', layer_wise_act)
if len(layer_wise_weight) != len(layer_wise_act):
    print('Error: layer_wise_weight and layer_wise_act have different lengths')
    print('layer_wise_weight', layer_wise_weight)
    print('layer_wise_act', layer_wise_act)
    print('layer_name_list', layer_name_list)
print(len(layer_wise_weight), len(layer_wise_act), len(layer_name_list))
del layer_wise_weight['module.conv1']
del layer_wise_act['module.conv1']

#remove '8' from layer_wise_weight and layer_wise_act
'''
for name in layer_wise_weight:
    if '8' in layer_wise_weight[name]:
        del layer_wise_weight[name]['8']
for name in layer_wise_act:
    if '8' in layer_wise_act[name]:
        del layer_wise_act[name]['8']
'''

#discard module.conv1 from layer_name_list
layer_name_list = [name for name in layer_name_list if name != 'module.conv1']
if len(layer_wise_weight) != len(layer_wise_act):
    print('Error: layer_wise_weight and layer_wise_act have different lengths')
    print('layer_wise_weight', layer_wise_weight)
    print('layer_wise_act', layer_wise_act)
    print('layer_name_list', layer_name_list)
print(len(layer_wise_weight), len(layer_wise_act), len(layer_name_list))

'''
ERASE_WIRED_POINT = True # the joint training causes some outliers, here we simply erase them
C = 1e8 # give wired points a huge value to make sure not choose them

if ERASE_WIRED_POINT:
    def pre_process_indicators(indicators):
        for k, v in indicators.items():
            wired_point = []
            for bit_width in bit_width_list[:-1]:
                if v[str(bit_width-1)] < v[str(bit_width)]:
                    wired_point.append(str(bit_width-1))
            
            for w in wired_point:
                print(v[w])
                v[w] = C
    
    pre_process_indicators(layer_wise_weight)
    pre_process_indicators(layer_wise_act)
'''
# --------------- ILP-based bit-width assignment ---------------
problem = LpProblem('Bit-width allocation', LpMinimize)

varible = {}
for i in range(len(layer_name_list)):
    for j in range(len(bit_width_list)):
        for k in range(len(bit_width_list)):
            varible[f"l{i}_w{j}_a{k}"] = LpVariable(
                f"l{i}_w{j}_a{k}", 0, 1, cat=LpInteger)

for i in range(len(layer_name_list)):
    problem += sum([varible[f"l{i}_w{j}_a{k}"]
                   for j in range(len(bit_width_list)) for k in range(len(bit_width_list))]) == 1

cons = lpSum([bit_ops[i] * avg_bit_w *
             avg_bit_a for i in range(len(layer_name_list))])
print(layer_wise_act[layer_name_list[i]])
problem += lpSum([(layer_wise_act[layer_name_list[i]][str(bit_width_list[k])] + alpha * layer_wise_weight[layer_name_list[i]][str(bit_width_list[j])])
                 * varible[f"l{i}_w{j}_a{k}"] for i in range(len(layer_name_list)) for j in range(len(bit_width_list)) for k in range(len(bit_width_list))])

if bitops_constraint:
    problem += lpSum([bit_ops[i]*varible[f"l{i}_w{j}_a{k}"]*bit_width_list[j]*bit_width_list[k] for i in range(
        len(layer_name_list)) for j in range(len(bit_width_list)) for k in range(len(bit_width_list))]) <= cons

# avg_bit cons
if AVG_BITS_CONS:
    problem += lpSum([varible[f"l{i}_w{j}_a{k}"]*bit_width_list[j] for i in range(len(layer_name_list))
                     for j in range(len(bit_width_list)) for k in range(len(bit_width_list))])/len(layer_name_list) <= avg_bit_w
    problem += lpSum([varible[f"l{i}_w{j}_a{k}"]*bit_width_list[k] for i in range(len(layer_name_list))
                     for j in range(len(bit_width_list)) for k in range(len(bit_width_list))])/len(layer_name_list) <= avg_bit_a

# compress radio cons

extra_model_size = (linear_number + first_layer_size) * 8
total_params = sum([32*i for i in number_of_tensor]) + \
    32 * (linear_number + first_layer_size)
print('total_params', total_params)

if compression_rate_constraint:
    problem += (extra_model_size+lpSum([varible[f"l{i}_w{j}_a{k}"]*bit_width_list[j]*number_of_tensor[i] for i in range(
        len(layer_name_list)) for j in range(len(bit_width_list)) for k in range(len(bit_width_list))])) <= total_params / compression_ratio

stime = time.time()
ret = problem.solve()
print(f'ILP solver consumes {time.time() - stime} s')

weight_bits = []
act_bits = []

w_sum = 0
a_sum = 0
total_bitops = 0
avg_bitops = 0
weight_sum = 0

for i in range(len(layer_name_list)):
    for j in range(len(bit_width_list)):
        for k in range(len(bit_width_list)):
            if value(varible[f"l{i}_w{j}_a{k}"]) == 1.:
                print(
                    'layer', layer_name_list[i], 'weight', bit_width_list[j], 'act', bit_width_list[k])
                total_bitops += bit_ops[i] * bit_width_list[j] * bit_width_list[k]
                avg_bitops += bit_ops[i] * avg_bit_a * avg_bit_w
                w_sum += bit_width_list[j]
                a_sum += bit_width_list[k]
                weight_bits.append(bit_width_list[j])
                act_bits.append(bit_width_list[k])
                weight_sum += bit_width_list[j] * number_of_tensor[i]

weight_sum = sum([weight_bits[i] * number_of_tensor[i]
                 for i in range(len(weight_bits))])

linear_weight_bits = 8
first_conv_layer_weight_bits = 8

quantized_model_size = weight_sum + linear_number * linear_weight_bits + first_layer_size * first_conv_layer_weight_bits

print("*"*80)
print('avg weight', w_sum/len(layer_name_list),
      'avg act', a_sum/len(layer_name_list))
print('compress radio', (total_params)/(quantized_model_size))
print("FP model size (MB)", round(total_params/MB, 3), 'searched model size (MB)', round((quantized_model_size)/MB, 3))
print("*"*80)

linear_act_bits = 8
first_conv_layer_act_bits = 8
print((first_layer_bitops * (first_conv_layer_act_bits * first_conv_layer_weight_bits))/GBITOPS, 'extra bitops for first layer')
fp_bitops = sum(
    [bit_ops[i] * 32 * 32 for i in range(len(layer_name_list))]) + extra_bitops * 32

extra_bitops = (first_layer_bitops * (first_conv_layer_act_bits * first_conv_layer_weight_bits) ) + (linear_number * (linear_act_bits * linear_weight_bits)) 
#extra_bitops *= (linear_act_bits * linear_weight_bits) #also for input we use (8,8)
target_bitops = sum([bit_ops[i] * avg_bit_w *
                    avg_bit_a for i in range(len(layer_name_list))]) + extra_bitops
total_bitops += extra_bitops
total_bitops = sum([bit_ops[i] * weight_bits[i] * act_bits[i]
                   for i in range(len(weight_bits))]) + extra_bitops
print('bitops radio (fp_bitops/target_bitops)', round(fp_bitops/total_bitops, 3), 'bitops radio (searched_bitops/target_bitops)',
      round(total_bitops/target_bitops, 3), 'searched bitops', round(total_bitops/GBITOPS, 3), 'target bitops', round(target_bitops/GBITOPS, 3))

print("*"*80)
print('searched weight bit-widths', weight_bits)
print('searched act bit-widths', act_bits)