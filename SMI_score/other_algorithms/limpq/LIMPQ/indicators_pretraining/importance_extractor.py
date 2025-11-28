import torch
import pickle

checkpoint_path = 'SMI_score/other_algorithms/limpq/LIMPQ/indicators_pretraining/out/resnet50_gluon_842/resnet50_gluon_842_checkpoint.pth.tar'
importance_indicators_save_path = 'SMI_score/other_algorithms/limpq/LIMPQ/importance_indicators.pkl'

cand = [8,4,2] # same bit-width candidates in the indicator pretraining YAML file. 

state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']

act, weight, layer_name = {}, {}, []

for k, v in state_dict.items():
    k: str

    if k.endswith('.quan_a_fn.s'):
        layer = k.split('.quan_a_fn.s')[0]
        assert layer not in act
        act[layer] = {}

        for bit_witdh, activation_scale_factor_value in zip(cand, v):
            # print(activation_scale_factor_value)
            act[layer][str(bit_witdh)] = activation_scale_factor_value.item()
    
    if k.endswith('.quan_w_fn.s'):
        layer = k.split('.quan_w_fn.s')[0]
        assert layer not in weight
        weight[layer] = {}

        for bit_witdh, weight_scale_factor_value in zip(cand, v):
            weight[layer][str(bit_witdh)] = weight_scale_factor_value.item()
        
        layer_name.append(layer)
    
print(weight)
with open(importance_indicators_save_path, 'wb') as f:
    pickle.dump({
        'weight': weight,
        'act': act,
        'layer_name': layer_name
    }, f)