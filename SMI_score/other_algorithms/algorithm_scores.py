import pickle
import numpy as np
import torch
import cvxpy as cp

from SMI.dataloaders import get_dataloader
from SMI_score.other_algorithms.ompq.calculate_ompq_score import get_ort_score, get_sensitivities

def getModuleByName(model,moduleName):
    '''
        replace module with name modelName.moduleName with newModule
    '''
    tokens = moduleName.split('.')
    m = model
    for tok in tokens:
        m = getattr(m,tok)
    return m



def init(modelname='mobilenetv2_x0_5',dataset='cifar10', BATCH_START=0,BATCH_END=16, aw_scheme=[(8, 2), (8, 4), (8, 8)]):
    batch_Ltildes_mpqco = []

    ref_index2layerscheme = []

    with open(f'SMI_score/other_algorithms/mpqco/MPQCO_DELTAL_{dataset}_{modelname}/MPQCO_DELTAL_{dataset}_{modelname}_batch{BATCH_START}(size64).pkl','rb') as f:
        hm = pickle.load(f)
        print(hm)
        for i in hm.keys():
            if i == 'n_samples':
                continue
            layer_name = i
            for sc in aw_scheme:
                ref_index2layerscheme.append((layer_name, str(sc)+'bits'))
    shape = (len(ref_index2layerscheme), len(ref_index2layerscheme))

    for batch_id in range(BATCH_START,BATCH_END):
        with open(f'SMI_score/other_algorithms/mpqco/MPQCO_DELTAL_{dataset}_{modelname}/MPQCO_DELTAL_{dataset}_{modelname}_batch{batch_id}(size64).pkl','rb') as f:
            hm = pickle.load(f)


        deltal = np.zeros(shape)

        for layer_id in range(len(ref_index2layerscheme)):
            layer_name,scheme = ref_index2layerscheme[layer_id]
            wbit = eval(scheme[:-4])[1]

            deltal[layer_id,layer_id] = hm[layer_name][wbit]

        batch_Ltildes_mpqco.append(deltal)
    batch_Ltildes_mpqco = np.array(batch_Ltildes_mpqco)
    ref_Ltilde_mpqco = batch_Ltildes_mpqco.mean(axis=0)
  

    return None, ref_Ltilde_mpqco, ref_index2layerscheme, shape

def get_hawqv2_sensitivities(eight_bit_model=None, four_bit_model=None, two_bit_model=None, quant_dict_base=None, modelname='mobilenetv2_x0_5', dataset='cifar10', BATCH_START=0, BATCH_END=16, aw_scheme=[(8, 2), (8, 4), (8, 8)], MPQ_scheme= [2,4,8]):

    _, ref_Ltilde_mpqco, ref_index2layerscheme, shape = init(modelname=modelname, dataset=dataset, BATCH_START=BATCH_START, BATCH_END=BATCH_END, aw_scheme=aw_scheme)

    batch_Ltildes_hawq = []
    dw_list = []
    for batch_id in range(BATCH_START, BATCH_END):
        with open(f'SMI_score/other_algorithms/hawq/HAWQ_DELTAL_{dataset}_{modelname}/TraceEST_batch{batch_id}.pkl', 'rb') as f:
            trace = pickle.load(f)

        deltal = np.zeros(shape)
        for layer_id in range(len(ref_index2layerscheme)):
            layer_name,scheme = ref_index2layerscheme[layer_id]
            w_bit = eval(scheme[:-4])[1]
            if w_bit == 8:
                tar_module = getModuleByName(eight_bit_model, layer_name)
            elif w_bit == 4:  
                tar_module = getModuleByName(four_bit_model, layer_name)
            elif w_bit == 2:
                tar_module = getModuleByName(two_bit_model, layer_name)
            else:
                raise ValueError(f"Unsupported bit width: {w_bit}")
            
            dw = tar_module.weight_fake_quant(tar_module.weight) - tar_module.weight
            dw_list.append((dw**2).sum().item())
            deltal[layer_id, layer_id] = (np.array(trace[layer_name]).mean()/dw.nelement()) * (dw**2).sum().item()
        batch_Ltildes_hawq.append(deltal)
    batch_Ltildes_hawq = np.array(batch_Ltildes_hawq)
    ref_Ltilde_hawq = batch_Ltildes_hawq.mean(axis=0)
    print(ref_Ltilde_hawq.shape)
    cached_grad = ref_Ltilde_hawq
    if cached_grad.__class__ == torch.Tensor:
        cached_grad = cached_grad.cpu().numpy()

    x = cp.Variable(cached_grad.shape[0], boolean=True)
    schemes_per_layer = len(aw_scheme)
    assert cached_grad.shape[0]%schemes_per_layer == 0, 'cached_gradient shape[0] does not divde schemes per layer'

    omegas =  np.diagonal(cached_grad)
    sensitivity_dic = {}
    dw_dict = {}
    for i in range(len(quant_dict_base.keys())):
        sensitivity_dic[i] = {}
        dw_dict[i] = {}
        for bwi, BW in enumerate(MPQ_scheme):
            sensitivity_dic[i][BW] = -(omegas[i * 3 + bwi].item())
            dw_dict[i][BW] = dw_list[i * 3 + bwi]
    
    return sensitivity_dic, dw_dict


def get_mpqco_sensitivities(quant_dict_base=None, modelname='mobilenetv2_x0_5', dataset='cifar10', BATCH_START=0, BATCH_END=16, aw_scheme=[(8, 2), (8, 4), (8, 8)], MPQ_scheme= [2,4,8]):
    _, ref_Ltilde_mpqco,_, shape = init(modelname, dataset, BATCH_START, BATCH_END, aw_scheme)
    cached_grad = ref_Ltilde_mpqco
    if cached_grad.__class__ == torch.Tensor:
        cached_grad = cached_grad.cpu().numpy()
    x = cp.Variable(cached_grad.shape[0], boolean=True)
    schemes_per_layer = len(aw_scheme)
    assert cached_grad.shape[0]%schemes_per_layer == 0, 'cached_gradient shape[0] does not divde schemes per layer'
    num_layers = cached_grad.shape[0]//schemes_per_layer
    omegas =  np.diagonal(cached_grad)
    sensitivity_dic = {}
    dw_dict = {}
    for i in range(len(quant_dict_base.keys())):
        sensitivity_dic[i] = {}
        dw_dict[i] = {}
        for bwi, BW in enumerate(MPQ_scheme):
            sensitivity_dic[i][BW] = -(omegas[i * 3 + bwi].item())
            dw_dict[i][BW] = 0
    # dw_dict[i][BW] = dw_list[i * 3 + bwi]
    return sensitivity_dic, dw_dict



BETAS = {
    'mobilenetv2_x0_5':  0.000000001, #from source code of OMPQ, for imagenet but use cifar also
    'resnet20': 10, # in source it is 100 but gives 0 for all ortg. calculation, decreased to 10 find best results
    'resnet18_gluon': 1, #from source code
    'resnet50_gluon': 1, #from source  code
    'mobilenetv2': 0.000000001, #from source code
}

scales = {
    'mobilenetv2_x0_5':  0,
    'resnet20': 65,
    'resnet18_gluon': 10, #
    'resnet50_gluon': 20, 
    'mobilenetv2': 0
}
def get_ompq_sensitivities(model, modelname,dataset, quant_dict_base, bitwidths, base_bit, discard_first=False, discard_last=True):
    _, _, train_loader_shuffled = get_dataloader(dataset=dataset, batch_size=64, subset=False)

    theta = get_sensitivities(model,    BETAS[modelname]  , train_loader_shuffled, model_name=modelname, dataset=dataset) # use beta given in the souce code of OMPQ
    if discard_first:
        theta = theta[1:]  # Discard the first layer sensitivity
    if discard_last:
        theta = theta[:-1]
    ompq_score = get_ort_score(quant_dict_base, theta, bitwidths, base_bit)
    print('ompq_score', ompq_score)
    scaled_ompq_score = {}
    for key, value in ompq_score.items():
        bit_dict = ompq_score[key]
        scaled_ompq_score[key] = {}
        for k, v in bit_dict.items():
            scaled_ompq_score[key][k] = v * 10**scales[modelname]  # Scale the score by the factor defined in scales dictionary

    return scaled_ompq_score
    

def get_limpq_sensitivities(model_name, cand = [8,4,2]):
    import torch
    import pickle
    cand_str = ''.join([str(i) for i in cand])
    base_path = 'SMI_score/other_algorithms/limpq/LIMPQ/indicators_pretraining/out/'
    checkpoint_path = base_path + f'{model_name}_{cand_str}/{model_name}_{cand_str}_checkpoint.pth.tar'
    #importance_indicators_save_path = 'importance_indicators.pkl'

     # same bit-width candidates in the indicator pretraining YAML file.

    state_dict = torch.load(checkpoint_path)['state_dict']

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

    importance_indicators = {
            'weight': weight,
            'act': act,
            'layer_name': layer_name
        }
    limpq_sensitivity_raw = importance_indicators['weight']
    limpq_sensitivity = {}
    keys = []
    for i,key in enumerate(limpq_sensitivity_raw.keys()):
        #remove module in key with ''
        #new_key = key.replace('module.','')
        keys.append(key)
        if limpq_sensitivity_raw[key]['2'] < limpq_sensitivity_raw[key]['4']:
            print('WIRED FOUND 2')
            limpq_sensitivity_raw[key]['2'] = 1e8
        elif limpq_sensitivity_raw[key]['4'] < limpq_sensitivity_raw[key]['8']:
            print('WIRED FOUND 4')
            limpq_sensitivity_raw[key]['4'] = 1e8            
        values = {
            2: -limpq_sensitivity_raw[key]['2'],
            4: -limpq_sensitivity_raw[key]['4'],
            8: -limpq_sensitivity_raw[key]['8'],
        }
        limpq_sensitivity[i] = values
    print(keys)
    return limpq_sensitivity