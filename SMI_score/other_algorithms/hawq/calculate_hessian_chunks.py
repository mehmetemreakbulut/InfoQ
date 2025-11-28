import os
import time
import torch,torchvision,time,pickle
import numpy as np

from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from copy import deepcopy
import torchvision as tv
import torch.nn.functional as F
import argparse
from pyhessian import hessian # Hessian computation
torch.manual_seed(42)
np.random.seed(42)

from SMI.dataloaders import get_dataloader
from models.models import create_model



def HAWQv2Calculate(modelname, dataset, batch_size_train=64, 
                    START_BATCH=0, END_BATCH=16):
    """
    Calculate HAWQv2 Hessian traces for a given model and dataset.
    :param model_name: Name of the model to be used.
    :param dataset: Dataset to be used for calibration.
    :param batch_size_train: Batch size for training.
    """
    # Set up the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mn = dataset.lower()+ '_' + modelname


    model = create_model(modelname, dataset)
    model = deepcopy(model) 


    _, _, train = get_dataloader(dataset=dataset, batch_size=batch_size_train, subset=False)

    calib_data = []
    i = 0
    START_BATCH = START_BATCH
    END_BATCH = END_BATCH
    with torch.no_grad():
        for i,batch in enumerate(train):
            img, label = batch['image'], batch['label']
            # calib_data is the sensitivity set (number of samples specified by user)

            if i>= START_BATCH and i< END_BATCH:
                img = img.to(device)
                label = label.to(device)
                calib_data.append((img,label))
            if i >= END_BATCH:
                break
            i += 1

    def getModuleByName(model,moduleName):
        '''
            replace module with name modelName.moduleName with newModule
        '''
        tokens = moduleName.split('.')
        m = model
        for tok in tokens:
            m = getattr(m,tok)
        return m
    
    criterion = torch.nn.CrossEntropyLoss()
    def estimate_trace(eval_data,layer_name):
        torch.manual_seed(0)
        np.random.seed(0)
        print(f'compute trace for layer {layer_name}')
        # first disable gradients of all modules
        for n,m in model.named_parameters():
            m.requires_grad = False
        # enable gradients for only that layer
        getModuleByName(model,layer_name).weight.requires_grad = True
        for m in model.parameters():
            if m.requires_grad:
                print('find m',m.shape)
        hessian_comp = hessian(model, criterion, data=eval_data, cuda=True)
        print('start trace estimation')
        s_time = time.time()
        res = hessian_comp.trace()
        print(f'{len(res)} samples:{time.time()-s_time:.2f} seconds')
        return res

    layers_to_check = []
    for n,m in model.named_modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):
            layers_to_check.append(n)
    if not os.path.exists(f'SMI_score/other_algorithms/hawq/HAWQ_DELTAL_{mn}'):
        os.mkdir(f'SMI_score/other_algorithms/hawq/HAWQ_DELTAL_{mn}')

    for i in range(len(calib_data)):
        print(f'batch {i+START_BATCH}')
        res = {}
        for n in layers_to_check:
            res[n] = estimate_trace(eval_data=calib_data[i],layer_name=n)
        with open(f'SMI_score/other_algorithms/hawq/HAWQ_DELTAL_{mn}/TraceEST_batch'+str(START_BATCH+i)+'.pkl','wb') as f:
            pickle.dump(res,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate HAWQv2 Hessian traces for a model and dataset.')
    parser.add_argument('--modelname', type=str, required=True, help='Name of the model to be used.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to be used for calibration.')
    parser.add_argument('--batch_size_train', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--start_batch', type=int, default=0, help='Starting batch index for calibration.')
    parser.add_argument('--end_batch', type=int, default=16, help='Ending batch index for calibration.')


    args = parser.parse_args()
    HAWQv2Calculate(args.modelname, args.dataset, args.batch_size_train, 
                    args.start_batch, args.end_batch)