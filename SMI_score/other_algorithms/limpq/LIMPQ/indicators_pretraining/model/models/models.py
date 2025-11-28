import logging

import timm
import torchvision.models as mds
import model.models.mobilenetv2_imagenet as mobilenetv2_imagenet
import model.models.mobilenet_cifar as mobilenet_cifar
import model.models.resnet_cifar as resnet_cifar
import importlib
importlib.reload(mobilenet_cifar)
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def create_model(model_name, dataset='imagenet'):
    
    if dataset == 'imagenet_256x256':
        if model_name == 'resnet18_gluon':
            
            model = timm.create_model('gluon_resnet18_v1b', pretrained=True)
        elif model_name == 'resnet18_a2_in1k':
            model = timm.create_model('resnet18.a2_in1k', pretrained=True)
        elif model_name == 'resnet50_gluon':
            model = timm.create_model('gluon_resnet50_v1b', pretrained=True)
        elif model_name == 'mobilenetv2':
            model = mobilenetv2_imagenet.mobilenetv2()
    elif dataset == 'cifar10':
        if model_name == 'mobilenetv2_x0_5':
            model = mobilenet_cifar.get_mobilenetv2_x0_5()
        elif model_name == 'mobilenetv2_x1_0':
            model = mobilenet_cifar.get_mobilenetv2_x1_0()
        elif model_name == 'resnet20':
            model = resnet_cifar.get_resnet20()
        elif model_name == 'resnet32':
            model = resnet_cifar.get_resnet32()
        
    model.eval()  # Set the model to evaluation mode
    logging.info(f"Created model: {model_name} for dataset: {dataset}")
    return model.to(device)
        