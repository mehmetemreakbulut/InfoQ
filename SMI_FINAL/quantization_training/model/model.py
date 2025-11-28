import logging

from .mobilenet import *
import timm
from .mobilenetv2_imagenet import * 

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
def create_model(args):
    logger = logging.getLogger()

    model = None
    if args.dataloader.dataset == 'imagenet':
        if args.arch == 'resnet18':
            model = timm.create_model('gluon_resnet18_v1b', pretrained=args.pre_trained)
        elif args.arch == 'resnet50':
            model = timm.create_model('gluon_resnet50_v1b', pretrained=args.pre_trained)
        elif args.arch == 'mobilenetv1':
            model = mobilenet_v1(pretrained=args.pre_trained)
        elif args.arch == 'mobilenetv2':
            import torch
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=MobileNet_V2_Weights.DEFAULT)

            #model = timm.create_model('mobilenetv2_100', pretrained=args.pre_trained)
            #model = mobilenetv2()
    #print model
    print(model)

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (args.arch, args.dataloader.dataset))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (args.arch, args.dataloader.dataset)
    msg += '\n          Use pre-trained model = %s' % args.pre_trained
    logger.info(msg)

    return model