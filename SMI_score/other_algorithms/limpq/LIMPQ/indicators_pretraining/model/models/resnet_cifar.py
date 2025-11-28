'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py
and adapted based on user-provided MobileNetV2 example.

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import sys
import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional, Tuple


cifar10_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
}

cifar100_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x): # Modified forward for simplicity, original had save_outputs
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block: Type[BasicBlock], layers: List[int], num_classes: int = 10, input_size: int = 32):
        super(CifarResNet, self).__init__()
        self.input_size = input_size # Store input size
        self.num_classes = num_classes # Store num_classes
        self.block = block # Store block type
        self.layers = layers # Store layer config

        self._norm_layer = nn.BatchNorm2d # Standard norm layer for ResNet

        self.inplanes = 16 # Initial number of input planes for layers
        self.dilation = 1 # Standard dilation

        # Initial convolution layer
        self.conv1 = conv3x3(3, self.inplanes)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # Maxpool layer removed for CIFAR as per standard implementations

        # Residual layers defined by 'layers' argument
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # Note: This was part of torchvision's ResNet, keeping it here. Might not apply if BN affine=False
        # for m in self.modules():
        #      if isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #          nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # Pass norm_layer if block needs it
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes)) # Pass norm_layer if block needs it

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x) # No maxpool in CifarResNet typically

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # --- Added Methods ---

    def feature_extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extracts intermediate feature maps from the network during a forward pass.
        Stores output after each major component (conv1, bn1, relu, each block, pool, fc).
        """
        tensor_list = []

        # Initial Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        tensor_list.append(x) # Store conv1 output

        # Layer 1 Blocks
        for block in self.layer1:
            if type(block) == BasicBlock:
                conv_out = block.relu(block.bn1(block.conv1(x)))
                tensor_list.append(conv_out)
                x = block(x)
                tensor_list.append(x)
                
        for block in self.layer2:
            if type(block) == BasicBlock:
          
                conv_out = block.relu(block.bn1(block.conv1(x)))
                tensor_list.append(conv_out)
                if block.downsample is not None:
                    conv2_out = block(x) #in the source code of OMPQ they use block output for conv2, code seems different because they did not use downsample layers
                    tensor_list.append(conv2_out)
                    downsample_out = block.downsample(x)
                    tensor_list.append(downsample_out)
                    x = block(x)
                else:
                    x = block(x)
                    tensor_list.append(x)
                    
        for block in self.layer3:
            if type(block) == BasicBlock:
                conv_out = block.relu(block.bn1(block.conv1(x)))
                tensor_list.append(conv_out)
                if block.downsample is not None:
                    conv2_out = block(x) #in the source code of OMPQ they use block output for conv2, code seems different because they did not use downsample layers
                    tensor_list.append(conv2_out)
                    downsample_out = block.downsample(x)
                    tensor_list.append(downsample_out)
                    x = block(x)
                else:
                    x = block(x)
                    tensor_list.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Keep flattened tensor
        x = self.fc(x); tensor_list.append(x)

        return tensor_list

    def get_layer_names(self) -> List[str]:
        """
        Return a list of layer names for Conv2d and Linear layers in the model.
        Mirrors the second get_layer_names implementation in the provided MobileNetV2.
        """
        layer_names = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_names.append(name)
        return layer_names

    def calculate_params_per_layer(self, input_channels: int = 3) -> Tuple[List[float], List[str]]:
        """
        Calculates the number of parameters for each logical sub-layer/layer
        (Conv+BN or Linear) based on the ACTUAL configuration of this model instance.
        Returns a list of parameter counts and corresponding layer names.
        """
        params_list = []
        layer_names = []

        current_channels = input_channels
        block_expansion = self.block.expansion

        # 1. Initial Conv + BN
        out_ch = self.conv1.out_channels
        # Conv params: out * in * k * k
        params_conv1 = out_ch * current_channels * self.conv1.kernel_size[0] * self.conv1.kernel_size[1]
        # BN params: 2 * out_ch (weight + bias)
        params_bn1 = 2 * out_ch
        params_list.append(params_conv1 + params_bn1)
        layer_names.append('conv1_bn1')
        current_channels = out_ch

        # --- Function to process layers ---
        def _calculate_layer_params(layer_module, layer_name_prefix, current_channels_in):
            layer_params = []
            layer_name_list = []
            ch_in = current_channels_in
            for idx, block in enumerate(layer_module):
                block_name = f"{layer_name_prefix}.{idx}"

                # Conv1 + BN1
                conv1 = block.conv1
                bn1 = block.bn1
                p_conv1 = conv1.out_channels * ch_in * conv1.kernel_size[0] * conv1.kernel_size[1]
                p_bn1 = 2 * conv1.out_channels
                layer_params.append(p_conv1 + p_bn1)
                layer_name_list.append(f"{block_name}.conv1_bn1")

                # Conv2 + BN2
                conv2 = block.conv2
                bn2 = block.bn2
                p_conv2 = conv2.out_channels * conv1.out_channels * conv2.kernel_size[0] * conv2.kernel_size[1]
                p_bn2 = 2 * conv2.out_channels
                layer_params.append(p_conv2 + p_bn2)
                layer_name_list.append(f"{block_name}.conv2_bn2")

                # Downsample (Conv1x1 + BN) if exists
                if block.downsample is not None:
                    ds_conv = block.downsample[0]
                    ds_bn = block.downsample[1]
                    p_ds_conv = ds_conv.out_channels * ch_in * ds_conv.kernel_size[0] * ds_conv.kernel_size[1]
                    p_ds_bn = 2 * ds_conv.out_channels
                    layer_params.append(p_ds_conv + p_ds_bn)
                    layer_name_list.append(f"{block_name}.downsample")
                    ch_in = ds_conv.out_channels # Output channels of downsample defines next input
                else:
                     # Output channels of conv2 defines next input if no downsample
                    ch_in = conv2.out_channels * block_expansion # This should match ds_conv.out_channels if present

            return layer_params, layer_name_list, ch_in # Return final output channels

        # 2. Layer 1
        l1_params, l1_names, current_channels = _calculate_layer_params(self.layer1, "layer1", current_channels)
        params_list.extend(l1_params)
        layer_names.extend(l1_names)

        # 3. Layer 2
        l2_params, l2_names, current_channels = _calculate_layer_params(self.layer2, "layer2", current_channels)
        params_list.extend(l2_params)
        layer_names.extend(l2_names)

        # 4. Layer 3
        l3_params, l3_names, current_channels = _calculate_layer_params(self.layer3, "layer3", current_channels)
        params_list.extend(l3_params)
        layer_names.extend(l3_names)

        # 5. Fully Connected Layer
        fc_layer = self.fc
        params_fc = (fc_layer.out_features * fc_layer.in_features) + fc_layer.out_features # W + b
        params_list.append(params_fc)
        layer_names.append('fc')

        return params_list, layer_names # Return count list and name list

    def calculate_flops_per_layer(self, input_channels: int = 3, input_resolution: int = 32) -> Tuple[List[float], int]:
        """
        Calculates the approximate FLOPs (MACs) for each logical sub-layer/layer
        (Conv or Linear) based on the ACTUAL configuration of this model instance.
        Ignores FLOPs from BN, Activations, Additions, Pooling, etc.
        Returns a list of FLOP counts for each Conv/Linear layer block.
        """
        flops_list = []

        current_channels = input_channels
        current_resolution = input_resolution
        block_expansion = self.block.expansion

        # FLOPs formulas (MACs)
        # Conv: C_out * C_in * K*K * H_out*W_out / groups
        # Linear: C_out * C_in

        # 1. Initial Conv
        out_ch = self.conv1.out_channels
        k = self.conv1.kernel_size[0]
        s = self.conv1.stride[0]
        # Output resolution after initial conv (stride=1 for CifarResNet)
        out_resolution = current_resolution // s
        flops_conv1 = out_ch * current_channels * k * k * out_resolution * out_resolution
        flops_list.append(flops_conv1)
        current_channels = out_ch
        current_resolution = out_resolution # Update resolution

        # --- Function to process layers ---
        def _calculate_layer_flops(layer_module, current_channels_in, current_resolution_in):
            layer_flops = []
            ch_in = current_channels_in
            res_in = current_resolution_in
            for idx, block in enumerate(layer_module):
                res_out_block = res_in // block.stride # Resolution after the strided conv (if any)

                # Conv1
                conv1 = block.conv1
                k1 = conv1.kernel_size[0]
                # Output res of conv1 depends on its stride
                res_out_conv1 = res_in // conv1.stride
                flops_conv1 = conv1.out_channels * ch_in * k1 * k1 * res_out_conv1 * res_out_conv1
                layer_flops.append(flops_conv1)

                # Conv2 (stride is always 1)
                conv2 = block.conv2
                k2 = conv2.kernel_size[0]
                # Input res to conv2 is output res of conv1
                res_in_conv2 = res_out_conv1
                # Output res of conv2 (stride=1)
                res_out_conv2 = res_in_conv2 # Stays the same
                flops_conv2 = conv2.out_channels * conv1.out_channels * k2 * k2 * res_out_conv2 * res_out_conv2
                layer_flops.append(flops_conv2)

                # Downsample (Conv1x1) if exists
                if block.downsample is not None:
                    ds_conv = block.downsample[0]
                    k_ds = ds_conv.kernel_size[0] # Should be 1
                    s_ds = ds_conv.stride[0]
                    # Output res of downsample (depends on its stride, matches block stride)
                    res_out_ds = res_in // s_ds
                    flops_ds = ds_conv.out_channels * ch_in * k_ds * k_ds * res_out_ds * res_out_ds
                    layer_flops.append(flops_ds)
                    ch_in = ds_conv.out_channels # Update channels based on downsample
                else:
                    # Update channels based on conv2 output if no downsample
                    ch_in = conv2.out_channels * block_expansion

                # Update resolution for the next block
                res_in = res_out_block # The output resolution of the block becomes input for the next

            return layer_flops, ch_in, res_in # Return flops, final output channels, final resolution

        # 2. Layer 1
        l1_flops, current_channels, current_resolution = _calculate_layer_flops(self.layer1, current_channels, current_resolution)
        flops_list.extend(l1_flops)

        # 3. Layer 2
        l2_flops, current_channels, current_resolution = _calculate_layer_flops(self.layer2, current_channels, current_resolution)
        flops_list.extend(l2_flops)

        # 4. Layer 3
        l3_flops, current_channels, current_resolution = _calculate_layer_flops(self.layer3, current_channels, current_resolution)
        flops_list.extend(l3_flops)

        # 5. Fully Connected Layer
        fc_layer = self.fc
        flops_fc = fc_layer.out_features * fc_layer.in_features
        flops_list.append(flops_fc)

        return flops_list, 0 # Return flops list and 0 (matching MobileNetV2 signature)

# Rest of the script (model loading functions) remains the same
# ... (keep the _resnet function and the loop setting attributes)

def _resnet(
    arch: str,
    layers: List[int],
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> CifarResNet:
    # Pass input_size if provided, otherwise CifarResNet defaults
    model = CifarResNet(BasicBlock, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def cifar10_resnet20(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet32(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet44(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet56(*args, **kwargs) -> CifarResNet: pass


def cifar100_resnet20(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet32(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet44(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet56(*args, **kwargs) -> CifarResNet: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for layers, model_name in zip([[3]*3, [5]*3, [7]*3, [9]*3],
                                  ["resnet20", "resnet32", "resnet44", "resnet56"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        # Ensure num_classes is passed correctly to the partial
        setattr(
            thismodule,
            method_name,
            partial(_resnet,
                    arch=model_name,
                    layers=layers,
                    model_urls=model_urls,
                    num_classes=num_classes) # Pass num_classes here
        )

import os
def get_resnet20():
    resnet20 = thismodule.cifar10_resnet20()
    weights_path = os.path.join(os.path.dirname(__file__), 'cifar10_resnet20-4118986f.pt')
    resnet20.load_state_dict(torch.load(weights_path))
    return resnet20

def get_resnet32():
    resnet32 = thismodule.cifar10_resnet32()
    weights_path = os.path.join(os.path.dirname(__file__), 'cifar10_resnet32-ef93fc4d.pt')
    resnet32.load_state_dict(torch.load(weights_path))
    return resnet32

def get_resnet44():
    resnet44 = thismodule.cifar10_resnet44()
    weights_path = os.path.join(os.path.dirname(__file__), 'cifar10_resnet44-2a3cabcb.pt')
    resnet44.load_state_dict(torch.load(weights_path))
    return resnet44

def get_resnet56():
    resnet56 = thismodule.cifar10_resnet56()
    weights_path = os.path.join(os.path.dirname(__file__), 'cifar10_resnet56-187c023a.pt')
    resnet56.load_state_dict(torch.load(weights_path))
    return resnet56