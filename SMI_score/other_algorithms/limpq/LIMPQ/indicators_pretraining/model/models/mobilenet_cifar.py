'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/mobilenetv2.py

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
from torch import nn
from torch import Tensor
import os
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional


cifar10_pretrained_weight_urls = {
    'mobilenetv2_x0_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x0_5-ca14ced9.pt',
    'mobilenetv2_x0_75': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x0_75-a53c314e.pt',
    'mobilenetv2_x1_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_0-fe6a5b48.pt',
    'mobilenetv2_x1_4': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt',
}

cifar100_pretrained_weight_urls = {
    'mobilenetv2_x0_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x0_5-9f915757.pt',
    'mobilenetv2_x0_75': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x0_75-d7891e60.pt',
    'mobilenetv2_x1_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_0-1311f9ff.pt',
    'mobilenetv2_x1_4': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_4-8a269f5e.pt',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()
        self.input_size = 32
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.round_nearest = round_nearest

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        cfg = [32, 16, 24, 32, 64, 96, 160, 320]
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10/100
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        self.inverted_residual_setting = inverted_residual_setting
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.input_channel = input_channel
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=1, norm_layer=norm_layer)]  # NOTE: change stride 2 -> 1 for CIFAR10/100
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def get_layer_names(self):
        """Return a list of layer names for the model block outputs"""
        layers = []
        
        # Initial layer
        layers.append('features.0')  # First ConvBNReLU
        
        # InvertedResidual blocks
        for i in range(1, len(self.features) - 1):  # Skip the last ConvBNReLU
            if isinstance(self.features[i], InvertedResidual):
                layers.append(f'features.{i}')
        
        # Final ConvBNReLU and classification layers
        layers.append(f'features.{len(self.features) - 1}')  # Last ConvBNReLU
        layers.append('avg_pool')
        layers.append('flatten')
        layers.append('classifier')
        layers.append('final_output')
        
        return layers

    def _get_layer_outputs(self, x):
        """Extract outputs from block layers during forward pass"""
        layer_outputs = {}
        
        # Process through feature layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Only store outputs from the blocks we're interested in
            if i == 0 or isinstance(layer, InvertedResidual) or i == len(self.features) - 1:
                layer_outputs[f'features.{i}'] = x
        
        # Final operations
        pool_out = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        flatten_out = torch.flatten(pool_out, 1)
        classifier_out = self.classifier(flatten_out)
        
        layer_outputs['pool'] = pool_out
        layer_outputs['flatten'] = flatten_out
        layer_outputs['classifier'] = classifier_out
        
        return layer_outputs, classifier_out

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
        
    def forward_layers(self, x: Tensor, all_layers=False) -> Tensor:
        if all_layers:
            # Collect all block outputs
            all_outputs, output = self._get_layer_outputs(x)
            all_outputs['final_output'] = output
            return all_outputs
        else:
            return self._forward_impl(x)
        
    def feature_extract(self, x):
        tensor = []
        for _layer in self.features:
            if type(_layer) is not InvertedResidual:
                x = _layer(x)
                tensor.append(x)
            else:
                if len(_layer.conv) == 3: #bn layer also in "conv sequential" so if it has 2 conv layer in total, length of sequential is 3. len is 4 if len(conv) = 3
                    tensor.append(_layer.conv[0](x))
                    tensor.append(_layer.conv(x))
                    x = _layer(x)
                else:
                    tensor.append(_layer.conv[0](x))
                    tensor.append(_layer.conv[1](_layer.conv[0](x)))
                    tensor.append(_layer(x))
                    x = _layer(x)
        
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        tensor.append(x)
        return tensor
    def calculate_params_per_layer(self, input_channels: int = 3) -> List[float]:
        """
        Calculates the number of parameters for each logical sub-layer/layer
        based on the ACTUAL configuration of this model instance. Mirrors __init__.
        Returns a list of parameter counts for each Conv/Linear layer block.
        """
        params_list = []
        layer_names = []

        # Use model's actual attributes
        width_mult = self.width_mult
        round_nearest = self.round_nearest
        # Base structure comes from self.inverted_residual_setting stored during init
        inverted_residual_setting = self.inverted_residual_setting

        current_channels = input_channels
        initial_stride = 1 if self.input_size < 128 else 2 # Match __init__

        # 1. First Layer (ConvBNReLU)
        actual_out_ch = _make_divisible(32 * width_mult, round_nearest) # Match __init__ calc
        params_layer = (actual_out_ch * current_channels * 3 * 3) + (2 * actual_out_ch) # K=3
        params_list.append(params_layer)
        layer_names.append('features.0')  # First ConvBNReLU
        current_channels = actual_out_ch

        # 2. Inverted Residual Blocks
        feature_index = 1
        for stage_idx, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            # Apply stride override logic
            effective_s = s
            if stage_idx == 1 and initial_stride == 1:
                effective_s = 1

            for i in range(n):
                stride = effective_s if i == 0 else 1
                hidden_dim = int(round(current_channels * t))
                conv_index = 0
                # Sub-layers within the InvertedResidual block
                if t != 1:
                    # Expansion Layer (Conv1x1 + BN)
                    params_exp = (hidden_dim * current_channels * 1 * 1) + (2 * hidden_dim)
                    params_list.append(params_exp)
                    #get layer name 
                    layer_names.append(f'features.{feature_index}.conv.{conv_index}')  # Expansion Conv1x1
                    conv_index += 1

                # Depthwise Layer (Conv3x3_DW + BN)
                params_dw = (hidden_dim * 1 * 3 * 3) + (2 * hidden_dim)
                params_list.append(params_dw)
                layer_names.append(f'features.{i}.conv.{conv_index}')  # Depthwise Conv3x3
                conv_index += 1

                # Projection Layer (Conv1x1 + BN)
                params_proj = (output_channel * hidden_dim * 1 * 1) + (2 * output_channel)
                params_list.append(params_proj)
                layer_names.append(f'features.{i}.conv.{conv_index}')  # Projection Conv1x1
                conv_index += 1

                current_channels = output_channel

        # 3. Final 1x1 Conv Layer (ConvBNReLU)
        final_conv_in_channels = current_channels
        final_conv_out_channels = self.last_channel # Use actual last_channel from __init__
        params_layer = (final_conv_out_channels * final_conv_in_channels * 1 * 1) + (2 * final_conv_out_channels)
        params_list.append(params_layer)
        layer_names.append(f'features.{len(inverted_residual_setting) + 1}')  # Last ConvBNReLU
        current_channels = final_conv_out_channels

        # 4. Classifier Layer (Linear)
        params_classifier = (self.num_classes * current_channels) + self.num_classes # W + b
        params_list.append(params_classifier)
        layer_names.append('classifier.0')  # Classifier Linear

        return params_list, 0  
    def calculate_flops_per_layer(self, input_channels: int = 3) -> List[float]:
        """
        Calculates the approximate FLOPs (MACs) for each logical sub-layer/layer
        based on the ACTUAL configuration of this model instance. Mirrors __init__.
        Ignores FLOPs from BN, Activations, Additions, Pooling, etc.
        Returns a list of FLOP counts for each Conv/Linear layer block.
        """
        flops_list = []

        # Use model's actual attributes
        width_mult = self.width_mult
        round_nearest = self.round_nearest
        inverted_residual_setting = self.inverted_residual_setting
        input_resolution = self.input_size # Use actual input size

        current_channels = input_channels
        current_resolution = input_resolution
        initial_stride = 1 if self.input_size < 128 else 2 # Match __init__

        # FLOPs formulas (MACs)
        # Conv: C_out * C_in * K*K * H_out*W_out / groups
        # Linear: C_out * C_in

        # 1. First Layer (ConvBNReLU)
        actual_out_ch = _make_divisible(32 * width_mult, round_nearest)
        kernel_size = 3
        stride = initial_stride
        out_resolution = current_resolution // stride
        flops_layer = actual_out_ch * current_channels * kernel_size * kernel_size * out_resolution * out_resolution
        flops_list.append(flops_layer)
        current_channels = actual_out_ch
        current_resolution = out_resolution

        # 2. Inverted Residual Blocks
        for stage_idx, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            # Apply stride override logic
            effective_s = s
            if stage_idx == 1 and initial_stride == 1:
                effective_s = 1

            for i in range(n):
                stride = effective_s if i == 0 else 1
                hidden_dim = int(round(current_channels * t))
                out_resolution = current_resolution // stride # Resolution after strided layer

                # Sub-layers within the InvertedResidual block
                if t != 1:
                    # Expansion Layer (Conv1x1, Stride=1)
                    kernel_size = 1
                    flops_exp = hidden_dim * current_channels * kernel_size * kernel_size * current_resolution * current_resolution
                    flops_list.append(flops_exp)
                    # Resolution doesn't change here

                # Depthwise Layer (Conv3x3_DW, Stride=stride)
                kernel_size = 3
                # Input res = current_resolution, Output res = out_resolution
                flops_dw = hidden_dim * 1 * kernel_size * kernel_size * out_resolution * out_resolution
                flops_list.append(flops_dw)

                # Projection Layer (Conv1x1, Stride=1)
                kernel_size = 1
                # Input res = out_resolution
                flops_proj = output_channel * hidden_dim * kernel_size * kernel_size * out_resolution * out_resolution
                flops_list.append(flops_proj)

                # Update state for next layer/block
                current_channels = output_channel
                current_resolution = out_resolution # Resolution carries over

        # 3. Final 1x1 Conv Layer (ConvBNReLU, Stride=1)
        final_conv_in_channels = current_channels
        final_conv_out_channels = self.last_channel
        kernel_size = 1
        # Input res = current_resolution
        flops_layer = final_conv_out_channels * final_conv_in_channels * kernel_size * kernel_size * current_resolution * current_resolution
        flops_list.append(flops_layer)
        current_channels = final_conv_out_channels
        # Resolution doesn't change here

        # 4. Classifier Layer (Linear)
        flops_classifier = self.num_classes * current_channels
        flops_list.append(flops_classifier)

        return flops_list, 0
    def get_layer_names(self):
        #print layer names
        layer_names = []
        for name in self.named_modules():
            if isinstance(name[1], nn.Conv2d):
                    layer_names.append(name[0])

                #print also if classifier
            if isinstance(name[1], nn.Linear):
                    layer_names.append(name[0])

        return layer_names
       


def _mobilenet_v2(
    arch: str,
    width_mult: List[int],
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> MobileNetV2:
    model = MobileNetV2(width_mult=width_mult, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def cifar10_mobilenetv2_x0_5(*args, **kwargs) -> MobileNetV2: pass
def cifar10_mobilenetv2_x0_75(*args, **kwargs) -> MobileNetV2: pass
def cifar10_mobilenetv2_x1_0(*args, **kwargs) -> MobileNetV2: pass
def cifar10_mobilenetv2_x1_4(*args, **kwargs) -> MobileNetV2: pass


def cifar100_mobilenetv2_x0_5(*args, **kwargs) -> MobileNetV2: pass
def cifar100_mobilenetv2_x0_75(*args, **kwargs) -> MobileNetV2: pass
def cifar100_mobilenetv2_x1_0(*args, **kwargs) -> MobileNetV2: pass
def cifar100_mobilenetv2_x1_4(*args, **kwargs) -> MobileNetV2: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for width_mult, model_name in zip([0.5, 0.75, 1.0, 1.4],
                                      ["mobilenetv2_x0_5", "mobilenetv2_x0_75", "mobilenetv2_x1_0", "mobilenetv2_x1_4"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_mobilenet_v2,
                    arch=model_name,
                    width_mult=width_mult,
            
                    model_urls=model_urls,
                    num_classes=num_classes)
        )

def get_mobilenetv2_x0_5():
    mobilenetv2 = thismodule.cifar10_mobilenetv2_x0_5()
    weight_path = os.path.join(os.path.dirname(__file__), 'cifar10_mobilenetv2_x0_5-ca14ced9.pt')
    mobilenetv2.load_state_dict(torch.load(weight_path))

    return mobilenetv2
def get_mobilenetv2_x1_0():
    mobilenetv2 = thismodule.cifar10_mobilenetv2_x1_0()
    weight_path = os.path.join(os.path.dirname(__file__), 'cifar10_mobilenetv2_x1_0-fe6a5b48.pt')

    mobilenetv2.load_state_dict(torch.load(weight_path))

    return mobilenetv2
