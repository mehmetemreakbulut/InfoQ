from types import MethodType


import torch
import torch.nn as nn
# No specific BasicBlock class import is needed here if we rely on the attribute
# structure (duck typing), which is suitable for a method you're adding to an
# existing instance that conforms to the printed structure.

def feature_extract_from_resnet18(self, x):
    """
    Extracts feature tensors after each convolutional layer and the final
    fully connected layer from the ResNet model.

    Args:
        self: The ResNet model instance.
        x: The input tensor to the model.

    Returns:
        A list of output tensors from specified layers.
    """
    tensors = []

    # Initial convolution stem
    # x after self.conv1
    x = self.conv1(x)
    tensors.append(x)
    x = self.bn1(x)
    x = self.act1(x) # In your structure, this is ReLU
    x = self.maxpool(x)

    # Iterating through layer1, layer2, layer3, layer4 (Sequential modules)
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer_module = getattr(self, layer_name)
        for block in layer_module:
            # Each 'block' is a BasicBlock instance
            
            shortcut = x  # Input to the current block

            # --- Main path of the BasicBlock ---
            # First convolution in the block
            # out after block.conv1
            conv1_out = block.conv1(x)            
            # Continue main path through bn1, drop_block, act1, aa
            main_path = block.bn1(conv1_out)
            main_path = block.drop_block(main_path) # Typically Identity if not active
            main_path = block.act1(main_path)       # ReLU
            main_path = block.aa(main_path)         # Typically Identity if not active AntiAliasing

            tensors.append(main_path)
            
            # Second convolution in the block
            # out after block.conv2
            conv2_out = block.conv2(main_path)
           
            
            main_path = block.bn2(conv2_out)
            main_path = block.act2(main_path)
            
            # Note: The 'aa' module is before conv2 in your BasicBlock structure.
            # No second 'aa' after bn2.

            # --- Shortcut path of the BasicBlock ---
            if hasattr(block, 'downsample') and block.downsample is not None:
                # The downsample module is a Sequential: (0): Conv2d, (1): BatchNorm2d
                # out after block.downsample[0] (Conv2d)
                conv_downsample_out = block.downsample[0](shortcut)
            
                shortcut = block.downsample[1](conv_downsample_out) # Pass through BatchNorm2d
                
                            # Add shortcut to the main path
                #x = main_path + shortcut
                #x = block.act2(x) # Final activation of the block (ReLU)

            x = block(x)
            tensors.append(x)
            if hasattr(block, 'downsample') and block.downsample is not None:
                tensors.append(shortcut) # in quant_dict_base downsample is after the conv2


            # 'x' is now the output of the current block

    # Global pooling layer
    # Your global_pool includes flattening:
    # SelectAdaptivePool2d(pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
    x = self.global_pool(x)

    # Final fully connected layer
    # x after self.fc
    x = self.fc(x)
    tensors.append(x)

    return tensors


def feature_extract_from_resnet50(self, x):
    tensors = []

    # Initial convolution stem
    # x after self.conv1
    x_conv1_stem = self.conv1(x)
    
    x = self.bn1(x_conv1_stem)
    x = self.act1(x) # In your structure, this is ReLU
    tensors.append(x)
    x = self.maxpool(x)

    # Iterating through layer1, layer2, layer3, layer4 (Sequential modules)
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        print(layer_name)
        layer_module = getattr(self, layer_name)
        for block_idx, block in enumerate(layer_module):
            print(block_idx)
            # Each 'block' is a Bottleneck instance
            
            shortcut = x  # Input to the current block

            # --- Main path of the Bottleneck block ---

            # First convolution in the bottleneck (conv1)
            # out after block.conv1
            conv1_out = block.conv1(x)
            
            
            path = block.bn1(conv1_out)
            path = block.act1(path) # ReLU
            tensors.append(path)

            # Second convolution in the bottleneck (conv2)
            # out after block.conv2
            conv2_out = block.conv2(path)
           

            path = block.bn2(conv2_out)
            path = block.drop_block(path) # Typically Identity
            path = block.act2(path)       # ReLU
            path = block.aa(path)         # Typically Identity (AntiAliasing)

            tensors.append(path)
            
            # Third convolution in the bottleneck (conv3)
            # out after block.conv3
            conv3_out = block.conv3(path)

            main_path_final = block.bn3(conv3_out)
            # Note: act3 is applied AFTER the residual connection

            # --- Shortcut path of the Bottleneck block ---
            if hasattr(block, 'downsample') and block.downsample is not None:
                # The downsample module is a Sequential: (0): Conv2d, (1): BatchNorm2d
                # out after block.downsample[0] (Conv2d)
                conv_downsample_out = block.downsample[0](shortcut)
            
                shortcut = block.downsample[1](conv_downsample_out) # Pass through BatchNorm2d
            # else: shortcut remains the original input 'x' to the block

            # Add shortcut to the main path
            x = main_path_final + shortcut
            x = block.act3(x) 
            tensors.append(x) #for conv3 of the block
            # 'x' is now the output of the current block
            if hasattr(block, 'downsample') and block.downsample is not None:
                tensors.append(shortcut) # in quant_dict_base downsample is after the conv2

    # Global pooling layer
    # Your global_pool includes flattening:
    # SelectAdaptivePool2d(pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
    x = self.global_pool(x)

    # Final fully connected layer
    # x after self.fc
    # Note: The output of fc is typically a 1D vector per sample if flatten=True in global_pool.
    # If you need the features *before* flattening for some reason, you'd tap before self.global_pool.flatten
    # However, "after self.fc" implies the final output.
    fc_out = self.fc(x)
    tensors.append(fc_out)

    return tensors


def add_feature_extract(net, model_name):
    if model_name == 'resnet18_gluon':
        net.feature_extract = MethodType(feature_extract_from_resnet18, net)
    elif model_name == 'resnet50_gluon':
        net.feature_extract = MethodType(feature_extract_from_resnet50, net)
    elif model_name=='mobilenetv2':
        pass
        
