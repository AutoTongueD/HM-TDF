
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from typing import Any, Callable, List, Optional, Type, Union

import sys
import os
script_path = os.path.abspath(__file__)
model_path = os.path.dirname(script_path)
sys.path.insert(0, model_path)
from KANConv import KAN_Convolutional_Layer as KAN_Conv
from KANLinear import KANLinear as KAN_Linear

def conv7x7(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=7,
        stride=stride,
        padding=dilation + 2,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=dilation + 1,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        kernel_size: int,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride) if kernel_size == 3 else (
                     conv5x5(inplanes, planes, stride) if kernel_size == 5 else 
                     conv7x7(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes) if kernel_size == 3 else (
                     conv5x5(planes, planes) if kernel_size == 5 else 
                     conv7x7(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x: Tensor) -> Tensor:
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

class Bottleneck(nn.Module):
    expansion: int = 2 # !!!!!

    def __init__(
        self,
        kernel_size: int,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        self.conv2 = conv3x3(width, width, stride, groups, dilation) if kernel_size == 3 else (
                conv5x5(width, width, stride, groups, dilation) if kernel_size == 5 else 
                conv7x7(width, width, stride, groups, dilation))
      
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResConvKAN(nn.Module): 
    def __init__(self, 
                 kernel_size: int, 
                 layers: list, 
                 zero_init_residual: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = 1
        self.base_width = 64
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, kernel_size, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, kernel_size, 128, layers[1], stride=2)

        # 24.09.12 Initiate weights for CNNs
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        stride_tuple = (2, 2)
        self.layerKAN = KAN_Conv(2, (3,3), stride=stride_tuple, padding=(1,1), device='cuda') if kernel_size == 3 else (  # !!!!!
                     KAN_Conv(1, (5,5), stride=stride_tuple, padding=(2,2), device='cuda') if kernel_size == 5 else    # !!!!!
                     KAN_Conv(1, (7,7), stride=stride_tuple, padding=(3,3), device='cuda'))      

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        kernel_size: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
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
        layers.append(
            block(
                kernel_size, self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    kernel_size,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, mask) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layerKAN(x) * mask
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class CNN_ConvKAN(nn.Module): 
    def __init__(self):
        super().__init__()
        self.base = ResConvKAN(3, [1, 1])
        self.mask_pool = nn.MaxPool2d(16, 16)

    def forward(self, X): # , mask0
        mask0 = ((X[:, 1, :, :] > -1.7)).float().unsqueeze(1)
        mask1 = self.mask_pool(mask0)
        mask_rate = mask1.shape[3] * mask1.shape[2] / mask1.sum(3).sum(2)
        X1 = self.base(X, mask1)
        return X1
   

class FeatureNet(nn.Module): 
    def __init__(self, num_labels, num_features, drop_rate):
        super().__init__()
        self.num_features = num_features
        self.kan_linears = []

        self.IE = CNN_ConvKAN()
        self.DE = nn.Sequential(KAN_Linear(num_features,        32, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(32,   128, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]))

        self.FFC = nn.Sequential(nn.BatchNorm1d(640),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(640, 32, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(32,        num_labels, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]))
        
        self.M_orient = nn.Parameter(torch.tensor([self.ordinal_multi_expert_labels_generation(num_labels, num_labels - 2)]), requires_grad=False)
        self.M_orient_abs = nn.Parameter(self.M_orient.data.abs(), requires_grad=False)
        self.expert_num = self.M_orient.shape[-1]
        self.MEC = nn.Sequential(nn.BatchNorm1d(640),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(640, 32, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(32,        self.expert_num, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]),
                                  nn.Softsign())

        for module in self.DE.modules():
            if isinstance(module, KAN_Linear):
                self.kan_linears.append(module)
        for module in self.FFC.modules():
            if isinstance(module, KAN_Linear):
                self.kan_linears.append(module)
        self.kan_linears.pop(-1)
        for module in self.MEC.modules():
            if isinstance(module, KAN_Linear):
                self.kan_linears.append(module)
        self.kan_linears.pop(-1)

    def ordinal_multi_expert_labels_generation(self, n_L, n_a):
        # Step 1: Initialize matrix M0 with -1s
        M0 = -1 * np.ones(((n_L - 1), n_L), dtype=int)
        # Step 2: Set the elements of M0 based on the rules in the pseudocode
        for i in range(1, n_L):
            M0[i - 1, :i] = 1
        # Step 3-4: Repeat n_a times
        for k in range(1, n_a + 1):
            M = M0.copy()  # Copy M0 to M
            # Step 6: Process each row in M
            for row in M:
                for j in range(n_L):
                    m_r = row.copy()
                    m_r[j] = 0  # Set m_r[j] to 0
                    # Step 10: Check the conditions
                    if 1 in m_r and -1 in m_r and m_r.tolist() not in M0.tolist():
                        M0 = np.vstack([M0] + [m_r])
        # Step 12: Return the result
        return M0.transpose().tolist()

    def forward(self, X, f_p):
        f_i = self.IE(X)
        f_p = self.DE(f_p)
        f_f = torch.cat((f_i, f_p), dim=1)
        ffc_out = self.FFC(f_f)

        encode = self.MEC(f_f) # (batch_size, classifier)
        distance = (encode.unsqueeze(1) * self.M_orient_abs - self.M_orient).pow(2).mean(2) # (batch_size, num_labels)
        mec_out = - distance
        return ffc_out, mec_out, distance, encode
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_linears
        )

# define net structure
def get_net(num_features, num_labels, drop_rate):
    net = FeatureNet(num_labels, num_features, drop_rate)
    return net


if __name__ == "__main__":
    device = 'cuda'
    from torchviz import make_dot
    model = get_net(10, 3, 0.2).to(device)
    input = torch.rand((2, 3, 224, 224)).to(device)
    fff = torch.rand((2, 10)).to(device)
    output = model(input, fff)
    dot = make_dot(output.mean(), params=dict(model.named_parameters()))
    dot.render("autograd_graph.pdf") # create a PDF in work_dir
    aaa = 1
