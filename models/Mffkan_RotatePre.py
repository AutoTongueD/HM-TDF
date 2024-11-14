
import torch
import torch.nn as nn
import sys
import os
script_path = os.path.abspath(__file__)
model_path = os.path.dirname(script_path)
sys.path.insert(0, model_path)
from KANLinear import KANLinear as KAN_Linear
from CNNKAN import CNNKan as CNN_Kan

class MffKan(nn.Module): 
    def __init__(self, num_labels, num_features, drop_rate):
        super().__init__()
        self.num_features = num_features
        self.kan_linears = []

        self.IE = CNN_Kan()
        self.RR = nn.Sequential(nn.BatchNorm1d(512),
                                  KAN_Linear(512, 32, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]),
                                  nn.BatchNorm1d(32),
                                  KAN_Linear(32,        1, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]))
        for module in self.RR.modules():
            if isinstance(module, KAN_Linear):
                self.kan_linears.append(module)
        self.kan_linears.pop(-1)

    def forward(self, X, f_p):
        f_i = self.IE(X)
        out = self.RR(f_i)
        return out
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_linears
        )

# define net structure
def get_net(num_features, num_labels, drop_rate):
    net = MffKan(num_labels, num_features, drop_rate)
    return net
