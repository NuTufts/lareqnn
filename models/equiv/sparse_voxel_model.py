import pytest
import torch
from e3nn.o3 import Irreps
from e3nn import o3
from e3nn.nn import BatchNorm
import math
from sparse_voxel_convolution import Convolution, EquivariantConvolutionBlock, \
    EquivariantDownSample
from utils import rotate_sparse_tensor
import MinkowskiEngine as ME
import torch.nn as nn
import numpy as np

class EquivModel(torch.nn.Module):
    def __init__(self, irreps_in=Irreps("0e"), irreps_out=Irreps("5x0e"), lmax=2, segment=False) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax, -1)
        self.activation = [torch.relu, torch.tanh]
        self.irreps_coef = [16, 16, 6, 6, 2, 2, 0, 0]  # list for coefficients of irreps in
        # the form "list[0]x0e + list[1]x0o + list[2]x1e + ...
        self.irreps_mid = o3.Irreps(f"{self.irreps_coef[0]}x0e + \
                                      {self.irreps_coef[1]}x0o + \
                                      {self.irreps_coef[2]}x1e + \
                                      {self.irreps_coef[3]}x1o + \
                                      {self.irreps_coef[4]}x2e + \
                                      {self.irreps_coef[5]}x2o + \
                                      {self.irreps_coef[6]}x3e + \
                                      {self.irreps_coef[7]}x3o").simplify()
        #irreps after downsample
        self.irreps_down = o3.Irreps(f"{self.irreps_coef[0]}x0e + \
                                       {self.irreps_coef[1]}x0o + \
                                       {int(self.irreps_coef[2]>0)}x1e + \
                                       {int(self.irreps_coef[3]>0)}x1o + \
                                       {int(self.irreps_coef[4]>0)}x2e + \
                                       {int(self.irreps_coef[5]>0)}x2o + \
                                       {int(self.irreps_coef[6]>0)}x3e + \
                                       {int(self.irreps_coef[7]>0)}x3o").simplify()
        self.segment = segment
        self.network_initialization()
        self.weight_initialization()

    def network_initialization(self):
        blocks = []
        diameters = [7, 3, 3, 3, 3, 3, 3, 3]
        downsample_locations = [1,5]
        
        steps = [(1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0)]
        

        for i, (diameter, step) in enumerate(zip(diameters, steps)):
            print(i, self.irreps_in)
            blocks.append(EquivariantConvolutionBlock(self.irreps_in,
                                                      self.irreps_mid,
                                                      self.irreps_sh,
                                                      diameter,
                                                      step,
                                                      self.activation))
            if i in downsample_locations:
                blocks.append(EquivariantDownSample(self.irreps_mid, kernel_size=3, stride=3))
                self.irreps_in = self.irreps_down
            else:
                self.irreps_in = self.irreps_mid  # update input irreps for next block to irreps_mid

        blocks.append(EquivariantConvolutionBlock(self.irreps_mid,
                                                  self.irreps_out,
                                                  self.irreps_sh,
                                                  diameter=3,
                                                  steps=(1.0, 1.0, 1.0),
                                                  activation=[self.activation[0]]  # only need ReLU here
                                                  ))

        self.blocks = nn.ModuleList(blocks)

        self.global_pool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, data) -> torch.Tensor:
        for i in range(len(self.blocks)):
            data = self.blocks[i](data)
            
        if not self.segment:
            data = self.global_pool(data)
        return data

    def weight_initialization(self):
        for m in self.modules():
            # if isinstance(m, ME.MinkowskiConvolution):
            #     ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if isinstance(m, o3.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if isinstance(m, Convolution):  # TODO: try some other initialization here and linear
                nn.init.xavier_uniform_(m.weight, 1)
