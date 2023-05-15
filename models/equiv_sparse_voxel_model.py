import pytest
import torch
from e3nn.o3 import Irreps
from e3nn import o3
from e3nn.nn import BatchNorm
import math
from equiv_sparse_voxel_convolution import Convolution, rotate_sparse_tensor, EquivariantConvolutionBlock
import MinkowskiEngine as ME
import torch.nn as nn
import numpy as np



def test_equivariance(input, model, rotations, irreps_in, irreps_out, device=torch.device("cpu")):
    """Test equivariance of a model
    Args:
        input (torch.Tensor): input tensor
        model (torch.nn.Module): model
        rotations (list): list of rotations
    """
    print("Testing equivariance:")
    print("Rotation i error: max(abs(y - y_rotated)) < 1e-6 * max(abs(y))")
    for i, abc in enumerate(rotations):
        abc = torch.tensor(abc)
        model = model.to(device)

        # (assert round false to avoid error)
        x2 = rotate_sparse_tensor(input, irreps_in, abc, device, assert_round=False)  # rotate input
        y2 = model(x2)

        y1 = model(input)
        y1 = rotate_sparse_tensor(y1, irreps_out, abc, device, assert_round=False)



        # check equivariance
        assert (y1.C[:, 1:] - y2.C[:, 1:]).abs().max() == 0
        if (y1.F - y2.F).abs().max() <= 1e-6 * y1.F.abs().max():
            print(f"Rotation {i} error: {(y1.F - y2.F).abs().max():.2e} < {1e-6 * y1.F.abs().max():.2e}")
        else:
            print(f"FAILED: Rotation {i} error: {(y1.F - y2.F).abs().max():.2e} > {1e-6 * y1.F.abs().max():.2e}")



class EquivModel(torch.nn.Module):
    def __init__(self, irreps_in=Irreps("1e"), irreps_out=Irreps("0e + 1e + 2e"), lmax = 2, segment = False) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax, -1)
        self.activation = [torch.relu,torch.tanh]
        self.irreps_coef = [16, 16, 6, 6, 2, 2, 0, 0] #list for coefficients of irreps in
        # the form "list[0]x0e + list[1]x0o + list[2]x1e + ...
        self.irreps_mid = o3.Irreps(f"{self.irreps_coef[0]}x0e + \
                                      {self.irreps_coef[1]}x0o + \
                                      {self.irreps_coef[2]}x1e + \
                                      {self.irreps_coef[3]}x1o + \
                                      {self.irreps_coef[4]}x2e + \
                                      {self.irreps_coef[5]}x2o + \
                                      {self.irreps_coef[6]}x3e + \
                                      {self.irreps_coef[7]}x3o").simplify()
        self.segment = segment
        self.network_initialization()
        self.weight_initialization()

    def network_initialization(self):
        blocks = []
        diameters = [11, 7, 7, 5, 3, 3, 3]
        steps = [(1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0)]


        for i, (diameter, step) in enumerate(zip(diameters, steps)):
            blocks.append(EquivariantConvolutionBlock(self.irreps_in,
                                                      self.irreps_mid,
                                                      self.irreps_sh,
                                                      diameter,
                                                      step,
                                                      self.activation))
            self.irreps_in = self.irreps_mid  # update input irreps for next block to irreps_mid
            
        blocks.append(EquivariantConvolutionBlock(self.irreps_mid,
                                                  self.irreps_out,
                                                  self.irreps_sh,
                                                  diameter = 3,
                                                  steps = (1.0, 1.0, 1.0),
                                                  activation=[self.activation[0]] # only need ReLU here
                                                  ))

        self.blocks = nn.ModuleList(blocks)

        self.global_pool = ME.MinkowskiGlobalAvgPooling()


    def forward(self, data) -> torch.Tensor:
        for i in range(len(self.blocks)):
            data = self.blocks[i](data)
        if not self.segment:
            data = self.global_pool(data)
        # x = self.softmax(x)
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

            if isinstance(m, Convolution): # TODO: try some other initialization here and linear
                nn.init.xavier_uniform_(m.weight, 1)


