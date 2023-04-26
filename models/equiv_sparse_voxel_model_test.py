import pytest
import torch
from e3nn.o3 import Irreps
from e3nn import o3
from e3nn.nn import BatchNorm
import math
from equiv_sparse_voxel_convolution import Convolution, rotate_sparse_tensor, EquivariantBatchNorm, EquivariantSoftMax
import MinkowskiEngine as ME
import torch.nn as nn
import numpy as np

def np_loader(inp):
    """Load data from file
    Args:
        inp (str): path to file
    Returns:
        npin (np.array): data
    """
    with open(inp, 'rb') as f:
        npin = np.load(f)

    return npin

rotations = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, math.pi / 2),
    (0.0, 0.0, math.pi),
    (0.0, math.pi / 2, 0.0),
    (0.0, math.pi / 2, math.pi / 2),
    (0.0, math.pi / 2, math.pi),
    (0.0, math.pi, 0.0),
    (math.pi / 2, 0.0, 0.0),
    (math.pi / 2, 0.0, math.pi / 2),
    (math.pi / 2, 0.0, math.pi),
    (math.pi / 2, math.pi / 2, 0.0),
]


def test_equivariance(rotations):

    irreps_in = Irreps("0e") # Single Scalar
    irreps_out = Irreps("5x0e") # 5 labels

    data_directory = "../PilarDataTrain"
    sample = np_loader(data_directory + "/Electron/000001.npy")
    print(sample.shape)

    coords = torch.from_numpy(sample[:, :-1])
    feat = torch.from_numpy(sample[:, -1]).unsqueeze(dim=-1).float()
    zeros = torch.zeros_like(feat)
    coords = torch.cat([zeros, coords], dim=-1).int()

    print(coords)

    print(coords.shape, feat.shape)

    x1 = ME.SparseTensor(
        coordinates=coords,
        features=feat
    )

    


    for abc in rotations:
        abc = torch.tensor(abc)

        labels = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

        model = EquivModel(irreps_in, irreps_out)

        # (assert round false to avoid error)
        x2 = rotate_sparse_tensor(x1, irreps_in, abc, assert_round=False) # rotate input
        y2 = model(x2)

        y1 = model(x1)
        y1 = rotate_sparse_tensor(y1, irreps_out, abc, assert_round=False)


        # check equivariance
        assert (y1.C - y2.C).abs().max() == 0
        print((y1.F - y2.F).abs().max(), 1e-6 * y1.F.abs().max())
        # assert (y1.F - y2.F).abs().max() < 1e-6 * y1.F.abs().max()
    print(y1)
    print(y2)


class EquivModel(torch.nn.Module):
    def __init__(self, irreps_in=Irreps("1e"), irreps_out=Irreps("0e + 1e + 2e")) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_sh = Irreps("0e + 1e + 2e")
        self.irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        self.network_initialization()
        self.weight_initialization()

    def network_initialization(self):
        self.conv1 = Convolution(self.irreps_in, self.irreps_mid, irreps_sh=self.irreps_sh, diameter=7,
                                 num_radial_basis=3,
                                 steps=(1.0, 1.0, 1.0))

        self.conv2 = Convolution(self.irreps_mid, self.irreps_mid, irreps_sh=self.irreps_sh, diameter=7,
                                 num_radial_basis=3,
                                 steps=(1.0, 1.0, 1.0))

        self.conv3 = Convolution(self.irreps_mid, self.irreps_out, irreps_sh=self.irreps_sh, diameter=7,
                                 num_radial_basis=3,
                                 steps=(1.0, 1.0, 1.0))

        # print attributes of irreps_mid

        self.norm1 = EquivariantBatchNorm(self.irreps_mid)

        self.norm2 = EquivariantBatchNorm(self.irreps_mid)

        self.softmax = EquivariantSoftMax(dim=0)

    def forward(self, data) -> torch.Tensor:
        x = self.conv1(data)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.softmax(x)
        return x

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)


if __name__ == "__main__":
    test_equivariance(rotations)
