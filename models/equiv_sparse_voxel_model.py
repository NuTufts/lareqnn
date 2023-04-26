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
import lovely_tensors as lt


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


def main():
    irreps_in = Irreps("0e")  # Single Scalar
    irreps_out = Irreps("5x0e")  # 5 labels

    data_directory = "../PilarDataTrain"
    sample = np_loader(data_directory + "/Electron/000001.npy")

    coords = torch.from_numpy(sample[:, :-1])
    feat = torch.from_numpy(sample[:, -1]).unsqueeze(dim=-1).float()
    zeros = torch.zeros_like(feat)
    coords = torch.cat([zeros, coords], dim=-1).int()

    print(f"coords: {coords.shape}, feat: {feat.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    x1 = ME.SparseTensor(
        coordinates=coords,
        features=feat,
        device=device
    )

    label = torch.Tensor([1.0]).long().to(device)

    model = EquivModel(irreps_in, irreps_out).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    for step in range(100):
        pred = model(x1)
        # print(pred.F)
        # print(pred.F.shape)
        # print(torch.argmax(pred.F, dim=-1))
        # print(torch.argmax(pred.F, dim=-1).shape)

        # print model parameters
        param_max = 0.0
        for param in model.parameters():
            if param.max() > param_max:
                param_max = param.max()
        #print(param_max)

        loss = loss_fn(pred.F, label)

        # print(torch.argmax(pred, dim=-1))
        # print(label.shape)
        # print(torch.argmax(pred, dim=-1).shape)
        #
        # loss = (torch.argmax(pred, dim=-1) - label).pow(2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 5 == 0:
            # accuracy = pred.round().eq(labels).all(dim=1).double().mean(dim=0).item()
            print(f"epoch {step:5d} | loss {loss:<10.1f} ")

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


    x_test = ME.SparseTensor(coordinates=x1.C, features=x1.F, device=torch.device("cpu"))
    test_equivariance(x_test, model.to(torch.device("cpu")), rotations, irreps_in, irreps_out)


def test_equivariance(input, model, rotations, irreps_in, irreps_out):
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
        # (assert round false to avoid error)
        x2 = rotate_sparse_tensor(input, irreps_in, abc, assert_round=False)  # rotate input
        y2 = model(x2)

        y1 = model(input)
        y1 = rotate_sparse_tensor(y1, irreps_out, abc, assert_round=False)

        # check equivariance
        assert (y1.C - y2.C).abs().max() == 0
        print(f"Rotation {i} error: {(y1.F - y2.F).abs().max():.2e} < {1e-6 * y1.F.abs().max():.2e}")
        assert (y1.F - y2.F).abs().max() < 1e-6 * y1.F.abs().max()


class EquivModel(torch.nn.Module):
    def __init__(self, irreps_in=Irreps("1e"), irreps_out=Irreps("0e + 1e + 2e"), segment = False) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_sh = Irreps("0e + 1e + 2e")
        self.irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        self.segment = segment
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
        #
        self.norm1 = EquivariantBatchNorm(self.irreps_mid)
        #
        self.norm2 = EquivariantBatchNorm(self.irreps_mid)

        self.global_pool = ME.MinkowskiGlobalAvgPooling()


    def forward(self, data) -> torch.Tensor:
        x = self.conv1(data)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        if not self.segment:
            x = self.global_pool(x)
        # x = self.softmax(x)
        return x

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



if __name__ == "__main__":
    main()
    # test_equivariance(rotations)
