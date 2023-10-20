# Code written by Mario Geiger
# taken from https://github.com/e3nn/e3nn/blob/main/tests/nn/models/v2203/sparse_voxel_convolution_test.py

import pytest
import torch
from e3nn.o3 import Irreps
import math
from utils import rotate_sparse_tensor
from torchsparse import SparseTensor
from sparse_voxel_convolution import Convolution

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


device = "cuda" if torch.cuda.is_available() else "cpu"


# run using "python -m pytest sparse_voxel_convolution_test.py"
@pytest.mark.parametrize("abc", rotations)
def test_equivariance(abc, device=device):

    abc = torch.tensor(abc)

    irreps_in = Irreps("1e")
    irreps_out = Irreps("0e + 1e + 2e")

    conv = Convolution(irreps_in, irreps_out, irreps_sh="0e + 1e + 2e", diameter=5, num_radial_basis=3, steps=(1.0, 1.0, 1.0)).to(device)

    coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.int32)
    feats = irreps_in.randn(4, -1)

    x1 = SparseTensor(
        coords = coords,
        feats = feats
    ).to(device)

    x2 = rotate_sparse_tensor(x1, irreps_in, abc, device=device)
    y2 = conv(x2)

    y1 = conv(x1)
    y1 = rotate_sparse_tensor(y1, irreps_out, abc, device=device)

    # check equivariance
    assert (y1.C - y2.C).abs().max() == 0
    assert (y1.F - y2.F).abs().max() < 1e-6 * y1.F.abs().max()

