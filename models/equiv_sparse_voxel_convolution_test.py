# Code written by Mario Geiger
# taken from https://github.com/e3nn/e3nn/blob/main/tests/nn/models/v2203/sparse_voxel_convolution_test.py

import pytest
import torch
from e3nn.o3 import Irreps
import math

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





@pytest.mark.parametrize("abc", rotations)
def test_equivariance(abc):
    pytest.importorskip("MinkowskiEngine")

    from MinkowskiEngine import SparseTensor
    from equiv_sparse_voxel_convolution import Convolution

    abc = torch.tensor(abc)

    irreps_in = Irreps("1e")
    irreps_out = Irreps("0e + 1e + 2e")

    conv = Convolution(irreps_in, irreps_out, irreps_sh="0e + 1e + 2e", diameter=7, num_radial_basis=3, steps=(1.0, 1.0, 1.0))

    x1 = SparseTensor(
        coordinates=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.int32),
        features=irreps_in.randn(4, -1),
    )

    x2 = rotate_sparse_tensor(x1, irreps_in, abc)
    y2 = conv(x2)

    y1 = conv(x1)
    y1 = rotate_sparse_tensor(y1, irreps_out, abc)

    # check equivariance
    assert (y1.C - y2.C).abs().max() == 0
    assert (y1.F - y2.F).abs().max() < 1e-6 * y1.F.abs().max()

