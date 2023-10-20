import numpy as np
import torch
from torchsparse import SparseTensor

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import FullyConnectedTensorProduct, Linear, Irreps
from e3nn.nn import BatchNorm, Activation, Gate


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


def rotate_sparse_tensor(x, irreps, abc, device):
    """Perform a rotation of angles abc to a sparse tensor"""

    coordinates = x.C[:, 1:].to(x.F.dtype).to(device)
    coordinates = torch.einsum("ij,bj->bi", Irreps("1e").D_from_angles(*abc).to(device), coordinates)
    # D_from_angles is YXY rotation applied right to left

    coordinates = coordinates.round().to(torch.int32)
    coordinates = torch.cat([x.C[:, :1], coordinates], dim=1)

    # rotate the features (according to `irreps`)
    features = x.F.to(device)
    print(f"{features.shape=}")
    print(f"{irreps.D_from_angles(*abc).shape=}")
    features = torch.einsum("ij,bj->bi", irreps.D_from_angles(*abc).to(device), features)

    return SparseTensor(coords=coordinates, feats=features)


def test_equivariance(input_tensor, model, rotations_list, irreps_in, irreps_out, mode="mean", device=torch.device("cpu")):
    """
    Test equivariance of a model

    Args:
        input_tensor (ME.SparseTensor): input tensor
        model (torch.nn.Module): model
        rotations_list (list of lists): list of rotations
        irreps_in (e3nn.o3.Irreps): input irreps
        irreps_out (e3nn.o3.Irreps): output irreps
        mode (str): "max" or "mean"
        device (torch.device): device
    """
    print("Testing equivariance:")
    print(f"Rotation i error: {mode}(abs(y - y_rotated)) < 1e-6 * max(abs(y))")

    for i, rotation in enumerate(rotations_list):
        error, relative_max = get_equiv_error(input_tensor,
                                              model,
                                              rotation,
                                              irreps_in,
                                              irreps_out,
                                              mode,
                                              device,
                                              return_max=True)

        if error < 1e-6 * relative_max:
            print(f"Rotation {i} error: {error:.2e} < 1e-6 * {relative_max:.2e}")
        else:
            print(f"FAILED: Rotation {i} error: {error:.2e} > 1e-6 * {relative_max:.2e}")


def get_equiv_error(input_tensor, model, rotation, irreps_in, irreps_out, mode="mean", device=torch.device("cpu"),
                    return_max=False):
    """
    Get rotation equivariance error of a model

    Args:
        input_tensor (ME.SparseTensor): input tensor
        model (torch.nn.Module): model
        rotation (list): rotation
        irreps_in (e3nn.o3.Irreps): input irreps
        irreps_out (e3nn.o3.Irreps): output irreps
        mode (str): "max" or "mean"
        device (torch.device): device
        return_max (bool): return max value of output_rotated.F.abs()
    """
    model = model.to(device)
    rotation_tensor = torch.tensor(rotation)

    rotated_input = rotate_sparse_tensor(input_tensor, irreps_in, rotation_tensor, device)  # rotate input
    rotated_output_model = model(rotated_input)

    output_model = model(input_tensor)
    output_rotated = rotate_sparse_tensor(output_model, irreps_out, rotation_tensor, device)
    # temporary fix
    rotated_output_model = SparseTensor(coords=rotated_output_model.C, feats=rotated_output_model.F,
                                        coordinate_manager=input_tensor.coordinate_manager)
    output_rotated = SparseTensor(coords=output_rotated.C, feats=output_rotated.F,
                                  coordinate_manager=input_tensor.coordinate_manager)

    diff = (rotated_output_model - output_rotated).F.abs()

    if mode == "max":
        diff = diff.max()
    elif mode == "mean":
        diff = diff.mean()
    else:
        raise ValueError(f"Unknown mode {mode}")

    if return_max:
        return diff, output_rotated.F.abs().max()

    return diff


def get_batch_jumps(sparse_tensor):
    """Get the indices where the batch index changes"""
    c_jumps = sparse_tensor.C.T[0]
    jumps = torch.where(c_jumps[:-1] != c_jumps[1:])[0] + 1
    end_value = torch.tensor([len(c_jumps)])
    z = torch.tensor([0])
    jumps_all = torch.cat((z, jumps, end_value)).T
    return jumps_all
