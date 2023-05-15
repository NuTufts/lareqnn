# Code written by Mario Geiger
# Taken from https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/v2203/sparse_voxel_convolution.py

import math

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import FullyConnectedTensorProduct, Linear, Irreps
from e3nn.nn import BatchNorm, Activation, Gate

try:
    from MinkowskiEngine import KernelGenerator, MinkowskiConvolutionFunction, SparseTensor
    from MinkowskiEngineBackend._C import ConvolutionMode
except ImportError:
    pass


class Convolution(torch.nn.Module):
    r"""convolution on voxels

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        input irreps

    irreps_out : `e3nn.o3.Irreps`
        output irreps

    irreps_sh : `e3nn.o3.Irreps`
        set typically to ``o3.Irreps.spherical_harmonics(lmax)``

    diameter : float
        diameter of the filter in physical units

    num_radial_basis : int
        number of radial basis functions

    steps : tuple of float
        size of the pixel in physical units

    no_linear : bool
        if True, the Linear layer is skipped and only a residual connection is used. (use only when irreps_in and
        irreps_out are the same)
    """

    def __init__(self, irreps_in, irreps_out, irreps_sh, diameter, num_radial_basis, steps=(1.0, 1.0, 1.0), no_linear=False):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)

        self.num_radial_basis = num_radial_basis
        self.no_linear = no_linear

        # self-connection
        if not self.no_linear:
            self.sc = Linear(self.irreps_in, self.irreps_out)

        # connection with neighbors
        r = diameter / 2

        s = math.floor(r / steps[0])
        x = torch.arange(-s, s + 1.0) * steps[0]

        s = math.floor(r / steps[1])
        y = torch.arange(-s, s + 1.0) * steps[1]

        s = math.floor(r / steps[2])
        z = torch.arange(-s, s + 1.0) * steps[2]

        # lattice = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)  # [x, y, z, R^3]
        lattice = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [x, y, z, R^3]
        self.register_buffer("lattice", lattice)

        emb = soft_one_hot_linspace(
            x=lattice.norm(dim=-1),
            start=0.0,
            end=r,
            number=self.num_radial_basis,
            basis="smooth_finite",
            cutoff=True,
        )
        self.register_buffer("emb", emb)

        sh = o3.spherical_harmonics(
            l=self.irreps_sh, x=lattice, normalize=True, normalization="component"
        )  # [x, y, z, irreps_sh.dim]
        self.register_buffer("sh", sh)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
            compile_left_right=False,
            compile_right=True,
        )

        self.weight = torch.nn.Parameter(torch.randn(self.num_radial_basis, self.tp.weight_numel))

        self.kernel_generator = KernelGenerator(lattice.shape[:3], dimension=3)

        self.conv_fn = MinkowskiConvolutionFunction()

    def kernel(self):
        weight = self.emb @ self.weight
        weight = weight / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2])
        kernel = self.tp.right(self.sh, weight)  # [x, y, z, irreps_in.dim, irreps_out.dim]

        # TODO: understand why this is necessary
        kernel = torch.einsum("xyzij->zyxij", kernel)  # [z, y, x, irreps_in.dim, irreps_out.dim]

        kernel = kernel.reshape(-1, *kernel.shape[-2:])  # [z * y * x, irreps_in.dim, irreps_out.dim]
        return kernel

    def forward(self, x):
        r"""
        Parameters
        ----------
        x : SparseTensor

        Returns
        -------
        SparseTensor
        """
        if self.no_linear:
            sc = x.F
        else:
            sc = self.sc(x.F)

        out = self.conv_fn.apply(
            x.F,
            self.kernel(),
            self.kernel_generator,
            ConvolutionMode.DEFAULT,
            x.coordinate_map_key,
            x.coordinate_map_key,
            x._manager,
        )

        return SparseTensor(
            sc + out,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x._manager,
        )


def get_batch_jumps(sparse_tensor):
    """Get the indices where the batch index changes"""
    c_jumps = sparse_tensor.coordinates.T[0]
    jumps = torch.where(c_jumps[:-1] != c_jumps[1:])[0] + 1
    end_value = torch.tensor([len(c_jumps)])
    z = torch.tensor([0])
    jumps_all = torch.cat((z, jumps, end_value)).T
    return jumps_all


class EquivariantBatchNorm(torch.nn.Module):
    r"""An equivariant batch normalization layer for a sparse tensor.
        See the pytorch :attr:`torch.nn.BatchNorm1d` for more details.
        """

    def __init__(
            self,
            irreps,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            reduce="mean",
            instance=False
    ):
        super(EquivariantBatchNorm, self).__init__()
        self.bn = BatchNorm(irreps,
                            eps,
                            momentum,
                            affine,
                            reduce=reduce,
                            instance=instance)
        self.irreps = irreps

    def forward(self, input):
        output = self.bn(input.F)

        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )

    def __repr__(self):
        s = "(irreps={}, eps={}, momentum={}, affine={}, reduce={}, instance={})".format(
            self.irreps,
            self.bn.eps,
            self.bn.momentum,
            self.bn.affine,
            self.bn.reduce,
            self.bn.instance
        )
        return self.__class__.__name__ + s


class EquivariantActivation(torch.nn.Module):
    r"""An equivariant activation layer for a sparse tensor. Uses Gate

    Parameters:
        irreps (Irreps): the irreps of the input
        acts (list of function): list of the activation function to use (Make sure even activations).
        To set up odd activations, modify this function and change irreps_gates
        """

    def __init__(
            self,
            irreps,
            acts,
    ):
        super(EquivariantActivation, self).__init__()

        self.irreps = irreps
        self.acts = acts

        irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps if ir.l == 0])
        irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps if ir.l > 0])
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")

        if irreps_gates.dim == 0:
            irreps_gates = irreps_gates.simplify()
            activation_gate = []
        else:
            activation_gate = [torch.sigmoid]
            # activation_gate = [torch.sigmoid, torch.tanh][:len(activation)]

        self.gate = Gate(irreps_scalars, self.acts, irreps_gates, activation_gate, irreps_gated)

        self.irreps_in = self.gate.irreps_in
        self.irreps_out = self.gate.irreps_out

    def forward(self, input):

        output = self.gate(input.F)

        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )

    def __repr__(self):
        s = "(irreps={}, act={})".format(
            self.irreps,
            self.acts
        )
        return self.__class__.__name__ + s


class EquivariantConvolutionBlock(torch.nn.Module):
    r"""An equivariant convolution block for a sparse tensor.

    Parameters:
        irreps_in (Irreps): the irreps of the input
        irreps_out (Irreps): the irreps of the output
        irreps_sh (Irreps): the irreps of the spherical harmonics
        diameter (float): the diameter of the convolution kernel
        steps (tuple of float): the step size of the convolution kernel
        activation (list of function): list of the activation functions to use for the gate
        """

    def __init__(
            self,
            irreps_in,
            irreps_out,
            irreps_sh,
            diameter,
            steps,
            activation,
    ):
        super(EquivariantConvolutionBlock, self).__init__()

        self.activation = EquivariantActivation(irreps_out, activation)
        self.conv = Convolution(irreps_in, self.activation.irreps_in, irreps_sh, diameter, num_radial_basis=3, steps=steps)
        self.BN = EquivariantBatchNorm(irreps_out)
        
    def forward(self, input):
        output = self.conv(input)
        output = self.activation(output)
        output = self.BN(output)

        return output



class EquivariantSoftMax(torch.nn.Module):
    r"""An equivariant softmax layer for a sparse tensor."""

    def __init__(
            self,
            dim=0,

    ):
        super(EquivariantSoftMax, self).__init__()
        self.softmax = torch.nn.Softmax(dim=dim)

    def forward(self, input):
        output = self.softmax(input.F)

        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )

    def __repr__(self):
        s = "(dim={})".format(
            self.softmax.dim,
        )
        return self.__class__.__name__ + s



def rotate_sparse_tensor(x, irreps, abc, device, assert_round=False):
    """Perform a rotation of angles abc to a sparse tensor"""

    coordinates = x.C[:, 1:].to(x.F.dtype).to(device)
    coordinates = torch.einsum("ij,bj->bi", Irreps("1e").D_from_angles(*abc).to(device), coordinates)
    #D_from_angles is YXY rotation applied right to left

    if assert_round:
        assert (coordinates - coordinates.round()).abs().max() < 1e-3
    coordinates = coordinates.round().to(torch.int32)
    coordinates = torch.cat([x.C[:, :1], coordinates], dim=1)

    # rotate the features (according to `irreps`)
    features = x.F.to(device)
    features = torch.einsum("ij,bj->bi", irreps.D_from_angles(*abc).to(device), features)

    return SparseTensor(coordinates=coordinates, features=features)
