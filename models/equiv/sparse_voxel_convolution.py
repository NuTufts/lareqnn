# Code written by Mario Geiger
# Taken from https://github.com/e3nn/e3nn/blob/main/e3nn/nn/models/v2203/sparse_voxel_convolution.py

import math

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import FullyConnectedTensorProduct, Linear, Irreps
from e3nn.nn import BatchNorm, Activation, Gate

try:
    from MinkowskiEngine import KernelGenerator, MinkowskiConvolutionFunction, SparseTensor, MinkowskiInterpolation,\
        MinkowskiMaxPooling, MinkowskiAvgPooling
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

    def __init__(self, irreps_in, irreps_out, irreps_sh, diameter, num_radial_basis, steps=(1.0, 1.0, 1.0),
                 no_linear=False):
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
        lattice = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)  # [x, y, z, R^3]
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


class EquivariantDownSample(torch.nn.Module):
    r"""An equivariant downsampling layer for a sparse tensor. 

    Parameters:
        irreps (Irreps): the irreps of the input
        kernel_size (int): kernel size for the pooling layer
        stride (int): stride length for the pooling
        """

    def __init__(
        self,
        irreps,
        kernel_size,
        stride,
        pooling_mode = "max"
    ):
        super(EquivariantDownSample, self).__init__()
        self.irreps = irreps
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling_mode = pooling_mode
        self.max_pool = MinkowskiMaxPooling(kernel_size=kernel_size, stride=stride, dimension=3)
        self.avg_pool = MinkowskiAvgPooling(kernel_size=kernel_size, stride=stride, dimension=3)
        assert input.F.shape[1] == irreps.dim, "Shape mismatch"

    def forward(self, input):
        cat_list = []

        start = 0
        max_pool = ME.MinkowskiMaxPooling(kernel_size=kernel_size, stride=stride, dimension=dim)
        for i in self.irreps.ls:
    
            end = start + 2*i+1
            temp = input.F[:,start:end,...]
            if i == 0:
                cat_list.append(temp)
            else:
                # stack the features and their norms together
                norm = temp.norm(dim=1, keepdim=True)
                cat_list.append(norm)
    
            start = end
    
        # stack all tensors along the feature dimension
        stacked_features = torch.cat(cat_list, dim=1)
    
        # create a sparse tensor from the stacked features
        stacked_tensors = ME.SparseTensor(coordinates=input.C, features=stacked_features)

        # perform pooling on the stacked tensor
        if self.pooling_mode == "max":
            pooled_tensors = self.max_pool(stacked_tensors)
        elif self.pooling_mode == "avg":
            pooled_tensors = self.avg_pool(stacked_tensors)
        else:
            raise ValueError(f"Unknown mode {mode}")
    
        return pooled_tensors


    def __repr__(self):
        s = "(irreps={}, kernel_size={}, stride={}, pooling_mode={})".format(
            self.irreps,
            self.kernel_size,
            self.stride,
            self.pooling_mode
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
        self.conv = Convolution(irreps_in, self.activation.irreps_in, irreps_sh, diameter, num_radial_basis=3,
                                steps=steps)
        self.BN = EquivariantBatchNorm(irreps_out)

    def forward(self, input):
        output = self.conv(input)
        output = self.activation(output)
        output = self.BN(output)

        return output


class EquivariantSoftMax(torch.nn.Module):
    """
    An equivariant softmax layer for a sparse tensor.
    """

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



