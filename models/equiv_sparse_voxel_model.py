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
    segment = False
    epochs = 1000

    data_directory = "../PilarDataTrain/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_locations = [data_directory + "/Electron/000005.npy",
                      data_directory + "/Muon/000001.npy",
                      data_directory + "/Gamma/000004.npy",
                      data_directory + "/Proton/000007.npy",
                      data_directory + "/Pion/000001.npy",
                      data_directory + "/Electron/000007.npy",
                      data_directory + "/Muon/000005.npy",
                      data_directory + "/Gamma/000008.npy",
                      ]



    true_labels = [0, 1, 2, 3, 4, 0, 1, 2]

    #
    # file_locations = [data_directory + "/Electron/000005.npy",
    #                   data_directory + "/Muon/000001.npy"]
    #
    # true_labels = [0, 1]


    data = []
    weights = torch.zeros(5)

    for i, file in enumerate(file_locations):
        sample = np_loader(file)
        coords = torch.from_numpy(sample[:, :-1])
        feat = torch.from_numpy(sample[:, -1]).unsqueeze(dim=-1).float()
        feat = torch.sqrt(feat)
        label = torch.Tensor([true_labels[i]]).int()
        if segment:
            labels = label*torch.ones_like(feat).int()
            data.append((coords, feat, labels))
            weights[int(label)] += len(sample)
        else:
            data.append((coords, feat, label))
            weights[int(label)] += 1

    weights = torch.Tensor(weights)

    weights = 1 - weights/weights.sum()

    print(weights)

    collate = ME.utils.batch_sparse_collate
    collated_data = collate(data)

    coords, feats, labels = collated_data

    print(f"collated_data: {collated_data=}")

    print(f"{labels=}")

    print(f"coords: {coords.shape}, feat: {feats.shape}")

    labels = labels.long().to(device)
    
    if segment:
        labels = labels.squeeze(1)



    x1 = ME.SparseTensor(
        coordinates=coords,
        features=feats,
        device=device
    )


    model = EquivModel(irreps_in, irreps_out, segment=segment).to(device)

    test_equivariance(x1, model, rotations, irreps_in, irreps_out, device=device)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

    losses = np.zeros(epochs)
    accuracies = np.zeros((epochs,5))

    for step in range(epochs):
        pred = model(x1)
        # print(pred.F)
        # print(pred.F.shape)
        # print(torch.argmax(pred.F, dim=-1))
        # print(torch.argmax(pred.F, dim=-1).shape)

        # print model parameters
        param_max = 0.0
        param_max_name = ""
        for name, param in model.named_parameters():
            if param.abs().max() > param_max:
                param_max = param.abs().max()
                param_max_name = name

        if step == 1000:
            optim.param_groups[0]['lr'] = 0.001
        if step == 1800:
            optim.param_groups[0]['lr'] = 0.001

        loss = loss_fn(pred.F, labels)
        # print(torch.argmax(pred, dim=-1))
        # print(label.shape)
        # print(torch.argmax(pred, dim=-1).shape)
        #
        # loss = (torch.argmax(pred, dim=-1) - label).pow(2).mean()


        optim.zero_grad()
        loss.backward()
        optim.step()

        losses[step] = loss.detach().cpu().numpy()

        accuracy = (torch.argmax(pred.F, dim=-1) == labels).float().mean()
        per_class_accuracy = []

        # calculate confusion matrix
        confusion_matrix = torch.zeros((5, 5))
        num_labels = torch.zeros((5, 1))
        for i in range(5):
            for j in range(5):
                confusion_matrix[i, j] = ((torch.argmax(pred.F.detach(), dim=-1) == i) & (labels == j)).float().sum()
            num_labels[i] = (labels == i).float().sum()
            accuracies[step, i] = confusion_matrix[i, i] / num_labels[i]

        if step % 5 == 0:
            print("Total number of instances per class:")
            print(num_labels.numpy().T)
            print("Confusion matrix:")
            with np.printoptions(precision=4, suppress=True):
                print(confusion_matrix.numpy())

            print(f"epoch {step:5d} | loss {loss:<3.2f} | accuracy {accuracy:<3.2f} | max {param_max_name} {param_max:4.3f}")

    np.save("losses.npy",losses)
    np.save("accuracies.npy",accuracies)
    test_equivariance(x1, model, rotations, irreps_in, irreps_out, device=device)


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


