import pytest
import torch
from e3nn.o3 import Irreps
from e3nn import o3
from e3nn.nn import BatchNorm
import math
from utils import rotate_sparse_tensor, test_equivariance, np_loader
from sparse_voxel_model import EquivModel
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


def main():
    irreps_in = Irreps("0e")  # Single Scalar
    irreps_out = Irreps("5x0e")  # 5 labels
    segment = False #keep false if downsampling
    epochs = 2000
    lr = 1e-2

    data_directory = "../PilarData/Train"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Selected data to test overfitting
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

    data = []
    weights = torch.zeros(5)

    for i, file in enumerate(file_locations):
        sample = np_loader(file)
        coords = torch.from_numpy(sample[:, :-1])
        feat = torch.from_numpy(sample[:, -1]).unsqueeze(dim=-1).float()
        feat = torch.sqrt(feat)
        label = torch.Tensor([true_labels[i]]).int()
        if segment:
            labels = label * torch.ones_like(feat).int()
            data.append((coords, feat, labels))
            weights[int(label)] += len(sample)
        else:
            data.append((coords, feat, label))
            weights[int(label)] += 1

    weights = torch.Tensor(weights)

    weights = 1 - weights / weights.sum()

    print(weights)

    collate = ME.utils.batch_sparse_collate
    collated_data = collate(data)

    coords, feats, labels = collated_data

    print(len(labels))
    
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

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    losses = np.zeros(epochs)
    accuracies = np.zeros((epochs, 5))

    for step in range(epochs):
        pred = model(x1)

        # print model parameters
        param_max = 0.0
        param_max_name = ""
        for name, param in model.named_parameters():
            if param.abs().max() > param_max:
                param_max = param.abs().max()
                param_max_name = name

        if step == 500:
            optim.param_groups[0]['lr'] = 0.005
        if step == 1000:
            optim.param_groups[0]['lr'] = 0.001

        loss = loss_fn(pred.F, labels)
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

            print(
                f"epoch {step:5d} | loss {loss:<3.2f} | accuracy {accuracy:<3.2f} | max {param_max_name} {param_max:4.3f}")

    np.save("losses.npy", losses)
    np.save("accuracies.npy", accuracies)
    test_equivariance(x1, model, rotations, irreps_in, irreps_out, device=device)


if __name__ == "__main__":
    main()
    # test_equivariance(rotations)
