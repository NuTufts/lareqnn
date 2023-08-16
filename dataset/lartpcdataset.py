import os, sys
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import MinkowskiEngine as ME
import h5py
from collections import Counter
from torch.utils.data import Sampler


class lartpcDatasetSparse(torchvision.datasets.DatasetFolder):
    def __init__(self,
                 root='./data3d',
                 extensions='.npy',
                 transform=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        super().__init__(root=root, loader=np_loader, extensions=extensions, transform=transform)

        #lartpcDatasetSparse.classnames = ["electron", "gamma", "muon", "proton", "pion"]
        lartpcDatasetSparse.device = device

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        coords = torch.from_numpy(sample[:, :-1])
        feat = torch.from_numpy(sample[:, -1])
        label = torch.tensor([target])
        sample = coords, feat
        
        if self.transform is not None:
            coords, feat = self.transform((coords, feat))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return coords, feat, label


class PilarDatasetHDF5(torch.utils.data.Dataset):
    def __init__(self,
                 inp_file,
                 lazy=True,  # takes a LOT of space but increases speed 10x
                 transform=None,
                 include_vertex=False,
                 target_transform=None):
        super(PilarDatasetHDF5, self).__init__()

        self.file = h5py.File(inp_file, 'r')
        # self.datasets = ["coordinates", "labels", "energy_deposit", "energy_init", "pos", "mom", "start_indices", "end_indices", "charge"] # uncomment to get more information
        self.datasets = ["coordinates", "charge", "labels", "start_indices", "end_indices"]
        if include_vertex:
            self.datasets.append("pos")
        if lazy:
            self.dset_references = {name: self.file[name] for name in self.datasets}
        else:
            self.dset_references = {name: self.file[name][:] for name in self.datasets}

        self.pdgtolabels = {11: 0, -11: 0, 13: 1, -13: 1, 22: 2, 211: 3, -211: 3, 2212: 4}
        self.classes = ["electron", "muon", "gamma", "pion", "proton"]
        self.idx_to_class = {i: self.classes[i] for i in range(len(self.classes))}
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.dset_references["labels"])
        self.include_vertex = include_vertex

    def __getitem__(self, index):
        start, end = int(self.dset_references["start_indices"][index]), int(self.dset_references["end_indices"][index])
        coords = self.dset_references["coordinates"][start:end].astype('int32')
        charge = self.dset_references["charge"][start:end].reshape(-1, 1).astype('float32')
        pdg = self.dset_references["labels"][index]
        label = self.pdgtolabels[int(pdg)]

        coords = torch.from_numpy(coords)
        charge = torch.from_numpy(charge)
        label = torch.tensor([label], dtype=torch.int32)

        if self.transform is not None:
            coords, charge = self.transform((coords, charge))
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.include_vertex:
            pos = self.dset_references["pos"][index]
            pos = torch.from_numpy(pos)
            return coords, charge, label, pos

        # energy_deposit = self.dset_references["energy_deposit"][index]
        # energy_init = self.dset_references["energy_init"][index]
        # pos = self.dset_references["pos"][index]
        # mom = self.dset_references["mom"][index]

        # pos = torch.from_numpy(pos)
        # mom = torch.from_numpy(mom)

        return coords, charge, label

    def __len__(self):
        return self.length

    def counts(self):
        pdg_values = self.dset_references["labels"][:]

        class_labels = [self.pdgtolabels[int(pdg)] for pdg in pdg_values]

        label_counts = Counter(class_labels)

        return label_counts


class HDF5Sampler(Sampler):
    def __init__(self, data_source, batch_size=8, shuffle="seq", drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.length = len(data_source)

        if shuffle not in ("random", "seq", "none"):
            raise TypeError('shuffle mode has to be either ("random", "seq", "none")')

    def __iter__(self):
        if self.shuffle == "random":
            indices = torch.randint(0, self.length - 1, (len(self), self.batch_size,))
        elif self.shuffle == "seq":
            random_starts = torch.randint(0, self.length - 1 - self.batch_size, (len(self), 1))
            indices = random_starts + torch.arange(0, self.batch_size)
        elif self.shuffle == "none":
            starts = torch.arange(0, self.length - 1, self.batch_size)

            # Check if the last batch exceeds the length of the dataset
            if starts[-1] + self.batch_size >= self.length:
                # Revert to the same behavior as "seq"
                random_start = torch.randint(0, self.length - 1 - self.batch_size, (1,))[0]
                starts[-1] = random_start

            indices = starts[:, None] + torch.arange(0, self.batch_size)

        return iter(indices.tolist())

    def __len__(self):
        if self.drop_last:
            return self.length // self.batch_size
        else:
            return (self.length + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    """
    a quick test program
    """
    # Remove . from .data_utils to run this test
    sparse = True
    if sparse:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = lartpcDatasetSparse(root="../../PilarDataTrain", device=device)
        #                          ,transform=transforms.Compose([
        #     ]))
        test_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=4,
            collate_fn=ME.utils.batch_sparse_collate,
            shuffle=True)

        it = iter(test_loader)
        batch = next(it)
        #     print(batch[0].sum())
        #     print(batch[1])
        x = batch
        print(x)
        classnames = data.class_to_idx
        idx_to_class = {v:k for k,v in classnames.items()}
        print([idx_to_class[int(i)] for i in x[2]])
        print(x[0].shape)
