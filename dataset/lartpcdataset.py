import os, sys
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate
from torchsparse import SparseTensor
import h5py
from collections import Counter
from torch.utils.data import Sampler

if __package__:
    from .data_utils import SparseToFull
else:
    from data_utils import SparseToFull


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
                 include_feats=False,
                 segment=False,
                 target_transform=None):
        super(PilarDatasetHDF5, self).__init__()

        self.file = h5py.File(inp_file, 'r')
        # self.datasets = ["coordinates", "labels", "energy_deposit", "energy_init", "pos", "mom", "start_indices", "end_indices", "charge"] # uncomment to get more information
        self.datasets = ["coordinates", "charge", "labels", "start_indices", "end_indices"]
        self.output_datasets = ["SparseImage", "labels"]
        if include_feats is not False:
            self.datasets.extend(["mom", "energy_deposit", "energy_init"])
            self.output_datasets.extend(["mom", "energy_deposit", "energy_init"])
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
        self.include_feats = include_feats
        self.segment = segment

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

        input = SparseTensor(coords=coords, feats=charge)

        if self.segment:
            label = label.repeat(coords.shape[0], 1)
            label = SparseTensor(coords=coords, feats=label)

        if self.include_feats is not False:
            mom = self.dset_references["mom"][index]
            mom = torch.from_numpy(mom)
            energy_init = self.dset_references["energy_init"][index]
            energy_init = torch.tensor([energy_init], dtype=torch.float32)
            energy_deposit = self.dset_references["energy_deposit"][index]
            energy_deposit = torch.tensor([energy_deposit], dtype=torch.float32)
            return (input, label, mom, energy_init, energy_deposit)

        return (input, label)

    def __len__(self):
        return self.length

    def counts(self):
        pdg_values = self.dset_references["labels"][:]

        class_labels = [self.pdgtolabels[int(pdg)] for pdg in pdg_values]

        label_counts = Counter(class_labels)

        return label_counts

class HDF5Sampler(Sampler):
    def __init__(self, data_source, batch_size=8, shuffle_mode="seq", drop_last=False, target_label=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_mode = shuffle_mode
        self.target_label = target_label
        self.length = len(data_source)

        if shuffle_mode not in ("random", "seq", None):
            raise TypeError('shuffle mode has to be either ("random", "seq", None)')

        # Filter indices by label if label is not None
        if target_label is not None:
            labels_tensor = torch.tensor(self.data_source.dset_references["labels"])
            self.indices = torch.nonzero(labels_tensor == self.target_label, as_tuple=True)[0]
            self.length = len(self.indices)
        else:
            self.indices = torch.arange(self.length)

    def __iter__(self):
        if self.shuffle_mode == "random":
            return iter(self._random_shuffle().tolist())
        elif self.shuffle_mode == "seq":
            return iter(self._sequential_shuffle().tolist())
        else:
            return iter(self._no_shuffle().tolist())

    def __len__(self):
        if self.drop_last:
            return self.length // self.batch_size
        else:
            return (self.length + self.batch_size - 1) // self.batch_size

    def _calculate_total_batches(self):
        total_batches = self.length // self.batch_size
        if self.length % self.batch_size != 0 and self.drop_last:
            total_batches -= 1
        return total_batches

    # Helper functions for shuffling indices
    # Fully random shuffle
    def _random_shuffle(self):
        shuffled_indices = self.indices[torch.randperm(self.length)]
        total_batches = self._calculate_total_batches()
        batched_indices = shuffled_indices[:total_batches * self.batch_size].view(total_batches,
                                                                                  self.batch_size)
        return batched_indices

    # sliding window shuffle wiht a random start
    def _sequential_shuffle(self):
        random_starts = torch.randint(0, self.length - self.batch_size,
                                      size=(self.length // self.batch_size,))
        batched_indices = torch.stack(
            [self.indices[start: start + self.batch_size] for start in random_starts])
        return batched_indices

    def _no_shuffle(self):
        total_batches = self._calculate_total_batches()
        batched_indices = torch.stack([self.indices[i: i + self.batch_size] for i in
                                       range(0, total_batches * self.batch_size, self.batch_size)])
        return batched_indices




class PilarDatasetHDF5Dense(torch.utils.data.Dataset):
    def __init__(self,
                 inp_file,
                 crop=True,
                 lazy=True,  # takes a LOT of space but increases speed 10x
                 transform=None,
                 include_feats=False,
                 segment=False,
                 projection=True,
                 image_size=(512,512),
                 target_transform=None):
        super(PilarDatasetHDF5Dense, self).__init__()

        self.file = h5py.File(inp_file, 'r')
        # self.datasets = ["coordinates", "labels", "energy_deposit", "energy_init", "pos", "mom", "start_indices", "end_indices", "charge"] # uncomment to get more information
        self.datasets = ["coordinates", "charge", "labels", "start_indices", "end_indices"]
        if include_feats is not False:
            self.datasets.extend(["mom", "energy_deposit", "energy_init"])
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
        self.include_feats = include_feats
        self.image_size = image_size
        self.crop = crop
        self.STF = SparseToFull(projection, image_size)

    def __getitem__(self, index):
        start, end = int(self.dset_references["start_indices"][index]), int(self.dset_references["end_indices"][index])
        coords = self.dset_references["coordinates"][start:end].astype('int32')
        charge = self.dset_references["charge"][start:end].reshape(-1, 1).astype('float32')
        pdg = self.dset_references["labels"][index]
        label = self.pdgtolabels[int(pdg)]

        if self.crop:
            coords -= coords.min(axis=0)

        coords = torch.from_numpy(coords)
        
        charge = torch.from_numpy(charge)
        label = torch.tensor([label], dtype=torch.int32)

        image = self.STF((coords, charge))


        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        # TODO: add segment option

        if self.include_feats is not False:
            mom = self.dset_references["mom"][index]
            mom = torch.from_numpy(mom)
            energy_init = self.dset_references["energy_init"][index]
            energy_init = torch.tensor([energy_init], dtype=torch.float32)
            energy_deposit = self.dset_references["energy_deposit"][index]
            energy_deposit = torch.tensor([energy_deposit], dtype=torch.float32)
            return image, label, mom, energy_init, energy_deposit

        return image, label

    def __len__(self):
        return self.length

    def counts(self):
        pdg_values = self.dset_references["labels"][:]

        class_labels = [self.pdgtolabels[int(pdg)] for pdg in pdg_values]

        label_counts = Counter(class_labels)

        return label_counts


def sparse_collate_fn_custom(inputs: List[Tuple[Any, ...]]) -> Tuple[Any, ...]:
    # Check if the first item in the list is a tuple
    if isinstance(inputs[0], tuple):
        output = []

        # Loop through each tuple item by index
        for idx in range(len(inputs[0])):
            item_type = type(inputs[0][idx])

            # Check the type of the item at the given index
            if item_type == np.ndarray:
                output.append(torch.stack([torch.tensor(input[idx]) for input in inputs], dim=0))
            elif item_type == torch.Tensor:
                output.append(torch.stack([input[idx] for input in inputs], dim=0))
            elif item_type == SparseTensor:
                output.append(sparse_collate([input[idx] for input in inputs]))
            else:
                output.append(torch.cat([torch.tensor(input[idx]) for input in inputs], dim=0))

        return tuple(output)
    else:
        return inputs


if __name__ == "__main__":
    """
    a quick test program
    """
    # Remove . from .data_utils to run this test
    dataset_location = "/home/oalterkait/PilarData/PilarDataTrain.h5"
    sparse = True
    include_feats = True
    segment = False

    if sparse:
        train_dataset = PilarDatasetHDF5(dataset_location, include_feats=include_feats, segment=segment)
        batch_size = 8
        sampler = HDF5Sampler(train_dataset, batch_size=batch_size, shuffle_mode="random")
        loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_sampler=sampler,
                                             collate_fn=sparse_collate_fn_custom,
                                             num_workers=1,
                                             prefetch_factor=2)


        it = iter(loader)
        batch = next(it)
        #     print(batch[0].sum())
        #     print(batch[1])
        x = batch


        classnames = train_dataset.classes
        idx_to_class = train_dataset.idx_to_class
        output_datasets = train_dataset.output_datasets

        for i, key in enumerate(output_datasets):
            print(key)

            if i==0:
                print("coords_shape = ", x[i].coords.shape)
                print(x[i].coords)
                print("feats_shape = ", x[i].feats.shape)
                print(x[i].feats)
                continue

            if key == "labels":
                if segment:
                    labels = x[i].F
                    print("labels_shape = ", labels.shape)
                else:
                    labels = x[i]
                    print([idx_to_class[int(i)] for i in labels])
                continue

            if isinstance(x[i], SparseTensor):
                print("shape = ", x[i].F.shape)
                print(x[i].F)
            else:
                print(x[i].shape)
                print(x[i])
            

        # for key, values in x.items():
        #     if key == "input": print("charge")
        #     else: print(key)
        #
        #     if key == "label":
        #         if segment:
        #             labels = x["label"].F
        #             print(labels.shape)
        #         else:
        #             labels = x["label"]
        #             print([idx_to_class[int(i)] for i in labels])
        #         continue
        #
        #     if isinstance(values, SparseTensor):
        #         print("shape = ", values.F.shape)
        #         print(values.F)
        #     else:
        #         print(values.shape)
        #         print(values)

    else:
        train_dataset = PilarDatasetHDF5Dense(dataset_location, include_feats=include_feats)
        batch_size = 8
        sampler = HDF5Sampler(train_dataset, batch_size=batch_size, shuffle_mode="random", target_label=2212)
        loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_sampler=sampler,
                                             num_workers=1,
                                             prefetch_factor=2)

        classnames = train_dataset.classes
        idx_to_class = train_dataset.idx_to_class

        it = iter(loader)
        batch = next(it)


        if include_feats:
            dict_labels = ["image", "label", "mom", "energy_init", "energy_deposit"]
        else:
            dict_labels = ["image", "label"]

        for i, key in enumerate(dict_labels):
            print(key)

            if key == "label":
                labels = batch[i]
                print("shape = ",labels.shape)
                print([idx_to_class[int(i)] for i in labels])
                continue

            print("shape = ",batch[i].shape)
            print(batch[i])