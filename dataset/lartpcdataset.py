import os, sys
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import MinkowskiEngine as ME
from .data_utils import np_loader


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
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return *sample, label


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
