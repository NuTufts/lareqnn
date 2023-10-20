from typing import List, Tuple, Union

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch import nn

from torchsparse import SparseTensor
from torchsparse import nn as spnn

import torchsparse.nn.functional as F

from .resnet_torchsparse_blocks import SparseConvBlock, SparseResBlock

__all__ = ['SparseResNet21D']

import torchsparse.backends

torchsparse.backends.allow_tf32 = True


class SparseResNet(nn.ModuleList):

    def __init__(
        self,
        blocks: List[Tuple[int, int, Union[int, Tuple[int, ...]],
                           Union[int, Tuple[int, ...]]]],
        *,
        in_channels: int = 4,
        width_multiplier: float = 1.0,
        out_classes: int = 1,

    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.in_channels = in_channels
        self.width_multiplier = width_multiplier
        self.out_classes = out_classes

        F.set_kmap_mode("hashmap")
        F.set_conv_mode(2)

        for num_blocks, out_channels, kernel_size, stride in blocks:
            out_channels = int(out_channels * width_multiplier)
            blocks = []
            for index in range(num_blocks):
                if index == 0:
                    blocks.append(
                        SparseConvBlock(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                        ))
                else:
                    blocks.append(
                        SparseResBlock(
                            in_channels,
                            out_channels,
                            kernel_size,
                        ))
                in_channels = out_channels
            self.append(nn.Sequential(*blocks))

        self.append(spnn.GlobalMaxPool())

        self.append(nn.Linear(out_channels, out_classes))


    def forward(self, x: SparseTensor) -> List[SparseTensor]:
        for module in self:
            x = module(x)
        return x


class SparseResNet14(SparseResNet):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            blocks=[
                (1, 32, 3, 2),
                (1, 64, 3, 2),
                (1, 128, 3, 2),
                (1, 256, 3, 2),
                (1, 512, 3, 1)
            ],
            **kwargs,
        )

class SparseResNet18(SparseResNet):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            blocks=[
                (1, 32, 3, 2),
                (2, 64, 3, 2),
                (2, 128, 3, 2),
                (2, 256, 3, 2),
                (2, 512, 3, 1)
            ],
            **kwargs,
        )


class SparseResNet34(SparseResNet):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            blocks=[
                (1, 32, 3, 2),
                (3, 64, 3, 2),
                (4, 128, 3, 2),
                (6, 256, 3, 2),
                (3, 512, 3, 1)
            ],
            **kwargs,
        )


class SparseResNet50(SparseResNet):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            width_multiplier=2.0,
            blocks=[
                (1, 32, 3, 2),
                (3, 64, 3, 2),
                (4, 128, 3, 2),
                (6, 256, 3, 2),
                (3, 512, 3, 2)
            ],
            **kwargs,
        )


if __name__ == '__main__':
    # test that the model works

    import argparse
    import random
    from typing import Any, Dict

    import numpy as np
    import torch
    import torch.utils.data
    from torch import nn
    from torch.cuda import amp

    import torchsparse
    from torchsparse import SparseTensor
    from torchsparse import nn as spnn
    from torchsparse.utils.collate import sparse_collate_fn
    from torchsparse.utils.quantize import sparse_quantize

    import torchsparse.nn.functional as F

    F.set_kmap_mode("hashmap")

    class RandomDataset:

        def __init__(self, input_size: int, voxel_size: float, num_feats: int, num_outputs: int) -> None:
            self.input_size = input_size
            self.voxel_size = voxel_size
            self.num_feats = num_feats
            self.num_outputs = num_outputs

        def __getitem__(self, _: int) -> Dict[str, Any]:
            inputs_coords = np.random.uniform(-100, 100, size=(self.input_size, 3))
            inputs_feats = np.random.uniform(-100, 100, size=(self.input_size, self.num_feats))
            label = np.random.choice(self.num_outputs, size=(1,))

            coords, feats = inputs_coords, inputs_feats
            coords -= np.min(coords, axis=0, keepdims=True)
            coords, indices = sparse_quantize(coords,
                                              self.voxel_size,
                                              return_index=True)

            coords = torch.tensor(coords, dtype=torch.int)
            feats = torch.tensor(feats[indices], dtype=torch.float)
            label = torch.tensor(label, dtype=torch.long)

            input = SparseTensor(coords=coords, feats=feats)
            return {'input': input, 'label': label}

        def __len__(self):
            return 100


    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = RandomDataset(input_size=1000, voxel_size=0.2, num_feats=4, num_outputs=10)
    dataflow = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=sparse_collate_fn,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SparseResNet21D(in_channels=4, out_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = amp.GradScaler(enabled=True)

    for k, feed_dict in enumerate(dataflow):
        inputs = feed_dict['input'].to(device)
        labels = feed_dict['label'].reshape(-1).to(device)

        with amp.autocast(enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        print(f'[step {k + 1}] loss = {loss.item()}')

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # enable torchsparse 2.0 inference
    model.eval()
    # enable fused and locality-aware memory access optimization
    torchsparse.backends.benchmark = True  # type: ignore

    with torch.no_grad():
        for k, feed_dict in enumerate(dataflow):
            inputs = feed_dict['input'].to(device).half()
            labels = feed_dict['label'].reshape(-1).to(device)

            with amp.autocast(enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            print(f'[inference step {k + 1}] loss = {loss.item()}')

    assert False

    epochs = 100
    num_batches = 8
    num_outputs = 5

    #device = torch.device('cpu')
    model = SparseResNet21D(in_channels=6, out_classes=num_outputs).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scalar = amp.GradScaler()

    print(model)

    # get sparse data, coordinates and features
    data_single = []

    for i in range(num_batches):
        length = torch.randint(100, 1000, (1,))
        coords = torch.randint(0, 32, (length, 3)).to(torch.int)
        feats = torch.rand(length, 6).to(torch.float)
        # 1 label per batch
        label = torch.randint(0, num_outputs, (1,))
        image = SparseTensor(coords=coords, feats=feats)
        data_dict_single = {"data": image, "label": label}
        data_single.append(data_dict_single)

    # collate data
    data = sparse_collate_fn(data_single)

    print(data["data"].coords)
    print(data["data"].feats)

    print(data["data"].to(device))

    # train
    for k, epoch in enumerate(range(epochs)):
        model.train()
        data_input = data["data"].to(device)
        label = data["label"].to(device)
        with amp.autocast():
            output = model(data_input)
            loss = criterion(output, label)

        print(f'[step {k + 1}] loss = {loss.item()}')

        optimizer.zero_grad()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()



