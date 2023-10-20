import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


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


class SparseToFull(object):
    """Change from sparse data format to a full 3D image.
    Args:
        imagesize (L x W x H): Size of full image
    """

    def __init__(self, imagesize=[512, 512, 512]):
        super().__init__()
        assert isinstance(imagesize, tuple)
        if len(imagesize)!=3:
            raise TypeError(f'only implemented for 3D images')
        self.imagesize = imagesize

    def __call__(self, tensor):
        """
        Args:
            tensor: tensor to be densified

        Returns:
            Tensor: full tensor
        """
        indices = tensor[0].T
        values = tensor[1].T.squeeze()
        tensor_input = torch.sparse_coo_tensor(indices, values, self.imagesize, dtype=torch.float32)
        tensor_dense = tensor_input.to_dense()
        dense_unsqueeze = torch.unsqueeze(tensor_dense, dim=0)

        return dense_unsqueeze

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Image Size={self.imagesize})"


class PreProcess(object):
    """Class to presprocess data
    Args:
        norm (bool): normalize the data
        clip (bool): clip the data
        sqrt (bool): take square root of data
        norm_mean (float): mean of noise to be added
        norm_std (float): std of noise to be added
        clip_min (float): min clip value
        clip_max (float): max clip value
        full (bool): true if data is not in sparse form (No implementation for full at the moment)
    """

    def __init__(self, norm=False, clip=True, sqrt=True, norm_mean=0.65, norm_std=0.57, clip_min=0.0, clip_max=1.0):
        assert isinstance(norm_mean, float)
        assert isinstance(norm_std, float)
        assert isinstance(clip_min, float)
        assert isinstance(clip_max, float)
        super().__init__()
        for name, value in vars().items():
            if name != "self" and name != "hparams":
                setattr(self, name, value)

    def __call__(self, batch):
        """
        Args:
            batch: coords, feat to process

        Returns:
            batch: processed batch
        """
        coords, feat = batch
        feat = feat.float()
        if self.sqrt:
            torch.sqrt(feat, out=feat)  # Take in place square root of tensor
        if self.norm:
            torch.sub(feat, self.norm_mean, alpha=1, out=feat)  # subtract mean
            torch.div(feat, self.norm_std, out=feat)  # divide by std
        if self.clip:
            torch.clamp(feat, self.clip_min, self.clip_max, out=feat)
        return coords, feat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm={self.norm}, sqrt={self.sqrt}, clip={self.clip}, mean={self.norm_mean}, std={self.norm_std}, clip_min={self.clip_min}, clip_max={self.clip_max})"


class AddNoise(object):
    """Adds gaussian noise to the data

    Args:
        noise_mean (float): mean of noise to be added
        noise_std (float): std of noise to be added
        full (bool): true if data is not in sparse form
    """

    def __init__(self, noise_mean=0.0, noise_std=0.1, full=False):
        super().__init__()
        assert isinstance(noise_mean, float)
        assert isinstance(noise_std, float)
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.full = full

    def __call__(self, batch):
        """
        Args:
            batch: batch to add noise to

        Returns:
            batch: tensor with noise
        """
        coords, feat = batch
        if self.full:
            noise = torch.normal(self.noise_mean, self.noise_std, size=feat.shape)
            feat = feat + noise

            return coords, feat

        else:
            noise = torch.normal(self.noise_mean, self.noise_std, size=feat.shape).to(feat.device)
            feat = feat + noise

            return coords, feat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Noise mean={self.noise_mean}, Noise std={self.noise_std})"


class ToSingle(object):
    """Change datatype to single precision (float32).

    Args:
        None
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        """
        Args:
            sample: tensor to be changed to float32.

        Returns:
            Tensor: changed sample to float32
        """
        return np.float32(sample)


def rotated(inp, angles):
    r = R.from_euler('zyx', angles, degrees=True).as_matrix()
    # print(torch.tensor(r))
    return torch.matmul(inp, torch.tensor(r, dtype=torch.half))


if __name__ == "__main__":
    """
    a quick test program
    """
    # Test sparse to full
    sparse = torch.tensor(
        [[0, 0, 0, 0.1], [0, 0, 1, 0.2], [0, 1, 0, 0.3], [0, 1, 1, 0.4], [1, 0, 0, 0.5], [1, 0, 1, 0.6], [1, 1, 0, 0.7],
         [2, 1, 1, 0.8]])
    sparse_to_full = SparseToFull((3, 3, 3))
    full = sparse_to_full(sparse)
    print("Sparse to full")
    print(full)
    print(f"sparse:\n{sparse}")

    # Test preprocess
    preprocess = PreProcess()
    preprocessed = preprocess(sparse[:, -1])
    print(f"preprocess:\n{torch.hstack((sparse[:, :-1], preprocessed.reshape(-1, 1)))}")

    # Test add noise
    add_noise = AddNoise(device="cpu")
    noisy = add_noise(sparse[:, -1])
    print(f"noisy:\n{torch.hstack((sparse[:, :-1], noisy.reshape(-1, 1)))}")
