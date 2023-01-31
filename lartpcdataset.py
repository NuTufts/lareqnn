import os,sys
import numpy as np
import torch
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import MinkowskiEngine as ME
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class lartpcDataset( torchvision.datasets.DatasetFolder ):
    CLASSNAMES = ["electron","gamma","muon","proton","pion"]
    def __init__(self, 
                 root='./data3d', 
                 extensions='.npy', 
                 norm = True, clip = True, sqrt = False,
                 norm_mean = 0.39, norm_std = 0.3, 
                 clip_min = -1.0, clip_max = 3.0,
                 transform = None,
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        
        super().__init__( root=root, loader=lartpcDataset.load_data, extensions=extensions, transform = transform)
        
        lartpcDataset.metadata = {}
        lartpcDataset.NORM = norm
        lartpcDataset.CLIP = clip
        lartpcDataset.SQRT = sqrt
        lartpcDataset.NORM_MEAN = norm_mean
        lartpcDataset.NORM_STD = norm_std
        lartpcDataset.CLIP_MIN = clip_min
        lartpcDataset.CLIP_MAX = clip_max
        lartpcDataset.device = device

    def load_data(inp):
        #print("lartpcDataset.load_data: path=",inp)
        with open(inp, 'rb') as f:
            npin = np.load(f)
            #npin = np.expand_dims(npin,axis=0)
            
        if lartpcDataset.SQRT:
            npin[:,-1] = np.sqrt(npin[:,-1])
            
        if lartpcDataset.NORM:
            npin[:,-1] -= lartpcDataset.NORM_MEAN
            npin[:,-1] /= lartpcDataset.NORM_STD
            
        if lartpcDataset.CLIP:
            np.clip(npin[:,-1],
                    lartpcDataset.CLIP_MIN,
                    lartpcDataset.CLIP_MAX, out = npin[:,-1])
            
        return npin
    
        
class lartpcDatasetSparse( torchvision.datasets.DatasetFolder ):
    CLASSNAMES = ["electron","gamma","muon","proton","pion"]
    def __init__(self, 
                 root='./data3d', 
                 extensions='.npy', 
                 norm = True, clip = True, sqrt = False,
                 norm_mean = 0.39, norm_std = 0.3, 
                 clip_min = -1.0, clip_max = 3.0,
                 transform = None,
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        
        super().__init__( root=root, loader=lartpcDataset.load_data, extensions=extensions, transform = transform)
        
        lartpcDataset.metadata = {}
        lartpcDataset.NORM = norm
        lartpcDataset.CLIP = clip
        lartpcDataset.SQRT = sqrt
        lartpcDataset.NORM_MEAN = norm_mean
        lartpcDataset.NORM_STD = norm_std
        lartpcDataset.CLIP_MIN = clip_min
        lartpcDataset.CLIP_MAX = clip_max
        lartpcDataset.device = device

    def load_data(inp):
        #print("lartpcDataset.load_data: path=",inp)
        with open(inp, 'rb') as f:
            npin = np.load(f)
            #npin = np.expand_dims(npin,axis=0)
            
        if lartpcDataset.SQRT:
            npin[:,-1] = np.sqrt(npin[:,-1])
            
        if lartpcDataset.NORM:
            npin[:,-1] -= lartpcDataset.NORM_MEAN
            npin[:,-1] /= lartpcDataset.NORM_STD
            
        if lartpcDataset.CLIP:
            np.clip(npin[:,-1],
                    lartpcDataset.CLIP_MIN,
                    lartpcDataset.CLIP_MAX, out = npin[:,-1])
            
        return npin
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        coords = torch.from_numpy(sample[:,:-1])
        feat = torch.from_numpy(sample[:,-1])
        label = torch.tensor([target])
        # mod's by Taritree
        #coords = torch.from_numpy(sample[:,:-1]).type(torch.LongTensor)
        #feat = torch.from_numpy(sample[:,-1]).unsqueeze(1).type(torch.FloatTensor) # ME needs feature to have 2D shape (N,1)
        #label = torch.tensor([target]).type(torch.LongTensor)
        return coords, feat, label        
    

class SparseToFull(object):
    """Change from sparse data format to a full 3D image.

    Args:
        imagesize (L x W x H): Size of full image
    """

    def __init__(self, imagesize = (512,512,512)):
        super().__init__()
        assert isinstance(imagesize,tuple)
        assert len(imagesize) == 3
        self.imagesize = imagesize

    def __call__(self, tensor):
        """
        Args:
            tensor to be changed to dense.

        Returns:
            Tensor: tensor in dense form.
        """
        indices = tensor[:,:-1].T
        values = tensor[:,-1].T
        tensorinput = torch.sparse_coo_tensor(indices, values, self.imagesize, dtype=torch.float32)
        tensordense = tensorinput.to_dense()
        denseunsqueeze = torch.unsqueeze(tensordense,dim=0)
        
        return denseunsqueeze

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Image Size={self.imagesize})"     
    
    
class AddNoise(object):
    """Addes gaussian noise to the data

    Args:
        noise_mean (float): mean of noise to be added
        noise_std (float): std of noise to be added
        full (bool): true if data is not in sparse form
    """

    def __init__(self, noise_mean = 0.0, noise_std = 0.1, full = False):
        super().__init__()
        assert isinstance(noise_mean,float)
        assert isinstance(noise_std,float)
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.full = full

    def __call__(self, tensor):
        """
        Args:
            tensor to add noise to

        Returns:
            Tensor: tensor with noise
        """
        if self.full:
            noise = torch.normal(self.noise_mean,self.noise_std, size = tensor.shape)

            return tensor+noise

        else:
            noise = torch.normal(self.noise_mean,self.noise_std, size = tensor.shape[0])
            return tensor[:,-1] + noise
        
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
    
    
    
    
    
    
if __name__ == "__main__":
    """
    a quick test program
    """
    sparse = False
    if sparse:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = lartpcDatasetSparse(root="../data3d",device=device)
    #                          ,transform=transforms.Compose([
    #     ]))
        test_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=4,
            collate_fn = ME.utils.batch_sparse_collate,
            shuffle=True)

        it = iter(test_loader)
        batch = next(it)
    #     print(batch[0].sum())
    #     print(batch[1])
        x = batch
        print(x)
    else:
        data = lartpcDataset(root="../data3d",transform=transforms.Compose([
            SparseToFull()
        ]))
        test_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=4,
            shuffle=True)

        it = iter(test_loader)
        batch = next(it)
    #     print(batch[0].sum())
    #     print(batch[1])
        x = batch
        print(x)
