import os,sys
import numpy as np
import torch
import torchvision
import torchvision.datasets
import torchvision.transforms as tranforms

class lartpcDataset( torchvision.datasets.DatasetFolder ):
    CLASSNAMES = ["electron","gamma","muon","proton","pion"]
    NORM = True
    CLIP = True
    NORM_MEAN = 0.0
    NORM_STD = 1.0
    CLIP_MIN = -1.0
    CLIP_MAX = 1.0
    def __init__(self, 
                 root='./data3d', 
                 extensions='.npy', 
                 norm = True, clip = False,
                 norm_mean = 0.1131, norm_std = 0.2687, 
                 clip_min = 6.0, clip_max = 10.0,
                 transform = None):
        
        super().__init__( root=root, loader=lartpcDataset.load_data, extensions=extensions, transform = transform)
        
        lartpcDataset.metadata = {}
        lartpcDataset.NORM = norm
        lartpcDataset.CLIP = clip
        lartpcDataset.NORM_MEAN = norm_mean
        lartpcDataset.NORM_STD = norm_std
        lartpcDataset.CLIP_MIN = clip_min
        lartpcDataset.CLIP_MAX = clip_max

    def load_data(inp):
        #print("lartpcDataset.load_data: path=",inp)
        with open(inp, 'rb') as f:
            npin = np.load(f)
            
        if lartpcDataset.NORM:
            npin[:,-1] -= lartpcDataset.NORM_MEAN
            npin[:,-1] /= lartpcDataset.NORM_STD
            
        if lartpcDataset.CLIP:
            np.clip(npin[:,-1],
                    lartpcDataset.CLIP_MIN,
                    lartpcDataset.CLIP_MAX, out = npin[:,-1])
            
        return npin
    

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
    """

    def __init__(self, noise_mean = 0.0, noise_std = 0.1):
        super().__init__()
        assert isinstance(noise_mean,float)
        assert isinstance(noise_std,float)
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __call__(self, tensor):
        """
        Args:
            tensor to add noise to

        Returns:
            Tensor: tensor with noise
        """
        noise = torch.normal(self.noise_mean,self.noise_std, size = tensor.shape)

        return tensor+noise

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
    import torch
    
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
