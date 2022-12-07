import torch
from torchvision import transforms
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
import gc
import MinkowskiEngine as ME
import MEresnet

from engine_lightning import LitEngineResNet, LitEngineResNetSparse
from lartpcdataset import lartpcDataset, lartpcDatasetSparse, SparseToFull
if __name__ == '__main__':
    wandb_logger = WandbLogger(project='lar-e3nn-sparse')

    pl.seed_everything(42, workers=True)

    DEVICE = torch.device("cuda")
    #DEVICE = torch.device("cpu")

    BATCHSIZE=4
    sparse = False
    
    if sparse:
        data_transform = transforms.Compose([
        ])
        dataset = lartpcDatasetSparse( root="../PilarDataTest",transform=data_transform,device = DEVICE)
    else:
        data_transform = transforms.Compose([
             SparseToFull()
        ])
        dataset = lartpcDataset( root="../PilarDataTest",transform=data_transform,device = DEVICE)

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[round(0.8*len(dataset)),round(0.2*len(dataset))])



    
    
    # model
    if sparse:
        model = LitEngineResNetSparse(batch_size=BATCHSIZE, train_dataset=train_dataset, val_dataset=valid_dataset)
    else:
        model = LitEngineResNet(batch_size=BATCHSIZE, train_dataset=train_dataset, val_dataset=valid_dataset,pin_memory=False)
    
    model.print_model()
    model = model
    
    wandb_logger.watch(model)

#     # testing block
#     print("//////// TESTING BLOCK //////////")
#     imgs, labels = next( iter(train_loader) )
#     imgs = imgs.to(DEVICE)
#     labels = labels.to(DEVICE)
#     print("imgs: ",imgs.shape)
#     print("labels: ",labels.shape," :: ",labels)
#     out = model.forward(imgs)
#     print(out.shape)

#     loss = model.calc_loss( out, labels )
#     print(loss)

    # training
    
    
    trainer = pl.Trainer(accelerator='gpu',
                         devices=2,
                         strategy='ddp',
                         precision=16,
                         accumulate_grad_batches=1,
                         #deterministic=True,
                         logger=wandb_logger, 
                         min_epochs=1,
                         max_epochs=500,
                         log_every_n_steps=1,
                         gradient_clip_val=0.5,
                         overfit_batches=1)
    
    if sparse:
        ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    
    trainer.fit(model)
