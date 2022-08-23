import torch
from torchvision import transforms
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
import gc

from engine_lightning import LitEngineResNet
from lartpcdataset import lartpcDataset, SparseToFull
if __name__ == '__main__':
    wandb_logger = WandbLogger(project='lar-e3nn-base')

    pl.seed_everything(42, workers=True)

    DEVICE = torch.device("cuda")
    #DEVICE = torch.device("cpu")

    BATCHSIZE=2

    # data
    data_transform = transforms.Compose([
            SparseToFull()
    ])

    dataset = lartpcDataset( root="../data3d",transform=data_transform)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[450,50])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=8, 
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=8, 
        pin_memory=True)


    # model
    model = LitEngineResNet()
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
    
    
    trainer = pl.Trainer(gpus=2,
                         strategy='dp',
                         precision=16,
                         accumulate_grad_batches=8,
                         #deterministic=True,
                         limit_train_batches=5,
                         logger=wandb_logger, 
                         min_epochs=1,
                         max_epochs=500,
                         log_every_n_steps=1)
    
    trainer.fit(model, train_loader, val_loader)
