import torch
from engine_lightning import LitEngineResNet
from lartpcdataset import lartpcDataset, SparseToFull
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project='tutorial-resnet-lightning')

pl.seed_everything(42, workers=True)

DEVICE = torch.device("cuda")
#DEVICE = torch.device("cpu")

BATCHSIZE=8

# data
dataset = lartpcDataset( root="../data3d",transform=transforms.Compose([
        SparseToFull()
    ]))
train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[450,50])

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCHSIZE,
    shuffle=True)
val_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=BATCHSIZE,
    shuffle=False)


# model
model = LitEngineResNet().to(DEVICE)
model.print_model()

# testing block
if False:
    print("//////// TESTING BLOCK //////////")
    imgs, labels = next( iter(train_loader) )
    imgs = imgs.to(DEVICE)
    labels = labels.to(DEVICE)
    print("imgs: ",imgs.shape)
    print("labels: ",labels.shape," :: ",labels)
    out = model.forward(imgs)
    print(out.shape)

    loss = model.calc_loss( out, labels )
    print(loss)

# training
trainer = pl.Trainer(gpus=1,
                     precision=16,
                     accumulate_grad_batches=4,
                     deterministic=True,
                     limit_train_batches=5,
                     logger=wandb_logger, 
                     min_epochs=1,
                     max_epochs=100)
trainer.fit(model, train_loader, val_loader)
