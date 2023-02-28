import torch
from torchvision import transforms
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
#from pl_bolts.callbacks import ModuleDataMonitor
import gc
import MinkowskiEngine as ME
import MEresnet
import wandb


from engine_lightning import LitEngineResNet, LitEngineResNetSparse
from lartpcdataset import lartpcDataset, lartpcDatasetSparse, SparseToFull
if __name__ == '__main__': 
    # Sweep parameters

    config = dict(
            train_datapath = "../PTrain",
            test_datapath = "../PilarDataTest",
            batch_size = 4,
            lr = 1e-4,
            weight_decay = 1e-2,
            grad_batches = 1,
            epochs = 500,
            pin_memory = True,
            grad_clip = 0.5,
            steps_per_epoch = 100,
            normalize = True, 
            clip = True, 
            sqrt = True, 
            norm_mean = 0.65, 
            norm_std = 0.57, 
            clip_min = -1.0, 
            clip_max = 1.0
        )
    
    
    
    

#     wandb.init(config=hyperparameter_defaults)
#     # Config parameters are automatically set by W&B sweep agent
#     config = wandb.config
#     #config = hyperparameter_defaults
    
    
    
    
    
    
    
    wandb_logger = WandbLogger(project='lar-e3nn-sparse')

    pl.seed_everything(42, workers=True)

    DEVICE = torch.device("cuda")
    #DEVICE = torch.device("cpu")

    
    sparse = True
    
    if sparse:
        data_transform = transforms.Compose([
        ])
        dataset = lartpcDatasetSparse( root=config["train_datapath"],transform=data_transform,device = DEVICE)
#         indices = list(range(0, config["max_length"], 2))
#         datasetnew = torch.utils.data.Subset(dataset,indices)
    else:
        data_transform = transforms.Compose([
             SparseToFull()
        ])
        dataset = lartpcDataset( root=config["train_datapath"],transform=data_transform,device = DEVICE)

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[round(0.8*len(dataset)),round(0.2*len(dataset))])



    
    
    # model
    if sparse:
        model = LitEngineResNetSparse(hparams = config, train_dataset=train_dataset, val_dataset=valid_dataset)
    else:
        model = LitEngineResNet(batch_size=BATCHSIZE, train_dataset=train_dataset, val_dataset=valid_dataset,pin_memory=False)
    
    model.print_model()
    model = model
    
    wandb_logger.watch(model, log = "all")

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
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    #monitor = ModuleDataMonitor(submodules=True)
    
    trainer = pl.Trainer(accelerator='gpu',
                     devices=2,
                     strategy='ddp',
                     precision=16,
                     accumulate_grad_batches=config["grad_batches"],
                     #deterministic=True,
                     logger=wandb_logger, 
                     min_epochs=1,
                     max_epochs=config["epochs"],
                     log_every_n_steps=10,
                     #overfit_batches=4,
                     #gradient_clip_val=config["grad_clip"],
                     limit_train_batches=config["steps_per_epoch"],
                     limit_val_batches=50,
                     callbacks=[lr_monitor])
                     #callbacks=[monitor])
    
    if sparse:
        ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    
    trainer.fit(model)
