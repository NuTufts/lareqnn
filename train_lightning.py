import os, sys
import yaml
import torch
from torchvision import transforms
import lightning.pytorch as pl
from argparse import ArgumentParser
from lightning.pytorch.loggers import WandbLogger
# from pl_bolts.callbacks import ModuleDataMonitor
import gc
import MinkowskiEngine as ME
import wandb

from engine_lightning import LitEngineResNetSparse
from dataset.lartpcdataset import PilarDatasetHDF5

from dataset.data_utils import SparseToFull, PreProcess, AddNoise


if __name__ == '__main__':
    # Sweep parameters
    workdir = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    config_loc = workdir+"/configs/default_config.yaml"

    with open(config_loc, "r") as yaml_file:
        hyperparameter_defaults = yaml.safe_load(yaml_file)
   
    # wandb.init(config=hyperparameter_defaults)
    # Config parameters are automatically set by W&B sweep agent
    config = hyperparameter_defaults
    # config = hyperparameter_defaults

    wandb_logger = WandbLogger(project=config["project"])

    wandb_logger.log_hyperparams(config)

    #pl.seed_everything(42, workers=True)

    DEVICE = torch.device("cuda")
    # DEVICE = torch.device("cpu")

    PreProcess = PreProcess(config["normalize"],
                            config["clip"],
                            config["sqrt"],
                            config["norm_mean"],
                            config["norm_std"],
                            config["clip_min"],
                            config["clip_max"])
    AddNoise = AddNoise()

    train_transform_cpu = transforms.Compose([PreProcess, AddNoise
    ])
    valid_transform_cpu = transforms.Compose([PreProcess
    ])
    train_transform_gpu = transforms.Compose([
                                                  ])
    valid_transform_gpu = transforms.Compose([
                                                  ])

    train_dataset = PilarDatasetHDF5(inp_file=config["train_datapath"], transform=train_transform_cpu)
    valid_dataset = PilarDatasetHDF5(inp_file=config["valid_datapath"], transform=valid_transform_cpu)

    #dataset = lartpcDatasetSparse(root=config["train_datapath"], transform=data_transform, device=DEVICE)
    #train_dataset = lartpcDatasetSparse(root=config["train_datapath"], transform=train_transform_cpu, device=DEVICE)
    #valid_dataset = lartpcDatasetSparse(root=config["valid_datapath"], transform=valid_transform_cpu, device=DEVICE)


    model = LitEngineResNetSparse(hparams=config,
                                  train_dataset=train_dataset,
                                  val_dataset=valid_dataset,
                                  classes=train_dataset.classes,
                                  idx_to_class=train_dataset.idx_to_class,
                                  train_transform_gpu=train_transform_gpu,
                                  valid_transform_gpu=valid_transform_gpu)

    model.print_model()

    wandb_logger.watch(model, log="all", log_freq=1000)

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
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', every_n_epochs=100)
    early_stopping = pl.callbacks.EarlyStopping('val_loss', patience=20)
    # monitor = ModuleDataMonitor(submodules=True)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=config["gpus"],
                         strategy='auto',
                         precision=32,
                         accumulate_grad_batches=config["grad_batches"],
                         # deterministic=True,
                         logger=wandb_logger,
                         min_epochs=1,
                         max_epochs=config["epochs"],
                         log_every_n_steps=10,
                         # overfit_batches=4,
                         gradient_clip_val=config["grad_clip"],
                         limit_train_batches=config["steps_per_epoch"],
                         limit_val_batches=100,
                         callbacks=[lr_monitor, checkpoint_callback, early_stopping])
    #                    callbacks=[lr_monitor, early_stopping])
    # callbacks=[monitor])

    ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    trainer.fit(model)
