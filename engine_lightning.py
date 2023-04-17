import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchmetrics
import pytorch_lightning as pl
import models.resnet as resnet
# import models.instance_resnet as instance_resnet
import MinkowskiEngine as ME
import wandb
from dataset.data_utils import PreProcess, AddNoise


class LitEngineResNetSparse(pl.LightningModule):
    def __init__(
            self,
            hparams,
            train_dataset,
            val_dataset,
            classes,
            class_to_idx,
            pretrained=False,
            input_channels=1,
            train_acc=torchmetrics.Accuracy(task="multiclass", num_classes=5),
            valid_acc=torchmetrics.Accuracy(task="multiclass", num_classes=5)
    ):

        super().__init__()
        for name, value in vars().items():
            if name not in ["self", "hparams"]:
                setattr(self, name, value)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        #         for key in hparams.keys():
        #             self.hparams[key]=hparams[key]
        #             print(key, hparams[key])
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.lr = hparams["lr"]
        self.weight_decay = hparams["weight_decay"]
        self.batch_size = hparams["batch_size"]
        self.pin_memory = hparams["pin_memory"]
        self.epochs = hparams["epochs"]
        self.steps_per_epoch = hparams["steps_per_epoch"]
        self.PreProcess = PreProcess(hparams["normalize"],
                                     hparams["clip"],
                                     hparams["sqrt"],
                                     hparams["norm_mean"],
                                     hparams["norm_std"],
                                     hparams["clip_min"],
                                     hparams["clip_max"]
                                     )
        self.AddNoise = AddNoise(self.device)

        if hparams["model"] == "ResNet14":
            self.model = resnet.ResNet14(in_channels=1, out_channels=5, D=3)
        elif hparams["model"] == "ResNet18":
            self.model = resnet.ResNet18(in_channels=1, out_channels=5, D=3)
        elif hparams["model"] == "ResNet34":
            self.model = resnet.ResNet34(in_channels=1, out_channels=5, D=3)
        elif hparams["model"] == "ResNet50":
            self.model = resnet.ResNet50(in_channels=1, out_channels=5, D=3)
        # elif hparams["model"]=="InstanceResNet14":
        #     self.model = MEresnet.ResNet14(in_channels=1, out_channels=5, D=3)
        # elif hparams["model"]=="InstanceResNet18":
        #     self.model = MEresnet.ResNet18(in_channels=1, out_channels=5, D=3)
        # elif hparams["model"]=="InstanceResNet34":
        #     self.model = MEresnet.ResNet34(in_channels=1, out_channels=5, D=3)
        # elif hparams["model"]=="InstanceResNet50":
        #     self.model = MEresnet.ResNet50(in_channels=1, out_channels=5, D=3)
        else:
            raise Exception("A valid model was not chosen")

    def print_model(self):
        print(self.model)

    def forward(self, x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #         optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, epochs=self.epochs, steps_per_epoch=1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ME.utils.batch_sparse_collate,
            num_workers=8,
            pin_memory=self.pin_memory)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=ME.utils.batch_sparse_collate,
            shuffle=False,
            num_workers=8,
            pin_memory=self.pin_memory)

    def calc_loss(self, pred, labels):
        loss = self.loss_fn(pred, labels)
        return loss

    def training_step(self, train_batch, batch_idx):
        coords, feats, labels = train_batch  # data batch, labels
        feats = self.PreProcess(feats)
        self.AddNoise.device = self.device
        feats = self.AddNoise(feats)
        stensor = ME.SparseTensor(coordinates=coords, features=feats.unsqueeze(dim=-1).float())
        z = self.model(stensor)
        loss = self.calc_loss(z.F, labels.long())
        self.log('train_loss', loss, sync_dist=True, on_step=False, on_epoch=True, batch_size=self.batch_size)
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        return {'loss': loss, 'preds': z.F, 'target': labels.long()}

    def training_step_end(self, outputs):
        self.train_acc(outputs['preds'], outputs['target'])
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        # self.log({"train_conf_mat" : wandb.plot.confusion_matrix(preds=outputs['preds'].argmax(axis=1).detach().cpu().numpy(), y_true=outputs['target'].detach().cpu().numpy(), class_names=self.class_names)})
        return torch.mean(outputs['loss'])

    #     def training_epoch_end(self, outputs):
    #         sch = self.lr_schedulers()
    #         #self.log('lr',self.lr, on_epoch=True)
    #         sch.step()

    def validation_step(self, val_batch, batch_idx):
        coords, feats, labels = val_batch
        feats = self.PreProcess(feats)
        stensor = ME.SparseTensor(coordinates=coords, features=feats.unsqueeze(dim=-1).float())
        z = self.model(stensor)
        loss = self.calc_loss(z.F, labels.long())
        self.log('val_loss', loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return {'loss': loss, 'preds': z.F, 'target': labels.long()}

    def validation_step_end(self, outputs):
        self.valid_acc(outputs['preds'], outputs['target'])
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)
        # self.log({"valid_conf_mat":wandb.plot.confusion_matrix(preds=outputs['preds'].argmax(axis=1).detach().cpu().numpy(), y_true=outputs['target'].detach().cpu().numpy(), class_names=self.class_names)})

#     def validation_epoch_end(self, outputs):
