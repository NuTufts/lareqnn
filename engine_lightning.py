import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchmetrics
import pytorch_lightning as pl
import MEresnet
import resnetbase
import MinkowskiEngine as ME
import wandb
from lartpcdataset import PreProcess

class LitEngineResNetSparse(pl.LightningModule):
    def __init__(
        self,
        hparams,
        train_dataset,
        val_dataset,
        pretrained=False,
        input_channels = 1,
        class_names = ["electron","gamma","muon","proton","pion"],
        train_acc = torchmetrics.Accuracy(num_classes=5),
        valid_acc = torchmetrics.Accuracy(num_classes=5)
    ):
        
        super().__init__()
        for name, value in vars().items():
            if name != "self" and name != "hparams":
                setattr(self, name, value)
        #self.wandb
        self.model = MEresnet.ResNet18(in_channels=1, out_channels=5, D=3)
        self.loss_fn = torch.nn.CrossEntropyLoss()
#         for key in hparams.keys():
#             self.hparams[key]=hparams[key]
#             print(key, hparams[key])
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
        #sys.exit(0)

    def print_model(self):
        print(self.model)
        
    def forward(self,x):
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
        collate_fn = ME.utils.batch_sparse_collate,
        num_workers=8,
        pin_memory=self.pin_memory)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
        dataset=self.val_dataset,
        batch_size=self.batch_size,
        collate_fn = ME.utils.batch_sparse_collate,
        shuffle=False,
        num_workers=8,
        pin_memory=self.pin_memory)
    
    def calc_loss( self, pred, labels ):
        loss = self.loss_fn( pred, labels )
        return loss
    
    
    def training_step(self, train_batch, batch_idx):
        coords, feats, labels = train_batch # data batch, labels
        feats = self.PreProcess(feats)
        stensor = ME.SparseTensor(coordinates=coords, features=feats.unsqueeze(dim=-1).float())
        z = self.model(stensor) 
        loss = self.calc_loss( z.F, labels.long() )
        self.log('train_loss', loss, sync_dist=True, on_step=False, on_epoch=True, batch_size=self.batch_size)
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        return {'loss': loss, 'preds': z.F, 'target': labels.long()}

    def training_step_end(self, outputs):
        self.train_acc(outputs['preds'], outputs['target'])
        self.log('train_acc',self.train_acc, on_step=False, on_epoch=True)
        #self.log({"train_conf_mat" : wandb.plot.confusion_matrix(preds=outputs['preds'].argmax(axis=1).detach().cpu().numpy(), y_true=outputs['target'].detach().cpu().numpy(), class_names=self.class_names)})
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
        loss = self.calc_loss( z.F, labels.long() )
        self.log('val_loss', loss, batch_size=self.batch_size,on_step=False, on_epoch=True)
        return {'loss': loss, 'preds': z.F, 'target': labels.long()}
        
    def validation_step_end(self, outputs):
        self.valid_acc(outputs['preds'], outputs['target'])
        self.log('valid_acc',self.valid_acc, on_step=False, on_epoch=True)
        #self.log({"valid_conf_mat":wandb.plot.confusion_matrix(preds=outputs['preds'].argmax(axis=1).detach().cpu().numpy(), y_true=outputs['target'].detach().cpu().numpy(), class_names=self.class_names)})
        
#     def validation_epoch_end(self, outputs):


class LitEngineResNet(pl.LightningModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        pin_memory = True,
        pretrained=False,
        lr=5.0e-4, 
        batch_size = 4, 
        input_channels = 1,
        class_names = ["electron","gamma","muon","proton","pion"],
        train_acc = torchmetrics.Accuracy(task='multiclass',num_classes=5),
        valid_acc = torchmetrics.Accuracy(task='multiclass',num_classes=5)
    
    ):
        
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)
        self.model = resnetbase.generate_model(10,num_classes=5,
                                      input_channels=input_channels)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def print_model(self):
        print(self.model)
        
    def forward(self,x):
        print(x)
        assert 0
        embedding = self.model(x)
        return embedding
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
        dataset=self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=self.pin_memory)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
        dataset=self.val_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=self.pin_memory)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch # data batch, labels
        z = self.model(x) 
        loss = self.calc_loss( z, y )
        self.log('train_loss', loss)
        return {'loss': loss, 'preds': z, 'target': y}

    def training_step_end(self, outputs):
        # update and log
        self.train_acc(outputs['preds'], outputs['target'])
        self.log('train_acc', self.train_acc)
        return torch.mean(outputs['loss'])

    def calc_loss( self, pred, labels ):
        loss = self.loss_fn( pred, labels )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.model(x)
        loss = self.calc_loss( z, y )
        self.log('val_loss', loss)
        return {'loss': loss, 'preds': z, 'target': y}
        
    def validation_step_end(self, outputs):
        self.valid_acc(outputs['preds'], outputs['target'])
        self.log('valid_acc', self.valid_acc)

        