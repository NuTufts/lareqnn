import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchmetrics
#import resnet
import pytorch_lightning as pl
import MEresnet
import MinkowskiEngine as ME

class LitEngineResNet(pl.LightningModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        pretrained=False,
        lr=1.0e-3, 
        batch_size = 4, 
        sparse = True,
        input_channels = 1,
        class_names = ["electron","gamma","muon","proton","pion"],
        train_acc = torchmetrics.Accuracy(num_classes=5),
        valid_acc = torchmetrics.Accuracy(num_classes=5)
    
    ):
        
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)
        #self.wandb
        if self.sparse:
            self.model = MEresnet.ResNet14(in_channels=1, out_channels=5, D=3)
        else:    
            self.model = resnet.generate_model(10,num_classes=5,
                                      input_channels=input_channels)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def print_model(self):
        print(self.model)
        
    def forward(self,x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
        dataset=self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        collate_fn = ME.utils.batch_sparse_collate,
        num_workers=8, 
        pin_memory=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
        dataset=self.val_dataset,
        batch_size=self.batch_size,
        collate_fn = ME.utils.batch_sparse_collate,
        shuffle=False,
        num_workers=8, 
        pin_memory=True)
    
    
    
    def training_step(self, train_batch, batch_idx):
        coords, feats, labels = train_batch # data batch, labels
        stensor = ME.SparseTensor(coordinates=coords, features=feats.unsqueeze(dim=-1).float())
        z = self.model(stensor) 
        loss = self.calc_loss( z.F, labels.long() )
        self.log('train_loss', loss, sync_dist=True)
        self.train_acc(z.F, labels.long())
        #trainacclog = {classn:self.train_acc[i] for i, classn in enumerate(self.class_names)}
        self.log('train_acc', self.train_acc , on_step=False, on_epoch=True)
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        return loss

    def training_step_end(self, outputs):
        return #outputs['loss'].sum()#/len(outputs['loss'])

    def calc_loss( self, pred, labels ):
        loss = self.loss_fn( pred, labels )
        return loss

    def validation_step(self, val_batch, batch_idx):
        coords, feats, labels = val_batch
        stensor = ME.SparseTensor(coordinates=coords, features=feats.unsqueeze(dim=-1).float())
        z = self.model(stensor) 
        loss = self.calc_loss( z.F, labels.long() )
        self.log('val_loss', loss, sync_dist=True)
        self.valid_acc(z.F, labels.long())
        #valacclog = {classn:self.valid_acc[i] for i, classn in enumerate(self.class_names)}
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        return loss
        
    def validation_step_end(self, outputs):
        return
        #self.valid_acc(outputs['preds'], outputs['target'])
        #self.log('valid_acc', self.valid_acc)
