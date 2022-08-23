import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchmetrics
import resnet
import pytorch_lightning as pl

class LitEngineResNet(pl.LightningModule):
    def __init__(self,pretrained=False,lr=1.0e-3, batch_size = 2):
        super().__init__()
        #self.wandb
        input_channels = 1
        if pretrained:
            # works on RGB images
            input_channels = 1
        self.lr = lr
        self.batch_size = batch_size
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
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
        return outputs['loss'].sum()/2

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
