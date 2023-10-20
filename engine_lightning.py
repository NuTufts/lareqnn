import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchmetrics
import lightning.pytorch as pl
import models.resnet_torchsparse as resnet
from torchsparse import SparseTensor
import wandb
from dataset.data_utils import PreProcess, AddNoise
from dataset.lartpcdataset import HDF5Sampler, sparse_collate_fn_custom
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class LitEngineResNetSparse(pl.LightningModule):
    def __init__(
            self,
            hparams,
            train_dataset,
            val_dataset,
            classes,
            idx_to_class,
            train_transform_gpu=None,
            valid_transform_gpu=None,
            test_dataset=None,
            pretrained=False,
            input_channels=1
    ):

        super().__init__()
        for name, value in vars().items():
            if name not in ["self", "hparams"]:
                setattr(self, name, value)


        self.idx_to_class = idx_to_class
        self.lr = hparams["lr"]
        self.weight_decay = hparams["weight_decay"]
        self.batch_size = hparams["batch_size"]
        self.epochs = hparams["epochs"]
        self.steps_per_epoch = hparams["steps_per_epoch"]
        self.prefetch_factor = hparams["prefetch_factor"]
        self.shuffle_mode = hparams["shuffle_mode"]
        
        self.train_transform_gpu = train_transform_gpu
        self.valid_transform_gpu = valid_transform_gpu

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5, average="none")
        self.train_conf = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=5)
        self.valid_conf = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=5)


        if hparams["model"] == "SparseResNet14":
            self.model = resnet.SparseResNet14(in_channels=1, out_classes=5)
        elif hparams["model"] == "SparseResNet18":
            self.model = resnet.SparseResNet18(in_channels=1, out_classes=5)
        elif hparams["model"] == "SparseResNet34":
            self.model = resnet.SparseResNet34(in_channels=1, out_classes=5)
        elif hparams["model"] == "SparseResNet50":
            self.model = resnet.SparseResNet50(in_channels=1, out_classes=5)
        else:
            raise Exception("A valid model was not chosen")

        # get counts of each parameter
        self.counts = [value for key, value in self.train_dataset.counts().items()]
        print(self.counts)
        # normalize counts
        self.normalized_counts = torch.tensor([1 / (value / sum(self.counts)) for value in self.counts])

        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.normalized_counts, label_smoothing=hparams["label_smoothing"])

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
        sampler = HDF5Sampler(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_mode)
        loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                             batch_sampler=sampler,
                                             collate_fn=sparse_collate_fn_custom,
                                             num_workers=1,
                                             prefetch_factor=self.prefetch_factor)
        return loader

    def val_dataloader(self):
        sampler = HDF5Sampler(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle_mode)
        loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                             batch_sampler=sampler,
                                             collate_fn=sparse_collate_fn_custom,
                                             num_workers=1,
                                             prefetch_factor=self.prefetch_factor)
        return loader

    def test_dataloader(self):
        sampler = HDF5Sampler(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle_mode)
        loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                             batch_sampler=sampler,
                                             collate_fn=sparse_collate_fn_custom,
                                             num_workers=1,
                                             prefetch_factor=self.prefetch_factor)
        return loader

    def calc_loss(self, pred, labels):
        loss = self.loss_fn(pred, labels)
        return loss

    def training_step(self, train_batch, batch_idx):
        input = train_batch[0]
        labels = train_batch[1].reshape(-1)
        
        # TODO: fix this so it has correct format for torchsparse
        if self.train_transform_gpu is not None:
            raise NotImplementedError
            coords, feats = self.train_transform_gpu(input)

        z = self.model(input)
        loss = self.calc_loss(z, labels.long())
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.train_acc.update(z, labels.long())
        self.train_conf.update(z, labels.long())

        self.log(f"train_acc", self.train_acc, on_step=False, on_epoch=True, sync_dist=True)

        self.log(f"data_steps", float(self.batch_size * self.global_step), on_step=False, on_epoch=True, sync_dist=True,\
                 batch_size=self.batch_size)

        if self.global_step % 10000 == 0:
            torch.cuda.empty_cache()
        return loss

    def on_train_epoch_end(self):

        # for i, cl in enumerate(self.classes):
        #     self.log(f"train_acc_{cl}", train_accuracies[i], sync_dist=True)

        # fig = plot_confusion_matrix(self.train_conf.compute().cpu().numpy(), self.classes)
        # plt.savefig(f"/plots/train_confusion_matrix_{self.global_step}.png")

        self.train_acc.reset()

    def validation_step(self, val_batch, batch_idx):
        input = val_batch[0]
        labels = val_batch[1].reshape(-1)

        # TODO: fix this so it has correct format for torchsparse
        if self.valid_transform_gpu is not None:
            raise NotImplementedError
            coords, feats = self.valid_transform_gpu(input)

        z = self.model(input)
        loss = self.calc_loss(z, labels.long())
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.valid_acc.update(z, labels.long())
        self.valid_conf.update(z, labels.long())

        return loss

    #
    def on_validation_epoch_end(self):
        valid_accuracies = self.valid_acc.compute().cpu().numpy()

        for i, cl in enumerate(self.classes):
            self.log(f"valid_acc_{cl}", valid_accuracies[i], sync_dist=True)

        self.log("valid_acc", np.mean(valid_accuracies), sync_dist=True)

        self.valid_acc.reset()
        self.valid_conf.reset()

    # def validation_step_end(self, outputs):
    #     self.valid_acc(outputs['preds'], outputs['target'])
    #     self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True, batch_size=self.batch_size)
    #     # self.log({"valid_conf_mat":wandb.plot.confusion_matrix(preds=outputs['preds'].argmax(axis=1).detach().cpu().numpy(), y_true=outputs['target'].detach().cpu().numpy(), class_names=self.class_names)})


#     def validation_epoch_end(self, outputs):


def plot_confusion_matrix(confusion_matrix, class_names):
    """Plots the confusion matrix.

    Parameters:
    - confusion_matrix: a 2D numpy array representing the confusion matrix.
    - class_names: a list of strings representing the names of the classes.

    Returns:
    - fig: matplotlib Figure instance with the plot.
    """
    # Create a new figure
    fig, ax = plt.subplots()

    # Display the confusion matrix as a color-encoded 2D plot
    cax = ax.matshow(confusion_matrix, cmap='coolwarm')  # Change colormap as needed

    # Add colorbar for reference
    fig.colorbar(cax)

    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Add tick marks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    # Label ticks with the respective class names
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations with the value of each cell
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, confusion_matrix[i, j],
                    ha="center", va="center", color="black")  # Adjust color as needed

    return fig
