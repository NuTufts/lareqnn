import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchmetrics
import lightning.pytorch as pl
import models.resnet as resnet
# import models.instance_resnet as instance_resnet
import MinkowskiEngine as ME
import wandb
from dataset.data_utils import PreProcess, AddNoise
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
            class_to_idx,
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


        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.lr = hparams["lr"]
        self.weight_decay = hparams["weight_decay"]
        self.batch_size = hparams["batch_size"]
        self.pin_memory = hparams["pin_memory"]
        self.epochs = hparams["epochs"]
        self.steps_per_epoch = hparams["steps_per_epoch"]
        
        self.train_transform_gpu = train_transform_gpu
        self.valid_transform_gpu = valid_transform_gpu

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5, average="none")
        self.train_conf = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=5)
        self.valid_conf = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=5)


        if hparams["model"] == "ResNet14":
            self.model = resnet.ResNet14(in_channels=1, out_channels=5, D=3)
        elif hparams["model"] == "ResNet18":
            self.model = resnet.ResNet18(in_channels=1, out_channels=5, D=3)
        elif hparams["model"] == "ResNet34":
            self.model = resnet.ResNet34(in_channels=1, out_channels=5, D=3)
        elif hparams["model"] == "ResNet50":
            self.model = resnet.ResNet50(in_channels=1, out_channels=5, D=3)
        elif hparams["model"] == "ResNet101":
            self.model = resnet.ResNet101(in_channels=1, out_channels=5, D=3)
        else:
            raise Exception("A valid model was not chosen")

        # get counts of each parameter
        self.counts = [value for key, value in dict(Counter(self.train_dataset.targets)).items()]
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
            shuffle=True,
            num_workers=8,
            pin_memory=self.pin_memory)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=ME.utils.batch_sparse_collate,
            shuffle=True,
            num_workers=8,
            pin_memory=self.pin_memory)

    def calc_loss(self, pred, labels):
        loss = self.loss_fn(pred, labels)
        return loss

    def training_step(self, train_batch, batch_idx):
        coords, feats, labels = train_batch  # data batch, labels
        if self.train_transform_gpu is not None:
            coords, feats = self.train_transform_gpu((coords, feats))
            
        stensor = ME.SparseTensor(coordinates=coords, features=feats.unsqueeze(dim=-1).float())
        z = self.model(stensor)
        loss = self.calc_loss(z.F, labels.long())
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.train_acc.update(z.F, labels.long())
        self.train_conf.update(z.F, labels.long())

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
        coords, feats, labels = val_batch
        if self.valid_transform_gpu is not None:
            coords, feats = self.valid_transform_gpu((coords,feat))
            
        stensor = ME.SparseTensor(coordinates=coords, features=feats.unsqueeze(dim=-1).float())
        z = self.model(stensor)
        loss = self.calc_loss(z.F, labels.long())
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.valid_acc.update(z.F, labels.long())
        self.valid_conf.update(z.F, labels.long())

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
