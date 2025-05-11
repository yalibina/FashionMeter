# Imports
import torch
from torch import nn
import pytorch_lightning as pl
from transformers import ViTForImageClassification
from sklearn.metrics import f1_score, fbeta_score

class LitViT(pl.LightningModule):
    def __init__(self, num_labels, id2label, label2id, class_weights, lr=1e-3, weight_decay=1e-2):
        super(LitViT, self).__init__()
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.class_weights = class_weights
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        self.save_hyperparameters()

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        accuracy = (preds == labels_np).mean()
        f1 = f1_score(labels_np, preds, average='macro')
        beta = 0.5
        f_beta = fbeta_score(labels_np, preds, beta=beta, average='macro')
        return loss, accuracy, f1, f_beta

    def training_step(self, batch, batch_idx):
        loss, accuracy, f1, f_beta = self.common_step(batch, batch_idx)
        self.log("train/loss", loss)
        self.log("train/accuracy", accuracy)
        self.log("train/f1", f1)
        self.log("train/fbeta", f_beta)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, f1, f_beta = self.common_step(batch, batch_idx)
        self.log("val/loss_epoch", loss, on_epoch=True)
        self.log("val/accuracy_epoch", accuracy, on_epoch=True)
        self.log("val/f1_epoch", f1, on_epoch=True)
        self.log("val/fbeta_epoch", f_beta, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, f1, f_beta = self.common_step(batch, batch_idx)
        self.log("test/loss_epoch", loss, on_epoch=True)
        self.log("test/accuracy_epoch", accuracy, on_epoch=True)
        self.log("test/f1_epoch", f1, on_epoch=True)
        self.log("test/fbeta_epoch", f_beta, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])