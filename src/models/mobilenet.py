import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.models import mobilenet_v2
from torch.nn import functional as F
from src.models.vit import LitViT
from sklearn.metrics import f1_score, fbeta_score

class MobileNetDistill(pl.LightningModule):
    def __init__(self, num_labels, class_weights, teacher_ckpt_path, id2label, label2id, lr=1e-3, weight_decay=1e-2, alpha=0.5, temperature=4.0):
        super().__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.class_weights = class_weights
        self.alpha = alpha
        self.temperature = temperature
        self.student = mobilenet_v2(weights=None, num_classes=num_labels)
        self.teacher = LitViT(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            class_weights=class_weights
        )
        self.teacher.load_state_dict(torch.load(teacher_ckpt_path)['state_dict'])
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.student(x)

    def distillation_loss(self, student_logits, teacher_logits, labels):
        ce_loss = nn.CrossEntropyLoss(weight=self.class_weights.to(student_logits.device))(student_logits, labels)
        log_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = F.kl_div(log_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss

    def common_step(self, batch, batch_idx):
        x = batch['pixel_values']
        y = batch['labels']
        student_logits = self.student(x)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        loss = self.distillation_loss(student_logits, teacher_logits, y)
        preds = student_logits.argmax(dim=1).detach().cpu().numpy()
        labels_np = y.detach().cpu().numpy()
        accuracy = (preds == labels_np).mean()
        f1 = f1_score(labels_np, preds, average='macro')
        beta = 0.5
        f_beta = fbeta_score(labels_np, preds, beta=beta, average='macro')
        return loss, accuracy, f1, f_beta

    def training_step(self, batch, batch_idx):
        loss, accuracy, f1, f_beta = self.common_step(batch, batch_idx)
        self.log('train/loss', loss)
        self.log('train/accuracy', accuracy)
        self.log('train/f1', f1)
        self.log('train/fbeta', f_beta)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, f1, f_beta = self.common_step(batch, batch_idx)
        self.log('val/loss_epoch', loss, on_epoch=True)
        self.log('val/accuracy_epoch', accuracy, on_epoch=True)
        self.log('val/f1_epoch', f1, on_epoch=True)
        self.log('val/fbeta_epoch', f_beta, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
