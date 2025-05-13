import torch
from pytorch_lightning import Trainer
from models.mobilenet import MobileNetDistill
from src.dataload.dataset import train_dataloader, val_dataloader, class_weights, NUM_LABELS, ids2label, label2ids

# Set these variables directly in Colab
teacher_ckpt = '/path/to/litvit.ckpt'  # Update this path in Colab
max_epochs = 10
lr = 1e-3
weight_decay = 1e-2
alpha = 0.5
temperature = 4.0

model = MobileNetDistill(
    num_labels=NUM_LABELS,
    class_weights=class_weights,
    teacher_ckpt_path=teacher_ckpt,
    id2label=ids2label,
    label2id=label2ids,
    lr=lr,
    weight_decay=weight_decay,
    alpha=alpha,
    temperature=temperature
)

trainer = Trainer(max_epochs=max_epochs)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)