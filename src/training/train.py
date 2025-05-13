import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.dataload.dataset import (
    train_dataloader,
    val_dataloader,
    class_weights,
    ids2label,
    label2ids,
    NUM_LABELS
)
from src.models.vit import LitViT
pl.seed_everything(42, workers=True)
# Load config
with open(os.path.join(os.path.dirname(__file__), '../../config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

N_EPOCHS = config['N_EPOCHS']
LR = config['LR']
WD = config['WD']
CHECKPOINT_DIR = config['CHECKPOINT_DIR']
PROJECT_NAME = config['PROJECT_NAME']
MODEL_NAME = config['MODEL_NAME']

wandb_logger = WandbLogger(
    project=PROJECT_NAME,
    name=f'{N_EPOCHS}epochs_lr{LR}_wd{WD}',
    checkpoint_name=f'{N_EPOCHS}epochs_lr{LR}_wd{WD}',
    log_model=True
)

trainer = pl.Trainer(
    logger=wandb_logger,
    log_every_n_steps=10,
    max_epochs=N_EPOCHS,
    deterministic=True,
    default_root_dir=CHECKPOINT_DIR,
    precision="16-mixed",
)

model = LitViT(
    num_labels=NUM_LABELS,
    id2label=ids2label,
    label2id=label2ids,
    class_weights=class_weights,
    lr=LR,
    weight_decay=WD,
    model_name=MODEL_NAME
)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

wandb.finish()