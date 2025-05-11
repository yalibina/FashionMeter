import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from transformers import ViTImageProcessor
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import yaml
import os

# Load config
with open(os.path.join(os.path.dirname(__file__), '../../config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

DATA_ROOT = config['DATA_ROOT']
BATCH_SIZE = config['BATCH_SIZE']
NUM_WORKERS = config['NUM_WORKERS']
MODEL_NAME = config['MODEL_NAME']
TRAIN_VAL_SPLIT = config['TRAIN_VAL_SPLIT']

dataset = ImageFolder(root=DATA_ROOT)
labels = dataset.classes
NUM_LABELS = len(labels)
ids2label = {id: label for id, label in enumerate(labels)}
label2ids = {label: id for id, label in ids2label.items()}

train_size = int(TRAIN_VAL_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_subset, val_subset = random_split(dataset, [train_size, val_size])

train_labels = [dataset.targets[i] for i in train_subset.indices]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

class WrapDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return {'image': image, 'label': label}

train_dataset = WrapDataset(train_subset)
val_dataset = WrapDataset(val_subset)

feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)

def collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    inputs = feature_extractor(images=images, return_tensors='pt')
    inputs['labels'] = torch.tensor(labels)
    return inputs

train_dataloader = DataLoader(
    train_dataset,
    num_workers=NUM_WORKERS,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=BATCH_SIZE,
    pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    batch_size=BATCH_SIZE,
    pin_memory=True
)
# test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)