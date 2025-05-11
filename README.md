# FashionMeter
HSE MMLS 2025 project

---
## Introduction

FashionMeter is a deep learning project created for fashion style image classification. Our model supports 11 fashion styles: `casual, formal, comfy, preppy, sporty, streetwear, grunge, boho, goth, vintage and military.`

## Repo structure

* [config.yaml](config.yaml) - main configuration file
* [requirements.txt](requirements.txt) - Python dependencies
* [training.sh](training.sh) - training script
* [src](src) - all source code
  * [dataload](src/dataload) - data loading utilities
    * [download.sh](src/dataload/download.sh) - data download script
    * [dataset.py](src/dataload/dataset.py) - init of dataloaders
    * [json_parser.ipynb](src/dataload/json_parser.ipynb) - Pinterest data parsing
  * [example](src/example) - example notebooks
    * [fashionmeter_train.ipynb](src/example/fashionmeter_train.ipynb) - old training example
    * [inference.ipynb](src/example/inference.ipynb) - inference example
  * [models](src/models) - model definitions
    * [vit.py](src/models/vit.py) -  model
  * [training](src/training) - training scripts and notebooks
    * [train.py](src/training/train.py) - main training script
    * [train.ipynb](src/training/train.ipynb) - training notebook
* [examples](examples) - additional example files
  * [example.txt](examples/example.txt)
* [README.md](README.md) - you are here

## Train your own model

We are fine-tuning a pre-trained Visual Transformer from Huggingface using Lightning modules. Logging is done with W&B.

Our training is done in jupyter notebooks, so feel free to download [fashion_meter.ipynb](src/training/train.ipynb) and try fine-tuning yourself.

## Inference

To use our model for inference on your own photos, use the [TBD] notebook.
