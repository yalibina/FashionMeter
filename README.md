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
  * [models](src/models) - model definitions
    * [vit.py](src/models/vit.py) -  ViT base and quantized models
    * [mobilenet.py](src/models/mobilenet.py) - distillation model
  * [training](src/training)
    * [train.py](src/training/train.py) - main training script
* [examples](examples) - example notebooks
  * [example_distil.ipynb](examples/example_distil.ipynb) - knowledge distillation example
  * [example_inference.ipynb](examples/example_inference.ipynb) - inference on single example notebook
  * [example_training.ipynb](examples/example_training.ipynb) - base model fine-tuning example
  * [pruned_inference.ipynb](examples/pruned_inference.ipynb) - evaluating pruned model performance
  * [quantized_inference.ipynb](examples/quantized_inference.ipynb) - evaluating quantized model performance
* [README.md](README.md) - you are here

## Train your own model

We are fine-tuning a pre-trained Visual Transformer from Huggingface using Lightning modules. Logging is done with W&B.

Try fine-tuning your own model with [example_training.ipynb](examples/example_training.ipynb) notebook.

## Inference

To use our model for inference on your own photos, use the [example_inference.ipynb](examples/example_inference.ipynb) notebook.
