# FashionMeter
HSE MMLS 2025 project

---
## Introduction

FashionMeter is a deep learning project created for fashion style image classification. Our model supports 11 fashion styles: `casual, formal, comfy, preppy, sporty, streetwear, grunge, boho, goth, vintage and military.`

## Repo structure

* [src](src) - all source code
  * [data](src/data) - data scraping code
    * [json_parser.ipynb](src/data/json_parser.ipynb) - downloading data from pinterest boards
  * [training](src/training) - contains code related to training the models
    * [fashion_meter.ipynb](src/training/fashion_meter.ipynb) - FashionMeter model training
* [examples](examples) - contains example notebooks for inference
* [README.md](README.md) - you are here
* [requirements.txt](requirements.txt) - library requirements that we use

## Train your own model

We are fine-tuning a pre-trained Visual Transformer from Huggingface using Lightning modules. Logging is done with W&B.

Our training is done in jupyter notebooks, so feel free to download [fashion_meter.ipynb](src/training/fashion_meter.ipynb) and try fine-tuning yourself.

## Inference

To use our model for inference, use the (inference link) notebook.
