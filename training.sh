#!/bin/bash

# Install dependencies
pip install -r req_mac.txt

# Download and extract dataset
cd src/dataload
bash download.sh
cd ../..

# Run training
python -m src.training.train