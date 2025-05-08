#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Populate environment with MERA software stack and its dependencies
pip install --upgrade pip
pip install numpy==1.24.3
pip install tqdm easydict wget notebook pandas matplotlib opencv-python gdown seaborn motpy
pip install tabulate

# for mera deps
pip install onednn-cpu-gomp==2022.0.2
pip install tensorflow==2.13.1
pip install tflite==2.10.0
pip install torch==1.12.1
pip install torchvision==0.13.1
pip install protobuf==4.25.5

# actual mera install
pip install mera-*.whl mera_tvm_full*.whl mera_tvm_internal*.whl mera2_runtime*.whl

echo "-- MERA successfuly installed."
echo "-- NOTE: Please ignore the typing-extensions conflict error message, it is OK and expected."
