#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# mera models
pip install mera_models*.whl
pip install onnx onnxruntime

# evaluate
pip install scikit-image scikit-learn cython evaluate tiktoken datasets

# chat-template
pip install Jinja2==3.1.4

echo "-- MERA-MODELS library successfuly installed."
