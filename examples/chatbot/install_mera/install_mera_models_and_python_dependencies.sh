#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# mera models
pip install mera_models*.whl
pip install onnx onnxruntime

# evaluate
pip install scikit-image scikit-learn cython evaluate tiktoken datasets jinja2

echo "-- MERA-MODELS library successfuly installed."
