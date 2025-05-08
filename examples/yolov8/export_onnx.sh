#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

cd source_model_files

if [ -f "yolov8m_hardswish_finetuned_640x640.onnx" ]; then
    printf "\nYolov8 onnx model already exported, stopping process.\n"
    exit 0
fi

# Detect active virtual env
PREV_VENV=""
if [[ -v VIRTUAL_ENV && -n "$VIRTUAL_ENV" ]]; then
    PREV_VENV="$VIRTUAL_ENV"
    printf "Detected active virtual environment: $PREV_VENV\n"
fi

# Activate virtual env & then deactivate
if [[ -n "$PREV_VENV" ]]; then
  source "$PREV_VENV/bin/activate"
  deactivate
  printf "\nCurrent virtual environment deactivated: $PREV_VENV\n"
fi

python3 -m venv venv
source venv/bin/activate
echo "Using new virtual environment..."
which python

# -- Install dependencies
pip install --upgrade pip
pip install numpy==1.24.3
pip install onnxoptimizer==0.3.13
pip install onnxruntime
pip install onnxsim
pip install onnxslim

# -- git clone ultralytics
# rm old files
rm -rfd ultralytics

# get version
git clone --recursive https://github.com/ultralytics/ultralytics.git && cd ultralytics && git checkout v8.3.68

# replace SiLU with hard swish --- apply patch BEFORE pip-installing/using the ultralytics repo, easy to forget
git apply ../mera_support.patch

# Install Ultralytics
pip install -e .

pwd

# -- Finetune and Export onnx model
MODEL_DIR=./runs/detect/train/weights
function export_func {
  rm -rfd ./runs
  yolo train data=coco128.yaml model=$1.pt epochs=$4 imgsz=640
  yolo export model=$MODEL_DIR/best.pt format=onnx imgsz=$2,$3
  cp -v $MODEL_DIR/best.onnx ../$1_hardswish_finetuned_$2x$3.onnx
}

export_func "yolov8s" "640" "640" "1"
export_func "yolov8m" "640" "640" "1"

# back to outside
cd ../
pwd

deactivate
# rm -rf venv

if [[ -n "$PREV_VENV" ]]; then
  source "$PREV_VENV/bin/activate"
  printf "\nOriginal virtual environment activated: $PREV_VENV\n"
fi

printf "\n\e[1;32m%s\e[0m\n\n" "Export ONNX complete"
