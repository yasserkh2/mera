#!/bin/bash

# Exit if a command exits with a non-zero status
set -e

# Check if yolov5 is already exported
if [ -f "source_model_files/yolov5s-448.tflite" ]; then
    printf "\nYolov5 model already exported\n"
    exit 0
fi

# Detect active virtual env
PREV_VENV=""
if [[ -n "$VIRTUAL_ENV" ]]; then
  PREV_VENV="$VIRTUAL_ENV"
  printf "Detected active virtual environment: $PREV_VENV\n"
fi

# Activate virtual env & then deactivate
if [[ -n "$PREV_VENV" ]]; then
  source "$PREV_VENV/bin/activate"
  deactivate
  printf "\nCurrent virtual environment deactivated: $PREV_VENV\n"
fi

# Check existing yolov5 dir
if [ -d "yolov5" ]; then
    printf "\nRemoving existing yolov5 directory\n"
    rm -rf yolov5
fi

# Check existing datasets dir
if [ -d "datasets" ]; then
    printf "\nRemoving existing datasets directory\n"
    rm -rf datasets
fi

# Clone yolov5 repo
printf "\nCloning yolov5 original repo\n"
git clone --recursive https://github.com/ultralytics/yolov5.git

# Check existing temp virtual env
if [ -d "yolov5env" ]; then
    printf "\nRemoving existing temp virtual environment: yolov5env\n"
    rm -rf yolov5env
fi

# Create temp virtual env & activate it
printf "\nCreating temporary virtual environment: yolov5env\n"
virtualenv -p python3 yolov5env
source yolov5env/bin/activate
printf "\nTemporary virtual environment activated: yolov5env\n"

# Checkout the specific commit and apply patch
printf "\nChecking out the specific commit and applying patch\n"
cd yolov5/
git checkout e80a09bbfa1ddb10
git apply ../demo.patch

# Install dependencies
printf "\nInstalling dependencies\n"
pip install -r requirements.txt

# Download pretrained yolov5 model
printf "\nDownloading pretrained yolov5 model\n"
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt

# Fine-tune the model
printf "\nFine-tuning the model\n"
PYTHONWARNINGS="ignore" python train.py --img 448 --batch 16 --epochs 4 --data coco128.yaml --weights ./yolov5s.pt

# Export fine-tuned model to tflite
printf "\nExporting fine-tuned model to tflite\n"
PYTHONWARNINGS="ignore" python export.py --weights runs/train/exp/weights/best.pt --img 448 --int8 --batch 1 --include tflite --data data/coco128.yaml

# Copy tflite model file to the source_model_files directory
printf "\nCopying tflite model file to the source_model_files directory\n"
cp runs/train/exp/weights/best-int8.tflite ../source_model_files/yolov5s-448.tflite
printf "\nFile saved to: ./source_model_files/yolov5s-448.tflite\n"

# Deactivate temp virtual env
deactivate
printf "\nTemporary virtual environment deactivated: yolov5env\n"

# Activate original env
if [[ -n "$PREV_VENV" ]]; then
  source "$PREV_VENV/bin/activate"
  printf "\nOriginal virtual environment activated: $PREV_VENV\n"
fi

# Change to the original directory
cd ..
