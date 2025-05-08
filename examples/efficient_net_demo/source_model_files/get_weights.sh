#!/usr/bin/env sh

wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite1.tar.gz
tar xvzf efficientnet-lite1.tar.gz
rm efficientnet-lite1.tar.gz
cp efficientnet-lite1/efficientnet-lite1-int8.tflite effnet-lite1-int8.tflite
cp efficientnet-lite1/efficientnet-lite1-fp32.tflite effnet-lite1-fp32.tflite
rm -rf efficientnet-lite1/

wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite4.tar.gz
tar xvzf efficientnet-lite4.tar.gz
rm efficientnet-lite4.tar.gz
cp efficientnet-lite4/efficientnet-lite4-int8.tflite effnet-lite4-int8.tflite
cp efficientnet-lite4/efficientnet-lite4-fp32.tflite effnet-lite4-fp32.tflite
rm -rf efficientnet-lite4/
