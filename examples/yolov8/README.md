
## Get the model

First, we need to download the YOLOv8m model from the Ultralytics repository and export it

This following scripts will do that
```
./export_onnx.sh
```

Console log: 
```
Deactivated existing virtual environment.
Using new virtual environment...
...
...
Success!
Modified model saved at 'source_model_files/yolov8m_hardswish_finetuned_640x640.onnx'.
```

## Quantize the model

In order to quantize the model 

```
python quantize_model.py \
  --model_path="./source_model_files/yolov8m_hardswish_finetuned_640x640.onnx" \
  --coco_dir="./data/calib_images" \
  --calib_imgs_num=20

```

For calibration, we recommend using this 500 coco images(train set) from mlperf calibration list [coco_cal_images_list.txt](https://github.com/mlcommons/inference/blob/master/calibration/COCO/coco_cal_images_list.txt)


Console log:
```
...
Done!
Quantized model saved at model_qtz.mera.
```


## Deploy 

```
python deploy.py \
  --model_path="./source_model_quantized/model_qtz.mera/model.mera" \
  --out_dir="./deploy_yolov8m" \
  --target="ip"

```


## Inference

```
python demo_model.py \
  --input_path="./data/input_images/bus.jpg" \
  --model_path="./deploy_yolov8m" \
  --target="ip"

```

