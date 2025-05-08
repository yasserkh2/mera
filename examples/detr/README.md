
# DETR Object Detection Demo

An Image Object Detection model, with transformer architecture, trained with COCO dataset. 

## Get original model file and export to onnx 

To generate the origina model file and export to onnx:
```bash
python export_detr.py
```

## Model compilation and deployment with MERA software stack

To compile the model for with target ip for the card:

```bash
python deploy.py --target ip
```

## Inference

To run the model on the source file exported to onnx:

```bash
python demo_model.py --model_path ./source_model_files/detr_600x400_nodict.onnx
```

To run on the compiled model (default target ip):

```bash
python demo_model.py --model_path ./deploy_detr --target ip
```
