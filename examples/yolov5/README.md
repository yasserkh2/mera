# YoloV5 Object Detection Demo

This demo shows how to modify a model from a third party repository and deploy it with MERA software stack.
The demo comprises two set of activities:
  * Generating the source model file from the official repository of YoloV5s
  * Model compilation and inference with MERA software stack


## Generating source model file

The code to generate the source model file is provided below. This will:
  * Clone the official YoloV5 repository
  * Create and change to a temporary virtual environment
  * Install the required dependencies (takes few minutes)
  * Apply the demo patch
  * Download the pretrained Yolov5 model
  * Perform a short fine-tuning to replace SiLU with Hardswish (takes few minutes)
  * Export the model to TFLite file
  * Copy the TFLite model file to source_model_files directory

```bash
./get_yolov5_source_model.sh
```


## Model compilation and inference with MERA software stack

### Compilation:

  * The code to compile the model is as follows.
```bash
python deploy.py --target ip
```

### Inference:

To run the model on the card with target `ip`

```bash
python demo_model.py --target ip
```

To run the model on `simulator`

```bash
python demo_model.py --target simulator
```

## Result
<img src="./data/input_output.png" alt="Sample input and output" width="1500"/>
