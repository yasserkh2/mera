# Text Classification models (Huggingface version)


## Setup

-  ``mera-models`` python library is required. To install use ``pip install mera-models``



## Compilation
 
To deploy from a quantized source model 

```bash
python deploy.py --model_id "./qtzed_tmp/" --target ip
```

Here `./qtzed_tmp/` is the output dir created by previous quantization step.

To compile directly from fp32 source using huggingface model id (without quantization)

```bash
python deploy.py --model_id "SamLowe/roberta-base-go_emotions" --target ip
```

or, if already exported to onnx (fp32), point to that folder

``` bash
python deploy.py --model_id "source_model_files/SamLowe__roberta-base-go_emotions_onnx" --target ip
```


## Inference

To run the mera deployed model on the card

```bash
python demo_model.py --model_path "./deploy_roberta-base-go_emotions/"  --target ip
```

To run source onnx(fp32)

``` bash
python demo_model.py --model_path  "source_model_files/SamLowe__roberta-base-go_emotions_onnx"
```
