# Text Generation Example (using Huggingface Models)


This example shows how to use a text generation model from Huggingface
with MERA software stack. The example comprises installation of the
required library followed by compilation of the model and running
inference with MERA software stack.

## Setup

-   `mera-models` python library is required. To install use
    `pip install mera-models`

## Model export and Quantization (Optional)

To export from huggingface directly and quantize using MERA quantizer

``` bash
python quantize_model.py
```

additional flags can be specified

* `--model_id {HF_MODEL_ID}` for specific model. it will automatical export onnx into `source_model_files` folder first, then do quantization. Later, the exported onnx model folder can be used as a starting point (to save time on export) such as `source_model_files/HuggingFaceTB__SmolLM2-135M-Instruct_onnx`
* ` --qtzed_path "./qtzed_tmp"` for output folder.

## Compilation

``` bash
python deploy.py
```

additional flags can be specified

* `--model_id {HF_MODEL_ID}` this flags works the same as in `quantize_model.py`, but this time it will go directly to compilation. for INT8 precision, use the output folder from `quantize_model.py` script (default is `./qtzed_tmp`). For BF16 precision, use onnx model folder or HF_MODEL_ID directly.
*  `--target ip` (default) This will deploy the model for the target for the Hardware device. Other target can be specified using *--target* argument, for example
    using `--target Simulator`, to get simulation on CPU.
* `--out_dir` to specify output deployment folder name

# Inference

``` bash
python demo_model.py
```

additional flags can be specified
* `--model_path` points to the deployment output folder obtained from `deploy.py`
*  `--target ip` specify the target to run
*  `--model_flags` specify the config specific to model/task. In text generation's case, the flags string could be `max_new_tokens=40,template=1` which means to generate maximum 40 new tokens, and use the Chat-template. NOTE: `template=1` is for use with model finetuned for instruction-following and chat-interface only. 

# Sample result

-   Prompt:
    -   *What is the capital city of Japan?*
-   Generated text:
    -   *The capital city of Japan is Tokyo.*
