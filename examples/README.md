# Examples main directory

This directory contains examples of different models demonstrating how to use MERA to deploy them on Sakura-2.

Please explore these folders for more detailed instructions.

## Important Considerations

#### Quantization 
For quantization calibration data, we prioritize speed and ease of understanding in these examples to demonstrate how our software works. As a result, we use small and generic calibration datasets. For higher accuracy, however, each model should utilize a tailored calibration dataset.

#### Scheduling Mode in Compilation

By default, the Scheduling mode is set to Simple, prioritizing quick compilation times. To optimize latency, you can switch the Scheduling mode to Performance. Please see an example in [resnet50 folder](resnet50) for detailed usage.

## Model Table

The table below lists example models, along with the corresponding code for exporting, quantizing, deploying, and performing inference.

### Note (*)

The target precision (INT8/BF16) support also depends on the source precision. For models that starts in INT8, the deployment will likewise remain in INT8 precision.


| **folder**                       | **model name**                                                                                                                                  | **params (M)** | **source precision** | **source type** | **INT8 support\*** | **BF16 support\*** |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | -------------------- | --------------- | ------------------ | ------------------ |
| detr                             | detr                                                                                                                                            | 41             | FP32                 | onnx            |                    | Y                  |
| efficient_net_demo               | efficientnet-lite4                                                                                                                              | 13             | INT8                 | tflite          | Y                  |                    |
| huggingface_image_classification | [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)                                                                               | 23             | FP32                 | onnx            | Y                  | Y                  |
| huggingface_image_classification | [vit-small-patch16-224](https://huggingface.co/WinKawaks/vit-small-patch16-224)                                                                 | 22             | FP32                 | onnx            | Y                  | Y                  |
| huggingface_image_classification | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)                                                               | 86             | FP32                 | onnx            | Y                  | Y                  |
| huggingface_image_classification | [google/vit-large-patch16-224](https://huggingface.co/google/vit-large-patch16-224) (New!)                                                      | 307            | FP32                 | onnx            | Y                  | Y                  |
| huggingface_image_segmentation   | [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)                                   | 4              | FP32                 | onnx            |                    | Y                  |
| huggingface_text_classification  | [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | 67             | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_classification  | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)                                                     | 125            | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_classification  | [lvwerra/distilbert-imdb](https://huggingface.co/lvwerra/distilbert-imdb) (New!)                                                                | 67             | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)                                                                           | 124            | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [openai-community/gpt2-medium](https://huggingface.co/openai-community/gpt2-medium)                                                             | 345            | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [openai-community/gpt2-large](https://huggingface.co/openai-community/gpt2-large)                                                               | 774            | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [openai-community/gpt2-xl](https://huggingface.co/openai-community/gpt2-xl)                                                                     | 1,500          | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [facebook/opt-125m](https://huggingface.co/facebook/opt-125m)                                                                                   | 125            | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)                                                                                   | 1,300          | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b) (New!)                                                                            | 6,700          | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [openlm-research/open_llama_7b_v2](https://huggingface.co/openlm-research/open_llama_7b_v2) (New!)                                              | 7,000          | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)                                                                     | 7,000          | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) (New!)                                                    | 7,000          | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (New!)                                                          | 8,000          | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (New!)                                        | 135            | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) (New!)                                        | 362            | FP32                 | onnx            | Y                  | Y                  |
| huggingface_text_gen             | [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) (New!)                                        | 1,710          | FP32                 | onnx            | Y                  | Y                  |
| resnet50                         | resnet50                                                                                                                                        | 23             | INT8                 | torchscript     | Y                  |                    |
| yolov5                           | yolov5s                                                                                                                                         | 7              | INT8                 | tflite          | Y                  |                    |
| yolov8                           | yolov8s (New!)                                                                                                                                  | 11             | FP32                 | onnx            | Y                  | Y                  |
| yolov8                           | yolov8m                                                                                                                                         | 26             | FP32                 | onnx            | Y                  | Y                  |

