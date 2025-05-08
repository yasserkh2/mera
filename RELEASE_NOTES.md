# RELEASE NOTES

## Release version 2.3.0
 This document contains details pertaining to the MERA Software release package. 
 
 * Supported HW and SW
   * OS:  Linux Ubuntu 22.04 LTS
   * HW: Sakura II PCIe Single or Dual cards and Sakura II M.2 M-Key modules
 * Model Frameworks:  Pytorch, TFLite, ONNX
 * Supported System for Compiling: x86 PC with 64G RAM
 * Supported Runtime target platforms: x86 with PCIe Gen 3 support

#### Note  

To use MERA for compiling and quantizing models, x86 PC is required, with a min of 32GB of RAM. It is recommended to use an x86 PC with min 64GB of RAM, so both small and large models (including LLMs) can be compiled and quantized; 

Using the MERA compiler, supported runtime deployment platform files can be generated for x86 systems.

For earlier OS version 20.04 a separate setup package maybe needed, please contact Edgecortix sales for further details.

 
### Release Highlights
 
- New Quantizer with greatly expanded capability and improvements
- Expanded model list in examples
- Frontend support
  - Yolov8 full capture, with CPU optimizations
  - Added features for front-ends and exporters (new patterns, transformation passes, conversions...)
- Backend support
  - Various bug fixes and improvements
  - new alias of scheduling mode ('Simple' for quick deployment and 'Performance' for optimized deployment)

### Bug fixes and Performance improvements

- multi-batch bug fixes of certain models
- Segformer latency improvement
- Yolov8 CPU Ops optimization
- mera-quantized convolution models latency improvement

#### MERA Quantizer

- New Quantizer with various improvements
  - Greatly improved quantization and evaluation time & memory consumption.
  - Better reported information, such as node precision breakdown in tabular format.
  - Fixed various bugs related to quantizer
  - Better accuracy on LLM transformer models.
  - Reduced latency overheads of some CNN models.
- Note: API changes, examples code updated.

#### Mera Visualizer

- Added support for new subgraph json format, with new information such as input/output shapes.

#### Installation scripts
- Better fallback handling, not closing terminal when script is stopped.

#### Examples
- Yolov8, updated export and inference script to reflect new model capture.
- Updated new args and alias to be in sync with mera software.
- Added main model table
- Improved ease of use for measure.py

### Limitations

- New Quantizer currently only supports onnx source model files, will fallback to legacy Quantizer for tflite and pytorch.
- MERA-Quantized model can still be further optimized for better latency, for future releases.

---

## Release version 2.2.0
 This document contains details pertaining to the MERA Software release package. 
 
 * Supported HW and SW
   * OS:  Linux Ubuntu 22.04 LTS or 20.04 LTS
   * HW: Sakura II PCIe Single or Dual cards and Sakura II M.2 M-Key modules
 * Model Frameworks:  Pytorch, TFLite, ONNX
 * Supported System for Compiling: x86 PC with 64G RAM
 * Supported Runtime target platforms: x86 with PCIe Gen 3 support

#### Note  

To use MERA for compiling and quantizing models, x86 PC is required, with a min of 32GB of RAM. It is recommended to use an x86 PC with min 64GB of RAM, so both small and large models (including LLMs) can be compiled and quantized; 

Using the MERA compiler, supported runtime deployment platform files can be generated for x86 systems.
 
### Release Highlights
 
- Leverages new MERA 2 heterogenous compilation framework
- SAKURA-II Dynamic Neural Accelerator and Reshaper support
- Includes BF16 and Int8 support via Mera Quantizer
- Transformer (including LLMs) and CNN support
- Mera Visualizer Support
- Support for Huggingface Models
- Jupyter Notebook

#### Mera Visualizer

- Enables users to see compiled model operation in graphical format
- Indicates operations/functions supported/run in Sakura II (in Green color) or moved to CPU for unsupported functions

#### MERA Quantizer

- Basic support for Int8 quantization of CNN and selected Transformer models

#### Model Frameworks Supported

##### PyTorch

- Legacy MERA 1 fallback for Int8 quantized CNN models

##### ONNX

- Support of BF16 and Int8 Transformer models
- Int8 requires usage of MERA quantizer
- Recommended format for BF16 (CNN and Transformer) deployments
- Recommended format for Transformer deployments

##### TFLite

- Support of BF16 and Int8 CNN models
- Int8 support for pre-quantized TFLite models
- Int8 support for MERA quantized models (fp32 source)
- BF16 support for fp32 TFLite models
- Recommended format for pre-quantized Int8 CNN deployments

#### Not Supported

- Dynamic Input Shapes
  - Input such as ONNX or PyTorch file can have dynamic input shape, but specific dimensions have to be supplied to compiler and the resulting compiled model will be limited to those input dimensions.
- Pre Quantized ONNX Models

#### Limitations

- Further improvements in LLM accuracy expected in next release
- For some CNNs latency is known to exceed similar TFLite deployment due to activations being computed in BF16

#### Legacy Support

- SAKURA-I support (through MERA 1 fallback for Int8 quantized CNN models)

#### Examples

- Sample demo examples are included in the package under the Examples folder that include, but not limited to, DETR, RESNET50, YOLOv5, YOLOv8, Huggingface models, Efficient Net
