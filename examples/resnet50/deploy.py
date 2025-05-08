# -*- coding: utf-8 -*-
"""
1. Compiling Resnet50
=====================

In this tutorial we will compile the ResNet50 model using MERA python API and run on an IP/Sakura device in full INT8 precision. 
Sakura-2 can run both on INT8 and BF16 precision. In this tutorial we will focus on INT8. The standard ResNet50 has float-32 weight and activations. 
So we will use PyTorch’s provided quantized version of ResNet50/ResNet18 and then quantize using Post-Training Static Quantization(PTQ) technique.


In order to PTQ, we need some changes in the source code. In the ``torchvision.models.quantization.resnet.QuantizableResNet``, 
PyTorch already did necessary modifications like inserting Quant/DeQuant stubs, fusing Conv2d+ReLU+BatchNorm2d in order to make the model quantization friendly.
Please refer to the source code for more details about QuantizableResNet.
"""


###############################################################################
# Step-1: Import the model and Quantize it
# ----------------------------------------
#
# The QuantizableResNet behaves the same as normal ResNet (torchvision.models.resnet50) except it has some extra customization to make it quantization friendly.
# So the QuantizableResNet can load a pre-trained ResNet50_Weights.DEFAULT float-32 weight file.
# After loading this way, this can be trained or fine-tuned the model as normal ResNet50.


import argparse
from pathlib import Path
import json
import torch
from torchvision.models.quantization import resnet as qresnet
from torchvision.models import ResNet50_Weights
import mera
from mera import Target, Platform
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import numpy as np
import cv2
from mera_models.utils import get_target, get_device_platform


###############################################################################
# Assuming the fine-tuning or re-training has been finished.
# Now we need to convert the float-32 weight into int8.
# For this, we will use pytorch eager mode quantization api.
# We also need some representative images for calibration.
# The purpose of calibration is to create a mapping from
# float-32 to int8 that minimizes the information loss
# between original float-32 inference and quantized int8 inference.


def get_quantized_resnet50(args):
    model = qresnet.resnet50(weights=ResNet50_Weights.DEFAULT)
    # train or fine tune if needed

    # Bring the model into inference mode
    model.eval()

    # Fuse BN layer and activations into preceding layers where possible
    model.fuse_model()

    # Set quantization config for Host/Server CPU (x86)
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    # Insert observers
    torch.quantization.prepare(model, inplace=True)

    # Calibrate the model and collect statistics
    # Using all .JPEG images found inside args.calib_dir
    calib_img_set = list(Path(args.calib_dir).rglob("*.JPEG"))
    sample_input = torch.rand(1, 3, 224, 224)
    if len(calib_img_set) == 0:
        print("WARNING: empty calibration set, use random input instead")
        model(sample_input)
    else:
        for img_path in tqdm(calib_img_set, desc="Calibrating"):
            img = cv2.imread(str(img_path))
            img = preprocess_input(img, as_numpy=False)
            model(img)

    # Convert to quantized version
    torch.quantization.convert(model, inplace=True)

    ## Save as torchscript:
    quant_model_out = "resnet50_int8.torchscript.pt"
    with torch.no_grad():
        script_module = torch.jit.trace(model, sample_input).eval()
        torch.jit.save(script_module, quant_model_out)
    print(f"Quantized model saved at {quant_model_out}\n")
    return quant_model_out


###############################################################################
# Here the calib_img_set is a list of calibration image paths.
# Normally 200 to 500 images from train sets are used as calibration images.
# The main thing we should consider is, the calibration image should represent the actual test time images.
# For this reason, we may also use images from the test set.


###############################################################################
# The preprocess_input method should have same pre-processing pipeline that will be used during the test time.
# For example, in imagenet dataset, the images are resized to resize_size=[232] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[224].
# Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
# All those should be used during calibration.


def preprocess_input(image, channel_order="NCHW", as_numpy=True):
    # assuming the image was read using cv2.imread(input_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = F.resize(img, 256, InterpolationMode.BILINEAR)
    img = F.center_crop(img, 224)
    img = F.to_tensor(img)
    img = F.normalize(
        img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
    )
    # The pytorch or onnx framework use channel first (NCHW) input
    # Sakura accelerator uses channel last (NHWC) input. So change the cannel order
    if channel_order == "NHWC":
        img = img.permute(1, 2, 0)  # NHWC
    img = img.unsqueeze(0)  # make it batch-1 by adding a dimension
    if as_numpy:
        img = np.asarray(img, dtype="float32")

    return img


###############################################################################
# Step-2: Compile with MERA
# -------------------------
#
# We are ready to compile the quantized model that we saved as resnet50_int8.torchscript.pt.
# We will use the ModelLoader.from_pytorch api from the mera package.
# Depending on the source file(int8 quantized model) type,
# the available options are ModelLoader.from_tflite, ModelLoader.from_onnx and ModelLoader.from_quantized_mera.


def compile_mera(model_path, out_dir, target, mera_platform, input_desc, build_cfg_dt, host_arch):
    with mera.Deployer(out_dir, overwrite=True) as deployer:
        model = mera.ModelLoader(deployer).from_pytorch(model_path, input_desc)

        # compilation
        deployer.deploy(
            model,
            mera_platform=mera_platform,
            target=target,
            host_arch=host_arch,
            build_config=build_cfg_dt,
        )
    print(f"\nCompiled model saved at: {out_dir}")
    return out_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="deploy_resnet50_sakura2/")
    parser.add_argument("--calib_dir", default="data/calibration_images")
    parser.add_argument("--target", default="ip")
    parser.add_argument(
        "--device",
        default="sakura2",
        type=str.lower,
    )
    parser.add_argument(
        "--build_cfg",
        default="{'scheduler_config': {'mode': 'Simple'}}",
        type=str,
    )
    parser.add_argument(
        "--host_arch",
        default="x86",
        type=str.lower,
    )
    return parser.parse_args()

def main(args):
    ## Step-1: Import the model and Quantize it
    quant_model_path = get_quantized_resnet50(args)

    ## Step-2: Compile with MERA
    mera_platform = get_device_platform(args.device)
    target = get_target(args.target)
    build_cfg_dt = json.loads(str(args.build_cfg).replace("'", '"'))
    compile_mera(
        model_path=quant_model_path,
        out_dir=args.out_dir,
        target=target,
        mera_platform=mera_platform,
        input_desc={"input0": ((1, 224, 224, 3), "float32")},
        build_cfg_dt=build_cfg_dt,
        host_arch=args.host_arch,
    )

if __name__ == "__main__":
    main(get_args())



###############################################################################
# The compiled output binary will be saved at ``Sakura2_resnet50/`` directory (mentioned in out_dir argument).
#
# The ``target=Target.IP`` will create a binary inside Sakura2_resnet50/build/IP
# which contains the instructions to run the execution on the Sakura card.
