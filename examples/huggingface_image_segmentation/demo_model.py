# -*- coding: utf-8 -*-
"""
2. Running Image Segmentation
=============================


"""

from os.path import isfile, join
import argparse
import warnings

from PIL import Image
import numpy as np

import torch
from torch import nn

warnings.filterwarnings("ignore")


def ade_palette():
    """ADE20K palette that maps each class to RGB values"""

    return [
        [204, 87, 92],
        [112, 185, 212],
        [45, 189, 106],
        [234, 123, 67],
        [78, 56, 123],
        [210, 32, 89],
        [90, 180, 56],
        [155, 102, 200],
        [33, 147, 176],
        [255, 183, 76],
        [67, 123, 89],
        [190, 60, 45],
        [134, 112, 200],
        [56, 45, 189],
        [200, 56, 123],
        [87, 92, 204],
        [120, 56, 123],
        [45, 78, 123],
        [156, 200, 56],
        [32, 90, 210],
        [56, 123, 67],
        [180, 56, 123],
        [123, 67, 45],
        [45, 134, 200],
        [67, 56, 123],
        [78, 123, 67],
        [32, 210, 90],
        [45, 56, 189],
        [123, 56, 123],
        [56, 156, 200],
        [189, 56, 45],
        [112, 200, 56],
        [56, 123, 45],
        [200, 32, 90],
        [123, 45, 78],
        [200, 156, 56],
        [45, 67, 123],
        [56, 45, 78],
        [45, 56, 123],
        [123, 67, 56],
        [56, 78, 123],
        [210, 90, 32],
        [123, 56, 189],
        [45, 200, 134],
        [67, 123, 56],
        [123, 45, 67],
        [90, 32, 210],
        [200, 45, 78],
        [32, 210, 90],
        [45, 123, 67],
        [165, 42, 87],
        [72, 145, 167],
        [15, 158, 75],
        [209, 89, 40],
        [32, 21, 121],
        [184, 20, 100],
        [56, 135, 15],
        [128, 92, 176],
        [1, 119, 140],
        [220, 151, 43],
        [41, 97, 72],
        [148, 38, 27],
        [107, 86, 176],
        [21, 26, 136],
        [174, 27, 90],
        [91, 96, 204],
        [108, 50, 107],
        [27, 45, 136],
        [168, 200, 52],
        [7, 102, 27],
        [42, 93, 56],
        [140, 52, 112],
        [92, 107, 168],
        [17, 118, 176],
        [59, 50, 174],
        [206, 40, 143],
        [44, 19, 142],
        [23, 168, 75],
        [54, 57, 189],
        [144, 21, 15],
        [15, 176, 35],
        [107, 19, 79],
        [204, 52, 114],
        [48, 173, 83],
        [11, 120, 53],
        [206, 104, 28],
        [20, 31, 153],
        [27, 21, 93],
        [11, 206, 138],
        [112, 30, 83],
        [68, 91, 152],
        [153, 13, 43],
        [25, 114, 54],
        [92, 27, 150],
        [108, 42, 59],
        [194, 77, 5],
        [145, 48, 83],
        [7, 113, 19],
        [25, 92, 113],
        [60, 168, 79],
        [78, 33, 120],
        [89, 176, 205],
        [27, 200, 94],
        [210, 67, 23],
        [123, 89, 189],
        [225, 56, 112],
        [75, 156, 45],
        [172, 104, 200],
        [15, 170, 197],
        [240, 133, 65],
        [89, 156, 112],
        [214, 88, 57],
        [156, 134, 200],
        [78, 57, 189],
        [200, 78, 123],
        [106, 120, 210],
        [145, 56, 112],
        [89, 120, 189],
        [185, 206, 56],
        [47, 99, 28],
        [112, 189, 78],
        [200, 112, 89],
        [89, 145, 112],
        [78, 106, 189],
        [112, 78, 189],
        [156, 112, 78],
        [28, 210, 99],
        [78, 89, 189],
        [189, 78, 57],
        [112, 200, 78],
        [189, 47, 78],
        [205, 112, 57],
        [78, 145, 57],
        [200, 78, 112],
        [99, 89, 145],
        [200, 156, 78],
        [57, 78, 145],
        [78, 57, 99],
        [57, 78, 145],
        [145, 112, 78],
        [78, 89, 145],
        [210, 99, 28],
        [145, 78, 189],
        [57, 200, 136],
        [89, 156, 78],
        [145, 78, 99],
        [99, 28, 210],
        [189, 78, 47],
        [28, 210, 99],
        [78, 145, 57],
    ]


def postprocessing(img, logits):
    """Apply post processing to logits & overlay on image"""

    # Resize logits to match image size
    logits = nn.functional.interpolate(
        logits,
        size=img.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    # Apply argmax on the class dimension
    seg = logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(img) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = Image.fromarray(img)

    return img


def load_predictor(model_path, target_str, device_str):
    """Return predictor based on either onnxruntime or meraruntime"""

    check_file = join(model_path, "model.onnx")
    if isfile(check_file):
        # Actual onnx api class
        from optimum.onnxruntime import ORTModelForSemanticSegmentation as mf

        print(f"Running inference on huggingface ONNX runtime...")
        predictor = mf.from_pretrained(
            model_path,
            use_cache=False,
            use_io_binding=False,
        )
    else:
        # MERA load api class
        from mera_models.hf.meraruntime import MERAModelForSemanticSegmentation as mf

        print(mf)
        predictor = mf.from_pretrained(
            model_path,
            use_cache=False,
            target=target_str,
            device_target=device_str,
            measure_latency=True,
        )

    return predictor


def run_inference(inputs, predictor, model_flags, **kwargs):
    """Run inference on input image using the predictor"""

    feat_ext = predictor.preprocessors[0]
    pixel_values = feat_ext(inputs, return_tensors="pt").to("cpu")
    outputs = predictor(**pixel_values)
    logits_cpu = outputs.logits
    overlay = postprocessing(inputs, logits_cpu)

    return overlay


def main(arg):
    """Main function to handle model demo"""

    seed = 1337  # to generate reproducible results
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    predictor = load_predictor(arg.model_path, arg.target, arg.device)

    image_data = Image.open(arg.input_path)

    # Run actual inference
    overlay = run_inference(
        image_data,
        predictor,
        arg.model_flags,
    )

    overlay.save(arg.save_path)
    print(f"Result saved at: {arg.save_path}")

    # Measure estimated latency
    if arg.target.lower() == "simulatorbf16":
        print(f" ** Estimated SimulatorBf16 latency {predictor.estimated_latency} ms")


def get_args():
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="data/scene_parse_150_image-18.jpeg",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./result.png",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="./deploy_segformer-b0-512-512",
        type=str,
    )
    parser.add_argument(
        "--model_flags",
        default="a=0",
        type=str,
        help="Please see the inference function for meaning of each flag.",
    )
    parser.add_argument(
        "--target",
        default="ip",
        type=str,
        help="MERA Target environment, ip is default",
    )
    parser.add_argument(
        "--device",
        default="sakura2",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
