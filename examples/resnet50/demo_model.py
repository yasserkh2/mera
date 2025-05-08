# -*- coding: utf-8 -*-
"""
2. Running Resnet50
===================


Running in Sakura Device
------------------------

In order to isolate the inference from the compilation code, let’s create a new file as demo_model.py.
"""

import argparse
from pathlib import Path
import mera
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import cv2
import numpy as np
import yaml

from mera_models.utils import get_target, get_device_target

# A dictionary that contains Imagenet class id to label mapping
IMAGENET_LABELS = yaml.safe_load(Path("data/imagenet_class_labels.yaml").read_text())


###############################################################################
# First, load the compiled model that we generated in previous step.
#


def load_model(args):
    # Decide where to execute. Run on IP or emulate on CPU?
    target = get_target(args.target)
    # Get the device.
    # NOTE: Due running in int8 precision, by default we will use sakura2
    device_target = get_device_target(args.device)
    # Load MERA model from pre-compiled binary
    dep_obj = mera.load_mera_deployment(args.src_model, target=target)
    # Get the runner(model) object that will do the prediction
    predictor = dep_obj.get_runner(device_target=device_target)
    return predictor


###############################################################################
# Before feeding the images, we need some pre-processing (see official doc for details)


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
    # Make it batch-1 by adding a dimension at the beginning
    img = img.unsqueeze(0)
    # If needed, return as numpy array instead of torch tensor
    if as_numpy:
        img = np.asarray(img, dtype="float32")

    return img


###############################################################################
# In order to visualize the output, let's create a overlay function


def get_top_k(arr: np.ndarray, k=5):
    idx = np.argpartition(arr, -k)[-k:]  # Indices not sorted
    idx = idx[np.argsort(arr[idx])][
        ::-1
    ]  # Indices sorted by value from largest to smallest

    return idx


def make_visual_output(pred, args, k=3):
    # Create a output dir if not exist
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_img = cv2.imread(args.input_image)
    h, w, c = src_img.shape
    # create a blank white image which have double the width of src_image
    viz_img = np.ones((h, w * 2, c), dtype=np.int8) * 255
    viz_img[0:h, 0:w] = src_img

    # keep top k (i.e., 3) predictions
    top_k = 3
    pred = pred[0][0, :]  # 1000 class pred
    ranked = get_top_k(pred, top_k)
    names = [IMAGENET_LABELS[p] for p in ranked]
    probs = [pred[p] for p in ranked]

    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_x, txt_y = w + 50, 50
    fontScale = 0.6
    color = (255, 0, 0)
    thickness = 1

    cv2.putText(
        viz_img,
        f"Top {k} Predictions:",
        (txt_x - 30, 20),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    for i, (name, prob) in enumerate(zip(names, probs)):
        txt_y += 20
        txt = f"{i+1}. {name} - {prob:.2f}%"
        print(txt)
        cv2.putText(
            viz_img, txt, (txt_x, txt_y), font, fontScale, color, thickness, cv2.LINE_AA
        )
    out_filename = out_dir / f"result.png"
    cv2.imwrite(str(out_filename), viz_img)
    print(f"Output Image Saved at {out_filename}")


###############################################################################
# Now we can run
def main(args):
    ## --- Step-1: Load the pre-compiled model
    predictor = load_model(args)

    ## --- Step-2: Run inference
    # Read image file
    img = cv2.imread(args.input_image)
    # Apply necessary pre-processing
    img = preprocess_input(img, channel_order="NHWC")
    # inference
    runner = predictor.set_input(img).run()
    pred = runner.get_outputs()

    # Show the output
    # top_cls = np.argmax(pred[0].flatten())
    # print(f"OUTPUT => Top-1 category: {top_cls} [{IMAGENET_LABELS[top_cls]}]")
    make_visual_output(pred, args)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model", default="deploy_resnet50_sakura2/")
    parser.add_argument("--target", default="ip")
    parser.add_argument("--device", default="sakura2")
    parser.add_argument("--input_image", default="data/input/cat.png")
    parser.add_argument("--out_dir", default="./")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
