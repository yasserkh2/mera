#!/usr/bin/env python3
import logging
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import mera
from mera_models.utils import get_target, get_device_target


if __name__ == "__main__":
    from yolov8_utils import YoloV8PostProcess, letterbox, draw_bbox
else:
    from .yolov8_utils import YoloV8PostProcess, letterbox, draw_bbox


logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO"),
)
logger = logging.getLogger(__name__)


def run_inference(image, predictor, arg):

    # Post processor object. For multiple run, keep it outside of the for loop
    PP = YoloV8PostProcess(
        inp_shape=(arg.input_size, arg.input_size),
        conf_thres=arg.score,
        iou_thres=arg.iou,
    )

    # --- preprocess
    image_data = preprocess_input(image, arg.input_size)

    # -- run raw inference
    runner = predictor.set_input(image_data).run()
    pred = runner.get_outputs()

    # ---- do the rest of post processing
    pred_final = PP.process(pred)

    # ---- Create overlay output image
    input_h = input_w = arg.input_size
    overlay = draw_bbox(pred_final, image.copy(), input_h, input_w)
    return overlay


def preprocess_input(image, size, use_letterbox=True):
    if isinstance(size, (list, tuple)):
        size_h, size_w = size
    else:
        size_h, size_w = size, size
    if use_letterbox:
        # will use padding to keep aspect ratio
        resized = letterbox(image, new_shape=(size_h, size_w), auto=False, stride=32)
    else:
        # by using this, the postprocessing of bbox might be incorrect,
        # due to not padding and distorting aspect ratio.
        resized = cv2.resize(np.copy(image), (size_h, size_w))

    image_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_data = image_data / 255.0
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    image_data = image_data.transpose(0, 3, 1, 2)
    image_data = np.ascontiguousarray(image_data)
    return image_data


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="./data/input_images/bus.jpg",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./result.png",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="./deploy_yolov8m",
        type=str,
    )
    parser.add_argument(
        "--input_size",
        default=640,
        type=int,
    )

    parser.add_argument(
        "--iou",
        default=0.45,
        type=float,
    )

    parser.add_argument(
        "--score",
        default=0.15,
        type=float,
    )

    parser.add_argument(
        "--target",
        default="ip",
        type=str.lower,
        help="Set the MERA target environment. Defaults to 'ip'."
    )
    parser.add_argument(
        "--device",
        default="sakura2",
        type=str,
    )
    arg = parser.parse_args()
    return arg


if __name__ == "__main__":
    arg = get_arg()

    target = get_target(arg.target)
    device_target = get_device_target(arg.device)
    print(f"Targeting {target} using {device_target} ...")

    # Load model and get the predictor
    ip1 = mera.load_mera_deployment(arg.model_path, target=target)
    predictor = ip1.get_runner(device_target=device_target)

    # Read input images
    image = cv2.imread(str(arg.input_path))
    print(f"Input Image shape: {image.shape}")

    # Make inference
    overlay = run_inference(image, predictor, arg)

    # Save the output
    save_path = arg.save_path
    cv2.imwrite(save_path, overlay)
    print(f"Saved {save_path}\n")
    print("Done.")
