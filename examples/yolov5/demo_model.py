import os
import logging
import cv2
import numpy as np
import torch
import argparse

import mera
from mera import Target
from mera.mera_deployment import DeviceTarget

from core.general import letterbox, non_max_suppression, scale_coords
from core.plots import Annotator, colors
from pathlib import Path
from mera_models.utils import flags_to_dict, get_device_target, get_target


logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def preprocess_input(image, input_h, input_w, stride, layout="NHWC"):
    """Preprocess the image"""
    image0 = letterbox(image, (input_h, input_w), stride=stride, auto=False)[0]
    image_data = image0.transpose((2, 0, 1))[::-1]
    image_data = np.ascontiguousarray(image_data)
    image_data = image_data.astype(np.float32)
    image_data /= 255  # 0 - 255 to 0.0 - 1.0
    image_data = image_data[np.newaxis, ...]
    if layout == "NHWC":
        image_data = image_data.transpose(0, 2, 3, 1)
    return image_data


def postprocess(
    pred,
    conf_thres,
    iou_thres,
    input_h,
    input_w,
    layout="NHWC",
    normalized=True,
):
    """Return detections based on predictions"""
    arr = pred[0]

    xc = arr[..., 4] > conf_thres
    arr_filtered = arr[xc]
    if arr_filtered.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32)]
    arr = arr_filtered[np.newaxis]

    if normalized:
        arr[..., 0] *= input_w  # x
        arr[..., 1] *= input_h  # y
        arr[..., 2] *= input_w  # w
        arr[..., 3] *= input_h  # h

    detections = non_max_suppression(
        torch.from_numpy(arr),
        conf_thres,
        iou_thres,
        classes=None,
        agnostic=False,
        max_det=300,
        multi_label=True,
    )

    return detections


def overlay_result(image_orig, detections, input_h, input_w):
    """Overlay detection results on the original image"""
    image_orig = np.ascontiguousarray(image_orig)
    annotator = Annotator(image_orig, line_width=1, example="abc")
    det = detections[0]
    if len(det):
        det[:, :4] = scale_coords(
            (input_h, input_w), det[:, :4], image_orig.shape
        ).round()

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = f"{COCO_CLASS_NAMES[c]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(c, True))

    im0 = annotator.result()

    return im0


def run_inference(image, predictor, model_flags):
    """Run inference on image using predictor & model flags"""
    model_flags = flags_to_dict(model_flags)
    conf_thres = float(model_flags["conf"])
    iou_thres = float(model_flags["iou"])
    input_h = int(model_flags["input_h"])
    input_w = int(model_flags["input_w"])
    stride = int(model_flags["stride"])

    preprocessed_input = preprocess_input(image, input_h, input_w, stride)
    runner = predictor.set_input(preprocessed_input).run()
    pred = runner.get_outputs()
    detections = postprocess(pred, conf_thres, iou_thres, input_h, input_w)
    overlay = overlay_result(image, detections, input_h, input_w)

    return np.ascontiguousarray(overlay)


def main(arg):
    """Main function to handle model demo"""
    image = cv2.imread(arg.input_path)
    print(f"Loaded input from: {arg.input_path}")
    print(f"Input shape: {image.shape}")

    print(f"Running inference on {arg.target}...")
    target = get_target(arg.target)
    ip_deployment = mera.load_mera_deployment(arg.model_path, target=target)
    iprt = ip_deployment.get_runner(device_target=get_device_target(arg.device))
    output = run_inference(image, iprt, arg.model_flags)
    print(f"Output shape: {output.shape}")
    cv2.imwrite(arg.save_path, output)
    print(f"Result saved at: {arg.save_path}")

    # Additionally saving with detailed name for comparison
    path = Path(arg.input_path)
    new_path = path.parent / f"{path.stem}_result_{arg.target}{path.suffix}"
    cv2.imwrite(str(new_path), output)
    print(f"Result also saved at: {str(new_path)}")


def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="./data/image.jpeg",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="./deploy_yolov5s_448",
        type=str,
    )
    parser.add_argument(
        "--model_flags",
        default="conf=0.35,iou=0.45,input_h=448,input_w=448,stride=64",
        type=str,
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
    parser.add_argument(
        "--save_path",
        default="result.png",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arg = get_args()
    main(arg)
