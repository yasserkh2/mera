import os
import sys
from os.path import splitext, join, basename

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np

import torch
import torchvision.transforms as T

import mera
from mera import Target

from mera_models.utils import get_target, get_device_target, flags_to_dict

# COCO classes
DETR_COCO_CLASSES = [
    "N/A",
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
    "N/A",
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
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
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
    "N/A",
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
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
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
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    source_img = pil_img.copy()
    draw = ImageDraw.Draw(source_img)
    fontsize = 14
    font = ImageFont.truetype("DejaVuSans.ttf", size=fontsize)
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        c = (0, 200, 100)
        cl = p.argmax()
        text = f"{DETR_COCO_CLASSES[cl]}: {p[cl]:0.2f}"
        draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=None, width=3, outline=c)
        draw.text((xmin + 3, ymin + 5), text, font=font)

    arr = np.array(source_img)
    print(arr.shape)

    return arr


def post_processing(logits, boxes, orig_w, orig_h, conf_threshold=0.7):
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
        boxes = torch.from_numpy(boxes)
    # keep only predictions with 0.7+ confidence
    probas = logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(boxes[0, keep], (orig_w, orig_h))

    return probas[keep], bboxes_scaled


def make_square_image(im):
    W, H = im.size
    background_color = (0, 0, 0)
    if W == H:
        return im
    elif W > H:
        result = Image.new(im.mode, (W, W), background_color)
        result.paste(im, (0, (W - H) // 2))
        return result
    else:
        result = Image.new(im.mode, (H, H), background_color)
        result.paste(im, ((H - W) // 2, 0))
        return result


def pre_processing(im, input_height, input_width):
    # standard PyTorch mean-std input image normalization
    detr_transform = T.Compose(
        [
            T.Resize((input_height, input_width)),  # orig is 800
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = detr_transform(im).unsqueeze(0)
    return img.detach().cpu().numpy()


def process_runtime_metrics(runner):
    try:
        metrics = runner.get_runtime_metrics()
        sim_us = [float(x["sim_time_us"]) for x in metrics]
        sim_ms = sum(sim_us) / 1000
        print(f" ** Estimated Simulator latency: {sim_ms} ms")
    except:
        print(f"<Could not extract SimulatorBf16 runtime metrics>")


def run_inference(image, predictor, model_flags):
    # --- take care of model flags
    flags = flags_to_dict(model_flags)

    is_onnx = bool(int(flags.get("onnx", 0)))
    input_height = int(flags.get("input_height", 320))
    input_width = int(flags.get("input_width", 320))

    conf_threshold = 0.7

    # --- preprocess
    orig_w, orig_h = image.size
    input_data = pre_processing(image, input_height, input_width)
    print(f"input: {input_data.shape}")

    # -- run raw inference
    if is_onnx:
        pred = predictor.run(
            ["output.logits", "output.boxes"],
            {"input.1": input_data},
        )
    else:
        runner = predictor.set_input(input_data).run()
        pred = runner.get_outputs()
        process_runtime_metrics(predictor)

    logits, boxes = pred
    keep_probs, bboxes_scaled = post_processing(
        logits,
        boxes,
        orig_w,
        orig_h,
        conf_threshold,
    )
    overlay = plot_results(image, keep_probs, bboxes_scaled)

    return np.ascontiguousarray(overlay)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="data/000000039769.jpg",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./result.png",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="./deploy_detr",
        # default="./source_model_files/detr-sim.onnx",
        type=str,
    )
    parser.add_argument(
        "--model_flags",
        default="a=0,input_height=400,input_width=600",
        type=str,
        help="please see the inference function for meaning of each flag.",
    )
    parser.add_argument(
        "--target",
        default="ip",
        type=str.lower,
        help="MERA Target environment, ip is (default)",
    )
    parser.add_argument(
        "--device",
        default="sakura2",
        type=str.lower,
    )
    arg = parser.parse_args()

    image = Image.open(arg.input_path)
    image = make_square_image(image)

    if splitext(arg.model_path)[-1] == ".onnx":
        import onnx
        import onnxruntime

        print(f"running inference on ONNX runtime...")
        arg.model_flags += ",onnx=1"
        predictor = onnxruntime.InferenceSession(arg.model_path)
    else:
        print(f"running inference on MERA {arg.target}...")
        target = get_target(arg.target)
        ip1 = mera.load_mera_deployment(arg.model_path, target=target)
        predictor = ip1.get_runner(device_target=get_device_target(arg.device))

    overlay = run_inference(image, predictor, arg.model_flags)
    print(f"original image size:{image.size}")
    print(f"overlay size:{overlay.shape}")
    Image.fromarray(overlay).save(arg.save_path)
