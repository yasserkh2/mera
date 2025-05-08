## Necessary utils function copied from  https://github.com/ultralytics/ultralytics

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import time
import yaml
import torchvision


class YoloV8PostProcess:
    def __init__(
        self,
        inp_shape=(640, 640),
        strides=(8, 16, 32),
        anchors_per_stride=16,
        conf_thres=0.25,
        iou_thres=0.45,
    ):
        self.anchors_per_stride = anchors_per_stride
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # fixed for specific image shape
        self.xy_anchor_grid = self.generate_anchor_grid(inp_shape, strides)
        self.xy_mul_grid = self.generate_xy_mul_grid(inp_shape, strides)

    def process(self, head_list: list):
        """
        Args:
            pred: list of numpy array
        """
        if len(head_list) == 6: # for previous compatibility
            tmp = self.combine_heads(head_list)
            tmp = torch.from_numpy(tmp)  # as NMS use pytorch library
        elif len(head_list) == 1: # from pure onnx
            tmp = torch.from_numpy(head_list[0])
        else:
            raise ValueError(f"no. of head is {len(head_list)}, not 1 or 6.")
        predictions = self.non_max_suppression(
            prediction=tmp,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
        )
        return predictions

    def generate_anchor_grid(self, inp_shape=(640, 640), strides=(8, 16, 32)):
        # get num passes for each stride -- length / stride
        passes = np.int32(np.divide(inp_shape, np.tile(strides, (2, 1)).T))
        a = np.zeros((np.sum(np.prod(passes, axis=1)), 2))
        start = 0
        end = 0
        for row in passes:
            y, x = tuple(row)
            end += x * y
            a[start:end, 0] = np.tile(np.arange(x), (1, y)) + 0.5
            a[start:end, 1] = (
                np.tile(np.arange(y).reshape((-1, 1)), (1, x)).reshape((-1)) + 0.5
            )
            start += x * y

        return np.expand_dims(a.T, axis=0)

    def generate_xy_mul_grid(self, inp_shape=(640, 640), strides=(8, 16, 32)):
        inp_shape = np.array(inp_shape)
        xy_strides = np.tile(strides, (2, 1)).T
        passes = np.int32(
            np.divide(inp_shape, xy_strides)
        )  # pass = side length / stride length
        g = np.hstack(
            [np.ones(np.prod(p)) * stride for p, stride in zip(passes, strides)]
        ).reshape((-1, 1))
        return g.T

    def combine_heads(
        self,
        pred: list,
    ):
        expected_shape = [
            (1, 64, 80, 80),  # 0
            (1, 80, 80, 80),  # 1
            (1, 64, 40, 40),  # 2
            (1, 80, 40, 40),  # 3
            (1, 64, 20, 20),  # 4
            (1, 80, 20, 20),  # 5
        ]
        for i in range(len(pred)):
            s1, s2 = pred[i].shape, expected_shape[i]
            assert s1 == s2, f"Expected shape {s2}, but got {s1} in {i}-th head."

        # transpose and reshape
        for i in range(len(pred)):
            x = pred[i]
            c_dim = x.shape[1]
            x = np.einsum("nchw->nhwc", x)
            x = np.reshape(x, (1, -1, c_dim))
            x = np.einsum("nkc->nck", x)
            pred[i] = x

        boxes = np.concatenate(pred[0::2], axis=2)
        class_probs = np.concatenate(pred[1::2], axis=2)

        boxes = boxes.reshape(
            (1, 4, self.anchors_per_stride, -1)
        )  # 4 is num coords per box
        boxes = np.exp(boxes) / np.sum(np.exp(boxes), axis=2, keepdims=True)  # softmax
        boxes = boxes.transpose((0, 2, 1, 3))

        boxes = np.sum(
            boxes
            * np.arange(self.anchors_per_stride).reshape(
                (1, self.anchors_per_stride, 1, 1)
            ),
            axis=1,
        )  # conv

        slice_a = boxes[:, :2, :]
        slice_b = boxes[:, 2:4, :]

        # get xy and wh
        xy = (self.xy_anchor_grid + (slice_b - slice_a) / 2) * self.xy_mul_grid
        wh = (slice_b + slice_a) * self.xy_mul_grid

        xywh = np.concatenate((xy, wh), axis=1)

        class_probs = 1.0 / (1.0 + np.exp(-class_probs))  # sigmoid on class probs

        out = np.concatenate((xywh, class_probs), axis=1)

        return out

    def xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
        Returns:
            y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def box_iou(self, box1, box2, eps=1e-7):
        """
        Calculate intersection-over-union (IoU) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

        Args:
            box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
            box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(
            2, 2
        )
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.001,
        iou_thres=0.7,  # 0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
    ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Arguments:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """

        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert (
            0 <= iou_thres <= 1
        ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        if isinstance(
            prediction, (list, tuple)
        ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = "mps" in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x.transpose(0, -1)[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)
            box = self.xywh2xyxy(
                box
            )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            if multi_label:
                i, j = (cls > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat(
                    (box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1
                )
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[
                    conf.view(-1) > conf_thres
                ]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
                # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                print.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
                break  # time limit exceeded

        return output


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border

    return img


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords



def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    balancer = round(0.02 * 0.5 * (img.shape[0] + img.shape[1]))
    tl = (line_thickness or balancer) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )



def draw_bbox(pred, overlay, input_h, input_w):
    # Get coco class label and corresponding color
    coco_info = yaml.safe_load(Path("./data/coco_classes.yaml").read_text())
    names = coco_info["class_label"]
    colors = coco_info["colors"]

    for _, det in enumerate(pred):  # detections per image
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(
            (input_h, input_w), det[:, :4], overlay.shape, ratio_pad=None
        ).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            txt = f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            print(txt)

        # Write results
        for *xyxy, conf, cls in reversed(det):
            label = f"{names[int(cls)]} {conf:.2f}"
            plot_one_box(
                xyxy, overlay, label=label, color=colors[int(cls)], line_thickness=1
            )
    overlay = overlay.astype(np.uint8)
    return overlay

