from pathlib import Path
import argparse
import numpy as np
import cv2

import mera
from mera_models.utils import get_device_platform

def quantize(model_fp32, qtzed_path, calib_imgs_num, eval_imgs_num, coco_dir, size, platform):
    print(f"Running MERA quantization ...")
    qtzed_path = Path(qtzed_path)
    calibration_imgs = get_calib_inputs(size, calib_imgs_num, coco_dir)
    calib_data = [{"images": v} for v in calibration_imgs]
    eval_imgs = get_calib_inputs(size, eval_imgs_num, coco_dir)
    eval_data = [{"images": v} for v in eval_imgs]

    with mera.Deployer(qtzed_path / "qtzer_deploy", overwrite=True) as deployer:
        model_fp32 = mera.ModelLoader(deployer).from_onnx(model_fp32)

        qtzer_cfg = mera.quantizer.QuantizerConfigPresets.DNA_SAKURA_II
        qtzer = mera.Quantizer(
            deployer, model_fp32, quantizer_config=qtzer_cfg, mera_platform=platform
        )

        model_qtz = qtzer.calibrate(calib_data).quantize()
        if not qtzed_path.exists(): 
            qtzed_path.mkdir(parents=True)
        model_qtz.save_to(qtzed_path / "model.mera")

        qtzer.evaluate_quality(eval_data)
    return qtzed_path 


def get_calib_inputs(size, image_num, coco_dir):

    # coco_val_txt = join(coco_dir, "mlperf_calibration_coco_cal_images_list.txt")
    # with open(coco_val_txt, "r") as f:
    #     path_lst = f.readlines()

    # # cleanup and add base dir
    # path_lst = [join(coco_dir, p[2:].strip()) for p in path_lst]
    path_lst = list(Path(coco_dir).rglob("*.jpg"))
    print(path_lst)

    print(f"using {image_num} images")

    # probably image_num not more than 5k
    arr_lst = []
    for image_path in path_lst[:image_num]:
        # image_path = "./data/horses.jpg"
        image = cv2.imread(image_path)
        arr = preprocess_input(image, size)
        arr_lst.append(arr)
    return arr_lst


def preprocess_input(image, size):
    image_data = cv2.resize(np.copy(image), (size, size))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    image_data = image_data / 255.0
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    image_data = image_data.transpose(0, 3, 1, 2)
    image_data = np.ascontiguousarray(image_data)
    return image_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_dir",
        default="./data/calib_images",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="./source_model_files/yolov8m_hardswish_finetuned_640x640.onnx",
        type=str,
    )
    parser.add_argument(
        "--qtzed_path",
        default="./source_model_quantized/model_qtz.mera",
        type=str,
    )
    parser.add_argument(
        "--input_size", default=640, type=int, help="input size, default is 640"
    )
    parser.add_argument(
        "--calib_imgs_num",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--eval_imgs_num",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--device",
        default="sakura2",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    platform = get_device_platform(args.device)
    print(f"Using Device Platform: {platform}")
    quantize(
        model_fp32=args.model_path,
        qtzed_path=args.qtzed_path,      
        calib_imgs_num=args.calib_imgs_num,
        eval_imgs_num=args.eval_imgs_num,
        coco_dir=args.coco_dir,
        size=args.input_size,
        platform=platform,
    )
