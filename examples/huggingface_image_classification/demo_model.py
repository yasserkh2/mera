from os.path import isfile, join
from pathlib import Path
import argparse
import warnings

from PIL import Image
import numpy as np
import torch

import mera_models as mm

warnings.filterwarnings("ignore")


def load_predictor(model_path, target_str, device_str):
    """Return predictor based on either onnxruntime or meraruntime"""
    check_file = join(model_path, "model.onnx")

    if isfile(check_file):
        # actual onnx api class
        from optimum.onnxruntime import ORTModelForImageClassification as mf

        print(f"running inference on huggingface ONNX runtime...")
        predictor = mf.from_pretrained(
            model_path,
            use_cache=False,
            use_io_binding=False,
        )

    else:
        # MERA load api class
        from mera_models.hf.meraruntime import MERAModelForImageClassification as mf

        print(mf)

        predictor = mf.from_pretrained(
            model_path,
            use_cache=False,
            target=target_str,
            device_target=device_str,
            measure_latency=True,
        )

    return predictor


def get_result_as_text(result):
    """Generate text (label + score) from result"""
    txt = ""
    for i, dt in enumerate(result):
        lb = dt["label"]
        score = dt["score"] * 100
        txt += f"{i+1}. {lb:<27}{score:>4,.1f}%\n"

    return txt


def run_inference(inputs, predictor, model_flags, **kwargs):
    """Run image classification inference"""

    if "MERA" in predictor.__class__.__name__:
        from mera_models.hf.pipelines import pipeline
    else:
        from transformers import pipeline

    if len(predictor.preprocessors) >= 1:
        feat_ext = predictor.preprocessors[0]
    else:
        from transformers import AutoImageProcessor

        feat_ext = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        print("-- WARNING: no preprocessor provided, load from resnet50.")

    if hasattr(predictor.config, "num_classes") and (
        predictor.config.num_classes != len(predictor.config.id2label)
    ):
        from transformers import AutoConfig

        config1 = AutoConfig.from_pretrained("microsoft/resnet-50")
        predictor.config.id2label = config1.id2label
        print("-- WARNING: no id2label provided, load from resnet50.")

    pipe_func = pipeline(
        task="image-classification",
        model=predictor,
        feature_extractor=feat_ext,
    )

    out_lst = pipe_func(inputs)
    result_txt = get_result_as_text(out_lst)

    return result_txt


def main(arg):
    """Main function to handle model demo"""
    seed = 1337  # to generate reproducible results
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    predictor = load_predictor(arg.model_path, arg.target, arg.device)

    image_data = Image.open(arg.input_path)
    image_cv = np.array(image_data).copy()
    image_cv = image_cv[:, :, ::-1]  # RGB -> BGR
    image_cv = np.ascontiguousarray(image_cv)

    result_txt = run_inference(
        image_data,
        predictor,
        arg.model_flags,
    )
    print(f"\nResult:")
    print(result_txt)

    mm.utils.save_text_as_image_file(
        arg.save_path,
        result_txt,
        overlay=image_cv,
        max_char_per_line=36,
        font_scale=0.4,
    )
    print(f"Result saved at: {arg.save_path}")

    # Additionally saving with detailed name for comparison
    path = Path(arg.input_path)
    new_path = str(path.parent / f"{path.stem}_result_{arg.target}{path.suffix}")
    mm.utils.save_text_as_image_file(
        new_path,
        result_txt,
        overlay=image_cv,
        max_char_per_line=36,
        font_scale=0.4,
    )
    print(f"Result also saved at: {str(new_path)}")

    # Measure estimated latency
    if arg.target.lower() == "simulatorbf16":
        print(f" ** Estimated SimulatorBf16 latency {predictor.estimated_latency} ms")


def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="data/cat.png",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./result.png",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="./deploy_vit-base-patch16-224",
        # default="./deploy_resnet50",
        type=str,
    )
    parser.add_argument(
        "--model_flags",
        default="a=0",
        type=str,
        help="please see the inference function for meaning of each flag.",
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
    arg = get_args()
    main(arg)
