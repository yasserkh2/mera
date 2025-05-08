import json
import argparse
import warnings

import numpy as np
from transformers import AutoFeatureExtractor
from datasets import load_dataset

from mera_models.hf.meraruntime import MERAModelForImageClassification as mf

warnings.filterwarnings("ignore")


def main(arg):
    # Replace quote for json compatibility
    build_dt = json.loads(str(arg.build_cfg).replace("'", '"'))

    source_dir = arg.model_id

    # define shape
    shape_mapping = {
        "batch_size": arg.batch_size,
        "num_channels": arg.num_channels,
        "height": arg.height,
        "width": arg.width,
    }

    mf.deploy(
        model_id=source_dir,
        out_dir=arg.out_dir,
        platform=arg.device,
        target=arg.target,
        shape_mapping=shape_mapping,
        build_cfg=build_dt,
        cache_dir="./source_model_files",
        host_arch=arg.host_arch,
    )
    print(f"Deployed folder at {arg.out_dir}")


def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="./deploy_vit-base-patch16-224",
        type=str,
    )
    parser.add_argument(
        "--model_id",
        default="./qtzed_tmp",
        # default="microsoft/resnet-50",
        type=str,
        help="either a huggingface model_id, a path to exported onnx folder, or quantized mera folder",
    )
    parser.add_argument(
        "--target",
        default="ip",
        type=str,
    )
    parser.add_argument(
        "--device",
        default="sakura2",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num_channels",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--height",
        default=224,
        type=int,
    )
    parser.add_argument(
        "--width",
        default=224,
        type=int,
    )
    parser.add_argument(
        "--build_cfg",
        default="{'scheduler_config': {'mode': 'Simple'}}",
        type=str,
    )
    parser.add_argument(
        "--calib_dataset",
        default="imagenet-sample",
        type=str,
    )
    parser.add_argument(
        "--calib_shuffle",
        action="store_true",
    )
    parser.add_argument(
        "--calib_num",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--eval_num",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--host_arch",
        default="x86",
        type=str.lower,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arg = get_args()
    main(arg)
