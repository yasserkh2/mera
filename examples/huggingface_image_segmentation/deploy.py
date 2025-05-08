# -*- coding: utf-8 -*-
"""
1. Compiling Image Segmentation
===============================


"""

import json
import argparse
import warnings

from mera_models.hf.meraruntime import MERAModelForSemanticSegmentation as mf

warnings.filterwarnings("ignore")


def main(arg):
    """Main function to handle model deployment"""

    # Define shape
    shape_mapping = {
        "batch_size": arg.batch_size,
        "num_channels": arg.num_channels,
        "height": arg.height,
        "width": arg.width,
    }

    # Replace quote for json compatibility
    build_dt = json.loads(str(arg.build_cfg).replace("'", '"'))

    # Deploy the model
    mf.deploy(
        model_id=arg.model_id,
        out_dir=arg.out_dir,
        platform=arg.device,
        target=arg.target,
        shape_mapping=shape_mapping,
        build_cfg=build_dt,
        cache_dir="./source_model_files",
        host_arch=arg.host_arch,
    )
    print(f"Deployed folder at: {arg.out_dir}")


def get_args():
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="./deploy_segformer-b0-512-512",
        type=str,
    )
    parser.add_argument(
        "--model_id",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        type=str,
        help="Either a huggingface model_id or a path to exported onnx folder",
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
        default=512,
        type=int,
    )
    parser.add_argument(
        "--width",
        default=512,
        type=int,
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


if __name__ == "__main__":
    args = get_args()
    main(args)
