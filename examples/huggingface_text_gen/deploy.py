# -*- coding: utf-8 -*-
"""
1. Compiling Text Generation
===============================


"""

import json
import argparse
import warnings

import numpy as np
from tqdm import tqdm

from mera_models.hf.meraruntime import MERAModelForCausalLM as mf

warnings.filterwarnings("ignore")


def compile_mera(arg):
    """Main function to handle model deployment"""
    # Define shape
    shape_mapping = {
        "batch_size": arg.batch_size,
        "sequence_length": arg.sequence_length,
    }
    # Replace quote for json compatibility
    build_dt = json.loads(str(arg.build_cfg).replace("'", '"'))

    source_dir = arg.model_id  # huggingface model id or quantize model dir
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
    print(f"deployed folder at {arg.out_dir}")
    return arg.out_dir


def get_args():
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="./deploy_SmolLM2-135M-Instruct",
        type=str,
    )
    parser.add_argument(
        "--model_id",
        default="qtzed_tmp",
        type=str,
        help="path to onnx/quantized mera folder.",
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
        "--sequence_length",
        default=128,
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
    compile_mera(args)
