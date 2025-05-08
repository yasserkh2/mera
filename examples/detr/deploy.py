#!/usr/bin/env python
import logging
import json

import mera
from mera import Target, Platform
from mera_models.utils import get_device_platform, get_target

logging.basicConfig(level=logging.INFO)


# Function for compiling the model and return the pre-preprocessed input data
def compile_mera(onnx_filename, out_dir, mera_platform, mera_target, build_cfg_dt, host_arch):
    # Compilation settings
    with mera.Deployer(out_dir, overwrite=True) as deployer:
        model = mera.ModelLoader(deployer).from_onnx(onnx_filename)

        deployer.deploy(
            model,
            mera_platform=mera_platform,
            target=mera_target,
            host_arch=host_arch,
            build_config=build_cfg_dt,
        )

        return out_dir


def main(arg):
    mera_platform = get_device_platform(arg.device)
    mera_target = get_target(arg.target)
    build_cfg_dt = json.loads(str(arg.build_cfg).replace("'", '"'))
    mera_path = compile_mera(
        arg.model_path, arg.out_dir, mera_platform, mera_target, build_cfg_dt, arg.host_arch
    )
    print(f"SUCCESS, saved at {mera_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="deploy_detr",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="./source_model_files/detr_600x400_nodict.onnx",
        type=str,
    )
    parser.add_argument(
        "--device",
        default="sakura2",
        type=str,
    )
    parser.add_argument(
        "--target",
        default="ip",
        type=str.lower,
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

    main(parser.parse_args())
