import argparse
import json

import numpy as np
import tensorflow as tf

import mera
from mera import Target, Platform
from mera_models.utils import get_device_platform, get_target


def load_image(image_path, input_size):
    """Image load & process"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    input_image = tf.cast(input_image, tf.float32)
    return input_image


def compile_mera(out_dir, model_path, mera_platform, host_arch, target, build_cfg_dt):
    """Compile the model using MERA"""
    with mera.Deployer(out_dir, overwrite=True) as deployer:
        model = mera.ModelLoader(deployer).from_tflite(model_path)

        print(f"\nCompiling for {target}")
        deployer.deploy(
            model,
            mera_platform=mera_platform,
            target=target,
            host_arch=host_arch,
            build_config=build_cfg_dt,
        )

        print(f"SUCCESS, saved at {out_dir}")


def main(arg):
    """Main function to handle model deployment"""
    mera_platform = get_device_platform(arg.device)
    target = get_target(arg.target)
    build_cfg_dt = json.loads(str(arg.build_cfg).replace("'", '"'))
    compile_mera(
        arg.out_dir,
        arg.model_path,
        mera_platform,
        arg.host_arch,
        target,
        build_cfg_dt,
    )


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
        default="source_model_files/yolov5s-448.tflite",
        type=str,
    )
    parser.add_argument("--out_dir", default="deploy_yolov5s_448", type=str)
    parser.add_argument(
        "--device",
        default="sakura2",
        type=str,
    )
    parser.add_argument(
        "--target",
        default="ip",
        type=str,
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
    arg = get_args()
    main(arg)
