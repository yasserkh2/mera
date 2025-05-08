import json

import mera
from mera import Target, Platform

from mera_models.utils import get_device_platform, get_target


# ### Compilation with MERA software stack helper
def compile_mera(tflite_filename, target, platform, host_arch, output_dir, build_cfg_dt):
    with mera.Deployer(output_dir, overwrite=True) as deployer:
        model = mera.ModelLoader(deployer).from_tflite(tflite_filename)

        deployer.deploy(
            model,
            mera_platform=platform,
            target=target,
            host_arch=host_arch,
            build_config=build_cfg_dt,
        )

    return output_dir


def main(args):
    platform = get_device_platform(args.device)
    host_arch = args.host_arch 

    model_path = args.model_path
    output_dir = args.out_dir
    target = get_target(args.target)
    build_cfg_dt = json.loads(str(args.build_cfg).replace("'", '"'))
    deploy_dir = compile_mera(
        model_path, target, platform, host_arch, output_dir, build_cfg_dt
    )
    print(f"compiled {deploy_dir}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=f"deploy_effnet_lite4",  # [1, 4]
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default=f"source_model_files/effnet-lite4-int8.tflite",  # [1, 4]
        type=str,
    )
    parser.add_argument(
        "--target",
        default="ip",
        type=str.lower,
    )
    parser.add_argument(
        "--device",
        default="sakura2",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
