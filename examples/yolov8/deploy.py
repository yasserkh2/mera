from pathlib import Path
import argparse
import json

import mera
from mera_models.utils import get_device_platform, get_target

def compile_mera(model_path, out_dir, platform, target, build_cfg_dt, host_arch):
    # ----- instantiate the model and save
    print(f"\nDeploying MERA model ...")
    with mera.Deployer(out_dir, overwrite=True) as deployer:
        suffix = Path(model_path).suffix
        if suffix == ".onnx":
            model = mera.ModelLoader(deployer).from_onnx(model_path)
        else:
            model = mera.ModelLoader(deployer).from_quantized_mera(model_path)

        deployer.deploy(
            model,
            mera_platform=platform,
            build_config=build_cfg_dt,
            target=target,  # Target.Simulator,
            host_arch=host_arch,
        )

    return out_dir


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./source_model_quantized/model_qtz.mera/model.mera",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default="./deploy_yolov8m",
        type=str,
    )
    parser.add_argument(
        "--device",
        default="sakura2",
        type=str,
    )
    parser.add_argument(
        "--target",
        default="ip",  # simulator, InterpreterHw
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
    arg = parser.parse_args()
    return arg


if __name__ == "__main__":
    arg = get_arg()

    platform = get_device_platform(arg.device)
    target = get_target(arg.target)
    model_path = arg.model_path
    build_cfg_dt = json.loads(str(arg.build_cfg).replace("'", '"'))
    print(f"Targeting {target} on {platform} ")
    mera_path = compile_mera(
        model_path,
        out_dir=arg.out_dir,
        platform=platform,
        target=target,
        build_cfg_dt=build_cfg_dt,
        host_arch=arg.host_arch,
    )

    print(f"SUCCESS, saved at {mera_path}")
