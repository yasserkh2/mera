import argparse
import json

from mera_models.hf.meraruntime import MERAModelForSequenceClassification as mf


def main(arg):

    # define shape
    shape_mapping = {
        "batch_size": arg.batch_size,
        "sequence_length": arg.sequence_length,
    }

    # replace quote for json compatibility
    build_dt = json.loads(str(arg.build_cfg).replace("'", '"'))

    source_dir = arg.model_id
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="./deploy_roberta-base-go_emotions",
        type=str,
    )

    parser.add_argument(
        "--model_id",
        default="./qtzed_tmp",
        type=str,
        help="either a huggingface model_id or a path to exported onnx folder.",
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
        default="{'enable_mm_graph_cutting': false, 'scheduler_config': {'mode': 'Simple'}}",
        type=str,
    )
    parser.add_argument(
        "--host_arch",
        default="x86",
        type=str.lower,
    )
    main(parser.parse_args())
