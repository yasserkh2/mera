import json

import numpy as np


def make_calib_and_eval_data(
    model_id,
    dataset,
    calib_num,
    eval_num,
    shuffle=False,
):
    print(f"making calibration data for {dataset}...")
    all_choice = {
        "imagenet-sample": {
            "id": "hunarbatra/imagenet1k_val_3k",
            "subset": None,
            "split": "validation",
            "col": "image",
        },
    }
    if dataset.lower() not in all_choice:
        raise ValueError(f"data generation only works with {all_choice.keys()}")

    dataset_id = all_choice[dataset]["id"]
    subset = all_choice[dataset]["subset"]
    split = all_choice[dataset]["split"]
    col = all_choice[dataset]["col"]

    # load datasets and pre-processing
    from transformers import AutoFeatureExtractor
    from datasets import load_dataset

    feat_ext = AutoFeatureExtractor.from_pretrained(model_id)
    dataset = load_dataset(dataset_id, name=subset, split=split)
    if shuffle:
        print("shuffle calibration data..")
        dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(calib_num + eval_num))
    data_lst = []
    for v in dataset[col]:
        processed = feat_ext(v)["pixel_values"][0]
        if len(processed.shape) == 3:  # add batch dim
            processed = processed[np.newaxis, ...]
        data_lst.append(processed)

    # make a list of dicts with input names
    data_lst = [{"pixel_values": da} for da in data_lst]

    # data packing
    CALIB_DATA = data_lst[:calib_num]
    EVAL_DATA = data_lst[-eval_num:]

    return CALIB_DATA, EVAL_DATA


def main(arg):

    # get quantizer calibration data
    calib_data, eval_data = make_calib_and_eval_data(
        model_id=arg.model_id,
        dataset=arg.calib_dataset,
        calib_num=arg.calib_num,
        eval_num=arg.eval_num,
        shuffle=arg.calib_shuffle,
    )

    # define shape
    shape_mapping = {
        "batch_size": arg.batch_size,
        "num_channels": arg.num_channels,
        "height": arg.height,
        "width": arg.width,
    }

    from mera_models.hf.meraruntime import MERAModelForImageClassification as mf

    source_dir = mf.quantize(
        model_id=arg.model_id,
        out_dir=arg.qtzed_path,
        platform=arg.device,
        shape_mapping=shape_mapping,
        cache_dir="./source_model_files",
        calib_data=calib_data,
        eval_data=eval_data,
        qtz_quality_check=False,
        qtz_debug_mode=True,
    )
    print(f"quantized folder at {source_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qtzed_path",
        default="./qtzed_tmp",
        type=str,
        help="temporary output folder in case of quantization.",
    )
    parser.add_argument(
        "--model_id",
        default="google/vit-base-patch16-224",
        # default="microsoft/resnet-50",
        type=str,
        help="either a huggingface model_id or a path to exported onnx folder.",
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
        default=1,
        type=int,
    )
    main(parser.parse_args())
