import os
import sys
import numpy as np
import torch
import mera_models as mm

from os.path import isfile, isdir, join
from PIL import Image
from transformers import AutoTokenizer
from mera_models.hf.auto import load_auto_model, load_auto_pipeline


def load_predictor(model_path, target_str, device_str):
    check_file = join(model_path, "model.onnx")
    runtime = "huggingface"
    if isfile(check_file):
        runtime = "onnx"
    elif isdir(model_path):
        runtime = "mera"
    print(f"running inference on {runtime.upper()} runtime...")

    predictor = load_auto_model(
        model_id=model_path,
        runtime=runtime,
        task="text-classification",
        target=target_str,
        device_target=device_str,
    )

    return predictor


def get_result_as_text(result):
    txt = ""
    for i, dt in enumerate(result):
        lb = dt["label"]
        score = dt["score"] * 100
        txt += f"{i+1}. {lb:<27}{score:>4,.1f}%\n"

    return txt


def run_inference(inputs, predictor, tokenizer, **kwargs):
    if "MERA" in predictor.__class__.__name__:
        runtime = "mera"
    elif "ORT" in predictor.__class__.__name__:
        runtime = "onnx"
    else:
        runtime = "huggingface"

    pipeline = load_auto_pipeline(runtime)

    pipe_func = pipeline(
        task="text-classification",
        model=predictor,
        tokenizer=tokenizer,
        top_k=3,
    )

    out_lst = pipe_func(inputs)
    result_txt = get_result_as_text(out_lst[0])

    return result_txt


def main(arg):
    seed = 1337  # will produce same result with same seed due to torch sampling
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    predictor = load_predictor(arg.model_path, arg.target, arg.device)
    tokenizer = AutoTokenizer.from_pretrained(arg.model_path)

    print("-- input --")
    print(arg.input_txt)

    # -- run actual inference
    result_txt = run_inference(
        arg.input_txt,
        predictor,
        tokenizer,
    )
    print("-- result --")
    print(result_txt)

    mm.utils.save_text_as_image_file(
        arg.save_path,
        result_txt,
        overlay=None,
        max_char_per_line=36,
        font_scale=0.6,
    )

    # Measure estimated latency
    if arg.target.lower() == "simulatorbf16":
        print(f" ** Estimated SimulatorBf16 latency {predictor.estimated_latency} ms")

    # Measure estimated latency
    if arg.target.lower() == "simulator":
        print(f" ** Estimated Simulator latency {predictor.estimated_latency} ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_txt",
        default="I love you.",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./result.png",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        # default="source_model_files/distilbert__distilbert-base-uncased-finetuned-sst-2-english_onnx",
        default="./deploy_roberta-base-go_emotions",
        type=str,
    )
    parser.add_argument(
        "--target",
        default="ip",
        type=str,
        help="MERA Target environment, ip is default",
    )
    parser.add_argument("--device", default="sakura2", type=str)
    main(parser.parse_args())
