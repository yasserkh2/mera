import argparse
import numpy as np
from tqdm import tqdm

# load datasets and pre-processing
from transformers import AutoTokenizer
from datasets import load_dataset
from mera_models.hf.meraruntime import MERAModelForSequenceClassification as mf


def quantize_model(arg):
    calib_data, eval_data = make_calib_and_eval_data(
        model_id=arg.model_id,
        sequence_length=arg.sequence_length,
        dataset=arg.calib_dataset,
        calib_num=arg.calib_num,
        eval_num=arg.eval_num,
        shuffle=arg.calib_shuffle,
    )

    # define shape
    shape_mapping = {
        "batch_size": arg.batch_size,
        "sequence_length": arg.sequence_length,
    }
    source_dir = mf.quantize(
        model_id=arg.model_id,
        out_dir=arg.qtzed_out_dir,
        platform=arg.device,
        shape_mapping=shape_mapping,
        cache_dir="./source_model_files",
        calib_data=calib_data,
        eval_data=eval_data,
        qtz_quality_check=False,
        qtz_debug_mode=True,
        apply_smooth_quant=False,
    )
    print(f"Quantized model saved at {source_dir}")
    return source_dir


def _pad_or_cut(input_id, sequence_length):
    curr_length = len(input_id)
    if sequence_length > curr_length:
        new_data = np.pad(np.array(input_id), (0, sequence_length - curr_length))
    elif sequence_length < curr_length:
        new_data = input_id[: -(curr_length - sequence_length)]
    else:
        new_data = input_id

    return np.array(new_data)


def make_input_data_pair(tokens, sequence_length):
    return [
        {k: _pad_or_cut(v, sequence_length)[np.newaxis, :] for k, v in token.items()}
        for token in tqdm(tokens)
    ]

def make_calib_and_eval_data(
    model_id, sequence_length, dataset, calib_num, eval_num, shuffle=False
):
    print(f"making calibration data for {dataset}...")
    all_choice = {
        "mnli": {
            "id": "nyu-mll/multi_nli",
            "subset": None,
            "split": "validation_matched",
            "col": "premise",
        },
        "emotions": {
            "id": "google-research-datasets/go_emotions",
            "subset": "simplified",
            "split": "test",
            "col": "text",
        },
    }
    if dataset.lower() not in all_choice:
        raise ValueError(f"data generation only works with {all_choice.keys()}")

    dataset_id = all_choice[dataset]["id"]
    subset = all_choice[dataset]["subset"]
    split = all_choice[dataset]["split"]
    col = all_choice[dataset]["col"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset(dataset_id, name=subset, split=split)
    if shuffle:
        print("shuffle calibration data..")
        dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(calib_num + eval_num))
    tokens = list(map(tokenizer, dataset[col]))

    data_pair = make_input_data_pair(tokens, sequence_length)

    # data packing
    CALIB_DATA = data_pair[:calib_num]
    EVAL_DATA = data_pair[-eval_num:]

    return CALIB_DATA, EVAL_DATA


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qtzed_out_dir",
        default="./qtzed_tmp",
        type=str,
        help="temporary output folder in case of quantization.",
    )
    parser.add_argument(
        "--model_id",
        default="SamLowe/roberta-base-go_emotions",
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
        "--sequence_length",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--calib_dataset",
        default="emotions",
        type=str,
    )
    parser.add_argument(
        "--calib_shuffle",
        action="store_true",
    )
    parser.add_argument(
        "--calib_num",
        default=30,
        type=int,
    )
    parser.add_argument(
        "--eval_num",
        default=3,
        type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":
    arg = get_arg()
    quantize_model(arg)
