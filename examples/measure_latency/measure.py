import collections
import json
import os
import sys
import fnmatch
import time
from pathlib import Path
from typing import Dict, List

import mera
import numpy as np
from mera import Target
from mera.deploy_project import MeraDeployProject
from mera.mera_deployment import DeviceTarget
from tqdm import tqdm


def find_dirs(root, name_pattern="deploy_*"):
    for entry in os.scandir(root):
        if entry.is_dir() and entry.name != "venv":
            if fnmatch.fnmatch(entry.name, name_pattern):
                yield entry.path
            yield from find_dirs(entry.path)


def extract_size_dt(model_path: str):
    try:
        prj = MeraDeployProject(Path(model_path), compile_flow="MERA")  # for mera2
    except TypeError:
        prj = MeraDeployProject(Path(model_path))  # for mera1
    json_path = prj.get_artifact("model", "input_desc.json")
    with open(json_path) as jp:
        size_container = json.load(jp)
    if isinstance(size_container, Dict):
        size_dt = size_container
    elif isinstance(size_container, List):
        size_dt = {}
        for obj in size_container:
            size_dt[obj[0]] = obj[1]
    else:
        raise ValueError("cannot deal with size_container other than dict and list")
    return size_dt


def make_input(size_dt):
    data = []
    for _, v in size_dt.items():
        if isinstance(v[0], int):
            data.append(np.random.rand(*v).astype(np.float32))
        else:
            data.append(np.random.rand(*v[0]).astype(v[1]))
    return data if len(data) > 1 else data[0]


def get_time_ms():
    return time.perf_counter() * 1000


def aggregate(raw_data):
    if len(raw_data) == 1:
        if isinstance(raw_data, List):  # MERA1 format
            return raw_data[0]
        return raw_data.popitem()[1]  # MERA2: value of first key of dict

    new_dt = {}
    try:  # MERA1 format
        for dt in raw_data:
            for k, v in dt.items():
                if k == "freq_mhz":
                    continue
                new_dt[k] = new_dt.get(k, 0) + v
    except AttributeError:  # MERA2 format
        for sub_name, sub_dt in raw_data.items():
            for k, v in sub_dt.items():
                if k == "freq_mhz" or k == "ip_ip_sub_time":
                    continue
                new_dt[k] = new_dt.get(k, 0) + v

    return new_dt


def run_model_forver(input_data, predictor):
    import sys
    import time

    runner = predictor.set_input(input_data).run()

    try:
        while True:
            runner1 = predictor.set_input(input_data)
            runner1.run()
            # time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nExiting from forever loop...")
        sys.exit(0)


def measure(input_data, predictor, loop_cnt):
    setinput_lst = []
    mera_metrics_lst = []
    end_to_end_lst = []

    # Warm-up
    for _ in tqdm(range(3), desc="warmup"):
        runner = predictor.set_input(input_data).run()

    for _ in tqdm(range(loop_cnt), desc="actual run"):
        p0 = get_time_ms()
        runner1 = predictor.set_input(input_data)
        p1 = get_time_ms()
        runner2 = runner1.run()
        p2 = get_time_ms()
        mera_metrics_lst.append(aggregate(runner2.get_runtime_metrics()))
        setinput_lst.append(p1 - p0)
        end_to_end_lst.append(p2 - p1)

    return setinput_lst, mera_metrics_lst, end_to_end_lst


def average_latency(lst):
    return sum(lst) / len(lst)


def get_time(mera_metrics_lst, k):
    extracted_mera_lst = [m[k] / 1000 for m in mera_metrics_lst]
    mera_result = average_latency(extracted_mera_lst)
    return mera_result


def summarize(setinput_lst, mera_metrics_lst, end_to_end_lst, sep):
    set_input_time = average_latency(setinput_lst)
    line = str(set_input_time)
    cpu_reorder_time = get_time(mera_metrics_lst, "reorder_in_time") + get_time(
        mera_metrics_lst, "reorder_out_time"
    )
    dma_time = get_time(mera_metrics_lst, "ip_in_time") + get_time(
        mera_metrics_lst, "ip_out_time"
    )
    ip_time = get_time(mera_metrics_lst, "ip_ip_time")

    line += sep + str(dma_time)
    line += sep + str(ip_time)
    line += sep + str(cpu_reorder_time)
    end_to_end = average_latency(end_to_end_lst)
    cpu_ops_time = end_to_end - (cpu_reorder_time + dma_time + ip_time)
    line += sep + str(cpu_ops_time)
    line += sep + str(end_to_end)
    return line


def run_deployment(
    model_path,
    rows,
    loop_count,
    sep,
    row_fetch=False,
    is_forever=False,
    is_llm=False,
):
    print(f"\nRunning {model_path} ...")
    size_dt = extract_size_dt(model_path)
    input_data = make_input(size_dt)

    try:
        ip = mera.load_mera_deployment(model_path, target=Target.IP)
    except Exception as e:
        raise ValueError("Failed to load model from", model_path, ":", e)

    device_target = DeviceTarget.SAKURA_2
    device_ids = 0 if is_forever else None
    predictor = ip.get_runner(
        device_target=device_target,
        device_ids=device_ids,
        dynamic_output_list=[0] if row_fetch else None,
    )
    setinput_lst, mera_metrics_lst, end_to_end_lst = [], [], []

    # --- actual running
    if is_forever:
        print("Running forever, press CTRL+C to exit...")
        run_model_forver(input_data, predictor)
    else:
        setinput_lst, mera_metrics_lst, end_to_end_lst = measure(
            input_data, predictor, loop_count
        )

    # --- summarize results
    line = summarize(setinput_lst, mera_metrics_lst, end_to_end_lst, sep)
    basename = os.path.basename(os.path.normpath(model_path))
    if basename == "mera_deployment":
        # Get second last component from normalized path
        components = os.path.normpath(model_path).split(os.sep)
        basename = components[-2] if len(components) >= 2 else basename
        llm_prefix = "[LLM] " if is_llm else ""
        basename = "[HF] " + llm_prefix + basename
    rows[basename] = line

    return rows


def print_csv(csv_file):
    import pandas as pd
    from tabulate import tabulate

    df = pd.read_csv(csv_file)
    table = [list(row) for row in df.values]
    # multi-line for last column
    headers = [col.replace(" ", "\n") for col in df.columns]

    print("\n\033[94mNote: All measurements are in milliseconds (ms)\033[0m")
    print(
        tabulate(
            table,
            headers=headers,
            tablefmt="psql",
            floatfmt=".4f",
            showindex=True,
            maxheadercolwidths=[None, None, 8, 8, 8, 8, 8, 10],
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./deployments",
        type=str,
        help="deployed model directory",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively search for models in subdirectories if enables",
    )
    parser.add_argument(
        "--loop_cnt",
        default=100,
        type=int,
        help="run the model x times to get average result. default = 100.",
    )
    parser.add_argument(
        "--name", default="", type=str, help="Prefixes for output file names"
    )
    parser.add_argument(
        "--run_forever",
        action="store_true",
        help="Run a single deployment forever, use for testing etc.",
    )
    parser.add_argument(
        "--row_fetch",
        action="store_true",
    )

    sep = ","
    arg = parser.parse_args()

    prefix = arg.name + "_" if arg.name else ""
    rows = collections.defaultdict(str)
    os.environ["MERA_PROFILING"] = "1"
    os.environ["MERA_BLOCKS_YOLOV5_CONF_THRESHOLD"] = "0.01"
    os.environ["MERA_BLOCKS_YOLOV8_CONF_THRESHOLD"] = "0.01"
    rows["Network"] = (
        "SetInput"
        + sep
        + "PCIe DMA"
        + sep
        + "DNA IP"
        + sep
        + "CPU Reorder"
        + sep
        + "CPU Ops"
        + sep
        + "EndToEnd (excl. SetInput)"
    )

    # --------- find all deployment paths ---------------------- #
    model_paths = []

    if arg.recursive:
        possible_paths = sorted(
            [Path(d) for d in find_dirs(arg.model_path, "deploy_*")]
        )
    else:
        possible_paths = [Path(arg.model_path)]
    for possible_dir in possible_paths:
        input_json_str = "model/input_desc.json"
        check_file1 = possible_dir / input_json_str
        check_file2 = possible_dir / "mera_deployment" / input_json_str
        if check_file1.is_file():
            model_paths.append(possible_dir)
        elif check_file2.is_file():
            model_paths.append(possible_dir / "mera_deployment")

    print("--- fetching the following deployment directories --")
    for model_path in model_paths:
        print(model_path)
    print("----------------------------------------------------")

    # ----------- run deployments ------------------------------ #
    if arg.run_forever and len(model_paths) != 1:
        raise ValueError("for running forever, need only 1 modelpath.")

    for model_path in model_paths:
        is_llm = (
            model_path.name == "mera_deployment"
            and (model_path.parent / "generation_config.json").is_file()
        )

        row_fetch = True if is_llm else arg.row_fetch
        rows = run_deployment(
            model_path=model_path,
            rows=rows,
            loop_count=arg.loop_cnt,
            sep=sep,
            row_fetch=row_fetch,
            is_forever=arg.run_forever,
            is_llm=is_llm,
        )

    # ----------- save results --------------------------------- #
    if len(rows) > 1:
        output_report = prefix + "latencies.csv"
        with open(output_report, "w") as f:
            for k, v in rows.items():
                f.write(k + sep + v)
                f.write("\n")
            print("\nResults saved to " + output_report)
        print_csv(output_report)
    else:
        print("Nothing to save")
