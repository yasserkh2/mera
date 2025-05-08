import os
import sys
from os.path import isfile, join

import numpy as np
import torch
import mera_models as mm


def load_predictor(model_path, target_str, device_str):
    check_file = join(model_path, "model.onnx")
    if isfile(check_file):
        # actual onnx api class
        from optimum.onnxruntime import ORTModelForCausalLM

        print(f"running inference on huggingface ONNX runtime...")
        predictor = ORTModelForCausalLM.from_pretrained(
            model_path,
            use_cache=False,
            use_io_binding=False,
        )
    else:
        # MERA load
        from mera_models.hf.meraruntime import MERAModelForCausalLM as mf

        print(mf)
        predictor = mf.from_pretrained(
            model_path,
            target=target_str,
            device_target=device_str,
            measure_latency=True,
        )

    return predictor


def run_predictor(inputs, predictor, max_new_tokens):
    # onnx or MERA runtime, same API
    pred = predictor.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=50726,
        do_sample=True,
        temperature=0.8,
    )

    return pred


def is_cls_onnx(predictor):
    return "ORTModel" in predictor.__class__.__name__


def _apply_template(prompt, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_inference(inputs, predictor, tokenizer, model_flags):
    # --- take care of model flags
    flags = mm.utils.flags_to_dict(model_flags)
    max_new_tokens = int(flags.get("max_new_tokens", 40))
    is_template = bool(flags.get("template", 0))  # 0 or 1

    if is_template:
        inputs = _apply_template(inputs, tokenizer)

    print(f"token eos: {tokenizer.eos_token_id}")

    inputs = tokenizer(inputs, return_tensors="pt")

    gen_tokens = run_predictor(inputs, predictor, max_new_tokens)
    decoded_data = tokenizer.batch_decode(gen_tokens, skip_special_tokens=False)

    return decoded_data[0]


def add_img_text(
    image,
    txt,
    pos_y,
    margin=30,
    fontScale=0.6,
    line_width=20,
    line_spacing=25,
    thickness=1,
    color=(255, 255, 255),
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_x, txt_y = margin, pos_y

    line = ""
    for word in txt.split(" "):
        line = line + f" {word}" if line != "" else word
        if len(line) >= line_width:
            image = cv2.putText(
                image,
                line,
                (txt_x, txt_y),
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            txt_y += line_spacing
            line = ""
    image = cv2.putText(
        image, line, (txt_x, txt_y), font, fontScale, color, thickness, cv2.LINE_AA
    )
    txt_y += line_spacing
    return txt_y, image


def get_overlay(txt):
    overlay = np.zeros((300, 800, 3), dtype=np.uint8)
    pos_y, overlay = add_img_text(
        overlay, txt, 30, margin=30, fontScale=0.6, line_width=68, line_spacing=30
    )
    return overlay


def run_inference_with_pipeline(prompt, predictor, tokenizer, model_flags):
    from mera_models.hf.pipelines import pipeline

    flags = mm.utils.flags_to_dict(model_flags)
    max_new_tokens = int(flags.get("max_new_tokens", 40))
    is_template = bool(flags.get("template", 0))  # 0 or 1

    if is_template:
        prompt = _apply_template(prompt, tokenizer)

    generator = pipeline(
        task="text-generation",
        model=predictor,
        tokenizer=tokenizer,
    )

    out_dt = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        pad_token_id=50726,
        do_sample=True,
        temperature=0.8,
    )

    return out_dt[0]["generated_text"]


def main(arg):
    seed = 1337  # will produce same result with same seed due to torch sampling
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    from transformers import AutoTokenizer

    # -- fill in input text
    prompt = "My name is Arthur and I live in" if not arg.input_txt else arg.input_txt

    # -- load stuff
    tokenizer = AutoTokenizer.from_pretrained(arg.model_path)
    predictor = load_predictor(arg.model_path, arg.target, arg.device)

    print(f"prompt: {prompt}")

    # -- run actual inference
    if arg.pipeline:
        txt = run_inference_with_pipeline(
            prompt=prompt,
            predictor=predictor,
            tokenizer=tokenizer,
            model_flags=arg.model_flags,
        )
    else:
        txt = run_inference(
            inputs=prompt,
            predictor=predictor,
            tokenizer=tokenizer,
            model_flags=arg.model_flags,
        )

    print(f"result txt: {txt}")

    mm.utils.save_text_as_image_file(
        arg.save_path,
        txt,
        max_char_per_line=52,
        font_scale=0.6,
    )

    # Measure estimated latency
    if arg.target.lower() in ["simulatorbf16", "simulator"]:
        print(f" ** Estimated {arg.target} latency {predictor.estimated_latency} ms")

    # Measure estimated latency
    if arg.target == "SimulatorBf16":
        print(f" ** Estimated SimulatorBf16 latency {predictor.estimated_latency} ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_txt",
        default="Tell me about Japan",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./result.png",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="./173_smollm2-1.7b-instruct-tk256",
        # default="./source_model_files/gpt2_onnx/",
        type=str,
    )
    parser.add_argument(
        "--model_flags",
        default="a=0,max_new_tokens=2048,template=1",
        type=str,
        help="please see the inference function for meaning of each flag.",
    )
    parser.add_argument(
        "--target",
        default="ip",
        type=str,
        help="MERA Target environment, bf16 is (default)",
    )
    parser.add_argument("--device", default="sakura2", type=str)
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use transformers pipeline to run inference",
    )
    main(parser.parse_args())
