# -*- coding: utf-8 -*-
"""
2. Running Text Generation
==========================

"""


from os.path import isfile, join
import argparse

import torch
import mera_models as mm
from transformers import AutoTokenizer


def load_predictor(model_path, target_str, device_str):
    """Return predictor based on either onnxruntime or meraruntime"""

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
    """Generate predictions using the loaded predictor"""

    pred = predictor.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=50726,
        do_sample=True,
        temperature=0.8,
    )

    return pred


def is_cls_onnx(predictor):
    """Check if predictor is using ONNX runtime"""
    return "ORTModel" in predictor.__class__.__name__


def _apply_template(prompt, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_inference(prompt, predictor, tokenizer, model_flags):
    """Run inference using the predictor & tokenizer"""

    # Convert model flags to dictionary
    flags = mm.utils.flags_to_dict(model_flags)
    max_new_tokens = int(flags.get("max_new_tokens", 40))  # max output token generated
    is_template = bool(int(flags.get("template", 0)))  # 1 means enable chat template

    if is_template:
        prompt = _apply_template(prompt, tokenizer)

    print(f"token eos: {tokenizer.eos_token_id}")

    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate & decode predictions
    gen_tokens = run_predictor(inputs, predictor, max_new_tokens)
    decoded_data = tokenizer.batch_decode(gen_tokens)

    return decoded_data[0]


def run_inference_with_pipeline(prompt, predictor, tokenizer, model_flags):
    """Run inference using pipeline api"""

    from mera_models.hf.pipelines import pipeline

    # Convert model flags to dictionary
    flags = mm.utils.flags_to_dict(model_flags)
    max_new_tokens = int(flags.get("max_new_tokens", 40))  # max output token generated
    is_template = bool(int(flags.get("template", 0)))  # 1 means enable chat template

    if is_template:
        prompt = _apply_template(prompt, tokenizer)

    generator = pipeline(
        task="text-generation",
        model=predictor,
        tokenizer=tokenizer,
    )

    # sample default arguments
    out_dt = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.8,
    )

    return out_dt[0]["generated_text"]


def main(arg):
    """Main function to handle model demo"""

    seed = 1337  # to generate reproducible results
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Fill in input text
    prompt = "My name is Arthur and I live in" if not arg.input_txt else arg.input_txt

    # Load stuff
    tokenizer = AutoTokenizer.from_pretrained(arg.model_path)
    predictor = load_predictor(arg.model_path, arg.target, arg.device)

    print(f"Prompt: {prompt}")

    # Run actual inference
    if arg.pipeline:
        txt = run_inference_with_pipeline(
            prompt=prompt,
            predictor=predictor,
            tokenizer=tokenizer,
            model_flags=arg.model_flags,
        )
    else:
        txt = run_inference(
            prompt=prompt,
            predictor=predictor,
            tokenizer=tokenizer,
            model_flags=arg.model_flags,
        )

    print(f"Result text: {txt}")

    # Save generated text as image file
    mm.utils.save_text_as_image_file(
        arg.save_path,
        txt,
        max_char_per_line=52,
        font_scale=0.6,
    )


def get_args():
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_txt",
        default="What is the capital city of Japan?",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./result.png",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="./deploy_SmolLM2-135M-Instruct",
        # default="HuggingFaceTB__SmolLM2-135M-Instruct_onnx",
        type=str,
    )
    parser.add_argument(
        "--model_flags",
        default="max_new_tokens=40,template=1",
        type=str,
        help="please see the inference function for meaning of each flag.",
    )
    parser.add_argument(
        "--target",
        default="ip",
        type=str,
        help="MERA Target environment, ip is default",
    )
    parser.add_argument("--device", default="sakura2", type=str)
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use transformers pipeline to run inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
