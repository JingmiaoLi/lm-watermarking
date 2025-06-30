"""
Batch Watermarking Script for LLaMA and GPT-style Language Models

This script loads a specified language model and tokenizer to generate watermarked text 
from input prompts stored in compressed JSON files. It supports configurable parameters 
for sampling, entropy-based watermarking (gamma and delta), batch processing, and GPU usage.

Key Features:
- Processes multiple .json.gz input files from a directory.
- Supports configurable sampling strategies and temperature.
- Applies a watermark logits processor to embed watermarks in generated text.
- Outputs results to a CSV file including original prompts and generated completions.
- Compatible with LLaMA, GPT-2, and other Hugging Face causal language models.

Usage:
    python my_generate.py --input_dir /path/to/input --output_csv /path/to/output.csv \
        --model_name_or_path /path/to/model --max_new_tokens 50 \
        --gamma 0.25 --delta 2.0 --use_sampling True --use_gpu True

Author:
    Jingmiao Li, adapted from the original lm-watermarking repository (MIT License).
"""

import os
import gzip
import json
import argparse
import logging
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from watermark_processor import WatermarkLogitsProcessor
import torch


def str2bool(v):
    """Utility function to parse boolean arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch generate watermarked text using LLaMA")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing .json.gz files")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file with watermarked texts")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or Hugging Face model identifier")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Maximum number of tokens to generate")
    parser.add_argument("--prompt_max_length", type=int, default=None, help="Maximum length of the prompt")
    parser.add_argument("--use_sampling", type=str2bool, default=True, help="Whether to use sampling (True) or greedy decoding (False)")
    parser.add_argument("--sampling_temp", type=float, default=0.7, help="Sampling temperature for multinomial sampling")
    parser.add_argument("--gamma", type=float, default=0.25, help="Fraction of vocabulary to partition into the greenlist")
    parser.add_argument("--delta", type=float, default=2.0, help="Logits bias added to greenlist tokens")
    parser.add_argument("--generation_seed", type=int, default=123, help="Random seed for text generation")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of prompts to process in one batch")
    parser.add_argument("--use_gpu", type=str2bool, default=True, help="Whether to use GPU for inference")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """Load the LLaMA model and tokenizer."""
    # Initialize device configuration
    if torch.cuda.is_available() and args.use_gpu:
        device = "cuda"
        print(f"Running on CUDA device(s): {torch.cuda.device_count()} available.")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    elif torch.backends.mps.is_available() and args.use_gpu:
        device = "mps"
        print("Running on MPS (Apple Silicon).")
    else:
        device = "cpu"
        print("CUDA and MPS are not available. Running on CPU.")

    # Load model with device_map for efficient multi-GPU handling
    if device == "cuda" and torch.cuda.device_count() > 1:
        print("Using device_map='auto' to utilize multiple GPUs.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    # Ensure the tokenizer has a pad_token
    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Set padding_side to 'left' for decoder-only models
    tokenizer.padding_side = "left"

    return model, tokenizer, device
def generate_batch(prompts, model, tokenizer, device, args):
    """Generate watermarked text for a batch of prompts with dynamic truncation."""
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme="simple_1",
        select_green_tokens=True
    )

    # Dynamically determine the maximum length for the prompt (Gradio logic)
    if args.prompt_max_length:
        dynamic_max_prompt_length = args.prompt_max_length
    elif hasattr(model.config, "max_position_embeddings"):
        dynamic_max_prompt_length = max(1, model.config.max_position_embeddings - args.max_new_tokens)
    else:
        dynamic_max_prompt_length = max(1, 2048 - args.max_new_tokens)

    # Tokenize input prompts with padding and truncation
    tokd_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=True,  # 保证和 Gradio 一致
        truncation=True,
        padding=True,  # 保证批量生成时张量形状一致
        max_length=dynamic_max_prompt_length
    ).to(device)

    # Debugging logs
    logging.info(f"dynamic_max_prompt_length: {dynamic_max_prompt_length}")
    for prompt, tokenized_prompt in zip(prompts, tokd_inputs["input_ids"]):
        logging.info(f"Original prompt: {prompt}")
        logging.info(f"Tokenized prompt length: {len(tokenized_prompt)}")
        logging.info(f"Decoded prompt: {tokenizer.decode(tokenized_prompt, skip_special_tokens=True)}")

    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.use_sampling,
        "temperature": args.sampling_temp if args.use_sampling else None,
        "num_beams": 1 if not args.use_sampling else None
    }

    # Apply watermark logits processor
    logits_processor = LogitsProcessorList([watermark_processor])

    # Set seed for reproducibility
    torch.manual_seed(args.generation_seed)

    # Generate watermarked text
    outputs = model.generate(**tokd_inputs, logits_processor=logits_processor, **gen_kwargs)

    # Extract only newly generated tokens
    generated_texts = tokenizer.batch_decode(
        outputs[:, tokd_inputs['input_ids'].shape[-1]:], skip_special_tokens=True
    )
    # Decode the actual prompts used
    used_prompts = tokenizer.batch_decode(tokd_inputs["input_ids"], skip_special_tokens=True)

    return generated_texts, used_prompts

def process_directory(input_dir, output_csv, model, tokenizer, device, args):
    """Process all .json.gz files in the input directory and save watermarked outputs to a CSV file."""
    # Get all .json.gz files in the directory
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json.gz")]

    if not input_files:
        logging.warning(f"No .json.gz files found in the directory: {input_dir}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Prepare output file
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["idx", "prompt", "watermarked_completion"])  # Single idx column

        idx = 1  # Start single index for all records
        for input_file in input_files:
            logging.info(f"Processing file: {input_file}")

            # Open and read each .json.gz file
            with gzip.open(input_file, "rt", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            # Extract prompts from the data
            prompts = []
            for item in data:
                if "text" in item:
                    tokenized_prompt = tokenizer(
                        item["text"],
                        truncation=True,
                        max_length=args.prompt_max_length,
                        return_tensors="pt",
                        add_special_tokens=True,  # Ensure consistency
                    )
                    decoded_prompt = tokenizer.decode(tokenized_prompt["input_ids"][0], skip_special_tokens=True).strip()
                    if decoded_prompt:
                        prompts.append(decoded_prompt)

            # Ensure prompts are not empty
            if not prompts:
                logging.warning(f"No valid prompts found in the file: {input_file}")
                logging.warning(f"File content: {data}")
                continue

            # Process in batches
            for i in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[i:i + args.batch_size]
                logging.info(f"Processing batch {i // args.batch_size + 1} of file {input_file}")
                watermarked_texts, used_prompts = generate_batch(batch_prompts, model, tokenizer, device, args)

                # Write results to CSV
                for prompt, watermarked_text in zip(used_prompts, watermarked_texts):
                    writer.writerow([idx, prompt, watermarked_text])
                    idx += 1

def main():
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="generation_log.log",  # Log to a file
        filemode="w"  # Overwrite the log file for each run
    )

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args)

    # Process prompts and generate watermarked text
    process_directory(args.input_dir, args.output_csv, model, tokenizer, device, args)


if __name__ == "__main__":
    main()




