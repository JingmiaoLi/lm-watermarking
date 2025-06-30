"""
KGW baseline detection: 

Main Detection Features:
- Optionally ignores repeated bigrams during scoring.
- Filters texts by token length (190–200 tokens).
- Outputs per-sample detection results to CSV files.
- Supports multi-GPU processing for parallel detection.

Example Usage:
python my_watermark_detection_baseline.py \
--folder_path ./data \
--gamma 0.25 \
--tokenizer meta-llama/Llama-3.1-8B \
--ignore_repeated_bigrams \
--max_tokens 200 \
--device cuda

Author:
Jingmiao Li
"""


import torch
import logging
import csv
import os
import gzip
import jsonlines
from transformers import AutoTokenizer
from watermark_processor import WatermarkDetector
import argparse
import gc
import glob
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


# Logging setup
logging.basicConfig(
    filename="process_log_llama2_190_200.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

import os
import re
import logging

def get_resume_checkpoints(log_file):

    checkpoints = {}
    pattern = re.compile(r"Checkpoint: File (.+), Processed (\d+)/(\d+) entries")

    if not os.path.exists(log_file):
        logging.info(f"Log file {log_file} not found. Starting fresh.")
        return checkpoints

    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                file_name = match.group(1)
                processed_entries = int(match.group(2))  # 已处理条目数
        
                checkpoints[file_name] = max(checkpoints.get(file_name, 0), processed_entries)

    return checkpoints

# Initialize watermark detector
def initialize_detector(tokenizer, gamma, device, ignore_repeated_bigrams, seed=None):
    vocab = list(tokenizer.get_vocab().values())
    rng = torch.Generator(device=device).manual_seed(seed) if seed else torch.Generator(device=device)
    detector = WatermarkDetector(
        vocab=vocab,
        tokenizer=tokenizer,
        gamma=gamma,
        z_threshold=4.0,
        device=device,
        ignore_repeated_bigrams=ignore_repeated_bigrams
    )
    detector.rng = rng
    return detector

# Load data from a json.gz file
def load_data(json_file_path):
    with gzip.open(json_file_path, 'rt', encoding='utf-8') as f:
        reader = jsonlines.Reader(f)
        for obj in reader:
            yield obj

# Process a single file
def process_single_file(file_path, rank, args, checkpoints=None):
    device = torch.device(f"cuda:{rank}")
    logging.info(f"Starting to process file: {file_path} on GPU {rank}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    detector = initialize_detector(tokenizer, args.gamma, device, args.ignore_repeated_bigrams, args.seed)
    
    csv_filename = f'detection_results_{args.gamma}_{os.path.basename(file_path)}_ignore_true_max_{args.max_tokens}.csv'
    total_records = 0

    # Count total records in the file
    try:
        total_records = sum(1 for _ in load_data(file_path))
    except Exception as e:
        logging.error(f"Error counting records in {file_path}: {str(e)}")
        return

    logging.info(f"Total records in file {file_path}: {total_records}")
    
    processed_count = checkpoints.get(file_path, 0) if checkpoints else 0
    valid_count = 0  # Only count entries with 190-200 tokens
    true_count = 0
    skipped_short = 0
    skipped_missing = 0

    try:
        with open(csv_filename, mode='a' if processed_count > 0 else 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if processed_count == 0:
                writer.writerow(["file id", "total entries", "text", "prediction", "z_score", "p_value", "num_tokens", "num_green_tokens", "green_fraction"])
            
            data_iterator = load_data(file_path)
            for i, document in enumerate(data_iterator, start=1):
                text = document.get('text') or document.get('watermarked_completion')
                if not text:
                    skipped_missing += 1
                    logging.warning(f"Document {i} in file {file_path} has no text. Skipping.")
                    continue

                # Get token length without truncation
                tokens = tokenizer.encode(text, truncation=False)
                token_len = len(tokens)

                if token_len < 190:
                    skipped_short += 1
                    logging.info(f"Document {i} in file {file_path} skipped due to token count {token_len} < 190.")
                    continue

                valid_count += 1
                if valid_count <= processed_count:
                    continue

                if token_len > 200:
                    tokens = tokens[:200]
                    text = tokenizer.decode(tokens, skip_special_tokens=True)

                try:
                    detection_result = detector.detect(text=text)
                    if detection_result.get('prediction') is True:
                        true_count += 1
                        num_tokens = detection_result.get('num_tokens_scored', 0)
                        num_green_tokens = detection_result.get('num_green_tokens', 0)
                        green_fraction = num_green_tokens / num_tokens if num_tokens > 0 else 0.0

                        writer.writerow([
                            os.path.basename(file_path), total_records, text,
                            detection_result.get('prediction'),
                            detection_result.get('z_score'),
                            detection_result.get('p_value'),
                            num_tokens,
                            num_green_tokens,
                            green_fraction
                        ])

                    processed_count += 1
                    if processed_count % 100 == 0:
                        logging.info(f"Checkpoint: File {file_path}, Processed {processed_count}/{total_records} entries.")
                except Exception as e:
                    logging.error(f"Error processing document {i} in file {file_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Error writing results for file {file_path}: {str(e)}")
    
    finally:
        gc.collect()
        logging.info(f"Finished processing file {file_path} on GPU {rank}. True detections: {true_count}. Valid count: {valid_count}. Skipped (short): {skipped_short}. Skipped (missing text): {skipped_missing}.")

# Process files on a single GPU
def process_on_single_gpu(rank, file_chunks, args, checkpoints):
    torch.cuda.set_device(rank)
    logging.info(f"GPU {rank} started with {len(file_chunks[rank])} files.")
    
    assigned_files = file_chunks[rank]
    print(f"[GPU {rank}] Starting process with {len(file_chunks[rank])} files.")


    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(lambda file: process_single_file(file, rank, args, checkpoints), assigned_files)

def distribute_files(file_paths, num_gpus):
    chunks = [[] for _ in range(num_gpus)]
    for idx, file_path in enumerate(file_paths):
        chunks[idx % num_gpus].append(file_path)
    return chunks

# Main multiprocessing entry
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process watermark detection on large datasets.")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder containing the JSON files.")
    parser.add_argument('--gamma', type=float, required=True, help="Gamma value for watermark detection.")
    parser.add_argument('--tokenizer', type=str, default='gpt2', help="Tokenizer model to use (default: gpt2).")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help="Device to run the detection on (default: cuda).")
    parser.add_argument('--ignore_repeated_bigrams', action='store_true', help="Whether to ignore repeated bigrams in the watermark detection process.")
    parser.add_argument('--max_tokens', type=int, default=None, help="Maximum number of tokens to process per text entry. Texts longer than this will be truncated.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for consistent watermark detection.")
    parser.add_argument('--log_file', type=str, default="process_log_llama2_190_200.log", help="Path to the log file for resuming progress.")
    
    args = parser.parse_args()

    file_paths = glob.glob(os.path.join(args.folder_path, "*.json.gz"))
    if not file_paths:
        raise ValueError(f"No files found in {args.folder_path} matching '*.json.gz'")

    num_gpus = torch.cuda.device_count()
    file_chunks = distribute_files(file_paths, num_gpus)
    print(f"[Main] Found {len(file_paths)} files: {file_paths}")

    checkpoints = get_resume_checkpoints(args.log_file)

    mp.spawn(
        process_on_single_gpu,
        args=(file_chunks, args, checkpoints),
        nprocs=num_gpus,
        join=True
    )
    logging.info("All processes completed.")
