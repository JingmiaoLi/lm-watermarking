"""
Step 1: Entropy-based filtering:

For the entropy-based experiments, first use this script (step1) to obtain entropy values and green masks. 
Then, use step2 to recalculate z-scores with different entropy_thresholds. 

Or this script can also be used with a entropy threshold value directly. 

- Token length filtering: skip entries with <190 tokens, truncate entries >200 tokens.
- Batch processing and optional entropy threshold filtering.
- Resumable processing via checkpoint logs.
- Multi-threshold detection and per-threshold summary CSV output.
- Optionally returning token-level and bigram-level green mask information.
- Obtain the entropy values and green masks 

Example Usage:
python my_watermark_detection_entropy_step1.py \
--folder_path ./data \
--gamma 0.25 \
--tokenizer meta-llama/Llama-3.1-8B \
--ignore_repeated_bigrams \
--max_tokens 200 \
--return_green_token_mask 

Author:
Jingmiao Li
"""

import torch
import logging
import csv
import os
import gc
import glob
import gzip
import jsonlines
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from my_watermark_processor2_entropy import WatermarkDetector

logging.basicConfig(
    filename="single_process_multi_gpu_copy1.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def load_data(json_file_path):
    with gzip.open(json_file_path, 'rt', encoding='utf-8') as f:
        reader = jsonlines.Reader(f)
        for obj in reader:
            yield obj

def count_records(file_path, tokenizer):
    count = 0
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        reader = jsonlines.Reader(f)
        for obj in reader:
            text = obj.get('text') or obj.get('watermarked_completion')
            if not text:
                continue
            tokens = tokenizer.encode(text, truncation=False)
            if len(tokens) >= 190:
                count += 1
    return count

def get_checkpoint_for_file(file_path):
    checkpoint = 0
    log_filename = "single_process_multi_gpu.log"
    if os.path.exists(log_filename):
        try:
            with open(log_filename, "r", encoding="utf-8") as log_file:
                for line in log_file:
                    if file_path in line and "Processed" in line and "valid entries" in line:
                        m = re.search(r"Processed (\d+)/", line)
                        if m:
                            i_val = int(m.group(1))
                            if i_val > checkpoint:
                                checkpoint = i_val
        except Exception as e:
            logging.error(f"Error reading checkpoint from log: {str(e)}")
    return checkpoint

def initialize_detector(tokenizer, gamma, device, ignore_repeated_bigrams):
    vocab = list(tokenizer.get_vocab().values())
    detector = WatermarkDetector(
        vocab=vocab,
        tokenizer=tokenizer,
        gamma=gamma,
        z_threshold=4.0,
        device=device,
        ignore_repeated_bigrams=ignore_repeated_bigrams
    )
    return detector

def process_single_file(file_path, tokenizer, model, detector, args, device):
    """
    Processes a .json.gz file, computes entropy and green mask for each text,
    and records z-score, p-value, prediction, and other information to a CSV file.
    Supports checkpointing for resuming progress.

    Logic modifications:
    - Skip records with fewer than 190 tokens.
    - Truncate texts longer than 200 tokens to 200 tokens.
    - The calling code only reads and truncates texts, collects them into batches,
        and passes them to the detector.
    - The detector handles tokenization, padding, and attention masks internally.
    - After processing each batch of valid records, detection results are written to CSV
        and statistics logs are updated.
    """

    logging.info(f"Starting to process file: {file_path}")

    try:
        total_records = count_records(file_path, tokenizer)
    except Exception as e:
        logging.error(f"Error counting records in {file_path}: {str(e)}")
        return

    logging.info(f"Total valid records in file {file_path} (num_token>=190): {total_records}")


    checkpoint = get_checkpoint_for_file(file_path)
    logging.info(f"Resuming processing from valid record index {checkpoint+1} for file {file_path}")

    thresholds = [None]

    csv_writers = {}
    csv_files = {}
    csv_header = [
        "idx", "text", "prediction", "z_score", "p_value",
        "num_tokens", "num_green_tokens", "green_token_fraction"
    ]
    if args.return_green_token_mask:
        csv_header.extend(["bigram_entropy_mask", "token_entropy_mask"])

    for thr in thresholds:
        thr_str = "none" if thr is None else thr
        csv_filename = f'detection_results_{os.path.basename(file_path)}_entr_{thr_str}.csv'
        f = open(csv_filename, mode='a' if checkpoint > 0 else 'w', newline='', encoding='utf-8')
        writer = csv.writer(f)
        if checkpoint == 0:
            writer.writerow(csv_header)
        csv_writers[thr] = writer
        csv_files[thr] = f

    summary_counts = {thr: {"true_count": 0, "false_count": 0} for thr in thresholds}
    record_buffer = {thr: [] for thr in thresholds}


    batch_size = 10 
    batch_texts = []  
    batch_indices = []  

    valid_count = 0  

    for raw_index, document in enumerate(load_data(file_path), start=1):
        text = document.get('text') or document.get('watermarked_completion')
        if not text:
            logging.warning(f"Document at raw index {raw_index} in file {file_path} has no text. Skipping.")
            continue

        # 获取 token 数量，不进行截断
        tokens = tokenizer.encode(text, truncation=False)
        token_len = len(tokens)
        if token_len < 190:
            logging.info(f"Document at raw index {raw_index} skipped due to token count {token_len} < 190.")
            continue

        valid_count += 1
        # 断点续跑：如果该有效记录已处理，则跳过
        if valid_count <= checkpoint:
            continue

        if token_len > 200:
            tokens = tokens[:200]
            text = tokenizer.decode(tokens, skip_special_tokens=True)

        batch_texts.append(text)
        batch_indices.append(valid_count)

        if len(batch_texts) >= batch_size:
            detection_results = detector.detect(
                text=batch_texts,
                model=model,
                tokenizer=tokenizer,
                device=device,
                return_green_token_mask=args.return_green_token_mask
            )

            for thr in thresholds:
                for i, single_result in enumerate(detection_results[thr]):
                    num_tokens_scored = single_result.get('num_tokens_scored', 1)
                    num_green_tokens = single_result.get('num_green_tokens', 0)
                    green_fraction = (num_green_tokens / num_tokens_scored) if num_tokens_scored > 0 else 0
                    prediction = single_result.get('prediction')
                    if prediction is True:
                        summary_counts[thr]["true_count"] += 1
                        logging.info(f"True entry detected: idx {batch_indices[i]}, z_score: {single_result.get('z_score')}, p_value: {single_result.get('p_value')}")
                    else:
                        summary_counts[thr]["false_count"] += 1

                    row_data = [
                        batch_indices[i],
                        batch_texts[i],
                        prediction,
                        single_result.get('z_score'),
                        single_result.get('p_value'),
                        num_tokens_scored,
                        num_green_tokens,
                        green_fraction,
                    ]
                    if args.return_green_token_mask:
                        row_data.append(single_result.get('bigram_entropy_mask'))
                        row_data.append(single_result.get('token_entropy_mask'))
                    record_buffer[thr].append(row_data)

     
            for thr in thresholds:
                if record_buffer[thr]:
                    csv_writers[thr].writerows(record_buffer[thr])
                    record_buffer[thr].clear()
    
            for thr in thresholds:
                thr_str = "none" if thr is None else thr
                true_count = summary_counts[thr]["true_count"]
                false_count = summary_counts[thr]["false_count"]
                logging.info(f"Processed {batch_indices[-1]}/{total_records} valid entries in file {file_path}: threshold {thr_str} -> true_count = {true_count}, false_count = {false_count}")
  
            batch_texts = []
            batch_indices = []


    if batch_texts:
        detection_results = detector.detect(
            text=batch_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_green_token_mask=args.return_green_token_mask
        )
        for thr in thresholds:
            for i, single_result in enumerate(detection_results[thr]):
                num_tokens_scored = single_result.get('num_tokens_scored', 1)
                num_green_tokens = single_result.get('num_green_tokens', 0)
                green_fraction = (num_green_tokens / num_tokens_scored) if num_tokens_scored > 0 else 0
                prediction = single_result.get('prediction')
                if prediction is True:
                    summary_counts[thr]["true_count"] += 1
                    logging.info(f"True entry detected: idx {batch_indices[i]}, z_score: {single_result.get('z_score')}, p_value: {single_result.get('p_value')}")
                else:
                    summary_counts[thr]["false_count"] += 1
                row_data = [
                    batch_indices[i],
                    batch_texts[i],
                    prediction,
                    single_result.get('z_score'),
                    single_result.get('p_value'),
                    num_tokens_scored,
                    num_green_tokens,
                    green_fraction,
                ]
                if args.return_green_token_mask:
                    row_data.append(single_result.get('bigram_entropy_mask'))
                    row_data.append(single_result.get('token_entropy_mask'))
                record_buffer[thr].append(row_data)


    for thr in thresholds:
        if record_buffer[thr]:
            csv_writers[thr].writerows(record_buffer[thr])
            record_buffer[thr].clear()

    summary_csv_filename = f'detection_summary_{os.path.basename(file_path)}.csv'
    with open(summary_csv_filename, mode='w', newline='', encoding='utf-8') as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_header = ["entropy_threshold", "true_count", "false_count", "total_records_processed"]
        summary_writer.writerow(summary_header)
        for thr in thresholds:
            true_count = summary_counts[thr]["true_count"]
            false_count = summary_counts[thr]["false_count"]
            total = true_count + false_count
            thr_str = "none" if thr is None else thr
            summary_writer.writerow([thr_str, true_count, false_count, total])
            logging.info(f"Finished processing file {file_path}. For threshold {thr_str}: true_count = {true_count}, false_count = {false_count}, total_records_processed: {valid_count}.")

    for f in csv_files.values():
        f.close()

    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Single-Process Multi-GPU Watermark Detection with modified truncation logic and checkpoint resume.")
    parser.add_argument('--folder_path', type=str, required=True, help="Folder containing .json.gz files.")
    parser.add_argument('--gamma', type=float, required=True, help="Gamma value for watermark detection.")
    parser.add_argument('--tokenizer', type=str, default='gpt2', help="Tokenizer model (default: gpt2).")
    parser.add_argument('--ignore_repeated_bigrams', action='store_true', help="Ignore repeated bigrams in detection.")
    parser.add_argument('--max_tokens', type=int, default=None, help="Max tokens per text (will be overridden by truncation logic).")
    parser.add_argument('--entropy_threshold', type=float, nargs='+', help="Entropy threshold(s) for filtering tokens. 可传入多个值。")
    parser.add_argument('--return_green_token_mask', action='store_true',
                        help="Whether to return green token mask (bigram_entropy_mask, token_entropy_mask).")
    args = parser.parse_args()

    file_paths = glob.glob(os.path.join(args.folder_path, "*.json.gz"))
    if not file_paths:
        raise ValueError(f"No .json.gz files found in {args.folder_path}.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    model = AutoModelForCausalLM.from_pretrained(
        args.tokenizer,
        device_map="auto",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    detector = initialize_detector(tokenizer, args.gamma, device, args.ignore_repeated_bigrams)

    for file_path in file_paths:
        summary_csv_filename = f'detection_summary_{os.path.basename(file_path)}.csv'
        if os.path.exists(summary_csv_filename):
            logging.info(f"File {file_path} has already been finished. Skipping.")
            continue
        process_single_file(file_path, tokenizer, model, detector, args, device)

    logging.info("All files processed in single-process multi-GPU mode.")

if __name__ == "__main__":
    main()

