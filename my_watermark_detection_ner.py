"""
Ner-based Weighting: 

This script processes all CSV files in a specified folder to detect watermarks in text columns.
It loads a pretrained language model and tokenizer to tokenize input texts and apply a watermark detector
with optional Named Entity Recognition (NER) to adjust token weighting.

Key Features:
- Supports processing multiple CSV files in batch.
- Compatible with any Hugging Face AutoTokenizer and AutoModelForCausalLM.
- Optionally adjusts token contributions for named entities via entity_scale.

Example usage:
python my_watermark_detection_ner.py \
    --folder_path ./data \
    --gamma 0.25 \
    --tokenizer gpt2 \
    --device cuda \
    --max_tokens 500 \
    --entity_scale 0.5 \
    --use_ner

Author:
Jingmiao Li
"""


import torch
import pandas as pd
from my_watermark_processor1_ner import WatermarkDetector
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import csv
import os
import argparse

# Set up logging
logging.basicConfig(
    filename="process_log_ner.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Load the detector and tokenizer
def initialize_detector(tokenizer, gamma, device, entity_scale=0.5, use_ner=True):
    # Get vocab from tokenizer
    vocab = list(tokenizer.get_vocab().values())
    detector = WatermarkDetector(
        vocab=vocab,
        tokenizer=tokenizer,
        gamma=gamma,
        z_threshold=4.0,
        device=device,
        use_ner=use_ner,
        entity_scale=entity_scale
    )
    return detector

def load_data(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            logging.info(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df.columns = df.columns.str.lower()  # Convert columns to lowercase for consistency
                
                # 检测列名是否存在
                column_name = None
                if 'text' in df.columns:
                    column_name = 'text'
                elif 'watermarked_completion' in df.columns:
                    column_name = 'watermarked_completion'
                else:
                    logging.error(f"CSV file {file_path} does not contain 'text' or 'watermarked_completion' column.")
                    continue  # 跳过当前文件
                
                # 遍历每一行数据
                for _, row in df.iterrows():
                    yield {
                        'idx': row.get('idx', None), 
                        'file_id': row.get('file_name', file_name),  # 取当前文件名作为 file_id
                        'text': row[column_name]  # 动态获取列名的数据
                    }
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

# Count number of records in the CSV file
def count_records(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    return len(df)

def process_folder(folder_path, gamma, tokenizer, model, device, max_tokens, entity_scale, use_ner):
    logging.info(f"Starting to process folder: {folder_path} with gamma={gamma}, entity_scale={entity_scale}, use_ner={use_ner}")
    
    detector = initialize_detector(tokenizer, gamma, device, entity_scale, use_ner)

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            result_file = f"ner_{entity_scale}_{os.path.splitext(file_name)[0]}.csv"
            logging.info(f"Processing file: {file_path} -> Result file: {result_file}")

            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df.columns = df.columns.str.lower()
                
                # 检测列名
                column_name = 'text' if 'text' in df.columns else 'watermarked_completion' if 'watermarked_completion' in df.columns else None
                if column_name is None:
                    logging.error(f"File {file_name} lacks 'text' or 'watermarked_completion' column.")
                    continue

                # 初始化 CSV 文件并写入标题
                with open(result_file, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        "idx", "text", "prediction", "z_score", "p_value",
                        "num_tokens", "num_green_tokens", "green_token_fraction",
                        "ner_entities", "green_token_mask"
                    ])
                    
                    true_count, false_count = 0, 0
                    
                    for i, row in df.iterrows():
                        text = row[column_name]
                        idx = row.get('idx', None)
                        logging.info(f"Processing idx={idx}")
                        
                        try:
                            # Tokenize and detect
                            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=min(max_tokens, 1024))
                            inputs = {key: value.to(device) for key, value in inputs.items()}
                            decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

                            # results
                            detection_result = detector.detect(text=decoded_text)
                            if not detection_result:
                                logging.warning(f"idx={idx}: No detection result, skipping.")
                                continue

                            green_token_mask = detection_result.get('green_token_mask', [])
                            num_green_tokens = detection_result.get('num_green_tokens', 0)
                            num_tokens_scored = detection_result.get('num_tokens_scored', 1)
                            green_token_fraction = num_green_tokens / num_tokens_scored if num_tokens_scored > 0 else 0

                            # write in csv
                            prediction = detection_result.get('prediction')
                            writer.writerow([
                                idx, text,
                                prediction,
                                detection_result.get('z_score'),
                                detection_result.get('p_value'),
                                num_tokens_scored,
                                num_green_tokens,
                                green_token_fraction,
                                str(detection_result.get('ner_entities', [])),
                                str(green_token_mask)
                            ])

                            if prediction == True:
                                true_count += 1
                                logging.info(f"idx={idx}: prediction=True, z_score={detection_result.get('z_score')}")
                            else:
                                false_count += 1
                                logging.info(f"idx={idx}: prediction=False, z_score={detection_result.get('z_score')}")
                        except Exception as e:
                            logging.error(f"idx={idx}: Error during processing - {e}")

                logging.info(f"Finished processing {file_name}. True: {true_count}, False: {false_count}")

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

    logging.info("Finished processing all files in folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process watermark detection on all CSV files in a folder.")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder containing CSV files.")
    parser.add_argument('--gamma', type=float, required=True, help="Gamma value for watermark detection.")
    parser.add_argument('--tokenizer', type=str, default='gpt2', help="Tokenizer model to use (default: gpt2).")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cuda', help="Device to run the detection on (default: cpu).")
    parser.add_argument('--max_tokens', type=int, default=None, help="Maximum number of tokens to process per text entry. Texts longer than this will be truncated.")
    parser.add_argument('--entity_scale', type=float, default=1.0, help="Scale factor for NER token contribution.")
    parser.add_argument('--use_ner', action='store_true', help="Enable NER for token contribution adjustments.")

    args = parser.parse_args()
    print(f"Arguments: {args}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.tokenizer,
        device_map="auto",  # using multiple gpus
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.eval()

    # log
    print("Model loaded successfully with device_map=auto")
    logging.info("Model loaded successfully with device_map=auto")

    process_folder(args.folder_path, args.gamma, tokenizer, model, args.device, args.max_tokens, args.entity_scale, args.use_ner)

    logging.info("Process completed")
    print("Process completed")
