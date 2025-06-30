"""
Step 2: Entropy-Based Watermark Detection Recalculation

This script takes CSV files generated in Step 1 (containing entropy values and masks)
and recomputes z-scores, p-values, and predictions for multiple entropy thresholds.

Key Features:
- Supports multiple entropy thresholds for filtering tokens or bigrams.
- For each threshold, outputs:
  - Recalculated detection results.
  - Per-threshold true-positive records.
  - Summary statistics.
- Merges all summary data into a combined CSV report.

Usage Example:
python my_watermark_detection_entropy_step2.py \
--input_dir ./step1_outputs \
--output_dir ./step2_outputs \
--gamma 0.25 \
--z_threshold 4.0 \
--entropy_threshold 0.5 1.0 2.0 \
--mask_type bigram

Author:
Jingmiao Li
"""

import argparse
import os
import glob
import time
import logging
import pandas as pd
import ast
import torch
import multiprocessing
import re

from transformers import AutoTokenizer
from my_watermark_processor2_entropy import WatermarkDetector


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("second_round_detection.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
# ==============================================

def extract_file_id(input_csv_file):
    """
    从文件名中提取 file_id。例如：
    detection_results_c4-train.00502-of-00512.json.gz_entr_default.csv -> c4-train.00502-of-00512
    """
    base = os.path.basename(input_csv_file)
    m = re.search(r"detection_results_(.*?)\.json\.gz", base)
    if m:
        return m.group(1)
    else:
        if base.startswith("detection_results_"):
            base = base[len("detection_results_"):]
        return os.path.splitext(base)[0]

def process_row_with_entropy_filter(row, gamma, z_threshold, entropy_threshold, mask_type, detector_instance):
    """
    对单行记录进行二次检测：根据给定的 entropy_threshold 过滤 token/bigram，
    并计算新的 z_score, p_value, 以及 True/False。
    """
    mask_col = "token_entropy_mask" if mask_type == "token" else "bigram_entropy_mask"
    mask_str = row.get(mask_col, "")
    if not mask_str or mask_str.strip() == "":
        try:
            T = float(row["num_tokens"])
            G = float(row["num_green_tokens"])
        except Exception:
            T, G = 0, 0
    else:
        try:
            mask_data = ast.literal_eval(mask_str)
        except Exception as e:
            logger.warning(f"Error parsing {mask_col} for idx {row.get('idx')}: {mask_str}. Error: {e}")
            mask_data = []
        filtered_t = 0
        filtered_g = 0
        for triple in mask_data:
            try:
                entropy_val = float(triple[1])
                is_green = triple[2]
            except Exception:
                continue
            if entropy_val >= entropy_threshold:
                filtered_t += 1
                if is_green:
                    filtered_g += 1
        T, G = filtered_t, filtered_g

    z = detector_instance._compute_z_score(G, T) if T > 0 else None
    p_val = detector_instance._compute_p_value(z) if z is not None else None
    pred = True if (z is not None and z > z_threshold) else False

    return {
        "idx": row["idx"],
        "text": row["text"],
        "entropy_threshold": entropy_threshold,
        "filtered_t": T,
        "filtered_g": G,
        "z_score": z,
        "p_value": p_val,
        "prediction": pred
    }

def process_rows_second_round(df, gamma, z_threshold, entropy_threshold, mask_type, detector_instance):
    """
    针对指定 entropy_threshold，对文件中每条记录进行处理，
    每 500 条记录输出一次日志，返回处理结果的 DataFrame（按 idx 升序排序）。
    """
    results = []
    total = len(df)
    for i, row in df.iterrows():
        results.append(process_row_with_entropy_filter(row, gamma, z_threshold, entropy_threshold, mask_type, detector_instance))
        if (i+1) % 500 == 0:
            logger.info(f"Processed {i+1}/{total} valid entries: threshold {entropy_threshold}")
    df_result = pd.DataFrame(results)
    df_result.sort_values(by="idx", inplace=True)
    return df_result

def process_default(df, file_id, input_csv_file):
    """
    对默认（第一轮原始检测）处理，即不做 entropy 过滤，
    每 500 条记录输出一次日志，返回处理结果的 DataFrame（按 idx 升序排序）。
    """
    results = []
    total = len(df)
    for i, row in df.iterrows():
        results.append({
            "idx": row["idx"],
            "file_id": file_id,
            "text": row["text"],
            "entropy_threshold": "default",
            "filtered_t": row.get("num_tokens", 0),
            "filtered_g": row.get("num_green_tokens", 0),
            "z_score": row.get("z_score", None),
            "p_value": row.get("p_value", None),
            "prediction": bool(row.get("prediction", False))
        })
        if (i+1) % 500 == 0:
            logger.info(f"Processed {i+1}/{total} valid entries in file {input_csv_file}: threshold default")
    df_default = pd.DataFrame(results)
    df_default.sort_values(by="idx", inplace=True)
    return df_default

def write_output_csv(threshold, df, output_dir, lock):
    """
    将预测为 True 的记录（true-only）追加写入对应的 CSV 文件，
    并保证追加的记录按 (file_id, idx) 升序排序后写入。
    输出文件包含 idx 和 file_id 字段，文件名格式：nowatermark_entropy_{threshold}_true_only.csv
    """
    out_cols = ["idx", "file_id", "text", "entropy_threshold", "filtered_t", "filtered_g", "z_score", "p_value", "prediction"]
    true_df = df[df["prediction"] == True][out_cols]
    true_df.sort_values(by=["file_id", "idx"], inplace=True)
    filename = os.path.join(output_dir, f"nowatermark_entropy_{threshold}_true_only.csv")
    with lock:
        if not os.path.exists(filename):
            true_df.to_csv(filename, index=False, mode="w", header=True)
        else:
            true_df.to_csv(filename, index=False, mode="a", header=False)

def write_summary_csv(summary_df, output_dir, lock):
    """
    将当前文件的汇总数据追加写入全局 summary 文件（combined_summary.csv）。
    """
    filename = os.path.join(output_dir, "combined_summary.csv")
    with lock:
        if not os.path.exists(filename):
            summary_df.to_csv(filename, index=False, mode="w", header=True)
        else:
            summary_df.to_csv(filename, index=False, mode="a", header=False)

def process_single_csv(input_csv_file, gamma, z_threshold, entropy_thresholds, mask_type, output_dir, lock):
    """
    处理单个 CSV 文件：
      1. 读取 CSV 文件，提取 file_id
      2. 分别对 default 以及每个 entropy_threshold 处理，
         每 500 条记录输出一次日志；
      3. 文件处理结束后，将 true-only 记录及汇总数据追加写入全局输出文件。
      4. 返回该文件的详细处理结果及汇总数据（供主进程汇总使用）。
    """
    start_t = time.time()
    pid = os.getpid()
    logger.info(f"[PID={pid}] 开始处理: {input_csv_file}")
    df = pd.read_csv(input_csv_file)
    total = len(df)
    logger.info(f"[PID={pid}] {input_csv_file}，共 {total} 行。")
    file_id = extract_file_id(input_csv_file)


    dummy_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dummy_device = torch.device("cpu")
    detector_helper = WatermarkDetector(vocab=[], gamma=gamma, device=dummy_device, tokenizer=dummy_tokenizer)


    default_df = process_default(df, file_id, input_csv_file)
    true_count_default = (default_df["prediction"] == True).sum()
    false_count_default = (default_df["prediction"] == False).sum()
    logger.info(f"Processed {total}/{total} valid entries in file {input_csv_file}: threshold default -> true_count = {true_count_default}, false_count = {false_count_default}")
    write_output_csv("default", default_df, output_dir, lock)

    # 处理其他 entropy_thresholds
    all_threshold_results = [default_df]
    for ent_thr in entropy_thresholds:
        df_thr = process_rows_second_round(df, gamma, z_threshold, ent_thr, mask_type, detector_helper)
        df_thr["file_id"] = file_id
        true_count_thr = (df_thr["prediction"] == True).sum()
        false_count_thr = (df_thr["prediction"] == False).sum()
        logger.info(f"Processed {total}/{total} valid entries in file {input_csv_file}: threshold {ent_thr} -> true_count = {true_count_thr}, false_count = {false_count_thr}")
        write_output_csv(ent_thr, df_thr, output_dir, lock)
        all_threshold_results.append(df_thr)

    end_t = time.time()
    logger.info(f"[PID={pid}] finished: {input_csv_file}, used {end_t - start_t:.2f} seconds")


    all_results_df = pd.concat(all_threshold_results, ignore_index=True)
    desired_cols = ["idx", "file_id", "text", "entropy_threshold", "filtered_t", "filtered_g", "z_score", "p_value", "prediction"]
    all_results_df = all_results_df[desired_cols]


    summary_df = all_results_df.groupby("entropy_threshold").agg(
        true_count=("prediction", lambda x: (x == True).sum()),
        false_count=("prediction", lambda x: (x == False).sum()),
        avg_filtered_t=("filtered_t", "mean"),
        avg_filtered_g=("filtered_g", "mean"),
        avg_z_score=("z_score", "mean"),
        avg_p_value=("p_value", "mean")
    ).reset_index()
    summary_df["total_records_processed"] = total
    summary_df["file_id"] = file_id
    desired_cols_summary = ["file_id", "entropy_threshold", "true_count", "false_count", "total_records_processed",
                              "avg_filtered_t", "avg_filtered_g", "avg_z_score", "avg_p_value"]
    summary_df = summary_df[desired_cols_summary]

    def check_anomaly(row):
        if row["true_count"] + row["false_count"] != row["total_records_processed"]:
            return "Mismatch!"
        return ""
    summary_df["anomaly_check"] = summary_df.apply(check_anomaly, axis=1)
    summary_df["overall_green_fraction"] = summary_df.apply(lambda row: row["avg_filtered_g"] / row["avg_filtered_t"] if row["avg_filtered_t"] > 0 else 0, axis=1)

    # 将当前文件的汇总数据追加写入全局 summary 文件
    write_summary_csv(summary_df, output_dir, lock)

    return all_results_df, summary_df

def sort_output_files(thresholds, output_dir):

    for thr in thresholds:
        filename = os.path.join(output_dir, f"nowatermark_entropy_{thr}_true_only.csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            if not df.empty:
                df.sort_values(by=["file_id", "idx"], inplace=True)
                df.to_csv(filename, index=False)
                logger.info(f"Sorted output file: {filename}")

    summary_file = os.path.join(output_dir, "combined_summary.csv")
    if os.path.exists(summary_file):
        df_summary = pd.read_csv(summary_file)
        df_summary["entropy_sort"] = df_summary["entropy_threshold"].apply(lambda x: -1 if x=="default" else float(x))
        df_summary.sort_values(by=["file_id", "entropy_sort"], inplace=True)
        df_summary.drop("entropy_sort", axis=1, inplace=True)
        df_summary.to_csv(summary_file, index=False)
        logger.info(f"Sorted global summary file: {summary_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Step 2 of entropy-based filtering: using entropy values and threshold to calculate z-scores"
    )
    parser.add_argument('--input_dir', type=str, required=True, help="input directory with csv files")
    parser.add_argument('--output_dir', type=str, required=True, help="output directory to store results")
    parser.add_argument('--gamma', type=float, default=0.25, help="expected green token ratio")
    parser.add_argument('--z_threshold', type=float, default=4.0, help="determine if the sample is watermarked")
    parser.add_argument('--entropy_threshold', type=float, nargs='*',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                        help="accept multipy entropy_thresholds")
    parser.add_argument('--mask_type', type=str, choices=["token", "bigram"], default="bigram",
                        help="use bigram as default")
    parser.add_argument('--num_workers', type=int, default=2, help="parallel process count")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_files:
        logger.info(f"{args.input_dir} doesn't have any csv files")
        return

    start_time = time.time()
    total_files = len(csv_files)
    logger.info(f"Found {total_files}  CSV files, use {args.num_workers} parallel workers to process.")

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    pool = multiprocessing.Pool(processes=args.num_workers)
    async_results = []
    for csv_file in csv_files:
        async_results.append(pool.apply_async(process_single_csv, 
                                (csv_file, args.gamma, args.z_threshold, args.entropy_threshold, args.mask_type, args.output_dir, lock)))
    pool.close()
    pool.join()

    all_detailed = []
    all_summary = []
    for r in async_results:
        detailed_df, summary_df = r.get()
        all_detailed.append(detailed_df)
        all_summary.append(summary_df)

    combined_summary = pd.concat(all_summary, ignore_index=True)
    combined_summary.sort_values(by=["file_id", "entropy_threshold"], inplace=True)
    combined_summary_file = os.path.join(args.output_dir, "combined_summary_merged.csv")
    combined_summary.to_csv(combined_summary_file, index=False)
    logger.info(f"sumary file: {combined_summary_file}")

    thresholds = ["default"] + [str(t) for t in sorted(args.entropy_threshold)]
    sort_output_files(thresholds, args.output_dir)

    end_time = time.time()
    logger.info(f"Finshed all processing, using {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
