from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import os
import pickle
import re

import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


parser = ArgumentParser(description="Analyze EFS results")
parser.add_argument(
    "--result_folder",
    type=str,
    default="./logs/benchmark_efs",
    help="Folder containing saved EFS result pickle files",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Expected number of structures. If provided, the script will check for missing or duplicated slices. -1 means no checking.",
)
args = parser.parse_args()

file_pattern = "*_efs.pkl"
files = glob(f"{args.result_folder}/{file_pattern}")
if len(files) == 0:
    raise FileNotFoundError(
        f"No result files found in {args.result_folder} with pattern {file_pattern}. Please check the folder and file naming convention."
    )

result_list = []
for file in tqdm(files, desc="Loading results"):
    with open(file, "rb") as f:
        result_list.append(pickle.load(f))

gt_e_list = np.concatenate([result["gt_e_list"] for result in result_list], axis=0)
pred_e_list = np.concatenate([result["pred_e_list"] for result in result_list], axis=0)
gt_f_list = np.concatenate([result["gt_f_list"] for result in result_list], axis=0)
pred_f_list = np.concatenate([result["pred_f_list"] for result in result_list], axis=0)
gt_s_list = np.concatenate([result["gt_s_list"] for result in result_list], axis=0)
pred_s_list = np.concatenate([result["pred_s_list"] for result in result_list], axis=0)

num_structures = sum(result["num_structures"] for result in result_list)

if num_structures != args.num_samples and args.num_samples != -1:
    print(f"Warning: Expected {args.num_samples} structures, but got {num_structures}. This may indicate some results are missing or duplicated.")

    log_files = glob(f"{args.result_folder}/slice_*_*.log")

    existing_results = []
    for file in tqdm(files, desc="Finding existing results"):
        match = re.search(r"slice_\d+_\d+", file)
        if match is not None:
            existing_results.append(os.path.join(args.result_folder, match[0] + ".log"))

    missing_results = []
    for file in tqdm(log_files, desc="Finding missing results"):
        if file not in existing_results:
            match = re.search(r"slice_\d+_\d+", file)
            if match is not None:
                slice_tag = "_".join(match[0].split("_")[-2:])
                missing_results.append(slice_tag)

    print(f"Number of missing logs: {len(missing_results)}")
    with open(Path(args.result_folder) / "missing_results.txt", "w", encoding="utf-8") as f:
        for item in missing_results:
            f.write(f"{item}\n")
    exit(1)

mae_e = mean_absolute_error(gt_e_list, pred_e_list)
mae_f = mean_absolute_error(gt_f_list, pred_f_list)
mae_s = mean_absolute_error(gt_s_list, pred_s_list)

print("\n" + "=" * 60)
print("Aggregate EFS metrics")
print("=" * 60)
print(f"Number of files: {len(files)}")
print(f"Number of structures: {num_structures}")
print(f"MAE Energy | MAE Forces | MAE Stress: {mae_e:.6f}, {mae_f:.6f}, {mae_s:.6f}")
