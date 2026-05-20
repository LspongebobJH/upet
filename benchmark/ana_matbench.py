from pathlib import Path
import pandas as pd
from matbench_discovery import today
from matbench_discovery.data import df_wbm, DataFilesCustomized
from matbench_discovery.enums import MbdKey
from matbench_discovery.metrics_old import stable_metrics
from pymatviz.enums import Key
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser
import re, os

e_form_pred_col = "e_form_per_atom_mattersim"

parser = ArgumentParser(description="Analyze Matbench IS2RE results")
parser.add_argument(
    "--result_folder", 
    type=str, 
    default="/home/lijiahang/OmniMat_mattersim_work/mattersim_worklogs/benchmark_matbench_chunk/test_torchrun_20250213", 
    help="Folder containing result CSV files"
)
parser.add_argument("--num_samples", type=int, default=df_wbm.shape[0])
args = parser.parse_args()

file_pattern="*wbm-IS2RE.csv.gz"
files = glob(f"{args.result_folder}/{file_pattern}")
if len(files) == 0:
    raise FileNotFoundError(f"No result files found in {args.result_folder} with pattern {file_pattern}. Please check the folder and file naming convention.")
df_result_list = [pd.read_csv(file) for file in tqdm(files, desc="Loading results")]
df_results = pd.concat(df_result_list, ignore_index=True)

# check results are all computed
if df_results.shape[0] != args.num_samples:
    print(f"Warning: Expected {df_wbm.shape[0]} results, but got {df_results.shape[0]}. This may indicate some results are missing or duplicated.")

    log_files = glob(f"{args.result_folder}/slice_*_*.log")

    exist_results = []
    for file in tqdm(files, desc="Finding existing results"):
        match = re.search(r"slice_\d+_\d+", file)[0]
        file = os.path.join(args.result_folder, match + ".log")
        exist_results.append(file)
    
    missing_results = []
    for file in tqdm(log_files, desc="Finding missing results"):
        if file not in exist_results:
            match = re.search(r"slice_\d+_\d+", file)[0]
            slice_tag = match.split("_")[-2:]
            slice_tag = "_".join(slice_tag)
            missing_results.append(slice_tag)
    print(f"Number of missing logs: {len(missing_results)}")
    with open(args.result_folder + "/missing_results.txt", "w") as f:
        for item in missing_results:
            f.write("%s\n" % item)
    # exit(1)

print("\n" + "=" * 60)
print("Calculate metrics...")
print("=" * 60)

# Merge results with WBM summary data
df_results_indexed = df_results.set_index(Key.mat_id)

# Merge predicted results with true values
df_eval = df_wbm.copy()
df_eval[e_form_pred_col] = df_results_indexed[e_form_pred_col]

# Keep only rows with predicted values
df_eval = df_eval.dropna(subset=[e_form_pred_col])

# check results are all computed
# assert df_eval.shape[0] == args.num_samples, f"Expected {args.num_samples} results after drop unpredicted rows, but got {df_eval.shape[0]}"

print(f"Number of structures used for evaluation: {len(df_eval):,}")


# Calculate predicted EACH (Energy Above Convex Hull)
# Formula: each_pred = each_true + e_form_pred - e_form_dft
each_pred = (
    df_eval[MbdKey.each_true] 
    + df_eval[e_form_pred_col] 
    - df_eval[MbdKey.e_form_dft]
)
each_true = df_eval[MbdKey.each_true]

# Calculate metrics
metrics = stable_metrics(each_true, each_pred, fillna=True)

print("\nEvaluation metrics:")
metric_names = ["Accuracy", "F1", "DAF", "Precision", "Recall", "TNR", "TPR", "MAE", "R2", "RMSE"]
metrics = [metrics[name] for name in metric_names]

display_metrics_names = ",".join([f"{name:>10s} " for name in metric_names])
display_metrics_values = ",".join([f"{value:>10.4f} " for value in metrics])
print(display_metrics_names)
print(display_metrics_values)