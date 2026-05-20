"""Evaluate MLIP-predicted thermal conductivity metrics against DFT results without
non-analytical correction term (NAC).
"""

import os
import sys
import glob
import pandas as pd
from pymatviz.enums import Key
from argparse import ArgumentParser
import gzip
import json

from matbench_discovery.data import DataFilesCustomized as DataFiles
from matbench_discovery.metrics import phonons


def main(args):
    """Evaluate kappa metrics and update model YAML files.

    Returns:
        Exit code: 0 if at least one model was evaluated, 1 otherwise.
    """

    # Compute metrics if we've collected all results
    pattern = f"{args.result_folder}/*kappa-103*.json.gz"
    file_list = list(glob.glob(pattern))
    # Load all results
    all_dfs = []
    for file_path in file_list:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        data = pd.DataFrame(data)
        all_dfs.append(data)

    print(f"\nProcessing...")    
    df_ml = pd.concat(all_dfs).reset_index()
    if "mat_id" in df_ml.columns:
        df_ml = df_ml.rename(columns={"mat_id": Key.mat_id})
    df_ml = df_ml.set_index(Key.mat_id)

    df_dft = pd.read_json(DataFiles.phonondb_pbe_103_kappa_no_nac.path)
    if "mp_id" in df_dft.columns:
        df_dft = df_dft.rename(columns={"mp_id": Key.mat_id})
    df_dft = df_dft.set_index(Key.mat_id)

    assert len(df_ml) == len(df_dft), f"ML and DFT results must have the same number of entries for comparison. ML: {len(df_ml)}, DFT: {len(df_dft)}"

    df_ml_metrics = phonons.calc_kappa_metrics_from_dfs(df_ml, df_dft)

    # Calculate metrics
    kappa_sre = df_ml_metrics[Key.sre].mean()
    kappa_srme = df_ml_metrics[Key.srme].mean()
    print(f"\t{kappa_srme=:.4f}")
    print(f"\t{kappa_sre=:.4f}")

    # Exit with error if no models were successfully evaluated
    print(f"\nDone")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate phonon thermal conductivity metrics.")
    parser.add_argument("--result_folder", type=str, required=True, help="Directory where results are saved.")
    args = parser.parse_args()
    main(args=args)