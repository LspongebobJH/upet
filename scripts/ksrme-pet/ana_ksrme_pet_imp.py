from pathlib import Path

import pandas as pd
from pymatviz.enums import Key
from matbench_discovery.data import DataFilesCustomized as DataFiles
from matbench_discovery.metrics.phonons import calc_kappa_metrics_from_dfs

kappa_dirs = [
    Path(
        "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/logs/ksrme/pet-oam-xl-v1.0.0-ispm"
    ),
    # Path(
    #     "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/logs/ksrme/pet-oam-xl-v1.0.0-nc-float32-2026-05-22-kappa-103-FIRE-dist=0.03-fmax=0.0001-symprec=1e-05-slice=52_103"
    # ),
]

kappa_files = [
    file_path
    for kappa_dir in kappa_dirs
    for file_path in sorted(kappa_dir.glob("*.json.gz"))
]
df_kappa = pd.concat([pd.read_json(file_path) for file_path in kappa_files])
df_kappa.index.name = Key.mat_id

print("Computing metrics against reference data...")
df_dft = pd.read_json(DataFiles.phonondb_pbe_103_kappa_no_nac.path).set_index(
    Key.mat_id
)

# WARNING: setting has_imag_ph_modes to False to compute the metrics anyway
df_kappa["has_imag_ph_modes"] = False
df_ml_metrics = calc_kappa_metrics_from_dfs(df_kappa, df_dft)
# Compute and print summary metrics
kappa_sre = df_ml_metrics[Key.sre].mean()
kappa_srme = df_ml_metrics[Key.srme].mean()
print(f"{kappa_sre=:.4f}")
print(f"{kappa_srme=:.4f}")
