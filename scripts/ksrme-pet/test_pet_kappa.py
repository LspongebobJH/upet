"""
Script for generating the predicted kappa-SRME values for the 103 structures in the
PhononDB-PBE dataset, using a PET model.

Templated from https://github.com/janosh/matbench-discovery/blob/main/models/nequip/test_nequip_kappa.py
"""

import argparse
import json
import os
import traceback
import warnings
from datetime import datetime
from importlib.metadata import version
from typing import Any, Literal

import ase.io
import pandas as pd
import torch
from calc_kappa import calc_kappa_for_structure
from metatomic.torch import load_atomistic_model
from metatomic.torch.ase_calculator import MetatomicCalculator, SymmetrizedCalculator
from pymatviz.enums import Key
from tqdm import tqdm

from matbench_discovery import today
from matbench_discovery.data import DataFilesCustomized as DataFiles
from matbench_discovery.metrics.phonons import calc_kappa_metrics_from_dfs
from matbench_discovery.phonons import KappaCalcParams

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_slice",
        type=str,
        default=None,
        help="Slice atoms_list with START_END semantics, e.g. 0_11 selects samples 0 through 10.",
    )
    parser.add_argument(
        "--non_conservative",
        action="store_true",
        default=False,
        help="Whether to use a non-conservative version of the model for force calculations. Default is False."
        
    )
    return parser.parse_args()


def parse_data_slice(data_slice: str, num_samples: int) -> tuple[int, int]:
    try:
        start_str, end_str = data_slice.split("_", maxsplit=1)
        start, end = int(start_str), int(end_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --data_slice {data_slice!r}. Expected format START_END, e.g. 0_11."
        ) from exc

    if not (0 <= start < end <= num_samples):
        raise ValueError(
            f"Invalid --data_slice {data_slice!r} for {num_samples} samples. "
            "Require 0 <= start < end <= len(atoms_list)."
        )

    return start, end

args = parse_args()

# Model configuration
module_dir = os.path.dirname(__file__)
model_name = "pet"
if args.non_conservative:
    model_variant = "oam-xl-v1.0.0-nc"  # get it with `mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-oam-xl-nc-v1.0.0.ckpt`
else:
    model_variant = "oam-xl-v1.0.0"  # get it with `mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-oam-xl-v1.0.0.ckpt`

precision = "float64"
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
dtype = torch.float64 if precision == "float64" else torch.float32
# model = load_atomistic_model(f"{model_name}-{model_variant}.pt") # jiahang: debug
model = load_atomistic_model(f"/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/checkpoints/pet-oam-xl-v1.0.0.pt")
model.capabilities().dtype = precision
model = model.to(dtype=dtype, device=device)
calc = MetatomicCalculator(model, device=device, non_conservative=args.non_conservative)
# calc = SymmetrizedCalculator(calc, batch_size=1, include_inversion=False) # jiahang: debug, since buggy to use symmetrized one in calculate_fc2_set(), which assumes using MetatomicCalculator.
batch_size = 1

# Relaxation parameters
ase_optimizer = "FIRE"
ase_filter: Literal["frechet", "exp"] = "frechet"  # recommended filter
max_steps = 300
fmax = 1e-4  # Run until the forces are smaller than this in eV/A

# Symmetry parameters
symprec = 1e-5  # symmetry precision for enforcing relaxation and conductivity calcs
enforce_relax_symm = True  # Enforce symmetry with during relaxation if broken
# Conductivity to be calculated if symmetry group changed during relaxation
conductivity_broken_symm = False
save_forces = True  # Save force sets to file
temperatures: list[float] = [300]
displacement_distance = 0.03
ignore_imaginary_freqs = True

# Task splitting:
data_slice = args.data_slice

job_name = f"kappa-103-{ase_optimizer}-dist={displacement_distance}-{fmax=}-{symprec=}"
if data_slice is not None:
    job_name += f"-slice={data_slice}"
out_dir = f"logs/ksrme/{model_name}-{model_variant}-{today}-{job_name}"
os.makedirs(out_dir, exist_ok=True)
timestamp = f"{datetime.now().astimezone():%Y-%m-%d@%H-%M-%S}"
print(f"\nJob {job_name} with {model_name} started {timestamp}")

atoms_list = ase.io.read(DataFiles.phonondb_pbe_103_structures.path, index=":")
# sort by size to get roughly even distribution of comp cost across GPUs
atoms_list = sorted(atoms_list, key=len)
if data_slice is not None:
    slice_start, slice_end = parse_data_slice(data_slice, len(atoms_list))
    atoms_list = atoms_list[slice_start:slice_end]
    print(
        f"Using atoms_list[{slice_start}:{slice_end}] -> "
        f"{slice_end - slice_start} samples (indices {slice_start} to {slice_end - 1})"
    )
if world_size > 1:
    num_samples = len(atoms_list)
    num_samples_per_proc = (num_samples + world_size - 1) // world_size
    start = local_rank * num_samples_per_proc
    end = min(start + num_samples_per_proc, num_samples)
    atoms_list = atoms_list[start:end]
    print("Be noted that atom_list has been sorted, such that index no longer corresponds to original mat_id. Use the mat_id in atoms.info to match with reference data.")
    print(f"Local rank {local_rank} handling samples from {start} to {end}")

# Save run parameters
kappa_params: KappaCalcParams = {
    "ase_optimizer": ase_optimizer,
    "ase_filter": ase_filter,
    "max_steps": max_steps,
    "force_max": fmax,
    "symprec": symprec,
    "enforce_relax_symm": enforce_relax_symm,
    "temperatures": temperatures,
    "out_dir": out_dir,
    "displacement_distance": displacement_distance,
    "save_forces": save_forces,
}
run_params = dict(
    **kappa_params,
    n_structures=len(atoms_list),
    data_slice=data_slice,
    struct_data_path=DataFiles.phonondb_pbe_103_structures.path,
    versions={dep: version(dep) for dep in ("numpy", "torch", "metatomic")},
)

with open(f"{out_dir}/run_params.json", mode="w") as file:
    json.dump(run_params, file, indent=4)

# Process results as they complete
kappa_results: dict[str, dict[str, Any]] = {}
force_results: dict[str, dict[str, Any]] = {}

for idx, atoms in enumerate(tqdm(atoms_list, desc="Calculating kappa...")):
    mat_id, result_dict, force_dict = calc_kappa_for_structure(
        atoms=atoms,
        calculator=calc,
        batch_size=batch_size,
        is_plusminus=True,
        ignore_imaginary_freqs=ignore_imaginary_freqs,
        formula_getter=lambda a: a.info.get("name", a.get_chemical_formula()),
        **kappa_params,
        task_id=idx,
    )
    kappa_results[mat_id] = result_dict
    if force_dict is not None:
        force_results[mat_id] = force_dict

    # Save intermediate results
    df_kappa = pd.DataFrame(kappa_results).T
    df_kappa.index.name = Key.mat_id
    df_kappa.reset_index(drop=True).to_json(
        f"{out_dir}/{local_rank}_kappa.json.gz"
    )
    df_kappa.to_json(f"{out_dir}/{local_rank}_kappa.json.gz")

    if save_forces:
        df_force = pd.DataFrame(force_results).T
        df_force = pd.concat([df_kappa, df_force], axis=1)
        df_force.index.name = Key.mat_id
        df_force.reset_index(drop=True).to_json(
            f"{out_dir}/{local_rank}_force-sets.json.gz"
        )

print(f"\nResults saved to {out_dir!r}")

try:
    print("Computing metrics against reference data...")
    df_dft = pd.read_json(DataFiles.phonondb_pbe_103_kappa_no_nac.path).set_index(
        Key.mat_id
    )
    if ignore_imaginary_freqs:
        # WARNING: setting has_imag_ph_modes to False to compute the metrics anyway
        df_kappa["has_imag_ph_modes"] = False
    df_ml_metrics = calc_kappa_metrics_from_dfs(df_kappa, df_dft)
    # Compute and print summary metrics
    kappa_sre = df_ml_metrics[Key.sre].mean()
    kappa_srme = df_ml_metrics[Key.srme].mean()
    print(f"{kappa_sre=:.4f}")
    print(f"{kappa_srme=:.4f}")

    df_ml_metrics.to_json(f"{out_dir}/metrics.json.gz")
    print(f"Saved metrics to {out_dir}/metrics.json.gz")
except Exception as exc:
    warnings.warn(f"Failed to calculate metrics: {exc!r}", stacklevel=2)
    traceback.print_exc()