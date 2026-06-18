# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import random
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from ase.units import GPa
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from omegaconf import OmegaConf

from metatrain.utils.data.ase_db_datasets import (
    AseDBCollateFn,
    AseDBDatasetCustomized,
    AseDBRawDataset,
)
from metatrain.utils.io import load_model as load_metatrain_model
from metatrain.utils.transfer import batch_to
from metatrain.utils.data import unpack_batch
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatensor.torch import Labels

import sys

sys.path.append("/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet")
from tools.utils import Logger

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))


def _dir_has_aselmdb(path: Union[str, Path]) -> bool:
    """Return True if path is a directory containing *.aselmdb files."""
    path = Path(path)
    return path.is_dir() and bool(glob(str(path / "*.aselmdb")))


def _resolve_aselmdb_src(read_from: str) -> Optional[Union[str, List[str]]]:
    """Resolve read_from to a fairchem src for an aselmdb dataset.

    Supports:
    - Single *.aselmdb file
    - Directory holding *.aselmdb files
    - Glob pattern expanding to directories with *.aselmdb files
    """
    if not isinstance(read_from, str):
        return None

    # Single aselmdb file
    if read_from.endswith(".aselmdb"):
        return read_from

    # Single directory of aselmdb files
    if _dir_has_aselmdb(read_from):
        return read_from

    # Glob pattern -> collect leaf directories that hold aselmdb files
    if any(ch in read_from for ch in "*?["):
        dirs = sorted(m for m in glob(read_from) if _dir_has_aselmdb(m))
        if dirs:
            return dirs

    return None


def save_results(args, results, logger, suffix="aggregated"):
    """Save evaluation results to pickle file."""
    logger("\nSaving results...")
    output_dir = Path(args.log_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_path)
    # Replace rank-specific name with suffix
    log_stem = log_path.stem
    if "rank" in log_stem:
        # Replace rankX with suffix
        import re
        log_stem = re.sub(r'rank\d+', suffix, log_stem)
    else:
        log_stem = f"{log_stem}_{suffix}"

    output_file = log_path.parent / f"{log_stem}_efs.pkl"

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    logger(f"Results saved to: {output_file}")
    return output_file


def gather_and_aggregate_results(results, device):
    """Gather results from all ranks and aggregate on rank 0."""
    # Gather all results to rank 0
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Collect results from all ranks
    gathered_results = [None] * world_size
    torch.distributed.all_gather_object(gathered_results, results)

    if rank == 0:
        # Aggregate results on rank 0
        aggregated = {'num_structures': 0}

        # Determine which modes are present
        modes = []
        if 'conservative' in gathered_results[0]:
            modes.append('conservative')
            aggregated['conservative'] = {
                'gt_e_list': [], 'pred_e_list': [],
                'gt_f_list': [], 'pred_f_list': [],
                'gt_s_list': [], 'pred_s_list': [],
            }
        if 'non_conservative' in gathered_results[0]:
            modes.append('non_conservative')
            aggregated['non_conservative'] = {
                'gt_e_list': [], 'pred_e_list': [],
                'gt_f_list': [], 'pred_f_list': [],
                'gt_s_list': [], 'pred_s_list': [],
            }

        # Concatenate results from all ranks
        for rank_results in gathered_results:
            aggregated['num_structures'] += rank_results['num_structures']

            for mode in modes:
                mode_data = rank_results[mode]
                aggregated[mode]['gt_e_list'].append(mode_data['gt_e_list'])
                aggregated[mode]['pred_e_list'].append(mode_data['pred_e_list'])
                aggregated[mode]['gt_f_list'].append(mode_data['gt_f_list'])
                aggregated[mode]['pred_f_list'].append(mode_data['pred_f_list'])
                aggregated[mode]['gt_s_list'].append(mode_data['gt_s_list'])
                aggregated[mode]['pred_s_list'].append(mode_data['pred_s_list'])

        # Final concatenation
        for mode in modes:
            aggregated[mode]['gt_e_list'] = np.concatenate(aggregated[mode]['gt_e_list'])
            aggregated[mode]['pred_e_list'] = np.concatenate(aggregated[mode]['pred_e_list'])
            aggregated[mode]['gt_f_list'] = np.concatenate(aggregated[mode]['gt_f_list'], axis=0)
            aggregated[mode]['pred_f_list'] = np.concatenate(aggregated[mode]['pred_f_list'], axis=0)
            aggregated[mode]['gt_s_list'] = np.concatenate(aggregated[mode]['gt_s_list'], axis=0)
            aggregated[mode]['pred_s_list'] = np.concatenate(aggregated[mode]['pred_s_list'], axis=0)

        return aggregated
    else:
        return None


def extract_energy_from_tensormap(energy_map, per_atom=True, systems=None):
    """Extract energy values from tensormap.

    Args:
        energy_map: TensorMap containing energy values
        per_atom: If True, divide total energy by number of atoms
        systems: List of System objects, needed to get atom counts when per_atom=True
    """
    values = energy_map.block().values.cpu().detach().numpy().flatten()

    if per_atom and systems is not None:
        # Divide each energy by the number of atoms in corresponding system
        atom_counts = np.array([len(system.positions) for system in systems])
        values = values / atom_counts

    return values


def extract_forces_from_tensormap(energy_map):
    """Extract forces from energy tensormap gradient (conservative mode)."""
    # if not energy_map.has_gradient("positions"):
    #     return None
    forces_block = energy_map.block().gradient("positions")
    # Forces are stored as negative position gradients
    forces = -forces_block.values.cpu().detach().numpy()
    return forces.reshape(-1, 3)


def extract_non_conservative_forces(forces_map):
    """Extract non-conservative forces (stored as raw +F)."""
    forces_block = forces_map.block()
    # Non-conservative forces are stored directly as +F
    forces = forces_block.values.cpu().detach().numpy()
    return forces.reshape(-1, 3)


def extract_stress_from_tensormap(energy_map, systems):
    """Extract stress from energy tensormap strain gradient (conservative mode)."""
    # if not energy_map.has_gradient("strain"):
    #     return None

    stress_block = energy_map.block().gradient("strain")
    stress_values = stress_block.values.cpu().detach().numpy()
    batch_size = len(systems)

    stress_reshaped = stress_values.reshape(batch_size, 3, 3, 1).squeeze(-1)

    volumes = []
    for system in systems:
        cell = system.cell.cpu().detach().numpy()
        volume = np.abs(np.linalg.det(cell))
        volumes.append(volume)
    volumes = np.array(volumes).reshape(-1, 1, 1)

    # Convert from strain gradient (virial*volume) to stress, then to GPa
    stress = stress_reshaped / volumes / GPa
    return stress.reshape(-1, 3, 3)


def extract_non_conservative_stress(stress_map):
    """Extract non-conservative stress (stored as raw stress from get_stress)."""
    stress_block = stress_map.block()
    # Non-conservative stress is stored directly as stress values in eV/Angstrom^3
    stress_values = stress_block.values.cpu().detach().numpy()
    # Convert to GPa units (same as original mae.py)
    return stress_values.reshape(-1, 3, 3) / GPa


def eval_batched(args, model, dataloader, logger, device, eval_conservative, eval_non_conservative, target_info, dtype):
    """Evaluate model on batched data from dataloader.

    Args:
        eval_conservative: bool, whether to evaluate conservative mode
        eval_non_conservative: bool, whether to evaluate non-conservative mode
        target_info: dict, target info from the dataset
        dtype: torch.dtype, dtype of the model
    """
    # model.eval()

    # Separate results for each mode
    results_dict = {}
    if eval_conservative:
        results_dict['conservative'] = {
            'gt_e_list': [], 'pred_e_list': [],
            'gt_f_list': [], 'pred_f_list': [],
            'gt_s_list': [], 'pred_s_list': [],
        }
    if eval_non_conservative:
        results_dict['non_conservative'] = {
            'gt_e_list': [], 'pred_e_list': [],
            'gt_f_list': [], 'pred_f_list': [],
            'gt_s_list': [], 'pred_s_list': [],
        }

    num_structures = 0

    # Get the neighbor lists required by the model
    requested_neighbor_lists = get_requested_neighbor_lists(model)

    # with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating batches")):
        # Unpack the serialized batch first
        systems, targets, extra_data = unpack_batch(batch)

        # Then move to device
        systems, targets, extra_data = batch_to(
            systems, targets, extra_data, dtype=dtype, device=device
        )

        # Compute neighbor lists for each system
        systems = [
            get_system_with_neighbor_lists(system, requested_neighbor_lists)
            for system in systems
        ]

        batch_size = len(systems)
        num_structures += batch_size

        # Build target_info for evaluate_model
        eval_targets = {}
        eval_targets["energy"] = target_info["energy"]
        if eval_non_conservative:
            eval_targets["non_conservative_forces"] = target_info["non_conservative_forces"]
            eval_targets["non_conservative_stress"] = target_info["non_conservative_stress"]

        # Get predictions using evaluate_model
        outputs = evaluate_model(
            model,
            systems,
            eval_targets,
            is_training=False,
        )

        # Extract ground truth and predictions
        # Energy is always computed
        gt_energies = extract_energy_from_tensormap(targets["energy"], per_atom=True, systems=systems)
        pred_energies = extract_energy_from_tensormap(outputs["energy"], per_atom=True, systems=systems)

        # Conservative mode: forces and stress from energy gradients
        if eval_conservative:
            results_dict['conservative']['gt_e_list'].append(gt_energies)
            results_dict['conservative']['pred_e_list'].append(pred_energies)

            gt_forces = extract_forces_from_tensormap(targets["energy"])
            pred_forces = extract_forces_from_tensormap(outputs["energy"])
            results_dict['conservative']['gt_f_list'].append(gt_forces)
            results_dict['conservative']['pred_f_list'].append(pred_forces)

            gt_stress = extract_stress_from_tensormap(targets["energy"], systems)
            pred_stress = extract_stress_from_tensormap(outputs["energy"], systems)
            results_dict['conservative']['gt_s_list'].append(gt_stress)
            results_dict['conservative']['pred_s_list'].append(pred_stress)

        # Non-conservative mode: forces and stress from direct predictions
        if eval_non_conservative:
            results_dict['non_conservative']['gt_e_list'].append(gt_energies)
            results_dict['non_conservative']['pred_e_list'].append(pred_energies)

            gt_forces = extract_non_conservative_forces(targets["non_conservative_forces"])
            pred_forces = extract_non_conservative_forces(outputs["non_conservative_forces"])
            results_dict['non_conservative']['gt_f_list'].append(gt_forces)
            results_dict['non_conservative']['pred_f_list'].append(pred_forces)

            gt_stress = extract_non_conservative_stress(targets["non_conservative_stress"])
            pred_stress = extract_non_conservative_stress(outputs["non_conservative_stress"])
            results_dict['non_conservative']['gt_s_list'].append(gt_stress)
            results_dict['non_conservative']['pred_s_list'].append(pred_stress)

        if (batch_idx + 1) % 10 == 0:
            logger(f"Processed {num_structures} structures")

    # Concatenate all results for each mode
    final_results = {'num_structures': num_structures}

    for mode_name, mode_results in results_dict.items():
        final_results[mode_name] = {
            "gt_e_list": np.concatenate(mode_results['gt_e_list']) if mode_results['gt_e_list'] else np.array([]),
            "pred_e_list": np.concatenate(mode_results['pred_e_list']) if mode_results['pred_e_list'] else np.array([]),
            "gt_f_list": np.concatenate(mode_results['gt_f_list'], axis=0) if mode_results['gt_f_list'] else np.array([]),
            "pred_f_list": np.concatenate(mode_results['pred_f_list'], axis=0) if mode_results['pred_f_list'] else np.array([]),
            "gt_s_list": np.concatenate(mode_results['gt_s_list'], axis=0) if mode_results['gt_s_list'] else np.array([]),
            "pred_s_list": np.concatenate(mode_results['pred_s_list'], axis=0) if mode_results['pred_s_list'] else np.array([]),
        }

    return final_results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file for the model"
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        required=True,
        help="Path to ASE DB file for validation data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs/benchmark_efs/slice_start_end.log",
        help="Path to log file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed evaluation across ranks"
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force rerun even if results exist"
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Slice of dataset to evaluate (format: start_end)"
    )
    parser.add_argument(
        "--non_conservative",
        action="store_true",
        help="Evaluate non-conservative forces and stress (direct predictions)"
    )
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="Evaluate conservative forces and stress (energy gradients)"
    )

    args = parser.parse_args()

    # Determine evaluation modes: if both specified, evaluate both; if neither, default to conservative
    eval_conservative = args.conservative or (not args.conservative and not args.non_conservative)
    eval_non_conservative = args.non_conservative

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.distributed.init_process_group(backend="nccl")
    else:
        device = torch.device(args.device)

    logger_func = Logger(log_path=args.log_path) if args.distributed else print

    logger_func("=" * 60)
    logger_func("Batched MAE Evaluation with ASE DB Dataloader")
    logger_func("=" * 60)
    logger_func(f"Model: {args.ckpt_path}")
    logger_func(f"Data: {args.valid_data_path}")
    logger_func(f"Device: {device}")
    logger_func(f"Batch size: {args.batch_size}")
    logger_func(f"Evaluation modes:")
    if eval_conservative:
        logger_func("  - Conservative (energy gradients)")
    if eval_non_conservative:
        logger_func("  - Non-conservative (direct predictions)")
    logger_func(f"Distributed: {args.distributed}")
    if args.distributed:
        logger_func(f"Rank: {rank}/{world_size}")
    logger_func("=" * 60)

    logger_func("\nLoading model...")
    model = load_metatrain_model(args.ckpt_path)
    model = model.to(device)
    model.eval()

    # Get the dtype from the model's parameters
    model_dtype = next(model.parameters()).dtype

    logger_func("\nCreating dataset...")
    # Read and expand target config exactly like training does
    config_path = "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/configs/pet-omat-xl-v1.0.0-lr-1e-3.yaml"
    full_config = OmegaConf.load(config_path)

    # Use metatrain's expand_dataset_config to apply all defaults
    from metatrain.utils.omegaconf import expand_dataset_config
    expanded_datasets = expand_dataset_config(full_config["validation_set"])

    # Extract the expanded validation targets (already has all defaults applied)
    expanded_config = expanded_datasets[0]  # First dataset in list
    all_targets = OmegaConf.to_container(expanded_config["targets"])

    # Select only the targets we need based on eval mode
    target_config = {
        "energy": all_targets["energy"]
    }

    # # Add conservative gradients if requested
    # if eval_conservative:
    #     target_config["energy"]["forces"] = all_targets["energy"]["forces"]
    #     target_config["energy"]["stress"] = all_targets["energy"]["stress"]

    # Add non-conservative targets if requested
    if eval_non_conservative:
        target_config["non_conservative_forces"] = all_targets["non_conservative_forces"]
        target_config["non_conservative_stress"] = all_targets["non_conservative_stress"]

    # Resolve glob pattern or directory path
    aselmdb_src = _resolve_aselmdb_src(args.valid_data_path)
    if aselmdb_src is None:
        raise ValueError(
            f"Could not resolve data path: {args.valid_data_path}\n"
            "Expected a *.aselmdb file, directory with *.aselmdb files, or glob pattern"
        )

    dataset = AseDBDatasetCustomized(
        target_config,  # conf_targets is first positional argument
        config={"src": aselmdb_src},
    )

    logger_func(f"Dataset size: {len(dataset)}")

    if args.slice:
        start, end = map(int, args.slice.split("_"))
        indices = list(range(start, min(end, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
        logger_func(f"Using slice [{start}:{end}), size: {len(dataset)}")

    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        logger_func(f"Rank {rank} will process {len(sampler)} samples")

        # Update log path to be per-rank for debugging
        log_path = Path(args.log_path)
        log_stem = log_path.stem.replace("slice_start_end", f"rank{rank}")
        args.log_path = str(log_path.with_name(log_stem + log_path.suffix))
        logger_func = Logger(log_path=args.log_path)

        # Check for existing aggregated results only on rank 0
        if rank == 0 and not args.force_rerun:
            # Check for final aggregated result file
            final_result_path = log_path.parent / f"{log_path.stem.replace(f'rank{rank}', 'aggregated')}_efs.pkl"
            if final_result_path.exists():
                logger_func(f"Aggregated results exist at {final_result_path}, skipping")
                if isinstance(logger_func, Logger):
                    logger_func.close()
                return
    else:
        sampler = None

    # Wrap dataset with AseDBRawDataset like the trainer does
    raw_dataset = AseDBRawDataset(dataset)

    collate_fn = AseDBCollateFn(
        target_keys=dataset.target_keys,
        callables=[],
        batch_atom_bounds=None,
        conf_targets=dataset.conf_targets,
    )

    dataloader = DataLoader(
        raw_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger_func(f"\nDataloader created with {len(dataloader)} batches")

    logger_func("\nStarting evaluation...")
    results = eval_batched(args, model, dataloader, logger_func, device, eval_conservative, eval_non_conservative, dataset.target_info, model_dtype)

    logger_func(f"\nEvaluated {results['num_structures']} structures")

    if args.distributed:
        # Aggregate results across all ranks
        logger_func("\nAggregating results across ranks...")
        aggregated_results = gather_and_aggregate_results(results, device)

        if rank == 0:
            # Only rank 0 saves and prints final results
            logger_func(f"\nTotal structures across all ranks: {aggregated_results['num_structures']}")

            # Save aggregated results
            save_results(args, aggregated_results, logger_func)

            # Compute and print MAE
            logger_func("\n" + "=" * 60)
            logger_func("Aggregated Results (All Ranks):")
            logger_func("=" * 60)

            for mode_name in ['conservative', 'non_conservative']:
                if mode_name not in aggregated_results:
                    continue

                mode_results = aggregated_results[mode_name]
                mae_e = mean_absolute_error(mode_results["gt_e_list"], mode_results["pred_e_list"])
                mae_f = mean_absolute_error(
                    mode_results["gt_f_list"].reshape(-1),
                    mode_results["pred_f_list"].reshape(-1),
                )
                mae_s = mean_absolute_error(
                    mode_results["gt_s_list"].reshape(-1),
                    mode_results["pred_s_list"].reshape(-1),
                )
                mode_label = "Conservative" if mode_name == "conservative" else "Non-Conservative"
                logger_func(f"\n{mode_label} Mode:")
                logger_func(f"  MAE Energy: {mae_e:.6f} eV/atom")
                logger_func(f"  MAE Forces: {mae_f:.6f} eV/Å")
                logger_func(f"  MAE Stress: {mae_s:.6f} GPa")

            logger_func("=" * 60)

        if isinstance(logger_func, Logger):
            logger_func.close()
    else:
        # Compute and print MAE for each mode
        logger_func("\n" + "=" * 60)
        logger_func("Results:")
        logger_func("=" * 60)

        for mode_name in ['conservative', 'non_conservative']:
            if mode_name not in results:
                continue

            mode_results = results[mode_name]
            if len(mode_results["gt_e_list"]) > 0:
                mae_e = mean_absolute_error(mode_results["gt_e_list"], mode_results["pred_e_list"])
                mae_f = mean_absolute_error(
                    mode_results["gt_f_list"].reshape(-1),
                    mode_results["pred_f_list"].reshape(-1),
                )
                mae_s = mean_absolute_error(
                    mode_results["gt_s_list"].reshape(-1),
                    mode_results["pred_s_list"].reshape(-1),
                )
                mode_label = "Conservative" if mode_name == "conservative" else "Non-Conservative"
                logger_func(f"\n{mode_label} Mode:")
                logger_func(f"  MAE Energy: {mae_e:.6f} eV/atom")
                logger_func(f"  MAE Forces: {mae_f:.6f} eV/Å")
                logger_func(f"  MAE Stress: {mae_s:.6f} GPa")
            else:
                logger_func(f"\n{mode_name}: No results to evaluate")

        logger_func("=" * 60)


if __name__ == "__main__":
    main()
