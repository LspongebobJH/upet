# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch

from ase.units import GPa
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from upet.calculator import UPETCalculator
from metatrain.utils.io import load_model as load_metatrain_model
from tools.utils import load_eval_data, Logger

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

def save_results(args, results, logger):
    logger("\nSaving results...")
    output_dir = Path(args.log_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_path)
    output_file = log_path.with_name(log_path.stem + "_efs.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    logger(f"Results saved to: {output_file}")
    return output_file


def eval(args, eval_data, logger):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    calc = UPETCalculator(
        # model="pet-oam-xl", 
        checkpoint_path=args.ckpt_path,
        version="1.0.0", 
        device='cuda'
    )

    gt_e_list = []
    pred_e_list = []
    gt_f_list = []
    pred_f_list = []
    gt_s_list = []
    pred_s_list = []

    for i, atoms in enumerate(tqdm(eval_data, desc="Evaluating on structures")):
        gt_energy = atoms.get_potential_energy() / len(atoms)
        gt_forces = atoms.get_forces()
        gt_stress = atoms.get_stress(voigt=False) / GPa

        atoms.set_calculator(calc)
        pred_energy = atoms.get_potential_energy() / len(atoms)
        pred_forces = atoms.get_forces()
        pred_stress = atoms.get_stress(voigt=False) / GPa

        gt_e_list.append(gt_energy)
        pred_e_list.append(pred_energy)
        gt_f_list.append(gt_forces)
        pred_f_list.append(pred_forces)
        gt_s_list.append(gt_stress)
        pred_s_list.append(pred_stress)

        if i % 100 == 0:
            logger(f"Processed {i}/{len(eval_data)} structures")

    results = {
        "gt_e_list": np.asarray(gt_e_list),
        "pred_e_list": np.asarray(pred_e_list),
        "gt_f_list": np.concatenate(gt_f_list, axis=0),
        "pred_f_list": np.concatenate(pred_f_list, axis=0),
        "gt_s_list": np.concatenate(gt_s_list, axis=0),
        "pred_s_list": np.concatenate(pred_s_list, axis=0),
        "num_structures": len(gt_e_list),
        "slice": args.slice,
    }

    if args.distributed:
        save_results(args, results, logger)
    else:
        mae_e = mean_absolute_error(results["gt_e_list"], results["pred_e_list"])
        mae_f = mean_absolute_error(results["gt_f_list"], results["pred_f_list"])
        mae_s = mean_absolute_error(results["gt_s_list"], results["pred_s_list"])
        print(f"MAE Energy | MAE Forces | MAE Stress: {mae_e:.6f}, {mae_f:.6f}, {mae_s:.6f}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--valid_data_path", type=str, default=None, help="valid data path")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a pre-trained model for fine-tuning. If not provided, trains a new model from scratch.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs/benchmark_efs/slice_start_end.log",
        help="Path to the log file.",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="slice is like 0_10, to take 0, 1, ..., 9 elements. In distributed mode, this slice is automatically sharded across ranks.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Whether to shard evaluation data across distributed ranks using WORLD_SIZE and RANK.",
    )
    parser.add_argument("--fidelity", type=str, default="pbe", choices=["pbe", "r2scan"], help="Fidelity level for the calculator")
    parser.add_argument(
        "--force_rerun",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Force re-running the evaluation.",
    )

    args = parser.parse_args()

    print("Loading evaluation data...")
    eval_data = load_eval_data(args.valid_data_path)
    num_samples = len(eval_data)

    if args.distributed:
        torch.cuda.set_device(local_rank)
        assert args.device == "cuda", "Manual specification of device is not supported in distributed mode."

        if args.slice is not None:
            slice_start, slice_end = (int(x) for x in args.slice.split("_"))
            sliced_num_samples = slice_end - slice_start
            num_samples_per_proc = (sliced_num_samples + world_size - 1) // world_size
            start = slice_start + rank * num_samples_per_proc
            end = min(start + num_samples_per_proc, slice_end)
        else:
            num_samples_per_proc = (num_samples + world_size - 1) // world_size
            start = rank * num_samples_per_proc
            end = min(start + num_samples_per_proc, num_samples)

        if start >= end:
            print(f"Rank {rank} has no samples assigned, skipping.")
            return

        args.slice = (start, end)

        log_path = Path(args.log_path)
        log_file_name = log_path.stem
        new_log_file_name = log_file_name.replace("slice_start_end", f"slice_{start}_{end}")
        new_log_path = log_path.with_name(new_log_file_name + log_path.suffix)
        args.log_path = str(new_log_path)

        result_path = new_log_path.with_name(new_log_file_name + "_efs.pkl")
        if glob(str(result_path)) and not args.force_rerun:
            print(f"Found existing EFS result file: {glob(str(result_path))[0]}, Skip.")
            return

        logger = Logger(log_path=args.log_path)

    else:
        if args.slice is not None:
            args.slice = tuple(int(x) if x else None for x in args.slice.split("_"))
        logger = print

    if args.slice is not None:
        start, end = args.slice
        eval_data = eval_data[start:end]
        logger(f"Evaluating slice [{start}:{end}) with {len(eval_data)} structures")
    else:
        logger(f"Evaluating full dataset with {num_samples} structures")

    logger("=" * 60)
    logger("mattersim model test on EFS dataset")
    logger("=" * 60)
    logger(f"Model path: {args.ckpt_path}")
    logger(f"Device: {args.device}")
    logger(f"Slice: {args.slice}")
    logger(f"logger path: {args.log_path}")
    logger("=" * 60)

    eval(args, eval_data, logger)
    if isinstance(logger, Logger):
        logger.close()


if __name__ == "__main__":
    main()