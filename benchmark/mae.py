# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import random
from glob import glob
from pathlib import Path

import numpy as np
import torch
from ase.db import connect
from ase.io import read
from ase.units import GPa
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from utils import resolve_aselmdb_paths, resolve_xyz_paths, Logger

from upet import get_upet, save_upet
from metatrain.cli.eval import eval_model
from omegaconf import OmegaConf

from ase.io import read

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))


def load_eval_data(valid_data_path):
    if valid_data_path.endswith(".pkl"):
        print(f"Detected pickle input: {valid_data_path}")
        with open(valid_data_path, "rb") as f:
            return pickle.load(f)

    if valid_data_path.endswith(".xyz"):
        xyz_paths = resolve_xyz_paths(valid_data_path)
        print(f"Detected XYZ input with {len(xyz_paths)} file(s)")
        eval_data = []
        for xyz_path in tqdm(xyz_paths, desc="Loading XYZ files"):
            atoms_list = read(xyz_path, index=":")
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            eval_data.extend(atoms_list)
        print(f"Total structures in concatenated atomlist: {len(eval_data)}")
        return eval_data

    if ".aselmdb" in valid_data_path:
        db_paths = resolve_aselmdb_paths(valid_data_path)
        print(f"Detected ASELMDB input with {len(db_paths)} file(s)")
        eval_data = []
        for db_path in tqdm(db_paths, desc="Loading ASELMDB files"):
            with connect(db_path, readonly=True, use_lock_file=False) as database:
                eval_data.extend(row.toatoms() for row in database.select())
        return eval_data

    raise ValueError(
        f"Unsupported evaluation data format for path: {valid_data_path}. "
        "Expected a .pkl file, .xyz path/glob/directory, or .aselmdb path/glob."
    )


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

import re
import numpy as np
from ase import Atoms

def read_xyz_efs(path):
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            natoms = int(line.strip())
            comment = f.readline()

            # --- cell ---
            lat = re.search(r'Lattice="([^"]+)"', comment)
            cell = np.array(lat.group(1).split(), dtype=float).reshape(3, 3) if lat else None

            # --- energy (frame-level scalar) ---
            energy = None
            e_match = re.search(r'(?<!\w)energy=([\d.eE+\-]+)', comment)
            if e_match:
                energy = float(e_match.group(1))

            # --- figure out which columns hold pos / forces / stress ---
            # Parse Properties= tolerantly: extract valid name:T:N triples only
            props_match = re.search(r'Properties=(\S+)', comment)
            col_map = {}   # name -> (start_col, count)
            cursor = 0
            if props_match:
                tokens = props_match.group(1).split(':')
                i = 0
                while i < len(tokens):
                    if not tokens[i]:          # skip empty (from ::)
                        i += 1
                        continue
                    # look ahead for type and integer count
                    if i + 2 >= len(tokens):
                        break
                    name, typ = tokens[i], tokens[i+1]
                    # find next integer token for count
                    j = i + 2
                    while j < len(tokens) and not re.fullmatch(r'\d+', tokens[j]):
                        j += 1
                    if j >= len(tokens):
                        break
                    count = int(tokens[j])
                    col_map[name] = (cursor, count)
                    cursor += count
                    i = j + 1

            # --- per-atom stress (9 components) comes from non_conservative_stress ---
            stress_key = next(
                (k for k in col_map if 'stress' in k.lower() and 'feature' not in k.lower()),
                None
            )
            force_key = next(
                (k for k in col_map if 'force' in k.lower() and 'feature' not in k.lower()),
                None
            )

            species, positions, forces, stresses = [], [], [], []
            for _ in range(natoms):
                cols = f.readline().split()

                # species is always col 0
                species.append(cols[0])

                # pos
                if 'pos' in col_map:
                    s, n = col_map['pos']
                    positions.append([float(x) for x in cols[s:s+n]])

                # forces
                if force_key:
                    s, n = col_map[force_key]
                    forces.append([float(x) for x in cols[s:s+n]])

                # stress
                if stress_key:
                    s, n = col_map[stress_key]
                    stresses.append([float(x) for x in cols[s:s+n]])

            atoms = Atoms(symbols=species, positions=positions, cell=cell, pbc=True)

            info = {}
            if energy is not None:
                info['energy'] = energy
            arrays = {}
            if forces:
                arrays['forces'] = np.array(forces)
            if stresses:
                arrays['stresses'] = np.array(stresses)   # shape (natoms, 9)

            atoms.info.update(info)
            atoms.arrays.update(arrays)
            frames.append(atoms)

    return frames

def eval(args, eval_data, logger):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_path = Path(args.log_path)
    if not log_path.exists() or args.force_rerun:
        logger("Loading calculator...")
        ckpt_path = Path(args.ckpt_path)
        exported_model_path = ckpt_path.with_suffix(".pt")
        if not exported_model_path.exists():
            save_upet(
                checkpoint_path=str(ckpt_path),
                output=str(exported_model_path)
            )
        exported_model = torch.jit.load(str(exported_model_path))
        options = {
            # "systems": "/mnt/shared-storage-user/lijiahang/datasets/non_equi_test_data/data.xyz"
            "systems": "/home/lijiahang/OmniMat_mattersim_work/mattersim_work/test_data/train/data.xyz",
        }
        cfg = OmegaConf.create(options)
        eval_model(
            exported_model, 
            options=cfg,
            output=args.log_path,
            batch_size=args.bs
        )

    # output_atoms_list = read_xyz_efs(args.log_path)
    # usage
    frames = read_xyz_efs(args.log_path)
    atoms = frames[0]
    print(atoms.info.get('energy'))
    print(atoms.arrays.get('forces'))
    print(atoms.arrays.get('stresses'))
    

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

    parser.add_argument(
        "--valid_data_path", 
        type=str, 
        default=None, 
        help="valid data path"
    )
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
    parser.add_argument(
        "--force_rerun",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Force re-running the evaluation.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=16,
        help="batch size",
    )

    # unused
    parser.add_argument("--fidelity", type=str, default="pbe", choices=["pbe", "r2scan"], help="Fidelity level for the calculator")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--device", type=str, default="cuda", help="device")

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