"""
Copyright (c) Meta Platforms, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import glob
import json
import random
import time
import traceback
import warnings
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE
from moyopy import MoyoDataset
from moyopy.interface import MoyoAdapter
from pymatviz.enums import Key
from tqdm import tqdm
from argparse import ArgumentParser
import os

from matbench_discovery import today
from matbench_discovery.data import DataFilesCustomized as DataFiles
from matbench_discovery.phonons import check_imaginary_freqs
from matbench_discovery.phonons import thermal_conductivity as ltc
from upet.calculator import UPETCalculator

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")


def seed_everywhere(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


rank, local_rank, group_rank, world_size = (
    os.environ.get("RANK", "0"),
    os.environ.get("LOCAL_RANK", "0"),
    os.environ.get("GROUP_RANK", "0"),
    os.environ.get("WORLD_SIZE", "1"),
)

rank, local_rank, group_rank, world_size = (
    int(rank),
    int(local_rank),
    int(group_rank),
    int(world_size),
)


class KappaSRMERunner:
    def __init__(
        self,
        seed: int,
        ckpt_path: str,
        save_dir: str,
        atom_disp: float,
        slice_tag: str | None = None,
        non_conservative: bool = False,
        is_plusminus: bool = False,
    ) -> None:
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.save_dir = save_dir
        self.non_conservative = non_conservative
        self.atom_disp = atom_disp
        self.slice_tag = slice_tag
        self.is_plusminus = is_plusminus

    def run(self, atoms_list: list) -> None:
        # Relaxation parameters
        max_steps = 300
        force_max = 1e-4  # Run until the forces are smaller than this in eV/A
        symprec = 1e-5
        enforce_relax_symm = True
        conductivity_broken_symm = False
        prog_bar = True
        save_forces = False  # Save force sets to file
        temperatures = [300]  # Temperatures to calculate conductivity at in Kelvin
        is_plusminus = self.is_plusminus  # Whether to use plus-minus displacements for fc calculations, which can improve accuracy at the cost of doubling the number of calculations. This is utilized by pet official ksrme eval codes.

        seed_everywhere(self.seed)

        # Setup model and calculator
        calculator = UPETCalculator(
            # model="pet-oam-xl",
            checkpoint_path=self.ckpt_path,
            version="1.0.0",
            device="cuda",
            non_conservative=self.non_conservative,
        )

        force_results: dict[str, dict[str, Any]] = {}
        kappa_results: dict[str, dict[str, Any]] = {}

        tqdm_bar = tqdm(
            atoms_list, desc="Conductivity calculation: ", disable=not prog_bar
        )

        test_metrics = {}
        save_dir = Path(self.save_dir)
        (save_dir).mkdir(parents=True, exist_ok=True)

        # Log run parameters
        timestamp = f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}"
        run_params = {
            "timestamp": timestamp,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "versions": {
                dep: version(dep) for dep in ("numpy", "torch", "matbench_discovery")
            },
            "optimizer": "FIRE",
            "max_steps": max_steps,
            "force_max": force_max,
            "symprec": symprec,
            "enforce_relax_symm": enforce_relax_symm,
            "conductivity_broken_symm": conductivity_broken_symm,
            "temperatures": temperatures,
            "displacement_distance": self.atom_disp,
            "n_structures": len(atoms_list),
        }
        with open(save_dir / "run_params.json", mode="w") as file:
            json.dump(run_params, file, indent=4)

        start_time = time.time()
        for atoms in tqdm_bar:
            mat_id = atoms.info[Key.mat_id]
            init_info = deepcopy(atoms.info)
            formula = atoms.get_chemical_formula()
            spg_num = MoyoDataset(MoyoAdapter.from_atoms(atoms)).number
            info_dict: dict[str, Any] = {
                str(Key.mat_id): mat_id,
                str(Key.formula): formula,
                str(Key.spg_num): spg_num,
            }
            err_dict: dict[str, list[str]] = {"errors": [], "error_traceback": []}

            tqdm_bar.set_postfix_str(mat_id, refresh=True)

            # Initialize relax_dict to avoid "possibly unbound" errors
            relax_dict = {
                "max_stress": None,
                "reached_max_steps": False,
                "broken_symmetry": False,
            }

            # Relaxation - using standard approach from other scripts
            try:
                atoms.calc = calculator
                if max_steps > 0:
                    if enforce_relax_symm:
                        atoms.set_constraint(FixSymmetry(atoms))

                    # Use standard mask for no-tilt constraint
                    filtered_atoms = FrechetCellFilter(
                        atoms, mask=[True] * 3 + [False] * 3
                    )

                    optimizer = FIRE(
                        filtered_atoms, logfile=save_dir / f"relax_{mat_id}.log"
                    )
                    optimizer.run(fmax=force_max, steps=max_steps)

                    reached_max_steps = optimizer.nsteps >= max_steps
                    if reached_max_steps:
                        print(f"{mat_id=} reached {max_steps=} during relaxation")

                    max_stress = (
                        atoms.get_stress().reshape((2, 3), order="C").max(axis=1)
                    )
                    atoms.calc = None
                    atoms.constraints = None
                    atoms.info = init_info | atoms.info

                    # Check if symmetry was broken during relaxation
                    relaxed_spg = MoyoDataset(MoyoAdapter.from_atoms(atoms)).number
                    broken_symmetry = spg_num != relaxed_spg
                    relax_dict = {
                        "max_stress": max_stress,
                        "reached_max_steps": reached_max_steps,
                        "relaxed_space_group_number": relaxed_spg,
                        "broken_symmetry": broken_symmetry,
                    }

            except Exception as exc:
                warnings.warn(
                    f"Failed to relax {formula=}, {mat_id=}: {exc!r}", stacklevel=2
                )
                traceback.print_exc()
                err_dict["errors"].append(f"RelaxError: {exc!r}")
                err_dict["error_traceback"].append(traceback.format_exc())
                kappa_results[mat_id] = info_dict | relax_dict | err_dict
                continue

            # Calculation of force sets
            try:
                # Initialize phono3py with the relaxed structure
                ph3 = ltc.init_phono3py(
                    atoms,
                    fc2_supercell=atoms.info["fc2_supercell"],
                    fc3_supercell=atoms.info["fc3_supercell"],
                    q_point_mesh=atoms.info["q_point_mesh"],
                    displacement_distance=self.atom_disp,
                    symprec=symprec,
                    is_plusminus=is_plusminus,
                )

                # Calculate force constants and frequencies
                ph3, fc2_set, freqs = ltc.get_fc2_and_freqs(
                    ph3,
                    calculator=calculator,
                    pbar_kwargs={"leave": False, "disable": not prog_bar},
                )

                # Check for imaginary frequencies
                has_imaginary_freqs = check_imaginary_freqs(freqs)
                freqs_dict = {
                    Key.has_imag_ph_modes: has_imaginary_freqs,
                    Key.ph_freqs: freqs,
                }

                # If conductivity condition is met, calculate fc3
                ltc_condition = not has_imaginary_freqs and (
                    not relax_dict["broken_symmetry"] or conductivity_broken_symm
                )

                if ltc_condition:  # Calculate third-order force constants
                    print(f"Calculating FC3 for {mat_id}")
                    fc3_set = ltc.calculate_fc3_set(
                        ph3,
                        calculator=calculator,
                        pbar_kwargs={"leave": False, "disable": not prog_bar},
                    )
                    ph3.produce_fc3(symmetrize_fc3r=True)
                else:
                    fc3_set = []

                if save_forces:
                    force_results[mat_id] = {"fc2_set": fc2_set, "fc3_set": fc3_set}

                if not ltc_condition:
                    kappa_results[mat_id] = info_dict | relax_dict | freqs_dict
                    warnings.warn(
                        f"{mat_id=} has imaginary frequencies or broken symmetry",
                        stacklevel=2,
                    )
                    continue

            except Exception as exc:
                warnings.warn(
                    f"Failed to calculate force sets {mat_id}: {exc!r}", stacklevel=2
                )
                traceback.print_exc()
                err_dict["errors"].append(f"ForceConstantError: {exc!r}")
                err_dict["error_traceback"].append(traceback.format_exc())
                kappa_results[mat_id] = info_dict | relax_dict | err_dict
                continue

            try:  # Calculate thermal conductivity
                ph3, kappa_dict, _cond = ltc.calculate_conductivity(
                    ph3, temperatures=temperatures
                )
                print(f"Finish calculating kappa for {mat_id}")
            except Exception as exc:
                warnings.warn(
                    f"Failed to calculate conductivity {mat_id}: {exc!r}", stacklevel=2
                )
                traceback.print_exc()
                err_dict["errors"].append(f"ConductivityError: {exc!r}")
                err_dict["error_traceback"].append(traceback.format_exc())
                kappa_results[mat_id] = info_dict | relax_dict | freqs_dict
                continue

            kappa_results[mat_id] = (
                info_dict | relax_dict | freqs_dict | kappa_dict | err_dict
            )

        elapsed = time.time() - start_time
        test_metrics["running_time"] = elapsed

        # Save results
        df_kappa = pd.DataFrame(kappa_results).T
        # df_kappa.index.name = Key.mat_id # jiahang (note): this line has bug
        json_path = f"{save_dir}/{today}-kappa-103-slice-{self.slice_tag}.json.gz"
        df_kappa.reset_index().to_json(json_path)
        print(f"Saved kappa results to {json_path}")

        if save_forces:
            force_out_path = (
                f"{save_dir}/{today}-kappa-103-" f"force-sets-{self.slice_tag}.json.gz"
            )
            df_force = pd.DataFrame(force_results).T
            df_force.index.name = Key.mat_id
            df_force.reset_index().to_json(force_out_path)
            print(f"Saved force results to {force_out_path}")


def main():
    parser = ArgumentParser(description="Benchmark phonon calculations with KappaSRME")
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint file"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="test_phonon",
        help="Path for the save directory",
    )
    parser.add_argument(
        "--distributed",
        default=False,
        action="store_true",
        help="Whether to run in distributed mode",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Optional slice of the dataset to run on, in the format 'start_end'",
    )

    parser.add_argument(
        "--non_conservative",
        action="store_true",
        help="Whether to use non-conservative version of the model (if supported)",
    )
    parser.add_argument(
        "--is_plusminus",
        action="store_true",
        help="whether to use minus-plus displacements for fc calculations, which can improve accuracy at the cost of doubling the number of calculations. This must be enabled to reproduce official pet ksrme eval results.",
    )
    args = parser.parse_args()

    atoms_list = read(
        DataFiles.phonondb_pbe_103_structures.path, format="extxyz", index=":"
    )

    if args.distributed:
        torch.cuda.set_device(local_rank)
        if args.slice is not None:
            warnings.warn(
                f"Distributed evaluation is only conducted in slice {args.slice}"
            )
            slice_start, slice_end = (int(x) for x in args.slice.split("_"))
            sliced_num_samples = slice_end - slice_start
            num_samples_per_proc = (sliced_num_samples + world_size - 1) // world_size
            start = slice_start + rank * num_samples_per_proc
            end = min(start + num_samples_per_proc, slice_end)
        else:
            num_samples = len(atoms_list)
            num_samples_per_proc = (num_samples + world_size - 1) // world_size
            start = rank * num_samples_per_proc
            end = min(start + num_samples_per_proc, num_samples)

        if start >= end:
            print(f"Rank {rank} has no samples assigned, skipping.")
            return

        args.slice = f"{start}_{end}"
        atoms_list = atoms_list[start:end]
        print(f"Process {rank} handling samples from {start} to {end}")

    runner = KappaSRMERunner(
        seed=42,
        ckpt_path=args.ckpt_path,
        save_dir=args.save_dir,
        atom_disp=0.03,
        slice_tag=args.slice,
        non_conservative=args.non_conservative,
        is_plusminus=args.is_plusminus,
    )

    runner.run(atoms_list=atoms_list)


if __name__ == "__main__":
    main()
