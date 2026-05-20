from __future__ import annotations

import argparse
import bz2
from glob import glob
import json
import os
import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING
import pickle
# from fairchem_omnimat.core import OCPCalculator
# from fairchem.core import OCPCalculator

warnings.filterwarnings("ignore", message="No Pauling electronegativity for")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ase")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

import numpy as np
import pandas as pd
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm
from copy import deepcopy

from matbench_discovery import today
from matbench_discovery.data import df_wbm, DataFilesCustomized
from matbench_discovery.energy import get_e_form_per_atom, mp_elemental_ref_energies
from matbench_discovery.metrics_old import stable_metrics
from matbench_discovery.enums import MbdKey
from batch_relax import BatchRelaxer
from pymatviz.enums import Key
from utils import Logger
import torch
from upet.calculator import UPETCalculator, get_upet
from metatomic_ase import MetatomicCalculator, SymmetrizedCalculator

if TYPE_CHECKING:
    import ase

rank, local_rank, world_size = \
    os.environ.get("RANK", "0"), \
    os.environ.get("LOCAL_RANK", "0"), \
    os.environ.get("WORLD_SIZE", "1")

rank, local_rank, world_size = \
    int(rank), \
    int(local_rank), \
    int(world_size)

def load_wbm_initial_atoms(
    file_path: str | Path, slice_tuple: tuple[int | None, int | None] | None = None, logger: Logger = None
) -> list[ase.Atoms]:
    file_path = Path(file_path)
    
    logger(f"Loading structures from {file_path}...")
    
    # Read bz2-compressed JSON file
    with bz2.open(file_path, "rt") as f:
        data = json.load(f)
    
    # Debug: check data structure
    logger(f"Data type: {type(data)}")
    if isinstance(data, dict):
        logger(f"Number of dictionary keys: {len(data)}")
        logger(f"First 10 keys: {list(data.keys())[:10]}")
        
        # Check the first value
        first_key = list(data.keys())[0]
        first_value = data[first_key]
        logger(f"Type of value for first key '{first_key}': {type(first_value)}")
        
        if isinstance(first_value, dict):
            logger(f"  Value keys: {list(first_value.keys())[:10]}")
        elif isinstance(first_value, list):
            logger(f"  List length: {len(first_value)}")
            if len(first_value) > 0:
                logger(f"  Type of first element: {type(first_value[0])}")

    
    # Convert JSON data to ASE Atoms object
    atoms_list = []
    
    if isinstance(data, dict):
        # Check data structure format
        # Case 1: {material_id: {initial_structure: ..., ...}, ...} - Each key is material_id
        # Case 2: {field_name: {index: value, ...}, ...} - Columnar dictionary format, needs to be combined by index
        # Case 3: {field_name: [values], ...} - Columnar list format, needs to be combined by index
        sample_key = list(data.keys())[0]
        sample_value = data[sample_key]
        
        # Check if it is a columnar format (field names as keys, values ​​are dictionaries or lists).
        is_columnar = False
        if "material_id" in data and "initial_structure" in data:
            # Check if the value is a dictionary (index as key)
            if isinstance(data["material_id"], dict) and isinstance(data["initial_structure"], dict):
                is_columnar = True
                logger("Detected columnar dictionary format; combine data by index...")
            elif isinstance(data["material_id"], list) and isinstance(data["initial_structure"], list):
                is_columnar = True
                logger("Detected columnar list format; combine data by index...")
        
        if is_columnar:
            # Columnar format: needs to be combined by index
            material_ids_dict = data["material_id"]
            structures_dict = data["initial_structure"]
            
            # Get all indices (from material_id or initial_structure)
            if isinstance(material_ids_dict, dict):
                indices = sorted(material_ids_dict.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
            else:
                indices = list(range(len(material_ids_dict)))
            
            if slice_tuple:
                start, end = slice_tuple
                indices = indices[start:end]
            
            # Combine data: get corresponding material_id and structure by index
            data_items = []
            for idx in indices:
                if isinstance(material_ids_dict, dict):
                    mat_id = material_ids_dict.get(str(idx), material_ids_dict.get(idx, f"unknown_{idx}"))
                else:
                    mat_id = material_ids_dict[int(idx)] if idx.isdigit() and int(idx) < len(material_ids_dict) else f"unknown_{idx}"
                
                if isinstance(structures_dict, dict):
                    struct_data = structures_dict.get(str(idx), structures_dict.get(idx))
                else:
                    struct_data = structures_dict[int(idx)] if idx.isdigit() and int(idx) < len(structures_dict) else None
                
                if struct_data is not None:
                    data_items.append((mat_id, struct_data))
        else:
            # Linear format: Each key is material_id
            items_list = list(data.items())
            if slice_tuple:
                start, end = slice_tuple
                items_list = items_list[start:end]
            data_items = items_list
    else:
        data_items = data
        if slice_tuple:
            start, end = slice_tuple
            data_items = data[start:end]
    
    for item in tqdm(data_items, desc="Loading structures"):
        try:
            if isinstance(data, dict):
                # Check if it is a columnar format (item is (mat_id, struct_data) tuple)
                # or a linear format (item is (key, entry_dict) tuple)
                if isinstance(item, tuple) and len(item) == 2:
                    first_elem, second_elem = item
                    # If the second element is a dictionary and contains initial_structure or computed_structure_entry, it is a linear format
                    if isinstance(second_elem, dict) and ("initial_structure" in second_elem or "computed_structure_entry" in second_elem):
                        # Linear format: {material_id: {initial_structure: ..., ...}, ...}
                        key, entry_dict = item
                        mat_id = key
                    else:
                        # Columnar format: (material_id, structure_data)
                        mat_id, struct_data = item
                        # Wrap struct_data into entry_dict format
                        entry_dict = {"initial_structure": struct_data}
                else:
                    # Default to linear format
                    key, entry_dict = item
                    mat_id = key
                
                # If entry_dict is not a dictionary, skip
                if not isinstance(entry_dict, dict):
                    continue
                
                # Check if it contains initial_structure or computed_structure_entry
                if "initial_structure" not in entry_dict and "computed_structure_entry" not in entry_dict:
                    # Might be metadata field, skip
                    continue
                
                # wbm_cses_plus_init_structs format: each entry is a dictionary containing CSE and initial structure
                from pymatgen.core import Structure
                
                structure = None
                
                # Prefer to get structure from initial_structure key
                if "initial_structure" in entry_dict:
                    struct_data = entry_dict["initial_structure"]
                    # struct_data might be a dict or already a Structure object
                    if isinstance(struct_data, Structure):
                        structure = struct_data
                    elif isinstance(struct_data, dict):
                        # Check if it is a valid Structure dict
                        if "lattice" in struct_data and "sites" in struct_data:
                            # Structure dict
                            structure = Structure.from_dict(struct_data)
                        elif "@module" in struct_data and struct_data.get("@module") == "pymatgen.core.structure":
                            structure = Structure.from_dict(struct_data)
                        elif all(k.isdigit() for k in struct_data.keys()):
                            # If all keys are digit strings, it might be a nested columnar format
                            # Try to get the first value (index '0')
                            if "0" in struct_data:
                                nested_struct = struct_data["0"]
                                if isinstance(nested_struct, dict) and ("lattice" in nested_struct and "sites" in nested_struct):
                                    structure = Structure.from_dict(nested_struct)
                                else:
                                    # Try to parse directly as Structure dict
                                    structure = Structure.from_dict(nested_struct)
                            else:
                                # If there is no '0', try the first key
                                first_key = list(struct_data.keys())[0]
                                nested_struct = struct_data[first_key]
                                if isinstance(nested_struct, dict) and ("lattice" in nested_struct and "sites" in nested_struct):
                                    structure = Structure.from_dict(nested_struct)
                                else:
                                    structure = Structure.from_dict(nested_struct)
                        else:
                            # Try to parse directly using Structure.from_dict
                            try:
                                structure = Structure.from_dict(struct_data)
                            except Exception as e:
                                # If all fails, print debug information
                                logger(f"\nDebug: Unable to parse initial_structure: {e}")
                                logger(f"  Type of struct_data: {type(struct_data)}")
                                logger(f"  Keys of struct_data: {list(struct_data.keys())[:10]}")
                                if isinstance(struct_data, dict) and len(struct_data) > 0:
                                    first_val = list(struct_data.values())[0]
                                    logger(f"  Type of first value: {type(first_val)}")
                                    if isinstance(first_val, dict):
                                        logger(f"  Keys of first value: {list(first_val.keys())[:10]}")
                                raise
                    else:
                        raise ValueError(f"Type of initial_structure is not as expected: {type(struct_data)}")
                
                # If there is no initial_structure, try to get it from computed_structure_entry
                if structure is None and "computed_structure_entry" in entry_dict:
                    cse_data = entry_dict["computed_structure_entry"]
                    from pymatgen.entries.computed_entries import ComputedStructureEntry
                    if isinstance(cse_data, ComputedStructureEntry):
                        structure = cse_data.structure
                    elif isinstance(cse_data, dict):
                        if "structure" in cse_data:
                            struct_data = cse_data["structure"]
                            if isinstance(struct_data, Structure):
                                structure = struct_data
                            elif isinstance(struct_data, dict):
                                structure = Structure.from_dict(struct_data)
                        else:
                            # Parse CSE
                            cse = ComputedStructureEntry.from_dict(cse_data)
                            structure = cse.structure
                
                # If entry_dict is structure data
                if structure is None and "lattice" in entry_dict and "sites" in entry_dict:
                    structure = Structure.from_dict(entry_dict)
                
                if structure is None:
                    logger(f"\n警告: 无法找到结构数据，key: {key}")
                    logger(f"  entry_dict 的键: {list(entry_dict.keys())[:10]}")
                    raise ValueError(f"无法从 entry_dict 中提取结构数据")
                
                atoms = AseAtomsAdaptor.get_atoms(structure)
                atoms.info[Key.mat_id] = mat_id
            else:
                # If data is a list, process directly
                if isinstance(item, dict):
                    from pymatgen.core import Structure
                    # Process different formats
                    if "initial_structure" in item:
                        struct_dict = item["initial_structure"]
                        if isinstance(struct_dict, dict):
                            structure = Structure.from_dict(struct_dict)
                        else:
                            structure = struct_dict
                    elif "lattice" in item and "sites" in item:
                        structure = Structure.from_dict(item)
                    elif "structure" in item:
                        struct_dict = item["structure"]
                        if isinstance(struct_dict, dict):
                            structure = Structure.from_dict(struct_dict)
                        else:
                            structure = struct_dict
                    else:
                        logger(f"\n警告: 无法解析结构数据")
                        logger(f"  item 的键: {list(item.keys())[:10]}")
                        raise ValueError("无法从 item 中提取结构数据")
                    atoms = AseAtomsAdaptor.get_atoms(structure)
                else:
                    # Try using ASE atoms
                    atoms = item
            
            atoms_list.append(atoms)
        except Exception as e:
            mat_id_str = ""
            if isinstance(data, dict):
                try:
                    mat_id_str = f", material_id: {item[0] if isinstance(item, tuple) else 'unknown'}"
                except:
                    pass
            logger(f"\nWarning: Failed to load structure: {e}{mat_id_str}")
            logger(f"  Skipping this structure...")
            continue
    
    return atoms_list


def relax_atoms_list(
    atoms_list: list[ase.Atoms],
    checkpoint_path: str | Path,
    device: str = "cuda",
    fmax: float = 0.02,
    steps: int = 500,
    logger: Logger = None,
    max_natoms_per_batch: int = 512,
    relaxer_type: str = "sequential",
    rot: bool = False
) -> list[ase.Atoms]:
    """Optimize a list of atomic structures.

    Args:
        atoms_list: List of ASE Atoms objects to be optimized
        checkpoint_path: Path to the mattersim model file
        device: Computing device ('cuda' or 'cpu')
        fmax: Maximum force threshold (eV/Å), optimization will stop when this threshold is reached
        steps: Maximum number of optimization steps

    Returns:
        List of optimized ASE Atoms objects
    """
    logger(f"Loading model: {checkpoint_path}")

    model = get_upet(checkpoint_path=checkpoint_path)
    calc = MetatomicCalculator(model, device=device)
    if rot:
        calc = SymmetrizedCalculator(calc, batch_size=16, include_inversion=False)
    logger(f"Model loaded... Device: {device}")
    relaxed_atoms_list = []

    if relaxer_type == "batch":

        relaxer = BatchRelaxer(
            calc, 
            fmax=fmax, 
            filter="FRECHETCELLFILTER", 
            max_natoms_per_batch=max_natoms_per_batch,
            rot=rot
        )
        _atoms_list = deepcopy(atoms_list)
        relaxer.relax(_atoms_list)
        relaxed_atoms_list = list(relaxer.final_atoms.values())

        for atoms in relaxed_atoms_list:
            atoms.set_calculator(calc)

    else:
        for i, atoms in enumerate(tqdm(atoms_list,  desc="Processing Structures...")):
            try:
                atoms = atoms.copy()  # Avoid modifying the original structure
                atoms.set_calculator(calc)
                cell_filter = FrechetCellFilter(atoms)
                opt = FIRE(cell_filter)
                opt.run(fmax=fmax, steps=steps)
                
                # 检查是否收敛
                if opt.get_number_of_steps() == steps:
                    atoms.info["converged"] = False
                else:
                    atoms.info["converged"] = True

                relaxed_atoms_list.append(atoms)

                if i % 10 == 0:
                    logger(f"Optimized {i}/{len(atoms_list)} structures on local rank {local_rank}...")
            except Exception as exc:
                logger(f"Warning: Optimization failed (material_id: {atoms.info.get('material_id', 'unknown')}): {exc}")
                # Even if failed, save but mark as unconverged
                atoms.info["converged"] = False
                relaxed_atoms_list.append(atoms)

    return relaxed_atoms_list


def parse_relaxed_atoms_list_as_df(
    atoms_list: list[ase.Atoms], *, keep_unconverged: bool = True, logger: Logger = None
) -> pd.DataFrame:
    """Parse a list of relaxed atomic structures into a DataFrame.

    Args:
        atoms_list: List of relaxed ASE Atoms objects
        keep_unconverged: Whether to keep unconverged structures
    Returns:
        DataFrame containing formation energies and other information
    """
    e_form_col = "e_form_per_atom_mattersim"

    logger("Loading WBM computed structure entries...")
    df_files = DataFilesCustomized
    wbm_cse_paths = df_files.wbm_computed_structure_entries.path
    df_wbm_cse = pd.read_json(wbm_cse_paths).set_index(Key.mat_id)

    def parse_cse_dict(dct):
        """Parse CSE dictionary, handling missing keys."""
        # Ensure required keys exist
        if "correction" not in dct:
            dct["correction"] = 0.0
        if "energy_adjustments" not in dct:
            dct["energy_adjustments"] = []
        # Try parsing
        try:
            return ComputedStructureEntry.from_dict(dct)
        except Exception as e:
            # If it still fails, try a more lenient approach
            logger(f"Warning: Failed to parse CSE: {e}")
            logger(f"  Dictionary keys: {list(dct.keys())[:10]}")
            # Try adding more default values
            if "entry_id" not in dct:
                dct["entry_id"] = None
            if "data" not in dct:
                dct["data"] = {}
            return ComputedStructureEntry.from_dict(dct)

    df_wbm_cse[Key.computed_structure_entry] = [
        parse_cse_dict(dct)
        for dct in tqdm(df_wbm_cse[Key.computed_structure_entry], desc="Parsing CSEs")
    ]

    logger(f"Found {len(df_wbm_cse):,} CSEs")
    logger(f"Found {len(atoms_list):,} relaxed structures")

    def parse_single_atoms(atoms: ase.Atoms) -> tuple[str, bool, float, float, float]:
        """Parse a single atomic structure and calculate formation energy."""
        structure = AseAtomsAdaptor.get_structure(atoms)
        energy = atoms.get_potential_energy()
        mat_id = atoms.info.get(Key.mat_id, "unknown")
        converged = atoms.info.get("converged", False)

        if mat_id not in df_wbm_cse.index:
            logger(f"Warning: CSE not found for material_id {mat_id}, skipping")
            return mat_id, converged, np.nan, energy, energy

        cse = df_wbm_cse.loc[mat_id, Key.computed_structure_entry]
        cse._energy = energy  # noqa: SLF001
        cse._structure = structure  # noqa: SLF001

        processed = MaterialsProject2020Compatibility(check_potcar=False).process_entry(cse)
        corrected_energy = processed.energy if processed is not None else energy
        
        # 使用 get_e_form_per_atom 计算形成能
        entry_to_use = processed if processed is not None else cse
        formation_energy = get_e_form_per_atom(
            entry_to_use,
            elemental_ref_energies=mp_elemental_ref_energies
        )

        return mat_id, converged, formation_energy, energy, corrected_energy

    mat_id_list, converged_list, e_form_list = [], [], []
    energy_list, corrected_energy_list = [], []

    for atoms in tqdm(atoms_list, desc="Processing relaxed structures"):
        mat_id, converged, formation_energy, energy, corrected_energy = (
            parse_single_atoms(atoms)
        )
        if not keep_unconverged and not converged:
            continue
        mat_id_list.append(mat_id)
        converged_list.append(converged)
        e_form_list.append(formation_energy)
        energy_list.append(energy)
        corrected_energy_list.append(corrected_energy)

    return pd.DataFrame(
        {
            Key.mat_id: mat_id_list,
            "converged": converged_list,
            e_form_col: e_form_list,
            "mattersim_energy": energy_list,
            "corrected_energy": corrected_energy_list,
        }
    )


def save_results(
    args: argparse.Namespace,
    df_results: pd.DataFrame,
    logger: Logger
) -> Path:
    # Save results
    logger("\nSaving results...")
    output_dir = Path(args.log_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    _path = Path(args.log_path)
    output_file = _path.with_name(_path.stem + f'_{today}-wbm-IS2RE.csv.gz')
    df_results.to_csv(output_file, index=False)
    logger(f"\nResults saved to: {output_file}")
    
    # no need to compute metrics for each slice separately. 
def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test mattersim model on Matbench Discovery dataset"
    )
    parser.add_argument(
        "--rot",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable rotation average test-time augmentation",
    )
    parser.add_argument(
        "--relaxer_type",
        type=str,
        choices=["sequential", "batch"],
        default="sequential",
        help="Optimizer type, 'sequential' or 'batch'",
    )
    parser.add_argument(
        "--max_natoms_per_batch",
        type=int,
        default=512,
        help="Maximum number of atoms per batch for batch relaxer",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/mnt/shared-storage-user/lijiahang/checkpoints/mattersim_30m_mptrj.pt",
        help="Path to the mattersim model file",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs/mattersim_matbench/mattersim_30m_mptrj_test.log",
        help="Path to the log file",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="slice is like 0_10, to take 0, 1, ..., 9 elements. Not supported in distributed mode, as the slice will be automatically assigned based on rank."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device. Not supported in distributed mode, as the device will be automatically assigned based on local_rank.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.02,
        help="Maximum force threshold (eV/Å)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum number of optimization steps",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Whether to use distributed optimization",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.distributed:
        # torch.distributed.init_process_group(
        #     backend="nccl", 
        #     device_id=torch.device(f"cuda:{local_rank}")
        # )
        # torch.distributed.barrier()
        torch.cuda.set_device(local_rank)
        # In distributed mode, it is difficult to specify slice for each process manually. We thus let RANK to specify it.
        
        # assert args.slice is None, "Manual specification of slice is not supported in distributed mode."
        assert args.device == "cuda", "Manual specification of device is not supported in distributed mode."
        if args.slice is not None:
            warnings.warn(f"Distributed evaluation is only conducted in slice {args.slice}")
            slice_tuple = args.slice.split("_")
            slice_tuple = tuple(int(x) for x in slice_tuple)
            slice_start, slice_end = slice_tuple
            num_samples = slice_end - slice_start
            num_samples_per_proc = (num_samples + world_size - 1) // world_size
            start = slice_start + rank * num_samples_per_proc
            end = min(start + num_samples_per_proc, slice_end)

            if start >= end:
                return # corner case
            args.slice = f"{start}_{end}"
            slice_tuple = (start, end)
            
        else:
            num_samples = len(df_wbm)
            num_samples_per_proc = (num_samples + world_size - 1) // world_size
            start = rank * num_samples_per_proc
            end = start + num_samples_per_proc
            args.slice = f"{start}_{end}"   
            slice_tuple = (start, end)

        # get the file name of log_path
        log_path = Path(args.log_path)
        log_file_name = log_path.stem
        # replace "slice_start_end" with "slice_{start}_{end}"
        new_log_file_name = log_file_name.replace("slice_start_end", f"slice_{start}_{end}")
        # jiahang: it's weird that with_name cannot deal with suffix
        # automatically though this logic has been implemented in with_name.
        new_log_path = log_path.with_name(new_log_file_name + log_path.suffix)
        args.log_path = str(new_log_path)

        # get wbm log path
        wbm_log_path = new_log_path.with_name(new_log_file_name + '*wbm-IS2RE.csv.gz')
        if glob(str(wbm_log_path)):
            print(f"Found existing WBM log file: {glob(str(wbm_log_path))[0]}, Skip.")
            # torch.distributed.barrier()
            return
        
    else:
        if args.slice is not None:
            slice_tuple = args.slice.split("_")
            slice_tuple = tuple(int(x) if x else None for x in slice_tuple)
        else:
            slice_tuple = None

    logger = Logger(log_path=args.log_path)

    # Check if the model file exists
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {checkpoint_path}")

    logger("=" * 60)
    logger("mattersim model test on Matbench Discovery dataset")
    logger("=" * 60)
    logger(f"Model path: {checkpoint_path}")
    logger(f"Device: {args.device}")
    logger(f"Slice: {args.slice}")
    logger(f"Maximum force threshold: {args.fmax} eV/Å")
    logger(f"Maximum number of steps: {args.steps}")
    logger(f"Log path: {args.log_path}")
    logger("=" * 60)

    # Load dataset
    logger("\nLoading WBM initial structures...")
    df_files = DataFilesCustomized

    logger("Using default data path, loading full dataset...")
    # Prefer using wbm_initial_structures, which is specifically for initial structures
    wbm_initial_structures_path = df_files.wbm_initial_structures
    logger(f"Using WBM initial structures file: {wbm_initial_structures_path}")
    init_wbm_atoms_list = load_wbm_initial_atoms(
        str(wbm_initial_structures_path), slice_tuple=slice_tuple,
        logger=logger
    )
    logger(f"Loaded {len(init_wbm_atoms_list):,} structures")

    # Optimize structures
    logger("\nStarting structure optimization...")
    relaxed_wbm_atoms_list = relax_atoms_list(
        init_wbm_atoms_list,
        checkpoint_path=checkpoint_path,
        device=args.device,
        fmax=args.fmax,
        steps=args.steps,
        logger=logger,
        max_natoms_per_batch=args.max_natoms_per_batch,
        relaxer_type=args.relaxer_type,
        rot=args.rot
    )

    # Parse results
    logger("\nParsing results...")
    df_results = parse_relaxed_atoms_list_as_df(
        relaxed_wbm_atoms_list,
        logger=logger
    )
    logger.log_df(df_results)

    # Save results
    save_results(args, df_results, logger)

    logger.close()

    # if args.distributed:
    #     torch.distributed.barrier()


if __name__ == "__main__":
    main()