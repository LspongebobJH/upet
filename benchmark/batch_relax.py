# -*- coding: utf-8 -*-
import sys
from typing import Dict, List, Union

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.filters import Filter
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import BFGS, FIRE
from ase.optimize.optimize import Optimizer
# from loguru import logger
from tqdm import tqdm

from metatomic_ase._symmetry import _rotate_atoms, _compute_rotational_average

import torch
import numpy as np


class DummyBatchCalculator(Calculator):
    def __init__(self):
        super().__init__()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        pass

    def get_potential_energy(self, atoms=None):
        return atoms.info["total_energy"]

    def get_forces(self, atoms=None):
        return atoms.arrays["forces"]

    def get_stress(self, atoms=None):
        return units.GPa * atoms.info["stress"]


class BatchRelaxer(object):
    """BatchRelaxer is a class for batch structural relaxation.
    It is more efficient than Relaxer when relaxing a large number of structures."""

    SUPPORTED_OPTIMIZERS = {"BFGS": BFGS, "FIRE": FIRE}
    SUPPORTED_FILTERS = {
        "EXPCELLFILTER": ExpCellFilter,
        "FRECHETCELLFILTER": FrechetCellFilter,
    }

    def __init__(
        self,
        calc,
        optimizer: Union[str, type[Optimizer]] = "FIRE",
        filter: Union[type[Filter], str, None] = None,
        fmax: float = 0.05,
        max_natoms_per_batch: int = 512,
        max_n_steps: int = 1_000_000,
        device: str = "cuda",
        rot: bool = False
    ):
        self.rot = rot
        self.calc = calc
        self.device = device
        if isinstance(optimizer, str):
            if optimizer.upper() not in self.SUPPORTED_OPTIMIZERS:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            self.optimizer = self.SUPPORTED_OPTIMIZERS[optimizer.upper()]
        elif issubclass(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        if isinstance(filter, str):
            if filter.upper() not in self.SUPPORTED_FILTERS:
                raise ValueError(f"Unsupported filter: {filter}")
            self.filter = self.SUPPORTED_FILTERS[filter.upper()]
        elif filter is None or issubclass(filter, Filter):
            self.filter = filter
        else:
            raise ValueError(f"Unsupported filter: {filter}")
        self.fmax = fmax
        self.max_natoms_per_batch = max_natoms_per_batch
        self.optimizer_instances: List[Optimizer] = []
        self.is_active_instance: List[bool] = []
        self.finished = False
        self.total_converged = 0
        self.trajectories: Dict[int, List[Atoms]] = {}
        self.max_n_steps = max_n_steps 
        self.final_atoms = {}

    def insert(self, atoms: Atoms):
        atoms.calc = DummyBatchCalculator()
        optimizer_instance = self.optimizer(
            self.filter(atoms) if self.filter else atoms
        )
        optimizer_instance.fmax = self.fmax
        optimizer_instance.nsteps = 0
        self.optimizer_instances.append(optimizer_instance)
        self.is_active_instance.append(True)

    def compute_efs(self, atoms_list):
        
        if self.rot:
            rotated_atoms_list = []
            for atoms in atoms_list:
                rotated_atoms_list.extend(_rotate_atoms(atoms, self.calc.quadrature_rotations))
            
            results: Dict[str, np.ndarray] = {}
            
            try:
                results = self.calc.base_calculator.compute_energy(
                    rotated_atoms_list,
                    True,
                )
                results['energy'] = np.array(results['energy'])
            except torch.cuda.OutOfMemoryError as e:
                raise RuntimeError(
                    "Out of memory error encountered during rotational averaging. "
                    "Please reduce the batch size or use lower rotational "
                    "averaging parameters. This can be done by setting the "
                    "`max_natoms_per_batch` and `l_max` parameters while initializing the "
                    "calculator."
                ) from e

            efs = {
                'energy': [],
                'forces': [],
                'stress': [],
            }

            for idx in range(len(atoms_list)):
                idx_st, idx_end = \
                    idx * len(self.calc.quadrature_rotations), (idx + 1) * len(self.calc.quadrature_rotations)
                curr_struct = {
                    'energy': results['energy'][idx_st:idx_end],
                    'forces': results['forces'][idx_st:idx_end],
                    'stress': results['stress'][idx_st:idx_end],
                }
                _results = \
                    _compute_rotational_average(
                        curr_struct,
                        self.calc.quadrature_rotations,
                        self.calc.quadrature_weights,
                        self.calc.store_rotational_std,
                    )
                
                efs['energy'].append(_results['energy'])
                efs['forces'].append(_results['forces'])
                efs['stress'].append(_results['stress'])

            efs['energy'] = np.array(efs['energy'])
            return efs
                
        else:
            try:
                return self.calc.compute_energy(
                    atoms_list,
                    compute_forces_and_stresses=True
                )
            except torch.cuda.OutOfMemoryError as e:
                raise RuntimeError(
                    "Out of memory error encountered during compute_energy. "
                    "Please reduce the batch size. "
                    "This can be done by setting the "
                    "`max_natoms_per_batch`"                    
                ) from e


    def step_batch(self):
        atoms_list = []
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                atoms_list.append(opt.atoms.atoms)

        # Note: we use a batch size of len(atoms_list)
        # because we only want to run one batch at a time
        results = \
            self.compute_efs(
                atoms_list
            )
        energy_batch, forces_batch, stress_batch = \
            results['energy'], results['forces'], results['stress']

        counter = 0
        self.finished = True
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                # Set the properties so the dummy calculator can
                # return them within the optimizer step
                opt.atoms.info["total_energy"] = energy_batch[counter]
                opt.atoms.arrays["forces"] = forces_batch[counter]
                opt.atoms.info["stress"] = stress_batch[counter]
                try:
                    self.trajectories[opt.atoms.info["structure_index"]].append(
                        opt.atoms.copy()
                    )
                except KeyError:
                    self.trajectories[opt.atoms.info["structure_index"]] = [
                        opt.atoms.copy()
                    ]

                opt.step()
                opt.nsteps += 1
                converged = opt.converged(opt.optimizable.get_gradient())
                over_max_steps = opt.nsteps >= self.max_n_steps
                if converged or over_max_steps:
                    self.is_active_instance[idx] = False
                    self.total_converged += 1
                    if self.total_converged % 10 == 0:
                        # logger.info(f"Relaxed {self.total_converged} structures.")
                        print(f"Relaxed {self.total_converged} structures.")
                    self.final_atoms[opt.atoms.info["structure_index"]] = opt.atoms.copy()
                    if converged and over_max_steps:
                        raise RuntimeError("This should not happen when both converged and over_max_steps.")
                    if converged:
                        self.final_atoms[opt.atoms.info["structure_index"]].info["converged"] = True
                    elif over_max_steps:
                        self.final_atoms[opt.atoms.info["structure_index"]].info["converged"] = False
                else:
                    self.finished = False
                counter += 1

        # remove inactive instances
        self.optimizer_instances = [
            opt
            for opt, active in zip(self.optimizer_instances, self.is_active_instance)
            if active
        ]
        self.is_active_instance = [True] * len(self.optimizer_instances)
        torch.cuda.empty_cache()

    def relax(
        self,
        atoms_list: List[Atoms],
    ) -> Dict[int, List[Atoms]]:
        self.trajectories = {}
        self.tqdmcounter = tqdm(total=len(atoms_list), file=sys.stdout)
        pointer = 0
        atoms_list_ = []
        for i in range(len(atoms_list)):
            atoms_list_.append(atoms_list[i].copy())
            atoms_list_[i].info["structure_index"] = i

        while (
            pointer < len(atoms_list) or not self.finished
        ):  # While there are unfinished instances or atoms left to insert
            while pointer < len(atoms_list) and (
                sum([len(opt.atoms) for opt in self.optimizer_instances])
                + len(atoms_list[pointer])
                <= self.max_natoms_per_batch
            ):
                # While there are enough n_atoms slots in the
                # batch and we have not reached the end of the list.
                self.insert(
                    atoms_list_[pointer]
                )  # Insert new structure to fire instances
                self.tqdmcounter.update(1)
                pointer += 1
            self.step_batch()
        self.tqdmcounter.close()

        assert len(self.final_atoms) == len(atoms_list) == len(self.trajectories)
        self.final_atoms = dict(sorted(self.final_atoms.items()))

        return self.trajectories
