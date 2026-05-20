# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
from tqdm import tqdm

from upet.calculator import UPETCalculator
import torch_sim as ts
from metatomic_torchsim import MetatomicModel
from torch_sim.autobatching import BinningAutoBatcher
from torch_sim.state import SimState
from torch_sim.models.interface import ModelInterface
from collections.abc import Callable
from typing import Any

local_rank = int(os.environ.get("LOCAL_RANK", 0))

def _chunked_apply[T: SimState](
    fn: Callable[..., T],
    states: SimState,
    model: ModelInterface,
    init_kwargs: Any,
    **batcher_kwargs: Any,
) -> T:
    """Apply a function to a state in chunks.

    This prevents us from running out of memory when applying a function to a large
    number of states.

    Args:
        fn (Callable): The state function to apply
        states (SimState): The states to apply the function to
        model (ModelInterface): The model to use for the autobatcher
        init_kwargs (Any): Unpacked into state init function.
        **batcher_kwargs: Additional keyword arguments for the autobatcher

    Returns:
        A state with the function applied
    """
    autobatcher = BinningAutoBatcher(model=model, **batcher_kwargs)
    autobatcher.load_states(states)

    initialized_states = [
        fn(model=model, state=system, **init_kwargs) \
        for system, _indices in \
        tqdm(autobatcher, total=len(autobatcher.batched_states), desc="_chunked_apply")
    ]

    ordered_states = autobatcher.restore_original_order(initialized_states)
    return ts.concatenate_states(ordered_states)

ts.runners._chunked_apply = _chunked_apply

def relax(args, atoms_list, logger):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger("Loading TorchSim model...")
    calc = UPETCalculator(
        # model="pet-oam-xl", 
        checkpoint_path=args.ckpt_path,
        version="1.0.0", 
        device=args.device
    )
    model = calc.calculator.model()
    model = MetatomicModel(
        model,
        device=args.device,
        compute_forces=True,
        compute_stress=True,
    )
    logger(f"Model loaded... Device: {model.device}")

    logger("Starting batched relaxation...")
    relaxed_state = ts.optimize(
        system=atoms_list,
        model=model,
        optimizer=ts.Optimizer.fire,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=args.fmax),
        max_steps=args.steps,
        autobatcher=True,
        pbar=(not args.distributed) or local_rank == 0,
        init_kwargs=dict(cell_filter=ts.CellFilter.frechet),
    )

    relaxed_atoms_list = relaxed_state.to_atoms()
    if not isinstance(relaxed_atoms_list, list):
        relaxed_atoms_list = [relaxed_atoms_list]

    for atoms, _atoms in zip(relaxed_atoms_list, atoms_list):
        atoms.calc = calc
        atoms.info = _atoms.info.copy()

    logger(f"Relaxed {len(relaxed_atoms_list)} structures")
    return relaxed_atoms_list