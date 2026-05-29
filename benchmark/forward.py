from metatrain.utils.io import load_model as load_metatrain_model
from metatrain.pet import PET
from metatomic.torch import System, ModelOutput
from metatomic_ase._calculator import _ase_to_torch_data
from metatomic_ase._neighbors import AllNeighborsCalculator
from upet.calculator import UPETCalculator

import torch

import sys
sys.path.append("/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet")

from tools.utils import load_eval_data

model_path = "/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/checkpoints/pet-omat-xs-v1.0.0.ckpt"
calc = UPETCalculator(
        # model="pet-omat-xs", 
        checkpoint_path=model_path,
        version="1.0.0", 
        device='cuda'
    )
_model = calc.calculator._model
outputs = {'energy': ModelOutput(quantity="energy", unit="eV")},

loaded_model: PET = load_metatrain_model(model_path)
loaded_model.to('cuda')
valid_data = load_eval_data("/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/test_data/data.aselmdb")

capabilities = _model.capabilities()
atoms_list = [valid_data[0]]
systems = []
for atoms in atoms_list:
    types, positions, cell, pbc = _ase_to_torch_data(
        atoms=atoms, dtype=torch.float32, device="cuda"
    )
    system = System(types, positions, cell, pbc)
    systems.append(system)
    # Get the additional inputs requested by the model

# Compute the neighbors lists requested by the model
_nl_calculators = AllNeighborsCalculator(
    requested_options=_model.requested_neighbor_lists(),
)

input_systems = _nl_calculators.compute(systems=systems)

loaded_model(
    input_systems, 
    outputs = {"energy": ModelOutput(quantity="", unit="meV")}
)
pass
