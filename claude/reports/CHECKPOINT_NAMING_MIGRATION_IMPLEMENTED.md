# PET Checkpoint Naming Migration - IMPLEMENTED ✅

## Problem

Old PET checkpoints use singular naming for non-conservative forces:
- `non_conservative_force` (WRONG - singular)

New checkpoints and all future code use plural naming:
- `non_conservative_forces` (CORRECT - plural)

This naming appears in two places:
1. `checkpoint['model_data']['dataset_info']['targets']` - target configuration
2. `checkpoint['model_state_dict']` - neural network weight keys

## Solution - IMPLEMENTED

### Changes Made to Metatrain

**File: `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/model.py`**
- Updated `__checkpoint_version__` from 14 to 15

**File: `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/checkpoints.py`**
- Added `model_update_v14_v15()` function that:
  1. Renames `non_conservative_force` → `non_conservative_forces` in `dataset_info.targets`
  2. Renames all model state_dict keys containing `non_conservative_force` → `non_conservative_forces`

### How It Works

When loading an old checkpoint (v14):
1. Metatrain detects version mismatch (checkpoint is v14, code expects v15)
2. Calls `model_update_v14_v15(checkpoint)` automatically
3. Migration renames targets and state_dict keys in-place
4. Checkpoint is updated to v15
5. Model loads successfully with new naming

### Testing

✅ Successfully tested with old checkpoint:
```bash
python3 -c "
from metatrain.utils.io import load_model as load_metatrain_model
model = load_metatrain_model('outputs/2026-06-11/21-13-51/model_1.ckpt')
print(list(model.dataset_info.targets.keys()))
# Output: ['energy', 'non_conservative_stress', 'non_conservative_forces']
"
```

### Usage

No user action required! Simply load old checkpoints as normal:

```python
from metatrain.utils.io import load_model

# Old checkpoint is automatically migrated on load
model = load_model("old_checkpoint.ckpt")

# Targets now use plural naming
print(model.dataset_info.targets.keys())
# ['energy', 'non_conservative_forces', 'non_conservative_stress']
```

### Migration Details

The migration function updates:

**1. Target Names (`dataset_info.targets`)**
```python
# Before: {'energy': ..., 'non_conservative_force': ..., 'non_conservative_stress': ...}
# After:  {'energy': ..., 'non_conservative_forces': ..., 'non_conservative_stress': ...}
```

**2. State Dict Keys (neural network weights)**
```python
# Before: 
#   - node_heads.non_conservative_force.0.0.weight
#   - scaler.non_conservative_force_scaler_buffer
#   - ...

# After:
#   - node_heads.non_conservative_forces.0.0.weight  
#   - scaler.non_conservative_forces_scaler_buffer
#   - ...
```

## Benefits

✅ Old checkpoints work seamlessly with new code  
✅ Automatic migration - no manual intervention  
✅ Standard metatrain checkpoint versioning system  
✅ All future code can use consistent plural naming  
✅ No backward compatibility shims needed

## Code Reference

### Migration Function

Location: `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/checkpoints.py`

```python
def model_update_v14_v15(checkpoint: dict) -> None:
    """
    Update a v14 checkpoint to v15.

    Renames non_conservative_force (singular) to non_conservative_forces (plural)
    in dataset_info.targets and model state_dict to match the standard naming convention.
    """
    # 1. Update dataset_info.targets
    if "model_data" in checkpoint and "dataset_info" in checkpoint["model_data"]:
        dataset_info = checkpoint["model_data"]["dataset_info"]
        if hasattr(dataset_info, "targets"):
            targets = dataset_info.targets
        elif isinstance(dataset_info, dict) and "targets" in dataset_info:
            targets = dataset_info["targets"]
        else:
            targets = None

        if targets and "non_conservative_force" in targets:
            targets["non_conservative_forces"] = targets.pop("non_conservative_force")

    # 2. Update model_state_dict keys
    for state_dict_key in ["model_state_dict", "best_model_state_dict"]:
        if state_dict_key in checkpoint and checkpoint[state_dict_key] is not None:
            state_dict = checkpoint[state_dict_key]
            keys_to_rename = [k for k in state_dict.keys() if "non_conservative_force" in k]
            for old_key in keys_to_rename:
                new_key = old_key.replace("non_conservative_force", "non_conservative_forces")
                state_dict[new_key] = state_dict.pop(old_key)
```

## Verification

To verify an old checkpoint will be migrated:

```bash
python3 << 'EOF'
import torch
checkpoint = torch.load("checkpoint.ckpt", map_location='cpu', weights_only=False)
print(f"Checkpoint version: {checkpoint['model_ckpt_version']}")

from metatrain.pet import PET  
print(f"Current PET version: {PET.__checkpoint_version__}")

if checkpoint['model_ckpt_version'] < PET.__checkpoint_version__:
    print("✓ Will be automatically migrated on load")
EOF
```

## Files Modified

1. `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/model.py`
   - Line 55: `__checkpoint_version__ = 15` (was 14)

2. `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/checkpoints.py`
   - Added function `model_update_v14_v15()` after line 296
