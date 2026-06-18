# PET Checkpoint Naming Migration - COMPLETE & TESTED ✅

## Problem

Old PET checkpoints use singular naming for non-conservative forces:
- `non_conservative_force` (WRONG - singular)

New checkpoints and all future code use plural naming:
- `non_conservative_forces` (CORRECT - plural)

This naming appears in two places:
1. `checkpoint['model_data']['dataset_info']['targets']` - target configuration
2. `checkpoint['model_state_dict']` - neural network weight keys (appears MULTIPLE times in some keys!)

**Important:** Some state_dict keys contain the target name multiple times, e.g.:
```
node_last_layers.non_conservative_forces.0.non_conservative_force___0.weight
                ^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^
                (1st: already plural)       (2nd: needs renaming)
```

## Solution - IMPLEMENTED & TESTED ✅

### Changes Made to Metatrain

**File: `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/model.py`**
- Line 55: Updated `__checkpoint_version__` from 14 to 15

**File: `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/checkpoints.py`**
- Added `model_update_v14_v15()` function (after line 296) that:
  1. Renames `non_conservative_force` → `non_conservative_forces` in `dataset_info.targets`
  2. Renames **ONLY singular occurrences** of `non_conservative_force` in state_dict keys

### Bug Fixes Applied

**Bug #1: Incomplete Renaming**
- **Problem:** Initial regex with `\b` word boundaries didn't match `non_conservative_force___0`
- **Fix:** Use simple `.replace()` to handle all occurrences

**Bug #2: Over-Renaming (The Critical Bug)**
- **Problem:** `.replace("non_conservative_force", "non_conservative_forces")` would match keys already containing `forces`:
  ```python
  # BUG: "non_conservative_forces" contains "non_conservative_force"
  "non_conservative_forces.weight".replace("non_conservative_force", "non_conservative_forces")
  # Result: "non_conservative_forcess.weight"  ❌ Double 's'!
  ```
- **Fix:** Use regex negative lookahead `r'non_conservative_force(?!s)'` to match ONLY when NOT followed by 's':
  ```python
  re.sub(r'non_conservative_force(?!s)', 'non_conservative_forces', key)
  ```

### Final Implementation

```python
def model_update_v14_v15(checkpoint: dict) -> None:
    """Update a v14 checkpoint to v15."""
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
    import re
    for state_dict_key in ["model_state_dict", "best_model_state_dict"]:
        if state_dict_key in checkpoint and checkpoint[state_dict_key] is not None:
            state_dict = checkpoint[state_dict_key]

            # Find keys with singular "non_conservative_force" (not followed by 's')
            keys_to_rename = []
            for k in state_dict.keys():
                if re.search(r'non_conservative_force(?!s)', k):
                    keys_to_rename.append(k)

            for old_key in keys_to_rename:
                # Replace singular occurrences only
                new_key = re.sub(
                    r'non_conservative_force(?!s)',
                    'non_conservative_forces',
                    old_key
                )
                state_dict[new_key] = state_dict.pop(old_key)
```

### How It Works

The regex pattern `r'non_conservative_force(?!s)'` uses a **negative lookahead**:
- Matches: `non_conservative_force.` ✓
- Matches: `non_conservative_force_` ✓  
- Matches: `non_conservative_force___0` ✓
- **Skips**: `non_conservative_forces` ✗ (already plural)
- **Skips**: `non_conservative_stress` ✗ (different target)

This ensures:
1. ✅ Singular forms are renamed to plural
2. ✅ Already-plural forms are left unchanged
3. ✅ Mixed keys (both plural and singular) only rename singular parts
4. ✅ Other targets (stress) are not affected

### Testing Results

✅ **Old checkpoint migration:**
```python
model = load_model('outputs/2026-06-11/21-13-51/model_1.ckpt')
print(list(model.dataset_info.targets.keys()))
# ['energy', 'non_conservative_stress', 'non_conservative_forces'] ✓
```

✅ **State dict verification:**
```python
state_dict = model.state_dict()
keys_with_nc = [k for k in state_dict.keys() if 'non_conservative' in k]
# All 30 keys use correct naming:
#   - node_heads.non_conservative_forces.0.0.weight ✓
#   - scaler.non_conservative_forces_scaler_buffer ✓
#   - node_last_layers.non_conservative_forces.0.non_conservative_forces___0.weight ✓
```

✅ **Conservative checkpoint safety:**
- Migration only acts when `non_conservative_force` exists
- Conservative checkpoints (with only `energy`, `forces`, `stress`) unchanged

✅ **Edge cases tested:**
- Keys with multiple occurrences: ✓
- Keys already using plural: ✓ (correctly skipped)
- Keys with underscores/dots: ✓
- Stress targets (stay singular): ✓

## Usage

No user action required! Simply load old checkpoints:

```python
from metatrain.utils.io import load_model

# Old checkpoint automatically migrated on load
model = load_model("old_checkpoint_v14.ckpt")

# Targets use plural naming
print(model.dataset_info.targets.keys())
# dict_keys(['energy', 'non_conservative_forces', 'non_conservative_stress'])
```

## Migration Details

**Target Names:**
```
Before: non_conservative_force
After:  non_conservative_forces
```

**State Dict Keys:**
```
Before: node_heads.non_conservative_force.0.0.weight
After:  node_heads.non_conservative_forces.0.0.weight

Before: scaler.non_conservative_force_scaler_buffer  
After:  scaler.non_conservative_forces_scaler_buffer

Before: node_last_layers.non_conservative_forces.0.non_conservative_force___0.weight
After:  node_last_layers.non_conservative_forces.0.non_conservative_forces___0.weight
        (only 2nd occurrence renamed - 1st already plural)
```

## Benefits

✅ Old checkpoints work seamlessly  
✅ Automatic migration - zero manual intervention  
✅ Standard metatrain versioning  
✅ Consistent plural naming across all code  
✅ No backward compatibility shims needed  
✅ Safe for both conservative and non-conservative checkpoints

## Files Modified

1. `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/model.py:55`
   - `__checkpoint_version__ = 15` (was 14)

2. `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/checkpoints.py:299`
   - Added `model_update_v14_v15()` with negative lookahead pattern
