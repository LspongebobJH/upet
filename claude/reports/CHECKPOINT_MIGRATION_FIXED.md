# PET Checkpoint Naming Migration - FINAL FIX ✅

## Problem

Old PET checkpoints have TWO naming issues:

1. **Target names**: `non_conservative_force` (singular) ❌ → should be `non_conservative_forces` (plural)
2. **Quantities**: `quantity='forces'` (plural) ❌ → should be `quantity='force'` (singular)

### Standard Naming Convention

```python
# Correct naming:
targets = {
    "forces": TargetInfo(quantity="force", ...),                    # Conservative
    "non_conservative_forces": TargetInfo(quantity="force", ...)    # Non-conservative
}

# WRONG naming in old checkpoints:
targets = {
    "forces": TargetInfo(quantity="forces", ...),                   # ❌ quantity wrong
    "non_conservative_force": TargetInfo(quantity="forces", ...)    # ❌ both wrong!
}
```

**Rule**: Target names use PLURAL (`forces`), but quantities use SINGULAR (`force`).

## Solution - IMPLEMENTED & TESTED ✅

### Changes Made to Metatrain

**File: `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/model.py:55`**
- Updated `__checkpoint_version__ = 15` (was 14)

**File: `/mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/checkpoints.py:299`**
- Added `model_update_v14_v15()` function that fixes:
  1. Target name: `non_conservative_force` → `non_conservative_forces`
  2. Quantity: `forces` → `force` (for ALL targets, including conservative)
  3. State dict keys: Renames all occurrences of `non_conservative_force`

### Migration Logic

```python
def model_update_v14_v15(checkpoint: dict) -> None:
    # 1. Rename target name: non_conservative_force → non_conservative_forces
    if "non_conservative_force" in targets:
        targets["non_conservative_forces"] = targets.pop("non_conservative_force")

    # 2. Fix quantity for ALL targets: forces → force
    for target_name, target_info in targets.items():
        if target_info.quantity == 'forces':  # Wrong (plural)
            target_info.quantity = 'force'     # Fix to singular

    # 3. Rename state_dict keys
    for key in state_dict.keys():
        if 'non_conservative_force' in key and 'non_conservative_forces' not in key:
            new_key = key.replace('non_conservative_force', 'non_conservative_forces')
            state_dict[new_key] = state_dict.pop(key)
```

### Why This Matters

The quantity field is used by metatensor/metatomic for unit validation:

```cpp
// In units.cpp:678
Warning: unknown dimension 'forces', only [charge energy force heat_flux ...] are supported
```

Using `quantity='forces'` (plural) causes this warning because metatensor only recognizes `'force'` (singular).

## Testing Results

✅ **Test 1: Non-conservative with wrong quantity**
```python
# Before: {'non_conservative_force': TargetInfo(quantity='forces', ...)}
# After:  {'non_conservative_forces': TargetInfo(quantity='force', ...)}
```

✅ **Test 2: Conservative with wrong quantity**
```python
# Before: {'forces': TargetInfo(quantity='forces', ...)}
# After:  {'forces': TargetInfo(quantity='force', ...)}
```

✅ **Test 3: Old checkpoint migration**
```python
model = load_model('outputs/2026-06-11/21-13-51/model_1.ckpt')
print(model.dataset_info.targets['non_conservative_forces'].quantity)
# Output: 'force' ✓ (correct, singular)
```

✅ **Test 4: No warning about unknown dimension**
```python
# Old: Warning: unknown dimension 'forces'
# New: No warning (uses correct 'force')
```

## Bug Fixes History

**Bug #1: Incomplete Renaming**
- Problem: Regex `\b` didn't match `non_conservative_force___0`
- Fix: Use `.replace()` for all occurrences

**Bug #2: Over-Renaming**  
- Problem: `.replace()` would match `forces` → `forcess`
- Fix: Use negative lookahead `r'non_conservative_force(?!s)'`

**Bug #3: Wrong Quantity (THE CRITICAL BUG)**
- Problem: Only fixed target names, didn't fix `quantity='forces'` → `quantity='force'`
- Fix: Loop through ALL targets and fix any `quantity='forces'` to `'force'`
- Impact: Affects BOTH conservative and non-conservative checkpoints

## Usage

No user action required! Old checkpoints are automatically migrated:

```python
from metatrain.utils.io import load_model

# Old checkpoint (v14) is automatically migrated to v15
model = load_model("old_checkpoint.ckpt")

# All targets now have correct naming
for name, info in model.dataset_info.targets.items():
    print(f"{name}: quantity={info.quantity}")

# Output:
#   energy: quantity=energy
#   non_conservative_forces: quantity=force  ✓
#   non_conservative_stress: quantity=pressure  ✓
```

## Files Modified

1. **[model.py:55](file:///mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/model.py#L55)**
   ```python
   __checkpoint_version__ = 15  # was 14
   ```

2. **[checkpoints.py:299](file:///mnt/shared-storage-gpfs2/lijiahang1/jobs/metatrain/src/metatrain/pet/checkpoints.py#L299)**
   ```python
   def model_update_v14_v15(checkpoint: dict):
       # 1. Rename non_conservative_force → non_conservative_forces
       # 2. Fix quantity='forces' → 'force' for ALL targets
       # 3. Update state_dict keys
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

## Summary

The migration now handles:
- ✅ Target name: `non_conservative_force` → `non_conservative_forces` (plural)
- ✅ Quantity: `forces` → `force` (singular) for ALL targets
- ✅ State dict: Rename all keys with `non_conservative_force`
- ✅ Conservative checkpoints: Fix quantity without changing target name
- ✅ Safety: Skip already-correct checkpoints
