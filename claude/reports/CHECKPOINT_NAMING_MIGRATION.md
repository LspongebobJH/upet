# PET Checkpoint Naming Migration

## Problem

Old PET checkpoints use singular naming for non-conservative forces:
- `non_conservative_force` (WRONG - singular)

New checkpoints and all future code should use plural naming:
- `non_conservative_forces` (CORRECT - plural)

This naming is stored in `checkpoint['hypers']['outputs']` and affects model loading.

## Current Solution

### Immediate Fix (Implemented)

Added `_normalize_model_outputs()` function in 3 files to create aliases at load time:

1. `benchmark/mae_parallel.py`
2. `benchmark/forward.py`  
3. `src/upet/_models.py`

This function checks the loaded model and adds the missing alias, ensuring both naming conventions work.

**Pros:**
- ✅ Works immediately without modifying checkpoints
- ✅ Handles both old and new checkpoints automatically
- ✅ No risk of corrupting checkpoint files

**Cons:**
- ❌ Backward compatibility code stays in codebase forever
- ❌ Doesn't fix the checkpoint itself
- ❌ Every piece of code that loads checkpoints needs the fix

### Permanent Solution (Recommended for Upstream)

The proper way to handle this in metatrain is through checkpoint versioning:

1. **Add migration function** in `metatrain/pet/checkpoints.py`:

```python
def model_update_v11_v12(checkpoint: dict) -> None:
    """
    Update a v11 checkpoint to v12.
    
    Renames non_conservative_force (singular) to non_conservative_forces (plural)
    in hypers.outputs to match the standard naming convention.
    
    :param checkpoint: The checkpoint to update.
    """
    if 'hypers' in checkpoint:
        hypers = checkpoint['hypers']
        
        # Handle object-style hypers
        if hasattr(hypers, 'outputs'):
            outputs = hypers.outputs
            if 'non_conservative_force' in outputs:
                outputs['non_conservative_forces'] = outputs.pop('non_conservative_force')
        
        # Handle dict-style hypers  
        elif isinstance(hypers, dict) and 'outputs' in hypers:
            outputs = hypers['outputs']
            if 'non_conservative_force' in outputs:
                outputs['non_conservative_forces'] = outputs.pop('non_conservative_force')
```

2. **Increment version** in `metatrain/pet/model.py`:

```python
__checkpoint_version__ = 12  # was 11
```

3. **Test the migration**:

```python
# Old checkpoint will be automatically upgraded when loaded
model = load_metatrain_model("old_checkpoint.ckpt")
# Outputs now include 'non_conservative_forces' instead of 'non_conservative_force'
```

**Pros:**
- ✅ Checkpoints are permanently fixed on first load
- ✅ Standard metatrain pattern
- ✅ Automatic migration
- ✅ Can eventually remove backward compatibility code

**Cons:**
- ❌ Requires upstream contribution to metatrain
- ❌ Users need to update metatrain version

## Recommendation

1. **Short term**: Keep the current `_normalize_model_outputs()` fix for immediate use
2. **Long term**: Contribute the checkpoint migration to metatrain upstream
3. **After metatrain update**: Remove the `_normalize_model_outputs()` workaround

## How Metatrain Checkpoint Migration Works

```
Load checkpoint
    ↓
Check version: checkpoint['model_ckpt_version'] vs model.__checkpoint_version__
    ↓
If version < current:
    ├─ Call model_update_v1_v2(checkpoint) 
    ├─ Call model_update_v2_v3(checkpoint)
    ├─ ...
    └─ Call model_update_v<N>_v<N+1>(checkpoint)
    ↓
Update checkpoint['model_ckpt_version']
    ↓
Return upgraded checkpoint
```

Each update function modifies the checkpoint dict in-place to match the next version's expectations.

## Testing

To test if a checkpoint needs migration:

```python
checkpoint = torch.load("model.ckpt", map_location='cpu', weights_only=False)

# Check hypers.outputs
if 'hypers' in checkpoint:
    if hasattr(checkpoint['hypers'], 'outputs'):
        outputs = checkpoint['hypers'].outputs
    else:
        outputs = checkpoint['hypers']['outputs']
    
    if 'non_conservative_force' in outputs:
        print("❌ OLD naming - needs migration")
    elif 'non_conservative_forces' in outputs:
        print("✅ NEW naming - OK")
```

## References

- metatrain checkpoint system: `metatrain/utils/io.py` (load_model)
- PET checkpoint migrations: `metatrain/pet/checkpoints.py`
- PET model: `metatrain/pet/model.py` (__checkpoint_version__ and upgrade_checkpoint)
- PET trainer: `metatrain/pet/trainer.py` (__checkpoint_version__ and upgrade_checkpoint)
