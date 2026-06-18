# Backward Compatibility Fix for PET Checkpoints

## Problem

Old checkpoints (e.g., `/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/outputs/2026-06-11/21-13-51/model_1.ckpt`) used a different naming convention:
- `non_conservative_force` (singular)

New checkpoints (e.g., `/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/outputs/2026-06-15/21-47-43/`) use:
- `non_conservative_forces` (plural)

When trying to evaluate an old checkpoint with the current code, it would fail with:
```
ValueError: Not all targets are within the model's supported outputs
```

This occurred because the evaluation script requested `non_conservative_forces`, but the old checkpoint only registered `non_conservative_force`.

## Solution

Added a `_normalize_model_outputs()` function that creates aliases for both naming conventions, ensuring old and new checkpoints work with all code.

## Files Modified

### 1. `benchmark/mae_parallel.py`
- Added `_normalize_model_outputs()` function (lines 45-71)
- Applied normalization after loading model (line 496)

### 2. `benchmark/forward.py`
- Added `_normalize_model_outputs()` function (lines 16-42)
- Applied normalization after loading model (line 55)

### 3. `src/upet/_models.py`
- Added `_normalize_model_outputs()` function (lines 31-57)
- Applied normalization after loading model in `_get_upet_exported_atomistic_model()` (line 249)

## How It Works

The `_normalize_model_outputs()` function:

1. Checks if the model has `hypers.outputs` attribute
2. If `non_conservative_force` exists but `non_conservative_forces` doesn't:
   - Creates alias: `outputs['non_conservative_forces'] = outputs['non_conservative_force']`
3. If `non_conservative_forces` exists but `non_conservative_force` doesn't:
   - Creates alias: `outputs['non_conservative_force'] = outputs['non_conservative_forces']`

This ensures both naming conventions are always available, regardless of which one the checkpoint was trained with.

## Testing

Run the evaluation with the old checkpoint:
```bash
NCCL_SOCKET_IFNAME=lo bash scripts/mae-pet-parallel.sh \
  --non_conservative \
  --model_variant omat-xs \
  --nproc_per_node 4 \
  --checkpoint /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/outputs/2026-06-11/21-13-51/model_1.ckpt
```

The checkpoint should now load successfully with the message:
```
INFO: Added alias non_conservative_forces -> non_conservative_force for backward compatibility
```

## Future Considerations

- All PET-related code that loads checkpoints now automatically handles both naming conventions
- No need to manually patch individual checkpoints
- New code can use either convention and it will work with all checkpoints
