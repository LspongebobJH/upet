# MAE Parallel Evaluation

Batched version of `mae.py` that uses the ASE DB dataloader for efficient parallel evaluation.

## Key Differences from `mae.py`

### 1. **Batch Processing**
- Uses `AseDBDatasetCustomized` for memory-efficient loading
- Processes multiple structures simultaneously via PyTorch `DataLoader`
- Configurable batch size for GPU memory optimization

### 2. **Conservative vs Non-Conservative Modes**

#### Conservative Mode (default)
- Forces computed as gradients of energy w.r.t. positions: `-∂E/∂r`
- Stress computed as gradient of energy w.r.t. strain: `∂E/∂ε * (1/volume)`
- Matches standard DFT convention
- **Advantage**: Thermodynamically consistent (forces are exact derivatives of energy)

#### Non-Conservative Mode (`--non_conservative`)
- Forces and stress predicted directly by separate model heads
- Not derived from energy gradients
- May have better accuracy for certain models (e.g., PET models trained with non-conservative targets)
- **Advantage**: Can be more accurate when trained specifically for this regime

#### Evaluate Both Simultaneously
- **Key Feature**: Specify both `--conservative` and `--non_conservative` flags together
- Same computational cost as single mode (model computes all outputs simultaneously)
- Enables direct comparison on identical structures
- Useful for:
  - Understanding trade-offs between thermodynamic consistency vs. accuracy
  - Model debugging and analysis
  - Choosing the best mode for your application

### 3. **Units**
All outputs match original `mae.py`:
- Energy: eV/atom
- Forces: eV/Å
- Stress: GPa

### 4. **Distributed Support**
- Uses PyTorch DDP via `DistributedSampler`
- Each rank processes a shard of data
- **Automatic aggregation**: Results from all ranks are automatically gathered and aggregated
- Rank 0 saves the final aggregated results and computes metrics
- **No need for separate analysis step** - MAE metrics printed at the end of distributed run
- Launch with `torchrun` or `torch.distributed.launch`

## Usage

### Single GPU (Conservative Mode - Default)
```bash
python benchmark/mae_parallel.py \
    --valid_data_path /path/to/data.db \
    --ckpt_path /path/to/model.ckpt \
    --batch_size 32 \
    --num_workers 8
```

### Non-Conservative Mode Only
```bash
python benchmark/mae_parallel.py \
    --valid_data_path /path/to/data.db \
    --ckpt_path /path/to/model.ckpt \
    --batch_size 32 \
    --non_conservative
```

### Evaluate Both Modes Simultaneously
Simply specify both flags:
```bash
python benchmark/mae_parallel.py \
    --valid_data_path /path/to/data.db \
    --ckpt_path /path/to/model.ckpt \
    --batch_size 32 \
    --conservative \
    --non_conservative
```

This will evaluate both conservative and non-conservative forces/stress in a single pass.

### Multi-GPU (Distributed)
```bash
torchrun --nproc_per_node=4 benchmark/mae_parallel.py \
    --valid_data_path /path/to/data.db \
    --ckpt_path /path/to/model.ckpt \
    --batch_size 32 \
    --distributed \
    --log_path ./logs/eval_rank.log
```

**Note:** Results are automatically aggregated across all ranks. Rank 0 will print the final MAE metrics and save aggregated results to `logs/eval_aggregated_efs.pkl`. Each rank maintains its own log file for debugging (`eval_rank0.log`, `eval_rank1.log`, etc.).

### With Data Slicing
```bash
python benchmark/mae_parallel.py \
    --valid_data_path /path/to/data.db \
    --ckpt_path /path/to/model.ckpt \
    --batch_size 32 \
    --slice 0_1000
```

## Arguments

- `--valid_data_path`: Path to ASE DB file (required)
- `--ckpt_path`: Path to model checkpoint (required)
- `--batch_size`: Batch size for evaluation (default: 16)
- `--num_workers`: Number of dataloader workers (default: 4)
- `--non_conservative`: Evaluate non-conservative forces/stress
- `--conservative`: Evaluate conservative forces/stress
- `--distributed`: Enable distributed evaluation
- `--slice`: Evaluate subset of data (format: `start_end`, e.g., `0_1000`)
- `--force_rerun`: Force rerun even if results exist
- `--log_path`: Path to log file
- `--seed`: Random seed (default: 42)
- `--device`: Device (cuda or cpu, default: cuda)

**Mode Selection:**
- If neither `--conservative` nor `--non_conservative` is specified: conservative mode (default)
- If only `--conservative` is specified: conservative mode only
- If only `--non_conservative` is specified: non-conservative mode only  
- If both `--conservative` and `--non_conservative` are specified: evaluate both modes simultaneously

## Output

Results are saved to a pickle file:

### Non-Distributed Mode
```
logs/benchmark_efs/slice_0_1000_efs.pkl
```

### Distributed Mode
```
logs/benchmark_efs/aggregated_efs.pkl
```

**Automatic aggregation**: In distributed mode, results from all ranks are gathered and combined automatically. No need to manually merge results or run `ana_mae_parallel.py` separately.

### Single Mode Evaluation
When evaluating a single mode (conservative or non-conservative), the structure is:
```python
{
    'num_structures': int,
    'conservative': {  # or 'non_conservative'
        'gt_e_list': np.array,      # Ground truth energies (eV/atom)
        'pred_e_list': np.array,    # Predicted energies (eV/atom)
        'gt_f_list': np.array,      # Ground truth forces (eV/Å)
        'pred_f_list': np.array,    # Predicted forces (eV/Å)
        'gt_s_list': np.array,      # Ground truth stress (GPa)
        'pred_s_list': np.array,    # Predicted stress (GPa)
    }
}
```

### Both Modes Evaluation (`--eval_both`)
When evaluating both modes simultaneously:
```python
{
    'num_structures': int,
    'conservative': {
        'gt_e_list': ..., 'pred_e_list': ...,
        'gt_f_list': ..., 'pred_f_list': ...,
        'gt_s_list': ..., 'pred_s_list': ...,
    },
    'non_conservative': {
        'gt_e_list': ..., 'pred_e_list': ...,
        'gt_f_list': ..., 'pred_f_list': ...,
        'gt_s_list': ..., 'pred_s_list': ...,
    }
}
```

This allows direct comparison between conservative and non-conservative predictions on the same structures.

## Analyzing Results

For **non-distributed** runs, use the provided `ana_mae_parallel.py` script:

```bash
python benchmark/ana_mae_parallel.py logs/benchmark_efs/results_efs.pkl
```

For **distributed** runs, MAE metrics are automatically computed and printed at the end. The aggregated results are also saved and can be analyzed with the same script if needed.

## Performance Tips

1. **Batch Size**: Larger batches utilize GPU better but use more memory. Start with 16-32.
2. **Num Workers**: Set to 4-8 for good I/O throughput without excessive overhead.
3. **Distributed**: For large datasets, use multi-GPU to parallelize across data shards.
4. **Memory**: If OOM, reduce `--batch_size` or use `--slice` to evaluate in chunks.

## Alignment with UPETCalculator

The implementation matches `UPETCalculator` behavior:
- Conservative mode uses energy gradients (same as `MetatomicCalculator` with `non_conservative=False`)
- Non-conservative mode uses direct force/stress predictions (same as `non_conservative=True`)
- Units and sign conventions match ASE/metatrain standards
