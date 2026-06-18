# MAE Parallel Evaluation Script

Script for running distributed MAE evaluation using `torchrun`.

## Script: `mae-pet-parallel.sh`

### Features

- ✅ **Multi-GPU support** via `torchrun`
- ✅ **Automatic result aggregation** across ranks
- ✅ **Flexible mode selection** (conservative, non-conservative, or both)
- ✅ **Configurable batch size and workers**
- ✅ **Follows same distributed setup as training** (`train_ddp.sh`)

### Usage

#### Basic Usage (Single Node, 8 GPUs, Conservative Mode)
```bash
bash scripts/mae-pet-parallel.sh \
    --model_variant omat-xl
```

#### Non-Conservative Mode
```bash
bash scripts/mae-pet-parallel.sh \
    --model_variant omat-xl \
    --non_conservative
```

#### Evaluate Both Modes Simultaneously
```bash
bash scripts/mae-pet-parallel.sh \
    --model_variant omat-xl \
    --conservative \
    --non_conservative
```

#### Custom Checkpoint
```bash
bash scripts/mae-pet-parallel.sh \
    --model_variant omat-xl \
    --checkpoint /path/to/your/model.ckpt \
    --non_conservative
```

#### Adjust GPU Count and Batch Size
```bash
bash scripts/mae-pet-parallel.sh \
    --model_variant omat-xl \
    --nproc_per_node 4 \
    --batch_size 128 \
    --num_workers 16
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_variant` | Model variant name (e.g., `omat-xl`, `omat-xs`) | `omat-xl` |
| `--checkpoint` | Path to checkpoint file | `checkpoints/pet-{variant}-v1.0.0.ckpt` |
| `--conservative` | Evaluate conservative mode | `false` |
| `--non_conservative` | Evaluate non-conservative mode | `false` |
| `--nproc_per_node` | Number of GPUs per node | `8` (or `$PROC_PER_NODE`) |
| `--batch_size` | Batch size per GPU | `64` |
| `--num_workers` | DataLoader workers per GPU | `8` |

**Mode Selection:**
- If neither `--conservative` nor `--non_conservative` is set: **conservative mode** (default)
- If only `--conservative` is set: **conservative mode only**
- If only `--non_conservative` is set: **non-conservative mode only**
- If both are set: **evaluate both modes simultaneously**

### Environment Variables

The script respects the following environment variables (same as `train_ddp.sh`):

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_COUNT` | Number of nodes | `1` |
| `NODE_RANK` | Current node rank | `0` |
| `PROC_PER_NODE` | GPUs per node | `8` |
| `MASTER_ADDR` | Master node address | `127.0.0.1` |
| `MASTER_PORT` | Master node port | `29501` |
| `NCCL_SOCKET_IFNAME` | Network interface | `bond0` |
| `JOB_ID` | Job ID for rendezvous | `mae_eval` |

### Multi-Node Execution

For multi-node distributed evaluation, set the environment variables before running:

**Node 0 (Master):**
```bash
export NODE_COUNT=2
export NODE_RANK=0
export MASTER_ADDR=<node0_ip>
export MASTER_PORT=29501

bash scripts/mae-pet-parallel.sh --model_variant omat-xl --non_conservative
```

**Node 1:**
```bash
export NODE_COUNT=2
export NODE_RANK=1
export MASTER_ADDR=<node0_ip>
export MASTER_PORT=29501

bash scripts/mae-pet-parallel.sh --model_variant omat-xl --non_conservative
```

### Output

Results are automatically aggregated and saved:

```
logs/mae/pet-omat-xl-v1.0.0-non_equi_test_data-2024-06-16-non_conservative/
├── eval_rank0.log        # Per-rank logs for debugging
├── eval_rank1.log
├── eval_rank2.log
├── ...
└── eval_aggregated_efs.pkl  # Final aggregated results
```

**Rank 0 output** includes:
- Total structures evaluated across all ranks
- MAE metrics for energy, forces, and stress
- Path to aggregated results file

### Examples

#### 1. Quick Evaluation (4 GPUs, Conservative)
```bash
bash scripts/mae-pet-parallel.sh \
    --model_variant omat-xl \
    --nproc_per_node 4
```

#### 2. Full Evaluation (8 GPUs, Both Modes)
```bash
bash scripts/mae-pet-parallel.sh \
    --model_variant omat-xl \
    --conservative \
    --non_conservative \
    --batch_size 64
```

#### 3. Single GPU (No Distribution)
```bash
bash scripts/mae-pet-parallel.sh \
    --model_variant omat-xs \
    --nproc_per_node 1 \
    --non_conservative
```

#### 4. Custom Checkpoint with Large Batch
```bash
bash scripts/mae-pet-parallel.sh \
    --checkpoint /path/to/experiment/checkpoint-epoch50.ckpt \
    --non_conservative \
    --batch_size 128 \
    --num_workers 16
```

### Notes

- **Single GPU mode**: When `--nproc_per_node 1`, the script runs without `torchrun`
- **Automatic aggregation**: Results from all GPUs are automatically combined on rank 0
- **Log files**: Each rank maintains its own log for debugging
- **Port conflict**: Uses port `29501` by default (different from training's `29500`)
