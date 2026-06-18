#!/usr/bin/env bash
set -ex

# --------------- Arguments ---------------
NON_CONSERVATIVE=false
CONSERVATIVE=false
MODEL_VARIANT="omat-xl"
CHECKPOINT=""
NPROC_PER_NODE="${PROC_PER_NODE:-8}"
BATCH_SIZE=64
NUM_WORKERS=8
MISC=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --non_conservative)
            NON_CONSERVATIVE=true
            shift
            ;;
        --conservative)
            CONSERVATIVE=true
            shift
            ;;
        --model_variant)
            MODEL_VARIANT="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --misc)
            MISC="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# --------------- Paths & Environment ---------------
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH

cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/envs/pet/bin:$PATH"

# --------------- Distributed Environment Variables ---------------
NNODES="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# --------------- Dataset & Model Configuration ---------------
# DATASET=/mnt/shared-storage-gpfs2/ailab-omnimat-shared/lijiahang/datasets/omat24/*/*
# DATASET_NAME="omat_100m"
# DATASET=/mnt/shared-storage-user/lijiahang/datasets/non_equi_test_data/data.aselmdb
# DATASET_NAME="non_equi_test_data"
# DATASET=/mnt/shared-storage-user/lijiahang/jielan-datasets/omat_compressed/**/*.aselmdb
# DATASET_NAME="omat_compressed"
DATASET=/mnt/shared-storage-gpfs2/ailab-omnimat-shared/lijiahang/datasets/omat24-1m-demo/omat24_1M_251210/*/*/*.aselmdb
DATASET_NAME="omat24-1m-demo"

MODEL_NAME="pet-${MODEL_VARIANT}-v1.0.0"

if [[ -n "$CHECKPOINT" ]]; then
    CKPT_PATH=${CHECKPOINT}
else
    CKPT_PATH=/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/checkpoints/${MODEL_NAME}.ckpt
fi

# --------------- Log Directory ---------------
CURRENT_DATE=$(date +%Y-%m-%d)
LOG_DIR=/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/logs/mae/${MODEL_NAME}-${DATASET_NAME}-${CURRENT_DATE}

# Add mode suffix to log directory
if [ "$NON_CONSERVATIVE" = true ] && [ "$CONSERVATIVE" = true ]; then
    LOG_DIR=${LOG_DIR}-both
elif [ "$NON_CONSERVATIVE" = true ]; then
    LOG_DIR=${LOG_DIR}-non_conservative
elif [ "$CONSERVATIVE" = true ]; then
    LOG_DIR=${LOG_DIR}-conservative
else
    LOG_DIR=${LOG_DIR}-conservative  # default
fi

if [ -n "$MISC" ]; then
    LOG_DIR=${LOG_DIR}-${MISC}
fi

mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/eval_rank.log

# --------------- Build Arguments ---------------
common_args=(
    benchmark/mae_parallel.py
    --config ${CONFIG}
    --valid_data_path "${DATASET}"
    --ckpt_path ${CKPT_PATH}
    --log_path ${LOG_PATH}
    --batch_size ${BATCH_SIZE}
    --num_workers ${NUM_WORKERS}
    --distributed
    --force_rerun
)

# Add mode flags
if [ "$NON_CONSERVATIVE" = true ]; then
    common_args+=(--non_conservative)
fi
if [ "$CONSERVATIVE" = true ]; then
    common_args+=(--conservative)
fi

# --------------- Run Evaluation ---------------
if [ "${NPROC_PER_NODE}" = "1" ]; then
    # Single GPU - no torchrun needed
    python "${common_args[@]}"
else
    # Multi-GPU - use torchrun
    torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank="$NODE_RANK" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --rdzv_id="${JOB_ID:-mae_eval}" \
        "${common_args[@]}"
fi

echo "Evaluation complete. Results saved to: ${LOG_DIR}/eval_aggregated_efs.pkl"

bash tools/train.sh --nproc_per_node 8