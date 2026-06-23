#!/usr/bin/env bash
set -ex

CONFIG_PATH=""
RESTART=""
RESTART_LR=""
WARMUP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --restart)
            RESTART="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --restart_lr)
            RESTART_LR="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/envs/pet/bin:$PATH"
export OUTPUTS_DIR="/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/outputs/$(date +%Y-%m-%d/%H-%M-%S)"

cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
# BRANCH=jiahang
# WORKTREE=/mnt/shared-storage-gpfs2/lijiahang1/tmp/job_$(date +%s)
# git worktree add --detach $WORKTREE $BRANCH
# cd $WORKTREE

NNODES="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${PROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"

common_args=(
    main.py
    train
    "${CONFIG_PATH:-configs/pet-omat-xl-v1.0.0-lr-1e-3.yaml}"
    -r architecture.training.distributed=True
    -r architecture.training.distributed_port=${MASTER_PORT}
)

if [ -n "$RESTART" ]; then
    common_args+=(--restart "$RESTART")
    if [ -n "$RESTART_LR" ]; then
        common_args+=(-r architecture.training.restart_lr="$RESTART_LR")
    fi
fi

if [ -n "$WARMUP" ]; then
    common_args+=(-r architecture.training.warmup_fraction="$WARMUP")
fi

if [ "${NPROC_PER_NODE}" = "1" ]; then
    python "${common_args[@]}" &
else
    torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank="$NODE_RANK" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --rdzv_id="${JOB_ID:-mattersim_work}" \
        "${common_args[@]}" &
fi

PID=$!
echo "Started job PID=$PID in $WORKTREE"

wait $PID
echo "Job PID=$PID completed"