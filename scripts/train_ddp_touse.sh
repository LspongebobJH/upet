#!/usr/bin/env bash
set -ex

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/envs/pet/bin:$PATH"

cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
BRANCH=main

WORKTREE=/tmp/job_$(date +%s)
git worktree add --detach $WORKTREE $BRANCH
cd $WORKTREE

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
    configs/pet-omat-xl-v1.0.0-lr-1e-3.yaml
    -r architecture.training.distributed=True
    -r architecture.training.distributed_port=${MASTER_PORT}
)

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