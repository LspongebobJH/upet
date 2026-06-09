#!/usr/bin/env bash
set -ex

# --------------- Arguments ---------------
NON_CONSERVATIVE=false
MODEL_VARIANT="oam-xl"
CHECKPOINT=""
NPROC_PER_NODE=8
while [[ $# -gt 0 ]]; do
    case "$1" in
        --non_conservative)
            NON_CONSERVATIVE=true
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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# --------------- Paths & Conda ---------------
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH

cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/envs/pet/bin:$PATH"

# --------------- Distributed Setting ---------------


NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
export WORLD_SIZE=$(( NPROC_PER_NODE * NODE_COUNT ))

DATASET=/mnt/shared-storage-user/lijiahang/datasets/non_equi_test_data/data.aselmdb
DATASET_NAME="non_equi_test_data"
MODEL_NAME="pet-${MODEL_VARIANT}-v1.0.0"
if [[ -n "$CHECKPOINT" ]]; then
    CKPT_PATH=${CHECKPOINT}
else
    CKPT_PATH=/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/checkpoints/${MODEL_NAME}.ckpt
fi
CURRENT_DATE=$(date +%Y-%m-%d)
LOG_DIR=/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/logs/mae/${MODEL_NAME}-${DATASET_NAME}-${CURRENT_DATE}

args=(
    --valid_data_path ${DATASET} \
    --ckpt_path ${CKPT_PATH} \
    --force_rerun \
    --distributed
)
if [ "$NON_CONSERVATIVE" = true ]; then
    args+=(--non_conservative)
    LOG_DIR=${LOG_DIR}-non_conservative
fi

mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/slice_start_end.log
args+=(--log_path ${LOG_PATH})

for (( i=0; i<NPROC_PER_NODE; i++ )); do
    export LOCAL_RANK=${i}
    export RANK=$(( NODE_RANK * NPROC_PER_NODE + LOCAL_RANK ))

    python benchmark/mae.py \
        "${args[@]}" &
done