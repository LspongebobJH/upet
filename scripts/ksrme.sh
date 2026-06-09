#!/usr/bin/env bash
set -ex

# --------------- Arguments ---------------
NON_CONSERVATIVE=false
MODEL_VARIANT="oam-xl"
CHECKPOINT=""
IS_PLUSMINUS=false
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
        --is_plusminus)
            IS_PLUSMINUS=true
            shift
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

NPROC_PER_NODE=8
NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
export WORLD_SIZE=$(( NPROC_PER_NODE * NODE_COUNT ))

if 
MODEL_NAME="pet-${MODEL_VARIANT}-v1.0.0"
if [[ -n "$CHECKPOINT" ]]; then
    CKPT_PATH=${CHECKPOINT}
else
    CKPT_PATH=/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/checkpoints/${MODEL_NAME}.ckpt
fi
CURRENT_DATE=$(date +%Y-%m-%d)
LOG_DIR=/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/logs/ksrme/${MODEL_NAME}-${CURRENT_DATE}

args=(
    --ckpt_path ${CKPT_PATH} \
    --distributed
)
if [ "$NON_CONSERVATIVE" = true ]; then
    args+=(--non_conservative)
    LOG_DIR=${LOG_DIR}-non_conservative
fi
if [ "$IS_PLUSMINUS" = true ]; then
    args+=(--is_plusminus)
    LOG_DIR=${LOG_DIR}-is_plusminus
fi

args+=(--save_dir ${LOG_DIR})
mkdir -p ${LOG_DIR}

# --------------- Running ---------------

for (( i=0; i<NPROC_PER_NODE; i++ )); do
    export LOCAL_RANK=${i}
    export RANK=$(( NODE_RANK * NPROC_PER_NODE + LOCAL_RANK ))
    python benchmark/ksrme.py \
        "${args[@]}" &    
done