#!/usr/bin/env bash
set -ex

SLICE_IDX=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)     usage ;;
        --slice_idx)   SLICE_IDX="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# --------------- Paths & Conda ---------------
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH

cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/envs/pet/bin:$PATH"

# --------------- Running arguments ---------------
slice=("0_52" "52_103")

# --------------- Distributed Setting ---------------

NPROC_PER_NODE=8
NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
export WORLD_SIZE=$(( NPROC_PER_NODE * NODE_COUNT ))

# --------------- Running ---------------

for (( i=0; i<NPROC_PER_NODE; i++ )); do
    export LOCAL_RANK=${i}
    export RANK=$(( NODE_RANK * NPROC_PER_NODE + LOCAL_RANK ))
    python scripts/ksrme-pet/test_pet_kappa.py \
        --non_conservative \
        --dtype float32 \
        --data_slice ${slice[${SLICE_IDX}]} &

    # echo RANK:${RANK}, LOCAL_RANK:${LOCAL_RANK}, WORLD_SIZE:${WORLD_SIZE}
done

wait
echo "ALL DONE"

bash tools/occupy.sh --nproc_per_node ${NPROC_PER_NODE}