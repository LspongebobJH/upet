# export CUDA_VISIBLE_DEVICES=1,2
NPROC_PER_NODE=4
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
    esac
done

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid --nproc_per_node: ${NPROC_PER_NODE}. Expected a positive integer"
    usage
fi

#!/usr/bin/env bash
set -ex

# --------------- Paths & Conda ---------------
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH

cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/envs/pet/bin:$PATH"

# --------------- Distributed Setting ---------------

NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
export WORLD_SIZE=$(( NPROC_PER_NODE * NODE_COUNT ))

for (( i=0; i<NPROC_PER_NODE; i++ )); do
    export LOCAL_RANK=${i}
    export RANK=$(( NODE_RANK * NPROC_PER_NODE + LOCAL_RANK ))

    python tools/train.py &
done

# wait
echo "ALL DONE"
sleep inf