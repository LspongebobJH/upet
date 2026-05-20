#!/usr/bin/env bash
set -ex

# --------------- Paths & Conda ---------------
cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/bin:$PATH"
. /mnt/shared-storage-user/lijiahang/miniconda3/etc/profile.d/conda.sh
conda activate pet

# --------------- Running arguments ---------------
save_dir="./logs/ksrme/pet-oam-xl-v1.0.0"
ckpt_path="./checkpoints/pet-oam-xl-v1.0.0.ckpt"

# --------------- Distributed Setting ---------------

NPROC_PER_NODE=2
NODE_RANK=${NODE_RANK}
export WORLD_SIZE=$(( NPROC_PER_NODE * NODE_COUNT ))

for (( i=0; i<NPROC_PER_NODE; i++ )); do
    export LOCAL_RANK=${i}
    export RANK=$(( NODE_RANK * NPROC_PER_NODE + LOCAL_RANK ))
    python benchmark/ksrme.py \
        --ckpt_path ${ckpt_path} \
        --save_dir ${save_dir} \
        --distributed

    # echo RANK:${RANK}, LOCAL_RANK:${LOCAL_RANK}, WORLD_SIZE:${WORLD_SIZE}
done

wait
echo "ALL DONE"