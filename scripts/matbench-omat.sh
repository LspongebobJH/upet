#!/usr/bin/env bash
set -ex

# --------------- Paths & Conda ---------------
cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/bin:$PATH"
. /mnt/shared-storage-user/lijiahang/miniconda3/etc/profile.d/conda.sh
conda activate pet

# --------------- Running arguments ---------------
log_path="./logs/matbench/pet-omat-xl-v1.0.0-0520/slice_start_end.log"
ckpt_path="/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/checkpoints/pet-omat-xl-v1.0.0.ckpt"

# --------------- Distributed Setting ---------------

NPROC_PER_NODE=8
NODE_RANK=${NODE_RANK}
export WORLD_SIZE=$(( NPROC_PER_NODE * NODE_COUNT ))

for (( i=0; i<NPROC_PER_NODE; i++ )); do
    export LOCAL_RANK=${i}
    export RANK=$(( NODE_RANK * NPROC_PER_NODE + LOCAL_RANK ))
    python benchmark/matbench.py \
        --relaxer_type batch \
        --max_natoms_per_batch 3072 \
        --distributed \
        --checkpoint_path ${ckpt_path} \
        --log_path ${log_path} &

    # echo RANK:${RANK}, LOCAL_RANK:${LOCAL_RANK}, WORLD_SIZE:${WORLD_SIZE}
done

wait
echo "ALL DONE"