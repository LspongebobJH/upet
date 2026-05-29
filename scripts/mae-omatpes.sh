#!/usr/bin/env bash
set -ex

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

datasets=(
    MP_ALOE_data_part_00000.aselmdb
    MP_ALOE_data_part_00001.aselmdb
    MP_ALOE_data_part_00002.aselmdb
    MP_ALOE_data_part_00003.aselmdb
    MP_ALOE_data_part_00004.aselmdb
    MP_ALOE_data_part_00005.aselmdb
    MP_ALOE_data_part_00006.aselmdb
    MP_ALOE_data_part_00007.aselmdb
)

for dataset in "${datasets[@]}"; do
    dataset_name="${dataset%.*}"
    for (( i=0; i<NPROC_PER_NODE; i++ )); do
        export LOCAL_RANK=${i}
        export RANK=$(( NODE_RANK * NPROC_PER_NODE + LOCAL_RANK ))
        python benchmark/mae.py \
            --valid_data_path /mnt/shared-storage-gpfs2/lijiahang1/jielan-datasets/${dataset} \
            --ckpt_path /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet/checkpoints/pet-omatpes-l-v0.1.0.ckpt \
            --log_path /mnt/shared-storage-gpfs2/lijiahang1/logs/mpaloe-efs-mae-pet-omatpes-l-0.1.0/${dataset_name}_slice_start_end.log \
            --force_rerun \
            --distributed &

        # echo RANK:${RANK}, LOCAL_RANK:${LOCAL_RANK}, WORLD_SIZE:${WORLD_SIZE}
    done

    wait
    echo "Dataset ${dataset} DONE"
done

# bash tools/occupy.sh --nproc_per_node ${NPROC_PER_NODE}