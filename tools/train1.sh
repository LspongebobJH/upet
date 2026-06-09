
NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=8

export WORLD_SIZE=$(( NPROC_PER_NODE * NODE_COUNT ))
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH

cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/envs/pet/bin:$PATH"

for (( i=0; i<NPROC_PER_NODE; i++ )); do
    export LOCAL_RANK=${i}
    export RANK=$(( NODE_RANK * NPROC_PER_NODE + LOCAL_RANK ))

    CUDA_VISIBLE_DEVICES=${i} python main.py train configs/pet-debug.yaml &
done
