export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH

cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/envs/pet/bin:$PATH"

CUDA_VISIBLE_DEVICES=0 python main.py train configs/pet-omat-xs-v1.0.0-lr-1e-3.yaml &
sleep 5
CUDA_VISIBLE_DEVICES=1 python main.py train configs/pet-omat-xs-v1.0.0-lr-5e-3.yaml & 
sleep 5
CUDA_VISIBLE_DEVICES=2 python main.py train configs/pet-omat-xs-v1.0.0-lr-5e-4.yaml

CUDA_VISIBLE_DEVICES=0,1,2 bash tools/occupy.sh --nproc_per_node 3
