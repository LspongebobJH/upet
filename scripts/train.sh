export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH

cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/envs/pet/bin:$PATH"

CUDA_VISIBLE_DEVICES=0 python \
    main.py train configs/pet-omat-xs-v1.0.0-lr-1e-3.yaml \
    -r architecture.training.num_epochs=20 &
sleep 5

CUDA_VISIBLE_DEVICES=1 python \
    main.py train configs/pet-omat-xs-v1.0.0-lr-1e-3.yaml \
    -r architecture.training.learning_rate=5e-3 &
sleep 5

CUDA_VISIBLE_DEVICES=2 python \
    main.py train configs/pet-omat-xs-v1.0.0-lr-1e-3.yaml \
    -r architecture.training.learning_rate=1e-2 &
sleep 5

CUDA_VISIBLE_DEVICES=3 python \
    main.py train configs/pet-omat-xs-v1.0.0-lr-1e-3.yaml \
    -r architecture.training.learning_rate=5e-2 &
sleep 5

CUDA_VISIBLE_DEVICES=4 python \
    main.py train configs/pet-omat-xs-v1.0.0-lr-1e-3.yaml &
sleep 5

CUDA_VISIBLE_DEVICES=5 python \
    main.py train configs/pet-omat-xs-v1.0.0-lr-1e-3.yaml \
    -r architecture.training.loss.energy.weight=1e-3 &
sleep 5


wait

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash tools/train.sh --nproc_per_node 6
