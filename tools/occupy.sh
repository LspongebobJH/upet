NPROC_PER_NODE=8
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

export JOB_DIR="/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet"
cd ${JOB_DIR}
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/bin:$PATH"
. /mnt/shared-storage-user/lijiahang/miniconda3/etc/profile.d/conda.sh
conda activate pet

NODE_RANK=${NODE_RANK} # muxi incorrectly set RANK to node rank
export WORLD_SIZE=$(( NPROC_PER_NODE * NODE_COUNT ))

for (( i=0; i<NPROC_PER_NODE; i++ )); do
    export LOCAL_RANK=${i}
    export RANK=$(( NODE_RANK * NPROC_PER_NODE + LOCAL_RANK ))

    python tools/test.py &
done

# wait
echo "ALL DONE"
sleep inf