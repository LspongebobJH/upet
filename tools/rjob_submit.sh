#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: rjob.sh [options] [-- script_args...]"
    echo ""
    echo "Options:"
    echo "  --nodes,   -n <num>    Number of nodes (default: 1)"
    echo "  --gpu,     -g <num>    Number of GPUs per node (default: 1)"
    echo "  --memory,  -m <mb>     Memory in MB (default: 128000)"
    echo "  --cpu,     -c <num>    Number of CPUs (default: 16)"
    echo "  --script,  -s <path>   Path to the script to run (required)"
    echo "  --priority, -p <num>   RJOB priority (default: 9)"
    echo "  --name                 RJOB name (default: pet)"
    echo "  --positive-tags <tags>     Positive nodes (machines) indices to use, where different nodes are seperated by commas, such as --positive-tags node/gpu-lg-cmc-h-h200-0114.host.h.pjlab.org.cn,node/gpu-lg-cmc-h-h200-0700.host.h.pjlab.org.cn (default: empty)"
    echo "  --negative-tags <tags>     Negative nodes (machines) indices to avoid, where different nodes are separated by commas, such as --negative-tags node/gpu-lg-cmc-h-h200-0114.host.h.pjlab.org.cn,node/gpu-lg-cmc-h-h200-0700.host.h.pjlab.org.cn (default: empty)"
    echo "  --help,    -h          Show this help message"
    echo ""
    echo "Any arguments after -- are passed directly to the target script."
    exit 0
}

export USER_NAME="lijiahang"

NODES=1
GPU=1
MEMORY=""
CPU=""
SCRIPT=""
PRIORITY=9
POSITIVE_TAGS=""
NEGATIVE_TAGS=""
NAME="pet"
SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)     usage ;;
        --nodes|-n)    NODES="$2";   shift 2 ;;
        --gpu|-g)      GPU="$2";     shift 2 ;;
        --memory|-m)   MEMORY="$2";  shift 2 ;;
        --cpu|-c)      CPU="$2";     shift 2 ;;
        --script|-s)   SCRIPT="$2";  shift 2 ;;
        --priority|-p) PRIORITY="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --positive-tags) POSITIVE_TAGS="$2"; shift 2 ;;
        --negative-tags) NEGATIVE_TAGS="$2"; shift 2 ;;
        --) shift; SCRIPT_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$SCRIPT" ]]; then
    echo "Error: --script is required."
    exit 1
fi

MEMORY=${MEMORY:-$((200000 * GPU))}
CPU=${CPU:-$((16 * GPU))}
HOST_NETWORK=$([[ "$GPU" -lt 8 ]] && echo false || echo true)


RJOB_ARGS=(
    --priority=${PRIORITY}
    --enable-sshd
    --name=${NAME}
    --gpu=${GPU}
    --memory=${MEMORY}
    --cpu=${CPU}
    --charged-group=omnimat_gpu
    --private-machine=group
    --mount=gpfs://gpfs1/${USER_NAME}:/mnt/shared-storage-user/${USER_NAME}
    --mount=gpfs://gpfs2/${USER_NAME}1:/mnt/shared-storage-gpfs2/${USER_NAME}1 \
    --mount=gpfs://gpfs2/ailab-omnimat-shared:/mnt/shared-storage-gpfs2/ailab-omnimat-shared
    --image=registry.h.pjlab.org.cn/ailab-omnimat/chenshuizhou-workspace:20250917184047
    -P ${NODES}
    --host-network=${HOST_NETWORK}
    --preemptible=no
    -e DISTRIBUTED_JOB=true
    --custom-resources rdma/mlnx_shared=8
)

[[ -n "$POSITIVE_TAGS" ]] && RJOB_ARGS+=(--positive-tags "$POSITIVE_TAGS")
[[ -n "$NEGATIVE_TAGS" ]] && RJOB_ARGS+=(--negative-tags "$NEGATIVE_TAGS")

SCRIPT_CMD=("$SCRIPT" "${SCRIPT_ARGS[@]}")
printf -v SCRIPT_CMD_ESCAPED '%q ' "${SCRIPT_CMD[@]}"

rjob submit "${RJOB_ARGS[@]}" \
    -- bash -exc "$SCRIPT_CMD_ESCAPED"
