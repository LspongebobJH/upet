#!/bin/bash

# Target directory
TARGET_DIR="/mnt/shared-storage-gpfs2/ailab-omnimat-shared/lijiahang/datasets/omat24"

# Create directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Base URL
BASE_URL="https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train"

# List of file names
NAMES=(
    "aimd-from-PBE-1000-npt"
    "aimd-from-PBE-3000-npt"
    "rattled-1000"
    "rattled-300"
    "rattled-500"
    "rattled-relax"
    "aimd-from-PBE-1000-nvt"
    "aimd-from-PBE-3000-nvt"
    "rattled-1000-subsampled"
    "rattled-300-subsampled"
    "rattled-500-subsampled"
)

# Loop through each name and download
for name in "${NAMES[@]}"; do
    LINK="${BASE_URL}/${name}.tar.gz"
    
    echo "=========================================="
    echo "Downloading: ${name}.tar.gz"
    echo "To: ${TARGET_DIR}"
    echo "URL: ${LINK}"
    echo "=========================================="
    
    aria2c \
      -x 16 -s 16 \
      -k 1M \
      --file-allocation=none \
      -c \
      -d "$TARGET_DIR" \
      "${LINK}"
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded: ${name}.tar.gz"
    else
        echo "✗ Failed to download: ${name}.tar.gz"
    fi
    
    echo ""
done

echo "All downloads completed!"