#!/bin/bash

# ==============================================================================
# Configuration
# ==============================================================================

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:$PATH

cd /mnt/shared-storage-gpfs2/lijiahang1/jobs/upet
export PATH="/mnt/shared-storage-user/lijiahang/miniconda3/envs/pet/bin:$PATH"

# Path to the python script
PYTHON_SCRIPT="./tools/aselmdb2xyz.py"

# Target output directory
TARGET_DIR="/mnt/shared-storage-gpfs2/lijiahang1/datasets/omat24-1m/omat24_1M_251210/train/xyz_files"

# List of specific folders containing train.aselmdb
FOLDERS=(
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

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "=================================================="
echo "Starting conversion for specific folders"
echo "Target Dir: $TARGET_DIR"
echo "=================================================="

# ==============================================================================
# Loop through the defined list
# ==============================================================================

for folder in "${FOLDERS[@]}"; do
    # Define input file path
    input_db="/mnt/shared-storage-gpfs2/lijiahang1/datasets/omat24-1m/omat24_1M_251210/train/${folder}/train.aselmdb"

    # Check if the input file actually exists in this folder
    if [ ! -f "$input_db" ]; then
        echo "WARNING: Skipping '$folder'. File not found: $input_db"
        continue
    fi

    # Define output file path: <folder_name>.xyz
    output_xyz="${TARGET_DIR}/${folder}.xyz"

    echo ""
    echo "Processing: $folder"
    echo "  Input:  $input_db"
    echo "  Output: $output_xyz"

    # Run the python converter
    python "$PYTHON_SCRIPT" \
        --input_path "$input_db" \
        --output_path "$output_xyz" \
        --force &

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Converted $folder"
    else
        echo "ERROR: Failed to convert $folder"
    fi
done

echo ""
echo "=================================================="
echo "All specified conversions completed."
echo "=================================================="