#!/bin/bash

# Target directory
DIR="/mnt/shared-storage-gpfs2/ailab-omnimat-shared/lijiahang/datasets/omat24"

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory $DIR does not exist."
    exit 1
fi

# Change to the target directory
cd "$DIR" || exit 1

# Loop through all .tar.gz files
for archive in *.tar.gz; do
    # Check if any files match the pattern (handle case where no files exist)
    if [ ! -f "$archive" ]; then
        echo "No .tar.gz files found in $DIR."
        break
    fi

    echo "----------------------------------------"
    echo "Processing: $archive"
    
    # Define the target folder name (remove .tar.gz extension)
    # Example: "aimd-from-PBE-1000-npt.tar.gz" -> "aimd-from-PBE-1000-npt"
    folder_name="${archive%.tar.gz}"
    
    # Create the directory
    mkdir -p "$folder_name"
    
    # Extract the archive into the new directory
    # -x: extract, -z: gzip, -f: file, -C: change to directory
    echo "Extracting to ./$folder_name ..."
    tar -xzf "$archive" -C "$folder_name"
    
    # Check if extraction was successful
    if [ $? -eq 0 ]; then
        echo "Extraction successful."
        
        # Delete the original archive to free up space
        echo "Deleting original archive: $archive"
        rm "$archive"
        
        if [ $? -eq 0 ]; then
            echo "✓ Done: $archive extracted and deleted."
        else
            echo "✗ Warning: Could not delete $archive (check permissions)."
        fi
    else
        echo "✗ Error: Failed to extract $archive."
        echo "Keeping the archive for manual inspection."
        # Optional: Remove the empty/incomplete folder if extraction failed
        # rmdir "$folder_name" 
    fi
    
    echo ""
done

echo "All operations completed."