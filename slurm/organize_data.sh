#!/bin/bash

# Check if the output directory is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_dir="$1"
output_dir="$2"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all the tar, tar.gz, and tar.bz2 files in the input directory
for file in "$input_dir"/*; do
    if [[ "$file" == *.tar.gz ]]; then
        echo "Extracting gzip archive: $file"
        tar -xzvf "$file" -C "$output_dir"

    elif [[ "$file" == *.tar.bz2 ]]; then
        echo "Extracting bzip2 archive: $file"
        tar -xjvf "$file" -C "$output_dir"

    elif [[ "$file" == *.tar ]]; then
        echo "Extracting tar archive: $file"
        tar -xvf "$file" -C "$output_dir"

    else
        echo "Skipping unsupported file type: $file"
    fi
done
