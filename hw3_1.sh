#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: bash hw3_1.sh <image_folder> <output_file>"
    exit 1
fi

image_folder=$1
output_file=$2

python3 hw3_1.py "$image_folder" "$output_file"
