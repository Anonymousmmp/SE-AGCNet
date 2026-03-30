#!/bin/bash

# Set your dataset path 
DATA_DIR="/path/to/LibriTTS"

# Output directory
OUTPUT_DIR="/path/to/output"

# Processing mode: test or train
MODE="train"

python DATAGEN/LibriAGC_gen.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mode ${MODE}
