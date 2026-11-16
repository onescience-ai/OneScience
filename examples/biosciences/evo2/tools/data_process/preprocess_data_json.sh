#!/bin/bash

# INPUT_DIR="../promoters" 
INPUT_DIR="/public/onestore/onedatasets/evo2/json/pretraining_or_both_phases/promoters"

OUTPUT_DIR="data/promoters/pretraining_data_promoters"
TOKENIZER="CharLevelTokenizer"

mkdir -p $OUTPUT_DIR

for SPLIT in train test valid; do
    INPUT_FILE="${INPUT_DIR}/data_promoters_${SPLIT}_chunk1.jsonl.gz"
    SPLIT_DIR="${OUTPUT_DIR}/data_promoters_${SPLIT}_text_${TOKENIZER}_document"
    OUTPUT_PREFIX="${OUTPUT_DIR}/data_promoters_${SPLIT}"

    mkdir -p "$SPLIT_DIR"
    echo "Processing $INPUT_FILE -> $OUTPUT_PREFIX"

    if python preprocess_data_json.py \
        --input "$INPUT_FILE" \
        --output-prefix "$OUTPUT_PREFIX" \
        --tokenizer-type "$TOKENIZER" \
        --dataset-impl mmap \
        --append-eod \
        # --enforce-sample-length 8192 \
        --workers 8 \
        --log-interval 100; then
        echo "Finished processing split: $SPLIT"
    else
        echo "Error processing split: $SPLIT, skipping..."
    fi
done
