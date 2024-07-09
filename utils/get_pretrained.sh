#!/bin/bash

FILE=$1

echo "Note: available models are densenet, inception, or resnet."
echo "specified $FILE"

# Declare associative arrays for URLs and model file paths
declare -A URLS
declare -A MODEL_FILES

URLS[densenet]="https://iffcloud.fz-juelich.de/s/xeoR4J75gaxbjPq/download/densenet.pt"
URLS[inception]="https://iffcloud.fz-juelich.de/s/72syL45p8YZBQD5/download/inception.pt"
URLS[resnet]="https://iffcloud.fz-juelich.de/s/gDbX7k6LxECgwnX/download/resnet.pt"

MODEL_FILES[densenet]="./parameters/densenet.pt"
MODEL_FILES[inception]="./parameters/inception.pt"
MODEL_FILES[resnet]="./parameters/resnet.pt"

download_model() {
    local model=$1
    wget "${URLS[$model]}" -O "${MODEL_FILES[$model]}"
}

if [[ $FILE == "all" ]]; then
    echo "Downloading all"
    for model in "${!URLS[@]}"; do
        download_model $model
    done
elif [[ -n ${URLS[$FILE]} ]]; then
    download_model $FILE
else
    echo "WRONG MODEL NAME! Use one of: densenet, inception, resnet"
    exit 1
fi