#!/bin/bash

# Unified training script for Navier-Stokes, Gray-Scott, and SNode DMD datasets
# Usage: ./train.sh [navier_stokes|gray_scott|snode_dmd]

# Check if dataset argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Dataset type required"
    echo "Usage: $0 [navier_stokes|gray_scott|snode_dmd]"
    exit 1
fi

DATASET=$1

# Validate dataset type
if [ "$DATASET" != "navier_stokes" ] && [ "$DATASET" != "gray_scott" ] && [ "$DATASET" != "snode_dmd" ]; then
    echo "Error: Invalid dataset type '$DATASET'"
    echo "Please choose: navier_stokes, gray_scott, or snode_dmd"
    exit 1
fi

# Run training
echo "Starting training for dataset: $DATASET"
python train_unified.py --dataset $DATASET

