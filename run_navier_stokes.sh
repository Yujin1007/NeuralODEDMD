#!/bin/bash

SAVE_DIR=$(python utils/get_save_dir.py)
echo $SAVE_DIR
python train_navier_stokes.py
python eval_navier_stokes.py --config_dir "$SAVE_DIR"