#!/bin/bash

SAVE_DIR=$(python get_save_dir.py)
python train_gray_scott.py
python eval_gray_scott.py --config_dir "$SAVE_DIR"