#!/bin/bash

# SAVE_DIR=$(python get_save_dir.py)
# python train_gray_scott.py
# python eval_gray_scott.py --config_dir ./results/gray_scott/run1_evolve
# python eval_gray_scott.py --config_dir ./results/gray_scott/run1_evolve --ckpt_name model_1000.pt
# python eval_gray_scott.py --config_dir ./results/gray_scott/run1_teacher_forcing
# python eval_navier_stokes.py --config_dir ./results/navier_stokes_obstacle/run1_autoreg
# python eval_navier_stokes_stochastic.py --config_dir ./results/navier_stokes_stochastic/evolve_consistency_from_start
# python stochasticity_test.py --dir evolve_consistency_from_start
python eval_SNode_DMD.py --config_dir ./results/stochastic/evolve_pe
python eval_SNode_DMD.py --config_dir ./results/stochastic/evolve
python eval_SNode_DMD.py --config_dir ./results/stochastic/teacher_forcing


cd /home/yk826/projects/NeuralODEDMD/results/navier_stokes/autoreg/teacher_forcing_reconstruction
cat uncertainty_metrics_summary.json
cd /home/yk826/projects/NeuralODEDMD/results/navier_stokes/autoreg/teacher_forcing_exploitation
cat uncertainty_metrics_summary.json
cd /home/yk826/projects/NeuralODEDMD/results/navier_stokes/autoreg/autoreg_reconstruction
cat uncertainty_metrics_summary.json
cd /home/yk826/projects/NeuralODEDMD/results/navier_stokes/autoreg/autoreg_exploitation
cat uncertainty_metrics_summary.json
