# Sparse-to-Field Reconstruction via Stochastic Neural Dynamic Mode Decomposition

Official implementation of Stochastic NODE--DMD-- Probabilistic modeling of DMD supporting sparse observation and uncertainty quantification. 
by [Yujin Kim](https://yujin1007.github.io/) and [Sarah Dean &dagger;](https://sdean-group.github.io/)


This is development code of **stochastic neural ordinary differential equation - Dynamic Mode Decomposition (CDMD-NODE)**.  
The project explores uncertainty-aware neural ODEs for dynamic mode decomposition with applications to complex dynamical systems.

## 🛠 Environment Setup

We recommend using **conda**:

```bash
conda create -n node_dmd python=3.10
conda activate node_dmd
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### ⚡ GPU Support (PyTorch)

By default `requirements.txt` installs the CPU version of PyTorch.  
If you want GPU acceleration, install PyTorch matching your CUDA version.

For CUDA 12.1:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

More info: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).


## 🚀 Training

### Train model
```bash
python train.py
```

### Evaluate (autoregressive rollout and teacher forcing rollout)
```bash
python eval.py
```


## 📌 Notes

- This repository is in **development** stage. Expect changes in API and structure.
- Contributions, bug reports, and feedback are welcome!

## 📜 Citation

If you use this code or build upon it, please cite appropriately (to be updated after publication).
