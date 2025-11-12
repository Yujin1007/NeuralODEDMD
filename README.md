# CDMD-NODE

This is development code of **Stochastic Neural Ordinary Differential Equation - Dynamic Mode Decomposition (NODE-NODE)**.  
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


## 🚀 Usage

### Train model
```bash
python train.py --dataset gray_scott
python train.py --dataset synthetic
python train.py --dataset vorticity
python train.py --dataset cylinder
```

### Evaluate (autoregressive rollout and teacher forcing rollout)
```bash
python eval.py --config_dir ./result/gray_scott/run1 --dataset gray_scott
python eval.py --config_dir ./result/synthetic/run1 --dataset synthetic
python eval.py --config_dir ./result/vorticity/run1 --dataset gray_scott
python eval.py --config_dir ./result/cylinder/run1 --dataset cylinder
```


## 📌 Notes

- This repository is in **development** stage. Expect changes in API and structure.
- Contributions, bug reports, and feedback are welcome!

## 📜 Citation

If you use this code or build upon it, please cite appropriately (to be updated after publication).
