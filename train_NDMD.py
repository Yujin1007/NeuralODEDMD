import os
import torch
import torch.optim as optim
from tqdm import trange
from config.config import Stochastic_Node_DMD_Config
from dataset.generate_synth_dataset import load_synth, SynthDataset
from models.ndmd import NeuralDMD
from utils.losses import pixel_loss_fn
from utils.utils import ensure_dir, reparameterize_full
import random
import numpy as np
from torch.utils.data import DataLoader
from utils.plots import plot_loss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_step(model, opt_state, xy, target_values, time_indices, optimizer, loss_fn):
    """
    Performs a single training step for the model.

    Args:
        model: PyTorch model (e.g., NeuralDMD).
        opt_state: PyTorch optimizer (e.g., torch.optim.Adam).
        xy: Tensor of shape (N, 2) for coordinates.
        target_values: Tensor of shape (N, T) for target pixel values.
        time_indices: Tensor of shape (T,) for time points.
        optimizer: PyTorch optimizer.
        beta: Hyperparameters for loss function (float or dict).
        frame_max: Maximum frame value for scaling.
        frame_min: Minimum frame value for scaling.
        loss_fn: Loss function (e.g., pixel_loss_fn).

    Returns:
        model: Updated model.
        opt_state: Updated optimizer.
        loss: Total loss for the batch.
        reconstruction_loss: Reconstruction loss.
        sparsity_loss: Sparsity loss.
        grads: List of gradients for model parameters.
    """
    # Ensure model is in training mode
    model.train()

    # Zero out gradients
    optimizer.zero_grad()

    # Compute loss and auxiliary outputs
    loss, aux = loss_fn(
        model, xy, target_values, time_indices,
        # beta_tv=beta, beta_neg=beta, beta_sparse=beta,  # Assuming beta is a float or dict
    )

    # Compute gradients
    loss.backward()
    grads = [param.grad for param in model.parameters()]

    # Update model parameters
    optimizer.step()

    # Extract auxiliary losses
    reconstruction_loss = aux["recon"]
    sparsity_loss = aux["sparse"]

    return model, opt_state, loss, reconstruction_loss, sparsity_loss, grads

def train_epoch(model, opt_state, xy_array, pix_array, time_array, optimizer, beta=0, frame_max=None, frame_min=None, initial_loss=None):
    """
    Trains the model for one epoch over the provided batches.
    
    Args:
        model: PyTorch model (e.g., NeuralDMD).
        opt_state: PyTorch optimizer (e.g., torch.optim.Adam).
        xy_array: Tensor of shape (num_batches, N, 2) for coordinates.
        pix_array: Tensor of shape (num_batches, N, T) for pixel data.
        time_array: Tensor of shape (num_batches, T) for time points.
        optimizer: PyTorch optimizer.
        beta: Hyperparameters for loss function (dict or tuple).
        frame_max: Maximum frame value for scaling.
        frame_min: Minimum frame value for scaling.
        initial_loss: Optional initial loss value (default: None).
    
    Returns:
        final_model: Updated model.
        final_opt_state: Updated optimizer.
        avg_loss: Sum of losses across batches.
        rec_avg: Sum of reconstruction losses.
        ortho_avg: Sum of orthogonal losses.
        grads: List of gradients from each batch.
        initial_loss: Initial loss value.
    """
    # Ensure model is in training mode
    model.train()
    
    # Calculate initial loss for the first batch if not provided
    if initial_loss is None:
        xy0 = xy_array[0]
        pixels0 = pix_array[0]
        times0 = time_array[0]
        _, _, initial_loss, _, _, _ = train_step(
            model, opt_state, xy0, pixels0, times0, optimizer, pixel_loss_fn
        )
        print(f"initial_loss: {initial_loss.item()}")

    # Initialize accumulators for losses and gradients
    num_batches = xy_array.shape[0]
    losses = []
    rec_losses = []
    ortho_losses = []
    grads = []

    # Iterate over batches
    for batch_idx in range(num_batches):
        # Extract batch data
        xy = xy_array[batch_idx]
        pixels = pix_array[batch_idx]
        times = time_array[batch_idx]

        # Add noise to xy
        noise = torch.randn_like(xy) * 0.01
        xy_noisy = xy + noise

        # Perform training step
        new_model, new_opt_state, loss, rec_loss, ortho_loss, batch_grads = train_step(
            model, opt_state, xy_noisy, pixels, times, optimizer, pixel_loss_fn
        )

        # Conditional update: only update if loss is less than or equal to initial_loss
        if loss <= initial_loss:
            model.load_state_dict(new_model.state_dict())
            opt_state.load_state_dict(new_opt_state.state_dict())
        
        # Accumulate losses and gradients
        losses.append(loss)
        rec_losses.append(rec_loss)
        ortho_losses.append(ortho_loss)
        grads.append(batch_grads)

    # Compute summed losses
    avg_loss = sum(losses)
    rec_avg = sum(rec_losses)
    ortho_avg = sum(ortho_losses)

    return model, opt_state, avg_loss, rec_avg, ortho_avg, grads, initial_loss

def run_train(cfg: Stochastic_Node_DMD_Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    t_list, coords_list, y_list, *_ = load_synth(device, T=cfg.data_len)
    dataset = SynthDataset(t_list, coords_list, y_list)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,  # Add  batch_size to your config
        shuffle=False,  # Set to True if you want to shuffle time steps
        num_workers=0,  # Set to >0 for parallel loading, but 0 is fine for small datasets
        drop_last=True  # Keep the last incomplete batch
    )   
    model = NeuralDMD(r=4, hidden_size=256, layers=4, num_frequencies=10)
    
    # pretrained_path = "results/stochastic/run11"
    # model = _prepare_model(cfg, pretrained_path)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    best = float("inf")
    ensure_dir(cfg.save_dir)
    
    avg_loss_list = []
    for epoch in trange(cfg.num_epochs, desc="Training"):
        total_loss = 0.0
        num_batches = 0
        # u_pred = y_list[0].unsqueeze(0).repeat(cfg.batch_size, *([1] * y_list[0].dim())).to(device)
        t_prev = torch.tensor(t_list[0], dtype=torch.float32, device=device).unsqueeze(0).repeat(cfg.batch_size, )
        u_pred = y_list[0]
        t_prev = t_prev.item()
        for batch in dataloader:
            t_next, coords, y_next, y_prev = [x.to(device) for x in batch]
            # print(f"shape t_next {t_next[0].shape}, coords {coords[0].shape}, y_next {y_next[0].shape}")
            t_next = t_next.item()          # converts 0-dim tensor or [1] tensor to a Python float
            coords   = coords.squeeze(0)       # [1,102,2] -> [102,2]
            y_next   = y_next.squeeze(0)       # [1,102,2] -> [102,2]
            y_prev   = y_prev.squeeze(0)       # [1,102,2] -> [102,2]

            model, opt_state, avg_loss, rec_loss, ortho_loss, grads, initial_loss = train_epoch(
                model,
                opt_state,
                coords,
                y_next,
                t_next,
                opt,

            )

        avg = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_loss_list.append(avg)
        if epoch % cfg.print_every == 0:
            print(f"Epoch {epoch:04d} | avg_loss={avg:.6f}")
            # torchㄴsave_dir, f'ckpt_model_{epoch}.pt'))
        if avg < best:
            best = avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_loss': best,
                'loss_list': avg_loss_list
            }, os.path.join(cfg.save_dir, 'best_model.pt'))
            plot_loss(avg_loss_list, cfg.save_dir)
    torch.save({
        'epoch': cfg.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'best_loss': best,
        'loss_list': avg_loss_list
    }, os.path.join(cfg.save_dir, 'final_model.pt'))
    plot_loss(avg_loss_list, cfg.save_dir, "final_loss.png")

    print(f"Training complete. Final model saved at {os.path.join(cfg.save_dir, 'final_model.pt')}")
    print(f"Best model saved at {os.path.join(cfg.save_dir, 'best_model.pt')} with loss {best:.6f}")


if __name__ == "__main__":
    run_train(Stochastic_Node_DMD_Config())
