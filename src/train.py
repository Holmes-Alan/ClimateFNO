"""
Training script for the Fourier Neural Operator (FNO) climate downscaler.

Usage
-----
    python src/train.py                      # default settings
    python src/train.py --epochs 200 \\
                         --batch_size 16 \\
                         --with_dem          # add DEM conditioning
    python src/train.py --coarsen_factor 8   # 8× downscaling

The script saves:
  - checkpoints/best.pt   – best validation checkpoint
  - checkpoints/last.pt   – end-of-epoch checkpoint
  - results/train_loss.png – loss curve (updated every epoch)
and logs to TensorBoard under runs/.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ERA5EuropeDataset
from model import build_fno


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train FNO for ERA5 downscaling")
    p.add_argument("--data_dir",      type=str,   default="../data")
    p.add_argument("--year_start",    type=int,   default=1979)
    p.add_argument("--year_end",      type=int,   default=2017)
    p.add_argument("--year_val",      type=int,   default=2017,
                   help="First year of validation split (year_val to year_val+1)")
    p.add_argument("--coarsen_factor",type=int,   default=4)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--hidden",        type=int,   default=64,
                   help="Hidden channel width of FNO")
    p.add_argument("--n_modes",       type=int,   default=16,
                   help="Number of Fourier modes per spatial dimension")
    p.add_argument("--n_layers",      type=int,   default=4,
                   help="Number of FNO blocks")
    p.add_argument("--with_dem",      action="store_true",
                   help="Add DEM + land-sea mask as input channels")
    p.add_argument("--dem_file",      type=str,
                   default="../data/ERA5_const_sfc_variables.nc")
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--accum",         type=int,   default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--checkpoint_dir",type=str,   default="../checkpoints")
    p.add_argument("--results_dir",   type=str,   default="../results")
    p.add_argument("--no_time_cond",  action="store_true",
                   help="Disable day-of-year / hour conditioning")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, device, accum, writer,
                global_step):
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)

    with tqdm(loader, desc="  train", leave=False, dynamic_ncols=True) as bar:
        for i, batch in enumerate(bar):
            x      = batch["inputs"].to(device)
            target = batch["targets"].to(device)
            doy    = batch["doy"].to(device)
            hour   = batch["hour"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(x, doy=doy, hour=hour)
                loss = torch.nn.functional.mse_loss(pred, target)
                loss = loss / accum

            scaler.scale(loss).backward()

            if (i + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item() * accum)
            bar.set_postfix(loss=f"{losses[-1]:.4f}")
            writer.add_scalar("Loss/train_step", losses[-1], global_step + i)

    return sum(losses) / len(losses)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    losses, maes = [], []

    for batch in loader:
        x      = batch["inputs"].to(device)
        target = batch["targets"].to(device)
        fine   = batch["fine"].to(device)
        coarse = batch["coarse"].to(device)
        doy    = batch["doy"].to(device)
        hour   = batch["hour"].to(device)

        pred = model(x, doy=doy, hour=hour)
        losses.append(torch.nn.functional.mse_loss(pred, target).item())

        # MAE in physical space (coarse + predicted residual vs fine)
        # inverse-normalise residual
        res_mean = loader.dataset.residual_mean.to(device)
        res_std  = loader.dataset.residual_std.to(device)
        pred_phys = coarse + (pred * res_std[None, :, None, None]
                              + res_mean[None, :, None, None])
        maes.append(torch.mean(torch.abs(fine - pred_phys)).item())

    return sum(losses) / len(losses), sum(maes) / len(maes)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_sample_figure(model, dataset, device, path, n=3):
    """Save a quick visual comparison of coarse / FNO / truth for n samples."""
    model.eval()
    indices = torch.randint(len(dataset), (n,)).tolist()

    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    var_labels = ["T2m [K]", "TP [m]"]

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        x      = sample["inputs"].unsqueeze(0).to(device)
        doy    = sample["doy"].unsqueeze(0).to(device)
        hour   = sample["hour"].unsqueeze(0).to(device)
        fine   = sample["fine"]
        coarse = sample["coarse"]

        pred_res = model(x, doy=doy, hour=hour).squeeze(0).cpu()
        res_mean = dataset.residual_mean
        res_std  = dataset.residual_std
        pred_phys = coarse + (pred_res * res_std[:, None, None]
                              + res_mean[:, None, None])

        # Plot T2m only (channel 0) for brevity
        ch = 0
        vmin = fine[ch].min().item()
        vmax = fine[ch].max().item()
        for col, (data, title) in enumerate([
            (coarse[ch],    "Coarse input"),
            (pred_phys[ch], "FNO output"),
            (fine[ch],      "Ground truth"),
        ]):
            ax = axes[row, col] if n > 1 else axes[col]
            im = ax.imshow(data, origin="upper", vmin=vmin, vmax=vmax,
                           cmap="RdBu_r")
            ax.set_title(f"{title}\n{var_labels[ch]}", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    dem_file = args.dem_file if args.with_dem else None

    print("Building training dataset …")
    ds_train = ERA5EuropeDataset(
        data_dir=args.data_dir,
        year_start=args.year_start,
        year_end=args.year_val,
        coarsen_factor=args.coarsen_factor,
        dem_file=dem_file,
    )
    print("Building validation dataset …")
    ds_val = ERA5EuropeDataset(
        data_dir=args.data_dir,
        year_start=args.year_val,
        year_end=args.year_val + 1,
        coarsen_factor=args.coarsen_factor,
        dem_file=dem_file,
        # Re-use training normalisation statistics
        mean=ds_train.mean,
        std=ds_train.std,
    )

    loader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_fno(
        in_channels=ds_train.n_input_channels,
        out_channels=2,
        hidden_channels=args.hidden,
        n_modes_h=args.n_modes,
        n_modes_w=args.n_modes,
        n_layers=args.n_layers,
        use_time_conditioning=not args.no_time_cond,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FNO parameters: {n_params:,}")

    # ── Optimiser & AMP ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    writer = SummaryWriter(log_dir="runs/fno_climate")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    train_losses, val_losses, val_maes = [], [], []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, loader_train, optimizer, scaler,
                                 device, args.accum, writer, global_step)
        global_step += len(loader_train)

        val_loss, val_mae = validate(model, loader_val, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val",         val_loss,   epoch)
        writer.add_scalar("MAE/val_physical", val_mae,    epoch)

        print(f"  train MSE = {train_loss:.5f}  |  "
              f"val MSE = {val_loss:.5f}  |  val MAE = {val_mae:.5f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  ✓ New best checkpoint saved (val_loss={val_loss:.5f})")

        # Always save last
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_loss": val_loss,
        }, os.path.join(args.checkpoint_dir, "last.pt"))

        # Sample figure every 10 epochs
        if epoch % 10 == 0:
            fig_path = os.path.join(args.results_dir, f"sample_epoch{epoch:04d}.png")
            save_sample_figure(model, ds_val, device, fig_path)
            print(f"  Sample figure saved → {fig_path}")

    # ── Final loss curve ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train MSE")
    ax.plot(val_losses,   label="val MSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.set_title("FNO training — ERA5 Europe downscaling")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "train_loss.png"), dpi=150)
    plt.close(fig)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
