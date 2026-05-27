"""
Evaluation script: load a trained FNO checkpoint and compute metrics on
the test set, then generate comparison plots.

Usage
-----
    python src/evaluate.py --checkpoint checkpoints/best.pt \\
                            --year_test 2018

Outputs are written to results/eval/.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import ERA5EuropeDataset
from model import build_fno


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, default="../checkpoints/best.pt")
    p.add_argument("--data_dir",       type=str, default="../data")
    p.add_argument("--year_test",      type=int, default=2018)
    p.add_argument("--coarsen_factor", type=int, default=4)
    p.add_argument("--with_dem",       action="store_true")
    p.add_argument("--dem_file",       type=str,
                   default="../data/ERA5_const_sfc_variables.nc")
    p.add_argument("--output_dir",     type=str, default="../results/eval")
    p.add_argument("--batch_size",     type=int, default=16)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=16):
    """
    Run the model over the full dataset and return arrays of predictions
    and ground truth in physical space.

    Returns
    -------
    fine_all   : np.ndarray (N, 2, H, W)
    coarse_all : np.ndarray (N, 2, H, W)
    pred_all   : np.ndarray (N, 2, H, W)
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    fine_list, coarse_list, pred_list = [], [], []

    res_mean = dataset.residual_mean.to(device)
    res_std  = dataset.residual_std.to(device)

    for batch in loader:
        x      = batch["inputs"].to(device)
        fine   = batch["fine"].to(device)
        coarse = batch["coarse"].to(device)
        doy    = batch["doy"].to(device)
        hour   = batch["hour"].to(device)

        pred_res  = model(x, doy=doy, hour=hour)
        pred_phys = coarse + (pred_res * res_std[None, :, None, None]
                              + res_mean[None, :, None, None])

        fine_list.append(fine.cpu().numpy())
        coarse_list.append(coarse.cpu().numpy())
        pred_list.append(pred_phys.cpu().numpy())

    return (np.concatenate(fine_list,   axis=0),
            np.concatenate(coarse_list, axis=0),
            np.concatenate(pred_list,   axis=0))


def compute_metrics(fine, coarse, pred):
    """Print and return RMSE and MAE for coarse baseline and FNO."""
    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2, axis=(0, 2, 3)))
    mae  = lambda a, b: np.mean(np.abs(a - b), axis=(0, 2, 3))

    var_names = ["T2m [K]", "TP [m]"]
    metrics = {}
    print("\n──────────────────────────────────────────────")
    print(f"{'Method':<20}  {'Metric':<6}  T2m         TP")
    print("──────────────────────────────────────────────")
    for name, pred_arr in [("Bilinear (coarse)", coarse), ("FNO", pred)]:
        r = rmse(fine, pred_arr)
        m = mae(fine, pred_arr)
        print(f"{name:<20}  RMSE   {r[0]:.4f}      {r[1]:.2e}")
        print(f"{'':<20}  MAE    {m[0]:.4f}      {m[1]:.2e}")
        metrics[name] = {"RMSE": r, "MAE": m}
    print("──────────────────────────────────────────────\n")
    return metrics


def plot_spatial_error(fine, coarse, pred, output_dir, var_idx=0):
    """
    Plot spatial maps of mean absolute error averaged over the test period,
    for the bilinear baseline and FNO side-by-side.
    """
    var_names = ["T2m [K]", "TP [m]"]
    vname = var_names[var_idx]

    mae_coarse = np.mean(np.abs(fine[:, var_idx] - coarse[:, var_idx]), axis=0)
    mae_fno    = np.mean(np.abs(fine[:, var_idx] - pred[:, var_idx]),   axis=0)

    vmax = mae_coarse.max()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title in zip(
        axes,
        [mae_coarse, mae_fno],
        [f"Bilinear MAE — {vname}", f"FNO MAE — {vname}"]
    ):
        im = ax.imshow(data, origin="upper", cmap="hot_r", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Spatial mean absolute error (test period)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f"spatial_mae_{vname.split()[0]}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_example_timestep(fine, coarse, pred, output_dir, t=0, var_idx=0):
    """
    Plot coarse / FNO prediction / ground truth for a single time step.
    """
    var_names = ["T2m [K]", "TP [m]"]
    vname     = var_names[var_idx]
    vmin      = fine[t, var_idx].min()
    vmax      = fine[t, var_idx].max()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title in zip(
        axes,
        [coarse[t, var_idx], pred[t, var_idx], fine[t, var_idx]],
        ["Coarse input (bilinear)", "FNO prediction", "Ground truth"]
    ):
        im = ax.imshow(data, origin="upper", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n{vname}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(output_dir, f"example_t{t}_{vname.split()[0]}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_power_spectrum(fine, coarse, pred, output_dir, var_idx=0):
    """
    Radially-averaged power spectrum comparison — checks whether FNO
    recovers fine-scale spectral energy lost by bilinear interpolation.
    """
    var_names = ["T2m [K]", "TP [m]"]

    def mean_spectrum(x):
        """Mean 1-D spectrum averaged over samples and rows."""
        specs = []
        for i in range(x.shape[0]):
            field = x[i, var_idx]                 # (H, W)
            ft    = np.abs(np.fft.rfft2(field))   # (H, W//2+1)
            specs.append(ft.mean(axis=0))          # average over rows
        return np.mean(specs, axis=0)

    freqs = np.fft.rfftfreq(fine.shape[-1])
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, arr, ls in [
        ("Ground truth", fine,   "-"),
        ("Bilinear",     coarse, "--"),
        ("FNO",          pred,   "-."),
    ]:
        ax.semilogy(freqs[1:], mean_spectrum(arr)[1:], ls, label=name)

    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("Power")
    ax.set_title(f"Power spectrum — {var_names[var_idx]}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"spectrum_{var_names[var_idx].split()[0]}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})

    # ── Build test dataset ────────────────────────────────────────────────────
    dem_file = args.dem_file if args.with_dem else None
    ds_test  = ERA5EuropeDataset(
        data_dir=args.data_dir,
        year_start=args.year_test,
        year_end=args.year_test + 1,
        coarsen_factor=args.coarsen_factor,
        dem_file=dem_file,
    )

    # ── Build and load model ──────────────────────────────────────────────────
    model = build_fno(
        in_channels=ds_test.n_input_channels,
        hidden_channels=saved_args.get("hidden", 64),
        n_modes_h=saved_args.get("n_modes", 16),
        n_modes_w=saved_args.get("n_modes", 16),
        n_layers=saved_args.get("n_layers", 4),
        use_time_conditioning=not saved_args.get("no_time_cond", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f})")

    # ── Run evaluation ────────────────────────────────────────────────────────
    print("Running inference …")
    fine, coarse, pred = evaluate(model, ds_test, device, args.batch_size)

    compute_metrics(fine, coarse, pred)

    for var_idx in range(2):
        plot_spatial_error(fine, coarse, pred, args.output_dir, var_idx)
        plot_example_timestep(fine, coarse, pred, args.output_dir,
                              t=0, var_idx=var_idx)
        plot_power_spectrum(fine, coarse, pred, args.output_dir, var_idx)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
