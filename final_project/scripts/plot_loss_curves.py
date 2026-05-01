"""
Plot training loss curves for the report.

Inputs:
  logs/dcgan_preset_{01..07}.csv   (per-epoch G/D loss for each unconditional DCGAN)
  logs/cdcgan.csv                  (per-epoch G/D loss for the partial cDCGAN)

Outputs:
  figures/loss_curves.pdf          7-panel grid: G & D loss per preset
  figures/loss_curves.png          same, raster
  figures/cdcgan_loss.pdf          single panel: cDCGAN G & D loss
  figures/cdcgan_loss.png          same, raster

Usage:
  python scripts/plot_loss_curves.py
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
LOG_DIR    = ROOT / "logs"
FIG_DIR    = ROOT / "figures"
PRESETS    = [f"preset_{i:02d}" for i in range(1, 8)]


def read_csv(path):
    epochs, g_loss, d_loss = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            g_loss.append(float(row["g_loss"]))
            d_loss.append(float(row["d_loss"]))
    return np.array(epochs), np.array(g_loss), np.array(d_loss)


def plot_dcgan_grid():
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True)
    axes = axes.ravel()

    for ax, preset in zip(axes[:7], PRESETS):
        path = LOG_DIR / f"dcgan_{preset}.csv"
        if not path.exists():
            ax.set_title(f"{preset} (missing)", fontsize=9)
            ax.axis("off")
            continue
        ep, g, d = read_csv(path)
        ax.plot(ep, g, label=r"$\mathcal{L}_G$", color="C0", linewidth=1.2)
        ax.plot(ep, d, label=r"$\mathcal{L}_D$", color="C3", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
        ax.set_title(preset, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].axis("off")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)

    for ax in axes[4:7]:
        ax.set_xlabel("epoch", fontsize=9)
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("loss", fontsize=9)

    fig.suptitle("Per-epoch WGAN-GP losses, unconditional DCGANs", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_pdf = FIG_DIR / "loss_curves.pdf"
    out_png = FIG_DIR / "loss_curves.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_pdf.relative_to(ROOT)}")
    print(f"  wrote {out_png.relative_to(ROOT)}")


def plot_cdcgan():
    path = LOG_DIR / "cdcgan.csv"
    if not path.exists():
        print(f"  skipping cDCGAN: {path} not found")
        return

    ep, g, d = read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ep, g, label=r"$\mathcal{L}_G$", color="C0", linewidth=1.4, marker="o", markersize=3)
    ax.plot(ep, d, label=r"$\mathcal{L}_D$", color="C3", linewidth=1.4, marker="o", markersize=3)
    ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"cDCGAN losses (incomplete: {len(ep)}/50 epochs)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    out_pdf = FIG_DIR / "cdcgan_loss.pdf"
    out_png = FIG_DIR / "cdcgan_loss.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_pdf.relative_to(ROOT)}")
    print(f"  wrote {out_png.relative_to(ROOT)}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    print(f"Output dir: {FIG_DIR}")
    plot_dcgan_grid()
    plot_cdcgan()


if __name__ == "__main__":
    main()
