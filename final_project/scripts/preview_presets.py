"""
Preview all terrain presets to help choose which to include in the dataset.

Usage:
    python preview_presets.py              # 3 samples per preset
    python preview_presets.py --samples 5  # more samples per preset
    python preview_presets.py --output my_preview.png

Output: preset_preview.png — open it, pick the presets you want, then edit
SELECTED_PRESETS in generate_dataset.py.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_dataset import ALL_PRESETS, generate_one_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3,
                        help="number of sample images per preset")
    parser.add_argument("--output",  default="preset_preview.png")
    cfg = parser.parse_args()

    n_rows = len(ALL_PRESETS)
    n_cols = cfg.samples
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, (name, params) in enumerate(ALL_PRESETS.items()):
        for col in range(n_cols):
            rgb, _ = generate_one_image((0, params, row * 1000 + col))
            axes[row, col].imshow(rgb)
            axes[row, col].axis("off")
        # Label on the leftmost cell
        axes[row, 0].set_title(
            f"{name}\n"
            f"scale={params['scale']:.0f}  depth={params['depth']}\n"
            f"factor={params['factor']:.2f}  pers={params['persistence']:.2f}",
            fontsize=7, loc="left", pad=3,
        )

    fig.suptitle(
        f"All presets — {cfg.samples} random samples each (jitter + offset applied)\n"
        "Edit SELECTED_PRESETS in generate_dataset.py after choosing.",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(cfg.output, dpi=120, bbox_inches="tight")
    print(f"Saved {cfg.output}")


if __name__ == "__main__":
    main()
