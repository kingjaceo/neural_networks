"""
Generate palette-snapped 4x4 sample grids for all seven presets.

Run from final_project/:
    python -m scripts.generate_snapped_figures

Outputs: samples/snapped/preset_XX.png
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from scripts.gan_terrain import (
    load_weights,
    _generator_forward,
    _snap_to_palette,
    LATENT_DIM,
    IMG_SIZE,
)

GRID_N = 4
SEED = 0
PRESETS = [f"preset_{i:02d}" for i in range(1, 8)]
CHECKPOINT_DIR = "checkpoints"
OUT_DIR = "samples/snapped"


def make_snapped_grid(preset, checkpoint_dir, n=4, seed=0):
    p = load_weights(preset, checkpoint_dir)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n * n, LATENT_DIM)).astype(np.float32)
    raw_batch = _generator_forward(z, p)  # (n*n, H, W, 3)
    grid = Image.new("RGB", (IMG_SIZE * n, IMG_SIZE * n))
    for idx in range(n * n):
        rgb = ((raw_batch[idx] * 0.5 + 0.5).clip(0.0, 1.0) * 255.0).astype(np.uint8)
        snapped, _ = _snap_to_palette(rgb)
        img = Image.fromarray(snapped)
        r, c = divmod(idx, n)
        grid.paste(img, (c * IMG_SIZE, r * IMG_SIZE))
    return grid


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for preset in PRESETS:
        npz_path = os.path.join(CHECKPOINT_DIR, preset, "generator.npz")
        if not os.path.isfile(npz_path):
            print(f"Skipping {preset}: no generator.npz found")
            continue
        print(f"Generating {preset}...", end=" ", flush=True)
        grid = make_snapped_grid(preset, CHECKPOINT_DIR, n=GRID_N, seed=SEED)
        out_path = os.path.join(OUT_DIR, f"{preset}.png")
        grid.save(out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
