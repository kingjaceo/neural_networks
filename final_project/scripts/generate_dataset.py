"""
Generate terrain dataset for GAN training.

Workflow:
  1. python preview_presets.py          # visual grid -> open preset_preview.png
  2. Edit SELECTED_PRESETS below
  3. python generate_dataset.py         # full run  (default: 1400 train + 143 test per preset)

Quick test:
  python generate_dataset.py --n_train 5 --n_test 2 --workers 4 --output_dir dataset_test
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
from opensimplex import noise2array

from noise import noise_to_image

# ── Edit this list after running preview_presets.py ──────────────────────────
SELECTED_PRESETS = [
    "preset_01",
    "preset_02",
    "preset_03",
    "preset_04",
    "preset_05",
    "preset_06",
    "preset_07",
]

ALL_PRESETS = {
    "preset_01": {"shape": (128, 128), "depth": 6, "scale": 200.0, "factor": 0.45, "persistence": 0.40},
    "preset_02": {"shape": (128, 128), "depth": 7, "scale": 180.0, "factor": 0.48, "persistence": 0.50},
    "preset_03": {"shape": (128, 128), "depth": 5, "scale": 250.0, "factor": 0.42, "persistence": 0.45},
    "preset_04": {"shape": (128, 128), "depth": 8, "scale": 160.0, "factor": 0.52, "persistence": 0.35},
    "preset_05": {"shape": (128, 128), "depth": 4, "scale": 300.0, "factor": 0.40, "persistence": 0.55},
    "preset_06": {"shape": (128, 128), "depth": 6, "scale": 220.0, "factor": 0.55, "persistence": 0.42},
    "preset_07": {"shape": (128, 128), "depth": 7, "scale": 140.0, "factor": 0.46, "persistence": 0.48},
    "preset_08": {"shape": (128, 128), "depth": 5, "scale": 270.0, "factor": 0.50, "persistence": 0.38},
    "preset_09": {"shape": (128, 128), "depth": 9, "scale": 190.0, "factor": 0.43, "persistence": 0.32},
    "preset_10": {"shape": (128, 128), "depth": 6, "scale": 170.0, "factor": 0.38, "persistence": 0.52},
}

SCALE_JITTER       = 0.15
PERSISTENCE_JITTER = 0.10
FACTOR_JITTER      = 0.10

N_TRAIN_PER_PRESET = 1400
N_TEST_PER_PRESET  = 143


def generate_one_image(args):
    """Worker: generate one terrain image with jitter. Returns (uint8 ndarray H×W×3, label_int)."""
    label_int, base_params, seed = args
    rng = np.random.default_rng(seed)

    shape       = base_params["shape"]
    depth       = base_params["depth"]
    scale       = base_params["scale"]       * rng.uniform(1.0 - SCALE_JITTER,       1.0 + SCALE_JITTER)
    factor      = base_params["factor"]      * rng.uniform(1.0 - FACTOR_JITTER,      1.0 + FACTOR_JITTER)
    persistence = base_params["persistence"] * rng.uniform(1.0 - PERSISTENCE_JITTER, 1.0 + PERSISTENCE_JITTER)

    # Random offset so each image samples a different region of noise space.
    # Without this, all images from the same preset would be near-identical.
    x_off = rng.uniform(0.0, 9999.0)
    y_off = rng.uniform(0.0, 9999.0)

    final_noise = np.zeros(shape)
    amplitude   = 1.0
    total_amp   = 0.0
    cur_scale   = scale
    for _ in range(depth):
        x = np.linspace(x_off, x_off + shape[0] / cur_scale, shape[0])
        y = np.linspace(y_off, y_off + shape[1] / cur_scale, shape[1])
        final_noise += noise2array(x, y) * amplitude
        total_amp   += amplitude
        amplitude   *= persistence
        cur_scale   *= factor

    rgb = np.array(noise_to_image(final_noise / total_amp), dtype=np.uint8)
    return rgb, label_int


def main():
    parser = argparse.ArgumentParser(description="Generate terrain dataset")
    parser.add_argument("--output_dir", default="dataset")
    parser.add_argument("--n_train",    type=int, default=N_TRAIN_PER_PRESET,
                        help="training images per preset")
    parser.add_argument("--n_test",     type=int, default=N_TEST_PER_PRESET,
                        help="test images per preset")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--workers",    type=int, default=min(mp.cpu_count(), 6),
                        help="parallel workers (default: min(cpu_count, 6))")
    cfg = parser.parse_args()

    presets     = {k: ALL_PRESETS[k] for k in SELECTED_PRESETS}
    class_names = list(presets.keys())
    label_map   = {name: i for i, name in enumerate(class_names)}

    os.makedirs(cfg.output_dir, exist_ok=True)
    master_rng = np.random.default_rng(cfg.seed)

    print(f"Presets : {class_names}")
    print(f"Workers : {cfg.workers}")

    for split, n_per in [("train", cfg.n_train), ("test", cfg.n_test)]:
        tasks = []
        for name, params in presets.items():
            label = label_map[name]
            for _ in range(n_per):
                seed = int(master_rng.integers(0, 2**31))
                tasks.append((label, params, seed))

        idx   = master_rng.permutation(len(tasks))
        tasks = [tasks[i] for i in idx]

        n_total   = len(tasks)
        chunksize = max(1, n_total // (cfg.workers * 4))
        print(f"\nGenerating {n_total} {split} images ({len(class_names)} classes × {n_per})...")
        t0 = time.time()

        sample_shape = next(iter(presets.values()))["shape"]
        images = np.empty((n_total, sample_shape[0], sample_shape[1], 3), dtype=np.uint8)
        labels = np.empty(n_total, dtype=np.int8)

        with mp.Pool(cfg.workers) as pool:
            for i, (img, lbl) in enumerate(
                pool.imap(generate_one_image, tasks, chunksize=chunksize)
            ):
                images[i] = img
                labels[i] = lbl

        img_path = os.path.join(cfg.output_dir, f"{split}_images.npy")
        lbl_path = os.path.join(cfg.output_dir, f"{split}_labels.npy")
        np.save(img_path, images)
        np.save(lbl_path, labels)
        print(f"  {img_path}  ({images.nbytes / 1e6:.0f} MB, {n_total} images, {time.time() - t0:.1f}s)")

    meta = {
        "class_names":         class_names,
        "seed":                cfg.seed,
        "n_train_per_preset":  cfg.n_train,
        "n_test_per_preset":   cfg.n_test,
        "jitter": {
            "scale":       SCALE_JITTER,
            "persistence": PERSISTENCE_JITTER,
            "factor":      FACTOR_JITTER,
        },
    }
    with open(os.path.join(cfg.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Metadata -> {cfg.output_dir}/metadata.json")


if __name__ == "__main__":
    main()
