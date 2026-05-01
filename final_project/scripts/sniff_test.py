"""
Dataset sniff test for the terrain GAN project.

Checks:
  - Array shapes, dtypes, and counts match expected values
  - Per-class counts are balanced
  - Pixel value statistics look reasonable (not all-black, not clipped)
  - Images are visually distinct across classes

Outputs:
  - Console report with PASS/FAIL/WARN per check
  - dataset_grid.png  — 7 rows (one per terrain class) × 8 random samples each

Usage:
  python sniff_test.py                         # checks dataset/
  python sniff_test.py --data_dir dataset_test # checks dataset_test/
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_PER_ROW   = 8    # columns in the output grid
EXPECTED_TRAIN    = 9800
EXPECTED_TEST     = 1001
EXPECTED_CLASSES  = 7
IMG_SHAPE         = (128, 128, 3)
PIXEL_MIN_OK      = 10   # warn if min mean-per-image is below this
PIXEL_MAX_OK      = 245  # warn if max mean-per-image is above this


# ── Helpers ───────────────────────────────────────────────────────────────────
_PASS  = "\033[32mPASS\033[0m"
_FAIL  = "\033[31mFAIL\033[0m"
_WARN  = "\033[33mWARN\033[0m"

def passed(msg):  print(f"  [{_PASS}] {msg}")
def failed(msg):  print(f"  [{_FAIL}] {msg}"); _failures.append(msg)
def warned(msg):  print(f"  [{_WARN}] {msg}")

_failures = []


def check(cond, pass_msg, fail_msg):
    if cond:
        passed(pass_msg)
    else:
        failed(fail_msg)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"\n{'='*60}")
    print(f"  Terrain GAN — Dataset Sniff Test")
    print(f"  Directory: {os.path.abspath(data_dir)}")
    print(f"{'='*60}\n")

    # ── 1. File existence ─────────────────────────────────────────────────────
    print("[ File existence ]")
    required = ["train_images.npy", "train_labels.npy",
                "test_images.npy",  "test_labels.npy", "metadata.json"]
    for fname in required:
        path = os.path.join(data_dir, fname)
        check(os.path.isfile(path), f"{fname} exists", f"{fname} MISSING at {path}")

    if _failures:
        print("\nCritical files missing — aborting.\n")
        sys.exit(1)

    # ── 2. Metadata ───────────────────────────────────────────────────────────
    print("\n[ Metadata ]")
    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    n_classes   = len(class_names)
    check(n_classes == EXPECTED_CLASSES,
          f"{n_classes} classes: {class_names}",
          f"Expected {EXPECTED_CLASSES} classes, got {n_classes}: {class_names}")

    n_train_per = meta.get("n_train_per_preset", "?")
    n_test_per  = meta.get("n_test_per_preset",  "?")
    print(f"  [INFO] train/preset={n_train_per}, test/preset={n_test_per}, seed={meta.get('seed')}")
    print(f"  [INFO] jitter: {meta.get('jitter')}")

    # ── 3. Load arrays ────────────────────────────────────────────────────────
    print("\n[ Loading arrays (mmap) ]")
    tr_img = np.load(os.path.join(data_dir, "train_images.npy"), mmap_mode="r")
    tr_lbl = np.load(os.path.join(data_dir, "train_labels.npy"), mmap_mode="r")
    te_img = np.load(os.path.join(data_dir, "test_images.npy"),  mmap_mode="r")
    te_lbl = np.load(os.path.join(data_dir, "test_labels.npy"),  mmap_mode="r")
    print(f"  [INFO] train_images: {tr_img.shape}, dtype={tr_img.dtype}")
    print(f"  [INFO] train_labels: {tr_lbl.shape}, dtype={tr_lbl.dtype}")
    print(f"  [INFO] test_images:  {te_img.shape}, dtype={te_img.dtype}")
    print(f"  [INFO] test_labels:  {te_lbl.shape}, dtype={te_lbl.dtype}")

    # ── 4. Shape / dtype checks ───────────────────────────────────────────────
    print("\n[ Shape and dtype ]")
    check(tr_img.ndim == 4 and tr_img.shape[1:] == IMG_SHAPE,
          f"train_images shape OK: {tr_img.shape}",
          f"train_images bad shape: {tr_img.shape}, expected (N,128,128,3)")
    check(tr_img.dtype == np.uint8,
          f"train_images dtype=uint8",
          f"train_images dtype={tr_img.dtype} (expected uint8)")
    check(te_img.ndim == 4 and te_img.shape[1:] == IMG_SHAPE,
          f"test_images shape OK: {te_img.shape}",
          f"test_images bad shape: {te_img.shape}")
    check(tr_lbl.shape == (len(tr_img),),
          f"train_labels length matches train_images",
          f"train_labels length {tr_lbl.shape} != {len(tr_img)}")
    check(te_lbl.shape == (len(te_img),),
          f"test_labels length matches test_images",
          f"test_labels length mismatch")

    # ── 5. Count checks ───────────────────────────────────────────────────────
    print("\n[ Image counts ]")
    # Relaxed: accept within ±10 of expected (for non-default runs)
    n_tr = len(tr_img)
    n_te = len(te_img)
    expected_tr = n_classes * (n_train_per if isinstance(n_train_per, int) else EXPECTED_TRAIN // EXPECTED_CLASSES)
    expected_te = n_classes * (n_test_per  if isinstance(n_test_per,  int) else EXPECTED_TEST  // EXPECTED_CLASSES)
    check(n_tr == expected_tr,
          f"train count = {n_tr} ({expected_tr} expected)",
          f"train count {n_tr} != expected {expected_tr}")
    check(n_te == expected_te,
          f"test count = {n_te} ({expected_te} expected)",
          f"test count {n_te} != expected {expected_te}")

    # ── 6. Label distribution ─────────────────────────────────────────────────
    print("\n[ Label distribution ]")
    for split_lbl, split_name, n_per in [(tr_lbl, "train", n_train_per), (te_lbl, "test", n_test_per)]:
        counts = {i: int(np.sum(split_lbl == i)) for i in range(n_classes)}
        balanced = all(abs(v - n_per) <= max(1, int(n_per * 0.05)) for v in counts.values())
        detail = ", ".join(f"{class_names[i]}:{counts[i]}" for i in range(n_classes))
        check(balanced,
              f"{split_name} labels balanced: {detail}",
              f"{split_name} labels NOT balanced: {detail}")

    label_vals = set(int(v) for v in tr_lbl)
    check(label_vals == set(range(n_classes)),
          f"All label values in [0,{n_classes-1}]",
          f"Unexpected label values: {label_vals}")

    # ── 7. Pixel statistics ───────────────────────────────────────────────────
    print("\n[ Pixel statistics per class (train) ]")
    print(f"  {'Class':<14} {'Mean':>7} {'Std':>7} {'Min':>6} {'Max':>6}")
    print(f"  {'-'*46}")
    for i, name in enumerate(class_names):
        mask = tr_lbl == i
        sample_imgs = tr_img[mask][:min(200, mask.sum())]  # load a subset to keep RAM low
        arr = np.array(sample_imgs, dtype=np.float32)
        mean_v = arr.mean()
        std_v  = arr.std()
        min_v  = arr.min()
        max_v  = arr.max()
        flag = ""
        if mean_v < PIXEL_MIN_OK or mean_v > PIXEL_MAX_OK:
            flag = " ← suspicious mean"
            warned(f"{name}: mean={mean_v:.1f} out of expected range ({PIXEL_MIN_OK}–{PIXEL_MAX_OK})")
        print(f"  {name:<14} {mean_v:>7.1f} {std_v:>7.1f} {min_v:>6.0f} {max_v:>6.0f}{flag}")

    # ── 8. Duplicate check (fast hash on a sample) ────────────────────────────
    print("\n[ Duplicate check (sample of 500 train images) ]")
    rng      = np.random.default_rng(0)
    idx_samp = rng.choice(len(tr_img), size=min(500, len(tr_img)), replace=False)
    samp     = np.array(tr_img[idx_samp])
    hashes   = [hash(img.tobytes()) for img in samp]
    n_unique = len(set(hashes))
    check(n_unique == len(hashes),
          f"No duplicates in sample ({n_unique}/{len(hashes)} unique)",
          f"Possible duplicates: only {n_unique}/{len(hashes)} unique hashes in sample")

    # ── 9. Build the visual grid ──────────────────────────────────────────────
    print("\n[ Building visual grid ]")
    rng     = np.random.default_rng(7)
    cols    = SAMPLES_PER_ROW
    rows    = n_classes

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 1.6, rows * 1.7),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.35})

    for r, (cname, cid) in enumerate(zip(class_names, range(n_classes))):
        mask     = np.where(tr_lbl == cid)[0]
        chosen   = rng.choice(mask, size=min(cols, len(mask)), replace=False)
        chosen.sort()
        row_imgs = np.array(tr_img[chosen])
        for c in range(cols):
            ax = axes[r][c]
            if c < len(row_imgs):
                ax.imshow(row_imgs[c])
            else:
                ax.set_visible(False)
            ax.axis("off")
            if c == 0:
                ax.set_title(cname, loc="left", fontsize=7, pad=2, fontweight="bold")

    fig.suptitle(
        f"Terrain Dataset — {n_tr} train images, {n_te} test images\n"
        f"7 classes × 128×128 RGB  |  8 random samples per class",
        fontsize=9, y=1.01
    )

    out_path = os.path.join(data_dir, "dataset_grid.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid saved -> {out_path}")

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if _failures:
        print(f"  RESULT: {len(_failures)} check(s) FAILED")
        for f in _failures:
            print(f"    ✗ {f}")
        sys.exit(1)
    else:
        print("  RESULT: All checks PASSED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
