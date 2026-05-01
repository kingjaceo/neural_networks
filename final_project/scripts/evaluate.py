#!/usr/bin/env python3
"""
evaluate.py — Evaluation metrics for the trained terrain WGAN-GP generators.

Metrics computed per preset (real held-out test set vs. generated samples):
  1. FID           — Fréchet Inception Distance (InceptionV3 pool_avg features)
  2. PSD slope     — log-log slope of radially-averaged 2D power spectrum
  3. Palette RMSE  — mean per-pixel distance to nearest terrain prototype color
  4. Tortuosity    — mean A* path length / Euclidean distance over random passable pairs
  5. Reachable %   — fraction of passable pixels reachable from a random start (BFS)

Outputs:
  results/eval_metrics.csv        — one row per preset, all metrics
  results/psd_{preset}.png        — power spectrum comparison plot per preset
  Console summary table

Usage (run from neural_networks/final_project/):
  python scripts/evaluate.py
  python scripts/evaluate.py --n_gen 200 --n_nav_imgs 50 --n_pairs 30 --seed 0

Notes:
  - FID with 200 samples is noisy but directionally useful; treat absolute values
    loosely and focus on relative ranking across presets.
  - Palette RMSE is 0 for real images (piecewise-constant palette) and rises when
    the generator outputs smooth gradients instead of discrete terrain colors.
"""
import argparse
import csv
import heapq
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

warnings.filterwarnings("ignore", category=np.ComplexWarning)


# ── Terrain palette (mirrors noise.py TERRAIN_COLORS exactly) ─────────────────
_PALETTE = np.array([
    [ 15,  40,  90],   # 0  deep water      — impassable
    [ 30,  80, 160],   # 1  shallow water   — impassable
    [210, 190, 130],   # 2  sand            — passable
    [ 50, 130,  50],   # 3  plains          — passable
    [110, 100,  90],   # 4  mountains       — passable
    [240, 240, 245],   # 5  mountain tops   — impassable
], dtype=np.float32)

_PASSABLE = np.array([False, False, True, True, True, False])


# ── Generator architecture (must match train_dcgan.py exactly) ─────────────────
def build_generator(latent_dim=100):
    z = layers.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(8 * 8 * 512, use_bias=False)(z)
    x = layers.Reshape((8, 8, 512))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    for f in [256, 128, 64]:
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(3, 3, padding="same", activation="tanh", use_bias=False)(x)
    return models.Model(z, x, name="generator")


def generate_images(G, n, latent_dim, batch_size, seed):
    """Sample n images from G; returns uint8 (N, H, W, 3) in [0, 255]."""
    z = tf.random.normal((n, latent_dim), seed=seed)
    chunks = []
    for i in range(0, n, batch_size):
        chunks.append(G(z[i:i + batch_size], training=False).numpy())
    f32 = np.concatenate(chunks, axis=0)          # float32, [-1, 1]
    return ((f32 * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)


# ── Palette helpers ────────────────────────────────────────────────────────────
def _nearest_terrain(img_uint8):
    """(H, W, 3) uint8 -> (H, W) int8 terrain class index."""
    flat = img_uint8.reshape(-1, 3).astype(np.float32)
    sq_d = np.sum((flat[:, None, :] - _PALETTE[None, :, :]) ** 2, axis=2)
    return sq_d.argmin(axis=1).reshape(img_uint8.shape[:2]).astype(np.int8)


def passable_mask(img_uint8):
    """(H, W, 3) uint8 -> (H, W) bool — True where terrain is traversable."""
    return _PASSABLE[_nearest_terrain(img_uint8)]


def palette_rmse(imgs_uint8):
    """Mean per-pixel L2 distance to nearest prototype color (lower = better)."""
    flat = imgs_uint8.reshape(-1, 3).astype(np.float32)
    dists = np.sqrt(np.sum((flat[:, None, :] - _PALETTE[None, :, :]) ** 2, axis=2))
    return float(dists.min(axis=1).mean())


# ── Power spectrum ─────────────────────────────────────────────────────────────
def _psd_slope_one(img_uint8):
    """Log-log slope of radially-averaged 2D power spectrum for one image."""
    gray = img_uint8.mean(axis=-1).astype(np.float64)
    H, W = gray.shape
    cy, cx = H // 2, W // 2
    f2 = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(f2) ** 2 + 1e-10
    y_idx, x_idx = np.indices((H, W))
    r_map = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
    r_max = min(cy, cx)
    radii = np.arange(1, r_max)
    ring_means = np.array([power[r_map == ri].mean() for ri in radii])
    valid = ring_means > 0
    if valid.sum() < 4:
        return np.nan
    return float(np.polyfit(np.log(radii[valid]), np.log(ring_means[valid]), 1)[0])


def psd_stats(imgs_uint8):
    """Returns (mean_slope, std_slope, radii_array, mean_power_curve)."""
    H, W = imgs_uint8.shape[1:3]
    cy, cx = H // 2, W // 2
    r_max = min(cy, cx)
    radii = np.arange(1, r_max)

    slopes, curves = [], []
    for img in imgs_uint8:
        gray = img.mean(axis=-1).astype(np.float64)
        f2 = np.fft.fftshift(np.fft.fft2(gray))
        power = np.abs(f2) ** 2 + 1e-10
        y_idx, x_idx = np.indices((H, W))
        r_map = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
        ring_means = np.array([power[r_map == ri].mean() for ri in radii])
        curves.append(ring_means)
        s = _psd_slope_one(img)
        if not np.isnan(s):
            slopes.append(s)

    mean_slope = float(np.mean(slopes)) if slopes else float("nan")
    std_slope  = float(np.std(slopes))  if slopes else float("nan")
    mean_curve = np.mean(curves, axis=0)
    return mean_slope, std_slope, radii, mean_curve


def save_psd_plot(radii, curve_real, curve_gen, slope_real, slope_gen, preset, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog(radii, curve_real, color="steelblue", label=f"Real (slope={slope_real:.2f})")
    ax.loglog(radii, curve_gen,  color="tomato", ls="--", label=f"Gen  (slope={slope_gen:.2f})")
    ax.set_xlabel("Spatial frequency (px⁻¹)")
    ax.set_ylabel("Mean power")
    ax.set_title(f"Radial power spectrum — {preset}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── FID ────────────────────────────────────────────────────────────────────────
def inception_features(imgs_uint8, model, batch_size):
    """(N, H, W, 3) uint8 -> (N, 2048) float32 InceptionV3 pool features."""
    feats = []
    for i in range(0, len(imgs_uint8), batch_size):
        batch = tf.image.resize(imgs_uint8[i:i + batch_size].astype(np.float32), (299, 299))
        feats.append(model(preprocess_input(batch), training=False).numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(real_feats, gen_feats, eps=1e-6):
    """Fréchet Inception Distance between two feature sets."""
    d = real_feats.shape[1]
    mu_r = real_feats.mean(0)
    mu_g = gen_feats.mean(0)
    sig_r = np.cov(real_feats, rowvar=False) + eps * np.eye(d)
    sig_g = np.cov(gen_feats,  rowvar=False) + eps * np.eye(d)
    diff = mu_r - mu_g
    covmean, _ = sqrtm(sig_r @ sig_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_g - 2 * covmean))


# ── A* path tortuosity ─────────────────────────────────────────────────────────
def _astar(passable, start, goal):
    """4-connected A* on bool (H, W) grid; returns step count or None."""
    H, W = passable.shape
    gy, gx = goal

    def h(y, x):
        return abs(y - gy) + abs(x - gx)

    heap = [(h(*start), 0, start[0], start[1])]
    g_best = {start: 0}
    closed = set()
    while heap:
        _, g, y, x = heapq.heappop(heap)
        if (y, x) == goal:
            return g
        if (y, x) in closed:
            continue
        closed.add((y, x))
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and passable[ny, nx] and (ny, nx) not in closed:
                ng = g + 1
                if ng < g_best.get((ny, nx), 1 << 30):
                    g_best[(ny, nx)] = ng
                    heapq.heappush(heap, (ng + h(ny, nx), ng, ny, nx))
    return None


def _bfs_reachable(passable, start):
    """Fraction of passable pixels reachable from start via 4-connected BFS."""
    total = int(passable.sum())
    if total == 0:
        return 0.0
    H, W = passable.shape
    visited = np.zeros((H, W), bool)
    visited[start] = True
    q = [start]
    head = 0
    while head < len(q):
        y, x = q[head]; head += 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and passable[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    return float(visited.sum()) / total


def nav_metrics(imgs_uint8, rng, n_pairs=30, n_reach_starts=5, min_dist=10):
    """
    Returns (mean_tortuosity, mean_reachable_fraction) over the batch.

    tortuosity: A* path steps / Euclidean distance, averaged over random passable
                pairs at least min_dist pixels apart.
    reachable:  fraction of passable pixels reachable from n_reach_starts random
                starts, averaged over all starts and images.
    """
    tort_vals, reach_vals = [], []
    for img in imgs_uint8:
        pm = passable_mask(img)
        pyx = np.argwhere(pm)
        if len(pyx) < 10:
            reach_vals.append(0.0)
            continue
        for _ in range(n_reach_starts):
            start = tuple(pyx[rng.integers(len(pyx))])
            reach_vals.append(_bfs_reachable(pm, start))
        for _ in range(n_pairs):
            i1, i2 = rng.choice(len(pyx), 2, replace=False)
            s, g = tuple(pyx[i1]), tuple(pyx[i2])
            euc = float(np.hypot(s[0] - g[0], s[1] - g[1]))
            if euc < min_dist:
                continue
            pl = _astar(pm, s, g)
            if pl is not None:
                tort_vals.append(pl / euc)
    mean_t = float(np.mean(tort_vals))  if tort_vals  else float("nan")
    mean_r = float(np.mean(reach_vals)) if reach_vals else float("nan")
    return mean_t, mean_r


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Evaluate terrain WGAN-GP generators.")
    ap.add_argument("--base_dir",   default=".",       help="Project root (contains dataset/, checkpoints/)")
    ap.add_argument("--n_gen",      type=int, default=200, help="Generated images per preset")
    ap.add_argument("--n_nav_imgs", type=int, default=50,  help="Images per preset used for nav metrics")
    ap.add_argument("--n_pairs",    type=int, default=30,  help="A* pairs per image for tortuosity")
    ap.add_argument("--latent_dim", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"\nTF {tf.__version__}  |  GPUs: {tf.config.list_physical_devices('GPU')}")

    # load metadata + test set
    with open(os.path.join(args.base_dir, "dataset", "metadata.json")) as fh:
        meta = json.load(fh)
    class_names = meta["class_names"]   # ["preset_01", ..., "preset_07"]

    te_img = np.load(os.path.join(args.base_dir, "dataset", "test_images.npy"), mmap_mode="r")
    te_lbl = np.load(os.path.join(args.base_dir, "dataset", "test_labels.npy"), mmap_mode="r")
    print(f"Test set: {te_img.shape}  dtype={te_img.dtype}")

    # build InceptionV3 once for FID (weights downloaded on first run)
    print("Loading InceptionV3 feature extractor...")
    inception = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))
    print("  done.")

    os.makedirs(os.path.join(args.base_dir, "results"), exist_ok=True)

    CSV_FIELDS = [
        "preset", "n_real", "n_gen",
        "fid",
        "psd_slope_real", "psd_slope_real_std",
        "psd_slope_gen",  "psd_slope_gen_std", "psd_slope_delta",
        "palette_rmse_real", "palette_rmse_gen",
        "tortuosity_real", "reachable_real",
        "tortuosity_gen",  "reachable_gen",
    ]
    rows = []

    for pidx, preset in enumerate(class_names):
        ckpt = os.path.join(args.base_dir, "checkpoints", preset, "G_final.weights.h5")
        if not os.path.exists(ckpt):
            print(f"\n[{preset}] G_final.weights.h5 not found — skipping.")
            continue

        print(f"\n{'='*66}")
        print(f"  {preset}  ({pidx + 1}/{len(class_names)})")
        print(f"{'='*66}")

        real = np.array(te_img[te_lbl == pidx])   # uint8 (200, 128, 128, 3)
        print(f"  Real test images : {len(real)}")

        # generate samples
        print(f"  Generating {args.n_gen} samples...")
        G = build_generator(args.latent_dim)
        G(tf.zeros((1, args.latent_dim)), training=False)   # build graph before loading weights
        G.load_weights(ckpt)
        gen = generate_images(G, args.n_gen, args.latent_dim, args.batch_size, args.seed)
        del G   # release model weights before next preset
        print(f"  Generated shape  : {gen.shape}")

        n = min(len(real), len(gen))

        # 1. FID
        print("  [1/4] FID...")
        fr = inception_features(real[:n], inception, args.batch_size)
        fg = inception_features(gen[:n],  inception, args.batch_size)
        fid = compute_fid(fr, fg)
        print(f"        FID = {fid:.1f}")

        # 2. Power spectrum
        print("  [2/4] Power spectrum...")
        sl_r, std_r, radii, curve_r = psd_stats(real[:n])
        sl_g, std_g, _,     curve_g = psd_stats(gen[:n])
        delta = sl_g - sl_r
        print(f"        real slope={sl_r:.3f}±{std_r:.3f}  gen slope={sl_g:.3f}±{std_g:.3f}  Δ={delta:+.3f}")
        psd_out = os.path.join(args.base_dir, "results", f"psd_{preset}.png")
        save_psd_plot(radii, curve_r, curve_g, sl_r, sl_g, preset, psd_out)

        # 3. Palette RMSE
        print("  [3/4] Palette RMSE...")
        pal_r = palette_rmse(real[:n])
        pal_g = palette_rmse(gen[:n])
        print(f"        real={pal_r:.1f}  gen={pal_g:.1f}")

        # 4. Navigability
        nav_n = min(args.n_nav_imgs, len(real), len(gen))
        print(f"  [4/4] Navigability (imgs={nav_n}, pairs/img={args.n_pairs})...")
        tort_r, reach_r = nav_metrics(real[rng.choice(len(real), nav_n, replace=False)], rng, args.n_pairs)
        tort_g, reach_g = nav_metrics(gen[rng.choice(len(gen),   nav_n, replace=False)], rng, args.n_pairs)
        print(f"        Tortuosity  real={tort_r:.3f}  gen={tort_g:.3f}")
        print(f"        Reachable   real={reach_r:.3f}  gen={reach_g:.3f}")

        def _f(v): return round(v, 4) if not np.isnan(v) else "nan"

        rows.append({
            "preset": preset, "n_real": len(real), "n_gen": len(gen),
            "fid": round(fid, 1),
            "psd_slope_real": round(sl_r, 4), "psd_slope_real_std": round(std_r, 4),
            "psd_slope_gen":  round(sl_g, 4), "psd_slope_gen_std":  round(std_g, 4),
            "psd_slope_delta": round(delta, 4),
            "palette_rmse_real": round(pal_r, 2), "palette_rmse_gen": round(pal_g, 2),
            "tortuosity_real": _f(tort_r), "reachable_real": _f(reach_r),
            "tortuosity_gen":  _f(tort_g), "reachable_gen":  _f(reach_g),
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*92}")
    hdr = "{:<12} {:>8} {:>10} {:>10} {:>13} {:>13} {:>13}"
    print(hdr.format("Preset", "FID↓", "PSD_real", "PSD_gen", "PalRMSE_gen↓",
                     "Tortuosity_g", "Reachable_g"))
    print("-" * 92)
    for r in rows:
        print(hdr.format(
            r["preset"], r["fid"],
            r["psd_slope_real"], r["psd_slope_gen"],
            r["palette_rmse_gen"],
            r["tortuosity_gen"], r["reachable_gen"],
        ))

    out_csv = os.path.join(args.base_dir, "results", "eval_metrics.csv")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOutputs saved:")
    print(f"  {out_csv}")
    print(f"  results/psd_{{preset}}.png  (one per preset)")


if __name__ == "__main__":
    main()
