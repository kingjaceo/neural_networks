"""
Load an exported DCGAN generator (NumPy .npz) and produce terrain at game time.

The .npz is produced once on a TF-equipped machine via scripts/export_generator.py
so this module has zero TensorFlow / PyTorch dependency.

Pipeline:
    z ~ N(0, I)  ->  generator_forward(z) -> RGB in [-1, 1], (128, 128, 3)
                 ->  uint8 [0, 255]
                 ->  snap each pixel to the nearest of the 6 palette colors
                 ->  speed_map by mapping each color index to its terrain speed

The forward pass below mirrors build_generator() in scripts/train_dcgan.py —
keep them in sync if the architecture changes.
"""
import os

import numpy as np
from PIL import Image

from scripts.noise import TERRAIN_COLORS

TERRAIN_SPEEDS = [0.0, 0.0, 0.6, 1.0, 0.4, 0.2]
PALETTE = np.array([c for _, c in TERRAIN_COLORS], dtype=np.int32)
SPEEDS  = np.array(TERRAIN_SPEEDS, dtype=np.float32)
LATENT_DIM = 100
IMG_SIZE   = 128

_weights_cache = {}


def _resolve_weights(preset, checkpoint_dir):
    path = os.path.join(checkpoint_dir, preset, "generator.npz")
    return path if os.path.isfile(path) else None


def load_weights(preset, checkpoint_dir="checkpoints"):
    path = _resolve_weights(preset, checkpoint_dir)
    if path is None:
        raise FileNotFoundError(
            f"No exported generator at "
            f"'{checkpoint_dir}/{preset}/generator.npz'. Run "
            f"scripts/export_generator.py --preset {preset} on the training "
            f"machine to produce it."
        )
    if path in _weights_cache:
        return _weights_cache[path]
    with np.load(path) as npz:
        weights = {k: npz[k] for k in npz.files}
    _weights_cache[path] = weights
    print(f"Loaded generator weights: {path}")
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# NumPy ops — mirror the layers in build_generator() of train_dcgan.py
# ─────────────────────────────────────────────────────────────────────────────

def _bn(x, gamma, beta, mean, var, eps):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)


def _upsample2x(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def _conv2d_same_3x3(x, w):
    """Stride-1, padding='same', kernel 3x3, no bias. Channels-last (B,H,W,C)."""
    B, H, W_, IC = x.shape
    OC = w.shape[-1]
    pad = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    # im2col: 9 spatial offsets concatenated along the channel dim, then matmul.
    cols = np.empty((B, H, W_, 9 * IC), dtype=x.dtype)
    k = 0
    for di in range(3):
        for dj in range(3):
            cols[..., k * IC:(k + 1) * IC] = pad[:, di:di + H, dj:dj + W_, :]
            k += 1
    return (cols.reshape(B * H * W_, 9 * IC) @ w.reshape(9 * IC, OC)
            ).reshape(B, H, W_, OC)


def _generator_forward(z, p):
    eps = float(p["bn_epsilon"])

    x = z @ p["dense_w"]
    x = x.reshape(-1, 8, 8, 512)
    x = _bn(x, p["bn0_gamma"], p["bn0_beta"], p["bn0_mean"], p["bn0_var"], eps)
    x = _leaky_relu(x)

    for i in range(3):                                  # 8 -> 16 -> 32 -> 64
        x = _upsample2x(x)
        x = _conv2d_same_3x3(x, p[f"conv{i}_w"])
        j = i + 1
        x = _bn(x, p[f"bn{j}_gamma"], p[f"bn{j}_beta"],
                p[f"bn{j}_mean"], p[f"bn{j}_var"], eps)
        x = _leaky_relu(x)

    x = _upsample2x(x)                                  # 64 -> 128
    x = _conv2d_same_3x3(x, p["conv3_w"])
    return np.tanh(x)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _snap_to_palette(rgb_uint8):
    """rgb_uint8: (H, W, 3) uint8 -> (snapped_rgb uint8, speed_map float32)."""
    diff = rgb_uint8.astype(np.int32)[..., None, :] - PALETTE[None, None, :, :]
    idx = (diff ** 2).sum(axis=-1).argmin(axis=-1)
    return PALETTE[idx].astype(np.uint8), SPEEDS[idx]


def generate_terrain(preset, checkpoint_dir="checkpoints", seed=None):
    """Sample one terrain map from the exported generator.

    Returns
    -------
    image     : PIL.Image, mode 'RGB', size 128x128 — palette-quantized terrain
    speed_map : np.ndarray, shape (128, 128), float32 — per-cell movement speed
    """
    p = load_weights(preset, checkpoint_dir)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((1, LATENT_DIM)).astype(np.float32)
    raw = _generator_forward(z, p)[0]
    rgb = ((raw * 0.5 + 0.5).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    snapped, speed_map = _snap_to_palette(rgb)
    return Image.fromarray(snapped, mode="RGB"), speed_map
