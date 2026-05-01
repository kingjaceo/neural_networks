"""
Baseline unconditional WGAN-GP (DCGAN-style architecture) for terrain generation.

Loss: Wasserstein with gradient penalty (lambda=10), n_disc_steps D updates per G update.
Adam: beta_1=0.5, beta_2=0.9 (WGAN-GP defaults).

Architecture (128x128 RGB):
  Generator:     z(100) -> Dense -> Reshape(8,8,512) -> 4x [UpSample + Conv2D + BN] -> 128x128x3
  Discriminator: 128x128x3 -> 4x [Conv2D stride 2 + LeakyReLU] -> Dense(1)  (no normalization;
                 BatchNorm in D is incompatible with the per-sample gradient penalty)

Usage (local sanity check):
  python train_dcgan.py --epochs 2 --batch_size 16

Usage (full run via SLURM):
  sbatch jobs/train_dcgan.sh

Outputs (all under --checkpoint_dir / --sample_dir / --log_dir):
  checkpoints/<preset>/G_epoch_NNNN.weights.h5
  checkpoints/<preset>/D_epoch_NNNN.weights.h5
  samples/<preset>/epoch_NNNN.png
  logs/dcgan_<preset>.csv
"""
import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DEFAULTS = dict(
    dataset_dir         = "dataset",
    preset              = "preset_01",
    epochs              = 100,   # target epoch (not additional epochs — so resume + epochs=100 trains to 100)
    batch_size          = 64,
    latent_dim          = 100,
    lr                  = 2e-4,
    beta1               = 0.5,
    n_disc_steps        = 5,    # D updates per G update (WGAN-GP standard)
    checkpoint_dir      = "checkpoints",
    sample_dir          = "samples",
    log_dir             = "logs",
    sample_interval     = 5,
    checkpoint_interval = 10,
    resume              = False, # load latest checkpoint and continue
)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir):
    """Return (epoch, G_path, D_path) for the highest saved epoch, or (0, None, None)."""
    import glob
    g_files = glob.glob(os.path.join(ckpt_dir, "G_epoch_*.weights.h5"))
    if not g_files:
        return 0, None, None
    # extract epoch numbers from filenames
    def epoch_of(path):
        return int(os.path.basename(path).split("_")[2].split(".")[0])
    latest = max(g_files, key=epoch_of)
    epoch  = epoch_of(latest)
    d_path = os.path.join(ckpt_dir, f"D_epoch_{epoch:04d}.weights.h5")
    return epoch, latest, d_path


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_generator(latent_dim):
    """z(latent_dim,) -> 128x128x3 float32 in [-1, 1]."""
    z = layers.Input(shape=(latent_dim,), name="z")

    x = layers.Dense(8 * 8 * 512, use_bias=False)(z)
    x = layers.Reshape((8, 8, 512))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    for filters in [256, 128, 64]:               # 8->16->32->64
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    x = layers.UpSampling2D(2)(x)               # 64->128, output
    x = layers.Conv2D(3, 3, padding="same",
        activation="tanh", use_bias=False)(x)

    return models.Model(z, x, name="generator")


def build_discriminator():
    """128x128x3 float32 in [-1, 1] -> unbounded scalar (Wasserstein critic)."""
    img = layers.Input(shape=(128, 128, 3), name="image")
    x = img
    for filters in [64, 128, 256, 512]:          # 128->64->32->16->8
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return models.Model(img, x, name="discriminator")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def make_train_fn(G, D, g_opt, d_opt, latent_dim, batch_size, n_disc_steps):
    lam = 10.0  # gradient penalty coefficient

    @tf.function
    def gradient_penalty(real, fake):
        alpha = tf.random.uniform((batch_size, 1, 1, 1))
        interp = alpha * real + (1.0 - alpha) * fake
        with tf.GradientTape() as tape:
            tape.watch(interp)
            d_interp = D(interp, training=True)
        grads = tape.gradient(d_interp, interp)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-8)
        return tf.reduce_mean((norm - 1.0) ** 2)

    @tf.function
    def d_step(real_imgs):
        z = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as tape:
            fake = G(z, training=False)
            loss = (tf.reduce_mean(D(fake, training=True))
                    - tf.reduce_mean(D(real_imgs, training=True))
                    + lam * gradient_penalty(real_imgs, fake))
        d_opt.apply_gradients(zip(tape.gradient(loss, D.trainable_variables),
                                  D.trainable_variables))
        return loss

    @tf.function
    def g_step():
        z = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as tape:
            fake = G(z, training=True)
            loss = -tf.reduce_mean(D(fake, training=False))
        g_opt.apply_gradients(zip(tape.gradient(loss, G.trainable_variables),
                                  G.trainable_variables))
        return loss

    def train_step(real_imgs):
        dl = None
        for _ in range(n_disc_steps):
            dl = d_step(real_imgs)
        gl = g_step()
        return gl, dl

    return train_step


def save_sample_grid(G, fixed_z, path, epoch):
    imgs = G(fixed_z, training=False).numpy()
    imgs = (imgs * 0.5 + 0.5).clip(0, 1)       # [-1,1] -> [0,1]
    n = int(len(imgs) ** 0.5)
    fig, axes = plt.subplots(n, n, figsize=(n * 1.5, n * 1.5))
    for img, ax in zip(imgs, axes.ravel()):
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(f"Epoch {epoch}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Baseline unconditional DCGAN — terrain")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    cfg = p.parse_args()

    print(f"TF {tf.__version__}  |  GPUs: {tf.config.list_physical_devices('GPU')}")

    # ── Load & filter data ────────────────────────────────────────────────────
    with open(os.path.join(cfg.dataset_dir, "metadata.json")) as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    if cfg.preset not in class_names:
        raise ValueError(f"--preset '{cfg.preset}' not found. Available: {class_names}")
    preset_idx = class_names.index(cfg.preset)

    images = np.load(os.path.join(cfg.dataset_dir, "train_images.npy"))
    labels = np.load(os.path.join(cfg.dataset_dir, "train_labels.npy"))
    images = images[labels == preset_idx]
    print(f"Preset '{cfg.preset}' ({preset_idx}): {len(images)} training images")

    images = images.astype(np.float32) / 127.5 - 1.0   # uint8 [0,255] -> float32 [-1,1]

    dataset = (
        tf.data.Dataset.from_tensor_slices(images)
        .shuffle(len(images), seed=42)
        .batch(cfg.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Build models ──────────────────────────────────────────────────────────
    G = build_generator(cfg.latent_dim)
    D = build_discriminator()
    G.summary()
    D.summary()

    g_opt = tf.keras.optimizers.Adam(cfg.lr, beta_1=cfg.beta1, beta_2=0.999)
    d_opt = tf.keras.optimizers.Adam(cfg.lr, beta_1=cfg.beta1, beta_2=0.999)

    train_step = make_train_fn(G, D, g_opt, d_opt,
                               cfg.latent_dim, cfg.batch_size, cfg.n_disc_steps)

    # ── Output dirs ───────────────────────────────────────────────────────────
    ckpt_dir   = os.path.join(cfg.checkpoint_dir, cfg.preset)
    sample_dir = os.path.join(cfg.sample_dir,     cfg.preset)
    os.makedirs(ckpt_dir,    exist_ok=True)
    os.makedirs(sample_dir,  exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if cfg.resume:
        ckpt_epoch, g_path, d_path = find_latest_checkpoint(ckpt_dir)
        if ckpt_epoch == 0:
            print("--resume set but no checkpoints found; starting from scratch.")
        else:
            # Build the graph by running one dummy forward pass before loading weights.
            # Note: optimizer state (Adam m/v accumulators) is not restored — it
            # re-warms over the first ~10 batches, which is fine in practice.
            dummy = tf.zeros((1, 128, 128, 3))
            D(dummy, training=False)
            G(tf.zeros((1, cfg.latent_dim)), training=False)
            G.load_weights(g_path)
            D.load_weights(d_path)
            start_epoch = ckpt_epoch + 1
            print(f"Resumed from epoch {ckpt_epoch} ({g_path})")

    if start_epoch > cfg.epochs:
        print(f"Already at epoch {start_epoch - 1} >= --epochs {cfg.epochs}. Nothing to do.")
        return

    fixed_z  = tf.random.normal((64, cfg.latent_dim), seed=0)
    log_path = os.path.join(cfg.log_dir, f"dcgan_{cfg.preset}.csv")
    log_mode = "a" if cfg.resume and start_epoch > 1 else "w"
    with open(log_path, log_mode, newline="") as f:
        if log_mode == "w":
            csv.writer(f).writerow(["epoch", "g_loss", "d_loss", "epoch_time_s"])

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        g_losses, d_losses = [], []

        for batch in dataset:
            gl, dl = train_step(batch)
            g_losses.append(float(gl))
            d_losses.append(float(dl))

        elapsed  = time.time() - t0
        mean_g   = float(np.mean(g_losses))
        mean_d   = float(np.mean(d_losses))

        print(f"Epoch {epoch:4d}/{cfg.epochs}  G {mean_g:.4f}  D {mean_d:.4f}  {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mean_g, mean_d, elapsed])

        if epoch % cfg.sample_interval == 0:
            save_sample_grid(G, fixed_z,
                             os.path.join(sample_dir, f"epoch_{epoch:04d}.png"), epoch)

        if epoch % cfg.checkpoint_interval == 0:
            G.save_weights(os.path.join(ckpt_dir, f"G_epoch_{epoch:04d}.weights.h5"))
            D.save_weights(os.path.join(ckpt_dir, f"D_epoch_{epoch:04d}.weights.h5"))

    G.save_weights(os.path.join(ckpt_dir, "G_final.weights.h5"))
    D.save_weights(os.path.join(ckpt_dir, "D_final.weights.h5"))
    save_sample_grid(G, fixed_z, os.path.join(sample_dir, "final.png"), cfg.epochs)
    print(f"\nDone.  Checkpoints -> {ckpt_dir}  Samples -> {sample_dir}  Log -> {log_path}")


if __name__ == "__main__":
    main()
