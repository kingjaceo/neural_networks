"""
Conditional WGAN-GP (DCGAN-style architecture) for terrain generation.

Loss: Wasserstein with gradient penalty (lambda=10), n_disc_steps D updates per G update.
Adam: beta_1=0.5, beta_2=0.9 (WGAN-GP defaults).

Architecture (128x128 RGB, 7 terrain types):
  Generator:     z(100) + label_onehot(7) -> Dense -> Reshape(8,8,512) -> 4x [UpSample + Conv2D + BN] -> 128x128x3
  Discriminator: 128x128x3 -> 4x [Conv2D stride 2 + LeakyReLU] -> Flatten + label_onehot -> Dense(1)
                 (no normalization in D: BatchNorm is incompatible with per-sample gradient penalty)

Usage (local sanity check):
  python train_cdcgan.py --epochs 2 --batch_size 16

Usage (full run via SLURM):
  sbatch jobs/train_cdcgan.sh

Outputs:
  checkpoints/cdcgan/G_epoch_NNNN.weights.h5
  checkpoints/cdcgan/D_epoch_NNNN.weights.h5
  samples/cdcgan/epoch_NNNN.png   (grid: one row per terrain type, 4 columns)
  logs/cdcgan.csv
"""
import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DEFAULTS = dict(
    dataset_dir         = "dataset",
    epochs              = 100,
    batch_size          = 64,
    latent_dim          = 100,
    lr                  = 2e-4,
    beta1               = 0.5,
    n_disc_steps        = 5,
    checkpoint_dir      = "checkpoints",
    sample_dir          = "samples",
    log_dir             = "logs",
    sample_interval     = 5,
    checkpoint_interval = 10,
    resume              = False,
)

SUBDIR = "cdcgan"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir):
    import glob as _glob
    g_files = _glob.glob(os.path.join(ckpt_dir, "G_epoch_*.weights.h5"))
    if not g_files:
        return 0, None, None
    def epoch_of(path):
        return int(os.path.basename(path).split("_")[2].split(".")[0])
    latest = max(g_files, key=epoch_of)
    epoch  = epoch_of(latest)
    d_path = os.path.join(ckpt_dir, f"D_epoch_{epoch:04d}.weights.h5")
    return epoch, latest, d_path


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_generator(latent_dim, num_classes):
    """[z(latent_dim), label_onehot(num_classes)] -> 128x128x3 float32 in [-1, 1]."""
    z     = layers.Input(shape=(latent_dim,),  name="z")
    label = layers.Input(shape=(num_classes,), name="label")
    x = layers.Concatenate()([z, label])        # (latent_dim + num_classes,)

    x = layers.Dense(8 * 8 * 512, use_bias=False)(x)
    x = layers.Reshape((8, 8, 512))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    for filters in [256, 128, 64]:              # 8->16->32->64
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    x = layers.UpSampling2D(2)(x)              # 64->128, output
    x = layers.Conv2D(3, 3, padding="same",
        activation="tanh", use_bias=False)(x)

    return models.Model([z, label], x, name="generator")


def build_discriminator(num_classes):
    """[128x128x3 float32, label_onehot(num_classes)] -> unbounded scalar (Wasserstein critic)."""
    img   = layers.Input(shape=(128, 128, 3), name="image")
    label = layers.Input(shape=(num_classes,), name="label")

    x = img
    for filters in [64, 128, 256, 512]:         # 128->64->32->16->8
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, label])        # inject conditioning after conv stack
    x = layers.Dense(1)(x)

    return models.Model([img, label], x, name="discriminator")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def make_train_fn(G, D, g_opt, d_opt, latent_dim, num_classes, batch_size, n_disc_steps):
    lam = 10.0  # gradient penalty coefficient

    @tf.function
    def gradient_penalty(real, fake, labels):
        alpha = tf.random.uniform((batch_size, 1, 1, 1))
        interp = alpha * real + (1.0 - alpha) * fake
        with tf.GradientTape() as tape:
            tape.watch(interp)
            d_interp = D([interp, labels], training=True)
        grads = tape.gradient(d_interp, interp)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-8)
        return tf.reduce_mean((norm - 1.0) ** 2)

    @tf.function
    def d_step(real_imgs, labels):
        z = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as tape:
            fake = G([z, labels], training=False)
            loss = (tf.reduce_mean(D([fake,      labels], training=True))
                    - tf.reduce_mean(D([real_imgs, labels], training=True))
                    + lam * gradient_penalty(real_imgs, fake, labels))
        d_opt.apply_gradients(zip(tape.gradient(loss, D.trainable_variables),
                                  D.trainable_variables))
        return loss

    @tf.function
    def g_step(labels):
        z = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as tape:
            fake = G([z, labels], training=True)
            loss = -tf.reduce_mean(D([fake, labels], training=False))
        g_opt.apply_gradients(zip(tape.gradient(loss, G.trainable_variables),
                                  G.trainable_variables))
        return loss

    def train_step(real_imgs, labels):
        dl = None
        for _ in range(n_disc_steps):
            dl = d_step(real_imgs, labels)
        gl = g_step(labels)
        return gl, dl

    return train_step


def save_sample_grid(G, fixed_z, fixed_labels, class_names, path, epoch):
    """Grid: one row per terrain type, 4 columns of independent samples."""
    imgs = G([fixed_z, fixed_labels], training=False).numpy()
    imgs = (imgs * 0.5 + 0.5).clip(0, 1)
    num_classes   = len(class_names)
    n_cols        = len(imgs) // num_classes
    fig, axes = plt.subplots(num_classes, n_cols,
                             figsize=(n_cols * 1.5, num_classes * 1.5))
    for r in range(num_classes):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(imgs[r * n_cols + c])
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(class_names[r], fontsize=7, rotation=45, labelpad=30)
    fig.suptitle(f"Epoch {epoch}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Conditional DCGAN — terrain (all presets)")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    cfg = p.parse_args()

    print(f"TF {tf.__version__}  |  GPUs: {tf.config.list_physical_devices('GPU')}")

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(os.path.join(cfg.dataset_dir, "metadata.json")) as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    num_classes = len(class_names)

    images = np.load(os.path.join(cfg.dataset_dir, "train_images.npy"))
    labels = np.load(os.path.join(cfg.dataset_dir, "train_labels.npy"))
    print(f"Loaded {len(images)} training images across {num_classes} presets")

    images        = images.astype(np.float32) / 127.5 - 1.0
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes).astype(np.float32)

    dataset = (
        tf.data.Dataset.from_tensor_slices((images, labels_onehot))
        .shuffle(len(images), seed=42)
        .batch(cfg.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Build models ──────────────────────────────────────────────────────────
    G = build_generator(cfg.latent_dim, num_classes)
    D = build_discriminator(num_classes)
    G.summary()
    D.summary()

    g_opt = tf.keras.optimizers.Adam(cfg.lr, beta_1=cfg.beta1, beta_2=0.999)
    d_opt = tf.keras.optimizers.Adam(cfg.lr, beta_1=cfg.beta1, beta_2=0.999)

    train_step = make_train_fn(G, D, g_opt, d_opt,
                               cfg.latent_dim, num_classes,
                               cfg.batch_size, cfg.n_disc_steps)

    # ── Output dirs ───────────────────────────────────────────────────────────
    ckpt_dir   = os.path.join(cfg.checkpoint_dir, SUBDIR)
    sample_dir = os.path.join(cfg.sample_dir,     SUBDIR)
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
            dummy_img   = tf.zeros((1, 128, 128, 3))
            dummy_z     = tf.zeros((1, cfg.latent_dim))
            dummy_label = tf.zeros((1, num_classes))
            D([dummy_img, dummy_label], training=False)
            G([dummy_z,   dummy_label], training=False)
            G.load_weights(g_path)
            D.load_weights(d_path)
            start_epoch = ckpt_epoch + 1
            print(f"Resumed from epoch {ckpt_epoch} ({g_path})")

    if start_epoch > cfg.epochs:
        print(f"Already at epoch {start_epoch - 1} >= --epochs {cfg.epochs}. Nothing to do.")
        return

    # Fixed noise + one-hot labels for consistent sample grids (4 samples per class)
    n_cols       = 4
    fixed_z      = tf.random.normal((num_classes * n_cols, cfg.latent_dim), seed=0)
    fixed_labels = tf.constant(
        np.repeat(np.eye(num_classes, dtype=np.float32), n_cols, axis=0)
    )

    log_path = os.path.join(cfg.log_dir, "cdcgan.csv")
    log_mode = "a" if cfg.resume and start_epoch > 1 else "w"
    with open(log_path, log_mode, newline="") as f:
        if log_mode == "w":
            csv.writer(f).writerow(["epoch", "g_loss", "d_loss", "epoch_time_s"])

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        g_losses, d_losses = [], []

        for batch_imgs, batch_labels in dataset:
            gl, dl = train_step(batch_imgs, batch_labels)
            g_losses.append(float(gl))
            d_losses.append(float(dl))

        elapsed = time.time() - t0
        mean_g  = float(np.mean(g_losses))
        mean_d  = float(np.mean(d_losses))

        print(f"Epoch {epoch:4d}/{cfg.epochs}  G {mean_g:.4f}  D {mean_d:.4f}  {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mean_g, mean_d, elapsed])

        if epoch % cfg.sample_interval == 0:
            save_sample_grid(G, fixed_z, fixed_labels, class_names,
                             os.path.join(sample_dir, f"epoch_{epoch:04d}.png"), epoch)

        if epoch % cfg.checkpoint_interval == 0:
            G.save_weights(os.path.join(ckpt_dir, f"G_epoch_{epoch:04d}.weights.h5"))
            D.save_weights(os.path.join(ckpt_dir, f"D_epoch_{epoch:04d}.weights.h5"))

    G.save_weights(os.path.join(ckpt_dir, "G_final.weights.h5"))
    D.save_weights(os.path.join(ckpt_dir, "D_final.weights.h5"))
    save_sample_grid(G, fixed_z, fixed_labels, class_names,
                     os.path.join(sample_dir, "final.png"), cfg.epochs)
    print(f"\nDone.  Checkpoints -> {ckpt_dir}  Samples -> {sample_dir}  Log -> {log_path}")


if __name__ == "__main__":
    main()
