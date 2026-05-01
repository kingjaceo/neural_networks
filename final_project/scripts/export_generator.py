"""
Convert a trained Keras DCGAN generator into a portable .npz so the game can
run inference in pure NumPy without TensorFlow.

Run on a machine with TensorFlow (i.e. the cluster), once per trained preset:
    python scripts/export_generator.py --preset preset_01

Output:
    checkpoints/<preset>/generator.npz

The .npz schema (keys consumed by scripts/gan_terrain.py):
    latent_dim                                    int32
    dense_w                                       (latent_dim, 8*8*512)
    bn{0..3}_{gamma,beta,mean,var}                (channels,)
    conv{0..3}_w                                  (3, 3, in_ch, out_ch)
    bn_epsilon                                    float32

If you change the architecture in train_dcgan.py, update gan_terrain.py too —
the NumPy forward pass mirrors the layer order here.
"""
import argparse
import os

import numpy as np

from train_dcgan import build_generator, find_latest_checkpoint


def main():
    p = argparse.ArgumentParser(description="Export DCGAN generator to NumPy .npz")
    p.add_argument("--preset", required=True)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--latent_dim", type=int, default=100)
    cfg = p.parse_args()

    import tensorflow as tf

    ckpt_dir = os.path.join(cfg.checkpoint_dir, cfg.preset)
    final = os.path.join(ckpt_dir, "G_final.weights.h5")
    if os.path.isfile(final):
        weights_path = final
    else:
        ep, weights_path, _ = find_latest_checkpoint(ckpt_dir)
        if weights_path is None:
            raise FileNotFoundError(f"No G_*.weights.h5 found in {ckpt_dir}")
        print(f"G_final.weights.h5 not found; using epoch {ep}: {weights_path}")

    G = build_generator(cfg.latent_dim)
    G(tf.zeros((1, cfg.latent_dim)), training=False)
    G.load_weights(weights_path)
    print(f"Loaded {weights_path}")

    out = {"latent_dim": np.int32(cfg.latent_dim)}
    bn_i = conv_i = 0
    bn_epsilon = None
    for layer in G.layers:
        cls = layer.__class__.__name__
        if cls == "Dense":
            (kernel,) = layer.get_weights()
            out["dense_w"] = kernel.astype(np.float32)
        elif cls == "Conv2D":
            (kernel,) = layer.get_weights()                  # use_bias=False
            out[f"conv{conv_i}_w"] = kernel.astype(np.float32)
            conv_i += 1
        elif cls == "BatchNormalization":
            gamma, beta, mean, var = layer.get_weights()
            out[f"bn{bn_i}_gamma"] = gamma.astype(np.float32)
            out[f"bn{bn_i}_beta"]  = beta.astype(np.float32)
            out[f"bn{bn_i}_mean"]  = mean.astype(np.float32)
            out[f"bn{bn_i}_var"]   = var.astype(np.float32)
            bn_epsilon = float(layer.epsilon)
            bn_i += 1
    if bn_epsilon is None:
        raise RuntimeError("No BatchNormalization layers found — architecture mismatch?")
    out["bn_epsilon"] = np.float32(bn_epsilon)

    out_path = os.path.join(ckpt_dir, "generator.npz")
    np.savez(out_path, **out)
    nbytes = sum(np.asarray(v).nbytes for v in out.values())
    print(f"Saved {out_path}  ({nbytes / 1e6:.1f} MB, {bn_i} BN layers, {conv_i} Conv2D layers)")


if __name__ == "__main__":
    main()
