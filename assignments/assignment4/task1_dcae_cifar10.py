"""
Task 1: Deep Convolutional Autoencoder (DCAE) on CIFAR-10
Reconstruct images from a 512-dimensional latent representation,
then cluster and visualize embeddings.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DCAE on CIFAR-10")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./outputs")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tsne-samples", type=int, default=10000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DCAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 32x16x16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x8x8
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 128x4x4
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 256x2x2
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(256 * 2 * 2, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 256 * 2 * 2)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        return self.encoder_fc(x)

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 256, 2, 2)
        return self.decoder_conv(x)

    def forward(self, x):
        return self.decode(self.encode(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for images, _ in loader:
        images = images.to(device)
        loss = criterion(model(images), images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for images, _ in loader:
        images = images.to(device)
        total += criterion(model(images), images).item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_reconstruction_metrics(model, loader, device):
    model.eval()
    mse_scores, ssim_scores = [], []
    for images, _ in loader:
        images = images.to(device)
        outputs = model(images)
        imgs_np = images.cpu().numpy()
        outs_np = outputs.cpu().numpy()
        for i in range(imgs_np.shape[0]):
            orig = np.transpose(imgs_np[i], (1, 2, 0))
            recon = np.transpose(outs_np[i], (1, 2, 0))
            mse_scores.append(np.mean((orig - recon) ** 2))
            ssim_scores.append(ssim(orig, recon, data_range=1.0, channel_axis=2))
    return np.mean(mse_scores), np.std(mse_scores), np.mean(ssim_scores), np.std(ssim_scores)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeddings, labels = [], []
    for images, lbls in loader:
        embeddings.append(model.encode(images.to(device)).cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def cluster_and_evaluate(embeddings, labels, n_clusters=10, pca_components=10, seed=42):
    pca = PCA(n_components=pca_components, random_state=seed)
    emb_pca = pca.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km_labels = kmeans.fit_predict(emb_pca)

    ari = adjusted_rand_score(labels, km_labels)
    nmi = normalized_mutual_info_score(labels, km_labels, average_method="arithmetic")

    cm = confusion_matrix(labels, km_labels)
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    acc = cm[row_ind, col_ind].sum() / cm.sum()

    return emb_pca, km_labels, pca, {
        "ari": ari,
        "nmi": nmi,
        "cluster_acc": acc,
        "pca_explained_variance_total": float(pca.explained_variance_ratio_.sum()),
        "pca_explained_variance_per_component": pca.explained_variance_ratio_.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_loss_curve(train_losses, test_losses, path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("DCAE Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_reconstructions(model, test_loader, device, path, n=10):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
    imgs_np = images.cpu().numpy()
    outs_np = outputs.cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(15, 3))
    for i in range(n):
        axes[0, i].imshow(np.transpose(imgs_np[i], (1, 2, 0)))
        axes[0, i].axis("off")
        axes[1, i].imshow(np.transpose(outs_np[i], (1, 2, 0)))
        axes[1, i].axis("off")
    fig.text(0.01, 0.75, "Original", fontsize=12, va="center")
    fig.text(0.01, 0.25, "Reconstructed", fontsize=12, va="center")
    plt.suptitle("DCAE Reconstruction Results on CIFAR-10 Test Set")
    plt.tight_layout(rect=[0.08, 0, 1, 0.95])
    plt.savefig(path, dpi=150)
    plt.close()


def save_tsne(embeddings_2d, true_labels, km_labels, class_names, indices, path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sc1 = axes[0].scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=true_labels[indices], cmap="tab10", s=5, alpha=0.6,
    )
    axes[0].set_title("t-SNE — Ground Truth Labels", fontsize=14)
    axes[0].set_xlabel("t-SNE Dim 1")
    axes[0].set_ylabel("t-SNE Dim 2")
    cb1 = plt.colorbar(sc1, ax=axes[0], ticks=range(10))
    cb1.set_ticklabels(class_names)

    sc2 = axes[1].scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=km_labels[indices], cmap="tab10", s=5, alpha=0.6,
    )
    axes[1].set_title("t-SNE — K-Means Cluster Labels", fontsize=14)
    axes[1].set_xlabel("t-SNE Dim 1")
    axes[1].set_ylabel("t-SNE Dim 2")
    cb2 = plt.colorbar(sc2, ax=axes[1], ticks=range(10))
    cb2.set_ticklabels([f"Cluster {i}" for i in range(10)])

    plt.suptitle("t-SNE of DCAE Embeddings (PCA 10-dim)", fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(root=args.data_dir, train=True,  download=True, transform=tfm)
    test_ds  = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=tfm)
    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds):,}  Test: {len(test_ds):,}  Classes: {class_names}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DCAE(latent_dim=args.latent_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── Training ──────────────────────────────────────────────────────────────
    train_losses, test_losses = [], []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss = evaluate(model, test_loader, criterion, device)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        print(f"Epoch [{epoch:3d}/{args.epochs}]  Train: {tr_loss:.6f}  Test: {te_loss:.6f}")

    save_loss_curve(train_losses, test_losses, os.path.join(args.out_dir, "loss_curve.png"))

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.out_dir, "dcae_cifar10.pt")
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    # ── Reconstruction quality ─────────────────────────────────────────────────
    save_reconstructions(model, test_loader, device,
                         os.path.join(args.out_dir, "reconstructions.png"))

    mse_mean, mse_std, ssim_mean, ssim_std = compute_reconstruction_metrics(
        model, test_loader, device
    )
    print(f"\n=== Reconstruction Metrics ===")
    print(f"MSE:  {mse_mean:.6f} ± {mse_std:.6f}")
    print(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")

    # ── Embeddings ─────────────────────────────────────────────────────────────
    full_ds = torch.utils.data.ConcatDataset([train_ds, test_ds])
    full_loader = DataLoader(full_ds, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    all_embeddings, all_labels = extract_embeddings(model, full_loader, device)
    print(f"\nEmbeddings: {all_embeddings.shape}  Labels: {all_labels.shape}")

    # ── Clustering ─────────────────────────────────────────────────────────────
    emb_pca, km_labels, pca, cluster_metrics = cluster_and_evaluate(
        all_embeddings, all_labels, seed=args.seed
    )

    print(f"\n=== Clustering Metrics ===")
    print(f"ARI:              {cluster_metrics['ari']:.4f}  (chance=0, perfect=1)")
    print(f"NMI:              {cluster_metrics['nmi']:.4f}  (0=no overlap, 1=perfect)")
    print(f"Clustering Acc:   {cluster_metrics['cluster_acc']:.4f}"
          f"  ({cluster_metrics['cluster_acc']*100:.2f}%)")
    print(f"PCA variance:     {cluster_metrics['pca_explained_variance_total']:.4f}")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(emb_pca), min(args.tsne_samples, len(emb_pca)), replace=False)
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30, max_iter=1000)
    emb_2d = tsne.fit_transform(emb_pca[indices])
    print(f"t-SNE shape: {emb_2d.shape}")

    save_tsne(emb_2d, all_labels, km_labels, class_names, indices,
              os.path.join(args.out_dir, "tsne.png"))

    # ── Summary JSON ──────────────────────────────────────────────────────────
    results = {
        "reconstruction": {
            "mse_mean": mse_mean, "mse_std": mse_std,
            "ssim_mean": ssim_mean, "ssim_std": ssim_std,
        },
        "clustering": cluster_metrics,
        "training": {
            "final_train_loss": train_losses[-1],
            "final_test_loss": test_losses[-1],
        },
        "config": vars(args),
    }
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
