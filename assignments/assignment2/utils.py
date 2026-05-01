"""Shared data loading, metrics computation, and plotting utilities."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — required on HPC nodes
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Tiny ImageNet validation dataset
# ---------------------------------------------------------------------------

class _TinyImageNetVal(Dataset):
    """Tiny ImageNet validation set (flat image layout).

    The official val split stores all images in val/images/ and maps filenames
    to WordNet IDs via val/val_annotations.txt.  This dataset parses that file
    and uses the class_to_idx mapping from the training ImageFolder so that
    integer labels are consistent across splits.
    """

    def __init__(self, val_dir: str, class_to_idx: dict, transform=None):
        val_dir = Path(val_dir)
        self.img_dir   = val_dir / 'images'
        self.transform = transform

        self.samples = []
        with open(val_dir / 'val_annotations.txt') as f:
            for line in f:
                parts = line.strip().split('\t')
                fname, wnid = parts[0], parts[1]
                if wnid in class_to_idx:
                    self.samples.append((fname, class_to_idx[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, label = self.samples[idx]
        img = Image.open(self.img_dir / fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

# Tiny ImageNet-200 (images originally 64x64, resized to 32x32)
_TINY_MEAN = (0.4802, 0.4481, 0.3975)
_TINY_STD  = (0.2770, 0.2691, 0.2821)

# CIFAR-100
_C100_MEAN = (0.5071, 0.4867, 0.4408)
_C100_STD  = (0.2675, 0.2565, 0.2761)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def get_tiny_imagenet_loaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    train_frac: float = 0.8,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for Tiny ImageNet-200.

    The official train split is divided 80/20 per-class into train and
    validation subsets using a fixed random seed so the split is reproducible.
    The official val split (flat layout) is used as the held-out test set.

    Args:
        data_dir   : Root directory (must contain train/ and val/).
        batch_size : Mini-batch size.
        num_workers: DataLoader worker count.
        train_frac : Fraction of per-class training samples kept for training.
    """
    train_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_TINY_MEAN, _TINY_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(_TINY_MEAN, _TINY_STD),
    ])

    # Two ImageFolder instances pointing to the same files but with different
    # transforms: train augmentations for train split, eval transforms for val.
    train_root     = os.path.join(data_dir, 'train')
    full_aug       = datasets.ImageFolder(train_root, transform=train_tf)
    full_eval      = datasets.ImageFolder(train_root, transform=eval_tf)

    # Per-class stratified 80/20 split
    class_to_indices: dict = defaultdict(list)
    for idx, (_, label) in enumerate(full_aug.samples):
        class_to_indices[label].append(idx)

    rng = np.random.default_rng(seed=0)
    train_idx, val_idx = [], []
    for label in sorted(class_to_indices):
        indices = rng.permutation(class_to_indices[label]).tolist()
        cut = int(len(indices) * train_frac)
        train_idx.extend(indices[:cut])
        val_idx.extend(indices[cut:])

    train_dataset = Subset(full_aug,  train_idx)
    val_dataset   = Subset(full_eval, val_idx)

    # Official val split used as test set
    test_dataset = _TinyImageNetVal(
        os.path.join(data_dir, 'val'),
        full_aug.class_to_idx,
        transform=eval_tf,
    )

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_dataset, shuffle=True,  **kw),
        DataLoader(val_dataset,   shuffle=False, **kw),
        DataLoader(test_dataset,  shuffle=False, **kw),
    )


def get_cifar100_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for CIFAR-100.

    The official train split (50 000 images) is divided 80/20 per-class.
    The official test split (10 000 images) is the held-out test set.
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_C100_MEAN, _C100_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_C100_MEAN, _C100_STD),
    ])

    full_aug  = datasets.CIFAR100(root='data', train=True,
                                  download=True, transform=train_tf)
    full_eval = datasets.CIFAR100(root='data', train=True,
                                  download=True, transform=eval_tf)
    test_ds   = datasets.CIFAR100(root='data', train=False,
                                  download=True, transform=eval_tf)

    # Per-class stratified 80/20 split (targets is a list of ints)
    targets = np.array(full_aug.targets)
    rng = np.random.default_rng(seed=0)
    train_idx, val_idx = [], []
    for label in range(100):
        indices = rng.permutation(np.where(targets == label)[0]).tolist()
        cut = int(len(indices) * 0.8)
        train_idx.extend(indices[:cut])
        val_idx.extend(indices[cut:])

    train_dataset = Subset(full_aug,  train_idx)
    val_dataset   = Subset(full_eval, val_idx)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_dataset, shuffle=True,  **kw),
        DataLoader(val_dataset,   shuffle=False, **kw),
        DataLoader(test_ds,       shuffle=False, **kw),
    )


# ---------------------------------------------------------------------------
# Metrics — core computation (works with pre-collected probabilities)
# ---------------------------------------------------------------------------

def compute_metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    threshold: float = None,
    plots_dir: str = None,
    tag: str = '',
) -> dict:
    """Compute all required metrics from pre-collected softmax probabilities.

    Args:
        probs      : (N, C) float32 array of softmax class probabilities.
        labels     : (N,)   int array of ground-truth class indices.
        num_classes: Number of output classes.
        threshold  : If set, only samples with max_prob >= threshold are
                     included.  Coverage (fraction kept) is always reported.
        plots_dir  : If set, saves ROC, PR, and confusion-matrix plots here.
        tag        : Filename prefix for saved plots.

    Returns:
        JSON-serialisable dict of metrics.
    """
    n_total  = len(labels)
    coverage = 1.0

    if threshold is not None:
        mask     = probs.max(axis=1) >= threshold
        coverage = float(mask.mean())
        probs    = probs[mask]
        labels   = labels[mask]

    if len(labels) == 0:
        return {'n_total': n_total, 'coverage': 0.0,
                'note': f'No samples above threshold={threshold}'}

    preds    = probs.argmax(axis=1)
    y_onehot = np.zeros((len(labels), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(labels)), labels] = 1.0

    acc  = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, average='macro', zero_division=0))
    rec  = float(recall_score(labels, preds, average='macro', zero_division=0))
    f1   = float(f1_score(labels, preds, average='macro', zero_division=0))
    cm   = confusion_matrix(labels, preds).tolist()

    try:
        roc_auc = float(roc_auc_score(y_onehot, probs,
                                      multi_class='ovr', average='macro'))
    except ValueError:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_onehot, probs, average='macro'))
    except ValueError:
        pr_auc = None

    result = {
        'n_total':          n_total,
        'n_evaluated':      len(labels),
        'coverage':         coverage,
        'accuracy':         acc,
        'precision_macro':  prec,
        'recall_macro':     rec,
        'f1_macro':         f1,
        'roc_auc_macro':    roc_auc,
        'pr_auc_macro':     pr_auc,
        'confusion_matrix': cm,
    }

    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
        _plot_roc(y_onehot, probs, num_classes, plots_dir, tag)
        _plot_pr(y_onehot, probs, num_classes, plots_dir, tag)
        _plot_confusion_matrix(cm, num_classes, plots_dir, tag)

    return result


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    threshold: float = None,
    plots_dir: str = None,
    tag: str = '',
) -> dict:
    """Collect softmax probabilities from model+loader, then compute metrics.

    Thin wrapper around compute_metrics_from_probs for use in train.py and
    finetune.py.  ensemble.py calls compute_metrics_from_probs directly with
    soft-voted probabilities.
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs  = F.softmax(model(images), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return compute_metrics_from_probs(
        all_probs, all_labels, num_classes,
        threshold=threshold, plots_dir=plots_dir, tag=tag,
    )


# ---------------------------------------------------------------------------
# Plot helpers (private)
# ---------------------------------------------------------------------------

def _plot_roc(y_onehot, probs, num_classes, plots_dir, tag):
    """Macro-averaged ROC curve, interpolated to a common FPR grid."""
    mean_fpr = np.linspace(0, 1, 500)
    tprs = []
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_onehot[:, c], probs[:, c])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    auc_val  = float(np.trapz(mean_tpr, mean_fpr))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_fpr, mean_tpr, label=f'Macro-avg ROC (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {tag}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'roc_{tag}.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def _plot_pr(y_onehot, probs, num_classes, plots_dir, tag):
    """Macro-averaged Precision-Recall curve."""
    mean_recall = np.linspace(0, 1, 500)
    precisions  = []
    for c in range(num_classes):
        prec_c, rec_c, _ = precision_recall_curve(y_onehot[:, c], probs[:, c])
        # sklearn returns recall in decreasing order; reverse for np.interp
        precisions.append(np.interp(mean_recall, rec_c[::-1], prec_c[::-1]))
    mean_prec = np.mean(precisions, axis=0)
    auc_val   = float(np.trapz(mean_prec, mean_recall))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_recall, mean_prec, label=f'Macro-avg PR (AUC={auc_val:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {tag}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'pr_{tag}.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def _plot_confusion_matrix(cm, num_classes, plots_dir, tag):
    """Confusion matrix heatmap (small cells for large class counts)."""
    size = max(8, num_classes // 10)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, aspect='auto', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {tag}')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'cm_{tag}.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loss curve plotting
# ---------------------------------------------------------------------------

def plot_curves(
    train_losses: list,
    val_losses: list,
    title: str,
    save_path: str,
) -> None:
    """Plot and save training vs. validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label='Train')
    ax.plot(epochs, val_losses,   label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
