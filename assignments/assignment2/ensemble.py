"""Task 2 — Ensemble of 3 independently trained CompactNet instances on CIFAR-100.

Workflow
--------
1. Identify the best Task 1 activation by comparing results/metrics_finetune_*.json.
2. Train 3 independent instances of that model on CIFAR-100 with different random
   seeds, optionally initialised with Tiny ImageNet pretrained weights.
3. Combine predictions via soft voting (average class-probability distributions).
4. Evaluate the ensemble on the CIFAR-100 test set with the full metric suite.

Outputs
-------
  checkpoints/ensemble_<act>_seed<N>.pth         — per-member checkpoints
  results/curves_ensemble_<act>_seed<N>.png       — per-member loss curves
  results/metrics_ensemble_<act>.json             — ensemble evaluation metrics

Usage
-----
  python ensemble.py \\
      --activation relu \\
      --pretrained_dir checkpoints \\
      --epochs 50 \\
      --seeds 42 43 44 \\
      --checkpoint_dir checkpoints \\
      --results_dir results
"""

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import get_model
from utils import (
    compute_metrics_from_probs,
    get_cifar100_loaders,
    plot_curves,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train and evaluate a 3-model CIFAR-100 ensemble')
    parser.add_argument('--activation', type=str, required=True,
                        choices=['relu', 'gelu', 'rswish'],
                        help='Activation of the best Task 1 model')
    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help='Directory with Tiny ImageNet .pth files; '
                             'if set, ensemble members are initialised with '
                             'transfer-learned weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seeds', type=int, nargs=3, default=[42, 43, 44])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Train one ensemble member
# ---------------------------------------------------------------------------

def train_member(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    epochs: int,
    device: torch.device,
    ckpt_path: str,
) -> tuple:
    """Train a single ensemble member, saving the best checkpoint by val loss.

    Returns:
        (train_losses, val_losses) — per-epoch loss lists.
    """
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        tr_loss, tr_correct, n = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * images.size(0)
            tr_correct += (logits.argmax(dim=1) == labels).sum().item()
            n          += images.size(0)
        tr_loss /= n
        tr_acc   = tr_correct / n

        # --- Validate ---
        model.eval()
        vl_loss, vl_correct, n = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits  = model(images)
                loss    = criterion(logits, labels)
                vl_loss    += loss.item() * images.size(0)
                vl_correct += (logits.argmax(dim=1) == labels).sum().item()
                n          += images.size(0)
        vl_loss /= n
        vl_acc   = vl_correct / n

        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(
            f'  Epoch {epoch:3d}/{epochs} | '
            f'train loss {tr_loss:.4f}  acc {tr_acc:.4f} | '
            f'val loss {vl_loss:.4f}  acc {vl_acc:.4f}'
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': vl_loss,
                    'val_acc': vl_acc,
                },
                ckpt_path,
            )
            print(f'    -> checkpoint saved  (val_loss={vl_loss:.4f})')

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Soft voting
# ---------------------------------------------------------------------------

def soft_vote(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Average softmax probabilities across all ensemble members.

    Returns:
        avg_probs  : (N, C) numpy array of averaged softmax probabilities.
        true_labels: (N,)   numpy array of ground-truth class indices.
    """
    for m in models:
        m.eval()

    all_avg_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Stack (n_members, B, C) then take mean over member axis
            probs_stack = torch.stack(
                [F.softmax(m(images), dim=1) for m in models], dim=0
            )
            avg_probs = probs_stack.mean(dim=0)   # (B, C)
            all_avg_probs.append(avg_probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_avg_probs, axis=0),
        np.concatenate(all_labels,    axis=0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_cifar100_model(activation: str, pretrained_path: str = None) -> nn.Module:
    """Build a 100-class CompactNet, optionally loading Tiny ImageNet weights."""
    model = get_model(num_classes=200, activation=activation)

    if pretrained_path is not None:
        ckpt = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])

    # Replace 200-class head with a He-initialised 100-class head
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 100)
    nn.init.kaiming_normal_(model.classifier.weight, mode='fan_out',
                            nonlinearity='relu')
    nn.init.zeros_(model.classifier.bias)

    return model


def main() -> None:
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f'[ensemble.py] device={device}  activation={args.activation}  '
        f'seeds={args.seeds}'
    )

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    pretrained_path = None
    if args.pretrained_dir is not None:
        pretrained_path = os.path.join(
            args.pretrained_dir, f'best_{args.activation}.pth'
        )
        print(f'Using pretrained weights: {pretrained_path}')
    else:
        print('Training from scratch (He init)')

    # ------------------------------------------------------------------
    # Train 3 ensemble members
    # ------------------------------------------------------------------
    member_ckpt_paths = []

    for i, seed in enumerate(args.seeds):
        print(f'\n=== Ensemble member {i + 1}/3  (seed={seed}) ===')
        _set_seed(seed)

        model = _build_cifar100_model(args.activation, pretrained_path).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
        )

        ckpt_path = os.path.join(
            args.checkpoint_dir, f'ensemble_{args.activation}_seed{seed}.pth'
        )
        member_ckpt_paths.append(ckpt_path)

        train_losses, val_losses = train_member(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=args.epochs,
            device=device,
            ckpt_path=ckpt_path,
        )

        plot_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            title=f'Ensemble Member {i + 1} ({args.activation.upper()}, seed={seed})',
            save_path=os.path.join(
                args.results_dir,
                f'curves_ensemble_{args.activation}_seed{seed}.png',
            ),
        )

    # ------------------------------------------------------------------
    # Load all checkpoints and soft-vote on test set
    # ------------------------------------------------------------------
    print('\n=== Evaluating ensemble ===')
    members = []
    for ckpt_path in member_ckpt_paths:
        model = _build_cifar100_model(args.activation).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        members.append(model)

    avg_probs, true_labels = soft_vote(members, test_loader, device)

    tag          = f'ensemble_{args.activation}'
    metrics_all  = compute_metrics_from_probs(
        avg_probs, true_labels, num_classes=100,
        plots_dir=args.results_dir, tag=tag,
    )
    metrics_conf = compute_metrics_from_probs(
        avg_probs, true_labels, num_classes=100,
        threshold=0.9,
        plots_dir=args.results_dir, tag=f'{tag}_conf90',
    )

    output = {
        'activation':         args.activation,
        'seeds':              args.seeds,
        'n_members':          3,
        'pretrained_dir':     args.pretrained_dir,
        'all_predictions':    metrics_all,
        'high_confidence_0.9': metrics_conf,
    }
    metrics_path = os.path.join(args.results_dir, f'metrics_{tag}.json')
    with open(metrics_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Ensemble metrics -> {metrics_path}')


if __name__ == '__main__':
    main()
