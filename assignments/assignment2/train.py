"""Task 1 — Train novel DCNN on Tiny ImageNet-200.

Trains one model variant (ReLU / GELU / RSwish) per invocation.
Run three times (or via SLURM array) to cover all activation functions.

Outputs per run
---------------
  checkpoints/best_<activation>.pth        — best model weights
  results/curves_train_<activation>.png    — train/val loss curves
  results/metrics_train_<activation>.json  — test metrics (all + ≥0.9 conf.)
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

from models import get_model
from utils import get_tiny_imagenet_loaders, compute_metrics, plot_curves


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train DCNN on Tiny ImageNet-200')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root of tiny-imagenet-200 (contains train/ and val/)')
    parser.add_argument('--arch', type=str, default='compactnet',
                        choices=['compactnet', 'resnet18'],
                        help='compactnet = novel architecture; resnet18 = baseline')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu', 'rswish'],
                        help='Activation for compactnet only (ignored for resnet18)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(arch: str, num_classes: int, activation: str) -> nn.Module:
    if arch == 'compactnet':
        return get_model(num_classes=num_classes, activation=activation)

    # ResNet-18 adapted for 32x32 input:
    #   - replace 7x7 stride-2 stem with 3x3 stride-1 (avoids immediate halving)
    #   - remove maxpool (would further halve to 8x8 before any residual blocks)
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    nn.init.zeros_(model.fc.bias)
    return model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Train / validation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full pass over the training set.

    Returns:
        (mean_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += images.size(0)

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a single loader without computing gradients.

    Returns:
        (mean_loss, accuracy)
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            n += images.size(0)

    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train.py] device={device}  arch={args.arch}  activation={args.activation}  epochs={args.epochs}')

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = get_tiny_imagenet_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model = build_model(args.arch, num_classes=200, activation=args.activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine annealing: lr decays from args.lr to 1% of args.lr over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    tag = args.activation if args.arch == 'compactnet' else args.arch
    ckpt_path = os.path.join(args.checkpoint_dir, f'best_{tag}.pth')
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'train loss {tr_loss:.4f}  acc {tr_acc:.4f} | '
            f'val loss {vl_loss:.4f}  acc {vl_acc:.4f}'
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(
                {
                    'epoch': epoch,
                    'activation': args.activation,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': vl_loss,
                    'val_acc': vl_acc,
                    'args': vars(args),
                },
                ckpt_path,
            )
            print(f'  -> checkpoint saved  (val_loss={vl_loss:.4f})')

    # ------------------------------------------------------------------
    # Loss curves
    # ------------------------------------------------------------------
    plot_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        title=f'Tiny ImageNet-200 — {tag.upper()}',
        save_path=os.path.join(args.results_dir, f'curves_train_{tag}.png'),
    )

    # ------------------------------------------------------------------
    # Test evaluation using the best checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    metrics_all  = compute_metrics(model, test_loader, device, num_classes=200,
                                   plots_dir=args.results_dir,
                                   tag=f'train_{tag}')
    metrics_conf = compute_metrics(model, test_loader, device, num_classes=200,
                                   threshold=0.9,
                                   plots_dir=args.results_dir,
                                   tag=f'train_{tag}_conf90')

    output = {
        'arch': args.arch,
        'activation': args.activation,
        'best_val_loss': best_val_loss,
        'all_predictions': metrics_all,
        'high_confidence_0.9': metrics_conf,
    }
    metrics_path = os.path.join(args.results_dir, f'metrics_train_{tag}.json')
    with open(metrics_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Test metrics -> {metrics_path}')


if __name__ == '__main__':
    main()
