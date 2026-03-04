"""Task 1 — Fine-tune or train from scratch on CIFAR-100.

Modes
-----
  scratch  : He-initialised weights, 100-class head, train from zero.
  finetune : Load best Tiny ImageNet-200 checkpoint, swap 200-class head for
             a new He-initialised 100-class head, fine-tune the full network.

Outputs per run
---------------
  checkpoints/cifar100_<mode>_<activation>.pth    — best model weights
  results/curves_<mode>_<activation>.png          — train/val loss curves
  results/metrics_<mode>_<activation>.json        — test metrics (all + ≥0.9 conf.)
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from models import get_model
from utils import get_cifar100_loaders, compute_metrics, plot_curves


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train/fine-tune DCNN on CIFAR-100')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['scratch', 'finetune'],
                        help='scratch = He init from random; '
                             'finetune = load Tiny ImageNet weights then adapt head')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu', 'rswish'])
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to .pth checkpoint (required when --mode=finetune)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Head adaptation
# ---------------------------------------------------------------------------

def adapt_classifier(model: nn.Module, num_classes: int = 100) -> nn.Module:
    """Replace model.classifier with a new He-initialised Linear(*, num_classes).

    Assumes the model exposes its classification head as model.classifier
    (an nn.Linear), which is the convention used in CompactNet.
    """
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.classifier.weight, mode='fan_out', nonlinearity='relu')
    nn.init.zeros_(model.classifier.bias)
    return model


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

    if args.mode == 'finetune' and args.pretrained_path is None:
        raise ValueError('--pretrained_path is required when --mode=finetune')

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[finetune.py] device={device}  mode={args.mode}  activation={args.activation}')

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    # Always build with 200-class head first so state dict keys match the
    # pretrained checkpoint (both scratch and finetune paths go through here).
    model = get_model(num_classes=200, activation=args.activation)

    if args.mode == 'finetune':
        ckpt = torch.load(args.pretrained_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        print(
            f'Loaded pretrained weights from {args.pretrained_path} '
            f'(epoch {ckpt["epoch"]}, val_loss {ckpt["val_loss"]:.4f})'
        )
    else:
        # He init is applied inside CompactNet.__init__; nothing extra needed.
        print('Training from scratch (He initialisation applied inside get_model)')

    # Replace the 200-class head with a fresh 100-class head.
    model = adapt_classifier(model, num_classes=100).to(device)

    # ------------------------------------------------------------------
    # Loss, optimiser, scheduler
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    tag = f'{args.mode}_{args.activation}'
    ckpt_path = os.path.join(args.checkpoint_dir, f'cifar100_{tag}.pth')
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
                    'mode': args.mode,
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
        title=f'CIFAR-100 ({args.mode}) — {args.activation.upper()}',
        save_path=os.path.join(args.results_dir, f'curves_{tag}.png'),
    )

    # ------------------------------------------------------------------
    # Test evaluation using the best checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    metrics_all  = compute_metrics(model, test_loader, device, num_classes=100,
                                   plots_dir=args.results_dir,
                                   tag=tag)
    metrics_conf = compute_metrics(model, test_loader, device, num_classes=100,
                                   threshold=0.9,
                                   plots_dir=args.results_dir,
                                   tag=f'{tag}_conf90')

    output = {
        'activation': args.activation,
        'mode': args.mode,
        'best_val_loss': best_val_loss,
        'all_predictions': metrics_all,
        'high_confidence_0.9': metrics_conf,
    }
    metrics_path = os.path.join(args.results_dir, f'metrics_{tag}.json')
    with open(metrics_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Test metrics -> {metrics_path}')


if __name__ == '__main__':
    main()
