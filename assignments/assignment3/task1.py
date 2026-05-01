"""
Assignment 3 — Task 1: Google Stock Price Prediction
Models: SimpleRNN, LSTM, GRU (PyTorch, CPU-only)
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StockRNN(nn.Module):
    """Wrapper around nn.RNN / nn.LSTM / nn.GRU for single-step regression."""

    def __init__(self, rnn_type: str, input_size: int = 1, hidden_size: int = 50,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(data_dir: str, timesteps: int = 60):
    """Load CSVs, scale, and create sliding-window sequences."""
    train_df = pd.read_csv(os.path.join(data_dir, "Google_Stock_Price_Train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "Google_Stock_Price_Test.csv"))

    # Use "Open" column; remove commas in numbers if present
    train_open = train_df["Open"].astype(str).str.replace(",", "").astype(float).values.reshape(-1, 1)
    test_open = test_df["Open"].astype(str).str.replace(",", "").astype(float).values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_open)

    # Build training sequences
    X_train, y_train = [], []
    for i in range(timesteps, len(train_scaled)):
        X_train.append(train_scaled[i - timesteps:i, 0])
        y_train.append(train_scaled[i, 0])
    X_train = np.array(X_train).reshape(-1, timesteps, 1)
    y_train = np.array(y_train).reshape(-1, 1)

    # For test set, prepend the last `timesteps` training values
    total = np.concatenate([train_open[-timesteps:], test_open], axis=0)
    total_scaled = scaler.transform(total)
    X_test, y_test_real = [], []
    for i in range(timesteps, len(total_scaled)):
        X_test.append(total_scaled[i - timesteps:i, 0])
        y_test_real.append(test_open[i - timesteps, 0])  # actual price
    X_test = np.array(X_test).reshape(-1, timesteps, 1)
    y_test_real = np.array(y_test_real).reshape(-1, 1)

    return (
        torch.FloatTensor(X_train), torch.FloatTensor(y_train),
        torch.FloatTensor(X_test), y_test_real,
        scaler,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train, epochs=100, lr=1e-3, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    n = X_train.size(0)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        avg_loss = epoch_loss / n
        history.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}")

    return history


# ---------------------------------------------------------------------------
# Evaluation & plotting
# ---------------------------------------------------------------------------

def predict(model, X_test, scaler):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test).numpy()
    pred = scaler.inverse_transform(pred_scaled)
    return pred


def plot_predictions(predictions: dict, y_real, results_dir: str):
    """Plot predicted vs actual stock prices for each model."""
    plt.figure(figsize=(14, 6))
    plt.plot(y_real, color="black", label="Actual Google Stock Price")
    colors = {"rnn": "red", "lstm": "blue", "gru": "green"}
    for name, pred in predictions.items():
        plt.plot(pred, color=colors[name], label=f"{name.upper()} Prediction")
    plt.title("Google Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "task1_predictions.png"), dpi=150)
    plt.close()


def plot_losses(all_losses: dict, results_dir: str):
    plt.figure(figsize=(10, 5))
    for name, losses in all_losses.items():
        plt.plot(losses, label=f"{name.upper()}")
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "task1_loss_curves.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 1: Stock Price Prediction")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=1,
                        help="CPU threads for PyTorch (set to cpus-per-task on SLURM)")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    print(f"[task1.py] seed={args.seed}  epochs={args.epochs}  lr={args.lr}")
    print("Loading data ...")
    X_train, y_train, X_test, y_test_real, scaler = load_data(args.data_dir, args.timesteps)
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    model_names = ["rnn", "lstm", "gru"]
    all_losses = {}
    predictions = {}
    metrics = {}

    for name in model_names:
        print(f"\n{'='*50}")
        print(f"Training {name.upper()} model")
        print(f"{'='*50}")
        model = StockRNN(rnn_type=name)
        history = train_model(model, X_train, y_train,
                              epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        all_losses[name] = history

        # Save checkpoint (rich format matching assignment2 conventions)
        ckpt_path = os.path.join(args.checkpoints_dir, f"task1_{name}.pth")
        torch.save(
            {
                "model_type": name,
                "model_state_dict": model.state_dict(),
                "final_train_loss": history[-1],
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"  -> checkpoint saved: {ckpt_path}")

        # Predict
        pred = predict(model, X_test, scaler)
        predictions[name] = pred

        # Metrics
        rmse = float(np.sqrt(np.mean((pred - y_test_real) ** 2)))
        mae = float(np.mean(np.abs(pred - y_test_real)))
        metrics[name] = {"rmse": rmse, "mae": mae}
        print(f"  RMSE={rmse:.2f}  MAE={mae:.2f}")

    # Save results
    plot_predictions(predictions, y_test_real, args.results_dir)
    plot_losses(all_losses, args.results_dir)

    with open(os.path.join(args.results_dir, "task1_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone. Results saved to:", args.results_dir)


if __name__ == "__main__":
    main()
