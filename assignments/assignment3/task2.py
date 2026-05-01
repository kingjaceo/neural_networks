"""
Assignment 3 — Task 2: Hotel Description Generation
Models: LSTM, GRU, Transformer (PyTorch, CPU-only)
Tokenization: Keras Tokenizer (as required by assignment)
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

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Tokenization (assignment-specified approach)
# ---------------------------------------------------------------------------

def build_tokenizer_and_sequences(corpus):
    """Fit Keras Tokenizer on corpus and return n-gram input sequences."""
    t = Tokenizer(
        num_words=None,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True, split=" ", char_level=False,
        oov_token=None, document_count=0,
    )
    t.fit_on_texts(corpus)
    total_words = len(t.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = t.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

    return t, input_sequences, total_words


def generate_padded_sequences(input_sequences, total_words):
    max_sequence_len = max(len(x) for x in input_sequences)
    input_sequences = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
    )
    predictors = input_sequences[:, :-1]
    label = input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TextLSTM(nn.Module):
    def __init__(self, total_words, embed_dim=10, hidden_size=100, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(total_words, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, total_words)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class TextGRU(nn.Module):
    def __init__(self, total_words, embed_dim=10, hidden_size=100, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(total_words, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, total_words)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class TextTransformer(nn.Module):
    def __init__(self, total_words, embed_dim=10, nhead=2, num_layers=2,
                 dim_feedforward=128, max_seq_len=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(total_words, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, total_words)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        x = self.embedding(x) + self.pos_embedding(positions)
        out = self.transformer(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X, y, epochs=50, lr=1e-3, batch_size=128):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = X.size(0)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        avg_loss = epoch_loss / n
        history.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

    return history


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_text(seed_text, next_words, model, max_seq_len, tokenizer):
    model.eval()
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")
        token_tensor = torch.LongTensor(token_list)
        with torch.no_grad():
            logits = model(token_tensor)
        predicted = torch.argmax(logits, dim=-1).item()
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text = seed_text + " " + output_word
    return seed_text.title()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_losses(all_losses, results_dir):
    plt.figure(figsize=(10, 5))
    for name, losses in all_losses.items():
        plt.plot(losses, label=name.upper())
    plt.title("Task 2 — Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "task2_loss_curves.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 2: Hotel Description Generation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--next_words", type=int, default=20,
                        help="Number of words to generate after seed text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=1,
                        help="CPU threads for PyTorch (set to cpus-per-task on SLURM)")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    print(f"[task2.py] seed={args.seed}  epochs={args.epochs}  lr={args.lr}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data ...")
    df = pd.read_csv(os.path.join(args.data_dir, "Seattle_Hotels_address_description.csv"), encoding="latin-1")
    all_descriptions = df["desc"].dropna().tolist()
    corpus = [x for x in all_descriptions]
    print(f"  {len(corpus)} hotel descriptions loaded")

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------
    print("Tokenizing ...")
    tokenizer, input_sequences, total_words = build_tokenizer_and_sequences(corpus)

    print(f"  Unique tokens: {len(tokenizer.word_index)}")
    print(f"  Total words (vocab+1): {total_words}")
    print(f"  N-gram sequences: {len(input_sequences)}")

    # Display tokenizer info (as required by assignment)
    print("\n--- Tokenizer Info ---")
    print(f"  word_counts (first 10): {dict(list(tokenizer.word_counts.items())[:10])}")
    print(f"  word_docs   (first 10): {dict(list(tokenizer.word_docs.items())[:10])}")
    print(f"  document_count: {tokenizer.document_count}")
    print(f"  Found {len(tokenizer.word_index)} unique tokens.")

    # ------------------------------------------------------------------
    # Pad sequences
    # ------------------------------------------------------------------
    print("\nPadding sequences ...")
    predictors, label, max_sequence_len = generate_padded_sequences(input_sequences, total_words)
    print(f"  predictors: {predictors.shape}  label: {label.shape}")
    print(f"  max_sequence_len: {max_sequence_len}")

    X = torch.LongTensor(predictors)
    # label is one-hot from to_categorical; convert to class indices for CrossEntropyLoss
    y = torch.LongTensor(np.argmax(label, axis=1))

    # ------------------------------------------------------------------
    # Train models
    # ------------------------------------------------------------------
    seed_texts = [
        "hilton seattle downtown",
        "best western seattle airport hotel",
        "located in the heart of downtown seattle",
    ]

    models_config = {
        "lstm": TextLSTM(total_words),
        "gru": TextGRU(total_words),
        "transformer": TextTransformer(total_words, max_seq_len=max_sequence_len),
    }

    all_losses = {}
    generated_texts = {}

    for name, model in models_config.items():
        print(f"\n{'='*60}")
        print(f"Training {name.upper()} model")
        print(f"{'='*60}")
        history = train_model(model, X, y,
                              epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        all_losses[name] = history

        # Save checkpoint (rich format matching assignment2 conventions)
        ckpt_path = os.path.join(args.checkpoints_dir, f"task2_{name}.pth")
        torch.save(
            {
                "model_type": name,
                "model_state_dict": model.state_dict(),
                "final_train_loss": history[-1],
                "total_words": total_words,
                "max_sequence_len": max_sequence_len,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"  -> checkpoint saved: {ckpt_path}")

        # Generate text
        generated_texts[name] = {}
        print(f"\n  --- Generated Descriptions ({name.upper()}) ---")
        for seed in seed_texts:
            text = generate_text(seed, args.next_words, model, max_sequence_len, tokenizer)
            generated_texts[name][seed] = text
            print(f"  Seed: \"{seed}\"")
            print(f"  Output: {text}\n")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    plot_losses(all_losses, args.results_dir)

    results = {
        "total_words": total_words,
        "max_sequence_len": max_sequence_len,
        "num_sequences": len(input_sequences),
        "generated_texts": generated_texts,
        "final_losses": {name: losses[-1] for name, losses in all_losses.items()},
    }
    with open(os.path.join(args.results_dir, "task2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone. Results saved to:", args.results_dir)


if __name__ == "__main__":
    main()
