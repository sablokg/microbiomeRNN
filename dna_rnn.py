#!/usr/bin/env python3
"""
dna_rnn.py — Train, save, and run predictions with an LSTM-based DNA
sequence classifier, driven by a `click` command-line interface.

Gaurav Sablok
gsablok@proton.me

Commands
--------
    python dna_rnn.py generate-demo-data   Create a toy labeled dataset (CSV)
    python dna_rnn.py train                Train a model and save a checkpoint
    python dna_rnn.py predict              Load a checkpoint and predict labels

Input formats
-------------
Training data: CSV with two columns, `sequence,label` (label = integer class,
e.g. 0/1). A header row is expected.

Prediction input: a single sequence via --sequence, a FASTA file via --fasta,
or a CSV file (one column of sequences) via --csv/--seq-col.

Example
-------
    python dna_rnn.py generate-demo-data --output demo.csv --n 2000
    python dna_rnn.py train --data demo.csv --output model.pt --epochs 15
    python dna_rnn.py predict --model model.pt --sequence ACGTACGTACGT...
    python dna_rnn.py predict --model model.pt --fasta new_seqs.fasta --output preds.csv
"""

import csv
import json
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# --------------------------------------------------------------------------
# Vocabulary / encoding
# --------------------------------------------------------------------------

NUCLEOTIDES = ["A", "C", "G", "T", "N"]  # N = unknown/padding/ambiguous base
CHAR2IDX = {c: i for i, c in enumerate(NUCLEOTIDES)}
PAD_IDX = CHAR2IDX["N"]


def encode_sequence(seq: str, max_len: int) -> List[int]:
    """Encode a DNA string into a fixed-length list of integer indices.
    Unknown characters (anything not ACGT) map to the 'N' index.
    Sequences longer than max_len are truncated; shorter ones are padded."""
    seq = seq.strip().upper()[:max_len]
    idxs = [CHAR2IDX.get(ch, PAD_IDX) for ch in seq]
    if len(idxs) < max_len:
        idxs = idxs + [PAD_IDX] * (max_len - len(idxs))
    return idxs


# --------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------

def read_labeled_csv(path: str, seq_col: str = "sequence", label_col: str = "label"
                      ) -> Tuple[List[str], List[int]]:
    sequences, labels = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if seq_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise click.ClickException(
                f"CSV must contain columns '{seq_col}' and '{label_col}'. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            sequences.append(row[seq_col])
            labels.append(int(row[label_col]))
    if not sequences:
        raise click.ClickException(f"No rows found in {path}")
    return sequences, labels


def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    """Returns (ids, sequences)."""
    ids, seqs = [], []
    cur_id, cur_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur_seq))
                cur_id = line[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            ids.append(cur_id)
            seqs.append("".join(cur_seq))
    if not ids:
        raise click.ClickException(f"No FASTA records found in {path}")
    return ids, seqs


def read_unlabeled_csv(path: str, seq_col: str = "sequence") -> Tuple[List[str], List[str]]:
    ids, seqs = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if seq_col not in reader.fieldnames:
            raise click.ClickException(
                f"CSV must contain column '{seq_col}'. Found: {reader.fieldnames}"
            )
        has_id = "id" in reader.fieldnames
        for i, row in enumerate(reader):
            ids.append(row["id"] if has_id else str(i))
            seqs.append(row[seq_col])
    return ids, seqs


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class DNADataset(Dataset):
    def __init__(self, sequences: List[str], labels: Optional[List[int]], max_len: int):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(encode_sequence(self.sequences[idx], self.max_len), dtype=torch.long)
        if self.labels is None:
            return x
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = len(NUCLEOTIDES)
    embed_dim: int = 16
    hidden_dim: int = 64
    num_layers: int = 2
    num_classes: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    max_len: int = 200


class DNARnn(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        mult = 2 if cfg.bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim * mult, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, x):
        emb = self.embedding(x)               # (B, L, E)
        out, (h, c) = self.lstm(emb)           # h: (num_layers*dirs, B, H)
        if self.cfg.bidirectional:
            h_final = torch.cat([h[-2], h[-1]], dim=1)   # last layer, both directions
        else:
            h_final = h[-1]
        return self.classifier(h_final)        # (B, num_classes) logits


# --------------------------------------------------------------------------
# Checkpoint save / load
# --------------------------------------------------------------------------

def save_checkpoint(path: str, model: DNARnn, cfg: ModelConfig, label_names: Optional[List[str]] = None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "label_names": label_names,
            "nucleotides": NUCLEOTIDES,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> Tuple[DNARnn, ModelConfig, Optional[List[str]]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = DNARnn(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt.get("label_names")


# --------------------------------------------------------------------------
# Training / evaluation loops
# --------------------------------------------------------------------------

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

@click.group()
def cli():
    """DNA sequence RNN — train, save, and predict, from the command line."""
    pass


@cli.command("generate-demo-data")
@click.option("--output", default="demo.csv", show_default=True, help="Output CSV path.")
@click.option("--n", default=2000, show_default=True, help="Number of sequences to generate.")
@click.option("--length", default=100, show_default=True, help="Sequence length.")
@click.option("--seed", default=42, show_default=True)
def generate_demo_data(output, n, length, seed):
    """Generate a toy binary-classification dataset: class 1 sequences are
    enriched for a 'TATAAA'-like motif (a stand-in for a real promoter
    signal); class 0 sequences are random. Useful for smoke-testing the
    train/predict commands."""
    random.seed(seed)
    motif = "TATAAA"
    rows = []
    for i in range(n):
        label = i % 2
        bases = "".join(random.choice("ACGT") for _ in range(length))
        if label == 1:
            pos = random.randint(0, max(0, length - len(motif) - 1))
            bases = bases[:pos] + motif + bases[pos + len(motif):]
        rows.append((bases, label))
    random.shuffle(rows)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "label"])
        writer.writerows(rows)
    click.echo(f"Wrote {n} rows to {output} (labels: 0 = random, 1 = motif-containing)")


@cli.command("train")
@click.option("--data", required=True, type=click.Path(exists=True), help="CSV with sequence,label columns.")
@click.option("--seq-col", default="sequence", show_default=True)
@click.option("--label-col", default="label", show_default=True)
@click.option("--output", default="model.pt", show_default=True, help="Path to save the trained checkpoint.")
@click.option("--max-len", default=200, show_default=True, help="Fixed sequence length (pad/truncate).")
@click.option("--embed-dim", default=16, show_default=True)
@click.option("--hidden-dim", default=64, show_default=True)
@click.option("--num-layers", default=2, show_default=True)
@click.option("--bidirectional/--unidirectional", default=True, show_default=True)
@click.option("--dropout", default=0.3, show_default=True)
@click.option("--epochs", default=15, show_default=True)
@click.option("--batch-size", default=32, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--val-split", default=0.15, show_default=True, help="Fraction of data held out for validation.")
@click.option("--device", default="auto", show_default=True, help="'auto', 'cpu', or 'cuda'.")
@click.option("--seed", default=42, show_default=True)
@click.option("--patience", default=5, show_default=True, help="Early-stopping patience on val loss (0 disables).")
def train(data, seq_col, label_col, output, max_len, embed_dim, hidden_dim, num_layers,
          bidirectional, dropout, epochs, batch_size, lr, val_split, device, seed, patience):
    """Train an LSTM classifier on labeled DNA sequences and save a checkpoint."""
    set_seed(seed)
    dev = get_device(device)
    click.echo(f"Using device: {dev}")

    sequences, labels = read_labeled_csv(data, seq_col, label_col)
    num_classes = len(set(labels))
    click.echo(f"Loaded {len(sequences)} sequences, {num_classes} classes")

    dataset = DNADataset(sequences, labels, max_len)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    cfg = ModelConfig(
        vocab_size=len(NUCLEOTIDES),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        bidirectional=bidirectional,
        dropout=dropout,
        max_len=max_len,
    )
    model = DNARnn(cfg).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, dev, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, dev, optimizer=None)
        click.echo(
            f"Epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                click.echo(f"Early stopping (no val improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    label_names = [str(c) for c in sorted(set(labels))]
    save_checkpoint(output, model, cfg, label_names)
    click.echo(f"Saved best checkpoint (val loss {best_val_loss:.4f}) to {output}")


@cli.command("predict")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True), help="Checkpoint from `train`.")
@click.option("--sequence", default=None, help="A single raw DNA sequence to classify.")
@click.option("--fasta", default=None, type=click.Path(exists=True), help="FASTA file of sequences to classify.")
@click.option("--csv", "csv_path", default=None, type=click.Path(exists=True), help="CSV file of sequences to classify.")
@click.option("--seq-col", default="sequence", show_default=True, help="Sequence column name (for --csv).")
@click.option("--output", default=None, type=click.Path(), help="Write predictions to this CSV instead of printing.")
@click.option("--device", default="auto", show_default=True)
@click.option("--batch-size", default=64, show_default=True)
def predict(model_path, sequence, fasta, csv_path, seq_col, output, device, batch_size):
    """Load a trained checkpoint and predict labels for new sequences.
    Provide exactly one of --sequence, --fasta, or --csv."""
    sources = [s for s in (sequence, fasta, csv_path) if s]
    if len(sources) != 1:
        raise click.UsageError("Provide exactly one of --sequence, --fasta, or --csv.")

    dev = get_device(device)
    model, cfg, label_names = load_checkpoint(model_path, dev)
    click.echo(f"Loaded model (max_len={cfg.max_len}, classes={label_names}) on {dev}")

    if sequence:
        ids, seqs = ["query"], [sequence]
    elif fasta:
        ids, seqs = read_fasta(fasta)
    else:
        ids, seqs = read_unlabeled_csv(csv_path, seq_col)

    ds = DNADataset(seqs, None, cfg.max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(dev)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    results = []
    for _id, seq, pred_idx, probs in zip(ids, seqs, all_preds, all_probs):
        label = label_names[pred_idx] if label_names else str(pred_idx)
        results.append({
            "id": _id,
            "sequence": seq if len(seq) <= 60 else seq[:57] + "...",
            "predicted_label": label,
            "confidence": round(max(probs), 4),
        })

    if output:
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "sequence", "predicted_label", "confidence"])
            writer.writeheader()
            writer.writerows(results)
        click.echo(f"Wrote {len(results)} predictions to {output}")
    else:
        for r in results:
            click.echo(f"{r['id']}\t{r['sequence']}\t-> {r['predicted_label']} (confidence {r['confidence']})")


if __name__ == "__main__":
    cli()
