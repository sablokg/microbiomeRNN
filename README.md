# DNA Sequence RNN (PyTorch + Click)

An LSTM-based classifier for DNA sequences with a full train / save / predict
workflow, driven entirely through a `click` command-line interface.

## Install
```bash
pip install -r requirements.txt
```

## Quick start
```bash
# 1. Generate a toy dataset (class 1 = contains a "TATAAA"-like motif)
python dna_rnn.py generate-demo-data --output demo.csv --n 2000

# 2. Train and save a checkpoint
python dna_rnn.py train --data demo.csv --output model.pt --epochs 15

# 3. Predict
python dna_rnn.py predict --model model.pt --sequence ACGTACGT...
python dna_rnn.py predict --model model.pt --fasta new_seqs.fasta --output preds.csv
python dna_rnn.py predict --model model.pt --csv new_seqs.csv --seq-col sequence
```

## Commands

### `generate-demo-data`
Creates a synthetic labeled CSV (`sequence,label`) for smoke-testing.

### `train`
- Input: CSV with `sequence,label` columns (configurable via `--seq-col`/`--label-col`)
- Key options: `--max-len`, `--embed-dim`, `--hidden-dim`, `--num-layers`,
  `--bidirectional/--unidirectional`, `--dropout`, `--epochs`, `--batch-size`,
  `--lr`, `--val-split`, `--patience` (early stopping), `--device` (`auto`/`cpu`/`cuda`)
- Saves the **best** checkpoint (lowest validation loss) to `--output`, including
  model weights, architecture config, and label names — everything needed to
  reload the model later without re-specifying hyperparameters.

### `predict`
- Provide exactly one of `--sequence`, `--fasta`, or `--csv`
- Prints results to stdout, or writes a CSV via `--output` with columns:
  `id, sequence, predicted_label, confidence`

## Model
Embedding → multi-layer (bi)LSTM → last hidden state → MLP classifier head.
Sequences are one-hot-index encoded over `{A, C, G, T, N}` (N covers unknown
bases and padding), truncated/padded to a fixed `max_len`.

## Notes
- Swap in your own real dataset any time — just point `--data` at a CSV with
  `sequence,label` columns (labels can be any integers, not just 0/1; the
  number of classes is inferred automatically).
- For multi-class problems, no code changes are needed — label count is
  detected from the training data.
