# Text Classification: CNN vs BiLSTM vs Transformer

Performance comparison of CNN, BiLSTM, and Transformer models on text
classification tasks, using pre-trained GloVe embeddings.

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages the Python environment)

## Setup

```bash
uv sync
```

Datasets are downloaded automatically from the Hugging Face Hub on first run:

| Key       | Hub dataset          | Classes | Task                   |
|-----------|----------------------|---------|------------------------|
| `imdb`    | `stanfordnlp/imdb`   | 2       | sentiment, long text   |
| `ag_news` | `fancyzhx/ag_news`   | 4       | topic, short text      |
| `bbc`     | `SetFit/bbc-news`    | 5       | topic, small data      |
| `sst5`    | `SetFit/sst5`        | 5       | fine-grained sentiment |

GloVe embeddings (`glove.6B.300d`, ~800 MB zip) are also downloaded
automatically into `pre-trained/` on first run.

The training split is further divided (stratified) into train/validation sets
using the per-dataset `val_ratio` (10% for IMDB and AG News; 20% for BBC,
whose training split has only 1225 samples). Validation loss drives early
stopping, LR scheduling, and best-model selection; the held-out test set is
evaluated exactly once after training. The vocabulary is built from the
training subset only.

## Usage

```bash
# Train all models on a dataset (imdb / ag_news / bbc)
uv run python src/train_all.py --model all --dataset imdb

# Train a single model
uv run python src/train_all.py --model cnn --dataset ag_news

# Force a device
uv run python src/train_all.py --model cnn --device cpu

# Run every dataset sequentially
uv run python src/train_all.py --model all --dataset all
```

The dataset can also be selected with the `TC_DATASET` environment variable
(used by the standalone `train_*.py` scripts). Training hyperparameters live
in `TRAIN_CONFIG` and model hyperparameters in `MODEL_CONFIGS` in
`src/config.py`.

Each `train_all.py` run wipes that dataset's previous outputs first, and
output filenames carry no timestamps — the directories always hold exactly
one (the latest) set of results per dataset.

## Outputs

All outputs are organized per dataset:

```
models/saved/<dataset>/      best checkpoints (bilstm.pth, cnn.pth, transformer.pth)
results/<dataset>/
├── plots/                   training-curve figures (PNG)
└── metrics/                 best-model metric tables and comparison table (CSV)
```

## Project Layout

```
src/
├── config.py            # datasets, training, and model configuration
├── training.py          # shared data pipeline and training loop (early stopping)
├── train_all.py         # CLI entry point; shares data across models, saves comparison CSV
├── train_bilstm.py      # thin wrapper: train BiLSTM only
├── train_cnn.py         # thin wrapper: train CNN only
├── train_transformer.py # thin wrapper: train Transformer only
├── data/
│   ├── loader.py        # Hugging Face dataset loading
│   ├── preprocessor.py  # text cleaning, vocabulary, PyTorch Dataset
│   └── embeddings.py    # GloVe download and embedding matrix
├── models/
│   ├── base.py          # shared model base class
│   ├── bilstm.py
│   ├── cnn.py
│   └── transformer.py
└── utils/
    ├── trainer.py       # train/eval loops
    └── metrics.py       # metrics and training-curve plots
```
