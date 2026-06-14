import os
import torch

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pre-trained embeddings directory
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, 'pre-trained')

# Output directories
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Pre-trained GloVe embeddings path
GLOVE_PATH = os.path.join(PRETRAINED_DIR, 'glove.6B.300d.txt')

# Dataset configurations (downloaded from the Hugging Face Hub)
DATASET_CONFIGS = {
    'imdb': {
        'name': 'IMDB',
        'hf_path': 'stanfordnlp/imdb',
        'text_column': 'text',
        'label_column': 'label',
        'num_classes': 2,
        'max_length': 400,
        'val_ratio': 0.1,
        'categories': ['negative', 'positive']
    },
    'ag_news': {
        'name': 'AG News',
        'hf_path': 'fancyzhx/ag_news',
        'text_column': 'text',
        'label_column': 'label',
        'num_classes': 4,
        'max_length': 64,
        'val_ratio': 0.1,
        'categories': ['World', 'Sports', 'Business', 'Science/Technology']
    },
    'bbc': {
        'name': 'BBC News',
        'hf_path': 'SetFit/bbc-news',
        'text_column': 'text',
        'label_column': 'label',
        'num_classes': 5,
        'max_length': 512,
        # larger validation share: only 1225 training samples in total
        'val_ratio': 0.2,
        'categories': ['business', 'entertainment', 'politics', 'sport', 'tech']
    },
    'sst5': {
        'name': 'SST-5',
        'hf_path': 'SetFit/sst5',
        'text_column': 'text',
        'label_column': 'label',
        'num_classes': 5,
        'max_length': 64,
        'val_ratio': 0.1,
        'categories': ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    }
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 512,
    'num_workers': 4,  # DataLoader workers; set 0 to debug crashes in worker processes
    'num_epochs': 50,
    'early_stopping_patience': 5,  # stop after this many epochs without val-loss improvement
    'early_stopping_min_delta': 1e-3,  # val-loss decrease below this does not count as improvement
    'learning_rate': 0.001,
    'embedding_dim': 300,  # must match the GloVe embedding dimension
    'random_seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Model configurations
MODEL_CONFIGS = {
    'bilstm': {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.5,
        'bidirectional': True,
        'batch_first': True
    },
    'cnn': {
        'num_filters': 100,
        'filter_sizes': [2, 3, 4],
        'dropout': 0.5,
        'max_pool': True,
        'batch_norm': True
    },
    'transformer': {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1
    }
}

# Selected via the TC_DATASET environment variable (or --dataset in train_all.py)
CHOSEN_DATASET = os.environ.get('TC_DATASET', 'imdb')
if CHOSEN_DATASET not in DATASET_CONFIGS:
    raise ValueError(
        f"Unknown dataset '{CHOSEN_DATASET}'; "
        f"expected one of {sorted(DATASET_CONFIGS)}"
    )

# Per-dataset output directories:
#   models/saved/<dataset>/<model>.pth
#   results/<dataset>/plots/    training-curve figures
#   results/<dataset>/metrics/  best-model metric tables (CSV)
DATASET_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, CHOSEN_DATASET)
PLOTS_DIR = os.path.join(RESULTS_DIR, CHOSEN_DATASET, 'plots')
METRICS_DIR = os.path.join(RESULTS_DIR, CHOSEN_DATASET, 'metrics')
for _output_dir in (DATASET_MODEL_DIR, PLOTS_DIR, METRICS_DIR):
    os.makedirs(_output_dir, exist_ok=True)

# Model checkpoint paths
MODEL_PATHS = {
    model: os.path.join(DATASET_MODEL_DIR, f'{model}.pth')
    for model in ('bilstm', 'cnn', 'transformer')
}
