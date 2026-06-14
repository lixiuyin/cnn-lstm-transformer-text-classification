import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable cuDNN optimizations
torch.backends.cudnn.benchmark = True

from config import (
    TRAIN_CONFIG, MODEL_CONFIGS, MODEL_PATHS,
    DATASET_CONFIGS, CHOSEN_DATASET, GLOVE_PATH,
    DATASET_MODEL_DIR, PLOTS_DIR, METRICS_DIR
)
from data.loader import load_split
from data.preprocessor import TextPreprocessor
from data.embeddings import GloVeEmbeddings
from utils.trainer import ModelTrainer
from utils.metrics import MetricsCalculator


def clean_output_dirs():
    """Remove artifacts of previous runs for the chosen dataset
    (checkpoints, plots, and metric tables)."""
    removed = 0
    for directory in (DATASET_MODEL_DIR, PLOTS_DIR, METRICS_DIR):
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
    if removed:
        logger.info(f"Cleaned {removed} files from previous runs for '{CHOSEN_DATASET}'")


def set_random_seeds():
    seed = TRAIN_CONFIG['random_seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    logger.info(f"Random seed: {seed}")


def prepare_data():
    """Run the full data pipeline once: GloVe embeddings, dataset download,
    preprocessing, vocabulary, data loaders, and the embedding matrix.

    The returned dict can be shared across model trainings so the (identical)
    pipeline is not repeated per model.
    """
    set_random_seeds()

    dataset_config = DATASET_CONFIGS[CHOSEN_DATASET]
    logger.info(f"Using dataset: {dataset_config['name']}")

    logger.info("Initializing text preprocessor and embeddings...")
    preprocessor = TextPreprocessor()
    embeddings = GloVeEmbeddings(GLOVE_PATH)

    # Load data from the Hugging Face Hub
    train_texts, train_labels = load_split(dataset_config, 'train')
    test_texts, test_labels = load_split(dataset_config, 'test')

    # Hold out a validation set from the training split. It drives early
    # stopping, LR scheduling, and model selection; the test set is only
    # evaluated once, after training.
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        list(train_texts), list(train_labels),
        test_size=dataset_config['val_ratio'],
        random_state=TRAIN_CONFIG['random_seed'],
        stratify=list(train_labels)
    )
    logger.info(
        f"Split sizes - train: {len(train_texts)}, "
        f"val: {len(val_texts)}, test: {len(test_texts)}"
    )

    logger.info("Preprocessing texts...")
    train_texts = [preprocessor.preprocess(text) for text in tqdm(train_texts, desc="Preprocessing train set")]
    val_texts = [preprocessor.preprocess(text) for text in tqdm(val_texts, desc="Preprocessing val set")]
    test_texts = [preprocessor.preprocess(text) for text in tqdm(test_texts, desc="Preprocessing test set")]

    logger.info("Creating datasets...")
    # The vocabulary is built from the training subset only
    max_length = dataset_config['max_length']
    train_dataset = preprocessor.create_dataset(train_texts, train_labels, max_length=max_length)
    vocab = train_dataset.vocab
    val_dataset = preprocessor.create_dataset(val_texts, val_labels, vocab=vocab, max_length=max_length)
    test_dataset = preprocessor.create_dataset(test_texts, test_labels, vocab=vocab, max_length=max_length)
    logger.info(f"Vocabulary size: {len(vocab)}")

    loader_kwargs = {
        'batch_size': TRAIN_CONFIG['batch_size'],
        'num_workers': TRAIN_CONFIG['num_workers'],
        'pin_memory': True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    logger.info(f"Batch size: {TRAIN_CONFIG['batch_size']}")

    logger.info("Building the pre-trained embedding matrix...")
    embedding_matrix = embeddings.load_embeddings(vocab)

    return {
        'dataset_config': dataset_config,
        'vocab': vocab,
        'embedding_matrix': embedding_matrix,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
    }


def train_model(model_key, model_cls, model_name, device=None, data=None):
    """Train one model with the shared training protocol
    (AdamW + ReduceLROnPlateau + early stopping) and save its outputs.

    Returns the best test metrics.
    """
    start_time = time.time()

    if device is None:
        device = TRAIN_CONFIG['device']
    logger.info(f"Using device: {device}")

    if data is None:
        data = prepare_data()
    set_random_seeds()  # identical weight init regardless of pipeline reuse

    dataset_config = data['dataset_config']

    logger.info(f"Initializing {model_name} model...")
    model = model_cls(
        vocab_size=len(data['vocab']),
        embedding_dim=TRAIN_CONFIG['embedding_dim'],
        pretrained_embeddings=data['embedding_matrix'],
        num_classes=dataset_config['num_classes'],
        **MODEL_CONFIGS[model_key]
    ).to(device)

    if 'cuda' in str(device) and torch.cuda.device_count() > 1:
        logger.info(f"Training with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        # use the same improvement threshold as early stopping
        threshold=TRAIN_CONFIG['early_stopping_min_delta'],
        threshold_mode='abs'
    )
    logger.info(f"Learning rate: {TRAIN_CONFIG['learning_rate']}")

    trainer = ModelTrainer(model, criterion, optimizer, device)

    logger.info(f"Training {model_name} on {dataset_config['name']}...")
    patience = TRAIN_CONFIG['early_stopping_patience']
    best_loss = float('inf')
    best_val_metrics = None
    best_train_metrics = None
    best_model_state = None
    epochs_without_improvement = 0

    train_metrics_list = []
    val_metrics_list = []

    for epoch in range(TRAIN_CONFIG['num_epochs']):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch + 1}/{TRAIN_CONFIG['num_epochs']}")

        train_loss, train_metrics = trainer.train_epoch(data['train_loader'])
        logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_metrics['accuracy']*100:.2f}%")

        if not np.isfinite(train_loss):
            # Abort before appending so the train/val metric lists stay aligned
            logger.error(f"Training loss is {train_loss}; aborting {model_name} training")
            break
        train_metrics_list.append(train_metrics)

        val_loss, val_metrics = trainer.evaluate(data['val_loader'])
        val_metrics_list.append(val_metrics)
        logger.info(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']*100:.2f}%")

        scheduler.step(val_loss)

        if val_loss < best_loss - TRAIN_CONFIG['early_stopping_min_delta']:
            best_loss = val_loss
            best_val_metrics = val_metrics
            best_train_metrics = train_metrics
            epochs_without_improvement = 0
            # Deep-copy the unwrapped state: state_dict() tensors are views of
            # the live parameters, and DataParallel adds a 'module.' prefix
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            best_model_state = copy.deepcopy(raw_model.state_dict())
            torch.save(best_model_state, MODEL_PATHS[model_key])
            logger.info(f"Saved best model, Loss: {best_loss:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping: no val-loss improvement for {patience} epochs "
                    f"(best: {best_loss:.4f})"
                )
                break

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} took {epoch_time:.2f}s")

    if best_model_state is None:
        raise RuntimeError(
            f"{model_name} training never achieved a finite validation loss; "
            "no model was saved. The run likely diverged (NaN loss)."
        )

    # Restore the best (val-selected) model and evaluate the test set ONCE
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(best_model_state)

    test_loss, test_metrics = trainer.evaluate(data['test_loader'])
    logger.info(f"Final test - Loss: {test_loss:.4f}, Accuracy: {test_metrics['accuracy']*100:.2f}%")

    total_time = time.time() - start_time
    logger.info(f"Training finished! Total time: {total_time:.2f}s ({total_time/60:.2f}min)")

    metrics_calculator = MetricsCalculator()
    metrics_calculator.display_best_metrics(
        best_train_metrics,
        best_val_metrics,
        test_metrics,
        model_name=model_name,
        dataset_name=CHOSEN_DATASET,
        save_dir=METRICS_DIR
    )

    metrics_calculator.plot_training_curves(
        train_metrics_list,
        val_metrics_list,
        model_name=model_name,
        dataset_name=CHOSEN_DATASET,
        save_dir=PLOTS_DIR
    )

    return test_metrics
