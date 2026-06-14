import argparse
import os
import subprocess
import sys

DATASETS = ('imdb', 'ag_news', 'bbc', 'sst5')

COMPARISON_METRICS = [
    ('Accuracy', 'accuracy'),
    ('Precision', 'precision'),
    ('Recall', 'recall'),
    ('F1-Score', 'f1_score'),
    ('Test Loss', 'loss'),
    ('Samples/Second', 'samples_per_second'),
]


def save_comparison(results, dataset, metrics_dir):
    """Print and save a cross-model comparison of best test metrics."""
    import pandas as pd

    table = {'Metric': [name for name, _ in COMPARISON_METRICS]}
    for model_name, metrics in results.items():
        column = []
        for name, key in COMPARISON_METRICS:
            value = metrics.get(key)
            if value is None:
                column.append('N/A')
            elif key in ('accuracy', 'precision', 'recall', 'f1_score'):
                column.append(f"{value*100:.1f}%")
            elif key == 'samples_per_second':
                column.append(f"{value:.0f}")
            else:
                column.append(f"{value:.4f}")
        table[model_name] = column

    df = pd.DataFrame(table)
    print(f"\nModel comparison on {dataset} (best test metrics):")
    print(df.to_string(index=False))

    csv_path = os.path.join(metrics_dir, f'comparison_{dataset}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Train text classification models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['bilstm', 'cnn', 'transformer', 'all'],
                        help='Model to train')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=[*DATASETS, 'all'],
                        help='Dataset to train on, or "all" to run every dataset '
                             '(default: TC_DATASET env var or imdb)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cuda/cpu)')
    args = parser.parse_args()

    if args.dataset == 'all':
        # One subprocess per dataset: config resolves dataset-dependent paths
        # at import time, so each dataset needs a fresh interpreter
        for dataset in DATASETS:
            print(f"\n========== Dataset: {dataset} ==========")
            command = [sys.executable, os.path.abspath(__file__),
                       '--model', args.model, '--dataset', dataset]
            if args.device:
                command += ['--device', args.device]
            subprocess.run(command, check=True)
        return

    # config resolves the dataset (and its output paths) at import time,
    # so the override must be set before importing it
    if args.dataset:
        os.environ['TC_DATASET'] = args.dataset

    from config import CHOSEN_DATASET, METRICS_DIR
    from training import prepare_data, clean_output_dirs
    from train_bilstm import train_bilstm
    from train_cnn import train_cnn
    from train_transformer import train_transformer

    trainers = {
        'bilstm': ('BiLSTM', train_bilstm),
        'cnn': ('CNN', train_cnn),
        'transformer': ('Transformer', train_transformer),
    }

    print(f"Using dataset: {CHOSEN_DATASET}")
    selected = list(trainers) if args.model == 'all' else [args.model]

    # Full runs start from a clean per-dataset output directory; single-model
    # runs only overwrite their own files, preserving the other models' results
    if args.model == 'all':
        clean_output_dirs()

    # Run the (identical) data pipeline once and share it across models
    data = prepare_data()

    results = {}
    for key in selected:
        model_name, train_fn = trainers[key]
        print(f"\nTraining {model_name} model...")
        results[model_name] = train_fn(device=args.device, data=data)

    if len(results) > 1:
        save_comparison(results, CHOSEN_DATASET, METRICS_DIR)


if __name__ == '__main__':
    main()
