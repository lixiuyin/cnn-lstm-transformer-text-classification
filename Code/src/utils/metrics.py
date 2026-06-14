import os

import matplotlib
matplotlib.use('Agg')  # plots are only saved to files, never shown interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsCalculator:
    @staticmethod
    def calculate_metrics(predictions, targets, loss=None):
        """Calculate evaluation metrics from raw predictions."""
        # Convert prediction scores to class labels
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-class case
            pred_labels = np.argmax(predictions, axis=1)
        else:
            # Binary case
            pred_labels = (predictions > 0.5).astype(int)

        accuracy = accuracy_score(targets, pred_labels)
        precision = precision_score(targets, pred_labels, average='weighted', zero_division=0)
        recall = recall_score(targets, pred_labels, average='weighted', zero_division=0)
        f1_score_value = f1_score(targets, pred_labels, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score_value
        }

        if loss is not None:
            metrics['loss'] = loss

        return metrics

    @staticmethod
    def plot_training_curves(train_metrics, val_metrics, model_name, dataset_name, save_dir=None):
        """Plot loss and accuracy curves for training and test sets."""
        train_losses = [m.get('loss', 0) for m in train_metrics]
        val_losses = [m.get('loss', 0) for m in val_metrics]
        train_accs = [m.get('accuracy', 0) for m in train_metrics]
        val_accs = [m.get('accuracy', 0) for m in val_metrics]
        epochs = range(1, len(train_metrics) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(epochs, train_losses, label='Training Loss', color='blue')
        ax1.plot(epochs, val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} on {dataset_name} - Loss Curves')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(epochs, train_accs, label='Training Accuracy', color='blue')
        ax2.plot(epochs, val_accs, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{model_name} on {dataset_name} - Accuracy Curves')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{model_name}_{dataset_name}_training_curves.png'
            plt.savefig(os.path.join(save_dir, filename))
            print(f"Training curves saved to: {os.path.join(save_dir, filename)}")

        plt.close(fig)

    def format_metric_value(self, metrics, key, default="N/A"):
        """Format a metric value; percentages keep one decimal place."""
        if key not in metrics:
            return default

        value = metrics[key]
        if key in ['accuracy', 'precision', 'recall', 'f1_score']:
            return f"{value*100:.1f}%"
        elif key in ['total_inference_time', 'avg_batch_time', 'min_batch_time', 'max_batch_time']:
            return f"{value:.4f}"
        elif key == 'samples_per_second':
            return f"{value:.2f}"
        else:
            return f"{value:.4f}"

    def display_best_metrics(self, best_train_metrics, best_val_metrics, test_metrics,
                             model_name, dataset_name, save_dir=None):
        """Print a table of the best model's training, validation, and test
        metrics, optionally saving it as a CSV file."""
        metric_keys = [
            ('Loss', 'loss'),
            ('Accuracy', 'accuracy'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1-Score', 'f1_score'),
            ('Total Inference Time (s)', 'total_inference_time'),
            ('Avg Batch Time (s)', 'avg_batch_time'),
            ('Min Batch Time (s)', 'min_batch_time'),
            ('Max Batch Time (s)', 'max_batch_time'),
            ('Samples/Second', 'samples_per_second')
        ]
        metrics_df = pd.DataFrame({
            'Metric': [name for name, _ in metric_keys],
            'Training': [self.format_metric_value(best_train_metrics, key) for _, key in metric_keys],
            'Validation': [self.format_metric_value(best_val_metrics, key) for _, key in metric_keys],
            'Test': [self.format_metric_value(test_metrics, key) for _, key in metric_keys]
        })

        print(f"\n{model_name} on {dataset_name} - Best Model Metrics:")
        print(metrics_df.to_string(index=False))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, f'{model_name}_{dataset_name}_best_metrics.csv')
            metrics_df.to_csv(csv_path, index=False)
            print(f"Best metrics saved to: {csv_path}")
