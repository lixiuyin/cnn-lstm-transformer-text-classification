import time

import numpy as np
import torch
from tqdm import tqdm

from .metrics import MetricsCalculator


MAX_GRAD_NORM = 1.0  # gradient clipping threshold (guards against LSTM gradient explosion)


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, train_loader):
        """Train for one epoch; metrics are computed from the predictions
        made during training, without a second pass over the data."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()

            text = batch['text'].to(self.device)
            labels = batch['label'].to(self.device)

            predictions = self.model(text).squeeze(1)
            loss = self.criterion(predictions, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        metrics = MetricsCalculator.calculate_metrics(
            np.array(all_predictions),
            np.array(all_targets),
            loss=avg_loss
        )

        return avg_loss, metrics

    def _cuda_sync(self):
        """Wait for pending CUDA kernels so wall-clock timings are accurate."""
        if torch.cuda.is_available() and 'cuda' in str(self.device):
            torch.cuda.synchronize()

    def evaluate(self, test_loader):
        """Evaluate in a single pass, recording per-batch inference times."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        batch_times = []
        warmed_shapes = set()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                text = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)

                if text.shape not in warmed_shapes:
                    # Discard one forward pass per input shape: cuDNN benchmark
                    # autotunes for every new shape (e.g. the smaller final
                    # batch) and would otherwise skew that batch's timing
                    self.model(text)
                    warmed_shapes.add(text.shape)

                self._cuda_sync()
                start_time = time.time()
                predictions = self.model(text).squeeze(1)
                self._cuda_sync()
                batch_times.append(time.time() - start_time)

                loss = self.criterion(predictions, labels)

                total_loss += loss.item()
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        metrics = MetricsCalculator.calculate_metrics(
            np.array(all_predictions),
            np.array(all_targets),
            loss=avg_loss
        )

        total_inference_time = sum(batch_times)
        metrics.update({
            'total_inference_time': total_inference_time,
            'avg_batch_time': float(np.mean(batch_times)),
            'min_batch_time': float(np.min(batch_times)),
            'max_batch_time': float(np.max(batch_times)),
            'samples_per_second': len(test_loader.dataset) / total_inference_time
        })

        return avg_loss, metrics

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")
