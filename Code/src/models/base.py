import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base model class providing functionality shared by all models."""

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        """Forward pass, must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method")

    def save(self, path):
        """Save model weights to the given path."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to: {path}")

    def load(self, path):
        """Load model weights from the given path."""
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from: {path}")

    def get_trainable_params(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_embeddings(self):
        """Freeze the embedding layer."""
        if hasattr(self, 'embedding'):
            self.embedding.weight.requires_grad = False
            print("Embedding layer frozen")

    def unfreeze_embeddings(self):
        """Unfreeze the embedding layer."""
        if hasattr(self, 'embedding'):
            self.embedding.weight.requires_grad = True
            print("Embedding layer unfrozen")
