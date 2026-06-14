import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class ResidualBlock(nn.Module):
    """A simple residual block."""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        # Padding keeps the sequence length unchanged
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)

        # Residual connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        # Even kernel sizes with padding=k//2 yield one extra position;
        # trim it (lossless) instead of resampling the feature map
        if out.size(2) != residual.size(2):
            out = out[:, :, :residual.size(2)]

        out += residual
        out = F.relu(out)

        return out


class CNNClassifier(BaseModel):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, dropout=0.5,
                 pretrained_embeddings=None, num_classes=2, max_pool=True, batch_norm=True):
        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            # Explicit copy (no aliasing of the possibly shared numpy buffer)
            self.embedding.weight.data.copy_(
                torch.as_tensor(pretrained_embeddings, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = True  # keep the embedding layer trainable

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(embedding_dim, num_filters, kernel_size=size, padding=size // 2)
            for size in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.max_pool = max_pool
        self.batch_norm = batch_norm

        if batch_norm:
            self.bn = nn.BatchNorm1d(num_filters * len(filter_sizes))

        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]

        # Reorder dimensions for Conv1d
        embedded = embedded.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]

        conv_outputs = []
        for residual_block in self.residual_blocks:
            conv_output = residual_block(embedded)
            if self.max_pool:
                conv_output = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            else:
                conv_output = F.avg_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            conv_outputs.append(conv_output)

        # Concatenate all convolution outputs
        cat = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        cat = self.dropout(cat)

        if self.batch_norm:
            cat = self.bn(cat)

        return self.fc(cat)
