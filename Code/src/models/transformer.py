import math

import torch
import torch.nn as nn

from .base import BaseModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Transformer encoder block: self-attention, feed-forward network,
    residual connections, and layer normalization."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask=None):
        # Self-attention + residual connection + layer normalization
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        # Feed-forward network + residual connection + layer normalization
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class TransformerClassifier(BaseModel):
    def __init__(self, vocab_size, embedding_dim, d_model, nhead, num_layers, dim_feedforward,
                 dropout=0.1, pretrained_embeddings=None, num_classes=2):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            # Explicit copy (no aliasing of the possibly shared numpy buffer)
            self.embedding.weight.data.copy_(
                torch.as_tensor(pretrained_embeddings, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = True  # keep the embedding layer trainable

        # Project embeddings to d_model if the dimensions differ
        if embedding_dim != d_model:
            self.projection = nn.Linear(embedding_dim, d_model)
        else:
            self.projection = nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )

        self.d_model = d_model
        # Initialize randomly only when no pre-trained embeddings were copied in
        self.init_weights(init_embedding=pretrained_embeddings is None)

    def init_weights(self, init_embedding=True):
        """Initialize model parameters."""
        initrange = 0.1
        if init_embedding:
            self.embedding.weight.data.uniform_(-initrange, initrange)
        if isinstance(self.projection, nn.Linear):
            self.projection.weight.data.uniform_(-initrange, initrange)
            self.projection.bias.data.zero_()

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text) * math.sqrt(self.d_model)  # [batch_size, seq_len, embedding_dim]
        embedded = self.projection(embedded)  # [batch_size, seq_len, d_model]
        embedded = self.pos_encoder(embedded)

        # Mask out padding positions
        src_key_padding_mask = (text == 0)  # [batch_size, seq_len]

        output = embedded
        for transformer_block in self.transformer_blocks:
            output = transformer_block(output, src_key_padding_mask)

        output = self.final_norm(output)

        # Mean-pool over real (non-padding) positions only
        token_mask = (~src_key_padding_mask).float().unsqueeze(-1)  # [batch, seq_len, 1]
        output = (output * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)

        return self.classifier(output)
