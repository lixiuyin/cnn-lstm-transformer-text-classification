import torch
import torch.nn as nn

from .base import BaseModel


class BiLSTMClassifier(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5,
                 pretrained_embeddings=None, num_classes=2, bidirectional=True, batch_first=True):
        super(BiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            # Explicit copy (no aliasing of the possibly shared numpy buffer)
            self.embedding.weight.data.copy_(
                torch.as_tensor(pretrained_embeddings, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = True  # keep the embedding layer trainable

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)

        # The fully connected layer input depends on directionality
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)

        # Use the hidden states of the last time step
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden)
