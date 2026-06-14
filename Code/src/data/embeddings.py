import os
import zipfile

import numpy as np
import requests
import torch
import torch.nn as nn
from tqdm import tqdm

from config import PRETRAINED_DIR, GLOVE_PATH, TRAIN_CONFIG

GLOVE_URL = 'https://nlp.stanford.edu/data/glove.6B.zip'
DOWNLOAD_CHUNK_SIZE = 1024


class GloVeEmbeddings:
    def __init__(self, glove_path=GLOVE_PATH):
        self.glove_path = glove_path
        self.embedding_dim = TRAIN_CONFIG['embedding_dim']
        self.embeddings_dict = {}
        self._download_if_needed()
        self._load_embeddings()

    def _download_if_needed(self):
        """Download and extract GloVe embeddings if they are missing."""
        if os.path.exists(self.glove_path):
            print(f"GloVe embeddings already exist: {self.glove_path}")
            return

        print(f"GloVe embeddings not found: {self.glove_path}")
        print("Downloading GloVe embeddings...")
        os.makedirs(PRETRAINED_DIR, exist_ok=True)

        zip_path = os.path.join(PRETRAINED_DIR, 'glove.6B.zip')
        if not os.path.exists(zip_path):
            response = requests.get(GLOVE_URL, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading GloVe embeddings",
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    size = f.write(data)
                    pbar.update(size)

        print("Extracting GloVe embeddings...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(PRETRAINED_DIR)

        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("GloVe embeddings ready!")

    def _load_embeddings(self):
        """Load GloVe embeddings into memory."""
        print(f"Loading GloVe embeddings from {self.glove_path}...")
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading embeddings"):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings_dict[word] = vector
        print(f"Loaded {len(self.embeddings_dict)} word vectors")

    def load_embeddings(self, vocab):
        """Build the embedding matrix for a vocabulary."""
        print("Building embedding matrix...")
        embedding_matrix = np.zeros((len(vocab), self.embedding_dim), dtype=np.float32)
        unknown_words = []

        for word, idx in tqdm(vocab.items(), desc="Processing vocabulary"):
            if word == '<pad>':
                continue  # padding must stay an all-zero vector
            if word in self.embeddings_dict:
                embedding_matrix[idx] = self.embeddings_dict[word]
            else:
                # Randomly initialize vectors for out-of-vocabulary words
                embedding_matrix[idx] = np.random.normal(
                    scale=0.6, size=(self.embedding_dim,)
                ).astype(np.float32)
                unknown_words.append(word)

        if unknown_words:
            print(f"Found {len(unknown_words)} unknown words, using random initialization")
            print("First 10 unknown words:", unknown_words[:10])

        return embedding_matrix


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None, trainable=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if pretrained_embeddings is not None:
            if isinstance(pretrained_embeddings, torch.Tensor):
                pretrained_embeddings = pretrained_embeddings.numpy()

            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

            if not trainable:
                # Freeze gradients only for pre-trained words; randomly
                # initialized (unknown) words keep training
                pretrained_mask = torch.ones_like(self.embedding.weight.data)
                for i in range(vocab_size):
                    if np.all(pretrained_embeddings[i] == 0):
                        pretrained_mask[i] = 0

                self.embedding.weight.requires_grad = True
                self.embedding.weight.register_hook(
                    lambda grad: grad * pretrained_mask
                )
        else:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
            if not trainable:
                self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)
