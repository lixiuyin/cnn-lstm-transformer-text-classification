import logging
import os
import re
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Negation words carry essential signal for sentiment classification, so they
# are exempt from stopword removal. Includes the contraction stems that
# punctuation stripping produces ("don't" -> "don", "can't" -> "can", ...).
NEGATION_WORDS = {
    'no', 'nor', 'not', 'can',
    'ain', 'aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven',
    'isn', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn',
}


class TextPreprocessor:
    def __init__(self):
        logger.info("Initializing text preprocessor...")

        # Bundled NLTK stopwords list
        nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
        stopwords_path = os.path.join(nltk_data_dir, 'stopwords', 'english')
        if not os.path.exists(stopwords_path):
            raise FileNotFoundError(f"Stopwords file not found: {stopwords_path}")
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            self.stop_words = set(f.read().splitlines()) - NEGATION_WORDS
        logger.info(f"Loaded {len(self.stop_words)} stopwords ({len(NEGATION_WORDS)} negation words exempt)")

        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z\s]')
        self.whitespace_pattern = re.compile(r'\s+')

    def preprocess(self, text: str) -> str:
        """Preprocess a single text."""
        if not isinstance(text, str):
            return ""
        # 1. Lowercase
        text = text.lower()
        # 2. Remove HTML tags and URLs
        text = self.html_pattern.sub(' ', text)
        text = self.url_pattern.sub(' ', text)
        # 3. Remove special characters, keep letters and whitespace
        text = self.special_chars_pattern.sub(' ', text)
        # 4. Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        # 5. Tokenize: only letters and spaces remain, so splitting suffices
        tokens = text.split()
        # 6. Remove stopwords and single-character tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]

        return ' '.join(tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in tqdm(texts, desc="Preprocessing texts")]

    def build_vocab(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
        """Build the vocabulary from texts."""
        logger.info("Building vocabulary...")
        word_freq = {}
        logger.info("Counting word frequencies...")
        for text in tqdm(texts, desc="Processing texts"):
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        logger.info("Filtering low-frequency words...")
        word_freq = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
        logger.info(f"{len(word_freq)} words remain after filtering")

        # Special tokens
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3
        }

        for word in tqdm(sorted(word_freq.keys()), desc="Adding words to vocabulary"):
            if word not in vocab:
                vocab[word] = len(vocab)

        logger.info(f"Vocabulary built with {len(vocab)} words")
        return vocab

    def create_dataset(self, texts, labels, vocab=None, max_length=200):
        """Create a TextDataset, building the vocabulary if not provided."""
        logger.info("Creating dataset...")
        if vocab is None:
            logger.info("No vocabulary provided, building one...")
            vocab = self.build_vocab(texts)

        dataset = TextDataset(texts, labels, vocab, max_length)
        logger.info("Dataset created")
        return dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=200):
        self.texts = texts
        self.labels = list(labels)  # copy: never alias the caller's list
        self.vocab = vocab
        self.max_length = max_length

        # Normalize labels once: map strings to integers, shift numeric
        # labels to start from 0
        self.has_string_labels = isinstance(self.labels[0], str)
        if self.has_string_labels:
            unique_labels = sorted(set(self.labels))
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
            logger.info(f"Label mapping: {self.label_map}")
        else:
            min_label = min(self.labels)
            if min_label != 0:
                self.labels = [label - min_label for label in self.labels]
                logger.info(
                    f"Shifted labels to start from 0 (was starting at {min_label})"
                )

    def text_to_indices(self, text):
        """Convert a text into a fixed-length sequence of vocabulary indices."""
        words = text.split()
        indices = []
        for word in words[:self.max_length]:  # truncate long sequences
            indices.append(self.vocab.get(word, self.vocab['<unk>']))

        if not indices:
            # Texts that become empty after preprocessing must keep one real
            # token: an all-padding row makes attention softmax produce NaN
            indices.append(self.vocab['<unk>'])

        # Pad short sequences
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<pad>']] * (self.max_length - len(indices)))

        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        indices = self.text_to_indices(text)

        if self.has_string_labels:
            label = self.label_map[label]
        return {
            'text': indices,
            'label': torch.tensor(label, dtype=torch.long)
        }
