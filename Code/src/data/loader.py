import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)

# The Hugging Face Hub client logs every HTTP request at INFO level
logging.getLogger('httpx').setLevel(logging.WARNING)


def load_split(dataset_config, split):
    """Load one split of a dataset from the Hugging Face Hub.

    Returns (texts, labels) as plain Python lists.
    """
    hf_path = dataset_config['hf_path']
    logger.info(f"Loading '{hf_path}' split '{split}' from the Hugging Face Hub...")
    dataset = load_dataset(hf_path, split=split)

    text_column = dataset_config['text_column']
    label_column = dataset_config['label_column']
    for column in (text_column, label_column):
        if column not in dataset.column_names:
            raise ValueError(
                f"Column '{column}' not found in '{hf_path}' "
                f"(available columns: {dataset.column_names})"
            )

    texts = dataset[text_column]
    labels = dataset[label_column]
    logger.info(f"Loaded {len(texts)} samples from '{hf_path}' split '{split}'")
    return texts, labels
