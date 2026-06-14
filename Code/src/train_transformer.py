import argparse

from models.transformer import TransformerClassifier
from training import train_model


def train_transformer(device=None, data=None):
    return train_model('transformer', TransformerClassifier, 'Transformer',
                       device=device, data=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Transformer model')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (default: auto-detect)')
    args = parser.parse_args()

    train_transformer(device=args.device)
