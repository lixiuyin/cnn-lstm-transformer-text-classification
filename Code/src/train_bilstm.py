import argparse

from models.bilstm import BiLSTMClassifier
from training import train_model


def train_bilstm(device=None, data=None):
    return train_model('bilstm', BiLSTMClassifier, 'BiLSTM', device=device, data=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the BiLSTM model')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (default: auto-detect)')
    args = parser.parse_args()

    train_bilstm(device=args.device)
