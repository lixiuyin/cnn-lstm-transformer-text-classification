import argparse

from models.cnn import CNNClassifier
from training import train_model


def train_cnn(device=None, data=None):
    return train_model('cnn', CNNClassifier, 'CNN', device=device, data=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the CNN model')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (default: auto-detect)')
    args = parser.parse_args()

    train_cnn(device=args.device)
