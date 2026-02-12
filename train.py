#!/usr/bin/env python3
"""
训练基础ResNet18模型
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.train_base import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet18 model')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (epochs)')

    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)