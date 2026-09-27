from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .config import LABELS
from .dataset import CoughDataset, discover_files
from .model import CoughCNN


def train(data_dir: str, output: str, epochs: int = 20, batch_size: int = 16, seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    files = discover_files(data_dir)
    if len(files) < 2:
        raise ValueError("Need at least two WAV files under labeled data folders.")
    generator = torch.Generator().manual_seed(seed)
    validation_size = max(1, int(len(files) * 0.2))
    train_size = len(files) - validation_size
    train_files, validation_files = random_split(files, [train_size, validation_size], generator=generator)
    train_set = CoughDataset(list(train_files), augment=True)
    validation_set = CoughDataset(list(validation_files))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    counts = np.bincount([LABELS.index(path.parent.name.lower()) for path in files], minlength=len(LABELS))
    weights = len(files) / (len(LABELS) * np.maximum(counts, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoughCNN(len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_accuracy = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for features, targets in validation_loader:
                predictions = model(features.to(device)).argmax(dim=1).cpu()
                correct += int((predictions == targets).sum())
                total += len(targets)
        accuracy = correct / max(1, total)
        print(f"epoch={epoch:03d} val_accuracy={accuracy:.3f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.cpu().state_dict(), "labels": LABELS, "accuracy": accuracy}, destination)
            model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the cough CNN")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="artifacts/cough_cnn.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train(args.data_dir, args.output, args.epochs, args.batch_size)
