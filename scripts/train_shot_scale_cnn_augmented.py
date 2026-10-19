from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader

SEED = 20261013
CLASSES = ["CS", "ECS", "FS", "LS", "MS"]
IMAGE_SIZE = 128
BATCH_SIZE = 32
MAX_EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 4
NUM_WORKERS = 0

ROOT = Path("data/processed")
TRAIN_DIR = ROOT / "shot_scale_frames"
VAL_DIR = ROOT / "shot_scale_validation_frames"
OUTPUT_DIR = ROOT / "models" / "shot_scale_cnn_augmented"

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(f"Output folder is not empty: {OUTPUT_DIR}")
if not (Path(__file__).resolve().parent / "train_shot_scale_cnn.py").is_file():
    raise FileNotFoundError("Baseline training script not found.")
if not (ROOT / "models" / "shot_scale_cnn" / "best_model.pt").is_file():
    raise FileNotFoundError("Baseline checkpoint not found.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
DEVICE = torch.device("cpu")


class FrameDataset(Dataset):
    def __init__(self, root, augment=False):
        self.root = Path(root)
        self.augment = augment
        self.items = []
        self.class_to_idx = {name: i for i, name in enumerate(CLASSES)}

        expected_per_class = 500 if self.root == TRAIN_DIR else 100
        for label in CLASSES:
            files = sorted(
                p for p in (self.root / label).glob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if len(files) != expected_per_class:
                raise ValueError(
                    f"{self.root}/{label}: expected {expected_per_class}, found {len(files)}"
                )
            self.items.extend((p, self.class_to_idx[label]) for p in files)

        expected_total = 2500 if self.root == TRAIN_DIR else 500
        if len(self.items) != expected_total:
            raise ValueError(
                f"Expected {expected_total} images in {self.root}, found {len(self.items)}"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(0.90, 1.10)
                )
            if random.random() < 0.5:
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(0.90, 1.10)
                )

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor, label


class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = FrameDataset(TRAIN_DIR, augment=True)
    val_dataset = FrameDataset(VAL_DIR, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = CustomCNN(len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {DEVICE}")
    print(f"Training images: {len(train_dataset)} (augmentation enabled)")
    print(f"Validation images: {len(val_dataset)} (augmentation disabled)")
    print(f"Classes: {CLASSES}")
    print(f"Maximum epochs: {MAX_EPOCHS}; early-stopping patience: {PATIENCE}")

    history = []
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        epoch_seconds = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "epoch_seconds": epoch_seconds,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"time={epoch_seconds:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": IMAGE_SIZE,
                    "normalization_mean": [0.5, 0.5, 0.5],
                    "normalization_std": [0.5, 0.5, 0.5],
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "best_validation_accuracy": best_val_accuracy,
                    "seed": SEED,
                    "augmentation": {
                        "horizontal_flip_probability": 0.5,
                        "brightness_range": [0.90, 1.10],
                        "contrast_range": [0.90, 1.10],
                    },
                },
                OUTPUT_DIR / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    total_seconds = time.time() - start

    with (OUTPUT_DIR / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "model": "CustomCNN with training-time augmentation",
        "device": str(DEVICE),
        "classes": CLASSES,
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "max_epochs": MAX_EPOCHS,
        "epochs_completed": len(history),
        "early_stopping_patience": PATIENCE,
        "best_epoch_by_validation_loss": best_epoch,
        "best_validation_loss": best_val_loss,
        "validation_accuracy_at_best_loss": best_val_accuracy,
        "total_training_seconds": total_seconds,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE,
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "brightness_range": [0.90, 1.10],
            "contrast_range": [0.90, 1.10],
        },
    }

    with (OUTPUT_DIR / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nExperiment 2 training complete.")
    print(f"Best epoch by validation loss: {best_epoch}")
    print(f"Validation loss at best epoch: {best_val_loss:.4f}")
    print(f"Validation accuracy at best epoch: {best_val_accuracy:.4f}")
    print(f"Total training time: {total_seconds / 60:.2f} minutes")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
