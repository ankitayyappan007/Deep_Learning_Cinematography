import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

SEED = 20261013
CLASSES = ["CS", "ECS", "FS", "LS", "MS"]
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_WORKERS = 0

ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
TRAIN_DIR = ROOT / "shot_scale_frames"
VAL_DIR = ROOT / "shot_scale_validation_frames"
OUTPUT_DIR = ROOT / "models" / "shot_scale_cnn"


if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {OUTPUT_DIR}. "
        "Choose a new output directory to preserve existing work."
    )


for folder in [TRAIN_DIR, VAL_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    for label in CLASSES:
        if not (folder / label).is_dir():
            raise FileNotFoundError(f"Class folder not found: {folder / label}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
DEVICE = torch.device("cpu")


class FrameDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.items = []
        self.class_to_idx = {label: i for i, label in enumerate(CLASSES)}

        for label in CLASSES:
            files = sorted((self.root / label).glob("*"))
            files = [
                p for p in files
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            
            
            expected_count = 500 if self.root == TRAIN_DIR else 100
            if len(files) != expected_count:
                raise ValueError(
                    f"{self.root}/{label}: expected {expected_count} images, "
                    f"found {len(files)}"
                )

            
            self.items.extend((p, self.class_to_idx[label]) for p in files)

        
        expected_total = 2500 if self.root == TRAIN_DIR else 500
        if len(self.items) != expected_total:
            raise ValueError(
                f"Expected {expected_total} images in {self.root}, "
                f"found {len(self.items)}"
            )


    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            array = np.asarray(image, dtype=np.float32) / 255.0

        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor, label


class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
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
    total_correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
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
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, total_correct / total


def main():
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    train_dataset = FrameDataset(TRAIN_DIR)
    val_dataset = FrameDataset(VAL_DIR)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = CustomCNN(num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {DEVICE}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Classes: {CLASSES}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    history = []
    best_val_accuracy = -1.0
    best_epoch = 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion
        )

        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "epoch_seconds": elapsed,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": IMAGE_SIZE,
                    "normalization_mean": [0.5, 0.5, 0.5],
                    "normalization_std": [0.5, 0.5, 0.5],
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best_val_accuracy,
                    "seed": SEED,
                },
                OUTPUT_DIR / "best_model.pt",
            )

    total_seconds = time.time() - start

    with (OUTPUT_DIR / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "model": "CustomCNN",
        "device": str(DEVICE),
        "classes": CLASSES,
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "total_training_seconds": total_seconds,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE,
    }

    with (OUTPUT_DIR / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Total training time: {total_seconds / 60:.2f} minutes")
    print(f"Saved model and logs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
