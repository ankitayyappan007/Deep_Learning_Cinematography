import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

CLASSES = ["CS", "ECS", "FS", "LS", "MS"]
IMAGE_SIZE = 128
BATCH_SIZE = 32
ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
TEST_DIR = ROOT / "shot_scale_test_frames"
MODEL_DIR = ROOT / "models" / "shot_scale_cnn_augmented"
CHECKPOINT = MODEL_DIR / "best_model.pt"
RESULTS_DIR = MODEL_DIR / "test_evaluation"
DEVICE = torch.device("cpu")


class FrameDataset(Dataset):
    def __init__(self, root):
        self.items = []
        self.class_to_idx = {name: i for i, name in enumerate(CLASSES)}

        for label in CLASSES:
            files = sorted(
                p for p in (root / label).glob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if len(files) != 100:
                raise ValueError(
                    f"{root / label}: expected 100 frames, found {len(files)}"
                )
            self.items.extend((p, self.class_to_idx[label]) for p in files)

        if len(self.items) != 500:
            raise ValueError(f"Expected 500 test frames, found {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor, label, str(path)


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


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test frames not found: {TEST_DIR}")
    if RESULTS_DIR.exists():
        raise FileExistsError(
            f"Results directory already exists: {RESULTS_DIR}. "
            "Nothing was overwritten."
        )

    dataset = FrameDataset(TEST_DIR)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    if checkpoint.get("classes") != CLASSES:
        raise ValueError("Checkpoint class mapping does not match.")

    model = CustomCNN(len(CLASSES)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred, frame_paths = [], [], []
    start = time.time()

    with torch.no_grad():
        for images, labels, paths in loader:
            predictions = model(images.to(DEVICE)).argmax(dim=1).cpu().tolist()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions)
            frame_paths.extend(paths)

    evaluation_seconds = time.time() - start
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0
    )
    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASSES)))
    )
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, output_dict=True, zero_division=0
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=False)

    with (RESULTS_DIR / "classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["class", "precision", "recall", "f1_score", "support"]
        )
        writer.writeheader()
        for i, label in enumerate(CLASSES):
            writer.writerow({
                "class": label,
                "precision": precision[i],
                "recall": recall[i],
                "f1_score": f1[i],
                "support": int(support[i]),
            })

    with (RESULTS_DIR / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted"] + CLASSES)
        for i, label in enumerate(CLASSES):
            writer.writerow([label] + matrix[i].tolist())

    with (RESULTS_DIR / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["frame_path", "actual", "predicted", "correct"])
        for path, actual, predicted in zip(frame_paths, y_true, y_pred):
            writer.writerow([
                path, CLASSES[actual], CLASSES[predicted], actual == predicted
            ])

    summary = {
        "model": "CustomCNN with training-time augmentation",
        "checkpoint": str(CHECKPOINT),
        "selected_epoch": checkpoint["best_epoch"],
        "best_validation_accuracy": checkpoint["best_validation_accuracy"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "test_frames": len(dataset),
        "test_accuracy": accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "evaluation_seconds": evaluation_seconds,
        "classes": CLASSES,
    }

    with (RESULTS_DIR / "evaluation_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nEXPERIMENT 2: HELD-OUT TEST RESULTS")
    print(f"Test frames: {len(dataset)}")
    print(f"Selected epoch: {checkpoint['best_epoch']}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro precision: {summary['macro_precision']:.4f}")
    print(f"Macro recall: {summary['macro_recall']:.4f}")
    print(f"Macro F1-score: {summary['macro_f1']:.4f}")
    print(f"Weighted F1-score: {summary['weighted_f1']:.4f}")
    print(f"Evaluation time: {evaluation_seconds:.2f} seconds")

    print("\nPER-CLASS REPORT")
    for label in CLASSES:
        row = report[label]
        print(
            f"{label}: precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"F1={row['f1-score']:.4f}, "
            f"support={int(row['support'])}"
        )

    print("\nCONFUSION MATRIX (rows=actual, columns=predicted)")
    print("       " + " ".join(f"{c:>6}" for c in CLASSES))
    for label, row in zip(CLASSES, matrix):
        print(f"{label:>6} " + " ".join(f"{v:6d}" for v in row))

    print(f"\nSaved results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
