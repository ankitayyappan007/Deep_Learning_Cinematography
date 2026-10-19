import argparse
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

SEED = 20261014
CLASSES = ["Static", "Motion", "Push", "Pull", "Multi_movement"]
SPLITS = ["train", "validation", "test"]
EXPECTED_COUNTS = {"train": 30288, "validation": 1683, "test": 1682}
FRAME_SHAPE = (5, 112, 112, 3)
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 4
NUM_WORKERS = 0
CPU_THREADS = 2
PROGRESS_INTERVAL = 25

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "processed" / "camera_movement_split.csv"
FRAME_DIR = ROOT / "data" / "processed" / "camera_movement_frames_5"
OUTPUT_DIR = ROOT / "data" / "processed" / "models" / "camera_movement_cnn"
CHECKPOINT_PATH = OUTPUT_DIR / "latest_checkpoint.pt"
BEST_PATH = OUTPUT_DIR / "best_model.pt"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"

CONFIG = {
    "model": "LightweightFrameDifferenceCNN",
    "classes": CLASSES,
    "input": "four consecutive signed RGB frame differences",
    "frame_shape": list(FRAME_SHAPE),
    "input_channels": 12,
    "batch_size": BATCH_SIZE,
    "max_epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "early_stopping_patience": PATIENCE,
    "seed": SEED,
    "cpu_threads": CPU_THREADS,
    "num_workers": NUM_WORKERS,
    "selection_metric": "validation_macro_f1",
    "loss": "square_root_inverse_frequency_weighted_cross_entropy",
    "device": "cpu",
    "label_warning": (
        "Original MovieShots labels only. These are not mapped to "
        "Pan, Tilt, Zoom, or Tracking/Dolly."
    ),
    "tradeoff": (
        "Frame differences reduce input size and emphasize temporal change, "
        "but discard much static scene context and may miss subtle movement."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from this pipeline's existing latest checkpoint.",
    )
    return parser.parse_args()


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)


def validate_manifest():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not FRAME_DIR.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {FRAME_DIR}")

    df = pd.read_csv(MANIFEST_PATH, dtype={"imdb_id": str})
    required = {"imdb_id", "shot_id", "movement", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(df.columns))}")

    if len(df) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"Expected 33653 manifest rows; found {len(df)}")
    if df.duplicated(["imdb_id", "shot_id"]).any():
        raise ValueError("Duplicate movie/shot keys found.")
    if set(df["movement"].unique()) != set(CLASSES):
        raise ValueError("Labels do not match the five original MovieShots classes.")
    if set(df["split"].unique()) != set(SPLITS):
        raise ValueError("Unexpected split names found.")

    counts = df["split"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected split counts: {counts}")

    overlap = {}
    movie_sets = {
        split: set(df.loc[df["split"] == split, "imdb_id"])
        for split in SPLITS
    }
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap[f"{a}_{b}"] = len(movie_sets[a] & movie_sets[b])
    if any(overlap.values()):
        raise ValueError(f"Movie overlap between splits: {overlap}")

    print("Manifest metadata validation passed.")
    print(f"Rows: {len(df)} | Split counts: {counts}")
    print("Movie overlap:", overlap)
    print("No frame archives were opened during metadata validation.")
    return df


class DifferenceDataset(Dataset):
    def __init__(self, frame):
        self.items = list(
            frame[["imdb_id", "shot_id", "movement"]].itertuples(index=False, name=None)
        )
        self.class_to_idx = {label: i for i, label in enumerate(CLASSES)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        imdb_id, shot_id, movement = self.items[index]
        filename = f"{imdb_id}_shot_{int(shot_id):04d}.npz"
        path = FRAME_DIR / filename

        try:
            with np.load(path, allow_pickle=False) as archive:
                frames = archive["frames"]
                if frames.shape != FRAME_SHAPE or frames.dtype != np.uint8:
                    raise ValueError(f"Unexpected array format: {path}")
                # Copy is limited to this sample. No full-dataset RAM cache.
                frames = frames.astype(np.float32, copy=True)
        except Exception as exc:
            raise RuntimeError(f"Cannot load verified frame array {path}: {exc}") from exc

        # Signed differences retain temporal order and motion direction.
        # Four RGB differences become 12 channels.
        differences = np.diff(frames, axis=0) / 255.0
        differences = differences.transpose(3, 0, 1, 2).reshape(12, 112, 112)
        x = torch.from_numpy(np.ascontiguousarray(differences, dtype=np.float32))
        y = self.class_to_idx[movement]
        return x, y


class LightweightDifferenceCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def class_weights(train_df):
    counts = train_df["movement"].value_counts()
    n = len(train_df)
    # Square-root inverse frequency moderates extreme rare-class weights.
    weights = [(n / (len(CLASSES) * counts[label])) ** 0.5 for label in CLASSES]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights / weights.mean()


def atomic_torch_save(payload, destination):
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(destination)


def write_history(history):
    temp_path = HISTORY_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    temp_path.replace(HISTORY_PATH)


def run_epoch(model, loader, criterion, optimizer, epoch, total_epochs, phase):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0
    all_true = []
    all_pred = []
    start = time.time()
    total_batches = len(loader)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (inputs, labels) in enumerate(loader, start=1):
            inputs = inputs.to("cpu")
            labels = labels.to("cpu")

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            seen += batch_n
            all_true.extend(labels.tolist())
            all_pred.extend(predictions.tolist())

            if batch_index % PROGRESS_INTERVAL == 0 or batch_index == total_batches:
                elapsed = time.time() - start
                rate = seen / max(elapsed, 1e-6)
                remaining = (len(loader.dataset) - seen) / max(rate, 1e-6)
                print(
                    f"{phase} epoch {epoch}/{total_epochs}: "
                    f"{seen}/{len(loader.dataset)} samples, "
                    f"elapsed={elapsed / 60:.1f} min, "
                    f"estimated remaining={remaining / 60:.1f} min",
                    flush=True,
                )

    macro_f1 = f1_score(
        all_true, all_pred, labels=list(range(len(CLASSES))),
        average="macro", zero_division=0
    )
    return {
        "loss": total_loss / max(seen, 1),
        "accuracy": correct / max(seen, 1),
        "macro_f1": macro_f1,
        "y_true": all_true,
        "y_pred": all_pred,
        "seconds": time.time() - start,
    }


def evaluate_and_save(model, loader, criterion, output_dir):
    result = run_epoch(model, loader, criterion, None, 0, 0, "TEST")
    y_true, y_pred = result["y_true"], result["y_pred"]

    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASSES)))
    )
    summary = {
        "model": CONFIG["model"],
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(
            y_true, y_pred, labels=list(range(len(CLASSES))),
            average="macro", zero_division=0
        ),
        "test_weighted_f1": f1_score(
            y_true, y_pred, labels=list(range(len(CLASSES))),
            average="weighted", zero_division=0
        ),
        "test_loss": result["loss"],
        "test_samples": len(y_true),
        "classes": CLASSES,
        "test_support_by_class": {
            label: int((np.array(y_true) == i).sum())
            for i, label in enumerate(CLASSES)
        },
        "classification_report": report,
        "confusion_matrix_labels_order": CLASSES,
        "confusion_matrix": matrix.tolist(),
        "label_warning": CONFIG["label_warning"],
        "tradeoff": CONFIG["tradeoff"],
    }

    with (output_dir / "test_evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(
        matrix, index=CLASSES, columns=CLASSES
    ).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv"
    )
    pd.DataFrame({
        "true_label": [CLASSES[i] for i in y_true],
        "predicted_label": [CLASSES[i] for i in y_pred],
    }).to_csv(output_dir / "test_predictions.csv", index=False)

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Accuracy: {summary['test_accuracy']:.4f}")
    print(f"Macro F1: {summary['test_macro_f1']:.4f}")
    print(f"Weighted F1: {summary['test_weighted_f1']:.4f}")
    print("\nPer-class metrics:")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, zero_division=0
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_string())
    print("\nSaved final evaluation artifacts.")


def main():
    args = parse_args()

    if args.show_config:
        print(json.dumps(CONFIG, indent=2))
        return

    set_seed()
    manifest = validate_manifest()
    if args.validate_only:
        print("Validation-only mode: no frame arrays opened; no training started.")
        return

    if args.resume:
        if not OUTPUT_DIR.is_dir() or not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                "--resume requires an existing output directory and latest checkpoint."
            )
    elif OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_DIR}. "
            "Use --resume only for a checkpoint created by this script."
        )

    # Training creates its own output folder only after preflight checks.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=args.resume)
    config_path = OUTPUT_DIR / "config.json"
    if args.resume:
        if not config_path.is_file():
            raise FileExistsError("Cannot safely resume: config.json is missing.")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != CONFIG:
            raise ValueError("Configuration mismatch; refusing unsafe resume.")
    else:
        with config_path.open("x", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=2)

    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "validation"].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    train_ds = DifferenceDataset(train_df)
    val_ds = DifferenceDataset(val_df)
    test_ds = DifferenceDataset(test_df)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )

    model = LightweightDifferenceCNN(len(CLASSES)).to("cpu")
    weights = class_weights(train_df)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history = []
    start_epoch = 1
    best_f1 = -1.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    if args.resume:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        if checkpoint.get("classes") != CLASSES or checkpoint.get("config") != CONFIG:
            raise ValueError("Checkpoint metadata mismatch; refusing to resume.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_validation_macro_f1"]
        best_epoch = checkpoint["best_epoch"]
        best_model_state = checkpoint["best_model_state_dict"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        # Restore the best checkpoint from the resumable snapshot as well.
        atomic_torch_save({
            "model_state_dict": best_model_state,
            "classes": CLASSES,
            "config": CONFIG,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "seed": SEED,
        }, BEST_PATH)
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        print(f"Resuming at epoch {start_epoch}; best validation macro F1={best_f1:.4f}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== TRAINING CONFIGURATION ===")
    print(f"Device: CPU | Threads: {CPU_THREADS} | Batch size: {BATCH_SIZE}")
    print(f"Train/validation/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Classes: {CLASSES}")
    print(f"Trainable parameters: {n_params:,}")
    print("Frame differences: 4 signed RGB differences, 12 input channels")
    print("Checkpoint selection: validation macro F1")
    print("Test set will only be evaluated after training/early stopping.")
    print("Progress is printed every 100 batches and at each epoch boundary.")

    overall_start = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        train_result = run_epoch(
            model, train_loader, criterion, optimizer, epoch, EPOCHS, "TRAIN"
        )
        val_result = run_epoch(
            model, val_loader, criterion, None, epoch, EPOCHS, "VALIDATION"
        )
        epoch_seconds = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "train_macro_f1": train_result["macro_f1"],
            "validation_loss": val_result["loss"],
            "validation_accuracy": val_result["accuracy"],
            "validation_macro_f1": val_result["macro_f1"],
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": time.time() - overall_start,
        }
        history.append(record)
        write_history(history)

        improved = val_result["macro_f1"] > best_f1
        if improved:
            best_f1 = val_result["macro_f1"]
            best_epoch = epoch
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            atomic_torch_save({
                "model_state_dict": best_model_state,
                "classes": CLASSES,
                "config": CONFIG,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_f1,
                "seed": SEED,
            }, BEST_PATH)
        else:
            epochs_without_improvement += 1

        # Checkpoint cadence: once per completed epoch, not per batch. This limits I/O overhead.
        # Includes current/best model, optimizer, RNG state, counters, and history.
        atomic_torch_save({
            "model_state_dict": model.state_dict(),
            "best_model_state_dict": best_model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": CLASSES,
            "config": CONFIG,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }, CHECKPOINT_PATH)

        avg_epoch = (time.time() - overall_start) / max(1, len(history))
        remaining_epochs = max(0, EPOCHS - epoch)
        eta_minutes = remaining_epochs * avg_epoch / 60
        print(
            f"\nEpoch {epoch}/{EPOCHS} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"train_acc={record['train_accuracy']:.4f} | "
            f"val_loss={record['validation_loss']:.4f} | "
            f"val_acc={record['validation_accuracy']:.4f} | "
            f"val_macro_f1={record['validation_macro_f1']:.4f} | "
            f"epoch={epoch_seconds / 60:.1f} min | "
            f"elapsed={(time.time() - overall_start) / 60:.1f} min | "
            f"ETA up to {eta_minutes:.1f} min",
            flush=True,
        )
        print(
            f"Checkpoint saved. Best epoch={best_epoch}, "
            f"best validation macro F1={best_f1:.4f}.",
            flush=True,
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping: no macro-F1 improvement for {PATIENCE} epochs.")
            break

    if not BEST_PATH.is_file():
        raise RuntimeError("No best checkpoint was produced; test evaluation cancelled.")

    best_checkpoint = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Never overwrite or repeat a completed test evaluation.
    if (OUTPUT_DIR.joinpath("test_evaluation.json").exists()):
        raise FileExistsError(
            "Test evaluation already exists. Refusing to overwrite or repeat it."
        )

    # Evaluate the test split exactly once, using the selected best checkpoint.
    evaluate_and_save(model, test_loader, criterion, OUTPUT_DIR)

    summary = {
        "config": CONFIG,
        "epochs_completed_this_run": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "total_training_minutes": (time.time() - overall_start) / 60,
        "test_evaluation_file": "test_evaluation.json",
        "rare_class_warning": (
            "Test support is only 16 Pull and 8 Multi_movement clips; "
            "their metrics are statistically unstable."
        ),
    }
    with (OUTPUT_DIR / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining and evaluation complete: {OUTPUT_DIR}")
    print(f"Best epoch: {best_epoch}; validation macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
