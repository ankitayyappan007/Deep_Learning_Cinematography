import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent / "data" / "processed" / "models"
OUTPUT_DIR = ROOT / "experiment_comparison"
OUTPUT_FILE = OUTPUT_DIR / "experiment_comparison.json"
CLASSES = ["CS", "ECS", "FS", "LS", "MS"]
TOL = 1e-9


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required source file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required source file missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a, b):
    return abs(float(a) - float(b)) <= TOL


def require(condition, message):
    if not condition:
        raise ValueError("VERIFICATION FAILED: " + message)


def read_experiment(name, folder, augmented):
    model_dir = ROOT / folder
    train_summary_path = model_dir / "training_summary.json"
    history_path = model_dir / "training_history.csv"
    eval_summary_path = model_dir / "test_evaluation" / "evaluation_summary.json"
    report_path = model_dir / "test_evaluation" / "classification_report.csv"
    matrix_path = model_dir / "test_evaluation" / "confusion_matrix.csv"
    checkpoint_path = model_dir / "best_model.pt"

    source_paths = [
        train_summary_path, history_path, eval_summary_path,
        report_path, matrix_path, checkpoint_path,
    ]
    for path in source_paths:
        require(path.is_file(), f"{name}: missing source artifact {path}")

    train = load_json(train_summary_path)
    evaluation = load_json(eval_summary_path)
    history = load_csv(history_path)
    report_rows = load_csv(report_path)
    matrix_rows = load_csv(matrix_path)

    require(train["classes"] == CLASSES, f"{name}: training class order mismatch")
    require(evaluation["classes"] == CLASSES, f"{name}: evaluation class order mismatch")
    require(int(train["train_images"]) == 2500, f"{name}: unexpected training size")
    require(int(train["validation_images"]) == 500, f"{name}: unexpected validation size")
    require(int(evaluation["test_frames"]) == 500, f"{name}: unexpected test size")
    require(len(history) == int(train["epochs_completed"]),
            f"{name}: history length differs from completed epochs")
    require(len(report_rows) == len(CLASSES), f"{name}: per-class report row count mismatch")
    require([r["class"] for r in report_rows] == CLASSES,
            f"{name}: per-class report order mismatch")
    require(len(matrix_rows) == len(CLASSES), f"{name}: confusion matrix row count mismatch")
    require([r["actual/predicted"] for r in matrix_rows] == CLASSES,
            f"{name}: confusion matrix class order mismatch")

    per_class = {}
    for row in report_rows:
        per_class[row["class"]] = {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1_score": float(row["f1_score"]),
            "support": int(row["support"]),
        }

    confusion = {
        row["actual/predicted"]: {
            label: int(row[label]) for label in CLASSES
        }
        for row in matrix_rows
    }

    for label in CLASSES:
        require(per_class[label]["support"] == 100,
                f"{name}: unexpected test support for {label}")
        require(sum(confusion[label].values()) == 100,
                f"{name}: confusion matrix row total mismatch for {label}")

    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in CLASSES)
    matrix_accuracy = correct / total
    require(total == int(evaluation["test_frames"]),
            f"{name}: confusion matrix total differs from test size")
    require(close(matrix_accuracy, evaluation["test_accuracy"]),
            f"{name}: confusion matrix accuracy differs from evaluation summary")

    mean_precision = sum(per_class[c]["precision"] for c in CLASSES) / len(CLASSES)
    mean_recall = sum(per_class[c]["recall"] for c in CLASSES) / len(CLASSES)
    mean_f1 = sum(per_class[c]["f1_score"] for c in CLASSES) / len(CLASSES)
    require(close(mean_precision, evaluation["macro_precision"]),
            f"{name}: report precision mean differs from saved macro precision")
    require(close(mean_recall, evaluation["macro_recall"]),
            f"{name}: report recall mean differs from saved macro recall")
    require(close(mean_f1, evaluation["macro_f1"]),
            f"{name}: report F1 mean differs from saved macro F1")

    if augmented:
        selected_epoch = int(train["best_epoch_by_validation_loss"])
        selected_row = next(
            (row for row in history if int(row["epoch"]) == selected_epoch), None
        )
        require(selected_row is not None, f"{name}: selected epoch missing from history")
        require(close(selected_row["validation_loss"], train["best_validation_loss"]),
                f"{name}: selected validation loss differs from history")
        require(close(selected_row["validation_accuracy"],
                      train["validation_accuracy_at_best_loss"]),
                f"{name}: selected validation accuracy differs from history")
        require(int(evaluation["selected_epoch"]) == selected_epoch,
                f"{name}: evaluation selected epoch mismatch")
    else:
        selected_epoch = int(train["best_epoch"])
        selected_row = next(
            (row for row in history if int(row["epoch"]) == selected_epoch), None
        )
        require(selected_row is not None, f"{name}: selected epoch missing from history")
        require(close(selected_row["validation_accuracy"],
                      train["best_validation_accuracy"]),
                f"{name}: selected validation accuracy differs from history")
        require(int(evaluation["selected_epoch"]) == selected_epoch,
                f"{name}: evaluation selected epoch mismatch")

    require(close(evaluation["total_training_seconds"], train["total_training_seconds"])
            if "total_training_seconds" in evaluation else True,
            f"{name}: training duration mismatch")

    metrics = {
        "test_accuracy": float(evaluation["test_accuracy"]),
        "macro_precision": float(evaluation["macro_precision"]),
        "macro_recall": float(evaluation["macro_recall"]),
        "macro_f1": float(evaluation["macro_f1"]),
        "weighted_f1": float(evaluation["weighted_f1"]),
        "selected_epoch": selected_epoch,
        "training_duration_seconds": float(train["total_training_seconds"]),
        "evaluation_duration_seconds": float(evaluation["evaluation_seconds"]),
        "test_frames": int(evaluation["test_frames"]),
        "train_frames": int(train["train_images"]),
        "validation_frames": int(train["validation_images"]),
        "epochs_completed": int(train["epochs_completed"]),
        "best_validation_accuracy": float(evaluation["best_validation_accuracy"]),
        "device": train["device"],
        "seed": int(train["seed"]),
        "batch_size": int(train["batch_size"]),
        "learning_rate": float(train["learning_rate"]),
        "image_size": int(train["image_size"]),
    }

    if augmented:
        metrics["best_validation_loss"] = float(train["best_validation_loss"])
        metrics["early_stopping_patience"] = int(train["early_stopping_patience"])
        augmentation = train["augmentation"]
        require(augmentation == {
            "horizontal_flip_probability": 0.5,
            "brightness_range": [0.9, 1.1],
            "contrast_range": [0.9, 1.1],
        }, "Experiment 2: augmentation settings differ from expected saved configuration")
    else:
        augmentation = None

    source_hashes = {str(p): sha256(p) for p in source_paths}

    return {
        "model": evaluation["model"],
        "checkpoint": str(checkpoint_path),
        "source_artifacts": {
            "training_summary": str(train_summary_path),
            "training_history": str(history_path),
            "evaluation_summary": str(eval_summary_path),
            "classification_report": str(report_path),
            "confusion_matrix": str(matrix_path),
        },
        "source_sha256": source_hashes,
        "metrics": metrics,
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "row_definition": "actual class",
            "column_definition": "predicted class",
            "class_order": CLASSES,
            "counts": confusion,
        },
        "augmentation": augmentation,
        "verification": {
            "confusion_matrix_accuracy_matches_summary": True,
            "per_class_macro_averages_match_summary": True,
            "class_supports_and_matrix_totals_verified": True,
            "selected_epoch_matches_training_artifacts": True,
        },
    }


def main():
    require(OUTPUT_DIR.is_dir(), f"Comparison folder missing: {OUTPUT_DIR}")
    require(not OUTPUT_FILE.exists(),
            f"Output already exists; refusing to overwrite: {OUTPUT_FILE}")

    baseline = read_experiment("Baseline CNN", "shot_scale_cnn", False)
    augmented = read_experiment(
        "Augmented CNN", "shot_scale_cnn_augmented", True
    )

    require(baseline["metrics"]["test_frames"] == augmented["metrics"]["test_frames"],
            "The two experiments use different test-set sizes")
    require(baseline["confusion_matrix"]["class_order"]
            == augmented["confusion_matrix"]["class_order"],
            "The two experiments use different class orders")

    result = {
        "title": "Shot-Scale CNN Experiment Comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "task": "Five-class cinematic shot-scale classification",
            "classes": CLASSES,
            "test_set_size_per_experiment": 500,
            "note": "Values are read from saved artifacts; no experiment was rerun.",
        },
        "experiments": {
            "baseline_cnn": baseline,
            "augmented_cnn": augmented,
        },
        "comparison": {
            "test_accuracy_difference_augmented_minus_baseline":
                augmented["metrics"]["test_accuracy"] - baseline["metrics"]["test_accuracy"],
            "macro_f1_difference_augmented_minus_baseline":
                augmented["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"],
            "best_by_test_accuracy": (
                "baseline_cnn"
                if baseline["metrics"]["test_accuracy"] > augmented["metrics"]["test_accuracy"]
                else "augmented_cnn"
                if augmented["metrics"]["test_accuracy"] > baseline["metrics"]["test_accuracy"]
                else "tie"
            ),
            "best_by_macro_f1": (
                "baseline_cnn"
                if baseline["metrics"]["macro_f1"] > augmented["metrics"]["macro_f1"]
                else "augmented_cnn"
                if augmented["metrics"]["macro_f1"] > baseline["metrics"]["macro_f1"]
                else "tie"
            ),
        },
        "verification": {
            "source_artifacts_present": True,
            "all_metric_consistency_checks_passed": True,
            "original_artifacts_modified": False,
        },
    }

    with OUTPUT_FILE.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("Comparison file created and source checks passed.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Baseline test accuracy: {baseline['metrics']['test_accuracy']:.4%}")
    print(f"Augmented test accuracy: {augmented['metrics']['test_accuracy']:.4%}")
    print(f"Baseline macro F1: {baseline['metrics']['macro_f1']:.4%}")
    print(f"Augmented macro F1: {augmented['metrics']['macro_f1']:.4%}")
    print(f"Baseline training duration: {baseline['metrics']['training_duration_seconds']:.2f} seconds")
    print(f"Augmented training duration: {augmented['metrics']['training_duration_seconds']:.2f} seconds")
    print("Per-class metrics, both confusion matrices, augmentation settings, and source hashes included.")
    print("Original checkpoints and reports were read only.")


if __name__ == "__main__":
    main()
