from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "data" / "processed" / "models"

SHOT_CLASSES = ["CS", "ECS", "FS", "LS", "MS"]
MOVEMENT_CLASSES = ["Static", "Motion", "Push", "Pull", "Multi_movement"]


def format_percent(value):
    if value is None or isinstance(value, bool):
        return "?"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "?"
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return "?"
    return f"{number * 100:.2f}%"

def _read_json(path, warnings):
    if not path.is_file():
        warnings.append(f"Missing artifact: {path.relative_to(ROOT)}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        warnings.append(f"Cannot read {path.relative_to(ROOT)}: {exc}")
        return None


def _read_csv(path, warnings):
    if not path.is_file():
        warnings.append(f"Missing artifact: {path.relative_to(ROOT)}")
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        warnings.append(f"Cannot read {path.relative_to(ROOT)}: {exc}")
        return None


def _valid_metric(value):
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _valid_count(value):
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0 and number.is_integer()


def _count_matrix_value(value):
    if not _valid_count(value):
        raise ValueError("invalid confusion-matrix count")
    return int(value)


def _validate_metric_summary(metrics, label, warnings):
    if not isinstance(metrics, dict):
        warnings.append(f"{label}: metric summary is missing or malformed.")
        return {}

    for key in ("test_accuracy", "macro_f1", "weighted_f1"):
        if key not in metrics:
            warnings.append(f"{label}: required metric {key} is missing.")
        elif not _valid_metric(metrics[key]):
            warnings.append(f"{label}: invalid or out-of-range {key}.")
            metrics.pop(key, None)

    return metrics


def _read_classification_report(path, expected_classes, warnings, label):
    rows = _read_csv(path, warnings)
    if rows is None:
        return {}
    result = {}
    for row in rows:
        name = (row.get("") or row.get("class") or row.get("label") or "").strip()
        if name in expected_classes:
            item = {}
            for key in ("precision", "recall", "f1-score", "f1_score"):
                raw = row.get(key)
                if raw not in (None, ""):
                    try:
                        number = float(raw)
                        if math.isfinite(number) and 0 <= number <= 1:
                            item[key] = number
                        else:
                            warnings.append(f"{label}: invalid {key} for class {name}.")
                    except (TypeError, ValueError):
                        warnings.append(f"{label}: unreadable {key} for class {name}.")
            raw_support = row.get("support")
            if raw_support not in (None, ""):
                try:
                    if not _valid_count(raw_support):
                        raise ValueError("invalid support")
                    item["support"] = int(float(raw_support))
                except (TypeError, ValueError, OverflowError):
                    warnings.append(f"{label}: invalid support for class {name}.")
            if item:
                result[name] = item
    return result


def _load_shot_scale(warnings):
    path = MODELS / "experiment_comparison" / "experiment_comparison.json"
    comparison = _read_json(path, warnings)
    group = {"source": str(path.relative_to(ROOT)), "experiments": {}}
    if not isinstance(comparison, dict):
        return group
    scope = comparison.get("scope")
    scope_classes = scope.get("classes") if isinstance(scope, dict) else None
    if scope_classes != SHOT_CLASSES:
        warnings.append("Shot-scale comparison class mapping is missing or differs from the expected saved mapping.")
    experiments = comparison.get("experiments")
    if not isinstance(experiments, dict):
        warnings.append("Shot-scale comparison JSON has no valid experiments object.")
        return group

    for key, label, folder in (
        ("baseline_cnn", "Baseline CNN", "shot_scale_cnn"),
        ("augmented_cnn", "Augmented CNN", "shot_scale_cnn_augmented"),
    ):
        item = experiments.get(key)
        if not isinstance(item, dict):
            warnings.append(f"Shot-scale comparison is missing experiment: {key}")
            continue
        raw_metrics = item.get("metrics")
        if not isinstance(raw_metrics, dict):
            warnings.append(f"{label}: metric summary is missing or malformed.")
            metrics = {}
        else:
            metrics = dict(raw_metrics)
            _validate_metric_summary(metrics, label, warnings)
        per_class = item.get("per_class_metrics")
        if not isinstance(per_class, dict):
            warnings.append(f"{label}: per-class metrics are missing or malformed.")
            per_class = {}
        if list(per_class.keys()) != SHOT_CLASSES:
            warnings.append(f"{label}: per-class labels/order do not match the expected shot-scale classes.")
        matrix_info = item.get("confusion_matrix")
        if not isinstance(matrix_info, dict):
            matrix_info = {}
        labels = matrix_info.get("class_order")
        counts = matrix_info.get("counts")
        matrix = None
        if labels == SHOT_CLASSES and isinstance(counts, dict):
            try:
                matrix = [[_count_matrix_value(counts[row][column]) for column in labels] for row in labels]
                if any(value < 0 for row in matrix for value in row):
                    raise ValueError("negative count")
                if any(sum(row) != 100 for row in matrix):
                    warnings.append(f"{label}: confusion-matrix row totals are not the expected 100 samples per class.")
            except (KeyError, TypeError, ValueError):
                warnings.append(f"{label}: confusion matrix is malformed.")
                matrix = None
        else:
            warnings.append(f"{label}: confusion matrix or class order is missing/mismatched.")

        artifact_dir = MODELS / folder / "test_evaluation"
        report = _read_classification_report(
            artifact_dir / "classification_report.csv", SHOT_CLASSES, warnings, label
        )
        if report:
            per_class = report
        group["experiments"][key] = {
            "label": label,
            "metrics": metrics,
            "per_class_metrics": per_class,
            "confusion_matrix": matrix,
            "confusion_matrix_labels_order": labels if matrix is not None else None,
            "source": str(path.relative_to(ROOT)),
        }
    return group


def _load_movement(warnings):
    group = {"source": "Saved movement test_evaluation.json files", "experiments": {}}
    for key, label, folder in (
        ("baseline", "Baseline movement CNN", "camera_movement_cnn"),
        ("balanced", "Balanced-sampling movement CNN", "camera_movement_cnn_balanced_sampling"),
    ):
        directory = MODELS / folder
        evaluation_path = directory / "test_evaluation.json"
        evaluation = _read_json(evaluation_path, warnings)
        if not isinstance(evaluation, dict):
            continue

        classes = evaluation.get("classes")
        if classes != MOVEMENT_CLASSES:
            warnings.append(f"{label}: original movement class order is missing or mismatched.")
            continue

        metrics = {
            key: evaluation.get(source_key)
            for key, source_key in (
                ("test_accuracy", "test_accuracy"),
                ("macro_f1", "test_macro_f1"),
                ("weighted_f1", "test_weighted_f1"),
                ("test_samples", "test_samples"),
            )
            if source_key in evaluation
        }
        _validate_metric_summary(metrics, label, warnings)
        per_class = evaluation.get("classification_report")
        if not isinstance(per_class, dict):
            warnings.append(f"{label}: classification report is missing or malformed.")
            per_class = {}
        if not all(name in per_class for name in MOVEMENT_CLASSES):
            warnings.append(f"{label}: one or more per-class metric entries are missing.")

        matrix = evaluation.get("confusion_matrix")
        matrix_labels = evaluation.get("confusion_matrix_labels_order")
        if (
            not isinstance(matrix, list) or matrix_labels != MOVEMENT_CLASSES
            or len(matrix) != len(MOVEMENT_CLASSES)
            or any(not isinstance(row, list) or len(row) != len(MOVEMENT_CLASSES) for row in matrix)
        ):
            warnings.append(f"{label}: confusion matrix or class order is invalid.")
            matrix = None
            matrix_labels = None
        else:
            try:
                matrix = [[_count_matrix_value(value) for value in row] for row in matrix]
                if any(value < 0 for row in matrix for value in row):
                    raise ValueError("negative count")
            except (TypeError, ValueError):
                warnings.append(f"{label}: confusion matrix contains invalid counts.")
                matrix = None
                matrix_labels = None

        group["experiments"][key] = {
            "label": label,
            "metrics": metrics,
            "per_class_metrics": per_class,
            "confusion_matrix": matrix,
            "confusion_matrix_labels_order": matrix_labels,
            "source": str(evaluation_path.relative_to(ROOT)),
            "label_warning": evaluation.get("label_warning", ""),
        }
    return group


def load_all():
    warnings = []
    shot = _load_shot_scale(warnings)
    movement = _load_movement(warnings)
    return {"shot_scale": shot, "movement": movement, "warnings": warnings}
