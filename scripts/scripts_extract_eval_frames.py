import csv
import random
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
SPLITS = {
    "validation": ROOT / "shot_scale_validation_manifest.csv",
    "test": ROOT / "shot_scale_test_manifest.csv",
}
OUTPUTS = {
    "validation": ROOT / "shot_scale_validation_frames",
    "test": ROOT / "shot_scale_test_frames",
}
CLASSES = ["CS", "ECS", "FS", "LS", "MS"]
SEED = 20261013

for name, manifest in SPLITS.items():
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if OUTPUTS[name].exists():
        raise FileExistsError(
            f"Output folder already exists: {OUTPUTS[name]}. "
            "Nothing was overwritten. Choose a new output folder."
        )

def extract_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()

for split_name, manifest_path in SPLITS.items():
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 500:
        raise ValueError(f"{split_name}: expected 500 rows, found {len(rows)}")

    counts = {label: 0 for label in CLASSES}
    for row in rows:
        if row["scale"] not in counts:
            raise ValueError(f"Unexpected class: {row['scale']}")
        counts[row["scale"]] += 1

    if any(counts[label] != 100 for label in CLASSES):
        raise ValueError(f"{split_name}: unexpected class counts {counts}")

    output_root = OUTPUTS[split_name]
    output_root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(SEED)
    rng.shuffle(rows)

    report = []
    for index, row in enumerate(rows, start=1):
        video_path = Path(row["video_path"])
        output_path = output_root / row["scale"] / (
            f"{row['imdb_id']}_shot_{int(row['shot_id']):04d}.jpg"
        )

        status = "failed"
        error = ""

        try:
            if not video_path.is_file():
                raise FileNotFoundError(f"Video not found: {video_path}")

            image = extract_frame(video_path)
            if image is None:
                raise RuntimeError("Could not decode a frame")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(output_path), bgr):
                raise RuntimeError("Could not save image")

            status = "success"
        except Exception as exc:
            error = str(exc)

        report.append({
            "split": split_name,
            "imdb_id": row["imdb_id"],
            "shot_id": row["shot_id"],
            "scale": row["scale"],
            "video_path": row["video_path"],
            "frame_path": str(output_path) if status == "success" else "",
            "status": status,
            "error": error,
        })

        if index % 50 == 0:
            print(f"{split_name}: processed {index}/{len(rows)} clips")

    report_path = ROOT / f"shot_scale_{split_name}_extraction_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)

    successes = sum(r["status"] == "success" for r in report)
    failures = len(report) - successes

    print(f"\n{split_name.upper()} RESULTS")
    print(f"Frames extracted: {successes}/{len(report)}")
    print(f"Failures: {failures}")
    for label in CLASSES:
        n = sum(
            r["status"] == "success" and r["scale"] == label
            for r in report
        )
        print(f"  {label}: {n} successful frames")
    print(f"Report: {report_path}")

print("\nExtraction finished. No model training was performed.")
