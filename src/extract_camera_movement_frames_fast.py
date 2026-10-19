from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "data" / "processed" / "camera_movement_split.csv"
OUTPUT_ROOT = ROOT / "data" / "processed" / "camera_movement_frames_3_fast"
REPORT_CSV = ROOT / "data" / "processed" / "camera_movement_extraction_report_3_fast.csv"

FRAMES_PER_CLIP = 3
FRAME_SIZE = (112, 112)
EXPECTED_LABELS = {"Static", "Motion", "Push", "Pull", "Multi_movement"}

def extract_clip(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "OPEN_FAILED"
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        return None, "NO_FRAMES"
    indices = [0] * FRAMES_PER_CLIP if count == 1 else [
        round(i * (count - 1) / (FRAMES_PER_CLIP - 1))
        for i in range(FRAMES_PER_CLIP)
    ]
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None, "FRAME_READ_FAILED"
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.stack(frames).astype(np.uint8), "OK"

def main():
    df = pd.read_csv(SPLIT_CSV, dtype={"imdb_id": str})
    required = {"imdb_id", "shot_id", "video_path", "movement", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if set(df["movement"].dropna().unique()) != EXPECTED_LABELS:
        raise ValueError("Unexpected movement labels.")
    if df[["imdb_id", "shot_id"]].duplicated().any():
        raise ValueError("Duplicate movie/shot records found.")
    if REPORT_CSV.exists():
        raise FileExistsError(f"Report exists already: {REPORT_CSV}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for number, row in enumerate(df.itertuples(index=False), start=1):
        filename = f"{row.imdb_id}_shot_{int(row.shot_id):04d}.npz"
        output_path = OUTPUT_ROOT / filename
        video_path = Path(row.video_path)
        if not video_path.is_absolute():
            video_path = ROOT / video_path
        status = "OK"
        if output_path.is_file():
            try:
                with np.load(output_path) as saved:
                    valid = "frames" in saved and saved["frames"].shape == (3, 112, 112, 3) and saved["frames"].dtype == np.uint8
                status = "EXISTING_VALID" if valid else "EXISTING_INVALID"
            except Exception:
                status = "EXISTING_INVALID"
        elif not video_path.is_file():
            status = "VIDEO_NOT_FOUND"
        else:
            frames, status = extract_clip(video_path)
            if frames is not None:
                np.savez(output_path, frames=frames)
        results.append({
            "imdb_id": row.imdb_id,
            "shot_id": int(row.shot_id),
            "movement": row.movement,
            "split": row.split,
            "video_path": str(video_path),
            "frame_path": str(output_path),
            "status": status,
        })
        if number % 250 == 0 or number == len(df):
            print(f"Processed {number}/{len(df)} clips", flush=True)
    report = pd.DataFrame(results)
    report.to_csv(REPORT_CSV, index=False)
    print("\nExtraction complete.")
    print("Records:", len(report))
    print("Output:", OUTPUT_ROOT)
    print("Report:", REPORT_CSV)
    print(report["status"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()
