from pathlib import Path
import cv2
import pandas as pd
import math

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(".")

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "movement_review"
    / "candidate_review.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "validation"
    / "movement_review"
)

FRAMES_ROOT = OUTPUT_ROOT / "frames"
CONTACT_ROOT = OUTPUT_ROOT / "contact_sheets"

NUM_FRAMES = 9

# ============================================================
# SETUP
# ============================================================

FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
CONTACT_ROOT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("CSV columns:")
for col in df.columns:
    print("  ", col)

print()
print("Total validation records:", len(df))

# ============================================================
# FIND VIDEO PATH COLUMN
# ============================================================

possible_path_columns = [
    "video_path",
    "local_video_path",
    "media_path",
    "clip_path",
    "file_path",
    "path",
]

path_column = None

for col in possible_path_columns:
    if col in df.columns:
        path_column = col
        break

if path_column is None:
    raise ValueError(
        "Could not find a video path column. "
        "Check the printed CSV columns and set path_column manually."
    )

print("Using video path column:", path_column)

# ============================================================
# HELPER: FIND A COLUMN
# ============================================================

def get_value(row, candidates, default="unknown"):
    for column in candidates:
        if column in df.columns:
            value = row[column]
            if pd.notna(value):
                return str(value)
    return default


# ============================================================
# CONTACT SHEET CREATION
# ============================================================

def create_contact_sheet(images, labels, output_path):
    if not images:
        return

    rows = 3
    cols = 3

    h, w = images[0].shape[:2]

    margin = 10
    label_height = 30

    sheet_width = cols * w + (cols + 1) * margin
    sheet_height = rows * (h + label_height) + (rows + 1) * margin

    sheet = 255 * (
        __import__("numpy").ones(
            (sheet_height, sheet_width, 3),
            dtype="uint8"
        )
    )

    for i, image in enumerate(images):

        r = i // cols
        c = i % cols

        x = margin + c * w
        y = margin + r * (h + label_height)

        sheet[y:y+h, x:x+w] = image

        cv2.putText(
            sheet,
            labels[i],
            (x + 5, y + h + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(str(output_path), sheet)


# ============================================================
# PROCESS EACH CLIP
# ============================================================

results = []

for index, row in df.iterrows():

    video_path = Path(str(row[path_column]))

    movie_id = get_value(
        row,
        ["movie_id", "imdb_id", "movie", "movieID"]
    )

    shot_id = get_value(
        row,
        ["shot_id", "shot", "shotID"]
    )

    movement = get_value(
        row,
        ["original_movement", "movement", "movement_label"]
    )

    scale = get_value(
        row,
        ["original_scale", "scale", "scale_label"]
    )

    # --------------------------------------------------------
    # Resolve relative paths
    # --------------------------------------------------------

    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    if not video_path.exists():
        print(f"[WARNING] Video not found: {video_path}")

        results.append({
            "movie_id": movie_id,
            "shot_id": shot_id,
            "movement": movement,
            "scale": scale,
            "video_path": str(video_path),
            "status": "VIDEO_NOT_FOUND",
        })

        continue

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARNING] Could not open: {video_path}")

        results.append({
            "movie_id": movie_id,
            "shot_id": shot_id,
            "movement": movement,
            "scale": scale,
            "video_path": str(video_path),
            "status": "OPEN_FAILED",
        })

        continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if frame_count <= 0:
        cap.release()

        results.append({
            "movie_id": movie_id,
            "shot_id": shot_id,
            "movement": movement,
            "scale": scale,
            "video_path": str(video_path),
            "status": "NO_FRAMES",
        })

        continue

    # --------------------------------------------------------
    # Determine 9 evenly spaced frame indices
    # --------------------------------------------------------

    if frame_count == 1:
        frame_indices = [0] * NUM_FRAMES
    else:
        frame_indices = [
            round(i * (frame_count - 1) / (NUM_FRAMES - 1))
            for i in range(NUM_FRAMES)
        ]

    safe_movie = movie_id.replace("/", "_")
    safe_shot = shot_id.replace("/", "_")

    clip_name = f"{safe_movie}_{safe_shot}"

    frame_output_dir = FRAMES_ROOT / clip_name
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    # --------------------------------------------------------
    # Extract frames
    # --------------------------------------------------------

    successful = 0

    for frame_number, frame_index in enumerate(frame_indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()

        if not success:
            print(
                f"[WARNING] Failed frame "
                f"{frame_index} from {video_path}"
            )
            continue

        frame_filename = (
            frame_output_dir
            / f"frame_{frame_number:02d}.jpg"
        )

        cv2.imwrite(
            str(frame_filename),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        # Resize only for the contact sheet.
        # Original extracted frame remains unchanged.
        preview = cv2.resize(
            frame,
            (640, 360),
            interpolation=cv2.INTER_AREA
        )

        images.append(preview)

        labels.append(
            f"{frame_number}: frame {frame_index}"
        )

        successful += 1

    cap.release()

    # --------------------------------------------------------
    # Contact sheet
    # --------------------------------------------------------

    contact_path = CONTACT_ROOT / f"{clip_name}.jpg"

    if images:
        create_contact_sheet(
            images,
            labels,
            contact_path
        )

    duration = (
        frame_count / fps
        if fps > 0
        else None
    )

    results.append({
        "movie_id": movie_id,
        "shot_id": shot_id,
        "original_movement": movement,
        "original_scale": scale,
        "video_path": str(video_path),
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": duration,
        "frames_requested": NUM_FRAMES,
        "frames_extracted": successful,
        "contact_sheet": str(contact_path),
        "status": "OK",
    })

    print(
        f"[{index + 1:02d}/{len(df)}] "
        f"{movie_id} / {shot_id} / {movement} "
        f"→ {successful}/{NUM_FRAMES} frames"
    )


# ============================================================
# SAVE EXTRACTION REPORT
# ============================================================

report_path = OUTPUT_ROOT / "frame_extraction_report.csv"

pd.DataFrame(results).to_csv(
    report_path,
    index=False
)

print()
print("=" * 60)
print("FRAME EXTRACTION COMPLETE")
print("=" * 60)
print("Clips:", len(df))
print("Contact sheets:", CONTACT_ROOT)
print("Frames:", FRAMES_ROOT)
print("Report:", report_path)