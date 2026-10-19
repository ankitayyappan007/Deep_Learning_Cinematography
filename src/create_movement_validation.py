
import json
import os
import re
import random
import shutil
import csv

PROJECT = os.getcwd()

# Find v3_full.json automatically
json_files = []
for root, dirs, files in os.walk(os.path.join(PROJECT, "data", "movieshots", "annotations")):
    for f in files:
        if f.lower() == "v3_full.json":
            json_files.append(os.path.join(root, f))

if not json_files:
    raise FileNotFoundError("v3_full.json was not found.")

JSON_PATH = json_files[0]

MEDIA_ROOT = os.path.join(
    PROJECT, "data", "media", "trailer", "trailer"
)

OUT_ROOT = os.path.join(
    PROJECT, "data", "validation", "movement_review"
)

os.makedirs(OUT_ROOT, exist_ok=True)

print("Annotation file:")
print(JSON_PATH)
print()

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

records = []

def walk(obj, path):
    if isinstance(obj, dict):

        # MovieShots shot record
        if "movement" in obj and isinstance(obj["movement"], dict):
            movement = obj["movement"].get("label")

            if movement:
                movie_id = None
                shot_id = None

                # Look through hierarchy path for IMDb movie ID and shot ID
                for item in path:
                    s = str(item)

                    if re.fullmatch(r"tt\d+", s):
                        movie_id = s

                    if re.fullmatch(r"\d{4}", s):
                        shot_id = s

                if movie_id and shot_id:
                    scale = None

                    if isinstance(obj.get("scale"), dict):
                        scale = obj["scale"].get("label")

                    records.append({
                        "movie_id": movie_id,
                        "shot_id": shot_id,
                        "original_movement": movement,
                        "original_scale": scale
                    })

        for key, value in obj.items():
            walk(value, path + [key])

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            walk(value, path + [i])

walk(data, [])

# Remove duplicates
unique = {}
for r in records:
    key = (
        r["movie_id"],
        r["shot_id"],
        r["original_movement"]
    )
    unique[key] = r

records = list(unique.values())

print("Total annotation records found:", len(records))

# Keep only clips that physically exist
available = []

for r in records:
    movie_dir = os.path.join(MEDIA_ROOT, r["movie_id"])
    filename = f"shot_{r['shot_id']}.mp4"
    video_path = os.path.join(movie_dir, filename)

    if os.path.isfile(video_path):
        r["video_path"] = video_path
        available.append(r)

print("Records with local video:", len(available))

# Required validation batch
targets = {
    "Static": 10,
    "Push": 10,
    "Pull": 10,
    "Motion": 50
}

# Group by label
groups = {}

for r in available:
    label = r["original_movement"]

    if label in targets:
        groups.setdefault(label, []).append(r)

# Deterministic shuffle
random.seed(42)

selected = []

for label, target in targets.items():

    candidates = groups.get(label, [])

    # Prefer different movies
    by_movie = {}
    for r in candidates:
        by_movie.setdefault(r["movie_id"], []).append(r)

    movies = list(by_movie.keys())
    random.shuffle(movies)

    chosen = []
    used_movies = set()

    # First pass: one clip per movie
    for movie in movies:
        if len(chosen) >= target:
            break

        candidate_list = by_movie[movie]
        r = random.choice(candidate_list)

        chosen.append(r)
        used_movies.add(movie)

    # Second pass if not enough unique movies
    remaining = [
        r for r in candidates
        if r not in chosen
    ]

    random.shuffle(remaining)

    for r in remaining:
        if len(chosen) >= target:
            break
        chosen.append(r)

    print(
        f"{label}: selected {len(chosen)} / {target} "
        f"from {len(candidates)} available"
    )

    for r in chosen:
        r["review_label"] = label
        r["manual_label"] = ""
        r["decision"] = ""
        r["review_notes"] = ""
        selected.append(r)

# Create review CSV
csv_path = os.path.join(
    OUT_ROOT,
    "candidate_review.csv"
)

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    fields = [
        "movie_id",
        "shot_id",
        "original_movement",
        "original_scale",
        "video_path",
        "review_label",
        "manual_label",
        "decision",
        "review_notes"
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for r in selected:
        writer.writerow(r)

# Create folders containing selected clips
for label in targets:
    os.makedirs(
        os.path.join(OUT_ROOT, label),
        exist_ok=True
    )

for r in selected:

    source = r["video_path"]

    destination = os.path.join(
        OUT_ROOT,
        r["review_label"],
        f"{r['movie_id']}_shot_{r['shot_id']}.mp4"
    )

    shutil.copy2(source, destination)

print()
print("======================================")
print("MOVEMENT VALIDATION BATCH CREATED")
print("======================================")
print("Output:", OUT_ROOT)
print("CSV:", csv_path)
print()

for label in targets:
    folder = os.path.join(OUT_ROOT, label)
    count = len([
        f for f in os.listdir(folder)
        if f.lower().endswith(".mp4")
    ])
    print(f"{label}: {count}")

print()
print("Done.")
