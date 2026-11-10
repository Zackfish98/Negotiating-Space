from pathlib import Path
import sys
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]  # Negotiating-Space/
sys.path.insert(0, str(ROOT))
from project_config import load_config

# ---------- settings ----------
MODEL_NAME = "yolov8m-pose.pt"
IMGSZ = 1920  # full resolution so distant people aren't shrunk away
CONF = 0.20
CHECK_EVERY = 50  # save a check image every N frames
TRACK = True  # link detections across frames into paths (False = raw detections)
TRACKER = "bytetrack.yaml"

# ---------- paths ----------
cfg = load_config()
frames_dir = cfg["frames_dir"]
CAMERA = cfg["camera"]  # e.g. C1, C6
out_dir = cfg["detections_dir"]  # Data/Sites/<site>/<cam>/detections
check_dir = cfg["checks_dir"] / f"check_frames_{Path(MODEL_NAME).stem}"  # one folder per model
check_dir.mkdir(parents=True, exist_ok=True)

images = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
print(f"Found {len(images)} frames in {frames_dir}")
if not images:
    raise SystemExit(
        "No frames found. Check working_root/site/camera in paths.yaml and site.json"
    )

# ---------- model (Apple GPU if available, else CPU) ----------
weights = (
    ROOT / MODEL_NAME
)  # weights live at the repo root; else Ultralytics downloads them
model = YOLO(str(weights) if weights.exists() else MODEL_NAME)
try:
    model(str(images[0]), imgsz=IMGSZ, device="mps", verbose=False)
    DEVICE = "mps"
except Exception:
    DEVICE = "cpu"
print(
    f"Running {MODEL_NAME} on {DEVICE}"
    + (f", tracking with {TRACKER}" if TRACK else "")
)

# ---------- detect (+ track) every frame ----------
# Tracking links detections into paths (needed for speed and dwell time). The
# tracker can drop or split some detections. WILDTRACK image subsets are believed
# to be ~2 fps, so people move about half a meter between frames and IDs may
# fragment. Set TRACK = False to get every raw detection (track_id = -1).
rows = []
for i, img in enumerate(images):
    if TRACK:
        r = model.track(
            str(img),
            persist=True,
            tracker=TRACKER,
            imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            verbose=False,
        )[0]
    else:
        r = model(str(img), imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)[0]

    if i % CHECK_EVERY == 0:
        r.save(filename=str(check_dir / f"frame_{i:04d}.jpg"))
        print(f"frame {i}/{len(images)}: {len(r.boxes)} people")

    if len(r.boxes) == 0:
        continue

    boxes = r.boxes.xyxy.tolist()
    confs = r.boxes.conf.tolist()
    kpts = r.keypoints.data.tolist()  # 17 x (x, y, confidence) per person
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)

    for det, ((x1, y1, x2, y2), c, kp, tid) in enumerate(zip(boxes, confs, kpts, ids)):
        row = {
            "frame": i,
            "file": img.name,
            "det": det,
            "track_id": tid,
            "conf": c,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "foot_x": (x1 + x2) / 2,
            "foot_y": y2,
        }  # ground point for the homography
        for k, (kx, ky, kc) in enumerate(kp):
            row[f"kp{k}_x"], row[f"kp{k}_y"], row[f"kp{k}_c"] = kx, ky, kc
        rows.append(row)

# ---------- save ----------
df = pd.DataFrame(rows)
csv_path = out_dir / f"detections_{CAMERA}_{Path(MODEL_NAME).stem}.csv"
df.to_csv(csv_path, index=False)
print(
    f"Saved {len(df)} detections across {df['frame'].nunique()} frames "
    f"(avg {len(df) / len(images):.1f} people per frame) to {csv_path.name}"
)

# ---------- tracking summary ----------
if TRACK:
    tracked = df[df["track_id"] >= 0]
    lengths = tracked.groupby("track_id")["frame"].nunique()
    print(
        f"Tracking: {len(lengths)} unique track_ids, "
        f"{len(df) - len(tracked)} detections without an ID; "
        f"frames per track: median {lengths.median():.0f}, max {lengths.max()}"
    )
