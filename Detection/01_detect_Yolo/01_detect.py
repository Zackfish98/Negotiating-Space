from pathlib import Path
import yaml
import pandas as pd
from ultralytics import YOLO

# ---------- settings ----------
MODEL_NAME = "yolov8m-pose.pt"
IMGSZ = 1920  # full resolution so distant people aren't shrunk away
CONF = 0.20
CHECK_EVERY = 50  # save a check image every N frames

# ---------- paths ----------
DETECTION = Path(__file__).resolve().parents[1]  # Detection/
ROOT = DETECTION.parent  # Negotiating-Space/
config = yaml.safe_load(open(ROOT / "Data" / "Data_Config" / "paths.yaml"))

frames_dir = Path(config["data_dir"]) / config["camera_dir"]
CAMERA = Path(config["camera_dir"]).name  # e.g. C1, C6
out_dir = DETECTION / config["output_dir"]
check_dir = out_dir / f"check_frames_{Path(MODEL_NAME).stem}"  # one folder per model
out_dir.mkdir(exist_ok=True)
check_dir.mkdir(exist_ok=True)

images = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
print(f"Found {len(images)} frames in {frames_dir}")
if not images:
    raise SystemExit("No frames found. Check camera_dir in paths.yaml")

# ---------- model (Apple GPU if available, else CPU) ----------
weights = ROOT / MODEL_NAME  # weights live at the repo root; else Ultralytics downloads them
model = YOLO(str(weights) if weights.exists() else MODEL_NAME)
try:
    model(str(images[0]), imgsz=IMGSZ, device="mps", verbose=False)
    DEVICE = "mps"
except Exception:
    DEVICE = "cpu"
print(f"Running {MODEL_NAME} on {DEVICE}")

# ---------- detect every frame (no tracking: keeps everyone) ----------
rows = []
for i, img in enumerate(images):
    r = model(str(img), imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)[0]

    if i % CHECK_EVERY == 0:
        r.save(filename=str(check_dir / f"frame_{i:04d}.jpg"))
        print(f"frame {i}/{len(images)}: {len(r.boxes)} people")

    if len(r.boxes) == 0:
        continue

    boxes = r.boxes.xyxy.tolist()
    confs = r.boxes.conf.tolist()
    kpts = r.keypoints.data.tolist()  # 17 x (x, y, confidence) per person

    for det, ((x1, y1, x2, y2), c, kp) in enumerate(zip(boxes, confs, kpts)):
        row = {
            "frame": i,
            "file": img.name,
            "det": det,
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
