from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, mapping

ROOT = Path(__file__).resolve().parents[2]  # Negotiating-Space/
sys.path.insert(0, str(ROOT))
from project_config import load_config

# ---------- settings ----------
TAG = "first_draft"  # "first_draft" for WILDTRACK; "master" when this is the field calibration for an own site
BUFFER_M = 2.0  # study area = point grid + this margin

# ---------- paths ----------
cfg = load_config()
cam = cfg["camera"]
calib_dir = cfg["calib_dir"]  # calibration artifacts (H, study area)
out_dir = cfg["detections_dir"]  # detection-derived data (Data/Sites/<site>/<cam>/detections)
checks_dir = cfg["checks_dir"]  # check images (Data/Sites/<site>/<cam>/checks)
frames_dir = cfg["frames_dir"]
DETECTIONS_CSV = f"detections_{cam}_yolov8m-pose.csv"  # output of 01_detect_Yolo/01_detect.py
pts = json.load(open(calib_dir / "homo_pick_points.json"))["points"]

fit = [p for p in pts if p["use"] == "fit"]
check = [p for p in pts if p["use"] == "check"]
if len(fit) < 4:
    raise SystemExit(f"Need at least 4 'fit' points, found {len(fit)}")

# ---------- 1. build the homography (pixels -> ground meters) ----------
src = np.array([p["pixel"] for p in fit], dtype=np.float64)
dst = np.array([p["ground"] for p in fit], dtype=np.float64)
H, _ = cv2.findHomography(src, dst, method=0)  # least squares over all fit points
H_inv = np.linalg.inv(H)  # ground -> pixels (for drawing on the frame)
np.save(calib_dir / f"H_{cam}_{TAG}.npy", H)


def transform(xy, M):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(xy, M).reshape(-1, 2)


# ---------- 2. error report ----------
print("Point errors (meters) — fit = used to build, check = independent test:")
for p in fit + check:
    err = np.linalg.norm(transform([p["pixel"]], H)[0] - np.array(p["ground"]))
    print(f"  {p['name']:<15} {p['use']:<6} {err:.2f} m")

# ---------- 3. study area: shape around all points + buffer ----------
ground_all = np.array([p["ground"] for p in pts])
study_area = MultiPoint(ground_all).convex_hull.buffer(
    BUFFER_M, join_style=2
)  # sharp corners
print(f"Study area: {study_area.area:.1f} m²")

# ---------- 4. convert detections, keep only those inside ----------
df = pd.read_csv(out_dir / DETECTIONS_CSV)
g = transform(df[["foot_x", "foot_y"]].values, H)
df["ground_x"], df["ground_y"] = g[:, 0], g[:, 1]
df["in_study_area"] = [study_area.contains(Point(x, y)) for x, y in g]
inside = df[df["in_study_area"]]

df.to_csv(out_dir / f"detections_{cam}_ground_all_{TAG}.csv", index=False)
inside.to_csv(out_dir / f"detections_{cam}_study_area_{TAG}.csv", index=False)
print(
    f"{len(inside)} of {len(df)} detections are inside the study area "
    f"(avg {len(inside) / df['frame'].nunique():.1f} people per frame)"
)

# ---------- 5. GeoJSON (site CRS) ----------
epsg = cfg["crs"].split(":")[-1]
crs = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::" + epsg}}
with open(calib_dir / f"study_area_{TAG}.geojson", "w") as f:
    json.dump(
        {
            "type": "FeatureCollection",
            "crs": crs,
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(study_area),
                    "properties": {"buffer_m": BUFFER_M},
                }
            ],
        },
        f,
    )
with open(out_dir / f"detections_{cam}_study_area_{TAG}.geojson", "w") as f:
    json.dump(
        {
            "type": "FeatureCollection",
            "crs": crs,
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [r.ground_x, r.ground_y],
                    },
                    "properties": {
                        "frame": int(r.frame),
                        "det": int(r.det),
                        "conf": float(r.conf),
                    },
                }
                for r in inside.itertuples()
            ],
        },
        f,
    )

# ---------- 6a. check image: study area drawn on the camera frame ----------
frame0 = sorted(frames_dir.glob("*.png"))[0]
img = cv2.imread(str(frame0))
outline_px = transform(np.array(study_area.exterior.coords), H_inv).astype(np.int32)
cv2.polylines(
    img, [outline_px], isClosed=True, color=(0, 255, 255), thickness=3
)  # yellow
for p in pts:
    color = (0, 0, 255) if p["use"] == "fit" else (255, 0, 0)  # red fit, blue check
    cv2.circle(img, tuple(int(v) for v in p["pixel"]), 8, color, -1)
for r in df[df["frame"] == 0].itertuples():
    color = (0, 255, 0) if r.in_study_area else (128, 128, 128)  # green in, grey out
    cv2.circle(img, (int(r.foot_x), int(r.foot_y)), 6, color, -1)
cv2.imwrite(str(checks_dir / f"check_study_area_camera_{TAG}.jpg"), img)

# ---------- 6b. check image: plan view ----------
plt.figure(figsize=(8, 8))
x, y = study_area.exterior.xy
plt.plot(x, y, color="orange", label=f"study area (+{BUFFER_M} m)")
plt.scatter(ground_all[:, 0], ground_all[:, 1], c="red", s=30, label="reference points")
plt.scatter(
    inside["ground_x"], inside["ground_y"], s=3, alpha=0.3, label="detections inside"
)
plt.gca().set_aspect("equal")
plt.xlabel("easting (m)")
plt.ylabel("northing (m)")
plt.legend()
plt.title(f"{cam} study area — plan view ({TAG})")
plt.savefig(checks_dir / f"check_study_area_plan_{TAG}.png", dpi=150)
print(f"Saved H_{cam}_{TAG}.npy and study_area_{TAG}.geojson to {calib_dir}")
print(f"Saved check_study_area_camera_{TAG}.jpg and check_study_area_plan_{TAG}.png to {checks_dir}")
