from pathlib import Path
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, LineString, mapping

ROOT = Path(__file__).resolve().parents[2]  # Negotiating-Space/
sys.path.insert(0, str(ROOT))
from project_config import load_config

# ---------- settings ----------
TAG = "master"  # imported calibration is checked against ground truth -> the master
POINTS_JSON = "homo_pick_points.json"
BUFFER_M = 4.0
GRID_STEP = 1.0

# ---------- paths ----------
cfg = load_config()
cam = cfg["camera"]  # camera id, e.g. C1
calib_dir = cfg["calib_dir"]  # calibration artifacts (H, study area)
out_dir = cfg["detections_dir"]  # detection-derived data (Data/Sites/<site>/<cam>/detections)
checks_dir = cfg["checks_dir"]  # check images (Data/Sites/<site>/<cam>/checks)
root = cfg["dataset_root"]
frames_dir = cfg["frames_dir"]
calib_name, view = cfg["calib_name"], cfg["view_num"]  # dataset's calibration file name, e.g. CVLab1
if not calib_name or calib_name == "TODO":
    raise SystemExit(f"Set calib_name for camera {cam} in {cfg['site_dir'] / 'site.json'}")
DETECTIONS_CSV = f"detections_{cam}_yolov8m-pose.csv"
pts = json.load(open(calib_dir / POINTS_JSON))["points"]


def transform(xy, M):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(xy, M).reshape(-1, 2)


# ================= 1. homography from WILDTRACK calibration =================
def read_numbers(path, tag):
    node = ET.parse(path).getroot().find(tag)
    data = node.find("data") if node.find("data") is not None else node
    return np.array([float(v) for v in data.text.split()])


K = read_numbers(
    root / "calibrations/intrinsic_zero" / f"intr_{calib_name}.xml", "camera_matrix"
).reshape(3, 3)
rvec = read_numbers(root / "calibrations/extrinsic" / f"extr_{calib_name}.xml", "rvec")
tvec = read_numbers(root / "calibrations/extrinsic" / f"extr_{calib_name}.xml", "tvec")
R, _ = cv2.Rodrigues(rvec)

H_ground_to_img = K @ np.column_stack([R[:, 0], R[:, 1], tvec])  # ground (cm) -> pixels
H_calib = np.diag([0.01, 0.01, 1.0]) @ np.linalg.inv(
    H_ground_to_img
)  # pixels -> ground (m)

cam_pos_cm = -R.T @ tvec  # camera center, WILDTRACK frame
cam_xy_local = cam_pos_cm[:2] / 100
print(f"Camera height above ground: {abs(cam_pos_cm[2]) / 100:.1f} m")

# ---- check against WILDTRACK ground truth ----
errors = []
for f in sorted((root / "annotations_positions").glob("*.json")):
    for person in json.load(open(f)):
        v = next(v for v in person["views"] if v["viewNum"] == view)
        if v["xmin"] == -1:
            continue
        g = transform([[(v["xmin"] + v["xmax"]) / 2, v["ymax"]]], H_calib)[0]
        pid = person["positionID"]
        truth = np.array([-3.0 + 0.025 * (pid % 480), -9.0 + 0.025 * (pid // 480)])
        errors.append(np.linalg.norm(g - truth))
errors = np.array(errors)
print(
    f"Calibration check vs ground truth ({len(errors)} people): "
    f"median {np.median(errors):.2f} m, 90% under {np.percentile(errors, 90):.2f} m"
)

# ================= 2–3. align WILDTRACK frame to UTM (north-up) =================
import itertools

local_pts = transform(
    [p["pixel"] for p in pts], H_calib
)  # true positions of your clicks
utm_pts = np.array([p["ground"] for p in pts], dtype=np.float64)
fit_mask = np.array([p["use"] == "fit" for p in pts])

# mirrored or not is decided by the calibration, not by best fit:
# camera z negative -> WILDTRACK's vertical axis points down -> frame is mirrored vs east/north
mirrored = cam_pos_cm[2] < 0
print(f"\nCamera z = {cam_pos_cm[2]:.0f} cm -> frame mirrored: {mirrored}")


def fit_rigid(a, b, mirror):
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    D = np.diag([1, -1 if mirror else 1])
    Rm = (
        Vt.T @ D @ U.T
        if np.linalg.det(Vt.T @ U.T) > 0
        else Vt.T @ np.diag([1, 1 if mirror else -1]) @ U.T
    )
    t = cb - Rm @ ca
    rms = np.sqrt(np.mean(np.sum((a @ Rm.T + t - b) ** 2, axis=1)))
    return Rm, t, rms


# try each row (group) as-is and reversed; keep the best fit
groups = sorted({p["group"] for p in pts})
results = []
for rev in itertools.product([False, True], repeat=len(groups)):
    u = utm_pts.copy()
    for g, r in zip(groups, rev):
        if r:
            idx = [i for i, p in enumerate(pts) if p["group"] == g]
            u[idx] = utm_pts[idx[::-1]]
    Rm_, t_, rms_ = fit_rigid(local_pts[fit_mask], u[fit_mask], mirrored)
    results.append((rms_, rev, Rm_, t_, u))
    label = ", ".join(
        f"{g} {'reversed' if r else 'as-is'}" for g, r in zip(groups, rev)
    )
    print(f"  {label:<35} RMS {rms_:.2f} m")

rms, rev, Rm, t, utm_pts = min(results, key=lambda r: r[0])
chosen = ", ".join(f"{g} {'reversed' if r else 'as-is'}" for g, r in zip(groups, rev))
print(f"Using: {chosen} (RMS {rms:.2f} m)")

S = np.eye(3)
S[:2, :2] = Rm
S[:2, 2] = t
H = S @ H_calib
H_inv = np.linalg.inv(H)
np.save(calib_dir / f"H_{cam}_{TAG}.npy", H)

aligned = local_pts @ Rm.T + t
print("How far each Google Earth placemark is from its true position:")
for p, a, u in zip(pts, aligned, utm_pts):
    print(f"  {p['name']:<15} {np.linalg.norm(a - u):.2f} m")

cam_utm = Rm @ cam_xy_local + t
d = cam_utm - aligned.mean(axis=0)
compass = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"][
    int(((np.degrees(np.arctan2(d[1], d[0])) + 22.5) % 360) // 45)
]
print(f"Camera is {np.hypot(*d):.1f} m to the {compass} of the reference points")

# ================= 4. study area from your clicked points (true positions) =================
study_area = MultiPoint(aligned).convex_hull.buffer(BUFFER_M, join_style=2)
print(f"\nStudy area: {study_area.area:.1f} m²")

# ---------- camera field of view on the ground ----------
MAX_RANGE_M = 30.0  # beyond this, people are too small to detect reliably

h_img, w_img = cv2.imread(str(sorted(frames_dir.glob("*.png"))[0])).shape[:2]
us, vs = np.meshgrid(np.linspace(0, w_img - 1, 60), np.linspace(0, h_img - 1, 60))
px = np.column_stack([us.ravel(), vs.ravel(), np.ones(us.size)])

g = px @ H.T  # raw projective coordinates
ref_sign = np.sign(
    (H @ [w_img / 2, h_img - 1, 1])[2]
)  # bottom-center pixel is surely ground
in_front = np.sign(g[:, 2]) == ref_sign  # drops points above the horizon
ground = g[in_front, :2] / g[in_front, 2:3]
near = np.hypot(*(ground - cam_utm).T) < MAX_RANGE_M

fov = MultiPoint(ground[near]).convex_hull
observed = study_area.intersection(fov)
unobserved = study_area.difference(fov)
print(
    f"Study area observed by camera: {observed.area / study_area.area:.0%} "
    f"({observed.area:.0f} of {study_area.area:.0f} m²)"
)

df = pd.read_csv(out_dir / DETECTIONS_CSV)
g = transform(df[["foot_x", "foot_y"]].values, H)
df["ground_x"], df["ground_y"] = g[:, 0], g[:, 1]
df["in_study_area"] = [study_area.contains(Point(x, y)) for x, y in g]
inside = df[df["in_study_area"]]
df.to_csv(out_dir / f"detections_{cam}_ground_all_{TAG}.csv", index=False)
inside.to_csv(out_dir / f"detections_{cam}_study_area_{TAG}.csv", index=False)
print(
    f"{len(inside)} of {len(df)} detections inside "
    f"(avg {len(inside) / df['frame'].nunique():.1f} people per frame)"
)

# ================= 5. GeoJSON (site CRS) =================
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
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": cam_utm.tolist()},
                    "properties": {"name": f"camera {cam}"},
                },
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

# ================= 6. check images =================
minx, miny, maxx, maxy = study_area.bounds
origin = np.array([np.floor(minx), np.floor(miny)])

grid_lines = []
for gx in np.arange(origin[0], maxx + GRID_STEP, GRID_STEP):
    grid_lines.append(
        LineString([(gx, miny - 1), (gx, maxy + 1)]).intersection(study_area)
    )
for gy in np.arange(origin[1], maxy + GRID_STEP, GRID_STEP):
    grid_lines.append(
        LineString([(minx - 1, gy), (maxx + 1, gy)]).intersection(study_area)
    )
grid_lines = [
    ln for ln in grid_lines if not ln.is_empty and ln.geom_type == "LineString"
]


def sample(line, n=40):
    return np.array(
        [line.interpolate(s, normalized=True).coords[0] for s in np.linspace(0, 1, n)]
    )


# ---- 6a. camera frame ----
img = cv2.imread(str(sorted(frames_dir.glob("*.png"))[0]))
overlay = img.copy()
for ln in grid_lines:
    cv2.polylines(
        overlay,
        [transform(sample(ln), H_inv).astype(np.int32)],
        False,
        (255, 255, 255),
        1,
    )
img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
cv2.polylines(
    img,
    [transform(np.array(study_area.exterior.coords), H_inv).astype(np.int32)],
    True,
    (0, 255, 255),
    3,
)
for p in pts:
    cv2.circle(img, tuple(int(v) for v in p["pixel"]), 8, (0, 0, 255), -1)
for r in df[df["frame"] == 0].itertuples():
    color = (0, 255, 0) if r.in_study_area else (128, 128, 128)
    cv2.circle(img, (int(r.foot_x), int(r.foot_y)), 6, color, -1)
cv2.imwrite(str(checks_dir / f"check_study_area_camera_{TAG}.jpg"), img)

# ---- 6b. plan view (north up, local meters, 1 m grid, camera) ----
L = lambda xy: np.asarray(xy) - origin
fig, ax = plt.subplots(figsize=(9, 9))
sa = L(np.array(study_area.exterior.coords))
fv = L(np.array(fov.exterior.coords))
ax.fill(
    fv[:, 0],
    fv[:, 1],
    color="grey",
    alpha=0.08,
    label=f"camera view (≤{MAX_RANGE_M:.0f} m)",
)
if not unobserved.is_empty:
    for poly in getattr(unobserved, "geoms", [unobserved]):
        up = L(np.array(poly.exterior.coords))
        ax.fill(
            up[:, 0],
            up[:, 1],
            facecolor="none",
            edgecolor="grey",
            hatch="///",
            lw=0,
            label="study area not observed",
        )
ax.plot(sa[:, 0], sa[:, 1], color="orange", label=f"study area (+{BUFFER_M} m)")
ax.scatter(
    *L(aligned).T, c="red", s=30, zorder=3, label="reference points (true position)"
)
ax.scatter(
    *L(utm_pts).T,
    facecolors="none",
    edgecolors="red",
    s=30,
    zorder=3,
    label="reference points (Google Earth)",
)
ax.scatter(
    inside["ground_x"] - origin[0],
    inside["ground_y"] - origin[1],
    s=3,
    alpha=0.3,
    label="people (feet, every frame)",
)
cam_l, centre = L(cam_utm), L(np.array(study_area.centroid.coords[0]))
ax.scatter(*cam_l, marker="^", s=120, c="black", zorder=4, label=f"camera {cam}")
ax.annotate(
    "", xy=centre, xytext=cam_l, arrowprops=dict(arrowstyle="->", color="grey", lw=1)
)

ax.set_xticks(
    np.arange(np.floor(ax.get_xlim()[0]), ax.get_xlim()[1], GRID_STEP), minor=True
)
ax.set_yticks(
    np.arange(np.floor(ax.get_ylim()[0]), ax.get_ylim()[1], GRID_STEP), minor=True
)
ax.grid(which="minor", color="grey", alpha=0.15, lw=0.5)
ax.grid(which="major", color="grey", alpha=0.3, lw=0.7)
ax.annotate(
    "N",
    xy=(0.95, 0.97),
    xytext=(0.95, 0.87),
    xycoords="axes fraction",
    ha="center",
    fontsize=12,
    arrowprops=dict(arrowstyle="->", lw=1.5),
)
ax.set_aspect("equal")
ax.set_xlabel("meters east")
ax.set_ylabel("meters north")
ax.set_title(f"{cam} study area — plan view (north up, 1 m grid, {TAG})")
ax.legend(loc="lower left", fontsize=8)
fig.savefig(checks_dir / f"check_study_area_plan_{TAG}.png", dpi=150)
print(f"Saved H_{cam}_{TAG}.npy and study_area_{TAG}.geojson to {calib_dir}")
print(f"Saved check_study_area_camera_{TAG}.jpg and check_study_area_plan_{TAG}.png to {checks_dir}")
