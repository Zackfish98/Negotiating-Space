# Data & Calibration — Negotiating Space

How footage is stored, described and calibrated before any detection runs.
This README covers **prep work only**: drives, paths, the footage manifest, and
turning a camera view into site-plan meters (homography + camera position).

Items marked **(to build)** don't exist yet.

---

## 1. Where things live

| What | Where | In git? |
|---|---|---|
| Raw footage (all of it) | **Master SSD**, stays at home, backed up to cloud | No |
| Footage being worked on now | **Working SSD / flash drive**, copies only | No |
| Code, configs, manifest, calibration points | This repo | Yes |
| Calibration results (`H_*.npy`, `study_area_*.geojson`, `camera_position.json`) | `Data/Sites/<site>/<camera>/` | Yes |
| Detection data and check images, per camera | `Data/Sites/<site>/<camera>/detections/` and `checks/` | No (regenerated) |
| Final outputs (behavior results) | `Naming_Behaviors/Outputs/` | No (regenerated) |
| YOLO weights (`*.pt`) | Repo root, ignored | No (Ultralytics downloads by name) |

**Rules**
- The working drive only ever holds **copies**. If it dies, nothing is lost.
- Never edit or rename footage on the working drive. Rename on the master, then re-copy.
- Anything *about* footage (metadata, which clips trained a model) lives in **this repo**,
  not on either drive. The drives hold pixels; the repo holds knowledge.

### Paths: `Data/Data_Config/paths.yaml`
Every script finds footage through this file plus the site's `site.json`, resolved in one
place: `project_config.py` at the repo root. Changing drives, sites or cameras means
editing this file, not the scripts.

```yaml
archive_root: ""                                   # master SSD (set when you have it)
working_root: "/Volumes/Backup Drive/GSAPP Flash"  # portable working copies
site: "wildtrack"                                  # which Data/Sites/<site>/ to use
camera: "C1"                                       # which camera position
```
Scripts read footage from `working_root`.

Dataset-specific settings (where frames sit inside the dataset, the CRS, per-camera
calibration names) live in `Data/Sites/<site>/site.json`:

```json
{
  "site": "wildtrack",
  "crs": "EPSG:32632",
  "dataset_root": "wildtrack/Wildtrack_dataset_full/Wildtrack_dataset",
  "frames_subdir": "Image_subsets/{camera}",
  "cameras": {
    "C1": {"calib_name": "CVLab1", "view_num": 0}
  }
}
```

---

## 2. Naming footage

```
{site}_{camera}_{YYYYMMDD}_{HHMM}.mp4
hvk_cam01_20261108_1400.mp4
```
- `site`: short lowercase id, matches a folder in `Data/Sites/`
- `camera`: camera position id (see §4). Move the camera → new id.
- Lowercase, no spaces.

Image-sequence datasets (e.g. WILDTRACK) keep their own frame names; the manifest
points at the folder instead of a file. WILDTRACK's cameras are named `C1`–`C7`.

---

## 3. The footage manifest: `Data/footage_manifest.csv`

One row per clip, filled in **when footage comes in**, before any processing. The
pipeline reads it to know what each clip is. Opens in Excel/Numbers; git tracks every edit.

| Field | Example | Why |
|---|---|---|
| `clip_id` | `hvk_cam01_20261108_1400` | Key every output is tagged with |
| `filename` | `hvk_cam01_20261108_1400.mp4` | File or folder under the drive root |
| `site_id` | `hvk` | Links to `Data/Sites/hvk/` |
| `camera_position_id` | `cam01` | Links to that camera's calibration |
| `date`, `start_time`, `end_time` | `2026-11-08`, `14:00`, `15:00` | Include timezone if not NYC |
| `day_of_week`, `holiday_or_event` | `Sat`, ` ` | Weekday vs weekend behaves differently |
| `device`, `lens`, `lens_mode` | `iPhone 15 Pro`, `0.5x`, `linear` | Linear/undistorted lens = better homography |
| `camera_height_m`, `camera_angle` | `3.2`, `30°` | From calibration; affects pose reliability |
| `mount` | `tripod` / `window` / `tree` | Diversity of viewpoints for training |
| `fps`, `resolution` | `30`, `1920x1080` | |
| `weather`, `temp`, `light` | `clear`, `12°C`, `daylight` | Rain empties benches, which isn't a design effect |
| `intervention_state` | `before` / `during` / `after` | Core comparison for activations |
| `intervention_notes` | `6 movable chairs added, NE corner` | |
| `source` | `self` / `partner:StreetPlans` / `public:WILDTRACK` | Plus permission notes |
| `signage_posted`, `anonymized` | `yes`, `no` | |
| `retain_until` | `2027-01-31` | When raw video is deleted, keeping only tracks |
| `homography_error_m` | `0.04` | From calibration check points |
| `split` | `train` / `validation` / `test` | **Clip-level.** Never let one clip feed both training and testing |
| `tracked`, `features_done` | `yes`, `no` | Pipeline skips finished clips |
| `notes` | `bus blocked view 14:20–14:25` | Anything odd |

**On `split`:** split by whole clips (better: whole days), not frames. Neighboring
frames are near-identical, so frame-level splits leak. Findings you report should come
from `test` clips; otherwise they're partly circular.

---

## 4. Calibration: camera view → site-plan meters

### Concepts
- **Homography** maps the *ground* in the image to the *ground* on the plan. It needs
  ≥4 ground points with known plan coordinates. It does **not** need the camera's location.
- **Camera position (x, y, z)** comes separately, from laser distances to the same ground points.
- A homography belongs to a **camera position**, not a site. Move the tripod a meter → recalibrate.
- People are placed by their **feet** (`foot_x, foot_y` = bottom-center of the box).
  Heads and hips drift away from the camera.
- Homography is 2D only: no heights (seat heights etc.).
- Work in a **projected CRS in meters** (UTM 18N `EPSG:32618` for NYC; WILDTRACK/Zurich
  is UTM 32N `EPSG:32632`). Meters in → speeds in m/s, no float-precision problems.

### Folder per camera position
```
Data/Sites/
└── wildtrack/
    ├── site.json                         # crs, dataset paths, per-camera calibration names
    ├── C1/
    │   ├── reference_frame.png           # frame used for picking points
    │   ├── GogEarth_points.kml           # points picked in Google Earth
    │   ├── homo_pick_points.json         # pixel ↔ ground pairs, fit/check
    │   ├── H_C1_master.npy               # pixels → meters, used by the pipeline
    │   ├── H_C1_first_draft.npy          # earlier manual fit, reference only
    │   ├── study_area_master.geojson
    │   ├── laser_measurements.csv        # own sites only (to build per site)
    │   └── camera_position.json          # own sites only (to build per site)
    └── C6/                               # not calibrated yet
```
Calibration results are *inputs* to every later step and took manual work, so they live
with the camera. Regenerable files sit beside them in `detections/` and `checks/`, which git ignores.

### Output tags
Every calibration output carries a tag so routes never overwrite each other:
- `_master`: what the pipeline uses. WILDTRACK: Route A (checked against ground truth).
  Own sites: the field-calibrated Route B.
- `_first_draft`: the earlier Google-Earth-only manual fit, kept for reference.

Detection files derived from a calibration carry the same tag, e.g.
`Data/Sites/wildtrack/C1/detections/detections_C1_study_area_master.csv`.

### Two routes

| Route | When | Scripts | Tag |
|---|---|---|---|
| **A. Imported** | The dataset ships its own calibration (WILDTRACK) | `homography_calibration_imported.py` | `master` |
| **B. Manual / field** | Your own footage | `homo_pick_points.py` → `homography_calibration_manual.py` (+ `homo_camera_position.py`, **to build**) | set `TAG` in the script: `first_draft` for WILDTRACK, `master` for own sites |

---

### Route B, part 1: Field session (own sites)

**Kit**
- Laser distance meter (Bosch / Leica DISTO class). A laser *pointer* won't measure.
- 4–6 target boards: matte white foam board (or Bosch/Leica target plates), each with a
  printed **ArUco marker** (`DICT_4X4_50`, white border ≥1 square)
- Tape measure, chalk or removable tape dots
- Phone for setup photos

**Marker size** (needs ~30 px per side; 1080p, ~70° lens; halve for 4K):

| Camera → marker | Marker side |
|---|---|
| 10 m | ~22 cm |
| 20 m | ~44 cm |
| 30 m | ~65 cm |

Test first: prop one up at the expected distance, photograph with the same device, run detection.

**Steps**
1. **Choose 6–10 control points** on the ground plane: curb corners, tree-pit corners,
   paving joints, drains. Spread near and far. Nothing raised (no bench tops). Durable.
2. **Mark and name them** (`A`, `B`, `C`…). Use 3-4-5 triangles (3 m, 4 m, 5 m) for true
   right angles if laying out a grid.
3. **Measure between points.** Meter on point A, aim at an upright target on B. Measure
   every pair you can: 4 points = 6 distances (extras = error check). Log in
   `laser_measurements.csv`.
4. **Place the camera.** Note mount (tripod / window / tree). Mark the spot (tape, chalk)
   and photograph the setup for re-placement.
5. **Measure camera → points.** From the camera, meter held at the lens, aim *down* at the
   target on each of **at least 3** points (4 = error check). Check the meter's reference
   edge (front/back) is aligned with the lens.
   - Every camera on a site measures the **same** points → all cameras share one frame.
   - If easy, add a plumb measurement (laser straight down) as a height check.
6. **Targets upright, facing the camera.** Bottom edge on the mark, aim at that edge, or
   record the target-center height if aiming at the center.
7. **Record the reference frame** with all boards in view (a still photo or high-quality
   frame, not a compressed one).
8. **Pick everything up.**

`laser_measurements.csv`
```
from,to,distance_m,note
A,B,4.212,
A,C,6.031,
...
CAM,A,9.874,meter front edge at lens
CAM,B,11.402,
CAM,C,8.995,
CAM,ground,3.210,plumb check
```

### Route B, part 2: Office

1. **Ground coordinates**
   - *From laser:* A = origin, A→B = x-axis; solve the rest from pairwise distances.
   - *From maps:* pick the same points in Google Earth → `GogEarth_points.kml` →
     `homo_convert_points.py` (converts to the site CRS). Google Earth is only good to
     ~1 m+, so prefer laser for precision and use maps to georeference.
2. **Pixel coordinates**: `homo_pick_points.py` (click each point on the reference frame;
   it also saves `reference_frame.png`), or ArUco auto-detection **(to build)**. If a far
   marker isn't detected, click that board by hand.
3. **Fill `homo_pick_points.json`**: mark most points `"use": "fit"` and 1–2 as
   `"use": "check"` (independent test, not used to build).
4. **Run `homography_calibration_manual.py`** with `TAG = "master"` → `H_<cam>_master.npy`,
   error report, study area, check images.
5. **Run `homo_camera_position.py`** **(to build)** → trilaterates camera x, y, z from
   ≥3 laser distances → `camera_position.json`.
6. **Check**
   - Error report: `check` points should land within a few cm (laser) / ~1 m (Google Earth).
   - `check_study_area_camera_<tag>.jpg`: grid should sit flat on the ground.
   - `check_study_area_plan_<tag>.png`: camera and study area where you expect.
   - Optional cross-check: camera height from the homography + lens specs vs laser height.
7. **Record** `homography_error_m` and `camera_height_m` in the manifest.

`camera_position.json`
```json
{
  "site": "hvk", "camera": "cam01", "crs": "EPSG:32618",
  "x": 586512.31, "y": 4506230.88, "z": 3.21,
  "mount": "window",
  "trilateration_rms_m": 0.03,
  "plumb_check_m": 3.21,
  "homography_error_m": 0.04,
  "date": "2026-11-08"
}
```

### Re-using a calibration between sessions
1. Put the camera back on its marked spot.
2. Overlay the saved control points on a new frame (or re-detect the ArUco boards).
3. Within a few pixels → same `camera_position_id`, reuse the homography.
4. Off → new id, redo part 2 steps 2–7 (a few minutes if boards are back out).

---

### Route A: Imported calibration (WILDTRACK)
`homography_calibration_imported.py` builds the homography and camera height from the
dataset's intrinsic/extrinsic files, checks it against WILDTRACK ground truth, then aligns
it to UTM using the Google Earth points in `homo_pick_points.json`. Writes `_master` outputs.
- Needs `paths.yaml` + `site.json` pointing at the dataset, and
  `Data/Sites/<site>/<cam>/detections/detections_<cam>_yolov8m-pose.csv` from `01_detect.py`.
- Use this for any dataset that provides calibration; it's more accurate than picking points.

---

## 5. Checklist for a new site

- [ ] Footage copied to master, named `{site}_{camera}_{date}_{time}`
- [ ] Row added to `Data/footage_manifest.csv` (incl. `split`)
- [ ] `Data/Sites/<site>/site.json` with CRS and camera entries
- [ ] Field session done → `laser_measurements.csv`, `reference_frame.png`
- [ ] `homo_pick_points.json` filled (fit + check)
- [ ] Homography run with `TAG = "master"`, errors acceptable, check images look right
- [ ] `camera_position.json` written
- [ ] Clip copied to working drive, `paths.yaml` set to the `site` + `camera`
- [ ] Ready for `Detection/01_detect_Yolo/01_detect.py`
