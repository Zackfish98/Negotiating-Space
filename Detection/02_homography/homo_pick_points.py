import cv2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # Negotiating-Space/
sys.path.insert(0, str(ROOT))
from project_config import load_config

# ---------- settings ----------
FRAME_INDEX = 39  # which frame of the sequence to click on

# ---------- paths ----------
cfg = load_config()
frame = sorted(cfg["frames_dir"].glob("*.png"))[FRAME_INDEX]

img = cv2.imread(str(frame))
ref_path = cfg["calib_dir"] / "reference_frame.png"
if not ref_path.exists():  # keep the frame the points were picked on
    cv2.imwrite(str(ref_path), img)
    print(f"Saved {ref_path}")


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}]")
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        cv2.imshow("frame", img)


cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("frame", click)
cv2.imshow("frame", img)
cv2.waitKey(0)  # press any key to close
cv2.destroyAllWindows()
