import cv2
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # Negotiating-Space/
config = yaml.safe_load(open(ROOT / "Data" / "Data_Config" / "paths.yaml"))
frame = sorted((Path(config["data_dir"]) / config["camera_dir"]).glob("*.png"))[39]

img = cv2.imread(str(frame))


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
