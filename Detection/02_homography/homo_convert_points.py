import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]  # Negotiating-Space/
sys.path.insert(0, str(ROOT))
from project_config import load_config

cfg = load_config()
kml_path = cfg["calib_dir"] / "GogEarth_points.kml"
ns = {"k": "http://www.opengis.net/kml/2.2"}
to_site = Transformer.from_crs(
    "EPSG:4326", cfg["crs"], always_xy=True
)  # site CRS from site.json, e.g. Zurich = UTM 32N

for pm in (
    ET.parse(kml_path).getroot().iter("{http://www.opengis.net/kml/2.2}Placemark")
):
    name = pm.find("k:name", ns).text
    lon, lat = map(float, pm.find(".//k:coordinates", ns).text.strip().split(",")[:2])
    x, y = to_site.transform(lon, lat)
    print(
        f'{{"name": "{name}", "pixel": [?, ?], "ground": [{x:.2f}, {y:.2f}], "use": "fit"}},'
    )
