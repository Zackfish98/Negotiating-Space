import xml.etree.ElementTree as ET
from pathlib import Path
from pyproj import Transformer

kml_path = Path(__file__).parent / "GogEarth_points.kml"
ns = {"k": "http://www.opengis.net/kml/2.2"}
to_utm = Transformer.from_crs(
    "EPSG:4326", "EPSG:32632", always_xy=True
)  # Zurich = UTM 32N

for pm in (
    ET.parse(kml_path).getroot().iter("{http://www.opengis.net/kml/2.2}Placemark")
):
    name = pm.find("k:name", ns).text
    lon, lat = map(float, pm.find(".//k:coordinates", ns).text.strip().split(",")[:2])
    x, y = to_utm.transform(lon, lat)
    print(
        f'{{"name": "{name}", "pixel": [?, ?], "ground": [{x:.2f}, {y:.2f}], "use": "fit"}},'
    )
