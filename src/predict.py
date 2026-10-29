import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
import torch
from src.common import load_config, make_model, device, normalize_image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True, help="Input georeferenced orthophoto GeoTIFF")
    ap.add_argument("--output", required=True, help="Output GeoPackage path")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    dev = device()
    model = make_model(cfg).to(dev)
    ckpt_path = Path(cfg["output_dir"]) / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"]); model.eval()
    bands = [int(x) for x in cfg.get("bands", [1,2,3])]
    tile = int(cfg.get("tile_size", 512))
    stride = int(cfg.get("stride", tile))
    pred_cfg = cfg.get("prediction", {})
    threshold = float(pred_cfg.get("threshold", 0.5))
    min_area = float(pred_cfg.get("min_polygon_area_m2", 4.0))
    simplify = float(pred_cfg.get("simplify_tolerance_m", 0.1))
    geometries = []
    with rasterio.open(args.input) as src:
        if src.crs is None:
            raise ValueError("Input GeoTIFF has no CRS.")
        if max(bands) > src.count:
            raise ValueError("Configured band index exceeds raster band count.")
        if not src.crs.is_projected and (min_area > 0 or simplify > 0):
            raise ValueError("Area filtering and simplification require projected CRS in metres; use min_polygon_area_m2=0 and simplify_tolerance_m=0 for geographic CRS.")
        for row in tqdm(range(0, src.height, stride), desc="Predicting"):
            for col in range(0, src.width, stride):
                h, w = min(tile, src.height-row), min(tile, src.width-col)
                # Pad edge windows to fixed tile dimensions, then crop predictions back.
                win = Window(col, row, w, h)
                raw = src.read(bands, window=win)
                img = np.moveaxis(raw, 0, -1)
                img = normalize_image(img)
                pad = np.zeros((tile, tile, img.shape[2]), dtype=np.float32)
                pad[:h, :w] = img
                x = torch.from_numpy(np.moveaxis(pad, -1, 0)[None]).float().to(dev)
                with torch.no_grad():
                    prob = torch.sigmoid(model(x))[0,0,:h,:w].cpu().numpy()
                mask = (prob >= threshold).astype(np.uint8)
                transform = rasterio.windows.transform(win, src.transform)
                for geom, val in shapes(mask, mask=mask.astype(bool), transform=transform):
                    if val != 1:
                        continue
                    poly = shape(geom)
                    if poly.is_empty:
                        continue
                    if simplify > 0:
                        poly = poly.simplify(simplify, preserve_topology=True)
                    if poly.area >= min_area:
                        geometries.append(poly)
        crs = src.crs
    if not geometries:
        print("No polygons detected at the current threshold.")
        gdf = gpd.GeoDataFrame({"class": [], "area": []}, geometry=[], crs=crs)
    else:
        gdf = gpd.GeoDataFrame({"class": ["building_roof"]*len(geometries),
                                "area": [float(g.area) for g in geometries]},
                               geometry=geometries, crs=crs)
        # Optional validity repair and remove empty results.
        gdf.geometry = gdf.geometry.make_valid()
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf.to_file(out_path, layer="roof_polygons", driver="GPKG")
    print(f"Saved {len(gdf)} polygons to {out_path}")

if __name__ == "__main__":
    main()
