import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm
from src.common import load_config, normalize_image, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    img_dir, mask_dir = out / "tiles" / "images", out / "tiles" / "masks"
    ensure_dir(img_dir); ensure_dir(mask_dir)

    layer = cfg.get("buildings_layer")
    buildings = gpd.read_file(cfg["buildings_vector"], layer=layer)
    buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
    buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if buildings.empty:
        raise ValueError("No polygon buildings found in the vector layer.")

    records = []
    seed = int(cfg.get("random_seed", 42))
    rng = random.Random(seed)
    raster_paths = [Path(p) for p in cfg["orthophotos"]]
    # Use independent rasters when there are at least three; otherwise split by spatial blocks after tiling.
    shuffled = list(range(len(raster_paths)))
    rng.shuffle(shuffled)
    n = len(shuffled)
    use_raster_split = n >= 3
    split_by_raster = {}
    if use_raster_split:
        n_test = max(1, int(round(n * float(cfg.get("test_fraction", 0.15)))))
        n_val = max(1, int(round(n * float(cfg.get("validation_fraction", 0.15)))))
        while n_test + n_val >= n:
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        for rank, idx in enumerate(shuffled):
            split_by_raster[str(raster_paths[idx])] = (
                "test" if rank < n_test else ("val" if rank < n_test + n_val else "train")
            )

    tile = int(cfg.get("tile_size", 512))
    stride = int(cfg.get("stride", tile))
    bands = [int(x) for x in cfg.get("bands", [1,2,3])]
    erosion = float(cfg.get("mask_erosion_m", 0.0))
    all_touched = bool(cfg.get("all_touched", False))
    min_pixels = int(cfg.get("min_building_pixels", 12))

    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise ValueError(f"{raster_path} has no CRS.")
            if max(bands) > src.count:
                raise ValueError(f"{raster_path} has {src.count} bands but requested {bands}.")
            b = buildings.to_crs(src.crs)
            if erosion:
                if not src.crs.is_projected:
                    raise ValueError("mask_erosion_m requires a projected CRS in metres.")
                # This checks projection, but projected units are not universally metres.
                unit = (src.crs.linear_units or "").lower()
                if "metre" not in unit and "meter" not in unit:
                    raise ValueError(f"CRS linear units are {src.crs.linear_units!r}, not metres.")
                b = b.copy()
                b["geometry"] = b.geometry.buffer(-erosion)
                b = b[b.geometry.notna() & ~b.geometry.is_empty]
            raster_bounds = box(*src.bounds)
            local = b[b.intersects(raster_bounds)]
            shapes = [(geom, 1) for geom in local.geometry if geom is not None and not geom.is_empty]
            if not shapes:
                print(f"Warning: no buildings intersect {raster_path}; skipping.")
                continue
            for row in tqdm(range(0, src.height - tile + 1, stride), desc=raster_path.name):
                for col in range(0, src.width - tile + 1, stride):
                    window = rasterio.windows.Window(col, row, tile, tile)
                    transform = rasterio.windows.transform(window, src.transform)
                    raw = src.read(bands, window=window)
                    if raw.shape[1] != tile or raw.shape[2] != tile:
                        continue
                    img = np.moveaxis(raw, 0, -1)
                    # Skip wholly nodata chips when nodata is defined.
                    if src.nodata is not None and np.all(raw == src.nodata):
                        continue
                    tile_bounds = rasterio.windows.bounds(window, src.transform)
                    tile_geom = box(*tile_bounds)
                    candidates = [(g, 1) for g in local.geometry if g is not None and not g.is_empty and g.intersects(tile_geom)]
                    if not candidates:
                        mask = np.zeros((tile, tile), dtype=np.uint8)
                    else:
                        mask = rasterize(
                            candidates, out_shape=(tile, tile), transform=transform,
                            fill=0, all_touched=all_touched, dtype="uint8"
                        )
                    if int(mask.sum()) < min_pixels and not candidates:
                        # Keep some background-only examples; useful for reducing false positives.
                        pass
                    stem = f"{raster_path.stem}_r{row}_c{col}"
                    ip = img_dir / f"{stem}.npy"
                    mp = mask_dir / f"{stem}.npy"
                    np.save(ip, normalize_image(img).astype(np.float32))
                    np.save(mp, mask.astype(np.uint8))
                    records.append({
                        "image": str(ip), "mask": str(mp),
                        "raster": str(raster_path), "row": row, "col": col,
                        "split": split_by_raster.get(str(raster_path), "pending"),
                        "positive_pixels": int(mask.sum())
                    })
    if not records:
        raise RuntimeError("No tiles generated. Check raster dimensions, CRS, and building overlap.")
    df = pd.DataFrame(records)
    if not use_raster_split:
        # Keep 4x4-tile spatial blocks together to reduce leakage between neighboring tiles.
        block_size_px = tile * 4
        df["block_id"] = (
            df["raster"].astype(str) + "_b" +
            (df["row"] // block_size_px).astype(str) + "_" +
            (df["col"] // block_size_px).astype(str)
        )
        blocks = df["block_id"].drop_duplicates().tolist()
        rng.shuffle(blocks)
        if len(blocks) < 3:
            raise ValueError(
                f"Only {len(blocks)} spatial blocks were generated. Need at least 3 blocks for train/val/test; "
                "use a larger raster or provide more independent areas."
            )
        n_test = max(1, int(round(len(blocks) * float(cfg.get("test_fraction", 0.15)))))
        n_val = max(1, int(round(len(blocks) * float(cfg.get("validation_fraction", 0.15)))))
        while n_test + n_val >= len(blocks):
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        block_split = {}
        for i, block in enumerate(blocks):
            block_split[block] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
        df["split"] = df["block_id"].map(block_split)
        df.drop(columns=["block_id"], inplace=True)
    df.to_csv(out / "splits.csv", index=False)
    print(df.groupby("split").size().to_string())
    if set(df["split"]) != {"train", "val", "test"}:
        print("WARNING: not all train/val/test splits are present. Increase study area or supply more independent spatial blocks.")
    print(f"Created {len(df)} tiles in {out / 'tiles'}")

if __name__ == "__main__":
    main()
