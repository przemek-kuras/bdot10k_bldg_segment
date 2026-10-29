# BDOT10k Building-Roof Segmentation

Pipeline for training a semantic segmentation model to extract visible roof footprints from orthophotos using BDOT10k building polygons as initial labels.

## Important limitations
- BDOT10k building footprints are not necessarily identical to visible roof outlines. Inspect overlays and tune `mask_erosion_m`.
- Use a projected CRS in metres for metric buffering/erosion. The code rejects geographic CRS for erosion.
- Split data by geographic area, not random neighboring tiles, to avoid spatial leakage.
- The included pipeline is a practical baseline, not a guarantee of cadastral accuracy.
- If imagery is RGB, the pipeline uses three bands. Set `bands` in the config if your GeoTIFF band order differs.

## 1. Environment
Recommended: Windows/Linux with NVIDIA GPU, Python 3.10 or 3.11, and a compatible PyTorch CUDA build.

Create an environment, then install PyTorch from https://pytorch.org/get-started/locally/ for your CUDA version. Then:
```bash
pip install -r requirements.txt
```
Check GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Input data
- One or more georeferenced orthophoto GeoTIFFs.
- BDOT10k building polygons as GeoPackage, Shapefile, or another vector format readable by GeoPandas.
- Ensure the vector layer contains building polygons only, or filter it before running.

Edit `configs/config.yaml`:
- `orthophotos`: list of GeoTIFF paths
- `buildings_vector`: BDOT10k vector file path
- `buildings_layer`: optional layer name (null for default)
- `output_dir`: output workspace
- `tile_size`: 512 by default
- `stride`: 512 by default; use 384 for overlap
- `bands`: 1-based band indices, e.g. `[1, 2, 3]`
- `mask_erosion_m`: default 0.0. Try 0, 0.25, 0.5, 1.0 m and visually inspect; only use a metric projected CRS.
- `validation_fraction` and `test_fraction`: split by source raster file. For a single large orthophoto, split by spatial blocks instead; see note below.

## 3. Run the pipeline
```bash
python -m src.prepare_data --config configs/config.yaml
python -m src.train --config configs/config.yaml
python -m src.evaluate --config configs/config.yaml
python -m src.predict --config configs/config.yaml --input data/orthophoto.tif --output outputs/roof_polygons.gpkg
```

You may also run inference over each input GeoTIFF:
```bash
python -m src.predict --config configs/config.yaml --input data/orthophoto.tif --output outputs/roof_polygons.gpkg
```

## 4. Spatial split warning
With at least three input rasters, the script assigns entire rasters to train/validation/test. With one or two rasters, it assigns 4×4-tile spatial blocks to splits. This reduces spatial leakage but is not a substitute for a genuinely independent test area. Do not randomly split neighboring tiles.

## 5. Outputs
- `output_dir/tiles/images/*.npy`: normalized RGB image chips
- `output_dir/tiles/masks/*.npy`: binary label masks
- `output_dir/splits.csv`: tile paths and split
- `output_dir/checkpoints/best.pt`: best validation checkpoint
- `output_dir/metrics.json`: test metrics
- GeoPackage: predicted roof polygons in input raster CRS

## 6. Practical improvements
1. Inspect 50–100 image/mask overlays before training.
2. Exclude cloud, shadow, nodata, and edge tiles where possible.
3. Consider hard-negative examples (bright concrete, roads, parking areas).
4. Compare U-Net against SegFormer after establishing a baseline.
5. Evaluate polygon geometry in addition to pixel metrics.
