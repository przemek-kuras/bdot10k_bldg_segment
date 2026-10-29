import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, bounds
from shapely.geometry import box


def main():
    parser = argparse.ArgumentParser(
        description="Wizualizacja kafla train/val/test z obrysami budynków BDOT10k."
    )
    parser.add_argument(
        "--splits",
        default="outputs/bdot_roofs/splits.csv",
        help="Ścieżka do splits.csv."
    )
    parser.add_argument(
        "--raster",
        default="data/ortofotomapa_ok.tif",
        help="Ścieżka do ortofotomapy."
    )
    parser.add_argument(
        "--buildings",
        required=True,
        help="Ścieżka do pliku wektorowego BDOT10k, np. GPKG lub SHP."
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Opcjonalna nazwa warstwy w pliku wektorowym."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Zbiór, z którego ma zostać wybrany kafel."
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Początkowy wiersz kafla w pikselach, zgodnie z splits.csv."
    )
    parser.add_argument(
        "--col",
        type=int,
        default=None,
        help="Początkowa kolumna kafla w pikselach, zgodnie z splits.csv."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Opcjonalna ścieżka zapisu wizualizacji PNG."
    )
    args = parser.parse_args()

    splits_path = Path(args.splits)
    raster_path = Path(args.raster)
    buildings_path = Path(args.buildings)

    for path in (splits_path, raster_path, buildings_path):
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    # 1. Odczyt istniejącego podziału — nie losujemy go ponownie.
    splits = pd.read_csv(splits_path)

    required_columns = {"row", "col", "split"}
    missing = required_columns - set(splits.columns)
    if missing:
        raise ValueError(f"Brak kolumn w splits.csv: {sorted(missing)}")

    selected = splits[splits["split"] == args.split].copy()

    if selected.empty:
        raise ValueError(f"Brak kafli w zbiorze {args.split!r}.")

    # 2. Wybór konkretnego kafla.
    if args.row is not None or args.col is not None:
        if args.row is None or args.col is None:
            raise ValueError("Parametry --row i --col należy podać razem.")

        selected = selected[
            (selected["row"] == args.row)
            & (selected["col"] == args.col)
        ]

        if selected.empty:
            raise ValueError(
                f"Nie znaleziono kafla row={args.row}, col={args.col} "
                f"w zbiorze {args.split!r}."
            )

    tile_info = selected.iloc[0]
    row = int(tile_info["row"])
    col = int(tile_info["col"])

    print(f"Zbiór: {args.split}")
    print(f"Kafel: row={row}, col={col}")

    # 3. Odczyt dokładnie tego samego fragmentu ortofotomapy.
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Ortofotomapa nie ma zdefiniowanego CRS.")
        if src.count < 3:
            raise ValueError("Do wizualizacji RGB potrzebne są co najmniej 3 kanały.")

        # Rozmiar kafla odczytujemy z metadanych zbioru, jeśli są dostępne.
        if "image" in splits.columns:
            image_path = str(tile_info["image"])
            tile_size = None

            if Path(image_path).exists():
                tile_size = np.load(image_path, mmap_mode="r").shape[0]

        else:
            tile_size = None

        # Standardowy rozmiar stosowany przez prepare_data.py.
        if tile_size is None:
            tile_size = 512

        window = Window(col, row, tile_size, tile_size)

        if (
            row < 0 or col < 0
            or row + tile_size > src.height
            or col + tile_size > src.width
        ):
            raise ValueError("Wybrany kafel wykracza poza granice ortofotomapy.")

        rgb = np.moveaxis(src.read([1, 2, 3], window=window), 0, -1)
        raster_crs = src.crs
        left, bottom, right, top = bounds(window, src.transform)

    # 4. Przygotowanie RGB do wyświetlania.
    rgb = rgb.astype(np.float32)
    for band in range(3):
        values = rgb[:, :, band]
        valid = np.isfinite(values)

        if valid.any():
            low, high = np.percentile(values[valid], [2, 98])
            if high > low:
                rgb[:, :, band] = np.clip(
                    (values - low) / (high - low), 0, 1
                )
            else:
                rgb[:, :, band] = 0

    # 5. Wczytanie budynków i przeliczenie ich do CRS ortofotomapy.
    if args.layer:
        buildings = gpd.read_file(buildings_path, layer=args.layer)
    else:
        buildings = gpd.read_file(buildings_path)

    if buildings.crs is None:
        raise ValueError(
            "Warstwa budynków nie ma CRS. "
            "Należy ustalić jej rzeczywisty układ współrzędnych."
        )

    buildings = buildings.to_crs(raster_crs)

    tile_geom = box(left, bottom, right, top)

    buildings = buildings[
        buildings.geometry.notna()
        & ~buildings.geometry.is_empty
        & buildings.geom_type.isin(["Polygon", "MultiPolygon"])
        & buildings.intersects(tile_geom)
    ].copy()

    # Przycinamy poligony do granic kafla.
    buildings["geometry"] = buildings.geometry.intersection(tile_geom)
    buildings = buildings[
        buildings.geometry.notna() & ~buildings.geometry.is_empty
    ]

    # 6. Wizualizacja.
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(
        rgb,
        extent=(left, right, bottom, top),
        origin="upper"
    )

    if not buildings.empty:
        buildings.boundary.plot(
            ax=ax,
            color="red",
            linewidth=0.8
        )

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_title(
        f"Kafel {args.split}: row={row}, col={col}\n"
        f"Liczba budynków: {len(buildings)}"
    )
    ax.set_xlabel(f"X ({raster_crs})")
    ax.set_ylabel(f"Y ({raster_crs})")

    plt.tight_layout()

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Zapisano wizualizację: {output}")

    plt.show()


if __name__ == "__main__":
    main()
