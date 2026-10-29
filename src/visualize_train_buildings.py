import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds
from shapely.geometry import box
from shapely.ops import unary_union


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Wyświetla całą ortofotomapę i budynki BDOT10k "
            "wykorzystane w zbiorze treningowym."
        )
    )
    parser.add_argument("--splits", required=True, help="Ścieżka do splits.csv")
    parser.add_argument("--raster", required=True, help="Ortofotomapa GeoTIFF")
    parser.add_argument("--buildings", required=True, help="Warstwa budynków BDOT10k")
    parser.add_argument("--layer", default=None, help="Nazwa warstwy wektorowej")
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="Rozmiar kafla w pikselach (domyślnie 512)"
    )
    parser.add_argument(
        "--max-side", type=int, default=3000,
        help="Maksymalny wymiar obrazu wyświetlanego w pikselach"
    )
    parser.add_argument("--output", default=None, help="Opcjonalny plik PNG")
    args = parser.parse_args()

    splits_path = Path(args.splits)
    raster_path = Path(args.raster)
    buildings_path = Path(args.buildings)

    for path in (splits_path, raster_path, buildings_path):
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    if args.tile_size <= 0 or args.max_side <= 0:
        raise ValueError("--tile-size i --max-side muszą być dodatnie.")

    # 1. Wczytaj istniejący podział zbioru danych.
    df = pd.read_csv(splits_path)

    required = {"row", "col", "split"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Brak wymaganych kolumn w splits.csv: {required - set(df.columns)}"
        )

    train = df[df["split"] == "train"].copy()
    if train.empty:
        raise ValueError("W splits.csv nie znaleziono kafli train.")

    # Jeśli CSV zawiera wiele ortofotomap, wybierz rekordy
    # odpowiadające wskazanemu plikowi.
    if "raster" in train.columns:
        raster_names = train["raster"].dropna().astype(str).unique()

        def same_path(value):
            p = Path(value)
            if not p.is_absolute():
                p = Path.cwd() / p
            try:
                return p.resolve() == raster_path.resolve()
            except OSError:
                return p.name == raster_path.name

        matched = train[train["raster"].map(same_path)]
        if not matched.empty:
            train = matched
        elif len(raster_names) > 1:
            raise ValueError(
                "splits.csv zawiera wiele rastrów, ale nie znaleziono "
                "wierszy odpowiadających podanej ortofotomapie. "
                "Sprawdź ścieżkę --raster."
            )
        elif len(raster_names) == 1:
            print(
                "Uwaga: ścieżka rastra w CSV różni się od --raster. "
                "Przyjęto, że chodzi o tę samą ortofotomapę."
            )

    print(f"Liczba kafli train: {len(train)}")

    # 2. Odtwórz geograficzny zasięg wszystkich kafli treningowych.
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Ortofotomapa nie ma zdefiniowanego CRS.")
        if src.count < 3:
            raise ValueError("Do wizualizacji RGB potrzebne są 3 kanały.")

        tile_geometries = []
        for row, col in train[["row", "col"]].drop_duplicates().itertuples(
            index=False, name=None
        ):
            row, col = int(row), int(col)

            if (
                row < 0 or col < 0
                or row + args.tile_size > src.height
                or col + args.tile_size > src.width
            ):
                continue

            window = Window(col, row, args.tile_size, args.tile_size)
            left, bottom, right, top = bounds(window, src.transform)
            tile_geometries.append(box(left, bottom, right, top))

        if not tile_geometries:
            raise ValueError(
                "Nie utworzono zasięgu kafli. Sprawdź rozmiar kafla "
                "oraz wartości row i col w splits.csv."
            )

        train_coverage = unary_union(tile_geometries)
        raster_crs = src.crs
        raster_bounds = src.bounds

        # 3. Wczytaj całą ortofotomapę w pomniejszonej rozdzielczości.
        scale = min(
            1.0,
            args.max_side / max(src.width, src.height)
        )
        out_width = max(1, round(src.width * scale))
        out_height = max(1, round(src.height * scale))

        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, out_height, out_width),
            resampling=Resampling.average
        )
        extent = (
            raster_bounds.left,
            raster_bounds.right,
            raster_bounds.bottom,
            raster_bounds.top
        )

    # Konwersja obrazu do postaci RGB i rozciągnięcie kontrastu.
    rgb = np.moveaxis(rgb, 0, -1).astype(np.float32)

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

    # 4. Wczytaj BDOT10k i przelicz geometrię do CRS ortofotomapy.
    if args.layer:
        buildings = gpd.read_file(buildings_path, layer=args.layer)
    else:
        buildings = gpd.read_file(buildings_path)

    if buildings.crs is None:
        raise ValueError(
            "Warstwa BDOT10k nie ma CRS. Ustal jej rzeczywisty układ "
            "współrzędnych przed kontynuacją."
        )

    buildings = buildings.to_crs(raster_crs)
    buildings = buildings[
        buildings.geometry.notna()
        & ~buildings.geometry.is_empty
        & buildings.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    # Ogranicz obliczenia do obiektów w obrębie ortofotomapy.
    raster_polygon = box(
        raster_bounds.left, raster_bounds.bottom,
        raster_bounds.right, raster_bounds.top
    )
    buildings = buildings[buildings.intersects(raster_polygon)].copy()

    # 5. Wybierz części budynków, które faktycznie pokrywają kafle train.
    buildings_train = buildings[
        buildings.intersects(train_coverage)
    ].copy()

    buildings_train["geometry"] = buildings_train.geometry.intersection(
        train_coverage
    )
    buildings_train = buildings_train[
        buildings_train.geometry.notna()
        & ~buildings_train.geometry.is_empty
    ].copy()

    print(f"Liczba budynków BDOT10k przecinających train: {len(buildings_train)}")

    # 6. Wyświetl całą ortofotomapę i obrysy budynków treningowych.
    fig, ax = plt.subplots(figsize=(16, 12))

    ax.imshow(rgb, extent=extent, origin="upper")

    buildings_train.boundary.plot(
        ax=ax,
        color="red",
        linewidth=0.45
    )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title(
        "Budynki BDOT10k na kaflach treningowych\n"
        f"Kafle train: {len(tile_geometries)} | "
        f"Budynki: {len(buildings_train)}"
    )
    ax.set_xlabel(f"X ({raster_crs})")
    ax.set_ylabel(f"Y ({raster_crs})")

    plt.tight_layout()

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Zapisano mapę: {output}")

    plt.show()


if __name__ == "__main__":
    main()
