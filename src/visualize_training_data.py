from pathlib import Path
import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.windows import from_bounds
from shapely.geometry import box


def main():
    parser = argparse.ArgumentParser(
        description="Wyświetlanie ortofotomapy z obrysami budynków BDOT10k."
    )
    parser.add_argument(
        "--raster",
        default="data/ortofotomapa_ok.tif",
        help="Ścieżka do ortofotomapy GeoTIFF."
    )
    parser.add_argument(
        "--buildings",
        default="data/budynki_bdot10k.gpkg",
        help="Ścieżka do warstwy budynków wektorowych."
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Nazwa warstwy w pliku wektorowym, jeśli jest wymagana."
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Opcjonalny zasięg mapy w układzie współrzędnych ortofotomapy."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Opcjonalna ścieżka zapisu obrazu PNG."
    )
    parser.add_argument(
        "--max-buildings",
        type=int,
        default=5000,
        help="Maksymalna liczba obrysów wyświetlanych jednocześnie."
    )
    args = parser.parse_args()

    raster_path = Path(args.raster)
    vector_path = Path(args.buildings)

    if not raster_path.exists():
        raise FileNotFoundError(f"Nie znaleziono rastra: {raster_path}")
    if not vector_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wektorowego: {vector_path}")

    # Wczytanie ortofotomapy.
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Ortofotomapa nie ma zdefiniowanego CRS.")

        raster_crs = src.crs
        raster_bounds = src.bounds

        if args.bounds:
            xmin, ymin, xmax, ymax = args.bounds
        else:
            # Domyślnie pokazujemy centralny fragment ortofotomapy.
            width = raster_bounds.right - raster_bounds.left
            height = raster_bounds.top - raster_bounds.bottom

            xmin = raster_bounds.left + width * 0.4
            xmax = raster_bounds.left + width * 0.6
            ymin = raster_bounds.bottom + height * 0.4
            ymax = raster_bounds.bottom + height * 0.6

        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Nieprawidłowy zasięg --bounds.")

        # Przycięcie zasięgu do granic rastra.
        xmin = max(xmin, raster_bounds.left)
        xmax = min(xmax, raster_bounds.right)
        ymin = max(ymin, raster_bounds.bottom)
        ymax = min(ymax, raster_bounds.top)

        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Wybrany zasięg leży poza ortofotomapą.")

        window = from_bounds(
            xmin, ymin, xmax, ymax,
            transform=src.transform
        ).round_offsets().round_lengths()

        # Odczyt pierwszych trzech kanałów RGB.
        if src.count < 3:
            raise ValueError(
                f"Raster ma tylko {src.count} kanał(y). "
                "Do tego skryptu potrzebna jest ortofotomapa RGB."
            )

        rgb = src.read([1, 2, 3], window=window)
        transform = src.window_transform(window)

        # Zmiana układu kanałów: (kanały, wiersze, kolumny) -> (wiersze, kolumny, kanały).
        rgb = reshape_as_image(rgb)

    # Przygotowanie obrazu do wyświetlania.
    if np.issubdtype(rgb.dtype, np.integer):
        max_value = np.iinfo(rgb.dtype).max
        rgb = rgb.astype(np.float32) / max_value
    else:
        rgb = rgb.astype(np.float32)

    # Proste rozciągnięcie kontrastu, przydatne m.in. dla danych 16-bitowych.
    for band in range(3):
        valid = np.isfinite(rgb[:, :, band])
        if valid.any():
            low, high = np.percentile(rgb[:, :, band][valid], [2, 98])
            if high > low:
                rgb[:, :, band] = np.clip(
                    (rgb[:, :, band] - low) / (high - low), 0, 1
                )

    # Wczytanie wektorów i transformacja do CRS ortofotomapy.
    if args.layer:
        buildings = gpd.read_file(vector_path, layer=args.layer)
    else:
        buildings = gpd.read_file(vector_path)

    if buildings.crs is None:
        raise ValueError(
            "Warstwa budynków nie ma CRS. "
            "Ustal jej rzeczywisty układ współrzędnych przed kontynuacją."
        )

    buildings = buildings.to_crs(raster_crs)

    # Zachowanie wyłącznie budynków przecinających wyświetlany fragment.
    view_polygon = box(xmin, ymin, xmax, ymax)
    buildings = buildings[
        buildings.geometry.notna()
        & ~buildings.geometry.is_empty
        & buildings.geometry.intersects(view_polygon)
    ].copy()

    if len(buildings) == 0:
        print("Uwaga: w wybranym fragmencie nie znaleziono budynków.")

    if args.max_buildings > 0 and len(buildings) > args.max_buildings:
        print(
            f"Znaleziono {len(buildings)} budynków. "
            f"Wyświetlam pierwsze {args.max_buildings}."
        )
        buildings = buildings.iloc[:args.max_buildings]

    # Rysowanie.
    fig, ax = plt.subplots(figsize=(14, 12))

    ax.imshow(
        rgb,
        extent=(xmin, xmax, ymin, ymax),
        origin="upper"
    )

    if not buildings.empty:
        buildings.boundary.plot(
            ax=ax,
            color="red",
            linewidth=0.8
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(
        f"Ortofotomapa i obrysy budynków BDOT10k\n"
        f"Liczba widocznych obiektów: {len(buildings)}"
    )
    ax.set_xlabel(f"X [{raster_crs}]")
    ax.set_ylabel(f"Y [{raster_crs}]")

    plt.tight_layout()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Zapisano obraz: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
