import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from PIL import Image


def resolve_path(value):
    """Rozwiązuje ścieżkę bezwzględną lub względną względem katalogu projektu."""
    path = Path(str(value))
    if path.exists():
        return path
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Wizualizacja masek budynków zbioru testowego na ortofotomapie."
    )
    parser.add_argument("--splits", required=True, help="Ścieżka do splits.csv")
    parser.add_argument("--raster", required=True, help="Ortofotomapa GeoTIFF")
    parser.add_argument(
        "--output",
        default="outputs/test_mask.png",
        help="Ścieżka wynikowego pliku PNG"
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=3000,
        help="Maksymalny wymiar obrazu wynikowego w pikselach"
    )
    args = parser.parse_args()

    splits_path = Path(args.splits)
    raster_path = Path(args.raster)
    output_path = Path(args.output)

    if not splits_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {splits_path}")
    if not raster_path.exists():
        raise FileNotFoundError(f"Nie znaleziono rastra: {raster_path}")

    if args.max_side <= 0:
        raise ValueError("--max-side musi być dodatnie.")

    # 1. Wczytanie istniejącego podziału.
    df = pd.read_csv(splits_path)

    required = {"split", "mask", "row", "col"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brak kolumn w splits.csv: {sorted(missing)}")

    test_df = df[df["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("W splits.csv nie znaleziono kafli test.")

    # Jeśli CSV zawiera wiele rastrów, wybierz tylko wskazany raster.
    if "raster" in test_df.columns:
        def same_raster(value):
            p = Path(str(value))
            if not p.is_absolute():
                p = Path.cwd() / p
            try:
                return p.resolve() == raster_path.resolve()
            except OSError:
                return p.name == raster_path.name

        matched = test_df[test_df["raster"].map(same_raster)]

        if not matched.empty:
            test_df = matched
        elif test_df["raster"].nunique() > 1:
            raise ValueError(
                "Nie znaleziono kafli test dla wskazanego rastra. "
                "Sprawdź ścieżkę --raster."
            )
        else:
            raise ValueError(
                "Ścieżka rastra w splits.csv nie odpowiada argumentowi --raster."
            )

    print(f"Liczba kafli test: {len(test_df)}")

    # 2. Odczyt całej ortofotomapy w pomniejszonej rozdzielczości.
    with rasterio.open(raster_path) as src:
        if src.count < 3:
            raise ValueError("Ortofotomapa musi mieć co najmniej 3 kanały RGB.")
        if src.crs is None:
            raise ValueError("Ortofotomapa nie ma zdefiniowanego CRS.")

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

        raster_width = src.width
        raster_height = src.height

    # Postać obrazu: wysokość x szerokość x kanały.
    rgb = np.moveaxis(rgb, 0, -1).astype(np.float32)

    # Rozciągnięcie kontrastu osobno dla każdego kanału.
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

    # 3. Utworzenie maski zbiorczej w rozdzielczości podglądu.
    # Wartość True oznacza piksel budynku w masce referencyjnej.
    test_mask = np.zeros((out_height, out_width), dtype=bool)

    for _, record in test_df.iterrows():
        mask_path = resolve_path(record["mask"])

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono maski z splits.csv: {mask_path}"
            )

        mask = np.load(mask_path)
        mask = np.squeeze(mask)

        if mask.ndim != 2:
            raise ValueError(
                f"Maska {mask_path} nie jest tablicą 2D: {mask.shape}"
            )

        row = int(record["row"])
        col = int(record["col"])
        tile_height, tile_width = mask.shape

        # Pominięcie kafli poza granicami rastra.
        if (
            row < 0 or col < 0
            or row + tile_height > raster_height
            or col + tile_width > raster_width
        ):
            print(f"Pomijam kafel poza rastrem: {mask_path}")
            continue

        # Przeliczenie granic kafla na piksele obrazu podglądowego.
        x0 = round(col * out_width / raster_width)
        x1 = round((col + tile_width) * out_width / raster_width)
        y0 = round(row * out_height / raster_height)
        y1 = round((row + tile_height) * out_height / raster_height)

        x0 = max(0, min(out_width, x0))
        x1 = max(0, min(out_width, x1))
        y0 = max(0, min(out_height, y0))
        y1 = max(0, min(out_height, y1))

        if x1 <= x0 or y1 <= y0:
            continue

        # Skalowanie maski metodą najbliższego sąsiada.
        # Dzięki temu nie powstają pośrednie wartości pikseli.
        mask_small = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (x1 - x0, y1 - y0),
                resample=Image.Resampling.NEAREST
            )
        ) > 0

        # Łączenie masek wszystkich kafli test.
        test_mask[y0:y1, x0:x1] |= mask_small

    # 4. Nałożenie czerwonego koloru na piksele budynków.
    overlay = rgb.copy()
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    alpha = 0.65

    overlay[test_mask] = (
        (1 - alpha) * overlay[test_mask] + alpha * red
    )

    # 5. Zapis PNG.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(overlay, origin="upper")
    ax.set_title(
        "Maski referencyjne budynków — zbiór testowy\n"
        f"Kafle: {len(test_df)} | "
        f"Piksele budynków w podglądzie: {int(test_mask.sum())}"
    )
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Zapisano PNG: {output_path}")


if __name__ == "__main__":
    main()
