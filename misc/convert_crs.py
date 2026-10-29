import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# 1. Definiowanie ścieżek do plików oraz układów współrzędnych
input_raster = "data/ortofotomapa2.tif"    # Ścieżka do Twojego pliku wejściowego
output_raster = "data/ortofotomapa2_ok.tif"  # Ścieżka, gdzie ma zapisać się nowy plik

src_crs = "EPSG:2178"  # Układ źródłowy (1965 strefa 4)
dst_crs = "EPSG:2180"  # Układ docelowy (1992)

# 2. Otwarcie wejściowego rastra
with rasterio.open(input_raster) as src:
    
    # 3. Obliczenie nowych wymiarów, transformacji (geotransformu) i granic dla EPSG:2180
    transform, width, height = calculate_default_transform(
        src.crs if src.crs else src_crs, # Używa CRS z pliku, a jeśli brak, to wymusza EPSG:2178
        dst_crs, 
        src.width, 
        src.height, 
        *src.bounds
    )
    
    # 4. Przygotowanie metadanych dla nowego pliku GeoTIFF
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    
    # 5. Zapisanie przetransformowanego rastra do nowego pliku
    with rasterio.open(output_raster, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs if src.crs else src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest # Zmień na Resampling.bilinear dla danych ciągłych (np. DEM)
            )

print(f"Sukces! Raster został przetransformowany i zapisany jako: {output_raster}")