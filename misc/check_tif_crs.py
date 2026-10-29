import os
import rasterio

def sprawdz_uklad_tif(sciezka_tif):
    print(f"Analizuję plik: {os.path.basename(sciezka_tif)}...")
    
    try:
        # Otwarcie pliku rastrowego w trybie odczytu
        with rasterio.open(sciezka_tif) as dataset:
            crs = dataset.crs
            
            if crs is None:
                print("\n[!] UWAGA: Ten plik TIF nie posiada przypisanego układu współrzędnych (brak metadanych GeoTIFF).")
                print("Możliwe, że potrzebujesz pliku tekstowego z georeferencją (np. .tfw).")
                return

            print("\n=== INFORMACJE O UKŁADZIE WSPÓŁRZĘDNYCH ===")
            
            # 1. Wyświetlenie kodu EPSG (najbardziej uniwersalny)
            if crs.is_epsg_code:
                print("To jest kod EPSG")
                print(f"Numer EPSG:{crs.to_epsg()}")
            else:
                print("Kod EPSG:    Brak bezpośredniego kodu EPSG w metadanych.")
                
            # 2. Wyświetlenie pełnej nazwy tekstowej układu
            print(f"Nazwa CRS:   {crs.to_string()}")
            
            # 3. Dodatkowe informacje o jednostkach i typie
            print(f"Jednostka:   {crs.linear_units}")
            print(f"Typ układu:  {'Rzutowany (Projected)' if crs.is_projected else 'Geograficzny (Geographic)'}")
            
            # Podstawowe parametry rastra dla pewności
            print("\n=== PARAMETRY RASTRA ===")
            print(f"Rozdzielczość (szerokość x wysokość): {dataset.width} x {dataset.height} px")
            print(f"Liczba kanałów (pasm):                {dataset.count}")
            print(f"Wymiary piksela (X, Y):               {dataset.res[0]:.3f}m x {dataset.res[1]:.3f}m")

    except rasterio.errors.RasterioIOError:
        print(f"\nBłąd: Nie można otworzyć pliku. Upewnij się, że ścieżka jest poprawna.")
    except Exception as e:
        print(f"\nWystąpił nieoczekiwany błąd: {e}")

# --- KONFIGURACJA ---
# Wpisz tutaj ścieżkę do swojego pliku TIF (np. "C:/dane/ortofoto.tif")
sciezka_do_pliku_tif = "data/ortofotomapa_ok.tif"

if os.path.exists(sciezka_do_pliku_tif):
    sprawdz_uklad_tif(sciezka_do_pliku_tif)
else:
    print(f"Błąd: Plik '{sciezka_do_pliku_tif}' nie istnieje. Sprawdź nazwę i ścieżkę.")