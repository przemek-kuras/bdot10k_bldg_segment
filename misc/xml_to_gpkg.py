import os
import geopandas as gpd

def konwertuj_bdot_xml_na_gpkg(sciezka_xml, sciezka_gpkg):
    print("Rozpoczynam wczytywanie pliku XML...")
    
    # BDOT10k zapisuje obrysy budynków w warstwie o nazwie 'OT_BUBD_A'
    layer_name = 'OT_BUBD_A'
    
    try:
        # Odczyt specyficznej warstwy budynków bezpośrednio z XML/GML
        print(f"Szukam warstwy {layer_name} w pliku źródłowym...")
        baza_budynkow = gpd.read_file(sciezka_xml, layer=layer_name)
        
        # Sprawdzenie czy układ współrzędnych (CRS) jest ustawiony, jeśli nie - wymuszenie EPSG:2180
        if baza_budynkow.crs is None:
            print("Nie wykryto układu współrzędnych. Ustawiam domyślny dla BDOT10k (EPSG:2180)...")
            baza_budynkow.set_crs(epsg=2180, inplace=True)
            
        print(f"Wczytano pomyślnie {len(baza_budynkow)} obiektów (budynków).")
        print(f"Zapisywanie do formatu GeoPackage: {sciezka_gpkg}...")
        
        # Zapis do pliku GPKG
        baza_budynkow.to_file(sciezka_gpkg, layer='budynki_bdot10k', driver="GPKG")
        print("Konwersja zakończona sukcesem!")
        
    except ValueError as ve:
        print(f"\nBłąd: Nie znaleziono warstwy {layer_name} w tym pliku XML.")
        print("Możliwe, że ten plik XML zawiera inne warstwy BDOT10k. Sprawdź strukturę danych.")
    except Exception as e:
        print(f"\nWystąpił błąd podczas konwersji: {e}")

# --- KONFIGURACJA ŚCIEŻEK ---
# Podaj nazwy swoich plików (mogą być ścieżki bezwzględne, np. 'C:/dane/bdot.xml')
plik_wejsciowy_xml = "data/bdot10k_budynki.xml" 
plik_wyjsciowy_gpkg = "data/budynki_bdot10k.gpkg"

if os.path.exists(plik_wejsciowy_xml):
    konwertuj_bdot_xml_na_gpkg(plik_wejsciowy_xml, plik_wyjsciowy_gpkg)
else:
    print(f"Błąd: Plik źródłowy '{plik_wejsciowy_xml}' nie istnieje w bieżącym katalogu.")