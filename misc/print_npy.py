import numpy as np

# 1. Wczytanie pliku .npy
macierz = np.load('outputs/bdot_roofs/tiles/images/ortofotomapa_ok_r7680_c4608.npy')
mmacierz = np.load('outputs/bdot_roofs/tiles/masks/ortofotomapa_ok_r7680_c4608.npy')

# 2. Wyświetlenie macierzy w konsoli
print(macierz[106][106])
print(mmacierz[106][106])