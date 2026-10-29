import torch

# 1. Sprawdzenie czy CUDA jest ogólnie dostępna
cuda_dostepna = torch.cuda.is_available()
print(f"Czy CUDA jest dostępna? {cuda_dostepna}")

if cuda_dostepna:
    # 2. Nazwa wykrytej karty graficznej
    print(f"Nazwa karty: {torch.cuda.get_device_name(0)}")
    
    # 3. Test przeniesienia obliczeń na GPU
    x = torch.rand(3, 3).to("cuda")
    print("\nPrzykładowy tensor na GPU:")
    print(x)
else:
    print("PyTorch nie widzi Twojej karty. Zainstalowałeś wersję CPU.")