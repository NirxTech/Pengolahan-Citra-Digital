import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_zone_plate(size=512):
    """Membuat pola Zone Plate (Lingkaran konsentris) untuk tes Aliasing"""
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    # Fungsi sinus dengan frekuensi meningkat ke arah luar
    km = 0.7 * np.pi
    img = np.sin(km * r**2 / size)
    
    # Normalisasi ke 0-255
    img_uint8 = ((img + 1) / 2 * 255).astype(np.uint8)
    return img_uint8

def simulate_image_aliasing(image, factor):
    """
    Simulasi aliasing dengan membandingkan teknik downsampling:
    1. Naive (Decimation) -> Menyebabkan Aliasing
    2. Proper (Interpolation) -> Anti-Aliasing
    """
    print(f"\nSimulasi Aliasing dengan faktor downsampling: {factor}x")
    
    h, w = image.shape[:2]
    new_h, new_w = h // factor, w // factor
    
    # 1. NAIVE DOWNSAMPLING (Ambil piksel loncat-loncat)
    # Ini mensimulasikan sampling tanpa filter anti-aliasing (Low Pass Filter)
    # Hasilnya akan muncul pola aneh (Moire Pattern)
    aliased_img = image[::factor, ::factor]
    
    # 2. ANTI-ALIASED DOWNSAMPLING (Area Interpolation / Gaussian Blur dulu)
    # Ini cara yang benar: filter frekuensi tinggi dulu, baru resize
    proper_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # --- VISUALISASI ---
    plt.figure(figsize=(15, 6))
    
    # A. Citra Asli
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Original High-Res ({h}x{w})")
    plt.axis('off')
    
    # B. Hasil Aliasing (Buruk)
    plt.subplot(1, 3, 2)
    plt.imshow(aliased_img, cmap='gray')
    plt.title(f"Naive Sampling (Aliasing!)\nPerhatikan Pola Moire")
    plt.axis('off')
    
    # C. Hasil Anti-Aliasing (Benar)
    plt.subplot(1, 3, 3)
    plt.imshow(proper_img, cmap='gray')
    plt.title(f"Anti-Aliased (Correct)\nPola lebih bersih/blur")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- MAIN PROGRAM ---
if __name__ == "__main__":
    # 1. Buat Pola Zone Plate (Paling bagus buat lihat aliasing)
    print("Membuat pola uji Zone Plate...")
    img_pattern = create_zone_plate(size=600)
    
    # 2. Lakukan simulasi dengan faktor 4x
    # (Artinya resolusi turun dari 600x600 jadi 150x150)
    simulate_image_aliasing(img_pattern, factor=4)