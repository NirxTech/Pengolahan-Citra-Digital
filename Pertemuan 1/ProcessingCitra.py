import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def proses_eksplorasi_citra(image_path):
    # 1. ACQUISITION: Membaca Citra Digital
    # OpenCV membaca dalam format BGR, kita konversi ke RGB untuk visualisasi yang benar
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Gambar tidak ditemukan. Pastikan path benar.")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape

    # 2. REPRESENTASI MATRIKS DAN VEKTOR
    print("=== REPRESENTASI DIGITAL ===")
    # Mengambil matriks 5x5 dari channel Red (R)
    matriks_5x5 = img_rgb[0:5, 0:5, 0] 
    print(f"Matriks 5x5 Piksel Pertama (Channel Red):\n{matriks_5x5}")
    
    # Flatten menjadi vektor
    vektor_citra = img_rgb.flatten()
    print(f"\nUkuran Vektor (Flattened): {vektor_citra.shape[0]} elemen")
    print(f"5 Elemen Pertama Vektor: {vektor_citra[:5]}")
    print("-" * 30)

    # 3. ANALISIS PARAMETER
    print("=== ANALISIS PARAMETER CITRA ===")
    bit_depth = 8  # Standar uint8
    aspect_ratio = w / h
    
    # Hitung ukuran memori asli (dalam MB)
    # Rumus: (Lebar * Tinggi * Channel * BitDepth) / (8 bit * 1024 * 1024)
    mem_size_mb = (w * h * c * bit_depth) / (8 * 1024 * 1024)
    
    print(f"Resolusi Spasial: {w} x {h} piksel")
    print(f"Bit Depth: {bit_depth}-bit per channel")
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
    print(f"Ukuran Memori Asli: {mem_size_mb:.2f} MB")

    # Skenario: Resolusi naik 2x (W*2, H*2) dan Bit Depth turun setengah (4-bit)
    new_w, new_h = w * 2, h * 2
    new_bit_depth = bit_depth / 2
    new_mem_size_mb = (new_w * new_h * c * new_bit_depth) / (8 * 1024 * 1024)
    
    print(f"\nPrediksi Memori (Res 2x, BitDepth 0.5x): {new_mem_size_mb:.2f} MB")
    print("-" * 30)

    # 4. MANIPULASI DASAR
    # A. Cropping (Mengambil bagian tengah citra)
    start_row, start_col = int(h * 0.25), int(w * 0.25)
    end_row, end_col = int(h * 0.75), int(w * 0.75)
    img_cropped = img_rgb[start_row:end_row, start_col:end_col]

    # B. Resizing (Mengecilkan citra menjadi 50%)
    img_resized = cv2.resize(img_rgb, (int(w/2), int(h/2)), interpolation=cv2.INTER_LINEAR)

    # C. Flipping (Horizontal Flip)
    img_flipped = cv2.flip(img_rgb, 1)

    # 5. VISUALISASI UNTUK LAPORAN
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original Citra\n({w}x{h})")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_cropped)
    plt.title("Hasil: Cropping (Center Region)")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_resized)
    plt.title("Hasil: Resizing (50% Scale)")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(img_flipped)
    plt.title("Hasil: Flip Horizontal")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

proses_eksplorasi_citra('TehPucuk.jpg')