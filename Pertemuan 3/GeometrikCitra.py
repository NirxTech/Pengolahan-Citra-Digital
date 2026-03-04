import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# ==========================================
# IDENTITAS
# Nama: Muhammad Zahran
# NIM : 24343077
# ==========================================

def print_identity():
    print("=" * 35)
    print("      PROGRAM REGISTRASI CITRA")
    print(f"Nama : Muhammad Zahran")
    print(f"NIM  : 24343077")
    print("=" * 35)

def evaluate_quality(target, result):
    # MSE
    mse = np.mean((target - result) ** 2)
    # PSNR
    if mse == 0:
        psnr = 100
    else:
        pixel_max = 255.0
        psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    return mse, psnr

def main():
    print_identity()
    
    # 1. LOAD KEDUA CITRA
    # Pastikan file 'lurus.jpg' dan 'miring.jpg' ada di folder yang sama
    img_lurus = cv2.imread('Pertemuan 3/lurus.jpg')
    img_miring = cv2.imread('Pertemuan 3/miring.jpg')

    if img_lurus is None or img_miring is None:
        print("Error: Pastikan file 'lurus.jpg' dan 'miring.jpg' tersedia!")
        return

    # Resize keduanya ke ukuran yang sama agar bisa dibandingkan (misal 600x800)
    width, height = 600, 800
    img_ref = cv2.resize(img_lurus, (width, height))
    img_src = cv2.resize(img_miring, (width, height))

    # --- TRANSFORMASI PERSPEKTIF (4 TITIK) ---
    # Klik/Cari koordinat 4 pojok dokumen pada foto 'miring.jpg'
    # Urutan: Kiri Atas, Kanan Atas, Kiri Bawah, Kanan Bawah
    pts_miring = np.float32([[100, 150], [520, 120], [30, 720], [580, 750]]) # Ganti sesuai foto
    pts_lurus = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix_p = cv2.getPerspectiveTransform(pts_miring, pts_lurus)

    # --- PENERAPAN 3 METODE INTERPOLASI ---
    methods = [
        ("Nearest", cv2.INTER_NEAREST),
        ("Bilinear", cv2.INTER_LINEAR),
        ("Bicubic", cv2.INTER_CUBIC)
    ]

    results_img = []
    print(f"{'Metode':<12} | {'MSE':<10} | {'PSNR':<10} | {'Waktu (ms)':<10}")
    print("-" * 50)

    for name, flag in methods:
        start = time.time()
        # Eksekusi Transformasi
        warped = cv2.warpPerspective(img_src, matrix_p, (width, height), flags=flag)
        end = time.time()
        
        duration = (end - start) * 1000
        mse, psnr = evaluate_quality(img_ref, warped)
        
        results_img.append(warped)
        print(f"{name:<12} | {mse:<10.2f} | {psnr:<10.2f} | {duration:<10.4f}")

    # --- VISUALISASI ---
    titles = ['Referensi (Lurus)', 'Input (Miring)', 'Hasil (Bicubic)']
    display = [img_ref, img_src, results_img[2]]

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(cv2.cvtColor(display[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()