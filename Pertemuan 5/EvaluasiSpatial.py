import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise
from skimage import img_as_ubyte

def compute_metrics(original, restored, process_time):
    mse = np.mean((original - restored) ** 2)
    p_val = psnr(original, restored, data_range=255)
    s_val = ssim(original, restored, data_range=255, channel_axis=None)
    return {"MSE": round(mse, 2), "PSNR": round(p_val, 2), "SSIM": round(s_val, 3), "Time": round(process_time, 4)}

# 1. Load Citra Asli (Ganti ke Path Lokal)
# Parameter 0 digunakan agar gambar dibaca langsung sebagai Grayscale
path_file = 'TehPucuk.jpg'
original_img = cv2.imread(path_file, 0) 

# Cek apakah gambar berhasil dimuat
if original_img is None:
    print(f"Error: File '{path_file}' tidak ditemukan! Pastikan file ada di folder yang sama.")
else:
    original_img = cv2.resize(original_img, (512, 512))

    # 2. Buat Variasi Noise
    noise_gaussian = img_as_ubyte(random_noise(original_img, mode='gaussian', var=0.01))
    noise_sp = img_as_ubyte(random_noise(original_img, mode='s&p', amount=0.05))
    noise_speckle = img_as_ubyte(random_noise(original_img, mode='speckle', var=0.05))

    noises = {
        "Gaussian": noise_gaussian,
        "Salt-and-Pepper": noise_sp,
        "Speckle": noise_speckle
    }

    # 3. Implementasi Filter
    results = []

    def apply_filters(noisy_img, noise_name):
        # --- Linear Filters ---
        # Mean Filter
        for k in [3, 5]:
            start = time.time()
            res = cv2.blur(noisy_img, (k, k))
            results.append((noise_name, f"Mean {k}x{k}", res, compute_metrics(original_img, res, time.time()-start)))
        
        # Gaussian Filter
        for s in [1.0, 2.0]:
            start = time.time()
            res = cv2.GaussianBlur(noisy_img, (5, 5), sigmaX=s)
            results.append((noise_name, f"Gaussian (s={s})", res, compute_metrics(original_img, res, time.time()-start)))

        # --- Non-Linear Filters ---
        # Median Filter
        for k in [3, 5]:
            start = time.time()
            res = cv2.medianBlur(noisy_img, k)
            results.append((noise_name, f"Median {k}x{k}", res, compute_metrics(original_img, res, time.time()-start)))
        
        # Min/Max Filter
        start = time.time()
        kernel = np.ones((3,3), np.uint8)
        res = cv2.erode(noisy_img, kernel) 
        results.append((noise_name, "Min Filter 3x3", res, compute_metrics(original_img, res, time.time()-start)))

    # Jalankan evaluasi
    for name, img in noises.items():
        apply_filters(img, name)

    # 4. Visualisasi & Tabel Performa
    print(f"{'Noise Type':<18} | {'Filter':<18} | {'PSNR':<6} | {'SSIM':<6} | {'Time (s)':<8}")
    print("-" * 75)

    for r in results:
        print(f"{r[0]:<18} | {r[1]:<18} | {r[3]['PSNR']:<6} | {r[3]['SSIM']:<6} | {r[3]['Time']:<8}")

    # Menampilkan perbandingan visual
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1); plt.title("Original"); plt.imshow(original_img, cmap='gray')
    plt.subplot(2, 3, 2); plt.title("S&P Noise"); plt.imshow(noise_sp, cmap='gray')

    res_median = next(x[2] for x in results if x[0] == "Salt-and-Pepper" and "Median 3x3" in x[1])
    res_mean = next(x[2] for x in results if x[0] == "Salt-and-Pepper" and "Mean 3x3" in x[1])

    plt.subplot(2, 3, 4); plt.title("Restored (Mean 3x3)"); plt.imshow(res_mean, cmap='gray')
    plt.subplot(2, 3, 5); plt.title("Restored (Median 3x3)"); plt.imshow(res_median, cmap='gray')
    plt.tight_layout()
    plt.show()