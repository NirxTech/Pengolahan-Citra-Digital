import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from sklearn.cluster import KMeans

def calculate_metrics(original_data, processed_data, start_time, end_time):
    mem_before = sys.getsizeof(original_data.tobytes())
    mem_after = sys.getsizeof(processed_data.tobytes())
    compression_ratio = mem_before / mem_after if mem_after > 0 else 0
    calc_time = end_time - start_time
    return mem_before, mem_after, compression_ratio, calc_time

def uniform_quantization(image, levels=16):
    start = time.time()
    h, w, c = image.shape
    factor = 256 // levels
    quantized = (image // factor) * factor
    end = time.time()
    return quantized, start, end

def nonuniform_quantization(image, n_clusters=16):
    start = time.time()
    h, w, c = image.shape
    image_2d = image.reshape(h * w, c)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(image_2d)
    quantized_2d = kmeans.cluster_centers_[kmeans.labels_]
    quantized_image = quantized_2d.reshape(h, w, c).astype('uint8')
    end = time.time()
    return quantized_image, start, end

def process_and_analyze(image_paths):
    for path in image_paths:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"Gambar {path} tidak ditemukan. Silakan ganti dengan path gambar yang sesuai.")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Konversi Ruang Warna
        t0 = time.time()
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        t1 = time.time()
        
        t2 = time.time()
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        t3 = time.time()
        
        t4 = time.time()
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        t5 = time.time()

        print(f"\n--- Analisis untuk {path} ---")
        print(f"Waktu Konversi RGB ke Grayscale: {t1-t0:.5f} detik")
        print(f"Waktu Konversi RGB ke HSV: {t3-t2:.5f} detik")
        print(f"Waktu Konversi RGB ke LAB: {t5-t4:.5f} detik")

        # 2. Implementasi Kuantisasi pada RGB (sebagai contoh perbandingan metrik)
        # Kuantisasi Uniform
        quant_uni_rgb, s_uni, e_uni = uniform_quantization(img_rgb, 16)
        mem_b_uni, mem_a_uni, cr_uni, time_uni = calculate_metrics(img_rgb, quant_uni_rgb, s_uni, e_uni)
        
        print("\nParameter Teknis Kuantisasi Uniform (RGB 256 -> 16 level):")
        print(f"Memori Sebelum: {mem_b_uni} bytes | Sesudah: {mem_a_uni} bytes")
        print(f"Rasio Kompresi: {cr_uni:.2f}x | Waktu: {time_uni:.5f} detik")

        # Kuantisasi Non-Uniform (K-Means)
        # Catatan: K-Means memakan waktu lebih lama, resize dilakukan agar lebih cepat jika diperlukan
        scale_percent = 50 
        w = int(img_rgb.shape[1] * scale_percent / 100)
        h = int(img_rgb.shape[0] * scale_percent / 100)
        img_rgb_small = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
        
        quant_nonuni_rgb, s_non, e_non = nonuniform_quantization(img_rgb_small, 16)
        mem_b_non, mem_a_non, cr_non, time_non = calculate_metrics(img_rgb_small, quant_nonuni_rgb, s_non, e_non)
        
        print("\nParameter Teknis Kuantisasi Non-Uniform (K-Means 16 clusters, skala 50%):")
        print(f"Memori Sebelum: {mem_b_non} bytes | Sesudah: {mem_a_non} bytes")
        print(f"Rasio Kompresi: {cr_non:.2f}x | Waktu: {time_non:.5f} detik")

        # 3. Visualisasi (Kualitas Subjektif & Histogram)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Visualisasi Kuantisasi & Histogram - {path} (Zahran - 24343077)')
        
        # Original
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title("RGB Original")
        axes[1, 0].hist(img_rgb.ravel(), 256, [0, 256], color='gray')
        axes[1, 0].set_title("Histogram Original")
        
        # Uniform
        axes[0, 1].imshow(quant_uni_rgb)
        axes[0, 1].set_title("Uniform Quantization (16)")
        axes[1, 1].hist(quant_uni_rgb.ravel(), 256, [0, 256], color='gray')
        axes[1, 1].set_title("Histogram Uniform")
        
        # Non-Uniform
        axes[0, 2].imshow(quant_nonuni_rgb)
        axes[0, 2].set_title("Non-Uniform (K-Means)")
        axes[1, 2].hist(quant_nonuni_rgb.ravel(), 256, [0, 256], color='gray')
        axes[1, 2].set_title("Histogram Non-Uniform")
        
        plt.tight_layout()
        plt.show()

# Panggilan eksekusi (Pastikan untuk mengganti nama file dengan gambar asli di direktori kamu)
image_paths = ['terang.jpg', 'normal.jpg', 'redup.jpg']
process_and_analyze(image_paths)