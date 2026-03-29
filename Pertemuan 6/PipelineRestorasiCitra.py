import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from skimage import img_as_float, util
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.restoration import richardson_lucy
from scipy.fftpack import fft2, ifft2

# ==========================================
# 1. INPUT GAMBAR SENDIRI DISINI
# ==========================================
NAMA_FILE = 'Pertemuan 6/Foto.jpg' 

def load_user_image(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Gagal! File '{filename}' tidak ditemukan di folder.")
    
    # Baca gambar
    img = cv2.imread(filename)
    
    # --- PERBAIKAN: Resize Gambar agar tidak macet ---
    # Kita batasi lebar maksimal 800px agar proses cepat
    max_dim = 800
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # Ubah ke Grayscale jika gambar berwarna
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalisasi ke float [0, 1]
    return img_as_float(img)

# ==========================================
# 2. FUNGSI RESTORASI & PSF
# ==========================================
def get_motion_psf(size_h, size_w, length, angle):
    psf = np.zeros((size_h, size_w))
    center_h, center_w = size_h // 2, size_w // 2
    angle_rad = np.deg2rad(angle)
    
    dx = int(length / 2 * np.cos(angle_rad))
    dy = int(length / 2 * np.sin(angle_rad))
    
    # Gambar garis blur pada PSF
    cv2.line(psf, (center_w - dx, center_h - dy), (center_w + dx, center_h + dy), 1, 1)
    return psf / psf.sum()

def inverse_filter(image_fft, psf_fft, threshold=0.1):
    # Stabilkan pembagian dengan threshold
    res = image_fft / (psf_fft + 1e-12)
    res[np.abs(psf_fft) < threshold] = 0
    return np.abs(ifft2(res))

def wiener_filter(image_fft, psf_fft, K=0.01):
    psf_fft_conj = np.conj(psf_fft)
    res = (psf_fft_conj / (np.abs(psf_fft)**2 + K)) * image_fft
    return np.abs(ifft2(res))

def evaluate(original, restored, compute_time):
    # Kliping agar nilai piksel tetap di 0-1
    original = np.clip(original, 0, 1)
    restored = np.clip(restored, 0, 1)
    
    p = psnr(original, restored, data_range=1)
    s = ssim(original, restored, data_range=1)
    m = mse(original, restored)
    return {"PSNR": round(p, 2), "SSIM": round(s, 4), "MSE": round(m, 5), "Time": round(compute_time, 4)}

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
try:
    original_img = load_user_image(NAMA_FILE)
    h, w = original_img.shape
    print(f"Berhasil memuat dan me-resize gambar: {NAMA_FILE} menjadi ({w}x{h})")

    # Parameter Degradasi
    L = 15     # Panjang blur
    theta = 30 # Arah 30 derajat

    # Bangun PSF (Point Spread Function)
    psf = get_motion_psf(h, w, L, theta)
    psf_fft = fft2(psf)

    # Pembuatan Variasi Degradasi
    # a. Motion Blur
    blur_fft = fft2(original_img) * psf_fft
    blur_only = np.abs(ifft2(blur_fft))

    # b. Gaussian Noise (sigma=20/255) + Motion Blur
    gaussian_noise = util.random_noise(blur_only, mode='gaussian', var=(20/255)**2)

    # c. Salt and Pepper (5%) + Motion Blur
    sp_noise = util.random_noise(blur_only, mode='s&p', amount=0.05)

    degradations = [
        ("Motion Blur", blur_only, 0.0001), # K kecil karena noise rendah
        ("Motion + Gaussian", gaussian_noise, 0.01),
        ("Motion + S&P", sp_noise, 0.05)
    ]

    results_table = []
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    for i, (name, degraded, K_val) in enumerate(degradations):
        deg_fft = fft2(degraded)
        
        # 1. Inverse Filtering
        start = time.time()
        res_inv = inverse_filter(deg_fft, psf_fft, threshold=0.1)
        t_inv = time.time() - start
        
        # 2. Wiener Filtering
        start = time.time()
        res_wie = wiener_filter(deg_fft, psf_fft, K=K_val)
        t_wie = time.time() - start
        
        # 3. Lucy-Richardson (Dibatasi 15 iterasi agar cepat)
        start = time.time()
        res_lr = richardson_lucy(degraded, psf, num_iter=15)
        t_lr = time.time() - start
        
        # Simpan evaluasi ke tabel
        results_table.append((name, "Inverse", evaluate(original_img, res_inv, t_inv)))
        results_table.append((name, "Wiener", evaluate(original_img, res_wie, t_wie)))
        results_table.append((name, "Lucy-R", evaluate(original_img, res_lr, t_lr)))
        
        # Tampilkan Hasil Visual
        axes[i, 0].imshow(degraded, cmap='gray'); axes[i, 0].set_title(f"Degraded: {name}")
        axes[i, 1].imshow(res_inv, cmap='gray'); axes[i, 1].set_title("Inverse Filter")
        axes[i, 2].imshow(res_wie, cmap='gray'); axes[i, 2].set_title(f"Wiener (K={K_val})")
        axes[i, 3].imshow(res_lr, cmap='gray'); axes[i, 3].set_title("Lucy-Richardson")

    plt.tight_layout()
    plt.show()

    # Cetak Tabel Analisis
    print(f"\n{'Degradasi':<20} | {'Metode':<10} | {'PSNR':<6} | {'SSIM':<6} | {'Waktu (s)':<8}")
    print("-" * 65)
    for deg, met, res in results_table:
        print(f"{deg:<20} | {met:<10} | {res['PSNR']:<6} | {res['SSIM']:<6} | {res['Time']:<8}")

except Exception as e:
    print(f"Terjadi Error: {e}")