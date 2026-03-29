import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data

# ==========================================
# 1. PERSIAPAN CITRA (LOAD 2 FOTO BERBEDA)
# ==========================================
def load_images(path1, path2):
    # Load Citra 1: Natural
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        print(f"File {path1} tidak ada, menggunakan sample 'Camera'.")
        img1 = pywt.data.camera().astype(np.float32)
    
    # Load Citra 2: Noise Periodik
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img2 is None:
        print(f"File {path2} tidak ada, membuat noise periodik sintetis.")
        rows, cols = img1.shape
        x = np.linspace(0, 1, cols); y = np.linspace(0, 1, rows)
        X, Y = np.meshgrid(x, y)
        noise = 40 * np.sin(2 * np.pi * 20 * X + 2 * np.pi * 20 * Y)
        img2 = np.clip(img1 + noise, 0, 255).astype(np.uint8)
        
    return img1, img2

# ==========================================
# 2. IMPLEMENTASI TRANSFORMASI FOURIER
# ==========================================
def process_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    mag_spectrum = 20 * np.log(np.abs(fshift) + 1)
    phase_spectrum = np.angle(fshift)
    
    # Rekonstruksi
    rec_mag = np.abs(np.fft.ifft2(np.abs(f)))  # Hanya Magnitudo
    rec_phase = np.abs(np.fft.ifft2(np.exp(1j * np.angle(f)))) # Hanya Fase
    
    return fshift, mag_spectrum, phase_spectrum, rec_mag, rec_phase

# ==========================================
# 3. FILTERING DOMAIN FREKUENSI
# ==========================================
def apply_filter(fshift, type='gaussian_lp', cutoff=30):
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    dist = np.sqrt(x*x + y*y)
    
    if type == 'ideal_lp':
        mask = np.uint8(dist <= cutoff)
    elif type == 'ideal_hp':
        mask = np.uint8(dist > cutoff)
    elif type == 'gaussian_lp':
        mask = np.exp(-(dist**2) / (2 * (cutoff**2)))
    elif type == 'gaussian_hp':
        mask = 1 - np.exp(-(dist**2) / (2 * (cutoff**2)))
    elif type == 'notch':
        mask = np.ones((rows, cols), np.float32)
        # Menghapus titik noise (sesuaikan koordinat jika perlu)
        cv2.circle(mask, (ccol+24, crow+24), 7, 0, -1)
        cv2.circle(mask, (ccol-24, crow-24), 7, 0, -1)
    
    filtered_f = fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_f)))
    return img_back, mask

# ==========================================
# 4. TRANSFORMASI WAVELET (2-LEVEL)
# ==========================================
def process_wavelet(img):
    # Dekomposisi 2-level dengan Daubechies 4
    coeffs = pywt.wavedec2(img, 'db4', level=2)
    LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs
    
    # Rekonstruksi hanya dari LL2 (Aproksimasi Terpilih)
    coeffs_filt = [LL2, (None, None, None), (None, None, None)]
    img_rec = pywt.waverec2(coeffs, 'db4') # Rekonstruksi penuh
    
    return [LL2, LH2, HL2, HH2], img_rec

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Load Data
img_nat, img_noise = load_images('Pertemuan 7/Foto.jpg', 'Pertemuan 7/TehPucuk.jpg')

# 2. Jalankan FFT & Filtering
fshift_nat, mag_nat, phase_nat, r_mag, r_phase = process_fft(img_nat)
img_gauss_lp, _ = apply_filter(fshift_nat, 'gaussian_lp', 40)
img_ideal_lp, _ = apply_filter(fshift_nat, 'ideal_lp', 40)

fshift_noise = np.fft.fftshift(np.fft.fft2(img_noise))
img_cleaned, _ = apply_filter(fshift_noise, 'notch')

# 3. Jalankan Wavelet
wt_coeffs, wt_full_rec = process_wavelet(img_nat)

# ==========================================
# VISUALISASI HASIL
# ==========================================
plt.figure(figsize=(16, 10))

# Baris 1: Analisis FFT Citra Natural
plt.subplot(3, 4, 1), plt.imshow(img_nat, cmap='gray'), plt.title('Original Natural')
plt.subplot(3, 4, 2), plt.imshow(mag_nat, cmap='gray'), plt.title('Magnitude Spectrum')
plt.subplot(3, 4, 3), plt.imshow(r_phase, cmap='gray'), plt.title('Rec. from Phase')
plt.subplot(3, 4, 4), plt.imshow(r_mag, cmap='gray'), plt.title('Rec. from Magnitude')

# Baris 2: Perbandingan Filter (Ringing Effect)
plt.subplot(3, 4, 5), plt.imshow(img_ideal_lp, cmap='gray'), plt.title('Ideal LP (Ringing)')
plt.subplot(3, 4, 6), plt.imshow(img_gauss_lp, cmap='gray'), plt.title('Gaussian LP (Smooth)')
plt.subplot(3, 4, 7), plt.imshow(img_noise, cmap='gray'), plt.title('Periodic Noise Image')
plt.subplot(3, 4, 8), plt.imshow(img_cleaned, cmap='gray'), plt.title('After Notch Filter')

# Baris 3: Analisis Wavelet
plt.subplot(3, 4, 9), plt.imshow(wt_coeffs[0], cmap='gray'), plt.title('Wavelet LL2')
plt.subplot(3, 4, 10), plt.imshow(wt_coeffs[1], cmap='gray'), plt.title('Wavelet LH2 (Detail)')
plt.subplot(3, 4, 11), plt.imshow(wt_coeffs[2], cmap='gray'), plt.title('Wavelet HL2 (Detail)')
plt.subplot(3, 4, 12), plt.imshow(wt_full_rec, cmap='gray'), plt.title('Wavelet Reconstruction')

plt.tight_layout()
plt.show()