import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift

def praktikum_7_1_fixed():
    print("PRAKTIKUM 7.1: TRANSFORMASI FOURIER DAN ANALISIS SPEKTRUM")
    print("=" * 60)
    
    # 1. Fungsi Pembuat Gambar (Tetap sama, hanya pembersihan kecil)
    def create_frequency_test_images():
        images = {}
        # Low Frequency
        img_low = np.zeros((256, 256), dtype=np.float32)
        cv2.rectangle(img_low, (50, 50), (200, 200), 1.0, -1)
        img_low = cv2.GaussianBlur(img_low, (31, 31), 10)
        images['Low Frequency'] = (img_low * 255).astype(np.uint8)
        
        # High Frequency
        img_high = np.zeros((256, 256), dtype=np.uint8)
        for i in range(0, 256, 16):
            for j in range(0, 256, 16):
                if (i//16 + j//16) % 2 == 0:
                    img_high[i:i+16, j:j+16] = 255
        images['High Frequency'] = img_high
        
        # Mixed Frequencies
        img_mixed = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_mixed, (30, 30), (150, 150), 200, -1)
        for i in range(0, 256, 8):
            cv2.line(img_mixed, (i, 0), (i, 255), 150, 1)
        for i in range(20, 236, 20):
            for j in range(20, 236, 20):
                cv2.circle(img_mixed, (i, j), 2, 255, -1)
        images['Mixed Frequencies'] = img_mixed
        
        # Periodic Pattern
        x, y = np.meshgrid(np.arange(256), np.arange(256))
        img_periodic = 127 + 127 * np.sin(2*np.pi*x/32) * np.sin(2*np.pi*y/64)
        images['Periodic Pattern'] = img_periodic.astype(np.uint8)
        
        return images

    def analyze_fourier_spectrum(image):
        img_float = image.astype(np.float32) / 255.0
        f = fft2(img_float)
        fshift = fftshift(f)
        mag = np.abs(fshift)
        return {
            'magnitude': mag,
            'log_magnitude': np.log(1 + mag),
            'phase': np.angle(fshift),
            'power': mag ** 2,
            'log_power': np.log(1 + mag**2),
            'fshift': fshift
        }

    def reconstruct_from_components(magnitude, phase):
        complex_spectrum = magnitude * np.exp(1j * phase)
        return np.clip(np.abs(ifft2(ifftshift(complex_spectrum))) * 255, 0, 255).astype(np.uint8)

    test_images = create_frequency_test_images()

    # --- VISUALISASI 1: SPEKTRUM (Gunakan plt.figure untuk memisahkan jendela) ---
    plt.figure(figsize=(16, 12))
    for idx, (title, image) in enumerate(test_images.items()):
        analysis = analyze_fourier_spectrum(image)
        plt.subplot(4, 4, idx*4 + 1); plt.imshow(image, cmap='gray'); plt.title(f'{title}\nOriginal'); plt.axis('off')
        plt.subplot(4, 4, idx*4 + 2); plt.imshow(analysis['log_magnitude'], cmap='magma'); plt.title('Log Magnitude'); plt.axis('off')
        plt.subplot(4, 4, idx*4 + 3); plt.imshow(analysis['phase'], cmap='hsv'); plt.title('Phase'); plt.axis('off')
        plt.subplot(4, 4, idx*4 + 4); plt.imshow(analysis['log_power'], cmap='viridis'); plt.title('Log Power'); plt.axis('off')
    plt.tight_layout()
    # Jangan panggil plt.show() di sini agar tidak memblock loop berikutnya

    # --- VISUALISASI 2: SWAPPING ---
    img1, img2 = test_images['Low Frequency'], test_images['High Frequency']
    a1, a2 = analyze_fourier_spectrum(img1), analyze_fourier_spectrum(img2)
    
    r1 = reconstruct_from_components(a1['magnitude'], a2['phase'])
    r2 = reconstruct_from_components(a2['magnitude'], a1['phase'])

    plt.figure(figsize=(12, 8))
    titles = ['Original Low Freq', 'Original High Freq', 'Mag(Low) + Phase(High)', 'Mag(High) + Phase(Low)']
    imgs = [img1, img2, r1, r2]
    for i in range(4):
        plt.subplot(2, 2, i+1); plt.imshow(imgs[i], cmap='gray'); plt.title(titles[i]); plt.axis('off')
    plt.suptitle("Importance of Phase Demonstration")
    plt.tight_layout()

    # --- VISUALISASI 3: ANALISIS DISTRIBUSI (Radial) ---
    print("\nPROSES ANALISIS DISTRIBUSI FREKUENSI...")
    plt.figure(figsize=(12, 10))
    for idx, (title, image) in enumerate(test_images.items()):
        analysis = analyze_fourier_spectrum(image)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        r = np.sqrt(x*x + y*y)
        r_bins = np.arange(0, crow, 1)
        
        radial_profile = [np.mean(analysis['magnitude'][(r >= i) & (r < i+1)]) for i in r_bins[:-1]]
        
        plt.subplot(2, 2, idx+1)
        plt.plot(r_bins[:-1], radial_profile, color='blue', lw=2)
        plt.fill_between(r_bins[:-1], 0, radial_profile, alpha=0.2)
        plt.title(f'Radial Distribution: {title}')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Tampilkan semua gambar sekaligus ---
    plt.show()

    # --- STATISTIK (DIPERBAIKI: Indeks Dinamis) ---
    print("\nFREQUENCY DOMAIN STATISTICS")
    print("-" * 75)
    print(f"{'Image Type':<20} | {'DC Component':<12} | {'Avg Mag (no DC)':<15} | {'Energy'}")
    print("-" * 75)
    for title, image in test_images.items():
        analysis = analyze_fourier_spectrum(image)
        h, w = analysis['magnitude'].shape
        cy, cx = h // 2, w // 2 # Titik pusat dinamis
        
        dc = analysis['magnitude'][cy, cx]
        mag_copy = analysis['magnitude'].copy()
        mag_copy[cy, cx] = 0
        avg_mag = np.mean(mag_copy)
        energy = np.sum(analysis['power'])
        
        print(f"{title:<20} | {dc:<12.2f} | {avg_mag:<15.6f} | {energy:<15.2f}")

# Jalankan fungsi
praktikum_7_1_fixed()