import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def manual_histogram_equalization(image):
    """
    Manual implementation of histogram equalization
    tanpa menggunakan OpenCV sedikitpun.
    
    Parameters:
    image: Input grayscale image (0-255)
    
    Returns:
    Equalized image and transformation function
    """
    # 1. Hitung histogram menggunakan NumPy
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # 2. Hitung cumulative histogram (CDF)
    cdf = hist.cumsum()
    
    # 3. Hitung transformation function (Normalisasi)
    # Masking untuk mengabaikan nilai 0 saat mencari nilai minimum
    cdf_masked = np.ma.masked_equal(cdf, 0)
    
    # Rumus normalisasi: T(v) = round((cdf(v) - cdf_min) / (M*N - cdf_min) * 255)
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    
    # Isi kembali nilai yang di-mask dengan 0 dan ubah ke integer 8-bit (LUT)
    transformation_function = np.ma.filled(cdf_normalized, 0).astype(np.uint8)
    
    # 4. Apply transformation (Pemetaan pixel lama ke baru)
    equalized_image = transformation_function[image]
    
    # 5. Return equalized image and transformation function
    return equalized_image, transformation_function


# ==========================================
# BLOK PENGUJIAN (TANPA OPENCV)
# ==========================================
if __name__ == "__main__":
    
    # --- OPSI 1: Menggunakan Citra Sintetis (Dummy) ---
    # Membangun citra gelap berukuran 256x256 pixel
    test_image = np.random.normal(50, 15, (256, 256))
    test_image = np.clip(test_image, 0, 255).astype(np.uint8)
    
    # --- OPSI 2: Jika ingin pakai foto asli, buka komentar di bawah ini ---
    # import urllib.request
    # from PIL import Image
    # img_pil = Image.open("nama_foto_kamu.jpg").convert('L') # 'L' untuk Grayscale
    # test_image = np.array(img_pil)
    
    # Eksekusi fungsi manual
    eq_image_manual, transform_curve = manual_histogram_equalization(test_image)
    
    # Visualisasi menggunakan murni Matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Baris 1: Citra Asli vs Hasil
    axes[0, 0].imshow(test_image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Citra Asli (Gelap)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(eq_image_manual, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title("Hasil Ekualisasi Manual")
    axes[0, 1].axis('off')
    
    # Baris 2: Histogram dan Kurva LUT
    axes[1, 0].hist(test_image.ravel(), 256, [0, 256], color='gray')
    axes[1, 0].set_title("Histogram Asli")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot histogram hasil
    axes[1, 1].hist(eq_image_manual.ravel(), 256, [0, 256], color='blue', alpha=0.6, label='Histogram Baru')
    
    # Tambahkan Kurva Transformasi di grafik yang sama dengan sumbu Y berbeda
    ax_curve = axes[1, 1].twinx()
    ax_curve.plot(transform_curve, color='red', linewidth=2, label='Kurva LUT')
    ax_curve.set_ylabel('Nilai Transformasi (0-255)', color='red')
    
    axes[1, 1].set_title("Histogram Baru & Kurva Transformasi")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()