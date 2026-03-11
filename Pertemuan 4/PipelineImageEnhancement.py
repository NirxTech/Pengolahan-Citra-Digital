import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy

# ==========================================
# IDENTITAS MAHASISWA
# ==========================================
NAMA = "Muhammad Zahran"
NIM = "24343077"

print(f"Program Enhancement Citra")
print(f"Nama : {NAMA}")
print(f"NIM  : {NIM}")
print("-" * 30)

def calculate_metrics(img):
    """Menghitung metrik kontras dan entropi."""
    contrast = img.std() # Standard Deviation sebagai proksi kontras
    entropy = shannon_entropy(img)
    return contrast, entropy

def point_processing(img):
    """Implementasi Negative, Log, dan Gamma Transformation."""
    # 1. Negative Transformation
    img_neg = 255 - img
    
    # 2. Log Transformation (s = c * log(1 + r))
    c = 255 / np.log(1 + np.max(img))
    img_log = c * (np.log(img + 1))
    img_log = np.array(img_log, dtype=np.uint8)
    
    # 3. Power-law (Gamma) Transformation (s = c * r^gamma)
    def apply_gamma(image, gamma):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    img_gamma_low = apply_gamma(img, 0.5)  # Terangkan
    img_gamma_mid = apply_gamma(img, 1.5)  # Kontras menengah
    img_gamma_high = apply_gamma(img, 2.5) # Gelapkan
    
    return img_neg, img_log, img_gamma_low, img_gamma_mid, img_gamma_high

def histogram_processing(img):
    """Implementasi Contrast Stretching, Global HE, dan CLAHE."""
    # 1. Contrast Stretching (Manual: Clip 5-95 percentile)
    p5, p95 = np.percentile(img, (5, 95))
    img_stretch_man = np.clip(img, p5, p95)
    img_stretch_man = ((img_stretch_man - p5) / (p95 - p5) * 255).astype(np.uint8)
    
    # 2. Contrast Stretching (Automatic: Min-Max)
    img_stretch_auto = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # 3. Global Histogram Equalization
    img_he = cv2.equalizeHist(img)
    
    # 4. Adaptive HE (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    
    return img_stretch_man, img_stretch_auto, img_he, img_clahe

def display_results(original, results, titles, category_name):
    """Menampilkan citra dan histogram untuk evaluasi visual."""
    n = len(results) + 1
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Analisis Citra: {category_name}", fontsize=16)
    
    images = [original] + results
    all_titles = ["Original"] + titles
    
    for i in range(n):
        # Plot Citra
        plt.subplot(2, n, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(all_titles[i])
        plt.axis('off')
        
        # Plot Histogram
        plt.subplot(2, n, i + 1 + n)
        plt.hist(images[i].ravel(), 256, [0, 256], color='black')
        contrast, entropy = calculate_metrics(images[i])
        plt.xlabel(f"C:{contrast:.1f} | E:{entropy:.2f}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
image_paths = {
    "Underexposed": "Praktikum 4/Gelap.jpg",
    "Overexposed": "Praktikum 4/Terang.jpg",
    "Uneven Illumination": "Praktikum 4/Bayangan.jpg"
}

for category, path in image_paths.items():
    # Load citra dalam grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Peringatan: File {path} tidak ditemukan. Melewati kategori {category}.")
        continue

    # Jalankan Enhancement
    neg, log, g_low, g_mid, g_high = point_processing(img)
    s_man, s_auto, he, clahe = histogram_processing(img)
    
    # List hasil dan judul
    results = [log, g_low, g_high, s_auto, he, clahe]
    titles = ["Log", "Gamma 0.5", "Gamma 2.5", "Stretch Auto", "Global HE", "CLAHE"]
    
    display_results(img, results, titles, category)