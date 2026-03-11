import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_enhancement_metrics(original, enhanced):
    """
    Menghitung metrik kualitas untuk evaluasi peningkatan citra medis.
    """
    metrics = {}
    
    # 1. Contrast Improvement Index (CII)
    # Mengukur seberapa besar peningkatan kontras lokal/global
    orig_std = np.std(original)
    enh_std = np.std(enhanced)
    metrics['CII'] = enh_std / orig_std if orig_std > 0 else 0
    
    # 2. Entropy Improvement (Informasi Visual)
    # Mengukur jumlah informasi (detail) yang berhasil dimunculkan
    orig_hist, _ = np.histogram(original.flatten(), 256, [0, 256])
    enh_hist, _ = np.histogram(enhanced.flatten(), 256, [0, 256])
    
    # Ditambah 1e-10 untuk menghindari error log(0)
    orig_entropy = stats.entropy(orig_hist + 1e-10) 
    enh_entropy = stats.entropy(enh_hist + 1e-10) 
    
    metrics['Original_Entropy'] = orig_entropy
    metrics['Enhanced_Entropy'] = enh_entropy
    metrics['Entropy_Gain'] = enh_entropy - orig_entropy
    
    return metrics

def medical_image_enhancement(medical_image, modality='X-ray'):
    """
    Adaptive enhancement pipeline khusus untuk citra medis
    """
    # Pastikan citra format grayscale 8-bit
    if len(medical_image.shape) > 2:
        img = cv2.cvtColor(medical_image, cv2.COLOR_BGR2GRAY)
    else:
        img = medical_image.copy()
        
    enhanced = np.zeros_like(img)
    pipeline_info = ""

    # --- IMPLEMENTASI BERDASARKAN MODALITAS ---
    
    if modality == 'X-ray':
        # Constraint: Tingkatkan kontras jaringan lunak tanpa over-ekspos tulang
        # Solusi: CLAHE dengan clip limit standar
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        pipeline_info = "CLAHE (clipLimit=2.0, grid=8x8)"
        
    elif modality == 'MRI':
        # Constraint: Angkat detail di area gelap tanpa merusak area terang
        # Solusi: Gamma Correction (gamma < 1) + Mild CLAHE
        img_float = img.astype(np.float32) / 255.0
        gamma_corrected = (np.power(img_float, 0.7) * 255).astype(np.uint8) #
        
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gamma_corrected)
        pipeline_info = "Gamma Correction (y=0.7) -> Mild CLAHE (clip=1.5)"
        
    elif modality == 'CT':
        # Constraint: Fokus pada jendela spesifik, abaikan outlier (artefak logam/udara)
        # Solusi: Robust Contrast Stretching berbasis persentil
        p2, p98 = np.percentile(img, (2, 98))
        img_clipped = np.clip(img, p2, p98)
        
        # Normalisasi ke 0-255
        enhanced = ((img_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
        pipeline_info = "Robust Contrast Stretching (2nd-98th Percentile)"
        
    elif modality == 'Ultrasound':
        # Constraint: Speckle noise sangat tinggi, dilarang amplifikasi ekstrem
        # Solusi: Sangat konservatif CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        pipeline_info = "Conservative CLAHE (clipLimit=1.0) to prevent speckle amp."
        
    else:
        # Fallback
        enhanced = cv2.equalizeHist(img) #
        pipeline_info = "Standard Global Histogram Equalization"

    # --- GENERATE REPORT ---
    metrics = calculate_enhancement_metrics(img, enhanced)
    
    report = {
        "Modality": modality,
        "Applied_Pipeline": pipeline_info,
        "Metrics": metrics
    }
    
    return enhanced, report


# ==========================================
# BLOK PENGUJIAN (TESTING)
# ==========================================
if __name__ == "__main__":
    # 1. Buat citra dummy yang menyerupai X-Ray (kontras rendah di tengah)
    np.random.seed(42)
    dummy_xray = np.random.normal(100, 20, (256, 256))
    dummy_xray[100:150, 100:150] = np.random.normal(110, 5, (50, 50)) # Simulasi massa/tumor samar
    dummy_xray = np.clip(dummy_xray, 0, 255).astype(np.uint8)
    
    # 2. Uji fungsi untuk berbagai modalitas
    modalities = ['X-ray', 'MRI', 'CT', 'Ultrasound']
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    for idx, mod in enumerate(modalities):
        # Jalankan Adaptive Enhancement
        enh_img, report = medical_image_enhancement(dummy_xray, modality=mod)
        
        # Plot Citra Hasil
        axes[0, idx].imshow(enh_img, cmap='gray')
        axes[0, idx].set_title(f"Modalitas: {mod}")
        axes[0, idx].axis('off')
        
        # Cetak Laporan ke Terminal
        print(f"\n{'='*40}")
        print(f"REPORT: {mod.upper()}")
        print(f"Pipeline : {report['Applied_Pipeline']}")
        print(f"CII      : {report['Metrics']['CII']:.3f} (Contrast Improv. Index)")
        print(f"Ent Gain : {report['Metrics']['Entropy_Gain']:.3f} bits/pixel")
        
        # Plot Histogram
        axes[1, idx].hist(enh_img.ravel(), 256, [0, 256], color='blue', alpha=0.7)
        axes[1, idx].set_title(f"Histogram ({mod})")
        axes[1, idx].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()