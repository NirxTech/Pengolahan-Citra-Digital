import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# PIPELINE SEGMENTASI CITRA REAL-WORLD - MUHAMMAD ZAHRAN (24343077)
# =====================================================================

DATA_DIR = './'

# Nama file gambar input yang Anda berikan
image_assignments = {
    'Bimodal': 'kunci.jpg',
    'Uneven Illumination': 'iluminasi.jpg',
    'Overlapping': 'overlapping.jpg'
}

def load_data(data_dir, assignment_dict):
    """Memuat citra real dan ground truth (opsional)."""
    datasets = []
    
    for characteristic, filename in assignment_dict.items():
        # Memuat citra input dalam grayscale
        img_path = os.path.join(data_dir, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Citra input '{img_path}' tidak ditemukan. Silakan taruh di folder '{DATA_DIR}'.")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Mencari dan memuat ground truth yang sesuai
        name_no_ext = os.path.splitext(filename)[0]
        gt_filename = name_no_ext + '_GT.png'
        gt_path = os.path.join(data_dir, gt_filename)
        
        gt_bin = None
        gt_source = "auto"
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            _, gt_bin = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY)
            gt_source = "file"
        else:
            gt_bin = generate_auto_gt(img)
        
        # Catatan Khusus untuk Beras (Overlapping):
        # Untuk tujuan evaluasi pixel-wise yang konsisten (IoU), user harus membuat
        # mask biner di mana SEMUA butir beras (dan hanya beras) adalah putih.
        # Ini akan sangat sulit dan membutuhkan pelabelan manual yang teliti.
        
        datasets.append((characteristic, img, gt_bin, gt_source))
        
    return datasets

def evaluate_metrics(gt, pred):
    """Menghitung metrik evaluasi segmentasi biner."""
    gt_bin = (gt > 127).astype(np.uint8).flatten()
    pred_bin = (pred > 127).astype(np.uint8).flatten()
    
    intersection = np.logical_and(gt_bin, pred_bin)
    union = np.logical_or(gt_bin, pred_bin)
    
    # Hitung pixel TRUE (Objek), FALSE (Background)
    tp = np.sum(intersection)
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))
    tn = np.sum((pred_bin == 0) & (gt_bin == 0))
    
    # Hitung metrik
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    acc = (tp + tn) / (gt_bin.size + 1e-6)
    
    return iou, dice, acc, prec, rec

def generate_auto_gt(img):
    """Membuat pseudo-ground-truth otomatis dari citra grayscale."""
    # Otsu + morfologi ringan untuk mengurangi noise
    _, gt_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    gt_clean = cv2.morphologyEx(gt_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    gt_clean = cv2.morphologyEx(gt_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    return gt_clean

# --- METODE THRESHOLDING ---
def apply_thresholding(img):
    results = {}
    
    # Global Thresholding (Manual T=127)
    start = time.time()
    _, res_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    results['Global (T=127)'] = (res_global, time.time() - start)
    
    # Otsu's Method
    start = time.time()
    _, res_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['Otsu'] = (res_otsu, time.time() - start)
    
    # Adaptive Mean (Disesuaikan untuk Real Data)
    start = time.time()
    # Gunakan neighborhood size yang lebih besar (misal 21) dan C yang lebih besar untuk real data
    res_adapt_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
    results['Adaptive Mean'] = (res_adapt_mean, time.time() - start)
    
    # Adaptive Gaussian
    start = time.time()
    res_adapt_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    results['Adaptive Gaussian'] = (res_adapt_gauss, time.time() - start)
    
    return results

# --- METODE EDGE DETECTION ---
def apply_edges(img):
    results = {}
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Sobel
    start = time.time()
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_res = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    # Threshold manual untuk binarisasi edge sobel
    _, sobel_bin = cv2.threshold(sobel_res, 30, 255, cv2.THRESH_BINARY)
    results['Sobel'] = (sobel_bin, time.time() - start)
    
    # Prewitt
    start = time.time()
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewittx = cv2.filter2D(img_blur, -1, kernelx)
    prewitty = cv2.filter2D(img_blur, -1, kernely)
    prewitt_res = cv2.add(np.abs(prewittx), np.abs(prewitty))
    # Threshold manual untuk binarisasi edge prewitt
    _, prewitt_bin = cv2.threshold(prewitt_res, 30, 255, cv2.THRESH_BINARY)
    results['Prewitt'] = (prewitt_bin, time.time() - start)
    
    # Canny (Disesuaikan untuk Real Data)
    start = time.time()
    # Threshold lebih rendah untuk menangkap tepi yang lebih lembut pada real data
    canny_res = cv2.Canny(img_blur, 30, 100)
    results['Canny'] = (canny_res, time.time() - start)
    
    return results

# --- METODE REGION-BASED ---
def apply_region(img):
    results = {}
    
    # Connected Components (menggunakan hasil Otsu sebagai input)
    start = time.time()
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels = cv2.connectedComponents(thresh)
    cc_res = np.zeros_like(img)
    if num_labels > 1:
        # Menemukan label terbesar (asumsi objek utama, untuk Beras ini akan salah)
        # Akan memakan waktu lebih lama untuk Beras
        largest_label_id = np.argmax(np.bincount(labels.flatten())[1:]) + 1
        cc_res[labels == largest_label_id] = 255
    results['Connected Comp'] = (cc_res, time.time() - start)
    
    # Watershed (Marker-based) - Disesuaikan untuk Beras
    start = time.time()
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Pre-processing untuk Beras: Thresholding awal yang sangat sensitif
    # Ini adalah bottleneck pada foto beras karena kontras rendah.
    _, thresh_ws = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV) 
    
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh_ws, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Threshold watershed: MENURUNKAN threshold untuk menangkap beras yang tumpang tindih
    # Namun, karena kontras rendah dengan tikar, ini mungkin akan menangkap anyaman bambu.
    _, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_color, markers)
    ws_res = np.zeros_like(img)
    ws_res[markers > 1] = 255
    results['Watershed'] = (ws_res, time.time() - start)
    
    return results

def main():
    print("="*60)
    print(" PIPELINE SEGMENTASI CITRA REAL-WORLD - MUHAMMAD ZAHRAN (24343077)")
    print("="*60)
    print(f"Data Dir: '{os.path.abspath(DATA_DIR)}'")
    
    # Memuat data
    try:
        datasets = load_data(DATA_DIR, image_assignments)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    all_methods = ["Global (T=127)", "Otsu", "Adaptive Mean", "Adaptive Gaussian", 
                   "Sobel", "Prewitt", "Canny", "Connected Comp", "Watershed"]
    
    # Konfigurasi Grid visualisasi layout
    n_rows = 4
    n_cols = 3
    
    for name, img, gt, gt_source in datasets:
        print(f"\n--- Evaluasi Citra: {name} ({image_assignments[name]}) ---")
        
        # Eksekusi Metode
        res_thresh = apply_thresholding(img)
        res_edges = apply_edges(img)
        res_region = apply_region(img)
        
        combined_results = {**res_thresh, **res_edges, **res_region}
        
        # Cetak Metrik
        if gt_source == "auto":
            pass
        print(f"{'Metode':<20} | {'IoU':<6} | {'Dice':<6} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'Waktu (s)':<8}")
        print("-" * 80)
        
        best_iou = -1
        best_method = ""
        
        for method_name in all_methods:
            if method_name not in combined_results: continue
            pred, t = combined_results[method_name]
            iou, dice, acc, prec, rec = evaluate_metrics(gt, pred)
            print(f"{method_name:<20} | {iou:.4f} | {dice:.4f} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {t:.6f}")
            if iou > best_iou:
                best_iou = iou
                best_method = method_name
                
        print(f">> Metode Terbaik untuk {name} (berdasarkan IoU): {best_method} (IoU: {best_iou:.4f})")
        
        # Visualisasi (Grid Comprehensive per image)
        plt.figure(figsize=(20, 15))
        plt.suptitle(f"Hasil Segmentasi Comprehensive - {name} ({image_assignments[name]})\nBy: M. Zahran (24343077)", fontsize=18)
        
        # Plot Original
        plt.subplot(n_rows, n_cols, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Citra Asli (Grayscale)', fontsize=14)
        plt.axis('off')
        
        # Plot Ground Truth
        plt.subplot(n_rows, n_cols, 2)
        plt.imshow(gt, cmap='gray')
        if gt_source == "auto":
            plt.title('Ground Truth (Auto / Pseudo)', fontsize=14)
        else:
            plt.title('Ground Truth (Binary Mask)', fontsize=14)
        plt.axis('off')
        
        # Plot 9 segmentation results in a grid (from 3 to 11)
        for i, method_name in enumerate(all_methods):
            if method_name not in combined_results: continue
            pred, _ = combined_results[method_name]
            plt.subplot(n_rows, n_cols, i + 3)
            plt.imshow(pred, cmap='gray')
            plt.title(f"{method_name} (IoU: {evaluate_metrics(gt, pred)[0]:.2f})", fontsize=12)
            plt.axis('off')
            
        # Plot Best Prediction Overlay Contour (di sel 12)
        best_pred = combined_results[best_method][0]
        plt.subplot(n_rows, n_cols, 12)
        
        # Mencari kontur pada prediksi terbaik
        contours, _ = cv2.findContours(best_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2) # Kontur Merah
        
        # Visualisasi contour dengan WM
        plt.imshow(img_color)
        plt.title(f'Visual Terbaik: Overlay Kontur Merah\nBy: M. Zahran (24343077)', fontsize=12, color='white', fontweight='bold', backgroundcolor='red')
        plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92) # Tambahan spasi untuk suptitle
        plt.show()

if __name__ == "__main__":
    main()