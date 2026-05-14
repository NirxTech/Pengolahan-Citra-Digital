import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def praktikum_9_1():
    """
    Perbandingan teknik thresholding: Global, Otsu, dan Adaptive
    """
    print("PRAKTIKUM 9.1: PERBANDINGAN TEKNIK THRESHOLDING")
    print("=" * 60)
    
    # Buat citra test dengan berbagai karakteristik
    def create_test_images():
        images = {}
        
        # 1. Citra bimodal (ideal untuk thresholding)
        img_bimodal = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_bimodal, (30, 30), (150, 150), 50, -1)  # Dark object
        cv2.rectangle(img_bimodal, (100, 100), (220, 220), 200, -1)  # Bright object
        images['Bimodal Image'] = img_bimodal
        
        # 2. Citra dengan uneven illumination
        img_uneven = np.zeros((256, 256), dtype=np.uint8)
        # Create gradient background
        for i in range(256):
            img_uneven[:, i] = i // 2
        # Add objects
        cv2.rectangle(img_uneven, (50, 50), (100, 100), 255, -1)
        cv2.rectangle(img_uneven, (150, 150), (200, 200), 100, -1)
        images['Uneven Illumination'] = img_uneven
        
        # 3. Citra dengan noise
        img_noisy = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_noisy, (50, 50), (150, 150), 128, -1)
        # Add Gaussian noise
        noise = np.random.normal(0, 30, img_noisy.shape)
        img_noisy = np.clip(img_noisy.astype(float) + noise, 0, 255).astype(np.uint8)
        images['Noisy Image'] = img_noisy
        
        # 4. Citra dengan multiple intensity levels
        img_multi = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img_multi, (30, 30), (90, 90), 80, -1)   # Dark gray
        cv2.rectangle(img_multi, (100, 30), (160, 90), 120, -1)  # Medium gray
        cv2.rectangle(img_multi, (170, 30), (230, 90), 180, -1)  # Light gray
        images['Multi-level Image'] = img_multi
        
        return images
    
    # Implementasi berbagai metode thresholding
    def apply_global_threshold(image, T=127):
        """Global thresholding"""
        _, binary = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
        return binary
    
    def apply_otsu_threshold(image):
        """Otsu's thresholding"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def apply_adaptive_threshold(image, block_size=11, C=2):
        """Adaptive thresholding"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block_size, C)
        return binary
    
    def apply_iterative_threshold(image, max_iter=100, tolerance=1):
        """Iterative threshold selection"""
        # Initialize threshold
        T = np.mean(image)
        
        for i in range(max_iter):
            # Segment image
            foreground = image[image > T]
            background = image[image <= T]
            
            # Compute means
            if len(foreground) > 0 and len(background) > 0:
                mu_fg = np.mean(foreground)
                mu_bg = np.mean(background)
                
                # New threshold
                T_new = (mu_fg + mu_bg) / 2
                
                # Check convergence
                if abs(T_new - T) < tolerance:
                    T = T_new
                    break
                    
                T = T_new
            else:
                break
        
        # Apply threshold
        _, binary = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
        return binary, T
    
    # Buat citra test
    test_images = create_test_images()
    
    # Terapkan berbagai metode thresholding
    results = {}
    
    for name, image in test_images.items():
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply different thresholding methods
        global_binary = apply_global_threshold(gray, 127)
        otsu_binary = apply_otsu_threshold(gray)
        adaptive_binary = apply_adaptive_threshold(gray, 11, 2)
        iterative_binary, T_iter = apply_iterative_threshold(gray)
        
        # FIX: Get Otsu threshold value as a scalar
        # cv2.threshold returns (retval, dst)
        T_otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure T_iter is also a scalar
        T_iter_val = float(T_iter[0]) if isinstance(T_iter, (np.ndarray, list)) else float(T_iter)

        results[name] = {
            'original': gray,
            'global': global_binary,
            'otsu': otsu_binary,
            'adaptive': adaptive_binary,
            'iterative': iterative_binary,
            'T_otsu': float(T_otsu_val), # Convert to float scalar
            'T_iter': T_iter_val         # Convert to float scalar
        }
    
    # Visualisasi hasil
    fig, axes = plt.subplots(len(test_images), 5, figsize=(20, 4*len(test_images)))
    
    for idx, (name, result) in enumerate(results.items()):
        # Column 1: Original image + histogram
        axes[idx, 0].imshow(result['original'], cmap='gray')
        axes[idx, 0].set_title(f'{name}\nOriginal')
        axes[idx, 0].axis('off')
        
        # Column 2-5: Thresholding results
        methods = ['global', 'otsu', 'adaptive', 'iterative']
        titles = ['Global (T=127)', f'Otsu (T={result["T_otsu"]:.0f})', 
                 'Adaptive', f'Iterative (T={result["T_iter"]:.0f})']
        
        for col, (method, title) in enumerate(zip(methods, titles), 1):
            axes[idx, col].imshow(result[method], cmap='gray')
            axes[idx, col].set_title(title)
            axes[idx, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analisis histogram dan threshold selection
    print("\nANALISIS HISTOGRAM DAN THRESHOLD SELECTION")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(list(results.items())[:4]):
        # Plot histogram
        hist = cv2.calcHist([result['original']], [0], None, [256], [0, 256])
        axes[idx].plot(hist, 'k-', linewidth=2)
        
        # Add threshold lines
        axes[idx].axvline(x=127, color='r', linestyle='--', label='Global (127)', alpha=0.7)
        axes[idx].axvline(x=result['T_otsu'], color='g', linestyle='--', 
                         label=f'Otsu ({result["T_otsu"]:.0f})', alpha=0.7)
        axes[idx].axvline(x=result['T_iter'], color='b', linestyle='--',
                         label=f'Iterative ({result["T_iter"]:.0f})', alpha=0.7)
        
        axes[idx].set_title(f'{name}\nHistogram with Thresholds')
        axes[idx].set_xlabel('Intensity')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 255])
    
    plt.tight_layout()
    plt.show()
    
    # Evaluasi kuantitatif (simulasi ground truth)
    print("\nEVALUASI KUANTITATIF (DENGAN SIMULASI GROUND TRUTH)")
    print("-" * 70)
    
    # Buat ground truth untuk bimodal image
    gt_bimodal = np.zeros((256, 256), dtype=np.uint8)
    gt_bimodal[30:150, 30:150] = 1  # First object
    gt_bimodal[100:220, 100:220] = 1  # Second object
    
    # Hitung metrics untuk setiap metode
    def calculate_metrics(binary, ground_truth):
        """Calculate segmentation metrics"""
        # Ensure binary images
        binary = (binary > 0).astype(np.uint8)
        ground_truth = (ground_truth > 0).astype(np.uint8)
        
        # True Positive, False Positive, etc.
        tp = np.sum((binary == 1) & (ground_truth == 1))
        fp = np.sum((binary == 1) & (ground_truth == 0))
        fn = np.sum((binary == 0) & (ground_truth == 1))
        tn = np.sum((binary == 0) & (ground_truth == 0))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou
        }
    
    # Evaluasi untuk bimodal image
    bimodal_result = results['Bimodal Image']
    
    print(f"{'Method':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10}")
    print("-" * 70)
    
    methods = ['global', 'otsu', 'adaptive', 'iterative']
    method_names = ['Global', 'Otsu', 'Adaptive', 'Iterative']
    
    for method, method_name in zip(methods, method_names):
        metrics = calculate_metrics(bimodal_result[method], gt_bimodal)
        print(f"{method_name:<15} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
              f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f} {metrics['iou']:<10.3f}")
    
    # Visual comparison dengan ground truth
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original and ground truth
    axes[0, 0].imshow(bimodal_result['original'], cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_bimodal, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')  # Empty
    
    # Row 2: Thresholding results dengan overlay errors
    for idx, (method, method_name) in enumerate(zip(methods[:3], method_names[:3])):
        result_binary = (bimodal_result[method] > 0).astype(np.uint8)
        
        # Create error visualization
        error_image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # True Positive: White
        tp_mask = (result_binary == 1) & (gt_bimodal == 1)
        error_image[tp_mask] = [255, 255, 255]
        
        # False Positive: Red (segmented but not in GT)
        fp_mask = (result_binary == 1) & (gt_bimodal == 0)
        error_image[fp_mask] = [255, 0, 0]
        
        # False Negative: Blue (in GT but not segmented)
        fn_mask = (result_binary == 0) & (gt_bimodal == 1)
        error_image[fn_mask] = [0, 0, 255]
        
        axes[1, idx].imshow(error_image)
        axes[1, idx].set_title(f'{method_name}\n(Red: FP, Blue: FN)')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Kesimpulan dan rekomendasi
    print("\nKESIMPULAN DAN REKOMENDASI")
    print("-" * 40)
    print("""
1. Global Thresholding:
   - Cocok untuk citra dengan kontras tinggi dan illumination uniform
   - Sensitif terhadap uneven illumination dan noise
   - Butuh manual threshold selection

2. Otsu's Method:
   - Fully automatic
   - Optimal untuk citra bimodal histogram
   - Tidak efektif untuk multi-modal atau uneven illumination

3. Adaptive Thresholding:
   - Cocok untuk uneven illumination
   - Robust terhadap variasi intensitas lokal
   - Parameter block size dan C perlu tuning

4. Iterative Thresholding:
   - Self-tuning threshold
   - Cocok untuk citra dengan distribusi intensity yang jelas
   - Computationally lebih expensive
    """)
    
    return results

thresholding_results = praktikum_9_1()

def praktikum_9_2():
    """
    Implementasi edge detection dan region-based segmentation
    """
    print("\nPRAKTIKUM 9.2: EDGE DETECTION DAN REGION-BASED SEGMENTATION")
    print("=" * 70)
    
    # 1. Buat citra test dengan berbagai edge types
    def create_edge_test_image():
        """Create test image with different edge types"""
        # Ukuran citra 300x400
        img = np.zeros((300, 400), dtype=np.uint8)
        
        # Step edge (sharp transition)
        cv2.rectangle(img, (50, 50), (150, 150), 100, -1)
        cv2.rectangle(img, (151, 50), (250, 150), 200, -1)
        
        # Ramp edge (gradual transition)
        for i in range(50, 150):
            img[160:240, i] = 50 + (i - 50) * 2
        
        # Roof edge (triangular)
        triangle_cnt = np.array([(300, 160), (350, 240), (250, 240)])
        cv2.drawContours(img, [triangle_cnt], 0, 150, -1)
        
        # Line edge (thin line)
        cv2.line(img, (50, 260), (350, 260), 200, 3)
        
        # Add noise untuk testing robustness
        noise = np.random.normal(0, 15, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    # 2. Implementasi berbagai edge detectors
    def apply_sobel_edge_detection(image, ksize=3):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        return magnitude.astype(np.uint8), direction
    
    def apply_prewitt_edge_detection(image):
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewittx = cv2.filter2D(image.astype(np.float64), -1, kernelx)
        prewitty = cv2.filter2D(image.astype(np.float64), -1, kernely)
        magnitude = np.sqrt(prewittx**2 + prewitty**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return magnitude.astype(np.uint8)
    
    def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
        blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges
    
    def apply_laplacian_edge_detection(image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
        return laplacian.astype(np.uint8)
    
    # 3. Region-based segmentation methods
    def region_growing_segmentation(image, seeds, threshold=20):
        segmented = np.zeros_like(image)
        visited = np.zeros_like(image, dtype=bool)
        h, w = image.shape
        
        if isinstance(seeds, tuple):
            seeds = [seeds]
        
        for seed in seeds:
            # Pastikan seed berada di dalam dimensi gambar (mencegah IndexError)
            if not (0 <= seed[0] < h and 0 <= seed[1] < w) or visited[seed]:
                continue
                
            stack = [seed]
            region_val = float(image[seed]) # Simpan nilai awal seed
            
            while stack:
                x, y = stack.pop()
                if visited[x, y]:
                    continue
                    
                visited[x, y] = True
                segmented[x, y] = 255
                
                # Check 4-connected neighbors
                for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                    if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                        neighbor_value = float(image[nx, ny])
                        # Perbaikan: cast ke float/int untuk mencegah overflow uint8
                        if abs(neighbor_value - region_val) < threshold:
                            stack.append((nx, ny))
        return segmented

    # Simplified Split & Merge untuk performa lebih stabil
    def region_splitting_merging(image, min_size=32, threshold=20):
        result = np.zeros_like(image)
        h, w = image.shape
        
        # Simulasi grid-based splitting sederhana untuk visualisasi
        for i in range(0, h, min_size):
            for j in range(0, w, min_size):
                region = image[i:i+min_size, j:j+min_size]
                if np.std(region) < threshold:
                    result[i:i+min_size, j:j+min_size] = np.mean(region)
                else:
                    result[i:i+min_size, j:j+min_size] = 255
        return result

    # 4. Eksekusi Program
    test_image = create_edge_test_image()
    edge_results = {}
    
    edge_results['Sobel'], sobel_direction = apply_sobel_edge_detection(test_image)
    edge_results['Prewitt'] = apply_prewitt_edge_detection(test_image)
    edge_results['Canny'] = apply_canny_edge_detection(test_image)
    edge_results['Laplacian'] = apply_laplacian_edge_detection(test_image)
    
    # SEEDS FIX: Index 300 diubah ke 299 agar tidak Out of Bounds
    seeds = [(75, 100), (200, 100), (299, 200)] 
    
    region_results = {}
    region_results['Region Growing'] = region_growing_segmentation(test_image, seeds, threshold=25)
    region_results['Split & Merge'] = region_splitting_merging(test_image, min_size=32, threshold=20)
    
    # 5. Visualisasi Hasil (FIXED INDEXING)
    print("\nVISUALIZING RESULTS...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Baris 1: Original dan Edge Detectors
    axes[0, 0].imshow(test_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    
    # Gunakan hanya 2 metode di baris atas agar total kolom pas (3)
    axes[0, 1].imshow(edge_results['Sobel'], cmap='gray')
    axes[0, 1].set_title('Sobel Edge')
    
    axes[0, 2].imshow(edge_results['Canny'], cmap='gray')
    axes[0, 2].set_title('Canny Edge')
    
    # Baris 2: Region Based
    axes[1, 0].imshow(region_results['Region Growing'], cmap='gray')
    axes[1, 0].set_title('Region Growing')
    
    axes[1, 1].imshow(region_results['Split & Merge'], cmap='gray')
    axes[1, 1].set_title('Split & Merge')
    
    # Overlay Boundaries pada Gambar Original
    boundaries = cv2.Canny(region_results['Region Growing'].astype(np.uint8), 50, 150)
    overlay = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
    overlay[boundaries > 0] = [0, 255, 0] # Warna Hijau
    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Region Boundaries Overlay')
    
    for ax in axes.ravel():
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    
    print("\nPROSES SELESAI.")
    return test_image, edge_results, region_results

# Menjalankan fungsi
if __name__ == "__main__":
    praktikum_9_2()