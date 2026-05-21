import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import urllib.request
import ssl
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. DATASET GENERATOR (MENGGUNAKAN CITRA ONLINE)
# ==========================================
def download_image(url):
    """Fungsi pembantu untuk mengunduh gambar dari internet secara aman."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req, context=ctx) as response:
        image_bytes = response.read()

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def create_synthetic_dataset():
    """Mengunduh objek dari internet dan membuat variasinya di memori."""
    categories = ["buku", "mug", "botol", "mainan", "remote"]
    
    # URL gambar beresolusi 400x400 (Tautan 'remote' telah diperbarui ke yang valid)
    image_urls = {
        "buku": "https://images.unsplash.com/photo-1544947950-fa07a98d237f?q=80&w=400&auto=format&fit=crop",
        "mug": "https://images.unsplash.com/photo-1514228742587-6b1558fcca3d?q=80&w=400&auto=format&fit=crop",
        "botol": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?q=80&w=400&auto=format&fit=crop",
        "mainan": "https://images.unsplash.com/photo-1596461404969-9ae70f2830c1?q=80&w=400&auto=format&fit=crop",
        "remote": "https://images.unsplash.com/photo-1558089687-f282ffcbc126?q=80&w=400&auto=format&fit=crop"
    }

    print("[INFO] Memulai pengunduhan citra referensi dari internet...")

    dataset = {}
    
    for cat in categories:
        try:
            ref_img = download_image(image_urls[cat])
            if ref_img is None:
                raise Exception("Gambar terunduh namun gagal dibaca OpenCV.")

            ref_img = cv2.resize(ref_img, (400, 400))
        except Exception as e:
            print(f"[ERROR] Gagal memproses gambar untuk '{cat}': {e}. Kategori ini akan dilewati.")
            continue # Lewati ke gambar berikutnya tanpa merusak sistem
        
        M_rot = cv2.getRotationMatrix2D((200, 200), 45, 1.0)
        rot_img = cv2.warpAffine(ref_img, M_rot, (400, 400), borderValue=(255,255,255))
        
        scale_img = cv2.resize(ref_img, (0,0), fx=0.6, fy=0.6)
        pad_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        offset_y = (400 - scale_img.shape[0]) // 2
        offset_x = (400 - scale_img.shape[1]) // 2
        pad_img[offset_y:offset_y+scale_img.shape[0], offset_x:offset_x+scale_img.shape[1]] = scale_img
        
        blur_img = cv2.GaussianBlur(ref_img, (5, 5), 0)
        dark_img = cv2.convertScaleAbs(blur_img, alpha=0.5, beta=-20)
        
        occ_img = ref_img.copy()
        cv2.rectangle(occ_img, (0, 0), (180, 180), (120, 120, 120), -1) 
        
        dataset[cat] = {
            "reference": ref_img,
            "variants": {
                "rotation": rot_img,
                "scale": pad_img,
                "illumination": dark_img,
                "occlusion": occ_img,
            },
        }

    print(f"[INFO] Dataset ({len(dataset)} kategori) berhasil dibuat di memori.\n")
    return dataset

def visualize_dataset(dataset):
    sns.set_theme(style="white", font_scale=0.9)

    categories = list(dataset.keys())
    if not categories:
        print("[WARNING] Dataset kosong, tidak ada visualisasi yang bisa ditampilkan.")
        return

    variant_names = ["reference", "rotation", "scale", "illumination", "occlusion"]
    title_map = {
        "reference": "Reference",
        "rotation": "Rotation",
        "scale": "Scale",
        "illumination": "Illumination",
        "occlusion": "Occlusion",
    }

    fig, axes = plt.subplots(len(categories), len(variant_names), figsize=(16, 3.2 * len(categories)))
    if len(categories) == 1:
        axes = np.expand_dims(axes, axis=0)

    palette = sns.color_palette("deep", len(categories))

    for row, cat in enumerate(categories):
        for col, variant in enumerate(variant_names):
            ax = axes[row, col]
            img = dataset[cat]["reference"] if variant == "reference" else dataset[cat]["variants"][variant]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if row == 0:
                ax.set_title(title_map[variant], color=palette[col % len(palette)])
            if col == 0:
                ax.set_ylabel(cat, rotation=0, labelpad=35, va="center", fontsize=11, color=palette[row % len(palette)])

    plt.suptitle("Visualisasi Dataset Sintetik", y=1.01, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# ==========================================
# 2. DETEKSI, DESKRIPSI, DAN PENCOCOKAN FITUR
# ==========================================
def get_detector(name):
    if name.upper() == "SIFT":
        return cv2.SIFT_create()
    elif name.upper() == "ORB":
        return cv2.ORB_create(nfeatures=1500)
    elif name.upper() == "SURF":
        try:
            return cv2.xfeatures2d.SURF_create()
        except AttributeError:
            print("[WARNING] SURF tidak tersedia pada build OpenCV ini. Mengalihkan otomatis ke SIFT.")
            return cv2.SIFT_create()
    else:
        raise ValueError(f"Metode {name} tidak dikenal.")

def extract_features(img, detector):
    if img is None:
        return [], None, 0.0

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    start_time = time.time()
    kp, des = detector.detectAndCompute(gray, None)
    elapsed_time = time.time() - start_time
    
    return kp, des, elapsed_time

def match_features(des1, des2, method="BF", feature_type="SIFT", ratio_thresh=0.75):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
        
    if feature_type.upper() == "ORB":
        norm_type = cv2.NORM_HAMMING
        index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        use_lsh = True
    else:
        norm_type = cv2.NORM_L2
        index_params = dict(algorithm=1, trees=5)
        use_lsh = False

    search_params = dict(checks=50)

    if method == "BF":
        matcher = cv2.BFMatcher(norm_type)
    else:  
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    if method != "BF":
        if use_lsh:
            des1_match = des1.astype(np.uint8) if des1.dtype != np.uint8 else des1
            des2_match = des2.astype(np.uint8) if des2.dtype != np.uint8 else des2
        else:
            des1_match = des1.astype(np.float32)
            des2_match = des2.astype(np.float32)
    else:
        des1_match = des1
        des2_match = des2

    raw_matches = matcher.knnMatch(des1_match, des2_match, k=2)

    good_matches = []
    for m_n in raw_matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return good_matches

def estimate_homography_ransac(kp1, kp2, matches, min_matches=4):
    if len(matches) < min_matches:
        return None, 0
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(np.sum(mask)) if mask is not None else 0
    return H, inliers

# ==========================================
# 3. BAG OF VISUAL WORDS (BoVW) IMPLEMENTATION
# ==========================================
class BagOfVisualWords:
    def __init__(self, n_clusters=20, detector_name="SIFT"):
        self.n_clusters = n_clusters
        self.detector_name = detector_name
        self.detector = get_detector(detector_name)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        
    def build_vocabulary(self, images):
        all_descriptors = []
        for img in images:
            _, des, _ = extract_features(img, self.detector)
            if des is not None:
                all_descriptors.append(des)
                
        if len(all_descriptors) == 0:
            raise ValueError("Fitur deskriptor tidak ditemukan pada data training.")
            
        all_descriptors = np.vstack(all_descriptors)
        self.kmeans.fit(all_descriptors.astype(float))
        
    def construct_histogram(self, img):
        _, des, _ = extract_features(img, self.detector)
        hist = np.zeros(self.n_clusters)
        
        if des is not None:
            words = self.kmeans.predict(des.astype(float))
            for word in words:
                hist[word] += 1
        sum_val = np.sum(hist)
        if sum_val > 0:
            hist /= sum_val
        return hist

    def transform_dataset(self, images):
        features = []
        for img in images:
            hist = self.construct_histogram(img)
            features.append(hist)
        return np.array(features)

# ==========================================
# 4. PIPELINE EVALUASI & ANALISIS UTAMA
# ==========================================
def main(show_visuals=False):
    dataset = create_synthetic_dataset()
    categories = list(dataset.keys())
    
    if len(categories) == 0:
        print("[ERROR] Tidak ada dataset yang bisa diproses. Hentikan eksekusi.")
        return

    visualize_dataset(dataset)

    methods = ["SIFT", "ORB"] 
    
    print("="*65)
    print(" EKSPLORASI TUGAS 1 & 2: DETEKSI, EKSTRAKSI DAN MATCHING METODE ")
    print("="*65)
    
    performance_log = {m: {"time": [], "kp_num": [], "dim": None, "accuracy": []} for m in methods}
    
    for m_name in methods:
        detector = get_detector(m_name)
        match_counts = []
        inlier_counts = []
        
        print(f"\n[METODE] Mengevaluasi Feature Extractor: {m_name}")
        
        for cat in categories:
            ref_img = dataset[cat]["reference"]
            if ref_img is None:
                continue
                
            kp_ref, des_ref, t_ref = extract_features(ref_img, detector)
            performance_log[m_name]["time"].append(t_ref)
            performance_log[m_name]["kp_num"].append(len(kp_ref))
            if des_ref is not None:
                performance_log[m_name]["dim"] = des_ref.shape[1]
                
            for test_name, test_img in dataset[cat]["variants"].items():
                if test_img is None:
                    continue
                
                kp_test, des_test, _ = extract_features(test_img, detector)
                
                matches = match_features(des_ref, des_test, method="FLANN", feature_type=m_name)
                _, inliers = estimate_homography_ransac(kp_ref, kp_test, matches)
                
                match_counts.append(len(matches))
                inlier_counts.append(inliers)
        
        avg_time = np.mean(performance_log[m_name]["time"]) if performance_log[m_name]["time"] else 0
        avg_kp = np.mean(performance_log[m_name]["kp_num"]) if performance_log[m_name]["kp_num"] else 0
        avg_inliers = np.mean(inlier_counts) if inlier_counts else 0
        
        print(f" -> Rata-rata Keypoints: {avg_kp:.2f}")
        print(f" -> Rata-rata Waktu Ekstraksi: {avg_time*1000:.3f} ms")
        print(f" -> Dimensi Deskriptor: {performance_log[m_name].get('dim', 0)}")
        print(f" -> Rata-rata Geometri Inliers (RANSAC): {avg_inliers:.2f}")

    # ==========================================
    # K-MEANS & BAG OF VISUAL WORDS CLASSIFICATION
    # ==========================================
    print("\n" + "="*65)
    print(" EKSPLORASI TUGAS 3: BAG OF VISUAL WORDS (BoVW) & SVM ")
    print("="*65)
    
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []
    
    for idx, cat in enumerate(categories):
        ordered_images = [
            dataset[cat]["reference"],
            dataset[cat]["variants"]["rotation"],
            dataset[cat]["variants"]["scale"],
            dataset[cat]["variants"]["illumination"],
            dataset[cat]["variants"]["occlusion"],
        ]

        for i, img in enumerate(ordered_images):
            if i < 3:
                train_paths.append(img)
                train_labels.append(idx)
            else:
                test_paths.append(img)
                test_labels.append(idx)
                
    k_values = [10, 20, 50, 100]
    
    for k in k_values:
        bovw = BagOfVisualWords(n_clusters=k, detector_name="SIFT")
        bovw.build_vocabulary(train_paths)
        
        X_train = bovw.transform_dataset(train_paths)
        X_test = bovw.transform_dataset(test_paths)
        
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(X_train, train_labels)
        acc = clf.score(X_test, test_labels)
        print(f"[BoVW] Akurasi SVM dengan K={k:<3}: {acc*100:.1f}%")
        
        if k == 20:
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(test_labels, y_pred)
            print("\n[Confusion Matrix untuk BoVW K=20]")
            print(cm)
            print("\nClassification Report:")
            
            # Perbaikan Logika Target Names yang Adaptif Sesuai Data Test yang Ada
            unique_labels = np.unique(test_labels)
            target_names_adaptif = [categories[i] for i in unique_labels]
            
            print(classification_report(
                test_labels, 
                y_pred, 
                labels=unique_labels, # Paksa Scikit-Learn mengecek label yang benar-benar eksis
                target_names=target_names_adaptif, 
                zero_division=0
            ))

    # ==========================================
    # PRINCIPAL COMPONENT ANALYSIS (PCA) REDUCTION
    # ==========================================
    print("\n" + "="*65)
    print(" EKSPLORASI TUGAS 4: REDUKSI DIMENSI DESKRIPTOR SIFT DENGAN PCA ")
    print("="*65)
    
    sift_detector = cv2.SIFT_create()
    all_sift_des = []
    for cat in categories:
        img = dataset[cat]["reference"]
        if img is not None:
            _, des, _ = extract_features(img, sift_detector)
            if des is not None:
                all_sift_des.append(des)
            
    if len(all_sift_des) > 0:
        all_sift_des = np.vstack(all_sift_des)
        pca_components = [16, 32, 64, 128]
        
        for comp in pca_components:
            if comp > all_sift_des.shape[1]:
                comp = all_sift_des.shape[1]
                
            pca = PCA(n_components=comp, random_state=42)
            pca.fit(all_sift_des)
            
            evr = np.sum(pca.explained_variance_ratio_)
            print(f"[PCA] Komponen: {comp:<3} | Total Variance Retained: {evr*100:.2f}%")
    else:
        print("[ERROR] Tidak ada deskriptor SIFT yang berhasil diekstraksi untuk PCA.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Matching System demo")
    parser.add_argument('--show', dest='show', action='store_true', help='Tampilkan visualisasi GUI')
    args = parser.parse_args()
    main(show_visuals=args.show)