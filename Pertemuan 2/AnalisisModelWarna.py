import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_dummy_image():
    """Membuat citra dummy untuk simulasi jika tidak ada file gambar"""
    # Ukuran 400x400
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 1. Background (Langit Biru)
    img[:] = (250, 206, 135) # BGR
    
    # 2. Objek Kulit (Wajah sederhana)
    cv2.circle(img, (200, 200), 100, (180, 200, 255), -1) # Warna kulit (BGR)
    
    # 3. Bayangan (Shadow) Gelap di separuh wajah
    # Kita buat overlay hitam transparan
    shadow = img.copy()
    cv2.rectangle(shadow, (200, 100), (300, 300), (0, 0, 0), -1)
    img = cv2.addWeighted(img, 0.7, shadow, 0.3, 0)
    
    return img

def analyze_color_model_suitability(image, application):
    """
    Menganalisis model warna mana yang terbaik untuk aplikasi tertentu.
    """
    print(f"\nAnalisis untuk aplikasi: {application.upper()}")
    
    if application == 'skin_detection':
        # --- KASUS 1: DETEKSI KULIT (SKIN DETECTION) ---
        # Hipotesis: HSV lebih baik daripada RGB karena memisahkan warna (Hue) dari cahaya.
        
        # A. Pendekatan RGB (Sangat terpengaruh cahaya/bayangan)
        # Warna kulit sederhana: R > 95, G > 40, B > 20, dll.
        b, g, r = cv2.split(image)
        mask_rgb = (r > 95) & (g > 40) & (b > 20) & ((np.maximum(r,np.maximum(g,b)) - np.minimum(r,np.minimum(g,b))) > 15) & (np.abs(r-g) > 15) & (r > g) & (r > b)
        mask_rgb = mask_rgb.astype(np.uint8) * 255
        
        # B. Pendekatan HSV (Lebih robust)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Range warna kulit di HSV (biasanya Hue 0-20)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Visualisasi
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Citra Asli (Ada Bayangan)")
        
        ax[1].imshow(mask_rgb, cmap='gray')
        ax[1].set_title("Deteksi RGB (Gagal di Bayangan)")
        
        ax[2].imshow(mask_hsv, cmap='gray')
        ax[2].set_title("Deteksi HSV (Lebih Stabil)")
        
        plt.suptitle(f"Analisis: {application}", fontsize=14, fontweight='bold')
        plt.show()
        
    elif application == 'shadow_removal':
        # --- KASUS 2: PENGHAPUSAN BAYANGAN (SHADOW REMOVAL) ---
        # Hipotesis: LAB lebih baik karena memisahkan Luminance (L) dari warna (a,b).
        
        # A. Equalization di RGB (Merusak warna)
        b, g, r = cv2.split(image)
        r_eq = cv2.equalizeHist(r)
        g_eq = cv2.equalizeHist(g)
        b_eq = cv2.equalizeHist(b)
        res_rgb = cv2.merge([b_eq, g_eq, r_eq])
        
        # B. Equalization di LAB (Hanya channel L)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b_lab = cv2.split(lab)
        
        # Kita gunakan CLAHE (Adaptive Histogram Equalization) agar lebih halus
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_eq = clahe.apply(l)
        
        res_lab = cv2.merge([l_eq, a, b_lab])
        res_lab_bgr = cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
        
        # Visualisasi
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Citra Asli (Gelap/Bayangan)")
        
        ax[1].imshow(cv2.cvtColor(res_rgb, cv2.COLOR_BGR2RGB))
        ax[1].set_title("RGB Equalization (Warna Berubah)")
        
        ax[2].imshow(cv2.cvtColor(res_lab_bgr, cv2.COLOR_BGR2RGB))
        ax[2].set_title("LAB L-Channel (Warna Asli Terjaga)")
        
        plt.suptitle(f"Analisis: {application}", fontsize=14, fontweight='bold')
        plt.show()

# --- MAIN PROGRAM ---
if __name__ == "__main__":
    # 1. Buat citra simulasi
    img_test = create_dummy_image()
    
    # 2. Jalankan analisis
    analyze_color_model_suitability(img_test, 'skin_detection')
    analyze_color_model_suitability(img_test, 'shadow_removal')