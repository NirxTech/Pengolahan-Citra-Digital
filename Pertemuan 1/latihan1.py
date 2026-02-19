import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import os

print("=== LATIHAN 1: ANALISIS PROPERTI CITRA ===")

def download_backup_image(filename="UNP.jpeg"):
    """Fungsi cadangan jika tidak ada foto pribadi"""
    print(f"File '{filename}' tidak ditemukan. Mendownload gambar contoh...")
    url = "isi dengan URL gambar yang valid" 
    try:
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print("Download selesai.")
        return True
    except Exception as e:
        print(f"Gagal download: {e}")
        return False

def analyze_my_image(image_path):
    """Fungsi utama untuk menganalisis gambar"""
    
    # Cek apakah file ada, jika tidak, download contoh
    if not os.path.exists(image_path):
        if not download_backup_image(image_path):
            return

    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Gambar rusak atau format tidak didukung.")
        return

    print(f"\n{'='*40}")
    print(f"HASIL ANALISIS: {image_path}")
    print(f"{'='*40}")

    # 1. Dimensi & Resolusi
    height, width, channels = img.shape
    resolution = width * height
    print(f"1. Dimensi     : {width} x {height}")
    print(f"   Resolusi    : {resolution:,} pixels")

    # 2. Aspect Ratio
    aspect_ratio = width / height
    print(f"2. Aspect Ratio: {aspect_ratio:.2f}")

    # 3. Konversi ke Grayscale & Bandingkan Ukuran
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Estimasi ukuran data mentah (Raw Data)
    size_color = img.nbytes
    size_gray = gray.nbytes
    
    print(f"3. Ukuran Memori (Raw Data):")
    print(f"   - RGB (Warna) : {size_color/1024:.2f} KB")
    print(f"   - Grayscale   : {size_gray/1024:.2f} KB")
    print(f"   - Penghematan : {(1 - size_gray/size_color)*100:.1f}%")

    # 4. Statistik Warna
    print("4. Statistik Warna (BGR):")
    colors = ('Blue', 'Green', 'Red')
    for i, color in enumerate(colors):
        mean_val = img[:, :, i].mean()
        std_val = img[:, :, i].std()
        print(f"   - {color:<5} : Mean={mean_val:.1f}, Std Dev={std_val:.1f}")

    # 5. Visualisasi Histogram
    print("5. Menampilkan Histogram...")
    
    plt.figure(figsize=(12, 6))
    
    # Subplot Kiri: Gambar Asli
    plt.subplot(1, 2, 1)
    # Convert BGR to RGB for Matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Citra Asli")
    plt.axis('off')
    
    # Subplot Kanan: Histogram
    plt.subplot(1, 2, 2)
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=colors[i])
    
    plt.title("Histogram Intensitas Warna")
    plt.xlabel("Intensitas (0-255)")
    plt.ylabel("Jumlah Piksel")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    nama_file = "TehPucuk.jpg" 
    
    analyze_my_image(nama_file)