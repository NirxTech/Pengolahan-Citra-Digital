import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

# 1. MEMUAT PRE-TRAINED MODEL
print("Memuat AI Model...")
base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# 2. MEMBACA DATASET
# Pastikan folder 'dataset' kamu sekarang berisi campuran X-ray (Normal dan COVID)
folder_dataset = "Dataset" 
format_didukung = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG')
file_gambar = [f for f in os.listdir(folder_dataset) if f.endswith(format_didukung)][:10] # Batasi 10 gambar

if not file_gambar:
    print(f"Error: Tidak ada gambar ditemukan di folder '{folder_dataset}'.")
    exit()

print(f"Memproses {len(file_gambar)} gambar untuk perbandingan dan grafik...")

# Variabel untuk menyimpan hasil guna keperluan grafik
hasil_skor_covid = []
gambar_ditampilkan = []
nama_file_list = []
label_prediksi = []

# 3. PROSES SCREENING & PREDIKSI
for nama_file in file_gambar:
    path_gambar = os.path.join(folder_dataset, nama_file)
    
    img_bgr = cv2.imread(path_gambar)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocessing
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)
    
    # Prediksi
    prediksi = model.predict(img_input, verbose=0)[0]
    skor_covid = prediksi[1] # Mengambil nilai probabilitas infeksi COVID-19
    
    # Menentukan Label
    if skor_covid > 0.5:
        label = "COVID-19"
    else:
        label = "NORMAL"
        
    # Menyimpan data untuk visualisasi
    hasil_skor_covid.append(skor_covid)
    gambar_ditampilkan.append(img_rgb)
    nama_file_list.append(nama_file)
    label_prediksi.append(label)

# 4. VISUALISASI HASIL & GRAFIK
# Membuat layout menggunakan GridSpec agar rapi
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Sistem PCD Screening COVID-19 & Analisis Grafik", fontsize=18, fontweight='bold')
gs = gridspec.GridSpec(3, 5, figure=fig) # 3 baris, 5 kolom

# --- BAGIAN A: Menampilkan 10 Gambar (Baris 1 dan 2) ---
for i in range(len(gambar_ditampilkan)):
    baris = i // 5
    kolom = i % 5
    ax = fig.add_subplot(gs[baris, kolom])
    
    ax.imshow(gambar_ditampilkan[i])
    
    # Format teks berdasarkan hasil
    skor = hasil_skor_covid[i]
    warna_teks = 'red' if label_prediksi[i] == "COVID-19" else 'green'
    
    ax.set_title(f"{nama_file_list[i]}\n{label_prediksi[i]} ({(skor*100):.1f}%)", color=warna_teks, fontsize=10, fontweight='bold')
    ax.axis('off')

# --- BAGIAN B: Menampilkan Grafik Perbandingan (Baris 3) ---
ax_grafik = fig.add_subplot(gs[2, :]) # Mengambil seluruh kolom di baris ke-3

# Menentukan warna batang grafik (Merah = Covid, Hijau = Normal)
warna_bar = ['red' if skor > 0.5 else 'green' for skor in hasil_skor_covid]
posisi_x = np.arange(len(nama_file_list))

ax_grafik.bar(posisi_x, hasil_skor_covid, color=warna_bar, edgecolor='black')
ax_grafik.set_xticks(posisi_x)
ax_grafik.set_xticklabels(nama_file_list, rotation=15, ha='right')
ax_grafik.set_ylabel('Probabilitas Terinfeksi COVID-19', fontweight='bold')
ax_grafik.set_title('Grafik Analisis Probabilitas Infeksi per Gambar', fontweight='bold')

# Menambahkan garis batas (Threshold) pada angka 0.5 (50%)
ax_grafik.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Batas Threshold Normal/Covid (0.5)')
ax_grafik.legend()

# Merapikan jarak antar elemen
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
print("Proses selesai! Menampilkan antarmuka hasil...")
plt.show()