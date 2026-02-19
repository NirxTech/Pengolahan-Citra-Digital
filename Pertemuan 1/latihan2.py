import numpy as np
import matplotlib.pyplot as plt

print("=== LATIHAN 2: SIMULASI SAMPLING & KUANTISASI ===")

def simulate_digitization(freq=1.0, duration=2.0, sampling_rate=10, quantization_levels=4):
    """
    Simulasi perubahan sinyal analog ke digital.
    
    Args:
    - freq: Frekuensi gelombang (Hz)
    - sampling_rate: Berapa kali ambil sampel per detik (Sumbu X)
    - quantization_levels: Berapa banyak tingkat kedalaman bit (Sumbu Y)
    """
    
    print(f"\nPARAMETER SIMULASI:")
    print(f"- Frekuensi Sinyal : {freq} Hz")
    print(f"- Sampling Rate    : {sampling_rate} Hz (Sample per detik)")
    print(f"- Level Kuantisasi : {quantization_levels} Levels (Bit Depth)")

    # 1. ANALOG SIGNAL (Sinyal Kontinu - High Resolution)
    # Kita pakai 1000 titik untuk mensimulasikan garis yang "mulus"
    t_analog = np.linspace(0, duration, 1000)
    y_analog = np.sin(2 * np.pi * freq * t_analog)

    # 2. SAMPLING (Diskrit dalam Waktu)
    # Hanya mengambil data pada detik-detik tertentu sesuai rate
    num_samples = int(duration * sampling_rate)
    t_sampled = np.linspace(0, duration, num_samples)
    y_sampled = np.sin(2 * np.pi * freq * t_sampled)

    # 3. QUANTIZATION (Diskrit dalam Amplitudo)
    # Membulatkan nilai y ke level terdekat
    
    # Normalisasi dulu dari range (-1 s/d 1) ke (0 s/d 1)
    y_normalized = (y_sampled + 1) / 2
    
    # Skala ke integer (0 s/d levels-1)
    max_level_val = quantization_levels - 1
    y_discrete_int = np.round(y_normalized * max_level_val)
    
    # Kembalikan ke range aslinya (-1 s/d 1) untuk ditampilkan
    y_quantized = (y_discrete_int / max_level_val) * 2 - 1

    # VISUALISASI
    plt.figure(figsize=(14, 6))

    # Plot 1: Proses Sampling (Titik-titik pengambilan)
    plt.subplot(1, 2, 1)
    plt.plot(t_analog, y_analog, label='Sinyal Asli (Analog)', color='lightgray', linewidth=2)
    plt.stem(t_sampled, y_sampled, linefmt='b-', markerfmt='bo', basefmt='r-', label='Titik Sample')
    plt.title(f'1. Proses Sampling\n(Rate: {sampling_rate} Hz)')
    plt.xlabel('Waktu (detik)')
    plt.ylabel('Amplitudo')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Hasil Digital (Kotak-kotak)
    plt.subplot(1, 2, 2)
    plt.plot(t_analog, y_analog, label='Sinyal Asli', color='lightgray', linestyle='--')
    
    # Step plot untuk menunjukkan efek digital yang "patah-patah"
    plt.step(t_sampled, y_quantized, where='mid', label='Sinyal Digital', color='red', linewidth=2)
    
    plt.title(f'2. Hasil Digitalisasi\n(Bit Depth: {quantization_levels} Levels)')
    plt.xlabel('Waktu (detik)')
    # Tampilkan garis grid horizontal sesuai level kuantisasi
    plt.yticks(np.linspace(-1, 1, quantization_levels)) 
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # SKENARIO 1: Kualitas Rendah (Patah-patah)
    print("\nMenampilkan Skenario 1: Kualitas Rendah...")
    simulate_digitization(freq=1.0, duration=2.0, sampling_rate=10, quantization_levels=4)
    
    # SKENARIO 2: Kualitas Tinggi (Lebih halus)
    # Uncomment baris di bawah ini jika ingin mencoba skenario 2 setelah menutup window pertama
    
    # print("\nMenampilkan Skenario 2: Kualitas Tinggi...")
    # simulate_digitization(freq=1.0, duration=2.0, sampling_rate=50, quantization_levels=32)