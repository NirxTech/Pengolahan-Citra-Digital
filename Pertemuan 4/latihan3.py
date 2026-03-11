import cv2
import numpy as np
import time

class RealTimeEnhancement:
    def __init__(self, target_fps=30, buffer_size=5):
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        
        # MEMORY CONSTRAINT: Hanya menyimpan tuple nilai min/max, BUKAN array gambar
        self.history_buffer = [] 
        
        # COMPUTATIONAL CONSTRAINT: Deklarasi objek CLAHE satu kali di memori
        # Menggunakan parameter tile 8x8 standar untuk keseimbangan speed/quality
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 
        
        self.prev_time = time.time()

    def enhance_frame(self, frame, enhancement_type='adaptive'):
        """
        Enhance single frame with real-time constraints
        """
        # 1. Optimasi Komputasi: Konversi ke HSV, proses HANYA kanal kecerahan (V)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        enhanced_v = v.copy()
        
        if enhancement_type == 'adaptive':
            # Ekstrak nilai ekstrem yang robust (mengabaikan noise/outlier)
            current_min = np.percentile(v, 2)
            current_max = np.percentile(v, 98)
            
            # Tambahkan ke buffer history
            self.history_buffer.append((current_min, current_max))
            
            # Batasi ukuran buffer (Memory Limit)
            if len(self.history_buffer) > self.buffer_size:
                self.history_buffer.pop(0)
            
            # 2. TEMPORAL CONSISTENCY: Hitung moving average untuk mencegah flickering
            smooth_min = np.mean([val[0] for val in self.history_buffer])
            smooth_max = np.mean([val[1] for val in self.history_buffer])
            
            # Terapkan Contrast Stretching menggunakan parameter yang sudah di-smooth
            if smooth_max > smooth_min:
                enhanced_v = ((v.astype(float) - smooth_min) / (smooth_max - smooth_min) * 255)
                enhanced_v = np.clip(enhanced_v, 0, 255).astype(np.uint8)
                
        elif enhancement_type == 'clahe':
            # CLAHE OpenCV sudah dioptimasi dalam C++ untuk real-time
            enhanced_v = self.clahe.apply(v)
            
        else:
            # Fallback: Ekualisasi global biasa (sangat berisiko flickering pada video)
            enhanced_v = cv2.equalizeHist(v)
            
        # 3. Gabungkan kembali kanal dan kembalikan ke format BGR
        hsv_enhanced = cv2.merge([h, s, enhanced_v])
        result_frame = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 4. Monitor Target FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        
        # Visualisasi Indikator FPS pada layar
        status_color = (0, 255, 0) if fps >= (self.target_fps * 0.8) else (0, 0, 255)
        cv2.putText(result_frame, f"FPS: {fps:.1f} | Mode: {enhancement_type}", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
        return result_frame

# ==========================================
# BLOK PENGUJIAN (SIMULASI WEBCAM/VIDEO)
# ==========================================
if __name__ == "__main__":
    print("Memulai simulasi real-time enhancement...")
    
    # Inisialisasi enhancer
    enhancer = RealTimeEnhancement(target_fps=30, buffer_size=10)
    
    # Buka webcam (0) atau file video ('video.mp4')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera tidak ditemukan. Menjalankan simulasi dengan noise array...")
        # (Blok ini hanya agar kode tidak error jika tidak ada webcam)
        for i in range(100):
            dummy_frame = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
            res = enhancer.enhance_frame(dummy_frame, enhancement_type='adaptive')
            cv2.imshow('Real-time Enhancement', res)
            if cv2.waitKey(30) & 0xFF == ord('q'): break
    else:
        print("Tekan 'q' untuk keluar, 'a' untuk Adaptive, 'c' untuk CLAHE")
        mode = 'adaptive'
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Ganti mode saat tombol ditekan
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('a'): mode = 'adaptive'
            elif key == ord('c'): mode = 'clahe'
            
            # Proses frame
            enhanced_frame = enhancer.enhance_frame(frame, enhancement_type=mode)
            
            # Tampilkan gambar asli berdampingan dengan hasil
            combined = np.hstack((frame, enhanced_frame))
            cv2.imshow('Kiri: Asli | Kanan: Enhanced', combined)

    cap.release()
    cv2.destroyAllWindows()