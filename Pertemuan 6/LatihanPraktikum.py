import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

# =================================================================
# FUNGSI PEMBANTU (UTILITY FUNCTIONS)
# =================================================================

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)

def calculate_ssim(img1, img2):
    """Simplified SSIM calculation"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

# =================================================================
# PRAKTIKUM 6.1: SIMULASI DEGRADASI CITRA
# =================================================================

def praktikum_6_1():
    print("\nPRAKTIKUM 6.1: SIMULASI DEGRADASI CITRA")
    print("=" * 50)
    
    def create_test_image():
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (100, 100), 200, -1)
        cv2.circle(img, (180, 80), 40, 150, -1)
        cv2.line(img, (50, 180), (200, 180), 100, 3)
        cv2.line(img, (180, 50), (180, 200), 100, 3)
        cv2.putText(img, 'TEST', (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 180, 2)
        return img

    def add_gaussian_noise(image, mean=0, sigma=25):
        noise = np.random.normal(mean, sigma, image.shape)
        return np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

    def add_salt_pepper_noise(image, prob=0.05):
        noisy = image.copy()
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = np.random.random()
                if rdn < prob:
                    noisy[i][j] = 0
                elif rdn > thres:
                    noisy[i][j] = 255
        return noisy

    def add_speckle_noise(image, sigma=0.1):
        noise = np.random.normal(1, sigma, image.shape)
        return np.clip(image.astype(float) * noise, 0, 255).astype(np.uint8)

    def add_motion_blur(image, length=15, angle=0):
        kernel = np.zeros((length, length))
        center = length // 2
        angle_rad = np.deg2rad(angle)
        x_start = int(center - (length/2) * np.cos(angle_rad))
        y_start = int(center - (length/2) * np.sin(angle_rad))
        x_end = int(center + (length/2) * np.cos(angle_rad))
        y_end = int(center + (length/2) * np.sin(angle_rad))
        cv2.line(kernel, (x_start, y_start), (x_end, y_end), 1, 1)
        kernel /= np.sum(kernel)
        blurred = cv2.filter2D(image.astype(float), -1, kernel)
        return np.clip(blurred, 0, 255).astype(np.uint8), kernel

    def add_out_of_focus_blur(image, radius=5):
        size = 2 * radius + 1
        kernel = np.zeros((size, size))
        cv2.circle(kernel, (radius, radius), radius, 1, -1)
        kernel /= np.sum(kernel)
        blurred = cv2.filter2D(image.astype(float), -1, kernel)
        return np.clip(blurred, 0, 255).astype(np.uint8), kernel

    clean_img = create_test_image()
    degradations = {
        'Clean Image': (clean_img, None),
        'Gaussian Noise': (add_gaussian_noise(clean_img), None),
        'Salt & Pepper': (add_salt_pepper_noise(clean_img), None),
        'Speckle Noise': (add_speckle_noise(clean_img), None),
        'Motion Blur': add_motion_blur(clean_img, 15, 30),
        'Out-of-Focus': add_out_of_focus_blur(clean_img, 5)
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for idx, (title, data) in enumerate(degradations.items()):
        img_show = data[0] if isinstance(data, tuple) else data
        axes[idx].imshow(img_show, cmap='gray')
        axes[idx].set_title(title)
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()

# =================================================================
# PRAKTIKUM 6.2: INVERSE DAN WIENER FILTER
# =================================================================

def praktikum_6_2():
    print("\nPRAKTIKUM 6.2: INVERSE FILTER VS WIENER FILTER")
    print("=" * 50)

    def get_psf_fft(image_shape, psf):
        psf_padded = np.zeros(image_shape)
        h, w = psf.shape
        psf_padded[:h, :w] = psf
        # Centering PSF
        psf_padded = np.roll(psf_padded, -(h//2), axis=0)
        psf_padded = np.roll(psf_padded, -(w//2), axis=1)
        return np.fft.fft2(psf_padded)

    def inverse_filter(degraded, psf, epsilon=1e-3):
        G = np.fft.fft2(degraded.astype(float))
        H = get_psf_fft(degraded.shape, psf)
        F_hat = G / (H + epsilon)
        return np.clip(np.abs(np.fft.ifft2(F_hat)), 0, 255).astype(np.uint8)

    def wiener_filter(degraded, psf, K=0.01):
        G = np.fft.fft2(degraded.astype(float))
        H = get_psf_fft(degraded.shape, psf)
        H_conj = np.conj(H)
        W = H_conj / (np.abs(H)**2 + K)
        F_hat = G * W
        return np.clip(np.abs(np.fft.ifft2(F_hat)), 0, 255).astype(np.uint8)

    # Setup
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    
    psf = cv2.getGaussianKernel(9, 2)
    psf = psf @ psf.T
    
    blurred = cv2.filter2D(img.astype(float), -1, psf)
    noise = np.random.normal(0, 5, blurred.shape)
    degraded = np.clip(blurred + noise, 0, 255).astype(np.uint8)

    res_inv = inverse_filter(degraded, psf, 0.01)
    res_wie = wiener_filter(degraded, psf, 0.01)

    titles = ['Original', 'Degraded', 'Inverse Filter', 'Wiener Filter']
    imgs = [img, degraded, res_inv, res_wie]
    
    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(f"{titles[i]}\nPSNR: {calculate_psnr(img, imgs[i]):.2f}")
        plt.axis('off')
    plt.show()

# =================================================================
# PRAKTIKUM 6.3: MOTION BLUR & RICHARDSON-LUCY
# =================================================================

def praktikum_6_3():
    print("\nPRAKTIKUM 6.3: MOTION BLUR ESTIMATION & DEBLURRING")
    print("=" * 50)

    def richardson_lucy(image, psf, iterations=30):
        image = image.astype(np.float32)
        psf = psf.astype(np.float32)
        estimate = np.full(image.shape, 0.5, dtype=np.float32)
        psf_flip = np.flip(psf)
        
        for i in range(iterations):
            blur = cv2.filter2D(estimate, -1, psf)
            blur = np.where(blur == 0, 1e-8, blur)
            ratio = image / blur
            correction = cv2.filter2D(ratio, -1, psf_flip)
            estimate *= correction
            estimate = np.clip(estimate, 0, 255)
        return estimate.astype(np.uint8)

    # Create Motion Blur
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(img, 'Zahran', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    
    length, angle = 20, 45
    psf = np.zeros((length, length))
    cv2.line(psf, (0, 0), (length-1, length-1), 1, 1)
    psf /= psf.sum()
    
    blurred = cv2.filter2D(img.astype(float), -1, psf)
    blurred_noisy = np.clip(blurred + np.random.normal(0, 2, blurred.shape), 0, 255).astype(np.uint8)
    
    restored_rl = richardson_lucy(blurred_noisy, psf)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(blurred_noisy, cmap='gray'); plt.title('Motion Blurred'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(restored_rl, cmap='gray'); plt.title('Richardson-Lucy'); plt.axis('off')
    plt.show()

# =================================================================
# MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    praktikum_6_1()
    praktikum_6_2()
    praktikum_6_3()
    print("\nSemua simulasi selesai dijalankan.")