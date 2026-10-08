import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("intel-image-classification/seg_train/seg_train/buildings/66.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Image could not be loaded")

dft = np.fft.fft2(image.astype(np.float64))

dft_shifted = np.fft.fftshift(dft)

magnitude = np.abs(dft_shifted)
magnitude_spectrum = np.log1p(magnitude)

def compress(dft, percent):
    """Keep the largest percent of the DFT coefficients and invert"""
    count = max(1, int(dft.size * percent / 100))
    magnitude = np.abs(dft)
    threshold = np.partition(magnitude.ravel(), -count)[-count]
    kept = np.where(magnitude >= threshold, dft, 0)
    return np.real(np.fft.ifft2(kept))

def quality(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    psnr = np.inf if mse == 0 else 10 * log10(255**2 / mse)
    return mse, psnr



plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("DFT magnitude (log)")
plt.axis("off")
plt.tight_layout()
plt.show()

# 2 low pass Filtering

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
radius = 30

y, x = np.ogrid[:rows, :cols]
low_pass = (x - ccol)**2 + (y - crow)**2 <= radius ** 2

filtered_dft = dft_shifted * low_pass
filtered = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_dft)))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(filtered, cmap="gray")
plt.title("Low-pass filtered")
plt.axis("off")
plt.tight_layout()
plt.show()
# 3 high pass
high_pass = ~low_pass

high_dft = dft_shifted * high_pass
edges = np.real(np.fft.ifft2(np.fft.ifftshift(high_dft)))

enhanced = image + edges

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap="gray")
plt.title("High-pass (edges)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(enhanced, cmap="gray")
plt.title("Edge enhanced")
plt.axis("off")
plt.tight_layout()
plt.show()

percentages = [1, 5, 10, 25, 50]
reconstructions = []

for percent in percentages:
    reconstructed = compress(dft, percent)
    mse, psnr = quality(original, reconstructed)
    reconstructions.append(reconstructed)
    print(f"{percent:3d}% MSE = {mse:8.2f} PSNR = {psnr:5.2f} dB")

plt.figure(figsize=(12, 4))
for i, (percent, reconstructed) in enumerate(zip(percentages, reconstructions), start=1):
    plt.subplot(1, len(percentages), i)
    plt.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
    plt.title(f"{percent}%")
    plt.axis("off")

plt.tight_layout()
plt.show()