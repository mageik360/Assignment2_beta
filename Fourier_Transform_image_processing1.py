import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

image = cv2.imread("intel-image-classification/seg_train/seg_train/buildings/66.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Image could not be loaded")

dft = np.fft.fft2(image.astype(np.float64))

dft_shifted = np.fft.fftshift(dft)

magnitude = np.abs(dft_shifted)
magnitude_spectrum = np.log1p(magnitude)

original = image.astype(np.float64)
dft = np.fft.fft2(original)

def compress(dft, percent):
    """Keep the largest percent of the DFT coefficients and invert"""
    count = max(1, int(dft.size * percent / 100))
    magnitude = np.abs(dft)
    threshold = np.partition(magnitude.ravel(), -count)[-count]
    kept = np.where(magnitude >= threshold, dft, 0)
    return np.real(np.fft.ifft2(kept))

def quality(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    psnr = np.inf if mse == 0 else 10 * np.log10(255**2 / mse)
    return mse, psnr



plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
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
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(filtered, cmap="gray", vmin=0, vmax=255)
plt.title("Low-pass filtered")
plt.axis("off")
plt.tight_layout()
plt.show()
# 3 high pass
high_pass = ~low_pass

high_dft = dft_shifted * high_pass
edges = np.real(np.fft.ifft2(np.fft.ifftshift(high_dft)))
alpha = 1.0
enhanced = image + alpha * edges

plt.figure(figsize=(12, 4))
edge_limit = np.max(np.abs(edges))
edge_norm = TwoSlopeNorm(vmin=-edge_limit, vcenter=0, vmax=edge_limit)

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap="gray", norm=edge_norm)
plt.title("High-pass (edges)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(enhanced, cmap="gray", vmin=0, vmax=255)
plt.title("Edge enhanced")
plt.axis("off")
plt.tight_layout()
plt.show()

# Low-pass comparison across radii
radii = [10, 20, 30]

plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

for i, radius in enumerate(radii):
    low_pass_r = (x - ccol) ** 2 + (y - crow) ** 2 <= radius ** 2
    low_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(dft_shifted * low_pass_r)))
    plt.subplot(1, 4, i + 2)
    plt.imshow(low_filtered, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Low-pass, r={radius}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# High-pass and edge enhancement across radii
high_pass_results = []
for radius in radii:
    high_pass_r = (x - ccol) ** 2 + (y - crow) ** 2 > radius ** 2
    edges_r = np.real(np.fft.ifft2(np.fft.ifftshift(dft_shifted * high_pass_r)))
    enhanced_r = image + edges_r
    high_pass_results.append((radius, edges_r, enhanced_r))

shared_edge_limit = max(np.max(np.abs(edges_r)) for _, edges_r, _ in high_pass_results)
shared_edge_norm = TwoSlopeNorm(vmin=-shared_edge_limit, vcenter=0, vmax=shared_edge_limit)

plt.figure(figsize=(9, 9))
for row, (radius, edges_r, enhanced_r) in enumerate(high_pass_results):
    panels = [
        (image, "Original", {"vmin": 0, "vmax": 255}),
        (edges_r, f"High-pass, r={radius}", {"norm": shared_edge_norm}),
        (enhanced_r, f"Edge enhanced, r={radius}", {"vmin": 0, "vmax": 255}),
    ]
    for col, (panel, title, image_args) in enumerate(panels):
        plt.subplot(len(radii), 3, row * 3 + col + 1)
        plt.imshow(panel, cmap="gray", **image_args)
        plt.title(title)
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

# Gaussian noise removed with the existing low-pass filter
rng = np.random.default_rng(0)
noisy = np.clip(original + rng.normal(0, 25, original.shape), 0, 255)
noisy_dft_shifted = np.fft.fftshift(np.fft.fft2(noisy))
denoised = np.real(np.fft.ifft2(np.fft.ifftshift(noisy_dft_shifted * low_pass)))

noisy_mse, _ = quality(original, noisy)
denoised_mse, _ = quality(original, denoised)
print(f"Noisy MSE = {noisy_mse:.2f}")
print(f"Low-pass filtered MSE = {denoised_mse:.2f}")

plt.figure(figsize=(12, 4))
for i, (panel, title) in enumerate(
    [
        (original, "Original"),
        (noisy, f"Noisy, MSE = {noisy_mse:.1f}"),
        (denoised, f"Low-pass filtered, MSE = {denoised_mse:.1f}"),
    ],
    start=1,
):
    plt.subplot(1, 3, i)
    plt.imshow(panel, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()