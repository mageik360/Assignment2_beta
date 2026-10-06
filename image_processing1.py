import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("intel-image-classification/seg_test/seg_test/buildings/24017.jpg")

if image is None:
    raise FileNotFoundError("Image could not be loaded")

dft = np.fft.fft2(image.astype(np.float64))

dft_shifted = np.fft.fftshift(dft)

magnitude = np.abs(dft_shifted)
magnitude_spectrum = np.log1p(magnitude)

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