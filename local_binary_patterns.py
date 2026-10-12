"""
Local binary patterns (LBP) for describing image texture.

The basic 8-neighbor LBP replaces each interior pixel with an 8-bit code.
Each bit records whether one of the eight surrounding pixels is at least as
bright as the center. The histogram of those codes counts how often each
local pattern occurs, so it describes the texture without recording where
the patterns sit in the image.

The Intel scene dataset used by the rest of this assignment has no face
image. The comparison below uses five photos: a forest path, a glacier
face, a glass tower, a street of shuttered buildings, and a breaking wave.
"""

import numpy as np
import matplotlib.pyplot as plt

import load_image


# Clockwise from the top-left. Bit 0 is the top-left neighbor and bit 7 is
# the left neighbor, so the code is sum_p s(neighbor_p - center) * 2^p.
NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
)

IMAGES = (
    (
        "Forest path",
        "intel-image-classification/seg_test/seg_test/forest/20056.jpg",
    ),
    (
        "Skyscraper",
        "intel-image-classification/seg_test/seg_test/buildings/20057.jpg",
    ),
    (
        "Glacier face",
        "intel-image-classification/seg_test/seg_test/glacier/20059.jpg",
    ),
    (
        "Street",
        "intel-image-classification/seg_test/seg_test/street/20066.jpg",
    ),
    (
        "Breaking wave",
        "intel-image-classification/seg_test/seg_test/sea/20072.jpg",
    ),
)


def compute_lbp(image):
    """
    Compute the basic 8-neighbor LBP of a grayscale image.

    A neighbor contributes a 1 when it is greater than or equal to the
    center pixel, which is the original LBP threshold. Equal neighbors
    therefore count as 1, and a perfectly flat patch is coded as 255.

    Pixels on the one-pixel border have no complete neighborhood, so they
    stay 0. The histogram helper ignores that border.

    Parameters
    ----------
    image : array of shape (height, width)
        One intensity value per pixel.

    Returns
    -------
    numpy.ndarray of uint8, same shape as image
        Each interior pixel is its LBP code in 0..255.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("LBP expects a grayscale image with shape (height, width)")
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError("LBP needs an image of at least 3 by 3 pixels")

    # int16 keeps the comparison well defined for both uint8 and float images.
    pixels = image.astype(np.int16)
    height, width = pixels.shape
    center = pixels[1:height - 1, 1:width - 1]

    code = np.zeros(center.shape, dtype=np.uint8)
    for bit, (row_offset, col_offset) in enumerate(NEIGHBOR_OFFSETS):
        neighbor = pixels[
            1 + row_offset:height - 1 + row_offset,
            1 + col_offset:width - 1 + col_offset,
        ]
        code |= (neighbor >= center).astype(np.uint8) << bit

    lbp_image = np.zeros((height, width), dtype=np.uint8)
    lbp_image[1:height - 1, 1:width - 1] = code
    return lbp_image


def lbp_histogram(lbp_image, exclude_border=True):
    """
    Count how often each LBP code occurs.

    There are 256 possible 8-bit codes, so the histogram has one bin per
    code. Bin 0 holds the number of pixels whose neighbors are all darker
    (a strict local bright spot). Bin 255 holds pixels whose neighbors are
    all at least as bright, including perfectly flat patches.

    Parameters
    ----------
    lbp_image : array of shape (height, width)
        Output of compute_lbp.
    exclude_border : bool
        Drop the one-pixel border. Those pixels were never assigned a real
        code, so including them would add a false spike at bin 0.

    Returns
    -------
    numpy.ndarray of shape (256,)
        Raw counts. Divide by the sum to compare images of different sizes.
    """
    lbp_image = np.asarray(lbp_image)
    region = lbp_image[1:-1, 1:-1] if exclude_border else lbp_image
    counts, _ = np.histogram(region, bins=256, range=(0, 256))
    return counts


def histogram_entropy(counts):
    """Shannon entropy of a histogram, in bits. Higher means more varied texture."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probability = counts[counts > 0] / total
    return float(-(probability * np.log2(probability)).sum())


def plot_lbp_results(samples):
    """
    Show each grayscale image, its LBP image, and its LBP histogram.

    The LBP image is a map of pattern codes, not a brightness image. A
    brighter LBP pixel means a larger code, which depends on the neighbor
    order rather than on how strong an edge is.
    """
    figure, axes = plt.subplots(len(samples), 3, figsize=(12, 3.2 * len(samples)))
    for row, (name, image, lbp_image, counts) in enumerate(samples):
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(name)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(lbp_image, cmap="gray", vmin=0, vmax=255)
        if row == 0:
            axes[row, 1].set_title("LBP codes")
        axes[row, 1].axis("off")

        axes[row, 2].plot(np.arange(256), counts, color="black", linewidth=0.8)
        if row == 0:
            axes[row, 2].set_title("LBP histogram")
        axes[row, 2].set_xlim(0, 255)
        #axes[row, 2].set_xlabel("LBP code")
        axes[row, 2].set_ylabel("Count")

    figure.tight_layout()
    return figure


def plot_histogram_comparison(samples):
    """Overlay the histograms as fractions of pixels, so the shapes can be compared."""
    figure, axis = plt.subplots(figsize=(10, 4))
    for name, _image, _lbp_image, counts in samples:
        fraction = counts / counts.sum()
        axis.plot(np.arange(256), fraction, label=name, linewidth=1.0)
    axis.set_xlim(0, 255)
    axis.set_xlabel("LBP code")
    axis.set_ylabel("Fraction of interior pixels")
    axis.set_title("LBP histogram comparison")
    axis.legend()
    figure.tight_layout()
    return figure


def describe_histograms(samples):
    """
    Print what the histograms say about these five textures.

    The numbers are measured from the images. A higher entropy, and a
    larger number of bins needed to cover the pixels, means the local
    patterns are more varied.
    """
    print(
        "Each histogram bin is one of the 256 local patterns. The height of a "
        "bin is how many pixels have that pattern. Code 0 is a pixel brighter "
        "than all eight neighbors. Code 255 is a pixel whose neighbors are all "
        "at least as bright, including flat areas. A peaked histogram means "
        "the same local patterns repeat. A spread histogram means the "
        "neighborhoods keep changing."
    )
    print()

    for name, _image, _lbp_image, counts in samples:
        total = int(counts.sum())
        order = np.argsort(counts)[::-1]
        covered = np.cumsum(counts[order]) / total
        bins_for_half = int(np.searchsorted(covered, 0.5) + 1)
        print(
            f"{name}: entropy {histogram_entropy(counts):.2f} bits, "
            f"code 255 = {counts[255] / total:.1%}, "
            f"code 0 = {counts[0] / total:.1%}, "
            f"{bins_for_half} codes cover half of the pixels"
        )

    print()


def main():
    samples = []
    for name, path in IMAGES:
        image = load_image.load_image(path)
        lbp_image = compute_lbp(image)
        counts = lbp_histogram(lbp_image)
        samples.append((name, image, lbp_image, counts))

    describe_histograms(samples)
    plot_lbp_results(samples)
    plot_histogram_comparison(samples)
    plt.show()


if __name__ == "__main__":
    main()
