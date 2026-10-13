#1. Write a Python script to compute the HOG features of a given image using a library such as
#OpenCV or scikit-image.
import os

import skimage
import matplotlib.pyplot as plt
from load_image import load_image_set
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def save_figure(fig, filename: str):
    """Save a figure into the plots directory."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {path}")
    plt.close(fig)


def compute_gradients(image: np.ndarray):
    """Compute gradient magnitude and direction using Sobel operators."""
    gx = skimage.filters.sobel_v(image)  # gradient in x-direction
    gy = skimage.filters.sobel_h(image)  # gradient in y-direction
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    return magnitude, direction


# ---------------------------------------------------------------------------
# HOG parameter comparison
#
# Note: skimage's built-in HOG visualization (visualize=True) is drawn from the
# raw per-cell orientation histograms *before* block normalization, so the
# block size has no effect on it at all. The functions below therefore
# re-implement the cell-histogram and block-normalization stages (matching
# skimage's math) and render the *block-normalized* histograms ourselves.
# ---------------------------------------------------------------------------


def _image_gradients(image: np.ndarray):
    """Gradient images along the row and column axes using centered
    differences, exactly like skimage's HOG implementation."""
    channel = image.astype(float)
    g_row = np.zeros_like(channel)
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
    g_col = np.zeros_like(channel)
    g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
    return g_row, g_col


def _cell_orientation_histograms(image: np.ndarray, orientations: int, pixels_per_cell):
    """Per-cell gradient orientation histograms: every pixel votes its
    gradient magnitude into one of `orientations` unsigned orientation bins
    over [0, 180) degrees, like skimage's HOG."""
    g_row, g_col = _image_gradients(image)
    magnitude = np.hypot(g_row, g_col)
    angle = np.rad2deg(np.arctan2(g_row, g_col)) % 180
    bin_width = 180.0 / orientations
    bins = np.minimum((angle / bin_width).astype(int), orientations - 1)

    height, width = image.shape[:2]
    c_row, c_col = pixels_per_cell
    n_cells_row = height // c_row
    n_cells_col = width // c_col
    cell_hists = np.zeros((n_cells_row, n_cells_col, orientations))
    for b in range(orientations):
        votes = np.where(bins == b, magnitude, 0.0)
        cell_hists[:, :, b] = (
            votes[: n_cells_row * c_row, : n_cells_col * c_col]
            .reshape(n_cells_row, c_row, n_cells_col, c_col)
            .sum(axis=(1, 3))
        )
    return cell_hists


def _l2_hys_normalize(block: np.ndarray) -> np.ndarray:
    """L2-Hys block normalization (L2 norm, clip at 0.2, renormalize by L2),
    matching skimage's default `block_norm`."""
    out = block / np.sqrt(np.sum(block**2) + 1e-5**2)
    out = np.minimum(out, 0.2)
    return out / np.sqrt(np.sum(out**2) + 1e-5**2)


def _block_normalize(cell_hists: np.ndarray, cells_per_block):
    """Average, for every cell, the L2-Hys normalized histograms of all blocks
    that contain it. This is what makes the block size visible: a 1x1 block
    normalizes each cell on its own (boosting weak cells), while a large block
    pools contrast over many cells (only strong edges stay bright)."""
    n_cells_row, n_cells_col, n_bins = cell_hists.shape
    b_row = min(cells_per_block[0], n_cells_row)
    b_col = min(cells_per_block[1], n_cells_col)
    accumulated = np.zeros_like(cell_hists)
    count = np.zeros((n_cells_row, n_cells_col))
    for r in range(n_cells_row - b_row + 1):
        for c in range(n_cells_col - b_col + 1):
            block = _l2_hys_normalize(cell_hists[r : r + b_row, c : c + b_col, :].ravel())
            accumulated[r : r + b_row, c : c + b_col, :] += block.reshape(b_row, b_col, n_bins)
            count[r : r + b_row, c : c + b_col] += 1
    return accumulated / count[..., None]


def _render_hog_canvas(cell_hists: np.ndarray, image_shape, pixels_per_cell, scale=4):
    """Render cell histograms as oriented line segments on a pixel canvas,
    using the same drawing convention as skimage's HOG visualization, but at a
    higher internal resolution (`scale` canvas pixels per image pixel) so the
    zoomed comparison stays crisp."""
    from skimage import draw

    height, width = image_shape[:2]
    c_row, c_col = pixels_per_cell
    c_row, c_col = c_row * scale, c_col * scale
    n_cells_row, n_cells_col, orientations = cell_hists.shape
    radius = max(min(c_row, c_col) // 2 - 1, 3)
    midpoints = np.pi * (np.arange(orientations) + 0.5) / orientations
    dr = radius * np.sin(midpoints)
    dc = radius * np.cos(midpoints)

    canvas = np.zeros((height * scale, width * scale))
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            centre_row = r * c_row + c_row // 2
            centre_col = c * c_col + c_col // 2
            for o in range(orientations):
                value = cell_hists[r, c, o]
                if value <= 0:
                    continue
                rr, cc = draw.line(
                    int(centre_row - dc[o]), int(centre_col + dr[o]),
                    int(centre_row + dc[o]), int(centre_col - dr[o]),
                )
                rr = np.clip(rr, 0, canvas.shape[0] - 1)
                cc = np.clip(cc, 0, canvas.shape[1] - 1)
                canvas[rr, cc] += value
    if canvas.max() > 0:
        canvas /= canvas.max()
    return canvas


def _find_crop_region(magnitude: np.ndarray, crop_size: int, stride=6):
    """Locate the crop_size x crop_size window with the strongest gradients,
    so that the zoomed comparison shows a region with plenty of HOG
    structure."""
    best_top, best_left, best_score = 0, 0, -np.inf
    for top in range(0, magnitude.shape[0] - crop_size + 1, stride):
        for left in range(0, magnitude.shape[1] - crop_size + 1, stride):
            score = magnitude[top : top + crop_size, left : left + crop_size].sum()
            if score > best_score:
                best_score, best_top, best_left = score, top, left
    return best_top, best_left


def plot_hog_with_varying_parameters(image: np.ndarray, name: str):
    """Plot large HOG visualizations for one image while varying cell size,
    block size and the number of orientation bins. One parameter is varied per
    row, and every panel zooms into the same edge-rich region of the image so
    that fine differences (segment angles, local contrast) are easy to see.

    Unlike skimage's built-in visualization, the segments here are drawn from
    *block-normalized* cell histograms, so the effect of the block size is
    actually visible (skimage's `visualize=True` ignores block normalization).
    """
    scale = 4  # canvas pixels per image pixel
    default_orientations = 9
    default_cell = (8, 8)
    default_block = (2, 2)

    # one row per parameter: (row label, [(panel title, orientations, pixels_per_cell, cells_per_block), ...])
    parameter_rows = [
        ("Cell size\n(pixels per cell)", [
            ("Cell 4x4 px", default_orientations, (4, 4), default_block),
            ("Cell 8x8 px", default_orientations, (8, 8), default_block),
            ("Cell 16x16 px", default_orientations, (16, 16), default_block),
        ]),
        ("Number of\norientation bins", [
            ("4 bins", 4, default_cell, default_block),
            ("9 bins", 9, default_cell, default_block),
            ("20 bins", 20, default_cell, default_block),
        ]),
        ("Block size\n(cells per block)", [
            ("Block 1x1 cells", default_orientations, default_cell, (1, 1)),
            ("Block 3x3 cells", default_orientations, default_cell, (3, 3)),
            ("Block 9x9 cells", default_orientations, default_cell, (9, 9)),
        ]),
    ]

    # Pick one edge-rich region to zoom into, shared by all panels.
    g_row, g_col = _image_gradients(image)
    magnitude = np.hypot(g_row, g_col)
    crop_size = min(72, image.shape[0], image.shape[1])
    crop_top, crop_left = _find_crop_region(magnitude, crop_size)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.85],
                          left=0.05, right=0.99, top=0.92, bottom=0.02,
                          wspace=0.12, hspace=0.25)

    for row_index, (row_label, settings) in enumerate(parameter_rows):
        for col_index, (title, orientations, pixels_per_cell, cells_per_block) in enumerate(settings):
            cell_hists = _cell_orientation_histograms(image, orientations, pixels_per_cell)
            canvas = _render_hog_canvas(
                _block_normalize(cell_hists, cells_per_block), image.shape, pixels_per_cell, scale
            )
            hog = skimage.feature.hog(
                image,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
            )
            ax = fig.add_subplot(gs[row_index, col_index])
            crop = canvas[crop_top * scale : (crop_top + crop_size) * scale,
                          crop_left * scale : (crop_left + crop_size) * scale]
            ax.imshow(crop, cmap="gray", interpolation="nearest")
            # faint cell grid to make the cell size easy to read off
            cell_px = pixels_per_cell[0] * scale
            for x in range(0, crop.shape[1] + 1, cell_px):
                ax.axvline(x - 0.5, color="gray", alpha=0.12, lw=0.5)
            for y in range(0, crop.shape[0] + 1, cell_px):
                ax.axhline(y - 0.5, color="gray", alpha=0.12, lw=0.5)
            ax.set_title(f"{title}\n(HOG descriptor: {hog.size:,} values)", fontsize=11)
            ax.axis("off")

    # Reference column: original image and whole-image HOG with the zoom region marked.
    ref_gs = gs[:, 3].subgridspec(2, 1, hspace=0.15)
    ax_original = fig.add_subplot(ref_gs[0])
    ax_original.imshow(image, cmap="gray")
    ax_original.add_patch(plt.Rectangle((crop_left, crop_top), crop_size, crop_size,
                                        edgecolor="red", facecolor="none", lw=2))
    ax_original.set_title("Original image", fontsize=11)
    ax_original.axis("off")

    default_hists = _block_normalize(
        _cell_orientation_histograms(image, default_orientations, default_cell), default_block
    )
    default_canvas = _render_hog_canvas(default_hists, image.shape, default_cell, scale)
    ax_default = fig.add_subplot(ref_gs[1])
    ax_default.imshow(default_canvas, cmap="gray")
    ax_default.add_patch(plt.Rectangle((crop_left * scale, crop_top * scale),
                                       crop_size * scale, crop_size * scale,
                                       edgecolor="red", facecolor="none", lw=2))
    ax_default.set_title("Whole-image HOG, default parameters\n(9 bins, 8x8 px cells, 2x2 block)", fontsize=11)
    ax_default.axis("off")

    fig.suptitle(
        f"Effect of HOG parameters",
        fontsize=15,
    )
    save_figure(fig, f"hog_parameters_{name}.png")


def extractHOG():
    categories = ["forest", "buildings", "street"]
    size = (150, 150)
    image_sets = [load_image_set(f"intel-image-classification/seg_train/seg_train/{category}/", max_images=1, size=size, random=True)
                  for category in categories]
    images = np.concat(image_sets)
    for name, image in zip(categories, images):
        magnitude, direction = compute_gradients(image)
        hog, hog_image = skimage.feature.hog(image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),visualize=True)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(magnitude, cmap="magma")
        axes[1].set_title("Gradient Magnitude Image")
        axes[1].axis("off")
        axes[2].imshow(hog_image, cmap="gray")
        axes[2].set_title("HOG Feature Image")
        axes[2].axis("off")
        axes[3].hist(hog, bins=50, color="steelblue", edgecolor="black")
        axes[3].set_title("Histogram of HOG Features")
        axes[3].set_xlabel("Feature Value")
        axes[3].set_ylabel("Frequency")
        plt.tight_layout()
        save_figure(fig, f"hog_overview_{name}.png")
        plot_hog_with_varying_parameters(image, name)

if __name__ == "__main__":
    extractHOG()

#2. Apply your implementation to at least three different images, including both simple and
#complex scenes.


#3. Visualize the original image, the gradient image, and the HOG feature image.


#4. Compare the HOG features extracted from different images. Discuss the impact of varying
#parameters like cell size, block size, and the number of bins on the resulting HOG
#descriptors.

