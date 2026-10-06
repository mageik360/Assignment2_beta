import numpy as np
import load_image as li
from matplotlib import pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

"""
  
1. Convert the image into a 2D matrix where each row represents an image and each column
represents a pixel value.
2. Compute the covariance matrix of the image data.
3. Calculate the eigenvalues and eigenvectors of the covariance matrix.
4. Sort the eigenvectors based on the eigenvalues in descending order.
5. Select the top k eigenvectors to form the principal components.
6. Project the original images onto the lower-dimensional subspace defined by the selected
principal components.
  
"""

def image_to_matrix(image: np.ndarray) -> np.ndarray:
    """
    Convert the image into a 2D matrix where each row represents an image and each column
    represents a pixel value.
    """
    return image.reshape(image.shape[0], -1)
  
def compute_covariance_matrix(centered_data: np.ndarray) -> np.ndarray:
    """
    Compute the covariance matrix of the image data in its small (Gram) form.

    The pixel covariance C = Xc^T Xc / (N-1) is d x d (22500 x 22500 for 150x150 images), which is
    too large to decompose. With fewer images than pixels we instead use G = Xc Xc^T / (N-1), which
    is only N x N and has the same non-zero eigenvalues as C (the "Gram trick" from Eigenfaces).
    """
    n_images = centered_data.shape[0]
    return np.dot(centered_data, centered_data.T) / (n_images - 1)

def calculate_eigenvalues_and_eigenvectors(gram_matrix: np.ndarray, centered_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the eigenvalues and eigenvectors of the pixel covariance matrix via the Gram matrix.

    If G u = lambda u, then C (Xc^T u) = lambda (Xc^T u), so Xc^T u (normalized to unit length) is an
    eigenvector of the pixel covariance with the same eigenvalue.
    """
    eigenvalues, gram_eigenvectors = np.linalg.eigh(gram_matrix)  # eigh: the matrix is symmetric
    # Centering leaves at most N-1 non-zero eigenvalues; the zero ones have no pixel-space eigenvector
    nonzero = eigenvalues > 1e-10 * eigenvalues.max()
    eigenvalues, gram_eigenvectors = eigenvalues[nonzero], gram_eigenvectors[:, nonzero]
    eigenvectors = np.dot(centered_data.T, gram_eigenvectors)
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
    return eigenvalues, eigenvectors
  
def sort_eigenvectors(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort the eigenvectors based on the eigenvalues in descending order.
    """
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors
  
def select_top_k_eigenvectors(eigenvectors: np.ndarray, k: int) -> np.ndarray:
    """
    Select the top k eigenvectors to form the principal components.
    """
    return eigenvectors[:, :k]
  
def project_onto_principal_components(data: np.ndarray, principal_components: np.ndarray) -> np.ndarray:
    """
    Project the original images onto the lower-dimensional subspace defined by the selected
    principal components.
    """
    return np.dot(data, principal_components)

def reconstruct_from_principal_components(projected_data: np.ndarray, principal_components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Map the projected data back to the original pixel space.
    """
    return np.dot(projected_data, principal_components.T) + mean

def pca(data: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA on the data and return the projected data onto the top k principal components,
    along with the principal components, the mean used for centering, and all sorted eigenvalues
    (the variance along each principal component).
    """
    
    data_matrix = data
    mean = np.mean(data_matrix, axis=0)
    centered_data = data_matrix - mean
    covariance_matrix = compute_covariance_matrix(centered_data)
    eigenvalues, eigenvectors = calculate_eigenvalues_and_eigenvectors(covariance_matrix, centered_data)
    if k > eigenvectors.shape[1]:
        raise ValueError(f"k={k} is larger than the {eigenvectors.shape[1]} non-zero components available; use more images or a smaller k")
    sorted_eigenvalues, sorted_eigenvectors = sort_eigenvectors(eigenvalues, eigenvectors)
    principal_components = select_top_k_eigenvectors(sorted_eigenvectors, k)
    projected_data = project_onto_principal_components(centered_data, principal_components)
    return projected_data, principal_components, mean, sorted_eigenvalues

IMAGE_COUNT = 500  # Number of images to load for PCA
K = 100  # Number of principal components to keep
IMAGE_SIZE = (150, 150)  # Native size of the dataset; odd-sized images are resized to match
N_SHOWN = 5  # Number of images to display in the comparison

def ready_images() -> np.ndarray:
    """
    Load and prepare images for PCA. Returns a matrix of shape (n_images, n_pixels).
    """
    image_dir_path = os.getenv("PATH_TEST_IMAGE_DIR")
    images = li.load_image_set(image_dir_path, IMAGE_COUNT, IMAGE_SIZE)
    normalized_images = li.nomalize_image(images)
    return image_to_matrix(normalized_images)

def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Mean squared error between the original and reconstructed images.
    """
    return float(np.mean((original - reconstructed) ** 2))

def compression_ratio(n_images: int, n_pixels: int, k: int) -> float:
    """
    Original values divided by stored values. The principal components (n_pixels x k) and the mean
    (n_pixels) are needed to reconstruct, so they count towards the compressed size.
    """
    original_size = n_images * n_pixels
    compressed_size = n_images * k + n_pixels * k + n_pixels
    return original_size / compressed_size

def components_for_variance(eigenvalues: np.ndarray, threshold: float) -> int:
    """
    Smallest number of principal components whose cumulative explained variance reaches the threshold.
    """
    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    return int(np.searchsorted(cumulative, threshold) + 1)

def plot_explained_variance(eigenvalues: np.ndarray, k: int, threshold: float = 0.95):
    """
    Plot the variance explained by each principal component (top) and the cumulative explained
    variance (bottom), marking the chosen k and the number of components needed for the threshold.
    """
    explained = eigenvalues / np.sum(eigenvalues) * 100
    cumulative = np.cumsum(explained)
    components = np.arange(1, len(eigenvalues) + 1)
    k_threshold = components_for_variance(eigenvalues, threshold)

    series_color, text_color, muted_color, grid_color = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    fig, (ax_each, ax_cum) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_each.plot(components, explained, color=series_color, linewidth=2)
    ax_each.set_title("Variance explained by each principal component", color=text_color, loc="left")
    ax_each.set_ylabel("Explained variance (%)", color=muted_color)

    ax_cum.plot(components, cumulative, color=series_color, linewidth=2)
    ax_cum.set_title("Cumulative variance explained", color=text_color, loc="left")
    ax_cum.set_ylabel("Cumulative explained variance (%)", color=muted_color)
    ax_cum.set_xlabel("Number of principal components (k)", color=muted_color)
    ax_cum.set_ylim(0, 100)
    ax_cum.axhline(threshold * 100, color=muted_color, linewidth=1, linestyle="--")
    ax_cum.annotate(f"{threshold:.0%} reached at k={k_threshold}", (k_threshold, threshold * 100),
                    xytext=(8, -16), textcoords="offset points", color=text_color)
    ax_cum.plot(k, cumulative[k - 1], "o", color=series_color, markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax_cum.annotate(f"chosen k={k}: {cumulative[k - 1]:.1f}%", (k, cumulative[k - 1]),
                    xytext=(8, -20), textcoords="offset points", color=text_color)

    for ax in (ax_each, ax_cum):
        ax.grid(True, color=grid_color, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(grid_color)
        ax.tick_params(colors=muted_color)
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    images = ready_images()
    n_images, n_pixels = images.shape
    print("Data matrix shape:", images.shape)

    k = K
    projected_data, principal_components, mean, eigenvalues = pca(data=images, k=k)
    reconstructed_images = reconstruct_from_principal_components(projected_data, principal_components, mean)
    print("Projected data shape:", projected_data.shape)
    print(f"MSE: {mean_squared_error(images, reconstructed_images):.5f}")
    print(f"Compression ratio: {compression_ratio(n_images, n_pixels, k):.2f}")
    print(f"Variance explained by k={k}: {np.sum(eigenvalues[:k]) / np.sum(eigenvalues):.1%}")
    for threshold in (0.90, 0.95, 0.99):
        print(f"Components needed for {threshold:.0%} variance: {components_for_variance(eigenvalues, threshold)}")

    # Show the original and reconstructed images for comparison, on the same [0, 1] scale
    plt.figure(figsize=(2 * N_SHOWN, 4))
    for i in range(N_SHOWN):
        plt.subplot(2, N_SHOWN, i + 1)
        plt.imshow(images[i].reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray', vmin=0, vmax=1)
        plt.title("Original")
        plt.axis('off')
        plt.subplot(2, N_SHOWN, N_SHOWN + i + 1)
        plt.imshow(np.clip(reconstructed_images[i], 0, 1).reshape(IMAGE_SIZE[1], IMAGE_SIZE[0]), cmap='gray', vmin=0, vmax=1)
        plt.title(f"k={k}")
        plt.axis('off')
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"PCA_N{n_images}_K{k}.png")
    plt.savefig(out_path, dpi=150)
    print("Saved plot to", out_path)

    variance_figure = plot_explained_variance(eigenvalues, k)
    variance_path = os.path.join(out_dir, f"PCA_variance_N{n_images}_K{k}.png")
    variance_figure.savefig(variance_path, dpi=150)
    print("Saved plot to", variance_path)
    plt.show()