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
  
def compute_covariance_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute the covariance matrix of the image data.
    """
    mean_centered_data = data - np.mean(data, axis=0)
    covariance_matrix = np.cov(mean_centered_data, rowvar=False)
    return covariance_matrix
  
def calculate_eigenvalues_and_eigenvectors(covariance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the eigenvalues and eigenvectors of the covariance matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # eigh: covariance is symmetric
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
    along with the principal components and the mean used for centering.
    """
    
    data_matrix = data
    mean = np.mean(data_matrix, axis=0)
    covariance_matrix = compute_covariance_matrix(data_matrix)
    eigenvalues, eigenvectors = calculate_eigenvalues_and_eigenvectors(covariance_matrix)
    sorted_eigenvalues, sorted_eigenvectors = sort_eigenvectors(eigenvalues, eigenvectors)
    principal_components = select_top_k_eigenvectors(sorted_eigenvectors, k)
    projected_data = project_onto_principal_components(data_matrix - mean, principal_components)
    return projected_data, principal_components, mean

IMAGE_COUNT = 500  # Number of images to load for PCA
K = 100  # Number of principal components to keep
IMAGE_SIZE = (64, 64)  # All images are resized to this so they share the same pixel columns
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

if __name__ == "__main__":
    images = ready_images()
    n_images, n_pixels = images.shape
    print("Data matrix shape:", images.shape)

    k = K
    projected_data, principal_components, mean = pca(data=images, k=k)
    reconstructed_images = reconstruct_from_principal_components(projected_data, principal_components, mean)
    print("Projected data shape:", projected_data.shape)
    print(f"MSE: {mean_squared_error(images, reconstructed_images):.5f}")
    print(f"Compression ratio: {compression_ratio(n_images, n_pixels, k):.2f}")

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
    plt.show()