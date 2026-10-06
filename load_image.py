from PIL import Image
import numpy as np

def load_image(file_path):
    """
    Load an image from a file, convert it to greyscale, and convert it to a numpy array.
    """
    image = Image.open(file_path).convert('L')  # Convert to greyscale
    return np.array(image)
  
def nomalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image pixel values to the range [0, 1].
    """
    return image / 255.0
