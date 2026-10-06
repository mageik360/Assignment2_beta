from PIL import Image
import numpy as np

def load_image(file_path):
    """
    Load an image from a file, convert it to greyscale, and convert it to a numpy array.
    """
    image = Image.open(file_path).convert('L')  # Convert to greyscale
    return np.array(image)
