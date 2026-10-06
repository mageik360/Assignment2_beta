from PIL import Image
import numpy as np
import os

def load_image(file_path):
    """
    Load an image from a file, convert it to greyscale, and convert it to a numpy array.
    """
    image = Image.open(file_path).convert('L')  # Convert to greyscale
    return np.array(image)
  
def load_image_set(directory_path, max_images, size=(64, 64)):
    """
    Load a set of images from a directory, convert them to greyscale, resize them to a common size,
    and stack them into a numpy array of shape (n_images, height, width).
    """
    images = []
    for filename in sorted(os.listdir(directory_path)):
        if len(images) >= max_images:
            break
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = resize_image(load_image(image_path), size)
            images.append(image)
    return np.array(images)
  
def nomalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image pixel values to the range [0, 1].
    """
    return image / 255.0
  
def flatten_image(image: np.ndarray) -> np.ndarray:
    """
    Flatten the image into a 1D array.
    """
    return image.flatten()

def resize_image(image: np.ndarray, new_size: tuple) -> np.ndarray:
    """
    Resize the image to a new size.
    """
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize(new_size, Image.LANCZOS)
    return np.array(resized_image)