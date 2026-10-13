import load_image
import cv2
import numpy as np
from pathlib import Path
from blob_detection import display_images_side_by_side, save_images_side_by_side
import matplotlib.pyplot as plt

def detect_contours(image, threshold=None, min_area=20, max_area=10000, blur_size=0,
                    mode=cv2.RETR_TREE, invert=False, method="otsu"):
    if invert:
        image = cv2.bitwise_not(image)
    blur_size = int(blur_size)
    if blur_size:
        if blur_size % 2 == 0:
            raise ValueError(f"blur_size must be odd, got {blur_size}")
        image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

    if threshold is not None:
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    elif method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "triangle":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 5)
    elif method == "canny":
        binary = cv2.Canny(image, 50, 150)
    else:
        raise ValueError(f"Unknown method: {method}")

    contours, _ = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours
            if cv2.contourArea(c) >= min_area
            and (max_area is None or cv2.contourArea(c) <= max_area)]

def distinct_colors(n):
    """
    Generate n visually distinct colors in BGR format.
    """
    if n == 0:
        return []
    hues = np.linspace(0, 180, n, endpoint=False)
    hsv = np.stack([hues, np.full(n, 255), np.full(n, 255)], axis=1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2BGR).reshape(-1, 3)
    return [tuple(int(v) for v in c) for c in bgr]
def draw_contours(image, contours, thickness=1):
    """
    Draw each contour outline in its own colour on a colour copy of the greyscale image.
    """
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for contour, color in zip(contours, distinct_colors(len(contours))):
        cv2.drawContours(output, [contour], -1, color, thickness)
    return output

def contour_detect_n_images(directory: str, n=10, threshold=None, min_area=20, blur_size=0, invert=False):
    """
    Run contour detection on the first n images (sorted by filename) in the directory.
    Returns a list of (path, image, contours, drawn_image) tuples.
    """
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:n]
    results = []
    for path in paths:
        image = load_image.load_image(str(path))
        contours = detect_contours(image, threshold=threshold, min_area=min_area, blur_size=blur_size, invert=invert)
        drawn_image = draw_contours(image, contours)
        results.append((path, image, contours, drawn_image))
    return results

def contour_stats(contours):
    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]
    return {
        "count": len(contours),
        "total_area": sum(areas),
        "mean_area": np.mean(areas) if areas else 0,
        "max_area": max(areas, default=0),
        "mean_perimeter": np.mean(perimeters) if perimeters else 0,
        "max_perimeter": max(perimeters, default=0),
        "min_area": min(areas, default=0),
        "min_perimeter": min(perimeters, default=0)
    }

def print_contour_stats(contours):
    stats = contour_stats(contours)
    print(f"Contours: {stats['count']}, Total Area: {stats['total_area']:.2f}, "
          f"Mean Area: {stats['mean_area']:.2f}, Max Area: {stats['max_area']:.2f}, "
          f"Mean Perimeter: {stats['mean_perimeter']:.2f}, Max Perimeter: {stats['max_perimeter']:.2f}")

def sweep_params(path, param_name, values, **fixed_params):
    """
    Run contour detection on one image once for each value in values for a
    single parameter. Extra keyword arguments are passed to detect_contours
    and kept fixed.
    """
    path = Path(path)
    image = load_image.load_image(str(path))
    n = len(values)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    axes = np.atleast_1d(axes)

    for ax, value in zip(axes, values):
        contours = detect_contours(image, **fixed_params, **{param_name: float(value)})
        drawn = draw_contours(image, contours)

        print(f"\n{param_name} = {value:.2f}")
        print_contour_stats(contours)

        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{param_name} = {value:.2f}\n{len(contours)} contours", fontsize=9)
        ax.axis("off")

    fig.suptitle(path.name)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = Path(__file__).resolve().parent / "out" / "contour_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_inverted" if fixed_params.get("invert") else ""
    if fixed_params.get("blur_size"):
        suffix += f"_blur{fixed_params['blur_size']}"
    out_path = out_dir / f"contour_sweep_{path.stem}_{param_name}{suffix}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":

    directory = "intel-image-classification (1)/seg_test/seg_test/buildings"
    """
    for path, image, contours, drawn_image in contour_detect_n_images(directory, n=5, invert=True):
        print(f"\nImage: {path.name}")
        print_contour_stats(contours)
        display_images_side_by_side(image, drawn_image)
    """
    path = "intel-image-classification (1)/seg_test/seg_test/buildings/20074.jpg"
    
    # Run contour detection on one image with the default parameters
    image = load_image.load_image(path)
    contours = detect_contours(image, method = "triangle")
    print_contour_stats(contours)
    drawn_image = draw_contours(image, contours)
    save_images_side_by_side(image, drawn_image, f"contours_{Path(path).stem}")
    display_images_side_by_side(image, drawn_image)
    
    #sweep_params(path, "blur_size", [0, 3, 5, 7], threshold = 90)
    plt.show()