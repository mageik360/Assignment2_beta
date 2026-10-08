import load_image
import cv2 
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def make_params(**overrides) -> cv2.SimpleBlobDetector_Params:
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 20
    params.maxThreshold = 220
    params.minRepeatability = 3
    params.minDistBetweenBlobs = 7
    params.minArea = 15
    params.maxArea = 5000
    params.filterByArea = True
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.05
    params.thresholdStep = 5
    params.filterByColor = True
    params.blobColor = 0
    

    for key, value in overrides.items():
        setattr(params, key, value)
    return params

# BGR colors: blobs found on the original image are red, blobs found on the inverted image are blue
RED = (0, 0, 255)
BLUE = (255, 0, 0)

def preprocess(image, blur=False):
    """
    Optionally blur the image before detection.
    """
    return cv2.GaussianBlur(image, (5, 5), 0) if blur else image

def detect_blobs(image, params, blur=False):
    """
    Detect blobs on both the image and its inverted version.
    Returns (keypoints, inverted_keypoints).
    """
    detector = cv2.SimpleBlobDetector_create(params)
    src = preprocess(image, blur)
    return detector.detect(src), detector.detect(cv2.bitwise_not(src))

def bbox_candidates(src, params):
    """
    Threshold src at the same levels as the detector and collect the contours that pass
    the area filter, together with their equivalent diameter.
    """
    # The detector always uses THRESH_BINARY, whatever blobColor is. Dark blobs are then
    # holes whose contour runs along the surrounding pixels, so the area must be measured
    # the same way here, or small blobs fall below minArea and get no (or a wrong) box.
    candidates = []
    for t in np.arange(params.minThreshold, params.maxThreshold, params.thresholdStep):
        _, binary = cv2.threshold(src, t, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            if params.minArea <= area <= params.maxArea:
                candidates.append((c, 2 * np.sqrt(area / np.pi)))  # equivalent diameter
    return candidates

def draw_bboxes(output, keypoints, candidates, color, tolerance=1.5):
    for kp in keypoints:
        # The keypoint is the average center over several thresholds, so for tiny blobs it
        # can land up to about a pixel outside the contour; allow for that
        around_kp = [(c, d) for c, d in candidates if cv2.pointPolygonTest(c, kp.pt, True) >= -tolerance]
        if not around_kp:
            continue
        best, _ = min(around_kp, key=lambda cd: abs(cd[1] - kp.size))
        x, y, w, h = cv2.boundingRect(best)
        cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), color, 1)

def draw_blobs_bbox(image, keypoints, inverted_keypoints, params, blur=False):
    """
    Draw a bounding box around each detected blob, red for blobs found on the image and
    blue for blobs found on the inverted image. SimpleBlobDetector only returns a center
    and a size, so the blob shape is recovered by thresholding at the same levels as the
    detector and picking, for each keypoint, the contour around it whose size matches best.
    The box is the bounding rectangle of that contour.
    """
    src = preprocess(image, blur)
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    draw_bboxes(output, keypoints, bbox_candidates(src, params), RED)
    draw_bboxes(output, inverted_keypoints, bbox_candidates(cv2.bitwise_not(src), params), BLUE)
    return output

def draw_blobs(image, keypoints, inverted_keypoints):
    """
    Draw each detected blob as a circle with the size returned by the detector,
    red for blobs found on the image and blue for blobs found on the inverted image.
    """
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for kps, color in ((keypoints, RED), (inverted_keypoints, BLUE)):
        for kp in kps:
            cv2.circle(output, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), color, 1)
    return output

def blob_detect_n_images(directory: str, n=10, params=None, blur=False, bbox=False):
    """
    Run blob detection on the first n images (sorted by filename) in the directory.
    Returns a list of (path, image, keypoints, inverted_keypoints, drawn_image) tuples.
    """
    params = params or make_params()
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:n]
    results = []
    for path in paths:
        image = load_image.load_image(str(path))
        keypoints, inverted_keypoints = detect_blobs(image, params, blur=blur)

        if bbox:
            drawn_image = draw_blobs_bbox(image, keypoints, inverted_keypoints, params, blur=blur)
        else:
          drawn_image = draw_blobs(image, keypoints, inverted_keypoints)
        results.append((path, image, keypoints, inverted_keypoints, drawn_image))
    return results

def display_images_side_by_side(image1: np.ndarray, image2: np.ndarray) -> None:
    """
    Display two images side by side for comparison.
    """
    image1_bgr = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    side_by_side = np.hstack([image1_bgr, image2])
    side_by_side = cv2.resize(side_by_side, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Before | After", side_by_side)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blob_stats(keypoints):
    sizes = [kp.size for kp in keypoints]
    return {
        "count": len(keypoints),
        "mean_size": np.mean(sizes) if sizes else 0,
        "min_size": min(sizes, default=0),
        "max_size": max(sizes, default=0),
        "positions": [(round(kp.pt[0]), round(kp.pt[1])) for kp in keypoints],
    }

def print_blob_stats(keypoints, label=""):
    stats = blob_stats(keypoints)
    print(f"\n{label}Count: {stats['count']}")
    print(f"Mean size: {stats['mean_size']:.2f}")
    print(f"Min size: {stats['min_size']:.2f}")
    print(f"Max size: {stats['max_size']:.2f}")
    print(f"Positions: {stats['positions']}")

def collect_stats(root: str, params, blur=False, max_per_category=None) -> pd.DataFrame:
    rows = []
    for category_dir in sorted(Path(root).iterdir()):
        if not category_dir.is_dir():
            continue
        paths = sorted(category_dir.glob("*.jpg"))
        if max_per_category:
            paths = paths[:max_per_category]
        for path in paths:
            image = load_image.load_image(str(path))
            keypoints, inverted_keypoints = detect_blobs(image, params, blur=blur)
            stats = blob_stats(keypoints)
            stats.update({f"inverted_{k}": v for k, v in blob_stats(inverted_keypoints).items()})
            stats["category"] = category_dir.name
            stats["file"] = path.name
            rows.append(stats)
    return pd.DataFrame(rows)


def sweep_param(path, param_name, start, end, n=5, blur=False, **fixed_params):
    """
    Run blob detection on one image while varying a single parameter
    from start to end in n evenly spaced steps. Extra keyword arguments are
    passed to make_params, e.g. to keep minArea <= maxArea during a maxArea sweep.
    """
    path = Path(path)
    image = load_image.load_image(str(path))
    values = np.linspace(start, end, n)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    axes = np.atleast_1d(axes)

    for ax, value in zip(axes, values):
        params = make_params(**fixed_params, **{param_name: float(value)})
        keypoints, inverted_keypoints = detect_blobs(image, params, blur=blur)
        drawn = draw_blobs_bbox(image, keypoints, inverted_keypoints, params, blur=blur)

        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{param_name} = {value:.2f}\n{len(keypoints)} red / {len(inverted_keypoints)} blue", fontsize=9)
        ax.axis("off")

    fig.suptitle(path.name)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    directory = "intel-image-classification (1)/seg_test/seg_test/buildings"

    """
    for path, image, keypoints, inverted_keypoints, drawn_image in blob_detect_n_images(directory, n=5, bbox=False, blur = False):
        print_blob_stats(keypoints, "Original (red) ")
        print_blob_stats(inverted_keypoints, "Inverted (blue) ")
        display_images_side_by_side(image, drawn_image)
    """
    path = "intel-image-classification (1)/seg_test/seg_test/buildings/20074.jpg"
    sweep_param(path, "minArea", 20, 35, n=4)
