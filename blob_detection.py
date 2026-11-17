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
    params.thresholdStep = 5
    params.minRepeatability = 3
    params.minDistBetweenBlobs = 7
    params.filterByArea = True
    params.minArea = 15
    params.maxArea = 5000
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.05
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

def make_side_by_side(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Stack a grayscale image and its drawn BGR version horizontally, scaled up 4x.
    """
    image1_bgr = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    side_by_side = np.hstack([image1_bgr, image2])
    return cv2.resize(side_by_side, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

def save_images_side_by_side(image1: np.ndarray, image2: np.ndarray, name: str) -> Path:
    """
    Save the same side by side image that display_images_side_by_side shows to out/<name>.png.
    """
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{name}.png"
    cv2.imwrite(str(out_path), make_side_by_side(image1, image2))
    print("Saved image to", out_path)
    return out_path

def display_images_side_by_side(image1: np.ndarray, image2: np.ndarray) -> None:
    """
    Display two images side by side for comparison.
    """
    cv2.imshow("Before | After", make_side_by_side(image1, image2))
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


DATASET = "intel-image-classification/seg_test/seg_test"

# One photo from each scene category in seg_test.
EXAMPLES = [
    ("buildings", "20057.jpg"),
    ("street", "20066.jpg"),
    ("forest", "20056.jpg"),
    ("sea", "20072.jpg"),
    ("glacier", "20059.jpg"),
    ("mountain", "20058.jpg"),
]

def print_blob_details(keypoints, label):
    """Print each blob's center and diameter, not only the summary."""
    print(f"\n{label} ({len(keypoints)})")
    for i, kp in enumerate(keypoints, 1):
        x, y = kp.pt
        print(f"  {i:3d}  x={x:6.1f}  y={y:6.1f}  size={kp.size:6.2f}")

def run_category_examples(root, examples, params=None, blur=False):
    """
    Detect blobs on one image per category, print per-blob stats, and save
    the marked image next to the grayscale original.
    """
    params = params or make_params()
    root = Path(root)
    for category, filename in examples:
        path = root / category / filename
        image = load_image.load_image(str(path))
        keypoints, inverted_keypoints = detect_blobs(image, params, blur=blur)
        drawn = draw_blobs(image, keypoints, inverted_keypoints)
        print(f"\n=== {category}/{filename} ===")
        print_blob_details(keypoints, "Dark blobs (red)")
        print_blob_details(inverted_keypoints, "Bright blobs (blue)")
        save_images_side_by_side(image, drawn, f"blobs_{category}_{path.stem}")

def summarize_categories(root, max_per_category=20, blur=False):
    """Average blob counts and sizes over the first images of each category."""
    df = collect_stats(root, make_params(), blur=blur, max_per_category=max_per_category)
    summary = (
        df.groupby("category")[["count", "inverted_count", "mean_size", "inverted_mean_size"]]
        .mean()
        .round(2)
    )
    print(f"\nMean over the first {max_per_category} images of each category")
    print(summary.to_string())
    out_path = Path(__file__).resolve().parent / "out" / "blob_category_summary.csv"
    out_path.parent.mkdir(exist_ok=True)
    summary.to_csv(out_path)
    print("Saved summary to", out_path)
    return df


def sweep_param(path, param_name, values, blur=False, **fixed_params):
    """
    Run blob detection on one image once for each value in values for a single
    parameter. Extra keyword arguments are passed to make_params, e.g. to keep
    minArea <= maxArea during a maxArea sweep.
    """
    path = Path(path)
    image = load_image.load_image(str(path))
    n = len(values)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    axes = np.atleast_1d(axes)

    for ax, value in zip(axes, values):
        params = make_params(**fixed_params, **{param_name: float(value)})
        keypoints, inverted_keypoints = detect_blobs(image, params, blur=blur)
        drawn = draw_blobs(image, keypoints, inverted_keypoints)

        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{param_name} = {value:.2f}\n{len(keypoints)} red / {len(inverted_keypoints)} blue", fontsize=9)
        ax.axis("off")

    fig.suptitle(path.name)
    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"blob_sweep_{path.stem}_{param_name}{'_blur' if blur else ''}.png"
    fig.savefig(out_path, dpi=150)
    print("Saved plot to", out_path)
    plt.show()

# Building images shown in Table 4 of the report.
TABLE4_BUILDINGS = ["20057.jpg", "20060.jpg", "20061.jpg", "20064.jpg", "20073.jpg", "20074.jpg"]

# Matplotlib colors matching the drawn images: red = dark blobs, blue = bright blobs.
# The marker shape is a second cue so the two types are not told apart by color alone.
BLOB_STYLE = {
    "dark": {"color": "#d62728", "marker": "o", "label": "Dark blobs (red)"},
    "bright": {"color": "#1f77b4", "marker": "s", "label": "Bright blobs (blue)"},
}

def centroid_table(path, params=None, blur=False) -> pd.DataFrame:
    """
    One row per detected blob with its centroid (x, y) in pixel coordinates
    (origin top left, y downwards) and its diameter. Blobs get labels D1, D2, ...
    (dark) and B1, B2, ... (bright), numbered by size, largest first.
    """
    params = params or make_params()
    image = load_image.load_image(str(path))
    keypoints, inverted_keypoints = detect_blobs(image, params, blur=blur)
    rows = []
    for kind, prefix, kps in (("dark", "D", keypoints), ("bright", "B", inverted_keypoints)):
        for i, kp in enumerate(sorted(kps, key=lambda k: -k.size), 1):
            rows.append({"file": Path(path).name, "label": f"{prefix}{i}", "type": kind,
                         "x": round(kp.pt[0], 1), "y": round(kp.pt[1], 1),
                         "diameter": round(kp.size, 2)})
    df = pd.DataFrame(rows, columns=["file", "label", "type", "x", "y", "diameter"])
    return image, df

def spatial_summary(df: pd.DataFrame, height: int) -> pd.DataFrame:
    """
    Per image and blob type: count, mean/std of the centroid coordinates, and the
    share of blobs in the upper half of the image (y < height / 2).
    """
    g = df.groupby(["file", "type"])
    summary = g.agg(count=("x", "size"), mean_x=("x", "mean"), std_x=("x", "std"),
                    mean_y=("y", "mean"), std_y=("y", "std"))
    summary["upper_half_pct"] = g["y"].apply(lambda y: 100 * (y < height / 2).mean())
    return summary.round(1)

def plot_blob_centroids(root, files, params=None, blur=False, n_labels=3):
    """
    For each image, show the grayscale image with the detected centroids (the
    n_labels largest of each type are labeled) next to a scatter of all centroid
    coordinates in pixel space, with marginal histograms of x and y.
    Saves the figure, a CSV with every centroid, a CSV with the spatial summary,
    and prints a LaTeX table with the labeled centroids.
    """
    root = Path(root)
    out_dir = Path(__file__).resolve().parent / "out" / "blob_centroids"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Two buildings per row, each as (image, centroid scatter). A lone building on
    # the last row is centered.
    per_row = 2
    n = len(files)
    n_rows = -(-n // per_row)
    fig = plt.figure(figsize=(9 * per_row, 3.6 * n_rows))
    outer = fig.add_gridspec(n_rows, 2 * per_row, wspace=0.3, hspace=0.45)
    tables = []
    for idx, filename in enumerate(files):
        row, slot = divmod(idx, per_row)
        in_row = min(per_row, n - row * per_row)
        col = 2 * slot + (per_row - in_row)
        image, df = centroid_table(root / filename, params, blur)
        tables.append(df)
        h, w = image.shape

        # Left: the image with centroids and labels
        ax_img = fig.add_subplot(outer[row, col])
        ax_img.imshow(image, cmap="gray", vmin=0, vmax=255)
        for kind, style in BLOB_STYLE.items():
            sub = df[df["type"] == kind]
            ax_img.scatter(sub["x"], sub["y"], s=14, c=style["color"], marker=style["marker"],
                           edgecolors="white", linewidths=0.6)
            for _, r in sub.head(n_labels).iterrows():
                ax_img.annotate(r["label"], (r["x"], r["y"]), xytext=(3, -3),
                                textcoords="offset points", fontsize=7, color="black",
                                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        ax_img.set_title(f"{filename}: {(df.type == 'dark').sum()} dark / "
                         f"{(df.type == 'bright').sum()} bright", fontsize=9)
        ax_img.set_xlabel("x (px)", fontsize=8)
        ax_img.set_ylabel("y (px)", fontsize=8)
        ax_img.tick_params(labelsize=7)

        # Right: centroid scatter with marginal histograms
        inner = outer[row, col + 1].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                          wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(inner[1, 0])
        ax_hx = fig.add_subplot(inner[0, 0], sharex=ax)
        ax_hy = fig.add_subplot(inner[1, 1], sharey=ax)
        bins = np.linspace(0, max(h, w), 11)
        for kind, style in BLOB_STYLE.items():
            sub = df[df["type"] == kind]
            ax.scatter(sub["x"], sub["y"], s=sub["diameter"] * 4, c=style["color"],
                       marker=style["marker"], alpha=0.7, edgecolors="white", linewidths=0.5,
                       label=style["label"])
            ax_hx.hist(sub["x"], bins=bins, color=style["color"], histtype="step", lw=1.5)
            ax_hy.hist(sub["y"], bins=bins, color=style["color"], histtype="step", lw=1.5, orientation="horizontal")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # image coordinates: y grows downwards
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(h / 2, color="0.7", lw=0.8, ls="--")
        ax.set_xlabel("centroid x (px)", fontsize=8)
        ax.set_ylabel("centroid y (px)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color="0.9", lw=0.5)
        ax_hx.tick_params(labelbottom=False, labelsize=6)
        ax_hy.tick_params(labelleft=False, labelsize=6)
        ax_hx.set_ylabel("n", fontsize=7)
        ax_hy.set_xlabel("n", fontsize=7)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.94))

    out_path = out_dir / "blob_centroids_buildings.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved plot to", out_path)

    all_df = pd.concat(tables, ignore_index=True)
    all_df.to_csv(out_dir / "blob_centroids_buildings.csv", index=False)
    summary = spatial_summary(all_df, height=image.shape[0])
    summary.to_csv(out_dir / "blob_centroid_summary.csv")
    print("\nSpatial distribution of centroids per image")
    print(summary.to_string())

    # A few labeled centroids per image, ready for the report
    labeled = all_df.groupby(["file", "type"]).head(3)
    print("\n% Labeled blob centroids (3 largest of each type per image)")
    print("\\begin{tabular}{llrrr}\n\\hline")
    print("Image & Blob & $x$ (px) & $y$ (px) & Diameter (px) \\\\\n\\hline")
    for _, r in labeled.iterrows():
        print(f"{r['file']} & {r['label']} & {r['x']:.1f} & {r['y']:.1f} & {r['diameter']:.1f} \\\\")
    print("\\hline\n\\end{tabular}")
    labeled.to_csv(out_dir / "blob_centroids_labeled.csv", index=False)
    plt.close(fig)
    return all_df, summary


if __name__ == "__main__":

    plot_blob_centroids("intel-image-classification (1)/seg_test/seg_test/buildings", TABLE4_BUILDINGS)

    run_category_examples(DATASET, EXAMPLES)
    summarize_categories(DATASET, max_per_category=20)

    directory = "intel-image-classification/seg_test/seg_test/buildings"
    if Path(directory).is_dir():
        for path, image, keypoints, inverted_keypoints, drawn_image in blob_detect_n_images(directory, n=5, bbox=False, blur = False):
            print_blob_stats(keypoints, "Original (red) ")
            print_blob_stats(inverted_keypoints, "Inverted (blue) ")
            save_images_side_by_side(image, drawn_image, f"blobs_{path.stem}")
            display_images_side_by_side(image, drawn_image)
    else:
        print("Skipping building sample; folder not found:", directory)
    """
    path = "intel-image-classification (1)/seg_test/seg_test/buildings/20074.jpg"

    # Do blobdetection on path with different parameters and display the results
    image = load_image.load_image(path)
    params = make_params()

    keypoints, inverted_keypoints = detect_blobs(image, params, blur=False)
    print_blob_stats(keypoints, "Original (red) ")
    print_blob_stats(inverted_keypoints, "Inverted (blue) ")

    drawn = draw_blobs(image, keypoints, inverted_keypoints)
    display_images_side_by_side(image, drawn)
    """
    # Thresholds
    #sweep_param(path, "minArea", [20, 25, 30, 35], blur=False)
    #sweep_param(path, "maxArea", [20, 75, 125, 180], blur=False)
    #sweep_param(path, "minThreshold", [0, 25, 50, 75], blur=False)
    #sweep_param(path, "maxThreshold", [100, 125, 150, 175], blur=False)
    #sweep_param(path, "thresholdStep", [5, 10, 15, 20], blur=False)         
    #sweep_param(path, "minDistBetweenBlobs", [10, 20, 30, 40], blur=False)
    #sweep_param(path, "minConvexity", [ 0.7, 0.8, 0.9, 0.95], blur=False)
    #sweep_param(path, "minCircularity", [0.2, 0.4, 0.6, 0.8], blur=False)