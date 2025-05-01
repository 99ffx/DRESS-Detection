import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import random
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
from Code.Utils.dataset import DRESSDataset


def get_patch_paths(slide_name: str) -> Tuple[Path, Path]:
    """Return paths to 10x and 20x patch files for a given slide."""
    base_dir = (
        Path(__file__).resolve().parent.parent
        / "Result"
        / "trident_processed_DRESS_OSU"
    )
    path_10x = (
        base_dir / "10x_224px_0px_overlap" / "patches" / f"{slide_name}_patches.h5"
    )
    path_20x = (
        base_dir / "20x_224px_0px_overlap" / "patches" / f"{slide_name}_patches.h5"
    )
    return path_10x, path_20x


def load_patch_coordinates(
    path_10x: Path, path_20x: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Load patch coordinates from H5 files."""
    with h5py.File(path_10x, "r") as f10, h5py.File(path_20x, "r") as f20:
        return f10["coords"][:], f20["coords"][:]


def find_nearest_patches(
    coords_10x: np.ndarray, coords_20x: np.ndarray, num_samples: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Find nearest 20x patches for random samples of 10x patches."""
    tree_20x = cKDTree(coords_20x)
    indices = random.sample(range(coords_10x.shape[0]), num_samples)
    sampled_10x = coords_10x[indices]
    dists, nearest_20x_indices = tree_20x.query(sampled_10x)
    return sampled_10x, coords_20x[nearest_20x_indices]


def plot_coordinate_matching(
    coords_10x: np.ndarray,
    coords_20x: np.ndarray,
    sampled_10x: np.ndarray,
    nearest_20x: np.ndarray,
    slide_name: str = "",
) -> None:
    """Visualize the patch coordinate matching."""
    plt.figure(figsize=(10, 10))
    plt.scatter(coords_20x[:, 0], coords_20x[:, 1], s=5, c="lightblue", label="All 20x")
    plt.scatter(coords_10x[:, 0], coords_10x[:, 1], s=5, c="lightgray", label="All 10x")
    plt.scatter(
        sampled_10x[:, 0], sampled_10x[:, 1], c="red", label="Sampled 10x", s=30
    )
    plt.scatter(
        nearest_20x[:, 0], nearest_20x[:, 1], c="green", label="Matched 20x", s=30
    )

    for pt10, pt20 in zip(sampled_10x, nearest_20x):
        plt.plot(
            [pt10[0], pt20[0]],
            [pt10[1], pt20[1]],
            c="orange",
            linestyle="--",
            linewidth=1,
        )

    title = "10x vs. 20x Patch Coordinate Matching"
    if slide_name:
        title += f" - {slide_name}"
    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_slide_patches(slide_name: str, num_samples: int = 20) -> Dict:
    """Main analysis function for a single slide."""
    path_10x, path_20x = get_patch_paths(slide_name)
    coords_10x, coords_20x = load_patch_coordinates(path_10x, path_20x)
    sampled_10x, nearest_20x = find_nearest_patches(coords_10x, coords_20x, num_samples)

    print(f"Analysis for {slide_name}:")
    print(f"10x coord shape: {coords_10x.shape}")
    print(f"20x coord shape: {coords_20x.shape}")

    return {
        "coords_10x": coords_10x,
        "coords_20x": coords_20x,
        "sampled_10x": sampled_10x,
        "nearest_20x": nearest_20x,
    }


def test_dataset():
    # Initialize your dataset (replace with your actual parameters)
    df = pd.read_csv("../Dataset_csv/dataset_split.csv")
    dataset = DRESSDataset(
        feats_path1="../Result/Features_Gigapath/Gigapath_10x",
        feats_path2="../Result/Features_Gigapath/Gigapath_20x",
        df=df,
        split="val",
        use_fusion=True,  # or False
    )

    print(f"\nTotal samples: {len(dataset)}")
    features, label = dataset
    print(f"length of features: {len(features)}")
    print(f"length of label: {len(label)}")

    # for i in range(3):
    #     features, label = dataset[i]
    #     print(f"\n--- Sample {i} ---")
    #     print(f"Label: {label.item()} (shape: {label.shape})")
    #     print("Feature shapes:")
    #     if dataset.use_fusion:
    #         print(f"  10x: {features['features_10x'].shape}")
    #         print(f"  20x: {features['features_20x'].shape}")
    #     else:
    #         print(f"  Features: {features['features'].shape}")
    #     print(f"Filename: {features.get('filename', 'MISSING')}")


def main():
    # Example usage
    slide_name = "S12-44902  I-3"  # Without '_patches.h5' suffix
    results = analyze_slide_patches(slide_name)

    # Visualize results
    plot_coordinate_matching(
        results["coords_10x"],
        results["coords_20x"],
        results["sampled_10x"],
        results["nearest_20x"],
        slide_name,
    )


if __name__ == "__main__":
    # main()
    test_dataset()
