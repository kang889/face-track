from pathlib import Path
import numpy as np

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

DATASET_DIR = Path("ml_dataset_point_neighbor")

# Choose the local patch point index you want to study.
# This is NOT the MediaPipe global landmark ID.
# This is the local index inside your cheek patch array.
SOURCE_POINT_INDEX = 24

# Ignore frames where the source point barely moved.
MIN_SOURCE_MAGNITUDE = 2.0

# Small number to avoid divide-by-zero.
EPS = 1e-6


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_direct_neighbors(patch_edges, source_index):
    """
    Return all direct neighbors of one source point from patch_edges.
    """
    neighbors = set()

    for a, b in patch_edges:
        a = int(a)
        b = int(b)

        if a == source_index:
            neighbors.add(b)
        elif b == source_index:
            neighbors.add(a)

    return sorted(neighbors)


def cosine_similarity(vec_a, vec_b, eps=1e-6):
    """
    Cosine similarity between two 2D vectors.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a < eps or norm_b < eps:
        return np.nan

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b + eps))


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

def main():
    files = sorted(DATASET_DIR.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in {DATASET_DIR}")

    first = np.load(files[0])
    patch_edges = first["patch_edges"]
    cheek_displacement = first["cheek_displacement"]

    num_points = cheek_displacement.shape[0]

    if SOURCE_POINT_INDEX < 0 or SOURCE_POINT_INDEX >= num_points:
        raise ValueError(
            f"SOURCE_POINT_INDEX={SOURCE_POINT_INDEX} is out of range. "
            f"Valid range: 0 to {num_points - 1}"
        )

    neighbors = get_direct_neighbors(patch_edges, SOURCE_POINT_INDEX)

    if not neighbors:
        raise RuntimeError(
            f"Point {SOURCE_POINT_INDEX} has no direct neighbors in patch_edges."
        )

    print("=" * 60)
    print(f"Dataset folder: {DATASET_DIR}")
    print(f"Number of files: {len(files)}")
    print(f"Number of cheek points in patch: {num_points}")
    print(f"Source point index: {SOURCE_POINT_INDEX}")
    print(f"Direct neighbors: {neighbors}")
    print(f"Minimum source magnitude filter: {MIN_SOURCE_MAGNITUDE}")
    print("=" * 60)

    # Store analysis values for each neighbor
    results = {
        neighbor: {
            "source_magnitudes": [],
            "neighbor_magnitudes": [],
            "magnitude_ratios": [],
            "cosine_similarities": [],
        }
        for neighbor in neighbors
    }

    used_frames = 0

    for file_path in files:
        data = np.load(file_path)
        cheek_disp = data["cheek_displacement"].astype(np.float32)

        source_vec = cheek_disp[SOURCE_POINT_INDEX]
        source_mag = float(np.linalg.norm(source_vec))

        if source_mag < MIN_SOURCE_MAGNITUDE:
            continue

        used_frames += 1

        for neighbor in neighbors:
            neighbor_vec = cheek_disp[neighbor]
            neighbor_mag = float(np.linalg.norm(neighbor_vec))
            ratio = neighbor_mag / (source_mag + EPS)
            cos_sim = cosine_similarity(source_vec, neighbor_vec, eps=EPS)

            results[neighbor]["source_magnitudes"].append(source_mag)
            results[neighbor]["neighbor_magnitudes"].append(neighbor_mag)
            results[neighbor]["magnitude_ratios"].append(ratio)
            results[neighbor]["cosine_similarities"].append(cos_sim)

    print(f"Frames used after source-magnitude filter: {used_frames}")
    print()

    if used_frames == 0:
        print("No usable frames. Lower MIN_SOURCE_MAGNITUDE or record more data.")
        return

    for neighbor in neighbors:
        source_mags = np.array(results[neighbor]["source_magnitudes"], dtype=np.float32)
        neighbor_mags = np.array(results[neighbor]["neighbor_magnitudes"], dtype=np.float32)
        ratios = np.array(results[neighbor]["magnitude_ratios"], dtype=np.float32)
        cos_sims = np.array(results[neighbor]["cosine_similarities"], dtype=np.float32)

        valid_cos = cos_sims[~np.isnan(cos_sims)]

        mean_source_mag = float(np.mean(source_mags)) if len(source_mags) else float("nan")
        mean_neighbor_mag = float(np.mean(neighbor_mags)) if len(neighbor_mags) else float("nan")
        mean_ratio = float(np.mean(ratios)) if len(ratios) else float("nan")
        std_ratio = float(np.std(ratios)) if len(ratios) else float("nan")
        mean_cos = float(np.mean(valid_cos)) if len(valid_cos) else float("nan")
        std_cos = float(np.std(valid_cos)) if len(valid_cos) else float("nan")

        print("-" * 60)
        print(f"Neighbor point: {neighbor}")
        print(f"Mean source magnitude:   {mean_source_mag:.4f}")
        print(f"Mean neighbor magnitude: {mean_neighbor_mag:.4f}")
        print(f"Mean magnitude ratio:    {mean_ratio:.4f}")
        print(f"Std magnitude ratio:     {std_ratio:.4f}")
        print(f"Mean direction similarity (cos): {mean_cos:.4f}")
        print(f"Std direction similarity  (cos): {std_cos:.4f}")

    print("-" * 60)
    print("Guide:")
    print("  Mean magnitude ratio: how much the neighbor moves compared to the source")
    print("  Mean direction similarity: 1.0 means same direction, 0 means unrelated, -1 means opposite")
    print("  Lower std means the relationship is more consistent")


if __name__ == "__main__":
    main()