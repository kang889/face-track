from __future__ import annotations

"""
pack_face_press_dataset.py

Convert raw multi-pass face-press recorder samples into a model-ready dataset.

What this packer does
---------------------
- loads all sample_*.npz files from one session folder (or recursively from a root)
- keeps only one chosen source point (defaults to the dominant source point)
- builds model inputs from the recorded fingertip / press condition
- builds model targets from full cheek landmark deformation (dx, dy, dz)
- builds visibility masks so blocked landmarks do not contribute to training loss
- splits data into train / val / test, preferably by trial index
- saves one packed .npz plus a metadata JSON summary

Intended first prototype input
------------------------------
X = [fingertip_rel_source_px_x,
     fingertip_rel_source_px_y,
     fingertip_xyz_x,
     fingertip_xyz_y,
     fingertip_xyz_z,
     pressure_tier]

Target
------
Y = flattened cheek deformation field:
    [p0_dx, p0_dy, p0_dz, p1_dx, p1_dy, p1_dz, ..., p24_dx, p24_dy, p24_dz]

Mask
----
M = same shape as Y. For a landmark that was blocked in a frame, all 3 channels
    for that landmark are 0. For a visible landmark, all 3 channels are 1.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

PRESSURE_TO_INDEX = {
    "light": 0,
    "medium": 1,
    "hard": 2,
}


def find_sample_files(root: Path) -> list[Path]:
    if root.is_file() and root.name.startswith("sample_") and root.suffix == ".npz":
        return [root]
    return sorted(root.rglob("sample_*.npz"))


def as_text_scalar(x) -> str:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return str(x.item())
        if x.size == 1:
            return str(x.reshape(-1)[0].item())
    return str(x)


def as_int_scalar(x) -> int:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return int(x.item())
        if x.size == 1:
            return int(x.reshape(-1)[0].item())
    return int(x)


def source_counts(sample_files: Iterable[Path]) -> Counter:
    counts: Counter = Counter()
    for path in sample_files:
        with np.load(path, allow_pickle=True) as data:
            counts[as_int_scalar(data["source_local_index"])] += 1
    return counts


def choose_dominant_source(sample_files: list[Path]) -> int:
    counts = source_counts(sample_files)
    if not counts:
        raise RuntimeError("No source points found in dataset")
    return counts.most_common(1)[0][0]


def trial_group_split(trial_indices: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    unique_trials = np.array(sorted(set(int(t) for t in trial_indices.tolist())), dtype=np.int32)
    if len(unique_trials) >= 3:
        shuffled_trials = unique_trials.copy()
        rng.shuffle(shuffled_trials)

        n_trials = len(shuffled_trials)
        n_train = max(1, int(round(n_trials * 0.7)))
        n_val = max(1, int(round(n_trials * 0.15)))
        if n_train + n_val >= n_trials:
            n_val = max(1, n_trials - n_train - 1)
        n_test = n_trials - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        train_trials = set(shuffled_trials[:n_train].tolist())
        val_trials = set(shuffled_trials[n_train:n_train + n_val].tolist())
        test_trials = set(shuffled_trials[n_train + n_val:].tolist())

        train_idx = np.array([i for i, t in enumerate(trial_indices) if int(t) in train_trials], dtype=np.int32)
        val_idx = np.array([i for i, t in enumerate(trial_indices) if int(t) in val_trials], dtype=np.int32)
        test_idx = np.array([i for i, t in enumerate(trial_indices) if int(t) in test_trials], dtype=np.int32)
        return train_idx, val_idx, test_idx, "trial_group_split"

    # Fallback: frame-level split.
    n = len(trial_indices)
    indices = np.arange(n, dtype=np.int32)
    rng.shuffle(indices)
    n_train = max(1, int(round(n * 0.7)))
    n_val = max(1, int(round(n * 0.15)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val:])
    return train_idx, val_idx, test_idx, "frame_random_split"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack raw face-press recorder data into a model-ready dataset")
    parser.add_argument("input_path", type=Path, help="Session folder or dataset root containing sample_*.npz files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save the packed dataset")
    parser.add_argument("--source-index", type=int, default=None, help="Keep only this source point. Default: dominant source")
    parser.add_argument("--min-visible-points", type=int, default=8, help="Skip samples below this visible-point count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = parser.parse_args()

    sample_files = find_sample_files(args.input_path)
    if not sample_files:
        raise RuntimeError(f"No sample_*.npz files found under: {args.input_path}")

    source_index = args.source_index if args.source_index is not None else choose_dominant_source(sample_files)

    if args.output_dir is None:
        input_root = args.input_path if args.input_path.is_dir() else args.input_path.parent
        output_dir = input_root / "packed"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    X_rows: list[np.ndarray] = []
    Y_rows: list[np.ndarray] = []
    M_rows: list[np.ndarray] = []

    trial_list: list[int] = []
    pressure_list: list[str] = []
    pressure_index_list: list[int] = []
    source_mp_id_list: list[int] = []
    alignment_error_list: list[float] = []
    visible_count_list: list[int] = []
    file_list: list[str] = []

    landmark_ids = None
    patch_edges = None
    neutral_template_px = None
    feature_names = [
        "fingertip_rel_source_px_x",
        "fingertip_rel_source_px_y",
        "fingertip_x",
        "fingertip_y",
        "fingertip_z",
        "pressure_tier",
    ]

    skipped = defaultdict(int)

    for path in sample_files:
        with np.load(path, allow_pickle=True) as data:
            sample_source = as_int_scalar(data["source_local_index"])
            if sample_source != source_index:
                skipped["wrong_source"] += 1
                continue

            visible_mask = data["visible_mask"].astype(bool)
            visible_count = int(np.sum(visible_mask))
            if visible_count < args.min_visible_points:
                skipped["too_few_visible"] += 1
                continue

            fingertip_px = data["fingertip_px"].astype(np.float32)
            fingertip_xyz = data["fingertip_xyz"].astype(np.float32)
            if np.isnan(fingertip_px).any() or np.isnan(fingertip_xyz).any():
                skipped["missing_fingertip"] += 1
                continue

            neutral_cheek_points_px = data["neutral_cheek_points_px"].astype(np.float32)
            cheek_disp_xyz = data["cheek_displacement_xyz"].astype(np.float32)
            if np.isnan(cheek_disp_xyz).any():
                skipped["nan_target"] += 1
                continue

            pressure_label = as_text_scalar(data["pressure_label"]).strip().lower()
            pressure_index = PRESSURE_TO_INDEX.get(pressure_label, 1)

            # First prototype input: fingertip location relative to the chosen source point + fingertip xyz + pressure tier.
            source_px = neutral_cheek_points_px[source_index]
            fingertip_rel_source_px = fingertip_px - source_px
            x = np.array([
                float(fingertip_rel_source_px[0]),
                float(fingertip_rel_source_px[1]),
                float(fingertip_xyz[0]),
                float(fingertip_xyz[1]),
                float(fingertip_xyz[2]),
                float(pressure_index),
            ], dtype=np.float32)

            # Target: all 25 cheek points, dx/dy/dz flattened.
            y = cheek_disp_xyz.reshape(-1).astype(np.float32)

            # Mask: one visibility bit per landmark, repeated across dx/dy/dz.
            m = np.repeat(visible_mask.astype(np.float32)[:, None], 3, axis=1).reshape(-1).astype(np.float32)

            X_rows.append(x)
            Y_rows.append(y)
            M_rows.append(m)
            trial_list.append(as_int_scalar(data["trial_index"]))
            pressure_list.append(pressure_label)
            pressure_index_list.append(pressure_index)
            source_mp_id_list.append(as_int_scalar(data["source_mp_id"]))
            alignment_error_list.append(float(as_int_scalar(data["anchor_alignment_error"]) if np.array(data["anchor_alignment_error"]).shape == () else np.array(data["anchor_alignment_error"]).item()))
            visible_count_list.append(visible_count)
            file_list.append(str(path))

            if landmark_ids is None:
                landmark_ids = data["left_cheek_ids"].astype(np.int32)
            if patch_edges is None:
                patch_edges = data["patch_edges"].astype(np.int32)
            if neutral_template_px is None:
                neutral_template_px = neutral_cheek_points_px.astype(np.float32)

    if not X_rows:
        raise RuntimeError("No usable samples remained after filtering")

    X = np.stack(X_rows, axis=0).astype(np.float32)
    Y = np.stack(Y_rows, axis=0).astype(np.float32)
    M = np.stack(M_rows, axis=0).astype(np.float32)
    trials = np.array(trial_list, dtype=np.int32)
    pressure_index_arr = np.array(pressure_index_list, dtype=np.int32)
    source_mp_id_arr = np.array(source_mp_id_list, dtype=np.int32)
    alignment_error_arr = np.array(alignment_error_list, dtype=np.float32)
    visible_count_arr = np.array(visible_count_list, dtype=np.int32)

    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, test_idx, split_method = trial_group_split(trials, rng)

    x_mean = X[train_idx].mean(axis=0).astype(np.float32)
    x_std = X[train_idx].std(axis=0).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)

    packed_npz_path = output_dir / "face_press_packed.npz"
    np.savez_compressed(
        packed_npz_path,
        X=X,
        Y=Y,
        M=M,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        trials=trials,
        pressure_index=pressure_index_arr,
        source_local_index=np.int32(source_index),
        source_mp_id=source_mp_id_arr,
        left_cheek_ids=landmark_ids,
        patch_edges=patch_edges,
        neutral_template_px=neutral_template_px,
        x_mean=x_mean,
        x_std=x_std,
    )

    summary = {
        "input_path": str(args.input_path),
        "output_dir": str(output_dir),
        "packed_npz": str(packed_npz_path),
        "num_samples_kept": int(len(X)),
        "num_features": int(X.shape[1]),
        "num_targets": int(Y.shape[1]),
        "num_landmarks": int(len(landmark_ids) if landmark_ids is not None else 0),
        "source_local_index": int(source_index),
        "source_mp_id": int(source_mp_id_arr[0]) if len(source_mp_id_arr) else None,
        "feature_names": feature_names,
        "split_method": split_method,
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "visible_points_per_sample": {
            "mean": float(visible_count_arr.mean()),
            "median": float(np.median(visible_count_arr)),
            "min": int(visible_count_arr.min()),
            "max": int(visible_count_arr.max()),
        },
        "alignment_error_px": {
            "mean": float(alignment_error_arr.mean()),
            "median": float(np.median(alignment_error_arr)),
            "min": float(alignment_error_arr.min()),
            "max": float(alignment_error_arr.max()),
        },
        "skipped_counts": dict(skipped),
        "pressure_distribution": dict(Counter(pressure_list)),
        "trial_distribution": {str(k): int(v) for k, v in Counter(trial_list).items()},
    }
    (output_dir / "face_press_packed_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "packed_files.txt").write_text("\n".join(file_list))

    print("=" * 72)
    print(f"Packed samples: {len(X)}")
    print(f"Source point kept: local {source_index} | MP {summary['source_mp_id']}")
    print(f"Input dim: {X.shape[1]}")
    print(f"Target dim: {Y.shape[1]}  ({summary['num_landmarks']} landmarks x 3)")
    print(f"Split: train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)} | {split_method}")
    print(f"Visible points/sample mean: {summary['visible_points_per_sample']['mean']:.2f}")
    print(f"Alignment error mean: {summary['alignment_error_px']['mean']:.2f}px")
    print(f"Skipped: {dict(skipped)}")
    print("Saved:")
    print(f"  {packed_npz_path}")
    print(f"  {output_dir / 'face_press_packed_summary.json'}")
    print(f"  {output_dir / 'packed_files.txt'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
