from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PRESSURE_TO_INDEX = {
    "light": 0,
    "medium": 1,
    "hard": 2,
}


def infer_packed_files_path(packed_npz: Path) -> Path:
    return packed_npz.parent / "packed_files.txt"


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_labels_from_indices(n: int, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    labels = np.full(n, "unused", dtype=object)
    labels[train_idx.astype(int)] = "train"
    labels[val_idx.astype(int)] = "val"
    labels[test_idx.astype(int)] = "test"
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a clean subset from the packed face-press dataset."
    )
    parser.add_argument("packed_npz", type=Path, help="Packed dataset .npz")
    parser.add_argument(
        "--packed-files",
        type=Path,
        default=None,
        help="Optional packed_files.txt from the packer. Used to recover per-sample raw metadata like alignment error.",
    )
    parser.add_argument("--pressure", type=str, default="medium", choices=["light", "medium", "hard"])
    parser.add_argument("--max-fingertip-dist", type=float, default=25.0, help="Max fingertip-to-source distance in px")
    parser.add_argument("--min-visible-points", type=int, default=20, help="Minimum visible cheek points")
    parser.add_argument("--max-align-error", type=float, default=5.0, help="Max anchor alignment error in px")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Default: packed/filtered_subset",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = args.packed_npz.parent / "filtered_subset"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    packed_files_path = args.packed_files if args.packed_files is not None else infer_packed_files_path(args.packed_npz)
    packed_file_list = load_lines(packed_files_path)
    have_raw_manifest = len(packed_file_list) > 0

    with np.load(args.packed_npz, allow_pickle=True) as packed:
        X = packed["X"].astype(np.float32)
        Y = packed["Y"].astype(np.float32)
        M = packed["M"].astype(np.float32)
        train_idx = packed["train_idx"].astype(np.int64)
        val_idx = packed["val_idx"].astype(np.int64)
        test_idx = packed["test_idx"].astype(np.int64)
        trials = packed["trials"].astype(np.int64)
        pressure_index = packed["pressure_index"].astype(np.int64)
        source_local_index = int(np.array(packed["source_local_index"]).item())
        source_mp_id = int(np.array(packed["source_mp_id"]).reshape(-1)[0])
        left_cheek_ids = packed["left_cheek_ids"].astype(np.int32)
        patch_edges = packed["patch_edges"].astype(np.int32)
        neutral_template_px = packed["neutral_template_px"].astype(np.float32)
        x_mean = packed["x_mean"].astype(np.float32)
        x_std = packed["x_std"].astype(np.float32)

    n = X.shape[0]
    if have_raw_manifest and len(packed_file_list) != n:
        raise RuntimeError(
            f"packed_files.txt length ({len(packed_file_list)}) does not match packed sample count ({n})."
        )

    split_labels = split_labels_from_indices(n, train_idx, val_idx, test_idx)

    # Features follow the packer:
    # 0 rel fingertip x, 1 rel fingertip y, 2 fingertip x, 3 fingertip y, 4 fingertip z, 5 pressure tier
    fingertip_dist = np.linalg.norm(X[:, 0:2], axis=1)
    visible_counts = (M.reshape(n, -1, 3)[:, :, 0] > 0.5).sum(axis=1)

    pressure_target = PRESSURE_TO_INDEX[args.pressure]
    pressure_keep = pressure_index == pressure_target
    dist_keep = fingertip_dist <= args.max_fingertip_dist
    visible_keep = visible_counts >= args.min_visible_points

    align_errors = np.full(n, np.nan, dtype=np.float32)
    align_keep = np.ones(n, dtype=bool)

    if have_raw_manifest:
        for i, path_str in enumerate(packed_file_list):
            with np.load(path_str, allow_pickle=True) as raw:
                align_val = float(np.array(raw["anchor_alignment_error"]).item())
                align_errors[i] = align_val
        align_keep = align_errors <= args.max_align_error

    keep = pressure_keep & dist_keep & visible_keep & align_keep
    kept_idx = np.nonzero(keep)[0]

    if len(kept_idx) == 0:
        raise RuntimeError("No samples remained after applying the requested filters.")

    X_f = X[kept_idx]
    Y_f = Y[kept_idx]
    M_f = M[kept_idx]
    trials_f = trials[kept_idx]
    pressure_index_f = pressure_index[kept_idx]
    split_labels_f = split_labels[kept_idx]
    fingertip_dist_f = fingertip_dist[kept_idx]
    visible_counts_f = visible_counts[kept_idx]
    align_errors_f = align_errors[kept_idx]

    out_npz = output_dir / "face_press_filtered_subset.npz"
    np.savez_compressed(
        out_npz,
        X=X_f,
        Y=Y_f,
        M=M_f,
        kept_idx=kept_idx.astype(np.int64),
        split_labels=split_labels_f.astype(object),
        trials=trials_f.astype(np.int64),
        pressure_index=pressure_index_f.astype(np.int64),
        fingertip_dist_px=fingertip_dist_f.astype(np.float32),
        visible_points=visible_counts_f.astype(np.int32),
        alignment_error_px=align_errors_f.astype(np.float32),
        source_local_index=np.int32(source_local_index),
        source_mp_id=np.int32(source_mp_id),
        left_cheek_ids=left_cheek_ids,
        patch_edges=patch_edges,
        neutral_template_px=neutral_template_px,
        x_mean=x_mean,
        x_std=x_std,
    )

    (output_dir / "kept_indices.txt").write_text("\n".join(str(int(i)) for i in kept_idx))
    if have_raw_manifest:
        kept_files = [packed_file_list[int(i)] for i in kept_idx]
        (output_dir / "kept_files.txt").write_text("\n".join(kept_files))

    split_counts = {}
    unique_splits, counts = np.unique(split_labels_f, return_counts=True)
    for s, c in zip(unique_splits.tolist(), counts.tolist()):
        split_counts[str(s)] = int(c)

    summary = {
        "packed_npz": str(args.packed_npz),
        "packed_files_used": str(packed_files_path) if have_raw_manifest else None,
        "filters": {
            "pressure": args.pressure,
            "max_fingertip_dist_px": float(args.max_fingertip_dist),
            "min_visible_points": int(args.min_visible_points),
            "max_align_error_px": float(args.max_align_error) if have_raw_manifest else None,
        },
        "source_local_index": int(source_local_index),
        "source_mp_id": int(source_mp_id),
        "kept_count": int(len(kept_idx)),
        "kept_fraction": float(len(kept_idx) / n),
        "split_counts": split_counts,
        "stats": {
            "fingertip_dist_px": {
                "mean": float(np.mean(fingertip_dist_f)),
                "median": float(np.median(fingertip_dist_f)),
                "min": float(np.min(fingertip_dist_f)),
                "max": float(np.max(fingertip_dist_f)),
            },
            "visible_points": {
                "mean": float(np.mean(visible_counts_f)),
                "median": float(np.median(visible_counts_f)),
                "min": int(np.min(visible_counts_f)),
                "max": int(np.max(visible_counts_f)),
            },
        },
    }
    if have_raw_manifest:
        summary["stats"]["alignment_error_px"] = {
            "mean": float(np.mean(align_errors_f)),
            "median": float(np.median(align_errors_f)),
            "min": float(np.min(align_errors_f)),
            "max": float(np.max(align_errors_f)),
        }

    summary_path = output_dir / "filtered_subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 72)
    print(f"Kept samples: {len(kept_idx)} / {n}")
    print(f"Source point: local {source_local_index} | MP {source_mp_id}")
    print(f"Pressure kept: {args.pressure}")
    print(f"Fingertip distance <= {args.max_fingertip_dist:.1f}px")
    print(f"Visible points >= {args.min_visible_points}")
    if have_raw_manifest:
        print(f"Alignment error <= {args.max_align_error:.1f}px")
    else:
        print("Alignment error filter skipped (no packed_files/raw manifest found)")
    print(f"Split counts: {split_counts}")
    print("Saved:")
    print(f"  {out_npz}")
    print(f"  {summary_path}")
    print(f"  {output_dir / 'kept_indices.txt'}")
    if have_raw_manifest:
        print(f"  {output_dir / 'kept_files.txt'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
