from __future__ import annotations

"""
export_face_press_inference.py

Load the trained face-press deformation model, run inference for one press condition,
and export a renderer-facing sparse deformation payload.

Two common ways to use it:
1) Export prediction for a real packed sample:
   python export_face_press_inference.py \
       --model ml_dataset_face_press_multipass/packed/model/best_model.pt \
       --packed ml_dataset_face_press_multipass/packed/face_press_packed.npz \
       --sample-index 0

2) Export prediction for a manual press condition:
   python export_face_press_inference.py \
       --model ml_dataset_face_press_multipass/packed/model/best_model.pt \
       --packed ml_dataset_face_press_multipass/packed/face_press_packed.npz \
       --fingertip-rel-x 0 --fingertip-rel-y 0 \
       --fingertip-x 0.45 --fingertip-y 0.40 --fingertip-z -0.02 \
       --pressure medium

What gets exported:
- left cheek landmark IDs
- source point metadata
- neutral cheek template pixel positions
- patch edges
- input feature vector used for inference
- predicted cheek deformation as (25, 3) dx/dy/dz
- predicted current cheek pixel positions (neutral px + predicted dx/dy)
- if using --sample-index, optional ground-truth deformation + visible mask for comparison
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

PRESSURE_TO_INDEX = {
    "light": 0,
    "medium": 1,
    "hard": 2,
}
INDEX_TO_PRESSURE = {v: k for k, v in PRESSURE_TO_INDEX.items()}


class FacePressMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (64, 128, 128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hid in hidden_dims:
            layers.append(nn.Linear(prev, hid))
            layers.append(nn.ReLU())
            prev = hid
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run trained face-press model and export sparse cheek deformation")
    p.add_argument("--model", type=Path, required=True, help="Path to best_model.pt")
    p.add_argument("--packed", type=Path, required=True, help="Path to face_press_packed.npz")
    p.add_argument("--output-dir", type=Path, default=None, help="Where to save the exported prediction")
    p.add_argument("--sample-index", type=int, default=None, help="Use one packed sample as the model input")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test", help="Which packed split sample-index refers to")
    p.add_argument("--fingertip-rel-x", type=float, default=None)
    p.add_argument("--fingertip-rel-y", type=float, default=None)
    p.add_argument("--fingertip-x", type=float, default=None)
    p.add_argument("--fingertip-y", type=float, default=None)
    p.add_argument("--fingertip-z", type=float, default=None)
    p.add_argument("--pressure", choices=["light", "medium", "hard"], default=None)
    return p


def load_checkpoint(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])
    model = FacePressMLP(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def get_split_indices(packed: np.lib.npyio.NpzFile, split_name: str) -> np.ndarray:
    if split_name == "all":
        return np.arange(len(packed["X"]), dtype=np.int64)
    key = f"{split_name}_idx"
    if key not in packed:
        raise KeyError(f"Packed dataset missing split key: {key}")
    return packed[key].astype(np.int64)


def build_input_from_sample(packed: np.lib.npyio.NpzFile, sample_index: int, split_name: str) -> tuple[np.ndarray, dict]:
    split_idx = get_split_indices(packed, split_name)
    if sample_index < 0 or sample_index >= len(split_idx):
        raise IndexError(f"sample-index {sample_index} out of range for split '{split_name}' with {len(split_idx)} samples")
    global_idx = int(split_idx[sample_index])
    x = packed["X"][global_idx].astype(np.float32)
    meta = {
        "sample_mode": "from_packed_sample",
        "split": split_name,
        "sample_index_within_split": int(sample_index),
        "global_sample_index": global_idx,
        "pressure_index": int(packed["pressure_index"][global_idx]) if "pressure_index" in packed else None,
    }
    return x, meta


def build_input_manual(args) -> tuple[np.ndarray, dict]:
    vals = [args.fingertip_rel_x, args.fingertip_rel_y, args.fingertip_x, args.fingertip_y, args.fingertip_z, args.pressure]
    if any(v is None for v in vals):
        raise ValueError("Manual mode requires fingertip-rel-x, fingertip-rel-y, fingertip-x, fingertip-y, fingertip-z, and pressure")
    x = np.array([
        float(args.fingertip_rel_x),
        float(args.fingertip_rel_y),
        float(args.fingertip_x),
        float(args.fingertip_y),
        float(args.fingertip_z),
        float(PRESSURE_TO_INDEX[args.pressure]),
    ], dtype=np.float32)
    meta = {
        "sample_mode": "manual",
        "split": None,
        "sample_index_within_split": None,
        "global_sample_index": None,
        "pressure_index": PRESSURE_TO_INDEX[args.pressure],
    }
    return x, meta


@torch.no_grad()
def run_inference(model: nn.Module, x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, device: torch.device) -> np.ndarray:
    xn = (x.astype(np.float32) - x_mean.astype(np.float32)) / x_std.astype(np.float32)
    xt = torch.from_numpy(xn.reshape(1, -1)).float().to(device)
    pred = model(xt).cpu().numpy().reshape(-1).astype(np.float32)
    return pred


def main() -> None:
    args = build_argparser().parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.packed.exists():
        raise FileNotFoundError(f"Packed dataset not found: {args.packed}")

    output_dir = args.output_dir or (args.model.parent / "export")
    output_dir.mkdir(parents=True, exist_ok=True)

    packed = np.load(args.packed, allow_pickle=True)
    required = ["X", "Y", "M", "x_mean", "x_std", "left_cheek_ids", "patch_edges", "neutral_template_px"]
    missing = [k for k in required if k not in packed]
    if missing:
        raise KeyError(f"Packed dataset missing keys: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.model, device)

    if args.sample_index is not None:
        x, sample_meta = build_input_from_sample(packed, args.sample_index, args.split)
        gt = packed["Y"][sample_meta["global_sample_index"]].astype(np.float32).reshape(-1, 3)
        mask = packed["M"][sample_meta["global_sample_index"]].astype(np.float32).reshape(-1, 3)
    else:
        x, sample_meta = build_input_manual(args)
        gt = None
        mask = None

    pred = run_inference(model, x, packed["x_mean"], packed["x_std"], device).reshape(-1, 3)

    left_cheek_ids = packed["left_cheek_ids"].astype(np.int32)
    patch_edges = packed["patch_edges"].astype(np.int32)
    neutral_template_px = packed["neutral_template_px"].astype(np.float32)
    predicted_current_px = neutral_template_px + pred[:, :2]

    input_feature_names = [
        "fingertip_rel_source_px_x",
        "fingertip_rel_source_px_y",
        "fingertip_x",
        "fingertip_y",
        "fingertip_z",
        "pressure_tier",
    ]
    pressure_value = float(x[5])
    pressure_label = INDEX_TO_PRESSURE.get(int(round(pressure_value)), f"index_{pressure_value}")

    export = {
        "model_path": str(args.model),
        "packed_path": str(args.packed),
        "source_local_index": int(checkpoint.get("source_local_index")) if checkpoint.get("source_local_index") is not None else None,
        "source_mp_id": int(checkpoint.get("source_mp_id")) if checkpoint.get("source_mp_id") is not None else None,
        "left_cheek_ids": left_cheek_ids.tolist(),
        "patch_edges": patch_edges.tolist(),
        "neutral_template_px": neutral_template_px.tolist(),
        "input_feature_names": input_feature_names,
        "input_vector": x.astype(float).tolist(),
        "pressure_label": pressure_label,
        "predicted_deformation_dxyz": pred.tolist(),
        "predicted_current_px": predicted_current_px.tolist(),
        "sample_meta": sample_meta,
    }

    if gt is not None:
        export["ground_truth_deformation_dxyz"] = gt.tolist()
        export["visible_mask_xyz"] = mask.tolist()
        visible = mask[:, 0] > 0.5
        if np.any(visible):
            err = pred[visible] - gt[visible]
            export["sample_metrics"] = {
                "mae_visible": float(np.mean(np.abs(err))),
                "rmse_visible": float(np.sqrt(np.mean(err ** 2))),
                "visible_landmark_count": int(np.sum(visible)),
            }

    stem = "manual_inference" if args.sample_index is None else f"sample_{args.split}_{args.sample_index:04d}"
    json_path = output_dir / f"{stem}.json"
    npz_path = output_dir / f"{stem}.npz"

    json_path.write_text(json.dumps(export, indent=2))
    np.savez_compressed(
        npz_path,
        left_cheek_ids=left_cheek_ids,
        patch_edges=patch_edges,
        neutral_template_px=neutral_template_px,
        input_vector=x.astype(np.float32),
        predicted_deformation_dxyz=pred.astype(np.float32),
        predicted_current_px=predicted_current_px.astype(np.float32),
        **({"ground_truth_deformation_dxyz": gt.astype(np.float32), "visible_mask_xyz": mask.astype(np.float32)} if gt is not None else {}),
    )

    print("=" * 72)
    print(f"Export mode: {sample_meta['sample_mode']}")
    if args.sample_index is not None:
        print(f"From split/sample: {args.split} / {args.sample_index}")
    print(f"Source: local {export['source_local_index']} | MP {export['source_mp_id']}")
    print(f"Pressure label: {pressure_label}")
    print(f"Predicted deformation shape: {pred.shape}")
    if gt is not None and 'sample_metrics' in export:
        print(f"Sample metrics: mae_visible={export['sample_metrics']['mae_visible']:.6f} | rmse_visible={export['sample_metrics']['rmse_visible']:.6f}")
    print("Saved:")
    print(f"  {json_path}")
    print(f"  {npz_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
