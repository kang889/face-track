from __future__ import annotations

"""
train_face_press_model.py

Train a first prototype press-to-deformation model from the packed face-press dataset.

Input
-----
Packed dataset created by pack_face_press_dataset.py containing:
- X : (N, input_dim)
- Y : (N, output_dim)
- M : (N, output_dim) visibility mask
- train/val/test indices
- metadata for source point and cheek patch

Model
-----
A simple MLP that predicts the full cheek deformation field (dx, dy, dz for all
25 cheek landmarks) from the press-condition input vector.

Loss
----
Masked MSE: only visible landmark outputs contribute to the loss.

Outputs
-------
- best_model.pt
- training_summary.json
- predictions_test.npz
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class PackedFacePressDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, M: np.ndarray, indices: np.ndarray):
        self.X = torch.from_numpy(X[indices]).float()
        self.Y = torch.from_numpy(Y[indices]).float()
        self.M = torch.from_numpy(M[indices]).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.M[idx]


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


class WeightedMaskedMSELoss(nn.Module):
    """Masked MSE with optional extra weight on z outputs."""

    def __init__(self, z_weight: float = 1.0):
        super().__init__()
        self.z_weight = float(z_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # output layout is [p0_dx, p0_dy, p0_dz, p1_dx, ...]
        if self.z_weight != 1.0:
            weights = torch.ones_like(target)
            weights[:, 2::3] = self.z_weight
        else:
            weights = 1.0

        sq = (pred - target) ** 2
        masked = sq * mask * weights
        denom = (mask * weights).sum().clamp_min(1.0)
        return masked.sum() / denom


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: WeightedMaskedMSELoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_weight = 0

    # Metrics on visible outputs only.
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    mask_sum = 0.0

    for x, y, m in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        pred = model(x)
        loss = loss_fn(pred, y, m)

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_weight += batch_size

        diff = (pred - y) * m
        abs_err_sum += float(diff.abs().sum().item())
        sq_err_sum += float((diff ** 2).sum().item())
        mask_sum += float(m.sum().item())

    mean_loss = total_loss / max(total_weight, 1)
    mae = abs_err_sum / max(mask_sum, 1.0)
    rmse = (sq_err_sum / max(mask_sum, 1.0)) ** 0.5
    return {
        "loss": mean_loss,
        "mae_visible": mae,
        "rmse_visible": rmse,
    }


@torch.no_grad()
def collect_test_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    xs, ys, ms, ps = [], [], [], []
    for x, y, m in loader:
        pred = model(x.to(device)).cpu().numpy().astype(np.float32)
        xs.append(x.numpy().astype(np.float32))
        ys.append(y.numpy().astype(np.float32))
        ms.append(m.numpy().astype(np.float32))
        ps.append(pred)
    return {
        "X": np.concatenate(xs, axis=0) if xs else np.empty((0, 0), np.float32),
        "Y": np.concatenate(ys, axis=0) if ys else np.empty((0, 0), np.float32),
        "M": np.concatenate(ms, axis=0) if ms else np.empty((0, 0), np.float32),
        "pred": np.concatenate(ps, axis=0) if ps else np.empty((0, 0), np.float32),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a first face-press deformation model.")
    parser.add_argument("packed_npz", type=Path, help="Path to face_press_packed.npz")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to packed file parent / model")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--z-weight", type=float, default=1.5, help="Extra loss weight on dz outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if not args.packed_npz.exists():
        raise FileNotFoundError(f"Packed dataset not found: {args.packed_npz}")

    output_dir = args.output_dir or (args.packed_npz.parent / "model")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    packed = np.load(args.packed_npz, allow_pickle=True)
    required = ["X", "Y", "M", "train_idx", "val_idx", "test_idx", "x_mean", "x_std"]
    missing = [k for k in required if k not in packed]
    if missing:
        raise KeyError(f"Packed dataset missing keys: {missing}")

    X = packed["X"].astype(np.float32)
    Y = packed["Y"].astype(np.float32)
    M = packed["M"].astype(np.float32)
    train_idx = packed["train_idx"].astype(np.int64)
    val_idx = packed["val_idx"].astype(np.int64)
    test_idx = packed["test_idx"].astype(np.int64)
    x_mean = packed["x_mean"].astype(np.float32)
    x_std = packed["x_std"].astype(np.float32)

    # Standardize inputs using train split statistics from the packer.
    Xn = (X - x_mean) / x_std

    train_ds = PackedFacePressDataset(Xn, Y, M, train_idx)
    val_ds = PackedFacePressDataset(Xn, Y, M, val_idx)
    test_ds = PackedFacePressDataset(Xn, Y, M, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = int(X.shape[1])
    output_dim = int(Y.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FacePressMLP(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = WeightedMaskedMSELoss(z_weight=args.z_weight)

    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    history = []
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x, y, m in train_loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y, m)
            loss.backward()
            optimizer.step()

            batch_size = x.shape[0]
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | train {train_loss:.6f} | "
            f"val {val_loss:.6f} | val_rmse {val_metrics['rmse_visible']:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "source_local_index": int(np.array(packed["source_local_index"]).item()) if "source_local_index" in packed else None,
                    "source_mp_id": int(np.array(packed["source_mp_id"]).reshape(-1)[0]) if "source_mp_id" in packed else None,
                    "left_cheek_ids": packed["left_cheek_ids"] if "left_cheek_ids" in packed else None,
                    "patch_edges": packed["patch_edges"] if "patch_edges" in packed else None,
                    "neutral_template_px": packed["neutral_template_px"] if "neutral_template_px" in packed else None,
                    "history_best_epoch": best_epoch,
                    "z_weight": args.z_weight,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # Final evaluation with best checkpoint.
    # PyTorch 2.6 defaults torch.load(..., weights_only=True), but this
    # checkpoint also stores NumPy arrays/metadata, so we explicitly set
    # weights_only=False for this locally generated trusted file.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, loss_fn, device)

    preds = collect_test_predictions(model, test_loader, device)
    pred_path = output_dir / "predictions_test.npz"
    np.savez_compressed(pred_path, **preds)

    summary = {
        "packed_npz": str(args.packed_npz),
        "output_dir": str(output_dir),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "train_count": int(len(train_ds)),
        "val_count": int(len(val_ds)),
        "test_count": int(len(test_ds)),
        "device": str(device),
        "epochs_requested": args.epochs,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "test_metrics": test_metrics,
        "z_weight": float(args.z_weight),
        "model_path": str(best_path),
        "predictions_test_path": str(pred_path),
        "source_local_index": int(np.array(packed["source_local_index"]).item()) if "source_local_index" in packed else None,
        "source_mp_id": int(np.array(packed["source_mp_id"]).reshape(-1)[0]) if "source_mp_id" in packed else None,
        "history": history,
    }

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 72)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val:.6f}")
    print(
        "Test metrics: "
        f"loss={test_metrics['loss']:.6f}, "
        f"mae_visible={test_metrics['mae_visible']:.6f}, "
        f"rmse_visible={test_metrics['rmse_visible']:.6f}"
    )
    print("Saved:")
    print(f"  {best_path}")
    print(f"  {summary_path}")
    print(f"  {pred_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
