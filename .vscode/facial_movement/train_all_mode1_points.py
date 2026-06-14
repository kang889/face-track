from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATASET_DIR = Path("ml_dataset_point_neighbor")
OUTPUT_DIR = Path("mode1_models")
SUMMARY_JSON = OUTPUT_DIR / "mode1_training_summary.json"

MIN_SOURCE_MAGNITUDE = 2.0
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
MIN_USABLE_SAMPLES = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Patch / graph helpers
# -----------------------------------------------------------------------------


def get_direct_neighbors(patch_edges: np.ndarray, source_index: int) -> list[int]:
    """Return all direct neighbors of one source point from patch_edges."""
    neighbors = set()

    for a, b in patch_edges:
        a = int(a)
        b = int(b)
        if a == source_index:
            neighbors.add(b)
        elif b == source_index:
            neighbors.add(a)

    return sorted(neighbors)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class OnePointToNeighborsDataset(Dataset):
    """Dataset for one source point and its direct-neighbor displacement targets."""

    def __init__(
        self,
        folder: Path,
        source_point_index: int,
        neighbor_indices: list[int],
        min_source_magnitude: float = 2.0,
    ):
        self.source_point_index = source_point_index
        self.neighbor_indices = neighbor_indices
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        files = sorted(folder.glob("*.npz"))
        if not files:
            raise RuntimeError(f"No .npz files found in {folder}")

        for file_path in files:
            data = np.load(file_path)
            cheek_disp = data["cheek_displacement"].astype(np.float32)

            source_vec = cheek_disp[source_point_index]
            source_mag = float(np.linalg.norm(source_vec))
            if source_mag < min_source_magnitude:
                continue

            neighbor_vecs = cheek_disp[neighbor_indices]  # shape (K, 2)
            x = source_vec.reshape(-1)                    # shape (2,)
            y = neighbor_vecs.reshape(-1)                 # shape (K*2,)
            self.samples.append((x, y))

        if not self.samples:
            raise RuntimeError(
                f"No usable samples for source point {source_point_index}. "
                f"Lower MIN_SOURCE_MAGNITUDE or record more data."
            )

        self.input_dim = 2
        self.output_dim = len(neighbor_indices) * 2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class NeighborMLP(nn.Module):
    """Same architecture as the current one-point training script."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Train / eval helpers
# -----------------------------------------------------------------------------


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# -----------------------------------------------------------------------------
# Main batch training
# -----------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATASET_DIR.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in {DATASET_DIR}")

    first = np.load(files[0])
    cheek_disp = first["cheek_displacement"]
    patch_edges = first["patch_edges"]
    left_cheek_ids = first.get("left_cheek_ids", np.arange(cheek_disp.shape[0], dtype=np.int32))
    num_points = int(cheek_disp.shape[0])

    print("=" * 72)
    print(f"Dataset folder: {DATASET_DIR}")
    print(f"Total files: {len(files)}")
    print(f"Cheek points: {num_points}")
    print(f"Device: {DEVICE}")
    print("=" * 72)

    summary: list[dict] = []

    for source_point_index in range(num_points):
        neighbor_indices = get_direct_neighbors(patch_edges, source_point_index)
        mp_id = int(left_cheek_ids[source_point_index])

        print("\n" + "-" * 72)
        print(f"Training source point {source_point_index} (MediaPipe ID {mp_id})")
        print(f"Direct neighbors: {neighbor_indices}")

        if not neighbor_indices:
            print("Skipped: no direct neighbors.")
            summary.append({
                "source_point_index": source_point_index,
                "source_mediapipe_id": mp_id,
                "neighbor_indices": neighbor_indices,
                "status": "skipped_no_neighbors",
            })
            continue

        try:
            dataset = OnePointToNeighborsDataset(
                folder=DATASET_DIR,
                source_point_index=source_point_index,
                neighbor_indices=neighbor_indices,
                min_source_magnitude=MIN_SOURCE_MAGNITUDE,
            )
        except RuntimeError as exc:
            print(f"Skipped: {exc}")
            summary.append({
                "source_point_index": source_point_index,
                "source_mediapipe_id": mp_id,
                "neighbor_indices": neighbor_indices,
                "status": "skipped_no_usable_samples",
                "reason": str(exc),
            })
            continue

        usable_samples = len(dataset)
        print(f"Usable samples: {usable_samples}")

        if usable_samples < MIN_USABLE_SAMPLES:
            print(f"Skipped: usable samples < {MIN_USABLE_SAMPLES}")
            summary.append({
                "source_point_index": source_point_index,
                "source_mediapipe_id": mp_id,
                "neighbor_indices": neighbor_indices,
                "usable_samples": usable_samples,
                "status": "skipped_too_few_samples",
            })
            continue

        n_total = usable_samples
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = n_total - n_train - n_val

        if n_val == 0 or n_test == 0:
            print("Skipped: dataset split too small.")
            summary.append({
                "source_point_index": source_point_index,
                "source_mediapipe_id": mp_id,
                "neighbor_indices": neighbor_indices,
                "usable_samples": usable_samples,
                "status": "skipped_split_too_small",
            })
            continue

        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = NeighborMLP(dataset.input_dim, dataset.output_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_ckpt_path = OUTPUT_DIR / f"point_{source_point_index:02d}_direct.pt"

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0

            for x, y in train_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)

            train_loss /= len(train_loader.dataset)
            val_loss = evaluate(model, val_loader, loss_fn)

            print(
                f"Src {source_point_index:02d} | "
                f"Epoch {epoch+1:03d} | train {train_loss:.6f} | val {val_loss:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "source_point_index": source_point_index,
                        "source_mediapipe_id": mp_id,
                        "neighbor_indices": neighbor_indices,
                        "input_dim": dataset.input_dim,
                        "output_dim": dataset.output_dim,
                        "patch_mode": "direct",
                    },
                    best_ckpt_path,
                )

        checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss = evaluate(model, test_loader, loss_fn)

        print(f"Saved: {best_ckpt_path.name}")
        print(f"Best val loss: {best_val:.6f}")
        print(f"Test loss: {test_loss:.6f}")

        summary.append({
            "source_point_index": source_point_index,
            "source_mediapipe_id": mp_id,
            "neighbor_indices": neighbor_indices,
            "usable_samples": usable_samples,
            "best_val_loss": float(best_val),
            "test_loss": float(test_loss),
            "checkpoint": best_ckpt_path.name,
            "status": "trained",
        })

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Finished. Summary saved to: {SUMMARY_JSON}")
    print(f"Checkpoints saved in: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
