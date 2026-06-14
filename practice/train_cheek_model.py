from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

DATASET_DIR = Path("ml_dataset_linear_skinning")
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CheekDataset(Dataset):
    def __init__(self, folder: Path):
        self.files = sorted(folder.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {folder}")

        first = np.load(self.files[0])
        self.input_dim = (
            first["cheek_displacement"].size +
            first["driver_offsets"].size
        )
        self.output_dim = first["patch_displacement"].size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        cheek = data["cheek_displacement"].astype(np.float32).reshape(-1)
        drivers = data["driver_offsets"].astype(np.float32).reshape(-1)
        target = data["patch_displacement"].astype(np.float32).reshape(-1)

        x = np.concatenate([cheek, drivers], axis=0)
        y = target

        return torch.from_numpy(x), torch.from_numpy(y)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, loss_fn):
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


def main():
    dataset = CheekDataset(DATASET_DIR)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP(dataset.input_dim, dataset.output_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")

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

        print(f"Epoch {epoch+1:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": dataset.input_dim,
                    "output_dim": dataset.output_dim,
                },
                "cheek_mlp.pt",
            )

    checkpoint = torch.load("cheek_mlp.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, loss_fn)
    print(f"Test loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()