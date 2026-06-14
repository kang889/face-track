from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

DATASET_DIR = Path("ml_dataset_point_neighbor")

# Your chosen source point
SOURCE_POINT_INDEX = 1

# Direct neighbors of point 13 from your analysis
# key in the neighbor points here for each individual/source point
NEIGHBOR_INDICES = [2, 5, 6, 7, 13, 16]

# Ignore samples where the source point barely moved
MIN_SOURCE_MAGNITUDE = 2.0

BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3

#checks whether PyTorch can use your GPU, if can program will use GPU, if not it will use cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class OnePointToNeighborsDataset(Dataset):
    def __init__(
        self,
        folder: Path,
        source_point_index: int,
        neighbor_indices: list[int],
        min_source_magnitude: float = 2.0,
    ):
        self.source_point_index = source_point_index
        self.neighbor_indices = neighbor_indices
        self.samples = []

        files = sorted(folder.glob("*.npz"))
        if not files:
            raise RuntimeError(f"No .npz files found in {folder}")

        for file_path in files:
            data = np.load(file_path)
        
            cheek_disp = data["cheek_displacement"].astype(np.float32) #looks for cheek_displacement array and cast the entire array to 32bit float, 32 bit float is necessary for pytorch

            source_vec = cheek_disp[source_point_index]
            source_mag = float(np.linalg.norm(source_vec)) #np.linalg computes the source vec to find the magnitude/length using pythagoras theorem

            if source_mag < min_source_magnitude: #ensure accurate data that uses only sufficient magnitude
                continue

            neighbor_vecs = cheek_disp[neighbor_indices]   # shape (K, 2)

            #intialising/creating a 1d array
            x = source_vec.reshape(-1)                     # shape (2,) # The -1 tells numPy to automatically calculate the exact length of the 1d array base on the total number of elements
            y = neighbor_vecs.reshape(-1)                  # shape (K*2,)

            self.samples.append((x, y))

        if not self.samples:
            raise RuntimeError(
                "No usable samples after filtering. "
                "Lower MIN_SOURCE_MAGNITUDE or record more data."
            )

        self.input_dim = 2
        self.output_dim = len(neighbor_indices) * 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class NeighborMLP(nn.Module):
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


# ------------------------------------------------------------
# Train / eval helpers
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    dataset = OnePointToNeighborsDataset(
        folder=DATASET_DIR,
        source_point_index=SOURCE_POINT_INDEX,
        neighbor_indices=NEIGHBOR_INDICES,
        min_source_magnitude=MIN_SOURCE_MAGNITUDE,
    )

#split datasets to 3 parts, first part 80% used for training, 10 % used as a quiz at the end of each round to see if model is learning, 10& used as final set for a unbias quiz to see model's  performance
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = NeighborMLP(dataset.input_dim, dataset.output_dim).to(DEVICE) #the brain, the model being trained
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # if the loss function is the grader pointing out mistakes, the optimizer is the mechanic that actually goes inside the network and adjusts the weights to fix those mistakes.
    loss_fn = nn.MSELoss() # the grader that calculates the loss

    best_val = float("inf")

    print("=" * 60)
    print(f"Dataset folder: {DATASET_DIR}")
    print(f"Source point: {SOURCE_POINT_INDEX}")
    print(f"Neighbors: {NEIGHBOR_INDICES}")
    print(f"Usable samples: {len(dataset)}")
    print(f"Output dimension: {dataset.output_dim}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x) # The model makes a guess on the neighbor movements.
            loss = loss_fn(pred, y) # the grader calculates how far off the guess was from the real answer

            optimizer.zero_grad() # PyTorch does calculus behind the scenes to figure out which internal weights caused the error and by how much.
            loss.backward()
            optimizer.step() # the Adam optimizer nudges the weights in the right direction so the model is slightly smarter for the next batch

            train_loss += loss.item() * x.size(0)


       # It checks: Did we get a lower error than our previous best?, if we did we use use torch.save() to save the model's current weights to a file
        train_loss /= len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, loss_fn)

        print(f"Epoch {epoch+1:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "source_point_index": SOURCE_POINT_INDEX,
                    "neighbor_indices": NEIGHBOR_INDICES,
                    "input_dim": dataset.input_dim,
                    "output_dim": dataset.output_dim,
                },
                "one_point_neighbors_model.pt",
            )

#After all epochs are finished, the script loads the absolute best version of the model from the saved file. 
#It runs it one last time on the Test Set (data the model has never, ever seen) to print out the final, official accuracy score.
    checkpoint = torch.load("one_point_neighbors_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(model, test_loader, loss_fn)
    print("=" * 60)
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print("Saved model: one_point_neighbors_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()