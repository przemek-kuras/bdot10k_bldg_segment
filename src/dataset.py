import numpy as np
import torch
from torch.utils.data import Dataset

class TileDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        x = np.load(row["image"]).astype(np.float32)  # H,W,C in [0,1]
        y = np.load(row["mask"]).astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
            if np.random.rand() < 0.5:
                x, y = x[::-1, :].copy(), y[::-1, :].copy()
            k = np.random.randint(0, 4)
            x, y = np.rot90(x, k).copy(), np.rot90(y, k).copy()
            if np.random.rand() < 0.3:
                x = np.clip(x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.03, 0.03), 0, 1)
        x = torch.from_numpy(np.moveaxis(x, -1, 0).copy()).float()
        y = torch.from_numpy(y[None, ...].copy()).float()
        return x, y
