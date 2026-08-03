import numpy as np
import torch
from torch.utils.data import Dataset


class NPYDataset(Dataset):

    def __init__(
        self,
        data_path,
        output_dir=None,
        **kwargs,
    ):
        self.data = np.load(data_path).astype(np.float32)

        assert self.data.ndim == 3

        self.window = self.data.shape[1]
        self.var_num = self.data.shape[2]

        # keep interface compatible
        self.auto_norm = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

class NPYImputationDataset(Dataset):

    def __init__(self, data_path, mask_path, **kwargs):

        self.data = np.load(data_path).astype(np.float32)
        self.mask = np.load(mask_path).astype(np.float32)

        assert self.data.shape == self.mask.shape

        self.window = self.data.shape[1]
        self.var_num = self.data.shape[2]

        self.auto_norm = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return (
            torch.from_numpy(self.data[idx]).float(),
            torch.from_numpy(self.mask[idx]).bool(),
        )