import numpy as np
from torch.utils.data import Dataset, DataLoader


def make_block_mask(shape, missing_ratio, seed=None):
    B, T, C = shape
    rng = np.random.default_rng(seed)

    mask = np.ones(shape, dtype=np.float32)
    block = max(1, int(T * missing_ratio))

    for i in range(B):
        for c in range(C):
            start = rng.integers(0, T - block + 1)
            mask[i, start:start + block, c] = 0.0

    return mask


def make_random_mask(shape, missing_ratio, seed=None):
    B, T, C = shape
    rng = np.random.default_rng(seed)

    mask = np.ones(shape, dtype=np.float32)
    num_missing = int(T * missing_ratio)

    for i in range(B):
        for c in range(C):
            idx = rng.choice(T, size=num_missing, replace=False)
            mask[i, idx, c] = 0.0

    return mask


class NPYDataset(Dataset):
    def __init__(
        self,
        data_path,
        missing_ratio=0.25,
        mask_type="block",
        seed=0,
        fixed_mask=None,
    ):
        self.data = np.load(data_path).astype(np.float32)

        # feature_idx = [0, 1]
        # feature_idx = [0, 1, 2, 3, 41, 42]
        # feature_idx = [4,5,6,7,8,9,43,44]
        # feature_idx = [10,11,12,13,14,45]
        # feature_idx = [15,16,17,18,19,46]
        # feature_idx = [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,47,48,49,50,51]

        # self.data = self.data[:, :, feature_idx]

        assert self.data.ndim == 3, (
            f"Expected [N,T,C], got {self.data.shape}"
        )

        N, T, C = self.data.shape
        self.eval_length = T
        self.target_dim = C

        self.observed_mask = np.ones_like(self.data, dtype=np.float32)

        if fixed_mask is not None:
            self.gt_mask = np.load(fixed_mask).astype(np.float32)

            if self.gt_mask.shape[-1] != self.data.shape[-1]:
                self.gt_mask = self.gt_mask[:, :, feature_idx]

            assert self.gt_mask.shape == self.data.shape, (
                f"Mask shape {self.gt_mask.shape} does not match data shape {self.data.shape}"
            )
        else:
            if mask_type == "block":
                self.gt_mask = make_block_mask(
                    self.data.shape,
                    missing_ratio,
                    seed=seed,
                )
            elif mask_type == "random":
                self.gt_mask = make_random_mask(
                    self.data.shape,
                    missing_ratio,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown mask_type: {mask_type}")

        self.timepoints = np.arange(T, dtype=np.float32)

    def __getitem__(self, idx):
        return {
            "observed_data": self.data[idx],
            "observed_mask": self.observed_mask[idx],
            "gt_mask": self.gt_mask[idx],
            "timepoints": self.timepoints,
        }

    def __len__(self):
        return len(self.data)


def get_dataloader(
    train_data,
    test_data,
    seed=1,
    batch_size=16,
    missing_ratio=0.25,
    mask_type="block",
    valid_ratio=0.1,
    fixed_mask=None,
):
    train_full = NPYDataset(
        train_data,
        missing_ratio=missing_ratio,
        mask_type=mask_type,
        seed=seed,
    )

    test_dataset = NPYDataset(
        test_data,
        missing_ratio=missing_ratio,
        mask_type=mask_type,
        seed=seed + 999,
        fixed_mask=fixed_mask,
    )

    n = len(train_full)
    idx = np.arange(n)

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_valid = int(n * valid_ratio)
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]

    train_subset = SubsetNPYDataset(train_full, train_idx)
    valid_subset = SubsetNPYDataset(train_full, valid_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader, train_full.target_dim


class SubsetNPYDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices
        self.eval_length = base_dataset.eval_length
        self.target_dim = base_dataset.target_dim

    def __getitem__(self, i):
        return self.base[self.indices[i]]

    def __len__(self):
        return len(self.indices)