import numpy as np
from pathlib import Path

# Input directory (existing diffusion-ready files)
DATA_DIR = Path("../../data/diffusion_ready/air_quality/")

# Input files
TRAIN_IN = DATA_DIR / "train_seq_small.npy"
TEST_IN = DATA_DIR / "test_seq_small.npy"

# Output files
TRAIN_OUT = DATA_DIR / "train_seq_no_time.npy"
TEST_OUT = DATA_DIR / "test_seq_no_time.npy"

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------

train = np.load(TRAIN_IN)
test = np.load(TEST_IN)

print("Original train shape:", train.shape)
print("Original test shape :", test.shape)

# ------------------------------------------------------------------
# Remove the last 4 features
# ------------------------------------------------------------------

train_11 = train[:, :, :-4]
test_11 = test[:, :, :-4]

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

np.save(TRAIN_OUT, train_11)
np.save(TEST_OUT, test_11)

print("\nSaved:")
print(f"  {TRAIN_OUT.name}: {train_11.shape}")
print(f"  {TEST_OUT.name}: {test_11.shape}")