import numpy as np

x = np.load("../data/diffusion_ready/air_quality/test_seq_small.npy")

print("mean:", x.mean(axis=(0,1)))
print("std:", x.std(axis=(0,1)))
print("min:", x.min())
print("max:", x.max())