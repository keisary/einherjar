import numpy as np

X = np.load(r"D:/midas_v2/midasV3/src/data/compiled/crypto/15m/BTCUSD_X.npy")
Y = np.load(r"D:/midas_v2/midasV3/src/data/compiled/crypto/15m/BTCUSD_Y_dir.npy")
R = np.load(r"D:/midas_v2/midasV3/src/data/compiled/crypto/15m/BTCUSD_Y_ret.npy")
T = np.load(r"D:/midas_v2/midasV3/src/data/compiled/crypto/15m/BTCUSD_ts.npy")

print(f"X shape={X.shape}, dtype={X.dtype}")
print(f"Y shape={Y.shape}, dtype={Y.dtype}")
print(f"R shape={R.shape}, dtype={R.dtype}")
print(f"T shape={T.shape}, dtype={T.dtype}")
print(f"X[0,:8]={X[0,:8]}")
print(f"Y[0]={Y[0]}")
print(f"R[0]={R[0]}")
print(f"T[0]={T[0]}")

# Check if there are any NaN/inf
print(f"X NaN count={np.isnan(X).sum()}, inf count={np.isinf(X).sum()}")
print(f"R NaN count={np.isnan(R).sum()}, inf count={np.isinf(R).sum()}")

# Sample of returns distribution
print(f"R returns min={R.min():.6f}, max={R.max():.6f}, mean={R.mean():.6f}")
