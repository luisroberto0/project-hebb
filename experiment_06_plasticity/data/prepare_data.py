"""Decodifica os parquets CIFAR-10 do HF -> cifar10.npz (uint8 arrays). Roda uma vez."""
import pandas as pd, io, numpy as np
from PIL import Image
import time

def decode(parquet):
    df = pd.read_parquet(parquet)
    n = len(df)
    X = np.zeros((n, 32, 32, 3), dtype=np.uint8)
    for i, b in enumerate(df['img']):
        X[i] = np.array(Image.open(io.BytesIO(b['bytes'])))
    y = df['label'].to_numpy().astype(np.int64)
    return X, y

t0 = time.time()
trX, trY = decode('cifar10_train.parquet')
teX, teY = decode('cifar10_test.parquet')
np.savez('cifar10.npz', train_x=trX, train_y=trY, test_x=teX, test_y=teY)
print(f"saved cifar10.npz: train {trX.shape} test {teX.shape} ({time.time()-t0:.0f}s)")
print("train label counts:", np.bincount(trY).tolist())
