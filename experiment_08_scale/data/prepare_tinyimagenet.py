"""Marco 5 (escala) — decodifica Tiny-ImageNet (200 classes) dos parquets HF -> npz, resize 32x32
(reusa a arquitetura do Marco 3). Roda uma vez."""
import pandas as pd, io, numpy as np, time
from PIL import Image

def decode(parquet, size=32):
    df = pd.read_parquet(parquet)
    X = np.zeros((len(df), size, size, 3), dtype=np.uint8)
    for i, r in enumerate(df['image']):
        X[i] = np.array(Image.open(io.BytesIO(r['bytes'])).convert('RGB').resize((size, size)))
    return X, df['label'].to_numpy().astype(np.int64)

t0 = time.time()
trX, trY = decode('tiny_train.parquet')
teX, teY = decode('tiny_valid.parquet')
np.savez('tinyimagenet32.npz', train_x=trX, train_y=trY, test_x=teX, test_y=teY)
print(f"saved tinyimagenet32.npz: train {trX.shape} test {teX.shape} classes {len(np.unique(trY))} ({time.time()-t0:.0f}s)")
