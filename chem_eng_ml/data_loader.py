import h5py
import numpy as np


def load_features(h5_path, index=None):
    f = h5py.File(h5_path, 'r')
    keys = index
    if index is None:
        keys = f.keys()
    dataset = np.zeros([len(keys), 129, 384, 1], dtype=np.int)
    for i, key in enumerate(keys):
        dataset[i] = np.reshape(f[str(key)][:], [129, 384, 1])
    return dataset
