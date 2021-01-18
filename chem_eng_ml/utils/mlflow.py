import numpy as np
import mlflow
import tempfile
import os


def log_array(a, name: str):
    """Logs an array as artifact to the currently active run.
    Args:
        a: (anything convertible to np.array) Array to store as artifact.
        name: (str) Name to give the folder and file under th current mlflow run.
    """
    a = np.array(a)
    tmpdir = tempfile.mkdtemp()
    save_path = os.path.join(tmpdir, name+".npy")
    np.save(save_path, a)
    mlflow.log_artifact(save_path, 'meta_data')
    return

