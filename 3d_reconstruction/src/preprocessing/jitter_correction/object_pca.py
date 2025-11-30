# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np



# ================================================================
# 1. Section: PCA for Object Orientation
# ================================================================
def get_object_orientation(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 1. Get coordinates of foreground pixels
    coords = np.column_stack(np.nonzero(mask))

    # 2. Compute centroid and center data
    centroid = coords.mean(axis=0)
    X = coords - centroid

    # 3. Covariance matrix and Eigen decomposition
    C = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C) 

    # 4. Sort by decreasing eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return centroid, eigvals, eigvecs