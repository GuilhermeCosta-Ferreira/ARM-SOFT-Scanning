# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
from .object_pca import get_object_orientation



# ================================================================
# 1. Section: Jitter Correction via PCA Alignment
# ================================================================
def fix_jitter(mask):
    H, W = mask.shape

    # Get PCA Alignment Parameters
    theta, eigvecs, centroid = get_pca_alignment(mask)

    # Generate Centered Grid    
    grid = generate_grid(H, W)

    # Inverse Rotation
    rotated_grid = apply_inverse_rotation(theta, grid, centroid)

    # NN Sampling
    mask_aligned = nearest_neighbour_sampling(mask, rotated_grid, H, W)

    return mask_aligned, centroid, eigvecs


# ──────────────────────────────────────────────────────
# 1.1 Subsection: PCA Alignment to Image Axes
# ──────────────────────────────────────────────────────
def get_pca_alignment(mask: np.ndarray) -> tuple:
    centroid, eigvals, eigvecs = get_object_orientation(mask)

    # Principal axis as (y, x)
    v = eigvecs[:, 0]
    vx, vy = v
    vx = -vx 

    theta = np.arctan2(vy, vx)

    return theta, eigvecs, centroid


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Generate Grid Centered at Given Point
# ──────────────────────────────────────────────────────
def generate_grid(H: int, W: int) -> tuple:
    cy_im = (H - 1) / 2.0
    cx_im = (W - 1) / 2.0

    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    x0 = X - cx_im
    y0 = Y - cy_im

    return x0, y0


# ──────────────────────────────────────────────────────
# 1.3 Subsection: Inverse Rotation
# ──────────────────────────────────────────────────────
def apply_inverse_rotation(theta: float, grid: tuple, centroid: tuple) -> tuple:
    # Unpacks the data
    x0, y0 = grid
    cy, cx = centroid

    # Inverse rotation: map output -> input
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # [x_rel; y_rel] = R(theta) * [x0; y0]
    x_rel = cos_t * x0 - sin_t * y0
    y_rel = sin_t * x0 + cos_t * y0

    # Add original centroid to get input coords
    x_in = x_rel + cx
    y_in = y_rel + cy

    return x_in, y_in


# ──────────────────────────────────────────────────────
# 1.4 Subsection: NN Sampling
# ──────────────────────────────────────────────────────
def nearest_neighbour_sampling(mask: np.ndarray, rotated_grid: tuple, H: int, W: int) -> np.ndarray:
    x_in, y_in = rotated_grid

    # Nearest-neighbor sampling
    x_in_round = np.round(x_in).astype(int)
    y_in_round = np.round(y_in).astype(int)

    mask_aligned = np.zeros_like(mask, dtype=bool)
    inside = (
        (x_in_round >= 0) & (x_in_round < W) &
        (y_in_round >= 0) & (y_in_round < H)
    )

    mask_aligned[inside] = mask[y_in_round[inside], x_in_round[inside]]

    return mask_aligned

