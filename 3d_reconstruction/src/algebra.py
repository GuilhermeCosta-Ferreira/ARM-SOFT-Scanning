# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Matrix Inversions
# ================================================================
def invert_extrinsic_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of an extrinsic transformation matrix.

    This function efficiently inverts a 4×4 extrinsic transformation matrix
    using the special structure of SE(3) transformations. For a transformation
    matrix T = [R | t; 0 0 0 1] where R is a 3×3 rotation matrix and t is
    a 3×1 translation vector, the inverse is T^(-1) = [R^T | -R^T*t; 0 0 0 1].

    Parameters
    ----------
    matrix : np.ndarray
        A 4×4 extrinsic transformation matrix of shape (4, 4) representing
        the transformation from world coordinates to camera coordinates.
        The matrix should have the structure:
        [[R11, R12, R13, tx],
            [R21, R22, R23, ty],
            [R31, R32, R33, tz],
            [0,   0,   0,   1 ]]

    Returns
    -------
    np.ndarray
        The inverse transformation matrix of shape (4, 4), representing
        the transformation from camera coordinates to world coordinates.

    Notes
    -----
    - This function assumes the input matrix represents a valid SE(3) transformation
        (i.e., the rotation part is orthogonal and the bottom row is [0, 0, 0, 1]).
    - The inversion uses the property that for rotation matrices, R^(-1) = R^T.
    - This is more efficient than computing the general matrix inverse.

    Examples
    --------
    >>> import numpy as np
    >>> # Identity transformation
    >>> T = np.eye(4)
    >>> T_inv = invert_extrinsic_matrix(T)
    >>> np.allclose(T_inv, T)
    True
    >>> # Translation only
    >>> T = np.eye(4)
    >>> T[:3, 3] = [1, 2, 3]
    >>> T_inv = invert_extrinsic_matrix(T)
    >>> T_inv[:3, 3]
    array([-1., -2., -3.])
    """
    # Extract rotation and translation components
    R = matrix[:3, :3]
    t = matrix[:3, 3]

    # Compute inverse using properties of rotation matrices
    R_inv = R.T
    t_inv = -R_inv @ t

    # Construct the inverse transformation matrix
    M_inv2 = np.eye(4)
    M_inv2[:3, :3] = R_inv
    M_inv2[:3, 3] = t_inv

    return M_inv2

def is_rotation_matrix(R, tol=1e-6):
    """Quick sanity check for a proper rotation matrix."""
    if R.shape != (3, 3):
        return False
    should_be_I = R.T @ R
    I = np.eye(3)
    return np.linalg.norm(should_be_I - I) < tol and np.linalg.det(R) > 0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def euler_from_matrix(R, degrees=False):
    R = np.asarray(R, dtype=float)
    if not is_rotation_matrix(R): raise ValueError("Input is not a proper rotation matrix (orthogonal with det +1).")

    # Numerical tolerance for gimbal detection
    eps = 1e-9

    sy = -R[2,0]
    sy = clamp(sy, -1.0, 1.0)  # numerical safety
    y = np.arcsin(sy)
    cy = np.cos(y)

    if abs(cy) > eps:
        x = np.arctan2(R[2,1], R[2,2])
        z = np.arctan2(R[1,0], R[0,0])
    else:
        # Gimbal lock (cy ~ 0): y ≈ ±pi/2
        # Collapse one DOF; choose z = 0 and compute x from a stable pair
        x = np.arctan2(-R[1,2], R[1,1])
        z = 0.0

    angles = (x, y, z)

    if degrees:
        angles = tuple(np.degrees(a) for a in angles)

    return angles


