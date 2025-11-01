# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
from tqdm import tqdm
from .camera_data.camera_data import CamerasData
from .scans import Scans



# ================================================================
# 1. Section: Scan Calculations
# ================================================================
def project_scan(camera_data: CamerasData, scans: Scans, camera_nr: int, resolution: int, reconstruction_range: float) -> tuple[np.ndarray, float]:
    """
    Project a 3D scan data into a volumetric representation using camera projection.
    This function takes camera calibration data and scan measurements to create
    a 3D volume by projecting world coordinates to image coordinates and
    accumulating scan values at corresponding voxels.

    Parameters
    ----------
    camera_data : CamerasData
        Container holding camera calibration parameters including projection
        matrices for multiple cameras.
    camera_nr : int
        The index/identifier of the camera to use for projection.
    scans : Scans
        Container holding scan data for multiple cameras/positions.
    resolution : int
        The number of voxels along each axis of the 3D reconstruction volume.
        Creates a resolution×resolution×resolution grid.
    reconstruction_range : float
        The physical extent of the reconstruction volume in world coordinates.
        Defines the cube size from -reconstruction_range to +reconstruction_range.

    Returns
    -------
    tuple[np.ndarray, float]
        A tuple containing:
        - camera_volume : np.ndarray
            3D volume of shape (resolution, resolution, resolution) with
            accumulated scan values at each voxel position.
        - voxel_size : float
            The physical size of each voxel in world coordinate units.

    Notes
    -----
    - Uses the camera's projection matrix P to transform 3D world coordinates
      to 2D image coordinates.
    - Only voxels that project to valid image coordinates (within scan bounds)
      contribute to the volume.
    - Progress is displayed using tqdm for the outer loop over x-coordinates.
    - The function assumes homogeneous coordinates (4D) for 3D points.

    Examples
    --------
    >>> volume, voxel_size = project_scan(cameras, scans, 0, 128, 5.0)
    >>> volume.shape
    (128, 128, 128)
    >>> # voxel_size represents the spacing between adjacent voxels
    """
    # Data extraction
    scan = scans.scan(camera_nr)
    P = camera_data.P(camera_nr)

    # Generate world coordinate grid
    coord_matrix_map, coord_range = generate_worl_grid_cube(resolution, reconstruction_range)
    camera_volume = np.zeros(coord_matrix_map.shape[:3], dtype=int)
    voxel_size = coord_range[1] - coord_range[0]

    # Project each voxel to image and accumulate scan values
    for idx, x in enumerate(tqdm(coord_range, desc="🔄 Processing Slices")):
        for idy, y in enumerate(coord_range):
            for idz, z in enumerate(coord_range):
                point_3d = np.array([x, y, z, 1])
                u,v = project_world_to_image(point_3d, P)

                is_valid = (u is not None and v is not None) and (0 <= u < scan.shape[0]) and (0 <= v < scan.shape[1])
                if is_valid: camera_volume[idx, idy, idz] += scan[u, v]
    
    return camera_volume, voxel_size


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Scan Projection Helpers
# ──────────────────────────────────────────────────────
def generate_worl_grid_cube(resolution: int, range_value: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 3D grid of world coordinates in a cubic volume.

    Creates a uniform 3D grid of coordinates within a cubic region centered at the origin,
    along with the coordinate range used for generation.

    Parameters
    ----------
    resolution : int
        The number of grid points along each axis. The total grid will have
        resolution³ points.
    range_value : float
        The half-extent of the cubic region. The grid spans from -range_value
        to +range_value along each axis.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - coord_matrix_map : np.ndarray
            A 4D array of shape (resolution, resolution, resolution, 3) where
            the last dimension contains the [x, y, z] coordinates for each grid point.
        - coord_range : np.ndarray
            A 1D array of shape (resolution,) containing the coordinate values
            along each axis from -range_value to +range_value.

    Examples
    --------
    >>> coord_map, coord_range = generate_worl_grid_cube(10, 1.0)
    >>> coord_map.shape
    (10, 10, 10, 3)
    >>> coord_range.shape
    (10,)
    >>> coord_range[0], coord_range[-1]
    (-1.0, 1.0)

    Notes
    -----
    - The coordinate system uses 'ij' indexing for meshgrid, meaning the first
      dimension corresponds to rows (X), second to columns (Y), and third to
      depth (Z).
    - The grid is uniformly spaced with (resolution-1) intervals between
      -range_value and +range_value.
    """
    coord_range = np.linspace(-range_value, range_value, resolution)

    X, Y, Z = np.meshgrid(coord_range, coord_range, coord_range, indexing='ij')
    coord_matrix_map = np.zeros((resolution, resolution, resolution, 3))
    coord_matrix_map[:,:,:,0] = X
    coord_matrix_map[:,:,:,1] = Y
    coord_matrix_map[:,:,:,2] = Z

    return coord_matrix_map, coord_range



# ================================================================
# 2. Section: Point Calculations
# ================================================================
def project_world_to_image(point_3d: np.ndarray, P: np.ndarray) -> tuple[int, int] | tuple[None, None]:
    """
    Project a 3D point in world coordinates to 2D image coordinates using camera projection matrix.

    Parameters
    ----------
    point_3d : np.ndarray
        3D point in world coordinates. Should be a homogeneous coordinate vector
        of shape (4,) or (3,) representing [x, y, z, 1] or [x, y, z].
    P : np.ndarray
        Camera projection matrix of shape (3, 4) that transforms world coordinates
        to image coordinates. This matrix combines intrinsic and extrinsic parameters.

    Returns
    -------
    tuple[int, int] | tuple[None, None]
        Image coordinates (u, v) as integer pixel positions if projection is valid.
        Returns (None, None) if the point projects to infinity (z-coordinate is 0).

    Notes
    -----
    - The function performs perspective projection using homogeneous coordinates.
    - Division by zero is handled by returning None values when z-coordinate is 0.
    - The resulting coordinates are rounded to integer pixel positions.
    - Points behind the camera (negative z after projection) are still projected
      but may result in negative or unexpected coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> point_3d = np.array([1.0, 2.0, 5.0, 1.0])  # Homogeneous coordinates
    >>> P = np.eye(3, 4)  # Simple projection matrix
    >>> u, v = project_world_to_image(point_3d, P)
    >>> print(f"Pixel coordinates: ({u}, {v})")
    Pixel coordinates: (0, 0)
    """
    x1, x2, x3 = P @ point_3d
    if x3 == 0: return None, None

    u = int(x1 / x3)
    v = int(x2 / x3)
    return u, v