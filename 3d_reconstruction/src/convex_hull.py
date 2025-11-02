# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from tqdm import tqdm

from .camera_data.camera_data import CamerasData
from .scans import Scans
from .mesh import build_mesh


# ================================================================
# 1. Section: Multi-View Reconstruction
# ================================================================
def multi_view_reconstruction(camera_data: CamerasData, scans: Scans, resolution: int = 200, reconstruction_range: float = 0.45, **kwargs) -> tuple[np.ndarray, float]:
    # Kwargs initialization
    get_mesh = kwargs.get('get_mesh', False)
    mesh_filename = kwargs.get('mesh_filename', f"XXd_convex_hull_mesh.ply")
    mesh_folder = kwargs.get('mesh_folder', 'reconstruction_mesh')

    # Threshold for convex hull (number of scans - 1)
    threshold = len(scans.nr_positions) - 1

    # Projection of all scans
    full_volume, voxel_size = project_all_scans(camera_data, scans, resolution=resolution, reconstruction_range=reconstruction_range)

    # Threshold the sum of the projection of all scans to get convex hull
    _, convex_hull = threshold_volume(full_volume, threshold=threshold)
    
    # Build mesh if required
    if get_mesh: build_mesh(convex_hull, file_name=mesh_filename, folder=mesh_folder, verbose=1)

    return convex_hull, voxel_size



# ================================================================
# 2. Section: All Scan Calculations
# ================================================================
def project_all_scans(camera_data: CamerasData, scans: Scans, resolution: int=200, reconstruction_range: float=0.45) -> tuple[list[np.ndarray], float]:
    """
    Project all scans from multiple cameras into 3D volumes.

    This function processes scanning data from multiple cameras by projecting each
    camera's scan data into a 3D volume representation. It iterates through all
    available cameras and creates volumetric reconstructions for each one.

    Parameters
    ----------
    camera_data : CamerasData
        Object containing camera configuration and calibration data for all cameras
        in the scanning system.
    scans : Scans
        Collection of scan data from all cameras to be processed and projected.
    resolution : int, optional
        The resolution for the 3D volume reconstruction grid. Higher values create
        more detailed volumes but require more memory and computation time.
        Default is 200.
    reconstruction_range : float, optional
        The spatial range (in meters) for the 3D reconstruction volume. Defines
        the physical extent of the reconstructed space. Default is 0.45.

    Returns
    -------
    tuple[list[np.ndarray], float]
        A tuple containing:
        - list[np.ndarray]: List of 3D volume arrays, one for each camera
        - float: The voxel size in meters for the reconstructed volumes

    Examples
    --------
    >>> volumes, voxel_size = project_all_scans(
    ...     camera_data, scans, resolution=300, reconstruction_range=0.5
    ... )
    >>> print(f"Generated {len(volumes)} volumes with voxel size {voxel_size:.4f}m")

    Notes
    -----
    - All cameras produce volumes with the same voxel size and spatial extent
    - The function prints a confirmation message upon successful completion
    - Memory usage scales with resolution³ and number of cameras
    """
    full_volume = []

    # RECONSTRUCTION LOOP
    for camera_nr in range(camera_data.nr_cameras):
        camera_volume, voxel_size = project_scan(camera_data, scans, camera_nr, resolution=resolution, reconstruction_range=reconstruction_range)
        full_volume.append(camera_volume)

    print(f"\n✅ All Scans Projected into Volume with Resolution {resolution} and Voxel Size {np.round(voxel_size, 4)} m")
    return np.array(full_volume), voxel_size

def threshold_volume(full_volume: np.ndarray, threshold: float = 11) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a multi-view convex hull from a 3D volume by summing across views and thresholding.

    This function takes a 4D volume array (views × depth × height × width) and computes
    a convex hull approximation by summing across all views and applying a threshold.
    Points with summed values above the threshold are considered part of the convex hull.

    Parameters
    ----------
    full_volume : np.ndarray
        A 4D numpy array of shape (n_views, depth, height, width) representing
        multiple views of a 3D volume. Each view contributes to the final hull computation.
    threshold : float, optional
        The threshold value for determining convex hull membership, by default 11.
        Points with summed values greater than this threshold are included in the hull.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - convex_hull : np.ndarray
            A 3D binary array of shape (depth, height, width) where 1 indicates
            points inside the convex hull and 0 indicates points outside.
        - sum_volume : np.ndarray
            A 3D array of shape (depth, height, width) containing the summed
            values across all views before thresholding.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a sample 4D volume with 3 views
    >>> volume = np.random.rand(3, 10, 10, 10)
    >>> hull, sum_vol = multi_view_convex_hull(volume, threshold=1.5)
    >>> hull.shape
    (10, 10, 10)
    >>> np.max(hull)
    1

    Notes
    -----
    - The function assumes the first axis (axis=0) represents different views.
    - The resulting convex hull is an approximation based on the threshold criterion.
    - Higher threshold values result in smaller, more conservative convex hulls.
    """
    sum_volume = np.sum(full_volume, axis=0)
    convex_hull = np.where(sum_volume > threshold, 1, 0)

    return convex_hull, sum_volume


# ================================================================
# 3. Section: Scan Calculations
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
    for idx, x in enumerate(tqdm(coord_range, desc=f"🔄 Processing Slices fo Camera Nr {camera_nr}")):
        for idy, y in enumerate(coord_range):
            for idz, z in enumerate(coord_range):
                point_3d = np.array([x, y, z, 1])
                u,v = project_world_to_image(point_3d, P)
                
                is_valid = (u is not None and v is not None) and (0 <= u < scan.shape[1]) and (0 <= v < scan.shape[0])
                if is_valid: camera_volume[idx, idy, idz] += scan[v, u]
    
    return camera_volume, voxel_size


# ──────────────────────────────────────────────────────
# 3.1 Subsection: Scan Projection Helpers
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
# 4. Section: Point Calculations
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