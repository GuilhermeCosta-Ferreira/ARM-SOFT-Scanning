# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np

from stl import mesh



# ================================================================
# 1. Section: STL Loading
# ================================================================
def load_stl_as_array(file_path: str, resolution: int | float) -> np.ndarray:
    """
    Load an STL file and convert it to a 3D numpy array representation.
    This function loads an STL mesh file, determines its bounding box, and converts
    it to a discretized 3D array representation at the specified resolution.

    Parameters
    ----------
    file_path : str
        Path to the STL file to be loaded.
    resolution : int | float
        The resolution for the 3D array conversion.
        - If `int`: used directly as the grid resolution
        - If `float`: interpreted as physical spacing, and resolution is calculated
          as ceil(max_dimension / spacing)

    Returns
    -------
    np.ndarray
        A 3D numpy array representing the discretized STL mesh at the specified
        resolution.

    Notes
    -----
    - When resolution is provided as a float, it represents the physical spacing
      between grid points in STL units.
    - The function calculates the bounding box from all vertices in the mesh.
    - The actual conversion to array is delegated to the `_extract_stl_array`
      helper function.

    Examples
    --------
    >>> # Load with integer resolution
    >>> array = load_stl_as_array("model.stl", 100)
    >>> # Load with physical spacing
    >>> array = load_stl_as_array("model.stl", 0.1)
    """
    # Load the STL file and extract vertices
    stl_mesh = mesh.Mesh.from_file(file_path)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    
    # Find bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    if(isinstance(resolution, float)):
        dims = max_coords - min_coords  # physical size (in STL units)
        resolution = np.ceil(dims / resolution).astype(int)[0]

    return convert_stl_to_array(stl_mesh, resolution, (min_coords, max_coords))

def convert_stl_to_array(stl_mesh: mesh.Mesh, resolution: int, coords: tuple) -> np.ndarray:
    """
    Convert an STL mesh to a 3D voxel array representation.
    This function performs a basic voxelization of an STL mesh by creating a 3D boolean
    array where True values indicate occupied voxels. The method uses a simple bounding
    box approach for each triangle in the mesh.

    Parameters
    ----------
    stl_mesh : mesh.Mesh
        The input STL mesh object containing triangle vectors to be voxelized.
    resolution : int
        The resolution of the output voxel grid in each dimension. The resulting
        array will have shape (resolution, resolution, resolution).
    coords : tuple
        A tuple containing (min_coords, max_coords) where:
        - min_coords : array-like of shape (3,)
            Minimum coordinates [x, y, z] of the bounding box
        - max_coords : array-like of shape (3,)
            Maximum coordinates [x, y, z] of the bounding box

    Returns
    -------
    tuple[np.ndarray, float]
        A tuple containing:
        - voxel_array : np.ndarray of shape (resolution, resolution, resolution)
            Boolean array where True indicates occupied voxels
        - voxel_size : float
            The size of each voxel in world units

    Notes
    -----
    - This implementation uses a basic triangle bounding box approach, which may
      result in false positives (marking voxels as occupied when they only
      intersect the triangle's bounding box but not the triangle itself).
    - Triangle indices are clamped to stay within the voxel grid bounds.
    - The voxel size is calculated as the maximum dimension of the bounding box
      divided by the resolution.

    Examples
    --------
    >>> import numpy as np
    >>> from stl import mesh
    >>> # Load STL mesh
    >>> stl_mesh = mesh.Mesh.from_file('model.stl')
    >>> # Define bounding box
    >>> min_coords = np.array([0, 0, 0])
    >>> max_coords = np.array([10, 10, 10])
    >>> coords = (min_coords, max_coords)
    >>> # Convert to voxel array
    >>> voxel_array, voxel_size = convert_stl_to_array(stl_mesh, 64, coords)
    >>> voxel_array.shape
    (64, 64, 64)
    >>> voxel_array.dtype
    dtype('bool')
    """
    # Simple inside/outside test (basic approach)
    min_coords, max_coords = coords
    voxel_array = np.zeros((resolution, resolution, resolution), dtype=bool)
    
    for triangle in stl_mesh.vectors:
        # Basic triangle bounding box check
        tri_min = np.min(triangle, axis=0)
        tri_max = np.max(triangle, axis=0)
        
        # Convert to voxel indices
        min_idx = np.floor((tri_min - min_coords) / (max_coords - min_coords) * (resolution - 1)).astype(int)
        max_idx = np.ceil((tri_max - min_coords) / (max_coords - min_coords) * (resolution - 1)).astype(int)
        
        # Clamp indices
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, resolution - 1)
        
        # Mark voxels in bounding box as occupied
        voxel_array[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1] = True
        voxel_size = np.max(max_coords - min_coords) / resolution

    return voxel_array, voxel_size
