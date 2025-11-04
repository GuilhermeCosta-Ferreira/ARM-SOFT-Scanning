import numpy as np

from stl import mesh

def load_stl_as_array(file_path: str, resolution: int | float) -> np.ndarray:
    # Load the STL file and extract vertices
    stl_mesh = mesh.Mesh.from_file(file_path)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    
    # Find bounding box
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    if(isinstance(resolution, float)):
        dims = max_coords - min_coords  # physical size (in STL units)
        resolution = np.ceil(dims / resolution).astype(int)[0]

    return _extract_stl_array(stl_mesh, resolution, (min_coords, max_coords))

def _extract_stl_array(stl_mesh: mesh.Mesh, resolution: int, coords: tuple) -> np.ndarray:
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
