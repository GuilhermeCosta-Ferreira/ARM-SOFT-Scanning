# ================================================================
# 0. Section: Imports
# ================================================================
import os
import trimesh

import numpy as np
from skimage import measure



# ================================================================
# 1. Section: Mesh Building
# ================================================================
def build_mesh(voxel_array: np.ndarray, **kwargs) -> trimesh.Trimesh:
    """
    Build a 3D mesh from a voxel array using marching cubes algorithm.

    This function converts a 3D voxel array into a triangular mesh by applying
    the marching cubes algorithm. It binarizes the input voxel data, extracts
    the surface mesh, and exports it as a PLY file to a specified directory.

    Parameters
    ----------
    voxel_array : np.ndarray
        3D numpy array representing voxel data. Values greater than 0 are
        considered as part of the object to be meshed.
    **kwargs : dict
        Additional keyword arguments:
        - file_name : str, optional
            Name of the output PLY file. Default is 'brain_mesh.ply'.
        - verbose : int, optional
            Verbosity level for output messages. Default is 1.
        - folder : str, optional
            Directory name where the mesh file will be saved. Default is 'blender'.

    Returns
    -------
    trimesh.Trimesh
        A trimesh object containing the extracted 3D mesh with vertices and faces.

    Notes
    -----
    - The function uses scikit-image's marching cubes algorithm with a level of 0.5
    - Input voxel array is binarized (values > 0 become 1, others become 0)
    - Creates the output directory if it doesn't exist
    - Mesh is exported in PLY format which is compatible with Blender and other 3D software
    - Mesh statistics (vertex count and face count) are reported when verbose > 0
    """
    # Kwarg defaults
    file_name = kwargs.get('file_name', 'brain_mesh.ply')
    verbose = kwargs.get('verbose', 1)
    folder = kwargs.get('folder', 'blender')

    # Prepare Data
    binary = np.where(voxel_array > 0, 1, 0)

    # Extract surface mesh
    verts, faces, _, _ = measure.marching_cubes(binary, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Make blender folder
    os.makedirs(folder, exist_ok=True)

    # Export mesh as PLY format
    mesh.export(f"{folder}/{file_name}")
    if verbose > 0: print(f"✅ Finished exporting {folder}/{file_name}\nMesh vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
    return mesh
