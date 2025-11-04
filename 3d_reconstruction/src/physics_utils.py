import numpy as np
from scipy.ndimage import center_of_mass, affine_transform

def center_volume(volume: np.ndarray) -> np.ndarray:
    # Calculate the center of mass of the volume
    center_of_mass_var = np.round(center_of_mass(volume)).astype(int)
    center = np.array(volume.shape) / 2
    difference = np.append(center_of_mass_var - center, 1)
    
    # Builds the translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:,3] = difference

    # Apply the translation to the volume
    centered_volume = affine_transform(volume, translation_matrix, order=0)

    return centered_volume

def volume_reshape(target_volume: np.ndarray, source_volume: np.ndarray) -> np.ndarray:
    # Get shapes
    target_shape = target_volume.shape
    source_shape = source_volume.shape

    # Check if source is larger than target in all dimensions
    assert all(s >= t for s, t in zip(source_shape, target_shape)), "Source volume must be larger than or equal to target volume in all dimensions"

    # Create output array with target shape
    reshaped_volume = np.zeros_like(source_volume)

    # Copy target volume into the center of the source volume
    reshaped_volume[:target_shape[0], :target_shape[1], :target_shape[2]] = target_volume

    return reshaped_volume