# ================================================================
# 0. Section: Imports
# ================================================================
import numpy as np
from scipy.ndimage import center_of_mass, affine_transform



# ================================================================
# 1. Section: Moving Volumes
# ================================================================
def center_volume(volume: np.ndarray) -> np.ndarray:
    """
    Center a 3D volume by translating it so its center of mass aligns with the geometric center.
    This function calculates the center of mass of the input volume, determines the 
    translation needed to move it to the geometric center of the volume, and applies 
    an affine transformation to perform the centering operation.

    Parameters
    ----------
    volume : np.ndarray
        A 3D numpy array representing the volume to be centered. The volume should
        contain numerical data where the center of mass can be meaningfully calculated.

    Returns
    -------
    np.ndarray
        A new 3D numpy array of the same shape as the input volume, but translated
        so that its center of mass is at the geometric center of the volume.

    Notes
    -----
    - The center of mass is calculated using scipy.ndimage.center_of_mass and 
      rounded to the nearest integer coordinates.
    - The geometric center is calculated as half the volume dimensions.
    - The translation is applied using scipy.ndimage.affine_transform with 
      order=0 (nearest neighbor interpolation).
    - A 4x4 homogeneous transformation matrix is constructed for the translation.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple 3D volume with an off-center mass
    >>> volume = np.zeros((10, 10, 10))
    >>> volume[2:4, 2:4, 2:4] = 1
    >>> centered = center_volume(volume)
    >>> # The mass should now be closer to the center of the volume
    """
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



# ================================================================
# 2. Section: Shape Mismatch
# ================================================================
def volume_reshape(target_volume: np.ndarray, source_volume: np.ndarray) -> np.ndarray:
    """
    Reshape a target volume to match the dimensions of a source volume by padding with zeros.

    This function takes a target volume and reshapes it to match the dimensions of a 
    source volume by copying the target volume into the beginning of a new array with 
    the same shape as the source volume. The remaining space is filled with zeros.

    Parameters
    ----------
    target_volume : np.ndarray
        The volume to be reshaped. Must have dimensions smaller than or equal to 
        the source volume in all axes.
    source_volume : np.ndarray
        The reference volume whose shape will be used for the output. Must be 
        larger than or equal to the target volume in all dimensions.

    Returns
    -------
    np.ndarray
        A new array with the same shape and dtype as source_volume, containing 
        the target_volume data in the top-left corner (indices [0:target_shape[0], 
        0:target_shape[1], 0:target_shape[2]]) and zeros elsewhere.

    Raises
    ------
    AssertionError
        If the source volume is smaller than the target volume in any dimension.

    Examples
    --------
    >>> import numpy as np
    >>> target = np.ones((2, 2, 2))
    >>> source = np.zeros((4, 4, 4))
    >>> result = volume_reshape(target, source)
    >>> result.shape
    (4, 4, 4)
    >>> result[0, 0, 0]
    1.0
    >>> result[3, 3, 3]
    0.0

    Notes
    -----
    - The function creates a new array with the same dtype as the source volume.
    - Only the first target_shape[i] elements along each axis i are filled with 
      target volume data; the rest remain as zeros.
    - This is useful for padding smaller volumes to match larger reference dimensions
      in 3D image processing or volumetric data analysis.
    """
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