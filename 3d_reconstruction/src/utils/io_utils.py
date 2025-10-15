# ================================================================
# 0. Section: Imports
# ================================================================
import os
import cv2
import numpy as np



# ================================================================
# 1. Section: Data Extraction
# ================================================================
def load_scans(path: str, file_type: str = ".png") -> tuple[np.ndarray, np.ndarray]:
    """
    Load and process scan images from a specified directory.

    This function reads all image files of a specified type from a given directory path,
    loads them as grayscale images using OpenCV, and returns both the image data and
    corresponding filenames as numpy arrays.

    Parameters
    ----------
    path : str
        The directory path containing the scan images to be loaded.
    file_type : str, optional
        The file extension to filter for when loading images. Default is ".png".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays:
        - First array: Stack of loaded grayscale images with shape (n_images, height, width)
        - Second array: Array of corresponding filenames as strings

    Notes
    -----
    - All images are loaded in grayscale mode using cv2.IMREAD_GRAYSCALE
    - Only files with the specified file extension are processed
    - The function uses os.listdir() so the order of loaded images depends on the
      filesystem's directory listing order
    - Requires OpenCV (cv2) and numpy to be imported
    """
    scans = []
    file_names = []

    # Check each file inside this folder and open it with opencv
    for file in os.listdir(path):
        if file.endswith(file_type):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            scans.append(img)
            file_names.append(file)

    return np.array(scans), np.array(file_names)



# ================================================================
# 2. Section: File Cleaning
# ================================================================
def clean_files(files: list | np.ndarray | str) -> None:
    if(isinstance(files, str)):
        files = [files]
        
    for file in files:
        if os.path.exists(file):
            os.remove(file)