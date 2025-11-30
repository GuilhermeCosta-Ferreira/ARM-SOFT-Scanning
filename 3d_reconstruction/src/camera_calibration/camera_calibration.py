import numpy as np
import cv2
import glob
import os
import pickle

# Camera calibration parameters
# You can modify these variables as needed
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners per chessboard row and column
SQUARE_SIZE = 2         # Size of a square in centimeters
CALIBRATION_IMAGES_PATH = './3d_reconstruction/figures/calibration_pictures/*.jpg'  # Path to calibration images
OUTPUT_DIRECTORY = './3d_reconstruction/figures/calibration_pictures/output'  # Directory to save calibration results
SAVE_UNDISTORTED = True   # Whether to save undistorted images

def calibrate_camera() -> tuple:
    """
    Calibrate the camera using chessboard images found by a glob pattern.
    This function searches for chessboard patterns in images matched by
    CALIBRATION_IMAGES_PATH, refines detected corner locations, accumulates
    object and image point correspondences, and computes the camera intrinsic
    matrix and distortion coefficients via OpenCV's cv2.calibrateCamera.
    Detected corners are drawn and saved to OUTPUT_DIRECTORY together with a
    pickle containing the calibration results and plaintext camera/disortion
    files.

    Parameters
    ----------
    None
        This function relies on module-level configuration constants:
        - CHESSBOARD_SIZE : tuple[int, int]
            Number of inner corners per chessboard row and column (cols, rows).
        - SQUARE_SIZE : float
            Physical size of one chessboard square (in chosen length units).
        - CALIBRATION_IMAGES_PATH : str
            Glob pattern pointing to calibration images (e.g. "data/*.png").
        - OUTPUT_DIRECTORY : str
            Directory where corner images and calibration results are saved.

    Returns
    -------
    ret : float | None
        RMS re-projection error returned by cv2.calibrateCamera. Returns None
        when calibration could not be performed (e.g. no images or no detections).
    mtx : np.ndarray | None
        Camera intrinsic (3×3) matrix, or None if calibration failed.
    dist : np.ndarray | None
        Distortion coefficients (e.g. k1, k2, p1, p2, k3, ...), or None on failure.
    rvecs : list[np.ndarray] | None
        List of rotation vectors for each successful calibration image, or None.
    tvecs : list[np.ndarray] | None
        List of translation vectors for each successful calibration image, or None.

    Raises
    ------
    OSError
        If creating OUTPUT_DIRECTORY or writing output files fails (permission
        problems, full disk, etc.).
    cv2.error
        If an underlying OpenCV call (image read, corner finding, calibration)
        raises an error.
    ValueError
        If input images have inconsistent/unsupported shapes that prevent
        calibration (this is typically forwarded from OpenCV).

    Examples
    --------
    >>> # Use module-level configuration before calling
    >>> ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    >>> if ret is None:
    ...     print("Calibration failed or no valid chessboard detections.")
    ... else:
    ...     print(f"RMS reprojection error: {ret:.4f}")
    ...     print("Camera matrix:\n", mtx)

    Notes
    -----
    - The function prints diagnostic messages for each processed image indicating
      whether a chessboard was found. If no images are found matching the glob
      pattern or no patterns are detected in any image, it returns five Nones.
    - Saved outputs (in OUTPUT_DIRECTORY):
      - corners_<original_filename> : images with drawn chessboard corners
      - calibration_data.pkl : pickle with keys 'camera_matrix', 'distortion_coefficients',
        'rotation_vectors', 'translation_vectors', 'reprojection_error'
      - camera_matrix.txt : plain-text camera matrix
      - distortion_coefficients.txt : plain-text distortion coefficients
    - Object points are prepared using CHESSBOARD_SIZE and scaled by SQUARE_SIZE,
      so the returned translation vectors are in the same physical units as SQUARE_SIZE.
    - The length/format of the distortion vector depends on OpenCV's selected model;
      users should inspect dist.shape before using.
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    
    # Scale object points by square size (for real-world measurements)
    objp = objp * SQUARE_SIZE
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    images = glob.glob(CALIBRATION_IMAGES_PATH)
    
    if not images:
        print(f"No calibration images found at {CALIBRATION_IMAGES_PATH}")
        return None, None, None, None, None
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    print(f"Found {len(images)} calibration images")
    
    # Process each calibration image
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            
            # Save image with corners drawn
            output_img_path = os.path.join(OUTPUT_DIRECTORY, f'corners_{os.path.basename(fname)}')
            cv2.imwrite(output_img_path, img)
            
            print(f"Processed image {idx+1}/{len(images)}: {fname} - Chessboard found")
        else:
            print(f"Processed image {idx+1}/{len(images)}: {fname} - Chessboard NOT found")
    
    if not objpoints:
        print("No chessboard patterns were detected in any images.")
        return None, None, None, None, None
    
    print("Calibrating camera...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Save calibration results
    calibration_data = {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        'rotation_vectors': rvecs,
        'translation_vectors': tvecs,
        'reprojection_error': ret
    }
    
    with open(os.path.join(OUTPUT_DIRECTORY, 'calibration_data.pkl'), 'wb') as f:
        pickle.dump(calibration_data, f)
    
    # Save camera matrix and distortion coefficients as text files
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'camera_matrix.txt'), mtx)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'distortion_coefficients.txt'), dist)
    
    print(f"Calibration complete! RMS re-projection error: {ret}")
    print(f"Results saved to {OUTPUT_DIRECTORY}")
    
    return ret, mtx, dist, rvecs, tvecs

def undistort_images(mtx: np.ndarray, dist: np.ndarray) -> None:
    """
    Undistort and save calibration images using a provided camera intrinsic matrix
    and distortion coefficients.
    This function searches for images using a module-level glob pattern, undistorts
    each found image with OpenCV using an optimal new camera matrix, optionally
    crops the result to the returned ROI, and writes the undistorted images to an
    "undistorted" subdirectory under a module-level output directory. Progress and
    diagnostic messages are printed to standard output.

    Parameters
    ----------
    mtx : np.ndarray
        Camera intrinsic matrix (3×3) as returned by camera calibration.
    dist : np.ndarray
        Distortion coefficients (1×N or Nx1), e.g. (k1, k2, p1, p2[, k3, ...]).

    Returns
    -------
    None
        The function has only side effects (reading input images, writing output
        images, and printing progress) and returns None.
    Raises
    ------
    ValueError
        If an image file is found but cannot be read (cv2.imread returns None) or
        the image array does not have a valid shape for processing.
    OSError
        If creating the output directory or writing image files fails due to OS
        level errors (permissions, disk full, etc.).
    RuntimeError
        If an OpenCV routine (for example cv2.getOptimalNewCameraMatrix or
        cv2.undistort) fails catastrophically; OpenCV-specific exceptions may also
        propagate.

    Notes
    -----
    - The function's behavior is controlled by module-level configuration variables:
      - SAVE_UNDISTORTED: when False, the function returns immediately without
        processing.
      - CALIBRATION_IMAGES_PATH: glob pattern used to discover input images.
      - OUTPUT_DIRECTORY: base directory where the "undistorted" output folder is
        created.
    - The refined camera matrix is computed with alpha=1 (no scaling of the
      result) and the ROI returned by cv2.getOptimalNewCameraMatrix is used to
      crop the undistorted image.
    - Output files are named "undistorted_<original_basename>" and written to
      OUTPUT_DIRECTORY/undistorted.
    - If no images match the glob pattern, the function prints a message and
      returns without raising an exception.
    - This function performs per-image prints to stdout to indicate progress; for
      non-interactive use, redirect or suppress stdout as needed.

    Examples
    --------
    >>> # Given a 3x3 camera matrix and distortion coefficients:
    >>> undistort_images(camera_matrix, dist_coeffs)
    """
    if not SAVE_UNDISTORTED:
        return
    
    images = glob.glob(CALIBRATION_IMAGES_PATH)
    
    if not images:
        print(f"No images found at {CALIBRATION_IMAGES_PATH}")
        return
    
    undistorted_dir = os.path.join(OUTPUT_DIRECTORY, 'undistorted')
    if not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)
    
    print(f"Undistorting {len(images)} images...")
    
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        
        # Refine camera matrix based on free scaling parameter
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # Undistort image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Crop the image (optional)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # Save undistorted image
        output_img_path = os.path.join(undistorted_dir, f'undistorted_{os.path.basename(fname)}')
        cv2.imwrite(output_img_path, dst)
        
        print(f"Undistorted image {idx+1}/{len(images)}: {fname}")
    
    print(f"Undistorted images saved to {undistorted_dir}")

def main():
    """
    Main function to run the camera calibration process.
    """
    print("Starting camera calibration...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    
    if mtx is None:
        print("Calibration failed. Exiting.")
        return
    
    # Undistort images
    undistort_images(mtx, dist)
    
    print("Camera calibration completed successfully!")

if __name__ == "__main__":
    main()