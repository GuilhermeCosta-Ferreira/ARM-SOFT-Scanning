import cv2

def list_cameras(max_tested: int = 10) -> list:
    """
    Scan and return indices of available video capture devices.

    This function probes system video capture devices by attempting to open
    consecutive integer device indices starting at 0 up to `max_tested - 1`.
    For each index it attempts to open a cv2.VideoCapture and (optionally)
    grabs a single frame to ensure the device is functional before including
    the index in the result. Opened capture handles are released immediately
    after the check.

    Parameters
    ----------
    max_tested : int, optional
        Maximum number of device indices to test (default is 10). The function
        will test indices 0..max_tested-1. A larger value increases the time
        required to run the check.

    Returns
    -------
    list[int]
        A list of integer device indices that were successfully opened and
        (optionally) returned a frame. The list will be empty if no working
        devices are found.

    Examples
    --------
    >>> # Check the first 8 indices for connected cameras
    >>> list_cameras(8)
    [0, 2]

    >>> # Use the default of 10 indices
    >>> available = list_cameras()
    >>> if available:
    ...     print(f"Found cameras at indices: {available}")
    ... else:
    ...     print("No cameras found")

    Notes
    -----
    - Requires OpenCV (cv2) to be installed and available in the runtime.
    - Behavior depends on the underlying OS and OpenCV backend; some backends
      may report devices differently or require different permissions.
    - The function performs a best-effort check by attempting to open and read
      a single frame; a device might be available but fail the frame read
      due to transient conditions (busy device, permission issues, etc.).
    - This function is intended for quick discovery and is not a substitute
      for robust device enumeration in production systems.
    """
    available_cams = []

    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            # This index doesn't correspond to a working camera
            continue

        # Optional: try to grab one frame to be extra sure
        ret, frame = cap.read()
        if ret:
            available_cams.append(i)

        cap.release()

    return available_cams

if __name__ == "__main__":
    cams = list_cameras(max_tested=10)
    print("Available camera IDs:", cams)
