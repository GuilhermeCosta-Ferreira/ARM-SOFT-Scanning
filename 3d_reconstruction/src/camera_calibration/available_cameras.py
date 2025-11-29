import cv2

def list_cameras(max_tested=10):
    """
    Try camera indices from 0 to max_tested-1 and return the ones that work.
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
