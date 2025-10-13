# ================================================================
# 0. Section: Imports
# ================================================================
import os
import cv2
import numpy as np



# ================================================================
# 1. Section: Data Extraction
# ================================================================
def load_scans(path: str, file_type: str = ".png") -> np.ndarray:
    scans = []
    # check each file inside this folder and open it with opencv
    for file in os.listdir(path):
        if file.endswith(file_type):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            scans.append(img)

    return np.array(scans)