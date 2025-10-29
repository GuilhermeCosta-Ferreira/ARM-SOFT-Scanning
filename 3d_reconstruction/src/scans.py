import cv2
import os

import numpy as np

from dataclasses import dataclass
from functools import singledispatchmethod

@dataclass
class Scan:
    @singledispatchmethod
    def __init__(self, data) -> None:
        raise TypeError(f"Unsupported init type: {type(data)!r}")
    
    @__init__.register
    def _(self, array: np.ndarray, position: int) -> None:
        self.scan = array
        self.position = position


    @property
    def scan(self) -> np.ndarray:
        return self._scan
    
    @scan.setter
    def scan(self, value: np.ndarray) -> None:
        if not(1 < len(value.shape) < 3):
            raise ValueError("Scans must be two dimensional")
        self._scan = value


    @property
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, value: int | None) -> None:
        if value is None:
            print("⚠️ Warning: No camera position was defined, position is set to None")
            self._position = value
            return
        elif not isinstance(value, int):
            raise TypeError("Position must be an integer")
        self._position = value

@dataclass
class Scans:
    @singledispatchmethod
    def __init__(self, data) -> None:
        raise TypeError(f"Unsupported init type: {type(data)!r}")
    
    @__init__.register
    def _(self, images: np.ndarray) -> None:
        self.scans = images
        self.positions = np.array([scan.position for scan in images])

    @__init__.register
    def _(self, images: list) -> None:
        self.scans = np.array(images)
        self.positions = np.array([scan.position for scan in images])

    @__init__.register
    def _(self, images: np.ndarray | list, positions: np.ndarray | list) -> None:
        self.scans = np.array(images)
        self.positions = np.array(positions)

    @__init__.register
    def _(self, scans_path: str) -> None:
        files = [f for f in os.listdir(scans_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
        files.sort()

        scans = []
        for idx, file in enumerate(files):
            img = cv2.imread(os.path.join(scans_path, file), cv2.IMREAD_GRAYSCALE)
            img = np.where(img > 0, 1, 0).astype(np.uint8)
            scan = Scan(img, idx)
            scans.append(scan)
        
        self.scans = np.array(scans)
        self.positions = np.array([scan.position for scan in scans])


    @property
    def scans(self) -> np.ndarray:
        return self._scans
    
    @scans.setter
    def scans(self, value: np.ndarray) -> None:

        test_shape = None
        for ext in value:
            if test_shape is None: test_shape = ext.scan.shape
            if not isinstance(ext, Scan):
                raise TypeError("All elements must be of type Scan")
            if ext.scan.shape != test_shape:
                raise ValueError("All scans must have the same shape")
        self._scans = value


    @property
    def nr_positions(self) -> int:
        return len(self.positions)
    
    @property
    def scan_shape(self) -> tuple[int, int]:
        return self.scans[0].scan.shape
    

    def scan(self, index: int) -> np.ndarray:
        return self.scans[index].scan
    
    def position(self, index: int) -> int:
        return self.scans[index].position