"""Core enhancement functions for dark camera trap images.

Week 1 scope: load an image as 8-bit grayscale and apply global
histogram equalization with OpenCV.
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def load_grayscale(path: PathLike) -> np.ndarray:
    """Read an image from disk as a single-channel uint8 array.

    ``cv2.imread`` silently returns ``None`` on failure, so the two failure
    modes are turned into explicit exceptions here.

    Raises:
        FileNotFoundError: the path does not point to an existing file.
        ValueError: the file exists but OpenCV cannot decode it as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image (unsupported or corrupted file): {path}")
    return img


def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization to an 8-bit image.

    ``cv2.equalizeHist`` only accepts 8-bit single-channel input, so:

    * 3-channel (BGR) and 4-channel (BGRA) images are converted to grayscale;
      the result is then a 2-D array with the same height and width.
    * Non-uint8 images are rejected instead of being silently rescaled.

    Returns:
        A new uint8 array; the input is not modified.

    Raises:
        TypeError: ``img`` is not a NumPy array.
        ValueError: empty image, unsupported dtype or channel layout.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected a numpy.ndarray, got {type(img).__name__}")
    if img.size == 0:
        raise ValueError("Cannot equalize an empty image")
    if img.dtype != np.uint8:
        raise ValueError(f"Expected an 8-bit (uint8) image, got dtype {img.dtype}")

    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 1:
            img = img[:, :, 0]
        elif channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
    elif img.ndim != 2:
        raise ValueError(f"Expected a 2-D or 3-D image array, got {img.ndim} dimensions")

    return cv2.equalizeHist(np.ascontiguousarray(img))
