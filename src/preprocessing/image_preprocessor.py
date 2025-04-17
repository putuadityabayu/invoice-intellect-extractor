
"""
Image preprocessing utilities to improve OCR accuracy
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Apply a series of preprocessing steps to improve image quality for OCR
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Apply sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Correct skewed images
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Deskewed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate skew angle
    # Find all non-zero points
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # The angle is between -90 and 0 degrees
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate the image to correct the skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def resize_image(
    image: np.ndarray, target_size: Tuple[int, int] = (1000, 1000)
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate target dimensions while preserving aspect ratio
    if w > h:
        new_w = target_w
        new_h = int(h * (target_w / w))
    else:
        new_h = target_h
        new_w = int(w * (target_h / h))
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image contrast and brightness
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply contrast enhancement
    normalized = clahe.apply(gray)
    
    return normalized
