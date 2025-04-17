
"""
Utility functions for image handling
"""

import os
import requests
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional


def download_image_from_url(url: str, save_path: str) -> Tuple[bool, str]:
    """
    Download an image from a URL and save it to a file
    
    Args:
        url: URL of the image
        save_path: Path to save the downloaded image
        
    Returns:
        Tuple of (success, message)
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Check if the content type is an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            return False, f"URL does not point to an image: {content_type}"
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        return True, "Image downloaded successfully"
    except requests.exceptions.RequestException as e:
        return False, f"Error downloading image: {str(e)}"


def is_valid_image(file_path: str) -> bool:
    """
    Check if the file is a valid image
    
    Args:
        file_path: Path to the image file
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(file_path) as img:
            # Try to verify the image
            img.verify()
        return True
    except:
        return False


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load an image from a file
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        return cv2.imread(file_path)
    except:
        return None


def save_image(image: np.ndarray, file_path: str) -> bool:
    """
    Save an image to a file
    
    Args:
        image: Image as numpy array
        file_path: Path to save the image
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        cv2.imwrite(file_path, image)
        return True
    except:
        return False
