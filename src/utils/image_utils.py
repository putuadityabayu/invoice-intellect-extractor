
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
    # ... keep existing code (download_image_from_url function implementation)


def is_valid_image(file_path: str) -> bool:
    """
    Check if the file is a valid image
    
    Args:
        file_path: Path to the image file
        
    Returns:
        True if valid image, False otherwise
    """
    # ... keep existing code (is_valid_image function implementation)


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load an image from a file
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Image as numpy array or None if failed
    """
    # ... keep existing code (load_image function implementation)


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
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def save_debug_info(data: any, file_path: str, format_type: str = "auto") -> bool:
    """
    Save debug information to a file
    
    Args:
        data: Data to save
        file_path: Path to save the file
        format_type: Type of format to save as ('json', 'txt', 'md', or 'auto')
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Determine format type if auto
        if format_type == "auto":
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.json':
                format_type = "json"
            elif ext == '.md':
                format_type = "md"
            else:
                format_type = "txt"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save data in the appropriate format
        if format_type == "json":
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        elif format_type == "md":
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)
        else:  # txt
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
                
        return True
    except Exception as e:
        print(f"Error saving debug info: {e}")
        return False
