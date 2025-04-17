
"""
OCR processor using doctr for text extraction with positional information
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


class OCRProcessor:
    """Class for handling OCR operations using doctr"""
    
    def __init__(self):
        """Initialize the OCR processor with doctr model"""
        # Use the doctr pretrained model for OCR
        self.model = ocr_predictor(pretrained=True)
        
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from an image with positional information
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of text blocks with text content and position
        """
        # Convert to RGB if grayscale (doctr expects RGB)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Create a document from the image
        doc = DocumentFile.from_images(image)
        
        # Run OCR prediction
        result = self.model(doc)
        
        # Extract text blocks with positions
        text_blocks = []
        
        # Get results from pages
        for page in result.pages:
            # Process each block in the page
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # Extract text and bounding box
                        text = word.value
                        # Bounding box is in relative coordinates (0-1)
                        # Format: [xmin, ymin, xmax, ymax]
                        bbox = word.geometry
                        
                        text_blocks.append({
                            "text": text,
                            "confidence": word.confidence,
                            "position": {
                                "x_min": bbox[0],
                                "y_min": bbox[1],
                                "x_max": bbox[2],
                                "y_max": bbox[3]
                            }
                        })
                        
        return text_blocks


# Initialize the OCR processor
ocr_processor = OCRProcessor()


def extract_text_with_positions(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Extract text from an image with positional information
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of text blocks with text content and position
    """
    return ocr_processor.extract_text(image)
