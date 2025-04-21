
"""
OCR processor using doctr for text extraction with positional information
"""

import numpy as np
import os
import json
from typing import List, Dict, Any, Tuple
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


class OCRProcessor:
    """Class for handling OCR operations using doctr"""
    
    def __init__(self):
        """Initialize the OCR processor with doctr model"""
        # Use the doctr pretrained model for OCR
        self.model = ocr_predictor("db_resnet50","crnn_mobilenet_v3_large",pretrained=True)
        
    def extract_text(self, image: str) -> List[Dict[str, Any]]:
        """
        Extract text from an image with positional information
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of text blocks with text content and position
        """
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
                                "x_min": bbox[0][0],
                                "y_min": bbox[0][1],
                                "x_max": bbox[1][0],
                                "y_max": bbox[1][1]
                            }
                        })
                        
        return text_blocks


# Initialize the OCR processor
ocr_processor = OCRProcessor()


def extract_text_with_positions(image: str) -> List[Dict[str, Any]]:
    """
    Extract text from an image with positional information
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of text blocks with text content and position
    """
    results = ocr_processor.extract_text(image)
    
    # Create a visual debug output
    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save raw OCR results as text with positions
    ocr_visual_output = ""
    for block in results:
        text = block["text"]
        pos = block["position"]
        confidence = block["confidence"]
        ocr_visual_output += f"Text: '{text}' (conf: {confidence:.2f})\n"
        ocr_visual_output += f"Position: x_min={pos['x_min']:.3f}, y_min={pos['y_min']:.3f}, "
        ocr_visual_output += f"x_max={pos['x_max']:.3f}, y_max={pos['y_max']:.3f}\n\n"
    
    with open(os.path.join(debug_dir, "ocr_visual_results.txt"), "w", encoding="utf-8") as f:
        f.write(ocr_visual_output)
    
    return results
