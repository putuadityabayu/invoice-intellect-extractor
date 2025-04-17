
"""
Main processing pipeline for invoice data extraction
"""

from typing import Dict, Any, List
import os
from src.ocr.processor import extract_text_with_positions
from src.preprocessing.image_preprocessor import preprocess_image
from src.models.entity_classifier import EntityClassifier
from src.models.relation_extractor import RelationExtractor
from src.utils.data_formatter import format_invoice_data

# Initialize models (load pre-trained models)
entity_classifier = EntityClassifier("models/entity_model.pth")
relation_extractor = RelationExtractor("models/relation_model.pth")


def process_invoice(image_path: str) -> Dict[str, Any]:
    """
    Main pipeline function to process an invoice and extract structured data
    
    Args:
        image_path: Path to the invoice image
        
    Returns:
        Structured invoice data as a dictionary
    """
    # Step 1: Preprocess the image
    # preprocessed_image = preprocess_image(image_path)
    
    # Step 2: Extract text with positions using OCR
    text_blocks = extract_text_with_positions(image_path)
    
    # Step 3: Classify entities (invoice number, date, customer, etc.)
    classified_entities = entity_classifier.classify(text_blocks)
    
    # Step 4: Extract relations (items, pricing)
    extracted_relations = relation_extractor.extract(text_blocks, classified_entities)
    
    # Step 5: Format the extracted data into the required JSON structure
    formatted_data = format_invoice_data(classified_entities, extracted_relations)
    
    return formatted_data
