
"""
Main processing pipeline for invoice data extraction
"""

from typing import Dict, Any, List
import os
import json
from src.ocr.processor import extract_text_with_positions
from src.preprocessing.image_preprocessor import preprocess_image
from src.models.entity_classifier import EntityClassifier
from src.models.relation_extractor import RelationExtractor
from src.utils.data_formatter import format_invoice_data

# Initialize models (load pre-trained models)
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(models_dir, exist_ok=True)

entity_model_path = os.path.join(models_dir, "entity_model.pth")
relation_model_path = os.path.join(models_dir, "relation_model.pth")

# Check if models exist
entity_model_exists = os.path.exists(entity_model_path)
relation_model_exists = os.path.exists(relation_model_path)

print(f"Entity model exists: {entity_model_exists}")
print(f"Relation model exists: {relation_model_exists}")

# Initialize models with paths if they exist
entity_classifier = EntityClassifier(entity_model_path if entity_model_exists else None)
relation_extractor = RelationExtractor(relation_model_path if relation_model_exists else None)


def process_invoice(image_path: str) -> Dict[str, Any]:
    """
    Main pipeline function to process an invoice and extract structured data
    
    Args:
        image_path: Path to the invoice image
        
    Returns:
        Structured invoice data as a dictionary
    """
    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Step 2: Extract text with positions using OCR
    text_blocks = extract_text_with_positions(image_path)
    
    # Debug: Print the extracted text blocks
    print(f"Extracted {len(text_blocks)} text blocks from the invoice")
    for i, block in enumerate(text_blocks[:5]):
        print(f"Block {i}: {block['text'][:50]}...")
    
    # Step 3: Classify entities (invoice number, date, customer, etc.)
    classified_entities = entity_classifier.classify(text_blocks)
    
    # Debug: Print the classified entities
    print(f"Classified entities: {json.dumps(classified_entities, indent=2)}")
    
    # Step 4: Extract relations (items, pricing)
    extracted_relations = relation_extractor.extract(text_blocks, classified_entities)
    
    # Debug: Print the extracted relations
    print(f"Extracted relations: {json.dumps(extracted_relations, indent=2)}")
    
    # Step 5: Format the extracted data into the required JSON structure
    formatted_data = format_invoice_data(classified_entities, extracted_relations)
    
    return formatted_data
