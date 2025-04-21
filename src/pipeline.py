
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
from src.models.layout_model import LayoutExtractor
from src.utils.data_formatter import format_invoice_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models (load pre-trained models)
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(models_dir, exist_ok=True)

entity_model_path = os.path.join(models_dir, "entity_model.pth")
relation_model_path = os.path.join(models_dir, "relation_model.pth")
layout_model_path = os.path.join(models_dir, "layout_model.pth")
spacy_model_path = os.path.join(models_dir, "spacy_ner")

# Check if models exist
entity_model_exists = os.path.exists(entity_model_path)
relation_model_exists = os.path.exists(relation_model_path)
layout_model_exists = os.path.exists(layout_model_path)
spacy_model_exists = os.path.exists(spacy_model_path)

logger.info(f"Entity model exists: {entity_model_exists}")
logger.info(f"Relation model exists: {relation_model_exists}")
logger.info(f"Layout model exists: {layout_model_exists}")
logger.info(f"SpaCy NER model exists: {spacy_model_exists}")

# Initialize models with paths if they exist
entity_classifier = EntityClassifier(
    entity_model_path if entity_model_exists else None,
    use_spacy=True  # Always use spaCy for NER
)
relation_extractor = RelationExtractor(relation_model_path if relation_model_exists else None)
layout_extractor = LayoutExtractor(layout_model_path if layout_model_exists else None)


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
    text_blocks = extract_text_with_positions(preprocessed_image)
    
    # Debug: Print the extracted text blocks
    logger.info(f"Extracted {len(text_blocks)} text blocks from the invoice")
    for i, block in enumerate(text_blocks[:5]):
        logger.info(f"Block {i}: {block['text'][:50]}...")
    
    # Step 3: Apply different extraction approaches
    
    # 3.1: Classic entity classification (with spaCy NER)
    classified_entities = entity_classifier.classify(text_blocks)
    logger.info(f"Entity classifier results: {json.dumps(classified_entities, indent=2)}")
    
    # 3.2: Layout-aware extraction
    layout_results = layout_extractor.process(text_blocks)
    logger.info(f"Layout extractor results: {json.dumps(layout_results, indent=2)}")
    
    # 3.3: Relation extraction for item relationships
    extracted_relations = relation_extractor.extract(text_blocks, classified_entities)
    logger.info(f"Relation extractor results: {json.dumps(extracted_relations, indent=2)}")
    
    # Step 4: Combine results from different approaches
    final_results = combine_extraction_results(classified_entities, layout_results, extracted_relations)
    logger.info(f"Combined results: {json.dumps(final_results, indent=2)}")
    
    # Step 5: Format the extracted data into the required JSON structure
    formatted_data = format_invoice_data(final_results, extracted_relations)
    
    return formatted_data


def combine_extraction_results(
    entity_results: Dict[str, Any],
    layout_results: Dict[str, Any],
    relation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine results from different extraction approaches
    
    Args:
        entity_results: Results from entity classifier
        layout_results: Results from layout extractor
        relation_results: Results from relation extractor
        
    Returns:
        Combined results
    """
    combined = {
        "invoice_number": None,
        "invoice_date": None,
        "name": None,
        "items": [],
        "subtotal": None,
        "total": None,
        "extra_price": []
    }
    
    # For header fields, prioritize: layout > entity classifier > relation extractor
    for field in ["invoice_number", "invoice_date"]:
        combined[field] = (
            layout_results.get(field) or 
            entity_results.get(field) or 
            relation_results.get(field)
        )
    
    # Customer name
    combined["name"] = (
        layout_results.get("customer_name") or 
        entity_results.get("customer_name") or 
        relation_results.get("name")
    )
    
    # For items, use layout's items if available, otherwise use relation extractor's
    if layout_results.get("items"):
        layout_items = layout_results.get("items", [])
        relation_items = relation_results.get("items", [])
        
        # If both sources have items, use the one with more information
        if layout_items and relation_items:
            # Compute average completeness (having name, quantity, unit_price, total_price)
            layout_completeness = sum(
                (1 if item.get("name") else 0) + 
                (1 if item.get("quantity") else 0) + 
                (1 if item.get("unit_price") else 0) + 
                (1 if item.get("total_price") else 0) 
                for item in layout_items
            ) / (len(layout_items) * 4) if layout_items else 0
            
            relation_completeness = sum(
                (1 if item.get("name") else 0) + 
                (1 if item.get("quantity") else 0) + 
                (1 if item.get("unit_price") else 0) + 
                (1 if item.get("total_price") else 0) 
                for item in relation_items
            ) / (len(relation_items) * 4) if relation_items else 0
            
            combined["items"] = layout_items if layout_completeness >= relation_completeness else relation_items
        else:
            combined["items"] = layout_items or relation_items
    else:
        combined["items"] = relation_results.get("items", [])
    
    # For total and subtotal
    for field in ["subtotal", "total"]:
        # First extract numeric values if possible
        layout_value = extract_numeric_value(layout_results.get(field))
        entity_value = extract_numeric_value(entity_results.get(field))
        relation_value = relation_results.get(field)
        
        # Prioritize values
        if isinstance(relation_value, (int, float)) and relation_value > 0:
            combined[field] = relation_value
        elif layout_value is not None:
            combined[field] = layout_value
        elif entity_value is not None:
            combined[field] = entity_value
        else:
            # Keep text value if no numeric value
            combined[field] = (
                layout_results.get(field) or 
                entity_results.get(field) or 
                relation_results.get(field)
            )
    
    # Combine extra price items
    extra_price_items = []
    
    # From layout results
    for item in layout_results.get("extra_price", []):
        if isinstance(item, str):
            label, value = parse_extra_price(item)
            if label and value is not None:
                extra_price_items.append({label: value})
        elif isinstance(item, dict):
            extra_price_items.append(item)
    
    # From entity results
    for item in entity_results.get("extra_price", []):
        if isinstance(item, str):
            label, value = parse_extra_price(item)
            if label and value is not None:
                extra_price_items.append({label: value})
        elif isinstance(item, dict):
            extra_price_items.append(item)
    
    # From relation results
    if isinstance(relation_results.get("extra_price"), list):
        extra_price_items.extend(relation_results.get("extra_price", []))
    
    # Remove duplicates by key
    seen_keys = set()
    unique_items = []
    for item in extra_price_items:
        if isinstance(item, dict) and len(item) > 0:
            key = next(iter(item.keys()))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_items.append(item)
    
    combined["extra_price"] = unique_items
    
    return combined


def extract_numeric_value(text):
    """
    Extract numeric value from text
    
    Args:
        text: Text containing a numeric value
        
    Returns:
        Extracted numeric value or None
    """
    if text is None:
        return None
        
    if isinstance(text, (int, float)):
        return text
        
    import re
    # Remove currency symbols and commas
    cleaned_text = re.sub(r'[$€£¥,]', '', str(text))
    
    # Find numbers with optional decimal point
    matches = re.findall(r'\d+\.\d+|\d+', cleaned_text)
    if matches:
        return float(matches[0])
    return None


def parse_extra_price(text):
    """
    Parse extra price text into label and value
    
    Args:
        text: Text containing extra price information
        
    Returns:
        Tuple of (label, value)
    """
    if not text:
        return None, None
        
    import re
    
    # Extract numeric value
    value = extract_numeric_value(text)
    if value is None:
        return None, None
    
    # Determine label based on keywords
    text_lower = text.lower()
    
    if "tax" in text_lower or "vat" in text_lower or "gst" in text_lower:
        return "tax", value
    elif "discount" in text_lower:
        return "discount", value
    elif "shipping" in text_lower or "delivery" in text_lower:
        return "shipping", value
    else:
        return "other", value
