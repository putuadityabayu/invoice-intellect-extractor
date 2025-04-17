
"""
Utilities for formatting extracted data into the required output structure
"""

from typing import Dict, Any, List, Optional
import re
from datetime import datetime


def extract_numeric_value(text: str) -> Optional[float]:
    """
    Extract a numeric value from text
    
    Args:
        text: Text containing a numeric value
        
    Returns:
        Extracted numeric value or None if not found
    """
    if not text:
        return None
        
    # Remove currency symbols and commas
    cleaned_text = re.sub(r'[$€£¥,]', '', text)
    
    # Find numbers with optional decimal point
    matches = re.findall(r'\d+\.\d+|\d+', cleaned_text)
    if matches:
        return float(matches[0])
    return None


def clean_invoice_number(text: str) -> str:
    """
    Clean and format invoice number
    
    Args:
        text: Raw invoice number text
        
    Returns:
        Cleaned invoice number
    """
    if not text:
        return ""
        
    # Extract the actual number part if present
    number_match = re.search(r'[#:]?\s*([A-Za-z0-9-]+)', text)
    if number_match:
        return number_match.group(1)
    return text


def format_date(text: str) -> str:
    """
    Attempt to format a date string
    
    Args:
        text: Raw date text
        
    Returns:
        Formatted date string
    """
    if not text:
        return ""
        
    # Try to extract a date pattern
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY or MM/DD/YYYY
        r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD
        r'\d{1,2}\s+[A-Za-z]+\s+\d{2,4}'   # DD Month YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    
    return text


def format_invoice_data(
    entities: Dict[str, Any], relations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format the extracted data into the required JSON structure
    
    Args:
        entities: Classified entities
        relations: Extracted relations
        
    Returns:
        Formatted invoice data
    """
    # Start with the relation data as it contains most of the structured information
    formatted_data = relations.copy()
    
    # Clean invoice number
    if formatted_data.get("invoice_number"):
        formatted_data["invoice_number"] = clean_invoice_number(formatted_data["invoice_number"])
    else:
        formatted_data["invoice_number"] = ""
    
    # Format date
    if formatted_data.get("invoice_date"):
        formatted_data["invoice_date"] = format_date(formatted_data["invoice_date"])
    else:
        formatted_data["invoice_date"] = ""
    
    # Ensure name is present
    if "name" not in formatted_data or not formatted_data["name"]:
        formatted_data["name"] = ""
    
    # Ensure all required fields are present for items
    for item in formatted_data.get("items", []):
        if "name" not in item:
            item["name"] = ""
        if "quantity" not in item:
            item["quantity"] = 0
        if "unit_price" not in item:
            item["unit_price"] = 0.0
        if "total_price" not in item:
            item["total_price"] = 0.0
    
    # Ensure subtotal exists
    if "subtotal" not in formatted_data or formatted_data["subtotal"] is None:
        formatted_data["subtotal"] = 0.0
    
    # Ensure total exists
    if "total" not in formatted_data or formatted_data["total"] is None:
        formatted_data["total"] = 0.0
    
    # Ensure extra_price exists
    if "extra_price" not in formatted_data:
        formatted_data["extra_price"] = []
    
    return formatted_data
