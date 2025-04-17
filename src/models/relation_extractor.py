
"""
Relation extractor model to identify relationships between entities (e.g., item details)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import re


class RelationModel(nn.Module):
    """Neural network model for relation extraction"""
    
    def __init__(self, input_dim: int = 768 * 2, hidden_dim: int = 256, num_classes: int = 4):
        """
        Initialize the relation extraction model
        
        Args:
            input_dim: Dimension of input features (2x entity features for pairs)
            hidden_dim: Dimension of hidden layer
            num_classes: Number of relation classes to predict
        """
        super(RelationModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class RelationExtractor:
    """Class for extracting relationships between entities in invoices"""
    
    # Relation types mapping
    RELATION_TYPES = {
        0: "none",
        1: "item_quantity",
        2: "item_price",
        3: "item_total"
    }
    
    def __init__(self, model_path: str = None):
        """
        Initialize the relation extractor
        
        Args:
            model_path: Path to saved model weights (if None, uses default rules)
        """
        # Initialize model architecture
        self.model = RelationModel()
        
        # Load model weights if available
        if model_path and torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.model = self.model.cuda()
        elif model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
        
        # Flag to indicate if using pretrained model or rule-based approach
        self.use_model = model_path is not None
    
    def extract_price(self, text: str) -> Optional[float]:
        """
        Extract price value from text
        
        Args:
            text: Text containing price
            
        Returns:
            Extracted price as float or None if not found
        """
        # Remove currency symbols and commas
        cleaned_text = re.sub(r'[$€£¥,]', '', text)
        
        # Find numbers with optional decimal point
        matches = re.findall(r'\d+\.\d+|\d+', cleaned_text)
        if matches:
            return float(matches[0])
        return None
    
    def extract_quantity(self, text: str) -> Optional[int]:
        """
        Extract quantity value from text
        
        Args:
            text: Text containing quantity
            
        Returns:
            Extracted quantity as int or None if not found
        """
        # Find integer numbers
        matches = re.findall(r'\d+', text)
        if matches:
            return int(matches[0])
        return None
    
    def rule_based_extraction(
        self, text_blocks: List[Dict[str, Any]], entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply rule-based relation extraction
        
        Args:
            text_blocks: List of text blocks with text and position
            entities: Classified entities
            
        Returns:
            Dictionary with extracted relations
        """
        # Enhanced entities structure to store the extracted relations
        relations = {
            "invoice_number": entities.get("invoice_number"),
            "invoice_date": entities.get("invoice_date"),
            "name": entities.get("customer_name"),
            "items": [],
            "subtotal": None,
            "extra_price": [],
            "total": None
        }
        
        # Extract subtotal
        if entities.get("subtotal"):
            relations["subtotal"] = self.extract_price(entities["subtotal"])
            
        # Extract total
        if entities.get("total"):
            relations["total"] = self.extract_price(entities["total"])
            
        # Process extra price items
        for extra in entities.get("extra_price", []):
            price = self.extract_price(extra)
            if price:
                # Try to determine the type of extra price
                if "tax" in extra.lower():
                    relations["extra_price"].append({"tax": price})
                elif "discount" in extra.lower():
                    relations["extra_price"].append({"discount": price})
                elif "shipping" in extra.lower():
                    relations["extra_price"].append({"shipping": price})
                else:
                    relations["extra_price"].append({"other": price})
        
        # Identify table structure for items
        # This is a simplified approach; a real implementation would use positional information
        potential_items = []
        for i, block in enumerate(text_blocks):
            text = block["text"]
            position = block["position"]
            
            # Skip likely header texts
            if any(header in text.lower() for header in ["invoice", "bill", "customer", "date"]):
                continue
                
            # Check if text might be a quantity (number only or with units)
            quantity_match = re.search(r'\b\d+\b', text)
            if quantity_match:
                potential_items.append({
                    "index": i,
                    "text": text,
                    "position": position,
                    "potential_quantity": int(quantity_match.group())
                })
                continue
                
            # Check if text might be a price (has currency symbol or decimal)
            price_match = re.search(r'[$€£¥]?\d+(\.\d{2})?', text)
            if price_match:
                potential_items.append({
                    "index": i,
                    "text": text,
                    "position": position,
                    "potential_price": self.extract_price(text)
                })
                continue
                
            # If not quantity or price, might be item name
            potential_items.append({
                "index": i,
                "text": text,
                "position": position,
                "potential_name": text
            })
        
        # Group items by y-position (assuming items are in rows)
        # This is simplified; real implementation would use more sophisticated clustering
        rows = {}
        for item in potential_items:
            y_center = (item["position"]["y_min"] + item["position"]["y_max"]) / 2
            # Group by rounding to nearest 0.05 to account for slight misalignments
            row_key = round(y_center * 20) / 20
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(item)
        
        # Process each row as a potential item
        for row_key, row_items in rows.items():
            # Skip rows with too few elements (likely not an item row)
            if len(row_items) < 2:
                continue
                
            # Try to identify item components
            item = {
                "name": "",
                "quantity": 0,
                "unit_price": 0.0,
                "total_price": 0.0
            }
            
            # Sort by x position
            row_items.sort(key=lambda x: x["position"]["x_min"])
            
            # Assign components based on position and content
            for i, component in enumerate(row_items):
                # First component is usually item name
                if i == 0 and "potential_name" in component:
                    item["name"] = component["text"]
                    
                # Check for quantity
                elif "potential_quantity" in component:
                    item["quantity"] = component["potential_quantity"]
                    
                # Check for prices
                elif "potential_price" in component:
                    # Last price in a row is typically the total
                    if i == len(row_items) - 1:
                        item["total_price"] = component["potential_price"]
                    else:
                        item["unit_price"] = component["potential_price"]
            
            # If we have at least a name and one numeric value, consider it a valid item
            if item["name"] and (item["quantity"] or item["unit_price"] or item["total_price"]):
                # Calculate missing values if possible
                if item["quantity"] and item["unit_price"] and not item["total_price"]:
                    item["total_price"] = item["quantity"] * item["unit_price"]
                elif item["quantity"] and item["total_price"] and not item["unit_price"]:
                    item["unit_price"] = item["total_price"] / item["quantity"]
                    
                relations["items"].append(item)
        
        return relations
    
    def model_extraction(
        self, text_blocks: List[Dict[str, Any]], entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply model-based relation extraction (placeholder for trained model)
        
        Args:
            text_blocks: List of text blocks with text and position
            entities: Classified entities
            
        Returns:
            Dictionary with extracted relations
        """
        # This is a placeholder for actual model-based extraction
        # For now, fall back to rule-based approach
        return self.rule_based_extraction(text_blocks, entities)
    
    def extract(
        self, text_blocks: List[Dict[str, Any]], entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract relationships between entities
        
        Args:
            text_blocks: List of text blocks with text and position
            entities: Classified entities
            
        Returns:
            Dictionary with extracted relations
        """
        if self.use_model:
            return self.model_extraction(text_blocks, entities)
        else:
            return self.rule_based_extraction(text_blocks, entities)
