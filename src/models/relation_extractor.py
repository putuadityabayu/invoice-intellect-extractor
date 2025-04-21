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
        Disabling manual rule-based extraction.
        Return entitas/relasi kosong agar pipeline hanya mengandalkan output model ML.
        """
        return {
            "invoice_number": None,
            "invoice_date": None,
            "name": None,
            "items": [],
            "subtotal": None,
            "extra_price": [],
            "total": None
        }
    
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
