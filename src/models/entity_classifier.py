
"""
Entity classifier model to identify invoice elements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple


class EntityModel(nn.Module):
    """Neural network model for entity classification"""
    
    def __init__(
        self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 8
    ):
        """
        Initialize the entity classifier model
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            num_classes: Number of entity classes to predict
        """
        super(EntityModel, self).__init__()
        
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


class EntityClassifier:
    """Class for classifying text entities in invoices"""
    
    # Entity types mapping
    ENTITY_TYPES = {
        0: "invoice_number",
        1: "invoice_date",
        2: "customer_name",
        3: "item_name",
        4: "item_quantity",
        5: "item_price",
        6: "subtotal",
        7: "total"
    }
    
    def __init__(self, model_path: str = None):
        """
        Initialize the entity classifier
        
        Args:
            model_path: Path to saved model weights (if None, uses default rules)
        """
        # Initialize model architecture
        self.model = EntityModel()
        # self.model_path = "models/entity_model.pth"
        
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
    
    def feature_extraction(self, text_blocks: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Extract features from text blocks for model input
        
        Args:
            text_blocks: List of text blocks with text and position
            
        Returns:
            Feature tensor
        """
        # In a real implementation, this would use embeddings from a language model
        # For now, this is a placeholder that would be replaced with actual feature extraction
        batch_size = len(text_blocks)
        # Create dummy features for demonstration
        features = torch.randn(batch_size, 768)
        return features
    
    def rule_based_classification(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply rule-based classification for entities
        
        Args:
            text_blocks: List of text blocks with text and position
            
        Returns:
            Dictionary of classified entities
        """
        entities = {
            "invoice_number": None,
            "invoice_date": None,
            "customer_name": None,
            "items": [],
            "subtotal": None,
            "total": None,
            "extra_price": []
        }
        
        # Simple rule-based approach using keywords
        for block in text_blocks:
            text = block["text"].lower()
            
            # Invoice number detection
            if "invoice" in text and ("#" in text or "no" in text or "number" in text):
                # Extract numeric part or entire text as invoice number
                entities["invoice_number"] = text
                
            # Date detection
            elif any(date_keyword in text for date_keyword in ["date", "issued", "due"]):
                entities["invoice_date"] = text
                
            # Customer detection
            elif any(customer_keyword in text for customer_keyword in ["customer", "bill to", "client"]):
                entities["customer_name"] = text
                
            # Subtotal detection
            elif "subtotal" in text or "sub-total" in text:
                entities["subtotal"] = text
                
            # Total detection
            elif "total" in text and not any(x in text for x in ["subtotal", "sub-total"]):
                entities["total"] = text
                
            # Tax or extra pricing detection
            elif any(tax_keyword in text for tax_keyword in ["tax", "vat", "gst", "discount"]):
                entities["extra_price"].append(text)
                
        return entities
    
    def model_classification(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply model-based classification for entities
        
        Args:
            text_blocks: List of text blocks with text and position
            
        Returns:
            Dictionary of classified entities
        """
        # Extract features
        features = self.feature_extraction(text_blocks)
        
        # Get model predictions
        with torch.no_grad():
            logits = self.model(features)
            predictions = torch.argmax(logits, dim=1)
        
        # Process predictions
        entities = {
            "invoice_number": None,
            "invoice_date": None,
            "customer_name": None,
            "items": [],
            "subtotal": None,
            "total": None,
            "extra_price": []
        }
        
        for i, pred in enumerate(predictions):
            entity_type = self.ENTITY_TYPES[pred.item()]
            print("entity_type",entity_type)
            
            if entity_type == "invoice_number":
                entities["invoice_number"] = text_blocks[i]["text"]
                
            elif entity_type == "invoice_date":
                entities["invoice_date"] = text_blocks[i]["text"]
                
            elif entity_type == "customer_name":
                entities["customer_name"] = text_blocks[i]["text"]
                
            elif entity_type == "subtotal":
                entities["subtotal"] = text_blocks[i]["text"]
                
            elif entity_type == "total":
                entities["total"] = text_blocks[i]["text"]
                
            elif entity_type in ["item_name", "item_quantity", "item_price"]:
                # Process item data
                # This is simplified; a real implementation would group these together
                if not any(text_blocks[i]["text"] in item.values() for item in entities["items"]):
                    entities["items"].append({"name": text_blocks[i]["text"]})
        
        return entities
    
    def classify(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify entities in the extracted text blocks
        
        Args:
            text_blocks: List of text blocks with text and position
            
        Returns:
            Dictionary of classified entities
        """
        if self.use_model:
            print("use model")
            return self.model_classification(text_blocks)
        else:
            return self.rule_based_classification(text_blocks)
