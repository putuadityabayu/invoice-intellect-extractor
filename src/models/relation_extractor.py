
"""
Relation extractor model to identify relationships between entities (e.g., item details)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import re


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize BiLSTM with attention
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(BiLSTMAttention, self).__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2, num_layers=num_layers, 
            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output dimension
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        """Forward pass"""
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        attn_applied = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Apply layer normalization
        normalized = self.layer_norm(attn_applied)
        
        return normalized


class RelationModel(nn.Module):
    """Neural network model for relation extraction using BiLSTM"""
    
    def __init__(self, input_dim: int = 768 * 2, hidden_dim: int = 256, num_classes: int = 4):
        """
        Initialize the relation extraction model
        
        Args:
            input_dim: Dimension of input features (2x entity features for pairs)
            hidden_dim: Dimension of hidden layer
            num_classes: Number of relation classes to predict
        """
        super(RelationModel, self).__init__()
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM with attention
        self.bilstm_attention = BiLSTMAttention(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.3
        )
        
        # Output classification layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """Forward pass"""
        # Reshape input for sequence processing
        batch_size = x.size(0)
        x_seq = x.unsqueeze(1)  # Add sequence dimension [batch, seq_len=1, features]
        
        # Apply input projection
        x_projected = self.input_projection(x)
        x_projected = F.relu(x_projected)
        x_projected = self.dropout(x_projected)
        
        # Reshape for LSTM
        x_lstm_input = x_projected.unsqueeze(1)  # [batch, seq_len=1, hidden_dim]
        
        # Apply BiLSTM with attention
        x_lstm = self.bilstm_attention(x_lstm_input)
        
        # Apply output layers
        x = F.relu(self.fc1(x_lstm))
        x = self.dropout(x)
        x = self.fc2(x)
        
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
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                self.model = self.model.cuda()
                print(f"Loaded relation model from {model_path} (CUDA)")
            except Exception as e:
                print(f"Error loading relation model: {e}")
                print("Using untrained model")
        elif model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.model.eval()
                print(f"Loaded relation model from {model_path}")
            except Exception as e:
                print(f"Error loading relation model: {e}")
                print("Using untrained model")
        
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
    
    def extract_features(self, source_block, target_block):
        """
        Extract features for a pair of text blocks
        
        Args:
            source_block: Source text block
            target_block: Target text block
            
        Returns:
            Feature tensor
        """
        # Create feature vector (placeholder implementation)
        features = torch.zeros(768 * 2)
        
        # Extract basic text features for source
        source_text = source_block.get("text", "").lower()
        for i, char in enumerate(source_text[:50]):
            if i < 100:
                features[i] = ord(char) / 255.0
        
        # Text length feature
        features[100] = min(len(source_text) / 100.0, 1.0)
        
        # Extract basic text features for target
        target_text = target_block.get("text", "").lower()
        for i, char in enumerate(target_text[:50]):
            if i < 100:
                features[768 + i] = ord(char) / 255.0
        
        # Text length feature for target
        features[768 + 100] = min(len(target_text) / 100.0, 1.0)
        
        # Spatial features for source
        if "position" in source_block:
            pos = source_block["position"]
            features[150] = pos.get("x_min", 0) / 1000.0
            features[151] = pos.get("y_min", 0) / 1000.0
            features[152] = pos.get("x_max", 0) / 1000.0
            features[153] = pos.get("y_max", 0) / 1000.0
        
        # Spatial features for target
        if "position" in target_block:
            pos = target_block["position"]
            features[768 + 150] = pos.get("x_min", 0) / 1000.0
            features[768 + 151] = pos.get("y_min", 0) / 1000.0
            features[768 + 152] = pos.get("x_max", 0) / 1000.0
            features[768 + 153] = pos.get("y_max", 0) / 1000.0
        
        # Relative position features
        if "position" in source_block and "position" in target_block:
            source_pos = source_block["position"]
            target_pos = target_block["position"]
            
            # Horizontal distance
            features[200] = (target_pos.get("x_min", 0) - source_pos.get("x_max", 0)) / 1000.0
            
            # Vertical distance
            features[201] = (target_pos.get("y_min", 0) - source_pos.get("y_max", 0)) / 1000.0
            
            # Same line feature
            same_line = (
                abs(source_pos.get("y_min", 0) - target_pos.get("y_min", 0)) < 20 or
                abs(source_pos.get("y_max", 0) - target_pos.get("y_max", 0)) < 20
            )
            features[202] = 1.0 if same_line else 0.0
        
        return features
    
    def model_extraction(
        self, text_blocks: List[Dict[str, Any]], entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply model-based relation extraction
        
        Args:
            text_blocks: List of text blocks with text and position
            entities: Classified entities
            
        Returns:
            Dictionary with extracted relations
        """
        if not self.use_model:
            return self.rule_based_extraction(text_blocks, entities)
        
        # Initialize result structure
        results = {
            "invoice_number": entities.get("invoice_number"),
            "invoice_date": entities.get("invoice_date"),
            "name": entities.get("customer_name"),
            "items": [],
            "subtotal": entities.get("subtotal"),
            "total": entities.get("total"),
            "extra_price": []
        }
        
        # Find item names
        item_blocks = []
        for block in text_blocks:
            entity_type = block.get("entity_type")
            if entity_type == "item_name":
                item_blocks.append(block)
        
        # For each item name, find related information
        for item_block in item_blocks:
            item_info = {"name": item_block.get("text", "")}
            
            # Evaluate all possible relations with other blocks
            for other_block in text_blocks:
                if other_block == item_block:
                    continue
                
                # Extract features for this pair
                features = self.extract_features(item_block, other_block)
                
                # Get model prediction
                with torch.no_grad():
                    features_tensor = features.unsqueeze(0)  # Add batch dimension
                    if torch.cuda.is_available():
                        features_tensor = features_tensor.cuda()
                    
                    logits = self.model(features_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, prediction].item()
                
                # Only consider high confidence predictions
                if confidence < 0.5:
                    continue
                
                # Process prediction based on relation type
                relation_type = self.RELATION_TYPES[prediction]
                
                if relation_type == "item_quantity":
                    quantity = self.extract_quantity(other_block.get("text", ""))
                    if quantity is not None:
                        item_info["quantity"] = quantity
                
                elif relation_type == "item_price":
                    price = self.extract_price(other_block.get("text", ""))
                    if price is not None:
                        item_info["unit_price"] = price
                
                elif relation_type == "item_total":
                    total = self.extract_price(other_block.get("text", ""))
                    if total is not None:
                        item_info["total_price"] = total
            
            # Calculate missing values
            if "quantity" in item_info and "unit_price" in item_info and "total_price" not in item_info:
                item_info["total_price"] = item_info["quantity"] * item_info["unit_price"]
            
            if "quantity" in item_info and "total_price" in item_info and "unit_price" not in item_info:
                if item_info["quantity"] > 0:
                    item_info["unit_price"] = item_info["total_price"] / item_info["quantity"]
            
            # Ensure required fields
            item_info["quantity"] = item_info.get("quantity", 1)
            item_info["unit_price"] = item_info.get("unit_price", 0.0)
            item_info["total_price"] = item_info.get("total_price", 0.0)
            
            # Add to results
            results["items"].append(item_info)
        
        return results
    
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

