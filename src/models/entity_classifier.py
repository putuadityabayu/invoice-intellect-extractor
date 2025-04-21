"""
Entity classifier model to identify invoice elements using both neural networks and NER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import re
from typing import List, Dict, Any, Tuple, Optional
import os
import numpy as np


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
    """Class for classifying text entities in invoices using multiple approaches"""
    
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
    
    def __init__(self, model_path: str = None, use_spacy: bool = True, use_nn: bool = True):
        """
        Initialize the entity classifier
        
        Args:
            model_path: Path to saved neural network model weights (if None, uses default rules)
            use_spacy: Whether to use spaCy NER
            use_nn: Whether to use neural network model
        """
        self.use_spacy = use_spacy
        self.use_nn = use_nn
        
        # Initialize neural network model if requested
        if use_nn:
            # Initialize model architecture
            self.model = EntityModel()
            
            # Load model weights if available
            if model_path and torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                self.model = self.model.cuda()
                print(f"Loaded neural network model from {model_path} (CUDA)")
            elif model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.model.eval()
                print(f"Loaded neural network model from {model_path}")
            else:
                print("Neural network requested but no valid model path provided")
        else:
            self.model = None
        
        # Initialize spaCy model if requested
        if use_spacy:
            try:
                # Try to load the custom model if it exists
                spacy_model_path = os.path.join(os.path.dirname(model_path), "spacy_ner") if model_path else None
                
                if spacy_model_path and os.path.exists(spacy_model_path):
                    self.nlp = spacy.load(spacy_model_path)
                    print(f"Loaded custom spaCy model from {spacy_model_path}")
                else:
                    # Fall back to standard spaCy model
                    self.nlp = spacy.load("en_core_web_sm")
                    print("Loaded standard spaCy en_core_web_sm model")
            except OSError as e:
                print(f"Error loading spaCy model: {e}")
                print("Downloading spaCy model en_core_web_sm...")
                try:
                    # Try to download the model
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                    print("Successfully downloaded and loaded spaCy model")
                except Exception as e:
                    print(f"Failed to download spaCy model: {e}")
                    self.nlp = None
                    self.use_spacy = False
        else:
            self.nlp = None
        
        # Flag for rule-based fallback
        self.use_rules = True
    
    def feature_extraction(self, text_blocks: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Extract features from text blocks for model input
        
        Args:
            text_blocks: List of text blocks with text and position
            
        Returns:
            Feature tensor
        """
        # In a real implementation, this would use embeddings from a language model
        batch_size = len(text_blocks)
        features = []
        
        for block in text_blocks:
            # Basic text features using TF-IDF like approach
            text = block["text"].lower()
            text_feature = np.zeros(768)  # Using 768 to match BERT dimensionality
            
            # Add position features
            if "position" in block:
                pos = block["position"]
                # Normalize position features and add to the first few dimensions
                text_feature[0] = pos.get("x_min", 0.0) / 1000.0
                text_feature[1] = pos.get("y_min", 0.0) / 1000.0
                text_feature[2] = pos.get("x_max", 1.0) / 1000.0
                text_feature[3] = pos.get("y_max", 1.0) / 1000.0
            
            # Add text length feature
            text_feature[4] = min(len(text) / 100.0, 1.0)  # Normalize text length
            
            # Add keyword presence features
            keywords = [
                "invoice", "bill", "receipt", "date", "due", "payment", "total",
                "subtotal", "tax", "amount", "item", "description", "quantity",
                "price", "unit", "discount", "shipping", "customer", "vendor"
            ]
            for i, keyword in enumerate(keywords):
                if keyword in text:
                    text_feature[5 + i] = 1.0
            
            features.append(text_feature)
        
        # Convert to tensor
        if not features:  # Handle empty list
            return torch.zeros((0, 768))
        return torch.tensor(np.array(features), dtype=torch.float32)
    
    def rule_based_classification(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply rule-based classification for entities

        NOTE: Di-nonaktifkan, return kosong supaya pipeline hanya mengandalkan hasil prediksi model ML (tanpa regex/manual pattern-matching)
        """
        return {
            "invoice_number": None,
            "invoice_date": None,
            "customer_name": None,
            "items": [],
            "subtotal": None,
            "total": None,
            "extra_price": []
        }
    
    def spacy_ner_classification(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use spaCy NER to classify entities
        
        Args:
            text_blocks: List of text blocks with text and position
            
        Returns:
            Dictionary of classified entities
        """
        if not self.nlp:
            return self.rule_based_classification(text_blocks)
        
        entities = {
            "invoice_number": None,
            "invoice_date": None,
            "customer_name": None,
            "items": [],
            "subtotal": None,
            "total": None,
            "extra_price": []
        }
        
        # Process each text block with spaCy
        for block in text_blocks:
            text = block["text"]
            doc = self.nlp(text)
            
            # Check for named entities
            for ent in doc.ents:
                if ent.label_ == "CARDINAL" and "invoice" in text.lower():
                    entities["invoice_number"] = text
                elif ent.label_ == "DATE":
                    entities["invoice_date"] = text
                elif ent.label_ == "ORG" or ent.label_ == "PERSON":
                    if not entities["customer_name"]:  # Only pick the first one
                        entities["customer_name"] = text
                
            # Apply custom rules with spaCy tokens for better accuracy
            text_lower = text.lower()
            
            # Check for monetary values
            money_pattern = re.compile(r'[$€£¥]\s*\d+[\d,\.]*|\d+[\d,\.]*\s*[$€£¥]')
            has_money = money_pattern.search(text)
            
            # Subtotal detection
            if ("subtotal" in text_lower or "sub-total" in text_lower or "sub total" in text_lower) and has_money:
                entities["subtotal"] = text
            
            # Total detection
            elif (("total" in text_lower or "amount due" in text_lower) and 
                  not any(x in text_lower for x in ["subtotal", "sub-total", "sub total"]) and has_money):
                entities["total"] = text
            
            # Tax or extra pricing detection
            elif any(tax_keyword in text_lower for tax_keyword in ["tax", "vat", "gst", "discount"]) and has_money:
                entities["extra_price"].append(text)
            
            # Item detection based on patterns
            item_patterns = [
                r'\d+\s*x\s*[$€£¥]?\s*\d+[\d,\.]*',  # quantity x price
                r'\d+\s*@\s*[$€£¥]?\s*\d+[\d,\.]*',   # quantity @ price
                r'\d+\s*ea\s*[$€£¥]?\s*\d+[\d,\.]*',  # quantity ea price
            ]
            if any(re.search(pattern, text) for pattern in item_patterns):
                # Check if it's not already detected as another entity
                if not any(text == v for v in [entities["invoice_number"], entities["invoice_date"], 
                                              entities["subtotal"], entities["total"]]):
                    # Find quantity and price if possible
                    quantity_match = re.search(r'\b(\d+)\s*[x@]', text)
                    price_match = re.search(r'[$€£¥]?\s*(\d+[\d,\.]*)', text)
                    
                    if quantity_match and price_match:
                        quantity = int(quantity_match.group(1))
                        price_str = price_match.group(0).replace(',', '')
                        price = float(re.sub(r'[$€£¥]', '', price_str))
                        
                        entities["items"].append({
                            "name": text,
                            "quantity": quantity,
                            "unit_price": price,
                            "total_price": quantity * price
                        })
        
        return entities
    
    def model_classification(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply model-based classification for entities
        
        Args:
            text_blocks: List of text blocks with text and position
            
        Returns:
            Dictionary of classified entities
        """
        if not self.model:
            return self.rule_based_classification(text_blocks)
        
        # Extract features
        features = self.feature_extraction(text_blocks)
        
        if len(features) == 0:
            return {"invoice_number": None, "invoice_date": None, "customer_name": None,
                    "items": [], "subtotal": None, "total": None, "extra_price": []}
        
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
        
        # Create mapping for item-related information
        item_info = {}
        
        for i, pred in enumerate(predictions):
            entity_type = self.ENTITY_TYPES[pred.item()]
            text = text_blocks[i]["text"]
            confidence = F.softmax(logits[i], dim=0)[pred].item()
            print(f"Classified '{text}' as {entity_type} with confidence {confidence:.4f}")
            
            if confidence < 0.5:
                print(f"Low confidence for '{text}' - skipping")
                continue
            
            if entity_type == "invoice_number":
                entities["invoice_number"] = text
                
            elif entity_type == "invoice_date":
                entities["invoice_date"] = text
                
            elif entity_type == "customer_name":
                entities["customer_name"] = text
                
            elif entity_type == "subtotal":
                entities["subtotal"] = text
                
            elif entity_type == "total":
                entities["total"] = text
                
            elif entity_type == "item_name":
                # Create a new item entry or update existing
                item_id = i  # Use index as temporary item ID
                if item_id not in item_info:
                    item_info[item_id] = {"name": text}
                else:
                    item_info[item_id]["name"] = text
            
            elif entity_type == "item_quantity":
                # Extract numeric quantity
                quantity_match = re.search(r'\b(\d+)\b', text)
                if quantity_match:
                    quantity = int(quantity_match.group(1))
                    
                    # Find or create item entry
                    item_id = i  # Use index as temporary item ID
                    if item_id not in item_info:
                        item_info[item_id] = {"quantity": quantity}
                    else:
                        item_info[item_id]["quantity"] = quantity
            
            elif entity_type == "item_price":
                # Extract price
                price = self.extract_price(text)
                if price:
                    # Find or create item entry
                    item_id = i  # Use index as temporary item ID
                    if item_id not in item_info:
                        item_info[item_id] = {"unit_price": price}
                    else:
                        item_info[item_id]["unit_price"] = price
        
        # Convert item_info to list of items
        for item_data in item_info.values():
            if "name" in item_data:  # Only add items that have at least a name
                # Fill in missing fields
                item = {
                    "name": item_data.get("name", "Unknown Item"),
                    "quantity": item_data.get("quantity", 1),
                    "unit_price": item_data.get("unit_price", 0.0)
                }
                
                # Calculate total price if possible
                if "quantity" in item_data and "unit_price" in item_data:
                    item["total_price"] = item_data["quantity"] * item_data["unit_price"]
                else:
                    item["total_price"] = 0.0
                
                entities["items"].append(item)
        
        return entities
    
    def extract_price(self, text: str) -> Optional[float]:
        """
        Extract price from text
        
        Args:
            text: Text containing a price
            
        Returns:
            Extracted price or None
        """
        # Remove currency symbols and commas
        cleaned_text = re.sub(r'[$€£¥,]', '', text)
        
        # Find numbers with optional decimal point
        matches = re.findall(r'\d+\.\d+|\d+', cleaned_text)
        if matches:
            return float(matches[0])
        return None
    
    def combine_results(self, nn_results: Dict, spacy_results: Dict, rule_results: Dict) -> Dict:
        """
        Combine results from different methods with priority
        
        Args:
            nn_results: Results from neural network
            spacy_results: Results from spaCy NER
            rule_results: Results from rule-based approach
            
        Returns:
            Combined results
        """
        combined = {
            "invoice_number": None,
            "invoice_date": None,
            "customer_name": None,
            "items": [],
            "subtotal": None,
            "total": None,
            "extra_price": []
        }
        
        # For each field, prioritize: neural network > spaCy > rule-based
        for field in ["invoice_number", "invoice_date", "customer_name", "subtotal", "total"]:
            combined[field] = nn_results.get(field) or spacy_results.get(field) or rule_results.get(field)
        
        # Combine items from all sources (ensuring no duplicates)
        all_items = []
        all_items.extend(nn_results.get("items", []))
        all_items.extend(spacy_results.get("items", []))
        all_items.extend(rule_results.get("items", []))
        
        # Remove duplicates (basic deduplication by name)
        seen_names = set()
        for item in all_items:
            item_name = item.get("name", "").lower()
            if item_name and item_name not in seen_names:
                seen_names.add(item_name)
                combined["items"].append(item)
        
        # Combine extra price items
        all_extra = []
        all_extra.extend(nn_results.get("extra_price", []))
        all_extra.extend(spacy_results.get("extra_price", []))
        all_extra.extend(rule_results.get("extra_price", []))
        
        # Keep only unique extra price items
        seen_extras = set()
        for extra in all_extra:
            if isinstance(extra, str):
                if extra.lower() not in seen_extras:
                    seen_extras.add(extra.lower())
                    combined["extra_price"].append(extra)
            elif isinstance(extra, dict):
                # For dictionary entries
                extra_key = list(extra.keys())[0] if extra else ""
                if extra_key and extra_key not in seen_extras:
                    seen_extras.add(extra_key)
                    combined["extra_price"].append(extra)
        
        return combined
    
    def classify(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify entities in the extracted text blocks
        
        Args:
            text_blocks: List of text blocks with text and position
            
        Returns:
            Dictionary of classified entities
        """
        # Get results from different methods
        nn_results = self.model_classification(text_blocks) if self.use_nn else {}
        spacy_results = self.spacy_ner_classification(text_blocks) if self.use_spacy else {}
        rule_results = self.rule_based_classification(text_blocks) if self.use_rules else {}
        
        # Combine results
        combined_results = self.combine_results(nn_results, spacy_results, rule_results)
        
        return combined_results
