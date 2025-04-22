
"""
Dataset classes for loading and processing invoice data for training
"""

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModel


class EntityDataset(Dataset):
    """Dataset for entity classification training"""
    
    def __init__(self, data_dir, split="train", transform=None):
        """
        Initialize entity classification dataset
        
        Args:
            data_dir: Root directory of dataset
            split: train, val, or test
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Initialize tokenizer and model for feature extraction
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Using random features instead")
            self.tokenizer = None
            self.model = None
        
        # Load annotations
        self.annotations_dir = os.path.join(data_dir, "annotations", split)
        self.image_dir = os.path.join(data_dir, "images", split)
        
        self.samples = []
        for ann_file in os.listdir(self.annotations_dir):
            if ann_file.endswith(".json"):
                with open(os.path.join(self.annotations_dir, ann_file), "r") as f:
                    annotation = json.load(f)
                    
                    # Extract features for each text block
                    for text_block in annotation["text_blocks"]:
                        self.samples.append({
                            "text": text_block["text"],
                            "position": text_block["position"],
                            "entity_type": text_block["entity_type"],
                            "image_path": os.path.join(self.image_dir, 
                                                      os.path.basename(annotation["image_path"]))
                        })
        
        # Entity type mapping - updated to include all entity types found in dataset
        self.entity_types = {
            "invoice_number": 0,
            "invoice_date": 1,
            "customer_name": 2,
            "item_name": 3,
            "item_quantity": 4,
            "item_price": 5,
            "item_total": 6,  # Added missing entity type
            "subtotal": 7,
            "subtotal_label": 8,
            "total": 9,
            "total_label": 10,
            "tax": 11,
            "tax_label": 12,
            "header": 13,
            "company_name": 14,
            "label": 15,
            "table_header": 16
        }
    
    def __len__(self):
        return len(self.samples)
    
    def extract_features(self, text):
        """
        Extract text features using pretrained model
        
        Args:
            text: Input text
            
        Returns:
            Feature tensor
        """
        if self.tokenizer is None or self.model is None:
            # Return random features if model not available
            return torch.randn(768)
        
        # Tokenize and get model outputs
        inputs = self.tokenizer(text, return_tensors="pt", 
                               padding="max_length", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token embedding as text features
        return outputs.last_hidden_state[:, 0, :].squeeze()
    
    def __getitem__(self, idx):
        """Get sample by index"""
        sample = self.samples[idx]
        
        # Extract text features
        features = self.extract_features(sample["text"])
        
        # Get label - add fallback for unknown entity types
        label = self.entity_types.get(sample["entity_type"], 0)  # Default to first class if unknown
        
        return features, torch.tensor(label)


class RelationDataset(Dataset):
    """Dataset for relation extraction training"""
    
    def __init__(self, data_dir, split="train", transform=None):
        """
        Initialize relation extraction dataset
        
        Args:
            data_dir: Root directory of dataset
            split: train, val, or test
            transform: Optional transforms to apply
        """
        # ... keep existing code (initialization part)
        
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Initialize tokenizer and model for feature extraction
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Using random features instead")
            self.tokenizer = None
            self.model = None
        
        # Load annotations
        self.annotations_dir = os.path.join(data_dir, "annotations", split)
        self.image_dir = os.path.join(data_dir, "images", split)
        
        self.samples = []
        for ann_file in os.listdir(self.annotations_dir):
            if ann_file.endswith(".json"):
                with open(os.path.join(self.annotations_dir, ann_file), "r") as f:
                    annotation = json.load(f)
                    
                    # Create a map of text blocks
                    text_blocks = {}
                    for i, block in enumerate(annotation.get("text_blocks", [])):
                        block_id = block.get("id", i)
                        text_blocks[block_id] = block
                    
                    # Extract features for each relation
                    for relation in annotation.get("relations", []):
                        source_id = relation.get("source_id", relation.get("head", None))
                        target_id = relation.get("target_id", relation.get("tail", None))
                        
                        if source_id is None or target_id is None:
                            continue
                            
                        if source_id not in text_blocks or target_id not in text_blocks:
                            continue
                            
                        source_block = text_blocks[source_id]
                        target_block = text_blocks[target_id]
                        
                        self.samples.append({
                            "source_text": source_block["text"],
                            "target_text": target_block["text"],
                            "source_position": source_block["position"],
                            "target_position": target_block["position"],
                            "relation_type": relation["relation_type"],
                            "image_path": os.path.join(self.image_dir, 
                                                     os.path.basename(annotation["image_path"]))
                        })
        
        # Relation type mapping
        self.relation_types = {
            "none": 0,
            "item_quantity": 1,
            "item_price": 2,
            "item_total": 3
        }
    
    def __len__(self):
        return len(self.samples)
    
    def extract_features(self, text):
        """
        Extract text features using pretrained model
        
        Args:
            text: Input text
            
        Returns:
            Feature tensor
        """
        if self.tokenizer is None or self.model is None:
            # Return random features if model not available
            return torch.randn(768)
        
        # Tokenize and get model outputs
        inputs = self.tokenizer(text, return_tensors="pt", 
                               padding="max_length", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token embedding as text features
        return outputs.last_hidden_state[:, 0, :].squeeze()
    
    def __getitem__(self, idx):
        """Get sample by index"""
        sample = self.samples[idx]
        
        # Extract text features for source and target
        source_features = self.extract_features(sample["source_text"])
        target_features = self.extract_features(sample["target_text"])
        
        # Concatenate features
        features = torch.cat([source_features, target_features], dim=0)
        
        # Get label with fallback
        label = self.relation_types.get(sample["relation_type"], 0)  # Default to first class if unknown
        
        return features, torch.tensor(label)
