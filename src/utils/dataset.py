
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
            # Use multilingual model to better support Indonesian
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
            self.model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Using random features instead")
            self.tokenizer = None
            self.model = None
        
        # Load annotations
        self.annotations_dir = os.path.join(data_dir, "annotations", split)
        self.image_dir = os.path.join(data_dir, "images", split)
        
        self.samples = []
        
        # Collect all entity types first to ensure comprehensive mapping
        all_entity_types = set()
        
        # First pass to collect all entity types
        for ann_file in os.listdir(self.annotations_dir):
            if ann_file.endswith(".json"):
                with open(os.path.join(self.annotations_dir, ann_file), "r") as f:
                    annotation = json.load(f)
                    for text_block in annotation.get("text_blocks", []):
                        if "entity_type" in text_block:
                            all_entity_types.add(text_block["entity_type"])
        
        # Create entity type mapping dynamically
        self.entity_types = {entity_type: i for i, entity_type in enumerate(sorted(all_entity_types))}
        print(f"Found {len(self.entity_types)} entity types: {self.entity_types}")
        
        # Now load the actual samples
        for ann_file in os.listdir(self.annotations_dir):
            if ann_file.endswith(".json"):
                with open(os.path.join(self.annotations_dir, ann_file), "r") as f:
                    annotation = json.load(f)
                    
                    # Extract features for each text block
                    for text_block in annotation.get("text_blocks", []):
                        if "entity_type" in text_block:
                            self.samples.append({
                                "text": text_block["text"],
                                "position": text_block["position"],
                                "entity_type": text_block["entity_type"],
                                "image_path": os.path.join(self.image_dir, 
                                                          os.path.basename(annotation.get("image_path", "")))
                            })
    
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
        if sample["entity_type"] not in self.entity_types:
            print(f"Warning: Unknown entity type {sample['entity_type']}")
            label = 0  # Default to first class if unknown
        else:
            label = self.entity_types[sample["entity_type"]]
        
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
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Initialize tokenizer and model for feature extraction
        try:
            # Use multilingual model to better support Indonesian
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
            self.model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Using random features instead")
            self.tokenizer = None
            self.model = None
        
        # Load annotations
        self.annotations_dir = os.path.join(data_dir, "annotations", split)
        self.image_dir = os.path.join(data_dir, "images", split)
        
        self.samples = []
        
        # Collect all relation types first to ensure comprehensive mapping
        all_relation_types = set()
        
        # First pass to collect all relation types
        for ann_file in os.listdir(self.annotations_dir):
            if ann_file.endswith(".json"):
                with open(os.path.join(self.annotations_dir, ann_file), "r") as f:
                    annotation = json.load(f)
                    for relation in annotation.get("relations", []):
                        if "relation_type" in relation:
                            all_relation_types.add(relation["relation_type"])
        
        # Add "none" relation type if not present
        all_relation_types.add("none")
        
        # Create relation type mapping dynamically
        self.relation_types = {relation_type: i for i, relation_type in enumerate(sorted(all_relation_types))}
        print(f"Found {len(self.relation_types)} relation types: {self.relation_types}")
        
        # Now load the actual samples
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
                                                     os.path.basename(annotation.get("image_path", "")))
                        })
    
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
        if sample["relation_type"] not in self.relation_types:
            print(f"Warning: Unknown relation type {sample['relation_type']}")
            label = 0  # Default to first class if unknown
        else:
            label = self.relation_types[sample["relation_type"]]
        
        return features, torch.tensor(label)
