
"""
Script for training the custom ML models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
import json
import random
import shutil
from tqdm import tqdm
from src.models.entity_classifier import EntityModel
from src.models.relation_extractor import RelationModel
from src.models.layout_model import LayoutModel
from src.utils.dataset import EntityDataset, RelationDataset

# Try to import spacy for NER training
try:
    import spacy
    from spacy.tokens import DocBin
    from spacy.training import Example
    SPACY_AVAILABLE = True
except ImportError:
    print("spaCy not available. NER training will be skipped.")
    SPACY_AVAILABLE = False


class LayoutDataset(Dataset):
    """Dataset for layout-aware model training"""
    
    def __init__(self, data_dir, split="train", transform=None):
        """
        Initialize layout dataset
        
        Args:
            data_dir: Root directory of dataset
            split: train, val, or test
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
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
                        if "position" in text_block and "entity_type" in text_block:
                            self.samples.append({
                                "text": text_block["text"],
                                "position": text_block["position"],
                                "entity_type": text_block["entity_type"],
                                "image_path": os.path.join(self.image_dir, 
                                                          os.path.basename(annotation["image_path"]))
                            })
        
        # Entity type mapping
        self.entity_types = {
            "invoice_number": 0,
            "invoice_date": 1,
            "customer_name": 2,
            "item_name": 3,
            "item_quantity": 4,
            "item_price": 5,
            "subtotal": 6,
            "total": 7
        }
    
    def __len__(self):
        return len(self.samples)
    
    def get_spatial_features(self, position):
        """Extract normalized spatial features"""
        # Normalize to 0-1 range
        page_width = 1000.0  # Assumed page width
        page_height = 1000.0  # Assumed page height
        
        x_min = position.get("x_min", 0.0) / page_width
        y_min = position.get("y_min", 0.0) / page_height
        x_max = position.get("x_max", 1.0) / page_width
        y_max = position.get("y_max", 1.0) / page_height
        
        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
    
    def get_text_features(self, text):
        """Generate simple text features (placeholder)"""
        # In a real implementation, this would use text embeddings
        # Here we create dummy features with text length and character frequencies
        features = torch.zeros(768)
        
        # Text length feature
        features[0] = min(len(text) / 100.0, 1.0)
        
        # Character frequency features
        for i, char in enumerate(text.lower()[:50]):
            features[i + 1] = ord(char) / 255.0
            
        # Simple keyword features
        keywords = {
            "invoice": 100, "number": 101, "date": 102, "customer": 103, 
            "bill": 104, "item": 105, "quantity": 106, "price": 107, 
            "total": 108, "subtotal": 109, "tax": 110, "amount": 111
        }
        
        for keyword, idx in keywords.items():
            if keyword in text.lower():
                features[idx] = 1.0
                
        return features
    
    def __getitem__(self, idx):
        """Get sample by index"""
        sample = self.samples[idx]
        
        # Extract text features (simplified)
        text_features = self.get_text_features(sample["text"])
        
        # Extract spatial features
        spatial_features = self.get_spatial_features(sample["position"])
        
        # Get label
        label = self.entity_types[sample["entity_type"]]
        
        return text_features, spatial_features, torch.tensor(label)


class DummyEntityDataset(Dataset):
    """
    Dummy dataset for entity classification training
    In a real scenario, this would load actual training data
    """
    
    def __init__(self, size=1000, feature_dim=768, num_classes=8):
        """Initialize dummy dataset"""
        self.size = size
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Generate dummy features and labels
        self.features = torch.randn(size, feature_dim)
        self.labels = torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DummyRelationDataset(Dataset):
    """
    Dummy dataset for relation extraction training
    In a real scenario, this would load actual training data
    """
    
    def __init__(self, size=1000, feature_dim=768*2, num_classes=4):
        """Initialize dummy dataset"""
        self.size = size
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Generate dummy features and labels
        self.features = torch.randn(size, feature_dim)
        self.labels = torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DummyLayoutDataset(Dataset):
    """
    Dummy dataset for layout model training
    In a real scenario, this would load actual training data
    """
    
    def __init__(self, size=1000, text_dim=768, spatial_dim=4, num_classes=8):
        """Initialize dummy dataset"""
        self.size = size
        self.text_dim = text_dim
        self.spatial_dim = spatial_dim
        self.num_classes = num_classes
        
        # Generate dummy features and labels
        self.text_features = torch.randn(size, text_dim)
        self.spatial_features = torch.rand(size, spatial_dim)  # 0-1 normalized
        self.labels = torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.text_features[idx], self.spatial_features[idx], self.labels[idx]


def prepare_spacy_training_data(data_dir):
    """
    Prepare training data for spaCy NER model
    
    Args:
        data_dir: Directory containing dataset
        
    Returns:
        True if successful, False otherwise
    """
    if not SPACY_AVAILABLE:
        print("spaCy not available. Skipping NER preparation.")
        return False
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(data_dir), "models", "spacy_ner")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base model
    try:
        nlp = spacy.blank("en")
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        return False
    
    # Create DocBin to store training documents
    train_doc_bin = DocBin()
    val_doc_bin = DocBin()
    
    # Entity labels we want to train
    entity_labels = ["INVOICE_NUMBER", "DATE", "CUSTOMER", "ITEM", "QUANTITY", "PRICE", "SUBTOTAL", "TOTAL"]
    
    # Add entity labels to NER pipe
    ner = nlp.add_pipe("ner")
    for label in entity_labels:
        ner.add_label(label)
    
    # Map our entity types to spaCy format
    entity_map = {
        "invoice_number": "INVOICE_NUMBER",
        "invoice_date": "DATE",
        "customer_name": "CUSTOMER",
        "item_name": "ITEM",
        "item_quantity": "QUANTITY",
        "item_price": "PRICE",
        "subtotal": "SUBTOTAL",
        "total": "TOTAL"
    }
    
    # Process training data
    print("Preparing spaCy NER training data...")
    
    for split in ["train", "val"]:
        annotations_dir = os.path.join(data_dir, "annotations", split)
        
        if not os.path.exists(annotations_dir):
            print(f"Directory not found: {annotations_dir}")
            continue
        
        for ann_file in tqdm(os.listdir(annotations_dir)):
            if not ann_file.endswith(".json"):
                continue
                
            with open(os.path.join(annotations_dir, ann_file), "r") as f:
                try:
                    annotation = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON file: {ann_file}")
                    continue
            
            # Create a dictionary to map position to text block
            text_blocks_by_position = {}
            for block in annotation.get("text_blocks", []):
                if "position" in block:
                    pos_key = (
                        block["position"].get("x_min", 0),
                        block["position"].get("y_min", 0),
                        block["position"].get("x_max", 0),
                        block["position"].get("y_max", 0)
                    )
                    text_blocks_by_position[pos_key] = block
            
            # Create a full text document by concatenating all text blocks
            full_text = " ".join(block.get("text", "") for block in annotation.get("text_blocks", []))
            doc = nlp.make_doc(full_text)
            
            # Map entity spans in the full text
            ents = []
            for block in annotation.get("text_blocks", []):
                if "entity_type" in block and block["entity_type"] in entity_map:
                    text = block.get("text", "")
                    if text in full_text:
                        start_idx = full_text.find(text)
                        end_idx = start_idx + len(text)
                        spacy_label = entity_map[block["entity_type"]]
                        span = doc.char_span(start_idx, end_idx, label=spacy_label)
                        if span:
                            ents.append(span)
            
            # Set entities on the document
            try:
                doc.ents = ents
                # Add to appropriate DocBin
                if split == "train":
                    train_doc_bin.add(doc)
                else:
                    val_doc_bin.add(doc)
            except Exception as e:
                print(f"Error adding entities to document: {e}")
    
    # Save DocBin to disk
    train_path = os.path.join(output_dir, "train.spacy")
    val_path = os.path.join(output_dir, "dev.spacy")
    
    train_doc_bin.to_disk(train_path)
    val_doc_bin.to_disk(val_path)
    
    print(f"Saved spaCy training data to {train_path} and {val_path}")
    
    # Create config file for training
    config = {
        "lang": "en",
        "pipeline": ["ner"],
        "components": {
            "ner": {
                "source": "en_core_web_sm"
            }
        },
        "training": {
            "dev_corpus": "corpora.dev",
            "train_corpus": "corpora.train",
            "batch_size": 8,
            "dropout": 0.2,
            "patience": 5,
            "max_epochs": 20,
            "optimizer": {
                "@optimizers": "Adam.v1",
                "beta1": 0.9,
                "beta2": 0.999,
                "L2_is_weight_decay": True,
                "L2": 0.01,
                "grad_clip": 1.0,
                "learn_rate": 0.001,
                "optimizer_params": {
                    "eps": 1e-8
                }
            }
        },
        "nlp": {
            "batch_size": 128,
            "tokenizer": {
                "@tokenizers": "spacy.Tokenizer.v1"
            }
        },
        "corpus": {
            "train": {
                "@readers": "spacy.Corpus.v1",
                "path": train_path,
                "max_length": 0
            },
            "dev": {
                "@readers": "spacy.Corpus.v1",
                "path": val_path,
                "max_length": 0
            }
        },
        "paths": {
            "train": train_path,
            "dev": val_path
        }
    }
    
    # Save config file
    config_path = os.path.join(output_dir, "config.cfg")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created spaCy config file at {config_path}")
    
    # Create train script
    train_script = f"""
# Train spaCy NER model
python -m spacy train {config_path} --output {output_dir} --paths.train {train_path} --paths.dev {val_path}
"""
    
    script_path = os.path.join(output_dir, "train.sh")
    with open(script_path, "w") as f:
        f.write(train_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created training script at {script_path}")
    print("To train the spaCy NER model, run the following command:")
    print(f"bash {script_path}")
    
    return True


def train_entity_model(
    data_dir: str, output_dir: str, epochs: int = 10, batch_size: int = 32, lr: float = 0.001
):
    """
    Train the entity classification model
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and data loaders
    if data_dir and os.path.exists(os.path.join(data_dir, "annotations")):
        print(f"Using real dataset from {data_dir}")
        train_dataset = EntityDataset(data_dir, split="train")
        val_dataset = EntityDataset(data_dir, split="val")
    else:
        print("Using dummy dataset for entity classification")
        train_dataset = DummyEntityDataset()
        val_dataset = DummyEntityDataset(size=200)  # Smaller validation set
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = EntityModel()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Print statistics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = os.path.join(output_dir, "entity_model.pth")
            torch.save(model.state_dict(), output_path)
            print(f"Model improved and saved to {output_path}")
    
    print(f"Training completed. Best model saved with validation loss: {best_val_loss:.4f}")


def train_relation_model(
    data_dir: str, output_dir: str, epochs: int = 10, batch_size: int = 32, lr: float = 0.001
):
    """
    Train the relation extraction model
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and data loaders
    if data_dir and os.path.exists(os.path.join(data_dir, "annotations")):
        print(f"Using real dataset from {data_dir}")
        train_dataset = RelationDataset(data_dir, split="train")
        val_dataset = RelationDataset(data_dir, split="val")
    else:
        print("Using dummy dataset for relation extraction")
        train_dataset = DummyRelationDataset()
        val_dataset = DummyRelationDataset(size=200)  # Smaller validation set
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = RelationModel()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Print statistics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = os.path.join(output_dir, "relation_model.pth")
            torch.save(model.state_dict(), output_path)
            print(f"Model improved and saved to {output_path}")
    
    print(f"Training completed. Best model saved with validation loss: {best_val_loss:.4f}")


def train_layout_model(
    data_dir: str, output_dir: str, epochs: int = 10, batch_size: int = 32, lr: float = 0.001
):
    """
    Train the layout-aware model
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and data loaders
    if data_dir and os.path.exists(os.path.join(data_dir, "annotations")):
        print(f"Using real dataset from {data_dir}")
        train_dataset = LayoutDataset(data_dir, split="train")
        val_dataset = LayoutDataset(data_dir, split="val")
    else:
        print("Using dummy dataset for layout model")
        train_dataset = DummyLayoutDataset()
        val_dataset = DummyLayoutDataset(size=200)  # Smaller validation set
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = LayoutModel()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for text_features, spatial_features, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(text_features, spatial_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for text_features, spatial_features, labels in val_loader:
                outputs = model(text_features, spatial_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Print statistics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = os.path.join(output_dir, "layout_model.pth")
            torch.save(model.state_dict(), output_path)
            print(f"Model improved and saved to {output_path}")
    
    print(f"Training completed. Best model saved with validation loss: {best_val_loss:.4f}")


def train_spacy_ner(data_dir: str, output_dir: str):
    """
    Prepare and train spaCy NER model
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Directory to save the trained model
    """
    if not SPACY_AVAILABLE:
        print("spaCy not available. Skipping NER training.")
        return
    
    # Prepare training data
    success = prepare_spacy_training_data(data_dir)
    
    if not success:
        print("Failed to prepare spaCy training data. Skipping NER training.")
        return
    
    # spaCy directory
    spacy_dir = os.path.join(output_dir, "spacy_ner")
    
    # Check if training script exists
    train_script = os.path.join(spacy_dir, "train.sh")
    if not os.path.exists(train_script):
        print(f"Training script not found at {train_script}")
        return
    
    print("Starting spaCy NER training...")
    
    # Run training script
    try:
        import subprocess
        result = subprocess.run(f"bash {train_script}", shell=True, check=True)
        print("spaCy NER training completed!")
    except Exception as e:
        print(f"Error during spaCy NER training: {e}")
        print("Please run the training script manually:")
        print(f"bash {train_script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train invoice extraction models")
    parser.add_argument(
        "--data_dir", type=str, default=None, 
        help="Directory containing the dataset (with images/ and annotations/ subdirectories)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models", help="Directory to save trained models"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--model", type=str, default="all", 
        choices=["entity", "relation", "layout", "spacy", "all"],
        help="Which model to train (entity, relation, layout, spacy, or all)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model in ["entity", "all"]:
        print("Training entity classification model...")
        train_entity_model(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr)
        
    if args.model in ["relation", "all"]:
        print("Training relation extraction model...")
        train_relation_model(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr)
    
    if args.model in ["layout", "all"]:
        print("Training layout-aware model...")
        train_layout_model(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr)
    
    if args.model in ["spacy", "all"] and SPACY_AVAILABLE:
        print("Training spaCy NER model...")
        train_spacy_ner(args.data_dir, args.output_dir)
