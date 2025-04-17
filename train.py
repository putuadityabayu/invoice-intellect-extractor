
"""
Script for training the custom ML models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.models.entity_classifier import EntityModel
from src.models.relation_extractor import RelationModel
import argparse
from pathlib import Path


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


def train_entity_model(
    output_dir: str, epochs: int = 10, batch_size: int = 32, lr: float = 0.001
):
    """
    Train the entity classification model
    
    Args:
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy dataset
    train_dataset = DummyEntityDataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = EntityModel()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
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
        
        # Print statistics
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    # Save the model
    output_path = os.path.join(output_dir, "entity_model.pth")
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


def train_relation_model(
    output_dir: str, epochs: int = 10, batch_size: int = 32, lr: float = 0.001
):
    """
    Train the relation extraction model
    
    Args:
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy dataset
    train_dataset = DummyRelationDataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = RelationModel()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
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
        
        # Print statistics
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    # Save the model
    output_path = os.path.join(output_dir, "relation_model.pth")
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train invoice extraction models")
    parser.add_argument(
        "--output_dir", type=str, default="models", help="Directory to save trained models"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--model", type=str, default="both", choices=["entity", "relation", "both"],
        help="Which model to train (entity, relation, or both)"
    )
    
    args = parser.parse_args()
    
    if args.model in ["entity", "both"]:
        print("Training entity classification model...")
        train_entity_model(args.output_dir, args.epochs, args.batch_size, args.lr)
        
    if args.model in ["relation", "both"]:
        print("Training relation extraction model...")
        train_relation_model(args.output_dir, args.epochs, args.batch_size, args.lr)
