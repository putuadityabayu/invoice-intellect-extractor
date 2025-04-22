
"""
Entity classification model for invoice extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityModel(nn.Module):
    """
    Simple neural network for entity classification
    Classifies text blocks into different entity types (invoice number, date, etc.)
    """
    
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=16):
        """
        Initialize the model
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            num_classes: Number of output classes
        """
        super(EntityModel, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Multilingual token patterns specific to invoices
        self.token_patterns = {
            # English tokens
            "invoice": 0,
            "receipt": 0,
            "bill": 0,
            "date": 1,
            "customer": 2,
            "item": 3,
            "quantity": 4,
            "price": 5,
            "total": 9,
            "subtotal": 7,
            "tax": 11,
            
            # Indonesian tokens
            "faktur": 0,
            "kwitansi": 0,
            "tagihan": 0,
            "tanggal": 1,
            "pelanggan": 2,
            "barang": 3,
            "jumlah": 4,
            "harga": 5,
            "total": 9,
            "subtotal": 7,
            "pajak": 11,
            "ppn": 11,
            "kepada": 2
        }
    
    def forward(self, x):
        """Forward pass"""
        # First hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        # Second hidden layer
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x
    
    def predict(self, features):
        """
        Predict entity type for given features
        
        Args:
            features: Text features
            
        Returns:
            Predicted entity type index
        """
        # Set model to evaluation mode
        self.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(features)
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item()
