
"""
Relation extraction model for invoice processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationModel(nn.Module):
    """
    Simple neural network for relation extraction
    Predicts relation types between pairs of text blocks
    """
    
    def __init__(self, input_dim=768*2, hidden_dim=512, num_classes=4):
        """
        Initialize the model
        
        Args:
            input_dim: Dimension of input features (combined features of source and target)
            hidden_dim: Dimension of hidden layer
            num_classes: Number of output classes
        """
        super(RelationModel, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Relation hints for different languages
        self.relation_hints = {
            # English relation patterns
            "quantity": "item_quantity",
            "qty": "item_quantity",
            "unit": "item_quantity",
            "price": "item_price",
            "cost": "item_price",
            "rate": "item_price",
            "amount": "item_total",
            "total": "item_total",
            "sum": "item_total",
            
            # Indonesian relation patterns
            "jumlah": "item_quantity",
            "unit": "item_quantity",
            "qty": "item_quantity",
            "kuantitas": "item_quantity",
            "harga": "item_price",
            "biaya": "item_price",
            "tarif": "item_price",
            "total": "item_total",
            "jumlah": "item_total",
            "subtotal": "item_total"
        }
        
        # BiLSTM for processing source-target pairs more contextually
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features (concatenated source and target features)
            
        Returns:
            Output logits
        """
        # First hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        # Add BiLSTM processing for batched inputs
        batch_size = x.size(0)
        if batch_size > 1:
            # Reshape for sequence processing
            x_seq = x.view(batch_size, 1, -1)
            x_seq, _ = self.bilstm(x_seq)
            # Flatten back
            x = x_seq.view(batch_size, -1)
        
        # Second hidden layer
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x
    
    def predict(self, features):
        """
        Predict relation type for given features
        
        Args:
            features: Source and target combined features
            
        Returns:
            Predicted relation type index
        """
        # Set model to evaluation mode
        self.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(features)
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item()


class RelationTransformer(nn.Module):
    """
    More advanced model for relation extraction based on transformer architecture
    Better for capturing contextual information from multilingual invoices
    """
    
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4):
        """
        Initialize the model
        
        Args:
            input_dim: Dimension of individual entity features
            hidden_dim: Dimension of hidden layer
            num_classes: Number of output classes
        """
        super(RelationTransformer, self).__init__()
        
        # Feature transformation
        self.fc_source = nn.Linear(input_dim, hidden_dim)
        self.fc_target = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*2,
            dropout=0.1,
            batch_first=True
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features (concatenated source and target features)
            
        Returns:
            Output logits
        """
        # Split the input into source and target features
        batch_size = x.size(0)
        split_size = x.size(1) // 2
        source_features = x[:, :split_size]
        target_features = x[:, split_size:]
        
        # Transform features
        source_hidden = F.relu(self.fc_source(source_features))
        target_hidden = F.relu(self.fc_target(target_features))
        
        # Prepare for transformer (sequence of length 2: source, target)
        sequence = torch.stack([source_hidden, target_hidden], dim=1)
        
        # Apply transformer
        transformed = self.transformer_layer(sequence)
        
        # Flatten and combine
        flat_transformed = torch.cat([
            transformed[:, 0, :], 
            transformed[:, 1, :]
        ], dim=1)
        
        # Classify
        output = self.classifier(flat_transformed)
        
        return output
