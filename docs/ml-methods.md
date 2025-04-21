
# Machine Learning Methods for Invoice Information Extraction

This document explains the key machine learning methods used in our invoice extraction system.

## 1. Neural Network Architecture

### Entity Classifier Model
We use a simple feedforward neural network with the following architecture:
- Input layer (768 dimensions) - Takes text features from transformer embeddings
- Hidden layer 1 (256 neurons) with ReLU activation and dropout (0.3)
- Hidden layer 2 (128 neurons) with ReLU activation and dropout (0.3)
- Output layer (8 neurons) - One for each entity type
  - invoice_number
  - invoice_date
  - customer_name
  - item_name
  - item_quantity
  - item_price
  - subtotal
  - total

### Relation Extractor Model
Similar architecture but designed for identifying relationships between text elements:
- Input layer (768*2 dimensions) - Takes concatenated features from two text elements
- Hidden layer 1 (256 neurons) with ReLU activation and dropout (0.3)
- Hidden layer 2 (128 neurons) with ReLU activation and dropout (0.3)
- Output layer (4 neurons) - One for each relation type
  - none
  - item_quantity
  - item_price
  - item_total

## 2. Feature Extraction Methods

### Text Embedding
- Uses DistilBERT model for generating text embeddings
- Each text block is tokenized and converted to 768-dimensional vectors
- CLS token embedding is used as the text representation
- This captures semantic meaning of text for classification

### Positional Features
- Incorporates spatial information from OCR output
- Uses normalized coordinates (x_min, y_min, x_max, y_max)
- Helps in understanding document layout and relationships

## 3. Training Approach

### Entity Classification
1. **Data Preparation**
   - Text blocks labeled with entity types
   - Augmentation with different fonts, sizes, and rotations
   - Balance dataset across all entity types

2. **Training Process**
   - Cross-entropy loss function
   - Adam optimizer with learning rate 0.001
   - Batch size of 32
   - Early stopping based on validation loss

### Relation Extraction
1. **Data Preparation**
   - Pairs of text blocks labeled with relation types
   - Negative sampling for "none" relations
   - Consideration of spatial proximity

2. **Training Process**
   - Cross-entropy loss function
   - Adam optimizer with learning rate 0.001
   - Batch size of 32
   - Early stopping based on validation loss

## 4. Rule-Based Fallback

When model confidence is low or in absence of trained models:
- Regular expressions for date and number formats
- Keyword matching for entity types
- Spatial relationship analysis
- Position-based table structure detection

## 5. Post-processing Methods

1. **Confidence Thresholding**
   - Only accept predictions above confidence threshold
   - Fall back to rule-based for low confidence

2. **Consistency Checks**
   - Validate numerical relationships (e.g., quantity * unit_price = total_price)
   - Date format validation
   - Invoice number format validation

## 6. Future Improvements

1. **Model Architecture**
   - Implement attention mechanisms
   - Add bidirectional LSTM layers
   - Graph neural networks for better relation extraction

2. **Training Data**
   - Increase dataset diversity
   - More augmentation techniques
   - Active learning for hard examples

3. **Feature Engineering**
   - Include font properties
   - Add more contextual features
   - Improved table structure detection

