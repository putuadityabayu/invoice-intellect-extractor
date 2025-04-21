
# Machine Learning Methods for Invoice Information Extraction

This document explains the key machine learning methods used in our invoice extraction system.

## 1. Neural Network Architectures

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

### LayoutLM-inspired Model
A model inspired by LayoutLM that combines text and spatial features:
- Separate encoding branches for text and spatial features
  - Text encoder: Linear(768) → ReLU → Linear(256)
  - Spatial encoder: Linear(4) → ReLU → Linear(64)
- Combined features processing
  - Concatenate text and spatial features
  - Hidden layer 1 (256 neurons) with ReLU activation and dropout (0.3)
  - Hidden layer 2 (128 neurons) with ReLU activation and dropout (0.3)
  - Output layer (8 neurons) - Same entity types as the first model

## 2. Named Entity Recognition with spaCy

### Custom spaCy NER Model
We train a custom Named Entity Recognition model using spaCy to identify entity types directly from text:
- Uses spaCy's entity recognition capabilities
- Custom entity types for invoice elements:
  - INVOICE_NUMBER
  - DATE
  - CUSTOMER
  - ITEM
  - QUANTITY
  - PRICE
  - SUBTOTAL
  - TOTAL
- Training process:
  - Convert annotated text blocks to spaCy training format
  - Train using spaCy's built-in training API
  - Apply transfer learning from base English model

### NER Integration
- The spaCy NER model provides an alternative prediction method
- Results are combined with neural network predictions for improved accuracy
- Especially effective for text with specific formats (dates, invoice numbers)

## 3. Feature Extraction Methods

### Text Embedding
- Uses DistilBERT model for generating text embeddings
- Each text block is tokenized and converted to 768-dimensional vectors
- CLS token embedding is used as the text representation
- This captures semantic meaning of text for classification

### Positional Features
- Incorporates spatial information from OCR output
- Uses normalized coordinates (x_min, y_min, x_max, y_max)
- Helps in understanding document layout and relationships

### Layout Analysis
- Identifies rows and columns in the document
- Groups text blocks into logical structures (headers, items, totals)
- Detects table structures for extracting item information
- Enhances extraction by understanding the document structure

## 4. Ensemble Approach

Our system combines multiple extraction methods:
1. **Neural network classification** - Primary entity type prediction
2. **spaCy NER** - Text-based entity recognition
3. **Layout-aware extraction** - Using spatial information
4. **Rule-based extraction** - Fallback for specific patterns

Results from these methods are combined using a weighted approach that prioritizes:
- Layout-based extraction for structured elements (tables)
- NER for named entities and dates
- Neural network for classification confidence
- Rule-based extraction as fallback

## 5. Training Approach

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

### Layout Model
1. **Data Preparation**
   - Text blocks with position information
   - Text features combined with spatial features
   - Normalization of spatial coordinates

2. **Training Process**
   - Cross-entropy loss function
   - Adam optimizer with learning rate 0.001
   - Batch size of 32
   - Early stopping based on validation loss

### spaCy NER
1. **Data Preparation**
   - Conversion of annotated text blocks to spaCy format
   - Creation of document spans with entity annotations
   - Generation of training and validation sets

2. **Training Process**
   - Transfer learning from base English model
   - Batch size of 8
   - Dropout rate of 0.2
   - Early stopping with patience of 5
   - Maximum of 20 epochs

## 6. Rule-Based Fallback

When model confidence is low or in absence of trained models:
- Regular expressions for date and number formats
- Keyword matching for entity types
- Spatial relationship analysis
- Position-based table structure detection
- Format validation for specific entity types

## 7. Post-processing Methods

1. **Confidence Thresholding**
   - Only accept predictions above confidence threshold
   - Fall back to rule-based for low confidence

2. **Consistency Checks**
   - Validate numerical relationships (e.g., quantity * unit_price = total_price)
   - Date format validation
   - Invoice number format validation

3. **Result Combination**
   - Merge results from different extraction methods
   - Resolve conflicts based on confidence scores
   - Ensure consistency in numerical values

## 8. Future Improvements

1. **Model Architecture**
   - Implement full LayoutLM or LayoutLMv2 with transformer architecture
   - Add bidirectional LSTM layers for sequence understanding
   - Incorporate attention mechanisms to focus on relevant text parts
   - Graph neural networks for better relation extraction

2. **Training Data**
   - Increase dataset diversity with more invoice formats
   - More augmentation techniques for robustness
   - Active learning for hard examples
   - Few-shot learning for rare entity types

3. **Feature Engineering**
   - Include font properties (size, style, weight)
   - Add more contextual features from surrounding text
   - Improved table structure detection
   - Document section classification

4. **Integration with LLMs**
   - Use large language models for text understanding
   - Zero-shot learning for unseen invoice formats
   - Prompt engineering for specific extraction tasks
   - Multimodal models that combine vision and language
