
# Invoice Processing Pipeline Documentation

This document provides a detailed explanation of how invoice images are processed through our machine learning pipeline to extract structured data.

## Overview

The invoice processing pipeline consists of the following main steps:

1. **Image Upload and Storage**
2. **Image Preprocessing**
3. **OCR Text Extraction**
4. **Entity Classification**
5. **Layout Analysis**
6. **Relation Extraction**
7. **Result Combination**
8. **Final Output Formatting**

Each step is explained in detail below with the techniques used and output formats.

## 1. Image Upload and Storage

### Process
- Images are uploaded through a web interface or API endpoint
- The system validates if the file is an actual image (PNG, JPG, JPEG, TIFF, or PDF)
- Images are temporarily stored in the server's file system for processing

### Code Example
```python
@app.route("/extract", methods=["POST"])
def extract_invoice_data():
    # Check if image URL is provided
    if "image_url" in request.form:
        image_url = request.form["image_url"]
        
        # Download image from URL
        temp_file_path = os.path.join(app.config["UPLOAD_FOLDER"], "temp_url_image.jpg")
        success, message = download_image_from_url(image_url, temp_file_path)
        
        if not success:
            return jsonify({"error": message}), 400
        
        # Validate image
        if not is_valid_image(temp_file_path):
            os.remove(temp_file_path)
            return jsonify({"error": "Invalid image file downloaded from URL"}), 400
            
        # Process the invoice (continues to next steps)...
    
    # Check if uploaded file is provided
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        # Validate image
        if not is_valid_image(file_path):
            os.remove(file_path)
            return jsonify({"error": "Invalid image file uploaded"}), 400
        
        # Process the invoice (continues to next steps)...
```

### Output
- Temporary image file (.jpg, .png, .tiff, or .pdf) stored in the server's file system
- Output Format: Original image file

## 2. Image Preprocessing

### Process
- The original image is preprocessed to improve OCR accuracy
- Multiple processing techniques are applied:
  - Conversion to grayscale to simplify processing
  - Adaptive thresholding to improve contrast
  - Denoising to remove noise from the image
  - Sharpening to enhance text edges
  - Optional deskewing to correct skewed images
  - Optional resizing to standardize dimensions

### Code Example
```python
def preprocess_image(image_path: str) -> np.ndarray:
    """Apply a series of preprocessing steps to improve image quality for OCR"""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Apply sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened
```

### Output
- Processed image as a numpy array in memory
- Output Format: Numpy array (can be saved as an image file for debugging)

## 3. OCR Text Extraction

### Process
- The preprocessed image is passed to an OCR engine (doctr)
- The OCR engine identifies text regions and recognizes text
- Each recognized text block includes:
  - Text content
  - Confidence score
  - Bounding box coordinates (x_min, y_min, x_max, y_max)

### Code Example
```python
def extract_text_with_positions(image: str) -> List[Dict[str, Any]]:
    """Extract text from an image with positional information"""
    # Create a document from the image
    doc = DocumentFile.from_images(image)
    
    # Run OCR prediction with doctr
    result = ocr_processor.model(doc)
    
    # Extract text blocks with positions
    text_blocks = []
    
    # Get results from pages
    for page in result.pages:
        # Process each block in the page
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    # Extract text and bounding box
                    text = word.value
                    bbox = word.geometry

                    text_blocks.append({
                        "text": text,
                        "confidence": word.confidence,
                        "position": {
                            "x_min": bbox[0][0],
                            "y_min": bbox[0][1],
                            "x_max": bbox[1][0],
                            "y_max": bbox[1][1]
                        }
                    })
                    
    return text_blocks
```

### Output
- List of text blocks with position information
- Output Format: JSON/Dictionary structure

## 4. Entity Classification

### Process
- Each text block is analyzed to identify what type of information it contains
- Multiple approaches are used:
  - spaCy NER: Identifies named entities (dates, organizations, amounts)
  - Custom ML classification: Uses a transformer-based model trained on invoice data
- The model considers:
  - Text content (embeddings)
  - Text position on the document
  - Contextual information

### Code Example
```python
def classify(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify entities in text blocks"""
    # Extract features for each text block
    features = [self._extract_features(block) for block in text_blocks]
    
    # Use spaCy for NER classification
    spacy_results = self._apply_spacy_ner(text_blocks)
    
    # Use custom ML model for classification if available
    if self.use_model:
        model_results = self._apply_model(features, text_blocks)
        
        # Combine results from both approaches
        combined_results = self._combine_results(spacy_results, model_results)
        return combined_results
    else:
        return spacy_results
```

### Output
- Classified entity data with keys like invoice_number, invoice_date, customer_name, etc.
- Output Format: JSON/Dictionary structure

## 5. Layout Analysis

### Process
- A LayoutLM-inspired model analyzes the document structure
- The model processes:
  - Text content
  - Spatial information (coordinates)
  - Visual features
- Transformer architecture with:
  - Multi-head self-attention
  - Bidirectional LSTM layers
  - Positional encodings

### Code Example
```python
def process(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process text blocks using layout-aware model"""
    # Extract spatial features
    spatial_features = []
    for block in text_blocks:
        if "position" in block:
            pos = block["position"]
            spatial_features.append([
                pos.get("x_min", 0),
                pos.get("y_min", 0),
                pos.get("x_max", 0),
                pos.get("y_max", 0)
            ])
        else:
            spatial_features.append([0, 0, 0, 0])
            
    # Extract text features
    text_features = []
    for block in text_blocks:
        # In a real system, this would use pre-trained embeddings
        text = block.get("text", "").lower()
        features = self._text_to_features(text)
        text_features.append(features)
    
    # Combine features
    combined_features = [
        torch.cat([t, torch.tensor(s)], dim=0)
        for t, s in zip(text_features, spatial_features)
    ]
    
    # Apply model if available
    if self.use_model:
        # Run through LayoutLM transformer model
        with torch.no_grad():
            batch = torch.stack(combined_features).unsqueeze(0)
            if torch.cuda.is_available():
                batch = batch.cuda()
                
            outputs = self.model(batch)
            predictions = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()
            
        # Format results
        return self._format_results(text_blocks, predictions)
    else:
        # Fallback to simpler approach
        return {}
```

### Output
- Layout analysis results with entity classification and structural information
- Output Format: JSON/Dictionary structure

## 6. Relation Extraction

### Process
- A BiLSTM-based model with attention is used to identify relationships between entities
- For each potential pair of text blocks, the model predicts if and how they are related
- Particularly useful for extracting line items (product name, quantity, price, etc.)
- The model processes:
  - Text content of both entities
  - Relative positions
  - Distance between entities

### Code Example
```python
def model_extraction(
    self, text_blocks: List[Dict[str, Any]], entities: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply model-based relation extraction"""
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
            
            # Process prediction based on relation type
            # ...
        
        # Add to results
        results["items"].append(item_info)
    
    return results
```

### Output
- Extracted relationships between entities, especially for line items
- Output Format: JSON/Dictionary structure

## 7. Result Combination

### Process
- Results from different extraction methods are combined
- Prioritization logic resolves conflicts between different extraction methods
- For each field, the system selects the most reliable value based on:
  - Confidence scores
  - Consistency with other extracted information
  - Completeness of the extracted data

### Code Example
```python
def combine_extraction_results(
    entity_results: Dict[str, Any],
    layout_results: Dict[str, Any],
    relation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Combine results from different extraction approaches"""
    combined = {
        "invoice_number": None,
        "invoice_date": None,
        "name": None,
        "items": [],
        "subtotal": None,
        "total": None,
        "extra_price": []
    }
    
    # For header fields, prioritize: layout > entity classifier > relation extractor
    for field in ["invoice_number", "invoice_date"]:
        combined[field] = (
            layout_results.get(field) or 
            entity_results.get(field) or 
            relation_results.get(field)
        )
    
    # Customer name
    combined["name"] = (
        layout_results.get("customer_name") or 
        entity_results.get("customer_name") or 
        relation_results.get("name")
    )
    
    # For items, use layout's items if available, otherwise use relation extractor's
    if layout_results.get("items"):
        layout_items = layout_results.get("items", [])
        relation_items = relation_results.get("items", [])
        
        # If both sources have items, use the one with more information
        if layout_items and relation_items:
            # Compute average completeness
            layout_completeness = sum(
                (1 if item.get("name") else 0) + 
                (1 if item.get("quantity") else 0) + 
                (1 if item.get("unit_price") else 0) + 
                (1 if item.get("total_price") else 0) 
                for item in layout_items
            ) / (len(layout_items) * 4) if layout_items else 0
            
            relation_completeness = sum(
                (1 if item.get("name") else 0) + 
                (1 if item.get("quantity") else 0) + 
                (1 if item.get("unit_price") else 0) + 
                (1 if item.get("total_price") else 0) 
                for item in relation_items
            ) / (len(relation_items) * 4) if relation_items else 0
            
            combined["items"] = layout_items if layout_completeness >= relation_completeness else relation_items
        else:
            combined["items"] = layout_items or relation_items
    else:
        combined["items"] = relation_results.get("items", [])
    
    # For total and subtotal
    for field in ["subtotal", "total"]:
        combined[field] = (
            layout_results.get(field) or 
            entity_results.get(field) or 
            relation_results.get(field)
        )
    
    return combined
```

### Output
- Combined and consolidated extraction results
- Output Format: JSON/Dictionary structure

## 8. Final Output Formatting

### Process
- The combined extraction results are formatted into the final output structure
- Ensures all required fields are present, even if empty
- Formats values for consistent representation (numeric formatting, date formatting)
- Validates the overall structure for completeness

### Code Example
```python
def format_invoice_data(final_results: Dict[str, Any], relation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Format extracted data into required JSON structure"""
    # Create standardized output structure
    formatted = {
        "invoice_number": final_results.get("invoice_number", ""),
        "invoice_date": final_results.get("invoice_date", ""),
        "name": final_results.get("name", ""),
        "items": [],
        "total": final_results.get("total", 0)
    }
    
    # Format items
    for item in final_results.get("items", []):
        formatted_item = {
            "name": item.get("name", ""),
            "quantity": item.get("quantity", 0),
            "unit_price": item.get("unit_price", 0),
            "total_price": item.get("total_price", 0)
        }
        formatted["items"].append(formatted_item)
    
    # Add extra price items if present
    if final_results.get("extra_price"):
        for extra in final_results.get("extra_price", []):
            if isinstance(extra, dict) and len(extra) > 0:
                key = next(iter(extra.keys()))
                value = extra[key]
                formatted[key] = value
    
    return formatted
```

### Output
- Final structured invoice data
- Output Format: JSON/Dictionary structure

## Machine Learning Methods

Our system uses a combination of the following machine learning approaches:

### 1. LayoutLM-inspired Model with Transformer Architecture
- **Transformer Blocks**: Multi-head self-attention mechanisms
- **Positional Encoding**: To capture the spatial layout of text
- **Bidirectional LSTM Layers**: To capture sequential dependencies
- **Spatial Features**: Normalized bounding box coordinates

### 2. Relation Extraction with BiLSTM and Attention
- **Bidirectional LSTM**: For understanding relationships between text blocks
- **Attention Mechanism**: To focus on relevant parts of the text
- **Layer Normalization**: For training stability

For more details on our machine learning methods, refer to the [ML Methods documentation](ml-methods.md).

## Debug Output Files

For debugging purposes, the system can generate intermediate output files at each step of the pipeline:

1. **Preprocessed Image**: `debug/preprocessed_image.jpg`
2. **OCR Results**: `debug/ocr_results.json`
3. **Entity Classification Results**: `debug/entity_results.json`
4. **Layout Analysis Results**: `debug/layout_results.json`
5. **Relation Extraction Results**: `debug/relation_results.json`
6. **Combined Results**: `debug/combined_results.json`
7. **Final Output**: `debug/final_output.json`

These files can be examined to understand how information flows through the pipeline and to identify potential issues at each stage.
