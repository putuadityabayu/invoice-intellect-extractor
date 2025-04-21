
# Invoice Processing Pipeline Schema

This document visualizes the data flow through the invoice processing pipeline.

```
┌────────────────┐
│                │
│  Input Image   │
│                │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│                │
│  Preprocessing │ -- Output: preprocessed_image.jpg
│                │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│                │
│  OCR Extraction│ -- Output: ocr_results.json, ocr_visual_results.txt
│                │
└───────┬────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│                                             │
│             Parallel Processing             │
│                                             │
├───────────────┬───────────────┬─────────────┤
│               │               │             │
│    Entity     │    Layout     │  Relation   │
│ Classification│   Analysis    │ Extraction  │
│               │               │             │
└───────┬───────┘       │       └─────┬───────┘
        │               │             │
        │               │             │
        ▼               ▼             ▼
┌───────────────┐┌───────────────┐┌───────────────┐
│               ││               ││               │
│entity_results.│││layout_results.││relation_results.│
│   json       ││    json       ││   json       │
│               ││               ││               │
└───────┬───────┘└───────┬───────┘└───────┬───────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
                ┌────────────────┐
                │                │
                │    Combine     │ -- Output: combined_results.json
                │    Results     │
                │                │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │                │
                │    Format      │ -- Output: final_output.json
                │    Output      │
                │                │
                └───────┬────────┘
                        │
                        ▼
                ┌────────────────┐
                │                │
                │  Final JSON    │
                │    Output      │
                │                │
                └────────────────┘
```

## Data Schema at Each Stage

### 1. Input Image
- Format: JPG, PNG, TIFF, or PDF
- Properties: Original invoice image

### 2. Preprocessed Image
- Format: Numpy array / JPG
- Properties: Grayscale, thresholded, denoised, sharpened

### 3. OCR Results
- Format: JSON array of objects
- Properties for each text block:
  ```json
  {
    "text": "Example text",
    "confidence": 0.98,
    "position": {
      "x_min": 0.1,
      "y_min": 0.2,
      "x_max": 0.3,
      "y_max": 0.4
    }
  }
  ```

### 4. Entity Classification Results
- Format: JSON object
- Properties:
  ```json
  {
    "invoice_number": "INV-12345",
    "invoice_date": "2023-04-15",
    "customer_name": "ACME Corp",
    "subtotal": 100.0,
    "total": 110.0,
    "extra_price": []
  }
  ```

### 5. Layout Analysis Results
- Format: JSON object
- Properties:
  ```json
  {
    "invoice_number": "INV-12345",
    "invoice_date": "2023-04-15",
    "customer_name": "ACME Corp",
    "items": [
      {
        "name": "Product A",
        "quantity": 2,
        "unit_price": 10.0,
        "total_price": 20.0
      }
    ],
    "subtotal": 100.0,
    "total": 110.0,
    "extra_price": []
  }
  ```

### 6. Relation Extraction Results
- Format: JSON object
- Properties:
  ```json
  {
    "invoice_number": "INV-12345",
    "invoice_date": "2023-04-15",
    "name": "ACME Corp",
    "items": [
      {
        "name": "Product A",
        "quantity": 2,
        "unit_price": 10.0,
        "total_price": 20.0
      }
    ],
    "subtotal": 100.0,
    "total": 110.0,
    "extra_price": []
  }
  ```

### 7. Combined Results
- Format: JSON object
- Properties: Same structure as layout and relation results, but with the best data from all sources

### 8. Final Output
- Format: JSON object
- Properties:
  ```json
  {
    "invoice_number": "INV-12345",
    "invoice_date": "2023-04-15",
    "name": "ACME Corp",
    "items": [
      {
        "name": "Product A",
        "quantity": 2,
        "unit_price": 10.0,
        "total_price": 20.0
      }
    ],
    "total": 110.0,
    "tax": 10.0
  }
  ```

## Debug Files Generated

The pipeline creates the following debug files in the `debug/` directory:

1. **preprocessed_image.jpg**: The preprocessed image
2. **ocr_results.json**: Raw OCR extraction results
3. **ocr_visual_results.txt**: Human-readable OCR results
4. **entity_results.json**: Entity classification results
5. **layout_results.json**: Layout analysis results
6. **layout_processing_debug.json**: Detailed debug info for layout model
7. **relation_results.json**: Relation extraction results
8. **combined_results.json**: Combined results
9. **final_output.json**: Final formatted output
