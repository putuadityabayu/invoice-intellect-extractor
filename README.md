
# Invoice Intellect Extractor

An AI-powered backend for extracting structured data from invoice images using custom machine learning models and OCR.

## Features

- Flask API for accepting invoice images (upload or URL)
- Custom OCR processing with doctr for text extraction
- Image preprocessing for enhanced quality and OCR accuracy
- Custom machine learning models for entity classification
- Relation extraction for identifying invoice items and pricing
- Structured JSON output with key invoice information

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Development with DevContainer

This project includes DevContainer configuration for Visual Studio Code, providing a consistent development environment.

1. Clone the repository:
```bash
git clone https://github.com/yourusername/invoice-intellect-extractor.git
cd invoice-intellect-extractor
```

2. Open in VS Code with DevContainer support:
```bash
code .
```

3. When prompted, click "Reopen in Container" or use the command palette (F1) and select "Remote-Containers: Reopen in Container".

4. VS Code will build the container and set up the environment automatically.

### Running the Server

Inside the DevContainer:

```bash
python app.py
```

The server will start at http://localhost:5000.

## API Usage

### Extract Invoice Data

**Endpoint**: `/extract`

**Method**: POST

**Parameters**:
- `file`: The invoice image file (multipart/form-data)
- OR `image_url`: URL to the invoice image

**Example using curl**:

```bash
# Using a local file
curl -X POST -F "file=@/path/to/invoice.jpg" http://localhost:5000/extract

# Using an image URL
curl -X POST -F "image_url=https://example.com/invoice.jpg" http://localhost:5000/extract
```

**Response**:

```json
{
  "invoice_number": "INV-2023-001",
  "invoice_date": "2023-10-15",
  "name": "John Doe",
  "items": [
    {
      "name": "Product A",
      "quantity": 2,
      "unit_price": 10.99,
      "total_price": 21.98
    },
    {
      "name": "Service B",
      "quantity": 1,
      "unit_price": 50.0,
      "total_price": 50.0
    }
  ],
  "subtotal": 71.98,
  "extra_price": [
    {
      "tax": 7.2
    }
  ],
  "total": 79.18
}
```

## Model Training

The system uses custom machine learning models for entity classification and relation extraction. These models need to be trained before use.

### Training Data

In a production implementation, you would need to create a dataset of invoice images and annotations. This dataset should include:

1. Invoice images in various formats
2. Text annotations with bounding boxes
3. Entity type labels for each text block
4. Relation annotations between entities

### Training the Models

1. Prepare the training data in the appropriate format

2. Run the training script:

```bash
python train.py --output_dir=models --epochs=20 --batch_size=32 --lr=0.001 --model=both
```

Options:
- `--output_dir`: Directory to save the trained models
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--model`: Which model to train (entity, relation, or both)

### Using Trained Models

After training, the models will be saved in the specified output directory. To use them, update the model paths in the code:

```python
# In src/models/entity_classifier.py
self.model_path = "path/to/trained/entity_model.pth"

# In src/models/relation_extractor.py
self.model_path = "path/to/trained/relation_model.pth"
```

## Custom ML Model Architecture

### Entity Classification Model

The entity classification model identifies text blocks as specific invoice elements like invoice numbers, dates, customer names, etc.

Architecture:
- Input: Text feature vector (768 dimensions)
- Hidden Layer 1: 256 neurons with ReLU activation
- Dropout Layer: 0.3 dropout rate
- Hidden Layer 2: 128 neurons with ReLU activation
- Dropout Layer: 0.3 dropout rate
- Output Layer: 8 neurons (one per entity type)

Entity Types:
1. Invoice Number
2. Invoice Date
3. Customer Name
4. Item Name
5. Item Quantity
6. Item Price
7. Subtotal
8. Total

### Relation Extraction Model

The relation extraction model identifies relationships between entities, especially for invoice items.

Architecture:
- Input: Pair of text feature vectors (1536 dimensions)
- Hidden Layer 1: 256 neurons with ReLU activation
- Dropout Layer: 0.3 dropout rate
- Hidden Layer 2: 128 neurons with ReLU activation
- Dropout Layer: 0.3 dropout rate
- Output Layer: 4 neurons (one per relation type)

Relation Types:
1. None
2. Item-Quantity
3. Item-Price
4. Item-Total

## Project Structure

```
invoice-intellect-extractor/
├── .devcontainer/                 # DevContainer configuration
│   ├── devcontainer.json
│   └── Dockerfile
├── src/
│   ├── preprocessing/             # Image preprocessing
│   │   ├── __init__.py
│   │   └── image_preprocessor.py
│   ├── ocr/                       # OCR processing
│   │   ├── __init__.py
│   │   └── processor.py
│   ├── models/                    # Machine learning models
│   │   ├── __init__.py
│   │   ├── entity_classifier.py
│   │   └── relation_extractor.py
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   └── data_formatter.py
│   ├── __init__.py
│   └── pipeline.py                # Main processing pipeline
├── app.py                         # Flask application
├── train.py                       # Model training script
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation
```

## Future Improvements

1. Implement a real feature extraction method using transformers like BERT
2. Add a proper dataset class for loading real training data
3. Implement more sophisticated table detection for invoice items
4. Add validation and test sets for model evaluation
5. Improve preprocessing for better OCR accuracy
6. Add support for multiple languages
7. Implement model fine-tuning for specific invoice formats

## License

This project is licensed under the MIT License - see the LICENSE file for details.
