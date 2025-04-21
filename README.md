
# Invoice Intellect Extractor

An AI-powered application for extracting structured data from invoice images using custom machine learning models and OCR.

## Features

- Flask API for accepting invoice images (upload or URL)
- Custom OCR processing with doctr for text extraction
- Image preprocessing for enhanced quality and OCR accuracy
- Multiple ML approaches for entity classification:
  - Neural network entity classification
  - Named Entity Recognition (NER) with spaCy
  - LayoutLM-inspired spatial-aware document understanding
- Relation extraction for identifying invoice items and pricing
- React frontend for uploading invoices and viewing extracted data
- Structured JSON output with key invoice information

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git
- Python 3.8+
- PyTorch
- spaCy

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

### Running the Application

#### Development Server

Inside the DevContainer:

```bash
# Start the backend
python app.py

# Start the frontend (in a separate terminal)
npm start
```

The backend server will start at http://localhost:5000.
The frontend will be available at http://localhost:3000.

#### Production Server with Gunicorn

For production deployments, use Gunicorn:

```bash
# Basic usage
gunicorn app:app -w 4 -b 0.0.0.0:5000

# With more workers and timeout configuration
gunicorn app:app -w 4 -b 0.0.0.0:5000 --timeout 120 --access-logfile - --error-logfile -

# As a background service
gunicorn app:app -w 4 -b 0.0.0.0:5000 --daemon
```

Options explained:
- `-w 4`: Spawns 4 worker processes (adjust based on CPU cores)
- `-b 0.0.0.0:5000`: Binds to all interfaces on port 5000
- `--timeout 120`: Sets worker timeout to 120 seconds for processing large images
- `--daemon`: Runs in the background

### Production Deployment with Gunicorn and Nginx

For production, it's recommended to use Gunicorn behind a reverse proxy like Nginx:

1. Create a systemd service file at `/etc/systemd/system/invoice-extractor.service`:

```
[Unit]
Description=Invoice Extractor Gunicorn Service
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/path/to/invoice-intellect-extractor
ExecStart=/path/to/venv/bin/gunicorn app:app -w 4 -b 127.0.0.1:5000 --timeout 120
Restart=always

[Install]
WantedBy=multi-user.target
```

2. Configure Nginx as a reverse proxy:

```
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 16M;
    }
}
```

3. Enable and start the service:

```bash
sudo systemctl enable invoice-extractor
sudo systemctl start invoice-extractor
sudo systemctl restart nginx
```

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

The system uses multiple machine learning approaches for entity classification and relation extraction:

1. Neural network models for entity classification and relation extraction
2. spaCy NER model for named entity recognition
3. LayoutLM-inspired model for layout-aware document understanding

### Creating a Training Dataset

#### Using the Sample Dataset Generator

The easiest way to start is by using the provided sample dataset generator:

```bash
# Generate a sample dataset with 20 training, 5 validation, and 5 test invoices
python utils/create_sample_dataset.py --output_dir dataset --train 20 --val 5 --test 5
```

This will create a complete dataset with images and annotations in the proper format.

#### Directory Structure

If you want to create your own dataset, use the following directory structure:

```
dataset/
├── images/             # Raw invoice images
│   ├── train/          # Training set images
│   │   ├── img1.jpg
│   │   └── ...
│   ├── val/            # Validation set images
│   │   ├── img101.jpg
│   │   └── ...
│   └── test/           # Test set images
│       ├── img201.jpg
│       └── ...
└── annotations/        # Annotations for each image
    ├── train/
    │   ├── img1.json
    │   └── ...
    ├── val/
    │   ├── img101.json
    │   └── ...
    └── test/
        ├── img201.json
        └── ...
```

#### Annotation Format

Each annotation file should be a JSON file with the following structure:

```json
{
  "image_path": "images/train/img1.jpg",
  "width": 1000,
  "height": 1414,
  "text_blocks": [
    {
      "id": 1,
      "text": "INVOICE #1234",
      "position": {
        "x_min": 100,
        "y_min": 50,
        "x_max": 300,
        "y_max": 80
      },
      "entity_type": "invoice_number"
    },
    {
      "id": 2,
      "text": "Date: 2023-10-15",
      "position": {
        "x_min": 500,
        "y_min": 50,
        "x_max": 650,
        "y_max": 80
      },
      "entity_type": "invoice_date"
    },
    // More text blocks with entity annotations
  ],
  "relations": [
    {
      "id": 1,
      "source_id": 5,
      "target_id": 6,
      "relation_type": "item_quantity"
    },
    {
      "id": 2,
      "source_id": 5,
      "target_id": 7,
      "relation_type": "item_price"
    },
    // More relation annotations
  ]
}
```

Where:
- `entity_type` can be: "invoice_number", "invoice_date", "customer_name", "item_name", "item_quantity", "item_price", "subtotal", "total"
- `relation_type` can be: "none", "item_quantity", "item_price", "item_total"

### Training the Models

Once your dataset is prepared, you can train all the models using the training script:

```bash
python train.py --data_dir=dataset --output_dir=models --epochs=20 --batch_size=32 --lr=0.001 --model=all
```

Options:
- `--data_dir`: Directory containing the dataset (with the structure as described above)
- `--output_dir`: Directory to save the trained models
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--model`: Which model to train:
  - `entity`: Only train the entity classifier
  - `relation`: Only train the relation extractor
  - `layout`: Only train the layout-aware model
  - `spacy`: Only prepare and train the spaCy NER model
  - `all`: Train all models (default)

### spaCy NER Training

When training the spaCy NER model, the script will:
1. Generate the necessary training data files
2. Create a configuration file
3. Generate a training script

You may need to manually run the training script for the spaCy model:

```bash
cd models/spacy_ner
bash train.sh
```

This will train the spaCy NER model using the generated training files.

### Using Trained Models

After training, the models will be saved in the specified output directory. The pipeline will automatically use them as long as they're located in the "models" directory with the following structure:

```
models/
├── entity_model.pth      # Entity classifier model
├── relation_model.pth    # Relation extractor model
├── layout_model.pth      # Layout-aware model
└── spacy_ner/            # spaCy NER model directory
    ├── model-best/       # Best model from spaCy training
    └── ...
```

## Machine Learning Methods

For details on the machine learning approaches used in this project, see the [ML Methods documentation](docs/ml-methods.md).

## Project Structure

```
invoice-intellect-extractor/
├── .devcontainer/                 # DevContainer configuration
│   ├── devcontainer.json
│   └── Dockerfile
├── docs/                          # Documentation
│   └── ml-methods.md              # ML methods documentation
├── src/
│   ├── preprocessing/             # Image preprocessing
│   │   ├── __init__.py
│   │   └── image_preprocessor.py
│   ├── ocr/                       # OCR processing
│   │   ├── __init__.py
│   │   └── processor.py
│   ├── models/                    # Machine learning models
│   │   ├── __init__.py
│   │   ├── entity_classifier.py   # Entity classification with spaCy
│   │   ├── relation_extractor.py  # Relation extraction
│   │   └── layout_model.py        # LayoutLM-inspired model
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   ├── dataset.py             # Dataset classes for training
│   │   └── data_formatter.py
│   ├── components/                # React components
│   │   ├── FileUpload.tsx
│   │   └── InvoicePreview.tsx
│   ├── pages/                     # React pages
│   │   ├── Index.tsx
│   │   └── NotFound.tsx
│   ├── types/                     # TypeScript types
│   │   └── invoice.ts
│   ├── __init__.py
│   └── pipeline.py                # Main processing pipeline
├── utils/
│   └── create_sample_dataset.py   # Sample dataset generator
├── models/                        # Trained models directory
├── app.py                         # Flask application
├── train.py                       # Model training script
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation
```

## Future Improvements

1. Implement a full LayoutLM or LayoutLMv2 model for better spatial-aware extraction
2. Improve the spaCy NER model with custom entity rules
3. Add a document classifier to handle different invoice formats
4. Implement more sophisticated table detection for invoice items
5. Add support for multiple languages
6. Integrate with large language models for zero-shot learning

## License

This project is licensed under the MIT License - see the LICENSE file for details.
