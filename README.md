
# Invoice Intellect Extractor

An AI-powered application for extracting structured data from invoice images using custom machine learning models and OCR.

## Features

- Flask API for accepting invoice images (upload or URL)
- Custom OCR processing with doctr for text extraction
- Image preprocessing for enhanced quality and OCR accuracy
- Custom machine learning models for entity classification
- Relation extraction for identifying invoice items and pricing
- React frontend for uploading invoices and viewing extracted data
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

### Dataset Preparation

#### Directory Structure

Create a dataset directory with the following structure:

```
dataset/
├── images/             # Raw invoice images
│   ├── train/          # Training set images
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── val/            # Validation set images
│   │   ├── img101.jpg
│   │   ├── img102.jpg
│   │   └── ...
│   └── test/           # Test set images
│       ├── img201.jpg
│       ├── img202.jpg
│       └── ...
└── annotations/        # Annotations for each image
    ├── train/
    │   ├── img1.json
    │   ├── img2.json
    │   └── ...
    ├── val/
    │   ├── img101.json
    │   ├── img102.json
    │   └── ...
    └── test/
        ├── img201.json
        ├── img202.json
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

### Sample Dataset for Testing

Here's a simple example of how to create a small test dataset:

1. Create the directory structure:
```bash
mkdir -p dataset/images/{train,val,test}
mkdir -p dataset/annotations/{train,val,test}
```

2. Place invoice images in the appropriate folders (train, val, test).

3. Create annotation files for each image following the format above.

#### Example Annotation File (dataset/annotations/train/invoice1.json):

```json
{
  "image_path": "images/train/invoice1.jpg",
  "width": 1000,
  "height": 1414,
  "text_blocks": [
    {
      "id": 1,
      "text": "INVOICE #INV-2023-001",
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
    {
      "id": 3,
      "text": "John Doe",
      "position": {
        "x_min": 100,
        "y_min": 120,
        "x_max": 250,
        "y_max": 150
      },
      "entity_type": "customer_name"
    },
    {
      "id": 4,
      "text": "Product A",
      "position": {
        "x_min": 100,
        "y_min": 200,
        "x_max": 300,
        "y_max": 230
      },
      "entity_type": "item_name"
    },
    {
      "id": 5,
      "text": "2",
      "position": {
        "x_min": 350,
        "y_min": 200,
        "x_max": 380,
        "y_max": 230
      },
      "entity_type": "item_quantity"
    },
    {
      "id": 6,
      "text": "$10.99",
      "position": {
        "x_min": 400,
        "y_min": 200,
        "x_max": 450,
        "y_max": 230
      },
      "entity_type": "item_price"
    }
  ],
  "relations": [
    {
      "id": 1,
      "source_id": 4,
      "target_id": 5,
      "relation_type": "item_quantity"
    },
    {
      "id": 2,
      "source_id": 4,
      "target_id": 6,
      "relation_type": "item_price"
    }
  ]
}
```

### Creating a Training Dataset from Scratch

For real-world applications, you would typically follow these steps:

1. **Collect Invoice Images**: 
   - Gather a diverse set of invoice images in different formats and layouts
   - Include varying fields, vendors, and styles for robust training

2. **Annotation Process**:
   - Use annotation tools like LabelImg, CVAT, or custom annotation scripts
   - For each image:
     a) Identify text blocks using OCR
     b) Manually label each text block with its entity type
     c) Define relations between text blocks (e.g., which price belongs to which item)
   - Save annotations in the specified JSON format

3. **Dataset Splitting**:
   - Split your dataset into training (70%), validation (15%), and test (15%) sets
   - Ensure each set contains a diverse range of invoice types

4. **Data Augmentation** (optional):
   - Generate additional training samples by applying transformations
   - Techniques include rotation, noise addition, contrast changes, etc.

### Training the Models

Once your dataset is prepared, you can train the models using the training script:

```bash
python train.py --data_dir=dataset --output_dir=models --epochs=20 --batch_size=32 --lr=0.001 --model=both
```

Options:
- `--data_dir`: Directory containing the dataset (with the structure as described above)
- `--output_dir`: Directory to save the trained models
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--model`: Which model to train (entity, relation, or both)

### Using Trained Models

After training, the models will be saved in the specified output directory. The pipeline will automatically use them as long as they're located in the "models" directory.

## Deployment

### Docker Deployment

Build and run with Docker:

```bash
# Build the Docker image
docker build -t invoice-extractor .

# Run the container
docker run -p 5000:5000 invoice-extractor
```

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
