
"""
Utility script for creating a sample dataset for training invoice extraction models
"""

import os
import json
import random
import shutil
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime, timedelta
import re
import uuid

# Constants
CANVAS_WIDTH = 2000
CANVAS_HEIGHT = 2800
BASE_FONT_SIZE = 36
HEADER_FONT_SIZE = 48
COMPANY_NAMES = [
    "Acme Corporation", "Globex Industries", "Stark Enterprises", "Wayne Enterprises",
    "Umbrella Corp", "Cyberdyne Systems", "Oscorp Industries", "LexCorp",
    "Massive Dynamic", "Soylent Corp", "Aperture Science", "Tyrell Corporation"
]
CUSTOMER_NAMES = [
    "John Smith", "Jane Doe", "Robert Johnson", "Emily Williams", 
    "Michael Brown", "Sarah Davis", "David Miller", "Jennifer Wilson",
    "ABC Company", "XYZ Ltd", "123 Industries", "Tech Solutions Inc."
]
PRODUCT_NAMES = [
    "Premium Widget", "Standard Gadget", "Deluxe Sprocket", "Basic Connector",
    "Advanced Mechanism", "Ultra Processor", "Essential Component", "Professional Tool",
    "Consulting Services", "Software License", "Maintenance Contract", "Technical Support"
]


def generate_invoice_number():
    """Generate a random invoice number"""
    prefixes = ["INV", "I", "INVOICE", ""]
    separators = ["-", "", "/", "#"]
    
    prefix = random.choice(prefixes)
    separator = random.choice(separators)
    number = random.randint(1000, 9999)
    year = random.randint(2020, 2023)
    
    formats = [
        f"{prefix}{separator}{number}",
        f"{prefix}{separator}{year}{separator}{number}",
        f"{prefix}{separator}{number}{separator}{year}",
        f"{year}{separator}{number}",
        f"{number}"
    ]
    
    return random.choice(formats)


def generate_date():
    """Generate a random invoice date"""
    today = datetime.now()
    days_back = random.randint(1, 365)
    invoice_date = today - timedelta(days=days_back)
    
    formats = [
        invoice_date.strftime("%Y-%m-%d"),
        invoice_date.strftime("%d/%m/%Y"),
        invoice_date.strftime("%m/%d/%Y"),
        invoice_date.strftime("%d-%m-%Y"),
        invoice_date.strftime("%d.%m.%Y"),
        invoice_date.strftime("%B %d, %Y")
    ]
    
    return random.choice(formats)


def generate_items(count=None):
    """Generate random invoice items"""
    if count is None:
        count = random.randint(1, 5)
    
    items = []
    
    for _ in range(count):
        name = random.choice(PRODUCT_NAMES)
        quantity = random.randint(1, 10)
        unit_price = round(random.uniform(10, 500), 2)
        total_price = round(quantity * unit_price, 2)
        
        items.append({
            "name": name,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_price": total_price
        })
    
    return items


def calculate_totals(items):
    """Calculate subtotal, tax, and total"""
    subtotal = sum(item["total_price"] for item in items)
    tax_rate = random.uniform(0.05, 0.25)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    
    return subtotal, tax, total


def create_invoice_image(invoice_data, output_path):
    """Create an invoice image with the given data"""
    # Create canvas
    canvas = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Try to load fonts
    try:
        header_font = ImageFont.truetype("Arial Bold.ttf", HEADER_FONT_SIZE)
        regular_font = ImageFont.truetype("Arial.ttf", BASE_FONT_SIZE)
        bold_font = ImageFont.truetype("Arial Bold.ttf", BASE_FONT_SIZE)
    except IOError:
        # Fallback to default font
        header_font = ImageFont.load_default()
        regular_font = ImageFont.load_default()
        bold_font = ImageFont.load_default()
    
    # Company header
    y_position = 100
    company_name = invoice_data["company_name"]
    draw.text((100, y_position), company_name, fill=(0, 0, 0), font=header_font)
    y_position += 80
    
    # Invoice details
    draw.text((100, y_position), "INVOICE", fill=(0, 0, 0), font=bold_font)
    y_position += 50
    
    # Two column layout
    left_column = 100
    right_column = CANVAS_WIDTH - 400
    
    # Invoice number and date
    draw.text((left_column, y_position), f"Invoice Number:", fill=(0, 0, 0), font=bold_font)
    draw.text((left_column + 300, y_position), invoice_data["invoice_number"], fill=(0, 0, 0), font=regular_font)
    
    draw.text((right_column, y_position), f"Date:", fill=(0, 0, 0), font=bold_font)
    draw.text((right_column + 150, y_position), invoice_data["invoice_date"], fill=(0, 0, 0), font=regular_font)
    y_position += 80
    
    # Customer section
    draw.text((left_column, y_position), "Bill To:", fill=(0, 0, 0), font=bold_font)
    y_position += 50
    draw.text((left_column, y_position), invoice_data["customer_name"], fill=(0, 0, 0), font=regular_font)
    y_position += 100
    
    # Items table
    table_headers = ["Item", "Quantity", "Unit Price", "Total"]
    col_widths = [800, 200, 300, 300]
    col_positions = [left_column]
    for width in col_widths[:-1]:
        col_positions.append(col_positions[-1] + width)
    
    # Table header
    for i, header in enumerate(table_headers):
        draw.text((col_positions[i], y_position), header, fill=(0, 0, 0), font=bold_font)
    
    y_position += 50
    
    # Table divider
    draw.line([(left_column, y_position), (col_positions[-1] + col_widths[-1], y_position)], 
              fill=(200, 200, 200), width=2)
    y_position += 20
    
    # Table rows
    for item in invoice_data["items"]:
        draw.text((col_positions[0], y_position), item["name"], fill=(0, 0, 0), font=regular_font)
        draw.text((col_positions[1], y_position), str(item["quantity"]), fill=(0, 0, 0), font=regular_font)
        draw.text((col_positions[2], y_position), f"${item['unit_price']:.2f}", fill=(0, 0, 0), font=regular_font)
        draw.text((col_positions[3], y_position), f"${item['total_price']:.2f}", fill=(0, 0, 0), font=regular_font)
        y_position += 60
    
    # Table divider
    draw.line([(left_column, y_position), (col_positions[-1] + col_widths[-1], y_position)], 
              fill=(200, 200, 200), width=2)
    y_position += 40
    
    # Summary
    summary_x = col_positions[2]
    
    draw.text((summary_x, y_position), "Subtotal:", fill=(0, 0, 0), font=bold_font)
    draw.text((col_positions[3], y_position), f"${invoice_data['subtotal']:.2f}", fill=(0, 0, 0), font=regular_font)
    y_position += 50
    
    tax_label = f"Tax ({invoice_data['tax_rate']*100:.0f}%):"
    draw.text((summary_x, y_position), tax_label, fill=(0, 0, 0), font=bold_font)
    draw.text((col_positions[3], y_position), f"${invoice_data['tax']:.2f}", fill=(0, 0, 0), font=regular_font)
    y_position += 50
    
    # Total
    draw.text((summary_x, y_position), "Total:", fill=(0, 0, 0), font=bold_font)
    draw.text((col_positions[3], y_position), f"${invoice_data['total']:.2f}", fill=(0, 0, 0), font=bold_font)
    
    # Save image
    canvas.save(output_path)
    print(f"Created invoice image: {output_path}")
    
    return canvas


def extract_text_blocks(canvas, invoice_data):
    """
    Extract text blocks from the invoice canvas
    
    This simulates OCR by using the known positions and text
    """
    text_blocks = []
    block_id = 0
    
    # Company header
    y_position = 100
    text_blocks.append({
        "id": block_id,
        "text": invoice_data["company_name"],
        "position": {
            "x_min": 100,
            "y_min": y_position,
            "x_max": 800,
            "y_max": y_position + 60
        },
        "entity_type": "company_name"
    })
    block_id += 1
    y_position += 80
    
    # Invoice label
    text_blocks.append({
        "id": block_id,
        "text": "INVOICE",
        "position": {
            "x_min": 100,
            "y_min": y_position,
            "x_max": 300,
            "y_max": y_position + 50
        },
        "entity_type": "label"
    })
    block_id += 1
    y_position += 50
    
    # Left column
    left_column = 100
    right_column = CANVAS_WIDTH - 400
    
    # Invoice number label
    text_blocks.append({
        "id": block_id,
        "text": "Invoice Number:",
        "position": {
            "x_min": left_column,
            "y_min": y_position,
            "x_max": left_column + 300,
            "y_max": y_position + 40
        },
        "entity_type": "label"
    })
    block_id += 1
    
    # Invoice number value
    text_blocks.append({
        "id": block_id,
        "text": invoice_data["invoice_number"],
        "position": {
            "x_min": left_column + 300,
            "y_min": y_position,
            "x_max": left_column + 600,
            "y_max": y_position + 40
        },
        "entity_type": "invoice_number"
    })
    block_id += 1
    
    # Date label
    text_blocks.append({
        "id": block_id,
        "text": "Date:",
        "position": {
            "x_min": right_column,
            "y_min": y_position,
            "x_max": right_column + 150,
            "y_max": y_position + 40
        },
        "entity_type": "label"
    })
    block_id += 1
    
    # Date value
    text_blocks.append({
        "id": block_id,
        "text": invoice_data["invoice_date"],
        "position": {
            "x_min": right_column + 150,
            "y_min": y_position,
            "x_max": right_column + 450,
            "y_max": y_position + 40
        },
        "entity_type": "invoice_date"
    })
    block_id += 1
    y_position += 80
    
    # Bill To label
    text_blocks.append({
        "id": block_id,
        "text": "Bill To:",
        "position": {
            "x_min": left_column,
            "y_min": y_position,
            "x_max": left_column + 200,
            "y_max": y_position + 40
        },
        "entity_type": "label"
    })
    block_id += 1
    y_position += 50
    
    # Customer name
    text_blocks.append({
        "id": block_id,
        "text": invoice_data["customer_name"],
        "position": {
            "x_min": left_column,
            "y_min": y_position,
            "x_max": left_column + 500,
            "y_max": y_position + 40
        },
        "entity_type": "customer_name"
    })
    block_id += 1
    y_position += 100
    
    # Table headers
    table_headers = ["Item", "Quantity", "Unit Price", "Total"]
    col_widths = [800, 200, 300, 300]
    col_positions = [left_column]
    for width in col_widths[:-1]:
        col_positions.append(col_positions[-1] + width)
    
    for i, header in enumerate(table_headers):
        text_blocks.append({
            "id": block_id,
            "text": header,
            "position": {
                "x_min": col_positions[i],
                "y_min": y_position,
                "x_max": col_positions[i] + col_widths[i],
                "y_max": y_position + 40
            },
            "entity_type": "table_header"
        })
        block_id += 1
    
    y_position += 70
    
    # Table rows
    item_blocks = []
    for item in invoice_data["items"]:
        # Item name
        item_name_block = {
            "id": block_id,
            "text": item["name"],
            "position": {
                "x_min": col_positions[0],
                "y_min": y_position,
                "x_max": col_positions[0] + col_widths[0],
                "y_max": y_position + 40
            },
            "entity_type": "item_name"
        }
        text_blocks.append(item_name_block)
        item_blocks.append(item_name_block)
        block_id += 1
        
        # Quantity
        quantity_block = {
            "id": block_id,
            "text": str(item["quantity"]),
            "position": {
                "x_min": col_positions[1],
                "y_min": y_position,
                "x_max": col_positions[1] + col_widths[1],
                "y_max": y_position + 40
            },
            "entity_type": "item_quantity"
        }
        text_blocks.append(quantity_block)
        block_id += 1
        
        # Unit price
        unit_price_block = {
            "id": block_id,
            "text": f"${item['unit_price']:.2f}",
            "position": {
                "x_min": col_positions[2],
                "y_min": y_position,
                "x_max": col_positions[2] + col_widths[2],
                "y_max": y_position + 40
            },
            "entity_type": "item_price"
        }
        text_blocks.append(unit_price_block)
        block_id += 1
        
        # Total price
        total_price_block = {
            "id": block_id,
            "text": f"${item['total_price']:.2f}",
            "position": {
                "x_min": col_positions[3],
                "y_min": y_position,
                "x_max": col_positions[3] + col_widths[3],
                "y_max": y_position + 40
            },
            "entity_type": "item_total"
        }
        text_blocks.append(total_price_block)
        block_id += 1
        
        y_position += 60
    
    y_position += 40
    
    # Summary
    summary_x = col_positions[2]
    
    # Subtotal label
    text_blocks.append({
        "id": block_id,
        "text": "Subtotal:",
        "position": {
            "x_min": summary_x,
            "y_min": y_position,
            "x_max": summary_x + 200,
            "y_max": y_position + 40
        },
        "entity_type": "label"
    })
    block_id += 1
    
    # Subtotal value
    text_blocks.append({
        "id": block_id,
        "text": f"${invoice_data['subtotal']:.2f}",
        "position": {
            "x_min": col_positions[3],
            "y_min": y_position,
            "x_max": col_positions[3] + col_widths[3],
            "y_max": y_position + 40
        },
        "entity_type": "subtotal"
    })
    block_id += 1
    y_position += 50
    
    # Tax label
    tax_label = f"Tax ({invoice_data['tax_rate']*100:.0f}%):"
    text_blocks.append({
        "id": block_id,
        "text": tax_label,
        "position": {
            "x_min": summary_x,
            "y_min": y_position,
            "x_max": summary_x + 200,
            "y_max": y_position + 40
        },
        "entity_type": "label"
    })
    block_id += 1
    
    # Tax value
    text_blocks.append({
        "id": block_id,
        "text": f"${invoice_data['tax']:.2f}",
        "position": {
            "x_min": col_positions[3],
            "y_min": y_position,
            "x_max": col_positions[3] + col_widths[3],
            "y_max": y_position + 40
        },
        "entity_type": "tax"
    })
    block_id += 1
    y_position += 50
    
    # Total label
    text_blocks.append({
        "id": block_id,
        "text": "Total:",
        "position": {
            "x_min": summary_x,
            "y_min": y_position,
            "x_max": summary_x + 200,
            "y_max": y_position + 40
        },
        "entity_type": "label"
    })
    block_id += 1
    
    # Total value
    text_blocks.append({
        "id": block_id,
        "text": f"${invoice_data['total']:.2f}",
        "position": {
            "x_min": col_positions[3],
            "y_min": y_position,
            "x_max": col_positions[3] + col_widths[3],
            "y_max": y_position + 40
        },
        "entity_type": "total"
    })
    block_id += 1
    
    return text_blocks, item_blocks


def create_relations(text_blocks, item_blocks):
    """Create relations between text blocks"""
    relations = []
    relation_id = 0
    
    # Find item blocks
    item_name_blocks = [block for block in text_blocks if block["entity_type"] == "item_name"]
    
    # Match item name blocks with quantity, price, and total blocks
    for item_block in item_name_blocks:
        item_y = (item_block["position"]["y_min"] + item_block["position"]["y_max"]) / 2
        
        # Find quantity, price, and total blocks for this item based on y position
        for block in text_blocks:
            block_y = (block["position"]["y_min"] + block["position"]["y_max"]) / 2
            
            # If blocks are in the same row (approximately)
            if abs(block_y - item_y) < 30:
                if block["entity_type"] == "item_quantity":
                    relations.append({
                        "id": relation_id,
                        "source_id": item_block["id"],
                        "target_id": block["id"],
                        "relation_type": "item_quantity"
                    })
                    relation_id += 1
                    
                elif block["entity_type"] == "item_price":
                    relations.append({
                        "id": relation_id,
                        "source_id": item_block["id"],
                        "target_id": block["id"],
                        "relation_type": "item_price"
                    })
                    relation_id += 1
                    
                elif block["entity_type"] == "item_total":
                    relations.append({
                        "id": relation_id,
                        "source_id": item_block["id"],
                        "target_id": block["id"],
                        "relation_type": "item_total"
                    })
                    relation_id += 1
    
    return relations


def create_annotation(image_path, invoice_data, text_blocks, relations):
    """Create annotation JSON file"""
    annotation = {
        "image_path": image_path,
        "width": CANVAS_WIDTH,
        "height": CANVAS_HEIGHT,
        "text_blocks": text_blocks,
        "relations": relations,
        "metadata": {
            "company_name": invoice_data["company_name"],
            "invoice_number": invoice_data["invoice_number"],
            "invoice_date": invoice_data["invoice_date"],
            "customer_name": invoice_data["customer_name"],
            "items": invoice_data["items"],
            "subtotal": invoice_data["subtotal"],
            "tax": invoice_data["tax"],
            "tax_rate": invoice_data["tax_rate"],
            "total": invoice_data["total"]
        }
    }
    
    return annotation


def create_sample_invoice(output_dir, split):
    """Create a sample invoice with image and annotation"""
    # Create random invoice data
    invoice_data = {
        "company_name": random.choice(COMPANY_NAMES),
        "invoice_number": generate_invoice_number(),
        "invoice_date": generate_date(),
        "customer_name": random.choice(CUSTOMER_NAMES),
        "items": generate_items()
    }
    
    # Calculate totals
    invoice_data["subtotal"], invoice_data["tax"], invoice_data["total"] = calculate_totals(invoice_data["items"])
    invoice_data["tax_rate"] = invoice_data["tax"] / invoice_data["subtotal"] if invoice_data["subtotal"] > 0 else 0.1
    
    # Create directories if they don't exist
    images_dir = os.path.join(output_dir, "images", split)
    annotations_dir = os.path.join(output_dir, "annotations", split)
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Generate unique ID
    invoice_id = str(uuid.uuid4())[:8]
    
    # Create image
    image_filename = f"invoice_{invoice_id}.png"
    image_path = os.path.join(images_dir, image_filename)
    relative_image_path = os.path.join("images", split, image_filename)
    
    canvas = create_invoice_image(invoice_data, image_path)
    
    # Extract text blocks and create relations
    text_blocks, item_blocks = extract_text_blocks(canvas, invoice_data)
    relations = create_relations(text_blocks, item_blocks)
    
    # Create annotation
    annotation = create_annotation(relative_image_path, invoice_data, text_blocks, relations)
    
    # Save annotation
    annotation_path = os.path.join(annotations_dir, f"invoice_{invoice_id}.json")
    with open(annotation_path, "w") as f:
        json.dump(annotation, f, indent=2)
    
    print(f"Created annotation: {annotation_path}")
    
    return invoice_id


def create_dataset(output_dir, num_train=20, num_val=5, num_test=5):
    """Create a full dataset with train, val, and test splits"""
    # Create base directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create samples for each split
    print(f"Creating {num_train} training samples...")
    train_ids = [create_sample_invoice(output_dir, "train") for _ in range(num_train)]
    
    print(f"Creating {num_val} validation samples...")
    val_ids = [create_sample_invoice(output_dir, "val") for _ in range(num_val)]
    
    print(f"Creating {num_test} test samples...")
    test_ids = [create_sample_invoice(output_dir, "test") for _ in range(num_test)]
    
    # Create dataset metadata
    metadata = {
        "dataset_name": "Invoice Extraction Sample Dataset",
        "created_at": datetime.now().isoformat(),
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "entity_types": [
            "invoice_number",
            "invoice_date",
            "customer_name",
            "item_name",
            "item_quantity",
            "item_price",
            "subtotal",
            "total",
            "tax",
            "company_name",
            "label",
            "table_header"
        ],
        "relation_types": [
            "item_quantity",
            "item_price",
            "item_total"
        ]
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset created at {output_dir}")
    print(f"Total samples: {num_train + num_val + num_test}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a sample dataset for invoice extraction")
    parser.add_argument(
        "--output_dir", type=str, default="dataset",
        help="Directory to store the dataset"
    )
    parser.add_argument(
        "--train", type=int, default=20,
        help="Number of training samples"
    )
    parser.add_argument(
        "--val", type=int, default=5,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--test", type=int, default=5,
        help="Number of test samples"
    )
    
    args = parser.parse_args()
    
    create_dataset(args.output_dir, args.train, args.val, args.test)
