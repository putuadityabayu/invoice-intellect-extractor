
"""
Script to create a sample dataset for training the invoice extraction models
"""

import os
import json
import shutil
import random
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta

# Configuration
OUTPUT_DIR = "dataset"
NUM_TRAIN_SAMPLES = 10
NUM_VAL_SAMPLES = 3
NUM_TEST_SAMPLES = 3
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1400

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "test"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "annotations", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "annotations", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "annotations", "test"), exist_ok=True)

# Sample data
COMPANY_NAMES = ["ABC Corp", "XYZ Inc", "Tech Solutions", "Global Services", "Modern Shop"]
CUSTOMER_NAMES = ["John Doe", "Jane Smith", "Robert Johnson", "Sarah Williams", "Michael Brown"]
PRODUCTS = [
    {"name": "Laptop", "price": 1299.99},
    {"name": "Smartphone", "price": 799.99},
    {"name": "Tablet", "price": 349.99},
    {"name": "Monitor", "price": 249.99},
    {"name": "Keyboard", "price": 59.99},
    {"name": "Mouse", "price": 29.99},
    {"name": "Headphones", "price": 129.99},
    {"name": "Printer", "price": 199.99},
]

def create_invoice_image(image_path, invoice_number, invoice_date, customer_name, items):
    """Create a simple invoice image with the given data"""
    # Create a blank white image
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to load a font, fall back to default if not available
        font_large = ImageFont.truetype("arial.ttf", 36)
        font_medium = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        # Use default font if arial.ttf is not available
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw company name
    company_name = random.choice(COMPANY_NAMES)
    draw.text((100, 50), company_name, font=font_large, fill="black")
    
    # Draw invoice title and number
    draw.text((100, 120), "INVOICE", font=font_medium, fill="black")
    inv_num_text = f"Invoice #: {invoice_number}"
    draw.text((100, 160), inv_num_text, font=font_small, fill="black")
    
    # Draw date
    date_text = f"Date: {invoice_date}"
    draw.text((100, 190), date_text, font=font_small, fill="black")
    
    # Draw customer info
    draw.text((100, 240), "Bill To:", font=font_small, fill="black")
    draw.text((100, 270), customer_name, font=font_small, fill="black")
    
    # Draw items header
    draw.text((100, 350), "Description", font=font_small, fill="black")
    draw.text((500, 350), "Quantity", font=font_small, fill="black")
    draw.text((600, 350), "Unit Price", font=font_small, fill="black")
    draw.text((750, 350), "Total", font=font_small, fill="black")
    
    # Draw separator line
    draw.line([(100, 380), (900, 380)], fill="black", width=2)
    
    # Draw items
    y_position = 400
    subtotal = 0
    
    for item in items:
        name = item["name"]
        quantity = item["quantity"]
        unit_price = item["price"]
        total_price = quantity * unit_price
        subtotal += total_price
        
        draw.text((100, y_position), name, font=font_small, fill="black")
        draw.text((500, y_position), str(quantity), font=font_small, fill="black")
        draw.text((600, y_position), f"${unit_price:.2f}", font=font_small, fill="black")
        draw.text((750, y_position), f"${total_price:.2f}", font=font_small, fill="black")
        
        y_position += 40
    
    # Draw separator line
    draw.line([(600, y_position + 20), (900, y_position + 20)], fill="black", width=1)
    
    # Draw subtotal
    draw.text((650, y_position + 40), "Subtotal:", font=font_small, fill="black")
    draw.text((750, y_position + 40), f"${subtotal:.2f}", font=font_small, fill="black")
    
    # Calculate tax (10%)
    tax = subtotal * 0.1
    draw.text((650, y_position + 70), "Tax (10%):", font=font_small, fill="black")
    draw.text((750, y_position + 70), f"${tax:.2f}", font=font_small, fill="black")
    
    # Calculate total
    total = subtotal + tax
    draw.text((650, y_position + 110), "Total:", font=font_medium, fill="black")
    draw.text((750, y_position + 110), f"${total:.2f}", font=font_medium, fill="black")
    
    # Save the image
    image.save(image_path)
    
    return {
        "company_name": company_name,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "customer_name": customer_name,
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "positions": {
            "invoice_number": {"x_min": 100, "y_min": 160, "x_max": 300, "y_max": 185},
            "invoice_date": {"x_min": 100, "y_min": 190, "x_max": 300, "y_max": 215},
            "customer_name": {"x_min": 100, "y_min": 270, "x_max": 400, "y_max": 295},
            "items_start_y": 400,
            "item_height": 40,
            "subtotal": {"x_min": 750, "y_min": y_position + 40, "x_max": 850, "y_max": y_position + 65},
            "tax": {"x_min": 750, "y_min": y_position + 70, "x_max": 850, "y_max": y_position + 95},
            "total": {"x_min": 750, "y_min": y_position + 110, "x_max": 850, "y_max": y_position + 135}
        }
    }

def create_annotation_file(image_data, relative_image_path, output_path):
    """Create an annotation file in the required format"""
    
    # Initialize lists for text blocks and relations
    text_blocks = []
    relations = []
    block_id = 1
    
    # Add invoice number
    text_blocks.append({
        "id": block_id,
        "text": f"Invoice #: {image_data['invoice_number']}",
        "position": image_data["positions"]["invoice_number"],
        "entity_type": "invoice_number"
    })
    block_id += 1
    
    # Add invoice date
    text_blocks.append({
        "id": block_id,
        "text": f"Date: {image_data['invoice_date']}",
        "position": image_data["positions"]["invoice_date"],
        "entity_type": "invoice_date"
    })
    block_id += 1
    
    # Add customer name
    text_blocks.append({
        "id": block_id,
        "text": image_data["customer_name"],
        "position": image_data["positions"]["customer_name"],
        "entity_type": "customer_name"
    })
    block_id += 1
    
    # Add items
    for i, item in enumerate(image_data["items"]):
        item_y = image_data["positions"]["items_start_y"] + (i * image_data["positions"]["item_height"])
        
        # Item name
        item_name_id = block_id
        text_blocks.append({
            "id": item_name_id,
            "text": item["name"],
            "position": {
                "x_min": 100, 
                "y_min": item_y, 
                "x_max": 400, 
                "y_max": item_y + 25
            },
            "entity_type": "item_name"
        })
        block_id += 1
        
        # Item quantity
        quantity_id = block_id
        text_blocks.append({
            "id": quantity_id,
            "text": str(item["quantity"]),
            "position": {
                "x_min": 500, 
                "y_min": item_y, 
                "x_max": 550, 
                "y_max": item_y + 25
            },
            "entity_type": "item_quantity"
        })
        block_id += 1
        
        # Item price
        price_id = block_id
        text_blocks.append({
            "id": price_id,
            "text": f"${item['price']:.2f}",
            "position": {
                "x_min": 600, 
                "y_min": item_y, 
                "x_max": 670, 
                "y_max": item_y + 25
            },
            "entity_type": "item_price"
        })
        block_id += 1
        
        # Item total
        total_id = block_id
        text_blocks.append({
            "id": total_id,
            "text": f"${item['quantity'] * item['price']:.2f}",
            "position": {
                "x_min": 750, 
                "y_min": item_y, 
                "x_max": 850, 
                "y_max": item_y + 25
            },
            "entity_type": "item_price"
        })
        block_id += 1
        
        # Add relations
        relation_id = 1
        
        # Item name to quantity relation
        relations.append({
            "id": relation_id,
            "source_id": item_name_id,
            "target_id": quantity_id,
            "relation_type": "item_quantity"
        })
        relation_id += 1
        
        # Item name to price relation
        relations.append({
            "id": relation_id,
            "source_id": item_name_id,
            "target_id": price_id,
            "relation_type": "item_price"
        })
        relation_id += 1
        
        # Item name to total relation
        relations.append({
            "id": relation_id,
            "source_id": item_name_id,
            "target_id": total_id,
            "relation_type": "item_total"
        })
        relation_id += 1
    
    # Add subtotal
    text_blocks.append({
        "id": block_id,
        "text": f"${image_data['subtotal']:.2f}",
        "position": image_data["positions"]["subtotal"],
        "entity_type": "subtotal"
    })
    block_id += 1
    
    # Add tax
    text_blocks.append({
        "id": block_id,
        "text": f"${image_data['tax']:.2f}",
        "position": image_data["positions"]["tax"],
        "entity_type": "item_price"
    })
    block_id += 1
    
    # Add total
    text_blocks.append({
        "id": block_id,
        "text": f"${image_data['total']:.2f}",
        "position": image_data["positions"]["total"],
        "entity_type": "total"
    })
    
    # Create the annotation object
    annotation = {
        "image_path": relative_image_path,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "text_blocks": text_blocks,
        "relations": relations
    }
    
    # Write to file
    with open(output_path, "w") as f:
        json.dump(annotation, f, indent=2)

def generate_random_invoice_data():
    """Generate random invoice data"""
    # Generate invoice number
    invoice_number = f"INV-{random.randint(1000, 9999)}"
    
    # Generate invoice date
    days_ago = random.randint(0, 60)
    invoice_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    # Select customer
    customer_name = random.choice(CUSTOMER_NAMES)
    
    # Generate items
    num_items = random.randint(1, 5)
    items = []
    
    for _ in range(num_items):
        product = random.choice(PRODUCTS)
        quantity = random.randint(1, 5)
        
        items.append({
            "name": product["name"],
            "quantity": quantity,
            "price": product["price"]
        })
    
    return invoice_number, invoice_date, customer_name, items

def create_sample_datasets():
    """Create sample datasets for training, validation, and testing"""
    
    # Create training samples
    for i in range(NUM_TRAIN_SAMPLES):
        invoice_number, invoice_date, customer_name, items = generate_random_invoice_data()
        
        image_path = os.path.join(OUTPUT_DIR, "images", "train", f"invoice_{i+1}.jpg")
        annotation_path = os.path.join(OUTPUT_DIR, "annotations", "train", f"invoice_{i+1}.json")
        relative_image_path = f"images/train/invoice_{i+1}.jpg"
        
        image_data = create_invoice_image(image_path, invoice_number, invoice_date, customer_name, items)
        create_annotation_file(image_data, relative_image_path, annotation_path)
        
        print(f"Created training sample {i+1}/{NUM_TRAIN_SAMPLES}")
    
    # Create validation samples
    for i in range(NUM_VAL_SAMPLES):
        invoice_number, invoice_date, customer_name, items = generate_random_invoice_data()
        
        image_path = os.path.join(OUTPUT_DIR, "images", "val", f"invoice_{i+1}.jpg")
        annotation_path = os.path.join(OUTPUT_DIR, "annotations", "val", f"invoice_{i+1}.json")
        relative_image_path = f"images/val/invoice_{i+1}.jpg"
        
        image_data = create_invoice_image(image_path, invoice_number, invoice_date, customer_name, items)
        create_annotation_file(image_data, relative_image_path, annotation_path)
        
        print(f"Created validation sample {i+1}/{NUM_VAL_SAMPLES}")
    
    # Create test samples
    for i in range(NUM_TEST_SAMPLES):
        invoice_number, invoice_date, customer_name, items = generate_random_invoice_data()
        
        image_path = os.path.join(OUTPUT_DIR, "images", "test", f"invoice_{i+1}.jpg")
        annotation_path = os.path.join(OUTPUT_DIR, "annotations", "test", f"invoice_{i+1}.json")
        relative_image_path = f"images/test/invoice_{i+1}.jpg"
        
        image_data = create_invoice_image(image_path, invoice_number, invoice_date, customer_name, items)
        create_annotation_file(image_data, relative_image_path, annotation_path)
        
        print(f"Created test sample {i+1}/{NUM_TEST_SAMPLES}")

if __name__ == "__main__":
    print("Creating sample datasets for invoice extraction...")
    create_sample_datasets()
    print(f"Done! Created {NUM_TRAIN_SAMPLES} training samples, {NUM_VAL_SAMPLES} validation samples, and {NUM_TEST_SAMPLES} test samples.")
    print(f"Dataset is located at: {os.path.abspath(OUTPUT_DIR)}")
