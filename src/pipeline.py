"""
Main processing pipeline for invoice data extraction
"""

from typing import Dict, Any, List
import os
import json
from src.ocr.processor import extract_text_with_positions
from src.preprocessing.image_preprocessor import preprocess_image, save_image
from src.models.entity_classifier import EntityClassifier
from src.models.relation_extractor import RelationExtractor
from src.models.layout_model import LayoutExtractor
from src.utils.data_formatter import format_invoice_data
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models (load pre-trained models)
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(models_dir, exist_ok=True)

entity_model_path = os.path.join(models_dir, "entity_model.pth")
relation_model_path = os.path.join(models_dir, "relation_model.pth")
layout_model_path = os.path.join(models_dir, "layout_model.pth")
spacy_model_path = os.path.join(models_dir, "spacy_ner")

# Check if models exist
entity_model_exists = os.path.exists(entity_model_path)
relation_model_exists = os.path.exists(relation_model_path)
layout_model_exists = os.path.exists(layout_model_path)
spacy_model_exists = os.path.exists(spacy_model_path)

logger.info(f"Entity model exists: {entity_model_exists}")
logger.info(f"Relation model exists: {relation_model_exists}")
logger.info(f"Layout model exists: {layout_model_exists}")
logger.info(f"SpaCy NER model exists: {spacy_model_exists}")

# Initialize models with paths if they exist
entity_classifier = EntityClassifier(
    entity_model_path if entity_model_exists else None,
    use_spacy=True  # Always use spaCy for NER
)
relation_extractor = RelationExtractor(relation_model_path if relation_model_exists else None)
layout_extractor = LayoutExtractor(layout_model_path if layout_model_exists else None)

# Create debug directory
debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug")
os.makedirs(debug_dir, exist_ok=True)


def save_debug_output(data, filename):
    """
    Save data to a debug file
    
    Args:
        data: Data to save
        filename: Filename to save to
    """
    filepath = os.path.join(debug_dir, filename)
    
    # Determine file type based on extension
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    elif ext == '.txt':
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(data))
    elif ext == '.md':
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)
    
    logger.info(f"Saved debug output to {filepath}")


def process_invoice(image_path: str) -> Dict[str, Any]:
    """
    Main pipeline function to process an invoice and extract structured data
    
    Args:
        image_path: Path to the invoice image
        
    Returns:
        Structured invoice data as a dictionary
    """
    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Save preprocessed image for debugging
    save_image(preprocessed_image, os.path.join(debug_dir, "preprocessed_image.jpg"))
    logger.info(f"Saved preprocessed image to debug/preprocessed_image.jpg")
    
    # Step 2: Extract text with positions using OCR
    text_blocks = extract_text_with_positions(preprocessed_image)
    
    # Save OCR results for debugging
    save_debug_output(text_blocks, "ocr_results.json")
    
    # Debug: Print the extracted text blocks
    logger.info(f"Extracted {len(text_blocks)} text blocks from the invoice")
    for i, block in enumerate(text_blocks[:5]):
        logger.info(f"Block {i}: {block['text'][:50]}...")
    
    # Step 3: Apply different extraction approaches
    
    # 3.1: Classic entity classification (with spaCy NER)
    classified_entities = entity_classifier.classify(text_blocks)
    logger.info(f"Entity classifier results: {json.dumps(classified_entities, indent=2)}")
    
    # Save entity classification results for debugging
    save_debug_output(classified_entities, "entity_results.json")
    
    # 3.2: Layout-aware extraction
    layout_results = layout_extractor.process(text_blocks)
    logger.info(f"Layout extractor results: {json.dumps(layout_results, indent=2)}")
    
    # Save layout analysis results for debugging
    save_debug_output(layout_results, "layout_results.json")
    
    # 3.3: Relation extraction for item relationships
    extracted_relations = relation_extractor.extract(text_blocks, classified_entities)
    logger.info(f"Relation extractor results: {json.dumps(extracted_relations, indent=2)}")
    
    # Save relation extraction results for debugging
    save_debug_output(extracted_relations, "relation_results.json")
    
    # Step 4: Combine results from different approaches
    final_results = combine_extraction_results(classified_entities, layout_results, extracted_relations)
    logger.info(f"Combined results: {json.dumps(final_results, indent=2)}")
    
    # Save combined results for debugging
    save_debug_output(final_results, "combined_results.json")
    
    # Step 5: Format the extracted data into the required JSON structure
    formatted_data = format_invoice_data(final_results, extracted_relations)
    
    # Save final output for debugging
    save_debug_output(formatted_data, "final_output.json")
    
    return formatted_data


def combine_extraction_results(
    entity_results: Dict[str, Any],
    layout_results: Dict[str, Any],
    relation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine results from different extraction approaches
    
    Args:
        entity_results: Results from entity classifier
        layout_results: Results from layout extractor
        relation_results: Results from relation extractor
        
    Returns:
        Combined results
    """
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
            # Compute average completeness (having name, quantity, unit_price, total_price)
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
        # First extract numeric values if possible
        layout_value = extract_numeric_value(layout_results.get(field))
        entity_value = extract_numeric_value(entity_results.get(field))
        relation_value = relation_results.get(field)
        
        # Prioritize values
        if isinstance(relation_value, (int, float)) and relation_value > 0:
            combined[field] = relation_value
        elif layout_value is not None:
            combined[field] = layout_value
        elif entity_value is not None:
            combined[field] = entity_value
        else:
            # Keep text value if no numeric value
            combined[field] = (
                layout_results.get(field) or 
                entity_results.get(field) or 
                relation_results.get(field)
            )
    
    # Combine extra price items
    extra_price_items = []
    
    # From layout results
    for item in layout_results.get("extra_price", []):
        if isinstance(item, str):
            label, value = parse_extra_price(item)
            if label and value is not None:
                extra_price_items.append({label: value})
        elif isinstance(item, dict):
            extra_price_items.append(item)
    
    # From entity results
    for item in entity_results.get("extra_price", []):
        if isinstance(item, str):
            label, value = parse_extra_price(item)
            if label and value is not None:
                extra_price_items.append({label: value})
        elif isinstance(item, dict):
            extra_price_items.append(item)
    
    # From relation results
    if isinstance(relation_results.get("extra_price"), list):
        extra_price_items.extend(relation_results.get("extra_price", []))
    
    # Remove duplicates by key
    seen_keys = set()
    unique_items = []
    for item in extra_price_items:
        if isinstance(item, dict) and len(item) > 0:
            key = next(iter(item.keys()))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_items.append(item)
    
    combined["extra_price"] = unique_items
    
    return combined


def extract_numeric_value(text):
    """
    Extract numeric value from text
    
    Args:
        text: Text containing a numeric value
        
    Returns:
        Extracted numeric value or None
    """
    if text is None:
        return None
        
    if isinstance(text, (int, float)):
        return text
        
    import re
    # Remove currency symbols and commas
    cleaned_text = re.sub(r'[$€£¥,]', '', str(text))
    
    # Find numbers with optional decimal point
    matches = re.findall(r'\d+\.\d+|\d+', cleaned_text)
    if matches:
        return float(matches[0])
    return None


def parse_extra_price(text):
    """
    Parse extra price text into label and value
    
    Args:
        text: Text containing extra price information
        
    Returns:
        Tuple of (label, value)
    """
    if not text:
        return None, None
        
    import re
    
    # Extract numeric value
    value = extract_numeric_value(text)
    if value is None:
        return None, None
    
    # Determine label based on keywords
    text_lower = text.lower()
    
    if "tax" in text_lower or "vat" in text_lower or "gst" in text_lower:
        return "tax", value
    elif "discount" in text_lower:
        return "discount", value
    elif "shipping" in text_lower or "delivery" in text_lower:
        return "shipping", value
    else:
        return "other", value

def prepare_spacy_training_data(data_dir):
    """
    Prepare training data for spaCy NER model with proper config format
    """
    if not SPACY_AVAILABLE:
        print("spaCy not available. Skipping NER preparation.")
        return False
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(data_dir), "models", "spacy_ner")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create config in proper .cfg format
    config_text = """
[paths]
train = "corpus/train.spacy"
dev = "corpus/dev.spacy"
vectors = null
init_tok2vec = null

[system]
gpu_allocator = "pytorch"

[nlp]
lang = "id"
pipeline = ["tok2vec","ner"]
batch_size = 128
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = ${components.tok2vec.model.encode.width}
attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
rows = [5000, 2500, 2500, 2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 256
depth = 8
window_size = 1
maxout_pieces = 3

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = false
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.encode.width}

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
accumulate_gradient = 1
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = 0
gpu_allocator = "pytorch"
dropout = 0.1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
before_to_disk = null
batcher = {"@batchers": "spacy.batch_by_words.v1", "discard_oversize": true, "size": 2000, "tolerance": 0.2, "get_length": null}

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 0.00005

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0

[pretraining]

[initialize]
vectors = null
init_tok2vec = null
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""
    
    # Save config in .cfg format
    config_path = os.path.join(output_dir, "config.cfg")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_text.strip())
    
    # Load base model
    try:
        nlp = spacy.blank("id")
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        return False
    
    # Create DocBin to store training documents
    train_doc_bin = DocBin()
    val_doc_bin = DocBin()
    
    # Entity labels we want to train
    entity_labels = ["INVOICE_NUMBER", "DATE", "CUSTOMER", "ITEM", "QUANTITY", "PRICE", "SUBTOTAL", "TOTAL"]
    
    # Add entity labels to NER pipe
    ner = nlp.add_pipe("ner")
    for label in entity_labels:
        ner.add_label(label)
    
    # Map our entity types to spaCy format
    entity_map = {
        "invoice_number": "INVOICE_NUMBER",
        "invoice_date": "DATE",
        "customer_name": "CUSTOMER",
        "item_name": "ITEM",
        "item_quantity": "QUANTITY",
        "item_price": "PRICE",
        "subtotal": "SUBTOTAL",
        "total": "TOTAL"
    }
    
    # Process training data
    print("Preparing spaCy NER training data...")
    
    for split in ["train", "val"]:
        annotations_dir = os.path.join(data_dir, "annotations", split)
        
        if not os.path.exists(annotations_dir):
            print(f"Directory not found: {annotations_dir}")
            continue
        
        for ann_file in tqdm(os.listdir(annotations_dir)):
            if not ann_file.endswith(".json"):
                continue
                
            with open(os.path.join(annotations_dir, ann_file), "r") as f:
                try:
                    annotation = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON file: {ann_file}")
                    continue
            
            # Create a dictionary to map position to text block
            text_blocks_by_position = {}
            for block in annotation.get("text_blocks", []):
                if "position" in block:
                    pos_key = (
                        block["position"].get("x_min", 0),
                        block["position"].get("y_min", 0),
                        block["position"].get("x_max", 0),
                        block["position"].get("y_max", 0)
                    )
                    text_blocks_by_position[pos_key] = block
            
            # Create a full text document by concatenating all text blocks
            full_text = " ".join(block.get("text", "") for block in annotation.get("text_blocks", []))
            doc = nlp.make_doc(full_text)
            
            # Map entity spans in the full text
            ents = []
            for block in annotation.get("text_blocks", []):
                if "entity_type" in block and block["entity_type"] in entity_map:
                    text = block.get("text", "")
                    if text in full_text:
                        start_idx = full_text.find(text)
                        end_idx = start_idx + len(text)
                        spacy_label = entity_map[block["entity_type"]]
                        span = doc.char_span(start_idx, end_idx, label=spacy_label)
                        if span:
                            ents.append(span)
            
            # Set entities on the document
            try:
                doc.ents = ents
                # Add to appropriate DocBin
                if split == "train":
                    train_doc_bin.add(doc)
                else:
                    val_doc_bin.add(doc)
            except Exception as e:
                print(f"Error adding entities to document: {e}")
    
    # Save DocBin to disk
    train_path = os.path.join(output_dir, "train.spacy")
    val_path = os.path.join(output_dir, "dev.spacy")
    
    train_doc_bin.to_disk(train_path)
    val_doc_bin.to_disk(val_path)
    
    print(f"Saved spaCy training data to {train_path} and {val_path}")
    
    # Create train script
    train_script = f"""
# Train spaCy NER model
python -m spacy train {config_path} --output {output_dir} --paths.train {train_path} --paths.dev {val_path}
"""
    
    script_path = os.path.join(output_dir, "train.sh")
    with open(script_path, "w") as f:
        f.write(train_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created training script at {script_path}")
    print("To train the spaCy NER model, run the following command:")
    print(f"bash {script_path}")
    
    return True
