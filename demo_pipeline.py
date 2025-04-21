
"""
Demo script to run the invoice processing pipeline and generate debug outputs
"""

import os
import sys
import argparse
import json
from src.pipeline import process_invoice
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Process an invoice image and generate debug outputs')
    parser.add_argument('image_path', help='Path to the invoice image file')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        logger.error(f"Image file not found: {args.image_path}")
        return 1
    
    # Ensure debug directory exists
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    logger.info(f"Processing invoice image: {args.image_path}")
    logger.info(f"Debug outputs will be saved to: {debug_dir}")
    
    # Process the invoice
    try:
        result = process_invoice(args.image_path)
        
        # Print the final result
        logger.info("Processing complete. Final result:")
        print(json.dumps(result, indent=2))
        
        logger.info(f"All debug outputs have been saved to: {debug_dir}")
        return 0
    except Exception as e:
        logger.error(f"Error processing invoice: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
