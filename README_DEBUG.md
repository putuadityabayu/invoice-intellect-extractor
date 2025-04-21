
# Invoice Processing Debug Guide

This guide explains how to process an invoice image and inspect the intermediate results from each stage of the pipeline.

## Running the Demo Script

You can process an invoice and generate debug outputs using the `demo_pipeline.py` script:

```bash
python demo_pipeline.py path/to/your/invoice.jpg
```

This will:
1. Process the invoice image through all pipeline stages
2. Save intermediate outputs to the `debug/` directory
3. Print the final extracted data to the console

## Examining Debug Outputs

After running the demo script, you'll find the following files in the `debug/` directory:

1. **preprocessed_image.jpg**
   - The preprocessed invoice image after grayscale conversion, thresholding, denoising, and sharpening.
   - Useful for checking if the preprocessing improved or degraded image quality.

2. **ocr_results.json**
   - Raw OCR output with text and position information for each detected text block.
   - Contains the text content, confidence score, and bounding box coordinates.

3. **ocr_visual_results.txt**
   - Human-readable version of the OCR results showing text and positions.

4. **entity_results.json**
   - Results from the entity classification stage.
   - Shows how text blocks were classified into invoice fields (invoice number, date, etc.).

5. **layout_results.json**
   - Results from the layout analysis stage.
   - Shows how document structure was understood based on both text and spatial information.

6. **relation_results.json**
   - Results from the relation extraction stage.
   - Shows how relationships between text blocks (especially for line items) were identified.

7. **combined_results.json**
   - Results after combining outputs from all extraction methods.
   - Shows how conflicts between different extraction methods were resolved.

8. **final_output.json**
   - The final formatted output that would be returned to the user.

## Interpreting the Debug Files

### For OCR Results

Look at the text recognition accuracy:
- Are all text blocks correctly identified?
- Are positions (bounding boxes) accurate?
- Are there any missing text blocks?

### For Entity Classification

Check how text blocks were classified:
- Were invoice fields (number, date, customer) correctly identified?
- Are there any misclassifications?

### For Layout Analysis

Examine how the document structure was understood:
- Did the model correctly identify headers, line items, and totals?
- Did it properly use spatial information?

### For Relation Extraction

Check relationship identification:
- Were item names correctly linked to quantities, prices, etc.?
- Were header fields correctly associated?

### For Final Output

Verify the quality of the final result:
- Does it contain all required information?
- Is the structure correct?
- Are there any errors or missing data?

## Customizing Debug Output

You can modify the `process_invoice` function in `src/pipeline.py` to add or change debug outputs for specific pipeline stages.

## Troubleshooting

If you encounter issues:

1. Check the preprocessed image to see if OCR problems are due to image quality
2. Examine OCR results to see if text was correctly recognized
3. Compare entity, layout, and relation results to understand which approach worked best
4. Look at combined results to see how conflicts were resolved

The debug outputs provide transparency into each step of the process, making it easier to identify and fix issues.
