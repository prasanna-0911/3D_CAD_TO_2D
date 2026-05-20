"""
12_easyocr_full.py - EasyOCR Full Extraction
=============================================
Extract everything from CAD drawing using EasyOCR.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from config import TARGET_PDF, OUTPUT_DIR

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = PROJECT_ROOT / "input_pdfs"
EASYOCR_OUTPUT = OUTPUT_DIR / "easyocr_full_extraction"
EASYOCR_OUTPUT.mkdir(exist_ok=True)

import fitz
from PIL import Image
import easyocr

def convert_pdf_to_images(pdf_path: str, output_dir: Path) -> list:
    """Convert PDF to images."""
    images = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_path = output_dir / f"page_{page_num + 1}.png"
        pix.save(img_path)
        images.append(str(img_path))
    doc.close()
    return images

def extract_all_easyocr(image_path: str) -> dict:
    """Extract all text using EasyOCR."""
    print("   [INIT] Loading EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    
    print("   [READ] Processing image...")
    results = reader.readtext(image_path, detail=1)
    
    # Organize results
    all_text = []
    text_with_positions = []
    
    for bbox, text, confidence in results:
        all_text.append(text)
        text_with_positions.append({
            'text': text,
            'confidence': confidence,
            'bbox': bbox
        })
    
    return {
        'all_text': all_text,
        'text_with_positions': text_with_positions,
        'total_text_items': len(all_text)
    }

def parse_extracted_text(raw_data: dict) -> dict:
    """Parse extracted text into categories."""
    all_text = raw_data['all_text']
    full_text = ' '.join(all_text)
    
    parsed = {}
    
    # 1. Extract dimensions (numbers + mm, degree symbols)
    dimensions = []
    for text in all_text:
        # Look for dimension-like patterns
        if re.search(r'\d+\.?\d*\s*(mm|cm|m)', text, re.IGNORECASE):
            dimensions.append(text)
        elif re.search(r'[øØ]\s*\d+', text):
            dimensions.append(text)
        elif re.search(r'R\s*\d+', text, re.IGNORECASE):
            dimensions.append(text)
    parsed['dimensions'] = dimensions[:20]
    
    # 2. Extract numbers that could be dimensions
    numbers = []
    for text in all_text:
        nums = re.findall(r'\d+\.?\d*', text)
        for n in nums:
            if float(n) > 1 and float(n) < 1000:  # Reasonable dimension range
                numbers.append(f"{n} mm")
    parsed['dimension_values'] = numbers[:20]
    
    # 3. Title block keywords
    title_block_keywords = ['drawing', 'part', 'name', 'number', 'revision', 'scale', 
                           'units', 'material', 'date', 'author', 'checked', 'approved']
    title_block_text = []
    for text in all_text:
        text_lower = text.lower()
        if any(kw in text_lower for kw in title_block_keywords):
            title_block_text.append(text)
    parsed['title_block_keywords'] = title_block_text[:10]
    
    # 4. All readable text (grouped)
    parsed['all_readable_text'] = all_text
    
    # 5. Check for specific patterns
    # GD&T related
    gdt_keywords = ['position', 'flatness', 'straightness', 'perpendicular', 
                   'parallelism', 'circularity', 'runout', 'profile', 'tolerance']
    gdt_found = [t for t in all_text if any(kw in t.lower() for kw in gdt_keywords)]
    parsed['gdt_mentions'] = gdt_found
    
    # Notes keywords
    note_keywords = ['note', 'warning', 'caution', 'note:', 'surface', 'finish', 'material']
    notes_found = [t for t in all_text if any(kw in t.lower() for kw in note_keywords)]
    parsed['note_keywords'] = notes_found
    
    # View keywords
    view_keywords = ['front', 'top', 'right', 'left', 'side', 'section', 'detail', 'isometric']
    views_found = [t for t in all_text if any(kw in t.lower() for kw in view_keywords)]
    parsed['view_mentions'] = views_found
    
    # Feature keywords
    feature_keywords = ['hole', 'fillet', 'chamfer', 'slot', 'pocket', 'rib', 'bore']
    features_found = [t for t in all_text if any(kw in t.lower() for kw in feature_keywords)]
    parsed['feature_mentions'] = features_found
    
    return parsed

def main():
    print("=" * 60)
    print("  EasyOCR Full Extraction")
    print("=" * 60)
    
    pdf_path = INPUT_DIR / TARGET_PDF
    print(f"\n[PDF] {TARGET_PDF}")
    
    temp_dir = EASYOCR_OUTPUT / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    print("\n[CONVERT] Converting PDF to images...")
    images = convert_pdf_to_images(str(pdf_path), temp_dir)
    print(f"   Created {len(images)} image(s)")
    
    # Save sample image
    sample_path = EASYOCR_OUTPUT / "sample_page_1.png"
    print(f"\n[SAVE] Saving sample image: {sample_path}")
    
    print("\n[EXTRACT] Extracting with EasyOCR...")
    raw_data = extract_all_easyocr(images[0])
    
    print("\n[PARSING] Categorizing extracted text...")
    parsed = parse_extracted_text(raw_data)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "drawing_name": TARGET_PDF,
        "model": "EasyOCR",
        "extraction_date": datetime.now().isoformat(),
        "total_text_items": raw_data['total_text_items'],
        "raw_extraction": raw_data,
        "parsed_data": parsed
    }
    
    output_file = EASYOCR_OUTPUT / f"easyocr_full_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Saved: {output_file}")
    
    # Summary
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"   Total text items: {raw_data['total_text_items']}")
    print(f"   Dimensions found: {len(parsed.get('dimension_values', []))}")
    print(f"   Title block keywords: {len(parsed.get('title_block_keywords', []))}")
    print(f"   GD&T mentions: {len(parsed.get('gdt_mentions', []))}")
    print(f"   Note keywords: {len(parsed.get('note_keywords', []))}")
    print(f"   View mentions: {len(parsed.get('view_mentions', []))}")
    print(f"   Feature mentions: {len(parsed.get('feature_mentions', []))}")
    
    # Save sample image
    img = Image.open(images[0])
    img.save(sample_path)
    print(f"\n[OK] Sample image saved: {sample_path}")

if __name__ == "__main__":
    main()