# ============================================================================
# MODEL 1 - EasyOCR (Ground Truth Baseline)
# ============================================================================
"""
EasyOCR Testing for CAD Drawings - Comprehensive 71 Parameter Test
===============================================================
This script extracts ALL visible text from drawings and maps it
to the 71 parameter categories.

Run on: Google Colab (CPU or GPU)
Output: JSON + Excel saved to Google Drive
"""

# ============================================================================
# CELL 1 - MOUNT DRIVE & SETUP
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

os.chdir("/content/drive/MyDrive/3D_CAD_TO_2D")

# Create output directories
output_dir = Path("outputs/easyocr_results")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print(f"Output directory: {output_dir}")

# ============================================================================
# CELL 2 - INSTALL EASYOCR
# ============================================================================
!pip install -q easyocr pymupdf pandas openpyxl tqdm
print("✅ EasyOCR installed!")

# ============================================================================
# CELL 3 - DEFINE PARAMETERS
# ============================================================================
PARAMETERS = {
    "Geometry_Design": [
        {"id": "1", "name": "Geometry & Design Changes"},
        {"id": "2", "name": "Shape modifications"},
        {"id": "3", "name": "Suppressed/unsuppressed features"},
        {"id": "4", "name": "Coordinate system shifts"},
        {"id": "5", "name": "Holes Feature"},
        {"id": "6", "name": "Fillet Feature"},
        {"id": "7", "name": "Chamfers Feature"},
        {"id": "8", "name": "Ribs Feature"},
    ],
    "Dimensions": [
        {"id": "9", "name": "Size changes"},
        {"id": "10", "name": "Length"},
        {"id": "11", "name": "Diameter"},
        {"id": "12", "name": "Thickness"},
        {"id": "13", "name": "Position changes"},
        {"id": "14", "name": "Location changes of features"},
        {"id": "19", "name": "Dimension position changes"},
        {"id": "22", "name": "Dimension value changes"},
    ],
    "Assembly": [
        {"id": "15", "name": "Assembly fit/interface changes"},
        {"id": "16", "name": "Exploded view differences"},
        {"id": "17", "name": "Sub-assembly changes"},
        {"id": "18", "name": "Fastener type/size updates"},
    ],
    "Tolerances": [
        {"id": "20", "name": "Tolerances change"},
        {"id": "21", "name": "Tolerances Location change"},
        {"id": "23", "name": "Tolerance updates"},
        {"id": "24", "name": "Fits (H7/g6)"},
    ],
    "GD_T": [
        {"id": "25", "name": "Position tolerance"},
        {"id": "26", "name": "Straightness"},
        {"id": "27", "name": "Flatness"},
        {"id": "28", "name": "Circularity"},
        {"id": "29", "name": "Parallelism"},
        {"id": "30", "name": "Perpendicularity"},
        {"id": "31", "name": "Angularity"},
        {"id": "32", "name": "Runout"},
    ],
    "Views_Drafting": [
        {"id": "33", "name": "View changes"},
        {"id": "34", "name": "View add/delete"},
        {"id": "35", "name": "View scale change"},
        {"id": "36", "name": "View representation change"},
        {"id": "37", "name": "Drafting view"},
        {"id": "38", "name": "Section line"},
        {"id": "39", "name": "Hatching change"},
        {"id": "40", "name": "Line type change"},
        {"id": "41", "name": "Fake dimension"},
        {"id": "42", "name": "Dimension arrow change"},
        {"id": "43", "name": "Dimension line thickness change"},
        {"id": "44", "name": "Dimension position change"},
        {"id": "45", "name": "View position changes"},
        {"id": "46", "name": "Leader line change"},
        {"id": "47", "name": "Additional curve creation"},
        {"id": "48", "name": "Section view shift"},
        {"id": "49", "name": "Broken view shift"},
        {"id": "50", "name": "Additional import photo/graphics"},
    ],
    "Notes_Annotations": [
        {"id": "51", "name": "General notes updates"},
        {"id": "52", "name": "Notes Location"},
        {"id": "53", "name": "Special instructions added/removed"},
        {"id": "54", "name": "Flag notes / key characteristics"},
        {"id": "55", "name": "Special symbols add/delete"},
        {"id": "56", "name": "Welding symbols"},
        {"id": "57", "name": "Datum changes"},
        {"id": "58", "name": "Datum Location changes"},
    ],
    "TitleBlock_Metadata": [
        {"id": "59", "name": "Revision number"},
        {"id": "60", "name": "Drawing number"},
        {"id": "61", "name": "Part name changes"},
        {"id": "62", "name": "Author"},
        {"id": "63", "name": "Checker/approver updates"},
        {"id": "64", "name": "Dates"},
        {"id": "65", "name": "Company or project info"},
    ],
    "BOM": [
        {"id": "66", "name": "BOM (Bill of Materials)"},
        {"id": "67", "name": "Component added/removed"},
        {"id": "68", "name": "Part number revisions"},
        {"id": "69", "name": "Part list"},
    ],
    "Scale": [
        {"id": "70", "name": "Sheet Scale changes"},
        {"id": "71", "name": "View Scale changes"},
    ]
}

# ============================================================================
# CELL 4 - CATEGORIZATION PATTERNS
# ============================================================================
import re

def categorize_element(text):
    """Categorize OCR text element based on patterns."""
    text_upper = text.upper()

    # Dimensions - diameter
    if re.search(r'^[F\[\(]?\d+', text) or 'Ø' in text or 'PHI' in text_upper:
        return "Dimensions", "Diameter symbol detected"

    # Tolerances (+/- values)
    if re.search(r'[+-]\d+\.\d+', text):
        return "Tolerances", "Tolerance value detected"

    # Radius
    if re.match(r'^R\s*\d+', text):
        return "Dimensions", "Radius value"

    # GD&T symbols
    gdt_patterns = ['FLATNESS', 'STRAIGHTNESS', 'PERPENDICULAR', 'PARALLEL', 'RUNOUT', 'POSITION', 'CIRCULAR']
    if any(p in text_upper for p in gdt_patterns):
        return "GD_T", "GD&T symbol detected"

    # Views
    view_patterns = ['FRONT VIEW', 'TOP VIEW', 'SIDE VIEW', 'SECTION', 'DETAIL', 'SCALE', 'A-A', 'B-B']
    if any(p in text_upper for p in view_patterns):
        return "Views_Drafting", "View annotation detected"

    # Title block keywords
    title_patterns = ['REV', 'DRAWN', 'CHECKED', 'APPROVED', 'DATE', 'DWG', 'PART', 'NAME', 'SCALE']
    if any(p in text_upper for p in title_patterns):
        return "TitleBlock_Metadata", "Title block element"

    # Notes
    notes_patterns = ['NOTE', 'WARNING', 'CAUTION', 'IMPORTANT', 'SPECIAL']
    if any(p in text_upper for p in notes_patterns):
        return "Notes_Annotations", "Note detected"

    return "Uncategorized", "Not matched to any category"


# ============================================================================
# CELL 5 - INITIALIZE EASYOCR
# ============================================================================
import torch
import easyocr

print(f"GPU Available: {torch.cuda.is_available()}")

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
print("✅ EasyOCR initialized!")

# ============================================================================
# CELL 6 - EXTRACT FROM SINGLE PDF
# ============================================================================
import fitz
from PIL import Image

def extract_from_pdf(pdf_path, zoom=3.0):
    """Extract all text from a PDF drawing."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Save temp image
    temp_path = "/tmp/temp_drawing.png"
    pix.save(temp_path)
    doc.close()

    # Run EasyOCR
    results = reader.readtext(temp_path, detail=1, paragraph=False)

    elements = []
    for bbox, text, confidence in results:
        # Calculate center
        x = sum(p[0] for p in bbox) / 4
        y = sum(p[1] for p in bbox) / 4

        # Fix Ø misreadings
        text = re.sub(r'^F(\d+)', r'Ø\1', text)
        text = re.sub(r'^\[(\d+)', r'Ø\1', text)
        text = re.sub(r'^p(\d+)', r'Ø\1', text)

        category, reason = categorize_element(text)

        elements.append({
            "text": text.strip(),
            "confidence": round(confidence, 3),
            "x": round(x, 1),
            "y": round(y, 1),
            "category": category,
            "category_reason": reason
        })

    return elements


# ============================================================================
# CELL 7 - PROCESS ALL PDFs
# ============================================================================
pdf_files = [
    "input_pdfs/HP_V1__Final Shaft_2d.pdf",
    "input_pdfs/HP_V2__Final Shaft_2d.pdf",
    "input_pdfs/HP_V1__bolt.pdf",
    "input_pdfs/HP_V2__bolt_2d.pdf",
    "input_pdfs/HP_Before__1st__NOVEX-GS 414 UFS DA-1__2d.pdf",
    "input_pdfs/HP_After__1st__NOVEX-GS 414 UFS DA-1__2d.pdf",
]

all_results = []

for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
    if not os.path.exists(pdf_file):
        print(f"❌ File not found: {pdf_file}")
        continue

    drawing_name = Path(pdf_file).stem
    print(f"\n📄 Processing: {drawing_name}")

    # Extract
    elements = extract_from_pdf(pdf_file, zoom=3.0)
    print(f"   Found {len(elements)} text elements")

    # Create result structure
    result = {
        "drawing_name": drawing_name,
        "model": "EasyOCR",
        "extraction_date": datetime.now().isoformat(),
        "total_elements": len(elements),
        "elements": elements,
        "category_counts": {}
    }

    # Count by category
    for elem in elements:
        cat = elem["category"]
        result["category_counts"][cat] = result["category_counts"].get(cat, 0) + 1

    # Save JSON
    json_path = output_dir / f"{drawing_name}_easyocr.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"   ✅ JSON saved: {json_path.name}")

    all_results.append(result)

    # Create Excel
    df = pd.DataFrame(elements)
    if len(df) > 0:
        excel_path = output_dir / f"{drawing_name}_easyocr.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"   ✅ Excel saved: {excel_path.name}")

# ============================================================================
# CELL 8 - SAVE COMBINED RESULTS
# ============================================================================
combined_path = output_dir / "ALL_DRAWINGS_easyocr.json"
with open(combined_path, 'w', encoding='utf-8') as f:
    json.dump({
        "model": "EasyOCR",
        "extraction_date": datetime.now().isoformat(),
        "total_drawings": len(all_results),
        "results": all_results
    }, f, indent=2)

print(f"\n✅ All results saved to: {output_dir}")

# ============================================================================
# CELL 9 - PRINT SUMMARY
# ============================================================================
print("\n" + "="*70)
print("EASYOCR EXTRACTION SUMMARY")
print("="*70)

for result in all_results:
    print(f"\n📄 {result['drawing_name']}")
    print(f"   Total elements: {result['total_elements']}")
    for cat, count in result['category_counts'].items():
        print(f"   {cat}: {count}")

print("\n" + "="*70)
print("✅ EasyOCR testing complete!")
print(f"   Results saved to: {output_dir}")
print("="*70)