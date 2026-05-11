# ============================================================================
# 1_test_easyocr.py - EasyOCR Ground Truth Extraction
# ============================================================================
"""
EasyOCR extraction script for CAD engineering drawings.
Uses GPU if available, falls back to CPU.

EasyOCR is the BEST performer for CAD text extraction because:
1. It reads actual pixel text (not hallucinating like VLMs)
2. GPU-accelerated (fast on T4/ RTX cards)
3. Good at technical/dense text
4. Provides confidence scores
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import (
    INPUT_DIR, OUTPUT_DIR, MODELS_CACHE_DIR,
    PDF_DPI_SETTINGS, MAX_IMAGE_SIZE_OCR, print_config
)
from scripts.utils import (
    pdf_to_image, save_json, save_excel,
    print_header, print_model_status, get_gpu_memory_info,
    categorize_text_elements, fix_diameter_symbol, merge_split_values,
    sanitize_filename, ensure_dir
)


def init_easyocr():
    """Initialize EasyOCR reader."""
    import easyocr

    gpu_available = get_gpu_memory_info()["mode"] == "GPU"
    print(f"   GPU available: {gpu_available}")

    reader = easyocr.Reader(['en'], gpu=gpu_available, verbose=False)
    print("   ✅ EasyOCR initialized")
    return reader


def extract_text(reader, image_path: str):
    """Extract text from image using EasyOCR."""
    results = reader.readtext(image_path, detail=1, paragraph=False)

    elements = []
    for idx, item in enumerate(results):
        bbox, text, conf = item

        # Calculate center point
        x = sum(float(p[0]) for p in bbox) / 4
        y = sum(float(p[1]) for p in bbox) / 4

        # Fix common misreadings
        text = fix_diameter_symbol(str(text).strip())

        elements.append({
            "id": idx + 1,
            "text": text,
            "confidence": round(float(conf), 3),
            "x": round(x, 1),
            "y": round(y, 1),
            "bbox": [[float(p[0]), float(p[1])] for p in bbox]
        })

    # Merge split values
    elements = merge_split_values(elements)

    return elements


def save_results(drawing_name: str, elements: list, output_dir: Path):
    """Save extraction results to JSON and Excel."""
    ensure_dir(output_dir)

    # Categorize elements
    categories = categorize_text_elements(elements)

    # Build summary
    total = len(elements)
    category_counts = {cat: len(items) for cat, items in categories.items()}

    summary = {
        "drawing_name": drawing_name,
        "model": "EasyOCR",
        "extraction_date": datetime.now().isoformat(),
        "total_text_elements": total,
        "by_category": category_counts,
        "image_zoom": PDF_DPI_SETTINGS["easyocr"],
    }

    # Save JSON with all data
    json_data = {
        **summary,
        "elements": elements,
        "categories": {cat: [{"id": e["id"], "text": e["text"]} for e in items]
                        for cat, items in categories.items()}
    }

    json_path = output_dir / f"{drawing_name}_easyocr.json"
    save_json(json_data, json_path)

    # Save Excel with categorization
    excel_data = []
    for category, items in categories.items():
        for elem in items:
            excel_data.append({
                "ID": elem["id"],
                "Category": category,
                "Text": elem["text"],
                "Confidence": elem["confidence"],
                "X": elem["x"],
                "Y": elem["y"]
            })

    if excel_data:
        excel_path = output_dir / f"{drawing_name}_easyocr.xlsx"
        save_excel(excel_data, excel_path, "EasyOCR Results")

    return summary


def process_all_drawings():
    """Process all PDF drawings in input directory."""
    print_header("EasyOCR - CAD Drawing Text Extraction")
    print_config()

    # Find all PDFs
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ No PDF files found in {INPUT_DIR}")
        return

    print(f"\n📄 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   • {pdf.name}")

    # Initialize EasyOCR
    print("\n📦 Initializing EasyOCR...")
    reader = init_easyocr()

    # Create output directory
    output_dir = OUTPUT_DIR / "easyocr_results"
    ensure_dir(output_dir)

    # Process each drawing
    all_results = []
    print("\n" + "=" * 60)

    for pdf_path in tqdm(pdf_files, desc="Processing drawings"):
        drawing_name = pdf_path.stem

        print(f"\n📄 {drawing_name}")

        # Convert PDF to image
        image = pdf_to_image(
            pdf_path,
            zoom=PDF_DPI_SETTINGS["easyocr"],
            max_size=MAX_IMAGE_SIZE_OCR
        )

        # Save temp image
        temp_image_path = output_dir / f"{drawing_name}_temp.png"
        image.save(temp_image_path, "PNG")
        image_path = str(temp_image_path)

        # Extract text
        print(f"   Extracting text...")
        elements = extract_text(reader, image_path)

        print(f"   Found {len(elements)} text elements")

        # Save results
        summary = save_results(drawing_name, elements, output_dir)
        all_results.append(summary)

        # Cleanup temp image
        temp_image_path.unlink(missing_ok=True)

    # Save combined results
    print("\n" + "=" * 60)
    print("📊 Summary - All Drawings")

    combined_path = output_dir / "ALL_DRAWINGS_easyocr_summary.json"
    combined_data = {
        "model": "EasyOCR",
        "extraction_date": datetime.now().isoformat(),
        "total_drawings": len(all_results),
        "results": all_results
    }
    save_json(combined_data, combined_path)

    print("\n✅ EasyOCR extraction complete!")
    print(f"   Results saved to: {output_dir}")

    return all_results


if __name__ == "__main__":
    try:
        process_all_drawings()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()