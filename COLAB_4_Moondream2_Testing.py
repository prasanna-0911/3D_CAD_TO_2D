# ============================================================================
# MODEL 4 - Moondream2 Comprehensive Testing
# ============================================================================
"""
Moondream2 Testing for CAD Drawings - 71 Parameter Extraction
===============================================================
Lightweight VLM good for technical images (~2GB VRAM).

Run on: Google Colab (any GPU)
Output: JSON + Excel saved to Google Drive
"""

# ============================================================================
# CELL 1 - MOUNT DRIVE & SETUP
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

import os
import gc
import torch

os.chdir("/content/drive/MyDrive/3D_CAD_TO_2D")

gc.collect()
torch.cuda.empty_cache()

output_dir = Path("outputs/moondream_results")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ============================================================================
# CELL 2 - INSTALL DEPENDENCIES
# ============================================================================
!pip install -q transformers pillow pymupdf pandas openpyxl tqdm

from transformers import AutoModelForVision2Seq, AutoProcessor

print("✅ Dependencies installed!")

# ============================================================================
# CELL 3 - LOAD MOONDREAM2
# ============================================================================
print("\n📦 Loading Moondream2...")

model = AutoModelForVision2Seq.from_pretrained(
    "vikhyatk/moondream2",
    revision="2024-08-26",
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("vikhyatk/moondream2")

print("✅ Moondream2 loaded!")

# ============================================================================
# CELL 4 - DEFINE EXTRACTION QUERIES
# ============================================================================
EXTRACTION_QUERIES = [
    ("Dimensions", "List ALL dimension values visible in this engineering drawing. Include lengths, diameters (Ø), radii (R), and tolerance values (+/-). Give exact numbers."),
    ("GD_T", "What GD&T (Geometric Dimensioning and Tolerancing) symbols are shown? Include position, flatness, perpendicularity, parallelism, runout, etc. with their values and datum references."),
    ("Views", "What views are shown in this drawing? List front view, top view, side view, section views (A-A, B-B), detail views, and their scales."),
    ("Geometry", "Describe all geometric features: holes with sizes, fillets with radii, chamfers, ribs, and any shape modifications."),
    ("TitleBlock", "Extract the title block information: part name, drawing number, revision (REV), scale, date, designer (DRAWN BY), checker, and approver."),
    ("Notes", "What notes and annotations are visible? Include general notes, surface finish requirements, welding symbols, and special instructions."),
    ("BOM", "Is there a Bill of Materials (BOM) or parts list? List all components with quantities and part numbers."),
    ("Assembly", "Describe the assembly structure: how many parts, fasteners shown, fit specifications, sub-assemblies."),
]

# ============================================================================
# CELL 5 - EXTRACT FUNCTION
# ============================================================================
from PIL import Image
import fitz

def extract_with_moondream(image_path, model, processor):
    """Extract information using Moondream2."""
    image = Image.open(image_path).convert("RGB")

    # Resize if needed
    if max(image.size) > 1280:
        ratio = 1280 / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    results = {}

    for category, question in EXTRACTION_QUERIES:
        try:
            inputs = processor(
                text=question,
                images=image,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512)

            answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            results[category] = {
                "question": question,
                "answer": answer.strip(),
                "detected": len(answer.strip()) > 10
            }

        except Exception as e:
            results[category] = {
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "detected": False
            }

        gc.collect()
        torch.cuda.empty_cache()

    return results

# ============================================================================
# CELL 6 - PROCESS ALL PDFs
# ============================================================================
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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

    # Convert to image
    doc = fitz.open(pdf_file)
    page = doc[0]
    mat = fitz.Matrix(1.5, 1.5)
    temp_path = "/tmp/moondream_temp.png"
    pix = page.get_pixmap(matrix=mat)
    pix.save(temp_path)
    doc.close()

    # Extract
    extraction = extract_with_moondream(temp_path, model, processor)

    # Create result
    result = {
        "drawing_name": drawing_name,
        "model": "Moondream2",
        "extraction_date": datetime.now().isoformat(),
        "extraction": extraction
    }

    # Save JSON
    json_path = output_dir / f"{drawing_name}_moondream.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"   ✅ JSON saved: {json_path.name}")

    # Create Excel
    rows = []
    for category, data in extraction.items():
        rows.append({
            "Drawing": drawing_name,
            "Category": category,
            "Question": data.get("question", ""),
            "Answer": data.get("answer", ""),
            "Detected": data.get("detected", False),
            "Model": "Moondream2"
        })

    if rows:
        df = pd.DataFrame(rows)
        excel_path = output_dir / f"{drawing_name}_moondream.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"   ✅ Excel saved: {excel_path.name}")

    all_results.append(result)
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# CELL 7 - SAVE COMBINED RESULTS
# ============================================================================
combined_path = output_dir / "ALL_DRAWINGS_moondream.json"
with open(combined_path, 'w', encoding='utf-8') as f:
    json.dump({
        "model": "Moondream2",
        "extraction_date": datetime.now().isoformat(),
        "total_drawings": len(all_results),
        "results": all_results
    }, f, indent=2)

print(f"\n✅ All results saved to: {output_dir}")

# ============================================================================
# CELL 8 - PRINT SUMMARY
# ============================================================================
print("\n" + "="*70)
print("MOONDREAM2 EXTRACTION SUMMARY")
print("="*70)

for result in all_results:
    print(f"\n📄 {result['drawing_name']}")
    detected = sum(1 for cat in result['extraction'].values() if cat.get('detected', False))
    print(f"   Categories detected: {detected}/{len(EXTRACTION_QUERIES)}")

print("\n✅ Moondream2 testing complete!")
print("="*70)