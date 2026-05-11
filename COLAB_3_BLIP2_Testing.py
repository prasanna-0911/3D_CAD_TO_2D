# ============================================================================
# MODEL 3 - BLIP2 OPT 2.7B Comprehensive Testing
# ============================================================================
"""
BLIP2 OPT 2.7B Testing for CAD Drawings - 71 Parameter Extraction
===============================================================
Fixed decoding version (slices input tokens from output).

Run on: Google Colab (T4 GPU)
Output: JSON + Excel saved to Google Drive
"""

# ============================================================================
# CELL 1 - MOUNT DRIVE & CLEAR MEMORY
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

import os
import gc
import torch

os.chdir("/content/drive/MyDrive/3D_CAD_TO_2D")

gc.collect()
torch.cuda.empty_cache()

output_dir = Path("outputs/blip2_results")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CELL 2 - INSTALL DEPENDENCIES
# ============================================================================
!pip install -q transformers pillow pymupdf pandas openpyxl tqdm

from transformers import Blip2Processor, Blip2ForConditionalGeneration

print("✅ Dependencies installed!")

# ============================================================================
# CELL 3 - DEFINE PARAMETERS (simplified for BLIP2's short Q&A style)
# ============================================================================
BLIP2_QUESTIONS = {
    "Geometry_Design": "What geometric features are visible? Are there holes, fillets, chamfers, or ribs?",
    "Dimensions": "What are all the dimension values shown? List every number you can see with mm units.",
    "Tolerances": "What tolerance specifications are shown? Look for +/- values and fits like H7/g6.",
    "GD_T": "What GD&T symbols are present? Look for position, flatness, perpendicularity symbols.",
    "Views_Drafting": "What views are shown? Is there a front view, top view, section view? What scales?",
    "Notes_Annotations": "What notes or annotations are visible? Look for surface finish, welding symbols.",
    "TitleBlock_Metadata": "What is in the title block? Give part name, drawing number, revision, scale, date.",
    "BOM": "Is there a bill of materials or parts list? List the components shown.",
    "Scale": "What is the scale of the drawing and views?"
}

# ============================================================================
# CELL 4 - LOAD BLIP2
# ============================================================================
print("\n📦 Loading BLIP2 OPT 2.7B...")

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

print("✅ BLIP2 OPT 2.7B loaded!")

if torch.cuda.is_available():
    print(f"   VRAM used: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")

# ============================================================================
# CELL 5 - EXTRACT FUNCTION
# ============================================================================
from PIL import Image
import fitz

def extract_with_blip2(image_path, model, processor):
    """Extract information using BLIP2 with SHORT questions."""
    image = Image.open(image_path).convert("RGB")

    # Resize for memory
    if max(image.size) > 1024:
        ratio = 1024 / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    results = {}

    for category, question in BLIP2_QUESTIONS.items():
        try:
            inputs = processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to("cuda", torch.float16)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=3
                )

            # ✅ FIXED DECODING: Slice off input tokens
            input_len = inputs["input_ids"].shape[1]
            answer = processor.batch_decode(
                generated_ids[:, input_len:],
                skip_special_tokens=True
            )[0].strip()

            # Check if answer is meaningful (not just prompt echo)
            detected = len(answer) > 5 and answer.lower() != question.lower()

            results[category] = {
                "question": question,
                "answer": answer,
                "detected": detected
            }

        except Exception as e:
            results[category] = {
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "detected": False
            }

        # Memory cleanup
        del inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

    return results

# ============================================================================
# CELL 6 - PROCESS ALL PDFs
# ============================================================================
import pandas as pd
from pathlib import Path
from datetime import datetime

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
    mat = fitz.Matrix(1.2, 1.2)
    temp_path = "/tmp/blip2_temp.png"
    pix.save(temp_path)
    doc.close()

    # Extract
    extraction = extract_with_blip2(temp_path, model, processor)

    # Create result
    result = {
        "drawing_name": drawing_name,
        "model": "BLIP2 OPT 2.7B",
        "extraction_date": datetime.now().isoformat(),
        "extraction": extraction
    }

    # Save JSON
    json_path = output_dir / f"{drawing_name}_blip2.json"
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
            "Model": "BLIP2 OPT 2.7B"
        })

    if rows:
        df = pd.DataFrame(rows)
        excel_path = output_dir / f"{drawing_name}_blip2.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"   ✅ Excel saved: {excel_path.name}")

    all_results.append(result)
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# CELL 7 - SAVE COMBINED RESULTS
# ============================================================================
combined_path = output_dir / "ALL_DRAWINGS_blip2.json"
with open(combined_path, 'w', encoding='utf-8') as f:
    json.dump({
        "model": "BLIP2 OPT 2.7B",
        "extraction_date": datetime.now().isoformat(),
        "total_drawings": len(all_results),
        "results": all_results
    }, f, indent=2)

print(f"\n✅ All results saved to: {output_dir}")

# ============================================================================
# CELL 8 - PRINT SUMMARY
# ============================================================================
print("\n" + "="*70)
print("BLIP2 OPT 2.7B EXTRACTION SUMMARY")
print("="*70)

for result in all_results:
    print(f"\n📄 {result['drawing_name']}")
    detected = sum(1 for cat in result['extraction'].values() if cat.get('detected', False))
    print(f"   Categories detected: {detected}/{len(BLIP2_QUESTIONS)}")

print("\n" + "="*70)
print("✅ BLIP2 testing complete!")
print("="*70)