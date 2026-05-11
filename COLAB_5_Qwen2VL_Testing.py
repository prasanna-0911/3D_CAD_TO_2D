# ============================================================================
# MODEL 5 - Qwen2-VL 7B Comprehensive Testing
# ============================================================================
"""
Qwen2-VL 7B Testing for CAD Drawings - 71 Parameter Extraction
===============================================================
Powerful VLM optimized for technical documents. Uses 8-bit quantization.

Run on: Google Colab (T4 GPU with 16GB VRAM recommended)
Output: JSON + Excel saved to Google Drive

⚠️ NOTE: If OOM errors, use 4-bit quantization or skip this model.
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

# Enable memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

output_dir = Path("outputs/qwen2vl_results")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# ============================================================================
# CELL 2 - INSTALL DEPENDENCIES
# ============================================================================
!pip install -q transformers accelerate bitsandbytes qwen-vl-utils pillow pymupdf pandas openpyxl tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

print("✅ Dependencies installed!")

# ============================================================================
# CELL 3 - LOAD QWEN2-VL (8-bit)
# ============================================================================
print("\n📦 Loading Qwen2-VL 7B (8-bit quantization)...")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True
    )

    print("✅ Qwen2-VL 7B loaded (8-bit)")

    if torch.cuda.is_available():
        print(f"   VRAM used: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")

except Exception as e:
    print(f"❌ Failed to load: {e}")
    print("\n⚠️ Try running with fewer drawings or use 4-bit quantization.")
    raise

# ============================================================================
# CELL 4 - DEFINE EXTRACTION PROMPTS
# ============================================================================
EXTRACTION_PROMPTS = [
    ("Dimensions", """Extract ALL dimension values from this CAD drawing:
- Linear dimensions: exact measurements with mm
- Diameter dimensions: Ø values like Ø25, Ø100.8
- Radius dimensions: R values like R5, R12.5
- Tolerances: +/- values like +0.1/-0.1
- Bilateral/unilateral tolerance specifications

List every number you can read exactly as shown."""),

    ("GD_T", """Identify all GD&T (Geometric Dimensioning and Tolerancing) symbols:
- Position tolerance (⊕) with value and datum reference
- Flatness, Straightness, Circularity with values
- Parallelism, Perpendicularity, Angularity with values
- Runout specifications
- Datum references (A, B, C in boxes)

Format: Symbol name | Value | Datum Reference"""),

    ("Views", """Analyze all views in this engineering drawing:
- Front view: present or not
- Top view: present or not
- Side view: present or not
- Section views (A-A, B-B, etc.): present and location
- Detail views: scale and location
- Isometric/3D view: present or not
- Each view's scale (SCALE 1:1, SCALE 2:1, etc.)"""),

    ("Geometry", """Describe geometric features visible:
- Holes: sizes (Ø), depths, locations
- Fillet radii: values like R3, R5
- Chamfers: values like 1x45°
- Ribs: thickness and location
- Pockets: dimensions
- bosses: dimensions
Describe any feature modifications or additions."""),

    ("TitleBlock", """Extract complete title block information:
- Part name/Title
- Drawing number
- Revision (REV): number and letter
- Scale: overall drawing scale
- Date: creation/modification dates
- Drawn by: designer name
- Checked by: checker name
- Approved by: approver name
- Company name
- Units (mm or inches)"""),

    ("Notes_Annotations", """Find all notes and annotations:
- General notes at top or bottom
- Surface finish requirements (Ra values)
- Welding symbols and specifications
- Heat treatment requirements
- Special instructions
- Flagged key characteristics
- Datum target specifications"""),

    ("Assembly", """Analyze assembly information:
- Number of parts in assembly
- Fasteners: bolts, screws, pins with sizes
- Fit specifications: H7/g6 type fits
- Sub-assembly groupings
- Interface dimensions between parts"""),

    ("BOM", """Extract Bill of Materials:
- Is BOM table present?
- Table headers (PC NO, PART NAME, QTY, etc.)
- List all components with:
  * Item number
  * Part name
  * Quantity
  * Part number
  * Material (if shown)"""),

    ("Tolerances", """Extract tolerance information:
- General tolerances shown
- Specific tolerances for dimensions
- Bilateral tolerances (+/- symmetric)
- Unilateral tolerances (+ only or - only)
- Fit tolerances (clearance, interference, transition)
- Angular tolerances in degrees"""),

    ("Scale", """Extract scale information:
- Overall drawing scale (e.g., SCALE 1:2, SCALE 1:5)
- Individual view scales
- Any changed scales (annotated with new values)
- Vendor/supplier scale notes"""),
]

# ============================================================================
# CELL 5 - EXTRACT FUNCTION
# ============================================================================
from PIL import Image
import fitz

def extract_with_qwen2vl(image_path, model, processor):
    """Extract information using Qwen2-VL."""
    # Load and resize image
    image = Image.open(image_path).convert("RGB")

    max_dim = 1400
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Save for processing
    temp_path = "/tmp/qwen2vl_temp.png"
    image.save(temp_path)

    results = {}

    for category, prompt in EXTRACTION_PROMPTS:
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_path},
                    {"type": "text", "text": prompt}
                ]
            }]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False
                )

            output_text = processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            results[category] = {
                "prompt": prompt,
                "answer": output_text.strip(),
                "detected": len(output_text.strip()) > 20
            }

        except Exception as e:
            results[category] = {
                "prompt": prompt,
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
    temp_path = "/tmp/qwen2vl_temp.png"
    pix = page.get_pixmap(matrix=mat)
    pix.save(temp_path)
    doc.close()

    # Extract
    extraction = extract_with_qwen2vl(temp_path, model, processor)

    # Create result
    result = {
        "drawing_name": drawing_name,
        "model": "Qwen2-VL 7B (8-bit)",
        "extraction_date": datetime.now().isoformat(),
        "extraction": extraction
    }

    # Save JSON
    json_path = output_dir / f"{drawing_name}_qwen2vl.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"   ✅ JSON saved: {json_path.name}")

    # Create Excel
    rows = []
    for category, data in extraction.items():
        rows.append({
            "Drawing": drawing_name,
            "Category": category,
            "Answer": data.get("answer", ""),
            "Characters": len(data.get("answer", "")),
            "Detected": data.get("detected", False),
            "Model": "Qwen2-VL 7B"
        })

    if rows:
        df = pd.DataFrame(rows)
        excel_path = output_dir / f"{drawing_name}_qwen2vl.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"   ✅ Excel saved: {excel_path.name}")

    all_results.append(result)
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# CELL 7 - SAVE COMBINED RESULTS
# ============================================================================
combined_path = output_dir / "ALL_DRAWINGS_qwen2vl.json"
with open(combined_path, 'w', encoding='utf-8') as f:
    json.dump({
        "model": "Qwen2-VL 7B (8-bit)",
        "extraction_date": datetime.now().isoformat(),
        "total_drawings": len(all_results),
        "results": all_results
    }, f, indent=2)

print(f"\n✅ All results saved to: {output_dir}")

# ============================================================================
# CELL 8 - PRINT SUMMARY
# ============================================================================
print("\n" + "="*70)
print("Qwen2-VL 7B EXTRACTION SUMMARY")
print("="*70)

for result in all_results:
    print(f"\n📄 {result['drawing_name']}")
    detected = sum(1 for cat in result['extraction'].values() if cat.get('detected', False))
    total_chars = sum(len(cat.get('answer', '')) for cat in result['extraction'].values())
    print(f"   Categories detected: {detected}/{len(EXTRACTION_PROMPTS)}")
    print(f"   Total characters: {total_chars}")

print("\n✅ Qwen2-VL testing complete!")
print("="*70)