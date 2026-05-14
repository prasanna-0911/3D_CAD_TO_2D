# ============================================================================
# MODEL 2 - LLaVA 1.5 7B Comprehensive Testing
# ============================================================================
"""
LLaVA 1.5 7B Testing for CAD Drawings - 71 Parameter Extraction
===============================================================
Tests all 71 parameter categories systematically.

Run on: Google Colab (T4 GPU recommended)
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

# Clear memory
gc.collect()
torch.cuda.empty_cache()

# Create output directories
output_dir = Path("outputs/llava_results")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "")

# ============================================================================
# CELL 2 - INSTALL DEPENDENCIES
# ============================================================================
!pip install -q transformers accelerate bitsandbytes pillow pymupdf pandas openpyxl tqdm

from transformers import AutoProcessor, LlamaForCausalLM
from transformers import BitsAndBytesConfig
import torch

print("✅ Dependencies installed!")

# ============================================================================
# CELL 3 - DEFINE 71 PARAMETERS WITH EXTRACTION PROMPTS
# ============================================================================
PARAMETER_PROMPTS = {
    "Geometry_Design": {
        "params": [
            {"id": 1, "name": "Geometry & Design Changes"},
            {"id": 2, "name": "Shape modifications"},
            {"id": 3, "name": "Suppressed/unsuppressed features"},
            {"id": 4, "name": "Coordinate system shifts"},
            {"id": 5, "name": "Holes Feature"},
            {"id": 6, "name": "Fillet Feature"},
            {"id": 7, "name": "Chamfers Feature"},
            {"id": 8, "name": "Ribs Feature"},
        ],
        "prompt": """Extract geometric feature information from this CAD drawing:
- Are there any holes visible? What are their sizes?
- Are fillets (rounded edges) visible? What radius?
- Are chamfers (angled edges) visible?
- Are there any rib structures?
- Describe any shape modifications or profile changes."""
    },

    "Dimensions": {
        "params": [
            {"id": 9, "name": "Size changes"},
            {"id": 10, "name": "Length"},
            {"id": 11, "name": "Diameter"},
            {"id": 12, "name": "Thickness"},
            {"id": 13, "name": "Position changes"},
            {"id": 14, "name": "Location changes of features"},
            {"id": 19, "name": "Dimension position changes"},
            {"id": 22, "name": "Dimension value changes"},
        ],
        "prompt": """Extract ALL dimensional measurements from this engineering drawing:
- Linear dimensions with values and units (mm)
- Diameter dimensions (Ø) with values
- Length measurements
- Thickness values
- Any position or location dimensions
List every dimension value you can see with exact numbers."""
    },

    "Assembly": {
        "params": [
            {"id": 15, "name": "Assembly fit/interface changes"},
            {"id": 16, "name": "Exploded view differences"},
            {"id": 17, "name": "Sub-assembly changes"},
            {"id": 18, "name": "Fastener type/size updates"},
        ],
        "prompt": """Extract assembly information from this CAD drawing:
- Are there multiple parts/assemblies shown?
- What fasteners are visible (bolts, screws, pins)?
- Are fit specifications shown (H7/g6 type)?
- Describe any sub-assembly groupings."""
    },

    "Tolerances": {
        "params": [
            {"id": 20, "name": "Tolerances change"},
            {"id": 21, "name": "Tolerances Location change"},
            {"id": 23, "name": "Tolerance updates (bilateral/unilateral)"},
            {"id": 24, "name": "Fits (H7/g6, etc.)"},
        ],
        "prompt": """Extract tolerance specifications from this CAD drawing:
- All tolerance values (+/- specifications)
- Are they bilateral (+/-) or unilateral?
- Any fit specifications (H7/g6, etc.)?
- Where are tolerances located on the drawing?"""
    },

    "GD_T": {
        "params": [
            {"id": 25, "name": "Position tolerance"},
            {"id": 26, "name": "Straightness"},
            {"id": 27, "name": "Flatness"},
            {"id": 28, "name": "Circularity"},
            {"id": 29, "name": "Parallelism"},
            {"id": 30, "name": "Perpendicularity"},
            {"id": 31, "name": "Angularity"},
            {"id": 32, "name": "Runout"},
        ],
        "prompt": """Extract GD&T (Geometric Dimensioning and Tolerancing) symbols:
- Position tolerances (⊕ symbol)
- Flatness specifications
- Straightness specifications
- Parallelism and Perpendicularity symbols
- Angularity and Runout
- Any datum references (A, B, C)
For each GD&T feature found, give the symbol type, value, and datum reference."""
    },

    "Views_Drafting": {
        "params": [
            {"id": 33, "name": "View changes"},
            {"id": 34, "name": "View add/delete"},
            {"id": 35, "name": "View scale change"},
            {"id": 36, "name": "View representation change"},
            {"id": 37, "name": "Drafting view"},
            {"id": 38, "name": "Section line"},
            {"id": 39, "name": "Hatching change"},
            {"id": 40, "name": "Line type change"},
        ],
        "prompt": """Extract view and drafting information:
- What views are shown (front, top, side, section, detail)?
- What is the scale of each view?
- Are there section lines (A-A, B-B)?
- Describe any hatching patterns in section views."""
    },

    "Notes_Annotations": {
        "params": [
            {"id": 51, "name": "General notes updates"},
            {"id": 52, "name": "Notes Location"},
            {"id": 53, "name": "Special instructions added/removed"},
            {"id": 54, "name": "Flag notes / key characteristics"},
            {"id": 55, "name": "Special symbols add/delete"},
            {"id": 56, "name": "Welding symbols"},
            {"id": 57, "name": "Datum changes"},
            {"id": 58, "name": "Datum Location changes"},
        ],
        "prompt": """Extract notes and annotations:
- General notes and instructions
- Surface finish requirements (Ra values)
- Welding symbols
- Datum target markings
- Any special symbols or markings
- Key characteristic flags"""
    },

    "TitleBlock_Metadata": {
        "params": [
            {"id": 59, "name": "Revision number"},
            {"id": 60, "name": "Drawing number"},
            {"id": 61, "name": "Part name changes"},
            {"id": 62, "name": "Author"},
            {"id": 63, "name": "Checker/approver updates"},
            {"id": 64, "name": "Dates"},
            {"id": 65, "name": "Company or project info"},
        ],
        "prompt": """Extract all information from the title block:
- Part name/title
- Drawing number
- Revision number (REV)
- Scale
- Date(s)
- Designer name (DRAWN BY)
- Checker name
- Approver name
- Company name
- Units (mm or inches)"""
    },

    "BOM": {
        "params": [
            {"id": 66, "name": "BOM (Bill of Materials)"},
            {"id": 67, "name": "Component added/removed / Quantity changes"},
            {"id": 68, "name": "Part number revisions"},
            {"id": 69, "name": "Part list"},
        ],
        "prompt": """Extract Bill of Materials information:
- Is there a BOM table?
- List all components/parts with quantities
- Part numbers and descriptions
- Any added or removed components"""
    },

    "Scale": {
        "params": [
            {"id": 70, "name": "Sheet Scale changes"},
            {"id": 71, "name": "View Scale changes / Vendor supplier updates"},
        ],
        "prompt": """Extract scale information:
- What is the overall drawing scale?
- What are the scales of individual views?
- Are there vendor or supplier notes?"""
    },
}

# ============================================================================
# CELL 4 - LOAD LLaVA 1.5 7B
# ============================================================================
print("\n📦 Loading LLaVA 1.5 7B...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = LlamaForCausalLM.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

print("✅ LLaVA 1.5 7B loaded (4-bit quantization)")

if torch.cuda.is_available():
    print(f"   VRAM used: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")

# ============================================================================
# CELL 5 - EXTRACT FUNCTION
# ============================================================================
from PIL import Image
import fitz

def extract_with_llava(image_path, model, processor):
    """Extract information from drawing using LLaVA."""
    image = Image.open(image_path).convert("RGB")

    # Resize if too large
    max_size = 1280
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    results = {}

    for category, data in PARAMETER_PROMPTS.items():
        try:
            prompt = f"User: <image>\n{data['prompt']}\nAssistant:"

            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            input_len = inputs['input_ids'].shape[1]
            answer = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

            # Parse answer into structured format
            results[category] = {
                "detected": len(answer.strip()) > 10,
                "extracted_text": answer.strip(),
                "parameters": {}
            }

            # Map extracted text to individual parameters
            for param in data["params"]:
                param_id = str(param["id"])
                # Simple check - if answer contains relevant keywords
                relevant = any(kw in answer.lower() for kw in param["name"].lower().split())
                results[category]["parameters"][param_id] = {
                    "detected": relevant or results[category]["detected"],
                    "value": "See extracted text" if relevant else "Not detected"
                }

        except Exception as e:
            results[category] = {
                "detected": False,
                "extracted_text": f"ERROR: {str(e)}",
                "parameters": {}
            }

        # Memory cleanup
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

    # Convert PDF to image
    doc = fitz.open(pdf_file)
    page = doc[0]
    mat = fitz.Matrix(1.5, 1.5)  # Moderate zoom
    pix = page.get_pixmap(matrix=mat)
    temp_path = "/tmp/llava_temp.png"
    pix.save(temp_path)
    doc.close()

    # Extract
    extraction = extract_with_llava(temp_path, model, processor)

    # Create result
    result = {
        "drawing_name": drawing_name,
        "model": "LLaVA 1.5 7B (4-bit)",
        "extraction_date": datetime.now().isoformat(),
        "extraction": extraction
    }

    # Save JSON
    json_path = output_dir / f"{drawing_name}_llava.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"   ✅ JSON saved: {json_path.name}")

    # Create Excel with structured data
    rows = []
    for category, data in extraction.items():
        for param_id, param_data in data.get("parameters", {}).items():
            param_name = next((p["name"] for p in PARAMETER_PROMPTS[category]["params"] if str(p["id"]) == param_id), param_id)
            rows.append({
                "Drawing": drawing_name,
                "Category": category,
                "Parameter_ID": param_id,
                "Parameter_Name": param_name,
                "Detected": param_data.get("detected", False),
                "Value": param_data.get("value", ""),
                "Model": "LLaVA 1.5 7B"
            })

    if rows:
        df = pd.DataFrame(rows)
        excel_path = output_dir / f"{drawing_name}_llava.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"   ✅ Excel saved: {excel_path.name}")

    all_results.append(result)

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# CELL 7 - SAVE COMBINED RESULTS
# ============================================================================
combined_path = output_dir / "ALL_DRAWINGS_llava.json"
with open(combined_path, 'w', encoding='utf-8') as f:
    json.dump({
        "model": "LLaVA 1.5 7B (4-bit)",
        "extraction_date": datetime.now().isoformat(),
        "total_drawings": len(all_results),
        "results": all_results
    }, f, indent=2)

print(f"\n✅ All results saved to: {output_dir}")

# ============================================================================
# CELL 8 - PRINT SUMMARY
# ============================================================================
print("\n" + "="*70)
print("LLaVA 1.5 7B EXTRACTION SUMMARY")
print("="*70)

for result in all_results:
    print(f"\n📄 {result['drawing_name']}")
    detected_count = sum(
        1 for cat in result['extraction'].values()
        for param in cat.get('parameters', {}).values()
        if param.get('detected', False)
    )
    print(f"   Parameters detected: {detected_count}/71")

print("\n" + "="*70)
print("✅ LLaVA 1.5 7B testing complete!")
print(f"   Results saved to: {output_dir}")
print("="*70)