"""
CAD VLM 71-Parameter Evaluation - Local PC Version
===================================================
Tests multiple VLMs against all 71 parameters from engineering drawings.

Models tested:
1. EasyOCR (baseline)
2. BLIP2 OPT 2.7B
3. LLaVA 1.5 7B (4-bit) - if VRAM allows
4. Qwen2-VL 7B (8-bit) - if VRAM allows

Run: python LOCAL_71Param_Test.py
"""

import os
import json
import gc
import torch
import re
from pathlib import Path
from datetime import datetime
from PIL import Image
import pandas as pd
import fitz

# ============================================================================
# CONFIGURATION
# ============================================================================
WORKING_DIR = Path(r"C:\Users\ADMIN\Downloads\3D_CAD_TO_2D")  # Change this!
INPUT_DIR = WORKING_DIR / "input_pdfs"
OUTPUT_DIR = WORKING_DIR / "outputs"

# Create output directories
for subdir in ["easyocr_results", "llava_results", "blip2_results", "qwen_results", "comparison_reports"]:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

os.chdir(WORKING_DIR)

# ============================================================================
# ALL 71 PARAMETERS DEFINITION
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

# Flatten parameters for easy access
ALL_PARAMS = {p["id"]: p for cat in PARAMETERS.values() for p in cat}
TOTAL_PARAMS = sum(len(v) for v in PARAMETERS.values())

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def pdf_to_image(pdf_path, dpi=150):
    """Convert PDF page to image"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return img

import io

def check_parameter_detection(text, param_name):
    """Check if a parameter's keywords appear in extracted text"""
    keywords = param_name.lower().split()
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords if len(kw) > 3)

def detect_parameters_in_text(extracted_text, model_name):
    """Detect which of the 71 parameters are present in extracted text"""
    detected = {}

    for param_id, param_info in ALL_PARAMS.items():
        name = param_info["name"]
        detected[param_id] = {
            "name": name,
            "detected": check_parameter_detection(extracted_text, name),
            "category": next(cat for cat, params in PARAMETERS.items()
                          for p in params if p["id"] == param_id)
        }

    return detected

# ============================================================================
# 1. EASYOCR TESTING
# ============================================================================
def test_easyocr():
    print("\n" + "="*70)
    print("1. TESTING EASYOCR (Baseline)")
    print("="*70)

    import easyocr

    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    results = []

    for pdf_file in INPUT_DIR.glob("*.pdf"):
        print(f"\n[PDF] Processing: {pdf_file.name}")

        # Convert PDF to image
        doc = fitz.open(pdf_file)
        page = doc[0]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        temp_path = OUTPUT_DIR / f"temp_{pdf_file.stem}.png"
        pix.save(str(temp_path))
        doc.close()

        # Run OCR
        ocr_results = reader.readtext(str(temp_path), detail=1)

        # Extract text
        elements = []
        for bbox, text, confidence in ocr_results:
            elements.append({
                "text": text.strip(),
                "confidence": round(confidence, 3)
            })

        # Combine all text for parameter detection
        all_text = " ".join([e["text"] for e in elements])
        detected = detect_parameters_in_text(all_text, "EasyOCR")

        result = {
            "drawing_name": pdf_file.stem,
            "model": "EasyOCR",
            "extraction_date": datetime.now().isoformat(),
            "total_elements": len(elements),
            "elements": elements,
            "parameters_detected": {pid: d["detected"] for pid, d in detected.items()}
        }

        # Save JSON
        json_path = OUTPUT_DIR / "easyocr_results" / f"{pdf_file.stem}_easyocr.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Save Excel
        df = pd.DataFrame(elements)
        excel_path = OUTPUT_DIR / "easyocr_results" / f"{pdf_file.stem}_easyocr.xlsx"
        df.to_excel(excel_path, index=False)

        detected_count = sum(1 for v in detected.values() if v["detected"])
        print(f"   Elements: {len(elements)}, Parameters detected: {detected_count}/{TOTAL_PARAMS}")

        results.append(result)

        # Cleanup
        temp_path.unlink(missing_ok=True)

    return results

# ============================================================================
# 2. BLIP2 TESTING
# ============================================================================
def test_blip2():
    print("\n" + "="*70)
    print("2. TESTING BLIP2 OPT 2.7B")
    print("="*70)

    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    questions = {
        "Dimensions": "What are all the dimension values shown? List every number with mm units.",
        "GD_T": "What GD&T symbols are present? Look for position, flatness, perpendicularity.",
        "TitleBlock": "What is in the title block? Give part name, drawing number, revision, scale.",
        "Notes": "What notes or annotations are visible?",
        "BOM": "Is there a bill of materials? List the components.",
        "Geometry": "What geometric features are visible? Holes, fillets, chamfers?",
        "Tolerances": "What tolerance specifications are shown?",
        "Views": "What views are shown? Front, top, side, section?",
        "Assembly": "What fasteners are visible?",
        "Scale": "What is the scale of the drawing?"
    }

    results = []

    for pdf_file in INPUT_DIR.glob("*.pdf"):
        print(f"\n[PDF] Processing: {pdf_file.name}")

        images = pdf_to_image(pdf_file, dpi=100)

        extracted_answers = {}
        for category, question in questions.items():
            try:
                inputs = processor(images=images, text=question, return_tensors="pt").to("cuda")
                output = model.generate(**inputs, max_new_tokens=256)
                answer = processor.decode(output[0], skip_special_tokens=True)
                extracted_answers[category] = answer
            except Exception as e:
                extracted_answers[category] = f"ERROR: {str(e)}"

        all_text = " ".join(extracted_answers.values())
        detected = detect_parameters_in_text(all_text, "BLIP2")

        result = {
            "drawing_name": pdf_file.stem,
            "model": "BLIP2 OPT 2.7B",
            "extraction_date": datetime.now().isoformat(),
            "extracted_answers": extracted_answers,
            "parameters_detected": {pid: d["detected"] for pid, d in detected.items()}
        }

        json_path = OUTPUT_DIR / "blip2_results" / f"{pdf_file.stem}_blip2.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        detected_count = sum(1 for v in detected.values() if v["detected"])
        print(f"   Parameters detected: {detected_count}/{TOTAL_PARAMS}")

        results.append(result)
        gc.collect()
        torch.cuda.empty_cache()

    return results

# ============================================================================
# 3. LLaVA TESTING (4-bit)
# ============================================================================
def test_llava():
    print("\n" + "="*70)
    print("3. TESTING LLaVA 1.5 7B (4-bit)")
    print("="*70)

    from transformers import pipeline, BitsAndBytesConfig

    pipe = pipeline(
        "image-text-to-text",
        model="llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompt = """Extract ALL information from this engineering drawing:
    - Dimensions (length, diameter, radius, thickness)
    - GD&T symbols (position, flatness, perpendicularity, etc.)
    - Title block (part name, drawing number, revision, scale)
    - Geometric features (holes, fillets, chamfers)
    - Tolerances and fits
    - Notes and annotations
    - Bill of Materials
    - Views and scales

    List everything you can see with exact values."""

    results = []

    for pdf_file in INPUT_DIR.glob("*.pdf"):
        print(f"\n[PDF] Processing: {pdf_file.name}")

        images = pdf_to_image(pdf_file, dpi=100)

        try:
            output = pipe(images, text=prompt, max_new_tokens=512)
            answer = output[0]['generated_text']
        except Exception as e:
            answer = f"ERROR: {str(e)}"

        detected = detect_parameters_in_text(answer, "LLaVA")

        result = {
            "drawing_name": pdf_file.stem,
            "model": "LLaVA 1.5 7B (4-bit)",
            "extraction_date": datetime.now().isoformat(),
            "extracted_text": answer,
            "parameters_detected": {pid: d["detected"] for pid, d in detected.items()}
        }

        json_path = OUTPUT_DIR / "llava_results" / f"{pdf_file.stem}_llava.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        detected_count = sum(1 for v in detected.values() if v["detected"])
        print(f"   Parameters detected: {detected_count}/{TOTAL_PARAMS}")

        results.append(result)
        gc.collect()
        torch.cuda.empty_cache()

    return results

# ============================================================================
# 4. QWEN2-VL TESTING (8-bit)
# ============================================================================
def test_qwen2vl():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

    if vram_gb < 16:
        print(f"\n[WARNING] Qwen2-VL requires ~16GB VRAM. You have {vram_gb:.1f}GB. Skipping...")
        return []

    print("\n" + "="*70)
    print("4. TESTING Qwen2-VL 7B (8-bit)")
    print("="*70)

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    prompt = """Extract ALL information from this engineering drawing:
    - Dimensions with exact values
    - GD&T symbols and their values
    - Title block information
    - Geometric features
    - Tolerances
    - Notes and BOM"""

    results = []

    for pdf_file in INPUT_DIR.glob("*.pdf"):
        print(f"\n[PDF] Processing: {pdf_file.name}")

        images = pdf_to_image(pdf_file, dpi=100)

        try:
            inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=512)
            answer = processor.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            answer = f"ERROR: {str(e)}"

        detected = detect_parameters_in_text(answer, "Qwen2-VL")

        result = {
            "drawing_name": pdf_file.stem,
            "model": "Qwen2-VL 7B (8-bit)",
            "extraction_date": datetime.now().isoformat(),
            "extracted_text": answer,
            "parameters_detected": {pid: d["detected"] for pid, d in detected.items()}
        }

        json_path = OUTPUT_DIR / "qwen_results" / f"{pdf_file.stem}_qwen.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        detected_count = sum(1 for v in detected.values() if v["detected"])
        print(f"   Parameters detected: {detected_count}/{TOTAL_PARAMS}")

        results.append(result)
        gc.collect()
        torch.cuda.empty_cache()

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("CAD VLM 71-PARAMETER EVALUATION - LOCAL PC")
    print("="*70)
    print(f"\nWorking directory: {WORKING_DIR}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total parameters: {TOTAL_PARAMS}")

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Get list of PDF files
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    print(f"\nPDF files found: {len(pdf_files)}")
    for pf in pdf_files:
        print(f"  - {pf.name}")

    all_results = {}

    # Run tests
    try:
        all_results["easyocr"] = test_easyocr()
    except Exception as e:
        print(f"[ERROR] EasyOCR failed: {e}")

    try:
        all_results["blip2"] = test_blip2()
    except Exception as e:
        print(f"[ERROR] BLIP2 failed: {e}")

    try:
        all_results["llava"] = test_llava()
    except Exception as e:
        print(f"[ERROR] LLaVA failed: {e}")

    try:
        all_results["qwen"] = test_qwen2vl()
    except Exception as e:
        print(f"[ERROR] Qwen2-VL failed: {e}")

    # =========================================================================
    # COMPARISON REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("CREATING COMPARISON REPORT")
    print("="*70)

    # Build comparison matrix
    comparison_data = []

    for model_name, results in all_results.items():
        for result in results:
            drawing = result["drawing_name"]
            params_detected = result.get("parameters_detected", {})
            detected_count = sum(1 for v in params_detected.values() if v)

            # Per-category counts
            for category, params in PARAMETERS.items():
                cat_detected = sum(1 for p in params if params_detected.get(p["id"], False))

                comparison_data.append({
                    "Model": model_name.upper(),
                    "Drawing": drawing,
                    "Category": category,
                    "Parameters_Detected": cat_detected,
                    "Total_in_Category": len(params)
                })

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)

        # Summary by model
        summary = df_comparison.groupby("Model").agg({
            "Parameters_Detected": "sum",
            "Total_in_Category": "sum"
        }).reset_index()
        summary["Detection_Rate_%"] = (summary["Parameters_Detected"] / summary["Total_in_Category"] * 100).round(1)

        print("\n[STATS] SUMMARY BY MODEL:")
        print(summary.to_string(index=False))

        # Save comparison report
        excel_path = OUTPUT_DIR / "comparison_reports" / f"71PARAM_COMPARISON_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            df_comparison.to_excel(writer, sheet_name="Detailed", index=False)

        print(f"\n[OK] Comparison report saved: {excel_path}")

    print("\n" + "="*70)
    print("[OK] ALL TESTING COMPLETE!")
    print("="*70)