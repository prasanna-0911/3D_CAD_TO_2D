"""
4_test_moondream.py - Moondream2 Parameter Extraction
=======================================================
Extracts 71 parameters using Moondream2 from the target PDF.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts_71params_extracting.config import (
    INPUT_DIR, OUTPUT_DIR, TARGET_PDF,
    PARAMETERS_71, ALL_PARAMS_71, MODELS_CONFIG
)
from scripts_71params_extracting.utils import (
    pdf_to_image, save_json, save_excel, print_header, free_gpu_memory, 
    get_gpu_memory_info, ensure_dir
)

MAX_IMAGE_SIZE = 1280
PDF_ZOOM = 1.2


def check_gpu_and_load():
    """Check GPU and load Moondream2 model."""
    import torch
    
    gpu_info = get_gpu_memory_info()
    
    if gpu_info["mode"] == "CPU":
        print("   ⚠️ Running on CPU - will be slow")
    else:
        print(f"   GPU: {gpu_info['device']} ({gpu_info['free_gb']}GB free)")
    
    if not get_gpu_memory_info()["available"] or gpu_info["free_gb"] < 3:
        print(f"   ⏭️ Skipping - insufficient GPU memory (need ~3GB)")
        return None, None
    
    from transformers import AutoModelForVision2Seq, AutoProcessor
    
    model_config = MODELS_CONFIG["moondream2"]
    print(f"   Loading {model_config['name']}...")
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_config["model_id"],
        revision="2024-08-26",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_config["model_id"])
    
    print("   ✅ Moondream2 loaded")
    
    return model, processor


def extract_with_moondream(model, processor, image):
    """Extract information using Moondream2."""
    questions = [
        "List all dimension values with units",
        "What GD&T symbols are present?",
        "What is in the title block?",
        "What views are shown?",
        "What notes are visible?"
    ]
    
    results = {}
    
    for i, question in enumerate(questions):
        try:
            inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=256)
            answer = processor.decode(output[0], skip_special_tokens=True)
            results[f"q{i+1}"] = answer
        except Exception as e:
            results[f"q{i+1}"] = f"ERROR: {str(e)}"
        
        free_gpu_memory()
    
    return results


def detect_71_params_vlm(extraction: dict) -> dict:
    """Detect 71 parameters from VLM extraction."""
    all_text = " ".join(extraction.values()).lower()
    
    detected = {}
    
    for param_id, param_info in ALL_PARAMS_71.items():
        is_detected = False
        param_lower = param_info["name"].lower()
        
        if param_id in ["5"]:
            is_detected = "hole" in all_text
        elif param_id in ["6"]:
            is_detected = "fillet" in all_text or "radius" in all_text
        elif param_id in ["10", "11"]:
            is_detected = "dimension" in all_text or "mm" in all_text
        elif param_id in ["25", "26", "27", "28", "29", "30", "31", "32"]:
            gdt_kw = ["position", "straightness", "flatness", "parallelism", "perpendicular", "angularity", "runout"]
            is_detected = any(kw in all_text for kw in gdt_kw)
        elif param_id in ["33", "34", "35", "36", "45"]:
            view_kw = ["front", "top", "side", "section", "view", "scale"]
            is_detected = any(kw in all_text for kw in view_kw)
        elif param_id in ["51", "52", "53", "54"]:
            is_detected = "note" in all_text or "warning" in all_text
        elif param_id in ["59", "60", "61", "62", "63", "64", "65"]:
            title_kw = ["rev", "drawing", "part", "date", "scale"]
            is_detected = any(kw in all_text for kw in title_kw)
        elif param_id in ["70", "71"]:
            is_detected = "scale" in all_text
        else:
            keywords = [w for w in param_lower.split() if len(w) > 3]
            is_detected = any(kw in all_text for kw in keywords)
        
        detected[param_id] = {
            "id": param_id,
            "name": param_info["name"],
            "detected": is_detected,
            "category": next(cat for cat, params in PARAMETERS_71.items() 
                          for p in params if p["id"] == param_id)
        }
    
    return detected


def detect_gt_params_from_vlm(extraction: dict) -> dict:
    """Detect ground truth parameters from VLM extraction."""
    import re
    
    all_text = " ".join(extraction.values())
    all_text_lower = all_text.lower()
    
    detected_gt = {}
    
    view_match = re.search(r'(\d+)\s*views?', all_text_lower)
    if view_match:
        detected_value = int(view_match.group(1))
        detected_gt["GT_10"] = {"name": "No_of_Views", "detected_value": detected_value, "ground_truth_value": 6, "match": detected_value == 6}
    
    scale_match = re.search(r'scale\s*(\d+)\s*[:/]\s*(\d+)', all_text_lower)
    if scale_match:
        numerator = int(scale_match.group(1))
        denominator = int(scale_match.group(2))
        detected_gt["GT_3"] = {"name": "Sheet_ScaleNumerator", "detected_value": numerator, "ground_truth_value": 1, "match": numerator == 1}
        detected_gt["GT_4"] = {"name": "Sheet_ScaleDenominator", "detected_value": denominator, "ground_truth_value": 1, "match": denominator == 1}
    
    view_scale_matches = re.findall(r'scale\s*(\d+)\s*[:/]\s*(\d+)', all_text_lower)
    for num, denom in view_scale_matches:
        if int(denom) == 2:
            detected_gt["GT_24"] = {"name": "View_D2_ScaleDenominator", "detected_value": int(denom), "ground_truth_value": 2, "match": True}
            detected_gt["GT_27"] = {"name": "View_D5_ScaleDenominator", "detected_value": int(denom), "ground_truth_value": 2, "match": True}
    
    return detected_gt


def save_results(drawing_name: str, extraction: dict, detected_71: dict, detected_gt: dict, output_dir: Path):
    """Save extraction results."""
    ensure_dir(output_dir)
    
    total_detected = sum(1 for p in detected_71.values() if p["detected"])
    total = len(detected_71)
    
    gt_matches = sum(1 for g in detected_gt.values() if g.get("match") == True)
    gt_total = len(detected_gt)
    
    summary = {
        "drawing_name": drawing_name,
        "model": "Moondream2",
        "extraction_date": datetime.now().isoformat(),
        "params_71_detected": total_detected,
        "params_71_total": total,
        "params_71_detection_rate": round((total_detected / total) * 100, 2),
        "gt_params_detected": gt_total,
        "gt_params_matched": gt_matches,
    }
    
    json_data = {**summary, "extraction": extraction, "params_71": detected_71, "gt_params": detected_gt}
    json_path = output_dir / f"{drawing_name}_moondream_71params.json"
    save_json(json_data, json_path)
    
    excel_data = [{"Parameter_ID": p["id"], "Parameter_Name": p["name"], 
                   "Category": p["category"], "Detected": "Yes" if p["detected"] else "No"}
                  for p in detected_71.values()]
    excel_path = output_dir / f"{drawing_name}_moondream_71params.xlsx"
    save_excel(excel_data, excel_path, "71 Parameters")
    
    return summary


def process_target_pdf():
    """Process the target PDF with Moondream2."""
    print_header("Moondream2 - 71 Parameters Extraction")
    
    pdf_path = INPUT_DIR / TARGET_PDF
    
    if not pdf_path.exists():
        print(f"\n❌ PDF not found: {pdf_path}")
        return
    
    print(f"\n📄 Processing: {pdf_path.name}")
    
    print("\n📦 Loading Moondream2 model...")
    model, processor = check_gpu_and_load()
    
    if model is None:
        print("\n⚠️ Skipping Moondream2 due to insufficient GPU memory")
        return
    
    output_dir = OUTPUT_DIR / "moondream_results"
    ensure_dir(output_dir)
    
    print("\n📄 Converting PDF to image...")
    image = pdf_to_image(pdf_path, zoom=PDF_ZOOM, max_size=MAX_IMAGE_SIZE)
    
    print(f"   Extracting with Moondream2...")
    extraction = extract_with_moondream(model, processor, image)
    
    print(f"   Detecting 71 parameters...")
    detected_71 = detect_71_params_vlm(extraction)
    
    print(f"   Detecting ground truth parameters...")
    detected_gt = detect_gt_params_from_vlm(extraction)
    
    drawing_name = pdf_path.stem
    summary = save_results(drawing_name, extraction, detected_71, detected_gt, output_dir)
    
    free_gpu_memory()
    
    print("\n" + "=" * 60)
    print("📊 Moondream2 Results Summary")
    print("=" * 60)
    print(f"   71 params detected: {summary['params_71_detected']}/{summary['params_71_total']} ({summary['params_71_detection_rate']}%)")
    print("\n✅ Moondream2 extraction complete!")
    
    return summary


if __name__ == "__main__":
    try:
        process_target_pdf()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()