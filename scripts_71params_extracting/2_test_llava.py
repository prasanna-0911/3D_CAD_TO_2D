"""
2_test_llava.py - LLaVA 1.5 7B Parameter Extraction
====================================================
Extracts 71 parameters using LLaVA 1.5 7B from the target PDF.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts_71params_extracting.config import (
    INPUT_DIR, OUTPUT_DIR, TARGET_PDF,
    PARAMETERS_71, ALL_PARAMS_71, MODELS_CONFIG, VLM_PROMPTS
)
from scripts_71params_extracting.utils import (
    pdf_to_image, save_json, save_excel, print_header, print_model_status,
    detect_params_71, free_gpu_memory, get_gpu_memory_info, ensure_dir
)

MAX_IMAGE_SIZE = 1280
PDF_ZOOM = 1.0


def check_gpu_and_load():
    """Check GPU and load LLaVA model."""
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from transformers import BitsAndBytesConfig
    
    gpu_info = get_gpu_memory_info()
    
    if gpu_info["mode"] == "CPU":
        print("   [WARNING] Running on CPU - will be slow")
    else:
        print(f"   GPU: {gpu_info['device']} ({gpu_info['free_gb']}GB free)")
    
    model_config = MODELS_CONFIG["llava_1.5_7b"]
    
    if not get_gpu_memory_info()["available"] or gpu_info["free_gb"] < 8:
        print(f"   [SKIP] Skipping - insufficient GPU memory (need ~8GB)")
        return None, None
    
    print(f"   Loading {model_config['name']}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_config["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    processor = AutoProcessor.from_pretrained(model_config["model_id"])
    
    print("   [OK] LLaVA 1.5 7B loaded (4-bit)")
    
    return model, processor


def extract_with_llava(model, processor, image_path: str):
    """Extract information using LLaVA."""
    import torch
    from PIL import Image
    
    image = Image.open(image_path).convert("RGB")
    
    if max(image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    questions = [
        ("dimensions", "What are all the dimension values visible in this engineering drawing? List every number with units (mm)."),
        ("gd_t", "What GD&T symbols are present? List position, flatness, perpendicularity, etc. with values."),
        ("title_block", "What is in the title block? Give part name, drawing number, revision, scale, units."),
        ("views", "What views are shown? (front, top, side, section, detail) List their positions and scales."),
        ("notes", "What notes or annotations are visible?"),
        ("geometry", "What geometric features are visible? Holes, fillets, chamfers, ribs? List sizes."),
    ]
    
    results = {}
    
    for task_name, question in questions:
        try:
            prompt = f"User: <image>\n{question}\nAssistant:"
            
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
            
            results[task_name] = answer.strip()
            
        except Exception as e:
            results[task_name] = f"ERROR: {str(e)}"
        
        free_gpu_memory()
    
    return results


def detect_gt_params_from_vlm(extraction: dict) -> dict:
    """Detect ground truth parameters from VLM extraction."""
    import re
    
    all_text = " ".join(extraction.values())
    all_text_lower = all_text.lower()
    
    detected_gt = {}
    
    # GT_10: No_of_Views
    view_match = re.search(r'(\d+)\s*views?', all_text_lower)
    if view_match:
        detected_value = int(view_match.group(1))
        detected_gt["GT_10"] = {
            "name": "No_of_Views",
            "detected_value": detected_value,
            "ground_truth_value": 6,
            "match": detected_value == 6
        }
    
    # Sheet scale
    scale_match = re.search(r'scale\s*(\d+)\s*[:/]\s*(\d+)', all_text_lower)
    if scale_match:
        numerator = int(scale_match.group(1))
        denominator = int(scale_match.group(2))
        detected_gt["GT_3"] = {"name": "Sheet_ScaleNumerator", "detected_value": numerator, "ground_truth_value": 1, "match": numerator == 1}
        detected_gt["GT_4"] = {"name": "Sheet_ScaleDenominator", "detected_value": denominator, "ground_truth_value": 1, "match": denominator == 1}
    
    # View scales
    view_scale_matches = re.findall(r'scale\s*(\d+)\s*[:/]\s*(\d+)', all_text_lower)
    for num, denom in view_scale_matches:
        if int(denom) == 2:
            detected_gt["GT_24"] = {"name": "View_D2_ScaleDenominator", "detected_value": int(denom), "ground_truth_value": 2, "match": True}
            detected_gt["GT_27"] = {"name": "View_D5_ScaleDenominator", "detected_value": int(denom), "ground_truth_value": 2, "match": True}
    
    # Text size
    if "text" in all_text_lower and "size" in all_text_lower:
        detected_gt["GT_5"] = {"name": "Drafting_TextSize", "detected_value": "Mentioned", "ground_truth_value": 5, "match": None}
    
    return detected_gt


def detect_71_params_vlm(extraction: dict) -> dict:
    """Detect 71 parameters from VLM extraction."""
    all_text = " ".join(extraction.values()).lower()
    
    detected = {}
    
    for param_id, param_info in ALL_PARAMS_71.items():
        is_detected = False
        param_lower = param_info["name"].lower()
        
        if param_id in ["5"]:  # Holes
            is_detected = "hole" in all_text or "bore" in all_text
        elif param_id in ["6"]:  # Fillet
            is_detected = "fillet" in all_text or "radius" in all_text or "r/" in all_text
        elif param_id in ["7"]:  # Chamfers
            is_detected = "chamfer" in all_text or "c " in all_text
        elif param_id in ["8"]:  # Ribs
            is_detected = "rib" in all_text
        
        elif param_id in ["10", "11"]:  # Length, Diameter
            is_detected = ("dimension" in all_text or "mm" in all_text) and any(c.isdigit() for c in all_text)
        elif param_id in ["12"]:
            is_detected = "thickness" in all_text or "thick" in all_text
        
        elif param_id in ["25", "26", "27", "28", "29", "30", "31", "32"]:
            gdt_keywords = {
                "25": ["position", "pos"],
                "26": ["straightness"],
                "27": ["flatness"],
                "28": ["circularity"],
                "29": ["parallelism"],
                "30": ["perpendicular"],
                "31": ["angularity"],
                "32": ["runout"]
            }
            is_detected = any(kw in all_text for kw in gdt_keywords.get(param_id, []))
        
        elif param_id in ["33", "34", "35", "36", "45"]:
            view_keywords = ["front", "top", "side", "section", "detail", "view", "scale"]
            is_detected = any(kw in all_text for kw in view_keywords)
        
        elif param_id in ["51", "52", "53", "54"]:
            note_keywords = ["note", "warning", "caution", "instruction"]
            is_detected = any(kw in all_text for kw in note_keywords)
        
        elif param_id in ["59", "60", "61", "62", "63", "64", "65"]:
            title_keywords = ["rev", "revision", "drawing", "part", "name", "date", "scale", "drawn", "checked"]
            is_detected = any(kw in all_text for kw in title_keywords)
        
        elif param_id in ["66", "67", "68", "69"]:
            bom_keywords = ["bom", "bill", "material", "parts", "item"]
            is_detected = any(kw in all_text for kw in bom_keywords)
        
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


def save_results(drawing_name: str, extraction: dict, detected_71: dict, detected_gt: dict, output_dir: Path):
    """Save extraction results."""
    ensure_dir(output_dir)
    
    total_detected = sum(1 for p in detected_71.values() if p["detected"])
    total = len(detected_71)
    
    gt_matches = sum(1 for g in detected_gt.values() if g.get("match") == True)
    gt_total = len(detected_gt)
    
    summary = {
        "drawing_name": drawing_name,
        "model": "LLaVA 1.5 7B (4-bit)",
        "extraction_date": datetime.now().isoformat(),
        "params_71_detected": total_detected,
        "params_71_total": total,
        "params_71_detection_rate": round((total_detected / total) * 100, 2),
        "gt_params_detected": gt_total,
        "gt_params_matched": gt_matches,
    }
    
    json_data = {
        **summary,
        "extraction": extraction,
        "params_71": detected_71,
        "gt_params": detected_gt
    }
    
    json_path = output_dir / f"{drawing_name}_llava_71params.json"
    save_json(json_data, json_path)
    
    excel_data = []
    for param_id, param_info in detected_71.items():
        excel_data.append({
            "Parameter_ID": param_id,
            "Parameter_Name": param_info["name"],
            "Category": param_info["category"],
            "Detected": "Yes" if param_info["detected"] else "No"
        })
    
    excel_path = output_dir / f"{drawing_name}_llava_71params.xlsx"
    save_excel(excel_data, excel_path, "71 Parameters")
    
    return summary


def process_target_pdf():
    """Process the target PDF with LLaVA."""
    print_header("LLaVA 1.5 7B - 71 Parameters Extraction")
    
    pdf_path = INPUT_DIR / TARGET_PDF
    
    if not pdf_path.exists():
        print(f"\n[ERROR] PDF not found: {pdf_path}")
        return
    
    print(f"\n[PDF] Processing: {pdf_path.name}")
    
    print("\n[LOADING] Loading LLaVA model...")
    model, processor = check_gpu_and_load()
    
    if model is None:
        print("\n[WARNING] Skipping LLaVA due to insufficient GPU memory")
        print("   This model requires ~8GB VRAM")
        return
    
    output_dir = OUTPUT_DIR / "llava_results"
    ensure_dir(output_dir)
    
    print("\n[PDF] Converting PDF to image...")
    image = pdf_to_image(pdf_path, zoom=PDF_ZOOM, max_size=MAX_IMAGE_SIZE)
    
    temp_path = output_dir / f"{pdf_path.stem}_temp.png"
    image.save(temp_path, "PNG")
    
    print(f"   Extracting with LLaVA...")
    extraction = extract_with_llava(model, processor, str(temp_path))
    
    print(f"   Detecting 71 parameters...")
    detected_71 = detect_71_params_vlm(extraction)
    
    print(f"   Detecting ground truth parameters...")
    detected_gt = detect_gt_params_from_vlm(extraction)
    
    drawing_name = pdf_path.stem
    summary = save_results(drawing_name, extraction, detected_71, detected_gt, output_dir)
    
    temp_path.unlink(missing_ok=True)
    free_gpu_memory()
    
    print("\n" + "=" * 60)
    print("[STATS] LLaVA Results Summary")
    print("=" * 60)
    print(f"   71 params detected: {summary['params_71_detected']}/{summary['params_71_total']} ({summary['params_71_detection_rate']}%)")
    
    print("\n[OK] LLaVA extraction complete!")
    print(f"   Results saved to: {output_dir}")
    
    return summary


if __name__ == "__main__":
    try:
        process_target_pdf()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()