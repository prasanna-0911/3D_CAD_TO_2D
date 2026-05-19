"""
1_test_easyocr.py - EasyOCR Parameter Extraction
===================================================
Extracts 71 parameters using EasyOCR from the target PDF.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts_71params_extracting.config import (
    INPUT_DIR, OUTPUT_DIR, TARGET_PDF,
    PARAMETERS_71, ALL_PARAMS_71, GROUND_TRUTH_PARAMS, PARAM_MAPPING
)
from scripts_71params_extracting.utils import (
    pdf_to_image, save_json, save_excel, print_header, print_model_status,
    detect_params_71, detect_ground_truth_params, compare_with_ground_truth,
    ensure_dir, get_gpu_memory_info
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
        
        x = sum(float(p[0]) for p in bbox) / 4
        y = sum(float(p[1]) for p in bbox) / 4
        
        elements.append({
            "id": idx + 1,
            "text": text.strip(),
            "confidence": round(float(conf), 3),
            "x": round(x, 1),
            "y": round(y, 1)
        })
    
    return elements


def detect_71_params(elements: list) -> dict:
    """Detect which of the 71 parameters are present."""
    all_text = " ".join([e["text"] for e in elements])
    
    detected = {}
    
    for param_id, param_info in ALL_PARAMS_71.items():
        text_lower = all_text.lower()
        param_lower = param_info["name"].lower()
        
        is_detected = False
        
        if param_id in ["5"]:  # Holes
            is_detected = "hole" in text_lower or "ø" in text_lower or "Ø" in text_lower
        elif param_id in ["6"]:  # Fillet
            is_detected = "fillet" in text_lower or "r" in text_lower and any(c.isdigit() for c in text_lower)
        elif param_id in ["7"]:  # Chamfers
            is_detected = "chamfer" in text_lower or "c" in text_lower
        elif param_id in ["8"]:  # Ribs
            is_detected = "rib" in text_lower
        
        elif param_id in ["10", "11"]:  # Length, Diameter
            is_detected = any(c.isdigit() for c in text_lower) and ("mm" in text_lower or "ø" in text_lower)
        elif param_id in ["12"]:  # Thickness
            is_detected = "thick" in text_lower or "th" in text_lower
        
        elif param_id in ["25", "26", "27", "28", "29", "30", "31", "32"]:  # GD&T
            gdt_keywords = {
                "25": ["position", "pos", "⊕"],
                "26": ["straightness", "—"],
                "27": ["flatness", "□"],
                "28": ["circularity", "○"],
                "29": ["parallelism", "//"],
                "30": ["perpendicular", "⊥"],
                "31": ["angularity", "∠"],
                "32": ["runout", "↗"]
            }
            is_detected = any(kw in text_lower for kw in gdt_keywords.get(param_id, []))
        
        elif param_id in ["33", "34", "35", "36", "45"]:  # Views
            view_keywords = ["front", "top", "side", "section", "detail", "view", "scale", "a-a", "b-b"]
            is_detected = any(kw in text_lower for kw in view_keywords)
        
        elif param_id in ["51", "52", "53", "54"]:  # Notes
            note_keywords = ["note", "warning", "caution", "1)", "2)", "3)", "general"]
            is_detected = any(kw in text_lower for kw in note_keywords)
        
        elif param_id in ["59", "60", "61", "62", "63", "64", "65"]:  # TitleBlock
            title_keywords = ["rev", "revision", "dwg", "drawing", "part", "name", "date", "drawn", "checked", "scale"]
            is_detected = any(kw in text_lower for kw in title_keywords)
        
        elif param_id in ["66", "67", "68", "69"]:  # BOM
            bom_keywords = ["bom", "bill", "material", "parts", "item", "qty", "quantity"]
            is_detected = any(kw in text_lower for kw in bom_keywords)
        
        elif param_id in ["70", "71"]:  # Scale
            is_detected = "scale" in text_lower or "1:" in text_lower
        
        else:
            keywords = [w for w in param_lower.split() if len(w) > 3]
            is_detected = any(kw in text_lower for kw in keywords)
        
        detected[param_id] = {
            "id": param_id,
            "name": param_info["name"],
            "detected": is_detected,
            "category": next(cat for cat, params in PARAMETERS_71.items() 
                          for p in params if p["id"] == param_id)
        }
    
    return detected


def detect_gt_params_from_drawing(elements: list) -> dict:
    """Try to detect ground truth parameters from drawing text."""
    import re
    
    all_text = " ".join([e["text"] for e in elements])
    all_text_lower = all_text.lower()
    
    detected_gt = {}
    
    # Load ground truth for comparison
    from config import GROUND_TRUTH_PARAMS, ALL_GT_PARAMS
    
    # GT_10: No_of_Views - Look for view count
    view_match = re.search(r'(\d+)\s*views?', all_text_lower)
    if view_match:
        detected_value = int(view_match.group(1))
        gt_value = 6  # From ground truth
        detected_gt["GT_10"] = {
            "name": "No_of_Views",
            "detected_value": detected_value,
            "ground_truth_value": gt_value,
            "match": detected_value == gt_value
        }
    
    # GT_3, GT_4: Sheet Scale - Look for scale notation
    scale_match = re.search(r'scale\s*(\d+)\s*[:/]\s*(\d+)', all_text_lower)
    if scale_match:
        numerator = int(scale_match.group(1))
        denominator = int(scale_match.group(2))
        
        detected_gt["GT_3"] = {
            "name": "Sheet_ScaleNumerator",
            "detected_value": numerator,
            "ground_truth_value": 1,
            "match": numerator == 1
        }
        detected_gt["GT_4"] = {
            "name": "Sheet_ScaleDenominator",
            "detected_value": denominator,
            "ground_truth_value": 1,
            "match": denominator == 1
        }
    
    # Look for view scale mentions (e.g., "SCALE 1:2", "SCALE 2:1")
    view_scale_matches = re.findall(r'scale\s*(\d+)\s*[:/]\s*(\d+)', all_text_lower)
    if view_scale_matches:
        # Check for View_D2_ScaleDenominator = 2
        for num, denom in view_scale_matches:
            if int(denom) == 2:
                detected_gt["GT_24"] = {
                    "name": "View_D2_ScaleDenominator",
                    "detected_value": int(denom),
                    "ground_truth_value": 2,
                    "match": True
                }
            if int(denom) == 2:
                detected_gt["GT_27"] = {
                    "name": "View_D5_ScaleDenominator",
                    "detected_value": int(denom),
                    "ground_truth_value": 2,
                    "match": True
                }
    
    # GT_5: Drafting_TextSize - Look for text size mentions
    text_size_match = re.search(r'text\s*size[:\s]*(\d+\.?\d*)', all_text_lower)
    if text_size_match:
        detected_value = float(text_size_match.group(1))
        detected_gt["GT_5"] = {
            "name": "Drafting_TextSize",
            "detected_value": detected_value,
            "ground_truth_value": 5,
            "match": abs(detected_value - 5) < 1
        }
    
    # GT_6: Drafting_ArrowSize
    arrow_size_match = re.search(r'arrow\s*size[:\s]*(\d+\.?\d*)', all_text_lower)
    if arrow_size_match:
        detected_value = float(arrow_size_match.group(1))
        detected_gt["GT_6"] = {
            "name": "Drafting_ArrowSize",
            "detected_value": detected_value,
            "ground_truth_value": 2.5,
            "match": abs(detected_value - 2.5) < 0.5
        }
    
    # GT_14, GT_15: View_Top_Scale
    # Check for "TOP" view scale in drawing
    if "top" in all_text_lower and "scale" in all_text_lower:
        top_scale_match = re.search(r'top[^scale]*scale\s*(\d+)\s*[:/]\s*(\d+)', all_text_lower)
        if top_scale_match:
            detected_gt["GT_14"] = {
                "name": "View_Top_ScaleNumerator",
                "detected_value": int(top_scale_match.group(1)),
                "ground_truth_value": 1,
                "match": int(top_scale_match.group(1)) == 1
            }
    
    return detected_gt


def save_results(drawing_name: str, elements: list, detected_71: dict, detected_gt: dict, output_dir: Path):
    """Save extraction results to JSON and Excel."""
    ensure_dir(output_dir)
    
    total_71_detected = sum(1 for p in detected_71.values() if p["detected"])
    total_71 = len(detected_71)
    
    summary = {
        "drawing_name": drawing_name,
        "model": "EasyOCR",
        "extraction_date": datetime.now().isoformat(),
        "total_text_elements": len(elements),
        "params_71_detected": total_71_detected,
        "params_71_total": total_71,
        "params_71_detection_rate": round((total_71_detected / total_71) * 100, 2),
        "gt_params_detected": len(detected_gt),
    }
    
    json_data = {
        **summary,
        "elements": elements,
        "params_71": detected_71,
        "gt_params": detected_gt
    }
    
    json_path = output_dir / f"{drawing_name}_easyocr_71params.json"
    save_json(json_data, json_path)
    
    # Save Excel with 71 parameters
    excel_data = []
    for param_id, param_info in detected_71.items():
        excel_data.append({
            "Parameter_ID": param_id,
            "Parameter_Name": param_info["name"],
            "Category": param_info["category"],
            "Detected": "Yes" if param_info["detected"] else "No"
        })
    
    excel_path = output_dir / f"{drawing_name}_easyocr_71params.xlsx"
    save_excel(excel_data, excel_path, "71 Parameters")
    
    return summary


def process_target_pdf():
    """Process the target PDF with EasyOCR."""
    print_header("EasyOCR - 71 Parameters Extraction")
    
    pdf_path = INPUT_DIR / TARGET_PDF
    
    if not pdf_path.exists():
        print(f"\n❌ PDF not found: {pdf_path}")
        return
    
    print(f"\n📄 Processing: {pdf_path.name}")
    
    print("\n📦 Initializing EasyOCR...")
    reader = init_easyocr()
    
    output_dir = OUTPUT_DIR / "easyocr_results"
    ensure_dir(output_dir)
    
    print("\n📄 Converting PDF to image...")
    image = pdf_to_image(pdf_path, zoom=3.0, max_size=2048)
    
    temp_image_path = output_dir / f"{pdf_path.stem}_temp.png"
    image.save(temp_image_path, "PNG")
    image_path = str(temp_image_path)
    
    print(f"   Extracting text...")
    elements = extract_text(reader, image_path)
    
    print(f"   Found {len(elements)} text elements")
    
    print(f"   Detecting 71 parameters...")
    detected_71 = detect_71_params(elements)
    
    print(f"   Detecting ground truth parameters...")
    detected_gt = detect_gt_params_from_drawing(elements)
    
    drawing_name = pdf_path.stem
    summary = save_results(drawing_name, elements, detected_71, detected_gt, output_dir)
    
    temp_image_path.unlink(missing_ok=True)
    
    print("\n" + "=" * 60)
    print("📊 EasyOCR Results Summary")
    print("=" * 60)
    print(f"   Text elements: {summary['total_text_elements']}")
    print(f"   71 params detected: {summary['params_71_detected']}/{summary['params_71_total']} ({summary['params_71_detection_rate']}%)")
    print(f"   GT params detected: {summary['gt_params_detected']}")
    
    print("\n✅ EasyOCR extraction complete!")
    print(f"   Results saved to: {output_dir}")
    
    return summary


if __name__ == "__main__":
    try:
        process_target_pdf()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()