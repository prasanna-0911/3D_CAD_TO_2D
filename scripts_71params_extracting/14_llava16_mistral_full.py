"""
14_llava16_mistral_full.py - LLaVA 1.6 Mistral 7B Full Extraction
==================================================================
Extract everything from CAD drawing in structured format (like llava15_extraction_structured.xlsx)
"""

import os
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = PROJECT_ROOT / "input_pdfs"
OUTPUT_DIR = PROJECT_ROOT / "outputs_71params"
LLAVA16_OUTPUT = OUTPUT_DIR / "llava16_mistral_full"
LLAVA16_OUTPUT.mkdir(exist_ok=True)

TARGET_PDF = "HP_58231-82P00_500_s_SUZUKI_DRAW_SH1.pdf"

import fitz
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

MAX_IMAGE_SIZE = 1280
MODEL_ID = "llava-v1.6-mistral-7b-hf"

def convert_pdf_to_images(pdf_path: str, output_dir: Path) -> list:
    """Convert PDF to images."""
    images = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_path = output_dir / f"page_{page_num + 1}.png"
        pix.save(img_path)
        images.append(str(img_path))
    doc.close()
    return images

def load_llava16():
    """Load LLaVA 1.6 Mistral 7B."""
    print(f"   Loading {MODEL_ID} (FP16)...")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("   [OK] LLaVA 1.6 Mistral 7B loaded (FP16)")
    return model, processor

def extract_task(model, processor, image_path: str, task: str, prompt: str) -> str:
    """Extract for a single task."""
    image = Image.open(image_path).convert("RGB")
    
    if max(image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    result = processor.decode(output[0], skip_special_tokens=True)
    
    if "ASSISTANT:" in result:
        result = result.split("ASSISTANT:")[-1].strip()
    
    return result

def extract_all_structured(model, processor, image_path: str) -> dict:
    """Extract all information in structured format."""
    results = {}
    
    # Task 1: All Text
    print("   [Task 1/6] Extracting all text...")
    prompt1 = """Extract ALL text visible in this engineering drawing. 
List every piece of text, number, label, and annotation you can see.
Be thorough - do not miss anything."""
    results["Task1_All_Text"] = extract_task(model, processor, image_path, "All Text", prompt1)
    
    # Task 2: Dimensions
    print("   [Task 2/6] Extracting dimensions...")
    prompt2 = """Extract ALL dimensions from this drawing.
Categorize them as:
1. Length dimensions (with values in mm)
2. Diameter dimensions (with values like Ø10)
3. Radius dimensions (with values like R5)
4. Angular dimensions (with values in degrees)
List each dimension with its category and value."""
    results["Task2_Dimensions"] = extract_task(model, processor, image_path, "Dimensions", prompt2)
    
    # Task 3: GD&T Symbols
    print("   [Task 3/6] Extracting GD&T symbols...")
    prompt3 = """Extract ALL GD&T (Geometric Dimensioning and Tolerancing) symbols.
For each symbol provide:
- Symbol type (position, flatness, straightness, perpendicularity, etc.)
- Tolerance value
- Datum references
If no GD&T symbols visible, state "No GD&T symbols visible"."""
    results["Task3_GDT"] = extract_task(model, processor, image_path, "GD&T", prompt3)
    
    # Task 4: Views
    print("   [Task 4/6] Extracting views...")
    prompt4 = """List ALL orthographic and pictorial views visible in this drawing.
Include: front, top, right, left, bottom, back, section, detail, auxiliary, isometric.
For each view note the type and any scale information."""
    results["Task4_Views"] = extract_task(model, processor, image_path, "Views", prompt4)
    
    # Task 5: Title Block
    print("   [Task 5/6] Extracting title block...")
    prompt5 = """Extract ALL information from the title block area.
Include:
1. Part name/title
2. Drawing number
3. Revision
4. Designer/Drawn by
5. Checker/Reviewed by
6. Dates
7. Company name
8. Material
9. Scale
10. Sheet size
11. Units
12. Projection type
If any field is not visible, state "Not specified in the image"."""
    results["Task5_TitleBlock"] = extract_task(model, processor, image_path, "Title Block", prompt5)
    
    # Task 6: Features (Geometry)
    print("   [Task 6/6] Extracting geometric features...")
    prompt6 = """Identify and list ALL geometric features visible:
- Holes (count and sizes/diameters)
- Fillets (count and radii)
- Chamfers (count and sizes)
- Slots/Pockets (count and dimensions)
- Ribs (count and positions)
- Threads (if visible)
Be specific about quantities and sizes."""
    results["Task6_Features"] = extract_task(model, processor, image_path, "Features", prompt6)
    
    return results

def create_structured_output(results: dict, image_path: str) -> dict:
    """Create structured output similar to llava15_extraction_structured.xlsx."""
    structured = {
        "Summary": {
            "Model Used": "LLaVA 1.6 (llava-v1.6-mistral-7b-hf)",
            "Drawing Name": TARGET_PDF,
            "Image Size": f"{Image.open(image_path).size}",
            "Extraction Date": datetime.now().isoformat(),
            "Total Tasks Completed": 6
        },
        "Tasks_Overview": [
            {"Task": "Task1_All_Text", "Status": "Completed"},
            {"Task": "Task2_Dimensions", "Status": "Completed"},
            {"Task": "Task3_GDT", "Status": "Completed"},
            {"Task": "Task4_Views", "Status": "Completed"},
            {"Task": "Task5_TitleBlock", "Status": "Completed"},
            {"Task": "Task6_Features", "Status": "Completed"},
        ],
        "Task1_All_Text": [],
        "Task2_Dimensions": [],
        "Task3_GDT": [],
        "Task4_Views": [],
        "Task5_TitleBlock": [],
        "Task6_Features": [],
        "Raw_Data": results
    }
    
    # Parse Dimensions into categories
    dim_text = results.get("Task2_Dimensions", "")
    categories = {
        "1. Length dimensions": [],
        "2. Diameter dimensions": [],
        "3. Radius dimensions": [],
        "4. Angular dimensions": []
    }
    
    for line in dim_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Extract dimension values
        import re
        numbers = re.findall(r'[\d.]+', line)
        
        if "diameter" in line.lower() or "ø" in line.lower() or "Ø" in line.lower():
            for n in numbers:
                categories["2. Diameter dimensions"].append({"Category": "2. Diameter dimensions", "Entry": line})
        elif "radius" in line.lower() or "r" in line.lower():
            for n in numbers:
                categories["3. Radius dimensions"].append({"Category": "3. Radius dimensions", "Entry": line})
        elif "°" in line or "degree" in line.lower():
            categories["4. Angular dimensions"].append({"Category": "4. Angular dimensions", "Entry": line})
        elif numbers:
            categories["1. Length dimensions"].append({"Category": "1. Length dimensions", "Entry": line})
    
    structured["Task2_Dimensions"] = [item for cat in categories.values() for item in cat][:50]
    
    # Parse Title Block
    tb_text = results.get("Task5_TitleBlock", "")
    tb_fields = [
        "1. Part name/title",
        "2. Drawing number",
        "3. Revision",
        "4. Designer/Drawn by",
        "5. Checker/Reviewed by",
        "6. Dates",
        "7. Company name",
        "8. Material",
        "9. Scale",
        "10. Sheet size",
        "11. Units",
        "12. Projection type"
    ]
    
    for field in tb_fields:
        if field in tb_text:
            structured["Task5_TitleBlock"].append({"Field": field, "Value": "Found"})
        else:
            structured["Task5_TitleBlock"].append({"Field": field, "Value": "Not specified in the image"})
    
    # Other tasks - add as raw text
    structured["Task1_All_Text"] = [{"Task": "Extract All Text", "Extracted Content": results.get("Task1_All_Text", "")[:2000]}]
    structured["Task3_GDT"] = [{"Task": "GD&T Symbols Extraction", "Result": results.get("Task3_GDT", "")}]
    structured["Task4_Views"] = [{"Task": "Views Extraction", "Result": results.get("Task4_Views", "")}]
    structured["Task6_Features"] = [{"Task": "Features Extraction", "Result": results.get("Task6_Features", "")}]
    
    return structured

def main():
    print("=" * 60)
    print("  LLaVA 1.6 Mistral 7B - Full Structured Extraction")
    print("=" * 60)
    
    pdf_path = INPUT_DIR / TARGET_PDF
    print(f"\n[PDF] {TARGET_PDF}")
    
    temp_dir = LLAVA16_OUTPUT / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    print("\n[CONVERT] Converting PDF to images...")
    images = convert_pdf_to_images(str(pdf_path), temp_dir)
    print(f"   Created {len(images)} image(s)")
    
    # Save sample image
    sample_path = LLAVA16_OUTPUT / "sample_page_1.png"
    print(f"\n[SAVE] Saving sample image: {sample_path}")
    Image.open(images[0]).save(sample_path)
    
    print("\n[LOAD] Loading LLaVA 1.6 Mistral 7B...")
    model, processor = load_llava16()
    
    print("\n[EXTRACT] Extracting all information...")
    results = extract_all_structured(model, processor, images[0])
    
    print("\n[STRUCTURE] Creating structured output...")
    structured = create_structured_output(results, images[0])
    
    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = LLAVA16_OUTPUT / f"llava16_mistral_{timestamp}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] JSON saved: {output_json}")
    
    # Summary
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"   Tasks completed: 6/6")
    print(f"   Dimensions found: {len(structured.get('Task2_Dimensions', []))}")
    print(f"   Title block fields: {len(structured.get('Task5_TitleBlock', []))}")
    print(f"   GD&T: {results.get('Task3_GDT', '')[:100]}")
    print(f"   Views: {results.get('Task4_Views', '')[:100]}")
    print(f"   Features: {results.get('Task6_Features', '')[:100]}")

if __name__ == "__main__":
    main()