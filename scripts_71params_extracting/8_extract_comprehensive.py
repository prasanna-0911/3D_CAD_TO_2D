"""
8_extract_comprehensive.py - Comprehensive Value Extraction
=============================================================
Extracts ACTUAL VALUES from CAD drawing using LLaVA.
Not just YES/NO - extracts real dimension values, GD&T, notes, etc.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime

# Import configs
from config_comprehensive import (
    COMPREHENSIVE_PARAMS, ALL_COMPREHENSIVE_PARAMS,
    ALL_GT_PARAMS, PARAM_MAPPING, EXTRACTION_CATEGORIES
)
from config import TARGET_PDF, OUTPUT_DIR, MODELS_CONFIG

import fitz
from PIL import Image

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = PROJECT_ROOT / "input_pdfs"
EXTRACTION_OUTPUT = OUTPUT_DIR / "comprehensive_extraction"
EXTRACTION_OUTPUT.mkdir(exist_ok=True)

MAX_IMAGE_SIZE = 1280

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

def load_llava():
    """Load LLaVA model."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    import torch
    
    from transformers import BitsAndBytesConfig
    
    model_config = MODELS_CONFIG["llava_1.5_7b"]
    print(f"   Loading {model_config['name']}...")
    
    processor = AutoProcessor.from_pretrained(
        model_config["model_id"],
        use_fast=False
    )
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_config["model_id"],
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print("   [OK] LLaVA loaded")
    return model, processor

def extract_values_with_llava(model, processor, image_path: str) -> dict:
    """Extract ACTUAL VALUES using LLaVA with detailed prompts."""
    import torch
    from PIL import Image
    
    image = Image.open(image_path).convert("RGB")
    
    if max(image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    results = {}
    
    # 1. DIMENSIONS - Extract actual values
    dim_prompt = """Extract ALL dimension values from this engineering drawing.
For each dimension, provide: value + unit.
Examples:
- Linear: "150 mm", "75.5 mm", "2.5 mm"
- Diameter: "ø10", "ø25 mm", "Ø0.5"
- Radius: "R5", "R12.5 mm"
- Angular: "45°", "30°"
List EVERY dimension visible in the drawing, one per line.
If no dimensions visible, write "NONE"."""
    
    prompt = f"USER: <image>\n{dim_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    results["dimensions"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 2. GD&T SYMBOLS - Extract actual values
    gdt_prompt = """Extract ALL GD&T (Geometric Dimensioning & Tolerancing) symbols.
For each symbol provide: symbol type + tolerance value + datum references.
Examples:
- Position: "ø0.05 A B C"
- Flatness: "0.02"
- Perpendicularity: "0.03 A"
- Straightness: "0.01"
- Runout: "0.05 A"
List EVERY GD&T symbol visible, one per line.
If none, write "NONE"."""
    
    prompt = f"USER: <image>\n{gdt_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    results["gdt_symbols"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 3. TITLE BLOCK - Extract actual values
    title_prompt = """Extract title block information.
Provide exact values for:
- Drawing Number:
- Part Name:
- Revision:
- Scale:
- Units:
- Date:
- Author:
If any field is empty or not visible, write "NOT VISIBLE"."""
    
    prompt = f"USER: <image>\n{title_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    results["title_block"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 4. VIEWS - Extract actual values
    views_prompt = """Identify ALL views in this drawing.
For each view provide: view name/type + position + scale.
Examples:
- Front view at (493.6, 521.9), scale 1:1
- Top view at (493.6, 705.7), scale 1:1
- Right view at (1079.8, 705.7), scale 1:1
List EVERY view visible, one per line."""
    
    prompt = f"USER: <image>\n{views_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    results["views"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 5. NOTES - Extract actual values
    notes_prompt = """Extract ALL notes, annotations, and text visible in this drawing.
Include:
- General notes at bottom
- Surface finish requirements (Ra, Rz)
- Welding symbols
- Datum identifiers (A, B, C)
- Any warning/caution text
Provide FULL TEXT of each note.
If no notes, write "NONE"."""
    
    prompt = f"USER: <image>\n{notes_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    results["notes"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 6. GEOMETRY FEATURES - Extract actual values
    geom_prompt = """Identify geometric features and provide counts/sizes:
- Holes: count + diameters (e.g., "4 holes, ø10mm each")
- Fillets: count + radii (e.g., "6 fillets, R3mm")
- Chamfers: count + sizes (e.g., "2 chamfers, 2x45°")
- Ribs: count + positions
If none of a type, write "0" for that type."""
    
    prompt = f"USER: <image>\n{geom_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    results["geometry"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 7. BOM - Extract actual values
    bom_prompt = """Extract Bill of Materials (BOM) if visible.
Provide: item number + part name + quantity.
If no BOM, write "NONE"."""
    
    prompt = f"USER: <image>\n{bom_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    results["bom"] = processor.decode(output[0], skip_special_tokens=True)
    
    return results

def parse_extracted_values(extracted: dict) -> dict:
    """Parse extracted text into structured values."""
    parsed = {}
    
    def get_answer_only(text: str) -> str:
        """Extract only the answer after ASSISTANT:"""
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:")[-1].strip()
        return text.strip()
    
    # Parse dimensions - extract numbers
    dim_text = get_answer_only(extracted.get("dimensions", ""))
    dimensions = []
    for line in dim_text.split("\n"):
        line = line.strip()
        if line and line != "NONE" and "Length:" in line:
            # Extract just the dimension value
            parts = line.split("Length:", 1)
            if len(parts) > 1:
                dimensions.append(parts[1].strip())
    parsed["dimensions"] = dimensions[:20]  # Limit to 20
    
    # Parse GD&T
    gdt_text = get_answer_only(extracted.get("gdt_symbols", ""))
    gdt_symbols = []
    for line in gdt_text.split("\n"):
        line = line.strip()
        if line and line != "NONE":
            gdt_symbols.append(line)
    parsed["gdt_symbols"] = gdt_symbols[:15]
    
    # Parse title block - extract key-value pairs
    title_text = get_answer_only(extracted.get("title_block", ""))
    title_block = {}
    for line in title_text.split("\n"):
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value:
                    title_block[key] = value
    parsed["title_block"] = title_block
    
    # Parse views
    views_text = get_answer_only(extracted.get("views", ""))
    views = []
    for line in views_text.split("\n"):
        line = line.strip()
        if line and line != "NONE" and "view" in line.lower():
            views.append(line)
    parsed["views"] = views[:10]
    
    # Parse notes
    notes_text = get_answer_only(extracted.get("notes", ""))
    notes = []
    for line in notes_text.split("\n"):
        line = line.strip()
        if line and line != "NONE":
            notes.append(line)
    parsed["notes"] = notes[:15]
    
    # Parse geometry
    geom_text = get_answer_only(extracted.get("geometry", ""))
    geometry = {}
    # Extract just the answer portion
    geometry["description"] = geom_text[:200] if geom_text else "NONE"
    parsed["geometry"] = geometry
    
    # Parse BOM
    bom_text = get_answer_only(extracted.get("bom", ""))
    bom = []
    for line in bom_text.split("\n"):
        line = line.strip()
        if line and line != "NONE":
            bom.append(line)
    parsed["bom"] = bom[:10]
    
    return parsed

def main():
    print("=" * 60)
    print("  Comprehensive Value Extraction (LLaVA)")
    print("=" * 60)
    
    pdf_path = INPUT_DIR / TARGET_PDF
    print(f"\n[PDF] {TARGET_PDF}")
    
    # Convert to images
    temp_dir = EXTRACTION_OUTPUT / "temp_images"
    temp_dir.mkdir(exist_ok=True)
    
    print("[CONVERT] Converting PDF to images...")
    images = convert_pdf_to_images(str(pdf_path), temp_dir)
    print(f"   Created {len(images)} image(s)")
    
    # Load model
    print("\n[LOAD] Loading LLaVA...")
    model, processor = load_llava()
    
    # Extract from first page (main drawing)
    print("\n[EXTRACT] Extracting values from drawing...")
    extracted_values = extract_values_with_llava(model, processor, images[0])
    
    # Parse into structured format
    parsed = parse_extracted_values(extracted_values)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result = {
        "drawing_name": TARGET_PDF,
        "model": "LLaVA 1.5 7B",
        "extraction_date": datetime.now().isoformat(),
        "raw_extraction": extracted_values,
        "parsed_values": parsed,
    }
    
    output_json = EXTRACTION_OUTPUT / f"comprehensive_extraction_{timestamp}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Saved: {output_json}")
    
    # Print summary
    print("\n=== EXTRACTED VALUES SUMMARY ===")
    print(f"Dimensions: {len(parsed.get('dimensions', []))} found")
    print(f"GD&T Symbols: {len(parsed.get('gdt_symbols', []))} found")
    print(f"Title Block: {len(parsed.get('title_block', {}))} fields")
    print(f"Views: {len(parsed.get('views', []))} found")
    print(f"Notes: {len(parsed.get('notes', []))} found")
    print(f"BOM: {len(parsed.get('bom', []))} items")

if __name__ == "__main__":
    main()