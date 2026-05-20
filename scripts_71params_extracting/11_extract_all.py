"""
11_extract_all.py - Full Extraction from CAD Drawing
======================================================
Extract EVERYTHING visible in the drawing using LLaVA.
Not limited to 71 params or ground truth - extract all possible information.
"""

import os
import json
from pathlib import Path
from datetime import datetime

from config import TARGET_PDF, OUTPUT_DIR, MODELS_CONFIG
import fitz
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = PROJECT_ROOT / "input_pdfs"
FULL_EXTRACT_OUTPUT = OUTPUT_DIR / "full_extraction"
FULL_EXTRACT_OUTPUT.mkdir(exist_ok=True)

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
    """Load LLaVA FP16."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    import torch
    
    model_config = MODELS_CONFIG["llava_1.5_7b"]
    print(f"   Loading {model_config['name']} (FP16)...")
    
    processor = AutoProcessor.from_pretrained(
        model_config["model_id"],
        use_fast=False
    )
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_config["model_id"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("   [OK] LLaVA loaded (FP16)")
    return model, processor

def extract_all_info(model, processor, image_path: str) -> dict:
    """Extract ALL information from the drawing."""
    import torch
    from PIL import Image
    
    image = Image.open(image_path).convert("RGB")
    
    if max(image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    results = {}
    
    # 1. COMPREHENSIVE TEXT EXTRACTION
    print("   [1/10] Extracting all visible text...")
    text_prompt = """You are an expert at reading engineering drawings.
Your task is to extract EVERY SINGLE piece of text visible in this drawing.
Do not miss anything - list all text, numbers, labels, annotations.

Format your response as:
TEXT: [list all text found, one per line]

If nothing, write "NONE"."""
    
    prompt = f"USER: <image>\n{text_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    results["all_text"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 2. ALL DIMENSIONS
    print("   [2/10] Extracting ALL dimensions...")
    dim_prompt = """List EVERY dimension visible in this drawing.
Include: linear, diameter, radius, angular, ordinate dimensions.
For each: value + location if visible.
Do not miss any dimension.

Format:
DIMENSIONS:
- [value] at [location if visible]
- [value] at [location if visible]
etc."""
    
    prompt = f"USER: <image>\n{dim_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    results["all_dimensions"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 3. ALL GD&T SYMBOLS
    print("   [3/10] Extracting ALL GD&T symbols...")
    gdt_prompt = """Find and list ALL GD&T (Geometric Dimensioning & Tolerancing) symbols.
Include: position, flatness, straightness, perpendicularity, parallelism, circularity, runout, profile.
For each: symbol type + tolerance value + datum references.

Format:
GD&T:
- [symbol]: [value] [datums]
etc."""
    
    prompt = f"USER: <image>\n{gdt_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    results["all_gdt"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 4. ALL SYMBOLS AND MARKERS
    print("   [4/10] Extracting all symbols and markers...")
    symbols_prompt = """List ALL symbols, markers, and special notations you can see.
Include: surface finish (Ra, Rz), welding symbols, datum targets, center marks, section lines,
break symbols, revision symbols, tolerance symbols, geometric symbols.

Format:
SYMBOLS:
- [symbol type]: [details]
etc."""
    
    prompt = f"USER: <image>\n{symbols_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    results["all_symbols"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 5. ALL VIEWS
    print("   [5/10] Extracting all views...")
    views_prompt = """Identify EVERY view visible in this drawing.
Include: front, top, right, left, bottom, back, section, detail, auxiliary, isometric, exploded.
For each: view type + scale if visible.

Format:
VIEWS:
- [view type]: [scale if visible]
etc."""
    
    prompt = f"USER: <image>\n{views_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    results["all_views"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 6. ALL ANNOTATIONS AND NOTES
    print("   [6/10] Extracting all annotations and notes...")
    notes_prompt = """Extract ALL notes, annotations, callouts, and flags.
Include: general notes, process notes, material notes, finish notes, reference notes.
Include EXACT text content.

Format:
NOTES:
- [note text]
etc."""
    
    prompt = f"USER: <image>\n{notes_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    results["all_notes"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 7. TITLE BLOCK (COMPLETE)
    print("   [7/10] Extracting complete title block...")
    title_prompt = """Extract EVERY piece of information from the title block area.
Do not leave anything out.
Include: company name, drawing number, part name, part number, revision, scale, units,
material, weight, sheet number, approval dates, checkers, drafters, etc.

Format:
TITLE BLOCK:
- [field]: [value]
etc."""
    
    prompt = f"USER: <image>\n{title_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    results["title_block_complete"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 8. BOM (IF PRESENT)
    print("   [8/10] Extracting BOM if present...")
    bom_prompt = """Extract the COMPLETE Bill of Materials (BOM) if visible.
Include ALL columns: Item No, Part Number, Part Name, Quantity, Material, Description, etc.

Format:
BOM:
[table content]
or "NONE" if not visible."""
    
    prompt = f"USER: <image>\n{bom_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    results["bom_complete"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 9. GEOMETRY FEATURES
    print("   [9/10] Extracting geometry features...")
    geom_prompt = """Identify ALL geometric features visible:
- All holes (count, sizes, positions)
- All fillets (count, radii)
- All chamfers (count, sizes)
- All slots/pockets
- All ribs/webs
- All drafts
- All threads

Format:
GEOMETRY:
- Holes: [count] - [sizes]
- Fillets: [count] - [radii]
etc."""
    
    prompt = f"USER: <image>\n{geom_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    results["geometry_features"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 10. LAYERS AND REGIONS
    print("   [10/10] Extracting layer information...")
    layers_prompt = """Look for any layer information, zones, grid lines, or region markers.
Include: zone letters/numbers, grid coordinates, layer names if visible.

Format:
LAYERS/ZONES:
- [information]
or "NONE"."""
    
    prompt = f"USER: <image>\n{layers_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    results["layers_zones"] = processor.decode(output[0], skip_special_tokens=True)
    
    return results

def main():
    print("=" * 60)
    print("  FULL EXTRACTION - Everything Visible")
    print("=" * 60)
    
    pdf_path = INPUT_DIR / TARGET_PDF
    print(f"\n[PDF] {TARGET_PDF}")
    
    temp_dir = FULL_EXTRACT_OUTPUT / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    print("[CONVERT] Converting PDF to images...")
    images = convert_pdf_to_images(str(pdf_path), temp_dir)
    print(f"   Created {len(images)} image(s)")
    
    print("\n[LOAD] Loading LLaVA...")
    model, processor = load_llava()
    
    print("\n[EXTRACT] Extracting ALL information...")
    full_results = extract_all_info(model, processor, images[0])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "drawing_name": TARGET_PDF,
        "model": "LLaVA 1.5 7B (FP16)",
        "extraction_date": datetime.now().isoformat(),
        "extraction_categories": list(full_results.keys()),
        "results": full_results
    }
    
    output_file = FULL_EXTRACT_OUTPUT / f"full_extraction_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Saved: {output_file}")
    
    # Summary
    print("\n=== EXTRACTION SUMMARY ===")
    for key in full_results.keys():
        text = full_results[key]
        # Count lines after ASSISTANT
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
        lines = [l for l in text.split("\n") if l.strip()]
        print(f"   {key}: {len(lines)} items")

if __name__ == "__main__":
    main()