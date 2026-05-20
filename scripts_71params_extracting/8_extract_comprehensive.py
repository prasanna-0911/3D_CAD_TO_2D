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
    
    # 1. DIMENSIONS - More explicit prompt
    dim_prompt = """You are an expert at reading engineering drawings. 
Look at the image carefully and extract ONLY the dimension values you can ACTUALLY SEE.
DO NOT invent or guess dimensions.
DO NOT number dimensions 1,2,3 - just list what you see.

For each dimension you see, write exactly as shown in the drawing:
- Linear dimensions: "45" (with unit from drawing)
- Diameter: "Ø12" or "ø10"
- Radius: "R5" 
- Angular: "45°"

CRITICAL: Only list dimensions that are VISIBLE in the image. If uncertain, write "NONE".
Start your answer with "DIMENSIONS:" """
    
    prompt = f"USER: <image>\n{dim_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    results["dimensions"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 2. GD&T SYMBOLS - More explicit
    gdt_prompt = """You are an expert at reading GD&T symbols in engineering drawings.
Look at the image carefully for feature control frames and GD&T symbols.
Extract ONLY symbols you can ACTUALLY SEE.

For each GD&T symbol, note:
- The geometric tolerance type (position, flatness, straightness, etc.)
- The tolerance value
- The datum references

Example format: "Position: Ø0.05 A B C" or "Flatness: 0.02"
DO NOT invent symbols that are not visible.
If none visible, write "NONE".
Start your answer with "GDT:" """
    
    prompt = f"USER: <image>\n{gdt_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    results["gdt_symbols"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 3. TITLE BLOCK - More explicit
    title_prompt = """Look at the TITLE BLOCK area of this engineering drawing (usually at bottom right or right side).
Extract ONLY the text you can ACTUALLY READ in these fields:
- Drawing Number (look for number in title block)
- Part Name (look for name/title)
- Revision (look for letter like A, B, C or number)
- Scale (look for like 1:1, 1:2, 2:1)
- Units (look for mm, inch, etc.)
- Date (look for date format)
- Author (look for name)

Write ONLY what you can read. If field is empty or unreadable, write "NOT FOUND".
Start your answer with "TITLE BLOCK:" """
    
    prompt = f"USER: <image>\n{title_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    results["title_block"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 4. VIEWS - More explicit
    views_prompt = """Look at this engineering drawing and count how many orthographic/pictorial views you can see.
Types to look for: front view, top view, right side view, left side view, section view, detail view, isometric view.
DO NOT list all possible views - only count what is ACTUALLY VISIBLE in the drawing.
If 3 views visible, say "3 views: front, top, right"
If 6 views visible, list them all.
Start with "VIEWS:" """
    
    prompt = f"USER: <image>\n{views_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    results["views"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 5. NOTES - More explicit
    notes_prompt = """Look at this engineering drawing for text notes and annotations.
Find and extract:
- General notes (usually in bottom left area, numbered 1, 2, 3...)
- Surface finish symbols (Ra, Rz values)
- Welding symbols
- Datum target points (A, B, C with circles)
- Any warning or reference notes

Extract the EXACT TEXT as written. If notes area is empty, write "NONE".
Start with "NOTES:" """
    
    prompt = f"USER: <image>\n{notes_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    results["notes"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 6. GEOMETRY FEATURES - More explicit
    geom_prompt = """Look at the drawing and count geometric features you can SEE:
- Count all CIRCULAR holes (with diameters visible)
- Count all FILLETS (rounded corners, R values)
- Count all CHAMFERS (angled cuts, often 45 degrees)
- Count all RIBS (thin vertical/horizontal supporting features)

For each type, give count and sizes if visible.
Example: "Holes: 4 holes visible, Ø8, Ø12"
Example: "Fillets: 6 fillets visible"
DO NOT guess - only count what you can see.
Start with "GEOMETRY:" """
    
    prompt = f"USER: <image>\n{geom_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    results["geometry"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 7. BOM - More explicit
    bom_prompt = """Look for a Bill of Materials (BOM) table in this drawing.
Usually appears as a table with: Item No, Part Name, Quantity, Material columns.
If BOM table is visible, extract: item number, part name, quantity.
If no BOM visible, write "NONE".
Start with "BOM:" """
    
    prompt = f"USER: <image>\n{bom_prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    results["bom"] = processor.decode(output[0], skip_special_tokens=True)
    
    return results

def parse_extracted_values(extracted: dict) -> dict:
    """Parse extracted text into structured values."""
    parsed = {}
    
    def get_answer_only(text: str, prefix: str = None) -> str:
        """Extract only the answer after ASSISTANT: and optional prefix."""
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
        
        if prefix and prefix in text.upper():
            text = text.split(prefix)[-1].strip()
        
        return text
    
    # Parse dimensions
    dim_text = get_answer_only(extracted.get("dimensions", ""), "DIMENSIONS:")
    dimensions = []
    for line in dim_text.split("\n"):
        line = line.strip()
        # Skip empty lines and "NONE"
        if line and line.upper() != "NONE":
            # Extract numbers with units
            numbers = re.findall(r'[\d.]+\s*(?:mm|°|ø|Ø|R)?', line, re.IGNORECASE)
            if numbers and len(line) < 50:  # Reasonable length for dimension
                dimensions.append(line)
    parsed["dimensions"] = dimensions[:20]
    
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