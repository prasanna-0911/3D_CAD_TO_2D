"""
13_qwen2vl_full.py - Qwen2-VL Full Extraction
=============================================
Extract everything from CAD drawing using Qwen2-VL (non-quantized).
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
QWEN_OUTPUT = OUTPUT_DIR / "qwen2vl_full_extraction"
QWEN_OUTPUT.mkdir(exist_ok=True)

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

def load_qwen2vl():
    """Load Qwen2-VL non-quantized."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    import torch
    
    model_config = MODELS_CONFIG["qwen2_vl_7b"]
    print(f"   Loading {model_config['name']} (FP16)...")
    
    processor = AutoProcessor.from_pretrained(
        model_config["model_id"],
        trust_remote_code=True
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config["model_id"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("   [OK] Qwen2-VL loaded (FP16)")
    return model, processor

def extract_all_qwen(model, processor, image_path: str) -> dict:
    """Extract all information using Qwen2-VL."""
    import torch
    from PIL import Image
    
    image = Image.open(image_path).convert("RGB")
    
    if max(image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    results = {}
    
    # 1. ALL TEXT
    print("   [1/10] Extracting all text...")
    text_prompt = """List EVERY piece of text visible in this engineering drawing.
Include all numbers, letters, labels, annotations.
Format: TEXT: [each item on new line]
If none: NONE"""
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    results["all_text"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 2. ALL DIMENSIONS
    print("   [2/10] Extracting dimensions...")
    dim_prompt = """List ALL dimensions visible: linear, diameter, radius, angular.
Format: DIMENSIONS: [each on new line]
If none: NONE"""
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": dim_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    results["all_dimensions"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 3. ALL GD&T
    print("   [3/10] Extracting GD&T...")
    gdt_prompt = """List ALL GD&T symbols with values and datums.
Format: GD&T: [each on new line]
If none: NONE"""
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": gdt_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    results["all_gdt"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 4. ALL SYMBOLS
    print("   [4/10] Extracting symbols...")
    symbols_prompt = """List ALL symbols: surface finish, welding, datum targets, etc.
Format: SYMBOLS: [each on new line]
If none: NONE"""
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": symbols_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    results["all_symbols"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 5. ALL VIEWS
    print("   [5/10] Extracting views...")
    views_prompt = """List ALL views visible.
Format: VIEWS: [each on new line]
If none: NONE"""
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": views_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    results["all_views"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 6. ALL NOTES
    print("   [6/10] Extracting notes...")
    notes_prompt = """List ALL notes and annotations.
Format: NOTES: [each on new line]
If none: NONE"""
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": notes_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    results["all_notes"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 7. TITLE BLOCK COMPLETE
    print("   [7/10] Extracting title block...")
    title_prompt = """Extract ALL title block info: drawing number, part name, revision, scale, units, date, author, material, etc.
Format: TITLE BLOCK: [field: value on new lines]
If none: NONE"""
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": title_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    results["title_block_complete"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 8. BOM
    print("   [8/10] Extracting BOM...")
    bom_prompt = """Extract Bill of Materials if visible.
Format: BOM: [table content]
If none: NONE"""
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": bom_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    results["bom_complete"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 9. GEOMETRY
    print("   [9/10] Extracting geometry...")
    geom_prompt = """List ALL geometry features: holes, fillets, chamfers, slots, ribs.
Format: GEOMETRY: [each on new line]
If none: NONE"""
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": geom_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    results["geometry_features"] = processor.decode(output[0], skip_special_tokens=True)
    
    # 10. LAYERS/ZONES
    print("   [10/10] Extracting layers/zones...")
    layers_prompt = """List any layer or zone information visible.
Format: LAYERS/ZONES: [info]
If none: NONE"""
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": layers_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    results["layers_zones"] = processor.decode(output[0], skip_special_tokens=True)
    
    return results

def main():
    print("=" * 60)
    print("  Qwen2-VL Full Extraction (FP16)")
    print("=" * 60)
    
    pdf_path = INPUT_DIR / TARGET_PDF
    print(f"\n[PDF] {TARGET_PDF}")
    
    temp_dir = QWEN_OUTPUT / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    print("\n[CONVERT] Converting PDF to images...")
    images = convert_pdf_to_images(str(pdf_path), temp_dir)
    print(f"   Created {len(images)} image(s)")
    
    print("\n[LOAD] Loading Qwen2-VL...")
    model, processor = load_qwen2vl()
    
    print("\n[EXTRACT] Extracting all information...")
    results = extract_all_qwen(model, processor, images[0])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "drawing_name": TARGET_PDF,
        "model": "Qwen2-VL 7B (FP16)",
        "extraction_date": datetime.now().isoformat(),
        "extraction_categories": list(results.keys()),
        "results": results
    }
    
    output_file = QWEN_OUTPUT / f"qwen2vl_full_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Saved: {output_file}")
    
    # Summary
    print("\n=== EXTRACTION SUMMARY ===")
    for key in results.keys():
        text = results[key]
        if "ASSISTANT:" in text or "Answer:" in text:
            text = text.split("Answer:")[-1].strip()
            text = text.split("ASSISTANT:")[-1].strip()
        lines = [l for l in text.split("\n") if l.strip()]
        print(f"   {key}: {len(lines)} lines")

if __name__ == "__main__":
    main()