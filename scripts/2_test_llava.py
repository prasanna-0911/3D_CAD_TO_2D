# ============================================================================
# 2_test_llava.py - LLaVA 1.5 7B Vision Language Model Testing
# ============================================================================
"""
LLaVA 1.5 7B testing script for CAD engineering drawings.
Uses 4-bit quantization to fit in 6-8GB VRAM.

IMPORTANT: LLaVA tends to hallucinate values on CAD drawings.
This is documented - do not expect perfect accuracy.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import (
    INPUT_DIR, OUTPUT_DIR, MODELS_CACHE_DIR,
    PDF_DPI_SETTINGS, MAX_IMAGE_SIZE_VLM, MODELS_CONFIG,
    print_config, get_gpu_info
)
from scripts.utils import (
    pdf_to_image, save_json, save_excel, print_header, print_model_status,
    get_gpu_memory_info, free_gpu_memory, sanitize_filename, ensure_dir
)


def init_llava():
    """Initialize LLaVA 1.5 7B with 4-bit quantization."""
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from transformers import BitsAndBytesConfig

    model_config = MODELS_CONFIG["llava_1.5_7b"]

    print(f"   Loading {model_config['name']}...")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_config["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_config["model_id"])

    print("   [OK] LLaVA 1.5 7B loaded (4-bit)")

    return model, processor


def extract_with_llava(model, processor, image_path: str):
    """Extract information using LLaVA."""
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    # Resize if too large
    if max(image.size) > MAX_IMAGE_SIZE_VLM:
        ratio = MAX_IMAGE_SIZE_VLM / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    questions = [
        ("dimensions", "What are all the dimension values visible in this engineering drawing? List every number with units (mm)."),
        ("gd_t", "What GD&T symbols are present? List position, flatness, perpendicularity, etc. with values."),
        ("title_block", "What is in the title block? Give part name, drawing number, revision, scale."),
        ("views", "What views are shown? (front, top, side, section, detail)"),
        ("notes", "What notes or annotations are visible?"),
    ]

    results = {}

    for task_name, question in questions:
        try:
            # Format prompt
            prompt = f"User: <image>\n{question}\nAssistant:"

            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            # Decode only new tokens
            input_len = inputs['input_ids'].shape[1]
            answer = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

            results[task_name] = answer.strip()

        except Exception as e:
            results[task_name] = f"ERROR: {str(e)}"

        # Clear memory
        free_gpu_memory()

    return results


def save_results(drawing_name: str, extraction: dict, output_dir: Path):
    """Save extraction results."""
    ensure_dir(output_dir)

    # Count total characters extracted
    total_chars = sum(len(str(v)) for v in extraction.values())

    summary = {
        "drawing_name": drawing_name,
        "model": "LLaVA 1.5 7B (4-bit)",
        "extraction_date": datetime.now().isoformat(),
        "total_chars_extracted": total_chars,
        "tasks_completed": len(extraction)
    }

    # Save JSON
    json_data = {
        **summary,
        "extraction": extraction
    }

    json_path = output_dir / f"{drawing_name}_llava.json"
    save_json(json_data, json_path)

    # Save Excel
    excel_data = []
    for task, answer in extraction.items():
        excel_data.append({
            "Task": task,
            "Extracted Text": answer,
            "Length": len(str(answer))
        })

    excel_path = output_dir / f"{drawing_name}_llava.xlsx"
    save_excel(excel_data, excel_path, "LLaVA Results")

    return summary


def process_all_drawings():
    """Process all PDF drawings."""
    print_header("LLaVA 1.5 7B - CAD Drawing VLM Testing")
    print_config()

    # Check GPU
    gpu_info = get_gpu_info()
    if not gpu_info["available"]:
        print("\n[WARNING] No GPU detected - LLaVA will be very slow on CPU")
    else:
        print(f"\nGPU: {gpu_info['device']} ({gpu_info['free_gb']}GB free)")

    # Find PDFs
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n[ERROR] No PDFs in {INPUT_DIR}")
        return

    print(f"\n[PDF] Found {len(pdf_files)} PDFs")
    for pdf in pdf_files:
        print(f"   • {pdf.name}")

    # Initialize LLaVA
    print("\n[LOADING] Loading LLaVA 1.5 7B...")
    model, processor = init_llava()

    # Output directory
    output_dir = OUTPUT_DIR / "llava_results"
    ensure_dir(output_dir)

    # Process drawings
    all_results = []
    print("\n" + "=" * 60)

    for pdf_path in tqdm(pdf_files, desc="Processing drawings"):
        drawing_name = pdf_path.stem
        print(f"\n[PDF] {drawing_name}")

        # Convert to image
        image = pdf_to_image(
            pdf_path,
            zoom=PDF_DPI_SETTINGS["vlm_large"],
            max_size=MAX_IMAGE_SIZE_VLM
        )

        temp_path = output_dir / f"{drawing_name}_temp.png"
        image.save(temp_path, "PNG")

        # Extract
        print(f"   Extracting with LLaVA...")
        extraction = extract_with_llava(model, processor, str(temp_path))

        for task, result in extraction.items():
            preview = result[:80] + "..." if len(result) > 80 else result
            print(f"      {task}: {preview}")

        # Save
        summary = save_results(drawing_name, extraction, output_dir)
        all_results.append(summary)

        # Cleanup
        temp_path.unlink(missing_ok=True)

        # Memory management
        free_gpu_memory()

    # Combined results
    print("\n" + "=" * 60)
    print("[STATS] Summary")

    combined_path = output_dir / "ALL_DRAWINGS_llava_summary.json"
    combined_data = {
        "model": "LLaVA 1.5 7B",
        "extraction_date": datetime.now().isoformat(),
        "total_drawings": len(all_results),
        "results": all_results
    }
    save_json(combined_data, combined_path)

    print("\n[OK] LLaVA testing complete!")
    print(f"   Results: {output_dir}")

    return all_results


if __name__ == "__main__":
    try:
        process_all_drawings()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interrupted")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()