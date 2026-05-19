# ============================================================================
# 4_test_moondream.py - Moondream2 Vision Language Model Testing
# ============================================================================
"""
Moondream2 testing script for CAD engineering drawings.
Moondream2 is lightweight (~2GB VRAM) and designed for technical images.
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
    pdf_to_image, save_json, save_excel, print_header,
    get_gpu_memory_info, free_gpu_memory, ensure_dir
)


def init_moondream():
    """Initialize Moondream2."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_config = MODELS_CONFIG["moondream2"]

    print(f"   Loading {model_config['name']}...")

    model = AutoModelForVision2Seq.from_pretrained(
        model_config["model_id"],
        revision=model_config["revision"],
        torch_dtype=torch.float16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_config["model_id"])

    print("   [OK] Moondream2 loaded")

    return model, processor


def extract_with_moondream(model, processor, image_path: str):
    """Extract information using Moondream2."""
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")

    # Resize if needed
    if max(image.size) > MAX_IMAGE_SIZE_VLM:
        ratio = MAX_IMAGE_SIZE_VLM / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    questions = [
        ("overall", "Describe this engineering drawing in detail."),
        ("dimensions", "List all dimension values with their units."),
        ("gd_t", "What GD&T symbols and tolerances are present?"),
        ("title_block", "What information is in the title block?"),
        ("views", "What views are shown and what is their scale?"),
    ]

    results = {}

    for task_name, question in questions:
        try:
            inputs = processor(
                text=question,
                images=image,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512
                )

            answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            results[task_name] = answer.strip()

        except Exception as e:
            results[task_name] = f"ERROR: {str(e)}"

        free_gpu_memory()

    return results


def save_results(drawing_name: str, extraction: dict, output_dir: Path):
    """Save Moondream2 results."""
    ensure_dir(output_dir)

    total_chars = sum(len(str(v)) for v in extraction.values()) if extraction else 0

    summary = {
        "drawing_name": drawing_name,
        "model": "Moondream2",
        "extraction_date": datetime.now().isoformat(),
        "total_chars_extracted": total_chars,
        "tasks_completed": len(extraction)
    }

    json_path = output_dir / f"{drawing_name}_moondream.json"
    save_json({**summary, "extraction": extraction}, json_path)

    excel_data = []
    for task, answer in extraction.items():
        excel_data.append({
            "Task": task,
            "Question": task.replace("_", " ").title(),
            "Extracted Text": answer,
            "Length": len(str(answer))
        })

    excel_path = output_dir / f"{drawing_name}_moondream.xlsx"
    save_excel(excel_data, excel_path, "Moondream2 Results")

    return summary


def process_all_drawings():
    """Process all PDF drawings with Moondream2."""
    print_header("Moondream2 - Vision Language Model Testing")
    print_config()

    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print(f"\nGPU: {gpu_info['device']} ({gpu_info['free_gb']}GB free)")
    else:
        print("\n[WARNING] No GPU - Moondream2 will be slow on CPU")

    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n[ERROR] No PDFs in {INPUT_DIR}")
        return

    print(f"\n[PDF] Found {len(pdf_files)} PDFs")

    print("\n[LOADING] Loading Moondream2...")
    model, processor = init_moondream()

    output_dir = OUTPUT_DIR / "moondream_results"
    ensure_dir(output_dir)

    all_results = []
    print("\n" + "=" * 60)

    for pdf_path in tqdm(pdf_files, desc="Processing"):
        drawing_name = pdf_path.stem
        print(f"\n[PDF] {drawing_name}")

        image = pdf_to_image(
            pdf_path,
            zoom=PDF_DPI_SETTINGS["vlm_small"],
            max_size=MAX_IMAGE_SIZE_VLM
        )

        temp_path = output_dir / f"{drawing_name}_temp.png"
        image.save(temp_path, "PNG")

        print(f"   Extracting...")
        extraction = extract_with_moondream(model, processor, str(temp_path))

        for task, result in extraction.items():
            preview = result[:60] + "..." if len(result) > 60 else result
            print(f"      {task}: {preview}")

        summary = save_results(drawing_name, extraction, output_dir)
        all_results.append(summary)

        temp_path.unlink(missing_ok=True)
        free_gpu_memory()

    combined_path = output_dir / "ALL_DRAWINGS_moondream_summary.json"
    save_json({
        "model": "Moondream2",
        "extraction_date": datetime.now().isoformat(),
        "total_drawings": len(all_results),
        "results": all_results
    }, combined_path)

    print("\n[OK] Moondream2 testing complete!")
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