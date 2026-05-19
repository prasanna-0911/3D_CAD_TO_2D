# ============================================================================
# 3_test_blip2.py - BLIP2 Vision Language Model Testing
# ============================================================================
"""
BLIP2 OPT 2.7B testing script with FIXED decoding.

CRITICAL FIX: The original script had a bug where it decoded input tokens
along with output tokens, causing the prompt to be returned as output.
FIX: Slice generated_ids[:, input_len:] to get only new tokens.
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


def init_blip2():
    """Initialize BLIP2 with OPT 2.7B."""
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    model_config = MODELS_CONFIG["blip2_opt_2.7b"]

    print(f"   Loading {model_config['name']}...")

    # Load model (FP16, no quantization needed for this size)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_config["model_id"],
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load processor
    processor = Blip2Processor.from_pretrained(model_config["model_id"])

    print("   [OK] BLIP2 OPT 2.7B loaded")

    return model, processor


def extract_with_blip2(model, processor, image_path: str):
    """
    Extract information using BLIP2.

    NOTE: BLIP2 is designed for SHORT VQA questions, not long prompts.
    We send one short question per task for best results.
    """
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    # Resize for memory efficiency
    if max(image.size) > 1024:
        ratio = 1024 / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Short, focused questions (not long prompts)
    questions = [
        ("overall", "What type of drawing is this?"),
        ("dimensions", "What are the dimension values?"),
        ("part_name", "What is the part name?"),
        ("scale", "What is the drawing scale?"),
    ]

    results = {}

    for task_name, question in questions:
        try:
            # Prepare inputs
            inputs = processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to("cuda", torch.float16)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=3
                )

            # [OK] FIXED DECODING: Slice off input tokens
            input_len = inputs["input_ids"].shape[1]
            answer = processor.batch_decode(
                generated_ids[:, input_len:],  # Only new tokens!
                skip_special_tokens=True
            )[0].strip()

            results[task_name] = answer

        except Exception as e:
            results[task_name] = f"ERROR: {str(e)}"

        # Clear memory
        del inputs, generated_ids
        free_gpu_memory()

    return results


def save_results(drawing_name: str, extraction: dict, output_dir: Path):
    """Save BLIP2 extraction results."""
    ensure_dir(output_dir)

    total_chars = sum(len(str(v)) for v in extraction.values())

    summary = {
        "drawing_name": drawing_name,
        "model": "BLIP2 OPT 2.7B",
        "extraction_date": datetime.now().isoformat(),
        "total_chars_extracted": total_chars,
        "tasks_completed": len(extraction)
    }

    # JSON
    json_path = output_dir / f"{drawing_name}_blip2.json"
    save_json({**summary, "extraction": extraction}, json_path)

    # Excel
    excel_data = []
    for task, answer in extraction.items():
        excel_data.append({
            "Task": task,
            "Question": task.replace("_", " ").title(),
            "Extracted Text": answer,
            "Length": len(str(answer))
        })

    excel_path = output_dir / f"{drawing_name}_blip2.xlsx"
    save_excel(excel_data, excel_path, "BLIP2 Results")

    return summary


def process_all_drawings():
    """Process all PDF drawings with BLIP2."""
    print_header("BLIP2 OPT 2.7B - Vision Language Model Testing")
    print_config()

    # Check GPU
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print(f"\nGPU: {gpu_info['device']} ({gpu_info['free_gb']}GB free)")
    else:
        print("\n[WARNING] No GPU - BLIP2 requires CUDA")

    # Find PDFs
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n[ERROR] No PDFs in {INPUT_DIR}")
        return

    print(f"\n[PDF] Found {len(pdf_files)} PDFs")

    # Initialize BLIP2
    print("\n[LOADING] Loading BLIP2...")
    model, processor = init_blip2()

    output_dir = OUTPUT_DIR / "blip2_results"
    ensure_dir(output_dir)

    all_results = []
    print("\n" + "=" * 60)

    for pdf_path in tqdm(pdf_files, desc="Processing"):
        drawing_name = pdf_path.stem
        print(f"\n[PDF] {drawing_name}")

        # Convert to image
        image = pdf_to_image(
            pdf_path,
            zoom=PDF_DPI_SETTINGS["vlm_small"],
            max_size=1024  # BLIP2 needs smaller images
        )

        temp_path = output_dir / f"{drawing_name}_temp.png"
        image.save(temp_path, "PNG")

        # Extract
        print(f"   Extracting...")
        extraction = extract_with_blip2(model, processor, str(temp_path))

        for task, result in extraction.items():
            preview = result[:60] + "..." if len(result) > 60 else result
            print(f"      {task}: {preview}")

        # Save
        summary = save_results(drawing_name, extraction, output_dir)
        all_results.append(summary)

        temp_path.unlink(missing_ok=True)
        free_gpu_memory()

    # Combined
    combined_path = output_dir / "ALL_DRAWINGS_blip2_summary.json"
    save_json({
        "model": "BLIP2 OPT 2.7B",
        "extraction_date": datetime.now().isoformat(),
        "total_drawings": len(all_results),
        "results": all_results
    }, combined_path)

    print("\n[OK] BLIP2 testing complete!")
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