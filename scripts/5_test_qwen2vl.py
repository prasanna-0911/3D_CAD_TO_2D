# ============================================================================
# 5_test_qwen2vl.py - Qwen2-VL 7B Vision Language Model Testing
# ============================================================================
"""
Qwen2-VL 7B testing script with 8-bit quantization.
Requires ~8GB VRAM - skip if not available.

Qwen2-VL is a powerful model specifically good at technical documents,
but needs careful memory management.
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
    get_gpu_memory_info, free_gpu_memory, ensure_dir, can_load_model
)


def init_qwen2vl():
    """Initialize Qwen2-VL with 8-bit quantization."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig

    model_config = MODELS_CONFIG["qwen2_vl_7b"]

    print(f"   Loading {model_config['name']}...")

    # Check if we have enough VRAM
    if not can_load_model(model_config["vram_estimate_gb"], safety_margin_gb=2.0):
        print(f"\n⚠️ Warning: May not have enough VRAM ({model_config['vram_estimate_gb']}GB needed)")
        print("   Trying anyway with 8-bit quantization...")

    # 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_config["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        processor = AutoProcessor.from_pretrained(
            model_config["model_id"],
            trust_remote_code=True
        )

        print("   ✅ Qwen2-VL 7B loaded (8-bit)")

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated(0) / 1e9
            print(f"   VRAM used: {vram_used:.2f} GB")

    except Exception as e:
        print(f"❌ Failed to load with 8-bit: {e}")
        print("   Try running on Google Colab with T4 GPU instead.")
        raise

    return model, processor


def extract_with_qwen2vl(model, processor, image_path: str):
    """
    Extract information using Qwen2-VL.
    Uses chat template for better results.
    """
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    # Resize if needed (Qwen2-VL handles up to ~1400px well)
    max_dim = 1400
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    image.save(image_path)  # Save for Qwen2-VL processing

    questions = [
        ("all_text", """Read and list ALL text visible in this engineering drawing:
- All dimension values with units
- All labels and annotations
- Title block information
- Scale and date information
- Any special notes or specifications

List everything exactly as written."""),

        ("dimensions", """Extract ALL dimensional information:
1. Linear dimensions: all length measurements with values
2. Diameter dimensions: all Ø values
3. Tolerances: all ± specifications
4. Radius dimensions: all R values
5. Angular dimensions: all degree measurements

Give exact numeric values."""),

        ("gdt", """Identify all GD&T (Geometric Dimensioning and Tolerancing) symbols:
- Position tolerance symbols
- Flatness, straightness, circularity
- Parallelism, perpendicularity, angularity
- Datum references (A, B, C in boxes)
- Tolerance values for each

List all found with their values."""),

        ("title_block", """Extract title block information:
- Part name/title
- Drawing number
- Revision number
- Scale
- Date(s)
- Designer/drafter
- Company name
- Units (mm or inches)"""),
    ]

    results = {}

    for task_name, question in questions:
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question}
                ]
            }]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False
                )

            output_text = processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            results[task_name] = output_text.strip()

        except Exception as e:
            results[task_name] = f"ERROR: {str(e)}"

        free_gpu_memory()

    return results


def save_results(drawing_name: str, extraction: dict, output_dir: Path):
    """Save Qwen2-VL results."""
    ensure_dir(output_dir)

    total_chars = sum(len(str(v)) for v in extraction.values()) if extraction else 0

    summary = {
        "drawing_name": drawing_name,
        "model": "Qwen2-VL 7B (8-bit)",
        "extraction_date": datetime.now().isoformat(),
        "total_chars_extracted": total_chars,
        "tasks_completed": len(extraction)
    }

    json_path = output_dir / f"{drawing_name}_qwen2vl.json"
    save_json({**summary, "extraction": extraction}, json_path)

    excel_data = []
    for task, answer in extraction.items():
        excel_data.append({
            "Task": task,
            "Extracted Text": answer,
            "Length": len(str(answer))
        })

    excel_path = output_dir / f"{drawing_name}_qwen2vl.xlsx"
    save_excel(excel_data, excel_path, "Qwen2-VL Results")

    return summary


def process_all_drawings():
    """Process all PDF drawings with Qwen2-VL."""
    print_header("Qwen2-VL 7B - Vision Language Model Testing")
    print_config()

    gpu_info = get_gpu_info()
    if not gpu_info["available"]:
        print("\n❌ Qwen2-VL requires GPU. Run on Google Colab instead.")
        return

    print(f"\nGPU: {gpu_info['device']} ({gpu_info['free_gb']}GB free)")

    if gpu_info["free_gb"] < 8:
        print("\n⚠️ Less than 8GB VRAM - Qwen2-VL may crash.")
        print("   Consider running on Google Colab with T4 GPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ No PDFs in {INPUT_DIR}")
        return

    print(f"\n📄 Found {len(pdf_files)} PDFs")

    print("\n📦 Loading Qwen2-VL 7B...")
    model, processor = init_qwen2vl()

    output_dir = OUTPUT_DIR / "qwen2vl_results"
    ensure_dir(output_dir)

    all_results = []
    print("\n" + "=" * 60)

    for pdf_path in tqdm(pdf_files, desc="Processing"):
        drawing_name = pdf_path.stem
        print(f"\n📄 {drawing_name}")

        image = pdf_to_image(
            pdf_path,
            zoom=PDF_DPI_SETTINGS["vlm_large"],
            max_size=1400
        )

        temp_path = output_dir / f"{drawing_name}_temp.png"
        image.save(temp_path, "PNG")

        print(f"   Extracting...")
        extraction = extract_with_qwen2vl(model, processor, str(temp_path))

        for task, result in extraction.items():
            preview = result[:80] + "..." if len(result) > 80 else result
            print(f"      {task}: {preview}")

        summary = save_results(drawing_name, extraction, output_dir)
        all_results.append(summary)

        temp_path.unlink(missing_ok=True)
        free_gpu_memory()

    combined_path = output_dir / "ALL_DRAWINGS_qwen2vl_summary.json"
    save_json({
        "model": "Qwen2-VL 7B (8-bit)",
        "extraction_date": datetime.now().isoformat(),
        "total_drawings": len(all_results),
        "results": all_results
    }, combined_path)

    print("\n✅ Qwen2-VL testing complete!")
    print(f"   Results: {output_dir}")

    return all_results


if __name__ == "__main__":
    try:
        process_all_drawings()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()