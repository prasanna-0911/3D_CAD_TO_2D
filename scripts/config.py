# ============================================================================
# config.py - Central Configuration for CAD VLM Extraction
# ============================================================================
"""
CAD VLM Extraction Project Configuration
==========================================
This file contains all central configuration settings for the project.
"""

import os
import sys
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = PROJECT_ROOT / "input_pdfs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_CACHE_DIR = PROJECT_ROOT / "models_cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [INPUT_DIR, OUTPUT_DIR, MODELS_CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ── Model cache environment variables ────────────────────────────────────
os.environ["HF_HOME"] = str(MODELS_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_CACHE_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ── GPU/CPU detection ────────────────────────────────────────────────────
def get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_gpu_info():
    """Get GPU information."""
    import torch
    if not torch.cuda.is_available():
        return {"available": False, "device": "CPU"}

    props = torch.cuda.get_device_properties(0)
    mem_total = props.total_memory / 1e9
    mem_allocated = torch.cuda.memory_allocated(0) / 1e9

    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "total_memory_gb": round(mem_total, 1),
        "allocated_gb": round(mem_allocated, 1),
        "free_gb": round(mem_total - mem_allocated, 1)
    }

DEVICE = get_device()
GPU_INFO = get_gpu_info()

# ── Image processing settings ─────────────────────────────────────────────
PDF_DPI_SETTINGS = {
    "easyocr": 3.0,      # 3x zoom for OCR (216 DPI)
    "vlm_small": 1.2,    # For BLIP2, Moondream (moderate)
    "vlm_large": 1.0,    # For LLaVA, Qwen2-VL (memory intensive)
}

MAX_IMAGE_SIZE_VLM = 1280   # Max dimension in pixels for VLMs
MAX_IMAGE_SIZE_OCR = 2048   # Max dimension for OCR

# ── Model configurations ────────────────────────────────────────────────────
MODELS_CONFIG = {
    "easyocr": {
        "enabled": True,
        "name": "EasyOCR",
        "languages": ['en'],
        "gpu": GPU_INFO["available"],
        "vram_estimate_gb": 2,
        "zoom": PDF_DPI_SETTINGS["easyocr"],
        "max_new_tokens": None,  # OCR doesn't use tokens
    },

    "llava_1.5_7b": {
        "enabled": True,
        "name": "LLaVA 1.5 7B",
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "quantization": "4bit",
        "vram_estimate_gb": 6,
        "zoom": PDF_DPI_SETTINGS["vlm_large"],
        "max_new_tokens": 512,
    },

    "qwen2_vl_7b": {
        "enabled": True,
        "name": "Qwen2-VL 7B",
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "quantization": "8bit",
        "vram_estimate_gb": 8,
        "zoom": PDF_DPI_SETTINGS["vlm_large"],
        "max_new_tokens": 1024,
    },

    "blip2_opt_2.7b": {
        "enabled": True,
        "name": "BLIP2 OPT 2.7B",
        "model_id": "Salesforce/blip2-opt-2.7b",
        "quantization": None,  # FP16
        "vram_estimate_gb": 4,
        "zoom": PDF_DPI_SETTINGS["vlm_small"],
        "max_new_tokens": 256,
    },

    "moondream2": {
        "enabled": True,
        "name": "Moondream2",
        "model_id": "vikhyatk/moondream2",
        "revision": "2024-08-26",
        "vram_estimate_gb": 2,
        "zoom": PDF_DPI_SETTINGS["vlm_small"],
        "max_new_tokens": 512,
    },
}

# ── 71 Parameters from Geometry.docx ──────────────────────────────────────
PARAMETER_EXTRACTION_SPEC = {
    "Geometry_Design": [
        "1. Geometry & Design Changes",
        "2. Shape modifications (profiles, contours)",
        "3. Suppressed/unsuppressed features",
        "4. Coordinate system shifts",
        "5. Holes Feature -- additions/removals",
        "6. Fillet Feature -- additions/removals",
        "7. Chamfers Feature -- additions/removals",
        "8. Ribs Feature -- additions/removals",
    ],

    "Dimensions": [
        "9. Size changes",
        "10. Length",
        "11. Diameter",
        "12. Thickness",
        "13. Position changes",
        "14. Location changes of features",
        "19. Dimension position changes",
        "22. Dimension value changes",
    ],

    "Assembly": [
        "15. Assembly fit/interface changes",
        "16. Exploded view differences",
        "17. Sub-assembly changes",
        "18. Fastener type/size updates",
    ],

    "Tolerances": [
        "20. Tolerances change",
        "21. Tolerances Location change",
        "23. Tolerance updates (bilateral/unilateral)",
        "24. Fits (H7/g6, etc.)",
    ],

    "GD_T": [
        "25. Position",
        "26. Straightness",
        "27. Flatness",
        "28. Circularity",
        "29. Parallelism",
        "30. Perpendicularity",
        "31. Angularity",
        "32. Runout",
    ],

    "Views_Drafting": [
        "33. View changes",
        "34. View add/delete",
        "35. View scale change",
        "36. View representation change",
        "37. Drafting view",
        "38. Section line",
        "39. Hatching change",
        "40. Line type change",
        "41. Fake dimension",
        "42. Dimension arrow change",
        "43. Dimension line thickness change",
        "44. Dimension position change",
        "45. View position changes",
        "46. Leader line change",
        "47. Additional curve creation for dimensioning",
        "48. Section view shift",
        "49. Broken view shift",
        "50. Additional import photo or graphics",
    ],

    "Notes_Annotations": [
        "51. General notes updates",
        "52. Notes Location",
        "53. Special instructions added/removed",
        "54. Flag notes / key characteristics",
        "55. Special symbols add/delete",
        "56. Welding symbols",
        "57. Datum changes",
        "58. Datum Location changes",
    ],

    "TitleBlock_Metadata": [
        "59. Revision number",
        "60. Drawing number",
        "61. Part name changes",
        "62. Author",
        "63. Checker/approver updates",
        "64. Dates",
        "65. Company or project info",
    ],

    "BOM": [
        "66. BOM (Bill of Materials)",
        "67. Component added/removed / Quantity changes",
        "68. Part number revisions",
        "69. Part list",
    ],

    "Scale": [
        "70. Sheet Scale changes",
        "71. View Scale changes / Vendor supplier updates",
    ],
}

# ── VLM Extraction prompts ─────────────────────────────────────────────────
VLM_EXTRACTION_PROMPTS = {
    "Dimensions": """Extract all dimension values from this engineering drawing:
- Linear/diameter/radius dimensions with values and units
- Tolerance values (+/- specifications)
- Angular dimensions with degree values

Return as JSON with keys: linear_dimensions, diameter_dimensions, radius_dimensions, tolerance_values, angular_dimensions.""",

    "GD_T": """Extract all GD&T (Geometric Dimensioning and Tolerancing) symbols:
- Position, flatness, straightness, circularity
- Parallelism, perpendicularity, angularity, runout
- Datum references (A, B, C, etc.)
- Tolerance values for each GD&T feature

Return as JSON with keys: gdt_symbols, datum_references, tolerance_values.""",

    "TitleBlock": """Extract all title block information:
- Part name, drawing number, revision
- Scale, units, date
- Designer, checker, approver
- Company name and logo text

Return as JSON with all extracted fields.""",

    "Views_Drafting": """Identify all views and drafting elements:
- View types (front, top, side, section, detail)
- View scales
- Section lines and hatching
- Dimension annotations and leaders

Return as JSON with keys: views, scales, section_views, dimensions.""",

    "Notes_Annotations": """Extract all notes, annotations, and special symbols:
- General notes and instructions
- Surface finish symbols
- Welding symbols
- Flag notes and key characteristics

Return as JSON with keys: general_notes, surface_finish, welding_symbols, flag_notes.""",

    "Geometry_Design": """Analyze geometric features shown in this drawing:
- Holes, fillets, chamfers mentioned in dimensions/annotations
- Feature types visible
- Shape modifications suggested

Return as JSON with keys: features_identified, feature_types.""",
}

# ── Summary ────────────────────────────────────────────────────────────────
def print_config():
    """Print configuration summary."""
    total_params = sum(len(p) for p in PARAMETER_EXTRACTION_SPEC.values())

    print("=" * 60)
    print("CAD VLM Extraction - Configuration Loaded")
    print("=" * 60)
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Input folder   : {INPUT_DIR}")
    print(f"Output folder  : {OUTPUT_DIR}")
    print(f"Models cache   : {MODELS_CACHE_DIR}")
    print(f"Device         : {DEVICE}")
    if GPU_INFO["available"]:
        print(f"GPU            : {GPU_INFO['device']}")
        print(f"VRAM           : {GPU_INFO['free_gb']}GB free / {GPU_INFO['total_memory_gb']}GB total")
    print(f"Parameters     : {total_params}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()