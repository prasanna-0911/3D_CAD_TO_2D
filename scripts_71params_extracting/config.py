"""
config.py - Configuration for 71 Parameters Extraction Project
================================================================
Contains:
- Our 71 parameters (Part A: What's visible in drawing)
- Ground truth parameters (Part B: CAD settings)
- Mapping between them
"""

import os
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = PROJECT_ROOT / "input_pdfs"
OUTPUT_DIR = PROJECT_ROOT / "outputs_71params"

# Target PDF
TARGET_PDF = "HP_58231-82P00_500_s_SUZUKI_DRAW_SH1.pdf"
GROUND_TRUTH_FILE = "DrawingInputSheet.xlsx"

# Create directories
for dir_path in [OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(PROJECT_ROOT / "models_cache")
os.environ["TRANSFORMERS_CACHE"] = str(PROJECT_ROOT / "models_cache")

# ============================================================================
# PART A: OUR 71 PARAMETERS (What's VISIBLE in drawing)
# ============================================================================
PARAMETERS_71 = {
    "Geometry_Design": [
        {"id": "1", "name": "Geometry & Design Changes"},
        {"id": "2", "name": "Shape modifications"},
        {"id": "3", "name": "Suppressed/unsuppressed features"},
        {"id": "4", "name": "Coordinate system shifts"},
        {"id": "5", "name": "Holes Feature"},
        {"id": "6", "name": "Fillet Feature"},
        {"id": "7", "name": "Chamfers Feature"},
        {"id": "8", "name": "Ribs Feature"},
    ],
    "Dimensions": [
        {"id": "9", "name": "Size changes"},
        {"id": "10", "name": "Length"},
        {"id": "11", "name": "Diameter"},
        {"id": "12", "name": "Thickness"},
        {"id": "13", "name": "Position changes"},
        {"id": "14", "name": "Location changes of features"},
        {"id": "19", "name": "Dimension position changes"},
        {"id": "22", "name": "Dimension value changes"},
    ],
    "Assembly": [
        {"id": "15", "name": "Assembly fit/interface changes"},
        {"id": "16", "name": "Exploded view differences"},
        {"id": "17", "name": "Sub-assembly changes"},
        {"id": "18", "name": "Fastener type/size updates"},
    ],
    "Tolerances": [
        {"id": "20", "name": "Tolerances change"},
        {"id": "21", "name": "Tolerances Location change"},
        {"id": "23", "name": "Tolerance updates"},
        {"id": "24", "name": "Fits (H7/g6)"},
    ],
    "GD_T": [
        {"id": "25", "name": "Position tolerance"},
        {"id": "26", "name": "Straightness"},
        {"id": "27", "name": "Flatness"},
        {"id": "28", "name": "Circularity"},
        {"id": "29", "name": "Parallelism"},
        {"id": "30", "name": "Perpendicularity"},
        {"id": "31", "name": "Angularity"},
        {"id": "32", "name": "Runout"},
    ],
    "Views_Drafting": [
        {"id": "33", "name": "View changes"},
        {"id": "34", "name": "View add/delete"},
        {"id": "35", "name": "View scale change"},
        {"id": "36", "name": "View representation change"},
        {"id": "37", "name": "Drafting view"},
        {"id": "38", "name": "Section line"},
        {"id": "39", "name": "Hatching change"},
        {"id": "40", "name": "Line type change"},
        {"id": "41", "name": "Fake dimension"},
        {"id": "42", "name": "Dimension arrow change"},
        {"id": "43", "name": "Dimension line thickness change"},
        {"id": "44", "name": "Dimension position change"},
        {"id": "45", "name": "View position changes"},
        {"id": "46", "name": "Leader line change"},
        {"id": "47", "name": "Additional curve creation"},
        {"id": "48", "name": "Section view shift"},
        {"id": "49", "name": "Broken view shift"},
        {"id": "50", "name": "Additional import photo/graphics"},
    ],
    "Notes_Annotations": [
        {"id": "51", "name": "General notes updates"},
        {"id": "52", "name": "Notes Location"},
        {"id": "53", "name": "Special instructions added/removed"},
        {"id": "54", "name": "Flag notes / key characteristics"},
        {"id": "55", "name": "Special symbols add/delete"},
        {"id": "56", "name": "Welding symbols"},
        {"id": "57", "name": "Datum changes"},
        {"id": "58", "name": "Datum Location changes"},
    ],
    "TitleBlock_Metadata": [
        {"id": "59", "name": "Revision number"},
        {"id": "60", "name": "Drawing number"},
        {"id": "61", "name": "Part name changes"},
        {"id": "62", "name": "Author"},
        {"id": "63", "name": "Checker/approver updates"},
        {"id": "64", "name": "Dates"},
        {"id": "65", "name": "Company or project info"},
    ],
    "BOM": [
        {"id": "66", "name": "BOM (Bill of Materials)"},
        {"id": "67", "name": "Component added/removed"},
        {"id": "68", "name": "Part number revisions"},
        {"id": "69", "name": "Part list"},
    ],
    "Scale": [
        {"id": "70", "name": "Sheet Scale changes"},
        {"id": "71", "name": "View Scale changes"},
    ]
}

ALL_PARAMS_71 = {p["id"]: p for cat in PARAMETERS_71.values() for p in cat}

# ============================================================================
# PART B: GROUND TRUTH PARAMETERS (CAD Settings from DrawingInputSheet)
# ============================================================================
GROUND_TRUTH_PARAMS = {
    "Sheet_Settings": [
        {"id": "GT_1", "name": "Sheet_Width", "value": 1189},
        {"id": "GT_2", "name": "Sheet_Height", "value": 841},
        {"id": "GT_3", "name": "Sheet_ScaleNumerator", "value": 1},
        {"id": "GT_4", "name": "Sheet_ScaleDenominator", "value": 1},
    ],
    "Drafting_Settings": [
        {"id": "GT_5", "name": "Drafting_TextSize", "value": 5},
        {"id": "GT_6", "name": "Drafting_ArrowSize", "value": 2.5},
        {"id": "GT_7", "name": "Drafting_LeaderStubSize", "value": 1.25},
        {"id": "GT_8", "name": "Drafting_DimensionPrecision", "value": 2},
        {"id": "GT_9", "name": "Drafting_AnglePrecision", "value": 2},
    ],
    "View_Settings": [
        {"id": "GT_10", "name": "No_of_Views", "value": 6},
        {"id": "GT_11", "name": "View_Top_Enable", "value": 1},
        {"id": "GT_12", "name": "View_Top_X", "value": 493.6},
        {"id": "GT_13", "name": "View_Top_Y", "value": 705.7},
        {"id": "GT_14", "name": "View_Top_ScaleNumerator", "value": 1},
        {"id": "GT_15", "name": "View_Top_ScaleDenominator", "value": 1},
        {"id": "GT_16", "name": "View_Front_Enable", "value": 1},
        {"id": "GT_17", "name": "View_Front_X", "value": 493.6},
        {"id": "GT_18", "name": "View_Front_Y", "value": 521.9},
        {"id": "GT_19", "name": "View_Right_Enable", "value": 1},
        {"id": "GT_20", "name": "View_Right_X", "value": 1079.8},
        {"id": "GT_21", "name": "View_Right_Y", "value": 705.7},
        {"id": "GT_22", "name": "View_D2_Enable", "value": 1},
        {"id": "GT_23", "name": "View_D2_ScaleNumerator", "value": 1},
        {"id": "GT_24", "name": "View_D2_ScaleDenominator", "value": 2},
        {"id": "GT_25", "name": "View_D5_Enable", "value": 1},
        {"id": "GT_26", "name": "View_D5_ScaleNumerator", "value": 1},
        {"id": "GT_27", "name": "View_D5_ScaleDenominator", "value": 2},
        {"id": "GT_28", "name": "View_Detail_Enable", "value": 1},
        {"id": "GT_29", "name": "View_Detail_X", "value": 900},
        {"id": "GT_30", "name": "View_Detail_Y", "value": 350},
    ],
    "Dimension_Settings": [
        {"id": "GT_31", "name": "Linear_Enable", "value": 1},
        {"id": "GT_32", "name": "Linear_TextSize", "value": 5},
        {"id": "GT_33", "name": "Linear_ArrowSize", "value": 2.5},
    ],
}

ALL_GT_PARAMS = {p["id"]: p for cat in GROUND_TRUTH_PARAMS.values() for p in cat}

# ============================================================================
# MAPPING: Our 71 Params ↔ Ground Truth Params
# ============================================================================
PARAM_MAPPING = {
    # View scale changes
    "35": ["GT_14", "GT_15", "GT_23", "GT_24", "GT_26", "GT_27"],  # View scale change
    "70": ["GT_3", "GT_4"],  # Sheet Scale changes
    
    # View changes
    "33": ["GT_10"],  # View changes
    "34": ["GT_11", "GT_16", "GT_19", "GT_22", "GT_25", "GT_28"],  # View add/delete
    
    # View position changes
    "45": ["GT_12", "GT_13", "GT_17", "GT_18", "GT_20", "GT_21", "GT_29", "GT_30"],  # View position changes
    
    # Dimension related
    "10": ["GT_31", "GT_32"],  # Length -> Linear settings
    "11": ["GT_31", "GT_32"],  # Diameter -> Linear settings
    
    # Notes
    "51": ["GT_5"],  # General notes -> Drafting_TextSize
    "52": ["GT_5"],  # Notes Location -> Drafting_TextSize
}

# ============================================================================
# VLM EXTRACTION PROMPTS
# ============================================================================
VLM_PROMPTS = {
    "dimensions": "Extract all dimension values: linear, diameter, radius with units (mm). List each dimension value found.",
    "gdt": "Extract GD&T symbols: position, flatness, straightness, perpendicularity, parallelism with values.",
    "title_block": "Extract title block: part name, drawing number, revision, scale, units, date, author.",
    "views": "Identify views: front, top, side, section, detail. Note their positions and scales.",
    "notes": "Extract all notes, annotations, warnings. Include general notes, surface finish, welding symbols.",
    "geometry": "Identify geometric features: holes, fillets, chamfers, ribs. Note their sizes and locations.",
    "tolerances": "Extract tolerances: bilateral, unilateral, fits (H7/g6), positional tolerances with values.",
    "gd_symbols": "Extract geometric dimensioning symbols: position, flatness, straightness, circularity, runout.",
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
MODELS_CONFIG = {
    "easyocr": {
        "enabled": True,
        "name": "EasyOCR",
        "vram_estimate_gb": 2,
    },
    "llava_1.5_7b": {
        "enabled": True,
        "name": "LLaVA 1.5 7B",
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "quantization": "4bit",
        "vram_estimate_gb": 6,
    },
    "blip2_opt_2.7b": {
        "enabled": True,
        "name": "BLIP2 OPT 2.7B",
        "model_id": "Salesforce/blip2-opt-2.7b",
        "vram_estimate_gb": 4,
    },
    "moondream2": {
        "enabled": True,
        "name": "Moondream2",
        "model_id": "vikhyatk/moondream2",
        "vram_estimate_gb": 2,
    },
    "qwen2_vl_7b": {
        "enabled": True,
        "name": "Qwen2-VL 7B",
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "quantization": "none",  # FP16 non-quantized
        "vram_estimate_gb": 14,
    },
}

def print_config():
    """Print configuration summary."""
    print("=" * 60)
    print("71 Parameters Extraction - Configuration Loaded")
    print("=" * 60)
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Input folder   : {INPUT_DIR}")
    print(f"Output folder  : {OUTPUT_DIR}")
    print(f"Target PDF    : {TARGET_PDF}")
    print(f"Our 71 params  : {len(ALL_PARAMS_71)}")
    print(f"Ground truth   : {len(ALL_GT_PARAMS)}")
    print(f"Mapped params : {len(PARAM_MAPPING)}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()