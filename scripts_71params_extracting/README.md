# 71 Parameters Extraction - Scripts

## Overview
This folder contains scripts to extract all 71 parameters from the target CAD drawing using 5 different VLM models, and compare the results against ground truth.

## Target PDF
- **File**: `HP_58231-82P00_500_s_SUZUKI_DRAW_SH1.pdf`
- **Location**: `input_pdfs/`

## Scripts Execution Order

### Step 1: Extract Ground Truth
```bash
python scripts_71params_extracting/0_extract_ground_truth.py
```
Extracts ground truth parameters from `DrawingInputSheet.xlsx`

### Step 2: Run VLM Models (in order)
```bash
# EasyOCR - Fast, CPU compatible (~2GB VRAM)
python scripts_71params_extracting/1_test_easyocr.py

# LLaVA 1.5 7B - Needs ~6GB VRAM
python scripts_71params_extracting/2_test_llava.py

# BLIP2 OPT 2.7B - Needs ~4GB VRAM
python scripts_71params_extracting/3_test_blip2.py

# Moondream2 - Needs ~2GB VRAM
python scripts_71params_extracting/4_test_moondream.py

# Qwen2-VL 7B - Needs ~8GB VRAM
python scripts_71params_extracting/5_test_qwen2vl.py
```

### Step 3: Generate Comparison Report
```bash
python scripts_71params_extracting/6_compare_all.py
```

## Output Files
Results are saved to `outputs_71params/` folder:
- `ground_truth/` - Ground truth parameters
- `easyocr_results/` - EasyOCR extraction results
- `llava_results/` - LLaVA extraction results
- `blip2_results/` - BLIP2 extraction results
- `moondream_results/` - Moondream2 extraction results
- `qwen2vl_results/` - Qwen2-VL extraction results
- `comparison_reports/` - Final comparison Excel report

## GPU Memory Requirements
| Model | VRAM Needed | Skipped if |
|-------|-------------|------------|
| EasyOCR | ~2GB | N/A (CPU OK) |
| Moondream2 | ~2GB | < 3GB free |
| BLIP2 | ~4GB | < 5GB free |
| LLaVA 1.5 7B | ~6GB | < 8GB free |
| Qwen2-VL 7B | ~8GB | < 10GB free |

## Parameters Extracted

### Part A: Our 71 Parameters (What's visible in drawing)
- Geometry_Design (8): holes, fillets, chamfers, ribs
- Dimensions (8): size, length, diameter, thickness
- Assembly (4): fit changes, fasteners
- Tolerances (4): tolerance values
- GD&T (8): position, flatness, runout, etc.
- Views_Drafting (18): view changes, scales
- Notes_Annotations (8): general notes, welding
- TitleBlock_Metadata (7): revision, drawing number
- BOM (4): bill of materials
- Scale (2): sheet/view scale

### Part B: Ground Truth Parameters (CAD settings)
- Sheet_Settings (4): width, height, scale
- Drafting_Settings (5): text size, arrow size
- View_Settings (21): view positions, scales
- Dimension_Settings (3): linear dimension settings

### Parameter Mapping
Some 71 parameters can be compared with ground truth:
- #35 (View scale change) ↔ View_Top_ScaleNumerator
- #70 (Sheet Scale) ↔ Sheet_ScaleNumerator
- #33 (View changes) ↔ No_of_Views
- etc.

## Running on GPU-enabled System (Colab/Server)
These scripts are designed to run on any GPU-enabled system. Just ensure:
1. PyTorch with CUDA is installed
2. transformers and other dependencies are installed
3. GPU has enough VRAM for the model you want to run