# CAD VLM Extraction Project

A complete VS Code setup for testing Vision Language Models (VLMs) and OCR on CAD engineering drawings. Extracts dimensional information, GD&T symbols, title block data, and more from 2D PDF drawings.

## Project Overview

This project tests multiple AI models for extracting information from CAD engineering drawings. The goal is to find the best approach for automated 2D drawing analysis as part of a 3D to 2D CAD conversion pipeline.

### Models Tested

| Model | Type | VRAM | Best For |
|-------|------|------|----------|
| **EasyOCR** | OCR | ~2GB | Ground truth text extraction |
| **LLaVA 1.5 7B** | VLM | ~6GB | General understanding |
| **BLIP2 OPT 2.7B** | VLM | ~4GB | Short Q&A |
| **Moondream2** | VLM | ~2GB | Technical images |
| **Qwen2-VL 7B** | VLM | ~8GB | Technical documents |

### Key Finding

**EasyOCR outperforms all VLMs** for CAD text extraction because it reads actual pixel text rather than hallucinating values. VLMs like LLaVA tend to generate fabricated dimension values instead of reading actual drawing content.

## Quick Start

### 1. Setup Environment

```batch
# Run setup script (one-time only)
setup.bat
```

This will:
- Create virtual environment (conda or venv)
- Install PyTorch with CUDA
- Install all dependencies
- Verify GPU detection

### 2. Add Your PDF Drawings

Copy your CAD drawing PDFs to the `input_pdfs/` folder:

```
input_pdfs/
├── HP_V1__Final Shaft_2d.pdf
├── HP_V2__Final Shaft_2d.pdf
├── HP_V1__bolt.pdf
├── HP_V2__bolt_2d.pdf
├── HP_Before__1st__NOVEX-GS 414 UFS DA-1__2d.pdf
└── HP_After__1st__NOVEX-GS 414 UFS DA-1__2d.pdf
```

### 3. Run Tests

```batch
# Option A: Run all tests
run_all_tests.bat

# Option B: Run individual tests
python scripts\1_test_easyocr.py   # Ground truth OCR
python scripts\2_test_llava.py     # LLaVA 1.5 7B
python scripts\3_test_blip2.py     # BLIP2
python scripts\4_test_moondream.py # Moondream2
python scripts\5_test_qwen2vl.py  # Qwen2-VL (needs 8GB+ VRAM)
```

### 4. View Results

Results are saved to `outputs/` folder:

```
outputs/
├── easyocr_results/           # EasyOCR extraction results
├── llava_results/             # LLaVA results
├── blip2_results/            # BLIP2 results
├── moondream_results/        # Moondream2 results
├── qwen2vl_results/          # Qwen2-VL results
└── comparison_reports/       # Comparison Excel files
```

## Project Structure

```
3D_CAD_TO_2D/
├── input_pdfs/               # Your PDF drawings here
├── outputs/                  # Extraction results
├── models_cache/             # Downloaded models (40GB+)
├── scripts/
│   ├── config.py            # Central configuration
│   ├── utils.py             # Helper functions
│   ├── 1_test_easyocr.py    # EasyOCR extraction
│   ├── 2_test_llava.py      # LLaVA 1.5 7B
│   ├── 3_test_blip2.py      # BLIP2 (fixed decoding)
│   ├── 4_test_moondream.py  # Moondream2
│   ├── 5_test_qwen2vl.py    # Qwen2-VL 7B
│   └── 6_compare_all.py     # Generate comparison report
├── requirements.txt          # Python dependencies
├── setup.bat               # Environment setup
├── run_all_tests.bat       # Run all tests
└── README.md               # This file
```

## 71 Parameters Extracted

Based on Geometry.docx, the following parameter categories are defined:

| Category | Parameters | Description |
|----------|------------|-------------|
| Geometry_Design | 1-8 | Holes, fillets, chamfers, ribs |
| Dimensions | 9-14, 19, 22 | Size, length, diameter, position |
| Assembly | 15-18 | Fit changes, fasteners |
| Tolerances | 20-21, 23-24 | Tolerance values, fits |
| GD&T | 25-32 | Position, flatness, runout, etc. |
| Views_Drafting | 33-50 | View changes, section lines |
| Notes_Annotations | 51-58 | General notes, welding symbols |
| TitleBlock_Metadata | 59-65 | Revision, drawing number |
| BOM | 66-69 | Bill of materials |
| Scale | 70-71 | Sheet/view scale |

## System Requirements

### Minimum
- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: 16GB
- **GPU**: NVIDIA with 6GB+ VRAM (RTX 3060 or better)
- **Storage**: 50GB free space
- **Python**: 3.9, 3.10, or 3.11

### Recommended
- **GPU**: RTX 4060 or better with 8GB+ VRAM
- **RAM**: 32GB
- **Storage**: 100GB free space

### GPU Check

```powershell
# Windows
nvidia-smi

# Should show GPU name and VRAM
```

## Troubleshooting

### Out of Memory (OOM) Errors

1. **Reduce image size**: Scripts auto-resize, but you can lower `MAX_IMAGE_SIZE` in config.py
2. **Use quantization**: LLaVA and Qwen2-VL use 4-bit/8-bit quantization
3. **Skip heavy models**: Qwen2-VL needs 8GB+ VRAM

### Model Download Issues

Models are cached in `models_cache/` folder. If downloads fail:

```bash
# Clear cache and retry
rmdir /s models_cache
python scripts\1_test_easyocr.py  # Will re-download
```

### GPU Not Detected

```python
# Test in Python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

If False:
1. Install NVIDIA drivers
2. Install CUDA Toolkit
3. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Known Issues

1. **Florence-2 Config Error**: transformers version mismatch. Use EasyOCR instead.
2. **PaddleOCR Abandoned**: OneDNN/PIR bugs on Colab. Use EasyOCR instead.
3. **VLMs Hallucinate**: LLaVA, BLIP2, Moondream2 may generate fake values. EasyOCR is reliable.

## Running on Google Colab

If local GPU is insufficient, run on Google Colab:

1. Upload scripts to Colab
2. Use T4 GPU (15GB VRAM)
3. Models order by VRAM needed:
   - EasyOCR → Moondream2 → BLIP2 → LLaVA → Qwen2-VL

## Output Format

### JSON Output
```json
{
  "drawing_name": "HP_V1__Final Shaft_2d",
  "model": "EasyOCR",
  "extraction_date": "2026-05-11T...",
  "total_text_elements": 40,
  "by_category": {
    "Dimensions_Tolerances": 11,
    "GD_T": 0,
    "Views_Drafting": 2,
    "TitleBlock_Metadata": 22,
    "Uncategorized": 5
  }
}
```

### Excel Output
- Per-drawing sheets with extracted values
- Category columns for EasyOCR
- Task-based columns for VLMs

## License

This is an academic project for CAD automation research.

## Contact

For issues or questions, refer to the chat history in:
- `claude_chat_3dto3d.txt`
- `chat789465132.pdf`
- `web_page_text.txt`