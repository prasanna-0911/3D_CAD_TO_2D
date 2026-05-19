# CAD VLM 71-Parameter Evaluation - Local PC Setup Guide

## Prerequisites Checklist

Before starting, make sure you have:
- [ ] Python 3.10 or 3.11 installed ([download](https://www.python.org/downloads/))
- [ ] NVIDIA GPU with CUDA support
- [ ] NVIDIA Driver installed (for CUDA)
- [ ] Git installed (optional, for cloning)

---

## Step 1: Check Python Installation

Open Command Prompt (Win+R, type `cmd`, press Enter):

```powershell
python --version
```

You should see `Python 3.10.x` or `Python 3.11.x`. If not, install Python from [python.org](https://www.python.org/downloads/).

**Important:** During installation, check **"Add Python to PATH"**.

---

## Step 2: Create Virtual Environment

Navigate to your project folder:

```powershell
cd C:\Users\ADMIN\Downloads\3D_CAD_TO_2D

# Create virtual environment
python -m venv cad_vlm_env
```

---

## Step 3: Activate Virtual Environment

```powershell
cad_vlm_env\Scripts\activate
```

You should see `(cad_vlm_env)` at the beginning of your command prompt.

---

## Step 4: Upgrade pip

```powershell
pip install --upgrade pip
```

---

## Step 5: Install PyTorch with CUDA

This is the most important step. Run this command:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8**, use:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 6: Install All Dependencies

Run this command to install all required packages:

```powershell
pip install transformers accelerate bitsandbytes pillow pymupdf pandas openpyxl tqdm easyocr pdf2image
```

**If you get errors**, try installing packages one by one:
```powershell
pip install transformers
pip install accelerate bitsandbytes
pip install pillow pymupdf pandas openpyxl tqdm
pip install easyocr
pip install pdf2image
```

---

## Step 7: Verify GPU Access

Test that Python can see your GPU:

```powershell
python
```

Then in Python:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
exit()
```

**Expected output:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX XXXX
VRAM: XX.X GB
```

---

## Step 8: Update Script Path (if needed)

Open `LOCAL_71Param_Test.py` in a text editor (Notepad++, VS Code, etc.)

Find line 17:
```python
WORKING_DIR = Path(r"C:\Users\ADMIN\Downloads\3D_CAD_TO_2D")
```

Change it to match your folder path (keep the `r` before the quotes for Windows paths).

---

## Step 9: Run the Test Script

Make sure your virtual environment is activated:
```powershell
cd C:\Users\ADMIN\Downloads\3D_CAD_TO_2D
cad_vlm_env\Scripts\activate
```

Run the script:
```powershell
python LOCAL_71Param_Test.py
```

---

## Expected Output

The script will:
1. Test EasyOCR on all PDFs
2. Test BLIP2 on all PDFs
3. Test LLaVA (if VRAM allows)
4. Test Qwen2-VL (if VRAM allows)
5. Create a comparison report

**Each model will take several minutes.** The total runtime depends on your GPU.

### Approximate Times:
| Model | Time per PDF | Total (6 PDFs) |
|-------|--------------|---------------|
| EasyOCR | 30 sec | 3 min |
| BLIP2 | 2 min | 12 min |
| LLaVA | 5 min | 30 min |
| Qwen2-VL | 8 min | 48 min |

---

## Output Files

After completion, you'll find:

```
outputs/
├── easyocr_results/
│   ├── HP_V1__Final Shaft_2d_easyocr.json
│   ├── HP_V1__Final Shaft_2d_easyocr.xlsx
│   └── ... (one per PDF)
├── blip2_results/
│   └── ... (one per PDF)
├── llava_results/
│   └── ... (one per PDF)
├── qwen_results/
│   └── ... (one per PDF)
└── comparison_reports/
    └── 71PARAM_COMPARISON_YYYYMMDD_HHMMSS.xlsx
```

---

## Troubleshooting

### "pip is not recognized"

Use `pip3` instead, or reinstall Python with "Add to PATH" checked.

### "No module named torch"

Make sure you activated the virtual environment:
```powershell
cad_vlm_env\Scripts\activate
```

### "CUDA out of memory"

This is expected for large models. Try:
- Closing other GPU applications
- The script will automatically skip models that need more VRAM

### "SSL certificate errors" when downloading models

Run this before installing:
```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org certifi
```

### EasyOCR slow on first run

EasyOCR downloads language models on first run. This is normal. Subsequent runs will be faster.

---

## Quick Reference Commands

```powershell
# Activate environment
cd C:\Users\ADMIN\Downloads\3D_CAD_TO_2D
cad_vlm_env\Scripts\activate

# Run test
python LOCAL_71Param_Test.py

# Deactivate environment when done
deactivate
```

---

## Need Help?

If you encounter errors, note:
1. The exact error message
2. Which step failed
3. Your GPU model and VRAM

And check:
- [PyTorch CUDA compatibility](https://pytorch.org/)
- [Transformers installation](https://huggingface.co/docs/transformers/installation)