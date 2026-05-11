@echo off
REM ============================================================================
REM setup.bat - CAD VLM Extraction Project Setup
REM ============================================================================
REM This script sets up the complete Python environment for the project.
REM Run this once after cloning/downloading the project.
REM ============================================================================

echo ============================================================
echo   CAD VLM Extraction - VS Code Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ first.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check for conda
where conda >nul 2>&1
if errorlevel 1 (
    echo [INFO] Anaconda not found. Using system Python.
    set USE_CONDA=0
) else (
    echo [OK] Anaconda found
    set USE_CONDA=1
)
echo.

REM Create virtual environment
echo [STEP 1] Creating virtual environment...
if "%USE_CONDA%"=="1" (
    echo Using Anaconda to create environment...
    conda create -n cad_vlm python=3.10 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment
        pause
        exit /b 1
    )
    echo Activating environment...
    call conda activate cad_vlm
) else (
    echo Using venv to create environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
    echo Activating environment...
    call venv\Scripts\activate.bat
)
echo.

REM Upgrade pip
echo [STEP 2] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 echo [WARNING] pip upgrade had issues, continuing...
echo.

REM Install PyTorch
echo [STEP 3] Installing PyTorch with CUDA support...
echo This may take 5-10 minutes on first run...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [WARNING] PyTorch installation had issues.
    echo Try: pip install torch torchvision torchaudio
)
echo.

REM Install requirements
echo [STEP 4] Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)
echo.

REM Verify GPU
echo [STEP 5] Checking GPU...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo.

REM Create folders
echo [STEP 6] Creating project folders...
if not exist "input_pdfs" mkdir input_pdfs
if not exist "outputs" mkdir outputs
if not exist "models_cache" mkdir models_cache
if not exist "logs" mkdir logs
if not exist "scripts" mkdir scripts
echo.

REM Verify EasyOCR
echo [STEP 7] Verifying EasyOCR...
python -c "import easyocr; print('EasyOCR version:', easyocr.__version__)"
echo.

echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Copy your PDF drawings to: input_pdfs\
echo   2. Run a test: python scripts\1_test_easyocr.py
echo   3. Run comparison: python scripts\6_compare_all.py
echo.
echo To activate environment later:
if "%USE_CONDA%"=="1" (
    echo   conda activate cad_vlm
) else (
    echo   venv\Scripts\activate.bat
)
echo.
pause