@echo off
REM ============================================================================
REM run_all_tests.bat - Run All Model Tests Sequentially
REM ============================================================================
REM This script runs all 5 model test scripts in order.
REM Results are saved to outputs/ folder.
REM ============================================================================

echo ============================================================
echo   CAD VLM Extraction - Running All Tests
echo ============================================================
echo.

REM Check if venv is activated
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Virtual environment not activated!
    echo Please run: venv\Scripts\activate.bat
    echo Or: conda activate cad_vlm
    pause
    exit /b 1
)

echo [OK] Environment ready
echo.

REM Check for PDFs
dir /b input_pdfs\*.pdf >nul 2>&1
if errorlevel 1 (
    echo [WARNING] No PDF files found in input_pdfs\
    echo Please copy your PDF drawings there first!
    echo.
    set /p CONTINUE=Continue anyway? (y/n):
    if /i not "%CONTINUE%"=="y" exit /b 1
)
echo.

REM Create logs directory
if not exist "logs" mkdir logs

echo ============================================================
echo   Starting Model Tests
echo ============================================================
echo.

REM Test 1: EasyOCR (Ground Truth)
echo [1/5] Running EasyOCR (Ground Truth)...
echo ----------------------------------------
python scripts\1_test_easyocr.py > logs\easyocr_log.txt 2>&1
if errorlevel 1 (
    echo [ERROR] EasyOCR failed. Check logs\easyocr_log.txt
) else (
    echo [OK] EasyOCR complete
)
echo.

REM Test 2: LLaVA 1.5 7B
echo [2/5] Running LLaVA 1.5 7B...
echo ----------------------------------------
python scripts\2_test_llava.py > logs\llava_log.txt 2>&1
if errorlevel 1 (
    echo [ERROR] LLaVA failed. Check logs\llava_log.txt
) else (
    echo [OK] LLaVA complete
)
echo.

REM Test 3: BLIP2
echo [3/5] Running BLIP2...
echo ----------------------------------------
python scripts\3_test_blip2.py > logs\blip2_log.txt 2>&1
if errorlevel 1 (
    echo [ERROR] BLIP2 failed. Check logs\blip2_log.txt
) else (
    echo [OK] BLIP2 complete
)
echo.

REM Test 4: Moondream2
echo [4/5] Running Moondream2...
echo ----------------------------------------
python scripts\4_test_moondream.py > logs\moondream_log.txt 2>&1
if errorlevel 1 (
    echo [ERROR] Moondream failed. Check logs\moondream_log.txt
) else (
    echo [OK] Moondream complete
)
echo.

REM Test 5: Qwen2-VL (if VRAM allows)
echo [5/5] Running Qwen2-VL...
echo ----------------------------------------
python scripts\5_test_qwen2vl.py > logs\qwen2vl_log.txt 2>&1
if errorlevel 1 (
    echo [WARNING] Qwen2-VL may have skipped (insufficient VRAM)
) else (
    echo [OK] Qwen2-VL complete
)
echo.

REM Generate Comparison
echo ============================================================
echo   Generating Comparison Report
echo ============================================================
python scripts\6_compare_all.py > logs\comparison_log.txt 2>&1
if errorlevel 1 (
    echo [ERROR] Comparison failed
) else (
    echo [OK] Comparison complete
)
echo.

REM Summary
echo ============================================================
echo   All Tests Complete!
echo ============================================================
echo.
echo Results saved to:
echo   outputs\easyocr_results\
echo   outputs\llava_results\
echo   outputs\blip2_results\
echo   outputs\moondream_results\
echo   outputs\qwen2vl_results\
echo   outputs\comparison_reports\
echo.
echo Logs saved to:
echo   logs\
echo.
echo ============================================================
pause