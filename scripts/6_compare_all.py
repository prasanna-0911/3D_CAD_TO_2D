# ============================================================================
# 6_compare_all.py - Compare All Model Results
# ============================================================================
"""
Comparison script to analyze and compare results from all models.
Creates a comprehensive comparison Excel with:
1. Summary table
2. Per-drawing comparison
3. Category-wise accuracy analysis
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import OUTPUT_DIR, print_config
from scripts.utils import (
    print_header, print_model_status, ensure_dir, save_json, save_excel,
    sanitize_filename
)
import pandas as pd


def find_result_files():
    """Find all result JSON files from each model."""
    result_files = {
        "EasyOCR": list(OUTPUT_DIR.glob("easyocr_results/*_easyocr.json")),
        "LLaVA": list(OUTPUT_DIR.glob("llava_results/*_llava.json")),
        "BLIP2": list(OUTPUT_DIR.glob("blip2_results/*_blip2.json")),
        "Moondream": list(OUTPUT_DIR.glob("moondream_results/*_moondream.json")),
        "Qwen2-VL": list(OUTPUT_DIR.glob("qwen2vl_results/*_qwen2vl.json")),
    }

    # Filter out summary files
    for model, files in result_files.items():
        result_files[model] = [f for f in files if "summary" not in f.name and "ALL" not in f.name]

    return result_files


def load_results(result_files):
    """Load all results into a structured format."""
    all_results = {}

    for model, files in result_files.items():
        if not files:
            print(f"   No results for {model}")
            continue

        model_results = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    drawing_name = file_path.stem.replace(f"_{model.lower()}", "").replace("_easyocr", "").replace("_llava", "").replace("_blip2", "").replace("_moondream", "").replace("_qwen2vl", "")
                    model_results.append({
                        "drawing": drawing_name,
                        "data": data
                    })
            except Exception as e:
                print(f"   Error loading {file_path}: {e}")

        if model_results:
            all_results[model] = model_results

    return all_results


def create_summary_table(all_results):
    """Create summary comparison table."""
    summary_data = []

    # Get all drawings
    all_drawings = set()
    for model, results in all_results.items():
        for r in results:
            all_drawings.add(r["drawing"])

    for drawing in sorted(all_drawings):
        row = {"Drawing": drawing}

        for model, results in all_results.items():
            # Find this drawing's result
            match = next((r for r in results if r["drawing"] == drawing), None)
            if match:
                data = match["data"]
                # Extract key metrics
                if "total_text_elements" in data:
                    row[f"{model}_Elements"] = data["total_text_elements"]
                    row[f"{model}_Categories"] = len(data.get("by_category", {}))
                else:
                    row[f"{model}_Chars"] = data.get("total_chars_extracted", 0)
                    row[f"{model}_Tasks"] = data.get("tasks_completed", 0)

        summary_data.append(row)

    return summary_data


def create_detailed_comparison(all_results):
    """Create detailed per-drawing comparison."""
    comparison_data = []

    # Get all drawings
    all_drawings = set()
    for model, results in all_results.items():
        for r in results:
            all_drawings.add(r["drawing"])

    for drawing in sorted(all_drawings):
        for model, results in all_results.items():
            match = next((r for r in results if r["drawing"] == drawing), None)
            if match:
                data = match["data"]

                if "elements" in data:
                    # EasyOCR format
                    comparison_data.append({
                        "Drawing": drawing,
                        "Model": model,
                        "Total Elements": data.get("total_text_elements", 0),
                        "Dimensions": data.get("by_category", {}).get("Dimensions_Tolerances", 0),
                        "GD&T": data.get("by_category", {}).get("GD_T", 0),
                        "Views": data.get("by_category", {}).get("Views_Drafting", 0),
                        "Title Block": data.get("by_category", {}).get("TitleBlock_Metadata", 0),
                        "Uncategorized": data.get("by_category", {}).get("Uncategorized", 0),
                    })
                else:
                    # VLM format
                    comparison_data.append({
                        "Drawing": drawing,
                        "Model": model,
                        "Total Chars": data.get("total_chars_extracted", 0),
                        "Tasks Completed": data.get("tasks_completed", 0),
                        "Extraction Date": data.get("extraction_date", ""),
                    })

    return comparison_data


def create_accuracy_analysis(all_results):
    """Analyze accuracy of each model (comparing to EasyOCR)."""
    # Use EasyOCR as ground truth
    if "EasyOCR" not in all_results:
        return []

    easyocr_results = {r["drawing"]: r["data"] for r in all_results["EasyOCR"]}

    accuracy_data = []

    for model, results in all_results.items():
        if model == "EasyOCR":
            continue

        for r in results:
            drawing = r["drawing"]
            data = r["data"]

            if drawing not in easyocr_results:
                continue

            easyocr = easyocr_results[drawing]
            easyocr_dims = easyocr.get("by_category", {}).get("Dimensions_Tolerances", 0)

            if "total_text_elements" in data:
                vlm_dims = data.get("by_category", {}).get("Dimensions_Tolerances", 0)
            else:
                vlm_dims = 0  # VLMs don't categorize like OCR

            accuracy_data.append({
                "Drawing": drawing,
                "Model": model,
                "EasyOCR Dimensions": easyocr_dims,
                "VLM Dimensions": vlm_dims,
                "Ratio": round(vlm_dims / easyocr_dims, 2) if easyocr_dims > 0 else "N/A"
            })

    return accuracy_data


def generate_report():
    """Generate comprehensive comparison report."""
    print_header("Model Comparison Report")
    print_config()

    # Find result files
    print("\n📂 Scanning for results...")
    result_files = find_result_files()

    available_models = [m for m, files in result_files.items() if files]
    print(f"   Found results for: {', '.join(available_models)}")

    if not available_models:
        print("\n❌ No results found! Run the test scripts first.")
        print("\n   Run in order:")
        print("   1. python scripts/1_test_easyocr.py")
        print("   2. python scripts/2_test_llava.py")
        print("   3. python scripts/3_test_blip2.py")
        print("   4. python scripts/4_test_moondream.py")
        print("   5. python scripts/5_test_qwen2vl.py")
        return

    # Load all results
    print("\n📊 Loading results...")
    all_results = load_results(result_files)

    # Create output directory
    report_dir = OUTPUT_DIR / "comparison_reports"
    ensure_dir(report_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = report_dir / f"MODEL_COMPARISON_{timestamp}.xlsx"

    print("\n📈 Generating comparison...")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Summary sheet
        summary = create_summary_table(all_results)
        if summary:
            df_summary = pd.DataFrame(summary)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # Detailed comparison
        detailed = create_detailed_comparison(all_results)
        if detailed:
            df_detailed = pd.DataFrame(detailed)
            df_detailed.to_excel(writer, sheet_name="Detailed_Comparison", index=False)

        # Accuracy analysis
        accuracy = create_accuracy_analysis(all_results)
        if accuracy:
            df_accuracy = pd.DataFrame(accuracy)
            df_accuracy.to_excel(writer, sheet_name="Accuracy_Analysis", index=False)

    print(f"\n✅ Report saved: {excel_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON SUMMARY")
    print("=" * 60)

    for model, results in all_results.items():
        total_elements = 0
        for r in results:
            data = r["data"]
            if "total_text_elements" in data:
                total_elements += data["total_text_elements"]
            else:
                total_elements += data.get("total_chars_extracted", 0)

        print(f"   {model}: {len(results)} drawings, {total_elements} total elements")

    print("\n" + "=" * 60)

    # Save JSON summary
    json_summary = {
        "generated": datetime.now().isoformat(),
        "models_tested": available_models,
        "total_drawings": len(set(r["drawing"] for results in all_results.values() for r in results)),
        "summary": {
            model: {
                "drawings_tested": len(results),
                "total_elements": sum(
                    r["data"].get("total_text_elements", 0) or r["data"].get("total_chars_extracted", 0)
                    for r in results
                )
            }
            for model, results in all_results.items()
        }
    }

    json_path = report_dir / f"COMPARISON_SUMMARY_{timestamp}.json"
    save_json(json_summary, json_path)

    print(f"✅ JSON summary: {json_path}")

    return excel_path


if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()