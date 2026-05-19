"""
6_compare_all.py - Compare All Models Results
==============================================
Compares extraction results from all models and generates comparison report.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts_71params_extracting.config import (
    OUTPUT_DIR, TARGET_PDF, PARAMETERS_71, ALL_PARAMS_71, PARAM_MAPPING
)
from scripts_71params_extracting.utils import (
    print_header, ensure_dir, save_json, save_excel
)


def find_result_files():
    """Find all result JSON files from each model."""
    result_files = {
        "EasyOCR": list(OUTPUT_DIR.glob("easyocr_results/*_71params.json")),
        "LLaVA": list(OUTPUT_DIR.glob("llava_results/*_71params.json")),
        "BLIP2": list(OUTPUT_DIR.glob("blip2_results/*_71params.json")),
        "Moondream": list(OUTPUT_DIR.glob("moondream_results/*_71params.json")),
        "Qwen2-VL": list(OUTPUT_DIR.glob("qwen2vl_results/*_71params.json")),
    }
    
    for model, files in result_files.items():
        result_files[model] = [f for f in files if "summary" not in f.name]
    
    return result_files


def load_model_results(result_files):
    """Load results from all models."""
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
                    model_results.append(data)
            except Exception as e:
                print(f"   Error loading {file_path}: {e}")
        
        if model_results:
            all_results[model] = model_results
    
    return all_results


def create_comparison_table(all_results):
    """Create comparison table across all models."""
    comparison_data = []
    
    for model, results in all_results.items():
        for result in results:
            params_71 = result.get("params_71", {})
            
            detected_count = sum(1 for p in params_71.values() if p["detected"])
            total_count = len(params_71)
            
            category_counts = {}
            for param_id, param_info in params_71.items():
                cat = param_info["category"]
                if cat not in category_counts:
                    category_counts[cat] = {"detected": 0, "total": 0}
                category_counts[cat]["total"] += 1
                if param_info["detected"]:
                    category_counts[cat]["detected"] += 1
            
            row = {
                "Model": model,
                "Total_Detected": detected_count,
                "Total_Params": total_count,
                "Detection_Rate_%": round((detected_count / total_count) * 100, 2)
            }
            
            for cat in PARAMETERS_71.keys():
                row[f"{cat}_Detected"] = category_counts.get(cat, {}).get("detected", 0)
                row[f"{cat}_Total"] = category_counts.get(cat, {}).get("total", 0)
            
            comparison_data.append(row)
    
    return comparison_data


def create_detailed_comparison(all_results):
    """Create detailed per-parameter comparison."""
    detailed_data = []
    
    for model, results in all_results.items():
        for result in results:
            params_71 = result.get("params_71", {})
            
            for param_id, param_info in params_71.items():
                detailed_data.append({
                    "Model": model,
                    "Parameter_ID": param_id,
                    "Parameter_Name": param_info["name"],
                    "Category": param_info["category"],
                    "Detected": "Yes" if param_info["detected"] else "No"
                })
    
    return detailed_data


def create_category_summary(all_results):
    """Create summary by parameter category."""
    category_summary = []
    
    for category, params in PARAMETERS_71.items():
        for model, results in all_results.items():
            detected_in_cat = 0
            total_in_cat = len(params)
            
            for result in results:
                params_71 = result.get("params_71", {})
                for param in params:
                    if param["id"] in params_71:
                        if params_71[param["id"]]["detected"]:
                            detected_in_cat += 1
            
            category_summary.append({
                "Category": category,
                "Model": model,
                "Detected": detected_in_cat,
                "Total": total_in_cat,
                "Detection_Rate_%": round((detected_in_cat / total_in_cat) * 100, 2)
            })
    
    return category_summary


def create_mapping_comparison(all_results):
    """Create comparison for mapped parameters (71 params ↔ Ground Truth)."""
    mapping_data = []
    
    for model, results in all_results.items():
        for result in results:
            params_71 = result.get("params_71", {})
            
            for our_param_id, gt_params in PARAM_MAPPING.items():
                if our_param_id in params_71:
                    our_param = params_71[our_param_id]
                    
                    for gt_id in gt_params:
                        mapping_data.append({
                            "Model": model,
                            "Our_Param_ID": our_param_id,
                            "Our_Param_Name": our_param["name"],
                            "GT_Param_ID": gt_id,
                            "Our_Detected": "Yes" if our_param["detected"] else "No",
                            "Can_Compare": "Yes"
                        })
    
    return mapping_data


def generate_report():
    """Generate comprehensive comparison report."""
    print_header("Model Comparison Report - 71 Parameters")
    
    print("\n📂 Scanning for results...")
    result_files = find_result_files()
    
    available_models = [m for m, files in result_files.items() if files]
    print(f"   Found results for: {', '.join(available_models)}")
    
    if not available_models:
        print("\n❌ No results found! Run the test scripts first.")
        print("\n   Run in order:")
        print("   1. python scripts_71params_extracting/0_extract_ground_truth.py")
        print("   2. python scripts_71params_extracting/1_test_easyocr.py")
        print("   3. python scripts_71params_extracting/2_test_llava.py")
        print("   4. python scripts_71params_extracting/3_test_blip2.py")
        print("   5. python scripts_71params_extracting/4_test_moondream.py")
        print("   6. python scripts_71params_extracting/5_test_qwen2vl.py")
        return
    
    print("\n📊 Loading results...")
    all_results = load_model_results(result_files)
    
    report_dir = OUTPUT_DIR / "comparison_reports"
    ensure_dir(report_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = report_dir f"COMPARISON_71PARAMS_{timestamp}.xlsx"
    
    print("\n📈 Generating comparison...")
    
    import pandas as pd
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_table = create_comparison_table(all_results)
        if summary_table:
            df_summary = pd.DataFrame(summary_table)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
        
        detailed_table = create_detailed_comparison(all_results)
        if detailed_table:
            df_detailed = pd.DataFrame(detailed_table)
            df_detailed.to_excel(writer, sheet_name="Detailed_Comparison", index=False)
        
        category_table = create_category_summary(all_results)
        if category_table:
            df_category = pd.DataFrame(category_table)
            df_category.to_excel(writer, sheet_name="By_Category", index=False)
        
        mapping_table = create_mapping_comparison(all_results)
        if mapping_table:
            df_mapping = pd.DataFrame(mapping_table)
            df_mapping.to_excel(writer, sheet_name="Mapping_Comparison", index=False)
    
    print(f"\n✅ Report saved: {excel_path}")
    
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    summary_table = create_comparison_table(all_results)
    for row in summary_table:
        print(f"   {row['Model']}: {row['Detected_Out_of']} ({row['Detection_Rate_%']}%)")
    
    print("\n" + "=" * 60)
    
    json_summary = {
        "generated": datetime.now().isoformat(),
        "models_tested": available_models,
        "summary": {
            model: {
                "detected": row["Total_Detected"],
                "total": row["Total_Params"],
                "rate": row["Detection_Rate_%"]
            }
            for row in summary_table
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