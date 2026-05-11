# ============================================================================
# FINAL COMPARISON SCRIPT - Compare All Models
# ============================================================================
"""
Comprehensive Model Comparison - All 5 Models
=============================================
Creates comparison Excel with:
1. Parameter detection matrix
2. Value extraction accuracy
3. Category-wise comparison

Run after all models have been tested.
"""

# ============================================================================
# CELL 1 - MOUNT DRIVE
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

os.chdir("/content/drive/MyDrive/3D_CAD_TO_2D")

# ============================================================================
# CELL 2 - DEFINE ALL 71 PARAMETERS
# ============================================================================
ALL_PARAMETERS = {
    "Geometry_Design": [1, 2, 3, 4, 5, 6, 7, 8],
    "Dimensions": [9, 10, 11, 12, 13, 14, 19, 22],
    "Assembly": [15, 16, 17, 18],
    "Tolerances": [20, 21, 23, 24],
    "GD_T": [25, 26, 27, 28, 29, 30, 31, 32],
    "Views_Drafting": list(range(33, 51)),
    "Notes_Annotations": list(range(51, 59)),
    "TitleBlock_Metadata": list(range(59, 66)),
    "BOM": [66, 67, 68, 69],
    "Scale": [70, 71]
}

# ============================================================================
# CELL 3 - FIND ALL RESULTS
# ============================================================================
output_base = Path("outputs")
models = ["easyocr_results", "llava_results", "blip2_results", "moondream_results", "qwen2vl_results"]

results = {}

for model_dir in models:
    model_name = model_dir.replace("_results", "").upper()
    model_path = output_base / model_dir

    if not model_path.exists():
        print(f"⚠️ {model_name} results not found: {model_path}")
        continue

    # Find JSON files (exclude summary files)
    json_files = list(model_path.glob("*_results.json")) + list(model_path.glob("*_*.json"))

    # Filter out summary files
    json_files = [f for f in json_files if "ALL_" not in f.name and "summary" not in f.name.lower()]

    if json_files:
        print(f"✅ {model_name}: {len(json_files)} drawings found")
        results[model_name] = [str(f) for f in json_files]
    else:
        print(f"❌ {model_name}: No result files found")

# ============================================================================
# CELL 4 - CREATE PARAMETER DETECTION MATRIX
# ============================================================================
print("\n📊 Creating Parameter Detection Matrix...")

detection_matrix = []

for drawing_name in ["HP_V1__Final Shaft_2d", "HP_V2__Final Shaft_2d", "HP_V1__bolt", "HP_V2__bolt_2d",
                      "HP_Before__1st__NOVEX-GS 414 UFS DA-1__2d", "HP_After__1st__NOVEX-GS 414 UFS DA-1__2d"]:

    row = {"Drawing": drawing_name}

    for model_name, files in results.items():
        # Find matching file
        matching_file = None
        for f in files:
            if drawing_name in f:
                matching_file = f
                break

        if matching_file:
            try:
                with open(matching_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Count detected parameters
                total_elements = data.get("total_elements", 0) or data.get("total_text_elements", 0)
                row[f"{model_name}_Elements"] = total_elements
                row[f"{model_name}_Detected"] = "Yes" if total_elements > 0 else "No"
            except:
                row[f"{model_name}_Elements"] = 0
                row[f"{model_name}_Detected"] = "Error"
        else:
            row[f"{model_name}_Elements"] = 0
            row[f"{model_name}_Detected"] = "No file"

    detection_matrix.append(row)

df_detection = pd.DataFrame(detection_matrix)

# ============================================================================
# CELL 5 - CREATE CATEGORY COMPARISON
# ============================================================================
print("📊 Creating Category Comparison...")

category_data = []

for drawing_name in detection_matrix:
    drawing = drawing_name["Drawing"]

    for model_name, files in results.items():
        matching_file = None
        for f in files:
            if drawing in f:
                matching_file = f
                break

        if matching_file:
            try:
                with open(matching_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Get category counts
                if "category_counts" in data:
                    for cat, count in data["category_counts"].items():
                        category_data.append({
                            "Drawing": drawing,
                            "Model": model_name,
                            "Category": cat,
                            "Count": count
                        })
                elif "by_category" in data:
                    for cat, count in data["by_category"].items():
                        category_data.append({
                            "Drawing": drawing,
                            "Model": model_name,
                            "Category": cat,
                            "Count": count
                        })
            except:
                pass

df_categories = pd.DataFrame(category_data)

# ============================================================================
# CELL 6 - SUMMARY STATISTICS
# ============================================================================
print("📊 Creating Summary Statistics...")

summary_data = []

for model_name, files in results.items():
    total_elements = 0
    drawings_processed = len(files)

    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
            total_elements += data.get("total_elements", 0) or data.get("total_text_elements", 0)
        except:
            pass

    summary_data.append({
        "Model": model_name,
        "Drawings_Processed": drawings_processed,
        "Total_Elements_Extracted": total_elements,
        "Avg_Elements_Per_Drawing": round(total_elements / drawings_processed, 1) if drawings_processed > 0 else 0
    })

df_summary = pd.DataFrame(summary_data)

# ============================================================================
# CELL 7 - SAVE COMPARISON REPORT
# ============================================================================
comparison_dir = Path("outputs/comparison_reports")
comparison_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = comparison_dir / f"MODEL_COMPARISON_{timestamp}.xlsx"

with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
    df_detection.to_excel(writer, sheet_name="Parameter_Detection", index=False)
    df_categories.to_excel(writer, sheet_name="Category_Comparison", index=False)
    df_summary.to_excel(writer, sheet_name="Summary_Statistics", index=False)

print(f"\n✅ Comparison report saved: {report_path}")

# ============================================================================
# CELL 8 - PRINT COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

print("\n📊 Summary Statistics:")
print(df_summary.to_string(index=False))

print("\n\n📊 Parameter Detection by Drawing:")
print(df_detection.to_string(index=False))

print("\n" + "="*80)

# Also create a pivot table for category comparison
if len(df_categories) > 0:
    print("\n📊 Category-wise Comparison:")
    pivot = df_categories.pivot_table(
        index="Category",
        columns="Model",
        values="Count",
        aggfunc="sum",
        fill_value=0
    )
    print(pivot.to_string())

print("\n" + "="*80)
print("✅ Comparison complete!")
print(f"   Report: {report_path}")
print("="*80)