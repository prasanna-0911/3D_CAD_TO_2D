"""
9_compare_comprehensive.py - Comprehensive Comparison
======================================================
Compare extracted values with ground truth side-by-side.
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

from config_comprehensive import (
    COMPREHENSIVE_PARAMS, ALL_COMPREHENSIVE_PARAMS,
    ALL_GT_PARAMS, PARAM_MAPPING
)
from config import OUTPUT_DIR

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
COMPREHENSIVE_OUTPUT = OUTPUT_DIR / "comprehensive_extraction"
COMPARISON_OUTPUT = OUTPUT_DIR / "comprehensive_comparison"

def load_extracted_values():
    """Load most recent extraction results."""
    files = list(COMPREHENSIVE_OUTPUT.glob("comprehensive_extraction_*.json"))
    if not files:
        return None
    
    latest = max(files, key=lambda x: x.stat().st_mtime)
    with open(latest, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_ground_truth():
    """Load ground truth from DrawingInputSheet.xlsx."""
    import pandas as pd
    
    gt_file = PROJECT_ROOT / "input_pdfs" / "DrawingInputSheet.xlsx"
    df = pd.read_excel(gt_file, sheet_name='Sheet1')
    
    gt_values = {}
    for _, row in df.iterrows():
        param_name = row.iloc[1]
        param_value = row.iloc[2]
        if pd.notna(param_name) and pd.notna(param_value):
            # Clean param name
            param_name = str(param_name).strip()
            # Convert value to appropriate type
            try:
                if isinstance(param_value, float):
                    gt_values[param_name] = param_value
                else:
                    gt_values[param_name] = param_value
            except:
                gt_values[param_name] = param_value
    
    return gt_values

def create_comparison_table(extracted_data: dict, gt_values: dict) -> pd.DataFrame:
    """Create side-by-side comparison table."""
    rows = []
    
    # 1. EXTRACTED VALUES (from PDF)
    parsed = extracted_data.get("parsed_values", {})
    
    # Dimensions
    dimensions = parsed.get("dimensions", [])
    for i, dim in enumerate(dimensions[:10], 1):
        rows.append({
            "Category": "Dimensions",
            "Parameter": f"Dimension_{i}",
            "Extracted_Value": dim,
            "Ground_Truth": "N/A (not in GT)",
            "Match": "N/A",
            "Source": "Extraction"
        })
    
    # GD&T Symbols
    gdt_symbols = parsed.get("gdt_symbols", [])
    for i, gdt in enumerate(gdt_symbols[:10], 1):
        rows.append({
            "Category": "GD&T",
            "Parameter": f"GD&T_{i}",
            "Extracted_Value": gdt,
            "Ground_Truth": "N/A (not in GT)",
            "Match": "N/A",
            "Source": "Extraction"
        })
    
    # Title Block
    title_block = parsed.get("title_block", {})
    title_mapping = {
        "Drawing Number": "Drawing Number",
        "Part Name": "Part Name",
        "Revision": "Revision",
        "Scale": "Scale",
        "Units": "Units",
    }
    
    for param, extracted_key in title_mapping.items():
        ext_val = title_block.get(extracted_key, "NOT FOUND")
        gt_val = gt_values.get(param, "N/A")
        
        match = "MATCH" if str(ext_val).strip() == str(gt_val).strip() else "NO MATCH"
        
        rows.append({
            "Category": "Title Block",
            "Parameter": param,
            "Extracted_Value": ext_val,
            "Ground_Truth": gt_val,
            "Match": match,
            "Source": "Extraction vs GT"
        })
    
    # Views
    views = parsed.get("views", [])
    gt_views = gt_values.get("No_of_Views", "N/A")
    rows.append({
        "Category": "Views",
        "Parameter": "Number of Views",
        "Extracted_Value": f"{len(views)} views found",
        "Ground_Truth": gt_views,
        "Match": "MATCH" if len(views) == gt_views else "NO MATCH",
        "Source": "Extraction vs GT"
    })
    
    for i, view in enumerate(views[:6], 1):
        rows.append({
            "Category": "Views",
            "Parameter": f"View_{i}",
            "Extracted_Value": view,
            "Ground_Truth": "N/A",
            "Match": "N/A",
            "Source": "Extraction"
        })
    
    # Notes
    notes = parsed.get("notes", [])
    for i, note in enumerate(notes[:5], 1):
        rows.append({
            "Category": "Notes",
            "Parameter": f"Note_{i}",
            "Extracted_Value": note[:100],
            "Ground_Truth": "N/A",
            "Match": "N/A",
            "Source": "Extraction"
        })
    
    # Geometry
    geometry = parsed.get("geometry", {})
    for feature, value in geometry.items():
        rows.append({
            "Category": "Geometry",
            "Parameter": f"{feature.capitalize()}",
            "Extracted_Value": value,
            "Ground_Truth": "N/A",
            "Match": "N/A",
            "Source": "Extraction"
        })
    
    # BOM
    bom = parsed.get("bom", [])
    for i, item in enumerate(bom[:5], 1):
        rows.append({
            "Category": "BOM",
            "Parameter": f"BOM_Item_{i}",
            "Extracted_Value": item,
            "Ground_Truth": "N/A",
            "Match": "N/A",
            "Source": "Extraction"
        })
    
    # 2. GROUND TRUTH PARAMETERS (not visible in image)
    # Key names from DrawingInputSheet.xlsx
    gt_only_params = [
        ("Sheet_Settings", "Sheet_Width", "Sheet_Width"),
        ("Sheet_Settings", "Sheet_Height", "Sheet_Height"),
        ("Drafting_Settings", "Text Size", "Drafting_TextSize"),
        ("Drafting_Settings", "Arrow Size", "Drafting_ArrowSize"),
        ("Drafting_Settings", "Leader Stub Size", "Drafting_LeaderStubSize"),
        ("Drafting_Settings", "Dimension Precision", "Drafting_DimensionPrecision"),
    ]
    
    for cat, display_name, gt_key in gt_only_params:
        gt_val = gt_values.get(gt_key, "NOT FOUND")
        rows.append({
            "Category": cat,
            "Parameter": display_name,
            "Extracted_Value": "NOT VISIBLE (CAD internal setting)",
            "Ground_Truth": gt_val,
            "Match": "N/A",
            "Source": "GT only"
        })
    
    return pd.DataFrame(rows)

def main():
    print("=" * 60)
    print("  Comprehensive Comparison - Extracted vs Ground Truth")
    print("=" * 60)
    
    COMPARISON_OUTPUT.mkdir(exist_ok=True)
    
    # Load extracted values
    print("\n[LOAD] Loading extracted values...")
    extracted_data = load_extracted_values()
    
    if not extracted_data:
        print("[ERROR] No extraction results found. Run 8_extract_comprehensive.py first.")
        return
    
    print("   Found extracted values")
    
    # Load ground truth
    print("\n[LOAD] Loading ground truth...")
    gt_values = load_ground_truth()
    print(f"   Found {len(gt_values)} ground truth parameters")
    
    # Create comparison table
    print("\n[COMPARE] Creating comparison table...")
    comparison_df = create_comparison_table(extracted_data, gt_values)
    
    # Save to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = COMPARISON_OUTPUT / f"COMPREHENSIVE_COMPARISON_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        comparison_df.to_excel(writer, sheet_name='Side_by_Side', index=False)
        
        # Summary sheet
        summary = {
            "Total Parameters": len(comparison_df),
            "Extracted from Image": len(comparison_df[comparison_df['Source'] == 'Extraction']),
            "Ground Truth Only": len(comparison_df[comparison_df['Source'] == 'GT only']),
            "Matches with GT": len(comparison_df[(comparison_df['Match'] == 'MATCH') & (comparison_df['Source'] == 'Extraction vs GT')]),
        }
        
        pd.DataFrame([summary]).to_excel(writer, sheet_name='Summary', index=False)
        
        # Raw extracted text
        raw = extracted_data.get("raw_extraction", {})
        raw_data = []
        for key, value in raw.items():
            raw_data.append({"Category": key, "Raw_Text": value[:1000]})
        pd.DataFrame(raw_data).to_excel(writer, sheet_name='Raw_Extraction', index=False)
    
    print(f"\n[OK] Comparison saved: {output_file}")
    
    # Print summary
    print("\n=== COMPARISON SUMMARY ===")
    print(f"Total rows: {len(comparison_df)}")
    print(f"Extracted: {summary['Extracted from Image']}")
    print(f"GT only: {summary['Ground Truth Only']}")
    print(f"Matches: {summary['Matches with GT']}")

if __name__ == "__main__":
    main()