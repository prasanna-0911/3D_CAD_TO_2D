"""
0_extract_ground_truth.py - Extract Ground Truth Parameters
=============================================================
Reads the DrawingInputSheet.xlsx and extracts ground truth parameter values.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts_71params_extracting.config import (
    INPUT_DIR, OUTPUT_DIR, GROUND_TRUTH_FILE, 
    GROUND_TRUTH_PARAMS, ALL_GT_PARAMS, PARAM_MAPPING
)
from scripts_71params_extracting.utils import ensure_dir, save_json, save_excel, print_header


def load_ground_truth():
    """Load ground truth parameters from Excel file."""
    import pandas as pd
    
    gt_file = INPUT_DIR / GROUND_TRUTH_FILE
    
    print(f"Loading ground truth from: {gt_file}")
    
    df = pd.read_excel(gt_file, sheet_name='Sheet1', header=None)
    
    gt_values = {}
    
    for idx, row in df.iterrows():
        param_name = row[1]
        param_value = row[2]
        
        if pd.notna(param_name) and pd.notna(param_value):
            gt_values[str(param_name).strip()] = param_value
    
    return gt_values


def extract_ground_truth():
    """Extract and organize ground truth parameters."""
    print_header("Ground Truth Parameters Extraction")
    
    gt_values = load_ground_truth()
    
    print(f"\nLoaded {len(gt_values)} ground truth values")
    
    organized_gt = {}
    
    for category, params in GROUND_TRUTH_PARAMS.items():
        organized_gt[category] = []
        
        for param in params:
            param_name = param["name"]
            param_value = gt_values.get(param_name, param["value"])
            
            organized_gt[category].append({
                "id": param["id"],
                "name": param_name,
                "value": param_value,
                "category": category
            })
    
    output_dir = OUTPUT_DIR / "ground_truth"
    ensure_dir(output_dir)
    
    result = {
        "extraction_date": datetime.now().isoformat(),
        "source_file": GROUND_TRUTH_FILE,
        "total_params": len(ALL_GT_PARAMS),
        "categories": organized_gt,
        "raw_values": gt_values
    }
    
    json_path = output_dir / "ground_truth_parameters.json"
    save_json(result, json_path)
    
    excel_data = []
    for category, params in organized_gt.items():
        for param in params:
            excel_data.append({
                "Parameter_ID": param["id"],
                "Parameter_Name": param["name"],
                "Category": param["category"],
                "Ground_Truth_Value": param["value"]
            })
    
    excel_path = output_dir / "ground_truth_parameters.xlsx"
    save_excel(excel_data, excel_path, "Ground Truth")
    
    print("\n" + "=" * 60)
    print("Ground Truth Parameters Summary")
    print("=" * 60)
    
    for category, params in organized_gt.items():
        print(f"\n{category}:")
        for param in params:
            print(f"   {param['id']}: {param['name']} = {param['value']}")
    
    print("\n✅ Ground truth extraction complete!")
    print(f"   Results saved to: {output_dir}")
    
    return result


if __name__ == "__main__":
    try:
        extract_ground_truth()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()