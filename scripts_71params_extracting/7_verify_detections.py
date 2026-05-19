"""
7_verify_detections.py - Manual Verification Script
=======================================================
Generates a side-by-side comparison for manual review of detections.
Run this to verify if detections are correct by checking against the PDF.
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import config
from config import PARAMETERS_71, ALL_PARAMS_71

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs_71params"
VERIFICATION_OUTPUT = OUTPUT_DIR / "verification_reports"

def load_results():
    """Load all model results."""
    results = {}
    
    # Load EasyOCR results
    easyocr_dir = OUTPUT_DIR / "easyocr_results"
    if easyocr_dir.exists():
        for f in easyocr_dir.glob("*_easyocr_71params.json"):
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                results['EasyOCR'] = data
                break
    
    # Load LLaVA results
    llava_dir = OUTPUT_DIR / "llava_results"
    if llava_dir.exists():
        for f in llava_dir.glob("*_llava_71params.json"):
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                results['LLaVA'] = data
                break
    
    return results

def get_category_for_param(param_id: str) -> str:
    """Get category for a param ID."""
    for cat, params in PARAMETERS_71.items():
        for p in params:
            if p['id'] == param_id:
                return cat
    return "Unknown"

def generate_verification_report(results: dict):
    """Generate verification report."""
    VERIFICATION_OUTPUT.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for verification sheet
    verification_rows = []
    
    # Sort params by ID numerically
    sorted_params = sorted(ALL_PARAMS_71.values(), key=lambda x: int(x['id']))
    
    for param in sorted_params:
        param_id = param['id']
        param_name = param['name']
        category = get_category_for_param(param_id)
        
        row = {
            'Param_ID': param_id,
            'Param_Name': param_name,
            'Category': category,
        }
        
        # Add detection status for each model
        for model_name, result in results.items():
            params_71 = result.get('params_71', {})
            detected = params_71.get(param_id, {}).get('detected', False)
            
            # Get raw extraction text for this category
            extraction = result.get('extraction', {})
            
            # Map category to extraction key
            key_map = {
                'Geometry_Design': 'geometry',
                'Dimensions': 'dimensions',
                'Assembly': 'dimensions',
                'Tolerances': 'tolerances',
                'GD_T': 'gd_t',
                'Views_Drafting': 'views',
                'Notes_Annotations': 'notes',
                'TitleBlock_Metadata': 'title_block',
                'BOM': 'title_block',
                'Scale': 'title_block',
            }
            
            extraction_key = key_map.get(category, 'dimensions')
            raw_text = extraction.get(extraction_key, '')[:500] if extraction.get(extraction_key) else 'N/A'
            
            # Truncate long text
            if len(raw_text) > 500:
                raw_text = raw_text[:500] + '...'
            
            row[f'{model_name}_Detected'] = 'YES' if detected else 'NO'
            row[f'{model_name}_Raw_Response'] = raw_text
        
        # Add manual verification columns
        row['Manual_Verify'] = ''
        row['Is_Correct?'] = ''
        row['Comments'] = ''
        
        verification_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(verification_rows)
    
    # Reorder columns - put manual verification at the end
    model_names = list(results.keys())
    col_order = ['Param_ID', 'Param_Name', 'Category']
    for model in model_names:
        col_order.extend([f'{model}_Detected', f'{model}_Raw_Response'])
    col_order.extend(['Manual_Verify', 'Is_Correct?', 'Comments'])
    
    df = df[col_order]
    
    # Save to Excel with formatting
    output_file = VERIFICATION_OUTPUT / f"VERIFICATION_TEMPLATE_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main verification sheet
        df.to_excel(writer, sheet_name='Verification', index=False)
        
        # Summary sheet
        summary_data = []
        for category, params in PARAMETERS_71.items():
            total = len(params)
            
            row = {'Category': category, 'Total_Params': total}
            
            for model in model_names:
                detected = sum(1 for p in params 
                    if results[model].get('params_71', {}).get(p['id'], {}).get('detected', False))
                row[f'{model}_Detected'] = detected
                row[f'{model}_Pct'] = f"{detected/total*100:.1f}%" if total > 0 else "0%"
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Instructions sheet
        instructions = [
            ["INSTRUCTIONS FOR MANUAL VERIFICATION"],
            [""],
            ["1. Open the original PDF: HP_58231-82P00_500_s_SUZUKI_DRAW_SH1.pdf"],
            ["2. For each parameter, visually check if the feature exists in the drawing"],
            ["3. In 'Manual_Verify' column, mark: 'VISIBLE' or 'NOT VISIBLE'"],
            ["4. In 'Is_Correct?' column, mark: 'TP' (True Positive) or 'FP' (False Positive)"],
            ["5. Add any comments in 'Comments' column"],
            [""],
            ["NOTE: The 'Raw_Response' column shows what the model actually extracted."],
            ["If the response is vague/generic, the detection may be a False Positive."],
            [""],
            ["Categories:"],
            ["- Geometry_Design: holes, fillets, chamfers, ribs"],
            ["- Dimensions: length, diameter, thickness, position changes"],
            ["- Tolerances: tolerance values, fits (H7/g6)"],
            ["- GD_T: position, flatness, straightness, etc."],
            ["- Views_Drafting: front/top/side views, section lines"],
            ["- Notes_Annotations: general notes, symbols, datums"],
            ["- TitleBlock_Metadata: part name, drawing number, revision"],
            ["- BOM: bill of materials"],
            ["- Scale: sheet scale, view scale"],
        ]
        
        instructions_df = pd.DataFrame(instructions)
        instructions_df.to_excel(writer, sheet_name='Instructions', index=False, header=False)
    
    print(f"\n[OK] Verification template saved: {output_file}")
    print(f"   - Verification sheet: {len(df)} parameters to review")
    print(f"   - Summary sheet: Detection rates by category")
    print(f"   - Instructions: How to verify manually")
    
    return output_file

def main():
    print("=" * 60)
    print("  Manual Verification Template Generator")
    print("=" * 60)
    
    # Load results
    results = load_results()
    
    if not results:
        print("[ERROR] No results found. Run extraction scripts first.")
        return
    
    print(f"\n[FOUND] Results from: {', '.join(results.keys())}")
    
    # Generate report
    output_file = generate_verification_report(results)
    
    print("\n" + "=" * 60)
    print("[DONE] Verification template ready!")
    print("=" * 60)

if __name__ == "__main__":
    main()