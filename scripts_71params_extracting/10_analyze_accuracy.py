"""
10_analyze_accuracy.py - Deep Accuracy Analysis
=================================================
Analyze patterns in extraction vs ground truth
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs_71params"

def load_extracted():
    """Load most recent extraction."""
    files = list((OUTPUT_DIR / "comprehensive_extraction").glob("comprehensive_extraction_*.json"))
    latest = max(files, key=lambda x: x.stat().st_mtime)
    with open(latest, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_ground_truth():
    """Load ground truth from Excel."""
    gt_file = PROJECT_ROOT / "input_pdfs" / "DrawingInputSheet.xlsx"
    df = pd.read_excel(gt_file, sheet_name='Sheet1')
    
    gt_values = {}
    for _, row in df.iterrows():
        param_name = row.iloc[1]
        param_value = row.iloc[2]
        if pd.notna(param_name) and pd.notna(param_value):
            gt_values[str(param_name).strip()] = param_value
    return gt_values

def analyze_accuracy():
    """Perform deep accuracy analysis."""
    print("=" * 60)
    print("  Deep Accuracy Analysis - LLaVA 1.5 7B (FP16)")
    print("=" * 60)
    
    extracted = load_extracted()
    gt_values = load_ground_truth()
    parsed = extracted.get("parsed_values", {})
    
    analysis_results = []
    
    # 1. DIMENSIONS ANALYSIS
    print("\n[1] DIMENSIONS ANALYSIS")
    print("-" * 40)
    dimensions = parsed.get("dimensions", [])
    
    # Look for number patterns in dimensions
    dim_numbers = []
    for dim in dimensions:
        import re
        numbers = re.findall(r'[\d.]+', dim)
        for n in numbers:
            try:
                dim_numbers.append(float(n))
            except:
                pass
    
    print(f"   Extracted numbers: {dim_numbers}")
    print(f"   Count: {len(dim_numbers)}")
    
    # 2. VIEWS ANALYSIS
    print("\n[2] VIEWS ANALYSIS")
    print("-" * 40)
    views_text = extracted.get("raw_extraction", {}).get("views", "")
    
    # Extract view count from various formats
    import re
    view_match = re.search(r'(\d+)\s*view', views_text, re.IGNORECASE)
    if view_match:
        extracted_views = int(view_match.group(1))
    else:
        extracted_views = 0
    
    gt_views = gt_values.get("No_of_Views", 6)
    print(f"   Extracted: {extracted_views} views")
    print(f"   Ground Truth: {gt_views} views")
    print(f"   Accuracy: {100 - abs(extracted_views - gt_views) / gt_views * 100:.1f}%")
    
    # 3. TITLE BLOCK ANALYSIS
    print("\n[3] TITLE BLOCK ANALYSIS")
    print("-" * 40)
    title_block = parsed.get("title_block", {})
    print(f"   Extracted fields: {len(title_block)}")
    
    # Check for partial matches (some characters might be correct)
    partial_matches = []
    
    # Drawing number - extract any numbers
    if title_block.get("* Drawing Number"):
        extracted_dwg = title_block.get("* Drawing Number", "")
        # Check if any digits match GT drawing number
        gt_dwg = "58231-82P00"
        for char in extracted_dwg:
            if char in gt_dwg:
                partial_matches.append(f"Char '{char}' found in GT")
    
    print(f"   Partial matches found: {len(partial_matches)}")
    
    # 4. GEOMETRY ANALYSIS
    print("\n[4] GEOMETRY ANALYSIS")
    print("-" * 40)
    geom = parsed.get("geometry", {})
    geom_desc = geom.get("description", "")
    
    # Extract hole counts and sizes
    hole_match = re.search(r'(\d+)\s*holes?', geom_desc, re.IGNORECASE)
    if hole_match:
        print(f"   Holes detected: {hole_match.group(1)}")
    
    fillet_match = re.search(r'(\d+)\s*fillets?', geom_desc, re.IGNORECASE)
    if fillet_match:
        print(f"   Fillets detected: {fillet_match.group(1)}")
    
    # 5. GD&T ANALYSIS
    print("\n[5] GD&T ANALYSIS")
    print("-" * 40)
    gdt = parsed.get("gdt_symbols", [])
    print(f"   GD&T symbols found: {len(gdt)}")
    print(f"   Raw: {gdt}")
    
    # 6. NOTES ANALYSIS
    print("\n[6] NOTES ANALYSIS")
    print("-" * 40)
    notes = parsed.get("notes", [])
    print(f"   Notes items found: {len(notes)}")
    
    # 7. CREATE COMPARISON TABLE
    print("\n[7] CREATING COMPARISON TABLE")
    print("-" * 40)
    
    comparison_data = []
    
    # Dimension comparison
    for i, dim in enumerate(dimensions[:5], 1):
        comparison_data.append({
            "Parameter": f"Dimension_{i}",
            "Category": "Dimensions",
            "Extracted": dim,
            "GT_Numeric_Value": dim_numbers[i-1] if i-1 < len(dim_numbers) else "N/A",
            "Note": "Check actual drawing for ground truth"
        })
    
    # View comparison
    comparison_data.append({
        "Parameter": "Number of Views",
        "Category": "Views",
        "Extracted": extracted_views,
        "GT_Numeric_Value": gt_views,
        "Note": f"Accuracy: {100 - abs(extracted_views - gt_views) / gt_views * 100:.1f}%"
    })
    
    # Title block comparison
    for key, value in title_block.items():
        comparison_data.append({
            "Parameter": key.replace("* ", ""),
            "Category": "Title Block",
            "Extracted": value,
            "GT_Numeric_Value": "N/A",
            "Note": "Manual verification needed"
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(comparison_data)
    
    output_file = OUTPUT_DIR / "comprehensive_comparison" / f"ACCURACY_ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Analysis', index=False)
        
        # Summary sheet
        summary = {
            "Metric": [
                "Dimensions extracted",
                "Views count extracted",
                "Title block fields",
                "GD&T symbols",
                "Geometry features detected",
                "Notes extracted"
            ],
            "Value": [
                len(dimensions),
                extracted_views,
                len(title_block),
                len(gdt),
                1 if geom else 0,
                len(notes)
            ]
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
        
        # Raw extraction
        raw_data = []
        for key, value in extracted.get("raw_extraction", {}).items():
            raw_data.append({"Category": key, "Raw_Response": value[:500]})
        pd.DataFrame(raw_data).to_excel(writer, sheet_name='Raw_Extraction', index=False)
    
    print(f"   Saved: {output_file}")
    
    # 8. OVERALL ACCURACY ESTIMATE
    print("\n" + "=" * 60)
    print("  OVERALL ACCURACY ESTIMATE")
    print("=" * 60)
    
    # Score based on what we can verify
    scores = []
    
    # Views - can verify
    view_accuracy = 100 - abs(extracted_views - gt_views) / gt_views * 100
    scores.append(("Views count", view_accuracy))
    
    # Dimensions - check if numbers are reasonable (not hallucinated)
    # We can't verify exact values without manual review
    
    print("\nVerifiable Metrics:")
    for name, score in scores:
        print(f"   {name}: {score:.1f}%")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_accuracy()