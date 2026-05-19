"""
config_comprehensive.py - Comprehensive Parameter List
=========================================================
Merges:
1. 71 parameters (our simplified list)
2. parameters list to extract.docx (100+ comprehensive params)
3. DrawingInputSheet.xlsx (ground truth CAD settings)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ============================================================================
# COMPREHENSIVE PARAMETERS FROM DOCX (3D Model + 2D Drawing)
# ============================================================================
COMPREHENSIVE_PARAMS = {
    # 3D MODEL METADATA
    "Model_Info": [
        {"id": "M1", "name": "Part Name", "category": "Model_Info", "source": "docx"},
        {"id": "M2", "name": "Part Number", "category": "Model_Info", "source": "docx"},
        {"id": "M3", "name": "Revision", "category": "Model_Info", "source": "docx"},
        {"id": "M4", "name": "Units", "category": "Model_Info", "source": "docx"},
    ],
    "Geometry_Size": [
        {"id": "G1", "name": "Bounding Box Min X", "category": "Geometry_Size", "source": "docx"},
        {"id": "G2", "name": "Bounding Box Min Y", "category": "Geometry_Size", "source": "docx"},
        {"id": "G3", "name": "Bounding Box Min Z", "category": "Geometry_Size", "source": "docx"},
        {"id": "G4", "name": "Bounding Box Max X", "category": "Geometry_Size", "source": "docx"},
        {"id": "G5", "name": "Bounding Box Max Y", "category": "Geometry_Size", "source": "docx"},
        {"id": "G6", "name": "Bounding Box Max Z", "category": "Geometry_Size", "source": "docx"},
        {"id": "G7", "name": "Overall Length", "category": "Geometry_Size", "source": "docx"},
        {"id": "G8", "name": "Overall Width", "category": "Geometry_Size", "source": "docx"},
        {"id": "G9", "name": "Overall Height", "category": "Geometry_Size", "source": "docx"},
    ],
    "Features_Holes": [
        {"id": "H1", "name": "Hole Count", "category": "Features_Holes", "source": "docx"},
        {"id": "H2", "name": "Hole Type", "category": "Features_Holes", "source": "docx"},
        {"id": "H3", "name": "Hole Diameter", "category": "Features_Holes", "source": "docx"},
        {"id": "H4", "name": "Hole Depth", "category": "Features_Holes", "source": "docx"},
        {"id": "H5", "name": "Hole Pattern Type", "category": "Features_Holes", "source": "docx"},
    ],
    "Features_Fillets_Chamfers": [
        {"id": "F1", "name": "Fillet Count", "category": "Features_Fillets", "source": "docx"},
        {"id": "F2", "name": "Fillet Radius", "category": "Features_Fillets", "source": "docx"},
        {"id": "F3", "name": "Chamfer Count", "category": "Features_Chamfers", "source": "docx"},
        {"id": "F4", "name": "Chamfer Distance", "category": "Features_Chamfers", "source": "docx"},
        {"id": "F5", "name": "Chamfer Angle", "category": "Features_Chamfers", "source": "docx"},
    ],
    "Edges_Faces": [
        {"id": "E1", "name": "Edge Count", "category": "Edges", "source": "docx"},
        {"id": "E2", "name": "Face Count", "category": "Faces", "source": "docx"},
        {"id": "E3", "name": "Face Types", "category": "Faces", "source": "docx"},
    ],
    "Datum_Geometry": [
        {"id": "D1", "name": "Datum Planes Count", "category": "Datum", "source": "docx"},
        {"id": "D2", "name": "Datum Axis Count", "category": "Datum", "source": "docx"},
    ],
    "Slots_Pockets": [
        {"id": "S1", "name": "Slot Count", "category": "Slots_Pockets", "source": "docx"},
        {"id": "S2", "name": "Pocket Count", "category": "Slots_Pockets", "source": "docx"},
    ],
    "Attribute_Metadata": [
        {"id": "A1", "name": "Material", "category": "Attributes", "source": "docx"},
        {"id": "A2", "name": "Thickness", "category": "Attributes", "source": "docx"},
    ],
    # 2D DRAWING SHEET PARAMETERS
    "Sheet_Info": [
        {"id": "SHT1", "name": "Sheet Size", "category": "Sheet_Info", "source": "docx"},
        {"id": "SHT2", "name": "Sheet Width", "category": "Sheet_Info", "source": "ground_truth"},
        {"id": "SHT3", "name": "Sheet Height", "category": "Sheet_Info", "source": "ground_truth"},
        {"id": "SHT4", "name": "Scale", "category": "Sheet_Info", "source": "docx"},
        {"id": "SHT5", "name": "Scale Numerator", "category": "Sheet_Info", "source": "ground_truth"},
        {"id": "SHT6", "name": "Scale Denominator", "category": "Sheet_Info", "source": "ground_truth"},
        {"id": "SHT7", "name": "Projection Type", "category": "Sheet_Info", "source": "docx"},
        {"id": "SHT8", "name": "Units", "category": "Sheet_Info", "source": "docx"},
    ],
    "View_Parameters": [
        {"id": "V1", "name": "Number of Views", "category": "Views", "source": "ground_truth"},
        {"id": "V2", "name": "View Names", "category": "Views", "source": "docx"},
        {"id": "V3", "name": "View Types", "category": "Views", "source": "docx"},
        {"id": "V4", "name": "View Scale", "category": "Views", "source": "docx"},
        {"id": "V5", "name": "View Origin X", "category": "Views", "source": "docx"},
        {"id": "V6", "name": "View Origin Y", "category": "Views", "source": "docx"},
    ],
    "Drafting_Settings": [
        {"id": "DR1", "name": "Text Size", "category": "Drafting", "source": "ground_truth"},
        {"id": "DR2", "name": "Arrow Size", "category": "Drafting", "source": "ground_truth"},
        {"id": "DR3", "name": "Leader Stub Size", "category": "Drafting", "source": "ground_truth"},
        {"id": "DR4", "name": "Dimension Precision", "category": "Drafting", "source": "ground_truth"},
        {"id": "DR5", "name": "Angle Precision", "category": "Drafting", "source": "ground_truth"},
    ],
    "Dimensions": [
        {"id": "DM1", "name": "Linear Dimensions Count", "category": "Dimensions", "source": "docx"},
        {"id": "DM2", "name": "Radial Dimensions Count", "category": "Dimensions", "source": "docx"},
        {"id": "DM3", "name": "Angular Dimensions Count", "category": "Dimensions", "source": "docx"},
        {"id": "DM4", "name": "Dimension Values", "category": "Dimensions", "source": "extraction"},
        {"id": "DM5", "name": "Dimension Tolerances", "category": "Dimensions", "source": "extraction"},
    ],
    "GDandT": [
        {"id": "GD1", "name": "GD&T Symbols Count", "category": "GDandT", "source": "docx"},
        {"id": "GD2", "name": "Position Tolerance", "category": "GDandT", "source": "extraction"},
        {"id": "GD3", "name": "Flatness", "category": "GDandT", "source": "extraction"},
        {"id": "GD4", "name": "Straightness", "category": "GDandT", "source": "extraction"},
        {"id": "GD5", "name": "Perpendicularity", "category": "GDandT", "source": "extraction"},
        {"id": "GD6", "name": "Parallelism", "category": "GDandT", "source": "extraction"},
    ],
    "Notes_Annotations": [
        {"id": "N1", "name": "General Notes Count", "category": "Notes", "source": "docx"},
        {"id": "N2", "name": "Notes Text Content", "category": "Notes", "source": "extraction"},
        {"id": "N3", "name": "Notes Position", "category": "Notes", "source": "extraction"},
        {"id": "N4", "name": "Welding Symbols", "category": "Notes", "source": "extraction"},
    ],
    "Title_Block": [
        {"id": "TB1", "name": "Drawing Number", "category": "TitleBlock", "source": "extraction"},
        {"id": "TB2", "name": "Part Name", "category": "TitleBlock", "source": "extraction"},
        {"id": "TB3", "name": "Revision", "category": "TitleBlock", "source": "extraction"},
        {"id": "TB4", "name": "Scale", "category": "TitleBlock", "source": "extraction"},
        {"id": "TB5", "name": "Author", "category": "TitleBlock", "source": "extraction"},
        {"id": "TB6", "name": "Date", "category": "TitleBlock", "source": "extraction"},
    ],
    "BOM": [
        {"id": "B1", "name": "BOM Present", "category": "BOM", "source": "extraction"},
        {"id": "B2", "name": "Component Count", "category": "BOM", "source": "docx"},
        {"id": "B3", "name": "Component Names", "category": "BOM", "source": "docx"},
    ],
    "Layers": [
        {"id": "L1", "name": "Layer Count", "category": "Layers", "source": "docx"},
        {"id": "L2", "name": "Layer Names", "category": "Layers", "source": "docx"},
        {"id": "L3", "name": "Visible Layers", "category": "Layers", "source": "docx"},
    ],
}

# Create flat dictionary
ALL_COMPREHENSIVE_PARAMS = {}
for cat, params in COMPREHENSIVE_PARAMS.items():
    for p in params:
        ALL_COMPREHENSIVE_PARAMS[p['id']] = p

# ============================================================================
# GROUND TRUTH PARAMETERS (from DrawingInputSheet.xlsx)
# ============================================================================
GROUND_TRUTH_PARAMS = {
    "Sheet_Settings": [
        {"id": "GT_1", "name": "Sheet_Width", "value": 1189, "category": "Sheet_Settings"},
        {"id": "GT_2", "name": "Sheet_Height", "value": 841, "category": "Sheet_Settings"},
        {"id": "GT_3", "name": "Sheet_ScaleNumerator", "value": 1, "category": "Sheet_Settings"},
        {"id": "GT_4", "name": "Sheet_ScaleDenominator", "value": 1, "category": "Sheet_Settings"},
    ],
    "Drafting_Settings": [
        {"id": "GT_5", "name": "Drafting_TextSize", "value": 5, "category": "Drafting_Settings"},
        {"id": "GT_6", "name": "Drafting_ArrowSize", "value": 2.5, "category": "Drafting_Settings"},
        {"id": "GT_7", "name": "Drafting_LeaderStubSize", "value": 1.25, "category": "Drafting_Settings"},
        {"id": "GT_8", "name": "Drafting_DimensionPrecision", "value": 2, "category": "Drafting_Settings"},
        {"id": "GT_9", "name": "Drafting_AnglePrecision", "value": 2, "category": "Drafting_Settings"},
    ],
    "View_Settings": [
        {"id": "GT_10", "name": "No_of_Views", "value": 6, "category": "View_Settings"},
        {"id": "GT_11", "name": "View_Top_Enable", "value": 1, "category": "View_Settings"},
        {"id": "GT_12", "name": "View_Top_X", "value": 493.6, "category": "View_Settings"},
        {"id": "GT_13", "name": "View_Top_Y", "value": 705.7, "category": "View_Settings"},
        {"id": "GT_14", "name": "View_Top_ScaleNumerator", "value": 1, "category": "View_Settings"},
        {"id": "GT_15", "name": "View_Top_ScaleDenominator", "value": 1, "category": "View_Settings"},
        {"id": "GT_16", "name": "View_Front_Enable", "value": 1, "category": "View_Settings"},
        {"id": "GT_17", "name": "View_Front_X", "value": 493.6, "category": "View_Settings"},
        {"id": "GT_18", "name": "View_Front_Y", "value": 521.9, "category": "View_Settings"},
        {"id": "GT_19", "name": "View_Right_Enable", "value": 1, "category": "View_Settings"},
        {"id": "GT_20", "name": "View_Right_X", "value": 1079.8, "category": "View_Settings"},
        {"id": "GT_21", "name": "View_Right_Y", "value": 705.7, "category": "View_Settings"},
    ],
    "Dimension_Settings": [
        {"id": "GT_31", "name": "Linear_Enable", "value": 1, "category": "Dimension_Settings"},
        {"id": "GT_32", "name": "Linear_TextSize", "value": 5, "category": "Dimension_Settings"},
        {"id": "GT_33", "name": "Linear_ArrowSize", "value": 2.5, "category": "Dimension_Settings"},
    ],
}

ALL_GT_PARAMS = {}
for cat, params in GROUND_TRUTH_PARAMS.items():
    for p in params:
        ALL_GT_PARAMS[p['id']] = p

# ============================================================================
# MAPPING: Ground Truth <-> Extracted Parameters
# ============================================================================
PARAM_MAPPING = {
    # Sheet settings
    "GT_1": {"extracted_param": "SHT2", "name": "Sheet Width"},
    "GT_2": {"extracted_param": "SHT3", "name": "Sheet Height"},
    "GT_3": {"extracted_param": "SHT5", "name": "Scale Numerator"},
    "GT_4": {"extracted_param": "SHT6", "name": "Scale Denominator"},
    
    # Drafting settings
    "GT_5": {"extracted_param": "DR1", "name": "Text Size"},
    "GT_6": {"extracted_param": "DR2", "name": "Arrow Size"},
    "GT_7": {"extracted_param": "DR3", "name": "Leader Stub Size"},
    "GT_8": {"extracted_param": "DR4", "name": "Dimension Precision"},
    "GT_9": {"extracted_param": "DR5", "name": "Angle Precision"},
    
    # View settings
    "GT_10": {"extracted_param": "V1", "name": "Number of Views"},
    "GT_14": {"extracted_param": "V4", "name": "View Scale"},
    
    # Dimension settings
    "GT_32": {"extracted_param": "DM1", "name": "Linear Text Size"},
}

# ============================================================================
# EXTRACTION CATEGORIES (what we can extract from image)
# ============================================================================
EXTRACTION_CATEGORIES = {
    "dimensions": "Extract ALL dimension values: linear (mm), diameter (ø), radius (R), angular (°). List each with value and position.",
    "gdt_symbols": "Extract ALL GD&T symbols: position tolerance (ø), flatness, straightness, perpendicularity, parallelism. List symbol, value, and datum references.",
    "title_block": "Extract title block: drawing number, part name, revision, scale, units, date, author, company.",
    "views": "Identify views: front, top, side, section, detail, isometric. List each view name and scale.",
    "notes": "Extract ALL notes: general notes, surface finish, welding symbols, datum identifiers. List full text.",
    "geometry": "Identify geometric features: holes (count, diameter), fillets (count, radius), chamfers (count, size), ribs.",
    "bom": "Extract BOM: component names, quantities, part numbers if visible.",
    "layers": "Note any layer information visible in drawing.",
}

def get_extraction_summary():
    """Get summary of all parameters."""
    print("\n=== COMPREHENSIVE PARAMETER SUMMARY ===")
    print(f"Total from docx: {len(ALL_COMPREHENSIVE_PARAMS)}")
    print(f"Total from ground truth: {len(ALL_GT_PARAMS)}")
    print(f"Mapped parameters: {len(PARAM_MAPPING)}")
    
    print("\n=== BY SOURCE ===")
    print("- extraction: Parameters extracted from PDF image")
    print("- docx: Parameters from 'parameters list to extract.docx'")
    print("- ground_truth: Parameters from DrawingInputSheet.xlsx")

if __name__ == "__main__":
    get_extraction_summary()