#!/usr/bin/env python3
"""Build a styled multi-sheet Excel (.xlsx) workbook from bench_results.json using stdlib only."""
import json
import os
import zipfile
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
IN_JSON = os.path.join(HERE, "bench_results.json")
OUT_XLSX = os.path.join(HERE, "benchmark_blender_vs_liquid_rust.xlsx")


def col_letter(idx: int) -> str:
    s = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def make_sheet_xml(rows, col_widths, freeze_row=1):
    # Cell value types:
    # ("str", text, style), ("num", val, style), ("formula", expr_without_eq, cached_or_none, style)
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
    ]
    if freeze_row:
        lines.append(
            f'<sheetViews><sheetView tabSelected="0" workbookViewId="0">'
            f'<pane ySplit="{freeze_row}" topLeftCell="A{freeze_row + 1}" activePane="bottomLeft" state="frozen"/>'
            f'</sheetView></sheetViews>'
        )
    lines.append("<cols>")
    for c_i, w in enumerate(col_widths, start=1):
        lines.append(f'<col min="{c_i}" max="{c_i}" width="{w}" customWidth="1"/>')
    lines.append("</cols><sheetData>")

    for r_idx, row in enumerate(rows, start=1):
        ht = ' ht="24" customHeight="1"' if r_idx == 1 else ' ht="19" customHeight="1"'
        lines.append(f'<row r="{r_idx}"{ht}>')
        for c_idx, cell in enumerate(row, start=1):
            if cell is None:
                continue
            ref = f"{col_letter(c_idx)}{r_idx}"
            kind = cell[0]
            if kind == "str":
                _, txt, sty = cell
                lines.append(
                    f'<c r="{ref}" s="{sty}" t="inlineStr"><is><t>{escape(str(txt))}</t></is></c>'
                )
            elif kind == "num":
                _, val, sty = cell
                if val is None:
                    continue
                lines.append(f'<c r="{ref}" s="{sty}"><v>{val}</v></c>')
            elif kind == "formula":
                _, expr, cached, sty = cell
                v_tag = f"<v>{cached}</v>" if cached is not None else ""
                lines.append(f'<c r="{ref}" s="{sty}"><f>{escape(expr)}</f>{v_tag}</c>')
        lines.append("</row>")

    lines.append("</sheetData></worksheet>")
    return "\n".join(lines)


def build_styles_xml():
    # Style indices:
    # 0: normal text
    # 1: header (navy fill, white bold)
    # 2: bold label
    # 3: integer (#,##0)
    # 4: float 3 decimals (0.000)
    # 5: float 2 decimals (0.00)
    # 6: speedup multiplier (0.0"x", bold green)
    # 7: title row (bold 13pt)
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="4">
    <numFmt numFmtId="164" formatCode="#,##0"/>
    <numFmt numFmtId="165" formatCode="0.000"/>
    <numFmt numFmtId="166" formatCode="0.00"/>
    <numFmt numFmtId="167" formatCode="0.0&quot;x&quot;"/>
  </numFmts>
  <fonts count="4">
    <font><sz val="11"/><name val="Calibri"/><color rgb="FF111827"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/><color rgb="FF065F46"/></font>
    <font><b/><sz val="13"/><name val="Calibri"/><color rgb="FF0F172A"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1E3A8A"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD1FAE5"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFD1D5DB"/></left>
      <right style="thin"><color rgb="FFD1D5DB"/></right>
      <top style="thin"><color rgb="FFD1D5DB"/></top>
      <bottom style="thin"><color rgb="FFD1D5DB"/></bottom>
    </border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" applyFont="1" applyFill="1" applyBorder="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="3" fillId="0" borderId="0" applyFont="1"/>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" applyNumberFormat="1" applyBorder="1"/>
    <xf numFmtId="165" fontId="0" fillId="0" borderId="1" applyNumberFormat="1" applyBorder="1"/>
    <xf numFmtId="166" fontId="0" fillId="0" borderId="1" applyNumberFormat="1" applyBorder="1"/>
    <xf numFmtId="167" fontId="2" fillId="3" borderId="1" applyFont="1" applyFill="1" applyNumberFormat="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" applyBorder="1"><alignment wrapText="1"/></xf>
  </cellXfs>
</styleSheet>"""


def find_row(data, sim_prefix, preset_prefix, mode):
    for i, r in enumerate(data):
        if r["simulator"].startswith(sim_prefix) and r["preset_or_res"].startswith(preset_prefix) and r["mode"] == mode:
            return i + 2, r  # 1-based Excel row in Raw Benchmark Data (header = row 1)
    return None, None


def main():
    with open(IN_JSON) as f:
        data = json.load(f)

    # Sheet 2: Raw Benchmark Data
    raw_headers = [
        ("str", "Simulator", 1),
        ("str", "Preset / Domain Res", 1),
        ("str", "Workload Mode", 1),
        ("str", "Compute Backend", 1),
        ("str", "Frames (3.0s @ 30fps)", 1),
        ("str", "Particles", 1),
        ("str", "Physics Time (s)", 1),
        ("str", "Mesh / Export Time (s)", 1),
        ("str", "Total Wall Time (s)", 1),
        ("str", "ms / Frame", 1),
        ("str", "Particle-Steps / sec", 1),
        ("str", "Mesh Vertices", 1),
        ("str", "Mesh Polygons", 1),
        ("str", "Notes", 1),
    ]
    raw_rows = [raw_headers]
    for r_i, rec in enumerate(data, start=2):
        ms_cached = round(rec["total_s"] * 1000.0 / rec["frames"], 2) if rec["frames"] else 0.0
        ps_cached = round(rec["particles"] * rec["frames"] / max(rec["total_s"], 1e-6)) if rec["total_s"] else 0
        raw_rows.append([
            ("str", rec["simulator"], 0),
            ("str", rec["preset_or_res"], 0),
            ("str", rec["mode"], 0),
            ("str", rec["backend"], 0),
            ("num", rec["frames"], 3),
            ("num", rec["particles"], 3),
            ("num", rec["physics_s"], 4),
            ("num", rec["mesh_s"], 4),
            ("num", rec["total_s"], 4),
            ("formula", f"I{r_i}/E{r_i}*1000", ms_cached, 5),
            ("formula", f"IF(I{r_i}>0,(F{r_i}*E{r_i})/I{r_i},0)", ps_cached, 3),
            ("num", rec["mesh_verts"], 3) if rec["mesh_verts"] is not None else ("str", "-", 0),
            ("num", rec["mesh_polys"], 3) if rec["mesh_polys"] is not None else ("str", "-", 0),
            ("str", rec["notes"], 7),
        ])

    for _ in ():
        pass

    # Sheet 1: Head-to-Head Summary (referencing Raw Benchmark Data!)
    sum_rows = [
        [
            ("str", "Scale Tier", 1),
            ("str", "Workload", 1),
            ("str", "Blender Config", 1),
            ("str", "Blender Particles", 1),
            ("str", "Blender Time (s)", 1),
            ("str", "LiquidRust Config", 1),
            ("str", "LiquidRust Particles", 1),
            ("str", "LiquidRust Time (s)", 1),
            ("str", "Speedup (Blender / LR)", 1),
        ]
    ]

    # Pick closest particle-count comparisons dynamically from captured runs
    lr_rows = [(i + 2, r) for i, r in enumerate(data) if r["simulator"].startswith("LiquidRust")]
    bl_rows = [(i + 2, r) for i, r in enumerate(data) if r["simulator"].startswith("Blender")]

    def closest_lr(target_p, backend_sub, mode):
        cands = [(i, r) for i, r in lr_rows if backend_sub in r["backend"] and r["mode"] == mode]
        if not cands:
            return None, None
        return min(cands, key=lambda pair: abs(pair[1]["particles"] - target_p))

    for bi, brec in bl_rows:
        p = brec["particles"]
        mode = brec["mode"]
        tier = f"~{max(1, round(p / 1000))}k particles"
        for backend_sub in (["Vulkan GPU", "CPU Rayon"] if mode == "Physics Only" else ["Vulkan GPU"]):
            li, lrec = closest_lr(p, backend_sub, mode)
            if lrec is None:
                continue
            r_num = len(sum_rows) + 1
            spd_cached = round(brec["total_s"] / max(lrec["total_s"], 1e-6), 2)
            sum_rows.append([
                ("str", tier, 0),
                ("str", f"{mode} ({backend_sub})", 0),
                ("str", brec["preset_or_res"], 0),
                ("formula", f"'Raw Benchmark Data'!F{bi}", brec["particles"], 3),
                ("formula", f"'Raw Benchmark Data'!I{bi}", brec["total_s"], 4),
                ("str", f"{lrec['preset_or_res']} ({backend_sub})", 0),
                ("formula", f"'Raw Benchmark Data'!F{li}", lrec["particles"], 3),
                ("formula", f"'Raw Benchmark Data'!I{li}", lrec["total_s"], 4),
                ("formula", f"IF(H{r_num}>0,E{r_num}/H{r_num},0)", spd_cached, 6),
            ])

    # Sheet 3: Hardware & Methodology
    meta_rows = [
        [("str", "Parameter", 1), ("str", "Value / Detail", 1)],
        [("str", "CPU", 0), ("str", "AMD Ryzen AI 9 HX 370 (24 hardware threads)", 0)],
        [("str", "GPU", 0), ("str", "AMD Radeon 890M Graphics (RADV STRIX1, Vulkan 1.4)", 0)],
        [("str", "Blender Version", 0), ("str", "Blender 5.3.0 (Mantaflow FLIP/APIC CPU multi-threaded, OPENVDB data cache)", 0)],
        [("str", "LiquidRust Version", 0), ("str", "LiquidRust v1.0.1 (Vulkan compute PBF / Rayon CPU PBF)", 0)],
        [("str", "Test Scene", 0), ("str", "3D Dam Break: 1.60m x 0.90m x 1.20m domain; 0.65m x 0.90m x 0.85m water block", 0)],
        [("str", "Timeline", 0), ("str", "90 frames (3.00 seconds simulated at 30 FPS)", 0)],
        [("str", "Note on Mesh Export", 0), ("str", "Blender bake_mesh() writes binary BOBJECT cache; LiquidRust --export-obj writes ASCII Wavefront .obj text files", 7)],
        [("str", "Note on Solvers", 0), ("str", "Blender uses Eulerian grid + FLIP particles; LiquidRust uses Position-Based Fluids (SPH/PBF)", 7)],
    ]

    sheets = [
        ("Comparison Summary", make_sheet_xml(sum_rows, [16, 26, 22, 16, 16, 26, 18, 17, 20], freeze_row=1)),
        ("Raw Benchmark Data", make_sheet_xml(raw_rows, [26, 22, 15, 16, 18, 13, 15, 19, 17, 13, 20, 14, 14, 48], freeze_row=1)),
        ("Hardware & Method", make_sheet_xml(meta_rows, [24, 88], freeze_row=1)),
    ]

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    wb_sheets = []
    wb_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '<Relationship Id="rIdStyle" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
    ]
    for i, (name, _) in enumerate(sheets, start=1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        wb_sheets.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
        wb_rels.append(
            f'<Relationship Id="rId{i}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
        )
    content_types.append("</Types>")
    wb_rels.append("</Relationships>")

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>\n'
        '</Relationships>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
        f'<sheets>{"".join(wb_sheets)}</sheets>\n'
        '</workbook>'
    )

    with zipfile.ZipFile(OUT_XLSX, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "\n".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", "\n".join(wb_rels))
        zf.writestr("xl/styles.xml", build_styles_xml())
        for i, (_, xml_str) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", xml_str)

    print(f"Created {OUT_XLSX} ({os.path.getsize(OUT_XLSX)} bytes, {len(data)} benchmark rows)")


if __name__ == "__main__":
    main()
