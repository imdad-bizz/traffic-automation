"""
Professional Excel Report Exporter for Traffic Violations & ALPR
================================================================
Generates high-precision, executive-ready Excel (.xlsx) workbooks.
Features:
- Real visual image thumbnails embedded directly into worksheet cells
- Clickable hyperlinks pointing to full-resolution evidence files
- PC clock date & time for every violation event
- Comprehensive element mapping (Vehicle, Speed, Plate, Confidence, Status)
- Executive KPI dashboard cards with automated statistics
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import openpyxl
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

logger = logging.getLogger(__name__)


def export_to_excel(
    records: List[Dict[str, Any]],
    output_path: str = "outputs/traffic_violations_alpr.xlsx",
    speed_limit_kmh: float = 60.0,
    speed_tolerance_kmh: float = 5.0,
) -> str:
    """
    Exports traffic detection records into an executive Excel workbook with embedded
    thumbnails, PC timestamps, and complete relational mapping.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Traffic Surveillance & ALPR"

    # Display gridlines
    ws.views.sheetView[0].showGridLines = True

    # ---------------------------------------------------------
    # Styles definition
    # ---------------------------------------------------------
    font_family = "Segoe UI"
    bengali_font_name = "Nirmala UI"

    title_fill = PatternFill(start_color="0D233A", end_color="0D233A", fill_type="solid")
    kpi_card_fill = PatternFill(start_color="F0F4F8", end_color="F0F4F8", fill_type="solid")
    kpi_viol_fill = PatternFill(start_color="FDE8E8", end_color="FDE8E8", fill_type="solid")
    kpi_norm_fill = PatternFill(start_color="E6F4EA", end_color="E6F4EA", fill_type="solid")

    header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    header_font = Font(name=font_family, size=10, bold=True, color="FFFFFF")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    violation_row_fill = PatternFill(start_color="FFF1F0", end_color="FFF1F0", fill_type="solid")
    violation_status_fill = PatternFill(start_color="F87171", end_color="F87171", fill_type="solid")
    violation_status_font = Font(name=font_family, size=10, bold=True, color="FFFFFF")

    normal_status_fill = PatternFill(start_color="34D399", end_color="34D399", fill_type="solid")
    normal_status_font = Font(name=font_family, size=10, bold=True, color="FFFFFF")

    regular_font = Font(name=font_family, size=10)
    bengali_font = Font(name=bengali_font_name, size=11, bold=True, color="0F172A")
    number_font = Font(name="Consolas", size=10, bold=True)
    link_font = Font(name=font_family, size=9, underline="single", color="1D4ED8")

    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1"),
    )
    card_border = Border(
        left=Side(style="medium", color="94A3B8"),
        right=Side(style="medium", color="94A3B8"),
        top=Side(style="medium", color="94A3B8"),
        bottom=Side(style="medium", color="94A3B8"),
    )

    # ---------------------------------------------------------
    # Row 1: Executive Title Banner
    # ---------------------------------------------------------
    ws.merge_cells("A1:Q1")
    t_cell = ws["A1"]
    t_cell.value = "BANGLADESH HIGHWAY TRAFFIC SURVEILLANCE & ALPR AUDIT REPORT"
    t_cell.font = Font(name=font_family, size=15, bold=True, color="FFFFFF")
    t_cell.fill = title_fill
    t_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 40

    # ---------------------------------------------------------
    # Row 2-4: KPI Summary Cards & Metadata
    # ---------------------------------------------------------
    total_records = len(records)
    total_violators = sum(1 for r in records if (
        r.get("speeding", 0) == 1 
        or float(r.get("speed_kmh", 0)) > (speed_limit_kmh + speed_tolerance_kmh)
        or float(r.get("peak_speed_kmh", 0)) > (speed_limit_kmh + speed_tolerance_kmh)
        or "VIOLATION" in str(r.get("reason", "")).upper()
    ))
    total_normal = total_records - total_violators
    compliance_rate = (total_normal / max(1, total_records)) * 100

    now_dt = datetime.datetime.now()
    generated_pc_time = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    # Card 1: Total Monitored
    ws.merge_cells("A2:C2")
    ws["A2"] = "TOTAL VEHICLES"
    ws["A2"].font = Font(name=font_family, size=9, bold=True, color="64748B")
    ws["A2"].fill = kpi_card_fill
    ws["A2"].alignment = Alignment(horizontal="center", vertical="center")

    ws.merge_cells("A3:C3")
    ws["A3"] = total_records
    ws["A3"].font = Font(name=font_family, size=16, bold=True, color="0F172A")
    ws["A3"].fill = kpi_card_fill
    ws["A3"].alignment = Alignment(horizontal="center", vertical="center")

    # Card 2: Violations
    ws.merge_cells("E2:G2")
    ws["E2"] = "SPEED VIOLATIONS"
    ws["E2"].font = Font(name=font_family, size=9, bold=True, color="B91C1C")
    ws["E2"].fill = kpi_viol_fill
    ws["E2"].alignment = Alignment(horizontal="center", vertical="center")

    ws.merge_cells("E3:G3")
    ws["E3"] = total_violators
    ws["E3"].font = Font(name=font_family, size=16, bold=True, color="DC2626")
    ws["E3"].fill = kpi_viol_fill
    ws["E3"].alignment = Alignment(horizontal="center", vertical="center")

    # Card 3: Normal Compliance
    ws.merge_cells("I2:K2")
    ws["I2"] = "NORMAL VEHICLES"
    ws["I2"].font = Font(name=font_family, size=9, bold=True, color="047857")
    ws["I2"].fill = kpi_norm_fill
    ws["I2"].alignment = Alignment(horizontal="center", vertical="center")

    ws.merge_cells("I3:K3")
    ws["I3"] = f"{total_normal} ({compliance_rate:.1f}%)"
    ws["I3"].font = Font(name=font_family, size=15, bold=True, color="059669")
    ws["I3"].fill = kpi_norm_fill
    ws["I3"].alignment = Alignment(horizontal="center", vertical="center")

    # Card 4: Parameters
    ws.merge_cells("M2:Q2")
    ws["M2"] = "ENFORCEMENT PARAMETERS & CLOCK"
    ws["M2"].font = Font(name=font_family, size=9, bold=True, color="475569")
    ws["M2"].fill = kpi_card_fill
    ws["M2"].alignment = Alignment(horizontal="center", vertical="center")

    ws.merge_cells("M3:Q3")
    ws["M3"] = f"Speed Limit: {speed_limit_kmh:.0f} km/h (+{speed_tolerance_kmh:.0f} buffer) | PC Clock: {generated_pc_time}"
    ws["M3"].font = Font(name=font_family, size=9, bold=True, color="334155")
    ws["M3"].fill = kpi_card_fill
    ws["M3"].alignment = Alignment(horizontal="center", vertical="center")

    ws.row_dimensions[2].height = 18
    ws.row_dimensions[3].height = 28
    ws.row_dimensions[4].height = 8

    # ---------------------------------------------------------
    # Row 5: Column Headers
    # ---------------------------------------------------------
    columns = [
        ("Record ID", 13),
        ("Violation Date (PC)", 18),
        ("Violation Time (PC)", 18),
        ("Video Time (s)", 14),
        ("Video Time (MM:SS)", 16),
        ("Track ID", 11),
        ("Vehicle Class", 15),
        ("Violation Type", 18),
        ("Measured Speed", 16),
        ("Speed Limit", 14),
        ("Excess Speed", 14),
        ("Violation Status", 18),
        ("License Plate (Bengali)", 25),
        ("License Plate (English)", 28),
        ("Extraction Source", 20),
        ("Confidence", 13),
        ("Clear Plate Snapshot", 28),
        ("Vehicle Context Snapshot", 28),
    ]

    header_row = 5
    ws.row_dimensions[header_row].height = 28

    for col_idx, (col_name, _) in enumerate(columns, start=1):
        c = ws.cell(row=header_row, column=col_idx, value=col_name)
        c.font = header_font
        c.fill = header_fill
        c.alignment = header_align
        c.border = thin_border

    # ---------------------------------------------------------
    # Data Rows (Row 6 onwards)
    # ---------------------------------------------------------
    row_idx = 6

    for idx, item in enumerate(records, start=1):
        track_id = int(item.get("track_id", 0))
        ts_sec = float(item.get("timestamp_sec", 0.0))
        mins = int(ts_sec // 60)
        secs = int(ts_sec % 60)
        video_time_mmss = f"{mins:02d}:{secs:02d}"

        # Determine PC clock date & time
        pc_date = item.get("pc_date")
        pc_time = item.get("pc_time")
        if not pc_date or not pc_time:
            # If not previously stored in record, calculate from now or base timestamp
            event_dt = now_dt
            pc_date = event_dt.strftime("%Y-%m-%d")
            pc_time = event_dt.strftime("%H:%M:%S")

        speed_kmh = float(item.get("speed_kmh", 0.0))
        peak_kmh = float(item.get("peak_speed_kmh", speed_kmh))
        eff_speed = max(speed_kmh, peak_kmh)

        v_type_raw = str(item.get("reason", "")).lower()
        is_speeding = (
            bool(item.get("speeding", 0))
            or eff_speed > (speed_limit_kmh + speed_tolerance_kmh)
            or "speed" in v_type_raw
        )

        excess_speed = max(0.0, eff_speed - speed_limit_kmh) if is_speeding else 0.0

        if is_speeding:
            viol_type = "Speed Violation"
            status_text = "🚨 SPEEDING"
        elif "opposite" in v_type_raw or "wrong" in v_type_raw:
            viol_type = "Wrong-Way / Lane"
            status_text = "🚨 WRONG-WAY"
        elif "red" in v_type_raw:
            viol_type = "Red Light Violation"
            status_text = "🚨 RED LIGHT"
        else:
            viol_type = "None (Compliant)"
            status_text = "✅ NORMAL"

        plate_bn = item.get("plate_bn", "")
        plate_en = item.get("plate_en", "")
        conf = float(item.get("plate_conf", 0.0))
        source = item.get("recognition_source") or ("GPT-4o Vision" if conf >= 0.90 else "Local EasyOCR")

        # Guess vehicle class based on plate or bounding box
        v_class = item.get("vehicle_class", "Vehicle")
        if not v_class or v_class == "Vehicle":
            if "ড" in plate_bn or "DA" in plate_en or "ট" in plate_bn or "TA" in plate_en:
                v_class = "Truck / Covered Van"
            elif "ন" in plate_bn or "NA" in plate_en:
                v_class = "Pickup / Microbus"
            elif "ক" in plate_bn or "KA" in plate_en:
                v_class = "Private Car"
            elif "হ" in plate_bn or "HA" in plate_en:
                v_class = "Motorcycle"
            else:
                v_class = "Commercial / Bus"

        record_id = f"REC-{idx:03d}"
        plate_crop_path = item.get("plate_crop_path", "")
        snap_path = item.get("snapshot_path", "")

        row_values = [
            record_id,
            pc_date,
            pc_time,
            f"{ts_sec:.2f}s",
            video_time_mmss,
            track_id,
            v_class,
            viol_type,
            f"{eff_speed:.1f} km/h",
            f"{speed_limit_kmh:.0f} km/h",
            f"+{excess_speed:.1f} km/h" if excess_speed > 0 else "0.0 km/h",
            status_text,
            plate_bn or "—",
            plate_en or "—",
            source,
            f"{conf * 100:.1f}%" if conf > 0 else "—",
            "",  # Plate image thumbnail placeholder
            "",  # Vehicle image thumbnail placeholder
        ]

        # Allocate comfortable height for visual thumbnails
        ws.row_dimensions[row_idx].height = 68
        is_even = (row_idx % 2 == 0)
        bg = violation_row_fill if is_speeding else (zebra_fill if is_even else None)

        for col_idx, val in enumerate(row_values, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.border = thin_border
            cell.font = regular_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

            if bg:
                cell.fill = bg

            # Numeric columns
            if col_idx in (1, 6, 9, 10, 11):
                cell.font = number_font

            # Bengali Plate font
            if col_idx == 13 and plate_bn:
                cell.font = bengali_font

            # Status column badge
            if col_idx == 12:
                if is_speeding or "VIOLATION" in status_text:
                    cell.fill = violation_status_fill
                    cell.font = violation_status_font
                else:
                    cell.fill = normal_status_fill
                    cell.font = normal_status_font

            # Hyperlinks on image placeholder cells
            if col_idx == 17 and plate_crop_path and Path(plate_crop_path).exists():
                cell.hyperlink = str(Path(plate_crop_path).resolve())
                cell.font = link_font
                cell.value = "Click to View Plate"

            if col_idx == 18 and snap_path and Path(snap_path).exists():
                cell.hyperlink = str(Path(snap_path).resolve())
                cell.font = link_font
                cell.value = "Click to View Vehicle"

        # Embed actual visual thumbnails into Col 17 (Plate) & Col 18 (Vehicle)
        try:
            if plate_crop_path and Path(plate_crop_path).exists():
                xl_p_img = XLImage(str(Path(plate_crop_path).resolve()))
                # Scale cleanly to fit in cell
                pw, ph = xl_p_img.width, xl_p_img.height
                target_ph = 60
                scale_p = target_ph / float(max(1, ph))
                xl_p_img.width = int(pw * scale_p)
                xl_p_img.height = target_ph
                col_letter = get_column_letter(17)
                ws.add_image(xl_p_img, f"{col_letter}{row_idx}")
        except Exception as img_err:
            logger.debug("Could not embed plate thumbnail in Excel: %s", img_err)

        try:
            if snap_path and Path(snap_path).exists():
                xl_v_img = XLImage(str(Path(snap_path).resolve()))
                vw, vh = xl_v_img.width, xl_v_img.height
                target_vh = 60
                scale_v = target_vh / float(max(1, vh))
                xl_v_img.width = int(vw * scale_v)
                xl_v_img.height = target_vh
                col_letter = get_column_letter(18)
                ws.add_image(xl_v_img, f"{col_letter}{row_idx}")
        except Exception as img_err:
            logger.debug("Could not embed vehicle thumbnail in Excel: %s", img_err)

        row_idx += 1

    # Auto-adjust column widths
    for col_idx, (_, default_width) in enumerate(columns, start=1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = default_width

    try:
        wb.save(str(out_file))
        logger.info("[ExcelExporter] Saved executive ALPR workbook with %d records to %s", len(records), out_file)
        return str(out_file)
    except PermissionError:
        ts_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = out_file.parent / f"{out_file.stem}_{ts_suffix}.xlsx"
        logger.warning(
            "[ExcelExporter] File %s is locked (e.g. open in Excel). Saving to fallback: %s",
            out_file,
            fallback_path,
        )
        wb.save(str(fallback_path))
        logger.info("[ExcelExporter] Saved executive ALPR workbook with %d records to %s", len(records), fallback_path)
        return str(fallback_path)
