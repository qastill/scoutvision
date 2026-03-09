"""
player_pdf.py — Generate dark-themed Player Report PDF using ReportLab
2 pages per player: Header/KPIs, Speed Zones/Movement
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas as pdf_canvas

# ── Palette ──────────────────────────────────────────────────────────────────
BG        = HexColor("#080C12")
GREEN     = HexColor("#00E676")
CYAN      = HexColor("#18FFFF")
AMBER     = HexColor("#FFD600")
WHITE     = HexColor("#FFFFFF")
GREY      = HexColor("#B0BEC5")
DARK_CARD = HexColor("#0D1520")
MID_CARD  = HexColor("#111B2A")
BORDER    = HexColor("#1E3050")
RED       = HexColor("#EF5350")

PAGE_W, PAGE_H = A4


def _page_bg(c, doc):
    c.saveState()
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(GREEN)
    c.rect(0, PAGE_H - 4, PAGE_W, 4, fill=1, stroke=0)
    c.setFillColor(BORDER)
    c.rect(0, 0, PAGE_W, 12, fill=1, stroke=0)
    c.setFillColor(GREY)
    c.setFont("Helvetica", 7)
    c.drawString(20, 3, "ScoutVision Analytics Platform")
    c.drawRightString(PAGE_W - 20, 3, f"Page {doc.page}")
    c.restoreState()


def _style(name, **kwargs):
    base = {"fontName": "Helvetica", "fontSize": 10, "textColor": WHITE, "leading": 14}
    base.update(kwargs)
    return ParagraphStyle(name, **base)


def generate_player_pdf(player: dict, match_results: dict, output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
    )
    story = []

    pid = player.get("id", "?")
    team = player.get("team", "Unknown")
    team_color = player.get("teamColor", "#FFFFFF")
    position = player.get("position", "CM")
    total_dist = player.get("totalDist", 0.0)
    sprint_dist = player.get("sprintDist", 0.0)
    top_speed = player.get("topSpeed", 0.0)
    sprints = player.get("sprints", 0)
    rating = player.get("rating", 5.0)
    fatigue = player.get("fatigue", "Normal")
    walk_km = player.get("walk", 0.0)
    jog_km = player.get("jog", 0.0)
    hirun_km = player.get("hiRun", 0.0)
    avg_x = player.get("avgX", 52.5)
    avg_y = player.get("avgY", 34.0)
    jersey = player.get("jerseyColor", [128, 128, 128])
    team_name_a = match_results.get("teamA", "Tim A")
    duration = match_results.get("duration", "90:00")
    date = match_results.get("date", "")

    jersey_hex = f"#{int(jersey[0]):02X}{int(jersey[1]):02X}{int(jersey[2]):02X}" if len(jersey) >= 3 else "#808080"

    # ── PAGE 1 ────────────────────────────────────────────────────────────────
    story += _page1(
        pid, team, team_color, position, total_dist, sprint_dist,
        top_speed, sprints, rating, fatigue, avg_x, avg_y,
        jersey_hex, duration, date
    )
    story.append(PageBreak())

    # ── PAGE 2 ────────────────────────────────────────────────────────────────
    story += _page2(
        pid, total_dist, walk_km, jog_km, hirun_km, sprint_dist,
        top_speed, sprints, position, avg_x, avg_y
    )

    doc.build(story, onFirstPage=_page_bg, onLaterPages=_page_bg)


def _page1(pid, team, team_color, position, total_dist, sprint_dist,
           top_speed, sprints, rating, fatigue, avg_x, avg_y,
           jersey_hex, duration, date):
    elems = []
    elems.append(Spacer(1, 3 * mm))

    # Header bar
    header_data = [[
        Paragraph(
            f'<font color="{team_color}"><b>PLAYER #{pid}</b></font>',
            _style("ph", fontSize=22, fontName="Helvetica-Bold")
        ),
        Paragraph(
            f'<font color="#B0BEC5">Position</font> <font color="#18FFFF"><b>{position}</b></font>  '
            f'<font color="#B0BEC5">  Team</font> <font color="{team_color}"><b>{team}</b></font>',
            _style("ph2", fontSize=11, alignment=TA_RIGHT)
        ),
    ]]
    ht = Table(header_data, colWidths=[90 * mm, 90 * mm])
    ht.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), MID_CARD),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (0, -1), 12),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elems.append(ht)
    elems.append(Spacer(1, 4 * mm))

    # KPI boxes row 1
    kpi1 = [
        _kpi_cell("Total Distance", f"{total_dist:.2f} km", GREEN, "🏃"),
        _kpi_cell("Top Speed", f"{top_speed:.1f} km/h", CYAN, "⚡"),
        _kpi_cell("Sprint Distance", f"{sprint_dist:.2f} km", AMBER, "💨"),
    ]
    kpi1_table = Table([kpi1], colWidths=[58 * mm, 58 * mm, 58 * mm])
    kpi1_table.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elems.append(kpi1_table)
    elems.append(Spacer(1, 3 * mm))

    # KPI boxes row 2
    kpi2 = [
        _kpi_cell("Sprint Count", str(sprints), HexColor("#FF6D00"), "🔥"),
        _kpi_cell("Rating", f"{rating:.1f} / 10", _rating_color(rating), "⭐"),
        _kpi_cell("Fatigue Level", fatigue, RED if fatigue == "Tinggi" else GREEN, "💪"),
    ]
    kpi2_table = Table([kpi2], colWidths=[58 * mm, 58 * mm, 58 * mm])
    kpi2_table.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elems.append(kpi2_table)
    elems.append(Spacer(1, 5 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 4 * mm))

    # Position zone indicator
    elems.append(Paragraph(
        '<font color="#FFD600">FIELD POSITION ZONE</font>',
        _style("sec", fontSize=12, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 3 * mm))

    zone_pct_x = avg_x / 105.0
    zone_pct_y = avg_y / 68.0
    zone_label = _zone_label(avg_x, avg_y)

    zone_info = [[
        Paragraph(
            f'<font color="#B0BEC5">Avg X Position:</font> '
            f'<font color="#18FFFF"><b>{avg_x:.1f}m</b></font> (0–105m field length)',
            _style("zi", fontSize=10)
        ),
        Paragraph(
            f'<font color="#B0BEC5">Avg Y Position:</font> '
            f'<font color="#18FFFF"><b>{avg_y:.1f}m</b></font> (0–68m field width)',
            _style("zi2", fontSize=10, alignment=TA_RIGHT)
        ),
    ]]
    zt = Table(zone_info, colWidths=[90 * mm, 90 * mm])
    zt.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elems.append(zt)
    elems.append(Spacer(1, 2 * mm))
    elems.append(Paragraph(
        f'<font color="#B0BEC5">Zone Classification: </font>'
        f'<font color="#00E676"><b>{zone_label}</b></font>',
        _style("zl", fontSize=11)
    ))
    elems.append(Spacer(1, 5 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 4 * mm))

    # Match context
    elems.append(Paragraph(
        '<font color="#FFD600">MATCH CONTEXT</font>',
        _style("sec", fontSize=12, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 3 * mm))
    ctx_data = [[
        Paragraph(f'<font color="#B0BEC5">Duration</font><br/><font color="#FFFFFF"><b>{duration}</b></font>',
                  _style("c", fontSize=10, alignment=TA_CENTER)),
        Paragraph(f'<font color="#B0BEC5">Date</font><br/><font color="#FFFFFF"><b>{date}</b></font>',
                  _style("c", fontSize=10, alignment=TA_CENTER)),
        Paragraph(f'<font color="#B0BEC5">Position</font><br/><font color="#18FFFF"><b>{position}</b></font>',
                  _style("c", fontSize=10, alignment=TA_CENTER)),
        Paragraph(f'<font color="#B0BEC5">Jersey</font><br/>'
                  f'<font color="#FFFFFF"><b>Player #{pid}</b></font>',
                  _style("c", fontSize=10, alignment=TA_CENTER)),
    ]]
    ct = Table(ctx_data, colWidths=[44 * mm, 44 * mm, 44 * mm, 44 * mm])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_CARD),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elems.append(ct)

    return elems


def _page2(pid, total_dist, walk_km, jog_km, hirun_km, sprint_dist,
           top_speed, sprints, position, avg_x, avg_y):
    elems = []
    elems.append(Spacer(1, 3 * mm))

    elems.append(Paragraph(
        f'<font color="#00E676">PLAYER #{pid}</font>'
        f' <font color="#FFFFFF">— Speed Zones & Movement</font>',
        _style("ph", fontSize=16, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 2 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 5 * mm))

    # Speed zone breakdown
    elems.append(Paragraph(
        '<font color="#FFD600">SPEED ZONE BREAKDOWN</font>',
        _style("sec", fontSize=12, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 3 * mm))

    total_safe = max(total_dist, 0.001)
    zones = [
        ("Walk",    walk_km,    "#4FC3F7", "< 8 km/h"),
        ("Jog",     jog_km,     "#00E676", "8–16 km/h"),
        ("Hi-Run",  hirun_km,   "#FFD600", "16–21 km/h"),
        ("Sprint",  sprint_dist, "#FF5252", "> 21 km/h"),
    ]

    for zone_name, dist_km, color_hex, threshold in zones:
        pct = dist_km / total_safe * 100
        bar_row = [[
            Paragraph(f'<font color="#B0BEC5">{zone_name}</font>',
                      _style("zn", fontSize=9, alignment=TA_LEFT)),
            Paragraph(f'<font color="#B0BEC5">{threshold}</font>',
                      _style("zt", fontSize=8, alignment=TA_CENTER, textColor=GREY)),
            Paragraph(f'<font color="{color_hex}"><b>{dist_km:.2f} km</b></font>',
                      _style("zv", fontSize=9, alignment=TA_RIGHT)),
            Paragraph(f'<font color="{color_hex}">{pct:.1f}%</font>',
                      _style("zp", fontSize=9, alignment=TA_RIGHT)),
        ]]
        bt = Table(bar_row, colWidths=[25 * mm, 35 * mm, 30 * mm, 20 * mm])
        bt.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elems.append(bt)

        # Bar visualization
        bar_fill = min(pct / 100.0, 1.0)
        bar_data = [[
            Paragraph(
                f'<font size="1" color="{color_hex}">{"█" * int(bar_fill * 60)}{"░" * (60 - int(bar_fill * 60))}</font>',
                _style("bar", fontSize=8, fontName="Courier", textColor=HexColor(color_hex))
            )
        ]]
        bar_t = Table(bar_data, colWidths=[175 * mm])
        bar_t.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        elems.append(bar_t)

    elems.append(Spacer(1, 5 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 5 * mm))

    # Movement summary table
    elems.append(Paragraph(
        '<font color="#FFD600">MOVEMENT SUMMARY</font>',
        _style("sec", fontSize=12, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 3 * mm))

    summary_data = [
        [_lbl("Total Distance"), _val(f"{total_dist:.2f} km"), _lbl("Sprint Episodes"), _val(str(sprints))],
        [_lbl("Walk"), _val(f"{walk_km:.2f} km"), _lbl("Top Speed"), _val(f"{top_speed:.1f} km/h")],
        [_lbl("Jog"), _val(f"{jog_km:.2f} km"), _lbl("Position"), _val(position)],
        [_lbl("Hi-Run"), _val(f"{hirun_km:.2f} km"), _lbl("Avg Field X"), _val(f"{avg_x:.1f}m")],
        [_lbl("Sprint"), _val(f"{sprint_dist:.2f} km"), _lbl("Avg Field Y"), _val(f"{avg_y:.1f}m")],
    ]

    st = Table(summary_data, colWidths=[40 * mm, 47 * mm, 45 * mm, 43 * mm])
    st.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [DARK_CARD, MID_CARD]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elems.append(st)

    return elems


def _kpi_cell(label, value, color, icon=""):
    inner = [
        [Paragraph(f'{icon} <font color="#B0BEC5">{label}</font>',
                   _style("kl", fontSize=8))],
        [Paragraph(f'<font color="{color.hexval() if hasattr(color, "hexval") else color}"><b>{value}</b></font>',
                   _style("kv", fontSize=14, fontName="Helvetica-Bold"))],
    ]
    inner_t = Table(inner, colWidths=[52 * mm])
    inner_t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_CARD),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("LINEABOVE", (0, 0), (-1, 0), 2, color),
    ]))
    return inner_t


def _lbl(text):
    return Paragraph(f'<font color="#B0BEC5">{text}</font>', _style("l", fontSize=9))


def _val(text):
    return Paragraph(f'<font color="#FFFFFF"><b>{text}</b></font>', _style("v", fontSize=9))


def _rating_color(r):
    if r >= 8.5:
        return GREEN
    elif r >= 7.0:
        return AMBER
    return RED


def _zone_label(avg_x, avg_y):
    if avg_x < 10 or avg_x > 95:
        return "Goalkeeper Zone"
    elif avg_x < 35:
        return "Defensive Third"
    elif avg_x < 70:
        return "Middle Third"
    else:
        return "Attacking Third"
