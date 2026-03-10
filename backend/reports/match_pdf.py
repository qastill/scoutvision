"""
match_pdf.py — Generate dark-themed Match Report PDF using ReportLab
3 pages: Cover, Team Comparison, Full Player Table
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import PageBreak
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.colors import HexColor
import io

# ── Palette ─────────────────────────────────────────────────────────────────
BG        = HexColor("#080C12")
GREEN     = HexColor("#00E676")
CYAN      = HexColor("#18FFFF")
AMBER     = HexColor("#FFD600")
WHITE     = HexColor("#FFFFFF")
GREY      = HexColor("#B0BEC5")
DARK_CARD = HexColor("#0D1520")
MID_CARD  = HexColor("#111B2A")
BORDER    = HexColor("#1E3050")

PAGE_W, PAGE_H = A4


# ── Dark background canvas callback ─────────────────────────────────────────
class DarkCanvas(SimpleDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)

    def handle_pageBegin(self):
        super().handle_pageBegin()
        self.canv.setFillColor(BG)
        self.canv.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)


def _page_background(c, doc):
    """Draw dark background on every page."""
    c.saveState()
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # Top accent bar
    c.setFillColor(GREEN)
    c.rect(0, PAGE_H - 4, PAGE_W, 4, fill=1, stroke=0)

    # Footer
    c.setFillColor(BORDER)
    c.rect(0, 0, PAGE_W, 12, fill=1, stroke=0)
    c.setFillColor(GREY)
    c.setFont("Helvetica", 7)
    c.drawString(20, 3, "ScoutVision Analytics Platform")
    c.drawRightString(PAGE_W - 20, 3, f"Page {doc.page}")
    c.restoreState()


def generate_match_pdf(results: dict, output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
    )

    story = []
    team_a = results.get("teamA", "Tim A")
    team_b = results.get("teamB", "Tim B")
    duration = results.get("duration", "90:00")
    date = results.get("date", "2024-01-01")
    players = results.get("players", [])
    team_stats = results.get("teamStats", {})

    # ── PAGE 1: Cover ────────────────────────────────────────────────────────
    story += _cover_page(team_a, team_b, duration, date)
    story.append(PageBreak())

    # ── PAGE 2: Team Comparison ──────────────────────────────────────────────
    story += _team_comparison_page(team_a, team_b, team_stats, players)
    story.append(PageBreak())

    # ── PAGE 3: Full Player Table ────────────────────────────────────────────
    story += _player_table_page(players)

    doc.build(story, onFirstPage=_page_background, onLaterPages=_page_background)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _style(name, **kwargs):
    base = {
        "fontName": "Helvetica",
        "fontSize": 10,
        "textColor": WHITE,
        "leading": 14,
    }
    base.update(kwargs)
    return ParagraphStyle(name, **base)


def _cover_page(team_a, team_b, duration, date):
    elems = []

    # Spacer to push content down
    elems.append(Spacer(1, 40 * mm))

    # SCOUTVISION logo text
    elems.append(Paragraph(
        '<font color="#00E676"><b>⚽ SCOUT</b></font><font color="#18FFFF"><b>VISION</b></font>',
        _style("logo", fontSize=28, alignment=TA_CENTER, leading=34)
    ))
    elems.append(Spacer(1, 3 * mm))
    elems.append(Paragraph(
        "Football Analytics Platform",
        _style("sub", fontSize=11, textColor=GREY, alignment=TA_CENTER)
    ))
    elems.append(Spacer(1, 12 * mm))

    # Accent line
    elems.append(HRFlowable(width="70%", thickness=1, color=GREEN, spaceAfter=10))

    # MATCH REPORT title
    elems.append(Paragraph(
        "MATCH REPORT",
        _style("title", fontSize=32, fontName="Helvetica-Bold",
               textColor=WHITE, alignment=TA_CENTER, leading=40)
    ))
    elems.append(Spacer(1, 10 * mm))

    # Teams
    teams_data = [[
        Paragraph(f'<font color="#00E676"><b>{team_a}</b></font>',
                  _style("ta", fontSize=20, alignment=TA_RIGHT)),
        Paragraph('<font color="#FFD600"><b>VS</b></font>',
                  _style("vs", fontSize=18, alignment=TA_CENTER)),
        Paragraph(f'<font color="#18FFFF"><b>{team_b}</b></font>',
                  _style("tb", fontSize=20, alignment=TA_LEFT)),
    ]]
    t = Table(teams_data, colWidths=[70 * mm, 30 * mm, 70 * mm])
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), DARK_CARD),
        ("ROUNDEDCORNERS", [8]),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 10 * mm))

    # Meta info
    meta = [[
        Paragraph(f'<font color="#B0BEC5">Duration</font><br/><font color="#FFFFFF"><b>{duration}</b></font>',
                  _style("m1", fontSize=11, alignment=TA_CENTER)),
        Paragraph(f'<font color="#B0BEC5">Date</font><br/><font color="#FFFFFF"><b>{date}</b></font>',
                  _style("m2", fontSize=11, alignment=TA_CENTER)),
    ]]
    mt = Table(meta, colWidths=[85 * mm, 85 * mm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), MID_CARD),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEAFTER", (0, 0), (0, -1), 1, BORDER),
    ]))
    elems.append(mt)

    return elems


def _team_comparison_page(team_a, team_b, team_stats, players):
    elems = []

    elems.append(Spacer(1, 5 * mm))
    elems.append(Paragraph(
        '<font color="#00E676">TEAM</font> <font color="#FFFFFF">COMPARISON</font>',
        _style("hdr", fontSize=20, fontName="Helvetica-Bold", alignment=TA_CENTER)
    ))
    elems.append(Spacer(1, 2 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=6))

    stats_a = team_stats.get(team_a, {})
    stats_b = team_stats.get(team_b, {})

    def val(d, key, default="-"):
        v = d.get(key, default)
        return str(v) if v != "-" else "-"

    # Find best players
    best_a_id = stats_a.get("bestPlayer", "-")
    best_b_id = stats_b.get("bestPlayer", "-")

    rows = [
        # Header
        [
            Paragraph(f'<font color="#00E676"><b>{team_a}</b></font>',
                      _style("h", fontSize=12, alignment=TA_CENTER)),
            Paragraph('<font color="#B0BEC5">Metric</font>',
                      _style("h", fontSize=10, alignment=TA_CENTER)),
            Paragraph(f'<font color="#18FFFF"><b>{team_b}</b></font>',
                      _style("h", fontSize=12, alignment=TA_CENTER)),
        ],
        _cmp_row("Total Distance", f'{val(stats_a, "totalDist")} km', f'{val(stats_b, "totalDist")} km'),
        _cmp_row("Avg Dist / Player", f'{val(stats_a, "avgDist")} km', f'{val(stats_b, "avgDist")} km'),
        _cmp_row("Total Sprint Dist", f'{val(stats_a, "totalSprint")} km', f'{val(stats_b, "totalSprint")} km'),
        _cmp_row("Top Speed", f'{val(stats_a, "topSpeed")} km/h', f'{val(stats_b, "topSpeed")} km/h'),
        _cmp_row("Avg Rating", val(stats_a, "avgRating"), val(stats_b, "avgRating")),
        _cmp_row("Players", val(stats_a, "players"), val(stats_b, "players")),
        _cmp_row("Best Player", f'#{best_a_id}', f'#{best_b_id}'),
        # Ball stats
        _cmp_row("Total Passes", val(stats_a, "totalPasses"), val(stats_b, "totalPasses")),
        _cmp_row("Avg Pass Accuracy", f'{val(stats_a, "avgPassAccuracy")}%', f'{val(stats_b, "avgPassAccuracy")}%'),
        _cmp_row("Total Shots", val(stats_a, "totalShots"), val(stats_b, "totalShots")),
        _cmp_row("Total xG", val(stats_a, "totalXg"), val(stats_b, "totalXg")),
        _cmp_row("Total Goals", val(stats_a, "totalGoals"), val(stats_b, "totalGoals")),
        _cmp_row("Total Tackles", val(stats_a, "totalTackles"), val(stats_b, "totalTackles")),
    ]

    col_w = [(PAGE_W - 30 * mm) / 3] * 3
    t = Table(rows, colWidths=col_w)
    style = [
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), MID_CARD),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        # Data rows alternating
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DARK_CARD, MID_CARD]),
        ("TOPPADDING", (0, 1), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, BORDER),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        # Metric column
        ("BACKGROUND", (1, 1), (1, -1), HexColor("#0A1018")),
    ]
    t.setStyle(TableStyle(style))
    elems.append(t)

    return elems


def _cmp_row(metric, val_a, val_b):
    return [
        Paragraph(f'<font color="#FFFFFF"><b>{val_a}</b></font>',
                  _style("v", fontSize=11, alignment=TA_CENTER)),
        Paragraph(f'<font color="#B0BEC5">{metric}</font>',
                  _style("m", fontSize=9, alignment=TA_CENTER)),
        Paragraph(f'<font color="#FFFFFF"><b>{val_b}</b></font>',
                  _style("v", fontSize=11, alignment=TA_CENTER)),
    ]


def _player_table_page(players):
    elems = []

    elems.append(Spacer(1, 5 * mm))
    elems.append(Paragraph(
        '<font color="#00E676">PLAYER</font> <font color="#FFFFFF">STATISTICS</font>',
        _style("hdr", fontSize=20, fontName="Helvetica-Bold", alignment=TA_CENTER)
    ))
    elems.append(Spacer(1, 2 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=6))

    headers = ["No", "No.Jrsy", "Pos", "Team", "Dist(km)", "Sprint(km)", "Top Spd", "Sprints", "Rating"]
    header_row = [
        Paragraph(f'<font color="#FFD600"><b>{h}</b></font>',
                  _style("th", fontSize=7, alignment=TA_CENTER))
        for h in headers
    ]

    data_rows = [header_row]
    for i, p in enumerate(players):
        team_col = p.get("teamColor", "#FFFFFF")
        pid = p["id"]
        display_name = p.get("name", f"Pemain #{pid}")
        display_number = str(p.get("number", pid))
        # Truncate long names for table
        display_name_short = display_name[:10] if len(display_name) > 10 else display_name
        row = [
            Paragraph(f'<font color="#B0BEC5">{i + 1}</font>', _style("td", fontSize=7, alignment=TA_CENTER)),
            Paragraph(f'<font color="#FFFFFF"><b>{display_number}</b></font>', _style("td", fontSize=7, alignment=TA_CENTER)),
            Paragraph(f'<font color="#18FFFF">{p.get("position", "CM")}</font>', _style("td", fontSize=7, alignment=TA_CENTER)),
            Paragraph(f'<font color="{team_col}">{p.get("team", "-")[:8]}</font>', _style("td", fontSize=6, alignment=TA_CENTER)),
            Paragraph(f'<font color="#FFFFFF">{p.get("totalDist", 0):.2f}</font>', _style("td", fontSize=7, alignment=TA_CENTER)),
            Paragraph(f'<font color="#FFFFFF">{p.get("sprintDist", 0):.2f}</font>', _style("td", fontSize=7, alignment=TA_CENTER)),
            Paragraph(f'<font color="#FFFFFF">{p.get("topSpeed", 0):.1f}</font>', _style("td", fontSize=7, alignment=TA_CENTER)),
            Paragraph(f'<font color="#FFFFFF">{p.get("sprints", 0)}</font>', _style("td", fontSize=7, alignment=TA_CENTER)),
            Paragraph(_rating_badge(p.get("rating", 5.0)), _style("td", fontSize=7, alignment=TA_CENTER)),
        ]
        data_rows.append(row)

    col_widths = [10 * mm, 12 * mm, 12 * mm, 22 * mm, 18 * mm, 20 * mm, 17 * mm, 16 * mm, 15 * mm]
    t = Table(data_rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0A1018")),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DARK_CARD, MID_CARD]),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
    ]
    t.setStyle(TableStyle(style))
    elems.append(t)

    return elems


def _rating_badge(rating: float) -> str:
    if rating >= 8.5:
        color = "#00E676"
    elif rating >= 7.0:
        color = "#FFD600"
    else:
        color = "#EF5350"
    return f'<font color="{color}"><b>{rating:.1f}</b></font>'
