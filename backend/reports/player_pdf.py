"""
player_pdf.py — Generate dark-themed Player Report PDF using ReportLab
2 pages per player:
  Page 1: Header (photo/swatch + match rating) + KPI boxes (3 rows) + Zone bar + Match context
  Page 2: Speed zones, movement summary, rating breakdown table, hexagonal radar chart
"""
import os
import math
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, Flowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor, Color
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
ORANGE    = HexColor("#FF6D00")

PAGE_W, PAGE_H = A4


# ── Custom Flowables ─────────────────────────────────────────────────────────

class JerseySwatch(Flowable):
    """Colored jersey swatch used when no player photo is available."""
    def __init__(self, color_hex, width=40 * mm, height=50 * mm):
        Flowable.__init__(self)
        try:
            self.swatch_color = HexColor(color_hex)
        except Exception:
            self.swatch_color = HexColor("#808080")
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        # Background
        c.setFillColor(DARK_CARD)
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        # Jersey body
        c.setFillColor(self.swatch_color)
        m = 5
        body_h = self.height * 0.58
        c.rect(m, m * 2, self.width - m * 2, body_h, fill=1, stroke=0)
        # Collar
        nw = (self.width - m * 2) * 0.3
        nx = (self.width - nw) / 2
        c.rect(nx, m * 2 + body_h - mm * 2, nw, mm * 5, fill=1, stroke=0)
        # Jersey icon
        c.setFillColor(Color(0, 0, 0, alpha=0.35))
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(self.width / 2, self.height * 0.18, "#")


class RadarChart(Flowable):
    """
    Hexagonal radar chart drawn directly on canvas.
    labels: list of 6 strings
    values: list of 6 floats in [0, 1]
    """
    def __init__(self, labels, values, width=140 * mm, height=100 * mm):
        Flowable.__init__(self)
        self.labels = labels
        self.values = [max(0.0, min(1.0, v)) for v in values]
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        cx = self.width / 2
        cy = self.height / 2
        r_max = min(cx, cy) * 0.72
        n = len(self.labels)

        def pt(i, radius):
            angle = math.pi / 2 + 2 * math.pi * i / n
            return cx + radius * math.cos(angle), cy + radius * math.sin(angle)

        # Background grid rings
        for level in (0.25, 0.5, 0.75, 1.0):
            ring_pts = [pt(i, r_max * level) for i in range(n)]
            c.setStrokeColor(HexColor("#1E3050"))
            c.setLineWidth(0.4 if level < 1.0 else 0.8)
            path = c.beginPath()
            path.moveTo(*ring_pts[0])
            for px, py in ring_pts[1:]:
                path.lineTo(px, py)
            path.close()
            c.drawPath(path, stroke=1, fill=0)

        # Spokes
        c.setLineWidth(0.4)
        c.setStrokeColor(HexColor("#1E3050"))
        for i in range(n):
            ex, ey = pt(i, r_max)
            c.line(cx, cy, ex, ey)

        # Data polygon — filled
        data_pts = [pt(i, r_max * v) for i, v in enumerate(self.values)]
        fill_col = Color(0.0, 0.9, 0.46, alpha=0.25)
        c.setFillColor(fill_col)
        c.setStrokeColor(GREEN)
        c.setLineWidth(1.5)
        path = c.beginPath()
        path.moveTo(*data_pts[0])
        for px, py in data_pts[1:]:
            path.lineTo(px, py)
        path.close()
        c.drawPath(path, stroke=1, fill=1)

        # Vertex dots
        c.setFillColor(GREEN)
        for px, py in data_pts:
            c.circle(px, py, 2.8, fill=1, stroke=0)

        # Labels
        c.setFont("Helvetica-Bold", 7)
        for i, label in enumerate(self.labels):
            lx, ly = pt(i, r_max + 10)
            c.setFillColor(GREY)
            # Slight value nudge for label positioning
            angle = math.pi / 2 + 2 * math.pi * i / n
            if math.cos(angle) < -0.1:
                c.drawRightString(lx, ly - 3, label)
            elif math.cos(angle) > 0.1:
                c.drawString(lx, ly - 3, label)
            else:
                c.drawCentredString(lx, ly - 3, label)

        # Centre value label
        c.setFillColor(GREY)
        c.setFont("Helvetica", 6)
        for level, label in ((0.5, "5"), (1.0, "10")):
            lx, ly = pt(0, r_max * level)
            c.drawCentredString(lx + 6, ly + 1, label)


class ZoneBar(Flowable):
    """Horizontal DEF / MID / ATK zone time bar."""
    def __init__(self, def_pct, mid_pct, atk_pct, width=175 * mm, height=10 * mm):
        Flowable.__init__(self)
        total = max(def_pct + mid_pct + atk_pct, 1)
        self.def_pct = def_pct / total
        self.mid_pct = mid_pct / total
        self.atk_pct = atk_pct / total
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        r = 3
        # Background rounded rect
        c.setFillColor(DARK_CARD)
        c.roundRect(0, 0, self.width, self.height, r, fill=1, stroke=0)

        segments = [
            (self.def_pct, HexColor("#4FC3F7"), f"DEF {self.def_pct*100:.0f}%"),
            (self.mid_pct, HexColor("#00E676"), f"MID {self.mid_pct*100:.0f}%"),
            (self.atk_pct, HexColor("#FFD600"), f"ATK {self.atk_pct*100:.0f}%"),
        ]
        x = 0
        for frac, col, lbl in segments:
            seg_w = self.width * frac
            if seg_w < 1:
                continue
            c.setFillColor(col)
            c.rect(x, 0, seg_w, self.height, fill=1, stroke=0)
            if seg_w > 18:
                c.setFillColor(Color(0, 0, 0, alpha=0.75))
                c.setFont("Helvetica-Bold", 6.5)
                c.drawCentredString(x + seg_w / 2, self.height * 0.28, lbl)
            x += seg_w

        # Border outline
        c.setStrokeColor(BORDER)
        c.setLineWidth(0.5)
        c.roundRect(0, 0, self.width, self.height, r, fill=0, stroke=1)


# ── Page background ───────────────────────────────────────────────────────────

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


# ── Main entry point ──────────────────────────────────────────────────────────

class HBarChart(Flowable):
    """Simple horizontal bar chart for ball involvement."""
    def __init__(self, bars, width=175 * mm, height=None):
        """
        bars: list of (label, value, max_value, color_hex, suffix)
        """
        Flowable.__init__(self)
        self.bars = bars
        self.width = width
        self.row_h = 9 * mm
        self.height = height or (len(bars) * self.row_h + 4 * mm)

    def draw(self):
        c = self.canv
        label_w = 45 * mm
        bar_area = self.width - label_w - 20 * mm
        y = self.height - self.row_h

        for label, value, max_val, color_hex, suffix in self.bars:
            # Label
            c.setFont("Helvetica", 7.5)
            c.setFillColor(GREY)
            c.drawString(0, y + self.row_h * 0.3, label)

            # Background track
            c.setFillColor(DARK_CARD)
            c.roundRect(label_w, y + 1, bar_area, self.row_h - 3, 2, fill=1, stroke=0)

            # Filled bar
            fill = min(max(value / max(max_val, 0.001), 0.0), 1.0)
            bar_w = bar_area * fill
            if bar_w > 2:
                c.setFillColor(HexColor(color_hex))
                c.roundRect(label_w, y + 1, bar_w, self.row_h - 3, 2, fill=1, stroke=0)

            # Value label
            c.setFont("Helvetica-Bold", 7.5)
            c.setFillColor(WHITE)
            c.drawRightString(self.width, y + self.row_h * 0.3,
                              f"{value:.1f}{suffix}" if isinstance(value, float) else f"{value}{suffix}")

            y -= self.row_h


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

    # ── Extract player fields ────────────────────────────────────────────────
    pid          = player.get("id", "?")
    team         = player.get("team", "Unknown")
    team_color   = player.get("teamColor", "#FFFFFF")
    position     = player.get("position", "CM")
    total_dist   = player.get("totalDist", 0.0)
    sprint_dist  = player.get("sprintDist", 0.0)
    top_speed    = player.get("topSpeed", 0.0)
    sprints      = player.get("sprints", 0)
    rating       = player.get("rating", 5.0)
    fatigue      = player.get("fatigue", "Normal")
    walk_km      = player.get("walk", 0.0)
    jog_km       = player.get("jog", 0.0)
    hirun_km     = player.get("hiRun", 0.0)
    avg_x        = player.get("avgX", 52.5)
    avg_y        = player.get("avgY", 34.0)
    jersey       = player.get("jerseyColor", [128, 128, 128])
    duration     = match_results.get("duration", "90:00")
    date         = match_results.get("date", "")

    # Auto-analytics
    avg_speed       = player.get("avgSpeed", 0.0)
    hi_km           = player.get("highIntensityKm", 0.0)
    activity_rate   = player.get("activityRate", 0.0)
    atk_pct         = player.get("attackPct", 0.0)
    mid_pct         = player.get("midPct", 0.0)
    def_pct         = player.get("defPct", 0.0)
    coverage_pct    = player.get("coveragePct", 0.0)
    avg_sprint_m    = player.get("avgSprintDistM", 0.0)
    pressing        = player.get("pressingEvents", 0)
    pos_consistency = player.get("posConsistency", 0.0)
    work_rate       = player.get("workRateIndex", 0.0)

    # Match rating
    match_rating  = player.get("matchRating", rating)
    rating_grade  = player.get("ratingGrade", "")
    rating_bd     = player.get("ratingBreakdown", {})

    # Roster fields
    player_name   = player.get("name", f"Pemain #{pid}")
    player_number = str(player.get("number", pid))
    photo_path    = player.get("photo_path", None)

    jersey_hex = (
        f"#{int(jersey[0]):02X}{int(jersey[1]):02X}{int(jersey[2]):02X}"
        if isinstance(jersey, (list, tuple)) and len(jersey) >= 3
        else "#808080"
    )

    # Ball event stats
    ev_goals         = player.get("goals", 0)
    ev_assists       = player.get("assists", 0)
    ev_xg            = player.get("xG", 0.0)
    ev_shots         = player.get("shots", 0)
    ev_shots_on_tgt  = player.get("shotsOnTarget", 0)
    ev_passes        = player.get("passes", 0)
    ev_pass_acc      = player.get("passAccuracy", 0.0)
    ev_tackles       = player.get("tackles", 0)
    ev_interceptions = player.get("interceptions", 0)
    ev_touches       = player.get("touches", 0)
    is_motm          = player.get("manOfTheMatch", False)

    story += _page1(
        pid, team, team_color, position, total_dist, sprint_dist,
        top_speed, sprints, fatigue, avg_x, avg_y,
        jersey_hex, duration, date,
        player_name=player_name, player_number=player_number, photo_path=photo_path,
        activity_rate=activity_rate, work_rate=work_rate, coverage_pct=coverage_pct,
        def_pct=def_pct, mid_pct=mid_pct, atk_pct=atk_pct,
        match_rating=match_rating, rating_grade=rating_grade,
        ev_goals=ev_goals, ev_assists=ev_assists, ev_xg=ev_xg, ev_shots=ev_shots,
        ev_shots_on_tgt=ev_shots_on_tgt, ev_passes=ev_passes,
        ev_pass_acc=ev_pass_acc, ev_tackles=ev_tackles,
        ev_interceptions=ev_interceptions, ev_touches=ev_touches,
        is_motm=is_motm,
    )
    story.append(PageBreak())

    story += _page2(
        pid, total_dist, walk_km, jog_km, hirun_km, sprint_dist,
        top_speed, sprints, position, avg_x, avg_y,
        player_name=player_name, player_number=player_number,
        avg_speed=avg_speed, hi_km=hi_km, activity_rate=activity_rate,
        avg_sprint_m=avg_sprint_m, pressing=pressing,
        pos_consistency=pos_consistency, work_rate=work_rate,
        coverage_pct=coverage_pct, rating_bd=rating_bd,
        match_rating=match_rating,
        ev_passes=ev_passes, ev_pass_acc=ev_pass_acc,
        ev_shots=ev_shots, ev_shots_on_tgt=ev_shots_on_tgt,
        ev_goals=ev_goals, ev_tackles=ev_tackles,
    )

    doc.build(story, onFirstPage=_page_bg, onLaterPages=_page_bg)


# ── PAGE 1 ────────────────────────────────────────────────────────────────────

def _page1(pid, team, team_color, position, total_dist, sprint_dist,
           top_speed, sprints, fatigue, avg_x, avg_y,
           jersey_hex, duration, date,
           player_name="", player_number="", photo_path=None,
           activity_rate=0.0, work_rate=0.0, coverage_pct=0.0,
           def_pct=0.0, mid_pct=0.0, atk_pct=0.0,
           match_rating=5.0, rating_grade="",
           ev_goals=0, ev_assists=0, ev_xg=0.0, ev_shots=0, ev_shots_on_tgt=0,
           ev_passes=0, ev_pass_acc=0.0, ev_tackles=0,
           ev_interceptions=0, ev_touches=0, is_motm=False):
    elems = []
    elems.append(Spacer(1, 3 * mm))

    # ── HEADER: [Photo/Swatch] | [Name/Number/Team/Pos] | [Match Rating] ────
    photo_w = 40 * mm
    photo_h = 52 * mm

    # Photo / jersey swatch
    if photo_path and os.path.exists(photo_path):
        try:
            media = Image(photo_path, width=photo_w, height=photo_h)
        except Exception:
            media = JerseySwatch(jersey_hex, width=photo_w, height=photo_h)
    else:
        media = JerseySwatch(jersey_hex, width=photo_w, height=photo_h)

    # Match rating colour
    mr_color = (
        "#00E676" if match_rating >= 7.0
        else "#FFD600" if match_rating >= 6.0
        else "#EF5350"
    )

    # Left info column
    info_paras = [
        Paragraph(
            f'<font color="{team_color}"><b>{player_name}</b></font>',
            _style("pn", fontSize=17, fontName="Helvetica-Bold", leading=21)
        ),
        Spacer(1, 1.5 * mm),
        Paragraph(
            f'<font color="#B0BEC5">No. </font>'
            f'<font color="#FFD600"><b>{player_number}</b></font>'
            f'  <font color="#B0BEC5">Pos: </font>'
            f'<font color="#18FFFF"><b>{position}</b></font>',
            _style("pnp", fontSize=11)
        ),
        Spacer(1, 1 * mm),
        Paragraph(
            f'<font color="#B0BEC5">Tim: </font>'
            f'<font color="{team_color}"><b>{team}</b></font>',
            _style("ptm", fontSize=10)
        ),
        Spacer(1, 1 * mm),
        Paragraph(
            f'<font color="#B0BEC5">Fatigue: </font>'
            f'<font color="{"#EF5350" if fatigue=="Tinggi" else "#00E676"}"><b>{fatigue}</b></font>',
            _style("pft", fontSize=10)
        ),
    ]

    # Right — BIG match rating
    rating_cell = [
        Paragraph(
            f'<font color="{mr_color}"><b>{match_rating:.1f}</b></font>',
            _style("mr", fontSize=42, fontName="Helvetica-Bold", leading=46, alignment=TA_CENTER)
        ),
        Paragraph(
            f'<font color="#B0BEC5">MATCH RATING</font>',
            _style("mrl", fontSize=7, alignment=TA_CENTER, textColor=GREY)
        ),
        Paragraph(
            f'<font color="{mr_color}"><b>{rating_grade}</b></font>',
            _style("mrg", fontSize=9, fontName="Helvetica-Bold", alignment=TA_CENTER)
        ),
    ]

    avail_w = PAGE_W - 30 * mm  # content width
    info_w  = avail_w - photo_w - 6 * mm - 38 * mm
    rating_w = 38 * mm

    hdr = Table(
        [[media, info_paras, rating_cell]],
        colWidths=[photo_w + 4 * mm, info_w, rating_w]
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), MID_CARD),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (0, -1),  10),
        ("LEFTPADDING",   (1, 0), (1, -1),  12),
        ("RIGHTPADDING",  (-1, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LINEAFTER",     (0, 0), (0, -1),  1, BORDER),
        ("LINEAFTER",     (1, 0), (1, -1),  1, BORDER),
    ]))
    elems.append(hdr)

    # ── MAN OF THE MATCH BADGE ────────────────────────────────────────────────
    if is_motm:
        motm_banner = Table([[
            Paragraph(
                f'<font color="#080C12"><b>🏆 MAN OF THE MATCH — {player_name} — Rating {match_rating:.1f} {rating_grade}</b></font>',
                _style("motm_b", fontSize=11, fontName="Helvetica-Bold",
                       alignment=TA_CENTER, textColor=HexColor("#080C12"))
            )
        ]], colWidths=[PAGE_W - 30 * mm])
        motm_banner.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#FFD700")),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elems.append(Spacer(1, 2 * mm))
        elems.append(motm_banner)

    elems.append(Spacer(1, 4 * mm))

    # ── KPI ROW 1 ────────────────────────────────────────────────────────────
    kpi1 = [
        _kpi_cell("Total Distance", f"{total_dist:.2f} km", GREEN, "🏃"),
        _kpi_cell("Top Speed",      f"{top_speed:.1f} km/h", CYAN,  "⚡"),
        _kpi_cell("Sprint Distance", f"{sprint_dist:.2f} km", AMBER, "💨"),
    ]
    elems.append(_kpi_row(kpi1))
    elems.append(Spacer(1, 3 * mm))

    # ── KPI ROW 2 ────────────────────────────────────────────────────────────
    kpi2 = [
        _kpi_cell("Sprint Count",   str(sprints),   ORANGE, "🔥"),
        _kpi_cell("Match Rating",   f"{match_rating:.1f} / 10", _hex_color(mr_color), "⭐"),
        _kpi_cell("Fatigue Level",  fatigue, RED if fatigue == "Tinggi" else GREEN, "💪"),
    ]
    elems.append(_kpi_row(kpi2))
    elems.append(Spacer(1, 3 * mm))

    # ── KPI ROW 3 ────────────────────────────────────────────────────────────
    kpi3 = [
        _kpi_cell("Activity Rate",    f"{activity_rate:.1f}%",    CYAN,  "📊"),
        _kpi_cell("Work Rate Index",  f"{work_rate:.1f} / 10",    GREEN, "⚙️"),
        _kpi_cell("Field Coverage",   f"{coverage_pct:.1f}%",     AMBER, "🗺️"),
    ]
    elems.append(_kpi_row(kpi3))
    elems.append(Spacer(1, 5 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 4 * mm))

    # ── MATCH EVENTS ─────────────────────────────────────────────────────────
    elems.append(Paragraph(
        '<font color="#FFD600">MATCH EVENTS</font>',
        _style("me_sec", fontSize=12, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 3 * mm))

    def _ev_cell(label, value, color):
        return Paragraph(
            f'<font color="#B0BEC5">{label}: </font>'
            f'<font color="{color}"><b>{value}</b></font>',
            _style(f"ev_{label}", fontSize=9)
        )

    events_data = [
        [_ev_cell("Goals",        str(ev_goals),                 "#FFD600"),
         _ev_cell("Assists",      str(ev_assists),               "#FFD600")],
        [_ev_cell("xG",           f"{ev_xg:.2f}",                "#FF6D00"),
         _ev_cell("Shots",        str(ev_shots),                 "#FF6D00")],
        [_ev_cell("On Target",    str(ev_shots_on_tgt),          "#FF6D00"),
         _ev_cell("Passes",       str(ev_passes),                "#18FFFF")],
        [_ev_cell("Pass Acc",     f"{ev_pass_acc:.1f}%",         "#18FFFF"),
         _ev_cell("Tackles",      str(ev_tackles),               "#00E676")],
        [_ev_cell("Interceptions",str(ev_interceptions),         "#00E676"),
         _ev_cell("Ball Touches", str(ev_touches),               "#B0BEC5")],
    ]
    ev_table = Table(events_data, colWidths=[87 * mm, 87 * mm])
    ev_table.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [DARK_CARD, MID_CARD]),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("GRID",           (0, 0), (-1, -1), 0.3, BORDER),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elems.append(ev_table)
    elems.append(Spacer(1, 4 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 4 * mm))

    # ── FIELD POSITION ZONE ──────────────────────────────────────────────────
    elems.append(Paragraph(
        '<font color="#FFD600">FIELD POSITION ZONE</font>',
        _style("sec", fontSize=12, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 3 * mm))

    zone_info = [[
        Paragraph(
            f'<font color="#B0BEC5">Avg X:</font> '
            f'<font color="#18FFFF"><b>{avg_x:.1f}m</b></font>  '
            f'<font color="#B0BEC5">Avg Y:</font> '
            f'<font color="#18FFFF"><b>{avg_y:.1f}m</b></font>',
            _style("zi", fontSize=10)
        ),
        Paragraph(
            f'<font color="#B0BEC5">Zone: </font>'
            f'<font color="#00E676"><b>{_zone_label(avg_x, avg_y)}</b></font>',
            _style("zi2", fontSize=10, alignment=TA_RIGHT)
        ),
    ]]
    zt = Table(zone_info, colWidths=[90 * mm, 90 * mm])
    zt.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elems.append(zt)
    elems.append(Spacer(1, 3 * mm))

    # Zone time bar
    elems.append(ZoneBar(def_pct, mid_pct, atk_pct, width=175 * mm, height=10 * mm))
    elems.append(Spacer(1, 5 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 4 * mm))

    # ── MATCH CONTEXT ────────────────────────────────────────────────────────
    elems.append(Paragraph(
        '<font color="#FFD600">MATCH CONTEXT</font>',
        _style("sec", fontSize=12, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 3 * mm))
    ctx_data = [[
        Paragraph(f'<font color="#B0BEC5">Duration</font><br/><font color="#FFFFFF"><b>{duration}</b></font>',
                  _style("c1", fontSize=10, alignment=TA_CENTER)),
        Paragraph(f'<font color="#B0BEC5">Date</font><br/><font color="#FFFFFF"><b>{date}</b></font>',
                  _style("c2", fontSize=10, alignment=TA_CENTER)),
        Paragraph(f'<font color="#B0BEC5">Position</font><br/><font color="#18FFFF"><b>{position}</b></font>',
                  _style("c3", fontSize=10, alignment=TA_CENTER)),
        Paragraph(f'<font color="#B0BEC5">Jersey No.</font><br/>'
                  f'<font color="#FFD600"><b>{player_number}</b></font>',
                  _style("c4", fontSize=10, alignment=TA_CENTER)),
    ]]
    ct = Table(ctx_data, colWidths=[44 * mm, 44 * mm, 44 * mm, 44 * mm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), DARK_CARD),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    elems.append(ct)
    return elems


# ── PAGE 2 ────────────────────────────────────────────────────────────────────

def _page2(pid, total_dist, walk_km, jog_km, hirun_km, sprint_dist,
           top_speed, sprints, position, avg_x, avg_y,
           player_name="", player_number="",
           avg_speed=0.0, hi_km=0.0, activity_rate=0.0,
           avg_sprint_m=0.0, pressing=0, pos_consistency=0.0,
           work_rate=0.0, coverage_pct=0.0,
           rating_bd=None, match_rating=5.0,
           ev_passes=0, ev_pass_acc=0.0, ev_shots=0, ev_shots_on_tgt=0,
           ev_goals=0, ev_tackles=0):
    rating_bd = rating_bd or {}
    elems = []
    elems.append(Spacer(1, 3 * mm))

    display_name = player_name if player_name and player_name != f"Pemain #{pid}" else f"Pemain #{pid}"
    elems.append(Paragraph(
        f'<font color="#00E676">{display_name}</font>'
        f' <font color="#FFFFFF">— Speed Zones & Analytics</font>',
        _style("ph2", fontSize=16, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 2 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 4 * mm))

    # ── SPEED ZONE BREAKDOWN ─────────────────────────────────────────────────
    elems.append(Paragraph(
        '<font color="#FFD600">SPEED ZONE BREAKDOWN</font>',
        _style("s1", fontSize=12, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 3 * mm))

    total_safe = max(total_dist, 0.001)
    zones = [
        ("Walk",   walk_km,    "#4FC3F7", "< 8 km/h"),
        ("Jog",    jog_km,     "#00E676", "8–16 km/h"),
        ("Hi-Run", hirun_km,   "#FFD600", "16–21 km/h"),
        ("Sprint", sprint_dist, "#FF5252", "> 21 km/h"),
    ]
    for zone_name, dist_km, color_hex, threshold in zones:
        pct = dist_km / total_safe * 100
        label_row = [[
            Paragraph(f'<font color="#B0BEC5">{zone_name}</font>',
                      _style("zn", fontSize=9)),
            Paragraph(f'<font color="#B0BEC5">{threshold}</font>',
                      _style("zt", fontSize=8, alignment=TA_CENTER)),
            Paragraph(f'<font color="{color_hex}"><b>{dist_km:.2f} km</b></font>',
                      _style("zv", fontSize=9, alignment=TA_RIGHT)),
            Paragraph(f'<font color="{color_hex}">{pct:.1f}%</font>',
                      _style("zp", fontSize=9, alignment=TA_RIGHT)),
        ]]
        bt = Table(label_row, colWidths=[25 * mm, 35 * mm, 30 * mm, 20 * mm])
        bt.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elems.append(bt)
        fill = min(pct / 100.0, 1.0)
        bar = Table([[Paragraph(
            f'<font size="1" color="{color_hex}">{"█" * int(fill * 58)}{"░" * (58 - int(fill * 58))}</font>',
            _style("bar", fontSize=8, fontName="Courier", textColor=HexColor(color_hex))
        )]], colWidths=[175 * mm])
        bar.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        elems.append(bar)

    elems.append(Spacer(1, 4 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 4 * mm))

    # ── TWO-COLUMN: Movement Summary  |  Rating Breakdown ────────────────────
    # Left: extended movement summary
    summary_rows = [
        [_lbl("Total Distance"),   _val(f"{total_dist:.2f} km"),  _lbl("Avg Speed"),       _val(f"{avg_speed:.1f} km/h")],
        [_lbl("Walk"),             _val(f"{walk_km:.2f} km"),     _lbl("Hi-Intensity Km"), _val(f"{hi_km:.2f} km")],
        [_lbl("Jog"),              _val(f"{jog_km:.2f} km"),      _lbl("Activity Rate"),   _val(f"{activity_rate:.1f}%")],
        [_lbl("Hi-Run"),           _val(f"{hirun_km:.2f} km"),    _lbl("Avg Sprint Dist"), _val(f"{avg_sprint_m:.1f} m")],
        [_lbl("Sprint"),           _val(f"{sprint_dist:.2f} km"), _lbl("Pressing Events"), _val(str(pressing))],
        [_lbl("Top Speed"),        _val(f"{top_speed:.1f} km/h"), _lbl("Pos. Consistency"),_val(f"{pos_consistency:.1f}")],
        [_lbl("Sprint Count"),     _val(str(sprints)),            _lbl("Work Rate Index"), _val(f"{work_rate:.1f}/10")],
        [_lbl("Field Coverage"),   _val(f"{coverage_pct:.1f}%"),  _lbl("Position"),        _val(position)],
    ]
    st = Table(summary_rows, colWidths=[38 * mm, 32 * mm, 42 * mm, 30 * mm])
    st.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [DARK_CARD, MID_CARD]),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("GRID",           (0, 0), (-1, -1), 0.3, BORDER),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))

    # Right: Rating breakdown table with mini bars
    bd_labels = [
        ("Distance",    rating_bd.get("distance",    0.0), "#00E676"),
        ("Sprint",      rating_bd.get("sprint",      0.0), "#FF6D00"),
        ("Speed",       rating_bd.get("speed",       0.0), "#18FFFF"),
        ("Intensity",   rating_bd.get("intensity",   0.0), "#FFD600"),
        ("Activity",    rating_bd.get("activity",    0.0), "#00E676"),
        ("Pressing",    rating_bd.get("pressing",    0.0), "#FF5252"),
        ("Consistency", rating_bd.get("consistency", 0.0), "#B0BEC5"),
    ]
    mr_color = (
        "#00E676" if match_rating >= 7.0
        else "#FFD600" if match_rating >= 6.0
        else "#EF5350"
    )
    bd_header = [[
        Paragraph(
            f'<font color="#FFD600">RATING BREAKDOWN</font>  '
            f'<font color="{mr_color}"><b>{match_rating:.1f}</b></font>',
            _style("rdhdr", fontSize=10, fontName="Helvetica-Bold")
        )
    ]]
    bd_data = [bd_header[0]]
    for label, score, col_hex in bd_labels:
        bar_filled = int(score / 10 * 20)
        bar_empty  = 20 - bar_filled
        bar_str    = "█" * bar_filled + "░" * bar_empty
        bd_data.append([
            Paragraph(
                f'<font color="#B0BEC5" size="8">{label:<11}</font>'
                f'<font color="{col_hex}" size="8" face="Courier"> {bar_str} </font>'
                f'<font color="{col_hex}" size="8"><b>{score:.1f}</b></font>',
                _style(f"bd_{label}", fontSize=8, fontName="Helvetica", leading=11)
            )
        ])

    bd_t = Table(bd_data, colWidths=[62 * mm])
    bd_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  MID_CARD),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [DARK_CARD, MID_CARD]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    two_col = Table([[st, Spacer(4 * mm, 1), bd_t]], colWidths=[142 * mm, 4 * mm, 64 * mm])
    two_col.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    elems.append(two_col)
    elems.append(Spacer(1, 5 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 4 * mm))

    # ── BALL INVOLVEMENT ─────────────────────────────────────────────────────
    elems.append(Paragraph(
        '<font color="#FFD600">BALL INVOLVEMENT</font>',
        _style("bi_sec", fontSize=12, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 3 * mm))

    # Contribution index = (passes*0.5 + tackles*1.5 + goals*5 + shots*1) normalized 0-10
    contrib_raw = ev_passes * 0.5 + ev_tackles * 1.5 + ev_goals * 5 + ev_shots * 1.0
    contrib_idx = round(min(10.0, contrib_raw / 5.0), 1)  # normalize: 50 raw = 10

    bi_bars = [
        ("Pass Accuracy",   ev_pass_acc,         100.0, "#18FFFF", "%"),
        ("Shots",           float(ev_shots),       10.0, "#FF6D00", ""),
        ("Shots on Target", float(ev_shots_on_tgt), 10.0, "#FFD600", ""),
        ("Contribution Idx",contrib_idx,           10.0, "#00E676", "/10"),
    ]
    elems.append(HBarChart(bi_bars, width=175 * mm))
    elems.append(Spacer(1, 4 * mm))
    elems.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    elems.append(Spacer(1, 4 * mm))

    # ── RADAR CHART ──────────────────────────────────────────────────────────
    elems.append(Paragraph(
        '<font color="#FFD600">PERFORMANCE RADAR</font>',
        _style("rdr", fontSize=12, fontName="Helvetica-Bold")
    ))
    elems.append(Spacer(1, 3 * mm))

    radar_labels = ["Speed", "Stamina", "Intensity", "Sprint\nPower", "Coverage", "Activity"]
    radar_values = [
        min(1.0, top_speed    / 40.0),
        min(1.0, total_dist   / 12.0),
        min(1.0, activity_rate / 100.0),
        min(1.0, avg_sprint_m / 40.0),
        min(1.0, coverage_pct / 25.0),
        min(1.0, work_rate    / 10.0),
    ]
    radar = RadarChart(radar_labels, radar_values, width=155 * mm, height=88 * mm)
    elems.append(radar)

    return elems


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kpi_cell(label, value, color, icon=""):
    color_str = color.hexval() if hasattr(color, "hexval") else str(color)
    inner = [
        [Paragraph(f'{icon} <font color="#B0BEC5">{label}</font>', _style("kl", fontSize=8))],
        [Paragraph(f'<font color="{color_str}"><b>{value}</b></font>',
                   _style("kv", fontSize=14, fontName="Helvetica-Bold"))],
    ]
    t = Table(inner, colWidths=[52 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), DARK_CARD),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("LINEABOVE",   (0, 0), (-1, 0),  2, color),
    ]))
    return t


def _kpi_row(cells):
    t = Table([cells], colWidths=[58 * mm, 58 * mm, 58 * mm])
    t.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 2),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _lbl(text):
    return Paragraph(f'<font color="#B0BEC5">{text}</font>', _style("l", fontSize=9))


def _val(text):
    return Paragraph(f'<font color="#FFFFFF"><b>{text}</b></font>', _style("v", fontSize=9))


def _hex_color(hex_str: str) -> HexColor:
    try:
        return HexColor(hex_str)
    except Exception:
        return WHITE


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
