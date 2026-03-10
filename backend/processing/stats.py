"""
stats.py — Player statistics calculation from tracking data
"""
import math
import numpy as np


# Speed thresholds in km/h
WALK_MAX = 8.0
JOG_MAX = 16.0
HIRUN_MAX = 21.0
SPRINT_MIN = 21.0

# Field dimensions (meters)
FIELD_WIDTH = 105.0
FIELD_HEIGHT = 68.0


def infer_position(avg_x: float, avg_y: float, team: str = "") -> str:
    """Infer player position from average field coordinates."""
    x, y = avg_x, avg_y

    if x < 10 or x > 95:
        return "GK"
    if (x < 25 or x > 80) and 20 < y < 48:
        return "CB"
    if (x < 30 or x > 75) and y <= 25:
        return "LB"
    if (x < 30 or x > 75) and y >= 43:
        return "RB"
    if (30 < x < 45 or 60 < x < 75) and not (20 < y < 48):
        return "CDM"
    if 30 < x < 45 or 60 < x < 75:
        return "CDM"
    if 40 < x < 65 and 20 < y < 48:
        return "CM"
    if x > 60 and y < 20:
        return "LW"
    if x > 60 and y > 50:
        return "RW"
    if x > 70:
        return "ST"
    if x < 35:
        return "ST"
    return "CM"


def pixel_to_field(px: float, py: float, frame_width: int, frame_height: int) -> tuple[float, float]:
    """Convert pixel coordinates to field coordinates (meters)."""
    fx = (px / frame_width) * FIELD_WIDTH
    fy = (py / frame_height) * FIELD_HEIGHT
    return fx, fy


def calculate_speed_kmh(pos1: tuple, pos2: tuple, dt_seconds: float) -> float:
    """Calculate speed in km/h between two field positions."""
    if dt_seconds <= 0:
        return 0.0
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dist_m = math.sqrt(dx * dx + dy * dy)
    speed_ms = dist_m / dt_seconds
    return speed_ms * 3.6  # m/s to km/h


def calculate_player_stats(
    track_id: int,
    positions: list[tuple],  # list of (frame_idx, field_x, field_y)
    video_fps: float,
    frame_sample_rate: int = 5
) -> dict:
    """
    Calculate complete player statistics from position timeline.

    positions: list of (frame_idx, field_x, field_y)
    Returns stat dict including auto-computed analytics.
    """
    if len(positions) < 2:
        return _empty_stats(track_id)

    # Sort by frame index
    positions = sorted(positions, key=lambda p: p[0])

    total_dist = 0.0
    walk_dist = 0.0
    jog_dist = 0.0
    hirun_dist = 0.0
    sprint_dist = 0.0
    top_speed = 0.0
    speeds = []

    in_sprint = False
    sprint_count = 0

    for i in range(1, len(positions)):
        _, x1, y1 = positions[i - 1]
        _, x2, y2 = positions[i]
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx * dx + dy * dy)

        # Sanity check — ignore teleports > 50m/step
        if dist > 50:
            in_sprint = False
            continue

        frame_gap = positions[i][0] - positions[i - 1][0]
        actual_dt = frame_gap / video_fps
        speed = (dist / actual_dt) * 3.6 if actual_dt > 0 else 0.0

        speeds.append(speed)
        total_dist += dist

        if speed >= SPRINT_MIN:
            sprint_dist += dist
            if not in_sprint:
                sprint_count += 1
                in_sprint = True
        else:
            in_sprint = False
            if speed >= HIRUN_MAX:
                hirun_dist += dist
            elif speed >= WALK_MAX:
                jog_dist += dist
            else:
                walk_dist += dist

        if speed > top_speed:
            top_speed = speed

    top_speed = min(top_speed, 45.0)  # cap unrealistic values

    # Positions for avg / std / coverage
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    avg_x = float(np.mean(xs))
    avg_y = float(np.mean(ys))

    # Distance in km
    total_km = total_dist / 1000.0
    sprint_km = sprint_dist / 1000.0
    hirun_km = hirun_dist / 1000.0
    jog_km = jog_dist / 1000.0
    walk_km = walk_dist / 1000.0

    # ── Rating & fatigue ────────────────────────────────────────────────────
    rating = min(10.0, max(5.0, total_km * 0.45 + sprint_count * 0.003 + top_speed * 0.08))
    rating = round(rating, 1)

    intense = hirun_km + sprint_km
    fatigue = "Tinggi" if total_km > 0 and (intense / (total_km + 1e-9)) > 0.40 else "Normal"

    # ── Auto-computed analytics ──────────────────────────────────────────────
    # Tracked duration from first to last frame
    duration_sec = max((positions[-1][0] - positions[0][0]) / video_fps, 1.0)

    # Average speed (km/h) = total_m / duration_sec * 3.6
    avg_speed = round((total_km * 1000 / max(duration_sec, 1)) * 3.6, 1)

    # High-intensity distance (hi-run + sprint)
    high_intensity_km = round(hirun_km + sprint_km, 2)

    # Activity rate: fraction of distance at jog or above
    activity_rate = round(
        (jog_km + hirun_km + sprint_km) / max(total_km, 0.001) * 100, 1
    )

    # Zone time percentages (by position count)
    n_pos = len(positions)
    def_count = sum(1 for p in positions if p[1] < 35)
    mid_count = sum(1 for p in positions if 35 <= p[1] <= 70)
    atk_count = sum(1 for p in positions if p[1] > 70)
    def_pct = round(def_count / n_pos * 100, 1)
    mid_pct = round(mid_count / n_pos * 100, 1)
    atk_pct = round(atk_count / n_pos * 100, 1)

    # Coverage: bounding box area / total field area
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_area = (max_x - min_x) * (max_y - min_y)
    coverage_pct = round(bbox_area / (105 * 68) * 100, 1)

    # Average sprint distance in meters
    avg_sprint_dist_m = round(sprint_km * 1000 / max(sprint_count, 1), 1)

    # Pressing events: frames at high-intensity run speed (>= 16 km/h)
    pressing_events = sum(1 for s in speeds if s >= 16.0)

    # Positional consistency: lower std = more consistent positioning
    pos_consistency = round(max(0.0, 100.0 - float(np.std(xs)) - float(np.std(ys))), 1)

    # Work rate index: composite of total km + high-intensity km
    work_rate_index = round(min(10.0, total_km * 0.8 + high_intensity_km * 2.0), 1)

    return {
        "trackId": track_id,
        "totalDist": round(total_km, 2),
        "sprintDist": round(sprint_km, 2),
        "hiRun": round(hirun_km, 2),
        "jog": round(jog_km, 2),
        "walk": round(walk_km, 2),
        "topSpeed": round(top_speed, 1),
        "sprints": sprint_count,
        "avgX": round(avg_x, 1),
        "avgY": round(avg_y, 1),
        "rating": rating,
        "fatigue": fatigue,
        # ── Auto analytics ──
        "avgSpeed": avg_speed,
        "highIntensityKm": high_intensity_km,
        "activityRate": activity_rate,
        "attackPct": atk_pct,
        "midPct": mid_pct,
        "defPct": def_pct,
        "coveragePct": coverage_pct,
        "avgSprintDistM": avg_sprint_dist_m,
        "pressingEvents": pressing_events,
        "posConsistency": pos_consistency,
        "workRateIndex": work_rate_index,
    }


def _empty_stats(track_id: int) -> dict:
    return {
        "trackId": track_id,
        "totalDist": 0.0,
        "sprintDist": 0.0,
        "hiRun": 0.0,
        "jog": 0.0,
        "walk": 0.0,
        "topSpeed": 0.0,
        "sprints": 0,
        "avgX": 52.5,
        "avgY": 34.0,
        "rating": 5.0,
        "fatigue": "Normal",
        # ── Auto analytics defaults ──
        "avgSpeed": 0.0,
        "highIntensityKm": 0.0,
        "activityRate": 0.0,
        "attackPct": 0.0,
        "midPct": 0.0,
        "defPct": 0.0,
        "coveragePct": 0.0,
        "avgSprintDistM": 0.0,
        "pressingEvents": 0,
        "posConsistency": 0.0,
        "workRateIndex": 0.0,
    }


def calculate_match_rating(stats: dict, position: str) -> dict:
    """
    Calculate a position-weighted match performance rating (3.0–10.0 scale).
    Returns dict with matchRating, ratingGrade, and a 7-component breakdown.
    """
    total_km  = stats.get("totalDist", 0)
    sprint_count = stats.get("sprints", 0)
    top_speed = stats.get("topSpeed", 0)
    hi_km     = stats.get("highIntensityKm", 0)
    activity  = stats.get("activityRate", 0)
    pressing  = stats.get("pressingEvents", 0)
    consistency = stats.get("posConsistency", 50)

    # Normalise each component to 0–10 (reference = avg pro 90-min match)
    d_s   = min(10, (total_km    / 10.5) * 7.5)
    sp_s  = min(10, (sprint_count / 40)  * 7.0)
    spd_s = min(10, (top_speed   / 32)   * 8.0)
    i_s   = min(10, (hi_km       / 2.5)  * 7.5)
    a_s   = min(10, (activity    / 65)   * 7.5)
    pr_s  = min(10, (pressing    / 80)   * 7.5)
    c_s   = min(10,  consistency / 10)

    # Position-specific weights [distance, sprints, speed, intensity, activity, pressing, consistency]
    pos = position.upper()
    if pos == "GK":
        w = [0.10, 0.08, 0.12, 0.08, 0.12, 0.05, 0.45]
    elif pos in ("CB", "RB", "LB"):
        w = [0.18, 0.12, 0.12, 0.13, 0.15, 0.15, 0.15]
    elif pos in ("CDM", "CM"):
        w = [0.22, 0.15, 0.10, 0.18, 0.18, 0.12, 0.05]
    elif pos in ("ST", "LW", "RW", "CAM"):
        w = [0.18, 0.22, 0.20, 0.20, 0.12, 0.05, 0.03]
    else:
        w = [0.20, 0.15, 0.13, 0.17, 0.15, 0.12, 0.08]

    base = sum(s * wt for s, wt in zip([d_s, sp_s, spd_s, i_s, a_s, pr_s, c_s], w))

    # Bonus / penalty modifiers
    bonus = 0.0
    if top_speed   >= 33:  bonus += 0.3
    if total_km    >= 12:  bonus += 0.4
    if sprint_count >= 60: bonus += 0.2
    if activity    < 40:   bonus -= 0.5
    if total_km    < 5:    bonus -= 0.8

    r = round(min(10.0, max(3.0, base + bonus)), 1)

    if   r >= 9: grade = "World Class 🌟"
    elif r >= 8: grade = "Excellent ⭐"
    elif r >= 7: grade = "Good 👍"
    elif r >= 6: grade = "Average ➡️"
    elif r >= 5: grade = "Below Average ⚠️"
    else:        grade = "Poor 🔴"

    return {
        "matchRating": r,
        "ratingGrade": grade,
        "ratingBreakdown": {
            "distance":    round(d_s,   1),
            "sprint":      round(sp_s,  1),
            "speed":       round(spd_s, 1),
            "intensity":   round(i_s,   1),
            "activity":    round(a_s,   1),
            "pressing":    round(pr_s,  1),
            "consistency": round(c_s,   1),
        },
    }


def build_team_stats(players: list[dict], team_name: str) -> dict:
    """Aggregate team-level stats from player list."""
    team = [p for p in players if p.get("team") == team_name]
    if not team:
        return {}

    total_dist = sum(p["totalDist"] for p in team)
    total_sprint = sum(p["sprintDist"] for p in team)
    top_speed = max(p["topSpeed"] for p in team)
    avg_dist = total_dist / len(team)
    avg_rating = sum(p["rating"] for p in team) / len(team)
    best = max(team, key=lambda p: p["rating"])

    return {
        "name": team_name,
        "players": len(team),
        "totalDist": round(total_dist, 2),
        "avgDist": round(avg_dist, 2),
        "totalSprint": round(total_sprint, 2),
        "topSpeed": round(top_speed, 1),
        "avgRating": round(avg_rating, 1),
        "bestPlayer": best["id"],
    }
