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
    Returns stat dict.
    """
    if len(positions) < 2:
        return _empty_stats(track_id)

    # Sort by frame index
    positions = sorted(positions, key=lambda p: p[0])

    # Time between sampled frames
    dt = frame_sample_rate / video_fps  # seconds per step

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

    # Positions for avg
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

    # Rating formula
    rating = min(10.0, max(5.0, total_km * 0.45 + sprint_count * 0.003 + top_speed * 0.08))
    rating = round(rating, 1)

    # Fatigue
    intense = hirun_km + sprint_km
    fatigue = "Tinggi" if total_km > 0 and (intense / (total_km + 1e-9)) > 0.40 else "Normal"

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
