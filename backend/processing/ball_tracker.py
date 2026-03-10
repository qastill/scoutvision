"""
ball_tracker.py — Ball detection and event detection for ScoutVision.
Uses YOLOv8n class 32 (sports ball) for detection.
"""
import numpy as np
import math
from collections import deque

# Field dimensions
FIELD_W, FIELD_H = 105.0, 68.0
GOAL_Y_MIN, GOAL_Y_MAX = 25.0, 43.0     # wider goal zone (was 30.34-37.66)
GOAL_ZONE_X_LEFT  = 3.0                  # ball x < 3 → left goal
GOAL_ZONE_X_RIGHT = 102.0                # ball x > 102 → right goal
GOAL_ZONE_DEPTH = 3.0                    # meters into field from goal line (legacy compat)

def pixel_to_field(px, py, frame_w, frame_h):
    return (px / frame_w) * FIELD_W, (py / frame_h) * FIELD_H


class BallTracker:
    def __init__(self, max_missing=15):
        self.pos = None          # (cx_px, cy_px)
        self.field_pos = None    # (fx, fy) in meters
        self.missing = 0
        self.max_missing = max_missing
        self.history = deque(maxlen=30)  # last 30 positions

    def update(self, detections, frame_w, frame_h):
        """detections: list of (cx, cy) pixel coords from YOLO ball class"""
        if detections:
            if self.pos is not None:
                # Pick closest to last known position
                best = min(detections, key=lambda d: abs(d[0]-self.pos[0])+abs(d[1]-self.pos[1]))
            else:
                best = detections[0]
            self.pos = best
            self.field_pos = pixel_to_field(best[0], best[1], frame_w, frame_h)
            self.missing = 0
        else:
            self.missing += 1
            if self.missing > self.max_missing:
                self.pos = None
                self.field_pos = None

        if self.field_pos:
            self.history.append(self.field_pos)
        return self.field_pos

    def get_velocity(self):
        """Returns (vx, vy) in m/frame based on last 5 frames."""
        if len(self.history) < 2:
            return (0.0, 0.0)
        recent = list(self.history)[-5:]
        if len(recent) < 2:
            return (0.0, 0.0)
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        n = len(recent) - 1
        return (dx/n, dy/n)


class EventDetector:
    """Detects passes, shots, goals, tackles from ball + player positions."""

    def __init__(self, fps, frame_w, frame_h):
        self.fps = fps
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Per-player event accumulators (player id 1..22)
        self.events = {i: {
            "passes": 0, "passesAttempted": 0,
            "shots": 0, "shotsOnTarget": 0,
            "goals": 0, "tackles": 0,
            "interceptions": 0, "touches": 0,
            "xG": 0.0,
        } for i in range(1, 23)}

        self.possession_id = None     # current possessing player id
        self.possession_team = None
        self.last_possession_id = None
        self.prev_possession_id = None   # two possessors ago — for assist tracking
        self.frames_in_possession = 0
        self.MIN_POSSESSION_FRAMES = 8  # must hold ball 8+ frames to count

        self.shot_cooldown = 0         # frames until next shot can be detected
        self.goal_cooldown = 0
        self.tackle_cooldown = {}      # player_id → cooldown

        self.POSSESSION_RADIUS_M = 3.0    # meters radius for possession
        self.SHOT_SPEED_THRESHOLD = 15.0  # m/s for shot detection
        self.TACKLE_RADIUS_M = 2.5

        # Goal events list — each entry: {type, player_id, assist_player_id, minute}
        self.goal_events: list[dict] = []

    def _find_possessor(self, ball_pos, players_current):
        """Find player closest to ball within possession radius."""
        if not ball_pos or not players_current:
            return None, None
        bx, by = ball_pos
        best_id, best_dist, best_team = None, float('inf'), None
        for pid, (px, py, team) in players_current.items():
            d = math.sqrt((px-bx)**2 + (py-by)**2)
            if d < self.POSSESSION_RADIUS_M and d < best_dist:
                best_dist = d
                best_id = pid
                best_team = team
        return (best_id, best_team) if best_id else (None, None)

    def _xg(self, shot_fx, shot_fy, attacking_right):
        """Simple xG model based on shot position."""
        # Goal center
        if attacking_right:
            goal_x, goal_y = FIELD_W, 34.0
        else:
            goal_x, goal_y = 0.0, 34.0
        distance = math.sqrt((shot_fx - goal_x)**2 + (shot_fy - goal_y)**2)
        # Angle to goal posts
        p1y, p2y = GOAL_Y_MIN, GOAL_Y_MAX
        try:
            angle_rad = abs(math.atan2(
                abs(shot_fx - goal_x) * (p2y - p1y),
                (shot_fx - goal_x)**2 + (shot_fy - p1y)*(shot_fy - p2y)
            ))
            angle_deg = math.degrees(angle_rad)
        except Exception:
            angle_deg = 15.0
        xg = 0.35 / (1 + distance * 0.07) * (min(angle_deg, 45) / 45) * 0.95
        return round(min(0.99, max(0.01, xg)), 3)

    def _is_goal_zone(self, fx, fy, attacking_right):
        """Check if ball is in goal zone (enhanced: x<3 or x>102, y 25-43m)."""
        in_y = GOAL_Y_MIN <= fy <= GOAL_Y_MAX
        if attacking_right:
            return fx > GOAL_ZONE_X_RIGHT and in_y
        else:
            return fx < GOAL_ZONE_X_LEFT and in_y

    def _is_any_goal(self, fx, fy):
        """Check if ball crossed either goal line."""
        in_y = GOAL_Y_MIN <= fy <= GOAL_Y_MAX
        return (fx < GOAL_ZONE_X_LEFT or fx > GOAL_ZONE_X_RIGHT) and in_y

    def update(self, frame_idx, ball_field_pos, players_current, ball_tracker):
        """
        Call every sampled frame.
        players_current: dict {player_id: (field_x, field_y, team_name)}
        """
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1
        if self.goal_cooldown > 0:
            self.goal_cooldown -= 1

        if not ball_field_pos:
            return

        bx, by = ball_field_pos
        possessor_id, possessor_team = self._find_possessor(ball_field_pos, players_current)

        # ── Ball touch
        if possessor_id:
            self.events[possessor_id]["touches"] += 1

        # ── Possession change detection
        if possessor_id != self.possession_id:
            if (self.possession_id and
                    self.frames_in_possession >= self.MIN_POSSESSION_FRAMES and
                    possessor_id):
                prev_team = self.possession_team
                new_team = possessor_team

                if prev_team == new_team:
                    # Same team → PASS
                    self.events[self.possession_id]["passes"] += 1
                    self.events[self.possession_id]["passesAttempted"] += 1
                else:
                    # Opponent team → TACKLE/INTERCEPTION
                    self.events[self.possession_id]["passesAttempted"] += 1  # failed pass
                    cdwn = self.tackle_cooldown.get(possessor_id, 0)
                    if cdwn <= 0:
                        # Check proximity for tackle vs interception
                        if (self.possession_id in players_current and
                                possessor_id in players_current):
                            px1, py1, _ = players_current[self.possession_id]
                            px2, py2, _ = players_current[possessor_id]
                            dist = math.sqrt((px1-px2)**2 + (py1-py2)**2)
                            if dist <= self.TACKLE_RADIUS_M:
                                self.events[possessor_id]["tackles"] += 1
                            else:
                                self.events[possessor_id]["interceptions"] += 1
                            self.tackle_cooldown[possessor_id] = int(self.fps * 3)

            self.prev_possession_id = self.last_possession_id
            self.possession_id = possessor_id
            self.possession_team = possessor_team
            self.frames_in_possession = 0
        else:
            self.frames_in_possession += 1

        # ── Shot detection
        vx, vy = ball_tracker.get_velocity()
        ball_speed_ms = math.sqrt(vx**2 + vy**2) * self.fps  # m/s
        if ball_speed_ms >= self.SHOT_SPEED_THRESHOLD and self.shot_cooldown <= 0:
            # Determine attacking direction: possessor's avg x > 52.5 → attacking right
            if self.last_possession_id and self.last_possession_id in players_current:
                shooter_id = self.last_possession_id
                sx, sy, _ = players_current[shooter_id]
                attacking_right = sx > 52.5
                # Ball moving toward goal
                toward_right_goal = vx > 0 and bx > 60 and attacking_right
                toward_left_goal  = vx < 0 and bx < 45 and not attacking_right
                if toward_right_goal or toward_left_goal:
                    xg_val = self._xg(bx, by, attacking_right)
                    self.events[shooter_id]["shots"] += 1
                    self.events[shooter_id]["xG"] = round(
                        self.events[shooter_id]["xG"] + xg_val, 3)
                    # On target: ball heading for goal y-range
                    if GOAL_Y_MIN <= by <= GOAL_Y_MAX:
                        self.events[shooter_id]["shotsOnTarget"] += 1
                    self.shot_cooldown = int(self.fps * 4)  # 4s cooldown

        # ── Goal detection (enhanced: x<3 or x>102, y 25-43m)
        if self.goal_cooldown <= 0 and self._is_any_goal(bx, by):
            scorer_id = self.last_possession_id
            assist_id = self.prev_possession_id if self.prev_possession_id != scorer_id else None
            if scorer_id and scorer_id in self.events:
                self.events[scorer_id]["goals"] += 1
            minute = int(frame_idx / (self.fps * 60)) if self.fps > 0 else 0
            self.goal_events.append({
                "type": "goal",
                "player_id": scorer_id,
                "assist_player_id": assist_id,
                "minute": minute,
            })
            # Credit assist
            if assist_id and assist_id in self.events:
                pass  # assists aggregated in detector.py from goal_events
            self.goal_cooldown = int(self.fps * 15)  # 15s cooldown

        if possessor_id:
            self.last_possession_id = possessor_id

        # Update tackle cooldowns
        for pid in list(self.tackle_cooldown):
            self.tackle_cooldown[pid] = max(0, self.tackle_cooldown[pid] - 1)

    def get_player_stats(self, player_id):
        e = self.events.get(player_id, {})
        attempted = max(e.get("passesAttempted", 0), 1)
        successful = e.get("passes", 0)
        accuracy = round(successful / attempted * 100, 1)
        # Count assists from goal_events
        assists = sum(
            1 for ev in self.goal_events
            if ev.get("assist_player_id") == player_id
        )
        return {
            "passes":          successful,
            "passesAttempted": e.get("passesAttempted", 0),
            "passAccuracy":    accuracy,
            "shots":           e.get("shots", 0),
            "shotsOnTarget":   e.get("shotsOnTarget", 0),
            "goals":           e.get("goals", 0),
            "assists":         assists,
            "tackles":         e.get("tackles", 0),
            "interceptions":   e.get("interceptions", 0),
            "touches":         e.get("touches", 0),
            "xG":              e.get("xG", 0.0),
        }
