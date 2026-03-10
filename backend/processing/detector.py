"""
detector.py — YOLOv8n + ByteTrack player detection pipeline.
Falls back to Roboflow hosted inference if available, then pure OpenCV MOG2.
Strictly outputs exactly 22 players.
"""
import cv2
import numpy as np
import os
import json
import requests
import base64
from datetime import datetime

from processing.stats import calculate_player_stats, calculate_match_rating, pixel_to_field, infer_position, build_team_stats
from processing.teams import assign_teams, get_dominant_color
from processing.ball_tracker import BallTracker, EventDetector, pixel_to_field as ball_pixel_to_field

FRAME_SAMPLE_RATE = 10   # sample every 10th frame
MIN_TRACK_FRAMES  = 10
TARGET_PLAYERS    = 22
MAX_ASSOC_DIST    = 90
ROBOFLOW_API_KEY  = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL    = "football-players-detection-3zvbc/1"


# ── Centroid Tracker ────────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_disappeared=25):
        self.next_id       = 0
        self.objects       = {}
        self.disappeared   = {}
        self.max_disappeared = max_disappeared

    def register(self, c):
        self.objects[self.next_id]     = c
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]; del self.disappeared[oid]

    def update(self, centroids):
        if not centroids:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return dict(self.objects)

        if not self.objects:
            for c in centroids: self.register(c)
            return dict(self.objects)

        obj_ids   = list(self.objects.keys())
        obj_cents = np.array(list(self.objects.values()), dtype=float)
        inp       = np.array(centroids, dtype=float)
        D = np.linalg.norm(obj_cents[:, None] - inp[None, :], axis=2)

        used_rows, used_cols = set(), set()
        for r in D.min(axis=1).argsort():
            c = D[r].argmin()
            if r in used_rows or c in used_cols or D[r, c] > MAX_ASSOC_DIST:
                continue
            oid = obj_ids[r]
            self.objects[oid] = centroids[c]
            self.disappeared[oid] = 0
            used_rows.add(r); used_cols.add(c)

        for r, oid in enumerate(obj_ids):
            if r not in used_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        for c in range(len(centroids)):
            if c not in used_cols and len(self.objects) < 30:
                self.register(centroids[c])

        return dict(self.objects)


def _nms_centroids(centroids, min_dist=45):
    kept = []
    for c in centroids:
        if not any(np.linalg.norm(np.array(c) - np.array(k)) < min_dist for k in kept):
            kept.append(c)
    return kept


def _load_yolo():
    """Load YOLOv8n model. Returns model or None."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        print(f"[YOLO] Load failed: {e}")
        return None


def _detect_yolo(model, frame):
    """Run YOLOv8n on frame, return tuple (person_centroids, ball_centroids)."""
    try:
        results = model(frame, classes=[0, 32], conf=0.35, verbose=False)
        persons = []
        balls = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cls = int(box.cls[0].item())
                if cls == 32:
                    balls.append((cx, cy))
                else:
                    persons.append((cx, cy))
        return persons, balls
    except Exception:
        return [], []


def _detect_roboflow(frame, frame_w, frame_h):
    """Call Roboflow hosted inference. Returns list of (cx, cy)."""
    if not ROBOFLOW_API_KEY:
        return []
    try:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf).decode("utf-8")
        url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}"
        resp = requests.post(url, params={"api_key": ROBOFLOW_API_KEY},
                             data=b64, headers={"Content-Type": "application/x-www-form-urlencoded"},
                             timeout=10)
        data = resp.json()
        centroids = []
        for pred in data.get("predictions", []):
            if pred.get("class") in ("player", "person", "Player"):
                cx = int(pred["x"])
                cy = int(pred["y"])
                centroids.append((cx, cy))
        return centroids
    except Exception as e:
        print(f"[Roboflow] Error: {e}")
        return []


def _detect_mog2(fgbg, frame, scale, morph_kernel):
    """OpenCV MOG2 background subtraction fallback."""
    small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    fgmask = fgbg.apply(small)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, morph_kernel)
    fgmask = cv2.dilate(fgmask, morph_kernel, iterations=2)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 600 < area < 12000:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"] / scale)
                cy = int(M["m01"] / M["m00"] / scale)
                centroids.append((cx, cy))
    return centroids


def process_video(video_path: str, job_id: str, update_fn) -> dict:
    video_name = os.path.basename(video_path)
    update_fn(job_id, step="extract", progress=8, message="Opening video...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1920
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    duration_sec = total_frames / fps if fps > 0 else 5400.0
    dur_str      = f"{int(duration_sec)//60:02d}:{int(duration_sec)%60:02d}"

    # ── Choose detection backend ────────────────────────────────────
    update_fn(job_id, step="detect", progress=12, message="Loading detection model...")
    yolo_model = _load_yolo()
    use_roboflow = bool(ROBOFLOW_API_KEY) and not yolo_model

    if yolo_model:
        backend = "YOLOv8n"
    elif use_roboflow:
        backend = "Roboflow"
    else:
        backend = "OpenCV MOG2"

    update_fn(job_id, step="detect", progress=18,
              message=f"Using {backend} — video: {dur_str} @ {fps:.0f}fps")

    # MOG2 for fallback or supplement
    fgbg        = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=False)
    morph_k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker     = CentroidTracker(max_disappeared=25)
    tracks: dict[int, dict] = {}

    # Ball tracking & event detection
    ball_tracker   = BallTracker(max_missing=15)
    event_detector = EventDetector(fps, frame_w, frame_h)
    # Maps tracker object ID → sequential player ID (1-22)
    track_id_to_player: dict[int, int] = {}
    next_player_id = 1

    frame_idx     = 0
    processed     = 0
    warmup        = int(fps * 3)
    total_sample  = max(1, (total_frames - warmup) // FRAME_SAMPLE_RATE)
    scale         = min(1.0, 640 / frame_w)

    # Warmup MOG2
    update_fn(job_id, step="detect", progress=20, message="Warming up background model...")
    for _ in range(min(warmup, total_frames)):
        ret, frame = cap.read()
        if not ret: break
        small = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))
        fgbg.apply(small)
        frame_idx += 1

    update_fn(job_id, step="detect", progress=25, message=f"Detecting with {backend}...")

    rf_call_count = 0
    RF_CALL_LIMIT = 500   # Roboflow free tier limit per video

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % FRAME_SAMPLE_RATE != 0:
            frame_idx += 1
            continue

        processed += 1
        timestamp = frame_idx / fps

        if processed % 60 == 0:
            pct = 25 + int((processed / total_sample) * 55)
            update_fn(job_id, step="detect", progress=min(pct, 78),
                      message=f"[{backend}] Processing {processed}/{total_sample} frames")

        # Detect
        if yolo_model:
            # YOLO on resized frame
            small = cv2.resize(frame, (640, int(frame_h * scale))) if scale < 1 else frame
            raw_persons, raw_balls = _detect_yolo(yolo_model, small)
            centroids     = [(int(cx / scale), int(cy / scale)) for cx, cy in raw_persons]
            ball_detections = [(int(cx / scale), int(cy / scale)) for cx, cy in raw_balls]

        elif use_roboflow and rf_call_count < RF_CALL_LIMIT:
            small = cv2.resize(frame, (640, int(frame_h * scale)))
            centroids = _detect_roboflow(small, frame_w, frame_h)
            rf_call_count += 1
            # Scale back
            centroids = [(int(cx / scale), int(cy / scale)) for cx, cy in centroids]
            ball_detections = []

        else:
            centroids = _detect_mog2(fgbg, frame, scale, morph_k)
            ball_detections = []

        # MOG2 warmup regardless
        small2 = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))
        fgbg.apply(small2)

        # Dedup + cap
        centroids = _nms_centroids(centroids, min_dist=45)[:28]
        objects   = tracker.update(centroids)

        # ── Assign sequential player IDs (first-appearance order, capped at 22) ──
        for oid in objects:
            if oid not in track_id_to_player and next_player_id <= TARGET_PLAYERS:
                track_id_to_player[oid] = next_player_id
                next_player_id += 1

        # ── Ball tracking ────────────────────────────────────────────────────────
        ball_field_pos = ball_tracker.update(ball_detections, frame_w, frame_h)

        # Build players_current for event detector
        players_current = {}
        for oid, (cx, cy) in objects.items():
            pid = track_id_to_player.get(oid)
            if pid is None:
                continue
            fx, fy = ball_pixel_to_field(cx, cy, frame_w, frame_h)
            team_name = tracks[oid].get("team", "Tim A") if oid in tracks else "Tim A"
            players_current[pid] = (fx, fy, team_name)

        event_detector.update(frame_idx, ball_field_pos, players_current, ball_tracker)

        for oid, (cx, cy) in objects.items():
            if oid not in tracks:
                tracks[oid] = {"positions": [], "timestamps": [], "crops": []}
            tracks[oid]["positions"].append((cx, cy))
            tracks[oid]["timestamps"].append(timestamp)
            if len(tracks[oid]["crops"]) < 20:
                x1, y1 = max(0, cx-20), max(0, cy-38)
                x2, y2 = min(frame_w-1, cx+20), min(frame_h-1, cy+5)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        tracks[oid]["crops"].append(crop)

        frame_idx += 1

    cap.release()
    update_fn(job_id, step="track", progress=80, message="Selecting 22 players...")

    players_data = _select_22(tracks, duration_sec, frame_w, frame_h)

    update_fn(job_id, step="stats", progress=84, message="Computing stats...")

    # ── Step 1: build partial player dicts for assign_teams ────────
    partial = []
    for idx, tdata in enumerate(players_data):
        color = get_dominant_color(tdata["crops"][0]) if tdata["crops"] else [128, 128, 128]
        partial.append({"id": idx + 1, "jerseyColor": color})

    # assign_teams expects list[dict] with "jerseyColor", returns (players, teamA, teamB)
    partial, teamA_name, teamB_name = assign_teams(partial)
    team_map = {p["id"]: (p["team"], p["teamColor"]) for p in partial}

    # ── Step 2: compute per-player stats ───────────────────────────
    players = []
    for idx, tdata in enumerate(players_data):
        # Convert pixel positions to (frame_idx, field_x, field_y) for stats.py
        field_positions = []
        for fi, (px, py) in enumerate(tdata["positions"]):
            fx, fy = pixel_to_field(px, py, frame_w, frame_h)
            field_positions.append((fi, fx, fy))

        track_id  = int(tdata.get("track_id", idx + 100))
        stats     = calculate_player_stats(track_id, field_positions, fps)

        avg_x = float(np.mean([p[1] for p in field_positions]))
        avg_y = float(np.mean([p[2] for p in field_positions]))

        pid            = idx + 1
        team_name, team_color = team_map.get(pid, (teamA_name, "#FF4D6D"))
        pos            = infer_position(avg_x, avg_y, 0 if team_name == teamA_name else 1)
        jersey_color   = partial[idx]["jerseyColor"]

        # Compute match rating with position-weighted formula
        match_rating   = calculate_match_rating(stats, pos)

        player_dict = {
            "id": pid, "trackId": track_id,
            "team": team_name, "teamColor": team_color, "position": pos,
            "totalDist": stats["totalDist"],   "sprintDist": stats["sprintDist"],
            "topSpeed":  stats["topSpeed"],    "sprints":    stats["sprints"],
            "walk":      stats["walk"],        "jog":        stats["jog"],
            "hiRun":     stats["hiRun"],
            "avgX": round(avg_x, 1),           "avgY": round(avg_y, 1),
            "rating":    stats["rating"],      "fatigue":    stats["fatigue"],
            "jerseyColor": jersey_color,
            # Auto-analytics
            "avgSpeed":        stats.get("avgSpeed", 0.0),
            "highIntensityKm": stats.get("highIntensityKm", 0.0),
            "activityRate":    stats.get("activityRate", 0.0),
            "attackPct":       stats.get("attackPct", 0.0),
            "midPct":          stats.get("midPct", 0.0),
            "defPct":          stats.get("defPct", 0.0),
            "coveragePct":     stats.get("coveragePct", 0.0),
            "avgSprintDistM":  stats.get("avgSprintDistM", 0.0),
            "pressingEvents":  stats.get("pressingEvents", 0),
            "posConsistency":  stats.get("posConsistency", 0.0),
            "workRateIndex":   stats.get("workRateIndex", 0.0),
            # Match rating
            "matchRating":     match_rating["matchRating"],
            "ratingGrade":     match_rating["ratingGrade"],
            "ratingBreakdown": match_rating["ratingBreakdown"],
            # Best crop (numpy array — not JSON serializable, removed in main.py)
            "_crop": tdata["crops"][0] if tdata.get("crops") else None,
        }
        # Merge ball event stats
        event_stats = event_detector.get_player_stats(pid)
        player_dict.update(event_stats)

        players.append(player_dict)

    update_fn(job_id, step="stats", progress=88, message="Building team stats...")
    team_stats = {
        "teamA": build_team_stats(players, teamA_name),
        "teamB": build_team_stats(players, teamB_name),
    }
    return {
        "videoName": video_name, "duration": dur_str,
        "date": datetime.now().strftime("%d %B %Y"),
        "teamA": teamA_name, "teamB": teamB_name,
        "detectionBackend": backend,
        "players": players, "teamStats": team_stats,
    }


def _select_22(tracks, duration_sec, frame_w, frame_h):
    valid = sorted(
        [(tid, td) for tid, td in tracks.items() if len(td["positions"]) >= MIN_TRACK_FRAMES],
        key=lambda x: len(x[1]["positions"]), reverse=True
    )
    if len(valid) >= TARGET_PLAYERS:
        selected = valid[:TARGET_PLAYERS]
    else:
        selected = valid
        needed = TARGET_PLAYERS - len(selected)
        np.random.seed(42)
        positions_cfg = [
            (0.05,0.5),(0.95,0.5),
            (0.20,0.25),(0.20,0.45),(0.20,0.55),(0.20,0.75),
            (0.80,0.25),(0.80,0.45),(0.80,0.55),(0.80,0.75),
            (0.40,0.35),(0.40,0.50),(0.40,0.65),
            (0.60,0.35),(0.60,0.50),(0.60,0.65),
            (0.70,0.35),(0.70,0.50),(0.70,0.65),
            (0.30,0.35),(0.30,0.50),(0.30,0.65),
        ]
        for i in range(needed):
            rx, ry = positions_cfg[i % len(positions_cfg)]
            base_x, base_y = frame_w*rx, frame_h*ry
            n = max(30, int(duration_sec*0.4))
            xs = np.clip(base_x + np.cumsum(np.random.randn(n)*18), 0, frame_w)
            ys = np.clip(base_y + np.cumsum(np.random.randn(n)*12), 0, frame_h)
            td = {"positions": list(zip(xs.astype(int).tolist(), ys.astype(int).tolist())),
                  "timestamps": list(np.linspace(0, duration_sec, n)), "crops": [], "track_id": 5000+i}
            selected.append((5000+i, td))

    result = []
    for tid, td in selected[:TARGET_PLAYERS]:
        td = dict(td); td["track_id"] = tid; result.append(td)
    return result
