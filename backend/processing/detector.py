"""
detector.py — YOLOv8 + ByteTrack player detection and tracking pipeline
"""
import cv2
import numpy as np
import os
import json
from datetime import datetime

from processing.stats import calculate_player_stats, pixel_to_field, infer_position, build_team_stats
from processing.teams import assign_teams, get_dominant_color


FRAME_SAMPLE_RATE = 5  # sample every 5th frame
MIN_TRACK_FRAMES = 10  # discard tracks with < 10 detections (noise)
MAX_PLAYERS = 30       # max tracked players to consider


def process_video(video_path: str, job_id: str, update_fn) -> dict:
    """
    Main pipeline: load video → detect → track → stats → return results dict.
    """
    video_name = os.path.basename(video_path)

    # --- Step 1: Open video ---
    update_fn(job_id, step="extract", progress=8, message="Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    duration_sec = total_frames / fps if fps > 0 else 5400.0

    # Format duration
    dur_min = int(duration_sec) // 60
    dur_sec = int(duration_sec) % 60
    duration_str = f"{dur_min:02d}:{dur_sec:02d}"

    update_fn(job_id, step="detect", progress=15, message=f"Loading YOLO model... (video: {duration_str})")

    # --- Step 2: Load YOLO ---
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
    except Exception as e:
        cap.release()
        raise RuntimeError(f"Failed to load YOLO: {e}")

    # --- Step 3: Setup ByteTrack via supervision ---
    try:
        import supervision as sv
        tracker = sv.ByteTrack()
    except Exception as e:
        cap.release()
        raise RuntimeError(f"Failed to init ByteTrack: {e}")

    # Track data: {track_id: {"positions": [...], "crops": [...]}}
    tracks = {}
    frame_idx = 0
    processed = 0
    total_to_process = max(1, total_frames // FRAME_SAMPLE_RATE)

    update_fn(job_id, step="detect", progress=20, message="Detecting players...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every Nth frame
        if frame_idx % FRAME_SAMPLE_RATE != 0:
            frame_idx += 1
            continue

        processed += 1

        # Progress update every 50 processed frames
        if processed % 50 == 0:
            pct = 20 + int((processed / total_to_process) * 50)
            pct = min(pct, 70)
            update_fn(job_id, step="detect", progress=pct,
                      message=f"Detecting players... frame {processed}/{total_to_process}")

        # YOLO detection — only detect 'person' class (class 0)
        try:
            results = model(frame, classes=[0], conf=0.35, verbose=False)
        except Exception:
            frame_idx += 1
            continue

        if not results or len(results) == 0:
            frame_idx += 1
            continue

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            frame_idx += 1
            continue

        # Convert to supervision Detections
        try:
            detections = sv.Detections.from_ultralytics(result)
        except Exception:
            frame_idx += 1
            continue

        # Filter to person class only
        if detections.class_id is not None:
            mask = detections.class_id == 0
            detections = detections[mask]

        if len(detections) == 0:
            frame_idx += 1
            continue

        # ByteTrack update
        try:
            tracked = tracker.update_with_detections(detections)
        except Exception:
            frame_idx += 1
            continue

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            frame_idx += 1
            continue

        # Process each tracked detection
        for i, track_id in enumerate(tracked.tracker_id):
            tid = int(track_id)
            if tid not in tracks:
                tracks[tid] = {"positions": [], "crops": []}

            # Bounding box center
            try:
                box = tracked.xyxy[i]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Field coords
                fx, fy = pixel_to_field(cx, cy, frame_w, frame_h)
                tracks[tid]["positions"].append((frame_idx, fx, fy))

                # Crop jersey (middle 30% of bbox height = torso)
                crop_y1 = y1 + int((y2 - y1) * 0.2)
                crop_y2 = y1 + int((y2 - y1) * 0.6)
                crop_x1 = max(0, x1)
                crop_x2 = min(frame_w, x2)

                if crop_y2 > crop_y1 and crop_x2 > crop_x1 and len(tracks[tid]["crops"]) < 5:
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop.size > 0:
                        tracks[tid]["crops"].append(crop)
            except Exception:
                continue

        frame_idx += 1

    cap.release()

    update_fn(job_id, step="track", progress=72, message="Analyzing tracks...")

    # --- Step 4: Filter valid tracks ---
    valid_tracks = {
        tid: data for tid, data in tracks.items()
        if len(data["positions"]) >= MIN_TRACK_FRAMES
    }

    # Sort by number of detections (most seen = most likely real players)
    sorted_tracks = sorted(valid_tracks.items(), key=lambda x: -len(x[1]["positions"]))
    sorted_tracks = sorted_tracks[:MAX_PLAYERS]

    update_fn(job_id, step="stats", progress=78, message=f"Computing stats for {len(sorted_tracks)} players...")

    # --- Step 5: Jersey color extraction ---
    track_colors = {}
    for tid, data in sorted_tracks:
        crops = data.get("crops", [])
        if crops:
            color = get_dominant_color(crops[0])
        else:
            color = [128, 128, 128]
        track_colors[tid] = color

    # --- Step 6: Compute per-player stats ---
    players_raw = []
    for player_num, (tid, data) in enumerate(sorted_tracks, start=1):
        stats = calculate_player_stats(tid, data["positions"], fps, FRAME_SAMPLE_RATE)
        stats["id"] = player_num
        stats["jerseyColor"] = track_colors.get(tid, [128, 128, 128])
        stats["team"] = ""
        stats["teamColor"] = "#FFFFFF"
        stats["position"] = infer_position(stats["avgX"], stats["avgY"])
        players_raw.append(stats)

    # --- Step 7: Team assignment via k-means on jersey colors ---
    update_fn(job_id, step="stats", progress=84, message="Assigning teams...")
    players_with_teams, team_a, team_b = assign_teams(players_raw)

    # Re-infer positions now that we know teams
    for p in players_with_teams:
        p["position"] = infer_position(p["avgX"], p["avgY"], p["team"])

    # --- Step 8: Team stats ---
    team_stats = {
        team_a: build_team_stats(players_with_teams, team_a),
        team_b: build_team_stats(players_with_teams, team_b),
    }

    update_fn(job_id, step="stats", progress=88, message="Finalizing results...")

    results = {
        "videoName": video_name,
        "duration": duration_str,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "teamA": team_a,
        "teamB": team_b,
        "players": players_with_teams,
        "teamStats": team_stats,
        "frameCount": frame_idx,
        "processedFrames": processed,
        "fps": round(fps, 1),
    }

    # Save results JSON
    results_path = f"/tmp/sv_{job_id}_results.json"
    try:
        with open(results_path, "w") as f:
            # Create serializable copy
            r_copy = dict(results)
            json.dump(r_copy, f, indent=2)
    except Exception:
        pass

    return results
