from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import uuid, json, os, asyncio
from processing.detector import process_video
from reports.match_pdf import generate_match_pdf
from reports.player_pdf import generate_player_pdf

app = FastAPI(title="ScoutVision API", max_upload_size=4 * 1024 * 1024 * 1024)  # 4GB max upload
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

jobs = {}  # {jobId: {status, progress, step, message, results, roster, error}}


def update_job(job_id, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


async def run_pipeline(job_id: str, video_path: str):
    """Run video processing pipeline. Stops at 'processed' status awaiting roster input."""
    try:
        update_job(job_id, status="processing", progress=5, step="extract", message="Extracting frames...")
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, process_video, video_path, job_id, update_job)

        # Store results and set status to 'processed' — wait for roster before generating PDFs
        update_job(
            job_id,
            status="processed",
            progress=85,
            step="processed",
            message="Video processed! Ready for roster input.",
            results=results
        )

        # Save player thumbnail crops to disk (from detector._crop field)
        import cv2 as _cv2
        for player in results.get("players", []):
            pid = player["id"]
            crop = player.pop("_crop", None)  # remove non-JSON-serializable numpy array
            if crop is not None and hasattr(crop, 'shape'):
                thumb_path = f"/tmp/sv_{job_id}_thumb_{pid}.jpg"
                try:
                    resized = _cv2.resize(crop, (80, 120), interpolation=_cv2.INTER_LINEAR)
                    _cv2.imwrite(thumb_path, resized)
                    player["_thumb_path"] = thumb_path
                except Exception:
                    player["_thumb_path"] = None
            else:
                player["_thumb_path"] = None

    except Exception as e:
        import traceback
        update_job(job_id, status="error", message=str(e), error=traceback.format_exc())
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


async def generate_pdfs(job_id: str):
    """Generate match + player PDFs, merging roster data into player records."""
    try:
        j = jobs[job_id]
        results = j["results"]
        roster = j.get("roster", {})

        update_job(job_id, status="generating", progress=90, step="pdf", message="Generating PDF reports...")

        # Merge roster info into player dicts
        players_with_roster = []
        for p in results["players"]:
            pid = p["id"]
            p_copy = dict(p)
            if pid in roster:
                if roster[pid].get("name"):
                    p_copy["name"] = roster[pid]["name"]
                if roster[pid].get("number"):
                    p_copy["number"] = roster[pid]["number"]
                if roster[pid].get("photo_path") and os.path.exists(roster[pid]["photo_path"]):
                    p_copy["photo_path"] = roster[pid]["photo_path"]
            players_with_roster.append(p_copy)

        results_with_roster = dict(results)
        results_with_roster["players"] = players_with_roster
        # Include manual events in PDF data
        manual_events = j.get("manual_events", [])
        if manual_events:
            results_with_roster["manualEvents"] = manual_events

        match_pdf_path = f"/tmp/sv_{job_id}_match.pdf"
        player_pdfs = {}
        generate_match_pdf(results_with_roster, match_pdf_path)
        for p in players_with_roster:
            pdf_path = f"/tmp/sv_{job_id}_player_{p['id']}.pdf"
            generate_player_pdf(p, results_with_roster, pdf_path)
            player_pdfs[p["id"]] = pdf_path

        results["_match_pdf"] = match_pdf_path
        results["_player_pdfs"] = player_pdfs
        update_job(
            job_id,
            status="done",
            progress=100,
            step="done",
            message="Analysis complete!",
            results=results
        )
    except Exception as e:
        import traceback
        update_job(job_id, status="error", message=str(e), error=traceback.format_exc())


@app.get("/health")
def health():
    return {"status": "ok", "service": "scoutvision"}


@app.get("/api/player/{job_id}/thumb/{player_id}")
def get_player_thumb(job_id: str, player_id: int):
    """Return cropped player thumbnail JPEG, or colored SVG placeholder."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j.get("status") not in ("processed", "generating", "done"):
        raise HTTPException(400, "Not ready")

    results = j.get("results", {})
    players = results.get("players", [])
    player = next((p for p in players if p["id"] == player_id), None)

    thumb_path = player.get("_thumb_path") if player else None
    if thumb_path and os.path.exists(thumb_path):
        return FileResponse(thumb_path, media_type="image/jpeg")

    # Colored SVG placeholder
    jersey = player.get("jerseyColor", [128, 128, 128]) if player else [128, 128, 128]
    r, g, b = int(jersey[0]), int(jersey[1]), int(jersey[2])
    pid_label = str(player_id)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="80" height="120" viewBox="0 0 80 120">
  <rect width="80" height="120" fill="#0D1520" rx="4"/>
  <circle cx="40" cy="35" r="18" fill="#{r:02X}{g:02X}{b:02X}" opacity="0.8"/>
  <text x="40" y="41" font-family="Arial" font-size="16" font-weight="bold" fill="white" text-anchor="middle">{pid_label}</text>
  <rect x="15" y="60" width="50" height="40" rx="4" fill="#{r:02X}{g:02X}{b:02X}" opacity="0.6"/>
  <text x="40" y="85" font-family="Arial" font-size="11" fill="white" text-anchor="middle">PLAYER</text>
  <text x="40" y="100" font-family="Arial" font-size="11" fill="white" text-anchor="middle">#{pid_label}</text>
</svg>'''
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/players/{job_id}")
def get_players_info(job_id: str):
    """Return player list with position/zone data when in processed state."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j.get("status") not in ("processed", "generating", "done"):
        raise HTTPException(400, "Not ready")
    players = j.get("results", {}).get("players", [])
    safe = []
    for p in players:
        safe.append({
            "id": p["id"],
            "team": p.get("team", ""),
            "teamColor": p.get("teamColor", "#FFFFFF"),
            "jerseyColor": p.get("jerseyColor", [128, 128, 128]),
            "position": p.get("position", "CM"),
            "avgX": p.get("avgX", 52.5),
            "avgY": p.get("avgY", 34.0),
            "attackPct": p.get("attackPct", 0),
            "midPct": p.get("midPct", 0),
            "defPct": p.get("defPct", 0),
        })
    return {"players": safe}


@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    video_path = f"/tmp/sv_{job_id}.mp4"

    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "step": "upload",
        "message": "Video uploaded, queued for processing"
    }
    background_tasks.add_task(run_pipeline, job_id, video_path)
    return {"jobId": job_id, "status": "queued"}


@app.post("/api/roster/{job_id}/skip")
@app.get("/api/roster/{job_id}/skip")
async def skip_roster(job_id: str, background_tasks: BackgroundTasks):
    """Skip roster input and generate PDFs — also used as the generate trigger after kejadian step."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] not in ("processed", "roster_saved"):
        raise HTTPException(400, f"Job not ready: {jobs[job_id]['status']}")
    if "roster" not in jobs[job_id]:
        jobs[job_id]["roster"] = {}
    background_tasks.add_task(generate_pdfs, job_id)
    return {"status": "generating", "jobId": job_id}


@app.post("/api/roster/{job_id}")
async def submit_roster(job_id: str, request: Request):
    """Accept roster data (names, numbers, photos). Does NOT trigger PDF generation.
    Call /api/roster/{id}/skip afterwards to generate."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] not in ("processed",):
        raise HTTPException(400, f"Job not ready for roster: {jobs[job_id]['status']}")

    form = await request.form()
    roster = {}

    for key in form:
        value = form[key]
        # Expected format: player_{id}_name, player_{id}_number, player_{id}_photo
        parts = key.split("_")
        if len(parts) < 3 or parts[0] != "player":
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        field = "_".join(parts[2:])  # handle multi-part field names

        if pid not in roster:
            roster[pid] = {}

        if field == "name":
            val = str(value).strip()
            if val:
                roster[pid]["name"] = val
        elif field == "number":
            val = str(value).strip()
            if val:
                roster[pid]["number"] = val
        elif field == "photo":
            # UploadFile object
            if hasattr(value, "read"):
                photo_data = await value.read()
                if photo_data and len(photo_data) > 100:
                    photo_path = f"/tmp/sv_{job_id}_roster_{pid}.jpg"
                    with open(photo_path, "wb") as f:
                        f.write(photo_data)
                    roster[pid]["photo_path"] = photo_path

    jobs[job_id]["roster"] = roster
    jobs[job_id]["status"] = "roster_saved"
    return {"status": "roster_saved", "jobId": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return {
        "jobId": job_id,
        "status": j["status"],
        "progress": j.get("progress", 0),
        "step": j.get("step"),
        "message": j.get("message", "")
    }


@app.post("/api/events/{job_id}")
async def submit_events(job_id: str, request: Request):
    """Store manual game events (goals, cards) and update player stats."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] not in ("processed", "roster_saved", "generating", "done"):
        raise HTTPException(400, f"Job not ready for events: {jobs[job_id]['status']}")
    data = await request.json()
    events = data.get("events", [])
    jobs[job_id]["manual_events"] = events
    # Update player goal/assist counts from manual events
    players = jobs[job_id].get("results", {}).get("players", [])
    if players and events:
        player_map = {p["id"]: p for p in players}
        for p in players:
            p["goals"] = 0
            p["assists"] = 0
        for ev in events:
            if ev.get("type") == "goal":
                pid = ev.get("playerId")
                apid = ev.get("assistPlayerId")
                if pid and pid in player_map:
                    player_map[pid]["goals"] = player_map[pid].get("goals", 0) + 1
                if apid and apid in player_map:
                    player_map[apid]["assists"] = player_map[apid].get("assists", 0) + 1
        # Recompute Man of the Match after manual updates
        if players:
            best = max(players, key=lambda p: p.get("matchRating", 5.0))
            for p in players:
                p["manOfTheMatch"] = (p is best)
    return {"status": "ok", "eventsCount": len(events)}


@app.get("/api/results/{job_id}")
def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "done":
        raise HTTPException(400, f"Job not done yet: {j['status']}")
    r = dict(j["results"])
    r.pop("_match_pdf", None)
    r.pop("_player_pdfs", None)
    # Include manual events if present
    if j.get("manual_events"):
        r["manualEvents"] = j["manual_events"]
    return r


@app.get("/api/report/{job_id}/match")
def get_match_pdf(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        raise HTTPException(404, "Report not ready")
    pdf_path = jobs[job_id]["results"]["_match_pdf"]
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"ScoutVision_Match_{job_id}.pdf")


@app.get("/api/report/{job_id}/player/{player_id}")
def get_player_pdf(job_id: str, player_id: int):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        raise HTTPException(404, "Report not ready")
    pdf_path = jobs[job_id]["results"]["_player_pdfs"].get(player_id)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(404, "Player PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"ScoutVision_Player_{player_id}_{job_id}.pdf")
