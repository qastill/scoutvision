from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uuid, json, os, asyncio
from processing.detector import process_video
from reports.match_pdf import generate_match_pdf
from reports.player_pdf import generate_player_pdf

app = FastAPI(title="ScoutVision API")
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


@app.post("/api/roster/{job_id}")
async def submit_roster(job_id: str, background_tasks: BackgroundTasks, request: Request):
    """Accept roster data (names, numbers, photos) and trigger PDF generation."""
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
    background_tasks.add_task(generate_pdfs, job_id)
    return {"status": "generating", "jobId": job_id}


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
