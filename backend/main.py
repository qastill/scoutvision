from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
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

jobs = {}  # {jobId: {status, progress, step, message, results, error}}


def update_job(job_id, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


async def run_pipeline(job_id: str, video_path: str):
    try:
        update_job(job_id, status="processing", progress=5, step="extract", message="Extracting frames...")
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, process_video, video_path, job_id, update_job)

        update_job(job_id, status="generating", progress=90, step="pdf", message="Generating reports...")
        match_pdf_path = f"/tmp/sv_{job_id}_match.pdf"
        player_pdfs = {}
        generate_match_pdf(results, match_pdf_path)
        for p in results["players"]:
            pdf_path = f"/tmp/sv_{job_id}_player_{p['id']}.pdf"
            generate_player_pdf(p, results, pdf_path)
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
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


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
