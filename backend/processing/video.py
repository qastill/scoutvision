"""
video.py — ffmpeg frame extraction utilities
"""
import ffmpeg
import os
import numpy as np


def extract_frames(video_path: str, output_dir: str, fps: float = 2.0) -> list[str]:
    """
    Extract frames from video at given fps using ffmpeg-python.
    Returns list of extracted frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")

    (
        ffmpeg
        .input(video_path)
        .filter("fps", fps=fps)
        .output(output_pattern, q=2)
        .overwrite_output()
        .run(quiet=True)
    )

    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".jpg")
    ])
    return frames


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata: duration, fps, width, height.
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (s for s in probe["streams"] if s["codec_type"] == "video"),
            None
        )
        if video_stream:
            duration = float(probe["format"].get("duration", 0))
            fps_str = video_stream.get("r_frame_rate", "25/1")
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 25.0
            width = int(video_stream.get("width", 1920))
            height = int(video_stream.get("height", 1080))
            return {"duration": duration, "fps": fps, "width": width, "height": height}
    except Exception:
        pass
    return {"duration": 5400.0, "fps": 25.0, "width": 1920, "height": 1080}


def format_duration(seconds: float) -> str:
    """Format seconds to MM:SS string."""
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"
