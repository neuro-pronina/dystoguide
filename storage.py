from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path

BASE = Path("data")
VIDEOS = BASE / "videos"
RESULTS = BASE / "results"
REPORTS = BASE / "reports"

for p in (VIDEOS, RESULTS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

def paths_for_patient(patient_id: str, ts: datetime) -> dict:
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    folder = VIDEOS / patient_id
    folder.mkdir(parents=True, exist_ok=True)
    video_path = folder / f"{stamp}.mp4"
    res_path = RESULTS / patient_id / f"{stamp}.json"
    rep_folder = REPORTS / patient_id
    rep_folder.mkdir(parents=True, exist_ok=True)
    pdf_path = rep_folder / f"{stamp}.pdf"
    return {"video": video_path, "json": res_path, "pdf": pdf_path}

def save_uploaded_video(patient_id: str, stamp: datetime, data: bytes) -> str:
    paths = paths_for_patient(patient_id, stamp)
    with open(paths["video"], "wb") as f:
        f.write(data)
    return str(paths["video"])
