from __future__ import annotations
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
from utils import DEG, find_first_csv, longest_true_segment, estimate_brightness, is_dark, estimate_instability

CalibrationMode = Literal["off", "auto", "cmor13"]

MAE = {"pitch": 3.5, "roll": 3.1, "yaw": 3.1}

@dataclass
class PoseResult:
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    involved_axes: list[str]
    severity_axes: dict[str, str]
    severity_overall: str
    qc: dict
    fps: float
    n_frames: int

class OpenFaceNotFound(RuntimeError):
    pass


def _run_openface(feature_exe: str, video_path: str, out_dir: str) -> str:
    if not feature_exe or not os.path.exists(feature_exe):
        raise OpenFaceNotFound(
            "OpenFace FeatureExtraction не найден. Укажите путь через переменную окружения OPENFACE_CMD "
            "или соберите в Dockerfile."
        )
    cmd = [
        feature_exe,
        "-f", video_path,
        "-out_dir", out_dir,
        "-pose", "-aus",
        "-2Dfp", "-3Dfp",
        "-q"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    csv_path = find_first_csv(out_dir)
    if not csv_path:
        raise RuntimeError("OpenFace не сгенерировал CSV")
    return csv_path


def _read_openface_csv(csv_path: str) -> Tuple[pd.DataFrame, float]:
    df = pd.read_csv(csv_path)
    for col in ("pose_Rx", "pose_Ry", "pose_Rz"):
        if col not in df.columns:
            raise RuntimeError(f"В CSV нет столбца {col}")
    df["pitch"] = np.degrees(df["pose_Rx"].astype(float))
    df["yaw"] = np.degrees(df["pose_Ry"].astype(float))
    df["roll"] = np.degrees(df["pose_Rz"].astype(float))
    if "timestamp" in df.columns and len(df) > 1:
        dt = np.diff(df["timestamp"].values)
        dt = dt[dt > 0]
        fps = float(1.0 / np.median(dt)) if len(dt) else 30.0
    else:
        fps = 30.0
    return df, fps


def _select_confident_segment(df: pd.DataFrame, conf_thr: float = 0.7) -> Tuple[int, int]:
    if "confidence" not in df.columns:
        raise RuntimeError("В CSV нет столбца confidence")
    mask = (df["confidence"].astype(float) >= conf_thr).values
    start, end = longest_true_segment(mask)
    return start, end


def _calibrate_pitch(series: np.ndarray, fps: float, mode: CalibrationMode) -> float:
    if mode == "off":
        return 0.0
    if mode == "cmor13":
        return -13.0
    window = int(max(1, min(2.0, len(series) / max(1.0, fps)) * fps))
    if window < 5:
        return 0.0
    return float(np.median(series[:window]))


def severity_label(angle_abs_deg: float) -> str:
    if angle_abs_deg < 3.0:
        return "ниже порога"
    if angle_abs_deg < 10.0:
        return "лёгкая"
    if angle_abs_deg < 20.0:
        return "умеренная"
    return "тяжёлая"


def analyze_video(video_path: str, calibration: CalibrationMode = "off") -> PoseResult:
    feature_exe = os.environ.get("OPENFACE_CMD", "")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _run_openface(feature_exe, video_path, tmp)
        df, fps = _read_openface_csv(csv_path)
        s, e = _select_confident_segment(df)
        if e <= s:
            raise RuntimeError("Нет подходящего интервала с confidence ≥ 0.7")
        seg = df.iloc[s:e].reset_index(drop=True)
        pitch = seg["pitch"].values.copy()
        off = _calibrate_pitch(pitch, fps, calibration)
        pitch = pitch - off
        roll = seg["roll"].values
        yaw = seg["yaw"].values
        m_pitch = float(np.mean(pitch))
        m_roll = float(np.mean(roll))
        m_yaw = float(np.mean(yaw))
        involved = []
        if abs(m_pitch) > MAE["pitch"]:
            involved.append("pitch")
        if abs(m_roll) > MAE["roll"]:
            involved.append("roll")
        if abs(m_yaw) > MAE["yaw"]:
            involved.append("yaw")
        sev_axes = {
            "pitch": severity_label(abs(m_pitch)),
            "roll": severity_label(abs(m_roll)),
            "yaw": severity_label(abs(m_yaw)),
        }
        overall = severity_label(max(abs(m_pitch), abs(m_roll), abs(m_yaw)))
        cap = cv2.VideoCapture(video_path)
        br_samples = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % max(1, int(fps//2)) == 0:
                br_samples.append(estimate_brightness(frame))
        cap.release()
        bright = float(np.mean(br_samples)) if br_samples else 100.0
        qc = {
            "brightness_mean": round(bright, 2),
            "is_dark": is_dark(bright),
            "unstable_camera": estimate_instability(yaw),
            "head_scale_ok": None,
            "other_faces": "unknown",
            "segment_frames": int(len(seg)),
            "calibration": calibration,
        }
        return PoseResult(
            pitch_deg=m_pitch,
            roll_deg=m_roll,
            yaw_deg=m_yaw,
            involved_axes=involved,
            severity_axes=sev_axes,
            severity_overall=overall,
            qc=qc,
            fps=fps,
            n_frames=int(len(seg)),
        )


def demo_pose_result() -> PoseResult:
    fps = 30.0
    yaw = 5.0
    sev = severity_label(abs(yaw))
    qc = {
        "brightness_mean": 100.0,
        "is_dark": False,
        "unstable_camera": False,
        "head_scale_ok": None,
        "other_faces": "unknown",
        "segment_frames": 300,
        "calibration": "off",
        "demo": True,
    }
    return PoseResult(
        pitch_deg=0.0,
        roll_deg=0.0,
        yaw_deg=yaw,
        involved_axes=["yaw"],
        severity_axes={"pitch": severity_label(0.0), "roll": severity_label(0.0), "yaw": sev},
        severity_overall=sev,
        qc=qc,
        fps=fps,
        n_frames=300,
    )
