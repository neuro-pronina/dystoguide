from __future__ import annotations
import os
import glob
import numpy as np
import cv2

DEG = 180.0 / np.pi

def find_first_csv(folder: str) -> str | None:
    candidates = sorted(glob.glob(os.path.join(folder, "*.csv")))
    return candidates[0] if candidates else None

def longest_true_segment(mask: np.ndarray) -> tuple[int, int]:
    """Возвращает (start, end_exclusive) для самого длинного интервала True в 1D-массиве.
    Если True нет — вернуть (0, 0).
    """
    best_len = 0
    best = (0, 0)
    start = None
    for i, v in enumerate(mask.astype(bool)):
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - start > best_len:
                best_len = i - start
                best = (start, i)
            start = None
    if start is not None:
        i = len(mask)
        if i - start > best_len:
            best = (start, i)
    return best

def estimate_brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())

def is_dark(mean_brightness: float, threshold: float = 60.0) -> bool:
    return mean_brightness < threshold

def estimate_instability(motion_series: np.ndarray, thresh: float = 1.5) -> bool:
    """Грубая оценка нестабильности: если медианная |Δ| по врем.ряду угла велика."""
    if len(motion_series) < 2:
        return False
    diffs = np.abs(np.diff(motion_series))
    return np.median(diffs) > thresh

def head_scale_ok(width_px: int, head_box_w_px: int | None, min_ratio: float = 0.1) -> bool | None:
    if head_box_w_px is None:
        return None
    return (head_box_w_px / max(1, width_px)) >= min_ratio
