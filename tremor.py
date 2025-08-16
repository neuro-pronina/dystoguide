from __future__ import annotations
import numpy as np
from scipy.signal import welch


def analyze_tremor(series_deg: np.ndarray, fps: float, fmin: float = 2.0, fmax: float = 12.0):
    if len(series_deg) < 32:
        return {
            "peak_hz": None,
            "rms_deg": float(np.sqrt(np.mean(series_deg**2))) if len(series_deg) else 0.0,
            "fmin": fmin, "fmax": fmax,
            "psd": ([], [])
        }
    nperseg = min(256, max(64, len(series_deg)//4))
    f, Pxx = welch(series_deg, fs=fps, nperseg=nperseg)
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        peak_hz = None
    else:
        idx = np.argmax(Pxx[band])
        peak_hz = float(f[band][idx])
    rms = float(np.sqrt(np.mean(series_deg**2)))
    return {"peak_hz": peak_hz, "rms_deg": rms, "fmin": fmin, "fmax": fmax, "psd": (f.tolist(), Pxx.tolist())}
