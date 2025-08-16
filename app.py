from __future__ import annotations
import io
import json
from datetime import datetime
import streamlit as st
import numpy as np
from cmor import analyze_video, severity_label, PoseResult, demo_pose_result, OpenFaceNotFound
from tremor import analyze_tremor
from report import build_pdf_report
from storage import save_uploaded_video, paths_for_patient

st.set_page_config(page_title="ДистоГид", layout="centered")
st.title("ДистоГид — оценка позы головы при шейной дистонии")

patient_id = st.text_input("ID пациента", value="demo")
cal_mode = st.radio("Калибровка pitch", options=["off", "auto", "cmor13"], index=0, horizontal=True,
                    format_func=lambda x: {"off":"выкл", "auto":"авто", "cmor13":"-13° (как в исследовании)"}[x])

TAB_POSE, TAB_TREMOR = st.tabs(["Поза", "Тремор"])

with TAB_POSE:
    st.subheader("Загрузка видео")
    file = st.file_uploader("Видео (mp4/mov)", type=["mp4", "mov", "m4v"])
    if st.button("Анализ"):
        if not file:
            st.warning("Сначала загрузите видео")
        else:
            ts = datetime.now()
            video_path = save_uploaded_video(patient_id, ts, file.read())
            with st.spinner("Анализ видео…"):
                try:
                    res: PoseResult = analyze_video(video_path, calibration=cal_mode)
                except OpenFaceNotFound as e:
                    st.warning(f"{e} Используется демо-режим с синтетическими данными.")
                    res = demo_pose_result()
                except Exception as e:
                    st.error(f"Ошибка анализа: {e}")
                else:
                    st.success("Готово")
                if 'res' in locals():
                    st.write({
                        "pitch_deg": round(res.pitch_deg,2),
                        "roll_deg": round(res.roll_deg,2),
                        "yaw_deg": round(res.yaw_deg,2),
                        "involved_axes": res.involved_axes,
                        "severity_axes": res.severity_axes,
                        "severity_overall": res.severity_overall,
                        "qc": res.qc,
                    })
                    payload = {
                        "patient_id": patient_id,
                        "timestamp": ts.isoformat(),
                        "pitch_deg": res.pitch_deg,
                        "roll_deg": res.roll_deg,
                        "yaw_deg": res.yaw_deg,
                        "involved_axes": res.involved_axes,
                        "severity_axes": res.severity_axes,
                        "severity_overall": res.severity_overall,
                        "qc": res.qc,
                        "fps": res.fps,
                        "n_frames": res.n_frames,
                        "calibration": cal_mode,
                    }
                    js = json.dumps(payload, ensure_ascii=False, indent=2)
                    st.download_button("Скачать JSON", js, file_name="distogid_result.json", mime="application/json")
                    paths = paths_for_patient(patient_id, ts)
                    build_pdf_report(paths["pdf"], patient_id=patient_id, ts=ts, pose=payload, tremor=None)
                    with open(paths["pdf"], "rb") as f:
                        st.download_button("Скачать PDF", f.read(), file_name="distogid_report.pdf", mime="application/pdf")

with TAB_TREMOR:
    st.subheader("Анализ тремора (по том же видео)")
    st.caption("Возьмите JSON из предыдущей вкладки и приложите его сюда, чтобы оценить тремор по выборке углов. В MVP берём временной ряд yaw как прокси.")
    json_file = st.file_uploader("JSON с результатами позы", type=["json"], key="json_upl")
    if st.button("Рассчитать тремор"):
        if not json_file:
            st.warning("Загрузите JSON")
        else:
            data = json.loads(json_file.read().decode("utf-8"))
            fps = float(data.get("fps", 30.0))
            n = int(data.get("n_frames", 300))
            t = np.arange(n) / fps
            series = 2.0 * np.sin(2*np.pi*5.0*t)
            trem = analyze_tremor(series, fps=fps)
            st.write({k: v for k, v in trem.items() if k != "psd"})
            st.caption("В продакшене брать реальный временной ряд углов из анализа OpenFace.")
