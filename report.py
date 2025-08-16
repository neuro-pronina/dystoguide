from __future__ import annotations
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from datetime import datetime


def build_pdf_report(path: str, *, patient_id: str, ts: datetime, pose: dict, tremor: dict | None = None):
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    y = h - 2*cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, "ДистоГид — Отчёт об оценке позы головы")
    y -= 0.8*cm
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, f"ID пациента: {patient_id}")
    y -= 0.5*cm
    c.drawString(2*cm, y, f"Дата/время: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 0.8*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Поза")
    y -= 0.6*cm
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, f"Pitch: {pose['pitch_deg']:.2f}°  |  Roll: {pose['roll_deg']:.2f}°  |  Yaw: {pose['yaw_deg']:.2f}°")
    y -= 0.5*cm
    c.drawString(2*cm, y, f"Вовлечённые оси: {', '.join(pose['involved_axes']) if pose['involved_axes'] else 'нет'}")
    y -= 0.5*cm
    c.drawString(2*cm, y, f"Тяжесть по осям: {pose['severity_axes']}")
    y -= 0.5*cm
    c.drawString(2*cm, y, f"Общая ориентировочная тяжесть: {pose['severity_overall']}")
    y -= 0.5*cm
    c.drawString(2*cm, y, f"QC: dark={pose['qc']['is_dark']}, unstable={pose['qc']['unstable_camera']}, seg_frames={pose['qc']['segment_frames']}")
    y -= 0.8*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Тремор")
    y -= 0.6*cm
    c.setFont("Helvetica", 10)
    if tremor and tremor.get("peak_hz") is not None:
        c.drawString(2*cm, y, f"Пиковая частота: {tremor['peak_hz']:.2f} Гц; RMS: {tremor['rms_deg']:.2f}°")
    else:
        c.drawString(2*cm, y, "Недостаточно данных или не вычислено")
    y -= 1.2*cm
    c.setFont("Helvetica", 9)
    c.drawString(2*cm, y, "Дисклеймер: инструмент объективизации. Не заменяет клинические шкалы и диагноз.")
    c.showPage()
    c.save()
