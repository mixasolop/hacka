"""Shared visual indicators for Streamlit screens."""

from __future__ import annotations

import streamlit as st


def status_color(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized in {"stable", "low", "mild", "high habitability"}:
        return "#2E8B57"
    if normalized in {"marginal", "elevated", "moderate"}:
        return "#D2A106"
    if normalized in {"cold", "stable cooling"}:
        return "#3B82C4"
    return "#B52A2A"


def kpi_card(label: str, value: str, status: str, *, font_size_px: int = 27, status_hex: str | None = None) -> None:
    color = status_hex or status_color(status)
    st.markdown(
        (
            "<div style='padding:10px 12px;border-radius:10px;border:1px solid #273249;"
            "background:#0f1626;margin-bottom:8px;'>"
            f"<div style='font-size:11px;letter-spacing:0.04em;text-transform:uppercase;color:#96a3bd;'>{label}</div>"
            f"<div style='font-size:{int(font_size_px)}px;font-weight:700;line-height:1.2;color:#f4f8ff;margin-top:4px;'>{value}</div>"
            f"<div style='margin-top:6px;font-size:12px;font-weight:600;color:{color};'>{status}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def badge(label: str, status: str) -> None:
    color = status_color(status)
    st.markdown(
        (
            "<div style='padding:10px 12px;border-radius:10px;border:1px solid #273249;"
            "margin-bottom:8px;background:#10192b;display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-weight:600;color:#d9e3f5;'>{label}</span>"
            f"<span style='color:{color};font-weight:700;border:1px solid {color};"
            f"padding:2px 8px;border-radius:999px;background:{color}22;'>{status}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def section_header(title: str, *, first: bool = False) -> None:
    if not first:
        st.markdown("<div style='height:1.3rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<h3 style='font-size:1.45rem;line-height:1.2;margin:0 0 0.6rem 0;'>{title}</h3>",
        unsafe_allow_html=True,
    )
