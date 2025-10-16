from __future__ import annotations

import html
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib_common import (
    header_ui,
    get_honorarios_df,
    clean_estado_cuota,
    money,
    sanitize_df,
    safe_markdown,
    style_table,
)
from lib_report import excel_bytes_single, generate_pdf_report

st.set_page_config(
    page_title="Honorarios",
    layout="wide",
    initial_sidebar_state="collapsed",
)
header_ui(
    "Honorarios",
    current_page="Honorarios",
    subtitle="Indicadores financieros y comportamiento de pago de honorarios.",
    nav_active="honorarios",
)

safe_markdown(
    """
    <style>
    .cm-coachmark {position: relative; z-index: 5;}
    .cm-modal-checkbox {display: none;}
    .cm-fab {
        position: fixed;
        bottom: 1.5rem;
        right: 1.5rem;
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.85rem 1.2rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #0f4c81, #0a2a4a);
        color: #f4f8ff;
        font-weight: 600;
        letter-spacing: 0.2px;
        box-shadow: 0 12px 32px rgba(8, 27, 51, 0.35);
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-decoration: none;
        font-family: inherit;
    }
    .cm-fab:hover {transform: translateY(-2px); box-shadow: 0 16px 36px rgba(8, 27, 51, 0.4);}
    .cm-fab:active {transform: translateY(0); box-shadow: 0 8px 20px rgba(8, 27, 51, 0.3);}
    .cm-fab-icon {
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .cm-fab-text {white-space: nowrap;}
    .cm-modal-overlay {
        position: fixed;
        inset: 0;
        background: rgba(7, 19, 33, 0.58);
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.25s ease;
        z-index: 30;
    }
    .cm-modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -42%) scale(0.96);
        width: min(520px, calc(100vw - 2.5rem));
        max-height: calc(100vh - 4rem);
        overflow-y: auto;
        background: #0e1d33;
        color: #f2f6ff;
        border-radius: 18px;
        padding: 1.8rem 2rem;
        box-shadow: 0 28px 48px rgba(2, 12, 26, 0.55);
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.25s ease, transform 0.25s ease;
        z-index: 40;
        font-family: inherit;
    }
    .cm-modal-checkbox:checked ~ .cm-modal-overlay {opacity: 1; pointer-events: auto;}
    .cm-modal-checkbox:checked ~ .cm-modal {opacity: 1; pointer-events: auto; transform: translate(-50%, -50%) scale(1);}
    .cm-modal__header {display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; margin-bottom: 1.25rem;}
    .cm-modal__title {margin: 0; font-size: 1.25rem; font-weight: 700; line-height: 1.35;}
    .cm-modal__close {
        cursor: pointer;
        background: rgba(255, 255, 255, 0.08);
        color: #f7fbff;
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: background 0.2s ease, color 0.2s ease;
        text-transform: uppercase;
    }
    .cm-modal__close:hover {background: rgba(255, 255, 255, 0.14); color: #fff;}
    .cm-modal__section {margin-bottom: 1.15rem;}
    .cm-modal__section-title {font-size: 0.95rem; font-weight: 700; margin-bottom: 0.45rem; text-transform: uppercase; letter-spacing: 0.3px; color: #9fb9d9;}
    .cm-modal__summary {font-size: 0.95rem; line-height: 1.55; margin: 0; color: #e5eeff;}
    .cm-steps {margin: 0.65rem 0 0; padding-left: 1.2rem; font-size: 0.95rem; line-height: 1.55; color: #d7e6ff; display: grid; gap: 0.55rem;}
    .cm-steps li {margin-left: 0.2rem;}
    .cm-route {display: block; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(157, 193, 237, 0.28); border-radius: 12px; padding: 0.85rem 1rem; font-size: 0.9rem; color: #cfe2ff; word-break: break-word;}
    .cm-highlight {display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.25rem 0.55rem; border: 2px dotted rgba(255, 214, 102, 0.9); border-radius: 10px; background: rgba(255, 214, 102, 0.12); color: #ffe7a1; font-weight: 700;}
    .cm-highlight-tag {font-size: 0.85rem; font-weight: 700; color: #ffefc1;}
    .cm-note {font-size: 0.85rem; color: #9fb9d9; margin-top: 0.6rem;}
    @media (max-width: 600px) {
        .cm-fab {bottom: 1rem; right: 1rem; padding: 0.75rem 1rem; gap: 0.45rem;}
        .cm-fab-text {font-size: 0.9rem;}
        .cm-modal {padding: 1.4rem 1.35rem; border-radius: 16px;}
        .cm-modal__title {font-size: 1.1rem;}
    }
    </style>
    <div class="cm-coachmark">
        <input type="checkbox" id="cm-honorarios-toggle" class="cm-modal-checkbox">
        <label for="cm-honorarios-toggle" class="cm-fab">
            <span class="cm-fab-icon">?</span>
            <span class="cm-fab-text">¿Dónde descargo?</span>
        </label>
        <label for="cm-honorarios-toggle" class="cm-modal-overlay"></label>
        <div class="cm-modal" role="dialog" aria-labelledby="cm-honorarios-title">
            <div class="cm-modal__header">
                <h2 class="cm-modal__title" id="cm-honorarios-title">Ruta para consultar y descargar información de Honorarios</h2>
                <label for="cm-honorarios-toggle" class="cm-modal__close">Cerrar</label>
            </div>
            <div class="cm-modal__section">
                <div class="cm-modal__section-title">Resumen</div>
                <p class="cm-modal__summary">Para consultar y descargar datos de honorarios, sigue esta ruta en el sistema de Contabilidad.</p>
            </div>
            <div class="cm-modal__section">
                <div class="cm-modal__section-title">Pasos</div>
                <ol class="cm-steps">
                    <li>Abre Contabilidad.</li>
                    <li>En la barra superior, entra a Honorarios.</li>
                    <li>Selecciona Convenio Honorarios.</li>
                    <li>Elige Consultas Honorarios.</li>
                    <li>Haz clic en <span class="cm-highlight">Cuotas Honorarios <span class="cm-highlight-tag">⬅ Aquí descargas</span></span> (aquí realizas la consulta y descarga).</li>
                </ol>
            </div>
            <div class="cm-modal__section">
                <div class="cm-modal__section-title">Árbol</div>
                <span class="cm-route">Contabilidad ▸ Honorarios ▸ Convenio Honorarios ▸ Consultas Honorarios ▸ Cuotas Honorarios</span>
            </div>
            <p class="cm-note">Tip: esta es la ruta estándar para consultar cuotas y descargar la información de honorarios.</p>
        </div>
    </div>
    """
)

df_hon = get_honorarios_df()
if df_hon is None or df_hon.empty:
    st.info("Cargue honorarios en 'Carga de Data' para analizar.")
    st.stop()

if "estado_cuota" not in df_hon.columns:
    st.warning("La base de honorarios cargada no incluye la columna estado_cuota.")
    st.stop()

df = clean_estado_cuota(df_hon.copy())
if df is None or df.empty:
    st.warning("No hay honorarios validos despues de limpiar estado_cuota.")
    st.stop()

_df = sanitize_df(df)
df = _df if isinstance(_df, pd.DataFrame) else df

df["estado_cuota"] = df["estado_cuota"].astype(str)

fecha_cols = {
    "emision": ["fecha_emision", "fac_fecha_factura"],
    "cuota": ["fecha_cuota", "fecha_cc"],
    "pago": ["fecha_ce", "fecha_pagado"],
}

def _pick_col(df_in: pd.DataFrame, options: list[str]) -> Optional[str]:
    for col in options:
        if col in df_in.columns:
            return col
    return None

fecha_emision_col = _pick_col(df, fecha_cols["emision"])
fecha_cuota_col = _pick_col(df, fecha_cols["cuota"])
fecha_pago_col = _pick_col(df, fecha_cols["pago"])

if fecha_emision_col:
    df["fecha_emision_ref"] = pd.to_datetime(df[fecha_emision_col], errors="coerce")
else:
    df["fecha_emision_ref"] = pd.NaT
if fecha_cuota_col:
    df["fecha_cuota_ref"] = pd.to_datetime(df[fecha_cuota_col], errors="coerce")
else:
    df["fecha_cuota_ref"] = pd.NaT
if fecha_pago_col:
    df["fecha_pago_ref"] = pd.to_datetime(df[fecha_pago_col], errors="coerce")
else:
    df["fecha_pago_ref"] = pd.NaT

pagadas = df[df["estado_cuota"] == "PAGADA"].copy()
no_pagadas = df[df["estado_cuota"] != "PAGADA"].copy()

def _sum_numeric(df_in: pd.DataFrame, columns: list[str]) -> float:
    for col in columns:
        if col in df_in.columns:
            return pd.to_numeric(df_in[col], errors="coerce").fillna(0).sum()
    return 0.0

monto_pagado_total = _sum_numeric(pagadas, ["monto_pagado", "liquido_cuota", "fac_monto_total"])
monto_no_pagado_total = _sum_numeric(no_pagadas, ["monto_autorizado", "monto_cuota", "fac_monto_total"])

total_docs = len(df)
count_pagadas = len(pagadas)
count_no_pagadas = len(no_pagadas)

mask_pago_valido = pagadas["fecha_pago_ref"].notna() & pagadas["fecha_cuota_ref"].notna() if count_pagadas else pd.Series(dtype=bool)
especial_condition = (pagadas["fecha_pago_ref"] < pagadas["fecha_emision_ref"]) | (
    (pagadas["fecha_pago_ref"] < pagadas["fecha_cuota_ref"]) & (pagadas["fecha_cuota_ref"] == pagadas["fecha_emision_ref"])
)
mask_atraso = mask_pago_valido & (pagadas["fecha_pago_ref"] > pagadas["fecha_cuota_ref"]) if count_pagadas else pd.Series(dtype=bool)
mask_dentro_plazo = mask_pago_valido & ~mask_atraso if count_pagadas else pd.Series(dtype=bool)
mask_same_day = mask_dentro_plazo & (pagadas["fecha_pago_ref"] == pagadas["fecha_cuota_ref"]) if count_pagadas else pd.Series(dtype=bool)
mask_especial = mask_pago_valido & especial_condition if count_pagadas else pd.Series(dtype=bool)
mask_especial_plazo = mask_especial & mask_dentro_plazo if count_pagadas else pd.Series(dtype=bool)
mask_anticipada_regular = mask_dentro_plazo & (pagadas["fecha_pago_ref"] < pagadas["fecha_cuota_ref"]) & ~especial_condition if count_pagadas else pd.Series(dtype=bool)

count_atraso = int(mask_atraso.sum()) if count_pagadas else 0
count_same_day = int(mask_same_day.sum()) if count_pagadas else 0
count_anticipada = int(mask_anticipada_regular.sum()) if count_pagadas else 0
count_especial = int(mask_especial_plazo.sum()) if count_pagadas else 0
count_en_plazo = int(mask_dentro_plazo.sum()) if count_pagadas else 0
count_sin_info = max(0, count_pagadas - (count_en_plazo + count_atraso)) if count_pagadas else 0

pct_pagadas_total = (count_pagadas / total_docs * 100.0) if total_docs else 0.0
pct_en_plazo = (count_en_plazo / count_pagadas * 100.0) if count_pagadas else 0.0
pct_atraso = (count_atraso / count_pagadas * 100.0) if count_pagadas else 0.0
pct_no_pagadas_total = (count_no_pagadas / total_docs * 100.0) if total_docs else 0.0

st.subheader("Resumen de honorarios")

safe_markdown(
    """
    <style>
    .honorarios-metric-card {
        position: relative;
        overflow: hidden;
        padding: 1.4rem 1.6rem;
        border-radius: var(--radius-md, 14px);
        background: linear-gradient(140deg, rgba(19, 37, 66, 0.92), rgba(11, 26, 48, 0.88));
        border: 1px solid rgba(79, 156, 255, 0.28);
        box-shadow: 0 24px 46px rgba(6, 17, 35, 0.45);
    }

    .honorarios-metric-card::before {
        content: "";
        position: absolute;
        inset: -60% 40% auto -40%;
        height: 180%;
        background: radial-gradient(circle at top, rgba(79, 156, 255, 0.38), transparent 65%);
        opacity: 0.65;
        pointer-events: none;
    }

    .honorarios-metric-card__title {
        position: relative;
        font-size: 0.82rem;
        letter-spacing: 0.45px;
        text-transform: uppercase;
        color: var(--app-text-muted, #9db4d5);
        margin: 0;
    }

    .honorarios-metric-card__value {
        position: relative;
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--app-text, #e6f1ff);
        margin: 0.55rem 0 0;
    }

    .honorarios-metric-card__footer {
        position: relative;
        margin-top: 0.55rem;
        font-size: 0.95rem;
        color: var(--app-text-muted, #9db4d5);
    }

    .honorarios-metric-card__breakdown {
        position: relative;
        margin-top: 0.75rem;
        font-size: 0.9rem;
        line-height: 1.45;
        color: var(--app-text-muted, #9db4d5);
    }

    .honorarios-metric-card__breakdown strong {
        color: var(--app-text, #e6f1ff);
        font-weight: 600;
    }
    </style>
    """,
)

ce_available = "cuenta_especial" in df.columns
ce_flag_all = df["cuenta_especial"].fillna(False).astype(bool) if ce_available else None

CE_COLORS = ["#4E79A7", "#A0A0A0"]

def _counts_html(ce_val: int, no_val: int) -> str:
    return (
        "<div class='honorarios-metric-card__breakdown'>"
        f"<span>Cuenta especial: <strong>{ce_val:,}</strong></span><br>"
        f"<span>No cuenta especial: <strong>{no_val:,}</strong></span>"
        "</div>"
    )

def _amount_html(ce_val: float, no_val: float) -> str:
    return (
        "<div class='honorarios-metric-card__breakdown'>"
        f"<span>Cuenta especial: <strong>{money(ce_val)}</strong></span><br>"
        f"<span>No cuenta especial: <strong>{money(no_val)}</strong></span>"
        "</div>"
    )

def _render_metric_block(
    title: str,
    main_value: str,
    footer: Optional[str] = None,
    breakdown_html: Optional[str] = None,
):
    parts: list[str] = [
        "<div class='honorarios-metric-card'>",
        f"<div class='honorarios-metric-card__title'>{title}</div>",
        f"<div class='honorarios-metric-card__value'>{main_value}</div>",
    ]
    if footer:
        parts.append(f"<div class='honorarios-metric-card__footer'>{footer}</div>")
    if breakdown_html:
        parts.append(breakdown_html)
    parts.append("</div>")
    safe_markdown("".join(parts))


# =========================================================
# Utilidades compartidas para deuda / presupuesto
# =========================================================
TODAY = pd.Timestamp(date.today()).normalize()

TABLE_HEADER_BG = "var(--app-primary)"
TABLE_HEADER_FG = "var(--app-table-header-fg)"
TABLE_STRIPED_BG = "#f2f5ff"
TABLE_HOVER_BG = "#e0e8ff"
TABLE_FONT_SIZE = "var(--app-table-font-size)"
TABLE_ROW_PADDING = "16px 22px"

_BUDGET_PANEL_STYLE = """
<style>
.budget-panel {
    background: var(--app-surface, #ffffff);
    border-radius: 18px;
    padding: 1.75rem 1.75rem 1.4rem;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(64, 86, 179, 0.18);
    box-shadow: 0 18px 45px rgba(26, 43, 90, 0.12);
}
.budget-panel__title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #1f2a55;
    letter-spacing: 0.01em;
}
.budget-panel__subtitle {
    font-size: 0.95rem;
    color: #5b6b95;
    margin-top: 0.35rem;
}
.budget-panel__input-wrapper {
    margin-top: 1.25rem;
}
.budget-panel input[type="text"] {
    font-size: 1.7rem !important;
    font-weight: 600;
    padding: 0.95rem 1.1rem !important;
    border-radius: 16px !important;
    border: 1px solid #c5ceff !important;
    color: #1f2a55 !important;
    box-shadow: inset 0 1px 2px rgba(19, 34, 88, 0.08) !important;
}
.budget-panel__resume {
    background: linear-gradient(135deg, #f4f7ff 0%, #ecf1ff 100%);
    border-radius: 16px;
    padding: 1rem 1.4rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.35rem;
    text-align: right;
}
.budget-panel__resume-label {
    font-size: 0.85rem;
    color: #4a5a82;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.budget-panel__resume-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1f2a55;
}
.budget-panel__resume-note {
    font-size: 0.78rem;
    color: #3f4d78;
    margin: 0.35rem 0 0;
    text-align: right;
}
.budget-panel__resume-note--muted {
    color: #6c7ba6;
}
.budget-panel__helper {
    font-size: 0.85rem;
    color: #667399;
    margin-top: 0.6rem;
}
@media (max-width: 992px) {
    .budget-panel__resume {
        margin-top: 1rem;
        text-align: left;
    }
}
</style>
"""


def _table_style(df_disp: pd.DataFrame):
    sty = df_disp.style if isinstance(df_disp, pd.DataFrame) else df_disp
    sty = sty.hide(axis="index")
    sty = sty.set_table_styles(
        [
            {
                "selector": "thead tr",
                "props": [
                    ("background-color", TABLE_HEADER_BG),
                    ("color", TABLE_HEADER_FG),
                    ("font-weight", "bold"),
                    ("font-size", TABLE_FONT_SIZE),
                    ("text-align", "center"),
                    ("border-radius", "12px 12px 0 0"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", "transparent"),
                    ("color", TABLE_HEADER_FG),
                    ("font-weight", "600"),
                    ("font-size", TABLE_FONT_SIZE),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "0.6px"),
                    ("padding", TABLE_ROW_PADDING),
                    ("text-align", "center"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("font-size", TABLE_FONT_SIZE),
                    ("padding", TABLE_ROW_PADDING),
                    ("text-align", "right"),
                    ("border-bottom", "1px solid #d9e1ff"),
                    ("color", "var(--app-table-body-fg)"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", TABLE_STRIPED_BG)],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", TABLE_HOVER_BG)],
            },
            {
                "selector": "tbody td:first-child",
                "props": [
                    ("text-align", "left"),
                    ("font-weight", "600"),
                    ("color", "#1f2a55"),
                ],
            },
        ],
        overwrite=False,
    )
    return sty


def _fmt_count(n: int) -> str:
    return f"{n:,} doc."


def _fmt_days(val: float) -> str:
    return "s/d" if pd.isna(val) else f"{val:,.1f} d".replace(",", "X").replace(".", ",").replace("X", ".")


def _card_html(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    *,
    tag: Optional[str] = None,
    tag_variant: str = "success",
    tone: str = "default",
    stats: Optional[List[Tuple[str, str]]] = None,
    compact: bool = True,
    tooltip: Optional[str] = None,
) -> str:
    classes = ["app-card", "app-card--frost"]
    if compact:
        classes.append("app-card__mini")
    if tone == "accent":
        classes.append("app-card--accent")
    tooltip_attr = f' title="{html.escape(str(tooltip))}"' if tooltip else ""
    title_html = html.escape(str(title))
    value_html = html.escape(str(value))
    subtitle_html = ""
    if subtitle:
        subtitle_html = html.escape(str(subtitle)).replace("\n", "<br>")
        subtitle_html = f'<p class="app-card__subtitle">{subtitle_html}</p>'
    tag_html = ""
    if tag:
        tag_cls = "app-card__tag"
        if tag_variant == "warning":
            tag_cls += " app-card__tag--warning"
        tag_html = f'<span class="{tag_cls}">{html.escape(str(tag))}</span>'
    stats_html = ""
    if stats:
        pills = "".join(
            f'<div class="app-inline-stats__item"><span class="app-inline-stats__label">{html.escape(str(label))}:</span> {html.escape(str(val))}</div>'
            for label, val in stats
        )
        stats_html = f'<div class="app-inline-stats">{pills}</div>'
    return (
        f'<div class="{" ".join(classes)}"{tooltip_attr}>'
        f'<div class="app-card__title">{title_html}</div>'
        f'<div class="app-card__value">{value_html}</div>'
        f'{subtitle_html}'
        f'{tag_html}'
        f'{stats_html}'
        '</div>'
    )


def _render_cards(cards: list[str], layout: str = "grid"):
    if not cards:
        return
    wrapper = {
        "grid": "app-card-grid",
        "grid-2": "app-grid-2",
        "grid-3": "app-grid-3",
    }.get(layout, "app-card-grid")
    safe_markdown(f'<div class="{wrapper}">{"".join(cards)}</div>')


def _format_currency_plain(val: float) -> str:
    try:
        return money(float(val)).replace("$", "").strip()
    except Exception:
        return str(val)


def _parse_currency_input(text: str, fallback: float) -> float:
    if not isinstance(text, str):
        return float(fallback)
    cleaned = text.strip()
    if not cleaned:
        return float(fallback)
    cleaned = cleaned.replace("$", "").replace(" ", "")
    cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return float(fallback)


def _avg_days_from_series(series: Optional[pd.Series]) -> float:
    if series is None:
        return float("nan")
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[clean.notna()]
    if clean.empty:
        return float("nan")
    return float(np.round(clean.mean(), 1))


def _ensure_importe_deuda_hon(dfin: pd.DataFrame) -> pd.DataFrame:
    d = dfin.copy()
    if "importe_deuda" in d.columns:
        d["importe_deuda"] = pd.to_numeric(d["importe_deuda"], errors="coerce").fillna(0.0)
    else:
        monto_cuota = pd.to_numeric(d.get("monto_cuota"), errors="coerce")
        base = monto_cuota
        if base is None or base.dropna().empty:
            base = pd.to_numeric(d.get("dcu_monto"), errors="coerce")
        if base is None or base.dropna().empty:
            base = pd.to_numeric(d.get("monto_autorizado"), errors="coerce")
        fallback = pd.to_numeric(d.get("fac_monto_total"), errors="coerce")
        if base is None:
            base = fallback
        if fallback is not None and base is not None:
            base = base.fillna(fallback)
        d["importe_deuda"] = base.fillna(0.0) if base is not None else 0.0

    if "monto_cuota" in d.columns:
        d["monto_cuota"] = pd.to_numeric(d["monto_cuota"], errors="coerce").fillna(0.0)
    else:
        d["monto_cuota"] = d["importe_deuda"].copy()
    return d


def _ensure_dias_a_vencer_hon(dfin: pd.DataFrame) -> pd.DataFrame:
    d = dfin.copy()
    if "fecha_venc_30" not in d.columns:
        d["fecha_venc_30"] = d.get("fecha_cuota_ref")
    d["fecha_venc_30"] = pd.to_datetime(d["fecha_venc_30"], errors="coerce")
    if "dias_a_vencer" not in d.columns:
        d["dias_a_vencer"] = (d["fecha_venc_30"] - TODAY).dt.days
    return d


def _classify_nivel_hon(estado: str) -> str:
    estado_norm = (estado or "").upper()
    if "PAGADA" in estado_norm:
        return "Pagada"
    if any(token in estado_norm for token in ("AUTORIZ", "CONTABIL")):
        return "Contabilizado Pendiente de Pago"
    return "Pendiente de Contabilización"


def _agg_block(d: pd.DataFrame, mask: pd.Series) -> Tuple[float, int]:
    sub = d.loc[mask] if isinstance(mask, pd.Series) else d
    monto = float(pd.to_numeric(sub.get("importe_deuda"), errors="coerce").fillna(0.0).sum())
    cant = int(len(sub))
    return monto, cant


def _kpis_deuda(dfin: pd.DataFrame) -> dict:
    if dfin is None or dfin.empty or "dias_a_vencer" not in dfin:
        return dict(vencido=0.0, c_venc=0, hoy=0.0, c_hoy=0, por_ven=0.0, c_por=0)
    dias = pd.to_numeric(dfin["dias_a_vencer"], errors="coerce")
    vencido, c_v = _agg_block(dfin, dias < 0)
    hoy_m, c_h = _agg_block(dfin, dias == 0)
    por_v, c_p = _agg_block(dfin, dias > 0)
    return dict(vencido=vencido, c_venc=c_v, hoy=hoy_m, c_hoy=c_h, por_ven=por_v, c_por=c_p)


def _deuda_detalle_metrics(dfin: pd.DataFrame, *, panel: str) -> dict[str, float]:
    if dfin is None or dfin.empty:
        return {
            "total_monto": 0.0,
            "total_docs": 0,
            "dias_prom": float("nan"),
        }

    d = dfin.copy()
    d["importe_deuda_num"] = pd.to_numeric(d.get("importe_deuda"), errors="coerce").fillna(0.0)
    total_monto = float(d["importe_deuda_num"].sum())
    total_docs = int(len(d))

    if panel == "Contabilizado Pendiente de Pago":
        if "fecha_cuota_ref" in d:
            dias = (TODAY - d["fecha_cuota_ref"]).dt.days
        else:
            dias = pd.Series(dtype=float)
    else:
        if "fecha_emision_ref" in d:
            dias = (TODAY - d["fecha_emision_ref"]).dt.days
        else:
            dias = pd.Series(dtype=float)

    dias_prom = _avg_days_from_series(dias)
    return {"total_monto": total_monto, "total_docs": total_docs, "dias_prom": dias_prom}


def _build_panel_stats(panel: str, detalle: dict, kpis: dict) -> List[Tuple[str, str]]:
    base_stats: List[Tuple[str, str]] = []
    if panel == "Contabilizado Pendiente de Pago":
        base_stats.append(
            (
                "Monto contabilizado",
                f"{money(detalle['total_monto'])} • {_fmt_count(detalle['total_docs'])}",
            )
        )
        base_stats.append(("Días prom. sin pagar", _fmt_days(detalle["dias_prom"])))
    else:
        base_stats.append(
            (
                "Monto pendiente de contabilización",
                f"{money(detalle['total_monto'])} • {_fmt_count(detalle['total_docs'])}",
            )
        )
        base_stats.append(("Días prom. sin contabilizar", _fmt_days(detalle["dias_prom"])))

    base_stats.extend(
        [
            ("Vencida", f"{money(kpis['vencido'])} • {_fmt_count(kpis['c_venc'])}"),
            ("Hoy", f"{money(kpis['hoy'])} • {_fmt_count(kpis['c_hoy'])}"),
            ("Por vencer", f"{money(kpis['por_ven'])} • {_fmt_count(kpis['c_por'])}"),
        ]
    )
    return base_stats


def _draw_debt_panel(title: str, panel_df: pd.DataFrame):
    safe_markdown(
        '<div class="app-title-block"><h3 style="color:#000;">'
        + html.escape(title)
        + '</h3><p>Desglose por cuenta especial</p></div>'
    )
    if "cuenta_especial" not in panel_df.columns:
        st.info("Sin campo de Cuenta Especial para desglosar.")
        return

    cards: list[str] = []
    for flag in (True, False):
        sub = panel_df[panel_df["cuenta_especial"] == flag]
        detalle = _deuda_detalle_metrics(sub, panel=title)
        kpis = _kpis_deuda(sub)
        cards.append(
            _card_html(
                title=f"CE {'Sí' if flag else 'No'}",
                value=f"Total pendiente: {money(detalle['total_monto'])}",
                subtitle=f"Cantidad total pendiente: {detalle['total_docs']:,} doc.",
                stats=_build_panel_stats(title, detalle, kpis),
                tone="accent" if flag else "default",
                compact=False,
            )
        )
    _render_cards(cards, layout="grid-2")


_LOCAL_FILTER_STATE_KEY = "hon_presupuesto_filters"
_LOCAL_AMOUNT_KEY = "hon_presupuesto_hoy"


def _compute_presupuesto_selection(
    prior: pd.DataFrame, presupuesto: float
) -> Tuple[pd.DataFrame, float, float, Optional[dict], dict]:
    presupuesto_val = float(presupuesto)

    if prior is None or prior.empty:
        empty_info = {
            "count": 0,
            "remaining_amount": 0.0,
            "full_budget": 0.0,
            "missing_budget": 0.0,
            "selected_count": 0,
        }
        return pd.DataFrame(), 0.0, presupuesto_val, None, empty_info

    tmp = prior.copy()
    if "importe_regla" not in tmp:
        tmp["importe_regla"] = pd.to_numeric(tmp.get("importe_deuda"), errors="coerce").fillna(0.0)
    tmp["importe_regla_num"] = pd.to_numeric(tmp.get("importe_regla"), errors="coerce").fillna(0.0)
    tmp["acum"] = tmp["importe_regla_num"].cumsum()

    tolerance = 1e-6
    selected_mask = tmp["acum"] <= presupuesto_val + tolerance
    seleccion = tmp[selected_mask].drop(columns=["acum", "importe_regla_num"], errors="ignore")
    suma_sel = float(tmp.loc[selected_mask, "importe_regla_num"].sum())

    restante_raw = presupuesto_val - suma_sel
    restante = max(0.0, restante_raw)

    pendientes_mask = ~selected_mask
    pendientes = tmp[pendientes_mask]
    siguiente = pendientes.head(1)
    next_info = None
    if not siguiente.empty:
        row = siguiente.iloc[0]
        next_amount = float(row.get("importe_regla_num", 0.0))
        prov_val = row.get("prr_razon_social") or row.get("Proveedor") or row.get("nombre_profesional")
        prov = str(prov_val) if pd.notna(prov_val) else "Proveedor sin identificar"
        doc_val = row.get("fac_numero") or row.get("numero_documento") or row.get("doc_id")
        doc = str(doc_val) if pd.notna(doc_val) else "s/d"
        adicional = max(0.0, next_amount - restante)
        next_info = {
            "monto": next_amount,
            "proveedor": prov,
            "documento": doc,
            "adicional": adicional,
        }

    remaining_amount = float(tmp.loc[pendientes_mask, "importe_regla_num"].sum())
    full_budget = float(tmp["acum"].max()) if not tmp.empty else 0.0
    missing_budget = max(0.0, full_budget - presupuesto_val)
    extra_info = {
        "count": int(pendientes_mask.sum()),
        "remaining_amount": remaining_amount,
        "full_budget": full_budget,
        "missing_budget": missing_budget,
        "selected_count": int(selected_mask.sum()),
    }

    return seleccion, suma_sel, restante, next_info, extra_info


def _update_presupuesto_session_state(filters: dict[str, str], default_value: float) -> None:
    stored_filters = st.session_state.get(_LOCAL_FILTER_STATE_KEY)
    text_key = f"{_LOCAL_AMOUNT_KEY}__text"

    if stored_filters is None:
        st.session_state[_LOCAL_FILTER_STATE_KEY] = filters
        st.session_state[_LOCAL_AMOUNT_KEY] = float(default_value)
        st.session_state[text_key] = _format_currency_plain(default_value)
        return

    if stored_filters != filters:
        st.session_state[_LOCAL_FILTER_STATE_KEY] = filters
        st.session_state[_LOCAL_AMOUNT_KEY] = float(default_value)
        st.session_state[text_key] = _format_currency_plain(default_value)


def _apply_local_filters(dfin: pd.DataFrame, *, ce_filter: str, prio_filter: str) -> pd.DataFrame:
    out = dfin.copy()
    if ce_filter == "Cuenta Especial" and "cuenta_especial" in out:
        out = out[out["cuenta_especial"] == True]
    elif ce_filter == "No Cuenta Especial" and "cuenta_especial" in out:
        out = out[out["cuenta_especial"] == False]
    if prio_filter == "Prioritario" and "prov_prioritario" in out:
        out = out[out["prov_prioritario"] == True]
    elif prio_filter == "No Prioritario" and "prov_prioritario" in out:
        out = out[out["prov_prioritario"] == False]
    return out


def _prioritize_documents(dfin: pd.DataFrame, criterio: str) -> pd.DataFrame:
    out = dfin.copy()
    if out.empty:
        return out

    if "dias_a_vencer" in out:
        dias = pd.to_numeric(out["dias_a_vencer"], errors="coerce")
        out["dias_a_vencer"] = dias
        out["vencida_flag"] = dias < 0
    else:
        out["dias_a_vencer"] = np.nan
        out["vencida_flag"] = False

    importe = pd.to_numeric(out.get("importe_deuda"), errors="coerce").fillna(0.0)
    out["importe_regla"] = importe

    if "Nivel" in out:
        out["_nivel_rank"] = out["Nivel"].map({
            "Pendiente de Contabilización": 0,
            "Contabilizado Pendiente de Pago": 1,
        }).fillna(2)
    else:
        out["_nivel_rank"] = 2

    if criterio == "Riesgo de aprobación":
        sort_cols = ["_nivel_rank", "vencida_flag", "dias_a_vencer", "importe_regla"]
        sort_order = [True, False, True, False]
    else:
        sort_cols = ["vencida_flag", "dias_a_vencer", "importe_regla"]
        sort_order = [False, True, False]

    existing_cols = [c for c in sort_cols if c in out.columns]
    if existing_cols:
        asc = [sort_order[sort_cols.index(c)] for c in existing_cols]
        out = out.sort_values(by=existing_cols, ascending=asc)

    return out


def _apply_horizon_filter(dfin: pd.DataFrame, horizonte: float) -> pd.DataFrame:
    if "dias_a_vencer" not in dfin:
        return dfin.copy()
    dias = pd.to_numeric(dfin["dias_a_vencer"], errors="coerce")
    return dfin[dias <= horizonte].copy()


BASE_KEEP_COLS = [
    "rut",
    "cnv_fecha_inicio",
    "cnv_fecha_termino",
    "dcu_correlativo",
    "cnv_cuotas",
    "fecha_emision",
    "fecha_cuota",
    "codigo_estado_cuota",
    "estado_cuota",
    "codigo_centro",
    "monto_cuota",
    "cuenta_especial",
    "cta_bnc_especial",
    "fecha_ce",
]


def _prep_show(d: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in BASE_KEEP_COLS if c in d.columns]
    show = d[keep].rename(
        columns={
            "rut": "RUT",
            "cnv_fecha_inicio": "Inicio Convenio",
            "cnv_fecha_termino": "Término Convenio",
            "dcu_correlativo": "Correlativo",
            "cnv_cuotas": "N° Cuotas",
            "fecha_emision": "Fecha Emisión",
            "fecha_cuota": "Fecha Cuota",
            "codigo_estado_cuota": "Código Estado",
            "estado_cuota": "Estado Cuota",
            "codigo_centro": "Código Centro",
            "monto_cuota": "Monto Cuota",
            "cuenta_especial": "Cuenta Especial",
            "cta_bnc_especial": "CTA Banco Especial",
            "fecha_ce": "Fecha Pago",
        }
    )

    for col in (
        "Inicio Convenio",
        "Término Convenio",
        "Fecha Emisión",
        "Fecha Cuota",
        "Fecha Pago",
    ):
        if col in show:
            fechas = pd.to_datetime(show[col], errors="coerce")
            show[col] = fechas.dt.strftime("%d-%m-%Y").fillna("s/d")
    if "Cuenta Especial" in show:
        show["Cuenta Especial"] = show["Cuenta Especial"].map({True: "Sí", False: "No"}).fillna("No")
    if "Monto Cuota" in show:
        show["Monto Cuota"] = (
            pd.to_numeric(show["Monto Cuota"], errors="coerce").fillna(0).map(money)
        )
    if "N° Cuotas" in show:
        cuotas = pd.to_numeric(show["N° Cuotas"], errors="coerce")
        show["N° Cuotas"] = cuotas.apply(lambda x: "" if pd.isna(x) else f"{int(x)}")
    return show


def _prep_export(d: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in BASE_KEEP_COLS if c in d.columns]
    out = d[keep].rename(
        columns={
            "rut": "RUT",
            "cnv_fecha_inicio": "Inicio Convenio",
            "cnv_fecha_termino": "Término Convenio",
            "dcu_correlativo": "Correlativo",
            "cnv_cuotas": "N° Cuotas",
            "fecha_emision": "Fecha Emisión",
            "fecha_cuota": "Fecha Cuota",
            "codigo_estado_cuota": "Código Estado",
            "estado_cuota": "Estado Cuota",
            "codigo_centro": "Código Centro",
            "monto_cuota": "Monto Cuota",
            "cuenta_especial": "Cuenta Especial",
            "cta_bnc_especial": "CTA Banco Especial",
            "fecha_ce": "Fecha Pago",
        }
    )
    if "Proveedor Prioritario" in out:
        out["Proveedor Prioritario"] = out["Proveedor Prioritario"].map({True: "Sí", False: "No"})
    if "Cuenta Especial" in out:
        out["Cuenta Especial"] = out["Cuenta Especial"].map({True: "Sí", False: "No"})
    if "Monto" in out:
        out["Monto"] = pd.to_numeric(out["Monto"], errors="coerce").fillna(0.0)
    if "AP" in out:
        ap_numeric = pd.to_numeric(out["AP"], errors="coerce")
        out["AP"] = pd.array(
            [pd.NA if pd.isna(val) else int(val) for val in ap_numeric],
            dtype="Int64",
        )
    return out

if ce_available and count_pagadas:
    pagadas_ce_flag = ce_flag_all.loc[pagadas.index]
else:
    pagadas_ce_flag = pd.Series(False, index=pagadas.index, dtype=bool)

if ce_available and count_no_pagadas:
    no_pagadas_ce_flag = ce_flag_all.loc[no_pagadas.index]
else:
    no_pagadas_ce_flag = pd.Series(False, index=no_pagadas.index, dtype=bool)

total_ce_count = int(ce_flag_all.sum()) if ce_available else 0
total_no_ce_count = total_docs - total_ce_count

pagadas_ce_count = int(pagadas_ce_flag.sum()) if ce_available else 0
pagadas_no_ce_count = max(0, count_pagadas - pagadas_ce_count)

en_plazo_ce_count = int((pagadas_ce_flag & mask_dentro_plazo).sum()) if ce_available else 0
en_plazo_no_ce_count = max(0, count_en_plazo - en_plazo_ce_count)
atraso_ce_count = int((pagadas_ce_flag & mask_atraso).sum()) if ce_available else 0
atraso_no_ce_count = max(0, count_atraso - atraso_ce_count)

no_pagadas_ce_count = int(no_pagadas_ce_flag.sum()) if ce_available else 0
no_pagadas_no_ce_count = max(0, count_no_pagadas - no_pagadas_ce_count)

if ce_available and count_pagadas:
    pagadas_ce_df = pagadas.loc[pagadas_ce_flag]
    pagadas_no_ce_df = pagadas.loc[~pagadas_ce_flag]
else:
    pagadas_ce_df = pagadas.iloc[0:0]
    pagadas_no_ce_df = pagadas.iloc[0:0]

if ce_available and count_no_pagadas:
    no_pagadas_ce_df = no_pagadas.loc[no_pagadas_ce_flag]
    no_pagadas_no_ce_df = no_pagadas.loc[~no_pagadas_ce_flag]
else:
    no_pagadas_ce_df = no_pagadas.iloc[0:0]
    no_pagadas_no_ce_df = no_pagadas.iloc[0:0]

monto_pagado_ce = _sum_numeric(pagadas_ce_df, ["monto_pagado", "liquido_cuota", "fac_monto_total"])
monto_pagado_no = max(0.0, monto_pagado_total - monto_pagado_ce)
monto_no_pagado_ce = _sum_numeric(no_pagadas_ce_df, ["monto_autorizado", "monto_cuota", "fac_monto_total"])
monto_no_pagado_no = max(0.0, monto_no_pagado_total - monto_no_pagado_ce)

count_same_day_total = count_same_day
count_anticipada_total = count_anticipada
count_especial_total = count_especial

metric_cols = st.columns(5)
with metric_cols[0]:
    breakdown = _counts_html(total_ce_count, total_no_ce_count) if ce_available else None
    _render_metric_block("Honorarios cargados", f"{total_docs:,}", breakdown_html=breakdown)
with metric_cols[1]:
    breakdown = _counts_html(pagadas_ce_count, pagadas_no_ce_count) if ce_available else None
    _render_metric_block("Pagadas", f"{count_pagadas:,}", footer=f"{pct_pagadas_total:.1f}% del total", breakdown_html=breakdown)
with metric_cols[2]:
    breakdown = _counts_html(en_plazo_ce_count, en_plazo_no_ce_count) if ce_available else None
    _render_metric_block("Pagadas dentro de plazo", f"{count_en_plazo:,}", footer=f"{pct_en_plazo:.1f}% de pagadas", breakdown_html=breakdown)
with metric_cols[3]:
    breakdown = _counts_html(atraso_ce_count, atraso_no_ce_count) if ce_available else None
    _render_metric_block("Pagadas con atraso", f"{count_atraso:,}", footer=f"{pct_atraso:.1f}% de pagadas", breakdown_html=breakdown)
with metric_cols[4]:
    breakdown = _counts_html(no_pagadas_ce_count, no_pagadas_no_ce_count) if ce_available else None
    _render_metric_block("No pagadas", f"{count_no_pagadas:,}", footer=f"{pct_no_pagadas_total:.1f}% del total", breakdown_html=breakdown)

if (count_same_day_total + count_anticipada_total + count_especial_total) > 0:
    st.caption(
        f"Dentro de plazo (total): {count_same_day_total:,} mismo dia | {count_anticipada_total:,} anticipadas | {count_especial_total:,} especiales"
    )
if count_sin_info > 0:
    st.caption(f"{count_sin_info:,} honorarios pagados sin fechas completas para clasificar.")

amount_cols = st.columns(2)
with amount_cols[0]:
    breakdown = _amount_html(monto_pagado_ce, monto_pagado_no) if ce_available else None
    _render_metric_block("Monto pagado", money(monto_pagado_total), breakdown_html=breakdown)
with amount_cols[1]:
    breakdown = _amount_html(monto_no_pagado_ce, monto_no_pagado_no) if ce_available else None
    _render_metric_block("Monto no pagado", money(monto_no_pagado_total), breakdown_html=breakdown)

if count_pagadas:
    pagadas["tiempo_pago_planeado"] = (pagadas["fecha_cuota_ref"] - pagadas["fecha_emision_ref"]).dt.days
    pagadas["tiempo_pago_real"] = (pagadas["fecha_pago_ref"] - pagadas["fecha_emision_ref"]).dt.days

    pagadas["tiempo_pago_planeado"] = pagadas["tiempo_pago_planeado"].clip(lower=0)
    if mask_especial.any():
        ajuste_especial = (pagadas.loc[mask_especial, "fecha_cuota_ref"] - pagadas.loc[mask_especial, "fecha_emision_ref"]).dt.days
        pagadas.loc[mask_especial, "tiempo_pago_real"] = ajuste_especial
    pagadas["tiempo_pago_real"] = pagadas["tiempo_pago_real"].clip(lower=0)

    st.markdown("#### Cumplimiento de pagos")
    pie_cols = st.columns(3) if ce_available else st.columns(1)

    with pie_cols[0]:
        st.markdown("**Pagadas por cumplimiento**")
        if (count_en_plazo + count_atraso) == 0:
            st.info("Sin registros con fechas completas para esta vista.")
        else:
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Dentro de plazo", "Con atraso"],
                        values=[count_en_plazo, count_atraso],
                        textinfo="label+percent",
                        hole=0.35,
                        marker=dict(colors=["#4E79A7", "#F28E2B"]),
                    )
                ]
            )
            fig_pie.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2))
            st.plotly_chart(fig_pie, use_container_width=True)
            if count_especial > 0:
                st.caption("Incluye pagos especiales dentro del plazo.")

    if ce_available:
        with pie_cols[1]:
            st.markdown("**Pagadas dentro de plazo por cuenta**")
            total_in_plazo_ce = en_plazo_ce_count + en_plazo_no_ce_count
            if total_in_plazo_ce == 0:
                st.info("Sin datos de pagadas dentro de plazo para graficar.")
            else:
                fig_in_plazo = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Cuenta especial", "No cuenta especial"],
                            values=[en_plazo_ce_count, en_plazo_no_ce_count],
                            textinfo="label+percent",
                            hole=0.35,
                            marker=dict(colors=CE_COLORS),
                        )
                    ]
                )
                fig_in_plazo.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2))
                st.plotly_chart(fig_in_plazo, use_container_width=True)

        with pie_cols[2]:
            st.markdown("**Pagadas con atraso por cuenta**")
            total_atraso_ce = atraso_ce_count + atraso_no_ce_count
            if total_atraso_ce == 0:
                st.info("Sin datos de pagos atrasados para graficar.")
            else:
                fig_atraso = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Cuenta especial", "No cuenta especial"],
                            values=[atraso_ce_count, atraso_no_ce_count],
                            textinfo="label+percent",
                            hole=0.35,
                            marker=dict(colors=CE_COLORS),
                        )
                    ]
                )
                fig_atraso.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2))
                st.plotly_chart(fig_atraso, use_container_width=True)

    if ce_available:
        ce_filter_options = ["Todas", "Cuenta especial", "No cuenta especial"]
        default_filter = st.session_state.get("honorarios_ce_hist_filter", "Todas")
        if default_filter not in ce_filter_options:
            default_filter = "Todas"
        ce_filter_value = st.radio(
            "Filtro de cuenta especial (solo histogramas)",
            ce_filter_options,
            index=ce_filter_options.index(default_filter),
            key="honorarios_ce_hist_filter",
            horizontal=True,
        )
    else:
        ce_filter_value = "Todas"

    mask_filter = pd.Series(True, index=pagadas.index)
    if ce_available:
        if ce_filter_value == "Cuenta especial":
            mask_filter = pagadas_ce_flag
        elif ce_filter_value == "No cuenta especial":
            mask_filter = ~pagadas_ce_flag

    selected_idx = pagadas.index[mask_filter]
    pagadas_view = pagadas.loc[selected_idx]

    if pagadas_view.empty:
        st.info("No hay honorarios pagados para el filtro seleccionado.")
    else:
        mask_same_day_view = mask_same_day.loc[selected_idx]
        mask_anticipada_view = mask_anticipada_regular.loc[selected_idx]
        mask_especial_view = mask_especial_plazo.loc[selected_idx]
        mask_atraso_view = mask_atraso.loc[selected_idx]

        count_same_day_view = int(mask_same_day_view.sum())
        count_anticipada_view = int(mask_anticipada_view.sum())
        count_especial_view = int(mask_especial_view.sum())

        serie_plan_total = pagadas_view["tiempo_pago_planeado"].dropna()
        serie_real_total = pagadas_view.loc[mask_atraso_view, "tiempo_pago_real"].dropna()

        st.markdown("#### Histogramas de tiempos (solo PAGADAS)")
        safe_markdown(
            f"<div style='font-size:20px; font-weight:600;'>Total observaciones planificado: {len(serie_plan_total):,} | Con atraso: {len(serie_real_total):,}</div>",
        )
        bin_size = st.slider("Ancho de clase (dias)", 1, 60, 1, key="hon_hist_bin")
        hist_cols = st.columns(2)

        def _stats(series: pd.Series) -> Optional[Dict[str, float]]:
            vals = series.dropna().to_numpy()
            if vals.size == 0:
                return None
            return {
                "mean": float(np.nanmean(vals)),
                "p50": float(np.nanpercentile(vals, 50)),
                "p75": float(np.nanpercentile(vals, 75)),
                "p90": float(np.nanpercentile(vals, 90)),
            }

        with hist_cols[0]:
            st.markdown("**Tiempo planificado (fecha cuota - fecha emision)**")
            if serie_plan_total.empty:
                st.info("Sin datos suficientes para el histograma de tiempo planificado.")
            else:
                fig_plan = go.Figure()
                vals_same_day = pagadas_view.loc[mask_same_day_view, "tiempo_pago_planeado"].dropna()
                vals_anticipada = pagadas_view.loc[mask_anticipada_view, "tiempo_pago_planeado"].dropna()
                vals_especial = pagadas_view.loc[mask_especial_view, "tiempo_pago_planeado"].dropna()
                safe_markdown(
                    "<div style='font-size:18px; font-weight:600;'>Desglose dentro de plazo</div>"
                    f"<div style='font-size:18px;'>Mismo dia: {count_same_day_view:,} | Anticipadas: {count_anticipada_view:,} | Especiales: {count_especial_view:,}</div>"
                )
                if not vals_same_day.empty:
                    fig_plan.add_trace(go.Histogram(x=vals_same_day, xbins=dict(size=bin_size), name="Mismo dia", opacity=0.7, marker_color="#4E79A7"))
                if not vals_anticipada.empty:
                    fig_plan.add_trace(go.Histogram(x=vals_anticipada, xbins=dict(size=bin_size), name="Anticipadas", opacity=0.7, marker_color="#59A14F"))
                if not vals_especial.empty:
                    fig_plan.add_trace(go.Histogram(x=vals_especial, xbins=dict(size=bin_size), name="Pagos especiales", opacity=0.7, marker_color="#AF7AA1"))
                fig_plan.update_layout(barmode="overlay", xaxis_title="Dias", yaxis_title="Cantidad de honorarios", legend=dict(orientation="h", yanchor="bottom", y=-0.15))
                stats_plan_view = _stats(serie_plan_total)
                if stats_plan_view:
                    fig_plan.add_vline(x=stats_plan_view["mean"], line_color="#E15759", line_dash="dash", annotation_text=f"Promedio {stats_plan_view['mean']:.1f}d", annotation_position="top")
                st.plotly_chart(fig_plan, use_container_width=True)
                if stats_plan_view:
                    safe_markdown(
                        f"<div style='font-size:24px; margin-top:4px;'>Total: {len(serie_plan_total):,} | Promedio: {stats_plan_view['mean']:.1f} dias | P50: {stats_plan_view['p50']:.1f} | P75: {stats_plan_view['p75']:.1f} | P90: {stats_plan_view['p90']:.1f}</div>"
                    )

                with st.expander('Detalle del calculo del tiempo planificado'):
                    st.markdown(
                        """- **Tiempo planificado** = `fecha_cuota` - `fecha_emision` (se fuerza a 0 si el resultado es negativo).
- **Mismo dia**: `fecha_pago` coincide con `fecha_cuota`.
- **Anticipadas**: `fecha_pago` es menor que `fecha_cuota`; se conserva el tiempo planificado original.
- **Pagos especiales**: `fecha_pago` es menor que `fecha_emision`; se usa `fecha_cuota` para el calculo del tiempo.
- Los conteos presentados incluyen solo filas validas con fechas completas y el filtro seleccionado."""
                    )

        with hist_cols[1]:
            st.markdown("**Pagadas con atraso (fecha pago - fecha emision)**")
            if serie_real_total.empty:
                st.info("Sin datos de pagos atrasados para el histograma.")
            else:
                fig_real = go.Figure()
                fig_real.add_trace(go.Histogram(x=serie_real_total, xbins=dict(size=bin_size), name="Pagadas con atraso", opacity=0.85, marker_color="#E15759"))
                safe_markdown(
                    f"<div style='font-size:20px; font-weight:600;'>Total observaciones (atraso): {len(serie_real_total):,}</div>"
                )
                stats_real_view = _stats(serie_real_total)
                if stats_real_view:
                    fig_real.add_vline(x=stats_real_view["mean"], line_color="#E15759", line_dash="dash", annotation_text=f"Promedio {stats_real_view['mean']:.1f}d", annotation_position="top")
                fig_real.update_layout(barmode="overlay", xaxis_title="Dias", yaxis_title="Cantidad de honorarios", legend=dict(orientation="h", yanchor="bottom", y=-0.15))
                st.plotly_chart(fig_real, use_container_width=True)
                if stats_real_view:
                    safe_markdown(
                        f"<div style='font-size:24px; margin-top:4px;'>Total: {len(serie_real_total):,} | Promedio: {stats_real_view['mean']:.1f} dias | P50: {stats_real_view['p50']:.1f} | P75: {stats_real_view['p75']:.1f} | P90: {stats_real_view['p90']:.1f}</div>"
                    )

                with st.expander('Detalle del calculo del tiempo real pagado'):
                    st.markdown(
                        """- **Tiempo real** = `fecha_pago` - `fecha_emision` (solo pagos con `fecha_pago` > `fecha_cuota`).
- Se fuerza a 0 cuando el resultado es negativo despues de los ajustes.
- La linea punteada indica el promedio de dias de atraso sobre las observaciones del filtro.
- El histograma excluye registros sin ambas fechas o sin atraso confirmado."""
                    )
else:
    st.info("Aun no hay honorarios marcados como PAGADA para analizar su cumplimiento.")


# =========================================================
# 3) Análisis de Deuda Pendiente
# =========================================================
st.subheader("3) Análisis de Deuda Pendiente")

df_nopag_all = no_pagadas.copy()
if not df_nopag_all.empty:
    df_nopag_all = _ensure_importe_deuda_hon(df_nopag_all)
    df_nopag_all = _ensure_dias_a_vencer_hon(df_nopag_all)
    if "estado_cuota" in df_nopag_all:
        df_nopag_all["Nivel"] = df_nopag_all["estado_cuota"].astype(str).apply(_classify_nivel_hon)
    else:
        df_nopag_all["Nivel"] = "Contabilizado Pendiente de Pago"
    if "cuenta_especial" in df_nopag_all:
        df_nopag_all["cuenta_especial"] = df_nopag_all["cuenta_especial"].fillna(False).astype(bool)
    else:
        df_nopag_all["cuenta_especial"] = False
    if "prov_prioritario" in df_nopag_all:
        df_nopag_all["prov_prioritario"] = df_nopag_all["prov_prioritario"].fillna(False).astype(bool)

    if {"fecha_emision_ref", "fecha_cuota_ref"}.issubset(df_nopag_all.columns):
        fechas_validas = df_nopag_all[["fecha_emision_ref", "fecha_cuota_ref"]].notna().all(axis=1)
        categoria = np.where(
            fechas_validas & (df_nopag_all["fecha_emision_ref"] > df_nopag_all["fecha_cuota_ref"]),
            "Emisión posterior a cuota",
            "Emisión anterior o igual a cuota",
        )
        categoria = np.where(~fechas_validas, "Sin información de fechas", categoria)
        df_nopag_all["clasificacion_plazo"] = categoria
        df_nopag_all["dias_sin_pago"] = (
            TODAY - pd.to_datetime(df_nopag_all.get("fecha_emision_ref"), errors="coerce")
        ).dt.days
        df_nopag_all["dias_para_cuota"] = (
            pd.to_datetime(df_nopag_all.get("fecha_cuota_ref"), errors="coerce") - TODAY
        ).dt.days
    else:
        df_nopag_all["clasificacion_plazo"] = "Sin información de fechas"
        df_nopag_all["dias_sin_pago"] = np.nan
        df_nopag_all["dias_para_cuota"] = np.nan

    df_nopag_all["monto_cuota_num"] = pd.to_numeric(df_nopag_all.get("monto_cuota"), errors="coerce").fillna(0.0)
else:
    df_nopag_all = df_nopag_all.iloc[0:0]

if df_nopag_all.empty:
    st.info("No hay honorarios pendientes de pago.")
else:
    contab_df = (
        df_nopag_all[df_nopag_all["Nivel"].eq("Contabilizado Pendiente de Pago")]
        if "Nivel" in df_nopag_all
        else df_nopag_all.iloc[0:0]
    )
    sincontab_df = (
        df_nopag_all[df_nopag_all["Nivel"].eq("Pendiente de Contabilización")]
        if "Nivel" in df_nopag_all
        else df_nopag_all.iloc[0:0]
    )
    _draw_debt_panel("Contabilizado Pendiente de Pago", contab_df)
    _draw_debt_panel("Pendiente de Contabilización", sincontab_df)

    categorias_visibles = [
        "Emisión posterior a cuota",
        "Emisión anterior o igual a cuota",
    ]
    cards_categoria: list[str] = []
    for label in categorias_visibles:
        subset = df_nopag_all[df_nopag_all["clasificacion_plazo"] == label]
        if subset.empty:
            continue
        total_monto = float(subset["monto_cuota_num"].sum())
        total_docs = int(len(subset))
        ce_mask = subset.get("cuenta_especial", pd.Series(False, index=subset.index)).astype(bool)
        ce_count = int(ce_mask.sum())
        ce_monto = float(subset.loc[ce_mask, "monto_cuota_num"].sum()) if ce_count else 0.0
        no_ce_count = total_docs - ce_count
        no_ce_monto = total_monto - ce_monto
        dias_prom = _avg_days_from_series(subset.get("dias_sin_pago"))

        stats = [
            ("Prom. días sin pago", _fmt_days(dias_prom)),
            ("Cuenta especial", f"{ce_count:,} • {money(ce_monto)}"),
            ("No cuenta especial", f"{no_ce_count:,} • {money(no_ce_monto)}"),
        ]
        cards_categoria.append(
            _card_html(
                title=label,
                value=money(total_monto),
                subtitle=f"{total_docs:,} honorarios sin pagar",
                stats=stats,
                tone="accent" if label == "Emisión posterior a cuota" else "default",
                compact=False,
            )
        )
    if cards_categoria:
        safe_markdown(
            "<div class='app-title-block'><h3 style='color:#000;'>Clasificación por fecha de cuota</h3>"
            "<p>Desglose basado en la comparación entre emisión y cuota.</p></div>"
        )
        _render_cards(cards_categoria, layout="grid-2")

    resumen_tipo = (
        df_nopag_all.groupby(["clasificacion_plazo", "estado_cuota"], dropna=False)
        .agg(
            Monto_Pendiente=("monto_cuota_num", "sum"),
            Honorarios=("monto_cuota_num", "count"),
            Prom_Dias_Sin_Pago=("dias_sin_pago", "mean"),
            Prom_Dias_Hasta_Cuota=("dias_para_cuota", "mean"),
        )
        .reset_index()
    )

    if not resumen_tipo.empty:
        display_tipo = resumen_tipo.copy()
        display_tipo["Monto_Pendiente"] = display_tipo["Monto_Pendiente"].map(money)
        display_tipo["Honorarios"] = display_tipo["Honorarios"].astype(int)
        display_tipo["Prom_Dias_Sin_Pago"] = display_tipo["Prom_Dias_Sin_Pago"].apply(_fmt_days)
        display_tipo["Prom_Dias_Hasta_Cuota"] = display_tipo["Prom_Dias_Hasta_Cuota"].apply(_fmt_days)
        display_tipo = display_tipo.rename(
            columns={
                "clasificacion_plazo": "Clasificación",
                "estado_cuota": "Estado Cuota",
                "Monto_Pendiente": "Monto Pendiente",
                "Honorarios": "Cant. Honorarios",
                "Prom_Dias_Sin_Pago": "Prom. días sin pago",
                "Prom_Dias_Hasta_Cuota": "Prom. días hasta cuota",
            }
        )
        display_tipo = sanitize_df(display_tipo)
        safe_markdown("**Resumen por clasificación y estado de cuota**")
        style_table(_table_style(display_tipo))

    resumen_ce = (
        df_nopag_all.groupby(["clasificacion_plazo", "cuenta_especial"], dropna=False)
        .agg(
            Monto_Pendiente=("monto_cuota_num", "sum"),
            Honorarios=("monto_cuota_num", "count"),
        )
        .reset_index()
    )
    if not resumen_ce.empty:
        display_ce = resumen_ce.copy()
        display_ce["Monto_Pendiente"] = display_ce["Monto_Pendiente"].map(money)
        display_ce["Honorarios"] = display_ce["Honorarios"].astype(int)
        display_ce["cuenta_especial"] = display_ce["cuenta_especial"].map({True: "Sí", False: "No"}).fillna("Sin dato")
        display_ce = display_ce.rename(
            columns={
                "clasificacion_plazo": "Clasificación",
                "cuenta_especial": "Cuenta especial",
                "Monto_Pendiente": "Monto Pendiente",
                "Honorarios": "Cant. Honorarios",
            }
        )
        display_ce = sanitize_df(display_ce)
        safe_markdown("**Desglose por cuenta especial**")
        style_table(_table_style(display_ce))


# =========================================================
# 4) Proyección de Vencimientos y Tablas de Pago
# =========================================================
st.subheader("4) Proyección de Vencimientos y Tablas de Pago")
st.caption("Los filtros siguientes impactan esta sección y el presupuesto automático.")

col_criterio, col_cuenta, col_prioritario = st.columns([1.4, 1, 1])

crit_sel = col_criterio.radio(
    "Criterio de Orden",
    ["Riesgo de aprobación", "Urgencia de vencimiento"],
    horizontal=True,
)

ce_local = col_cuenta.radio(
    "Cuenta Especial (local)",
    ["Todas", "Cuenta Especial", "No Cuenta Especial"],
    horizontal=True,
    index=0,
)

prio_local = col_prioritario.radio(
    "Proveedor Prioritario (local)",
    ["Todos", "Prioritario", "No Prioritario"],
    horizontal=True,
    index=0,
)

df_nopag_loc = _apply_local_filters(
    df_nopag_all,
    ce_filter=ce_local,
    prio_filter=prio_local,
)

vencidos_m, vencidos_c = _agg_block(df_nopag_loc, df_nopag_loc.get("dias_a_vencer", pd.Series(dtype=float)) < 0) if not df_nopag_loc.empty and "dias_a_vencer" in df_nopag_loc else (0.0, 0)
hoy_m, hoy_c = _agg_block(df_nopag_loc, df_nopag_loc.get("dias_a_vencer", pd.Series(dtype=float)) == 0) if not df_nopag_loc.empty and "dias_a_vencer" in df_nopag_loc else (0.0, 0)
criticos_m = vencidos_m + hoy_m
criticos_c = vencidos_c + hoy_c

_render_cards(
    [
        _card_html(
            "Deuda vencida",
            money(vencidos_m),
            subtitle="Honorarios con atraso",
            tag=_fmt_count(vencidos_c),
            tone="accent",
        ),
        _card_html(
            "Vencen hoy",
            money(hoy_m),
            subtitle="Monto que vence el día de hoy",
            tag=_fmt_count(hoy_c),
        ),
        _card_html(
            "Monto crítico",
            money(criticos_m),
            subtitle="Suma de vencidos + hoy",
            tag=_fmt_count(criticos_c),
            tag_variant="warning",
        ),
    ]
)

with st.container():
    col_horizonte, col_tabla = st.columns([0.6, 0.7])
    horizonte = col_horizonte.number_input("Horizonte (días)", min_value=7, max_value=90, value=30, step=7)
    dias_tabla = col_tabla.number_input(
        "Días tabla a proyectar",
        min_value=1,
        max_value=int(horizonte),
        value=min(3, int(horizonte)),
        step=1,
    )

fig_proyeccion: Optional[go.Figure] = None
if not df_nopag_loc.empty and "fecha_venc_30" in df_nopag_loc:
    horizonte_td = pd.to_timedelta(int(horizonte), "D")
    proy = df_nopag_loc[df_nopag_loc["fecha_venc_30"].between(TODAY, TODAY + horizonte_td)]
    if not proy.empty:
        flujo = (
            proy.groupby(proy["fecha_venc_30"].dt.date)
            .agg(
                Monto_a_Pagar=("importe_deuda", "sum"),
                Cant_Honorarios=("importe_deuda", "count"),
            )
            .reset_index()
            .rename(columns={"fecha_venc_30": "Fecha"})
        )
        flujo["Flujo_Acumulado"] = flujo["Monto_a_Pagar"].cumsum()
        st.markdown(f"**Proyección Próximos {int(dias_tabla)} días**")
        small = flujo.head(int(dias_tabla)).rename(
            columns={
                "Fecha": "Día",
                "Monto_a_Pagar": "Monto a Pagar",
                "Cant_Honorarios": "Cant. Honorarios",
                "Flujo_Acumulado": "Flujo Acumulado",
            }
        )
        small_display = small.copy()
        for col in ("Monto a Pagar", "Flujo Acumulado"):
            if col in small_display:
                small_display[col] = small_display[col].map(money)
        small_display = sanitize_df(small_display)
        style_table(_table_style(small_display))

        accent_color = "#3f51b5"
        accent_soft = "#9aa5ff"
        secondary_color = "#ff7043"
        accumulated_color = "#1e88e5"

        selected_count = int(dias_tabla)
        selected = flujo.iloc[:selected_count]

        fig = go.Figure()
        fig.add_bar(
            x=flujo["Fecha"],
            y=flujo["Cant_Honorarios"],
            name="Cant. Honorarios",
            yaxis="y2",
            marker=dict(
                color=[accent_color if i < selected_count else accent_soft for i in range(len(flujo))],
                line=dict(color="#2c3c8f", width=0.5),
            ),
            opacity=0.85,
        )
        fig.add_scatter(
            x=flujo["Fecha"],
            y=flujo["Monto_a_Pagar"],
            name="Monto Diario",
            mode="lines",
            line=dict(color=secondary_color, width=3, shape="spline"),
        )
        fig.add_scatter(
            x=flujo["Fecha"],
            y=flujo["Flujo_Acumulado"],
            name="Acumulado",
            mode="lines",
            line=dict(color=accumulated_color, width=3, dash="dash"),
        )
        if not selected.empty:
            fig.add_scatter(
                x=selected["Fecha"],
                y=selected["Monto_a_Pagar"],
                mode="markers+text",
                name="Monto Selección",
                marker=dict(size=12, color=secondary_color, line=dict(color="#ffffff", width=2)),
                text=selected["Monto_a_Pagar"].map(money),
                textposition="top center",
                hovertemplate="<b>%{x}</b><br>Monto: %{text}<extra></extra>",
                showlegend=False,
            )

        fig.update_layout(
            height=420,
            template="plotly_white",
            bargap=0.25,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=40, l=50, r=50),
            yaxis=dict(title="Monto ($)", zeroline=False, showgrid=True, gridcolor="rgba(63, 81, 181, 0.12)"),
            yaxis2=dict(
                overlaying="y",
                side="right",
                title="Cantidad",
                zeroline=False,
                showgrid=False,
            ),
            xaxis=dict(showgrid=False),
            plot_bgcolor="rgba(248, 250, 255, 0.9)",
            paper_bgcolor="rgba(255, 255, 255, 1)",
        )
        st.plotly_chart(fig, use_container_width=True)
        fig_proyeccion = fig
    else:
        st.info("Sin vencimientos en el horizonte seleccionado.")
else:
    st.info("No hay datos de vencimientos para proyectar con los filtros seleccionados.")

candidatas_base = _apply_horizon_filter(df_nopag_loc, horizonte)
candidatas_prior = _prioritize_documents(candidatas_base, crit_sel)


# =========================================================
# 5) Presupuesto del Día (Selección Automática)
# =========================================================
st.subheader("5) Presupuesto del Día (Selección Automática)")

seleccion = pd.DataFrame()
seleccion_df = pd.DataFrame()

if candidatas_base.empty or "importe_deuda" not in candidatas_base:
    st.info("No hay honorarios pendientes para priorizar con los filtros locales.")
else:
    prior = candidatas_prior
    total_criticos = float(prior.loc[prior.get("dias_a_vencer", pd.Series(dtype=float)) <= 0, "importe_regla"].sum()) if "importe_regla" in prior else float(prior.loc[prior.get("dias_a_vencer", pd.Series(dtype=float)) <= 0, "importe_deuda"].sum())
    default_presu = total_criticos if total_criticos > 0 else 0.0

    local_filters = {"ce": ce_local, "prio": prio_local, "crit": crit_sel}
    _update_presupuesto_session_state(local_filters, float(default_presu))

    current_amount = float(st.session_state.get(_LOCAL_AMOUNT_KEY, default_presu))
    budget_input_key = f"{_LOCAL_AMOUNT_KEY}__text"
    if budget_input_key not in st.session_state:
        st.session_state[budget_input_key] = _format_currency_plain(current_amount)

    safe_markdown(_BUDGET_PANEL_STYLE)

    with st.container():
        safe_markdown(
            """
            <div class="budget-panel">
                <div class="budget-panel__title">Monto disponible hoy</div>
                <div class="budget-panel__subtitle">Define tu presupuesto diario considerando cuentas críticas y prioridades.</div>
                <div class="budget-panel__input-wrapper">
            """
        )
        col_input, col_resume = st.columns([2.5, 1.5])
        with col_input:
            presupuesto_text = st.text_input(
                "Monto disponible hoy",
                value=st.session_state[budget_input_key],
                label_visibility="collapsed",
                help="Ingresa el monto en CLP. Puedes usar puntos o comas como separador.",
            )
            monto_presu = float(round(_parse_currency_input(presupuesto_text, current_amount)))
            st.session_state[_LOCAL_AMOUNT_KEY] = monto_presu
            formatted_text = _format_currency_plain(monto_presu)
            if presupuesto_text != formatted_text:
                st.session_state[budget_input_key] = formatted_text
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()

        (
            seleccion_df,
            suma_sel,
            restante,
            next_info,
            pendientes_info,
        ) = _compute_presupuesto_selection(prior, float(monto_presu))

        with col_resume:
            amount_col, action_col = st.columns([1.7, 1.3])
            with amount_col:
                resume_html = (
                    f"""
                    <div class=\"budget-panel__resume\">
                        <span class=\"budget-panel__resume-label\">Equivale a</span>
                        <span class=\"budget-panel__resume-value\">{money(monto_presu)}</span>
                    </div>
                    """
                )
                safe_markdown(resume_html)
            with action_col:
                pendientes_count = int(pendientes_info.get("count", 0))
                missing_budget_all = float(pendientes_info.get("missing_budget", 0.0))
                full_budget = float(pendientes_info.get("full_budget", 0.0))
                remaining_amount_all = float(pendientes_info.get("remaining_amount", 0.0))
                if next_info:
                    help_text = f"Se sumarán {money(next_info['adicional'])} para incorporar el siguiente honorario."
                    button_label = "➕ Agregar honorario extra"
                    if st.button(
                        button_label,
                        key="hon_add_extra_invoice",
                        help=(
                            f"{next_info['proveedor']} — Documento {next_info['documento']} por {money(next_info['monto'])}. "
                            + help_text
                        ),
                        use_container_width=True,
                    ):
                        nuevo_monto = float(monto_presu + next_info["adicional"])
                        st.session_state[_LOCAL_AMOUNT_KEY] = nuevo_monto
                        st.session_state[budget_input_key] = _format_currency_plain(nuevo_monto)
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
                    safe_markdown(
                        f"<p class='budget-panel__resume-note'>Próximo: {next_info['proveedor']} — Documento {next_info['documento']} ({money(next_info['monto'])})</p>"
                    )
                    safe_markdown(
                        f"<p class='budget-panel__resume-note budget-panel__resume-note--muted'>Ajuste necesario: {money(next_info['adicional'])}</p>"
                    )
                else:
                    safe_markdown(
                        "<p class='budget-panel__resume-note budget-panel__resume-note--muted'>No hay honorarios adicionales por agregar.</p>"
                    )
                if pendientes_count > 0 and missing_budget_all > 1e-6:
                    button_help = (
                        "El presupuesto se ajustará para cubrir todas las candidatas. "
                        + f"Nuevo total: {money(full_budget)}."
                    )
                    if st.button(
                        "🧾 Agregar todos los honorarios pendientes",
                        key="hon_add_all_invoices",
                        help=button_help,
                        use_container_width=True,
                    ):
                        nuevo_monto = float(monto_presu + missing_budget_all)
                        st.session_state[_LOCAL_AMOUNT_KEY] = nuevo_monto
                        st.session_state[budget_input_key] = _format_currency_plain(nuevo_monto)
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
                    safe_markdown(
                        "<p class='budget-panel__resume-note budget-panel__resume-note--muted'>"
                        + f"Pendientes: {pendientes_count} honorarios, {money(remaining_amount_all)}"
                        + "</p>"
                    )

        safe_markdown(
            """
                </div>
                <div class="budget-panel__helper">El valor se guarda automáticamente para esta sesión y se utilizará en el informe PDF.</div>
            </div>
            """
        )

        seleccion = seleccion_df

        resumen_cards = [
            _card_html("Presupuesto ingresado", money(float(monto_presu)), subtitle="Disponible hoy", tone="accent"),
            _card_html("Suma selección", money(suma_sel), subtitle="Total comprometido"),
            _card_html("Saldo sin asignar", money(restante), subtitle="Presupuesto - selección", tag_variant="warning"),
        ]
        if next_info:
            resumen_cards.append(
                _card_html(
                    "Próximo honorario a incorporar",
                    money(next_info["monto"]),
                    subtitle=f"{next_info['proveedor']} — Documento {next_info['documento']}",
                    tag=f"Faltan {money(next_info['adicional'])}" if next_info["adicional"] > 0 else "Disponible",
                    tag_variant="warning" if next_info["adicional"] > 0 else "success",
                    tone="accent",
                    stats=[
                        ("Saldo libre actual", money(restante)),
                        ("Ajuste necesario", money(next_info["adicional"])),
                    ],
                    compact=False,
                )
            )
        _render_cards(resumen_cards)

tab_candidatas, tab_seleccion = st.tabs([
    "Candidatas a Pago",
    "Selección a Pagar Hoy",
])

candidatas_display = _prep_show(candidatas_prior)
seleccion_display = _prep_show(seleccion)
candidatas_display = sanitize_df(candidatas_display)
seleccion_display = sanitize_df(seleccion_display)

with tab_candidatas:
    st.markdown("**Candidatas a Pago (todas las cuotas — ordenadas por prioridad)**")
    style_table(_table_style(candidatas_display), visible_rows=15)
    st.download_button(
        "⬇️ Descargar Candidatas",
        data=excel_bytes_single(_prep_export(candidatas_prior), "HonorariosCandidatas"),
        file_name="honorarios_candidatas.xlsx",
        disabled=candidatas_prior.empty,
    )

with tab_seleccion:
    safe_markdown(
        """
        <div class="app-note">
            <strong>Selección a pagar hoy</strong> — bloque crítico de honorarios.
        </div>
        """
    )
    style_table(_table_style(seleccion_display), visible_rows=15)
    st.download_button(
        "⬇️ Descargar Selección de Hoy",
        data=excel_bytes_single(_prep_export(seleccion), "HonorariosPagoHoy"),
        file_name="honorarios_pago_hoy.xlsx",
        disabled=seleccion.empty,
    )


# =========================================================
# 6) Reporte PDF
# =========================================================
st.subheader("6) Reporte PDF")
if st.button("📄 Generar PDF Honorarios"):
    pagos_criticos_pdf = pd.DataFrame()
    if isinstance(seleccion, pd.DataFrame) and not seleccion.empty:
        pagos_criticos_pdf = seleccion.rename(columns={"importe_deuda": "importe_deuda"}).copy()

    pdf_bytes = generate_pdf_report(
        secciones={"kpis": False, "rankings": False, "deuda": True, "presupuesto": True},
        kpis={},
        rankings_df=pd.DataFrame(),
        niveles_kpis={},
        proyeccion_chart=fig_proyeccion,
        pagos_criticos_df=pagos_criticos_pdf,
        sugerencias=["Priorizar 'Selección a Pagar Hoy' para los honorarios críticos."],
        filtros=None,
        presupuesto_monto=st.session_state.get(_LOCAL_AMOUNT_KEY),
        seleccion_hoy_df=seleccion,
    )

    st.download_button(
        "Guardar PDF",
        data=pdf_bytes,
        file_name=f"Informe_Honorarios_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )
