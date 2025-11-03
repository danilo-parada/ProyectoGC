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

def _paid_amount_series(df_in: pd.DataFrame) -> pd.Series:
    """Serie de monto pagado efectiva para registros PAGADOS.

    Prioriza columnas típicas en orden y retorna 0.0 cuando faltan.
    """
    amount = pd.Series(np.nan, index=df_in.index, dtype=float)
    for col in ("monto_pagado", "liquido_cuota", "fac_monto_total"):
        if col in df_in.columns:
            cand = pd.to_numeric(df_in[col], errors="coerce")
            amount = amount.fillna(cand)
    return amount.fillna(0.0)

def _quota_amount_series(df_in: pd.DataFrame) -> pd.Series:
    """Serie de monto de cuota para calculos base.

    Prioriza `monto_cuota` y cae a `monto_autorizado` o `fac_monto_total`.
    """
    amount = pd.Series(np.nan, index=df_in.index, dtype=float)
    for col in ("monto_cuota", "monto_autorizado", "fac_monto_total"):
        if col in df_in.columns:
            cand = pd.to_numeric(df_in[col], errors="coerce")
            amount = amount.fillna(cand)
    return amount.fillna(0.0)

monto_pagado_total = _sum_numeric(pagadas, ["monto_cuota", "monto_pagado", "liquido_cuota", "fac_monto_total"])
monto_no_pagado_total = _sum_numeric(no_pagadas, ["monto_cuota", "monto_autorizado", "fac_monto_total"])

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
    # Acepta DataFrame o Styler; si viene None, usa tabla vacía
    if isinstance(df_disp, pd.DataFrame):
        sty = df_disp.style
    else:
        sty = df_disp
        if sty is None:
            sty = pd.DataFrame().style
    # Oculta índice si es posible (versiones antiguas de pandas pueden diferir)
    try:
        sty = sty.hide(axis="index")
    except Exception:
        try:
            sty = sty.hide_index()
        except Exception:
            pass
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


def _resolve_pending_amount(dfin: pd.DataFrame) -> pd.Series:
    if dfin.empty:
        return pd.Series(dtype=float)

    amount = pd.Series(np.nan, index=dfin.index, dtype=float)
    preference = ["monto_autorizado", "monto_cuota", "fac_monto_total", "importe_deuda"]
    for col in preference:
        if col in dfin.columns:
            candidate = pd.to_numeric(dfin[col], errors="coerce")
            amount = amount.combine_first(candidate)
    return amount.fillna(0.0)


def _agg_block(d: pd.DataFrame, mask: pd.Series) -> Tuple[float, int]:
    sub = d.loc[mask] if isinstance(mask, pd.Series) else d
    amount_col = "importe_pendiente" if "importe_pendiente" in sub.columns else "importe_deuda"
    monto = float(pd.to_numeric(sub.get(amount_col), errors="coerce").fillna(0.0).sum())
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


def _prioritize_documents(dfin: pd.DataFrame, estados: list[str]) -> pd.DataFrame:
    out = dfin.copy()
    if out.empty:
        return out

    estados = estados or []
    dias = pd.to_numeric(out.get("dias_a_vencer"), errors="coerce")
    out["dias_a_vencer"] = dias
    out["vencida_flag"] = dias < 0

    importe = pd.to_numeric(out.get("importe_deuda"), errors="coerce").fillna(0.0)
    out["importe_regla"] = importe

    estado_order_map = {estado: idx for idx, estado in enumerate(estados)}
    default_estado_order = len(estado_order_map)
    if "estado_cuota" in out:
        out["_estado_order"] = (
            out["estado_cuota"].astype(str).map(estado_order_map).fillna(default_estado_order)
        )
    else:
        out["_estado_order"] = default_estado_order

    sort_cols = ["_estado_order", "estado_cuota", "dias_a_vencer", "importe_regla"]
    ascending = [True, True, True, False]
    existing_cols = [c for c in sort_cols if c in out.columns]
    if existing_cols:
        asc = [ascending[sort_cols.index(c)] for c in existing_cols]
        out = out.sort_values(by=existing_cols, ascending=asc)

    return out.drop(columns=["_estado_order"], errors="ignore")


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
    "dias_a_vencer",
    "Dias_Hasta_Cuota",
    "codigo_estado_cuota",
    "estado_cuota",
    "codigo_centro",
    "monto_cuota",
    "cuenta_especial",
    "banco",
    "cuenta_corriente",
    "centro_costo_costeo",
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
            "banco": "Banco",
            "cuenta_corriente": "Cuenta Corriente",
            "centro_costo_costeo": "Centro Costo Costeo",
        }
    )
    # Asegurar columna Dias_Hasta_Cuota en exporte y remover campo técnico
    if "Dias_Hasta_Cuota" not in show.columns and "dias_a_vencer" in show.columns:
        _vexp = pd.to_numeric(show["dias_a_vencer"], errors="coerce")
        show["Dias_Hasta_Cuota"] =  _vexp
    if "dias_a_vencer" in show.columns:
        show = show.drop(columns=["dias_a_vencer"])  # no exportar nombre técnico

    for col in (
        "Inicio Convenio",
        "Término Convenio",
        "Fecha Emisión",
        "Fecha Cuota",

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
        # Ordenar columnas como el PDF (y omitir las no listadas)
    order_pdf_like = [
        "RUT",
        "Inicio Convenio",
        "T?rmino Convenio",
        "Fecha Emisi??n",
        "Fecha Cuota",
        "Dias_Hasta_Cuota",
        "Estado Cuota",
        "C??digo Centro",
        "Monto Cuota",
        "Cuenta Especial",
        "Banco",
        "Cuenta Corriente",
        "Centro Costo Costeo",
    ]
    existing = [c for c in order_pdf_like if c in show.columns]
    if existing:
        show = show[existing]
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
            "banco": "Banco",
            "cuenta_corriente": "Cuenta Corriente",
            "centro_costo_costeo": "Centro Costo Costeo",
        }
    )
    # Garantizar columna Dias_Hasta_Cuota en exporte y remover campo técnico
    if "Dias_Hasta_Cuota" not in out.columns and "dias_a_vencer" in out.columns:
        _vexp = pd.to_numeric(out["dias_a_vencer"], errors="coerce")
        out["Dias_Hasta_Cuota"] = _vexp
    if "dias_a_vencer" in out.columns:
        out = out.drop(columns=["dias_a_vencer"])  # no exportar nombre técnico
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
    # Ordenar columnas como el PDF (y omitir las no listadas)
    order_pdf_like = [
        "RUT",
        "Inicio Convenio",
        "TǸrmino Convenio",
        "Fecha Emisi��n",
        "Fecha Cuota",
        "Dias_Hasta_Cuota",
        "Estado Cuota",
        "C��digo Centro",
        "Monto Cuota",
        "Cuenta Especial",
        "Banco",
        "Cuenta Corriente",
        "Centro Costo Costeo",
    ]
    _exist = [c for c in order_pdf_like if c in out.columns]
    if _exist:
        out = out[_exist]
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

monto_pagado_ce = _sum_numeric(pagadas_ce_df, ["monto_cuota", "monto_pagado", "liquido_cuota", "fac_monto_total"])
monto_pagado_no = max(0.0, monto_pagado_total - monto_pagado_ce)
monto_no_pagado_ce = _sum_numeric(no_pagadas_ce_df, ["monto_cuota", "monto_autorizado", "fac_monto_total"])
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
    _render_metric_block("Monto cuota (pagadas)", money(monto_pagado_total), breakdown_html=breakdown)
with amount_cols[1]:
    breakdown = _amount_html(monto_no_pagado_ce, monto_no_pagado_no) if ce_available else None
    _render_metric_block("Monto no pagado", money(monto_no_pagado_total), breakdown_html=breakdown)

# ---------------------------------------------------------
# Tipología de pagos (solo PAGADAS) basada en emisión, cuota y CE
# ---------------------------------------------------------
if count_pagadas:
    em = pd.to_datetime(pagadas.get("fecha_emision_ref"), errors="coerce")
    cu = pd.to_datetime(pagadas.get("fecha_cuota_ref"), errors="coerce")
    ce_src = pd.to_datetime(pagadas.get("fecha_ce"), errors="coerce") if "fecha_ce" in pagadas.columns else pd.Series(pd.NaT, index=pagadas.index)
    ce = ce_src.fillna(pd.to_datetime(pagadas.get("fecha_pago_ref"), errors="coerce"))

    em_lt_cu = (em < cu)
    em_eq_cu = (em == cu)
    em_gt_cu = (em > cu)
    ce_lt_cu = (ce < cu)
    ce_eq_cu = (ce == cu)
    ce_gt_cu = (ce > cu)
    ce_eq_em = (ce == em)
    ce_gt_em = (ce > em)
    ce_lt_em = (ce < em)

    # Pago a tiempo vs atrasado y tipo de emisión
    pago_tiempo = np.where(ce_le_cu := (ce <= cu), "A tiempo", "Atrasado")
    tipo_emision_lbl = np.where(em_gt_cu, "Emision vencida", np.where((em.notna() & cu.notna()), "Emision valida", "Sin info"))

    # Fecha de pago efectiva (para pagadas): max(ce, cuota)
    fecha_pago_efectiva = np.where((ce_gt_cu) & ce.notna() & cu.notna(), ce, cu)
    fecha_pago_efectiva = pd.to_datetime(fecha_pago_efectiva, errors="coerce")

    # Clasificación global según condiciones provistas
    conds = [
        # C1: em<cu  ce<cu
        (em_lt_cu & ce_lt_cu),
        # C2: em<cu  ce=cu
        (em_lt_cu & ce_eq_cu),
        # C3: em<cu  ce>cu
        (em_lt_cu & ce_gt_cu),
        # C4: em=cu  ce<cu
        (em_eq_cu & ce_lt_cu),
        # C5: em=cu  ce=cu
        (em_eq_cu & ce_eq_cu),
        # C6: em=cu  ce>cu
        (em_eq_cu & ce_gt_cu),
        # C7: em>cu  ce<=cu
        (em_gt_cu & (ce <= cu)),
        # C8: em>cu  ce=em  ce>cu
        (em_gt_cu & ce_eq_em & ce_gt_cu),
        # C9: em>cu  ce>em  ce>cu (extemporaneo total)
        (em_gt_cu & ce_gt_em & ce_gt_cu),
        # C10: em>cu  ce<em  ce>cu (pago previo a emision)
        (em_gt_cu & ce_lt_em & ce_gt_cu),
    ]
    labels = [
        "Flujo normal de pago - Pago anticipado",
        "Flujo normal de pago - Pago mismo dia de fecha cuota",
        "Pago tardio - Emision valida",
        "Flujo normal de pago - Pago anticipado",
        "Flujo normal de pago - Pago mismo dia de fecha cuota",
        "Pago tardio - Emision valida",
        "Pago a tiempo - Emision vencida",
        "Fuera de plazo - CE coincide con emision",
        "Extemporaneo total",
        "Pago previo a emision",
    ]
    clas_global = np.select(conds, labels, default="Caso no clasificado")

    extemporaneo = np.select(
        [ (em_gt_cu & ce_gt_em & ce_gt_cu), (em_gt_cu & ce_lt_em & ce_gt_cu) ],
        [ True, True ],
        default=False,
    )

    pagadas["tipo_emision_label"] = tipo_emision_lbl
    pagadas["pago_tiempo_label"] = pago_tiempo
    pagadas["fecha_pago_efectiva"] = fecha_pago_efectiva
    pagadas["clasificacion_global"] = clas_global
    pagadas["extemporaneo"] = extemporaneo
    pagadas["monto_cuota_eff"] = _quota_amount_series(pagadas)

    # (Eliminado) Resumen por tipo de emision y pago a tiempo
    # Se omite la tabla "Tipologia de pagos (PAGADAS)" porque la detallada es suficiente.

    # Resumen por clasificacion global
    resumen_global = (
        pagadas.groupby(["clasificacion_global", "tipo_emision_label"], dropna=False)
        .agg(Boletas=("monto_cuota_eff", "count"), Monto_Cuota=("monto_cuota_eff", "sum"))
        .reset_index()
        .rename(columns={
            "clasificacion_global": "Clasificacion",
            "tipo_emision_label": "Tipo de Emision",
        })
    )
    if not resumen_global.empty:
        disp_g = resumen_global.copy()
        # Agregar columna de condiciones de fecha segun Clasificacion (reglas generales por grupo)
        cond_map = {
            "Flujo normal de pago - Pago anticipado": "emision  cuota  ce < cuota".replace("\u007f", "<="),
            "Flujo normal de pago - Pago mismo dia de fecha cuota": "emision  cuota  ce = cuota".replace("\u007f", "<="),
            "Pago tardio - Emision valida": "emision  cuota  ce > cuota".replace("\u007f", "<="),
            "Pago a tiempo - Emision vencida": "emision > cuota  ce  cuota".replace("\u007f", "<="),
            "Fuera de plazo - CE coincide con emision": "emision > cuota  ce = emision  ce > cuota".replace("\u007f", "<="),
            "Extemporaneo total": "emision > cuota  ce > emision  ce > cuota".replace("\u007f", "<="),
            "Pago previo a emision": "emision > cuota  ce < emision  ce > cuota".replace("\u007f", "<="),
        }
        disp_g["Condiciones de fecha"] = disp_g["Clasificacion"].map(cond_map).fillna("")
        # Reescritura en formato con separadores ';' para mayor claridad
        cond_map2 = {
            "Flujo normal de pago - Pago anticipado": "emision <= cuota; ce < cuota",
            "Flujo normal de pago - Pago mismo dia de fecha cuota": "emision <= cuota; ce = cuota",
            "Pago tardio - Emision valida": "emision <= cuota; ce > cuota",
            "Pago a tiempo - Emision vencida": "emision > cuota; ce <= cuota",
            "Fuera de plazo - CE coincide con emision": "emision > cuota; ce = emision; ce > cuota",
            "Extemporaneo total": "emision > cuota; ce > emision; ce > cuota",
            "Pago previo a emision": "emision > cuota; ce < emision; ce > cuota",
        }
        disp_g["Condiciones de fecha"] = disp_g["Clasificacion"].map(cond_map2).fillna("")

        # Agregar columna Pago (A tiempo / Atrasado) segun la Clasificacion
        pago_map = {
            "Flujo normal de pago - Pago anticipado": "A tiempo",
            "Flujo normal de pago - Pago mismo dia de fecha cuota": "A tiempo",
            "Pago a tiempo - Emision vencida": "A tiempo",
            "Pago tardio - Emision valida": "Atrasado",
            "Fuera de plazo - CE coincide con emision": "Atrasado",
            "Extemporaneo total": "Atrasado",
            "Pago previo a emision": "Atrasado",
        }
        disp_g["Pago"] = disp_g["Clasificacion"].map(pago_map).fillna("")

        # Ordenar por Tipo de Emision (Valida primero, luego Vencida) y luego por Clasificacion
        order_map = {"Emision valida": 0, "Emision vencida": 1}
        disp_g["__ord"] = disp_g["Tipo de Emision"].map(order_map).fillna(99)
        disp_g = disp_g.sort_values(by=["__ord", "Clasificacion"]).drop(columns=["__ord"]) 
        disp_g["Monto_Cuota"] = disp_g["Monto_Cuota"].map(money)
        # Reordenar columnas: Tipo de Emision, Pago, Clasificacion y luego condiciones
        cols_order = ["Tipo de Emision", "Pago", "Clasificacion", "Condiciones de fecha", "Boletas", "Monto_Cuota"]
        existing = [c for c in cols_order if c in disp_g.columns]
        disp_g = disp_g[existing + [c for c in disp_g.columns if c not in existing]]
        disp_g = sanitize_df(disp_g)
        st.markdown("#### Tipologia detallada (Clasificacion Global)")
        style_table(_table_style(disp_g))
        st.download_button(
            "Descargar Tipologia Detallada",
            data=excel_bytes_single(
                disp_g.rename(columns={"Monto_Cuota": "Monto_Cuota"}),
                "Tipologia_Clasificacion_Pagadas"
            ),
            file_name="pagadas_tipologia_clasificacion.xlsx",
            disabled=resumen_global.empty,
        )

if count_pagadas:
    pagadas["tiempo_pago_planeado"] = (pagadas["fecha_cuota_ref"] - pagadas["fecha_emision_ref"]).dt.days
    pagadas["tiempo_pago_real"] = (pagadas["fecha_pago_ref"] - pagadas["fecha_emision_ref"]).dt.days

    pagadas["tiempo_pago_planeado"] = pagadas["tiempo_pago_planeado"].clip(lower=0)
    if mask_especial.any():
        ajuste_especial = (pagadas.loc[mask_especial, "fecha_cuota_ref"] - pagadas.loc[mask_especial, "fecha_emision_ref"]).dt.days
        pagadas.loc[mask_especial, "tiempo_pago_real"] = ajuste_especial
    pagadas["tiempo_pago_real"] = pagadas["tiempo_pago_real"].clip(lower=0)

    st.markdown("#### Cumplimiento de pagos")
    pie_cols = st.columns(3)

    with pie_cols[0]:
        st.markdown("**A tiempo vs Atrasado (tipologia)**")
        if (count_en_plazo + count_atraso) == 0:
            st.info("Sin datos suficientes.")
        else:
            vals_tip = [int(count_en_plazo), int(count_atraso)]
            fig_pie2 = go.Figure(
                data=[
                    go.Pie(
                        labels=["A tiempo", "Atrasado"],
                        values=vals_tip,
                        textinfo="label+percent",
                        hole=0.45,
                        marker=dict(colors=["#59A14F", "#E15759"]),
                        showlegend=False,
                    )
                ]
            )
            fig_pie2.update_layout(
                margin=dict(l=0, r=0, t=20, b=0), height=260,
            )
            st.plotly_chart(fig_pie2, use_container_width=True)

    with pie_cols[1]:
        st.markdown("**Pagadas dentro de plazo por cuenta**")
        total_in_plazo_ce = en_plazo_ce_count + en_plazo_no_ce_count
        if total_in_plazo_ce == 0 or not ce_available:
            st.info("Sin datos de CE/No CE.")
        else:
            fig_in_plazo = go.Figure(
                data=[
                    go.Pie(
                        labels=["Cuenta especial", "No cuenta especial"],
                        values=[en_plazo_ce_count, en_plazo_no_ce_count],
                        textinfo="label+percent",
                        hole=0.45,
                        marker=dict(colors=CE_COLORS),
                        showlegend=False,
                    )
                ]
            )
            fig_in_plazo.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=260)
            st.plotly_chart(fig_in_plazo, use_container_width=True)

    with pie_cols[2]:
        st.markdown("**Pagadas con atraso por cuenta**")
        total_atraso_ce = atraso_ce_count + atraso_no_ce_count
        if total_atraso_ce == 0 or not ce_available:
            st.info("Sin datos de CE/No CE.")
        else:
            fig_atraso = go.Figure(
                data=[
                    go.Pie(
                        labels=["Cuenta especial", "No cuenta especial"],
                        values=[atraso_ce_count, atraso_no_ce_count],
                        textinfo="label+percent",
                        hole=0.45,
                        marker=dict(colors=CE_COLORS),
                        showlegend=False,
                    )
                ]
            )
            fig_atraso.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=260)
            st.plotly_chart(fig_atraso, use_container_width=True)

    st.markdown("#### Histogramas de tiempos (solo PAGADAS)")
    # NUEVO: Histograma de tiempo = fecha_pago_efectiva - fecha_emision, con filtros y promedios general/filtro
    try:
        tpe_all = (pd.to_datetime(pagadas.get("fecha_pago_efectiva"), errors="coerce") - pd.to_datetime(pagadas.get("fecha_emision_ref"), errors="coerce")).dt.days.dropna()
        # Orden de filtros: CE -> Pago -> Emision -> Clasificacion
        cc1, cc2, cc3, cc4 = st.columns((1.2, 1.4, 1.6, 1.8))
        # 1) Cuenta especial
        with cc1:
            ce_val_new = st.radio("Cuenta especial", ["Todas", "Cuenta especial", "No cuenta especial"], horizontal=True, index=0, key="hon_hist_ce_new")

        # Base para construir opciones dependientes
        df_opts = pagadas.copy()
        if ce_available:
            if ce_val_new == "Cuenta especial":
                df_opts = df_opts.loc[pagadas_ce_flag]
            elif ce_val_new == "No cuenta especial":
                df_opts = df_opts.loc[~pagadas_ce_flag]

        # 2) Tipo de pago (A tiempo / Atrasado) basado en CE
        pago_values = []
        if "pago_tiempo_label" in df_opts:
            pago_values = sorted([x for x in df_opts["pago_tiempo_label"].dropna().unique().tolist() if x in ("A tiempo", "Atrasado")])
        pago_opts = ["Todos"] + pago_values
        with cc2:
            pago_sel = st.selectbox("Pago", pago_opts, index=0, key="hon_hist_pago")
        if pago_sel != "Todos" and "pago_tiempo_label" in df_opts:
            df_opts = df_opts[df_opts["pago_tiempo_label"] == pago_sel]

        # 3) Tipo de Emision dependiente de CE y Pago
        tipo_values = []
        if "tipo_emision_label" in df_opts:
            tipo_values = sorted(df_opts["tipo_emision_label"].dropna().unique().tolist())
        tipo_opts_new = ["Todos"] + tipo_values
        with cc3:
            tipo_sel_new = st.selectbox("Tipo de Emision", tipo_opts_new, index=0, key="hon_hist_tipo")
        if tipo_sel_new != "Todos" and "tipo_emision_label" in df_opts:
            df_opts = df_opts[df_opts["tipo_emision_label"] == tipo_sel_new]

        # 4) Clasificacion dependiente de CE, Pago y Emision
        class_values = []
        if "clasificacion_global" in df_opts:
            class_values = sorted(df_opts["clasificacion_global"].dropna().unique().tolist())
        class_opts_new = ["Todos"] + class_values if class_values else ["Todos"]
        with cc4:
            class_sel_new = st.selectbox("Clasificacion", class_opts_new, index=0, key="hon_hist_cls")

        # Ajustar coherencia: si la clasificacion seleccionada deja sin opciones de Pago, volver a "Todos"
        df_check = df_opts.copy()
        if class_sel_new != "Todos" and "clasificacion_global" in df_check:
            df_check = df_check[df_check["clasificacion_global"] == class_sel_new]
        if "pago_tiempo_label" in df_check:
            pago_after_cls = set([x for x in df_check["pago_tiempo_label"].dropna().unique().tolist() if x in ("A tiempo", "Atrasado")])
            if pago_sel != "Todos" and pago_sel not in pago_after_cls:
                st.session_state["hon_hist_pago"] = "Todos"
                pago_sel = "Todos"
        cbin, cmax = st.columns((1,1))
        with cbin:
            bin_size_new = st.slider("Ancho de columnas (dias)", 1, 60, 2)
        max_domain_new = int(max(1, np.ceil(np.nanmax(tpe_all)))) if not tpe_all.empty else 1
        with cmax:
            # Por defecto 60 dias; si el dominio es menor, usar el maximo disponible
            default_max = min(60, int(max_domain_new)) if int(max_domain_new) >= 60 else int(max_domain_new)
            max_days_new = st.slider("Max dias a mostrar", 1, int(max_domain_new), default_max)
        mask_new = pd.Series(True, index=pagadas.index)
        if ce_available:
            if ce_val_new == "Cuenta especial":
                mask_new &= pagadas_ce_flag
            elif ce_val_new == "No cuenta especial":
                mask_new &= ~pagadas_ce_flag
        if "pago_tiempo_label" in pagadas and pago_sel != "Todos":
            mask_new &= (pagadas["pago_tiempo_label"] == pago_sel)
        if "tipo_emision_label" in pagadas and tipo_sel_new != "Todos":
            mask_new &= (pagadas["tipo_emision_label"] == tipo_sel_new)
        if "clasificacion_global" in pagadas and class_sel_new != "Todos":
            mask_new &= (pagadas["clasificacion_global"] == class_sel_new)
        tpe_fil = (pd.to_datetime(pagadas.loc[mask_new, "fecha_pago_efectiva"], errors="coerce") - pd.to_datetime(pagadas.loc[mask_new, "fecha_emision_ref"], errors="coerce")).dt.days.dropna()
        tpe_all_vis = tpe_all[tpe_all <= max_days_new]
        tpe_fil_vis = tpe_fil[tpe_fil <= max_days_new]
        mean_all = float(np.nanmean(tpe_all)) if not tpe_all.empty else float("nan")
        mean_fil = float(np.nanmean(tpe_fil)) if not tpe_fil.empty else float("nan")
        fig_new = go.Figure()
        # Un solo color de barra acorde al proyecto (azul)
        base_color = "#4f9cff"
        source_vals = tpe_fil_vis if not tpe_fil_vis.empty else tpe_all_vis
        fig_new.add_histogram(
            x=source_vals,
            xbins=dict(size=bin_size_new),
            name="Distribucion",
            opacity=0.95,
            marker=dict(color=base_color, line=dict(width=0)),
        )
        fig_new.update_layout(
            bargap=0.08,
            height=600,
            margin=dict(l=28, r=28, t=36, b=36),
            template="plotly_white",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#1f2a55"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            xaxis_title="Dias",
            yaxis_title="Cantidad de honorarios",
        )
        fig_new.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.35)", zeroline=False, linecolor="#1f2a55")
        fig_new.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.35)", zeroline=False, linecolor="#1f2a55")
        if not np.isnan(mean_all):
            fig_new.add_vline(x=mean_all, line_color="#3f51b5", line_width=2, line_dash="dash",
                              annotation=dict(text=f"Prom gral {mean_all:.1f}d", font=dict(color="#3f51b5"), bgcolor="#ffffff"))
        if not np.isnan(mean_fil):
            fig_new.add_vline(x=mean_fil, line_color="#2ecc71", line_width=2, line_dash="dot",
                              annotation=dict(text=f"Prom filtro {mean_fil:.1f}d", font=dict(color="#2ecc71"), bgcolor="#ffffff"))
        st.plotly_chart(fig_new, use_container_width=True)
        # Indicadores como tarjetas horizontales (estilo app-card)
        def _fmt_days(x: float) -> str:
            try:
                return f"{float(x):.1f} d"
            except Exception:
                return "s/d"

        cards_metrics: list[str] = []
        # Conteos dentro y fuera del rango (filtro de max_dias solo aplica a estas tarjetas)
        base_vals = tpe_fil if not tpe_fil.empty else tpe_all
        base_vals = pd.to_numeric(base_vals, errors="coerce").dropna()
        n_total = int(len(base_vals))
        n_le = int((base_vals <= max_days_new).sum())
        n_gt = max(0, n_total - n_le)
        cards_metrics.append(_card_html("Total observaciones", f"{n_total:,}", compact=True))
        cards_metrics.append(_card_html(f"≤ {max_days_new} d", f"{n_le:,}", compact=True))
        cards_metrics.append(_card_html(f"> {max_days_new} d", f"{n_gt:,}", compact=True))
        cards_metrics.append(_card_html("Prom general", _fmt_days(mean_all) if not np.isnan(mean_all) else "s/d", compact=True))
        cards_metrics.append(_card_html("Prom filtro", _fmt_days(mean_fil) if not np.isnan(mean_fil) else "s/d", compact=True, tone="accent"))

        if not tpe_fil.empty:
            p50 = float(np.nanpercentile(tpe_fil, 50))
            p75 = float(np.nanpercentile(tpe_fil, 75))
            p90 = float(np.nanpercentile(tpe_fil, 90))
            cards_metrics.append(_card_html("P50 (filtro)", _fmt_days(p50), compact=True))
            cards_metrics.append(_card_html("P75 (filtro)", _fmt_days(p75), compact=True))
            cards_metrics.append(_card_html("P90 (filtro)", _fmt_days(p90), compact=True))

        _render_cards(cards_metrics, layout="grid")
    except Exception:
        pass
    # LEGACY HISTOGRAM BLOCK DISABLED
# legacy removed
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
        df_nopag_all["margen_emision"] = (
            pd.to_datetime(df_nopag_all.get("fecha_cuota_ref"), errors="coerce")
            - pd.to_datetime(df_nopag_all.get("fecha_emision_ref"), errors="coerce")
        ).dt.days
    else:
        df_nopag_all["clasificacion_plazo"] = "Sin información de fechas"
        df_nopag_all["dias_sin_pago"] = np.nan
        df_nopag_all["dias_para_cuota"] = np.nan
        df_nopag_all["margen_emision"] = np.nan

    df_nopag_all["importe_pendiente"] = _resolve_pending_amount(df_nopag_all)
    df_nopag_all["importe_deuda"] = df_nopag_all["importe_pendiente"]
    df_nopag_all["dias_a_vencer"] = pd.to_numeric(df_nopag_all.get("dias_para_cuota"), errors="coerce")
    df_nopag_all["monto_cuota_num"] = df_nopag_all["importe_pendiente"]
else:
    df_nopag_all = df_nopag_all.iloc[0:0]

if df_nopag_all.empty:
    st.info("No hay honorarios pendientes de pago.")
else:
    estado_values_source_s3 = (
        df_nopag_all["estado_cuota"]
        if "estado_cuota" in df_nopag_all.columns
        else pd.Series(dtype=str)
    )
    estado_options_s3 = sorted(
        pd.Series(estado_values_source_s3)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    default_estados_s3: List[str] = []
    if "CONTABILIZADA" in estado_options_s3:
        default_estados_s3 = ["CONTABILIZADA"]
    elif estado_options_s3:
        default_estados_s3 = [estado_options_s3[0]]

    _ESTADO_FILTER_KEY_S3 = "honorarios_s3_estado_multiselect"
    if _ESTADO_FILTER_KEY_S3 not in st.session_state:
        st.session_state[_ESTADO_FILTER_KEY_S3] = default_estados_s3
    else:
        current_selection_s3 = st.session_state.get(_ESTADO_FILTER_KEY_S3, [])
        if estado_options_s3:
            valid_selection_s3 = [
                opt for opt in current_selection_s3 if opt in estado_options_s3
            ]
            if len(valid_selection_s3) != len(current_selection_s3):
                st.session_state[_ESTADO_FILTER_KEY_S3] = (
                    valid_selection_s3 if valid_selection_s3 else default_estados_s3
                )

    filtro_estado_col, _, _ = st.columns([1.6, 1, 1])
    with filtro_estado_col:
        label_col_s3, add_col_s3, remove_col_s3 = st.columns([0.7, 0.15, 0.15])
        with label_col_s3:
            st.markdown("**Estado de cuota (orden)**")
        selected_estados_state_s3 = st.session_state.get(
            _ESTADO_FILTER_KEY_S3, default_estados_s3
        )
        available_extra_states_s3 = [
            opt for opt in estado_options_s3 if opt not in selected_estados_state_s3
        ]
        with add_col_s3:
            add_clicked_s3 = st.button(
                "➕",
                key="honorarios_s3_estado_add",
                help="Agregar otro estado de cuota al filtro y orden",
                disabled=len(available_extra_states_s3) == 0,
            )
        with remove_col_s3:
            remove_clicked_s3 = st.button(
                "➖",
                key="honorarios_s3_estado_remove",
                help="Quitar el último estado del filtro",
                disabled=len(selected_estados_state_s3) <= 1,
            )
        if add_clicked_s3 and available_extra_states_s3:
            st.session_state[_ESTADO_FILTER_KEY_S3] = selected_estados_state_s3 + [
                available_extra_states_s3[0]
            ]
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        if remove_clicked_s3 and len(selected_estados_state_s3) > 1:
            st.session_state[_ESTADO_FILTER_KEY_S3] = selected_estados_state_s3[:-1]
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        selected_estados_s3 = st.multiselect(
            "Estado de cuota (orden)",
            estado_options_s3,
            default=st.session_state.get(_ESTADO_FILTER_KEY_S3, default_estados_s3),
            key=_ESTADO_FILTER_KEY_S3,
            label_visibility="collapsed",
            placeholder="Selecciona uno o más estados",
        )
        if estado_options_s3:
            st.caption(
                "Usa los botones ➕/➖ para ajustar el orden de prioridad de los estados."
            )

    selected_estados_s3 = (
        selected_estados_s3
        if selected_estados_s3
        else st.session_state.get(_ESTADO_FILTER_KEY_S3, default_estados_s3)
    )

    df_section3_base = df_nopag_all.copy()
    if selected_estados_s3 and "estado_cuota" in df_section3_base.columns:
        df_section3_base = df_section3_base[
            df_section3_base["estado_cuota"].isin(selected_estados_s3)
        ].copy()

    resumen_filter = st.radio(
        "Tipo de cuenta especial (resumen)",
        ["Todas", "Cuenta especial", "No cuenta especial"],
        horizontal=True,
        index=0,
    )

    if resumen_filter == "Cuenta especial":
        df_resumen = df_section3_base[df_section3_base["cuenta_especial"] == True]
    elif resumen_filter == "No cuenta especial":
        df_resumen = df_section3_base[
            df_section3_base["cuenta_especial"] == False
        ]
    else:
        df_resumen = df_section3_base

    categorias_visibles = [
        "Emisión posterior a cuota",
        "Emisión anterior o igual a cuota",
    ]
    tooltip_plazo = (
        "Días sin pago = hoy - fecha de emisión. "
        "Días hasta cuota = fecha de cuota - hoy (negativo indica atraso). "
        "Margen al emitir = fecha de cuota - fecha de emisión; si es negativo, la cuota nace con atraso."
    )
    cards_categoria: list[str] = []
    cards_sin_info: list[str] = []
    categorias_iterables = categorias_visibles + ["Sin información de fechas"]
    for label in categorias_iterables:
        subset = df_resumen[df_resumen["clasificacion_plazo"] == label]
        if subset.empty:
            continue
        for flag, flag_label in ((True, "Cuenta especial"), (False, "No cuenta especial")):
            if "cuenta_especial" in subset.columns:
                sub = subset[subset["cuenta_especial"] == flag]
            else:
                sub = subset if flag else subset.iloc[0:0]
            if sub.empty:
                continue
            amount_series = (
                pd.to_numeric(sub.get("importe_pendiente"), errors="coerce").fillna(0.0)
                if "importe_pendiente" in sub
                else pd.Series(dtype=float)
            )
            total_monto = float(amount_series.sum())
            total_docs = int(len(sub))
            dias_prom_pago = _avg_days_from_series(sub.get("dias_sin_pago"))
            dias_prom_cuota = _avg_days_from_series(sub.get("dias_para_cuota"))
            dias_prom_margen = _avg_days_from_series(sub.get("margen_emision"))

            stats = [
                ("Prom. días sin pago", _fmt_days(dias_prom_pago)),
                ("Prom. días hasta cuota", _fmt_days(dias_prom_cuota)),
                ("Margen al emitir", _fmt_days(dias_prom_margen)),
            ]
            card_html = _card_html(
                title=f"{label}  {flag_label}",
                value=money(total_monto),
                subtitle=f"{total_docs:,} honorarios sin pagar",
                stats=stats,
                tone="accent" if label == "Emisión posterior a cuota" else "default",
                compact=False,
                tooltip=tooltip_plazo,
            )
            if label == "Sin información de fechas":
                cards_sin_info.append(card_html)
            else:
                cards_categoria.append(card_html)
    if cards_categoria:
        safe_markdown(
            "<div class='app-title-block'><h3 style='color:#000;'>Clasificación por fecha de cuota</h3>"
            "<p>Desglose basado en la comparación entre emisión y cuota.</p></div>"
        )
        _render_cards(cards_categoria, layout="grid-2")
    if cards_sin_info:
        safe_markdown(
            "<div class='app-title-block'><h3 style='color:#000;'>Sin información de fechas</h3>"
            "<p>Honorarios sin datos completos de emisión o fecha de cuota.</p></div>"
        )
        _render_cards(cards_sin_info, layout="grid-2")

    resumen_tipo = (
        df_resumen.groupby(["clasificacion_plazo", "estado_cuota"], dropna=False)
        .agg(
            Monto_Pendiente=("importe_pendiente", "sum"),
            Honorarios=("importe_pendiente", "count"),
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
        display_tipo = display_tipo.rename(columns={
            "clasificacion_plazo": "Clasificacion",
            "estado_cuota": "Estado Cuota",
            "Monto_Pendiente": "Monto Pendiente",
            "Honorarios": "Cant. Honorarios",
        })
        safe_markdown("""
        <div class="app-note">
          <strong>Como leer estos promedios</strong><br>
          <b>Prom. dias sin pago</b>: dias desde la emision hasta hoy (antiguedad).<br>
          <b>Prom. dias hasta cuota</b>: dias desde hoy hasta la fecha de cuota (negativo = vencido).
        </div>
        """)
        display_tipo = sanitize_df(display_tipo)
        safe_markdown("**Resumen por clasificacion y estado de cuota**")
        style_table(_table_style(display_tipo))

    resumen_ce = (
        df_resumen.groupby(["clasificacion_plazo", "cuenta_especial"], dropna=False)
        .agg(
            Monto_Pendiente=("importe_pendiente", "sum"),
            Honorarios=("importe_pendiente", "count"),
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

col_estado, col_cuenta, col_prioritario = st.columns([1.6, 1, 1])

estado_values_source = None
if "estado_cuota" in df_nopag_all.columns and not df_nopag_all.empty:
    estado_values_source = df_nopag_all["estado_cuota"]
elif "estado_cuota" in df.columns:
    estado_values_source = df["estado_cuota"]
else:
    estado_values_source = pd.Series(dtype=str)

estado_options = sorted(
    pd.Series(estado_values_source).dropna().astype(str).unique().tolist()
)
default_estados = []
if "CONTABILIZADA" in estado_options:
    default_estados = ["CONTABILIZADA"]
elif estado_options:
    default_estados = [estado_options[0]]

_ESTADO_FILTER_KEY = "hon_estado_multiselect"
if _ESTADO_FILTER_KEY not in st.session_state:
    st.session_state[_ESTADO_FILTER_KEY] = default_estados
else:
    current_selection = st.session_state.get(_ESTADO_FILTER_KEY, [])
    if estado_options:
        valid_selection = [opt for opt in current_selection if opt in estado_options]
        if len(valid_selection) != len(current_selection):
            st.session_state[_ESTADO_FILTER_KEY] = (
                valid_selection if valid_selection else default_estados
            )

with col_estado:
    label_col, add_col, remove_col = st.columns([0.7, 0.15, 0.15])
    with label_col:
        st.markdown("**Estado de cuota (orden)**")
    selected_estados_state = st.session_state.get(_ESTADO_FILTER_KEY, default_estados)
    available_extra_states = [
        opt for opt in estado_options if opt not in selected_estados_state
    ]
    with add_col:
        add_clicked = st.button(
            "➕",
            key="hon_estado_add",
            help="Agregar otro estado de cuota al filtro y orden",
            disabled=len(available_extra_states) == 0,
        )
    with remove_col:
        remove_clicked = st.button(
            "➖",
            key="hon_estado_remove",
            help="Quitar el último estado del filtro",
            disabled=len(selected_estados_state) <= 1,
        )
    if add_clicked and available_extra_states:
        st.session_state[_ESTADO_FILTER_KEY] = selected_estados_state + [
            available_extra_states[0]
        ]
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    if remove_clicked and len(selected_estados_state) > 1:
        st.session_state[_ESTADO_FILTER_KEY] = selected_estados_state[:-1]
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    selected_estados = st.multiselect(
        "Estado de cuota (orden)",
        estado_options,
        default=st.session_state.get(_ESTADO_FILTER_KEY, default_estados),
        key=_ESTADO_FILTER_KEY,
        label_visibility="collapsed",
        placeholder="Selecciona uno o más estados",
    )
    if estado_options:
        st.caption(
            "Usa los botones ➕/➖ para ajustar el orden de prioridad de los estados."
        )

selected_estados = (
    selected_estados
    if selected_estados
    else st.session_state.get(_ESTADO_FILTER_KEY, default_estados)
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
if selected_estados and "estado_cuota" in df_nopag_loc.columns:
    df_nopag_loc = df_nopag_loc[df_nopag_loc["estado_cuota"].isin(selected_estados)].copy()

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
candidatas_prior = _prioritize_documents(candidatas_base, selected_estados)

# Asegurar columna de prioridad de tiempo para tablas (negativo = vencido)
if "dias_a_vencer" in candidatas_prior.columns:
    _vals = pd.to_numeric(candidatas_prior["dias_a_vencer"], errors="coerce")
    candidatas_prior["Dias_Hasta_Cuota"] = pd.array(
        [pd.NA if pd.isna(v) else int(v) for v in _vals], dtype="Int64"
    )


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

    local_filters = {
        "ce": ce_local,
        "prio": prio_local,
        "estado": tuple(selected_estados),
    }
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
                    button_label = " Agregar honorario extra"
                    if st.button(
                        button_label,
                        key="hon_add_extra_invoice",
                        help=(
                            f"{next_info['proveedor']}  Documento {next_info['documento']} por {money(next_info['monto'])}. "
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
                        f"<p class='budget-panel__resume-note'>Próximo: {next_info['proveedor']}  Documento {next_info['documento']} ({money(next_info['monto'])})</p>"
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
                        " Agregar todos los honorarios pendientes",
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
        if isinstance(seleccion, pd.DataFrame) and "dias_a_vencer" in seleccion.columns:
            _vals2 = pd.to_numeric(seleccion["dias_a_vencer"], errors="coerce")
            seleccion["Dias_Hasta_Cuota"] = pd.array(
                [pd.NA if pd.isna(v) else int(v) for v in _vals2], dtype="Int64"
            )

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
                    subtitle=f"{next_info['proveedor']}  Documento {next_info['documento']}",
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
    st.markdown("**Candidatas a Pago (todas las cuotas  ordenadas por prioridad)**")
    style_table(_table_style(candidatas_display), visible_rows=15)
    st.download_button(
        "️ Descargar Candidatas",
        data=excel_bytes_single(_prep_export(candidatas_prior), "HonorariosCandidatas"),
        file_name="honorarios_candidatas.xlsx",
        disabled=candidatas_prior.empty,
    )

with tab_seleccion:
    safe_markdown(
        """
        <div class="app-note">
            <strong>Selección a pagar hoy</strong>  bloque crítico de honorarios.
        </div>
        """
    )
    style_table(_table_style(seleccion_display), visible_rows=15)
    st.download_button(
        "️ Descargar Selección de Hoy",
        data=excel_bytes_single(_prep_export(seleccion), "HonorariosPagoHoy"),
        file_name="honorarios_pago_hoy.xlsx",
        disabled=seleccion.empty,
    )


# =========================================================
# 6) Reporte PDF
# =========================================================
st.subheader("6) Reporte PDF")
if st.button(" Generar PDF Honorarios"):
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
        honorarios=True,
    )

    st.download_button(
        "Guardar PDF",
        data=pdf_bytes,
        file_name=f"Informe_Honorarios_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )





