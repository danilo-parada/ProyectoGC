# 10_Tablero_KPI.py — KPI con promedios GLOBAL/LOCAL y histogramas alineados
from __future__ import annotations

import html
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.utils import LABELS, TOOLTIPS
from lib_common import (
    get_df_norm, general_date_filters_ui,
    advanced_filters_ui, money, one_decimal, header_ui,
    ESTADO_LABEL, sanitize_df, safe_markdown
)
from lib_metrics import (
    ensure_derived_fields,
    compute_kpis,
    apply_common_filters,
    compute_monto_pagado_real,
    compute_dic_split,
)
from lib_report import excel_bytes_multi

BLUE = "#1f77b4"   # pagadas
RED  = "#e57373"   # no pagadas
PURP = "#8e44ad"   # línea promedio GLOBAL
GRN  = "#2ecc71"

def _metric_card(
    title: str,
    value: str,
    caption: Optional[str] = None,
    *,
    tooltip: Optional[str] = None,
) -> str:
    extra = f'<p class="app-card__subtitle">{caption}</p>' if caption else ""
    tooltip_attr = f' title="{html.escape(tooltip)}"' if tooltip else ""
    return (
        f'<div class="app-card"{tooltip_attr}>'
        f'<div class="app-card__title">{title}</div>'
        f'<div class="app-card__value">{value}</div>'
        f'{extra}'
        '</div>'
    )


def _stat_pill(label: str, value: str) -> str:
    return (
        '<div class="app-inline-stats__item">'
        f'<span class="app-inline-stats__label">{label}:</span> {value}'
        '</div>'
    )


def _segment_card(
    title: str,
    primary: str,
    stats: List[Tuple[str, str]],
    *,
    tooltip: Optional[str] = None,
) -> str:
    pills = "".join(_stat_pill(name, val) for name, val in stats)
    tooltip_attr = f' title="{html.escape(tooltip)}"' if tooltip else ""
    return (
        f'<div class="app-card"{tooltip_attr}>'
        f'<div class="app-card__title">{title}</div>'
        f'<div class="app-card__value">{primary}</div>'
        f'<div class="app-inline-stats">{pills}</div>'
        '</div>'
    )


def _range_card(title: str, value: str) -> str:
    return (
        '<div class="app-range-card">'
        f'<div class="app-range-card__title">{html.escape(title)}</div>'
        f'<div class="app-range-card__value">{html.escape(value)}</div>'
        '</div>'
    )


def _render_range_cards(items: list[tuple[str, str]]):
    if not items:
        return
    cards_html = "".join(_range_card(title, value) for title, value in items)
    safe_markdown('<div class="app-range-card-grid">' + cards_html + '</div>')


def _render_percentile_cards(items: List[Tuple[str, str, Optional[str]]]):
    if not items:
        return
    cards = []
    for label, value, helper in items:
        label_html = html.escape(label)
        value_html = html.escape(value)
        helper_html = (
            f'<div class="app-percentile-helper">{html.escape(helper)}</div>' if helper else ""
        )
        cards.append(
            f'<div class="app-percentile-card">'
            f'<div class="app-percentile-label">{label_html}</div>'
            f'<div class="app-percentile-value">{value_html}</div>'
            f'{helper_html}'
            '</div>'
        )
    safe_markdown('<div class="app-percentile-grid">' + "".join(cards) + '</div>')
   # línea promedio LOCAL

st.set_page_config(page_title="Tablero KPI", layout="wide")
header_ui(
    "Métricas de Pagos y Ciclo de Facturas",
    current_page="Resumen facturas",
    subtitle="Resumen de indicadores generales de facturas",
)

# ===================== Helpers CE robusto =====================
TRUE_TOKENS = {"true", "1", "si", "sí", "s", "y", "t", "yes", "x"}

def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """Convierte Serie a booleana (acepta 1/0, Sí/No, True/False, etc.). NaN -> False."""
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) > 0
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(TRUE_TOKENS)

def _subset_by_ce(dfin: pd.DataFrame, choice: str) -> pd.DataFrame:
    if "cuenta_especial" not in dfin.columns:
        return dfin
    ce_mask = _coerce_bool_series(dfin["cuenta_especial"])
    if choice == "Cuenta Especial":
        return dfin[ce_mask]
    if choice == "No Cuenta Especial":
        return dfin[~ce_mask]
    return dfin

# ===================== Carga base =====================
df0 = get_df_norm()
if df0 is None:
    st.warning("Cargue y mapee datos en 'Carga de Data'.")
    st.stop()

df0 = ensure_derived_fields(df0)

# ===================== Filtros globales =====================
fac_ini, fac_fin, pay_ini, pay_fin = general_date_filters_ui(df0)
sede, org, prov, cc, oc, est, _prio_removed = advanced_filters_ui(
    df0, show_controls=["sede", "org", "prov", "cc", "oc"]
)

common_filters = {
    "fac_range": (fac_ini, fac_fin),
    "pay_range": (pay_ini, pay_fin),
    "sede": sede,
    "org": org,
    "prov": prov,
    "cc": cc,
    "oc": oc,
    "est": est,
    "estado_doc": None,
    "prio": [],
}

df_common_no_estado = apply_common_filters(df0, common_filters).copy()
df_common_no_estado["monto_pagado_real"] = compute_monto_pagado_real(df_common_no_estado)
df_filtered_common = df_common_no_estado.copy()

df = df_common_no_estado

# Particiones útiles
pagadas_series = df.get("monto_pagado_real")
if pagadas_series is None:
    pagadas_series = pd.Series(0.0, index=df.index)
else:
    pagadas_series = pd.to_numeric(pagadas_series, errors="coerce").fillna(0.0)

pagadas_mask_bool = pagadas_series > 0
df_pag = df[pagadas_mask_bool].copy()
df_no_pag = df[~pagadas_mask_bool].copy()

# ===================== KPIs principales =====================
st.subheader("Metricas principales del periodo")

kpi = compute_kpis(df_filtered_common)
dic_split = compute_dic_split(df_filtered_common)

# ===== Base de pagos contabilizados (para DPP/DIC/DCP y promedios globales) =====
pagadas_mask_full = pd.to_numeric(df.get("monto_pagado_real"), errors="coerce") if "monto_pagado_real" in df else pd.Series(0.0, index=df.index)
pagadas_mask_full = pagadas_mask_full.fillna(0.0) > 0
df_pag_contab = df[pagadas_mask_full &
                   (pd.to_datetime(df.get("fecha_cc"), errors="coerce").notna())].copy()

mean_dso_kpi = kpi["dpp"]
mean_tfc_kpi = kpi["dic"]
mean_tpc_kpi = kpi["dcp"]

def _fmt_days_metric(value: float) -> str:
    return "s/d" if pd.isna(value) else f"{one_decimal(value)} días"


def _fmt_dic_avg(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "s/d"
    return f"{one_decimal(value)} días"


def _dic_stats_entries(dic_split: dict) -> list[tuple[str, str]]:
    pagadas_avg = _fmt_dic_avg(dic_split.get("dic_pagadas_avg"))
    pagadas_n = int(dic_split.get("dic_pagadas_n", 0))
    contab_avg = _fmt_dic_avg(dic_split.get("dic_contab_unpaid_avg"))
    contab_n = int(dic_split.get("dic_contab_unpaid_n", 0))
    no_contab_n = int(dic_split.get("no_contab_n", 0))
    return [
        ("Pagadas", f"{pagadas_avg} • {pagadas_n:,} doc."),
        ("Contab. sin pago", f"{contab_avg} • {contab_n:,} doc."),
        ("No contabilizadas", f"{no_contab_n:,} doc."),
    ]


metric_cards = [
    _segment_card(
        "Facturado: Pagado vs Sin pagar",
        money(kpi["total_facturado"]),
        [
            ("Pagado", money(kpi["facturado_pagado"])),
            ("Sin pagar", money(kpi["facturado_sin_pagar"])),
        ],
        tooltip=TOOLTIPS["desglose_facturado"],
    ),
    _metric_card(
        "Total pagado (real)",
        money(kpi["total_pagado_real"]),
        caption=f"Pagadas: {int(kpi['docs_pagados']):,}",
        tooltip=TOOLTIPS["total_pagado_real"],
    ),
    _metric_card(
        LABELS["dpp_emision_pago"],
        _fmt_days_metric(mean_dso_kpi),
        tooltip=TOOLTIPS["dpp_emision_pago"],
    ),
    _metric_card(
        LABELS["dcp_contab_pago"],
        _fmt_days_metric(mean_tpc_kpi),
        tooltip=TOOLTIPS["dcp_contab_pago"],
    ),
]
metric_cards.append(
    _segment_card(
        LABELS["dic_emision_contab"],
        f"{len(df_filtered_common):,} doc.",
        _dic_stats_entries(dic_split),
        tooltip=TOOLTIPS["dic_emision_contab"],
    )
)
safe_markdown('<div class="app-card-grid">' + "".join(metric_cards) + '</div>')

# ====== KPIs por cuenta especial (Si/No) ======
if "cuenta_especial" in df_filtered_common.columns:
    safe_markdown('<div class="app-separator"></div>')
    st.markdown("### Metricas por cuenta especial")

    ce_mask_all = _coerce_bool_series(df_filtered_common["cuenta_especial"])
    segment_cards: list[str] = []
    for flag, title in [(True, "Cuenta especial: si"), (False, "Cuenta especial: no")]:
        sub = df_filtered_common[ce_mask_all] if flag else df_filtered_common[~ce_mask_all]
        if sub.empty:
            continue
        k = compute_kpis(sub)
        dic_split_seg = compute_dic_split(sub)

        stats = [
            ("Documentos", f"{int(k['docs_total']):,}"),
            ("Total pagado (real)", money(k["total_pagado_real"])),
            (LABELS["dpp_emision_pago"], _fmt_days_metric(k["dpp"])),
            (LABELS["dcp_contab_pago"], _fmt_days_metric(k["dcp"])),
        ]
        stats.extend(_dic_stats_entries(dic_split_seg))
        segment_cards.append(
            _segment_card(title, money(k["total_facturado"]), stats)
        )

    if segment_cards:
        safe_markdown('<div class="app-card-grid">' + "".join(segment_cards) + '</div>')
    else:
        st.info("Sin registros suficientes para evaluar segmentos de cuenta especial.")

# ===================== Analisis interactivo de tiempos (filtros LOCALES) =====================
safe_markdown('<div class="app-separator"></div>')
st.subheader("Analisis interactivo de tiempos")

ctrl_row1 = st.columns(4)
max_dias_pag = ctrl_row1[0].slider("Dia maximo a visualizar", 30, 365, 100, 10, key="max_dias_pag")
bins_pag     = ctrl_row1[1].slider("Numero de clases (columnas)", 10, 100, 25, 5, key="bins_pag")

ce_local = "Todas"
if "cuenta_especial" in df.columns:
    ce_local = ctrl_row1[2].radio("Cuenta Especial (local)", ["Todas","Cuenta Especial","No Cuenta Especial"],
                                  horizontal=True, index=0, key="ce_local")

prio_local = ctrl_row1[3].radio("Prioritario (local)", ["Todos","Prioritario","No Prioritario"],
                                horizontal=True, index=0, key="prio_local")

def _apply_locals(dfin: pd.DataFrame) -> pd.DataFrame:
    d = _subset_by_ce(dfin, ce_local)  # <-- CE robusto
    if "prov_prioritario" in d.columns:
        if prio_local == "Prioritario":
            d = d[d["prov_prioritario"] == True]
        elif prio_local == "No Prioritario":
            d = d[d["prov_prioritario"] == False]
    return d

# ===================== PAGADAS (pagos contabilizados) =====================
st.markdown("### Pagadas (base: pagos contabilizados)")

df_pag_contab_loc = _apply_locals(df_pag_contab)

fac_l   = pd.to_datetime(df_pag_contab_loc.get("fac_fecha_factura"), errors="coerce")
cc_l    = pd.to_datetime(df_pag_contab_loc.get("fecha_cc"), errors="coerce")
pago_l  = pd.to_datetime(df_pag_contab_loc.get("fecha_pagado"), errors="coerce")

dso_loc = (pago_l - fac_l).dt.days
tfc_loc = (cc_l   - fac_l).dt.days
tpc_loc = (pago_l - cc_l).dt.days

# Promedios GLOBAL (fijos) ya calculados arriba; Promedios LOCAL (dependen de filtros locales)
mean_dso_loc = float(np.nanmean(dso_loc)) if dso_loc.notna().any() else np.nan
mean_tfc_loc = float(np.nanmean(tfc_loc)) if tfc_loc.notna().any() else np.nan
mean_tpc_loc = float(np.nanmean(tpc_loc)) if tpc_loc.notna().any() else np.nan

def _hist_with_two_means(series: pd.Series, nbins: int, color: str,
                         xmax: int, title: str, mean_global: float, mean_local: float):
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[(vals >= 0) & (vals <= xmax)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals, nbinsx=nbins, marker_color=color))
    # GLOBAL
    if pd.notna(mean_global):
        fig.add_vline(x=mean_global, line_dash="solid", line_color=PURP)
        fig.add_annotation(x=mean_global, y=1.12, xref="x", yref="paper",
                           text=f"Promedio global: {one_decimal(mean_global)} días",
                           showarrow=False, font=dict(color=PURP))
    # LOCAL
    if pd.notna(mean_local):
        fig.add_vline(x=mean_local, line_dash="dash", line_color=GRN)
        fig.add_annotation(x=mean_local, y=1.04, xref="x", yref="paper",
                           text=f"Promedio local: {one_decimal(mean_local)} días",
                           showarrow=False, font=dict(color=GRN))
    fig.update_layout(height=320, margin=dict(l=20,r=20,t=56,b=20), title=title)
    return fig, vals

def _stats_and_counts(vals: pd.Series, xmax: int):
    vals = pd.to_numeric(vals, errors="coerce")
    vals = vals[vals >= 0]
    n_total = int(vals.notna().sum())
    n_le    = int((vals <= xmax).sum())
    n_gt    = n_total - n_le
    p50 = float(np.nanpercentile(vals, 50)) if n_total else np.nan
    p75 = float(np.nanpercentile(vals, 75)) if n_total else np.nan
    p90 = float(np.nanpercentile(vals, 90)) if n_total else np.nan
    return n_total, n_le, n_gt, p50, p75, p90

# ======= NOTA DE VALIDEZ (N total / usados / excluidos negativos) =======
def _validity_note(series_days: pd.Series, label: str):
    """Muestra una nota de validez: N total vs N usados (≥0) y excluidos por tiempos negativos.
    Importante para explicar diferencias de conteo con lo monetario.
    """
    raw = pd.to_numeric(series_days, errors="coerce")
    raw = raw[raw.notna()]
    n_total = int(len(raw))
    n_neg   = int((raw < 0).sum())
    n_used  = n_total - n_neg

    st.caption(
        f"ℹ️ {label}: N total={n_total:,} • N usados (≥0)={n_used:,} • Excluidos por tiempos negativos={n_neg:,}. "
        "Los histogramas cuentan solo días ≥ 0; los promedios y agregados monetarios del tablero no se ven afectados."
    )

# DPP ancho completo + contadores + P50/P75/P90
if dso_loc.notna().any():
    dso_num = pd.to_numeric(dso_loc, errors="coerce")
    _render_range_cards([
        ("Pagadas ≤ 30 días", f"{int((dso_num <= 30).sum()):,}"),
        (f"Pagadas 31–{max_dias_pag} días", f"{int(((dso_num > 30) & (dso_num <= max_dias_pag)).sum()):,}"),
        (f"Pagadas > {max_dias_pag} días", f"{int((dso_num > max_dias_pag).sum()):,}"),
    ])

    fig_dso, _ = _hist_with_two_means(dso_loc, bins_pag, BLUE, max_dias_pag,
                                      f"Emisión → Pago • {LABELS['dpp_emision_pago']}",
                                      mean_global=mean_dso_kpi, mean_local=mean_dso_loc)
    st.plotly_chart(fig_dso, use_container_width=True)

    # Nota de validez (N total / usados / excluidos negativos)
    _validity_note(dso_loc, LABELS["dpp_emision_pago"])

    n_total, n_le, n_gt, p50, p75, p90 = _stats_and_counts(dso_loc, max_dias_pag)
    _render_percentile_cards([
        ("N total", f"{n_total:,}", "Documentos"),
        ("P50", f"{one_decimal(p50)} d" if pd.notna(p50) else "s/d", "Mediana"),
        ("P75", f"{one_decimal(p75)} d" if pd.notna(p75) else "s/d", "Percentil 75"),
        ("P90", f"{one_decimal(p90)} d" if pd.notna(p90) else "s/d", "Percentil 90"),
        (f"N ≤ {max_dias_pag} d", f"{n_le:,}", "Documentos"),
        (f"N > {max_dias_pag} d", f"{n_gt:,}", "Documentos"),
    ])
else:
    st.info("Sin datos de Días Promedio a Pago (DPP) en pagos contabilizados (con los filtros locales).")

# DIC / DCP en 2 columnas (solo P50/P75/P90 y N≤/N>)
colA, colB = st.columns(2)
with colA:
    if tfc_loc.notna().any():
        fig_tfc, _ = _hist_with_two_means(tfc_loc, bins_pag, BLUE, max_dias_pag,
                                          f"Emisión → Contabilización • {LABELS['dic_emision_contab']}",
                                          mean_global=mean_tfc_kpi, mean_local=mean_tfc_loc)
        st.plotly_chart(fig_tfc, use_container_width=True)

        # Nota de validez
        _validity_note(tfc_loc, LABELS["dic_emision_contab"])

        n_total, n_le, n_gt, p50, p75, p90 = _stats_and_counts(tfc_loc, max_dias_pag)
        _render_percentile_cards([
            ("N total", f"{n_total:,}", "Documentos"),
            ("P50", f"{one_decimal(p50)} d" if pd.notna(p50) else "s/d", "Mediana"),
            ("P75", f"{one_decimal(p75)} d" if pd.notna(p75) else "s/d", "Percentil 75"),
            ("P90", f"{one_decimal(p90)} d" if pd.notna(p90) else "s/d", "Percentil 90"),
            (f"N <= {max_dias_pag} d", f"{n_le:,}", "Documentos"),
            (f"N > {max_dias_pag} d", f"{n_gt:,}", "Documentos"),
        ])
    else:
        st.info("Sin datos de Días a Ingreso Contable (DIC) en pagos contabilizados.")
with colB:
    if tpc_loc.notna().any():
        fig_tpc, _ = _hist_with_two_means(tpc_loc, bins_pag, BLUE, max_dias_pag,
                                          f"Contabilización → Pago • {LABELS['dcp_contab_pago']}",
                                          mean_global=mean_tpc_kpi, mean_local=mean_tpc_loc)
        st.plotly_chart(fig_tpc, use_container_width=True)

        # Nota de validez
        _validity_note(tpc_loc, LABELS["dcp_contab_pago"])

        n_total, n_le, n_gt, p50, p75, p90 = _stats_and_counts(tpc_loc, max_dias_pag)
        _render_percentile_cards([
            ("N total", f"{n_total:,}", "Documentos"),
            ("P50", f"{one_decimal(p50)} d" if pd.notna(p50) else "s/d", "Mediana"),
            ("P75", f"{one_decimal(p75)} d" if pd.notna(p75) else "s/d", "Percentil 75"),
            ("P90", f"{one_decimal(p90)} d" if pd.notna(p90) else "s/d", "Percentil 90"),
            (f"N <= {max_dias_pag} d", f"{n_le:,}", "Documentos"),
            (f"N > {max_dias_pag} d", f"{n_gt:,}", "Documentos"),
        ])
    else:
        st.info("Sin datos de Días desde Contabilización al Pago (DCP) en pagos contabilizados.")

# ===================== NO PAGADAS — tiempos hasta hoy =====================
safe_markdown('<div class="app-separator"></div>')
st.subheader("No Pagadas — tiempos *hasta hoy*")

with st.expander("¿Cómo se calculan estos tiempos?"):
    st.markdown(
        """
**Base temporal:** desde la **fecha de factura** (`fac_fecha_factura`) **hasta hoy**.  
Se muestran dos grupos:
1) **Facturado sin Contabilizar**: no tiene `fecha_cc` ⇒ días = *hoy − emisión*.  
2) **Contabilizado sin Pago**: sí tiene `fecha_cc` y no `fecha_pagado` ⇒ días = *hoy − contabilización*.  
Las líneas muestran **Promedio global** (fijo) y **Promedio local** (según filtros CE/Prioritario).
        """
    )

# Promedios globales (referencia fija)
today = pd.Timestamp.today().normalize()
fac_np_g = pd.to_datetime(df_no_pag.get("fac_fecha_factura", pd.Series(dtype="datetime64[ns]")), errors="coerce")
cc_np_g  = pd.to_datetime(df_no_pag.get("fecha_cc", pd.Series(dtype="datetime64[ns]")), errors="coerce")
pag_np_g = pd.to_datetime(df_no_pag.get("fecha_pagado", pd.Series(dtype="datetime64[ns]")), errors="coerce")

mean_np_sin_contab = float(np.nanmean((today - fac_np_g[cc_np_g.isna()]).dt.days)) if cc_np_g.isna().any() else np.nan
mean_np_contab_sin_pago = float(np.nanmean((today - cc_np_g[(cc_np_g.notna()) & (pag_np_g.isna())]).dt.days)) \
                          if ((cc_np_g.notna()) & (pag_np_g.isna())).any() else np.nan

# Local (afectado por CE/Prioritario)
df_no_pag_loc = _apply_locals(df_no_pag)

fac_np = pd.to_datetime(df_no_pag_loc.get("fac_fecha_factura", pd.Series(dtype="datetime64[ns]")), errors="coerce")
cc_np  = pd.to_datetime(df_no_pag_loc.get("fecha_cc", pd.Series(dtype="datetime64[ns]")), errors="coerce")
pag_np = pd.to_datetime(df_no_pag_loc.get("fecha_pagado", pd.Series(dtype="datetime64[ns]")), errors="coerce")

mask_sin_cc   = cc_np.isna()
mask_con_cc   = cc_np.notna()
mask_sin_pago = pag_np.isna()

df_no_cc = df_no_pag_loc[mask_sin_cc].copy()
df_cc_sp = df_no_pag_loc[mask_con_cc & mask_sin_pago].copy()

row_np = st.columns(2)
max_dias_no = row_np[0].slider("Dia maximo a visualizar (No Pagadas)", 30, 365, 100, 10, key="max_dias_no")
bins_no     = row_np[1].slider("Número de clases (No Pagadas)", 10, 100, 25, 5, key="bins_no")

def _hist_np_two_means(series: pd.Series, nbins: int, color: str, xmax: int,
                       title: str, mean_global: float, mean_local: float):
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[(vals >= 0) & (vals <= xmax)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals, nbinsx=nbins, marker_color=color))
    # GLOBAL
    if pd.notna(mean_global):
        fig.add_vline(x=mean_global, line_dash="solid", line_color=PURP)
        fig.add_annotation(x=mean_global, y=1.12, xref="x", yref="paper",
                           text=f"Promedio global: {one_decimal(mean_global)} días",
                           showarrow=False, font=dict(color=PURP))
    # LOCAL
    if pd.notna(mean_local):
        fig.add_vline(x=mean_local, line_dash="dash", line_color=GRN)
        fig.add_annotation(x=mean_local, y=1.04, xref="x", yref="paper",
                           text=f"Promedio local: {one_decimal(mean_local)} días",
                           showarrow=False, font=dict(color=GRN))
    fig.update_layout(height=320, margin=dict(l=20,r=20,t=56,b=20), title=title)
    return fig, vals

def _stats_block(vals: pd.Series, xmax: int):
    vals = pd.to_numeric(vals, errors="coerce")
    vals = vals[vals >= 0]
    n_total = int(vals.notna().sum())
    n_le    = int((vals <= xmax).sum())
    n_gt    = n_total - n_le
    p50 = float(np.nanpercentile(vals, 50)) if n_total else np.nan
    p75 = float(np.nanpercentile(vals, 75)) if n_total else np.nan
    p90 = float(np.nanpercentile(vals, 90)) if n_total else np.nan
    _render_percentile_cards([
        ("N total", f"{n_total:,}", "Documentos"),
        ("P50", f"{one_decimal(p50)} d" if pd.notna(p50) else "s/d", "Mediana"),
        ("P75", f"{one_decimal(p75)} d" if pd.notna(p75) else "s/d", "Percentil 75"),
        ("P90", f"{one_decimal(p90)} d" if pd.notna(p90) else "s/d", "Percentil 90"),
        (f"N ≤ {xmax} d", f"{n_le:,}", "Documentos"),
        (f"N > {xmax} d", f"{n_gt:,}", "Documentos"),
    ])

c1, c2 = st.columns(2)
with c1:
    if not df_no_cc.empty:
        dias_fac_hoy = (today - pd.to_datetime(df_no_cc["fac_fecha_factura"], errors="coerce")).dt.days
        mean_local_np1 = float(np.nanmean(dias_fac_hoy)) if dias_fac_hoy.notna().any() else np.nan
        fig4, _ = _hist_np_two_means(dias_fac_hoy, bins_no, RED, max_dias_no,
                                     "Facturado sin Contabilizar (emisión → hoy)",
                                     mean_global=mean_np_sin_contab, mean_local=mean_local_np1)
        st.plotly_chart(fig4, use_container_width=True)

        # Nota de validez
        _validity_note(dias_fac_hoy, "Facturado sin contabilizar (emisión→hoy)")

        _stats_block(dias_fac_hoy, max_dias_no)
    else:
        st.info("Sin registros en **Facturado sin Contabilizar** con los filtros actuales.")
with c2:
    if not df_cc_sp.empty:
        dias_cc_hoy = (today - pd.to_datetime(df_cc_sp["fecha_cc"], errors="coerce")).dt.days
        mean_local_np2 = float(np.nanmean(dias_cc_hoy)) if dias_cc_hoy.notna().any() else np.nan
        fig5, _ = _hist_np_two_means(dias_cc_hoy, bins_no, RED, max_dias_no,
                                     "Contabilizado sin Pago (contab. → hoy)",
                                     mean_global=mean_np_contab_sin_pago, mean_local=mean_local_np2)
        st.plotly_chart(fig5, use_container_width=True)

        # Nota de validez
        _validity_note(dias_cc_hoy, "Contabilizado sin pago (contab.→hoy)")

        _stats_block(dias_cc_hoy, max_dias_no)
    else:
        st.info("Sin registros en **Contabilizado sin Pago** con los filtros actuales.")

# ===================== Composición del Portafolio =====================
safe_markdown('<div class="app-separator"></div>')
st.subheader("Composición del Portafolio")

modo = st.radio("Medir por…", ["Conteo de documentos","Monto Autorizado"], horizontal=True)
serie = df.copy()
serie["_peso"] = 1.0 if modo == "Conteo de documentos" else pd.to_numeric(serie.get("monto_autorizado", 0), errors="coerce").fillna(0.0)

# OC robusto
if "fac_oc_numero" in serie:
    has_oc = pd.to_numeric(serie["fac_oc_numero"], errors="coerce").fillna(0) > 0
elif "oc_numero" in serie:
    has_oc = pd.to_numeric(serie["oc_numero"], errors="coerce").fillna(0) > 0
else:
    has_oc = pd.to_numeric(serie.get("con_oc", 0), errors="coerce").fillna(0) > 0
serie["_oc_flag"] = has_oc.map({True:"Con OC", False:"Sin OC"})

serie["_estado_legible"] = serie["estado_pago"].map(ESTADO_LABEL).fillna("Desconocido")
serie["_prio_legible"] = serie.get("prov_prioritario", False).map({True:"Prioritario", False:"No Prioritario"})

# CE legible robusto
if "cuenta_especial" in serie.columns:
    serie["_ce_legible"] = _coerce_bool_series(serie["cuenta_especial"]).map({True:"Cuenta Especial", False:"No Cuenta Especial"})

def _pie(df_in, name_col, title):
    t = df_in.groupby(name_col)["_peso"].sum().reset_index()
    fig = go.Figure(data=[go.Pie(labels=t[name_col], values=t["_peso"], hole=0.4)])
    fig.update_layout(title=title)
    return fig

cols = st.columns(4)
cols[0].plotly_chart(_pie(serie, "_oc_flag", "Distribución por OC"), use_container_width=True)
cols[1].plotly_chart(_pie(serie, "_estado_legible", "Distribución por Tipo de Documento"), use_container_width=True)
cols[2].plotly_chart(_pie(serie, "_prio_legible", "Proveedor Prioritario"), use_container_width=True)
if "_ce_legible" in serie.columns:
    cols[3].plotly_chart(_pie(serie, "_ce_legible", "Cuenta Especial vs No"), use_container_width=True)

# ===================== Exportación a Excel =====================
safe_markdown('<div class="app-separator"></div>')
st.subheader("Exportar Dashboard a Excel (multi-hoja)")
sheets = {}
sheets["Pagadas"] = df_pag if not df_pag.empty else pd.DataFrame()
sheets["No Pagadas"] = df_no_pag if not df_no_pag.empty else pd.DataFrame()

def _group(df_in, col, label):
    t = df_in.groupby(col)["_peso"].sum().reset_index().rename(columns={col:label, "_peso":"Valor"})
    return t

tmp = serie.copy()
sheets["Comp_OC"] = _group(tmp, "_oc_flag", "OC")
sheets["Comp_TipoDoc"] = _group(tmp, "_estado_legible", "Tipo Doc")
sheets["Comp_Prioritario"] = _group(tmp, "_prio_legible", "Prioritario")
if "_ce_legible" in tmp.columns:
    sheets["Comp_CuentaEspecial"] = _group(tmp, "_ce_legible", "Cuenta Especial")

st.download_button(
    "⬇️ Descargar Excel del Dashboard",
    data=excel_bytes_multi(sheets),
    file_name="dashboard_kpis.xlsx"
)
