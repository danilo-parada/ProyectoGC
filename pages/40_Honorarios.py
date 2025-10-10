from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Optional, Dict

from lib_common import (
    header_ui,
    get_honorarios_df,
    clean_estado_cuota,
    money,
    sanitize_df,
    safe_markdown,
)

st.set_page_config(page_title="Honorarios", layout="wide")
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
