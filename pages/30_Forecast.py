from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from lib_common import (
    get_df_norm,
    get_honorarios_df,
    header_ui,
    general_date_filters_ui,
    advanced_filters_ui,
    apply_general_filters,
    apply_advanced_filters,
    money,
    style_table,
    safe_markdown,
)
from lib_report import excel_bytes_single


def _excel_download(df: pd.DataFrame, sheet: str, name: str):
    return st.download_button(
        label=f"Descargar Excel - {sheet}",
        data=excel_bytes_single(df, sheet_name=sheet),
        file_name=name,
        disabled=df.empty,
    )


def _build_facturas_base(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=["fecha_pagado", "monto_autorizado", "cuenta_especial"])
    d = df_in.copy()
    d["fecha_pagado"] = pd.to_datetime(d.get("fecha_pagado"), errors="coerce")
    d["monto_autorizado"] = pd.to_numeric(d.get("monto_autorizado"), errors="coerce").fillna(0.0)
    if "cuenta_especial" not in d.columns:
        d["cuenta_especial"] = False
    d = d[(d["fecha_pagado"].notna()) & (d["monto_autorizado"] > 0)]
    return d[["fecha_pagado", "monto_autorizado", "cuenta_especial"]].copy()


def _build_honorarios_base() -> pd.DataFrame:
    d0 = get_honorarios_df()
    if d0 is None or getattr(d0, "empty", True):
        return pd.DataFrame(columns=["fecha_pagado", "monto_autorizado", "cuenta_especial"])
    d = d0.copy()
    if "estado_cuota" in d.columns:
        d = d[d["estado_cuota"].astype(str) == "PAGADA"]
    if "fecha_ce" in d.columns:
        d["fecha_pagado"] = pd.to_datetime(d["fecha_ce"], errors="coerce")
    else:
        d["fecha_pagado"] = pd.to_datetime(d.get("fecha_pagado"), errors="coerce")
    amt = pd.Series(np.nan, index=d.index, dtype=float)
    for col in ("monto_cuota", "monto_pagado", "liquido_cuota", "monto_autorizado", "fac_monto_total"):
        if col in d.columns:
            cand = pd.to_numeric(d[col], errors="coerce")
            amt = amt.fillna(cand)
    d["monto_autorizado"] = amt.fillna(0.0)
    if "cuenta_especial" in d.columns:
        ce = d["cuenta_especial"].fillna(False).astype(bool)
    else:
        ce = (
            (d.get("banco").notna() if "banco" in d.columns else False)
            | (d.get("cuenta_corriente").notna() if "cuenta_corriente" in d.columns else False)
        )
    d["cuenta_especial"] = ce
    d = d[(d["fecha_pagado"].notna()) & (d["monto_autorizado"] > 0)]
    return d[["fecha_pagado", "monto_autorizado", "cuenta_especial"]].copy()


def _aggregate(df: pd.DataFrame, gran: str):
    if gran == "Mes":
        df["fecha_g"] = pd.to_datetime(df["fecha_pagado"]).dt.to_period("M").dt.to_timestamp()
        freq = "MS"
    elif gran == "Semana":
        df["fecha_g"] = pd.to_datetime(df["fecha_pagado"]).dt.to_period("W").dt.start_time
        freq = "W-MON"
    else:
        df["fecha_g"] = pd.to_datetime(df["fecha_pagado"]).dt.to_period("D").dt.to_timestamp()
        freq = "D"
    ser = df.groupby("fecha_g")["monto_autorizado"].sum().sort_index().asfreq(freq, fill_value=0.0)
    return ser, freq


def _fit_hw(ts: pd.Series, steps: int, seasonal_periods: int, damped: bool):
    sp = max(2, int(seasonal_periods)) if seasonal_periods else None
    use_seasonal = sp is not None and sp >= 2 and len(ts) >= max(4, (sp or 2) + 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            if use_seasonal:
                model = ExponentialSmoothing(
                    ts, trend="add", seasonal="add", seasonal_periods=sp,
                    damped_trend=bool(damped), initialization_method="estimated"
                )
            else:
                model = ExponentialSmoothing(
                    ts, trend="add", seasonal=None,
                    damped_trend=bool(damped), initialization_method="estimated"
                )
            fit = model.fit(optimized=True)
            fc = fit.forecast(steps)
            fitted_vals = fit.fittedvalues
            common_idx = fitted_vals.index.intersection(ts.index)
            resid = (ts.loc[common_idx] - fitted_vals.loc[common_idx]).astype(float)
            sigma = float(np.nanstd(resid.values, ddof=1)) if len(resid) > 2 else 0.0
            return fc, fit, sigma
    except Exception:
        pass
    last = float(ts.iloc[-1]) if len(ts) else 0.0
    idx = pd.date_range(start=ts.index[-1], periods=steps + 1, freq=ts.index.freq)[1:] if len(ts) else pd.Index([])
    fc = pd.Series([last] * steps, index=idx)
    sigma = float(np.nanstd(ts.values, ddof=1)) if len(ts) > 2 else 0.0
    return fc, None, sigma


def _ci_from_fc(fc: pd.Series, sigma: float, alpha: float):
    z = float(norm.ppf(0.5 + alpha / 2.0))
    low = fc - z * sigma
    high = fc + z * sigma
    return low, high


def _mape_label(actual: pd.Series, fitted: pd.Series):
    try:
        a = pd.to_numeric(pd.Series(actual), errors="coerce")
        f = pd.to_numeric(pd.Series(fitted), errors="coerce")
        if isinstance(actual, pd.Series) and isinstance(fitted, pd.Series):
            idx = a.index.intersection(f.index)
            a = a.loc[idx]
            f = f.loc[idx]
        mask = a.notna() & f.notna() & (a > 0)
        if mask.sum() == 0:
            return None, "-"
        mape = float((np.abs(a[mask] - f[mask]) / a[mask]).mean() * 100.0)
        if mape <= 10:
            label = "Excelente"
        elif mape <= 20:
            label = "Bueno"
        elif mape <= 30:
            label = "Aceptable"
        else:
            label = "Mejorar"
        return mape, label
    except Exception:
        return None, "-"


def _period_label(gran: str) -> str:
    return "mes" if gran == "Mes" else ("semana" if gran == "Semana" else "dia")


HIST_COLOR = "#1f77b4"
FC_COLOR = "#ff7f0e"
IC_FILL = "rgba(255,127,14,0.14)"
IC_LINE = "rgba(255,127,14,0.55)"


def _format_hover_y():
    return "$%{y:,.0f}"


def _build_fig(
    ts: pd.Series,
    fc: pd.Series,
    low: pd.Series,
    high: pd.Series,
    title: str,
    *,
    rec_y: float | None = None,
    rec_enfoque: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts.values,
            name="Historico",
            mode="lines+markers",
            line=dict(color=HIST_COLOR, width=2.5),
            marker=dict(size=5, color=HIST_COLOR, line=dict(width=0)),
            hovertemplate="Periodo=%{x|%Y-%m-%d}<br>Monto=" + _format_hover_y() + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc.index,
            y=fc.values,
            name="Forecast",
            mode="lines+markers",
            line=dict(color=FC_COLOR, width=2.5, dash="dash"),
            marker=dict(size=6, color=FC_COLOR, line=dict(width=0)),
            hovertemplate="Periodo=%{x|%Y-%m-%d}<br>Forecast=" + _format_hover_y() + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(fc.index) + list(fc.index[::-1]),
            y=list(high.values) + list(low.values[::-1]),
            fill="toself",
            fillcolor=IC_FILL,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="IC",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc.index,
            y=high.values,
            name="IC Superior",
            mode="lines",
            line=dict(color=IC_LINE, width=1.5, dash="dot"),
            hovertemplate="Periodo=%{x|%Y-%m-%d}<br>IC Superior=" + _format_hover_y() + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc.index,
            y=low.values,
            name="IC Inferior",
            mode="lines",
            line=dict(color=IC_LINE, width=1.5, dash="dot"),
            hovertemplate="Periodo=%{x|%Y-%m-%d}<br>IC Inferior=" + _format_hover_y() + "<extra></extra>",
        )
    )
    try:
        if rec_y is not None and len(fc) > 0:
            rec_x = fc.index[0]
            low0 = float(low.iloc[0]) if len(low) > 0 else None
            high0 = float(high.iloc[0]) if len(high) > 0 else None
            ic_txt = ""
            if low0 is not None and high0 is not None:
                ic_txt = f"<br>IC: {money(low0)} a {money(high0)}"
            fig.add_trace(
                go.Scatter(
                    x=[rec_x],
                    y=[rec_y],
                    name="Forecast proximo periodo",
                    mode="markers",
                    marker=dict(size=12, color="#2ecc71", symbol="star"),
                    hovertemplate="Forecast proximo periodo=" + _format_hover_y() + "<extra></extra>",
                    showlegend=True,
                )
            )
            fig.add_annotation(
                x=rec_x,
                y=rec_y,
                text=(
                    (f"Forecast proximo periodo: {money(float(rec_y))}" + (f" ({rec_enfoque})" if rec_enfoque else ""))
                    + ic_txt
                ),
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.2,
                arrowcolor="#2ecc71",
                bgcolor="rgba(46,204,113,0.12)",
                bordercolor="#2ecc71",
                borderwidth=1,
                font=dict(size=11, color="#1f2a55"),
                ay=-40,
            )
    except Exception:
        pass

    fig.update_layout(
        title=title,
        template="plotly_white",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_x=0.01,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1.5,
        linecolor="#1f2a55",
        mirror=True,
        showgrid=True,
        gridcolor="rgba(200,200,200,0.35)",
        ticks="outside",
        ticklen=6,
        tickcolor="#1f2a55",
        title_text="Periodo",
        title_font=dict(size=12, color="#1f2a55"),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1.5,
        linecolor="#1f2a55",
        mirror=True,
        showgrid=True,
        gridcolor="rgba(200,200,200,0.35)",
        zeroline=False,
        ticks="outside",
        ticklen=6,
        tickcolor="#1f2a55",
        title_text="Monto",
        title_font=dict(size=12, color="#1f2a55"),
        tickprefix="$",
    )
    return fig


def _table_style(df_in: pd.DataFrame):
    sty = df_in.style
    try:
        sty = sty.hide(axis="index")
    except Exception:
        try:
            sty = sty.hide_index()
        except Exception:
            pass
    try:
        sty = sty.set_properties(subset=[df_in.columns[0]], **{"text-align": "left", "font-weight": "600"})
        if df_in.shape[1] > 1:
            sty = sty.set_properties(subset=df_in.columns[1:], **{"text-align": "right"})
    except Exception:
        pass
    return sty


def _metric_card(title: str, value: str, caption: str | None = None, *, cls: str = "") -> str:
    cap_html = f'<p class="app-card__subtitle">{caption}</p>' if caption else ""
    classes = "app-card" + (f" {cls}" if cls else "")
    return (
        f'<div class="{classes}">'
        f'<div class="app-card__title">{title}</div>'
        f'<div class="app-card__value">{value}</div>'
        f'{cap_html}'
        '</div>'
    )


def _advisor_card(gran: str, fc: pd.Series, low: pd.Series, high: pd.Series, mape: float | None, mape_lbl: str | None) -> str:
    period = _period_label(gran)
    try:
        fc_next = float(fc.iloc[0]) if len(fc) else 0.0
        low_next = float(low.iloc[0]) if len(low) else 0.0
        high_next = float(high.iloc[0]) if len(high) else 0.0
    except Exception:
        fc_next = low_next = high_next = 0.0

    calidad_txt = f"Calidad: {mape_lbl} ({mape:.1f}%)" if mape is not None else "Calidad: -"
    caption = (
        f"Forecast: {money(fc_next)}"
        f"<br>Conservador: {money(low_next)} (IC minimo)"
        f"<br>Agresivo: {money(high_next)} (IC maximo)"
        f"<br>{calidad_txt}"
        f"<br>Enfoque: Forecast"
        f"<br><span class=\"app-card__subtitle--muted\">Regla: Forecast como base; Conservador = IC minimo; Agresivo = IC maximo.</span>"
    )
    return _metric_card(f"Forecast proximo {_period_label(gran)}", money(fc_next), caption, cls="app-card--advice app-card--wide")


# Page config
st.set_page_config(page_title="Forecast", layout="wide", initial_sidebar_state="collapsed")
header_ui(
    title="Laboratorio de Forecast de Pagos",
    current_page="Forecast",
    subtitle="Proyecciones sobre pagos",
    nav_active="forecast",
)


df0 = get_df_norm()
if df0 is None:
    st.warning("Carga tus datos en 'Carga de Data' primero.")
    st.stop()

fac_ini, fac_fin, pay_ini, pay_fin = general_date_filters_ui(df0)
st.subheader("1. Filtros del Escenario")
sede, org, prov, cc, oc, est, prio = advanced_filters_ui(
    df0, show_controls=["sede", "prov", "cc", "oc"], labels={"oc": "Con OC"}
)

df_filtrado = apply_general_filters(df0, fac_ini, fac_fin, pay_ini, pay_fin)
df_filtrado = apply_advanced_filters(df_filtrado, sede, [], prov, cc, oc, [], prio)

fuente = st.radio("Fuente de datos", ["Facturas", "Honorarios", "Ambos"], horizontal=True, index=0)

df_fac_base = _build_facturas_base(df_filtrado)
df_hon_base = _build_honorarios_base()
if fuente == "Facturas":
    df_base = df_fac_base
elif fuente == "Honorarios":
    df_base = df_hon_base
else:
    df_base = pd.concat([df_fac_base, df_hon_base], ignore_index=True)

if df_base.empty:
    st.info("No hay datos de pagos con los filtros seleccionados.")
    st.stop()

# Model params
st.subheader("2. Parametros del Modelo")
c1, c2, c3 = st.columns(3)
c1.markdown("Modelo: Holt-Winters (aditivo)")
gran = c2.selectbox("Granularidad", ["Mes", "Semana", "Dia"], index=0)
n_steps = c3.slider("Horizonte a Proyectar", 1, 36, 6, 1)

h1, h2, h3 = st.columns(3)
seasonal_ui = h1.slider("Periodos estacionales", 1, 36, 12, 1)
prev_src = st.session_state.get("_last_fuente_for_damped")
if fuente != prev_src:
    if fuente == "Ambos":
        st.session_state["damped_trend_flag"] = True
    st.session_state["_last_fuente_for_damped"] = fuente
damped_trend = h2.checkbox("Tendencia amortiguada (damped)", value=bool(st.session_state.get("damped_trend_flag", fuente == "Ambos")), key="damped_trend_flag")
alpha_ci = h3.slider("Confianza", 0.80, 0.99, 0.95, 0.01)

clip_ci0 = st.checkbox("Recortar IC inferior < 0 a 0", value=True)
clip_fc0 = st.checkbox("Evitar forecast negativo (>= 0)", value=True)

# Aggregate + fit (General)
ts_all, freq = _aggregate(df_base.copy(), gran)
max_period = max(2, len(ts_all) // 2)
seasonal_periods = min(int(seasonal_ui), int(max_period)) if len(ts_all) >= 2 else 2
fc_all, fit_all, sigma_all = _fit_hw(ts_all, n_steps, seasonal_periods, damped_trend)
low_all, high_all = _ci_from_fc(fc_all, sigma_all, alpha_ci)
if clip_ci0:
    low_all = low_all.clip(lower=0.0)
if clip_fc0:
    fc_all = fc_all.clip(lower=0.0)

# Metrics
mape_all, mape_lbl_all = (None, "-")
try:
    if fit_all is not None:
        mape_all, mape_lbl_all = _mape_label(ts_all, fit_all.fittedvalues)
except Exception:
    pass

rec_global = float(fc_all.iloc[0]) if len(fc_all) else None
fig_all = _build_fig(
    ts_all, fc_all, low_all, high_all, title="Proyeccion General",
    rec_y=rec_global, rec_enfoque="Forecast",
)
st.plotly_chart(fig_all, use_container_width=True)

# Table (General)
def _tick_fmt(gran):
    return "%Y-%m" if gran == "Mes" else ("%Y-%m-%d")

tick_format = _tick_fmt(gran)
tbl_all = pd.DataFrame({
    "Periodo": pd.to_datetime(fc_all.index).to_series().dt.strftime(tick_format).values,
    "Forecast": [money(v) for v in fc_all.values],
    "IC Inferior": [money(v) for v in low_all.values],
    "IC Superior": [money(v) for v in high_all.values],
})
display_tbl_all = tbl_all.copy()
if mape_all is not None:
    resumen_row = pd.DataFrame([
        {
            "Periodo": "Resumen (MAPE)",
            "Forecast": f"{mape_all:.1f}% ({mape_lbl_all})",
            "IC Inferior": "",
            "IC Superior": "",
        }
    ])
    display_tbl_all = pd.concat([display_tbl_all, resumen_row], ignore_index=True)
style_table(_table_style(display_tbl_all), visible_rows=min(15, len(display_tbl_all) + 1))
_excel_download(tbl_all, sheet="General", name="forecast_general.xlsx")

# Indicadores
hist_total = float(ts_all.sum()) if len(ts_all) else 0.0
hist_avg_12 = float(ts_all.tail(min(12, len(ts_all))).mean()) if len(ts_all) else 0.0
sum_fc = float(fc_all.sum())
sum_low = float(low_all.sum())
sum_high = float(high_all.sum())

cards = []
cards.append(_metric_card("Historico total", money(hist_total)))
cards.append(_metric_card("Promedio ultimos 12", money(hist_avg_12)))
cards.append(_metric_card("Horizonte acumulado", money(sum_fc)))
cards.append(_metric_card("IC rango (inferior-superior)", f"{money(sum_low)} a {money(sum_high)}"))
_mape_value_txt = f"{mape_all:.1f}%" if mape_all is not None else "-"
_mape_caption = ""
if mape_all is not None:
    _mape_caption = (
        f"Calidad: {mape_lbl_all}<br>"
        "<span class=\"app-card__subtitle--muted\">"
        "MAPE: error absoluto promedio del pronostico vs real; menor es mejor. "
        "Ej: 10% \u2248 el pronostico se desvia ~10% del real."
        "</span>"
    )
cards.append(_metric_card("MAPE", _mape_value_txt, caption=_mape_caption))
cards.append(_advisor_card(gran, fc_all, low_all, high_all, mape_all, mape_lbl_all))
safe_markdown('<div class="app-card-grid">' + "".join(cards) + '</div>')

# Historico (ultimos 12)
with st.expander("Tabla historico (ultimos 12)"):
    if fuente == "Ambos":
        ts_fac, _ = _aggregate(df_fac_base.copy(), gran) if isinstance(df_fac_base, pd.DataFrame) else (pd.Series(dtype=float), None)
        ts_hon, _ = _aggregate(df_hon_base.copy(), gran) if isinstance(df_hon_base, pd.DataFrame) else (pd.Series(dtype=float), None)
        base_idx = ts_all.index
        fac_align = ts_fac.reindex(base_idx, fill_value=0.0) if len(ts_fac) else pd.Series(0.0, index=base_idx)
        hon_align = ts_hon.reindex(base_idx, fill_value=0.0) if len(ts_hon) else pd.Series(0.0, index=base_idx)
        ambos_align = fac_align + hon_align
        last_n = min(12, len(base_idx))
        if last_n:
            idx_tail = base_idx[-last_n:]
            periodos = pd.to_datetime(idx_tail).to_series().dt.strftime(tick_format).values
            hist_tbl = pd.DataFrame({
                "Periodo": periodos,
                "Real Facturas": [money(v) for v in fac_align.loc[idx_tail].values],
                "Real Honorarios": [money(v) for v in hon_align.loc[idx_tail].values],
                "Real Ambos": [money(v) for v in ambos_align.loc[idx_tail].values],
            })
            style_table(_table_style(hist_tbl), visible_rows=min(last_n, len(hist_tbl) + 1))
            _excel_download(hist_tbl, sheet="Historico", name="historico_ultimos12.xlsx")
    else:
        last_n = min(12, len(ts_all))
        if last_n:
            hist_tbl = pd.DataFrame({
                "Periodo": pd.to_datetime(ts_all.tail(last_n).index).to_series().dt.strftime(tick_format).values,
                "Real": [money(v) for v in ts_all.tail(last_n).values],
            })
            style_table(_table_style(hist_tbl), visible_rows=min(last_n, len(hist_tbl) + 1))
            _excel_download(hist_tbl, sheet="Historico", name="historico_ultimos12.xlsx")


st.markdown("---")
st.subheader("3. Desglose por Cuenta Especial")


def _section_ce(df_in: pd.DataFrame, title: str, flag_val: bool):
    d = df_in[df_in["cuenta_especial"].fillna(False).astype(bool) == flag_val]
    if d.empty:
        st.info(f"Sin datos para '{title}'.")
        return
    ts, _ = _aggregate(d.copy(), gran)
    if ts.empty or ts.sum() <= 0:
        st.info(f"Serie vacia para '{title}'.")
        return
    max_p = max(2, len(ts) // 2)
    sp = min(int(seasonal_ui), int(max_p)) if len(ts) >= 2 else 2
    fc, fit, sigma = _fit_hw(ts, n_steps, sp, damped_trend)
    low, high = _ci_from_fc(fc, sigma, alpha_ci)
    if clip_ci0:
        low = low.clip(lower=0.0)
    if clip_fc0:
        fc = fc.clip(lower=0.0)

    rec_local = float(fc.iloc[0]) if len(fc) else None
    fig = _build_fig(ts, fc, low, high, title=title, rec_y=rec_local, rec_enfoque="Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # Cards (igual esquema que general)
    hist_total_ce = float(ts.sum()) if len(ts) else 0.0
    hist_avg12_ce = float(ts.tail(min(12, len(ts))).mean()) if len(ts) else 0.0
    sum_fc = float(fc.sum())
    sum_low = float(low.sum())
    sum_high = float(high.sum())
    mape_val, mape_lbl = (None, "-")
    try:
        if fit is not None:
            mape_val, mape_lbl = _mape_label(ts, fit.fittedvalues)
    except Exception:
        pass
    cards = [
        _metric_card("Historico total", money(hist_total_ce)),
        _metric_card("Promedio ultimos 12", money(hist_avg12_ce)),
        _metric_card("Horizonte acumulado", money(sum_fc)),
        _metric_card("IC rango (inferior-superior)", f"{money(sum_low)} a {money(sum_high)}"),
        _metric_card("MAPE", f"{mape_val:.1f}%" if mape_val is not None else "-", caption=(f"Calidad: {mape_lbl}" if mape_val is not None else "")),
        _advisor_card(gran, fc, low, high, mape_val, mape_lbl),
    ]
    safe_markdown('<div class="app-card-grid">' + "".join(cards) + '</div>')

    tbl = pd.DataFrame({
        "Periodo": pd.to_datetime(fc.index).to_series().dt.strftime(tick_format).values,
        "Forecast": [money(v) for v in fc.values],
        "IC Inferior": [money(v) for v in low.values],
        "IC Superior": [money(v) for v in high.values],
    })
    style_table(_table_style(tbl), visible_rows=min(12, len(tbl) + 1))
    base_name = "forecast_ce.xlsx" if flag_val else "forecast_no_ce.xlsx"
    sheet_name = "CE" if flag_val else "No CE"
    _excel_download(tbl, sheet=sheet_name, name=base_name)


c1, c2 = st.columns(2)
with c1:
    _section_ce(df_base, "Cuenta Especial", True)
with c2:
    _section_ce(df_base, "No Cuenta Especial", False)
