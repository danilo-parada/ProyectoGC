# 30_Forecast.py — Forecast con IC, métricas y explicación en General/CE/No CE.
# Mantiene estilo previo. Descargas en Excel. Gráficos con IC y marcadores.

import html
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from math import sqrt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.stats import norm

from lib_common import (
    get_df_norm, general_date_filters_ui, apply_general_filters,
    advanced_filters_ui, apply_advanced_filters, header_ui, money, one_decimal,
    style_table
)
from lib_report import excel_bytes_single


# ------------------------ utilidades ------------------------ #
def _agg_paid_series_like_general(df: pd.DataFrame, freq_alias: str) -> pd.DataFrame:
    """Serie de pagos (pagadas) agregada por fecha de pago y monto_autorizado>0."""
    if df.empty:
        return pd.DataFrame(columns=["fecha_g", "y"])
    d = df.copy()
    d = d[d["estado_pago"] == "pagada"]
    if d.empty:
        return pd.DataFrame(columns=["fecha_g", "y"])

    d["fecha_pagado"] = pd.to_datetime(d.get("fecha_pagado"), errors="coerce")
    d["monto_autorizado"] = pd.to_numeric(d.get("monto_autorizado"), errors="coerce").fillna(0.0)
    d = d[(d["fecha_pagado"].notna()) & (d["monto_autorizado"] > 0)]
    if d.empty:
        return pd.DataFrame(columns=["fecha_g", "y"])

    if freq_alias == "MS":
        d["fecha_g"] = d["fecha_pagado"].dt.to_period("M").dt.to_timestamp()
    elif freq_alias.startswith("W-"):
        d["fecha_g"] = d["fecha_pagado"].dt.to_period("W").dt.start_time
    else:
        d["fecha_g"] = d["fecha_pagado"].dt.to_period("D").dt.to_timestamp()

    ts = (
        d.groupby("fecha_g")["monto_autorizado"].sum()
         .sort_index()
         .asfreq(freq_alias, fill_value=0.0)
    )
    return ts.reset_index().rename(columns={"monto_autorizado": "y"})


def _ci_from_residuals(y_true: np.ndarray, yhat_in: np.ndarray, yhat_out: np.ndarray, z: float, clip_zero: bool):
    """IC usando desviación estándar de residuos in-sample."""
    y_true = np.asarray(y_true, float)
    yhat_in = np.asarray(yhat_in, float)
    yhat_out = np.asarray(yhat_out, float)
    resid = y_true - yhat_in
    sigma = np.nanstd(resid)
    low = yhat_out - z * sigma
    high = yhat_out + z * sigma
    if clip_zero:
        low = np.maximum(low, 0.0)
    return low, high, sigma


def _excel_download(df: pd.DataFrame, sheet: str, name: str):
    return st.download_button(
        f"⬇️ Descargar Excel — {sheet}",
        data=excel_bytes_single(df, sheet),
        file_name=name,
        disabled=df.empty
    )


def _metrics(y_true, y_hat):
    y_true = np.asarray(y_true, float)
    y_hat = np.asarray(y_hat, float)
    mae = np.mean(np.abs(y_true - y_hat))
    rmse = sqrt(np.mean((y_true - y_hat) ** 2))
    mask = y_true > 0
    mape = np.nan
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_hat[mask]) / y_true[mask])) * 100.0
    return mae, rmse, mape


def _metrics_explainer_block(title: str, thr_exc: int, thr_good: int, thr_ok: int):
    """Bloque didáctico con fórmulas (latex), lectura del IC y guía dinámica por umbrales."""
    st.markdown(f"### ℹ️ Explicación de métricas — {title}")

    # Promedio histórico
    st.markdown("**Promedio histórico**")
    st.latex(r"\overline{H}=\frac{\sum_{t=1}^{n}\text{monto}_t}{n}")
    st.markdown("Base para comparar el nivel esperado de pagos.")

    # Variación %
    st.markdown("**Variación % Forecast vs Promedio**")
    st.latex(r"\%\Delta=\left(\frac{\overline{F}}{\overline{H}}-1\right)\times 100")
    st.markdown("donde ")
    st.latex(r"\overline{F}\ \text{es el promedio del forecast en el horizonte.}")
    st.markdown("• > 0%: pagar más que el histórico &nbsp;&nbsp;• < 0%: pagar menos.")

    # MAE / RMSE / MAPE con fórmulas
    st.markdown("**MAE** (error absoluto medio, en $):")
    st.latex(r"\mathrm{MAE}=\frac{1}{n}\sum_{t=1}^{n}\left|y_t-\hat{y}_t\right|")

    st.markdown("**RMSE** (raíz del error cuadrático medio, en $): penaliza más los errores grandes.")
    st.latex(r"\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{t=1}^{n}\left(y_t-\hat{y}_t\right)^2}")

    st.markdown("**MAPE** (error porcentual medio, solo con valores reales > 0):")
    st.latex(r"\mathrm{MAPE}=\frac{100}{n}\sum_{t=1}^{n}\left|\frac{y_t-\hat{y}_t}{y_t}\right|\quad\text{con }y_t>0")

    # Guía por umbrales dinámicos
    st.markdown(
        f"**Guía**: ≤ **{thr_exc}%** Excelente, ≤ **{thr_good}%** Bueno, "
        f"≤ **{thr_ok}%** Aceptable, > **{thr_ok}%** Débil."
    )

    # Recomendación operativa basada en IC
    st.markdown("**Recomendación operativa (usando el IC):**")
    st.markdown(
        "- En las tablas aparece **IC Bajo** y **IC Alto**. Interprétalo como un rango donde es **probable** "
        "que caiga el pago real."
    )
    st.markdown("- Para decidir un **monto de pago** por período puedes usar:")
    st.markdown("  - **Conservador:** *IC Bajo* (minimiza riesgo de sobre-pago).")
    st.markdown("  - **Base/Esperado:** *Forecast* (punto central del modelo).")
    st.markdown("  - **Prudente:** promedio del intervalo de confianza (si la variabilidad es alta):")
    st.latex(r"\text{Monto prudente}=\frac{\mathrm{IC\ Bajo}+\mathrm{IC\ Alto}}{2}")

    st.markdown(
        "- Si el **IC es muy ancho** o el **MAPE** es alto, segmenta por **CE/No CE**, revisa estacionalidad "
        "o cambia/ajusta el modelo."
    )


# ------------------------ componentes de UI ------------------------ #
def _render_metric_cards(cards: list[dict[str, str]]):
    if not cards:
        return
    pieces = ["<div class='forecast-card-grid'>"]
    for card in cards:
        label = html.escape(card.get("label", ""))
        value = html.escape(card.get("value", ""))
        foot = card.get("foot")
        foot_is_html = card.get("foot_is_html", False)
        if foot:
            foot_html = foot if foot_is_html else html.escape(foot)
            foot_block = f"<p class=\"forecast-card__foot\">{foot_html}</p>"
        else:
            foot_block = ""
        card_markup = (
            "<div class=\"forecast-card\">"
            f"<span class=\"forecast-card__label\">{label}</span>"
            f"<span class=\"forecast-card__value\">{value}</span>"
            f"{foot_block}"
            "</div>"
        )
        pieces.append(card_markup)
    pieces.append("</div>")
    st.markdown("".join(pieces), unsafe_allow_html=True)


def _mape_status(mape_val: float, thr_exc: int, thr_good: int, thr_ok: int):
    if np.isnan(mape_val):
        return "neutral", "MAPE no disponible"
    if mape_val <= thr_exc:
        return "success", f"Excelente · ≤ {thr_exc}%"
    if mape_val <= thr_good:
        return "info", f"Bueno · ≤ {thr_good}%"
    if mape_val <= thr_ok:
        return "warning", f"Aceptable · ≤ {thr_ok}%"
    return "danger", f"Débil · > {thr_ok}%"


def _mape_card(mape_val: float, thr_exc: int, thr_good: int, thr_ok: int) -> dict[str, str]:
    variant, message = _mape_status(mape_val, thr_exc, thr_good, thr_ok)
    if np.isnan(mape_val):
        return {
            "label": "MAPE",
            "value": "N/A",
            "foot": "Requiere valores reales positivos para calcularse.",
        }
    badge = (
        f"<span class='forecast-chip forecast-chip--{variant}'>{html.escape(message)}</span>"
        "<span class='forecast-foot-note'>Precisión in-sample</span>"
    )
    return {
        "label": "MAPE",
        "value": f"{one_decimal(mape_val)}%",
        "foot": badge,
        "foot_is_html": True,
    }


def _forecast_table_style(df_in: pd.DataFrame) -> pd.io.formats.style.Styler:
    sty = df_in.style.hide(axis="index")
    sty = sty.set_table_styles([
        {"selector": "thead tr", "props": [
            ("background-color", "#0d2f66"),
            ("color", "#FFFFFF"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.6px"),
            ("font-weight", "600"),
            ("font-size", "0.9rem"),
        ]},
        {"selector": "th", "props": [
            ("background-color", "transparent"),
            ("color", "#FFFFFF"),
            ("padding", "12px 16px"),
        ]},
        {"selector": "tbody td", "props": [
            ("font-size", "0.95rem"),
            ("padding", "12px 16px"),
            ("border-bottom", "1px solid #e0e6ff"),
            ("color", "#132542"),
        ]},
        {"selector": "tbody tr:nth-child(even)", "props": [
            ("background-color", "#f5f7ff"),
        ]},
        {"selector": "tbody tr:hover", "props": [
            ("background-color", "#e8edff"),
        ]},
    ], overwrite=False)
    if df_in.shape[1] > 1:
        first_col = df_in.columns[0]
        sty = sty.set_properties(subset=[first_col], **{"text-align": "left", "font-weight": "600"})
        if df_in.shape[1] > 1:
            sty = sty.set_properties(subset=df_in.columns[1:], **{"text-align": "right"})
    return sty


def _render_note(text: str):
    st.markdown(
        f"<p class='forecast-note'>{html.escape(text)}</p>",
        unsafe_allow_html=True,
    )


# ------------------------ cabecera y filtros ------------------------ #
st.set_page_config(page_title="Forecast", layout="wide")
header_ui(
    title="Laboratorio de Forecast de Pagos",
    current_page="Forecast",
    subtitle="Proyecciones sobre facturas pagadas (monto_autorizado en fecha_pagado)"
)

st.html('<div class="content-container">')
df0 = get_df_norm()
if df0 is None:
    st.warning("Carga tus datos en 'Carga de Data' primero.")
    st.html("</div>")
    st.stop()

fac_ini, fac_fin, pay_ini, pay_fin = general_date_filters_ui(df0)

st.subheader("1. Filtros del Escenario (Locales a Forecast)")
sede, org, prov, cc, oc, est, prio = advanced_filters_ui(
    df0,
    show_controls=['sede', 'prov', 'cc', 'oc', 'prio'],
    labels={"oc": "Con OC"},
    helps={"oc": "Filtra por presencia de Orden de Compra (valor > 0)."}
)

df_filtrado = apply_general_filters(df0, fac_ini, fac_fin, pay_ini, pay_fin)
df_filtrado = apply_advanced_filters(df_filtrado, sede, [], prov, cc, oc, [], prio)

df = df_filtrado[df_filtrado["estado_pago"] == "pagada"].copy()
if df.empty or "fecha_pagado" not in df.columns or df["fecha_pagado"].isna().all():
    st.info("No hay datos de pagos con los filtros seleccionados.")
    st.html("</div>")
    st.stop()

with st.expander("¿Qué datos se usan para el forecast?", expanded=True):
    st.markdown(
        """
- **Estado:** solo `pagada`.  
- **Valor:** `monto_autorizado` (siempre > 0).  
- **Fecha:** `fecha_pagado` (eje temporal).  
Esto pronostica **flujos realizados** (útil para caja/presupuesto).
        """
    )

df["fecha_pagado"] = pd.to_datetime(df["fecha_pagado"])
df["importe"] = pd.to_numeric(df["monto_autorizado"], errors="coerce").fillna(0.0)

# ------------------------ parámetros del modelo ------------------------ #
st.subheader("2. Parámetros del Modelo")
c1, c2, c3 = st.columns(3)
modelo_sel = c1.selectbox("Modelo", ["Proyección de Media Móvil", "Regresión Lineal", "Holt-Winters (Aditivo)"])
gran = c2.selectbox("Granularidad", ["Mes", "Semana", "Día"])
n_steps = c3.slider("Horizonte a Proyectar", 1, 36, 6, 1)

h1, h2, h3 = st.columns(3)
seasonal_periods_ui = h1.slider("Períodos Estacionales", 1, 36, 12, 1)
damped_trend = h2.checkbox("Tendencia amortiguada (damped)", value=False)
alpha_ci = h3.slider("Confianza (α)", 0.80, 0.99, 0.95, 0.01)

st.subheader("2.1 Alertas y Reglas")
a1, a2, a3, a4 = st.columns(4)
thr_exc = a1.slider("MAPE Excelente ≤", 5, 20, 10, 1)
thr_good = a2.slider("MAPE Bueno ≤", 10, 30, 20, 1)
thr_ok   = a3.slider("MAPE Aceptable ≤", 20, 50, 30, 1)
clip_ci  = a4.checkbox("Recortar IC inferior < 0 a 0", value=True)
clip_fc  = st.checkbox("Evitar forecast negativo (≥ 0)", value=True)

# ------------------------ agregación temporal (GENERAL) ------------------------ #
if gran == "Mes":
    df["fecha_g"] = df["fecha_pagado"].dt.to_period("M").dt.to_timestamp()
    freq = "MS"
elif gran == "Semana":
    df["fecha_g"] = df["fecha_pagado"].dt.to_period("W").dt.start_time
    freq = "W-MON"
else:
    df["fecha_g"] = df["fecha_pagado"].dt.to_period("D").dt.to_timestamp()
    freq = "D"

ts = (
    df.groupby("fecha_g")["importe"].sum()
      .sort_index().asfreq(freq, fill_value=0.0)
      .reset_index()
)

max_period = max(2, len(ts) // 2)
seasonal_periods = min(seasonal_periods_ui, max_period)
z = float(norm.ppf(0.5 + alpha_ci / 2.0))

last_date = ts["fecha_g"].iloc[-1]
future_idx = pd.date_range(start=last_date, periods=n_steps + 1, freq=freq)[1:]

# ------------------------ modelado (GENERAL) ------------------------ #
y = ts["importe"].astype(float)
ts["tendencia"] = np.nan
fc = pd.DataFrame({"fecha_g": future_idx, "forecast": np.nan})
ci_low = ci_high = None


def _ci_bounds(series_fitted: pd.Series, fc_values: np.ndarray, zval: float):
    resid = series_fitted.dropna()
    if resid.empty:
        return None, None
    sigma = np.std(resid)
    lower = fc_values - zval * sigma
    upper = fc_values + zval * sigma
    return lower, upper


if modelo_sel == "Proyección de Media Móvil":
    ventana = st.slider("Ventana de Media Móvil", 1, max_period, min(12, max_period), 1)
    ts["tendencia"] = y.rolling(window=ventana, min_periods=1).mean()
    forecast_value = ts["tendencia"].iloc[-1]
    fc["forecast"] = forecast_value
    resid = y - ts["tendencia"]
    lb, ub = _ci_bounds(resid, fc["forecast"].values, z)
    ci_low, ci_high = lb, ub

elif modelo_sel == "Regresión Lineal":
    ts["time_index"] = np.arange(len(ts.index))
    model = LinearRegression().fit(ts[["time_index"]], y)
    ts["tendencia"] = model.predict(ts[["time_index"]])
    fut_idx = np.arange(len(ts.index), len(ts.index) + n_steps).reshape(-1, 1)
    fc["forecast"] = model.predict(fut_idx)
    resid = y - ts["tendencia"]
    lb, ub = _ci_bounds(resid, fc["forecast"].values, z)
    ci_low, ci_high = lb, ub

else:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = ExponentialSmoothing(
                y, trend="add", seasonal="add",
                seasonal_periods=seasonal_periods, damped_trend=damped_trend,
                initialization_method="estimated"
            ).fit()
        ts["tendencia"] = model.fittedvalues
        fc["forecast"] = model.forecast(n_steps).values
        resid = y - ts["tendencia"]
        lb, ub = _ci_bounds(resid, fc["forecast"].values, z)
        ci_low, ci_high = lb, ub
    except Exception as e:
        st.error(f"Error con Holt-Winters (Aditivo): {e}")
        st.html("</div>")
        st.stop()

# ------------------------ visualización (GENERAL) ------------------------ #
st.subheader("3. Visualización del Pronóstico — **General**")

tick_format = "%b %Y" if gran == "Mes" else ("Sem %W, %Y" if gran == "Semana" else "%d-%m-%Y")

fc_plot = fc.copy()
ci_low_plot = np.array(ci_low, float).copy() if ci_low is not None else None
ci_high_plot = np.array(ci_high, float).copy() if ci_high is not None else None

if clip_fc:
    fc_plot["forecast"] = np.maximum(fc_plot["forecast"], 0.0)
if clip_ci and (ci_low_plot is not None):
    ci_low_plot = np.maximum(ci_low_plot, 0.0)

err_plus = err_minus = None
if (ci_low_plot is not None) and (ci_high_plot is not None):
    err_plus = np.maximum(ci_high_plot - fc_plot["forecast"].values, 0.0)
    err_minus = np.maximum(fc_plot["forecast"].values - ci_low_plot, 0.0)

fig = go.Figure()
fig.add_scatter(x=ts["fecha_g"], y=y, mode="lines", name="Histórico Real")
fig.add_scatter(x=ts["fecha_g"], y=ts["tendencia"], mode="lines", name="Tendencia (Modelo)", line=dict(dash="dot"))
fig.add_scatter(
    x=fc_plot["fecha_g"], y=fc_plot["forecast"], mode="lines+markers", name="Forecast",
    line=dict(dash="dash"),
    error_y=dict(
        type="data",
        array=err_plus if err_plus is not None else None,
        arrayminus=err_minus if err_minus is not None else None,
        visible=True if err_plus is not None else False
    )
)
if (ci_low_plot is not None) and (ci_high_plot is not None):
    fig.add_scatter(
        x=list(fc_plot["fecha_g"]) + list(fc_plot["fecha_g"][::-1]),
        y=list(ci_high_plot) + list(ci_low_plot[::-1]),
        fill="toself", fillcolor="rgba(0,0,200,0.08)", line=dict(width=0),
        hoverinfo="skip", name=f"IC {int(alpha_ci*100)}%"
    )
fig.update_xaxes(tickformat=tick_format)
st.plotly_chart(fig, use_container_width=True)
if clip_fc or clip_ci:
    _render_note("Se recortó en 0 el forecast o el límite inferior por la naturaleza de los pagos.")

# ------------------------ resumen + explicación (GENERAL) ------------------------ #
st.subheader("4. Resumen Numérico del Pronóstico")
avg_hist = float(y.mean())
avg_fc = float(np.mean(fc_plot["forecast"].values))
var_pct = ((avg_fc - avg_hist) / avg_hist) * 100 if avg_hist > 0 else 0.0
mae, rmse, mape = _metrics(y_true=y, y_hat=ts["tendencia"].fillna(y).values)

cards_general = [
    {
        "label": "Promedio Histórico",
        "value": money(avg_hist),
        "foot": "Promedio del monto pagado en el histórico.",
    },
    {
        "label": "Forecast Promedio",
        "value": money(avg_fc),
        "foot": "Valor medio proyectado en el horizonte seleccionado.",
    },
    {
        "label": "Variación % Forecast vs Promedio",
        "value": f"{one_decimal(var_pct)}%",
        "foot": "Positivo: forecast por encima del histórico.",
    },
    {
        "label": "MAE",
        "value": money(mae),
        "foot": "<span class='forecast-foot-note'>Error absoluto medio (histórico).</span>",
        "foot_is_html": True,
    },
    {
        "label": "RMSE",
        "value": money(rmse),
        "foot": "<span class='forecast-foot-note'>Raíz del error cuadrático medio.</span>",
        "foot_is_html": True,
    },
    _mape_card(mape, thr_exc, thr_good, thr_ok),
]
_render_metric_cards(cards_general)

fc_display = fc_plot.copy()
fc_display["fecha_g"] = pd.to_datetime(fc_display["fecha_g"])
fc_display["Período"] = fc_display["fecha_g"].dt.strftime(tick_format)
display_general = fc_display[["Período", "forecast"]].rename(columns={"forecast": "Valor Estimado"})
if (ci_low_plot is not None) and (ci_high_plot is not None):
    display_general["IC Bajo"] = ci_low_plot
    display_general["IC Alto"] = ci_high_plot
cols = display_general.columns.tolist()

display_general_fmt = display_general.copy()
for col in cols[1:]:
    display_general_fmt[col] = display_general_fmt[col].map(money)

st.markdown("#### Horizonte proyectado (General)")
styled_general = _forecast_table_style(display_general_fmt[cols])
style_table(styled_general)
export_general = display_general.rename(columns={"Valor Estimado": "Valor_Estimado"})
if "IC Bajo" in export_general.columns:
    export_general = export_general.rename(columns={"IC Bajo": "IC_Bajo", "IC Alto": "IC_Alto"})
_excel_download(export_general, "Forecast_General", "forecast_general.xlsx")

with st.expander("¿Cómo leer este bloque? (General)"):
    _metrics_explainer_block("General", thr_exc, thr_good, thr_ok)

# ------------------------ Forecast por Tipo de Cuenta ------------------------ #
st.markdown("---")
st.subheader("5. Forecast por Tipo de Cuenta")
freq_alias = "MS" if gran == "Mes" else ("W-MON" if gran == "Semana" else "D")

# ====== 5.1 Cuentas Especiales ====== #
st.markdown("#### 5.1 Cuentas Especiales")
df_especial = df_filtrado[df_filtrado.get("cuenta_especial", False) == True].copy()
ts_ce = _agg_paid_series_like_general(df_especial, freq_alias)

if ts_ce.empty or ts_ce["y"].sum() == 0:
    st.info("Sin pagos suficientes en **Cuentas Especiales**.")
else:
    ts_ce = ts_ce.sort_values("fecha_g").set_index("fecha_g")["y"].astype(float)
    max_period_ce = max(2, len(ts_ce) // 2)
    seasonal_periods_ce = min(seasonal_periods_ui, max_period_ce)
    last_date_ce = ts_ce.index[-1]
    future_idx_ce = pd.date_range(start=last_date_ce, periods=n_steps + 1, freq=freq_alias)[1:]

    # ajuste + forecast + IC
    if modelo_sel == "Proyección de Media Móvil":
        w = min(12, max_period_ce)
        yhat_in_ce = pd.Series(ts_ce).rolling(window=w, min_periods=1).mean().values
        yhat_out_ce = np.full(n_steps, float(pd.Series(ts_ce).rolling(w, min_periods=1).mean().iloc[-1]))
    elif modelo_sel == "Regresión Lineal":
        t = np.arange(len(ts_ce)).reshape(-1, 1)
        reg = LinearRegression().fit(t, ts_ce.values)
        yhat_in_ce = reg.predict(t)
        tf = np.arange(len(ts_ce), len(ts_ce) + n_steps).reshape(-1, 1)
        yhat_out_ce = reg.predict(tf)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                hw = ExponentialSmoothing(
                    ts_ce.values, trend="add", seasonal="add",
                    seasonal_periods=seasonal_periods_ce, damped_trend=damped_trend,
                    initialization_method="estimated"
                ).fit()
            yhat_in_ce = hw.fittedvalues
            yhat_out_ce = hw.forecast(n_steps)
        except Exception as e:
            st.warning(f"Holt-Winters no disponible en CE: {e}")
            w = min(12, max_period_ce)
            yhat_in_ce = pd.Series(ts_ce).rolling(window=w, min_periods=1).mean().values
            yhat_out_ce = np.full(n_steps, float(pd.Series(ts_ce).rolling(w, min_periods=1).mean().iloc[-1]))

    if clip_fc:
        yhat_out_ce = np.maximum(yhat_out_ce, 0.0)

    ci_low_ce, ci_high_ce, _ = _ci_from_residuals(
        ts_ce.values, np.nan_to_num(yhat_in_ce, nan=0.0), yhat_out_ce, z, clip_zero=clip_ci
    )

    err_plus_ce = np.maximum(ci_high_ce - yhat_out_ce, 0.0)
    err_minus_ce = np.maximum(yhat_out_ce - ci_low_ce, 0.0)

    col_plot, col_tab = st.columns([2, 1])
    with col_plot:
        fig_ce = go.Figure()
        fig_ce.add_scatter(x=ts_ce.index, y=ts_ce.values, mode="lines+markers", name="Pagos reales")
        fig_ce.add_scatter(
            x=ts_ce.index, y=np.nan_to_num(yhat_in_ce, nan=0.0), mode="lines",
            name="Tendencia (Modelo)", line=dict(dash="dot")
        )
        fig_ce.add_scatter(
            x=future_idx_ce, y=yhat_out_ce, mode="lines+markers", name="Forecast",
            line=dict(dash="dash"),
            error_y=dict(type="data", array=err_plus_ce, arrayminus=err_minus_ce, visible=True)
        )
        fig_ce.add_scatter(
            x=list(future_idx_ce) + list(future_idx_ce[::-1]),
            y=list(ci_high_ce) + list(ci_low_ce[::-1]),
            fill="toself", fillcolor="rgba(0,0,200,0.08)", line=dict(width=0),
            hoverinfo="skip", name=f"IC {int(alpha_ci*100)}%"
        )
        fig_ce.update_xaxes(tickformat=("%b %Y" if gran == "Mes" else ("Sem %W, %Y" if gran == "Semana" else "%d-%m-%Y")))
        fig_ce.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=420,
                             legend_orientation="h", legend_y=1.02, legend_x=0)
        st.plotly_chart(fig_ce, use_container_width=True)

    with col_tab:
        tb = pd.DataFrame({
            "Período": pd.to_datetime(future_idx_ce).strftime(
                "%b %Y" if gran == "Mes" else ("Sem %W, %Y" if gran == "Semana" else "%d-%m-%Y")
            ),
            "Valor Estimado": yhat_out_ce,
            "IC Bajo": ci_low_ce,
            "IC Alto": ci_high_ce,
        })
        tb_display = tb.copy()
        for col in ["Valor Estimado", "IC Bajo", "IC Alto"]:
            tb_display[col] = tb_display[col].map(money)
        style_table(_forecast_table_style(tb_display))
        export_ce = tb.rename(
            columns={"Valor Estimado": "Valor_Estimado", "IC Bajo": "IC_Bajo", "IC Alto": "IC_Alto"}
        )
        _excel_download(export_ce, "Forecast_CE", "forecast_cuentas_especiales.xlsx")

    st.markdown("#### Resumen Numérico — Cuentas Especiales")
    avg_hist_ce = float(ts_ce.mean())
    avg_fc_ce = float(np.mean(yhat_out_ce))
    var_pct_ce = ((avg_fc_ce - avg_hist_ce) / avg_hist_ce) * 100 if avg_hist_ce > 0 else 0.0
    mae_ce, rmse_ce, mape_ce = _metrics(ts_ce.values, np.nan_to_num(yhat_in_ce, nan=ts_ce.values))

    cards_ce = [
        {
            "label": "Promedio Histórico",
            "value": money(avg_hist_ce),
            "foot": "Nivel medio pagado en el histórico CE.",
        },
        {
            "label": "Forecast Promedio",
            "value": money(avg_fc_ce),
            "foot": "Proyección media para Cuentas Especiales.",
        },
        {
            "label": "Variación % vs Prom.",
            "value": f"{one_decimal(var_pct_ce)}%",
            "foot": "Impacto porcentual frente al histórico.",
        },
        {
            "label": "MAE",
            "value": money(mae_ce),
            "foot": "<span class='forecast-foot-note'>Error absoluto medio (histórico).</span>",
            "foot_is_html": True,
        },
        {
            "label": "RMSE",
            "value": money(rmse_ce),
            "foot": "<span class='forecast-foot-note'>Penaliza errores grandes.</span>",
            "foot_is_html": True,
        },
        _mape_card(mape_ce, thr_exc, thr_good, thr_ok),
    ]
    _render_metric_cards(cards_ce)

    with st.expander("¿Cómo leer este bloque? (Cuentas Especiales)"):
        _metrics_explainer_block("Cuentas Especiales", thr_exc, thr_good, thr_ok)

# ====== 5.2 Cuentas No Especiales ====== #
st.markdown("#### 5.2 Cuentas No Especiales")
df_noesp = df_filtrado[df_filtrado.get("cuenta_especial", False) != True].copy()
ts_ne = _agg_paid_series_like_general(df_noesp, freq_alias)

if ts_ne.empty or ts_ne["y"].sum() == 0:
    st.info("Sin pagos suficientes en **Cuentas No Especiales**.")
else:
    ts_ne = ts_ne.sort_values("fecha_g").set_index("fecha_g")["y"].astype(float)
    max_period_ne = max(2, len(ts_ne) // 2)
    seasonal_periods_ne = min(seasonal_periods_ui, max_period_ne)
    last_date_ne = ts_ne.index[-1]
    future_idx_ne = pd.date_range(start=last_date_ne, periods=n_steps + 1, freq=freq_alias)[1:]

    if modelo_sel == "Proyección de Media Móvil":
        w = min(12, max_period_ne)
        yhat_in_ne = pd.Series(ts_ne).rolling(window=w, min_periods=1).mean().values
        yhat_out_ne = np.full(n_steps, float(pd.Series(ts_ne).rolling(w, min_periods=1).mean().iloc[-1]))
    elif modelo_sel == "Regresión Lineal":
        t = np.arange(len(ts_ne)).reshape(-1, 1)
        reg = LinearRegression().fit(t, ts_ne.values)
        yhat_in_ne = reg.predict(t)
        tf = np.arange(len(ts_ne), len(ts_ne) + n_steps).reshape(-1, 1)
        yhat_out_ne = reg.predict(tf)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                hw = ExponentialSmoothing(
                    ts_ne.values, trend="add", seasonal="add",
                    seasonal_periods=seasonal_periods_ne, damped_trend=damped_trend,
                    initialization_method="estimated"
                ).fit()
            yhat_in_ne = hw.fittedvalues
            yhat_out_ne = hw.forecast(n_steps)
        except Exception as e:
            st.warning(f"Holt-Winters no disponible en No CE: {e}")
            w = min(12, max_period_ne)
            yhat_in_ne = pd.Series(ts_ne).rolling(window=w, min_periods=1).mean().values
            yhat_out_ne = np.full(n_steps, float(pd.Series(ts_ne).rolling(w, min_periods=1).mean().iloc[-1]))

    if clip_fc:
        yhat_out_ne = np.maximum(yhat_out_ne, 0.0)

    ci_low_ne, ci_high_ne, _ = _ci_from_residuals(
        ts_ne.values, np.nan_to_num(yhat_in_ne, nan=0.0), yhat_out_ne, z, clip_zero=clip_ci
    )

    err_plus_ne = np.maximum(ci_high_ne - yhat_out_ne, 0.0)
    err_minus_ne = np.maximum(yhat_out_ne - ci_low_ne, 0.0)

    col_plot, col_tab = st.columns([2, 1])
    with col_plot:
        fig_ne = go.Figure()
        fig_ne.add_scatter(x=ts_ne.index, y=ts_ne.values, mode="lines+markers", name="Pagos reales")
        fig_ne.add_scatter(
            x=ts_ne.index, y=np.nan_to_num(yhat_in_ne, nan=0.0), mode="lines",
            name="Tendencia (Modelo)", line=dict(dash="dot")
        )
        fig_ne.add_scatter(
            x=future_idx_ne, y=yhat_out_ne, mode="lines+markers", name="Forecast",
            line=dict(dash="dash"),
            error_y=dict(type="data", array=err_plus_ne, arrayminus=err_minus_ne, visible=True)
        )
        fig_ne.add_scatter(
            x=list(future_idx_ne) + list(future_idx_ne[::-1]),
            y=list(ci_high_ne) + list(ci_low_ne[::-1]),
            fill="toself", fillcolor="rgba(0,0,200,0.08)", line=dict(width=0),
            hoverinfo="skip", name=f"IC {int(alpha_ci*100)}%"
        )
        fig_ne.update_xaxes(tickformat=("%b %Y" if gran == "Mes" else ("Sem %W, %Y" if gran == "Semana" else "%d-%m-%Y")))
        fig_ne.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=420,
                             legend_orientation="h", legend_y=1.02, legend_x=0)
        st.plotly_chart(fig_ne, use_container_width=True)

    with col_tab:
        tb = pd.DataFrame({
            "Período": pd.to_datetime(future_idx_ne).strftime(
                "%b %Y" if gran == "Mes" else ("Sem %W, %Y" if gran == "Semana" else "%d-%m-%Y")
            ),
            "Valor Estimado": yhat_out_ne,
            "IC Bajo": ci_low_ne,
            "IC Alto": ci_high_ne,
        })
        tb_display = tb.copy()
        for col in ["Valor Estimado", "IC Bajo", "IC Alto"]:
            tb_display[col] = tb_display[col].map(money)
        style_table(_forecast_table_style(tb_display))
        export_ne = tb.rename(
            columns={"Valor Estimado": "Valor_Estimado", "IC Bajo": "IC_Bajo", "IC Alto": "IC_Alto"}
        )
        _excel_download(export_ne, "Forecast_NoCE", "forecast_cuentas_no_especiales.xlsx")

    st.markdown("#### Resumen Numérico — Cuentas No Especiales")
    avg_hist_ne = float(ts_ne.mean())
    avg_fc_ne = float(np.mean(yhat_out_ne))
    var_pct_ne = ((avg_fc_ne - avg_hist_ne) / avg_hist_ne) * 100 if avg_hist_ne > 0 else 0.0
    mae_ne, rmse_ne, mape_ne = _metrics(ts_ne.values, np.nan_to_num(yhat_in_ne, nan=ts_ne.values))

    cards_ne = [
        {
            "label": "Promedio Histórico",
            "value": money(avg_hist_ne),
            "foot": "Nivel medio pagado en el histórico No CE.",
        },
        {
            "label": "Forecast Promedio",
            "value": money(avg_fc_ne),
            "foot": "Proyección media para cuentas No Especiales.",
        },
        {
            "label": "Variación % vs Prom.",
            "value": f"{one_decimal(var_pct_ne)}%",
            "foot": "Impacto porcentual frente al histórico.",
        },
        {
            "label": "MAE",
            "value": money(mae_ne),
            "foot": "<span class='forecast-foot-note'>Error absoluto medio (histórico).</span>",
            "foot_is_html": True,
        },
        {
            "label": "RMSE",
            "value": money(rmse_ne),
            "foot": "<span class='forecast-foot-note'>Raíz del error cuadrático medio.</span>",
            "foot_is_html": True,
        },
        _mape_card(mape_ne, thr_exc, thr_good, thr_ok),
    ]
    _render_metric_cards(cards_ne)

    with st.expander("¿Cómo leer este bloque? (Cuentas No Especiales)"):
        _metrics_explainer_block("Cuentas No Especiales", thr_exc, thr_good, thr_ok)

st.html("</div>")